#!/usr/bin/env python
"""Write a municipality's whole street network as one GeoJSON, from OSM alone.

The 2D layer's output for an AREA rather than a junction: every named street in the boundary,
the kerb actually traced along it, every crossing, and the route decision that applies. A render
is then a crop of this, not a separately configured site.

    scripts/export_network.py                       -> output/network/hopewell_borough.geojson
    scripts/export_network.py --area pennington_borough

WGS84, because GeoJSON is lon/lat by spec and the consumer is a map. The geometry is built in
state-plane feet and reprojected once at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

from src.geometry.model import NJ_STATE_PLANE_FT
from src.geometry.corridor_paint import paint_facility
from src.geometry.markings import (BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE,
                                   BIKE_LANE_SURFACE)
from src.geometry.network.area import area_context, area_corridors, corridor_pavement
from src.sources.observations import ELEMENT_FROM_OSM
from src.geometry.treatments import route_decision_for
from src.geometry.treatments.corridor import CorridorFacility

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "network"

WGS84_EPSG = 4326



def _decision_name(road: str, municipality: str) -> str | None:
    decision = route_decision_for(road, municipality)
    return type(decision).__name__ if decision is not None else None


def _paint_rows(corridor, facility, town: str) -> list[dict]:
    """The facility this route proposes, drawn along the whole corridor.

    `paint_facility` never needed a junction model - it takes a Corridor - so the design comes
    out network-wide for free. Broad St: 3,953 ft of 6,868 placed on the north kerb in 11 runs,
    0 refusals, 15 spans where the kerb is untraced and the section cannot be tested.
    """
    common = {"name": corridor.name, "municipality": town}
    paint = paint_facility(corridor, facility)
    rows: list[dict] = []
    for run in paint.runs:
        span = {"start_ft": round(run.start_ft, 2), "end_ft": round(run.end_ft, 2),
                "side": paint.side, "compass_side": paint.compass_side,
                "width_ft": round(run.section.width_ft, 2),
                "buffer_ft": round(run.section.buffer_ft, 2),
                "constrained": bool(run.section.constrained)}
        # `paint_kind` is the markings.PaintKind NAME, and it is what makes a slice renderable
        # without a second channel table: the 3D bridge looks the kind up in that registry and
        # hands real PaintPieces to the exporter's own serializer, so a marking is drawn in the
        # channel - and therefore the colour and the builder - its kind already declares.
        rows.append({"kind": "bikeway", "paint_kind": BIKE_LANE_SURFACE.name,
                     **common, **span, "geometry": run.lane_surface})
        rows.append({"kind": "bikeway_buffer", "paint_kind": BIKE_BUFFER_FILL.name,
                     **common, **span, "geometry": run.buffer_zone})
        rows += [{"kind": "edge_line", "paint_kind": BIKE_LANE_EDGE_LINE.name,
                  **common, **span, "geometry": line}
                 for line in run.edge_lines]
        rows += [{"kind": "bollard", **common, **span, "geometry": Point(xy)}
                 for xy in run.bollards]
    return rows


def _context_rows(area: str, pavement) -> list[dict]:
    """The street's surroundings, AS OSM RECORDS THEM: geometry plus the element's own tags.

    Filed with no `name` because they belong to the AREA - a building fronts whichever street it
    fronts, and deciding that here would be a join nothing downstream asked for.

    THE PROPERTIES ARE THE TAGS. There is no schema of ours here to keep in step with OSM's: a
    building's height and a crossing's markings are functions of the tags, so a `height_m` or
    `markings` column beside them would be a second copy free to disagree - and the functions
    that derive them (`height_from_tags`, `_markings_from_tags`) already exist and already run
    at read time. One JSON column rather than a column per key, because the borough's tag
    vocabulary is hundreds of keys and a sparse table of them is mostly nulls.

    EVERY LAYER, not a chosen few: `area_context` sweeps whatever the fetchers in
    src/sources/osm_context.py know how to ask for, and this writes all of it out under
    `osm_<layer>`. Carrying three of the ten is what left a slice with no driveways, no parking
    and no surveyed footway while the same junction drawn as a site had 33 paved surfaces.
    `other` and `relations` are the same fix one level further: a tagged way none of those ten
    predicates claims, and a `type=multipolygon` relation whose tags live only on the relation
    (a park or a soil-survey polygon whose member ways are themselves untagged), land here too
    rather than being the one thing this document still can't answer for.

    ONLY WHAT CANNOT BE DERIVED is still left out: the crossing BARS are not stored, because
    export_scenario paints them from the way against the surveyor's own crossing:markings.
    """
    context = area_context(area)
    # OSM footprints are coarser than the traced kerbs, so a few sit in the carriageway. Dropped
    # rather than drawn standing in the road - src/render/export.py does the same, against the
    # same geometry. Skipped where nothing is paved, which would drop every building.
    rows = [_osm_row("building", item) for item in context.pop("buildings")
            if pavement is None or not item["geometry"].intersects(pavement)]
    rows += [_osm_row("osm_node", item) for item in context.pop("nodes")]
    # `crossing_way` and `kerb_way` keep the names they were first written under, so a document
    # read by an older checkout still finds them; every layer added since is `osm_<layer>`.
    named = {"crossings": "crossing_way", "kerb_ways": "kerb_way"}
    rows += [_osm_row(named.get(layer, f"osm_{layer}"), item)
             for layer, items in context.items() for item in items]
    return rows


def _osm_row(kind: str, item: dict) -> dict:
    """One OSM element as one feature: its geometry, its tags, its way id and its node ids.

    `node_ids` is carried because a tactile pad is placed at a node SHARED by a crossing way and
    a tactile_paving kerb way - the topology IS the observation, and no geometry expresses it.
    The way `id` because a PavedSurface keeps one, and the kerb-opening rules match on it.

    `provenance` because an element this project OBSERVED must be tellable from one a mapper
    surveyed. Without it the carrier's whole bargain is unauditable: the document would assert a
    signal mast in the same voice as a traced kerb, and nothing downstream could ask which is
    which. Computed in area_context and dropped here is the same defect a `label` field had -
    a fact derived and then thrown away at the boundary.
    """
    return {"kind": kind, "tags": json.dumps(item["tags"], sort_keys=True),
            "way_ids": str(item["id"]) if item.get("id") is not None else "",
            "node_ids": ",".join(str(i) for i in item.get("node_ids") or ()),
            "provenance": item.get("provenance", ELEMENT_FROM_OSM),
            "geometry": item["geometry"]}


def network_features(area: str) -> gpd.GeoDataFrame:
    """Every corridor, kerb run, crossing and proposed marking in one frame, tagged by `kind`.

    One frame rather than several files: a reader asking "what is on this street" wants them
    together, and `kind` is cheaper to filter than a pile of joins.
    """
    rows: list[dict] = []
    for corridor in area_corridors(area):
        town = corridor.municipalities[0] if corridor.municipalities else ""
        facility = route_decision_for(corridor.name, town)
        decision = _decision_name(corridor.name, town)
        traced_ft = sum(run.length_ft for run in corridor.kerb_runs)
        rows.append({"kind": "street", "name": corridor.name, "municipality": town,
                     "length_ft": round(corridor.length_ft, 2),
                     "decision": decision,
                     "traced_kerb_ft": round(traced_ft, 2),
                     # Of BOTH kerbs, so 1.0 is a street traced down each side.
                     "kerb_coverage": round(traced_ft / (2 * corridor.length_ft), 4),
                     "crossings": len(corridor.cross_street_ft),
                     "geometry": corridor.centerline})
        # The asphalt, so a slice of this document is a street and not paint floating in space.
        pavement = corridor_pavement(corridor)
        if pavement is not None:
            rows.append({"kind": "pavement", "name": corridor.name, "municipality": town,
                         "width_ft": round(corridor.nominal_width_ft, 2),
                         "area_sqft": round(pavement.area, 1), "geometry": pavement})
        rows += [{"kind": "kerb", "name": corridor.name, "municipality": town,
                  "side": run.side, "source": run.source,
                  "start_ft": round(run.start_ft, 2), "end_ft": round(run.end_ft, 2),
                  "length_ft": round(run.length_ft, 2),
                  "way_ids": ",".join(str(i) for i in run.way_ids),
                  "geometry": run.line}
                 for run in corridor.kerb_runs]
        rows += [{"kind": "crossing", "name": corridor.name, "municipality": town,
                  "station_ft": round(station, 2),
                  "geometry": Point(corridor.centerline.interpolate(station).coords[0])}
                 for station in corridor.cross_street_ft]
        # Only a facility DRAWS something. CorridorCalming places no new section - it narrows an
        # existing one - so it contributes a `decision` on the street and no paint of its own.
        if isinstance(facility, CorridorFacility):
            rows += _paint_rows(corridor, facility, town)

    paved = [r["geometry"] for r in rows if r["kind"] == "pavement"]
    rows += _context_rows(area, unary_union(paved) if paved else None)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=NJ_STATE_PLANE_FT)


def export_network(area: str, out_dir: Path = OUT_DIR) -> Path:
    features = network_features(area).to_crs(WGS84_EPSG)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{area}.geojson"
    # Written whole each time rather than appended: GeoJSON has no update, and a stale half-file
    # reads as a smaller town. Same reason export_all_scenarios.py clears before it writes.
    features.to_file(path, driver="GeoJSON")
    return path


def _summarise(features: gpd.GeoDataFrame) -> str:
    streets = features[features["kind"] == "street"]
    decided = streets[streets["decision"].notna()]
    return (f"{len(streets)} streets ({streets['length_ft'].sum() / 5280:.2f} mi), "
            f"{(features['kind'] == 'kerb').sum()} kerb runs, "
            f"{(features['kind'] == 'crossing').sum()} crossings, "
            f"{features[features['kind'] == 'pavement']['area_sqft'].sum() / 43560:.1f} acres "
            f"of asphalt, "
            f"{len(decided)} street(s) carrying a route decision "
            f"({', '.join(sorted(decided['name'])) or 'none'}), "
            f"{(features['kind'] == 'bikeway').sum()} bikeway run(s) totalling "
            f"{_bikeway_ft(features):,.0f} ft, {(features['kind'] == 'bollard').sum()} bollards, "
            f"{(features['kind'] == 'building').sum()} buildings, "
            + ", ".join(f"{n} {k}" for k, n in sorted(features[
                features["kind"].str.startswith(("crossing_way", "kerb_way", "osm_"))
            ]["kind"].value_counts().items())))


def _bikeway_ft(features: gpd.GeoDataFrame) -> float:
    runs = features[features["kind"] == "bikeway"]
    return float((runs["end_ft"] - runs["start_ft"]).sum()) if len(runs) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    features = network_features(args.area)
    print(f"{args.area}: {_summarise(features)}")
    path = export_network(args.area, args.out_dir)
    print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
