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
from src.geometry.treatments import route_decision_for
from src.geometry.treatments.corridor import CorridorFacility

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "network"

WGS84_EPSG = 4326

#: How far from a crossing a traced kerb can be and still be the kerb it ENDS at. A crossing way
#: is mapped kerb to kerb, so its own length plus a little is the honest reach.
CROSSING_KERB_REACH_FT = 30.0


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


def _context_rows(area: str, kerbs: list, pavement) -> list[dict]:
    """Buildings, sidewalks and crossing markings: the street's surroundings rather than the
    street. Filed with no `name` because they belong to the AREA - a building fronts whichever
    street it fronts, and deciding that here would be a join nothing downstream asked for.

    The bars come from the surveyor's own `crossing:markings`, so an unmarked crossing paints
    nothing. Trimmed against the traced kerbs the document already holds, which is why the kerbs
    are passed in rather than re-read: two reads are two chances to disagree.
    """
    from shapely import STRtree

    from src.geometry.surveyed import crossing_bars_ft, crossing_lines_ft

    context = area_context(area)
    # Only the kerbs NEAR each crossing. `carriageway_geometry_ft` trims the crossing where it
    # meets a kerb, and handed all 75 borough-wide runs it trims against one on another street
    # and collapses the way to a point.
    tree = STRtree(kerbs) if kerbs else None
    # OSM footprints are coarser than the traced kerbs, so a few sit in the carriageway. Dropped
    # rather than drawn standing in the road - src/render/export.py does the same, against the
    # same geometry. Skipped entirely where nothing is paved, which would drop every building.
    rows: list[dict] = [{"kind": "building", "height_m": round(height, 2), "geometry": ring}
                        for ring, height in context["buildings"]
                        if pavement is None or not ring.intersects(pavement)]
    rows += [{"kind": "sidewalk", "geometry": line} for line in context["sidewalks"]]
    for crossing in context["crossings"]:
        marks = crossing.markings
        near = ([kerbs[i] for i in tree.query(crossing.geometry.buffer(CROSSING_KERB_REACH_FT))]
                if tree is not None else [])
        rows += [{"kind": "crossing_bar", "markings": marks, "geometry": bar}
                 for bar in crossing_bars_ft(crossing, near)]
        rows += [{"kind": "crossing_line", "markings": marks, "geometry": line}
                 for line in crossing_lines_ft(crossing, near)]
    return rows


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
    rows += _context_rows(area, [r["geometry"] for r in rows if r["kind"] == "kerb"],
                          unary_union(paved) if paved else None)
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
            f"{(features['kind'] == 'sidewalk').sum()} sidewalks, "
            f"{(features['kind'] == 'crossing_bar').sum()} crossing bars")


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
