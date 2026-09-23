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

from src.geometry.model import NJ_STATE_PLANE_FT
from src.geometry.network.area import area_corridors
from src.geometry.treatments import route_decision_for

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "network"

WGS84_EPSG = 4326


def _decision_name(road: str, municipality: str) -> str | None:
    decision = route_decision_for(road, municipality)
    return type(decision).__name__ if decision is not None else None


def network_features(area: str) -> gpd.GeoDataFrame:
    """Every corridor, kerb run and crossing in one frame, tagged by `kind`.

    One frame rather than three files: a reader asking "what is on this street" wants them
    together, and `kind` is cheaper to filter than three joins.
    """
    rows: list[dict] = []
    for corridor in area_corridors(area):
        town = corridor.municipalities[0] if corridor.municipalities else ""
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
            f"{len(decided)} street(s) carrying a route decision: "
            f"{', '.join(sorted(decided['name'])) or 'none'}")


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
