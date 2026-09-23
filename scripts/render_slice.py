#!/usr/bin/env python
"""Draw a 2D sheet from a SLICE of a network GeoJSON. No site, no scenario, no model.

A "site" here is a window onto the borough document, so a sheet is a crop plus a scale:

    scripts/render_slice.py --street "Broad Street"
    scripts/render_slice.py --around -74.76196,40.38918 --radius-ft 300 --name broad_greenwood

Everything drawn comes from the file `scripts/export_network.py` wrote. Nothing is recomputed
here - if a marking is missing from the picture it is missing from the document, which is the
property that makes the two impossible to disagree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import matplotlib
from shapely.geometry import Point, box

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # after matplotlib.use: the backend must be set first

from src.geometry.model import NJ_STATE_PLANE_FT

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = REPO_ROOT / "output" / "network"
OUT_DIR = REPO_ROOT / "output" / "slices"

#: Matched to src/render/plan_view.py so a slice and a junction sheet read as the same drawing.
STYLE: dict[str, dict] = {
    "street":         dict(color="#9a9a9a", linewidth=0.8, linestyle=(0, (6, 4)), zorder=1),
    "kerb":           dict(color="#2b2b2b", linewidth=1.4, zorder=4),
    "bikeway":        dict(color="mediumseagreen", alpha=0.45, zorder=2),
    "bikeway_buffer": dict(color="#c8c8c8", alpha=0.7, zorder=2),
    "edge_line":      dict(color="seagreen", linewidth=1.6, zorder=3),
    "bollard":        dict(color="darkorange", markersize=2.5, zorder=6),
    "crossing":       dict(color="#4a4a4a", markersize=3.0, marker="+", zorder=5),
}
DRAW_ORDER = ("street", "bikeway", "bikeway_buffer", "kerb", "edge_line", "crossing", "bollard")

FT_PER_IN = 120.0


def load_network(area: str, network_dir: Path = NETWORK_DIR) -> gpd.GeoDataFrame:
    """The document in state-plane FEET, which is the frame every distance here is in."""
    path = network_dir / f"{area}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist - run scripts/export_network.py first")
    return gpd.read_file(path).to_crs(NJ_STATE_PLANE_FT)


def slice_around(network: gpd.GeoDataFrame, center_ft: Point, radius_ft: float
                 ) -> gpd.GeoDataFrame:
    """Everything within a square window. CLIPPED, not filtered by centroid: a 1,050 ft bikeway
    run whose centre is outside the window still crosses it, and dropping it would draw a hole."""
    return gpd.clip(network, box(center_ft.x - radius_ft, center_ft.y - radius_ft,
                                 center_ft.x + radius_ft, center_ft.y + radius_ft))


def slice_for_street(network: gpd.GeoDataFrame, street: str, pad_ft: float = 80.0
                     ) -> gpd.GeoDataFrame:
    """Everything within `pad_ft` of a named street, its own features and its neighbours'."""
    named = network[network["name"] == street]
    if named.empty:
        raise SystemExit(f"no street named {street!r} in this document. Have: "
                         + ", ".join(sorted(network['name'].dropna().unique())))
    return gpd.clip(network, named.union_all().buffer(pad_ft).envelope)


def draw(features: gpd.GeoDataFrame, title: str, out_path: Path) -> Path:
    minx, miny, maxx, maxy = features.total_bounds
    fig, ax = plt.subplots(figsize=((maxx - minx) / FT_PER_IN, (maxy - miny) / FT_PER_IN))

    for kind in DRAW_ORDER:
        part = features[features["kind"] == kind]
        if part.empty:
            continue
        style = dict(STYLE[kind])
        if part.geom_type.isin(("Point", "MultiPoint")).all():
            part.plot(ax=ax, marker=style.pop("marker", "o"), linestyle="none", **style)
        else:
            part.plot(ax=ax, **style)

    ax.set_title(title, fontsize=9, loc="left")
    ax.set_aspect("equal")
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _center_ft(lonlat: str) -> Point:
    lon, lat = (float(v) for v in lonlat.split(","))
    return gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(NJ_STATE_PLANE_FT).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--street", help="slice to one named street")
    parser.add_argument("--around", help="lon,lat to centre a square window on")
    parser.add_argument("--radius-ft", type=float, default=300.0)
    parser.add_argument("--name", help="output stem (default: derived from the slice)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    network = load_network(args.area)
    if args.street:
        features, stem = slice_for_street(network, args.street), args.street.lower().replace(" ", "_")
    elif args.around:
        features = slice_around(network, _center_ft(args.around), args.radius_ft)
        stem = f"{args.around.replace(',', '_')}_{args.radius_ft:.0f}ft"
    else:
        features, stem = network, args.area

    if features.empty:
        raise SystemExit("that slice is empty - nothing to draw")
    counts = ", ".join(f"{n} {k}" for k, n in features["kind"].value_counts().items())
    print(f"{stem}: {len(features)} feature(s) - {counts}")
    print(f"wrote {draw(features, stem, args.out_dir / f'{args.name or stem}.png')}")


if __name__ == "__main__":
    main()
