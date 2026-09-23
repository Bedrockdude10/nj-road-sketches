#!/usr/bin/env python
"""Render a 2D sheet and a 3D still from a SLICE of a network GeoJSON. No site, no scenario file.

    scripts/render_slice.py --street "Broad Street"
    scripts/render_slice.py --around=-74.76196,40.38918 --radius-ft 300 --name broad_greenwood --3d

A "site" here is a window onto the borough document, so a drawing is a crop plus a decision:
`slice_design` turns the crop into the (model, state) pair this project's renderers already
take, `route_decision_for` says what is proposed along each street in it, and `plot_design_state`
and `export_scenario` draw it. Nothing about the geometry is computed here - if a marking is
missing from the picture it is missing from the document or from the treatment, which is the
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
from src.geometry.network.slice_design import slice_design, slice_pavement
from src.geometry.treatments import route_decision_for
from src.render.export import export_scenario
from src.render.plan_view import plot_design_state

REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = REPO_ROOT / "output" / "network"
OUT_DIR = REPO_ROOT / "output" / "slices"

WGS84_EPSG = 4326


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


def _parts(geom):
    """A clip can split one footprint or way into several; each is its own record."""
    return [g for g in getattr(geom, "geoms", [geom]) if g is not None and not g.is_empty]


def slice_context(features: gpd.GeoDataFrame) -> dict[str, list[dict]]:
    """The document's own context, in the shape the OSM fetchers return it.

    Passed to the renderers so they do NOT fetch: `fetch_buildings` and friends take a centre and
    a radius, and the largest radius fitting the declared snapshot bbox is smaller than the
    borough, so a slice near its edge would silently lose its surroundings.
    """
    wgs84 = features.to_crs(WGS84_EPSG)
    buildings = wgs84[wgs84["kind"] == "building"]
    crossings = wgs84[wgs84["kind"] == "crossing_way"]
    return {
        "buildings": [{"coords_wgs84": list(part.exterior.coords), "height_m": row.height_m,
                       "height_source": "osm", "tags": {}}
                      for row in buildings.itertuples() for part in _parts(row.geometry)
                      if part.geom_type == "Polygon"],
        "crossings": [{"coords_wgs84": list(part.coords), "node_ids": [],
                       "tags": {"crossing:markings": row.markings}
                                if isinstance(row.markings, str) else {}}
                      for row in crossings.itertuples() for part in _parts(row.geometry)
                      if part.geom_type == "LineString"],
    }


def design_for(features: gpd.GeoDataFrame):
    """(model, state, pavement) for a slice, with every route decision in it applied.

    EVERY street in the window is offered its decision, not one named street: that is what makes
    this a network drawing rather than a site. A street with no decision simply gets none back.
    """
    model, state = slice_design(features)
    for street in sorted({leg.name for leg in model.legs.values()}):
        decision = route_decision_for(street, features["municipality"].dropna().iloc[0])
        if decision is not None:
            state = decision.apply_to(state, model)
    return model, state, slice_pavement(features)


def draw_2d(features: gpd.GeoDataFrame, name: str, out_dir: Path) -> Path:
    model, state, pavement = design_for(features)
    context = slice_context(features)
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_design_state(ax, model, state, name, crossings=context["crossings"],
                      sidewalks=[], traffic_control=[], street_furniture=[], pavement=pavement)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_3d(features: gpd.GeoDataFrame, name: str, out_dir: Path) -> Path:
    from scripts.phase4_render_3d import find_blender, render_all

    model, state, pavement = design_for(features)
    context = slice_context(features)
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry, png = out_dir / f"{name}_3d.json", out_dir / f"{name}_3d.png"
    export_scenario(model, state, name, geometry, pavement=pavement,
                    buildings=context["buildings"], crossings=context["crossings"],
                    traffic_control=[], street_furniture=[])
    render_all(find_blender(), [(geometry, png)])
    return png


def _center_ft(lonlat: str) -> Point:
    lon, lat = (float(v) for v in lonlat.split(","))
    return gpd.GeoSeries([Point(lon, lat)], crs=WGS84_EPSG).to_crs(NJ_STATE_PLANE_FT).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--street", help="slice to one named street")
    parser.add_argument("--around", help="lon,lat to centre a square window on")
    parser.add_argument("--radius-ft", type=float, default=300.0)
    parser.add_argument("--name", help="output stem (default: derived from the slice)")
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--3d", dest="three_d", action="store_true", help="also render in 3D")
    args = parser.parse_args()

    network = load_network(args.area, args.network_dir)
    if args.street:
        features, stem = slice_for_street(network, args.street), args.street.lower().replace(" ", "_")
    elif args.around:
        features = slice_around(network, _center_ft(args.around), args.radius_ft)
        stem = f"{args.around.replace(',', '_')}_{args.radius_ft:.0f}ft"
    else:
        features, stem = network, args.area

    if features.empty:
        raise SystemExit("that slice is empty - nothing to draw")
    stem = args.name or stem
    counts = ", ".join(f"{n} {k}" for k, n in features["kind"].value_counts().items())
    print(f"{stem}: {len(features)} feature(s) - {counts}")
    print(f"wrote {draw_2d(features, stem, args.out_dir)}")
    if args.three_d:
        print(f"wrote {draw_3d(features, stem, args.out_dir)}")


if __name__ == "__main__":
    main()
