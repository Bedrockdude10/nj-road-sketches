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
import json
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
from src.geometry.treatments import (existing_conditions, osm_derived_baseline,
                                     route_decision_for)
from src.sources.osm_context import (height_from_tags, is_kerb, is_street_furniture,
                                     is_traffic_control)
from src.render.export import export_scenario
from src.render.frame import Frame
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


#: {layer: (document kind, the geometry type it is carried as)} - every OSM layer a renderer can
#: be handed, keyed by the name its FETCHER uses, because that is the name the consumer knows it
#: by. The document writes `osm_<layer>` for everything added after the first three; see
#: scripts/export_network.py:_context_rows.
SLICE_LAYERS: dict[str, tuple[str, str]] = {
    "buildings": ("building", "Polygon"),
    "parking_lots": ("osm_parking_lots", "Polygon"),
    "crossings": ("crossing_way", "LineString"),
    "sidewalks": ("osm_sidewalks", "LineString"),
    "kerb_ways": ("kerb_way", "LineString"),
    "driveways": ("osm_driveways", "LineString"),
    "parking_aisles": ("osm_parking_aisles", "LineString"),
    "stop_lines": ("osm_stop_lines", "LineString"),
    "roads": ("osm_roads", "LineString"),
}


def slice_context(features: gpd.GeoDataFrame) -> dict[str, list[dict]]:
    """The document's own OSM context, in the shape the OSM fetchers return it.

    Same keys, same dicts, so nothing downstream can tell a slice from a junction - and the
    renderers never fetch. That matters twice over: `fetch_buildings` and friends take a centre
    and a radius, and the largest radius fitting the declared snapshot bbox is smaller than the
    borough, so a slice near its edge would silently lose its surroundings.

    The tags are OSM's own, carried verbatim through the document, so everything derived FROM
    them - a building's height, a crossing's markings - is derived here by the same functions a
    site uses rather than read from a column that could disagree.

    EVERY layer, because the consumer decides what it needs and this cannot: `_paved_surfaces_ft`
    reads four of them to build one list, and handing it three drew the driveways and lost the
    streets around them. A layer the document holds and this does not pass on is invisible.
    """
    wgs84 = features.to_crs(WGS84_EPSG)
    # A dict OR a JSON string: GDAL tags the column as a JSON subtype on write and parses it
    # back to a dict on read, but a document written by another driver - or read straight off
    # disk - still carries the string. Accepting only one of the two silently emptied every tag,
    # and an empty tag dict fails every predicate rather than raising.
    def tags_of(row) -> dict:
        tags = getattr(row, "tags", None)
        return tags if isinstance(tags, dict) else json.loads(tags) if isinstance(tags, str) else {}

    of_kind = lambda kind: wgs84[wgs84["kind"] == kind].itertuples()

    def ids(row, field: str) -> list[int]:
        return [int(i) for i in str(getattr(row, field, "")).split(",") if i.strip().isdigit()]

    def layer(name: str) -> list[dict]:
        kind, geom_type = SLICE_LAYERS[name]
        return [{"coords_wgs84": list(part.exterior.coords if geom_type == "Polygon"
                                      else part.coords),
                 "tags": tags_of(row), "id": next(iter(ids(row, "way_ids")), None),
                 "node_ids": ids(row, "node_ids")}
                for row in of_kind(kind) for part in _parts(row.geometry)
                if part.geom_type == geom_type]

    nodes = [(tags_of(row), part, row) for row in of_kind("osm_node")
             for part in _parts(row.geometry) if part.geom_type == "Point"]
    context = {name: layer(name) for name in SLICE_LAYERS}
    # height_m None where nobody recorded one, which is what fetch_buildings means by it -
    # "nobody said" is a different answer from the default, and export.py looks elsewhere.
    for item in context["buildings"]:
        found = height_from_tags(item["tags"])
        item["height_m"], item["height_source"] = found if found else (None, None)
    # Kerb NODES, appended to the kerb ways exactly as `fetch_kerbs` returns them in one list:
    # OSM tags a dropped kerb on the node where the footway crosses, and the two are one layer.
    context["kerb_ways"] += [{"coords_wgs84": None, "lon": p.x, "lat": p.y, "tags": t,
                              "id": next(iter(ids(row, "way_ids")), None)}
                             for t, p, row in nodes if is_kerb(t)]
    context["traffic_control"] = [{"lon": p.x, "lat": p.y, "tags": t}
                                  for t, p, _row in nodes if is_traffic_control(t)]
    context["street_furniture"] = [{"lon": p.x, "lat": p.y, "tags": t}
                                   for t, p, _row in nodes if is_street_furniture(t)]
    return context


def window_frame(features: gpd.GeoDataFrame) -> Frame:
    """The extent the drawing covers: the WINDOW, not a radius derived from the legs.

    `junction_frame` sizes itself on how far the longest leg reaches from the junction, which is
    the right answer when the drawing is one junction and the wrong one when it is a crop: on a
    600 ft window it returned a 437 ft half-width, so the coverage report demanded surveyed
    features from 137 ft outside the clip the slice was built from and called them missing.
    A Frame is already a square about a centre, so a window IS one.
    """
    minx, miny, maxx, maxy = features.total_bounds
    return Frame(Point((minx + maxx) / 2, (miny + maxy) / 2),
                 max(maxx - minx, maxy - miny) / 2)


def context_layers(context: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The layers BOTH views take, so neither can be handed a set the other was not.

    Two layers are deliberately not here, and neither is a view drawing something the other
    cannot. `buildings` the 2D sheet does not draw. `sidewalks` is the surveyed OSM CENTRELINE,
    which is a 2D annotation - the 3D footway is a BAND derived from the pavement both views
    share (`build_sidewalk_pieces`), so the walkable surface is in both and only the blue
    reference line is in one. Everything else goes to both, which is what keeps a hydrant from
    existing in one view and not the other.
    """
    return {key: context[key]
            for key in ("crossings", "traffic_control", "street_furniture", "kerb_ways",
                        "stop_lines")}


def _route_decisions(state, model, features: gpd.GeoDataFrame):
    """Whatever `route_decision_for` proposes along each street in the window.

    EVERY street is offered its decision, not one named street: that is what makes this a
    network drawing rather than a site. A street with no decision simply gets none back, and the
    side a facility takes is the route's own (Broad St's two-way bikeway is on the north kerb -
    see CORRIDOR_SIDE), not something the crop chooses.
    """
    town = features["municipality"].dropna().iloc[0]
    # The street name off the CONFIG, not off `Leg.name` - a Leg is named for its own key, as at
    # a site, and `config["legs"][key]["street_name"]` is the one place the street is recorded.
    # See `legs_on_road`, which reads the same field for the same reason.
    streets = {cfg.get("street_name") for cfg in model.config.get("legs", {}).values()}
    for street in sorted(s for s in streets if s):
        decision = route_decision_for(street, town)
        if decision is not None:
            state = decision.apply_to(state, model)
    return state


#: The scenarios a window can be drawn in, each composed from treatments that take a (state,
#: model) and no leg names - so there is nothing per-site here and nothing to add for a new crop.
#: `sites/*/scenarios.py` builds the same three out of hand-written leg tuples; these do not.
SCENARIOS = {
    "existing": lambda state, model, features: state,
    "proposed": lambda state, model, features: osm_derived_baseline(state, model),
    # NOT on top of `proposed`: osm_derived_baseline paints the kerbs the way OSM says they are
    # used, and that parking competes with the facility for the same width - stacked, the section
    # lands 14.6 ft wide and leaves a 10.3 ft travel lane, which travel_lane_too_narrow refuses.
    # The route decision IS the proposal here; what the kerbs do under it is the decision's own.
    "two_way_bikeway": lambda state, model, features: _route_decisions(state, model, features),
}


def design_for(features: gpd.GeoDataFrame, scenario: str = "two_way_bikeway"):
    """(model, state, pavement, context) for a slice, drawn in one scenario.

    The baseline is `existing_conditions`, the same state every site pipeline labels "Existing
    Conditions", so "existing" here means what it means everywhere else in this repo.
    """
    context = slice_context(features)
    model, _ = slice_design(features, osm=context)
    state = SCENARIOS[scenario](existing_conditions(model), model, features)
    return model, state, slice_pavement(features), context


def draw_2d(features: gpd.GeoDataFrame, name: str, out_dir: Path,
             scenario: str = "two_way_bikeway", dpi: int = 200) -> Path:
    model, state, pavement, context = design_for(features, scenario)
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_design_state(ax, model, state, name, pavement=pavement,
                      frame=window_frame(features), sidewalks=context["sidewalks"],
                      **context_layers(context))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    # A double yellow's strokes are 4 in apart (DOUBLE_YELLOW_GAP_FT): 1.2 px at 200 dpi over a
    # 600 ft window, narrower than the strokes, so they merge. The geometry is a true double and
    # the sheet cannot resolve it - raise the density rather than widening the paint.
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_3d(features: gpd.GeoDataFrame, name: str, out_dir: Path,
             scenario: str = "two_way_bikeway") -> Path:
    from scripts.phase4_render_3d import find_blender, render_all

    model, state, pavement, context = design_for(features, scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry, png = out_dir / f"{name}_3d.json", out_dir / f"{name}_3d.png"
    export_scenario(model, state, name, geometry, pavement=pavement,
                    frame=window_frame(features),
                    buildings=context["buildings"], **context_layers(context))
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
    parser.add_argument("--scenario", default="two_way_bikeway", choices=sorted(SCENARIOS),
                        help="which design to draw (default: two_way_bikeway)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="2D sheet density; 600+ resolves a double yellow's 4 in gap")
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
    print(f"wrote {draw_2d(features, stem, args.out_dir, args.scenario, args.dpi)}")
    if args.three_d:
        print(f"wrote {draw_3d(features, stem, args.out_dir, args.scenario)}")


if __name__ == "__main__":
    main()
