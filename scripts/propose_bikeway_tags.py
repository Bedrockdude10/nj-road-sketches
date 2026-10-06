#!/usr/bin/env python
"""Write Broad St's two-way bikeway as OSM inline tags: proposals/<area>/two_way_bikeway.yaml.

    scripts/propose_bikeway_tags.py [--area hopewell_borough]

Runs the route's ladder (BROAD_ST_TWO_WAY_BIKEWAY) ONCE over the existing world and records, for
each OSM way an approach lies on, the section that approach landed on - in OSM's own schema:

    cycleway:<side>=track, cycleway:<side>:oneway=no, cycleway:<side>:width,
    cycleway:<side>:buffer, cycleway:<side>:separation:left=flex_post,
    parking:<side>:restriction=no_parking

and the far kerb's hatching as what OSM maps hatching as: a new closed way (negative id, OSM's
convention for an element not yet uploaded) tagged road_marking=restriction + pattern=chevron,
traced off the zone the ladder's design actually painted. The far kerb's PARKING is OSM's
street-parking schema on the way, and where it changes from one leg to the next the way is SPLIT
at the junction between them (wiki Street parking: "The roadway needs to be split up where any
of the properties changes"), written as osmChange writes a split.

`<side>` is the WAY's side, as OSM's always is. A way carrying several approaches gets the
narrowest section any of them took, because one way is one set of tags. From then on THE FILE IS
THE PROPOSAL: the two_way_bikeway scenario draws what it says (render_slice.SCENARIOS), so a width
is changed by editing the file, not by re-running this. Re-running overwrites it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from scripts.render_slice import load_network, slice_context
from src.geometry.network.slice_design import slice_design
from src.geometry.targets import LegSide, Side
from src.geometry.treatments import (BROAD_ST_TWO_WAY_BIKEWAY, LaneNarrowing, MarkedParking,
                                     existing_conditions)
from src.geometry.treatments.bikeways import AddBikeLaneBollards, AddTwoWayBikeLane
from src.sources.proposals import proposal_path


def _feet(value_ft: float) -> str:
    """OSM's feet notation, so the width the ladder chose is the width written: `8'`."""
    return f"{value_ft:g}'"


#: The kinds a hatched kerbside zone is painted with (LaneNarrowing.paint): its fill, its taper,
#: and the lines that bound them. Their union is the area a mapper would trace.
_HATCH_KINDS = ("lane_narrowing_fill", "taper_fill", "lane_edge_line", "taper_line",
                "zone_end_line")


def _hatched_areas(model, state, network, carrying: set[str], new_ids) -> list[dict]:
    """[{element, tags, coords, source}] - each far-kerb hatched zone the ladder's design PAINTS,
    as one new road_marking=restriction way. Traced off the drawn paint rather than rebuilt from
    the treatment's numbers, so the area is the zone as drawn - cut at its crossings and swept
    round its openings - and redrawing it from the way reproduces it."""
    import pyproj
    from shapely.ops import unary_union

    from src.geometry.model import NJ_STATE_PLANE_FT, WGS84
    from src.geometry.network.slice_design import slice_pavement
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT, MIN_ZONE_AREA_SQ_FT
    from src.render.props import build_props
    from src.render.scene import SceneGeometry

    hatched = {(zone.target.leg, str(side)) for zone in state.treatments_of(LaneNarrowing)
               if zone.target.leg in carrying and not zone.line_only for side in zone.sides}
    if not hatched:
        return []
    pavement = slice_pavement(network, state.corner_fillets, state.legs, model.osm)
    with contextlib.redirect_stdout(io.StringIO()):
        scene = SceneGeometry.resolve(model, state, pavement=pavement)
        props = build_props(model, state, scene.crosswalk_offsets, pavement=scene.pavement)
        paint, _ = scene.build_paint_and_posts(props)
    to_wgs84 = pyproj.Transformer.from_crs(NJ_STATE_PLANE_FT, WGS84, always_xy=True)
    out = []
    for leg_name, side in sorted(hatched):
        pieces = [p for p in paint if p.leg == leg_name and str(p.side) == side
                  and str(p.kind) in _HATCH_KINDS]
        shapes = [p.geometry if p.kind.covers_area
                  else p.geometry.buffer(LANE_EDGE_LINE_WIDTH_FT / 2, cap_style=2, join_style=2)
                  for p in pieces]
        if not shapes:
            continue
        # Closed by a hair and opened again, so the line laid along the fill's edge and the
        # fill itself read as one area rather than two that touch.
        union = unary_union(shapes).buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
        for part in getattr(union, "geoms", [union]):
            if part.geom_type != "Polygon" or part.area < MIN_ZONE_AREA_SQ_FT:
                continue
            xs, ys = zip(*part.exterior.coords)
            lons, lats = to_wgs84.transform(xs, ys)
            out.append({"element": f"way/{next(new_ids)}",
                        "tags": {"road_marking": "restriction", "pattern": "chevron",
                                 "colour": "white"},
                        "coords": [[round(lon, 8), round(lat, 8)] for lon, lat in zip(lons, lats)],
                        "source": f"Proposal: the hatched far kerb of Broad St's two-way "
                                  f"bikeway, as BROAD_ST_TWO_WAY_BIKEWAY's ladder painted it on "
                                  f"{leg_name} {side} (scripts/propose_bikeway_tags.py). "
                                  f"Not a survey."})
    return out


def _leg_span(leg, aligned: bool, node_ids: list[int]) -> tuple[int, int] | None:
    """(first, last) index into the way's node list that this leg covers. A leg whose far end is
    no junction (the area stops) runs on to the way's own end in its direction."""
    at = {node: i for i, node in enumerate(node_ids)}
    start, end = at.get(leg.start_node), at.get(leg.end_node)
    if start is None and end is None:
        return None
    if end is None:
        end = len(node_ids) - 1 if aligned else 0
    if start is None:
        start = 0 if aligned else len(node_ids) - 1
    return min(start, end), max(start, end)


def _pieces(node_ids: list[int], spans: list[tuple[int, int, object]]) -> list[tuple[list[int], object]]:
    """The way cut wherever the answer changes from one leg to the next - at the node the two
    legs share, which is a junction OSM already has. [(node ids, answer)], in way order; one
    piece, the whole way, where every leg agrees."""
    groups: list[list] = []
    for lo, hi, answer in sorted(spans, key=lambda span: span[0]):
        if groups and groups[-1][2] == answer:
            groups[-1][1] = max(groups[-1][1], hi)
        else:
            groups.append([lo, hi, answer])
    cuts = [0, *(group[0] for group in groups[1:]), len(node_ids) - 1]
    return [(node_ids[a:b + 1], group[2]) for (a, b), group in zip(itertools.pairwise(cuts), groups)]


def _to_the_inch(depth_ft: float) -> int:
    """A stall depth in whole inches, rounded DOWN so the stall is never deeper than the ladder
    sized it. Also what keeps two legs a hundredth of a foot apart from splitting a way - the
    wiki's "avoid over-fragmenting the street line"."""
    return int(depth_ft * 12 + 1e-9)


def _parking_tags(osm_side: str, depth_in: int | None) -> dict:
    """OSM's street-parking schema for a marked parallel lane `depth_in` inches deep (feet and
    inches notation, which osm_width_ft reads), or nothing."""
    if depth_in is None:
        return {}
    feet, inches = divmod(depth_in, 12)
    return {f"parking:{osm_side}": "lane", f"parking:{osm_side}:orientation": "parallel",
            f"parking:{osm_side}:markings": "yes",
            f"parking:{osm_side}:width": f"{feet}'{inches}\"" if inches else f"{feet}'"}


def proposed_tags(area: str) -> list[dict]:
    """[{element, tags, source, nodes?, coords?}]: the bikeway's tags on each way it runs along,
    the far kerb's parking on each stretch of way the ladder parked - the way split where that
    changes - and one new way per far-kerb hatched zone it painted."""
    network = load_network(area)
    osm = slice_context(network)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = slice_design(network, osm=osm, osm_area=area)
        state = BROAD_ST_TWO_WAY_BIKEWAY.apply_to(existing_conditions(model), model, quiet=True)
    posts = {(t.target.leg, str(t.target.side)) for t in state.treatments_of(AddBikeLaneBollards)}
    by_way: dict[int, dict] = {}
    for lane in state.treatments_of(AddTwoWayBikeLane):
        leg_name, side = lane.target.leg, str(lane.target.side)
        leg = model.legs[leg_name]
        aligned = model.leg_osm_aligned.get(leg_name, True)
        osm_side = side if aligned else str(Side(side).other)
        way = by_way.setdefault(leg.osm_way_id, {"sides": {}, "legs": []})
        entry = way["sides"].setdefault(osm_side, {"width_ft": lane.width_ft,
                                                   "buffer_ft": lane.buffer_ft, "posts": True})
        entry["width_ft"] = min(entry["width_ft"], lane.width_ft)
        entry["buffer_ft"] = min(entry["buffer_ft"], lane.buffer_ft)
        entry["posts"] = entry["posts"] and (leg_name, side) in posts
        # THE FAR KERB, per leg: the parking lane hold_travel_lane_at_target marked there, or
        # None where it hatched or left nothing. Kept per leg, because the way is split by it.
        far = str(Side(side).other)
        stalls = state.treatment_for(MarkedParking, LegSide(leg_name, far))
        depth_in = (_to_the_inch(stalls.depth_ft)
                    if stalls is not None and not stalls.curb_offset_ft else None)
        way["legs"].append((leg_name, aligned, far if aligned else side, depth_in))
    carrying = {lane.target.leg for lane in state.treatments_of(AddTwoWayBikeLane)}
    new_ids = itertools.count(-1, -1)
    out = [*_hatched_areas(model, state, network, carrying, new_ids)]
    roads = {road["id"]: road for road in osm["roads"]
             if len(road.get("node_ids") or []) == len(road.get("coords_wgs84") or [])}
    splits = 0
    for way_id, way in sorted(by_way.items()):
        bikeway = {}
        for osm_side, entry in sorted(way["sides"].items()):
            key = f"cycleway:{osm_side}"
            bikeway.update({key: "track", f"{key}:oneway": "no",
                            f"{key}:width": _feet(entry["width_ft"]),
                            f"{key}:buffer": _feet(entry["buffer_ft"]),
                            f"parking:{osm_side}:restriction": "no_parking"})
            if entry["posts"]:
                bikeway[f"{key}:separation:left"] = "flex_post"
        legs = ", ".join(sorted(name for name, *_ in way["legs"]))
        source = (f"Proposal: Broad St two-way protected bikeway, the rung "
                  f"BROAD_ST_TWO_WAY_BIKEWAY's ladder chose on {legs}, and the far kerb's parking "
                  f"lane where it marked one (scripts/propose_bikeway_tags.py). Not a survey.")
        road = roads[way_id]
        spans = []
        for leg_name, aligned, osm_far, depth_in in way["legs"]:
            span = _leg_span(model.legs[leg_name], aligned, road["node_ids"])
            if span is not None:
                spans.append((*span, (osm_far, depth_in)))
        pieces = _pieces(road["node_ids"], spans) if spans else [(road["node_ids"], None)]
        for i, (nodes, answer) in enumerate(pieces):
            parking = _parking_tags(*answer) if answer is not None else {}
            if i == 0:
                entry = {"element": f"way/{way_id}", "tags": {**bikeway, **parking},
                         "source": source}
                if len(pieces) > 1:
                    entry["nodes"] = nodes        # restated, shortened: osmChange's split
            else:
                # A CREATED WAY CARRIES ITS WHOLE TAG SET, as in osmChange: OSM's own tags for
                # the street, then what the proposal adds.
                entry = {"element": f"way/{next(new_ids)}", "nodes": nodes,
                         "tags": {**roads[way_id]["tags"], **bikeway, **parking},
                         "source": f"{source} Split from way/{way_id} where the far kerb's "
                                   f"parking changes (wiki: Street parking)."}
                splits += 1
            out.append(entry)
    if splits:
        print(f"  split {splits} piece(s) off their ways where the far kerb's parking changes.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    args = parser.parse_args()
    entries = proposed_tags(args.area)
    path = proposal_path(args.area, "two_way_bikeway")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Broad St's two-way protected bikeway, AS OSM TAGS on the ways it runs along.\n"
              "# Written by scripts/propose_bikeway_tags.py from the route's ladder; from then on\n"
              "# this file IS the proposal - edit a width here, not in code. Sides are each way's\n"
              "# own (OSM's convention); a negative id is a way the proposal adds.\n"
              "# Schema: src/sources/proposals.py.\n")
    path.write_text(header + yaml.safe_dump({"observations": entries}, sort_keys=False,
                                            allow_unicode=True, width=100))
    print(f"wrote {path} ({len(entries)} element(s))")


if __name__ == "__main__":
    main()
