"""A design, written as the osmChange that would make OSM say it.

The route ladders (BROAD_ST_TWO_WAY_BIKEWAY) DECIDE a section per approach; this states each
decision in OSM's vocabulary - cycleway and parking tags on the ways, the way split where a kerb's
parking changes, a road_marking=restriction area for each hatched kerb - built from the design's
own numbers and never from paint. From then on the file is the proposal and the drawing reads it
like any other OSM (src/sources/osm_change.py).
"""
from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import pyproj
from shapely.geometry import Polygon

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84
from src.geometry.model.approach import End, approach_id, leg_side, leg_station
from src.geometry.model.corners import junction_mouth_ft
from src.geometry.paint.datum import Along, Centre, Kerb, Narrowest, place
from src.geometry.targets import LegSide, Side
from src.sources.osm_change import NewNode, OsmChange, WayChange

if TYPE_CHECKING:
    from src.geometry.model import Leg
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.state import DesignState


def _feet(value_ft: float) -> str:
    """OSM's feet notation, so the width the ladder chose is the width written: `8'`."""
    return f"{value_ft:g}'"


def _parking_tags(osm_side: str, depth_in: int | None) -> dict:
    """OSM's street-parking schema for a marked parallel lane `depth_in` inches deep (feet and
    inches notation, which osm_width_ft reads), or nothing."""
    if depth_in is None:
        return {}
    feet, inches = divmod(depth_in, 12)
    return {f"parking:{osm_side}": "lane", f"parking:{osm_side}:orientation": "parallel",
            f"parking:{osm_side}:markings": "yes",
            f"parking:{osm_side}:width": f"{feet}'{inches}\"" if inches else f"{feet}'"}


def to_the_inch(depth_ft: float) -> int:
    """A stall depth in whole inches, rounded DOWN so the stall is never deeper than sized."""
    return int(depth_ft * 12 + 1e-9)


def leg_span(leg: Leg, aligned: bool, node_ids: Sequence[int]) -> tuple[int, int] | None:
    """(first, last) index into the way's node list that this leg covers."""
    at = {node: i for i, node in enumerate(node_ids)}
    start, end = at.get(leg.start_node), at.get(leg.end_node)
    if start is None and end is None:
        return None
    if end is None:
        end = len(node_ids) - 1 if aligned else 0
    if start is None:
        start = 0 if aligned else len(node_ids) - 1
    return min(start, end), max(start, end)


def way_pieces(node_ids: Sequence[int], spans: Sequence[tuple[int, int, Hashable]]
               ) -> list[tuple[list[int], Hashable]]:
    """The way cut wherever the answer changes from one leg to the next."""
    groups: list[list] = []
    for lo, hi, answer in sorted(spans, key=lambda span: span[0]):
        if groups and groups[-1][2] == answer:
            groups[-1][1] = max(groups[-1][1], hi)
        else:
            groups.append([lo, hi, answer])
    cuts = [0, *(group[0] for group in groups[1:]), len(node_ids) - 1]
    return [(list(node_ids[a:b + 1]), group[2])
            for (a, b), group in zip(itertools.pairwise(cuts), groups, strict=True)]


def hatched_zone_ft(state: DesignState, leg_name: str, side: str,
                    zone: LaneNarrowing) -> Polygon | None:
    """The hatched kerbside zone `zone` states on this kerb, between the junction mouths."""
    if zone.line_only or Side(side) not in zone.sides:
        return None
    leg = state.legs[leg_name]
    inner = (Centre(zone.lane_edge_ft) if zone.lane_edge_ft is not None
             else Narrowest(zone.stripe_width_ft))
    near = junction_mouth_ft(leg_name, side, state.legs, state.corner_fillets)
    start_ft = near[1] if near else 0.0
    if zone.end_ft is not None:
        end_ft = zone.end_ft
    else:
        far = (junction_mouth_ft(approach_id(leg_name, End.END), leg_side(End.END, side),
                                 state.legs, state.corner_fillets)
               if leg.end_node is not None else None)
        end_ft = leg_station(leg, End.END, far[1]) if far else leg.centerline.length
    if end_ft - start_ft <= 0:
        return None
    geometry = place(leg, side, Along((start_ft, end_ft), Kerb(0), inner)).geometry
    if geometry is None or geometry.is_empty:
        return None
    parts = [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type == "Polygon"]
    return max(parts, key=lambda p: p.area) if parts else None


def proposal_from_design(model, state: DesignState, osm: dict[str, list[dict]],
                         note: str) -> OsmChange:
    """The osmChange that makes OSM carry `state`'s bikeway, far-kerb parking and hatching."""
    from src.geometry.treatments.bikeways.bollards import AddBikeLaneBollards
    from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.parking import MarkedParking

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
        depth_in = (to_the_inch(stalls.depth_ft)
                    if stalls is not None and not stalls.curb_offset_ft else None)
        way["legs"].append((leg_name, aligned, far if aligned else side, depth_in))
    carrying = {lane.target.leg for lane in state.treatments_of(AddTwoWayBikeLane)}
    new_ids = itertools.count(-1, -1)

    # HATCHED AREAS FIRST
    to_wgs84 = pyproj.Transformer.from_crs(NJ_STATE_PLANE_FT, WGS84, always_xy=True)
    nodes, ways = [], []
    zones = sorted(((z.target.leg, str(s), z) for z in state.treatments_of(LaneNarrowing)
                    if z.target.leg in carrying for s in z.sides), key=lambda t: (t[0], t[1]))
    for leg_name, side, zone in zones:
        polygon = hatched_zone_ft(state, leg_name, side, zone)
        if polygon is None:
            continue
        xs, ys = zip(*list(polygon.exterior.coords)[:-1])
        lons, lats = to_wgs84.transform(xs, ys)
        ids = []
        for lon, lat in zip(lons, lats):
            node = NewNode(next(new_ids), round(lon, 7), round(lat, 7), {})
            nodes.append(node)
            ids.append(node.id)
        ways.append(WayChange(next(new_ids), "create", (*ids, ids[0]),
                              {"road_marking": "restriction", "pattern": "chevron",
                               "colour": "white",
                               "note": f"{note} Far-kerb hatching on {leg_name} {side}."}))

    # THEN the ways
    roads = {road["id"]: road for road in osm["roads"]
             if len(road.get("node_ids") or []) == len(road.get("coords_wgs84") or [])}
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
        road = roads[way_id]
        legs = ", ".join(sorted(name for name, *_ in way["legs"]))
        existing = road["tags"].get("note")
        merged = f"{existing}; {note} Legs: {legs}." if existing else f"{note} Legs: {legs}."
        spans = []
        for leg_name, aligned, osm_far, depth_in in way["legs"]:
            span = leg_span(model.legs[leg_name], aligned, road["node_ids"])
            if span is not None:
                spans.append((*span, (osm_far, depth_in)))
        pieces = way_pieces(road["node_ids"], spans) if spans else [(list(road["node_ids"]), None)]
        split = (f" Split from way/{way_id} where the far kerb's parking changes"
                 f" (wiki: Street parking).")
        # The first piece keeps the way's id; every other piece is a new way with its full tags.
        for i, (piece, answer) in enumerate(pieces):
            tags = {**road["tags"], **bikeway, **(_parking_tags(*answer) if answer else {}),
                    "note": merged if i == 0 else merged + split}
            ways.append(WayChange(way_id, "modify", tuple(piece), tags) if i == 0
                        else WayChange(next(new_ids), "create", tuple(piece), tags))
    # Order is osm_change's to keep: load, write and apply all put a change in canonical order.
    return OsmChange(tuple(nodes), tuple(ways))

