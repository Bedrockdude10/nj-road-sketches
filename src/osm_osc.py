"""The .osc files: OSM edits saying what the tags do not yet say, and what is proposed.

The reader (src/osm_world.py) draws tags and decides nothing; this module writes them, as an
osmChange to read in JOSM, fix by hand and upload where it describes the street as it is. Nothing
here measures a radius from a junction: a junction is a node where three or more legs of two or
more streets meet, and a leg is the run of ways out of it, node by node, to the next junction or a
dead end. Two relationships OSM does not record are read off its geometry, each by one rule:
  - which street a kerb bounds. `barrier=kerb` ways are not joined to roads, so each kerb segment
    is the edge of the street whose centreline is nearest it.
  - which junction a crosswalk belongs to. A leg's first crossing out of its junction is that
    junction's - unless it is the only crossing on a leg between two junctions, when it is the
    nearer junction's along the leg. (In Hopewell every such crossing is 17-40 ft from one end
    and 120 ft or more from the other: none is mid-block.)

`existing_markings(area)` - what OSM's tags do not yet say about the street as it is:
  - each street's `width`, where OSM has none, one per block (the stretch between two
    junctions): the median kerb to kerb over the block's cross-sections that meet this street's
    kerb on both sides (a cross-section's first kerb each way, if it is this street's). Ways are
    split at junctions so each piece is one block's, and a width steps only at a junction. The
    reader lays every marking from that width about the way, never from a kerb.
  - the street as built: `area:highway` polygons named like the street, each cross-section out to
    its own kerbs - the kerb line carried straight across each gap in it and straight on past its
    ends - and to half its block's width from the way on a side with no kerb at all; and each
    junction's area, the face around its node enclosed by the mapped kerbs and each leg's first
    cross-section that meets its own kerbs. This is the concrete; the paint stays on it.
  - where lane markings are not painted - centre line or lane lines: the way split and the stretch tagged
    `lane_markings=no`. Across every marked crossing, its width; and on each leg of each junction,
    from the junction node out to the further of where the junction ends (_mouths: its crosswalk's
    outer edge, else the cross street's kerb line carried straight across the mouth, else half the
    cross street's width) and its stop line (the one across the lane approaching the junction:
    traffic keeps right) - so no piece inside a junction, whose kerbside lanes are gone, has a lane
    marking to lay.
  - where the law forbids standing: within NO_STANDING_FT of an intersection (N.J.S.A. 39:4-138),
    on both kerbs of each leg, out from the further of the statute's two arms: the far side line
    of the junction's crosswalk on it (any mapped crossing, marked or not), and the cross street's
    side line (its kerb line as above, else half its `width`). The way is split there and tagged
    `parking:<side>=no` + `parking:<side>:restriction=no_standing` +
    `parking:<side>:restriction:reason=junction` (wiki Street parking), which the proposal reads
    like any other restriction - NO_STANDING_FT on every kerb, a curb extension or not: the
    statute's shorter zone beside one is not taken, so no car stands near the corner. With
    `kerb_extensions` - the daylighting proposal's base - every crosswalk a junction claims is
    given `crossing:kerb_extension=both` first.

`two_way_bikeway(area, base)` - `base` plus the proposal, applied universally: one design table
over every street way in the world, every rule a vector operation over all of them (_design).
  - the road diet, everywhere a centre line is painted and the width holds it: two LANE_FT lanes,
    and each kerb's spare, lane outward, a parking lane (`parking:<side>=lane`, up to
    PARKING_LANE_FT) against the travel lane where OSM does not restrict the kerb and a stall
    fits, then the rest to the edge a hatched shoulder (`shoulder:<side>=yes`, its `:width`, and
    `shoulder:<side>:markings=hatched`, a project convention), which the reader hatches out to
    the kerb as built. `width:lanes` is what lays the parking against the lane - see
    src/osm_world.py:_Reader.cross_section. Each parked piece is marked in as many STALL_LENGTH_M
    stalls as fit (`parking:<side>:capacity`); where traffic crosses a parked kerb - a
    `kerb=lowered` kerb, or a driveway or parking aisle leaving from it - the way is split and that piece is not parked, nor is one
    too short for a stall.
  - every marked crosswalk, and every crossing a junction claims, continental
    (`crossing:markings=zebra`).
  - on the ways the proposal's file chooses (proposals/<area>/<name>.toml, Proposal - the only
    place a street is named) its track tags, on its compass kerb: a two-way protected track, TRACK_FT
    (CONSTRAINED_TRACK_FT where only that fits; the road diet alone where neither does), a
    BUFFER_FT buffer with flex posts, both lanes still LANE_FT, all the spare to the other kerb,
    and a 0 ft hatched shoulder outside the track. Where traffic crosses the track the way is
    split and that piece tagged with no posts and dotted green
    (`cycleway:<side>:crossing:markings=dashes`, a project convention: OSM has no tag for
    skip-paint). The track runs to each junction's crosswalk, and across the junction it is a
    `highway=cycleway` + `cycleway=crossing` way (wiki Tag:cycleway=crossing), straight from its
    end on one leg to its end on the next (_track_ends).
  - inside every junction - from its node out to each leg's crosswalk's outer edge, the furthest
    it reaches across the street, else the cross street's kerb line (_mouths) - no kerbside
    markings: the street's section starts past the crosswalk, however skewed the crosswalk is.
  - restriping moves every centre line, so every stop line on a restriped street is laid across
    its approach lane only, from the new centre line to that lane's edge.
  - with `kerb_extensions` (the daylighting proposal), on a base whose crosswalks carry
    `crossing:kerb_extension=both`: each extension a `road_marking=restriction` +
    `pattern=solid` + `colour=tan` area (NYC's painted curb extension), one per corner: the drawn
    street outside the drawn travel lanes, its corner rounded to the EFFECTIVE_RADIUS_M arc on their
    edges (NJDOT 6.4.3), and off every bikeway and its buffer, out along each leg to
    its no-standing zone's end - the zone's hatching, painted as the extension, so the zone's
    shoulder is unmarked, `shoulder:<side>:markings=no` (_section_tags) - with
    `barrier=bollard` + `bollard=flexible` nodes down its middle (_extensions); the crosswalk is
    left as mapped, and the reader paints the extension over it. The kerbs are re-signed: a
    restriction OSM records with no `:restriction:reason` - the blanket corner sign the extension
    replaces - is lifted (_may_park), so the statute's zone is the limit.
  - where the track stops - a junction where no other leg of the route carries it - a two-stage
    turn box over its first TURN_BOX_LENGTH_FT out from the crosswalk (_track_ends, MUTCD 9E.11).

`retagged(area, base, proposal)` - a `[retag]` proposal file: `base` (existing conditions) with its
tags on every street way it chooses, and nothing else.
"""
from __future__ import annotations

import bisect
import dataclasses
import functools
import itertools
import math
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import shapely.ops
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from src.osm_world import (BOLLARD_SPACING_M, PROHIBITIONS, FT_TO_M, Leg, LocalFrame,
                           SEAM_M, STATION_STEP_M, TANGENT_M, Stations, StreetGraph, carriageway_width_m,
                           crossing_markings, ease, has_centre_line, has_lane_lines, heading, is_corner, is_street,
                           kerb_section, lane_count, lane_widths, line_parts, meets, read, rides_along, sided, street_key,
                           tagged_width, tangent, taper_rate, vertex_stations, width_m, width_or_default)
from src.sources.osm_change import PROPOSALS_DIR, NewNode, OsmChange, WayChange, write_change
from src.sources.osm_context import PAINT_LAYERS, SNAPSHOT_AREAS, fetch_borough_osm, osm_layers
from src.standards import in_unit, si
from src.turning import EFFECTIVE_RADIUS_M, Edge, arc, centre, effective_radius

# Every figure is declared in standards.toml; these are in feet, the unit the section is designed in.
NODE_MATCH_M, ACROSS_M = si("numerical.node_match"), si("numerical.across")
OVERSHOOT_M = si("numerical.overshoot")
OSM_GRID_M = si("numerical.osm_grid")

TRACK_FT, CONSTRAINED_TRACK_FT = in_unit("track.width", "ft"), in_unit("track.constrained_width", "ft")
BUFFER_FT, LANE_FT = in_unit("track.buffer", "ft"), in_unit("lane.width", "ft")
MIN_STALL_FT = in_unit("parking.min_stall", "ft")          # spare narrower than this is hatched, not parked
PARKING_LANE_FT = in_unit("parking.lane_depth", "ft")
MIN_HATCH_FT = in_unit("hatch.min_width", "ft")            # spare narrower than this is left unmarked
STALL_LENGTH_M = si("stall.parallel_length")             # a piece shorter than one stall is not parked
DRIVEWAY_CLEARANCE_FT = in_unit("parking.driveway_clearance", "ft")
NO_STANDING_FT = in_unit("no_standing.from_crosswalk", "ft")

AREA_NOTE = ("Existing conditions: the street's surface out to its mapped kerbs (each kerb the edge "
             "of the street nearest it), and half its block's width where a side has none, by "
             "src/osm_osc.py. Check before upload.")
EXISTING_NOTE = ("Existing conditions: `width` measured between the mapped kerbs, and where the "
                 "centre line is not painted, from the mapped crossings, stop lines and kerbs at each "
                 "junction, by src/osm_osc.py. Check on the ground or imagery before upload.")
def no_standing_note(feet: float) -> str:
    return (f"No standing within {feet:g} ft of the nearest crosswalk or the intersecting street's "
            "side line, R.S. 39:4-138(e) - the law, not a sign.")


TURN_BOX_NOTE = ("Proposal: a two-stage turn box where the two-way track ends (MUTCD 9E.11), so a "
                 "rider can cross to or from the far side of the street (src/osm_osc.py). "
                 "`cycleway=two_stage_box` is a value from the 2020 draft Proposal:Two-stage bicycle "
                 "turn, not an approved tag.")
EXTENSION_NOTE = ("Proposal: a painted curb extension at the corner, tan as NYC paints them and edged "
                  "in flexible posts, so the crosswalk is shorter, over the corner's "
                  f"{NO_STANDING_FT:g} ft no-standing zone, R.S. 39:4-138(e) (src/osm_osc.py).")
RECENTRED_NOTE = ("Existing conditions: the way moved to the middle of its mapped kerbs (wiki "
                  "Key:placement), by src/osm_osc.py. Check before upload.")
COMPASS = {"north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0)}


@dataclasses.dataclass(frozen=True)
class Proposal:
    """proposals/<area>/<name>.toml: the tags a proposal puts on the ways it chooses - those whose
    OSM tags match `ways` (key -> the values it may have). Under `[track]` every street way chosen
    takes `tags` on its `kerb` (a compass side; `{side}` in a key is each way's own left or right
    there), fitted to the kerbs by _design. Under each `[[retag]]` (`retag`, its rules) every street
    or crossing way chosen takes `tags` as they are, on existing conditions (retagged), a value in
    braces a standards.toml figure written in feet. Nothing in src names a street: the file does."""
    name: str
    note: str
    kerb_extensions: bool
    ways: dict[str, tuple[str, ...]]
    kerb: str
    tags: dict[str, str]
    retag: tuple[tuple[dict[str, tuple[str, ...]], dict[str, str]], ...] = ()

    @classmethod
    def load(cls, path: Path) -> Proposal:
        raw = tomllib.loads(path.read_text())
        section = "retag" if "retag" in raw else "track"
        rules = raw["retag"] if section == "retag" else [raw.get("track", {})]
        rules = rules if isinstance(rules, list) else [rules]
        problems = [f"unknown key {k!r}" for k in raw if k not in ("note", "kerb_extensions", section)]
        for chosen in rules:
            problems += [f"{section}: unknown key {k!r}" for k in chosen
                         if k not in ("ways", "tags", *(("kerb",) if section == "track" else ()))]
            if not chosen.get("ways"):
                problems.append(f"{section}.ways chooses nothing")
        if section == "track" and rules[0].get("kerb") not in COMPASS:
            problems.append(f"track.kerb {rules[0].get('kerb')!r} is not one of {sorted(COMPASS)}")
        if not raw.get("note"):
            problems.append("no note: every edit a proposal writes says what it is")
        if problems:
            raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
        chosen = [({k: tuple([v] if isinstance(v, str) else v) for k, v in rule["ways"].items()},
                   dict(rule.get("tags", {}))) for rule in rules]
        if section == "track":
            return cls(path.stem, raw["note"], bool(raw.get("kerb_extensions", False)),
                       chosen[0][0], rules[0]["kerb"], chosen[0][1])
        return cls(path.stem, raw["note"], bool(raw.get("kerb_extensions", False)), {}, "", {},
                   tuple((ways, {k: _ft(in_unit(v[1:-1], "ft")) if v.startswith("{") else v
                                 for k, v in tags.items()}) for ways, tags in chosen))

    def chooses(self, tags: dict, ways: dict[str, tuple[str, ...]] | None = None) -> bool:
        return all(tags.get(key) in values for key, values in (ways or self.ways).items())

    def track_tags(self, side: str) -> dict[str, str]:
        return {key.format(side=side): value for key, value in self.tags.items()}


# --- where a junction ends, along each of its legs (the graph is src/osm_world.py:StreetGraph) ---

def _claimed(leg: Leg, crossing_nodes, stops: _Stops) -> int | None:
    """The crossing node whose crosswalk ends this leg's centre line at its junction: the
    first out along the leg - unless it is the only one on a leg between two junctions and
    nearer the other, whose crosswalk it then is. Never one beyond the stop line serving the
    approach: a stop line is placed in advance of its junction's crosswalk (MUTCD 3B.16, as
    cited), so a crossing past it is mid-block, dark only across its own band."""
    stop_at = _stop_line(leg, stops)
    on_leg = list(dict.fromkeys(
        n for k, n in enumerate(leg.nodes) if k > 0 and n in crossing_nodes and n != leg.end
        and (stop_at is None or float(leg.d[k]) < stop_at)))
    if not on_leg:
        return None
    if leg.end is not None and len(on_leg) == 1:
        out = float(leg.d[leg.nodes.index(on_leg[0])])
        return on_leg[0] if out < leg.length - out else None
    return on_leg[0]


def _stop_line(leg: Leg, stops: _Stops) -> float | None:
    """How far out along `leg` its junction's stop line is: the first that lies across the lane
    approaching the junction - right of that traffic, which keeps right. A stop line joined to
    the leg at a node stands at that node; one joined to no street (wiki Key:road_marking
    allows a separate way) stands where it lies along the leg of its nearest street. A oneway
    leg whose traffic only leaves the junction has none: nothing approaches it there."""
    if leg.outbound:
        return None
    found = []
    for k in range(1, len(leg.steps)):
        if leg.nodes[k] == leg.end:
            break
        back = np.flatnonzero(leg.d[:k] < leg.d[k])
        if not len(back):
            continue
        toward = leg.xy[back[-1]] - leg.xy[k]
        right = np.array([toward[1], -toward[0]])
        for stop in stops.get(leg.nodes[k], []):
            far = max(stop.coords, key=lambda c: float(np.linalg.norm(np.asarray(c) - leg.xy[k])))
            if float(np.dot(np.asarray(far) - leg.xy[k], right)) > 0:
                found.append(float(leg.d[k]))
                break
        if found:
            break
    ways = {way_id for way_id, _i in leg.steps}
    line = leg.line()
    for stop, way_id in getattr(stops, "floating", []):
        if way_id not in ways or line.length == 0:
            continue
        middle = stop.interpolate(0.5, normalized=True)
        d = line.project(middle)
        if not 0 < d < line.length:
            continue
        here = np.asarray(line.interpolate(d).coords[0])
        toward = np.asarray(line.interpolate(max(d - TANGENT_M, 0.0)).coords[0]) - here
        if float(np.dot(np.asarray(middle.coords[0]) - here, [toward[1], -toward[0]])) > 0:
            found.append(float(d))
    return min(found, default=None)


def _kerb_line(network: StreetGraph, leg: Leg, kerbs: _Kerbs) -> float | None:
    """How far out along `leg` the cross street's kerb line, carried straight across the
    mouth, crosses it. On each side of the leg the next leg round the junction, where it is
    another street's; that street's kerb in the corner between the two, at its least distance
    from that street's centreline (carried back across the junction along its first segment);
    and where the leg leaves that distance - where the corner has none of its kerb mapped, half
    its width, which is where its edge is drawn. The further side; None where neither side of
    the leg is another street's."""
    ordered = network.around(leg.junction)
    if leg not in ordered:
        return None
    here = leg.xy[0]
    i = ordered.index(leg)
    own = heading(leg.direction())
    line = leg.line()
    found = []
    for neighbour, ccw in ((ordered[(i + 1) % len(ordered)], True), (ordered[i - 1], False)):
        if neighbour is leg or neighbour.street == leg.street:
            continue
        theirs = heading(neighbour.direction())
        start, span = (own, (theirs - own) % (2 * math.pi)) if ccw else (
            theirs, (own - theirs) % (2 * math.pi))
        cross = LineString([here - neighbour.direction() * leg.length,
                            *neighbour.line().coords])
        half = kerbs.least_offset(neighbour.street, cross, leg.length + neighbour.length,
                                  here, start, span, line)
        if half is None:
            # No kerb of it mapped in the corner: its own surface as drawn, half its width
            # either side of it - not carried across the junction, which on a street meeting
            # this one at a shallow angle would reach far down this one.
            half = _half_width(network, neighbour)
            edge = neighbour.line().buffer(half, cap_style="flat")
        else:
            edge = cross.buffer(half)
        inside = [LineString(p) for p in line_parts(line.intersection(edge))]
        for piece in inside:
            at = sorted((line.project(Point(piece.coords[0])), line.project(Point(piece.coords[-1]))))
            # Where the kerb line crosses the leg: one that covers the whole leg crosses nothing.
            if at[0] <= NODE_MATCH_M and at[1] < line.length - NODE_MATCH_M:
                found.append(at[1])
    return max(found) if found else None


def _side_line(network: StreetGraph, leg: Leg) -> float:
    """Where the cross street's side line is by its tags, where no kerb of it is mapped: half
    the widest carriageway among the junction's other streets' first ways."""
    return max((_half_width(network, other) for other in network.legs(leg.junction)
                if other.street != leg.street), default=0.0)


def _half_width(network: StreetGraph, leg: Leg) -> float:
    """Half a leg's carriageway, by its first way's tags: where its edge is drawn."""
    return carriageway_width_m(network.streets[leg.steps[0][0]]["tags"])[0] / 2


def _lines_xy(geometry) -> list[tuple[float, float]]:
    """Every vertex of a (multi)line, or of nothing."""
    return [xy for part in getattr(geometry, "geoms", [geometry]) if not part.is_empty
            for xy in part.coords]


# --- the street's kerbs ------------------------------------------------------------------------

class _Kerbs:
    """The mapped `barrier=kerb` ways, in segments, each the edge of the street whose centreline is
    nearest it (OSM does not say which street a kerb bounds), and which are `kerb=lowered` or
    `flush` and no crossing's ramp - a dropped kerb, where traffic may cross it."""

    def __init__(self, layers: dict, frame: LocalFrame, network: StreetGraph,
                 bbox: tuple[float, float, float, float]):
        segments, lowered = [], []
        for way in layers["kerbs"]:
            line = frame.line(way.get("coords_wgs84") or [])
            if line is None:
                continue
            # A dropped kerb traffic crosses - not a crossing's ramp: one a footway crossing meets
            # carries pedestrians, and its crosswalk's band already cuts the paint there.
            low = (way["tags"].get("kerb") in ("lowered", "flush")
                   and not meets(way, line, layers["crossings"], frame))
            for a, b in itertools.pairwise(line.coords):
                if a != b:
                    segments.append(LineString([a, b]))
                    lowered.append(low)
        self.segments = np.array(segments, dtype=object)
        self.lowered = np.array(lowered, dtype=bool)
        self.tree = STRtree(self.segments)
        ways = list(network.streets)
        streets = STRtree([network.lines[w] for w in ways])
        middles = shapely.line_interpolate_point(self.segments, 0.5, normalized=True)
        nearest = streets.query_nearest(middles, all_matches=False)
        self.owner = np.empty(len(segments), dtype=object)
        self.owner[nearest[0]] = [street_key(network.streets[ways[k]]) for k in nearest[1]]
        by_street: dict[str, list[LineString]] = defaultdict(list)
        for w in ways:
            by_street[street_key(network.streets[w])].append(network.lines[w])
        self.street_lines = {key: shapely.union_all(lines) for key, lines in by_street.items()}
        for geometry in self.street_lines.values():
            shapely.prepare(geometry)
        self.ends = np.array([c for s in segments for c in s.coords])
        self.end_owner = np.repeat(self.owner, 2)
        # A cross-section is cast across the whole area: the first kerb it meets is its edge.
        west, south, east, north = bbox
        self.ray_m = float(np.linalg.norm(np.subtract(frame.point(east, north),
                                                      frame.point(west, south))))

    def hits(self, street: str, stations: Stations, sign: int) -> np.ndarray:
        """Each cross-section's distance on one side to the first kerb it meets, where that kerb is
        `street`'s AND across from this cross-section - no part of the street nearer the kerb
        than the cross-section is (ACROSS_M, numerical only). A kerb of the street's that the
        ray reaches only past a bend, beside another stretch of it, is not this one's edge: North
        Elm's ray from one station met its own kerb 1,147 ft away, and its surface was drawn out
        to it. NaN where it is another street's, not across, or there is none."""
        offsets = np.full(len(stations.s), np.nan)
        for i, (point, normal) in enumerate(zip(stations.xy, stations.left, strict=True)):
            ray = LineString([point, point + sign * normal * self.ray_m])
            met = self.tree.query(ray, predicate="intersects")
            if not len(met):
                continue
            far = shapely.distance(Point(point), shapely.intersection(ray, self.segments[met]))
            first = met[int(np.argmin(far))]
            hit = Point(point + sign * normal * float(far.min()))
            if (self.owner[first] == street
                    and far.min() - hit.distance(self.street_lines[street]) <= ACROSS_M):
                offsets[i] = float(far.min())
        return offsets

    def band_reach(self, leg: Leg, band, sign: int) -> float | None:
        """How far out along `leg` a crosswalk's `band` reaches where it meets the leg's street's
        own kerb on one side (`sign` +1 is left, looking out from the junction): the furthest
        point of their meeting. None where that kerb is not mapped through the band."""
        line, found = leg.line(), []
        for k in self.tree.query(band, predicate="intersects"):
            if self.owner[k] != leg.street:
                continue
            for x, y in _lines_xy(self.segments[k].intersection(band)):
                at, offset = _station(line, (x, y))
                if offset * sign > 0:
                    found.append(at)
        return max(found) if found else None

    def least_offset(self, street: str, centre: LineString, along_m: float, apex: np.ndarray,
                     start: float, span: float, leg: LineString) -> float | None:
        """The least distance from `centre` of `street`'s kerb vertices in the corner: the wedge
        `span` radians anticlockwise from `start` about `apex`, along `centre`'s first `along_m`,
        and short of `leg`'s end - the corner between two legs ends where they do. Only a kerb
        that is the street's edge there: the first kerb out from its centreline, with no other
        between them (a parking lot's kerb behind the street's own is not its edge)."""
        mine = self.end_owner == street
        if not mine.any():
            return None
        ends = self.ends[mine]
        turn = (np.arctan2(ends[:, 1] - apex[1], ends[:, 0] - apex[0]) - start) % (2 * math.pi)
        points = shapely.points(ends)
        on = shapely.line_locate_point(centre, points)         # past either end of it: no corner of it
        within = ((turn > 0) & (turn < span) & (on > 0) & (on < min(along_m, centre.length))
                  & (shapely.line_locate_point(leg, points) < leg.length))
        if not within.any():
            return None
        found = []
        for k in np.flatnonzero(mine)[within]:
            point = Point(self.ends[k])
            foot = centre.interpolate(centre.project(point))
            ray = LineString([foot, point])
            blocked = any(other != k // 2 and self.segments[other].distance(point) > ACROSS_M
                          for other in self.tree.query(ray, predicate="intersects"))
            if not blocked:
                found.append(ray.length)
        return min(found) if found else None


# --- small helpers -----------------------------------------------------------------------------

def _has_centre_line(tags: dict) -> bool:
    """A street whose centre line the reader paints (src/osm_world.py:has_centre_line), before
    `lane_markings=no` says where it does not."""
    return is_street(tags) and has_centre_line(
        {k: v for k, v in tags.items() if (k, v) != ("lane_markings", "no")})


def _has_lane_markings(tags: dict) -> bool:
    """A street whose lane markings - its centre line or its lane lines - the reader paints,
    before `lane_markings=no` says where it does not."""
    unmarked = {k: v for k, v in tags.items() if (k, v) != ("lane_markings", "no")}
    return is_street(tags) and (has_centre_line(unmarked) or has_lane_lines(unmarked))


def _marked(tags: dict) -> bool:
    return crossing_markings(tags) not in (None, "no")


def _half(crossing: dict) -> float:
    """Half a crossing's `width`, else of the default the reader draws it at."""
    return width_or_default(crossing["tags"], "width", "crossing")[0] / 2


def _with_note(tags: dict, note: str) -> dict:
    existing = tags.get("note")
    return {**tags, "note": f"{existing}; {note}" if existing and note not in existing else note}


def _ids_below(change: OsmChange | None) -> Iterator[int]:
    used = [0] + ([n.id for n in change.nodes] + [w.id for w in change.ways] if change else [])
    return itertools.count(min(used) - 1, -1)


def _ft(value_ft: float) -> str:
    feet, inches = divmod(int(round(value_ft * 12)), 12)
    return f"{feet}'{inches}\"" if inches else f"{feet}'"


def _merged(intervals: list[tuple[float, float]], length: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((max(0.0, a), min(length, b)) for a, b in intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _split(line: LineString, node_ids, node_st, cuts: list[float], ids: Iterator[int],
           frame: LocalFrame, nodes: list[NewNode]) -> tuple[list[int], list[float], list[int]]:
    """A way's nodes and their stations with a node at each station in `cuts` - the one already
    there, else a new one - and the index of each cut's node."""
    node_ids, node_st = list(node_ids), list(node_st)
    for cut in sorted(cuts):
        i = bisect.bisect_left(node_st, cut - NODE_MATCH_M)
        if i < len(node_st) and abs(node_st[i] - cut) <= NODE_MATCH_M:
            continue
        point = line.interpolate(cut)
        new = NewNode(next(ids), *frame.wgs84(point.x, point.y), {})
        nodes.append(new)
        node_ids.insert(i, new.id)
        node_st.insert(i, cut)
    at = sorted({bisect.bisect_left(node_st, cut - NODE_MATCH_M) for cut in cuts})
    return node_ids, node_st, at


def _carried(s: np.ndarray, offsets: np.ndarray, otherwise: np.ndarray) -> np.ndarray:
    """A kerb's offsets with its line carried straight across each gap in it - from the last
    mapped cross-section before to the first after - and straight on past its ends; `otherwise`
    where a side has no kerb at all."""
    mapped = np.isfinite(offsets)
    return np.interp(s, s[mapped], offsets[mapped]) if mapped.any() else otherwise


# Highway classes, most important first - a junction's area takes its most important street's.
_RANK = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
         "living_street")


def _junction_face(network: StreetGraph, kerbs: _Kerbs, junction: int) -> Polygon | None:
    """The face around `junction` enclosed by the mapped kerbs and, across each leg, its first
    cross-section that meets its own kerb on both sides - None where that face is not closed."""
    closing = []
    for leg in network.legs(junction):
        if leg.direction() is None:
            continue
        stations = Stations(leg.line())
        left = kerbs.hits(leg.street, stations, 1)
        right = kerbs.hits(leg.street, stations, -1)
        both = np.flatnonzero(np.isfinite(left) & np.isfinite(right))
        if not len(both):
            return None
        i = both[0]
        # Run on just past each kerb, so the line crosses the kerb it ends on; the face it
        # closes is the same for any length that does.
        closing.append(LineString([stations.xy[i] + stations.left[i] * (left[i] + OVERSHOOT_M),
                                   stations.xy[i] - stations.left[i] * (right[i] + OVERSHOOT_M)]))
    if len(closing) < 2:
        return None
    centre = Point(network.node_xy[junction])
    hull = shapely.MultiLineString(closing).convex_hull
    near = [kerbs.segments[k] for k in kerbs.tree.query(hull, predicate="intersects")]
    for face in shapely.get_parts(shapely.polygonize(shapely.get_parts(
            shapely.unary_union([*closing, *near])))):
        if face.contains(centre):
            # Closed: inside its legs' cross-sections (to numerical precision).
            return face if face.difference(hull).area <= face.area * 1e-9 else None
    return None


def _ramp_gaps(layers: dict, frame: LocalFrame, report: Counter[str]) -> None:
    """Report each lowered kerb a footway crossing meets that has no `wheelchair` tag: the reader
    draws a curb ramp flush only where OSM says `wheelchair=yes`, so these are mapping gaps."""
    for kerb in layers["kerbs"]:
        line = frame.line(kerb.get("coords_wgs84") or [])
        if (line is not None and kerb["tags"].get("kerb") == "lowered" and kerb["tags"].get("wheelchair") is None
                and meets(kerb, line, layers["crossings"], frame)):
            report["lowered kerbs at a crossing with no wheelchair tag: not drawn as ramps"] += 1


def _closed_way(ring: list, tags: dict, ids: Iterator[int], frame: LocalFrame,
                nodes: list[NewNode]) -> WayChange:
    new = [NewNode(next(ids), *frame.wgs84(x, y), {}) for x, y in ring[:-1]]
    nodes.extend(new)
    return WayChange(next(ids), "create", (*(n.id for n in new), new[0].id), tags)


def _pieces(count: int, at: list[int]) -> list[tuple[int, int]]:
    """(first, last) node index of each piece of a way of `count` nodes split at indices `at`."""
    bounds = [0, *(i for i in at if 0 < i < count - 1), count - 1]
    return [(a, b) for a, b in itertools.pairwise(bounds) if b > a]


def _at_beyond(line: LineString, s: float, offset: float) -> np.ndarray:
    """`_at`, carried straight on past either end of `line` along its end segment."""
    if 0.0 <= s <= line.length:
        return _at(line, s, offset)
    end = 0.0 if s < 0 else line.length
    t = tangent(line, end)
    return _at(line, end, offset) + t * (s - end)


def _station(line: LineString, point) -> tuple[float, float]:
    """`point`'s station along `line`, and its offset from it (left positive)."""
    s = line.project(Point(point))
    t, v = tangent(line, s), np.asarray(point) - np.asarray(line.interpolate(s).coords[0])
    return s, float(t[0] * v[1] - t[1] * v[0])


def _at(line: LineString, s: float, offset: float) -> np.ndarray:
    """The point `offset` to the left of `line` at station `s`."""
    t = tangent(line, s)
    return np.asarray(line.interpolate(s).coords[0]) + offset * np.array([-t[1], t[0]])


# --- existing conditions -----------------------------------------------------------------------

class _Stops(defaultdict):
    """Stop lines by each street node they are joined to, and `floating`: (line, nearest street
    way) for each joined to none - which a stop line need not be (wiki Key:road_marking)."""

    def __init__(self) -> None:
        super().__init__(list)
        self.floating: list[tuple[LineString, int]] = []


def _stops(layers: dict, frame: LocalFrame, network: StreetGraph, report: Counter[str]) -> _Stops:
    """Every stop line: by each street node it is joined to, else with its nearest street - the
    street it belongs to as every kerb does (_Kerbs)."""
    stops = _Stops()
    ways = list(network.streets)
    tree = STRtree([network.lines[w] for w in ways])
    for stop_way in layers["stop_lines"]:
        stop = frame.line(stop_way["coords_wgs84"])
        if stop is None:
            continue
        joined = [n for n in stop_way["node_ids"] if network.on_street(n)]
        for n in joined:
            stops[n].append(stop)
        if not joined and ways:
            nearest = ways[int(tree.query_nearest(stop.interpolate(0.5, normalized=True))[0])]
            stops.floating.append((stop, nearest))
            report["stop lines on no street, given to their nearest"] += 1
    return stops


def _bands(layers: dict, network: StreetGraph, frame: LocalFrame) -> dict[int, tuple[float, Polygon]]:
    """Each crossing node on a street: half its crossing's `width`, and its band - the crossing
    way that wide, square-ended - the crosswalk as built. A crossing split where it passes over an
    island or a curb extension is still one crosswalk (wiki Tag:footway=traffic_island): its band
    runs on across a footway that joins it end to end to another crossing, both along its line."""
    crossings = {c["id"] for c in layers["crossings"]}
    ends: dict[int, list[dict]] = defaultdict(list)
    for way in [c for c in layers["crossings"] if c.get("node_ids")] + [w for w in layers["roads"] + layers["sidewalks"]
                                      if w["tags"].get("highway") == "footway" and w.get("node_ids")]:
        for node in {way["node_ids"][0], way["node_ids"][-1]}:
            ends[node].append(way)
    out = {}
    for crossing in layers["crossings"]:
        line = frame.line(crossing["coords_wgs84"])
        if line is None or not any(network.on_street(n) for n in crossing["node_ids"]):
            continue
        half = _half(crossing)
        band = _crosswalk(crossing, half, ends, crossings, frame).buffer(half, cap_style="flat")
        for node in crossing["node_ids"]:
            if network.on_street(node):
                out[node] = (half, band)
    return out


def _crosswalk(crossing: dict, half: float, ends: dict[int, list[dict]], crossings: set[int],
               frame: LocalFrame) -> LineString:
    """`crossing`'s line, run on across each footway joining it end to end to another crossing
    where both lie within `half` of its line: the whole crosswalk it was split from."""
    line = frame.line(crossing["coords_wgs84"])
    a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    ux, uy = (b - a) / np.linalg.norm(b - a)

    def along(way: dict) -> bool:
        return all(abs(ux * (y - a[1]) - uy * (x - a[0])) <= half
                   for x, y in (frame.point(*c) for c in way["coords_wgs84"]))

    parts, frontier = {crossing["id"]: crossing}, [crossing]
    while frontier:
        way = frontier.pop()
        for end in {way["node_ids"][0], way["node_ids"][-1]}:
            for walk in ends[end]:
                far = walk["node_ids"][-1] if walk["node_ids"][0] == end else walk["node_ids"][0]
                for then in ends[far]:
                    if (walk["id"] not in crossings and then["id"] in crossings and then["id"] not in parts
                            and along(walk) and along(then)):
                        parts |= {walk["id"]: walk, then["id"]: then}
                        frontier.append(then)
    joined = shapely.ops.linemerge([frame.line(w["coords_wgs84"]) for w in parts.values()])
    return joined if isinstance(joined, LineString) else line


def _band_span(line: LineString, band: Polygon, lo: float, hi: float) -> tuple[float, float] | None:
    """The nearest and furthest stations along `line` that `band` covers between lateral offsets
    `lo` and `hi` (left positive; along one offset where they are equal): the furthest is where
    paint between them starts for none of it to be on the band. None where the band is not there."""
    # Stationed on only the stretch the band can reach - no point of it is further than half its
    # perimeter from its centroid - so a street that comes back near it does not catch it. Its
    # vertices behind that stretch's start all station at 0, which can fold the ring: made valid.
    reach, about = band.exterior.length / 2, line.project(band.centroid)
    start = max(0.0, about - reach)
    near = shapely.ops.substring(line, start, about + reach)
    framed = shapely.make_valid(Polygon([_station(near, xy) for xy in band.exterior.coords]))
    lo, hi = sorted((lo, hi))
    clip = LineString([(0, lo), (near.length, lo)]) if lo == hi else shapely.box(0, lo, near.length, hi)
    part = framed.intersection(clip)
    return None if part.is_empty else (start + float(part.bounds[0]), start + float(part.bounds[2]))


def _datums(network: StreetGraph, kerbs: _Kerbs, stops: _Stops, bands: dict[int, tuple[float, Polygon]],
            report: Counter[str]) -> list[tuple[Leg, dict[str, float], int | None]]:
    """Each junction leg, how far out along it R.S. 39:4-138(e)'s setback is measured from on
    each kerb (left and right looking out from the junction), and the node of the crosswalk it
    claims (None where it has none). On each kerb, the further of the statute's two arms: the
    crosswalk's side line - any mapped crossing, a crosswalk in law, marked
    or not - where its band (the crossing way, its `width` wide) meets that kerb's mapped
    `barrier=kerb` (_Kerbs.band_reach), so a skewed crosswalk is measured where it is; and the
    cross street's side line (its kerb line, else half its width)."""
    half = {n: h for n, (h, _band) in bands.items()}
    out = []
    for junction in sorted(network.junctions):
        for leg in network.legs(junction):
            if (kerb_at := _kerb_line(network, leg, kerbs)) is not None:
                side_line = (kerb_at, "the cross street's kerb line")
            else:
                side_line = (_side_line(network, leg), "half the cross street's width")
            claimed = _claimed(leg, half, stops)
            if claimed is None:
                report["no-standing legs with no crossing mapped: the crosswalk arm not applied"] += 1
            datum = {}
            for side, sign in (("left", 1), ("right", -1)):
                arms = [side_line]
                if claimed is not None:
                    met = kerbs.band_reach(leg, bands[claimed][1], sign)
                    arms.append((met, "its crosswalk, where it meets the kerb") if met is not None else
                                (float(leg.d[leg.nodes.index(claimed)]) + half[claimed],
                                 "its crosswalk, at its node (no kerb mapped there)"))
                datum[side], why = max(arms)
                report[f"no-standing zones measured from {why} (leg kerbs)"] += 1
            out.append((leg, datum, claimed))
    return out


def _junction_interiors(mouths: list[tuple[Leg, float]],
                        network: StreetGraph) -> dict[int, list[tuple[float, float]]]:
    """Each way's stretches inside a junction - from its node out to each leg's mouth (_mouths)."""
    inside: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for leg, mouth in mouths:
        for way_id, a, b in leg.dark(min(mouth, leg.length), network.stations):
            inside[way_id].append((a, b))
    return inside


# What a piece inside a junction carries (_inside_junction), and so how one is known (_past_mouth).
INSIDE_JUNCTION = {"shoulder:both": "no", "cycleway:both": "no", "parking:left": "no", "parking:right": "no"}


def _inside_junction(tags: dict) -> dict:
    """A piece inside a junction, short of its crosswalk: no kerbside markings - no shoulder,
    cycle lane or track, or parking - only the restrictions its kerbs carry. A street's kerbside
    lanes end at the crosswalk; across the junction is the cross street."""
    out = {k: v for k, v in tags.items()
           if not k.startswith(("shoulder", "cycleway", "parking", "placement")) or ":restriction" in k}
    return out | INSIDE_JUNCTION


def _mouths(network: StreetGraph, kerbs: _Kerbs, stops: _Stops,
            bands: dict[int, tuple[float, Polygon]]) -> list[tuple[Leg, float]]:
    """Each junction leg, and how far out along it the junction ends: at the outer edge of the
    crosswalk it claims, the furthest the crosswalk reaches across the street, so none of the
    street's own paint is on the junction side of it however skewed it is. The band is straight,
    so that is where it crosses the centre line or where it meets either kerb (_Kerbs.band_reach)
    - a kerb not mapped there, that side's edge by the street's width (_band_span). One station
    for the whole street, as a section is laid square across it: ending each side at its own
    would shift the lanes the far side is laid against (src/osm_world.py:_Reader.cross_section).
    With no crosswalk, at the cross street's kerb line, else at half the cross street's width.
    Inside it a street carries no kerbside markings; the reader cuts the band out of whatever
    reaches it (src/osm_world.py:_Reader.on_carriageway)."""
    half = {n: h for n, (h, _band) in bands.items()}
    out = []
    for junction in sorted(network.junctions):
        for leg in network.legs(junction):
            claimed = _claimed(leg, half, stops)
            if claimed is None:
                mouth = _kerb_line(network, leg, kerbs)
                out.append((leg, _side_line(network, leg) if mouth is None else mouth))
                continue
            line, band = leg.line(), bands[claimed][1]
            edge = _half_width(network, leg)
            spans = [_band_span(line, band, 0.0, 0.0)]
            for sign in (1, -1):
                met = kerbs.band_reach(leg, band, sign)
                spans.append(_band_span(line, band, 0.0, sign * edge) if met is None else (met, met))
            out.append((leg, max([float(leg.d[leg.nodes.index(claimed)])] + [s[1] for s in spans if s])))
    return out


class _World:
    """A world as this module reads it, once: its layers, frame and street graph, its kerbs, and
    as they are asked for its stop lines, crosswalk bands, and from them where each junction ends
    along its legs (_mouths) and each leg's no-standing datums (_datums). What these find that OSM
    is missing goes in `report`."""

    def __init__(self, area: str, layers: dict, frame: LocalFrame, network: StreetGraph,
                 report: Counter[str] | None = None):
        self.layers, self.frame, self.network = layers, frame, network
        self.report: Counter[str] = Counter() if report is None else report
        self.kerbs = _Kerbs(layers, frame, network, SNAPSHOT_AREAS[area])

    @classmethod
    def of(cls, area: str, change: OsmChange | None = None, report: Counter[str] | None = None) -> _World:
        layers, frame = osm_layers(area, change), LocalFrame(SNAPSHOT_AREAS[area])
        return cls(area, layers, frame, StreetGraph(layers["roads"], frame), report)

    @functools.cached_property
    def stops(self) -> _Stops:
        return _stops(self.layers, self.frame, self.network, self.report)

    @functools.cached_property
    def bands(self) -> dict[int, tuple[float, Polygon]]:
        return _bands(self.layers, self.network, self.frame)

    @functools.cached_property
    def mouths(self) -> list[tuple[Leg, float]]:
        return _mouths(self.network, self.kerbs, self.stops, self.bands)

    @functools.cached_property
    def datums(self) -> list[tuple[Leg, dict[str, float], int | None]]:
        return _datums(self.network, self.kerbs, self.stops, self.bands, self.report)


def _leg_forward(leg: Leg) -> dict[int, bool]:
    """Each way of `leg`, and whether it runs outward from the junction (its own direction)."""
    return {way: j > i for (way, i), (other, j) in itertools.pairwise(leg.steps) if way == other}


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


def _extension_sides(leg: Leg, node: int | None, node_tags: dict[int, dict]) -> set[str]:
    """Which kerbs of `leg` - left and right looking out from its junction - its crosswalk's node
    gives a curb extension: `crossing:kerb_extension` (wiki), whose left and right are the
    direction of the way the node is on."""
    value = node_tags.get(node, {}).get("crossing:kerb_extension") if node is not None else None
    if value == "both":
        return {"left", "right"}
    if value not in ("left", "right"):
        return set()
    k = leg.nodes.index(node)
    way = leg.steps[min(k + 1, len(leg.steps) - 1)][0]
    return {value} if _leg_forward(leg).get(way, True) else {_other(value)}


def existing_markings(area: str, kerb_extensions: bool = False) -> tuple[OsmChange, Counter[str]]:
    """Every street way split where its centre line is not painted, the stretch tagged
    `lane_markings=no`, and each piece's `width` measured between its kerbs - and what each was
    read from, and what OSM is missing. With `kerb_extensions` (the daylighting proposal's base),
    every crosswalk a junction claims is given a curb extension on each kerb, and the no-standing
    zones follow from that tag as from one OSM already has."""
    report: Counter[str] = Counter()
    world = _World.of(area, report=report)
    layers, frame, network, kerbs, stops = world.layers, world.frame, world.network, world.kerbs, world.stops
    ids = _ids_below(None)
    nodes: list[NewNode] = []
    ways: list[WayChange] = []

    # Where the centre line is not painted: across each marked crossing, and out of each junction.
    crossing_of: dict[int, dict] = {}
    for crossing in layers["crossings"]:
        if _marked(crossing["tags"]):
            crossing_of |= {n: crossing for n in crossing["node_ids"] if network.on_street(n)}
    half_of = {n: _half(c) for n, c in crossing_of.items()}

    intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for node, half in half_of.items():
        for way_id, i in network.on_street(node):
            at = network.stations[way_id][i]
            intervals[way_id].append((at - half, at + half))
    mouths = world.mouths
    for leg, mouth in mouths:
        marks = [(mouth, "where the junction ends")]
        if (stop_at := _stop_line(leg, stops)) is not None:
            marks.append((stop_at, "its stop line"))
        reach, why = max(marks)
        report[f"junction legs: lane markings end at {why}"] += 1
        for way_id, a, b in leg.dark(min(reach, leg.length), network.stations):
            intervals[way_id].append((a, b))

    datums = world.datums
    if kerb_extensions:
        # The daylighting proposal's curb extensions: `crossing:kerb_extension` on the node of
        # every crosswalk a junction claims (wiki Key:crossing:kerb_extension, which needs
        # `highway=crossing` on that node - given it where OSM has it only on the way), on both kerbs.
        raw_nodes = fetch_borough_osm(SNAPSHOT_AREAS[area])["nodes"]
        crossing_of = {n: c for c in layers["crossings"] for n in c["node_ids"]}
        for node in {node for _leg, _datum, node in datums if node is not None}:
            tags = dict(raw_nodes[node].get("tags") or {})
            tags.setdefault("highway", "crossing")
            if "crossing" not in tags and crossing_of[node]["tags"].get("crossing"):
                tags["crossing"] = crossing_of[node]["tags"]["crossing"]
            tags["crossing:kerb_extension"] = "both"
            nodes.append(NewNode(node, raw_nodes[node]["lon"], raw_nodes[node]["lat"], tags))
            report["curb extensions given (kerbs)"] += 2

    # Where the law forbids standing (R.S. 39:4-138(e)): out along each leg from its datum, on each
    # kerb NO_STANDING_FT.
    no_standing: dict[tuple[int, str], list[tuple[float, float, float]]] = defaultdict(list)
    for leg, datum, _node in datums:
        forward = _leg_forward(leg)
        for side in ("left", "right"):
            feet = NO_STANDING_FT
            reach = datum[side] + feet * FT_TO_M
            stop_at = _stop_line(leg, stops)
            if stop_at is not None and stop_at > reach:
                report["stop lines further out than the no-standing zone (leg kerbs)"] += 1
            for way_id, a, b in leg.dark(min(reach, leg.length), network.stations):
                own = side if forward.get(way_id, True) else _other(side)
                no_standing[(way_id, own)].append((a, b, feet))

    # Each street's kerb to kerb, at every cross-section that meets its own kerb on both sides.
    sides: dict[int, tuple[Stations, np.ndarray, np.ndarray]] = {}   # way -> (stations, left, right)
    across: dict[int, tuple[np.ndarray, np.ndarray]] = {}    # way -> (stations, width or NaN)
    for way_id, way in network.streets.items():
        # Each end's cross-section spans the join, as the reader's does: two ways of one street
        # share it, so their areas meet with no wedge between them where the street bends.
        stations = Stations(network.lines[way_id], *network.beyond(way_id))
        left = kerbs.hits(street_key(way), stations, 1)
        right = kerbs.hits(street_key(way), stations, -1)
        sides[way_id] = (stations, left, right)
        across[way_id] = (stations.s, left + right)

    # One width per block - the stretch of a street between two junctions, whichever ways carry
    # it: the median of its measured cross-sections. A street's width only steps at a junction.
    block_at: dict[int, list[tuple[float, float, frozenset]]] = defaultdict(list)
    block_width: dict[frozenset, float | None] = {}
    for junction in network.junctions:
        for leg in network.legs(junction):
            cover: dict[int, list[float]] = {}
            for (way_id, i), (other, j) in itertools.pairwise(leg.steps):
                if way_id == other:
                    a, b = sorted((network.stations[way_id][i], network.stations[way_id][j]))
                    lo, hi = cover.get(way_id, [a, b])
                    cover[way_id] = [min(lo, a), max(hi, b)]
            key = frozenset((w, lo, hi) for w, (lo, hi) in cover.items())
            if key in block_width:
                continue                                 # the same block, walked from its far end
            values = np.concatenate([across[w][1][(across[w][0] >= lo) & (across[w][0] <= hi)]
                                     for w, lo, hi in key] or [np.array([])])
            values = values[np.isfinite(values)]
            block_width[key] = float(np.median(values)) if len(values) else None
            for w, lo, hi in key:
                block_at[w].append((lo, hi, key))
            report["blocks measured between their kerbs" if block_width[key] else
                   "blocks with no kerb pair mapped: the width fallback"] += 1

    # The street as it is built: its surface out to its own kerbs, and on a side with none, half its
    # block's width from the way - as `area:highway`, named like the street. Paint is laid from the
    # way's `width`, not from this; the reader keeps it on this surface.
    for way_id, way in network.streets.items():
        stations, left, right = sides[way_id]
        half = np.full(len(stations.s), carriageway_width_m(way["tags"])[0] / 2)
        if tagged_width(way["tags"]) is None:
            for lo, hi, key in block_at[way_id]:
                if block_width[key]:
                    half[(stations.s >= lo) & (stations.s <= hi)] = block_width[key] / 2
        tags = {"area:highway": way["tags"]["highway"], "note": AREA_NOTE}
        if way["tags"].get("surface"):
            tags["surface"] = way["tags"]["surface"]     # the street's surface is its area's
        if way["tags"].get("name"):
            tags["name"] = way["tags"]["name"]
        for ring in stations.strips(-_carried(stations.s, right, half), _carried(stations.s, left, half)):
            ways.append(_closed_way(ring, tags, ids, frame, nodes))

    # Each junction's area: the face around its node that the mapped kerbs enclose, each leg closed
    # by its first cross-section that meets its own kerb on both sides. Where a corner's kerb is
    # not mapped the face is not closed - it runs out past those cross-sections - and none is
    # written.
    for junction in sorted(network.junctions):
        face = _junction_face(network, kerbs, junction)
        if face is None:
            report["junctions with no area: a corner's kerb not mapped"] += 1
            continue
        report["junctions with an area, enclosed by their kerbs"] += 1
        legs = network.legs(junction)
        rank = max((network.streets[leg.steps[0][0]]["tags"]["highway"] for leg in legs),
                   key=lambda h: -_RANK.index(h) if h in _RANK else -len(_RANK))
        ways.append(_closed_way(list(face.exterior.coords),
                                {"area:highway": rank, "junction": "yes", "note": AREA_NOTE},
                                ids, frame, nodes))

    _ramp_gaps(layers, frame, report)

    interiors = _junction_interiors(mouths, network)
    # Each way split where its lane markings stop and at each junction it runs through - so every
    # piece is one block's - and each piece given its block's width. A `width` OSM has is kept.
    for way_id, way in network.streets.items():
        line = network.lines[way_id]
        dark = (_merged(intervals.get(way_id, []), line.length)
                if _has_lane_markings(way["tags"]) and way["tags"].get("lane_markings") != "no" else [])
        tagged = tagged_width(way["tags"]) is not None
        junctions = [network.stations[way_id][i] for i, n in enumerate(way["node_ids"])
                     if n in network.junctions and 0 < i < len(way["node_ids"]) - 1]
        widths = [] if tagged else [block_width[key] for _lo, _hi, key in block_at[way_id]]
        spans = {side: no_standing.get((way_id, side), []) for side in ("left", "right")}
        zones = {side: _merged([(a, b) for a, b, _feet in found], line.length)
                 for side, found in spans.items()}
        interior = _merged(interiors.get(way_id, []), line.length)
        if not dark and not any(widths) and not any(zones.values()) and not interior:
            continue
        report["no-standing zones ending inside an unpainted stretch"] += sum(
            lo < e < hi for side in zones for span in zones[side] for e in span for lo, hi in dark)
        node_ids, node_st, at = _split(line, way["node_ids"], network.stations[way_id],
                                       [e for span in dark + interior + zones["left"] + zones["right"]
                                        for e in span]
                                       + junctions, ids, frame, nodes)
        for k, (a, b) in enumerate(_pieces(len(node_ids), at)):
            middle = (node_st[a] + node_st[b]) / 2
            tags = dict(way["tags"])
            if any(lo <= middle <= hi for lo, hi in dark):
                tags["lane_markings"] = "no"
            if any(lo <= middle <= hi for lo, hi in interior):
                tags = _inside_junction(tags)
                report["inside a junction: no kerbside markings (ft of street)"] += round(
                    (node_st[b] - node_st[a]) / FT_TO_M)
            within = [side for side in ("left", "right")
                      if any(lo <= middle <= hi for lo, hi in zones[side])]
            if within:
                feet = min(f for side in within for a_, b_, f in spans[side] if a_ <= middle <= b_)
                tags = _with_note(_no_standing(tags, within), no_standing_note(feet))
                report[f"no standing within {feet:g} ft of the intersection (ft of kerb)"] += round(
                    len(within) * (node_st[b] - node_st[a]) / FT_TO_M)
            if not tagged:
                width = next((block_width[key] for lo, hi, key in block_at[way_id]
                              if lo <= middle <= hi), None)
                if width is not None:
                    tags["width"] = f"{width:.1f}"
            report["street pieces: width " + ("already in OSM" if tagged else
                                              "measured" if "width" in tags else
                                              "left to the fallback") + " (ft)"] += round(
                (node_st[b] - node_st[a]) / FT_TO_M)
            tags = _with_note(tags, EXISTING_NOTE)
            piece = tuple(node_ids[a:b + 1])
            ways.append(WayChange(way_id, "modify", piece, tags) if k == 0
                        else WayChange(next(ids), "create", piece, tags))
    return OsmChange(tuple(nodes), tuple(ways)), report


# --- the proposal ------------------------------------------------------------------------------



def _no_standing(tags: dict, sides=("left", "right")) -> dict:
    """`tags` with the kerbs on `sides` not to be stood at by law: each side's
    `:restriction=no_standing` for `:reason=junction` - a side already `no_stopping`, the stricter,
    kept - and any `parking:both:*` written out per side so nothing contradicts it. The side's
    position (`parking:<side>=lane`, ...) is left as it is: the bay is still there, and only
    standing in it is not allowed (wiki Street parking)."""
    out = _per_side(tags)
    for side in sides:
        if out.get(f"parking:{side}:restriction") != "no_stopping":
            out[f"parking:{side}:restriction"] = "no_standing"
            out[f"parking:{side}:restriction:reason"] = "junction"
        out.setdefault(f"parking:{side}:restriction:reason", "junction")
    return out


def _may_park(tags: dict, side: str, resign: bool = False) -> bool:
    """Whether OSM lets this kerb be parked: not `parking:<side>=no|separate`, and no
    `no_parking` / `no_standing` / `no_stopping` restriction (wiki Street parking). Untagged is
    not restricted. With `resign` - a proposal that re-signs the kerbs, the daylighting one - a
    restriction OSM records with no `:restriction:reason` is lifted: it is the blanket sign the
    curb extension replaces, and the statute's zone (reason `junction`) is what stays."""
    kind = sided(tags, "parking", side)
    restriction = sided(tags, "parking", side, ":restriction") or "none"
    reason = sided(tags, "parking", side, ":restriction:reason")
    if resign and restriction in PROHIBITIONS and reason is None:
        restriction = "none"
    return kind not in ("no", "separate") and restriction not in PROHIBITIONS


def _lifted(tags: dict) -> list[str]:
    """The kerbs `_may_park(resign=True)` opens that OSM's restriction shuts: the blanket sign."""
    return [side for side in ("left", "right") if _may_park(tags, side) != _may_park(tags, side, True)]


def _resigned(tags: dict, sides: list[str]) -> dict:
    """`tags` without the restriction (and its reason) on `sides`, `parking:both:*` written out per
    side so the other kerb keeps its own: what a proposal that re-signs the kerbs but does not
    re-section them (retagged) does to a way - _section_tags drops them the same way."""
    gone = tuple(f"parking:{side}:restriction" for side in sides)
    return {k: v for k, v in _per_side(tags).items() if not k.startswith(gone)}


def _per_side(tags: dict) -> dict:
    """`tags` with every `parking:both:*` written out as `parking:left:*` and `parking:right:*`,
    where the side does not say otherwise - so one side can change and the other keep its own."""
    out = {k: v for k, v in tags.items() if not k.startswith("parking:both")}
    for key, value in tags.items():
        if key.startswith("parking:both"):
            for side in ("left", "right"):
                out.setdefault(key.replace("parking:both", f"parking:{side}", 1), value)
    return out


def _openings(way: dict, line: LineString, sign: int, kerbs: _Kerbs, branches: list[dict],
              frame: LocalFrame, report: Counter[str]) -> list[tuple[float, float]]:
    """Stretches of `way` where traffic crosses its edge on one side (`sign` +1 is left): each of
    its street's lowered kerb segments on that side, projected onto the way, and each driveway or
    parking aisle (`branches`) leaving that side from a node it shares with the way where no
    lowered kerb is mapped, its `width` about that node. Where both are mapped the traced kerb is
    the opening: an untagged driveway's width is the service default, a guess at it."""
    out = []
    for k in np.flatnonzero(kerbs.lowered & (kerbs.owner == street_key(way))):
        ends = [_station(line, c) for c in kerbs.segments[k].coords]
        # Only a kerb beside this way: one past its ends would project onto an end node.
        if not any(0 < s < line.length for s, _ in ends):
            continue
        if all(offset * sign > 0 for _s, offset in ends):
            out.append((min(s for s, _ in ends), max(s for s, _ in ends)))
    lowered = _merged(out, line.length)
    node_st = vertex_stations(line)
    for branch in branches:
        for end in (0, -1):
            node = branch["node_ids"][end]
            if node not in way["node_ids"] or len(branch["coords_wgs84"]) < 2:
                continue
            # Its first segment out of the shared node says which side it leaves from.
            _s, offset = _station(line, frame.point(*branch["coords_wgs84"][1 if end == 0 else -2]))
            if offset * sign > 0:
                at = node_st[way["node_ids"].index(node)]
                if any(a <= at <= b for a, b in lowered):
                    report["driveways opened at their lowered kerb"] += 1
                    break
                width, tagged = width_or_default(branch["tags"], "width", "service")
                if not tagged:
                    report["driveways with no width tag, the service default"] += 1
                out.append((at - width / 2, at + width / 2))
            break
    return _merged(out, line.length)


# Where a street is narrower than its section, what gives and how far: the buffer first, to this,
# then the track to CONSTRAINED_TRACK_FT, then parking, then the lanes.
MIN_BUFFER_FT = in_unit("track.min_buffer", "ft")
# A two-stage turn box's length along the track: its queue of bicycles nose to tail.
BICYCLE_LENGTH_FT = in_unit("turn_box.bicycle_length", "ft")
TURN_BOX_QUEUE_BICYCLES = int(in_unit("turn_box.queue", "1"))
TURN_BOX_LENGTH_FT = BICYCLE_LENGTH_FT * TURN_BOX_QUEUE_BICYCLES


def _samples(line: LineString, kerbs: _Kerbs, street: str) -> Stations:
    """Where to measure a way's kerbs: every STATION_STEP_M, and at every vertex of its own kerbs
    - a kerb is straight between its vertices, so together they hold every place it comes
    closest to the way."""
    own = kerbs.ends[kerbs.end_owner == street]
    at = [line.project(Point(c)) for c in own]
    s = np.concatenate([Stations(line).s, [a for a in at if 0 < a < line.length]])
    return Stations(line, s=np.unique(np.round(s, 3)))


def _narrow(line: LineString, kerbs: _Kerbs, street: str, need_m: float) -> list[tuple[float, float, float]]:
    """The stretches of a way where its own kerbs stand closer together than `need_m`, as (from,
    to, the least kerb to kerb there) - each end where the kerbs cross back to `need_m`, between
    the samples either side of it."""
    sampled = _samples(line, kerbs, street)
    across = kerbs.hits(street, sampled, 1) + kerbs.hits(street, sampled, -1)
    short = np.isfinite(across) & (across < need_m)
    n, spans, i = len(sampled.s), [], 0

    def end(inside: int, outside: int) -> float:
        if 0 <= outside < n and np.isfinite(across[outside]):
            a, b = across[outside] - need_m, across[inside] - need_m
            return float(sampled.s[outside] + (sampled.s[inside] - sampled.s[outside]) * a / (a - b))
        return float(sampled.s[inside])

    while i < n:
        if not short[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and short[j + 1]:
            j += 1
        spans.append((end(i, i - 1), end(j, j + 1), float(across[i:j + 1].min())))
        i = j + 1
    return spans


def _need_ft(row) -> float:
    """What a design row's section needs, kerb to kerb, bar its hatched shoulders - the spare."""
    track = row.track_ft + row.buffer_ft if row.track_ft else 0.0
    return 2 * LANE_FT + track + row.parking_left_ft + row.parking_right_ft


def _blocks(network: StreetGraph, kerbs: _Kerbs, applied: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Each applied way's block's (need, least): what the block's section needs kerb to kerb, in
    feet - the most any of its ways' does, so a corner's unparked piece is sized like the parked
    block it is part of - and, in metres, the least kerb to kerb over every stretch of the block
    narrower than that (_narrow), else 0. A block is a junction leg."""
    need = {way_id: _need_ft(row) for way_id, row in applied.iterrows()}
    blocks = {frozenset(w for w, _i in leg.steps if w in need)
              for junction in network.junctions for leg in network.legs(junction)}
    blocks |= {frozenset([w]) for w in need if not any(w in block for block in blocks)}
    out: dict[int, tuple[float, float]] = {}
    for block in filter(None, blocks):
        ft = max(need[w] for w in block)
        least = min((there for w in block
                     for _lo, _hi, there in _narrow(network.lines[w], kerbs,
                                                    street_key(network.streets[w]), ft * FT_TO_M)),
                    default=0.0)
        out |= dict.fromkeys(block, (ft, least))
    return out


def _route(network: StreetGraph, proposal: Proposal) -> set[int]:
    """The ways `proposal` chooses."""
    return {w["id"] for w in network.streets.values() if proposal.chooses(w["tags"])}


def _abuts_zone(network: StreetGraph, way: dict, end: int, side: str) -> bool:
    """Whether `way`'s kerb on `side` carries on, past its end (0 or -1), into a corner's
    no-standing zone: the same street's way that continues there tagged
    `parking:<side>:restriction:reason=junction` on that kerb, its sides read in its own direction."""
    on = network.onward(way["id"], end)
    if on is None:
        return False
    other, other_end = on
    own = side if (other_end == -1) == (end == 0) else _other(side)   # sides swap where it turns round
    return network.ways[other]["tags"].get(f"parking:{own}:restriction:reason") == "junction"


def _free(spans: list[tuple[float, float]], length: float) -> list[tuple[float, float]]:
    """The stretches of [0, length] outside `spans` (sorted, merged)."""
    out, at = [], 0.0
    for lo, hi in spans:
        if lo > at:
            out.append((at, lo))
        at = max(at, hi)
    return out + ([(at, length)] if length > at else [])


def _stall_layout(network: StreetGraph, applied: pd.DataFrame,
                  crossed: dict[int, dict[str, list[tuple[float, float]]]],
                  ) -> dict[tuple[int, str], tuple[list[float], list[tuple[float, float]]]]:
    """Each parked kerb's stalls, laid along the kerb rather than way by way: its stretches between
    openings (`crossed`, way -> side -> spans) joined end to end across a way's end where the same
    street's way that carries on parks that kerb too - so a way split for the other kerb's sake
    leaves no gap. Along each run, as many whole STALL_LENGTH_M stalls as fit, kept
    DRIVEWAY_CLEARANCE_FT from an end at an opening; what is left over goes to the opening end,
    else hard against a corner's no-standing zone (_abuts_zone), else half to each end, and is
    hatched like any other unparked kerb. Per (way, side): each stall's middle, and each stretch
    of stalls, in the way's own stations."""
    parked = {(w, side) for w, row in applied.iterrows() for side in ("left", "right")
              if row[f"parking_{side}_ft"]}
    length = {w: network.lines[w].length for w, _side in parked}
    free = {(w, side): _free(crossed[w][side], length[w]) for w, side in parked}

    def reaches(w: int, side: str, end: int) -> bool:          # its kerb is free up to that end
        spans = free.get((w, side))
        return bool(spans) and (spans[0][0] <= NODE_MATCH_M if end == 0
                                else spans[-1][1] >= length[w] - NODE_MATCH_M)

    def onward(w: int, side: str, end: int) -> tuple[int, str, int, int] | None:
        """The parked kerb that carries on past `w`'s `end`: (way, its side, span, end entered)."""
        if not reaches(w, side, end):
            return None
        on = network.onward(w, end)
        if on is None:
            return None
        o, o_end = on
        own = side if (o_end == 0) == (end == -1) else _other(side)   # turned round: sides swap
        if not reaches(o, own, o_end):
            return None
        return o, own, 0 if o_end == 0 else len(free[(o, own)]) - 1, o_end

    out = {key: ([], []) for key in parked}
    seen: set[tuple[int, str, int]] = set()
    # Walked from each end of each run - a span end at an opening, or at a way's end nothing
    # parked carries on from - then from anywhere, for a run that closes on itself.
    entries = [(w, side, k, True) for (w, side), spans in free.items() for k in range(len(spans))
               if k > 0 or onward(w, side, 0) is None]
    entries += [(w, side, k, False) for (w, side), spans in free.items() for k in range(len(spans))
                if k < len(spans) - 1 or onward(w, side, -1) is None]
    entries += [(w, side, k, True) for (w, side), spans in free.items() for k in range(len(spans))]
    for start in entries:
        w, side, k, forward = start
        chain = []
        while (w, side, k) not in seen:
            seen.add((w, side, k))
            chain.append((w, side, k, forward))
            spans = free[(w, side)]
            if k != (len(spans) - 1 if forward else 0):
                break
            nxt = onward(w, side, -1 if forward else 0)
            if nxt is None:
                break
            w, side, k, entered = nxt
            forward = entered == 0
        if not chain:
            continue

        def at_end(item: tuple[int, str, int, bool], leaving: bool) -> tuple[bool, bool]:
            """Whether the run ends here at an opening, and else against a corner's zone."""
            w, side, k, forward = item
            lo, hi = free[(w, side)][k]
            low = forward != leaving                                  # this end is the span's lo
            opening = lo > NODE_MATCH_M if low else hi < length[w] - NODE_MATCH_M
            return opening, not opening and _abuts_zone(network, network.streets[w], 0 if low else -1, side)

        (open_a, zone_a), (open_b, zone_b) = at_end(chain[0], False), at_end(chain[-1], True)
        total = sum(hi - lo for w, side, k, _f in chain for lo, hi in [free[(w, side)][k]])
        clear = DRIVEWAY_CLEARANCE_FT * FT_TO_M
        a, b = clear * open_a, total - clear * open_b
        count = int(max(0.0, b - a) // STALL_LENGTH_M)
        if not count:
            continue
        share = (1.0 if open_a else 0.0) if open_a != open_b else (
            1.0 if zone_a and not zone_b else 0.0 if zone_b and not zone_a else 0.5)
        first = a + ((b - a) - count * STALL_LENGTH_M) * share
        last = first + count * STALL_LENGTH_M
        at = 0.0
        for w, side, k, forward in chain:
            lo, hi = free[(w, side)][k]

            def station(x: float, lo=lo, hi=hi, at=at, forward=forward) -> float:
                return lo + (x - at) if forward else hi - (x - at)

            middles, stretches = out[(w, side)]
            middles += [station(first + (j + 0.5) * STALL_LENGTH_M) for j in range(count)
                        if at <= first + (j + 0.5) * STALL_LENGTH_M < at + (hi - lo)]
            if max(first, at) < min(last, at + (hi - lo)):
                stretches.append(tuple(sorted((station(max(first, at)), station(min(last, at + (hi - lo)))))))
            at += hi - lo
    return out


def _tag_pieces(way: dict, line: LineString, network: StreetGraph, kerbs: _Kerbs, layers: dict,
                frame: LocalFrame, report: Counter[str], ids: Iterator[int], nodes: list[NewNode],
                edits: dict, created: list[WayChange], tags: dict, parked: dict[str, int],
                crossed_at: dict[str, list[tuple[float, float]]],
                stalls: dict[tuple[int, str], tuple[list[float], list[tuple[float, float]]]],
                track: tuple[str, str, int] | None = None, need_ft: float = 0.0,
                least_m: float = 0.0, inside: list[tuple[float, float]] = (), note: str = "") -> None:
    """`way` split where traffic crosses a kerb - a lowered kerb, or where none is mapped a
    driveway or parking aisle leaving from that side, its width - and each piece tagged `tags`, but where it is crossed, on
    that side: no hatching on its shoulder, no parking, and across the track (`track` = its side,
    the side its posts face, and its sign) no posts and dotted green. Its openings on each side are
    `crossed_at` (_openings). Each parked kerb (`parked`, side -> sign) is parked where `stalls`
    (_stall_layout) lays whole stalls and hatched elsewhere, its parking lane given to its
    shoulder. A parked piece's `parking:<side>:capacity` is the stalls whose middle is on it; the
    reader lays them along the whole run (src/osm_world.py:_Reader.stall_all). Where the
    kerbs stand closer than its block's section needs (`need_ft`, _blocks), the way is split there
    too and that piece narrowed (_narrowed) to the block's least kerb to kerb (`least_m`) - the
    markings give locally, never the whole street, and every narrowed piece of a block alike."""
    sides = {"left": 1, "right": -1}
    starts = {side: stalls[(way["id"], side)][0] for side in parked}
    runs = {side: stalls[(way["id"], side)][1] for side in parked}
    narrow = _narrow(line, kerbs, street_key(way), need_ft * FT_TO_M) if need_ft else []
    # Into and out of each narrowed stretch, on its wider side, a `placement=transition` piece
    # (wiki Key:placement) as long as the MUTCD taper for the most any marking moves - so the
    # markings have finished narrowing where the kerbs pinch them.
    # Every narrowed stretch of a block takes ONE layout - the block's section narrowed for its
    # least kerb to kerb, its parked sides parked - so the markings run straight through, across
    # way ends, whichever piece has a stall on it, an opening, or a corner's no-standing zone.
    tapers, layouts = [], []
    short = need_ft - least_m / FT_TO_M
    for lo, hi, _least in narrow if short > 0 else ():
        layouts.append((lo, hi, _narrowed(dict(tags), short, least_m, report, round((hi - lo) / FT_TO_M))))
        reach = _shift_m(tags, layouts[-1][2]) / taper_rate(tags)
        if lo - reach < 0 or hi + reach > line.length:
            report["transitions cut short by the end of their way"] += 1
        tapers += [(max(0.0, lo - reach), lo), (hi, min(line.length, hi + reach))]
    cuts = [e for spans in [*crossed_at.values(), *runs.values()] for span in spans for e in span]
    cuts += [e for lo, hi, _least in narrow for e in (lo, hi)] + [e for span in tapers for e in span]
    cuts += [e for span in inside for e in span]
    node_ids, node_st, at = _split(line, way["node_ids"], network.stations[way["id"]], cuts, ids,
                                   frame, nodes)
    for k, (a, b) in enumerate(_pieces(len(node_ids), at)):
        middle = (node_st[a] + node_st[b]) / 2
        length_ft = round((node_st[b] - node_st[a]) / FT_TO_M)
        crossed = {side: any(lo <= middle <= hi for lo, hi in spans)
                   for side, spans in crossed_at.items()}
        piece_tags = dict(tags)
        for side in sides:
            if crossed[side] and piece_tags.get(f"shoulder:{side}:markings") == "hatched":
                del piece_tags[f"shoulder:{side}:markings"]
                report["hatching broken where traffic crosses the kerb (ft)"] += length_ft
        if track:
            side, facing, _sign = track
            piece_tags[f"cycleway:{side}:separation:{facing}"] = "no" if crossed[side] else "flex_post"
            if crossed[side]:
                piece_tags[f"cycleway:{side}:crossing:markings"] = "dashes"
                report["of which crossed by traffic, no posts (ft)"] += length_ft
        for side in parked:
            if any(lo <= middle <= hi for lo, hi in runs[side]):
                stalls = sum(node_st[a] - NODE_MATCH_M <= st < node_st[b] - NODE_MATCH_M
                             for st in starts[side])
                piece_tags[f"parking:{side}:capacity"] = str(stalls)
                report["parking stalls marked"] += stalls
                continue
            lane_ft = (width_m(piece_tags.get(f"parking:{side}:width")) or 0.0) / FT_TO_M
            piece_tags = {key: v for key, v in piece_tags.items()
                          if not key.startswith(f"parking:{side}")}
            piece_tags[f"parking:{side}"] = "no"
            if not crossed[side]:
                # No whole stall here: its parking lane is hatched with the shoulder.
                shoulder_ft = (width_m(piece_tags.get(f"shoulder:{side}:width")) or 0.0) / FT_TO_M
                piece_tags[f"shoulder:{side}:width"] = _ft(shoulder_ft + lane_ft)
                report["kerb too short for a whole stall, hatched (ft)"] += length_ft
        layout = next((layout for lo, hi, layout in layouts if lo <= middle <= hi), None)
        if layout is not None:
            piece_tags = _laid_as(piece_tags, layout)
        elif any(a <= middle <= b for a, b in tapers):
            piece_tags["placement"] = "transition"
            report["transitions into and out of narrow stretches (ft of street)"] += length_ft
        if any(lo <= middle <= hi for lo, hi in inside):
            piece_tags = _inside_junction(piece_tags)
            report["inside a junction: no kerbside markings (ft of street)"] += length_ft
        piece_tags = _with_note(piece_tags, note)
        piece = tuple(node_ids[a:b + 1])
        if k == 0:
            edits[way["id"]] = (piece, piece_tags)
        else:
            created.append(WayChange(next(ids), "create", piece, piece_tags))


def _shift_m(a: dict, b: dict) -> float:
    """The most any marking moves between two sections of a street: at most half the change in
    `width` plus every change in a width laid across it - shoulders, track, buffer, parking, lanes."""
    keys = [f"{part}:{side}:{size}" for side in ("left", "right")
            for part, size in (("shoulder", "width"), ("cycleway", "width"), ("cycleway", "buffer"),
                               ("parking", "width"))]

    def w(tags: dict, key: str) -> float:
        return width_m(tags.get(key)) or 0.0

    def lanes(tags: dict) -> list[float]:
        return sorted(w or 0.0 for w in lane_widths(tags))

    la, lb = lanes(a), lanes(b)
    return (abs(w(a, "width") - w(b, "width")) / 2 + sum(abs(w(a, k) - w(b, k)) for k in keys)
            + (sum(abs(x - y) for x, y in zip(la, lb, strict=True)) if len(la) == len(lb) else 0.0))


def _laid_as(piece: dict, layout: dict) -> dict:
    """A piece of a narrowed stretch in its stretch's layout: its width, lanes, track and buffer;
    a parked side's parking lane as wide as the layout's (or gone, where the layout drops it);
    a side with no stall here hatching the layout's parking lane with its shoulder."""
    out = dict(piece)
    for key, value in layout.items():
        if key in ("width", "width:lanes") or (key.startswith("cycleway:") and key.endswith((":width", ":buffer"))):
            out[key] = value
    for side in ("left", "right"):
        shoulder = width_m(layout.get(f"shoulder:{side}:width")) or 0.0
        lane = width_m(layout.get(f"parking:{side}:width")) or 0.0 if layout.get(f"parking:{side}") == "lane" else 0.0
        if out.get(f"parking:{side}") == "lane" and lane:
            out[f"parking:{side}:width"] = _ft(lane / FT_TO_M)
            out[f"shoulder:{side}:width"] = _ft(shoulder / FT_TO_M)
        else:
            if out.get(f"parking:{side}") == "lane":            # the layout drops this side's parking
                out = {k: v for k, v in out.items() if not k.startswith(f"parking:{side}")}
                out[f"parking:{side}"] = "no"
            out[f"shoulder:{side}:width"] = _ft((shoulder + lane) / FT_TO_M)
    return out


def _narrowed(tags: dict, short_ft: float, least_m: float, report: Counter[str], length_ft: float) -> dict:
    """A piece's section narrowed by `short_ft` to fit kerbs `least_m` apart, giving in order: the
    buffer to MIN_BUFFER_FT, the track to CONSTRAINED_TRACK_FT, each parking lane to MIN_STALL_FT
    (or, past that, the lane - its width then hatched), and last the two lanes, equally. The piece
    carries its own `width`, and no shoulder but what dropping a parking lane leaves."""
    out, left = dict(tags), short_ft
    report["narrow stretches: markings narrowed locally (ft of street)"] += length_ft
    track = next((key.split(":")[1] for key, v in tags.items()
                  if key.count(":") == 1 and key.startswith("cycleway:") and v == "track"), None)
    if track:
        for key, floor, what in ((f"cycleway:{track}:buffer", MIN_BUFFER_FT, "buffer"),
                                 (f"cycleway:{track}:width", CONSTRAINED_TRACK_FT, "track")):
            have = (width_m(out.get(key)) or 0.0) / FT_TO_M
            give = min(left, max(0.0, have - floor))
            if give > 0:
                kept = max(floor, math.floor((have - give) * 12) / 12)   # down to the inch: never wider
                out[key] = _ft(kept)
                left -= have - kept
                report[f"narrow stretches: {what} narrowed (ft of street)"] += length_ft
    for side in ("left", "right"):
        out[f"shoulder:{side}:width"] = "0'"
    for side in ("left", "right"):
        if left <= 0 or out.get(f"parking:{side}") != "lane":
            continue
        have = (width_m(out.get(f"parking:{side}:width")) or 0.0) / FT_TO_M
        if have - MIN_STALL_FT >= left:
            out[f"parking:{side}:width"] = _ft(math.floor((have - left) * 12) / 12)
            left = 0.0
        else:
            out = {k: v for k, v in out.items() if not k.startswith(f"parking:{side}")}
            out[f"parking:{side}"] = "no"
            out[f"shoulder:{side}:width"] = _ft(max(0.0, have - left))
            left = max(0.0, left - have)
            report["narrow stretches: parking dropped (ft of street)"] += length_ft
    if left > 0:
        out["width:lanes"] = "|".join([_ft(math.floor((LANE_FT - left / 2) * 12) / 12)] * 2)
        report["narrow stretches: lanes narrowed (ft of street)"] += length_ft
    out["width"] = f"{math.floor(least_m * 100) / 100:.2f}"   # to the centimetre, down: the kerbs hold it
    return out


def _design(network: StreetGraph, route: set[int], kerb: str, resign: bool = False) -> pd.DataFrame:
    """The proposal as one table over every street way in the world: a row per way, every quantity
    a column, and every rule a vector operation over all of them at once.

    The section, kerb to kerb: two LANE_FT lanes everywhere it is applied; on `route` (the
    proposal's ways) the widest
    track that leaves them at exactly that - TRACK_FT, else CONSTRAINED_TRACK_FT - and its
    BUFFER_FT buffer, which gives first (the give order) down to MIN_BUFFER_FT
    before a block loses its track, on `kerb` (a compass side), with all the spare to the other; elsewhere the spare
    halved between the kerbs. Each kerb's spare, lane outward: a parking lane against the lane, up
    to PARKING_LANE_FT, where OSM lets that kerb be parked and the spare holds MIN_STALL_FT; then
    the rest a hatched shoulder. Applied to every way on the route, and every
    other whose centre line is painted, that holds two lanes."""
    ways = list(network.streets.values())
    lines = [network.lines[w["id"]] for w in ways]
    d = pd.DataFrame({
        "id": [w["id"] for w in ways],
        "length_m": [line.length for line in lines],
        "width_ft": [carriageway_width_m(w["tags"])[0] / FT_TO_M for w in ways],
        "width_tagged": [tagged_width(w["tags"]) is not None for w in ways],
        "route": [w["id"] in route for w in ways],
        # A street whose centre line is painted - anywhere: `lane_markings=no` on a piece of it
        # (out of a junction) drops the centre line there, not the street's section.
        "painted": [_has_centre_line(w["tags"]) for w in ways],
        "track_left": [float((Stations(line).left @ np.array(COMPASS[kerb])).mean()) > 0 for line in lines],
        "may_left": [_may_park(w["tags"], "left", resign) for w in ways],
        "may_right": [_may_park(w["tags"], "right", resign) for w in ways],
        "inside_left_ft": 0.0, "inside_right_ft": 0.0,   # its track has no parking behind it
    })
    room = d["width_ft"]
    on_route = d["route"]
    d["track_ft"] = np.select([on_route & (room >= TRACK_FT + BUFFER_FT + 2 * LANE_FT),
                               on_route & (room >= CONSTRAINED_TRACK_FT + MIN_BUFFER_FT + 2 * LANE_FT)],
                              [TRACK_FT, CONSTRAINED_TRACK_FT], 0.0)
    tracked = d["track_ft"] > 0
    # The buffer the block holds, to the inch down: BUFFER_FT where it fits, else what is left.
    d["buffer_ft"] = np.where(tracked, np.minimum(
        BUFFER_FT, np.floor((room - 2 * LANE_FT - d["track_ft"]) * 12) / 12), 0.0)
    spare = room - 2 * LANE_FT - np.where(tracked, d["track_ft"] + d["buffer_ft"], 0.0)
    d["applies"] = (on_route | d["painted"]) & (spare >= 0)
    # Which kerb the track takes, and so whose spare is none.
    track_is = {"left": d["track_left"], "right": ~d["track_left"]}
    for side in ("left", "right"):
        mine = np.where(tracked, np.where(track_is[side], 0.0, spare), spare / 2)
        # Whether a stretch of this kerb is long enough for a stall is _tag_pieces', which sees
        # the kerb's openings; here a kerb parks where its spare holds a parking lane.
        parks = d[f"may_{side}"] & (mine >= MIN_STALL_FT) & ~(tracked & track_is[side])
        d[f"parking_{side}_ft"] = np.where(parks, np.floor(np.minimum(mine, PARKING_LANE_FT) * 12) / 12, 0.0)
        d[f"shoulder_{side}_ft"] = np.maximum(0.0, np.floor((mine - d[f"parking_{side}_ft"]) * 12) / 12)
    return d.set_index("id")


def _section_tags(osm: dict, row, proposal: Proposal) -> dict:
    """A way's tags with its row of the design written onto them, and the proposal's track tags."""
    near = "left" if row.track_left else "right"
    far = "right" if near == "left" else "left"
    tags = {k: v for k, v in osm.items() if not k.startswith(("cycleway", "shoulder", "parking:"))}
    tags |= {"lanes": "2", "width:lanes": f"{_ft(LANE_FT)}|{_ft(LANE_FT)}"}
    if row.track_ft:
        tags |= proposal.track_tags(near) | {
            f"cycleway:{near}:width": _ft(row.track_ft), f"cycleway:{near}:buffer": _ft(row.buffer_ft),
            f"cycleway:{far}": "no"}
    for side in ("left", "right"):
        parking = getattr(row, f"parking_{side}_ft")
        if parking:
            tags |= {f"parking:{side}": "lane", f"parking:{side}:orientation": "parallel",
                     f"parking:{side}:markings": "yes", f"parking:{side}:width": _ft(parking)}
        else:
            # An unparked kerb keeps what OSM says forbids it - the restriction and its reason.
            tags[f"parking:{side}"] = "no"
            tags |= {key: v for key, v in _per_side(osm).items() if key.startswith(f"parking:{side}:restriction")}
        tags |= {f"shoulder:{side}": "yes", f"shoulder:{side}:width": _ft(getattr(row, f"shoulder_{side}_ft"))}
        # Hatched - but with curb extensions a corner's no-standing zone is unmarked: the extension
        # is that zone's marking (_extensions), and the turn inside its arc is the street's.
        zone = proposal.kerb_extensions and tags.get(f"parking:{side}:restriction:reason") == "junction"
        tags[f"shoulder:{side}:markings"] = "no" if zone else "hatched"
    return tags


def _left(line: LineString, s: float) -> np.ndarray:
    """The unit vector square to `line` at station `s`, to its left."""
    t = tangent(line, s)
    return np.array([-t[1], t[0]])


def _junction_moves(network: StreetGraph, blocks: list) -> dict[int, np.ndarray]:
    """How far each junction moves when the blocks into it are re-centred (`blocks`: leg, line,
    offset or None): on to every street through it at once - the move along each street's left
    there is that street's own offset at the junction, the mean of its re-centred blocks', or 0
    where none of its blocks moved, so a street left where OSM has it still runs through the node.
    Least squares where three streets disagree; only junctions some street moved."""
    at: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for leg, line, shift, _bumps in blocks:
        for node, s in ((leg.junction, 0.0), (leg.end, line.length)):
            if node in network.junctions:
                at[node][leg.street].append((_left(line, s), shift))
    moves = {}
    for node, streets in at.items():
        normals, shifts = [], []
        for legs in streets.values():
            normal = legs[0][0]
            moved = [shift * float(left @ normal) for left, shift in legs if shift is not None]
            normals.append(normal)
            shifts.append(float(np.mean(moved)) if moved else 0.0)
        if any(shifts):
            moves[node] = np.linalg.lstsq(np.array(normals), np.array(shifts), rcond=None)[0]
    return moves


def _recentred(area: str, base: OsmChange, report: Counter[str],
               proposal: Proposal | None = None) -> OsmChange:
    """`base` with every street re-centred between its kerbs.

    A way runs down the middle of its carriageway (wiki Key:placement, the default) and every
    marking is laid from it, but a mapped way is often off that middle - Seminary Avenue by 4.4 ft,
    W Broad east of Louellen by 1.2-3.2 ft, which squeezed a lane there to 8.4 ft. OSM has no tag
    for "this far off centre"; the way itself is moved.
    - Each block (a junction leg) takes ONE offset: the median, over its cross-sections that meet
      its own kerbs on both sides, of where the kerbs' midline lies from it - so the paint laid
      from it cannot jitter. Every node inside the block moves square to it by that much.
    - Where a stretch of it is narrower than its section (_narrow), the way moves on to that
      stretch's own midline - the median there - so the narrowed section is centred between the
      kerbs that pinch it, shifting in and out over the MUTCD taper for its speed (taper_rate),
      with a node added at each end of each taper and of the stretch.
    A junction moves on to every re-centred street through it (_junction_moves) - pinning it where
    OSM has it bent an off-centre street back to OSM's line at every cross street - and each block
    tapers from there to its own offset. A block with no kerb pair mapped stays where OSM has it,
    but for the taper off a junction that moved. Without a `proposal` (existing.osc) only the
    blocks move: there is no section to fit, so no stretch is narrow."""
    world = _World.of(area, base)
    frame, network, kerbs = world.frame, world.network, world.kerbs
    raw_nodes = fetch_borough_osm(SNAPSHOT_AREAS[area])["nodes"]
    need: dict[int, float] = {}
    if proposal is not None:
        design = _design(network, _route(network, proposal), proposal.kerb, proposal.kerb_extensions)
        need = {way_id: ft * FT_TO_M
                for way_id, (ft, _least) in _blocks(network, kerbs, design[design["applies"]]).items()}
    ids = _ids_below(base)
    moves: dict[int, np.ndarray] = {}
    added: dict[int, list[tuple[int, float, int]]] = defaultdict(list)   # way -> (after index, d, node)
    new_nodes: list[NewNode] = []
    done: set[frozenset] = set()
    blocks: list[tuple[Leg, LineString, float | None, list]] = []   # (leg, its line, offset, stretches)
    for junction in network.junctions:
        for leg in network.legs(junction):
            key = frozenset(leg.nodes)
            if key in done:
                continue                                 # the same block, walked from its far end
            done.add(key)
            line = leg.line()
            if line.length == 0:
                continue
            offsets, bumps = [], []                      # bumps: (from, to, correction, taper) in d
            for way_id in {way_id for way_id, _i in leg.steps}:
                way, wline = network.streets[way_id], network.lines[way_id]
                sampled = _samples(wline, kerbs, street_key(way))
                left = kerbs.hits(street_key(way), sampled, 1)
                right = kerbs.hits(street_key(way), sampled, -1)
                both = np.isfinite(left) & np.isfinite(right)
                middle = sampled.xy + sampled.left * ((left - right) / 2)[:, None]
                offsets += [_station(line, point)[1] for point in middle[both]]
                if way_id not in need:
                    continue
                for lo, hi, _least in _narrow(wline, kerbs, street_key(way), need[way_id]):
                    there = both & (sampled.s >= lo) & (sampled.s <= hi)
                    if not there.any():
                        continue
                    # The stretch's own midline, sample by sample, rate-limited to the taper both ways
                    # so a wobble in the tracing cannot make the street wiggle.
                    rate = taper_rate(way["tags"])
                    at = sorted((_station(line, sampled.xy[i])[0], _station(line, middle[i])[1])
                                for i in np.flatnonzero(there))
                    ds, mids = np.array([a for a, _m in at]), np.array([m for _a, m in at])
                    for order in (range(1, len(ds)), range(len(ds) - 2, -1, -1)):
                        for i in order:
                            j = i - 1 if order.step == 1 else i + 1
                            step = rate * abs(ds[i] - ds[j])
                            mids[i] = min(max(mids[i], mids[j] - step), mids[j] + step)
                    d_lo, d_hi = sorted(_station(line, wline.interpolate(x).coords[0])[0] for x in (lo, hi))
                    bumps.append((d_lo, d_hi, (ds, mids), rate))
            if not offsets:
                report["blocks with no kerb pair mapped: not re-centred"] += 1
                blocks.append((leg, line, None, []))
                continue
            shift = float(np.median(offsets))
            report["blocks re-centred between their kerbs"] += 1
            report[f"blocks re-centred by {abs(shift) / FT_TO_M:.0f} ft"] += 1
            # Each stretch: its midline profile less the block's offset, and how long the taper on
            # to it must be at each end - for the larger of its two end corrections.
            bumps = [(lo, hi, (ds, mids - shift),
                      max(abs(mids[0] - shift), abs(mids[-1] - shift)) / rate)
                     for lo, hi, (ds, mids), rate in bumps]
            report["narrow stretches the street shifts on to the middle of"] += len(bumps)
            blocks.append((leg, line, shift, bumps))
    shared = _junction_moves(network, blocks)
    for junction, move in shared.items():
        moves[junction] = network.node_xy[junction] + move
    report["junctions moved on to the streets re-centred through them"] = len(shared)
    for leg, line, shift, bumps in blocks:
        # From where its junction went, on to the block's offset over the taper for its speed: a
        # junction no street through it moved stays, and a dead end is the block's own.
        own = shift or 0.0
        ends = [float(shared[node] @ _left(line, s)) if node in shared
                else own if node not in network.junctions else 0.0
                for node, s in ((leg.junction, 0.0), (leg.end, line.length))]
        # And how far each junction moved along the block, into it positive: the block's nodes
        # slide with it, so none is left behind it - over at least pi times that, where the
        # slide's ease cannot carry one node past the next.
        alongs = [float(shared[node] @ tangent(line, s)) * (1 if s == 0.0 else -1) if node in shared else 0.0
                  for node, s in ((leg.junction, 0.0), (leg.end, line.length))]
        if shift is None and not any(ends) and not any(alongs):
            continue
        rate = taper_rate(network.streets[leg.steps[0][0]]["tags"])
        leads = [max(abs(end - own) / rate, math.pi * abs(along)) for end, along in zip(ends, alongs, strict=True)]
        if sum(leads) > line.length:
            report["tapers cut short by a junction"] += 1
            leads = [lead * line.length / sum(leads) for lead in leads]

        def station(d: float, alongs: list = alongs, leads: list = leads, length: float = line.length) -> float:
            start, end = ((along * (1.0 - ease(x / lead)) if lead > 0 else 0.0)
                          for along, lead, x in zip(alongs, leads, (d, length - d), strict=True))
            return d + start - end

        def at(d: float, line: LineString = line) -> np.ndarray:
            return _at_beyond(line, station(d), offset(d))

        def offset(d: float, own: float = own, ends: list = ends, leads: list = leads,
                   bumps: list = bumps, length: float = line.length) -> float:
            base = own + sum((end - own) * (1.0 - ease(x / lead)) for end, lead, x
                             in zip(ends, leads, (d, length - d), strict=True) if lead > 0)
            best = 0.0
            for lo, hi, (ds, corrections), taper in bumps:
                if lo <= d <= hi:
                    value = float(np.interp(d, ds, corrections))
                elif taper > 0 and lo - taper < d < lo:
                    value = float(corrections[0]) * ease((d - (lo - taper)) / taper)
                elif taper > 0 and hi < d < hi + taper:
                    value = float(corrections[-1]) * ease(((hi + taper) - d) / taper)
                else:
                    continue
                if abs(value) > abs(best):
                    best = value
            return base + best

        # Every ramp drawn as a curve: a node every STATION_STEP_M along it.
        ramps = [span for span, lead in zip(((0.0, leads[0]), (line.length - leads[1], line.length)), leads,
                                            strict=True) if lead > 0]
        ramps += [span for lo, hi, _c, taper in bumps for span in ((lo - taper, lo), (lo, hi), (hi, hi + taper))]

        for k, node in enumerate(leg.nodes):
            if node in (leg.junction, leg.end) or node in network.junctions or node in moves:
                continue
            moves[node] = at(float(leg.d[k]))
        for start, stop in ramps:
            if start < 0 or stop > line.length:
                report["tapers cut short by a junction"] += 1
            steps = max(1, math.ceil((stop - start) / STATION_STEP_M))
            for d in np.linspace(start, stop, steps + 1):
                if not 0 < d < line.length:
                    continue
                k = int(np.searchsorted(leg.d, d, side="right")) - 1
                if not 0 <= k < len(leg.steps) - 1:
                    continue
                (way_a, i_a), (way_b, i_b) = leg.steps[k], leg.steps[k + 1]
                if way_a != way_b or abs(i_a - i_b) != 1 or min(abs(d - leg.d[k]), abs(d - leg.d[k + 1])) < NODE_MATCH_M:
                    continue
                node = next(ids)
                new_nodes.append(NewNode(node, *frame.wgs84(*at(d)), {}))
                added[way_a].append((min(i_a, i_b), d if i_b > i_a else -d, node))
    report["nodes moved to re-centre streets"] = len(moves)
    report["nodes added for the tapers on to narrow stretches"] = len(new_nodes)
    kept = [node for node in base.nodes if node.id not in moves]
    # A moved node keeps its tags as `base` has them - retagged there (a crossing's kerb
    # extension), else as OSM has them.
    in_base = {node.id: node.tags for node in base.nodes}
    moved = [NewNode(node, *frame.wgs84(*xy), in_base[node] if node in in_base
                     else raw_nodes.get(node, {}).get("tags") or {})
             for node, xy in moves.items()]
    ways = {w.id: w for w in base.ways}
    for way_id, inserts in added.items():
        old = ways[way_id].node_ids if way_id in ways else tuple(network.streets[way_id]["node_ids"])
        after: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for index, order, node in inserts:
            after[index].append((order, node))
        node_ids = tuple(n for i, existing in enumerate(old)
                         for n in (existing, *(node for _o, node in sorted(after.get(i, [])))))
        ways[way_id] = (dataclasses.replace(ways[way_id], node_ids=node_ids) if way_id in ways else
                        WayChange(way_id, "modify", node_ids,
                                  _with_note(network.streets[way_id]["tags"],
                                             proposal.note if proposal else RECENTRED_NOTE)))
    return OsmChange((*kept, *moved, *new_nodes), tuple(ways.values()))


def _past_mouth(leg: Leg, mouth: float, network: StreetGraph) -> Iterator[int]:
    """Each piece of the leg past the junction's mouth, from it outward: the step it begins at. Past
    the pieces its tags say are the junction's own (INSIDE_JUNCTION) too: existing_markings split
    them off at the mouth as it found it, and re-centring the street (_recentred) moves the mouth
    found after."""
    return (k for k in range(len(leg.steps) - 1)
            if leg.d[k + 1] > mouth + NODE_MATCH_M and leg.steps[k + 1][0] == leg.steps[k][0]
            and not INSIDE_JUNCTION.items() <= network.streets[leg.steps[k][0]]["tags"].items())


def _track_ends(mouths: list[tuple[Leg, float]], network: StreetGraph, applied: pd.DataFrame,
                frame: LocalFrame, ids: Iterator[int], nodes: list[NewNode],
                created: list[WayChange], report: Counter[str], proposal: Proposal) -> None:
    """Where the track meets each junction: it ends at the mouth (_mouths) of every leg of the
    route that carries it - `applied`'s `track_ft` wide on its `track_left` (else right) kerb,
    inside any shoulder there. Two such legs at a junction are joined across it by a cycle crossing -
    `highway=cycleway` + `cycleway=crossing` (wiki), straight from the track's end on one to its
    end on the other: `oneway=no`, unless the track carries its riders one way (`track_along`,
    whether they ride in the way's direction: rides_along), when it is `oneway=yes` from the end
    they ride in at to the one they ride out at. A leg with none across from it is where the
    track stops: a two-stage turn box over the TURN_BOX_LENGTH_FT past its crosswalk (MUTCD
    9E.11) - `area:highway=cycleway` + `cycleway=two_stage_box` + the track's `surface:colour`,
    the value the 2020 draft Proposal:Two-stage bicycle turn's; OSM has no approved tag for the box."""
    by_junction: dict[int, list[tuple[Leg, float, int, float, float, bool | None, str | None]]] = defaultdict(list)
    for leg, mouth in mouths:
        k = next(_past_mouth(leg, mouth, network), None)
        if k is None or leg.steps[k][0] not in applied.index:
            continue
        row = applied.loc[leg.steps[k][0]]
        if not row["track_ft"]:
            continue
        forward = leg.steps[k + 1][1] > leg.steps[k][1]
        near = (1 if row["track_left"] else -1) * (1 if forward else -1)   # the leg's left, +1
        own = "left" if row["track_left"] else "right"
        along = row.get("track_along")
        along = None if pd.isna(along) else bool(along)        # None: ridden both ways
        # Across the junction and in the box, the track's own colour: its way's, else as the
        # proposal tags it.
        colour = row.get("track_colour") or proposal.tags.get("cycleway:{side}:surface:colour")
        by_junction[leg.junction].append((leg, mouth, near, (row["width_ft"] / 2 - row[f"shoulder_{own}_ft"]) * FT_TO_M,
                                          row["track_ft"], None if along is None else along != forward, colour))
    for ends in by_junction.values():
        colour = {"surface:colour": ends[0][-1]} if ends[0][-1] else {}
        if len(ends) >= 2:
            points = [(_at(leg.line(), mouth, near * (half - track * FT_TO_M / 2)), track, inbound)
                      for leg, mouth, near, half, track, inbound, _colour in ends]
            (a, wa, _a_in), (b, wb, b_in) = max(itertools.combinations(points, 2),
                                                key=lambda pair: float(np.linalg.norm(pair[0][0] - pair[1][0])))
            if b_in:                                           # from where its riders ride in
                a, b = b, a
            oneway = "no" if b_in is None else "yes"
            crossing = [NewNode(next(ids), *frame.wgs84(*xy), {}) for xy in (a, b)]
            nodes.extend(crossing)
            created.append(WayChange(next(ids), "create", tuple(n.id for n in crossing),
                                     {"highway": "cycleway", "cycleway": "crossing", "oneway": oneway,
                                      "width": _ft(min(wa, wb)), "crossing:markings": "dashes",
                                      **colour, "note": proposal.note}))
            continue
        leg, start, near, half, track, _inbound, _colour = ends[0]
        line = leg.line()
        at = np.unique(np.concatenate([np.arange(start, start + TURN_BOX_LENGTH_FT * FT_TO_M, STATION_STEP_M),
                                       [start + TURN_BOX_LENGTH_FT * FT_TO_M]]))
        outer = [_at(line, x, near * half) for x in at]
        inner = [_at(line, x, near * (half - track * FT_TO_M)) for x in at[::-1]]
        created.append(_closed_way([tuple(xy) for xy in [*outer, *inner, outer[0]]],
                                   {"area:highway": "cycleway", "cycleway": "two_stage_box",
                                    **colour, "note": TURN_BOX_NOTE},
                                   ids, frame, nodes))
        report["two-stage turn boxes where the track ends"] += 1


def _zone_end(leg: Leg, side: str, network: StreetGraph) -> float:
    """How far out along `leg` its kerb on `side` (looking out from the junction) is a corner's
    no-standing zone: its ways from the junction out tagged `parking:<side>:restriction:reason=junction`
    on that kerb, each read in its own direction; 0 where the first is not."""
    forward, end = _leg_forward(leg), 0.0
    for ((way, _i), (other, _j)), d in zip(itertools.pairwise(leg.steps), leg.d[1:], strict=True):
        if way != other:
            continue
        own = side if forward.get(way, True) else _other(side)
        if network.ways[way]["tags"].get(f"parking:{own}:restriction:reason") != "junction":
            break
        end = float(d)
    return end


def _beside(edge: Edge, end: float, depth: float) -> Polygon:
    """The street on a tangent edge's corner side (src/turning.py): `depth` in from it, and along it
    from as far behind the junction out to `end`."""
    p, o, n = edge.point, edge.out, edge.inward
    return Polygon([p - depth * o, p + end * o, p + end * o + depth * n, p - depth * o + depth * n])


def _paintable(piece) -> bool:
    """Whether a piece of an extension is a polygon at least MIN_HATCH_FT wide."""
    return isinstance(piece, Polygon) and not piece.buffer(-MIN_HATCH_FT * FT_TO_M / 2).is_empty


def _extensions(area: str, change: OsmChange, ids: Iterator[int],
                report: Counter[str]) -> tuple[list[NewNode], list[WayChange]]:
    """Each painted curb extension in `change`'s world as drawn (src/osm_world.py:read), as OSM
    maps one: a painted area, a closed way `road_marking=restriction` + `pattern=solid` +
    `colour=tan` (wiki Key:road_marking, its no-parking marking, filled in the tan NYC paints a
    curb extension). One shape per corner - between a leg's left kerb and the next leg
    anticlockwise's right - where its crosswalk's node gives either kerb one
    (`crossing:kerb_extension`, _extension_sides): the drawn street outside both legs' drawn travel
    lanes - each leg's tangent edge, as a turn is measured (_tangent_edge), the corner between them
    rounded to the EFFECTIVE_RADIUS_M arc tangent to both (src/turning.py), the turn inside it left
    to traffic - and off every bikeway and its buffer, out along each leg to the end of its
    no-standing zone (_zone_end): the zone's hatching, painted as the extension; each piece against
    a travel lane, car or bike. A piece under MIN_HATCH_FT wide is not painted; each is
    snapped to OSM_GRID_M, so it is whole as written. Flexible posts, `barrier=bollard` +
    `bollard=flexible` nodes (wiki Tag:barrier=bollard), run down it, half its narrower end's width
    in from the travel lanes (_posts). The crosswalk is left as OSM maps it: the reader paints the
    extension over it (src/osm_world.py:_Reader.crossing). The nodes and ways it adds."""
    layers, reader = read(area, change)
    world = _World(area, layers, reader.frame, reader.graph)
    frame, network, mouths = reader.frame, reader.graph, dict(world.mouths)
    node_tags = {n["id"]: n["tags"] for n in layers["traffic_control"] if "lon" in n}
    node_of = {leg: node for leg, _datum, node in world.datums}
    bikeway = shapely.union_all(reader.bikeway)
    free = reader.carriageway().difference(bikeway)                     # the street off every bikeway
    nodes: list[NewNode] = []
    ways: list[WayChange] = []
    for junction in sorted(network.junctions):
        for a, b in network.corners(junction):
            pair = ((a, "left", 1), (b, "right", -1))
            if not any(side in _extension_sides(leg, node_of.get(leg), node_tags) for leg, side, _sign in pair):
                continue
            edges = [_tangent_edge(reader, network, leg, sign, mouths.get(leg, 0.0)) for leg, _side, sign in pair]
            reach = [_zone_end(leg, side, network) for leg, side, _sign in pair]
            # Past both kerbs: each leg's zone starts past the other's kerb line (_datums).
            depth = sum(reach) + OVERSHOOT_M
            shape = free.intersection(_beside(edges[0], reach[0], depth)).intersection(_beside(edges[1], reach[1], depth))
            ends = [e.point + e.out * out_to for e, out_to in zip(edges, reach, strict=True)]
            meet = centre(*edges, 0.0)                         # where the two edges meet; None where they run on
            corner = [] if meet is None else [meet]
            # The corner rounded to the effective radius on the lane edges (NJDOT 6.4.3, src/turning.py):
            # the turn inside the arc stays the street's.
            if (turn := arc(*edges, EFFECTIVE_RADIUS_M)) is not None:
                shape = shape.difference(Polygon([meet, *turn.coords]))
                corner = list(turn.coords)
            shape = shapely.set_precision(shape, OSM_GRID_M)
            lanes = LineString([ends[0], *corner, ends[1]])    # the travel lanes' edge, round the corner
            # Each piece carries the kerb out to a travel lane, car or bike: against one.
            pieces = [piece for piece in shapely.get_parts(shape) if _paintable(piece)
                      and min(piece.distance(lanes), piece.distance(bikeway)) < SEAM_M]
            if not pieces:
                report["curb extensions not painted: nothing outside the travel lanes there (corners)"] += 1
                continue
            wide = min((w for e, end in zip(edges, ends, strict=True) if (w := _width(free, end, e.inward, depth)) > 0),
                       default=0.0)
            middle = lanes.offset_curve(-wide / 2)
            for piece in pieces:
                ways.append(_closed_way(list(piece.exterior.coords),
                                        {"road_marking": "restriction", "pattern": "solid",
                                         "colour": "tan", "note": EXTENSION_NOTE}, ids, frame, nodes))
                _posts(middle, piece, frame, ids, nodes)
            report["corners with a curb extension, with flexible posts"] += 1
    return nodes, ways


def _width(area: BaseGeometry, point: np.ndarray, inward: np.ndarray, depth: float) -> float:
    """How far `area` runs in from `point` along `inward`, up to `depth`; 0 where it does not start there."""
    ray = LineString([point, point + inward * depth]).intersection(area)
    return next((LineString(part).length for part in line_parts(ray) if Point(part[0]).distance(Point(point)) < SEAM_M), 0.0)


def _posts(path: LineString, area: Polygon, frame: LocalFrame, ids: Iterator[int], nodes: list[NewNode]) -> None:
    """Flexible posts along `path` where it is in `area`, at most BOLLARD_SPACING_M apart, half
    that in from each end."""
    for piece in line_parts(path.intersection(area)):
        middle = LineString(piece)
        count = max(1, math.ceil(middle.length / BOLLARD_SPACING_M))
        for d in (np.arange(count) + 0.5) * middle.length / count:
            nodes.append(NewNode(next(ids), *frame.wgs84(*middle.interpolate(d).coords[0]),
                                 {"barrier": "bollard", "bollard": "flexible"}))


def turns(area: str, change: OsmChange) -> list[tuple[tuple, str, float]]:
    """Every corner's effective turning radius (src/turning.py) in `change`'s world as drawn
    (src/osm_world.py:read): the corner - its junction and its two legs, by street and the junction
    each reaches - its name and the radius. A corner is between a leg and the next anticlockwise, the right
    turn from the first into the second; none where a oneway's traffic cannot make it. The arc may
    not leave the drawn carriageway or touch a post, or cross into a strip a car may stand in."""
    layers, reader = read(area, change)
    network, mouth = reader.graph, dict(_World(area, layers, reader.frame, reader.graph).mouths)
    drivable = reader.carriageway().buffer(SEAM_M)
    # A parked strip shrunk by a seam: its edge is the lane's, which the arc may run along.
    blocked = shapely.union_all([Point(p["position_m"][:2]) for p in reader.out["props"] if p["type"] == "bollard"]
                                + [strip.buffer(-SEAM_M) for strip in reader.parked])
    shapely.prepare(drivable)
    shapely.prepare(blocked)
    found = []
    for junction in sorted(network.junctions):
        for a, b in network.corners(junction):
            if a.outbound or b.inbound or not is_corner(a, b):
                continue
            radius = effective_radius(*(_tangent_edge(reader, network, leg, side, mouth[leg]) for leg, side in ((a, 1), (b, -1))),
                                      drivable, blocked)
            if radius is None:
                continue
            found.append(((junction, a.street, a.end, b.street, b.end), f"{a.street} to {b.street}", radius))
    return found


def _tangent_edge(reader, network: StreetGraph, leg: Leg, side: int, mouth: float) -> Edge:
    """The leg's tangent edge on `side` (+1 its left; src/turning.py): its travel lane's edge as
    the reader drew it where its first piece past the junction's mouth begins, straight along the
    leg's direction there and from the junction out."""
    k = next((k for k in _past_mouth(leg, mouth, network) if leg.steps[k][0] in reader.lane_edges), None)
    line, s = leg.line(), min(mouth, leg.length)
    edge = 0.0
    if k is not None:
        (way, i), (_way, j) = leg.steps[k], leg.steps[k + 1]
        left, right = reader.lane_edges[way]
        forward, at = j > i, 0 if j > i else -1
        on_left = (side > 0) == forward              # the way's own left, where the leg's side is
        edge = float(np.atleast_1d(left if on_left else -right)[at])
    out = tangent(line, s)
    return Edge(_at(line, s, side * edge) - out * s, out, side)    # carried straight back to the junction


def two_way_bikeway(area: str, base: OsmChange, proposal: Proposal) -> tuple[OsmChange, Counter[str]]:
    """`base` with the proposal applied across the whole world - the road diet, continental
    crossings, and the two-way bikeway on the ways `proposal` chooses - laid on streets first
    re-centred between their kerbs (_recentred) - and how much of each. With the proposal's
    `kerb_extensions` (on a `base` from
    existing_markings(kerb_extensions=True), whose crosswalks carry `crossing:kerb_extension`),
    the kerbs are re-signed (_may_park) and each extension painted and posted (_extensions)."""
    report: Counter[str] = Counter()
    kerb_extensions = proposal.kerb_extensions
    base = _recentred(area, base, report, proposal)
    world = _World.of(area, base)
    layers, frame, network, kerbs = world.layers, world.frame, world.network, world.kerbs
    ids = _ids_below(base)
    edits: dict[int, tuple[tuple[int, ...] | None, dict]] = {}   # way -> (new node list, tags)
    nodes: list[NewNode] = []
    created: list[WayChange] = []
    route = _route(network, proposal)
    centre_of: dict[int, tuple[LineString, int, float]] = {}   # way -> (line, sign, centre)
    design = _design(network, route, proposal.kerb, resign=kerb_extensions)
    applied = design[design["applies"]]
    # Each junction leg's mouth - its crosswalk - and the stretch of each way inside it.
    mouths = world.mouths
    inside = _junction_interiors(mouths, network)
    if kerb_extensions:
        kept = _design(network, route, proposal.kerb)
        for side in ("left", "right"):
            lifted = design[f"may_{side}"] & ~kept[f"may_{side}"]
            report["kerbs re-signed: OSM no-parking with no reason, lifted (ft)"] += round(
                (design["length_m"][lifted] / FT_TO_M).sum())
    feet = applied["length_m"] / FT_TO_M
    for track in (TRACK_FT, CONSTRAINED_TRACK_FT):
        report[f"{proposal.name}: {int(track)} ft track (ft)"] = round(
            feet[applied["route"] & (applied["track_ft"] == track)].sum())
    report[f"{proposal.name}: the track does not fit, road diet only (ft)"] = round(
        feet[applied["route"] & (applied["track_ft"] == 0)].sum())
    report["road diet: two 11 ft lanes (ft)"] = round(feet.sum())
    report["road diet: no width tag, on the fallback (ft)"] = round(feet[~applied["width_tagged"]].sum())
    report["road diet: narrower than two 11 ft lanes, left as is (ft)"] = round(
        (design["length_m"][(design["route"] | design["painted"]) & ~design["applies"]] / FT_TO_M).sum())
    for side in ("left", "right"):
        report["kerbs parked (ft)"] += round(feet[applied[f"parking_{side}_ft"] > 0].sum())
        report["kerbs hatched (ft)"] += round(feet[applied[f"shoulder_{side}_ft"] >= MIN_HATCH_FT].sum())

    # Writing it: each way's tags, split where traffic crosses a kerb (_tag_pieces), its stalls
    # laid along each kerb (_stall_layout).
    blocks = _blocks(network, kerbs, applied)
    branches = layers["driveways"] + layers["parking_aisles"]
    crossed = {w: {side: _openings(network.streets[w], network.lines[w], sign, kerbs, branches,
                                   frame, report) for side, sign in (("left", 1), ("right", -1))}
               for w in applied.index}
    stalls = _stall_layout(network, applied, crossed)
    for row in applied.itertuples():
        way, line = network.streets[row.Index], network.lines[row.Index]
        near = "left" if row.track_left else "right"
        far = "right" if near == "left" else "left"
        sign = 1 if near == "left" else -1           # signed offsets are left-positive
        parked = {side: s for side, s in (("left", 1), ("right", -1))
                  if getattr(row, f"parking_{side}_ft")}
        _tag_pieces(way, line, network, kerbs, layers, frame, report, ids, nodes, edits, created,
                    _section_tags(way["tags"], row, proposal), parked=parked, crossed_at=crossed[row.Index],
                    stalls=stalls,
                    track=(near, far, sign) if row.track_ft else None,
                    need_ft=blocks[row.Index][0], least_m=blocks[row.Index][1],
                    inside=_merged(inside.get(row.Index, []), line.length), note=proposal.note)
        # The restriped centre line, left-positive off the way: past the left kerb's spare - and
        # the track and its buffer, where they are on the left - one lane in. Stop lines are laid
        # from it.
        left_spare = row.shoulder_left_ft + row.parking_left_ft + (
            row.track_ft + row.buffer_ft if row.track_ft and near == "left" else 0.0)
        centre_of[row.Index] = (line, 1, (row.width_ft / 2 - left_spare - LANE_FT) * FT_TO_M)

    _track_ends(mouths, network, applied, frame, ids, nodes, created, report, proposal)

    # Stop lines, moved with the centre line: on every restriped street, each across its approach
    # lane only (MUTCD 3B.16), from the new centre line to that lane's edge - never on across the
    # kerbside hatching or parking. Joined to a restriped way at a node, else that of its nearest
    # street (a stop line need not be joined: wiki Key:road_marking).
    restriped = [w for w in centre_of if w in network.streets]
    nearest_of = STRtree([network.lines[w] for w in restriped]) if restriped else None
    for stop_way in layers["stop_lines"]:
        stop = frame.line(stop_way["coords_wgs84"])
        if stop is None or nearest_of is None:
            continue
        on = {w for n in stop_way["node_ids"] for w, _i in network.on_street(n)}
        joined = [w for w in restriped if w in on]
        if not joined and not on:
            joined = [restriped[int(nearest_of.query_nearest(stop.interpolate(0.5, normalized=True))[0])]]
        if not joined:
            continue
        line, sign, centre = centre_of[joined[0]]
        s, offset = _station(line, stop.interpolate(0.5, normalized=True).coords[0])
        edge = centre + (LANE_FT if sign * offset > centre else -LANE_FT) * FT_TO_M
        ends = [NewNode(next(ids), *frame.wgs84(*_at(line, s, sign * o)), {}) for o in (centre, edge)]
        nodes.extend(ends)
        edits[stop_way["id"]] = (tuple(n.id for n in ends), _with_note(stop_way["tags"], proposal.note))

    # A street restriped is restriped whole: paint OSM maps on its carriageway - a hatched area
    # (`road_marking=restriction`, `area:highway=prohibited`) within half its `width` of a
    # restriped way, a flush median say - gives way to the new section, so it is deleted.
    lines_of = [network.lines[w] for w in applied.index]
    halves = [row.width_ft * FT_TO_M / 2 for row in applied.itertuples()]
    on_street = STRtree(lines_of)
    deleted = []
    for marking in layers["road_markings"]:
        ring = frame.line(marking["coords_wgs84"])
        if marking["id"] < 0 or ring is None or len(ring.coords) < 4:
            continue
        middle = Polygon(ring.coords).representative_point()
        if any(lines_of[k].distance(middle) <= halves[k] for k in on_street.query(middle.buffer(max(halves)))):
            deleted.append(WayChange(marking["id"], "delete", (), {}))
            report["painted areas on restriped streets, deleted"] += 1

    # Continental crossings, everywhere: every marked crosswalk, every crossing a junction claims,
    # and every crossing of the bikeway's route.
    crossing_of = {n: c for c in layers["crossings"] for n in c["node_ids"]}
    zebra = {c["id"] for c in layers["crossings"] if _marked(c["tags"])}
    zebra |= {crossing_of[n]["id"] for w in route for n in network.streets[w]["node_ids"] if n in crossing_of}
    for junction in network.junctions:
        for leg in network.legs(junction):
            if (n := _claimed(leg, crossing_of, world.stops)) is not None:
                zebra.add(crossing_of[n]["id"])
    report["crosswalks made continental"] = len(zebra)
    for crossing in layers["crossings"]:
        if crossing["id"] in zebra:
            edits[crossing["id"]] = (None, _with_note(
                {**crossing["tags"], "crossing:markings": "zebra"}, proposal.note))

    by_id = {w.id: w for w in base.ways}
    in_layers = {w["id"]: w for layer in ("roads", "crossings", "stop_lines") for w in layers[layer]}
    for way_id, (node_ids, tags) in edits.items():
        if way_id in by_id:
            by_id[way_id] = dataclasses.replace(
                by_id[way_id], tags=tags, node_ids=node_ids or by_id[way_id].node_ids)
        else:
            by_id[way_id] = WayChange(way_id, "modify",
                                      node_ids or tuple(in_layers[way_id]["node_ids"]), tags)
    if kerb_extensions:
        added, extended = _extensions(area, OsmChange((*base.nodes, *nodes), (*by_id.values(), *created, *deleted)),
                                      ids, report)
        nodes += added
        by_id |= {way.id: way for way in extended}
    return OsmChange((*base.nodes, *nodes), (*by_id.values(), *created, *deleted)), report


# --- writing an area's changes ------------------------------------------------------------------

def write_changes(area: str, proposals: bool = True, into: Path = PROPOSALS_DIR) -> dict[str, Counter[str]]:
    """Write `area`'s existing.osc - every area has one - and, with `proposals`, the .osc of each
    proposal file it has (proposals/<area>/<name>.toml, Proposal), all from the pull now cached,
    to `into`/<area>/; what each found, by file."""
    existing, found = existing_markings(area)
    existing = _recentred(area, existing, found)
    write_change(existing, into / area / "existing.osc")
    reports = {"existing": found}
    today = {key: radius for key, _name, radius in turns(area, existing)} if proposals else {}
    for path in sorted((PROPOSALS_DIR / area).glob("*.toml")) if proposals else []:
        proposal = Proposal.load(path)
        change, report = _proposed(area, proposal)
        # Corners the proposal leaves narrower than today's and under EFFECTIVE_RADIUS_M, by name;
        # today's own are only counted - where OSM maps no kerb the corner is drawn square, so a
        # radius there measures what OSM is missing.
        for key, name, radius in turns(area, change):
            if radius < min(EFFECTIVE_RADIUS_M, today.get(key, 0.0)):
                report[f"corners narrowed under {EFFECTIVE_RADIUS_M / FT_TO_M:g} ft (NJDOT 6.4.3): "
                       f"{name}, {today[key] / FT_TO_M:.0f} to {radius / FT_TO_M:.0f} ft"] += 1
            report[f"corners under {EFFECTIVE_RADIUS_M / FT_TO_M:g} ft effective radius"] += radius < EFFECTIVE_RADIUS_M
        reports[proposal.name] = report
        write_change(change, into / area / f"{proposal.name}.osc")
    return reports


def retagged(area: str, base: OsmChange, proposal: Proposal) -> tuple[OsmChange, Counter[str]]:
    """`base` with each retag rule's tags on every street or crossing way it chooses, and nothing
    else - but that
    where those tags move a side's cycle lane from outside its parking to the kerb
    (parking_at_the_kerb), a painted area at that kerb (`road_marking=restriction`, on the
    carriageway nearest that way) moves out with the parking: square to the way, by the lane's
    width and its buffer, off any node it shares with something built, with any paint it shares one with.
    With `kerb_extensions` (on a `base` whose crosswalks carry `crossing:kerb_extension`), every
    street's travel lanes held at LANE_FT by the shoulders its width leaves (_shoulders), and each
    curb extension painted and posted across its kerb's parking lane and shoulder, outside a cycle
    lane at the kerb (_extensions). Where a rule lays a track (`cycleway:<side>=track`) at the
    kerb, it is carried across each junction, or ends in a turn box (_track_ends)."""
    report: Counter[str] = Counter()
    world = _World.of(area, base)
    layers, frame, network = world.layers, world.frame, world.network
    edits = {way.id: way for way in base.ways}
    streets = [(way, network.lines[way_id]) for way_id, way in network.streets.items()]
    lengths = {way["id"]: line.length for way, line in streets}
    shifts: dict[tuple[int, int], float] = {}                     # (way, side sign) -> metres
    retagged_ids: set[int] = set()
    for way in [way for way, _line in streets] + [c for c in layers["crossings"] if c.get("node_ids")]:
        rules = [tags for ways, tags in proposal.retag if proposal.chooses(way["tags"], ways)]
        lifted = _lifted(way["tags"]) if proposal.kerb_extensions else []
        tags = (_resigned(way["tags"], lifted) if lifted else way["tags"]) | {
            k: v for rule in rules for k, v in rule.items()}
        shoulders = _shoulders(tags) if proposal.kerb_extensions and is_street(tags) else {}
        if not rules and not lifted and not shoulders:
            continue
        tags |= shoulders
        report["kerbs given a shoulder outside LANE_FT travel lanes"] += len(shoulders) // 2
        if lifted:
            report["kerbs re-signed: OSM no-parking with no reason, lifted (ft)"] += round(
                len(lifted) * lengths[way["id"]] / FT_TO_M)
        old = edits.get(way["id"])
        edits[way["id"]] = WayChange(way["id"], old.action if old else "modify",
                                     old.node_ids if old else tuple(way["node_ids"]),
                                     _with_note(tags, proposal.note))
        report[f"{proposal.name}: ways retagged"] += 1
        if rules:
            retagged_ids.add(way["id"])
        for sign, side in ((1, "left"), (-1, "right")):
            if shift := _inside_m(tags, side) - _inside_m(way["tags"], side):
                shifts[way["id"], sign] = shift
    tree = STRtree([line for _way, line in streets])
    nodes = {node.id: node for node in base.nodes}
    ids = _ids_below(base)
    built = Counter(n for name, items in layers.items() if name not in PAINT_LAYERS
                    for item in items if isinstance(item, dict) for n in set(item.get("node_ids") or ()))
    for marking in layers["road_markings"]:
        ring = frame.line(marking["coords_wgs84"])
        if ring is None or len(ring.coords) < 4:
            continue
        middle = Polygon(ring.coords).representative_point()
        way, line = streets[int(tree.nearest(middle))]
        s, offset = _station(line, middle.coords[0])
        sign = 1 if offset > 0 else -1
        if (way["id"], sign) not in shifts or abs(offset) > carriageway_width_m(way["tags"])[0] / 2:
            continue
        # Paint the area shares a node with - a stop line drawn to the hatching's edge - moves with
        # it, so still ends there; anything built - the kerb it was drawn to - stays, and the area
        # takes a new node of its own there.
        move = -sign * shifts[way["id"], sign] * _left(line, s)
        own = {n: next(ids) if built[n] else n for n in marking["node_ids"]}
        for node_id, lonlat in zip(marking["node_ids"], marking["coords_wgs84"], strict=True):
            nodes[own[node_id]] = NewNode(own[node_id], *frame.wgs84(*(np.asarray(frame.point(*lonlat)) + move)),
                                          nodes[node_id].tags if node_id in nodes else {})
        if own.keys() != set(own.values()):
            edits[marking["id"]] = WayChange(marking["id"], "modify", tuple(own[n] for n in marking["node_ids"]),
                                             _with_note(marking["tags"], proposal.note))
        report["painted areas moved out with the parking"] += 1
    # Each kerb's shoulder as built after the retag, read from its tags as the reader lays it
    # (kerb_section), and on the ways the proposal retags a track at the kerb: for _track_ends,
    # which carries the track across each junction.
    rows = {}
    for way_id, way in network.streets.items():
        tags = edits[way_id].tags if way_id in edits else way["tags"]
        row = {"width_ft": carriageway_width_m(tags)[0] / FT_TO_M, "track_ft": 0.0}
        for side in ("left", "right"):
            kerb = kerb_section(tags, side)
            row[f"shoulder_{side}_ft"] = kerb.shoulder_m / FT_TO_M
            if way_id in retagged_ids and tags.get(f"cycleway:{side}") == "track" and kerb.cycleway_at_kerb:
                row |= {"track_ft": width_or_default(tags, f"cycleway:{side}:width", "cycleway")[0] / FT_TO_M,
                        "track_left": side == "left",
                        "track_along": None if tags.get(f"cycleway:{side}:oneway") == "no" else rides_along(tags, side),
                        "track_colour": tags.get(f"cycleway:{side}:surface:colour")}
        rows[way_id] = row
    applied = pd.DataFrame.from_dict(rows, orient="index")
    extended: list[NewNode] = []
    created: list[WayChange] = []
    if applied["track_ft"].any():
        _track_ends(world.mouths, network, applied, frame, ids, extended, created, report, proposal)
    if proposal.kerb_extensions:
        added, ways = _extensions(area, OsmChange(tuple((nodes | {n.id: n for n in extended}).values()),
                                                  tuple((edits | {w.id: w for w in created}).values())), ids, report)
        extended += added
        created += ways
    nodes |= {node.id: node for node in extended}
    edits |= {way.id: way for way in created}
    return OsmChange(tuple(nodes.values()), tuple(edits.values())), report


def _shoulders(tags: dict) -> dict:
    """The `shoulder:<side>=yes` + `shoulder:<side>:width` (wiki Key:shoulder) that hold a way's
    travel lanes at LANE_FT, as the two-way design's section does: each side's half of what its
    lanes (lane_count) and the cycle lane, parking and shoulder its tags lay leave of its `width`,
    to the inch, added to any shoulder it has. None where nothing is left, or where its tags say a
    kerb has no shoulder (`shoulder:<side>=no`), as a piece inside a junction's do (_inside_junction)."""
    if "no" in (sided(tags, "shoulder", side) for side in ("left", "right")):
        return {}
    kerbs = {side: kerb_section(tags, side) for side in ("left", "right")}
    left_m = carriageway_width_m(tags)[0] - lane_count(tags) * LANE_FT * FT_TO_M - sum(
        k.cycleway_m + k.parking_m + k.shoulder_m for k in kerbs.values())
    each_ft = math.floor(left_m / 2 / FT_TO_M * 12) / 12
    if each_ft <= 0:
        return {}
    return {key: value for side, kerb in kerbs.items() for key, value in (
        (f"shoulder:{side}", "yes"), (f"shoulder:{side}:width", _ft(kerb.shoulder_m / FT_TO_M + each_ft)))}


def _inside_m(tags: dict, side: str) -> float:
    """How far a side's cycle lane and its buffer lie between the kerb and its parking: none unless
    the lane is at the kerb."""
    kerb = kerb_section(tags, side)
    return kerb.cycleway_m if kerb.cycleway_at_kerb else 0.0


def _proposed(area: str, proposal: Proposal) -> tuple[OsmChange, Counter[str]]:
    """`proposal` laid on `area`'s existing conditions, with the curb extensions it asks for."""
    base = existing_markings(area, kerb_extensions=proposal.kerb_extensions)[0]
    if proposal.retag:
        return retagged(area, _recentred(area, base, Counter()), proposal)
    return two_way_bikeway(area, base, proposal)
