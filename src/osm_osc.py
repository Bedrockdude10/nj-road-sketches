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
  - where the centre line is not painted: the way split and the stretch tagged
    `lane_markings=no`. Across every marked crossing, its width; and on each leg of each junction,
    from the junction node out to the further of the leg's stop line (the one across the lane
    approaching the junction: traffic keeps right) and the far edge of the junction's crosswalk on
    it; with neither, to where the cross street's kerb line, carried straight across the mouth,
    crosses the leg; with no kerb of any cross street mapped at the junction, UNMARKED_MOUTH_FT.
  - where the law forbids standing: within NO_STANDING_FT of an intersection (N.J.S.A. 39:4-138),
    on both kerbs of each leg, out from the further of the statute's two arms: the far side line
    of the junction's crosswalk on it (any mapped crossing, marked or not), and the cross street's
    side line (its kerb line as above, else half its `width`). The way is split there and tagged
    `parking:<side>=no` + `parking:<side>:restriction=no_standing` +
    `parking:<side>:restriction:reason=junction` (wiki Street parking), which the proposal reads
    like any other restriction.

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
  - on the bikeway's route - Broad Street, BROAD_STREET, the one input that names a street - a
    two-way protected track on the north kerb as the old design ladder had it: TRACK_FT
    (CONSTRAINED_TRACK_FT where only that fits; the road diet alone where neither does), a
    BUFFER_FT buffer with flex posts, both lanes still LANE_FT, all the spare to the south kerb,
    and a 0 ft hatched shoulder outside the track. Where traffic crosses the track the way is
    split and that piece tagged with no posts and dotted green
    (`cycleway:<side>:crossing:markings=dashes`, a project convention: OSM has no tag for
    skip-paint). Through each junction the track is a `highway=cycleway` + `cycleway=crossing`
    way (wiki Tag:cycleway=crossing), straight from its end on one leg to its start on the next;
    the junction stretch carries no track. Restriping moves the centre line, so the route's stop
    lines are moved with it, across their approach lane.
"""
from __future__ import annotations

import bisect
import dataclasses
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon

from src.osm_world import (CARRIAGEWAY, DEFAULT_WIDTHS_M, FT_TO_M, LocalFrame,
                           STATION_STEP_M, Stations, carriageway_width_m, ease, has_centre_line,
                           taper_rate, width_m)
from src.sources.osm_change import NewNode, OsmChange, WayChange, apply_change
from src.sources.osm_context import SNAPSHOT_AREAS, fetch_borough_osm, osm_layers

# NOT OSM: Danny's rule for a junction leg with no stop line, no crosswalk, and no kerb of any
# cross street mapped - the centre line stops this far from the junction node.
UNMARKED_MOUTH_FT = 11.0
NODE_MATCH_M = 0.001              # numerical only: a cut this close to a node is at that node

BROAD_STREET = ("West Broad Street", "East Broad Street")
TRACK_FT, CONSTRAINED_TRACK_FT, BUFFER_FT, LANE_FT = 10.0, 8.0, 3.0, 11.0
MIN_STALL_FT = 7.0                # spare narrower than this is hatched, not parked
PARKING_LANE_FT = 8.0             # a parking lane's depth, where the spare allows
MIN_HATCH_FT = 0.5                # spare narrower than this is left unmarked
STALL_LENGTH_M = 20 * FT_TO_M     # a piece shorter than one parallel stall is not parked
NO_STANDING_FT = 25.0             # R.S. 39:4-138 (e), STANDARDS.md 1: no standing this near an intersection

AREA_NOTE = ("Existing conditions: the street's surface out to its mapped kerbs (each kerb the edge "
             "of the street nearest it), and half its block's width where a side has none, by "
             "src/osm_osc.py. Check before upload.")
EXISTING_NOTE = ("Existing conditions: `width` measured between the mapped kerbs, and where the "
                 "centre line is not painted, from the mapped crossings, stop lines and kerbs at each "
                 "junction, by src/osm_osc.py. Check on the ground or imagery before upload.")
NO_STANDING_NOTE = (f"No standing within {NO_STANDING_FT:g} ft of the nearest crosswalk or the "
                    "intersecting street's side line, N.J.S.A. 39:4-138 - the law, not a sign.")
PROPOSAL_NOTE = ("Proposal: Broad St two-way protected bikeway on the north kerb "
                 "(src/osm_osc.py:two_way_bikeway). Not a survey.")


# --- the street network ------------------------------------------------------------------------

def _street_key(way: dict) -> str:
    """Which street a way is part of: its name, else the way itself."""
    return way["tags"].get("name") or f"way {way['id']}"


def _vertex_stations(line: LineString) -> np.ndarray:
    """Each vertex's distance along `line` - a node's station on its way."""
    xy = np.asarray(line.coords)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


@dataclasses.dataclass(eq=False)
class _Leg:
    """The run of street ways out of a junction, node by node, to the next junction or a dead end."""
    junction: int
    street: str
    steps: list[tuple[int, int]]      # (way id, node index), from the junction out
    nodes: list[int]
    xy: np.ndarray
    d: np.ndarray                     # each step's distance out from the junction along the leg
    end: int | None                   # the junction it reaches; None at a dead end

    @property
    def length(self) -> float:
        return float(self.d[-1])

    def direction(self) -> np.ndarray | None:
        """The leg's first segment out of the junction, as a unit vector."""
        ahead = np.flatnonzero(self.d > 0)
        if not len(ahead):
            return None
        out = self.xy[ahead[0]] - self.xy[0]
        return out / np.linalg.norm(out)

    def line(self) -> LineString:
        keep = np.concatenate([[True], np.diff(self.d) > 0])
        return LineString(self.xy[keep])

    def dark(self, reach: float, stations: dict[int, np.ndarray]) -> list[tuple[int, float, float]]:
        """(way, from, to) stations of each way the leg covers from the junction out to `reach`."""
        out = []
        for k in range(len(self.steps) - 1):
            (way, i), (other, j) = self.steps[k], self.steps[k + 1]
            if way != other or self.d[k + 1] == self.d[k]:
                continue
            if self.d[k] >= reach:
                break
            a, b = stations[way][i], stations[way][j]
            t = min(1.0, (reach - self.d[k]) / (self.d[k + 1] - self.d[k]))
            end = a + (b - a) * t
            out.append((way, min(a, end), max(a, end)))
        return out


class _Network:
    """The streets as a graph of junctions and the legs between them."""

    def __init__(self, layers: dict, frame: LocalFrame):
        self.streets: dict[int, dict] = {}
        self.lines: dict[int, LineString] = {}
        self.stations: dict[int, np.ndarray] = {}
        self.on_node: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.node_xy: dict[int, np.ndarray] = {}
        for way in layers["roads"]:
            line = frame.line(way.get("coords_wgs84") or [])
            node_ids = way.get("node_ids") or []
            if not _is_street(way["tags"]) or line is None or len(node_ids) != len(line.coords):
                continue
            self.streets[way["id"]], self.lines[way["id"]] = way, line
            self.stations[way["id"]] = _vertex_stations(line)
            for i, (node, xy) in enumerate(zip(node_ids, line.coords, strict=True)):
                self.on_node[node].append((way["id"], i))
                self.node_xy[node] = np.asarray(xy)
        self.junctions = {node for node, at in self.on_node.items()
                          if len(self._starts(node)) >= 3
                          and len({_street_key(self.streets[w]) for w, _ in at}) >= 2}
        self._legs: dict[int, list[_Leg]] = {}

    def _starts(self, node: int) -> list[tuple[int, int, int]]:
        return [(way, i, step) for way, i in self.on_node[node] for step in (-1, 1)
                if 0 <= i + step < len(self.streets[way]["node_ids"])]

    def legs(self, junction: int) -> list[_Leg]:
        if junction not in self._legs:
            self._legs[junction] = [self._walk(junction, *start) for start in self._starts(junction)]
        return self._legs[junction]

    def _walk(self, junction: int, way: int, i: int, step: int) -> _Leg:
        steps, seen, end = [(way, i)], {(way, i)}, None
        while True:
            ids = self.streets[way]["node_ids"]
            if 0 <= i + step < len(ids):
                i += step
                if (way, i) in seen:
                    break
                steps.append((way, i))
                seen.add((way, i))
                if ids[i] in self.junctions:
                    end = ids[i]
                    break
                continue
            # Off the end of this way, at a node that is no junction: on along the way ending there.
            on = [(w, k) for w, k in self.on_node[ids[i]] if (w, k) not in seen
                  and k in (0, len(self.streets[w]["node_ids"]) - 1)]
            if not on:
                break
            way, i = on[0]
            step = 1 if i == 0 else -1
            steps.append((way, i))
            seen.add((way, i))
        nodes = [self.streets[w]["node_ids"][k] for w, k in steps]
        xy = np.array([self.node_xy[n] for n in nodes])
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        return _Leg(junction, _street_key(self.streets[steps[0][0]]), steps, nodes, xy, d, end)

    @staticmethod
    def claimed(leg: _Leg, crossing_nodes, stops: _Stops) -> int | None:
        """The crossing node whose crosswalk ends this leg's centre line at its junction: the
        first out along the leg - unless it is the only one on a leg between two junctions and
        nearer the other, whose crosswalk it then is. Never one beyond the stop line serving the
        approach: a stop line is placed in advance of its junction's crosswalk (MUTCD 3B.16, as
        cited), so a crossing past it is mid-block, dark only across its own band."""
        stop_at = _Network.stop_line(leg, stops)
        on_leg = list(dict.fromkeys(
            n for k, n in enumerate(leg.nodes) if k > 0 and n in crossing_nodes and n != leg.end
            and (stop_at is None or float(leg.d[k]) < stop_at)))
        if not on_leg:
            return None
        if leg.end is not None and len(on_leg) == 1:
            out = float(leg.d[leg.nodes.index(on_leg[0])])
            return on_leg[0] if out < leg.length - out else None
        return on_leg[0]

    @staticmethod
    def stop_line(leg: _Leg, stops: _Stops) -> float | None:
        """How far out along `leg` its junction's stop line is: the first that lies across the lane
        approaching the junction - right of that traffic, which keeps right. A stop line joined to
        the leg at a node stands at that node; one joined to no street (wiki Key:road_marking
        allows a separate way) stands where it lies along the leg of its nearest street."""
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
            toward = np.asarray(line.interpolate(max(d - 0.5, 0.0)).coords[0]) - here
            if float(np.dot(np.asarray(middle.coords[0]) - here, [toward[1], -toward[0]])) > 0:
                found.append(float(d))
        return min(found, default=None)

    def kerb_line(self, leg: _Leg, kerbs: _Kerbs) -> float | None:
        """How far out along `leg` the cross street's kerb line, carried straight across the
        mouth, crosses it. On each side of the leg the next leg round the junction, where it is
        another street's; that street's kerb in the corner between the two, at its least distance
        from that street's centreline (carried back across the junction along its first segment);
        and where the leg leaves that distance. The further side; None where neither corner has
        the cross street's kerb mapped."""
        legs = [other for other in self.legs(leg.junction) if other.direction() is not None]
        if leg not in legs:
            return None
        here = leg.xy[0]

        def angle(v: np.ndarray) -> float:
            return math.atan2(v[1], v[0]) % (2 * math.pi)

        ordered = sorted(legs, key=lambda other: angle(other.direction()))
        i = ordered.index(leg)
        own = angle(leg.direction())
        line = leg.line()
        found = []
        for neighbour, ccw in ((ordered[(i + 1) % len(ordered)], True), (ordered[i - 1], False)):
            if neighbour is leg or neighbour.street == leg.street:
                continue
            theirs = angle(neighbour.direction())
            start, span = (own, (theirs - own) % (2 * math.pi)) if ccw else (
                theirs, (own - theirs) % (2 * math.pi))
            cross = LineString([here - neighbour.direction() * leg.length,
                                *neighbour.line().coords])
            half = kerbs.least_offset(neighbour.street, cross, leg.length + neighbour.length,
                                      here, start, span)
            if half is None:
                continue
            inside = [LineString(p) for p in _lines(line.intersection(cross.buffer(half)))]
            for piece in inside:
                at = sorted((line.project(Point(piece.coords[0])), line.project(Point(piece.coords[-1]))))
                if at[0] <= NODE_MATCH_M:
                    found.append(at[1])
        return max(found) if found else None

    def side_line(self, leg: _Leg) -> float:
        """Where the cross street's side line is by its tags, where no kerb of it is mapped: half
        the widest carriageway among the junction's other streets' first ways."""
        return max((carriageway_width_m(self.streets[other.steps[0][0]]["tags"])[0] / 2
                    for other in self.legs(leg.junction) if other.street != leg.street), default=0.0)


def _lines(geometry) -> list[list]:
    return [list(part.coords) for part in getattr(geometry, "geoms", [geometry])
            if isinstance(part, LineString) and len(part.coords) >= 2]


# --- the street's kerbs ------------------------------------------------------------------------

class _Kerbs:
    """The mapped `barrier=kerb` ways, in segments, each the edge of the street whose centreline is
    nearest it (OSM does not say which street a kerb bounds), and which are `kerb=lowered` or
    `flush` and no crossing's ramp - a dropped kerb, where traffic may cross it."""

    def __init__(self, layers: dict, frame: LocalFrame, network: _Network,
                 bbox: tuple[float, float, float, float]):
        segments, lowered = [], []
        for way in layers["kerbs"]:
            line = frame.line(way.get("coords_wgs84") or [])
            if line is None:
                continue
            # A dropped kerb traffic crosses - not a crossing's ramp: one a footway crossing meets
            # carries pedestrians, and its crosswalk's band already cuts the paint there.
            low = (way["tags"].get("kerb") in ("lowered", "flush")
                   and not _meets(way, line, layers["crossings"], frame))
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
        self.owner[nearest[0]] = [_street_key(network.streets[ways[k]]) for k in nearest[1]]
        self.ends = np.array([c for s in segments for c in s.coords])
        self.end_owner = np.repeat(self.owner, 2)
        # A cross-section is cast across the whole area: the first kerb it meets is its edge.
        west, south, east, north = bbox
        self.ray_m = float(np.linalg.norm(np.subtract(frame.point(east, north),
                                                      frame.point(west, south))))

    def hits(self, street: str, stations: Stations, sign: int) -> np.ndarray:
        """Each cross-section's distance on one side to the first kerb it meets, where that kerb is
        `street`'s (NaN where it is another street's, or there is none)."""
        offsets = np.full(len(stations.s), np.nan)
        for i, (point, normal) in enumerate(zip(stations.xy, stations.left, strict=True)):
            ray = LineString([point, point + sign * normal * self.ray_m])
            met = self.tree.query(ray, predicate="intersects")
            if not len(met):
                continue
            far = shapely.distance(Point(point), shapely.intersection(ray, self.segments[met]))
            first = met[int(np.argmin(far))]
            if self.owner[first] == street:
                offsets[i] = float(far.min())
        return offsets

    def least_offset(self, street: str, centre: LineString, along_m: float, apex: np.ndarray,
                     start: float, span: float) -> float | None:
        """The least distance from `centre` of `street`'s kerb vertices in the wedge `span`
        radians anticlockwise from `start` about `apex`, along `centre`'s first `along_m`."""
        mine = self.end_owner == street
        if not mine.any():
            return None
        ends = self.ends[mine]
        turn = (np.arctan2(ends[:, 1] - apex[1], ends[:, 0] - apex[0]) - start) % (2 * math.pi)
        points = shapely.points(ends)
        within = (turn > 0) & (turn < span) & (shapely.line_locate_point(centre, points) <= along_m)
        if not within.any():
            return None
        return float(shapely.distance(points[within], centre).min())


# --- small helpers -----------------------------------------------------------------------------

def _is_street(tags: dict) -> bool:
    return tags.get("highway") in CARRIAGEWAY and tags.get("highway") != "service"


def _has_centre_line(tags: dict) -> bool:
    """A street whose centre line the reader paints (src/osm_world.py:has_centre_line), before
    `lane_markings=no` says where it does not."""
    return _is_street(tags) and has_centre_line(
        {k: v for k, v in tags.items() if (k, v) != ("lane_markings", "no")})


def _marked(tags: dict) -> bool:
    markings = tags.get("crossing:markings")
    return (markings not in (None, "no")
            or (markings is None and tags.get("crossing") in ("marked", "zebra", "uncontrolled")))


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


def _junction_face(network: _Network, kerbs: _Kerbs, junction: int) -> Polygon | None:
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
        closing.append(LineString([stations.xy[i] + stations.left[i] * (left[i] + 0.5),
                                   stations.xy[i] - stations.left[i] * (right[i] + 0.5)]))
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


def _meets(kerb: dict, line: LineString, ways: list[dict], frame: LocalFrame) -> list[LineString]:
    """The `ways` that meet a kerb way - at a node they share, or across it."""
    nodes = set(kerb.get("node_ids") or [])
    return [path for way in ways if (path := frame.line(way.get("coords_wgs84") or [])) is not None
            and (nodes & set(way["node_ids"]) or path.intersects(line))]


def _ramp_gaps(layers: dict, frame: LocalFrame, report: Counter[str]) -> None:
    """Report each lowered kerb a footway crossing meets that has no `wheelchair` tag: the reader
    draws a curb ramp flush only where OSM says `wheelchair=yes`, so these are mapping gaps."""
    for kerb in layers["kerbs"]:
        line = frame.line(kerb.get("coords_wgs84") or [])
        if (line is not None and kerb["tags"].get("kerb") == "lowered" and kerb["tags"].get("wheelchair") is None
                and _meets(kerb, line, layers["crossings"], frame)):
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


def _tangent(line: LineString, s: float) -> np.ndarray:
    a = np.asarray(line.interpolate(max(s - 0.5, 0.0)).coords[0])
    b = np.asarray(line.interpolate(min(s + 0.5, line.length)).coords[0])
    return (b - a) / np.linalg.norm(b - a)


def _station(line: LineString, point) -> tuple[float, float]:
    """`point`'s station along `line`, and its offset from it (left positive)."""
    s = line.project(Point(point))
    t, v = _tangent(line, s), np.asarray(point) - np.asarray(line.interpolate(s).coords[0])
    return s, float(t[0] * v[1] - t[1] * v[0])


def _at(line: LineString, s: float, offset: float) -> np.ndarray:
    """The point `offset` to the left of `line` at station `s`."""
    t = _tangent(line, s)
    return np.asarray(line.interpolate(s).coords[0]) + offset * np.array([-t[1], t[0]])


# --- existing conditions -----------------------------------------------------------------------

class _Stops(defaultdict):
    """Stop lines by each street node they are joined to, and `floating`: (line, nearest street
    way) for each joined to none - which a stop line need not be (wiki Key:road_marking)."""

    def __init__(self) -> None:
        super().__init__(list)
        self.floating: list[tuple[LineString, int]] = []


def _stops(layers: dict, frame: LocalFrame, network: _Network, report: Counter[str]) -> _Stops:
    """Every stop line: by each street node it is joined to, else with its nearest street - the
    street it belongs to as every kerb does (_Kerbs)."""
    stops = _Stops()
    ways = list(network.lines)
    tree = STRtree([network.lines[w] for w in ways])
    for stop_way in layers["stop_lines"]:
        stop = frame.line(stop_way["coords_wgs84"])
        if stop is None:
            continue
        joined = [n for n in stop_way["node_ids"] if n in network.on_node]
        for n in joined:
            stops[n].append(stop)
        if not joined and ways:
            nearest = ways[int(tree.query_nearest(stop.interpolate(0.5, normalized=True))[0])]
            stops.floating.append((stop, nearest))
            report["stop lines on no street, given to their nearest"] += 1
    return stops


def existing_markings(area: str) -> tuple[OsmChange, Counter[str]]:
    """Every street way split where its centre line is not painted, the stretch tagged
    `lane_markings=no`, and each piece's `width` measured between its kerbs - and what each was
    read from, and what OSM is missing."""
    layers = osm_layers(area)
    frame = LocalFrame(SNAPSHOT_AREAS[area])
    network = _Network(layers, frame)
    kerbs = _Kerbs(layers, frame, network, SNAPSHOT_AREAS[area])
    report: Counter[str] = Counter()
    ids = _ids_below(None)
    nodes: list[NewNode] = []
    ways: list[WayChange] = []

    # Where the centre line is not painted: across each marked crossing, and out of each junction.
    crossing_of: dict[int, dict] = {}
    for crossing in layers["crossings"]:
        if _marked(crossing["tags"]):
            crossing_of |= {n: crossing for n in crossing["node_ids"] if n in network.on_node}
    half_of = {n: (width_m(c["tags"].get("width")) or DEFAULT_WIDTHS_M["crossing"]) / 2
               for n, c in crossing_of.items()}
    stops = _stops(layers, frame, network, report)

    intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for node, half in half_of.items():
        for way_id, i in network.on_node[node]:
            at = network.stations[way_id][i]
            intervals[way_id].append((at - half, at + half))
    for junction in network.junctions:
        for leg in network.legs(junction):
            marks = []
            claimed = network.claimed(leg, half_of, stops)
            if claimed is not None:
                k = leg.nodes.index(claimed)
                marks.append((float(leg.d[k]) + half_of[claimed], "the far edge of its crosswalk"))
            stop_at = network.stop_line(leg, stops)
            if stop_at is not None:
                marks.append((stop_at, "its stop line"))
            if marks:
                reach, why = max(marks)
            elif (reach := network.kerb_line(leg, kerbs)) is not None:
                why = "the cross street's kerb line"
            else:
                reach, why = UNMARKED_MOUTH_FT * FT_TO_M, f"{UNMARKED_MOUTH_FT:g} ft, no kerb mapped"
            report[f"junction legs: centre line ends at {why}"] += 1
            for way_id, a, b in leg.dark(min(reach, leg.length), network.stations):
                intervals[way_id].append((a, b))

    # Where the law forbids standing (N.J.S.A. 39:4-138, STANDARDS.md 1): NO_STANDING_FT out along
    # each leg from the further of its two arms - the far side line of the junction's crosswalk on
    # the leg (any mapped crossing: a crosswalk in law, marked or not), and the cross street's side
    # line (its kerb line, else half its width).
    every_half = {n: (width_m(c["tags"].get("width")) or DEFAULT_WIDTHS_M["crossing"]) / 2
                  for c in layers["crossings"] for n in c["node_ids"] if n in network.on_node}
    no_standing: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for junction in network.junctions:
        for leg in network.legs(junction):
            arms = []
            if (claimed := network.claimed(leg, every_half, stops)) is not None:
                arms.append((float(leg.d[leg.nodes.index(claimed)]) + every_half[claimed], "its crosswalk"))
            else:
                report["no-standing legs with no crossing mapped: the crosswalk arm not applied"] += 1
            if (kerb_at := network.kerb_line(leg, kerbs)) is not None:
                arms.append((kerb_at, "the cross street's kerb line"))
            else:
                arms.append((network.side_line(leg), "half the cross street's width"))
            datum, why = max(arms)
            reach = datum + NO_STANDING_FT * FT_TO_M
            report[f"no standing within {NO_STANDING_FT:g} ft of the intersection, from {why} (legs)"] += 1
            stop_at = network.stop_line(leg, stops)
            if stop_at is not None and stop_at > reach:
                report["stop lines further out than the no-standing zone (legs)"] += 1
            for way_id, a, b in leg.dark(min(reach, leg.length), network.stations):
                no_standing[way_id].append((a, b))

    # Each street's kerb to kerb, at every cross-section that meets its own kerb on both sides.
    sides: dict[int, tuple[Stations, np.ndarray, np.ndarray]] = {}   # way -> (stations, left, right)
    across: dict[int, tuple[np.ndarray, np.ndarray]] = {}    # way -> (stations, width or NaN)
    for way_id, way in network.streets.items():
        stations = Stations(network.lines[way_id])
        left = kerbs.hits(_street_key(way), stations, 1)
        right = kerbs.hits(_street_key(way), stations, -1)
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
            if leg.street in BROAD_STREET:
                report["Broad St blocks measured at " + (
                    f"{block_width[key] / FT_TO_M:.0f} ft" if block_width[key] else
                    "nothing: no kerb pair mapped, the width fallback")] += 1

    # The street as it is built: its surface out to its own kerbs, and on a side with none, half its
    # block's width from the way - as `area:highway`, named like the street. Paint is laid from the
    # way's `width`, not from this; the reader keeps it on this surface.
    for way_id, way in network.streets.items():
        stations, left, right = sides[way_id]
        half = np.full(len(stations.s), carriageway_width_m(way["tags"])[0] / 2)
        if not (way["tags"].get("width") or way["tags"].get("width:carriageway")):
            for lo, hi, key in block_at[way_id]:
                if block_width[key]:
                    half[(stations.s >= lo) & (stations.s <= hi)] = block_width[key] / 2
        tags = {"area:highway": way["tags"]["highway"], "note": AREA_NOTE}
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

    # Each way split where its centre line stops and at each junction it runs through - so every
    # piece is one block's - and each piece given its block's width. A `width` OSM has is kept.
    for way_id, way in network.streets.items():
        line = network.lines[way_id]
        dark = (_merged(intervals.get(way_id, []), line.length)
                if _has_centre_line(way["tags"]) and way["tags"].get("lane_markings") != "no" else [])
        tagged = bool(way["tags"].get("width") or way["tags"].get("width:carriageway"))
        junctions = [network.stations[way_id][i] for i, n in enumerate(way["node_ids"])
                     if n in network.junctions and 0 < i < len(way["node_ids"]) - 1]
        widths = [] if tagged else [block_width[key] for _lo, _hi, key in block_at[way_id]]
        zones = _merged(no_standing.get(way_id, []), line.length)
        if not dark and not any(widths) and not zones:
            continue
        report["no-standing zones ending inside an unpainted stretch"] += sum(
            lo < e < hi for span in zones for e in span for lo, hi in dark)
        node_ids, node_st, at = _split(line, way["node_ids"], network.stations[way_id],
                                       [e for span in dark + zones for e in span] + junctions, ids,
                                       frame, nodes)
        for k, (a, b) in enumerate(_pieces(len(node_ids), at)):
            middle = (node_st[a] + node_st[b]) / 2
            tags = dict(way["tags"])
            if any(lo <= middle <= hi for lo, hi in dark):
                tags["lane_markings"] = "no"
            if any(lo <= middle <= hi for lo, hi in zones):
                tags = _with_note(_no_standing(tags), NO_STANDING_NOTE)
                report[f"no standing within {NO_STANDING_FT:g} ft of the intersection (ft of street)"] += round(
                    (node_st[b] - node_st[a]) / FT_TO_M)
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


# --- the Broad St proposal ---------------------------------------------------------------------

PROHIBITIONS = ("no_parking", "no_standing", "no_stopping")


def _no_standing(tags: dict) -> dict:
    """`tags` with neither kerb to be stood at by law: each side `parking:<side>=no` and
    `:restriction=no_standing` for `:reason=junction` - a side already `no_stopping`, the stricter,
    kept - and any `parking:both:*` written out per side so nothing contradicts it."""
    both = {k[len("parking:both"):]: v for k, v in tags.items() if k.startswith("parking:both")}
    out = {k: v for k, v in tags.items() if not k.startswith("parking:both")}
    for side in ("left", "right"):
        for rest, value in both.items():
            out.setdefault(f"parking:{side}{rest}", value)
        out[f"parking:{side}"] = "no"
        if out.get(f"parking:{side}:restriction") != "no_stopping":
            out[f"parking:{side}:restriction"] = "no_standing"
            out[f"parking:{side}:restriction:reason"] = "junction"
    return out


def _may_park(tags: dict, side: str) -> bool:
    """Whether OSM lets this kerb be parked: not `parking:<side>=no|separate`, and no
    `no_parking` / `no_standing` / `no_stopping` restriction (wiki Street parking). Untagged is
    not restricted."""
    kind = tags.get(f"parking:{side}") or tags.get("parking:both")
    restriction = (tags.get(f"parking:{side}:restriction")
                   or tags.get("parking:both:restriction") or "none")
    return kind not in ("no", "separate") and restriction not in PROHIBITIONS


def _openings(way: dict, line: LineString, sign: int, kerbs: _Kerbs, branches: list[dict],
              frame: LocalFrame, report: Counter[str]) -> list[tuple[float, float]]:
    """Stretches of `way` where traffic crosses its edge on one side (`sign` +1 is left): each of
    its street's lowered kerb segments on that side, projected onto the way, and each driveway or
    parking aisle (`branches`) leaving that side from a node it shares with the way where no
    lowered kerb is mapped, its `width` about that node. Where both are mapped the traced kerb is
    the opening: an untagged driveway's width is the service default, a guess at it."""
    out = []
    for k in np.flatnonzero(kerbs.lowered & (kerbs.owner == _street_key(way))):
        ends = [_station(line, c) for c in kerbs.segments[k].coords]
        # Only a kerb beside this way: one past its ends would project onto an end node.
        if not any(0 < s < line.length for s, _ in ends):
            continue
        if all(offset * sign > 0 for _s, offset in ends):
            out.append((min(s for s, _ in ends), max(s for s, _ in ends)))
    lowered = _merged(out, line.length)
    node_st = _vertex_stations(line)
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
                width = width_m(branch["tags"].get("width"))
                if width is None:
                    report["driveways with no width tag, the service default"] += 1
                    width = DEFAULT_WIDTHS_M["service"]
                out.append((at - width / 2, at + width / 2))
            break
    return _merged(out, line.length)


# Where a street is narrower than its section, what gives and how far: the buffer first, to this
# (Danny, 2026-10-07; STANDARDS.md 6c), then the track to CONSTRAINED_TRACK_FT, then parking, then
# the lanes.
MIN_BUFFER_FT = 2.0


class _Samples:
    """Cross-sections of a line at stations `s` - Stations' shape, for _Kerbs.hits."""

    def __init__(self, line: LineString, s: np.ndarray):
        self.s = s
        self.xy = np.array([line.interpolate(at).coords[0] for at in s])
        tangents = [_tangent(line, at) for at in s]
        self.left = np.array([[-t[1], t[0]] for t in tangents])


def _samples(line: LineString, kerbs: _Kerbs, street: str) -> _Samples:
    """Where to measure a way's kerbs: every STATION_STEP_M, and at every vertex of its own kerbs
    - a kerb is straight between its vertices, so together they hold every place it comes
    closest to the way."""
    own = kerbs.ends[kerbs.end_owner == street]
    at = [line.project(Point(c)) for c in own]
    s = np.concatenate([Stations(line).s, [a for a in at if 0 < a < line.length]])
    return _Samples(line, np.unique(np.round(s, 3)))


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
    track = row.track_ft + BUFFER_FT if row.track_ft else 0.0
    return 2 * LANE_FT + track + row.parking_left_ft + row.parking_right_ft


def _route(network: _Network) -> tuple[list[dict], set[int], set[int]]:
    """The bikeway's route - BROAD_STREET's ways - its junctions with other streets, and which of
    its ways are junction stretches (dark, at such a junction), which the cycle crossing carries
    instead of the track."""
    broad = [w for w in network.streets.values() if w["tags"].get("name") in BROAD_STREET]
    others = {n for w in network.streets.values() if w["tags"].get("name") not in BROAD_STREET
              for n in w["node_ids"]}
    junction_nodes = {n for w in broad for n in w["node_ids"] if n in others}
    through = {w["id"] for w in broad if w["tags"].get("lane_markings") == "no"
               and set(w["node_ids"]) & junction_nodes}
    return broad, junction_nodes, through


def _free(spans: list[tuple[float, float]], length: float) -> list[tuple[float, float]]:
    """The stretches of [0, length] outside `spans` (sorted, merged)."""
    out, at = [], 0.0
    for lo, hi in spans:
        if lo > at:
            out.append((at, lo))
        at = max(at, hi)
    return out + ([(at, length)] if length > at else [])


def _tag_pieces(way: dict, line: LineString, network: _Network, kerbs: _Kerbs, layers: dict,
                frame: LocalFrame, report: Counter[str], ids: Iterator[int], nodes: list[NewNode],
                edits: dict, created: list[WayChange], tags: dict, parked: dict[str, int],
                track: tuple[str, str, int] | None = None, need_ft: float = 0.0) -> None:
    """`way` split where traffic crosses a kerb - a lowered kerb, or where none is mapped a
    driveway or parking aisle leaving from that side, its width - and each piece tagged `tags`, but where it is crossed, on
    that side: no hatching on its shoulder, no parking, and across the track (`track` = its side,
    the side its posts face, and its sign) no posts and dotted green. Each parked kerb (`parked`,
    side -> sign) holds, between its openings, as many whole STALL_LENGTH_M stalls as fit,
    centred, with the rest - and any stretch too short for one - hatched: its parking lane given
    to its shoulder. A parked piece's `parking:<side>:capacity` is the stalls that start on it;
    the reader lays them along the whole run (src/osm_world.py:_Reader.stall_all). Where the
    kerbs stand closer than the section needs (`need_ft`, _narrow), the way is split there too
    and that piece narrowed (_narrowed) - the markings give locally, never the whole street."""
    sides = {"left": 1, "right": -1}
    crossed_at = {side: _openings(way, line, sign, kerbs,
                                  layers["driveways"] + layers["parking_aisles"], frame, report)
                  for side, sign in sides.items()}
    starts: dict[str, list[float]] = {}
    runs: dict[str, list[tuple[float, float]]] = {}
    for side in parked:
        starts[side], runs[side] = [], []
        for lo, hi in _free(crossed_at[side], line.length):
            count = int((hi - lo) // STALL_LENGTH_M)
            if count:
                first = lo + ((hi - lo) - count * STALL_LENGTH_M) / 2
                runs[side].append((first, first + count * STALL_LENGTH_M))
                starts[side] += [first + k * STALL_LENGTH_M for k in range(count)]
    narrow = _narrow(line, kerbs, _street_key(way), need_ft * FT_TO_M) if need_ft else []
    # Into and out of each narrowed stretch, on its wider side, a `placement=transition` piece
    # (wiki Key:placement) as long as the MUTCD taper for the most any marking moves - so the
    # markings have finished narrowing where the kerbs pinch them.
    # Each narrowed stretch takes ONE layout end to end - the section narrowed for the most it
    # holds, its parked sides parked - so the markings run straight through it, whichever piece
    # has a stall on it and whichever an opening.
    tapers, layouts = [], []
    for lo, hi, least in narrow:
        short = _section_ft(tags) - least / FT_TO_M
        if short <= 0:
            continue
        layouts.append((lo, hi, _narrowed(dict(tags), short, least, report, round((hi - lo) / FT_TO_M))))
        reach = _shift_m(tags, layouts[-1][2]) / taper_rate(tags)
        if lo - reach < 0 or hi + reach > line.length:
            report["transitions cut short by the end of their way"] += 1
        tapers += [(max(0.0, lo - reach), lo), (hi, min(line.length, hi + reach))]
    cuts = [e for spans in [*crossed_at.values(), *runs.values()] for span in spans for e in span]
    cuts += [e for lo, hi, _least in narrow for e in (lo, hi)] + [e for span in tapers for e in span]
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
        piece_tags = _with_note(piece_tags, PROPOSAL_NOTE)
        piece = tuple(node_ids[a:b + 1])
        if k == 0:
            edits[way["id"]] = (piece, piece_tags)
        else:
            created.append(WayChange(next(ids), "create", piece, piece_tags))


def _section_ft(tags: dict) -> float:
    """What a piece's tagged section needs, kerb to kerb, bar its shoulders: its lanes
    (`width:lanes`), its track and buffer, and each side's parking lane where it is parked."""
    lanes = sum(width_m(w) or 0.0 for w in (tags.get("width:lanes") or "").split("|")) / FT_TO_M
    track = next((key.split(":")[1] for key, v in tags.items()
                  if key.count(":") == 1 and key.startswith("cycleway:") and v == "track"), None)
    extra = sum((width_m(tags.get(f"cycleway:{track}:{part}")) or 0.0) for part in ("width", "buffer")) if track else 0.0
    parking = sum((width_m(tags.get(f"parking:{side}:width")) or 0.0) for side in ("left", "right")
                  if tags.get(f"parking:{side}") == "lane")
    return lanes + (extra + parking) / FT_TO_M


def _shift_m(a: dict, b: dict) -> float:
    """The most any marking moves between two sections of a street: at most half the change in
    `width` plus every change in a width laid across it - shoulders, track, buffer, parking, lanes."""
    keys = [f"{part}:{side}:{size}" for side in ("left", "right")
            for part, size in (("shoulder", "width"), ("cycleway", "width"), ("cycleway", "buffer"),
                               ("parking", "width"))]

    def w(tags: dict, key: str) -> float:
        return width_m(tags.get(key)) or 0.0

    def lanes(tags: dict) -> list[float]:
        return sorted(width_m(v) or 0.0 for v in (tags.get("width:lanes") or "").split("|"))

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


def _design(network: _Network, route: set[int], through: set[int]) -> pd.DataFrame:
    """The proposal as one table over every street way in the world: a row per way, every quantity
    a column, and every rule a vector operation over all of them at once.

    The section, kerb to kerb: two LANE_FT lanes everywhere it is applied; on `route` (Broad St's
    ways, bar `through` - its junction stretches, which the cycle crossing carries) the widest
    track that leaves them at exactly that - TRACK_FT, else CONSTRAINED_TRACK_FT - and its
    BUFFER_FT buffer, on the north kerb, with all the spare to the south; elsewhere the spare
    halved between the kerbs. Each kerb's spare, lane outward: a parking lane against the lane, up
    to PARKING_LANE_FT, where OSM lets that kerb be parked, the spare holds MIN_STALL_FT and the
    way one stall; then the rest a hatched shoulder. Applied to every way on the route, and every
    other whose centre line is painted, that holds two lanes."""
    ways = list(network.streets.values())
    lines = [network.lines[w["id"]] for w in ways]
    d = pd.DataFrame({
        "id": [w["id"] for w in ways],
        "length_m": [line.length for line in lines],
        "width_ft": [carriageway_width_m(w["tags"])[0] / FT_TO_M for w in ways],
        "width_tagged": [carriageway_width_m(w["tags"])[1] == "from OSM" for w in ways],
        "route": [w["id"] in route and w["id"] not in through for w in ways],
        "painted": [_has_centre_line(w["tags"]) and w["tags"].get("lane_markings") != "no"
                    for w in ways],
        "north_left": [Stations(line).left[:, 1].mean() > 0 for line in lines],
        "may_left": [_may_park(w["tags"], "left") for w in ways],
        "may_right": [_may_park(w["tags"], "right") for w in ways],
    })
    room = d["width_ft"]
    on_route = d["route"]
    d["track_ft"] = np.select([on_route & (room >= TRACK_FT + BUFFER_FT + 2 * LANE_FT),
                               on_route & (room >= CONSTRAINED_TRACK_FT + BUFFER_FT + 2 * LANE_FT)],
                              [TRACK_FT, CONSTRAINED_TRACK_FT], 0.0)
    tracked = d["track_ft"] > 0
    spare = room - 2 * LANE_FT - np.where(tracked, d["track_ft"] + BUFFER_FT, 0.0)
    d["applies"] = (on_route | d["painted"]) & (spare >= 0)
    # Which kerb the track takes, and so whose spare is none: the north.
    north_is = {"left": d["north_left"], "right": ~d["north_left"]}
    for side in ("left", "right"):
        mine = np.where(tracked, np.where(north_is[side], 0.0, spare), spare / 2)
        parks = (d[f"may_{side}"] & (mine >= MIN_STALL_FT) & (d["length_m"] >= STALL_LENGTH_M)
                 & ~(tracked & north_is[side]))
        d[f"parking_{side}_ft"] = np.where(parks, np.floor(np.minimum(mine, PARKING_LANE_FT) * 12) / 12, 0.0)
        d[f"shoulder_{side}_ft"] = np.maximum(0.0, np.floor((mine - d[f"parking_{side}_ft"]) * 12) / 12)
    return d.set_index("id")


def _section_tags(osm: dict, row) -> dict:
    """A way's tags with its row of the design written onto them."""
    north = "left" if row.north_left else "right"
    south = "right" if north == "left" else "left"
    tags = {k: v for k, v in osm.items() if not k.startswith(("cycleway", "shoulder", "parking:"))}
    tags |= {"lanes": "2", "width:lanes": f"{_ft(LANE_FT)}|{_ft(LANE_FT)}"}
    if row.track_ft:
        tags |= {f"cycleway:{north}": "track", f"cycleway:{north}:oneway": "no",
                 f"cycleway:{north}:width": _ft(row.track_ft),
                 f"cycleway:{north}:buffer": _ft(BUFFER_FT),
                 f"cycleway:{north}:surface:colour": "green", f"cycleway:{south}": "no"}
    for side in ("left", "right"):
        parking = getattr(row, f"parking_{side}_ft")
        if parking:
            tags |= {f"parking:{side}": "lane", f"parking:{side}:orientation": "parallel",
                     f"parking:{side}:markings": "yes", f"parking:{side}:width": _ft(parking)}
        else:
            # An unparked kerb keeps what OSM says forbids it - the restriction and its reason.
            tags[f"parking:{side}"] = "no"
            tags |= {key.replace("parking:both", f"parking:{side}", 1): v for key, v in osm.items()
                     if key.startswith("parking:both:restriction")}
            tags |= {key: v for key, v in osm.items() if key.startswith(f"parking:{side}:restriction")}
        tags |= {f"shoulder:{side}": "yes", f"shoulder:{side}:markings": "hatched",
                 f"shoulder:{side}:width": _ft(getattr(row, f"shoulder_{side}_ft"))}
    return tags


def _recentred(area: str, base: OsmChange, report: Counter[str]) -> OsmChange:
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
    Junction nodes stay: the cross street shares them. A block with no kerb pair mapped stays
    where OSM has it. Only the proposal moves ways: the existing render draws OSM as it is."""
    layers = apply_change(osm_layers(area), base)
    frame = LocalFrame(SNAPSHOT_AREAS[area])
    network = _Network(layers, frame)
    kerbs = _Kerbs(layers, frame, network, SNAPSHOT_AREAS[area])
    raw = fetch_borough_osm(bbox=SNAPSHOT_AREAS[area])
    raw_nodes = raw["nodes"] if isinstance(raw["nodes"], dict) else {n["id"]: n for n in raw["nodes"]}
    broad, _junction_nodes, through = _route(network)
    design = _design(network, {w["id"] for w in broad}, through)
    need = {way_id: _need_ft(row) * FT_TO_M for way_id, row in design[design["applies"]].iterrows()}
    ids = _ids_below(base)
    moves: dict[int, np.ndarray] = {}
    added: dict[int, list[tuple[int, float, int]]] = defaultdict(list)   # way -> (after index, d, node)
    new_nodes: list[NewNode] = []
    done: set[frozenset] = set()
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
                sampled = _samples(wline, kerbs, _street_key(way))
                left = kerbs.hits(_street_key(way), sampled, 1)
                right = kerbs.hits(_street_key(way), sampled, -1)
                both = np.isfinite(left) & np.isfinite(right)
                middle = sampled.xy + sampled.left * ((left - right) / 2)[:, None]
                offsets += [_station(line, point)[1] for point in middle[both]]
                if way_id not in need:
                    continue
                for lo, hi, _least in _narrow(wline, kerbs, _street_key(way), need[way_id]):
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
            # Off a pinned junction node and on to the block's offset over the same taper.
            rate = taper_rate(network.streets[leg.steps[0][0]]["tags"])
            lead = abs(shift) / rate
            pinned_end = leg.end in network.junctions

            def offset(d: float, shift: float = shift, bumps: list = bumps, lead: float = lead,
                       length: float = line.length, pinned_end: bool = pinned_end) -> float:
                base = shift * min(ease(d / lead) if lead > 0 else 1.0,
                                   ease((length - d) / lead) if pinned_end and lead > 0 else 1.0)
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
            ramps = [(0.0, lead)] + ([(line.length - lead, line.length)] if pinned_end else [])
            ramps += [span for lo, hi, _c, taper in bumps for span in ((lo - taper, lo), (lo, hi), (hi, hi + taper))]

            for k, node in enumerate(leg.nodes):
                if node in (leg.junction, leg.end) or node in network.junctions or node in moves:
                    continue
                moves[node] = _at(line, float(leg.d[k]), offset(float(leg.d[k])))
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
                    new_nodes.append(NewNode(node, *frame.wgs84(*_at(line, d, offset(d))), {}))
                    added[way_a].append((min(i_a, i_b), d if i_b > i_a else -d, node))
    report["nodes moved to re-centre streets"] = len(moves)
    report["nodes added for the tapers on to narrow stretches"] = len(new_nodes)
    kept = [node for node in base.nodes if node.id not in moves]
    moved = [NewNode(node, *frame.wgs84(*xy), raw_nodes.get(node, {}).get("tags") or {}
                     if node > 0 else next(n.tags for n in base.nodes if n.id == node))
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
                                  _with_note(network.streets[way_id]["tags"], PROPOSAL_NOTE)))
    return OsmChange((*kept, *moved, *new_nodes), tuple(ways.values()))


def two_way_bikeway(area: str, base: OsmChange) -> tuple[OsmChange, Counter[str]]:
    """`base` with the proposal applied across the whole world - the road diet, continental
    crossings, and Broad Street's two-way bikeway - laid on streets first re-centred between
    their kerbs (_recentred) - and how much of each."""
    report: Counter[str] = Counter()
    base = _recentred(area, base, report)
    layers = apply_change(osm_layers(area), base)
    frame = LocalFrame(SNAPSHOT_AREAS[area])
    network = _Network(layers, frame)
    kerbs = _Kerbs(layers, frame, network, SNAPSHOT_AREAS[area])
    ids = _ids_below(base)
    edits: dict[int, tuple[tuple[int, ...] | None, dict]] = {}   # way -> (new node list, tags)
    nodes: list[NewNode] = []
    created: list[WayChange] = []
    broad, junction_nodes, through = _route(network)
    track_at: dict[int, tuple[list[float], float]] = {}   # end node -> (track centre, width ft)
    centre_of: dict[int, tuple[LineString, int, float]] = {}   # way -> (line, north sign, centre)
    design = _design(network, {w["id"] for w in broad}, through)
    applied = design[design["applies"]]
    feet = applied["length_m"] / FT_TO_M
    for track in (TRACK_FT, CONSTRAINED_TRACK_FT):
        report[f"Broad St: {int(track)} ft track (ft)"] = round(feet[applied["track_ft"] == track].sum())
    report["Broad St: the track does not fit, road diet only (ft)"] = round(
        feet[applied["route"] & (applied["track_ft"] == 0)].sum())
    report["road diet: two 11 ft lanes (ft)"] = round(feet.sum())
    report["road diet: no width tag, on the fallback (ft)"] = round(feet[~applied["width_tagged"]].sum())
    report["road diet: narrower than two 11 ft lanes, left as is (ft)"] = round(
        (design["length_m"][(design["route"] | design["painted"]) & ~design["applies"]] / FT_TO_M).sum())
    for side in ("left", "right"):
        report["kerbs parked (ft)"] += round(feet[applied[f"parking_{side}_ft"] > 0].sum())
        report["kerbs hatched (ft)"] += round(feet[applied[f"shoulder_{side}_ft"] >= MIN_HATCH_FT].sum())

    # Writing it: each way's tags, split where traffic crosses a kerb (_tag_pieces).
    for row in applied.itertuples():
        way, line = network.streets[row.Index], network.lines[row.Index]
        north = "left" if row.north_left else "right"
        south = "right" if north == "left" else "left"
        sign = 1 if north == "left" else -1          # signed offsets are left-positive
        parked = {side: s for side, s in (("left", 1), ("right", -1))
                  if getattr(row, f"parking_{side}_ft")}
        _tag_pieces(way, line, network, kerbs, layers, frame, report, ids, nodes, edits, created,
                    _section_tags(way["tags"], row), parked=parked,
                    track=(north, south, sign) if row.track_ft else None, need_ft=_need_ft(row))
        if row.track_ft:
            # The track's centre at each end, and the restriped centre line, north-positive off
            # the way: the north edge is half the width out.
            half = row.width_ft * FT_TO_M / 2
            for end, at in ((0, 0.0), (-1, line.length)):
                centre = _at(line, at, sign * (half - row.track_ft * FT_TO_M / 2))
                track_at[way["node_ids"][end]] = (centre.tolist(), row.track_ft)
            centre_of[row.Index] = (line, sign, half - (row.track_ft + BUFFER_FT + LANE_FT) * FT_TO_M)

    # The track through each junction: straight across, from its end on one leg to its start on
    # the next, as the cycle crossing it is. The junction stretch itself carries no track.
    for way in broad:
        if way["id"] in through:
            edits[way["id"]] = (None, _with_note({k: v for k, v in way["tags"].items()
                                                  if not k.startswith("cycleway")}, PROPOSAL_NOTE))
    for junction in junction_nodes:
        pieces = [w for w in broad if w["id"] in through and junction in w["node_ids"]]
        outer = [n for w in pieces for n in (w["node_ids"][0], w["node_ids"][-1])
                 if n != junction and n in track_at]
        if len(outer) < 2:
            continue
        a, b = max(itertools.combinations(outer, 2),
                   key=lambda pair: Point(track_at[pair[0]][0]).distance(Point(track_at[pair[1]][0])))
        ends = [NewNode(next(ids), *frame.wgs84(*track_at[n][0]), {}) for n in (a, b)]
        nodes.extend(ends)
        created.append(WayChange(next(ids), "create", tuple(n.id for n in ends),
                                 {"highway": "cycleway", "cycleway": "crossing", "oneway": "no",
                                  "width": _ft(min(track_at[a][1], track_at[b][1])),
                                  "crossing:markings": "dashes", "surface:colour": "green",
                                  "note": PROPOSAL_NOTE}))

    # Stop lines, moved with the centre line: each joined to a restriped way, across its approach
    # lane, from the new centre line to that lane's edge.
    for stop_way in layers["stop_lines"]:
        stop = frame.line(stop_way["coords_wgs84"])
        joined = [w["id"] for w in broad if w["id"] in centre_of
                  and set(stop_way["node_ids"]) & set(w["node_ids"])]
        if stop is None or not joined:
            continue
        line, sign, centre = centre_of[joined[0]]
        s, offset = _station(line, stop.interpolate(0.5, normalized=True).coords[0])
        edge = centre + (LANE_FT if sign * offset > centre else -LANE_FT) * FT_TO_M
        ends = [NewNode(next(ids), *frame.wgs84(*_at(line, s, sign * o)), {}) for o in (centre, edge)]
        nodes.extend(ends)
        edits[stop_way["id"]] = (tuple(n.id for n in ends), _with_note(stop_way["tags"], PROPOSAL_NOTE))

    # Continental crossings, everywhere: every marked crosswalk, every crossing a junction claims,
    # and every crossing of the bikeway's route.
    crossing_of = {n: c for c in layers["crossings"] for n in c["node_ids"]}
    zebra = {c["id"] for c in layers["crossings"] if _marked(c["tags"])}
    zebra |= {crossing_of[n]["id"] for w in broad for n in w["node_ids"] if n in crossing_of}
    stops = _stops(layers, frame, network, Counter())
    for junction in network.junctions:
        for leg in network.legs(junction):
            if (n := network.claimed(leg, crossing_of, stops)) is not None:
                zebra.add(crossing_of[n]["id"])
    report["crosswalks made continental"] = len(zebra)
    for crossing in layers["crossings"]:
        if crossing["id"] in zebra:
            edits[crossing["id"]] = (None, _with_note(
                {**crossing["tags"], "crossing:markings": "zebra"}, PROPOSAL_NOTE))

    by_id = {w.id: w for w in base.ways}
    in_layers = {w["id"]: w for layer in ("roads", "crossings", "stop_lines") for w in layers[layer]}
    for way_id, (node_ids, tags) in edits.items():
        if way_id in by_id:
            by_id[way_id] = dataclasses.replace(
                by_id[way_id], tags=tags, node_ids=node_ids or by_id[way_id].node_ids)
        else:
            by_id[way_id] = WayChange(way_id, "modify",
                                      node_ids or tuple(in_layers[way_id]["node_ids"]), tags)
    return OsmChange((*base.nodes, *nodes), (*by_id.values(), *created)), report
