"""OSM, read and drawn: an area's cached OSM with a proposal (.osc) applied, as the world JSON
scripts/blender/blender_scene.py `--build` reads.

Everything drawn is something OSM maps, placed where OSM puts it:
  carriageway   a street's `area:highway` polygons where it has them - the street as built -
                else half each way's `width` either side of it. Every marking below is laid from
                the way's `width` (the way runs down the middle of its carriageway), never from a
                kerb, and then kept on that surface: paint stops at the kerb
  bike lanes    `cycleway[:<side>]=lane|track`, a band of `cycleway:<side>:width` inside that
                edge - green where `cycleway:<side>:surface:colour=green`, in skip bars with
                dotted edges where `cycleway:<side>:crossing:markings=dashes` - then
                `cycleway:<side>:buffer`, with a bollard every BOLLARD_SPACING_M down its centre
                where `cycleway:<side>:separation:*=flex_post|bollard`;
                `cycleway:<side>:oneway=no` adds the dashed yellow divider down its middle;
                a `lane` (not a `track`) gets MUTCD's bicycle symbol and arrow at the start of
                each block (_Reader.markings_all)
  shoulder      `shoulder:<side>=yes`, `shoulder:<side>:width` at the edge; where
                `shoulder:<side>:markings=hatched` (a project convention), hatched from its inner
                line out to its street's own `area:highway` edge - everything the kerb as built
                leaves outside the markings; not hatched (a driveway), its lane edge dotted
  parking      `parking:<side>=lane`, an edge line `parking:<side>:width` inside that edge;
                where `parking:<side>:markings=yes`, its `parking:<side>:capacity` stalls marked,
                in equal stalls along each run of parked pieces joined end to end
                `street_side` bays paved beside the carriageway, `on_kerb` parking on the
                pavement beyond it, `half_on_kerb` half on each (_Reader.street_parking)
  lane lines    white where two `lanes` running the same way meet, the travel way shared
                equally among all of them unless `width:lanes` places them; broken, solid where
                `change:lanes` keeps traffic from crossing; none where `lane_markings=no`
  centre line   two-way ways with 2+ lanes, or with `overtaking*` or `lane_markings=yes` tagged
                (has_centre_line), unless `lane_markings=no` - so where it stops is
                where the way is split and tagged, never decided here. Double yellow where
                `overtaking[:forward|:backward]=no`, dashed where `=yes` or untagged. Where the
                backward lanes meet the forward ones - the lanes equal, or as `width:lanes` has
                them, so off the way beside a turn lane - see _Reader.cross_section
  cycle crossings `highway=cycleway` + `cycleway=crossing` ways: a band of their `width`, green
                where `surface:colour=green`; where `crossing:markings=dashes`, edged in
                CROSSBIKE_DASH_M dots and the green in skip bars of the same pattern; a dashed
                yellow divider where `oneway=no`
  sidewalks     `footway=sidewalk` ways, and paved footpaths (_footpath), buffered to their
                `width`, concrete unless `surface=asphalt`
  crossings     `footway=crossing` ways, painted as their `crossing:markings` says, on the
                carriageway only: two edge lines for `lines`/`dashes`/`dots`, bars for `zebra`,
                both for `ladder`; any other kind - `yes` among them - drawn as zebra and counted
  markings      `road_marking=stop_line` bars (STOP_BAR_M wide), `road_marking=restriction`
                hatched areas - filled solid in their `colour`, edged white, where
                `pattern=solid` (a painted curb extension)
  lane arrows   `turn:lanes[:forward|:backward]`: in each lane, MUTCD's arrow for its `left`,
                `through` and `right` - two to a run of ways with the same indications up to a
                junction, one at each end, held clear of what crosses the lane (_Reader.markings_all)
  transitions   a `placement=transition` way (wiki Key:placement) interpolates every width of
                its section from the piece before to the piece after, on an S-curve; every
                other way's layout is constant along it, as its tags are
  kerbs         `barrier=kerb` ways, at the height their `kerb` / `kerb:height` says, and a
                dark red detectable warning pad behind each `tactile_paving=yes` one
  paved ground  driveways, parking aisles, `amenity=parking` areas and `highway=*` areas
                (src/sources/osm_context.py:is_highway_area) - except `parking=lane`
                areas, which are on the carriageway and drawn as their marked outline - and
                a lowered kerb's driveway mouth (_Reader.mouths), and any mapped driveway
                surface (`area:highway=service` + `service=driveway`)
  buildings     their footprints, at `height` / `building:levels`
  surface       a street, its `area:highway`, paved ground or an unpaved path is drawn in its
                `surface`'s material (SURFACE_MATERIAL): concrete, gravel, dirt, else asphalt
  turn boxes    `area:highway=cycleway` + `cycleway=two_stage_box` areas (MUTCD 9E.11): green,
                outlined in white, a bicycle symbol and a through arrow into the bikeway; the
                track's divider stops at the box
  bollards      `barrier=bollard`: a post at a node, and along a way (a row) one at each end
                and evenly between, at most BOLLARD_SPACING_M apart

Where OSM records no width, DEFAULT_WIDTHS_M stands in, and every use of it is counted in
`stats` so a render says how much of it is OSM and how much is the fallback.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyproj
import shapely
import shapely.affinity
from shapely import STRtree, unary_union
from shapely.affinity import scale as _mirror
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, nearest_points, split

from src.geometry.model.crs import NJ_STATE_PLANE_FT, WGS84
from src.sources.osm_change import OsmChange, apply_change, load_change
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers
from src.standards import FIGURES, TO_SI, si

FT_TO_M = TO_SI["ft"]

# What each `surface` value (wiki Key:surface) is drawn in - its material, and the texture
# src/render/theme.py fetches for it. Any other value, or none, is asphalt.
SURFACE_MATERIAL = {**dict.fromkeys(("concrete", "concrete:plates", "concrete:lanes"), "concrete"),
                    **dict.fromkeys(("gravel", "fine_gravel", "compacted", "pebblestone"), "gravel"),
                    **dict.fromkeys(("unpaved", "ground", "dirt", "earth", "mud", "sand"), "dirt")}
UNPAVED = ("gravel", "dirt")
PAVEMENT_KEYS = ("pavement", "pavement_concrete", "pavement_gravel", "pavement_dirt")
ON_FOOT = ("footway", "pedestrian", "path", "track", "bridleway")

# Vehicular highway values (wiki Key:highway, "Roads" and "Link roads"), plus non-driveway service.
CARRIAGEWAY = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
               "residential", "living_street", "service", "motorway_link", "trunk_link",
               "primary_link", "secondary_link", "tertiary_link", "busway"}


def is_street(tags: dict) -> bool:
    """A carriageway that is a street - a junction is where two of them meet - and not a
    `highway=service` way: a driveway, alley or parking aisle leaves a street, it does not cross it."""
    return tags.get("highway") in CARRIAGEWAY and tags.get("highway") != "service"

# Every figure below is declared in standards.toml, with its source and status; read in SI.
# NOT OSM: what is drawn where a way carries no width tag. Counted every time it is used.
DEFAULT_WIDTHS_M = {"lane": si("lane.width"), "service": si("default_width.service"),
                    "cycleway": si("cycleway.width"), "parking": si("parking.lane_depth"),
                    "sidewalk": si("default_width.sidewalk"), "crossing": si("default_width.crossing"),
                    "shoulder": si("default_width.shoulder")}
DEFAULT_BUILDING_HEIGHT_M = si("building.default_height")
# A kerb's reveal above the gutter by `kerb=*`, where `kerb:height` is absent; a curb ramp's.
KERB_HEIGHT_M = {"raised": si("kerb.raised"), "regular": si("kerb.raised"), "rolled": si("kerb.rolled"),
                 "lowered": si("kerb.lowered"), "flush": si("kerb.flush")}
RAMP_HEIGHT_M = si("kerb.ramp")
TACTILE_DEPTH_M, RAMP_WIDTH_M = si("tactile.depth"), si("tactile.width")
SIGNAL_CLEARANCE_M = si("signal.clearance")
STATION_STEP_M, CHUNK_M = si("numerical.station_step"), si("numerical.chunk")
CENTRE_PAIR_OFFSET_M = si("line.double_offset")
DASH_M, DASH_CYCLE_M = si("line.dash"), si("line.dash_cycle")
STOP_BAR_M = si("stop_line.width")
BOLLARD_SPACING_M = si("bollard.spacing")
CROSSBIKE_DASH_M = si("line.dot")
STALL_WIDTH_M, STALL_BODY_M, DIAGONAL_DEG = si("stall.width"), si("stall.length"), si("stall.diagonal_angle")
PARALLEL_STALL_M = si("stall.parallel_length")
SYMBOL_LENGTH_M, SYMBOL_WIDTH_M = si("symbol.length"), si("symbol.width")
TURN_ARROW_CLEAR_M, TURN_ARROW_STEP_M = si("arrow.clearance"), si("arrow.search_step")
ZEBRA_BAR_M = si("crosswalk.bar_width")
HATCH_ANGLE_DEG, HATCH_SPACING_M = si("hatch.angle"), si("hatch.spacing")
HATCH_WIDE_MPH = si("hatch.wide_speed")
SEAM_M, TANGENT_M = si("numerical.seam"), si("numerical.tangent_probe")
SEARCH_STEP_M = si("numerical.search_step")
# The `colour`s a `road_marking=restriction` area is hatched in other than white, each in its own
# channels (`<colour>_hatch_edge_lines`, `<colour>_hatch_stroke_lines`) - the channel is what
# decides a stripe's colour in 3D (scripts/blender/blender_scene.py:PAINT_COLOUR_CHANNELS).
HATCH_COLOURS = ("yellow", "blue")
# The `colour`s a `road_marking=restriction` + `pattern=solid` area is filled in, each its own
# `<colour>_fill_polygons` channel: tan is NYC's painted curb extension (`colour=tan`).
FILL_COLOURS = ("tan",)
# The line channels that are paint, kept on the street surface (_Reader.on_carriageway).
PAINT_LINES = ("bike_lane_edge_lines", "parking_edge_lines", "bike_lane_contraflow_lines",
               "lane_narrowing_edge_lines", "lane_narrowing_hatch_lines", "lane_narrowing_hatch_wide_lines",
               *(f"{c}_hatch_{k}_lines" for c in HATCH_COLOURS for k in ("edge", "stroke")),
               "parking_stall_divider_lines",
               "cycle_crossing_edge_lines",
               "cycle_crossing_divider_lines", "lane_lines")
# A street's own longitudinal paint, which stops where another street's surface begins (_Reader.
# off_other_streets): not across a junction or a side street's mouth.
STREET_PAINT = ("bike_lane_edge_lines", "parking_edge_lines", "lane_lines", "bike_lane_contraflow_lines",
                "lane_narrowing_edge_lines", "bike_lane_surface_polygons")
# Parking restrictions that forbid parking in a bay (wiki Street parking, "Restrictions").
PROHIBITIONS = ("no_parking", "no_standing", "no_stopping")
# Street parking positions off the carriageway (wiki Street_parking, "Parking position").
OFF_CARRIAGEWAY = ("street_side", "on_kerb", "half_on_kerb")

_WIDTH = re.compile(r"""^\s*(?:(?P<ft>\d+(?:\.\d+)?)\s*'\s*(?:(?P<in>\d+(?:\.\d+)?)\s*")?
                       |(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>m|ft)?)\s*$""", re.VERBOSE)


def width_m(value: str | None) -> float | None:
    """An OSM width in metres: `3`, `3.5 m`, `12 ft`, `7'8"` (wiki Key:width). None if absent or
    unreadable."""
    if not value:
        return None
    match = _WIDTH.match(value.split(";")[0])
    if match is None:
        return None
    if match["ft"] is not None:
        return (float(match["ft"]) + float(match["in"] or 0) / 12) * FT_TO_M
    return float(match["num"]) * (FT_TO_M if match["unit"] == "ft" else 1.0)


class LocalFrame:
    """WGS84 -> local metres about the area's centre, through NJ State Plane."""

    def __init__(self, bbox: tuple[float, float, float, float]):
        self._to_ft = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)
        west, south, east, north = bbox
        self.origin_ft = self._to_ft.transform((west + east) / 2, (south + north) / 2)

    def line(self, coords_wgs84: list) -> LineString | None:
        if not coords_wgs84 or len(coords_wgs84) < 2:
            return None
        xs, ys = self._to_ft.transform(*zip(*coords_wgs84, strict=True))
        ox, oy = self.origin_ft
        return LineString([((x - ox) * FT_TO_M, (y - oy) * FT_TO_M)
                           for x, y in zip(xs, ys, strict=True)])

    def point(self, lon: float, lat: float) -> list[float]:
        x, y = self._to_ft.transform(lon, lat)
        return [(x - self.origin_ft[0]) * FT_TO_M, (y - self.origin_ft[1]) * FT_TO_M]

    def wgs84(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Local metres back to (lon, lat), rounded to OSM's 7 decimal places."""
        lon, lat = self._to_ft.transform(x_m / FT_TO_M + self.origin_ft[0],
                                         y_m / FT_TO_M + self.origin_ft[1],
                                         direction=pyproj.enums.TransformDirection.INVERSE)
        return round(lon, 7), round(lat, 7)


def _rings(geometry: BaseGeometry) -> list[list[list[float]]]:
    parts = getattr(geometry, "geoms", [geometry])
    return [[list(p) for p in part.exterior.coords]
            for part in parts if isinstance(part, Polygon) and not part.is_empty]


def _lines(geometry: BaseGeometry) -> list[list[list[float]]]:
    parts = getattr(geometry, "geoms", [geometry])
    return [[list(p) for p in part.coords]
            for part in parts if isinstance(part, LineString) and len(part.coords) >= 2]


def _strip(line: LineString, half_m: float) -> list[list[list[float]]]:
    """`line` widened to `half_m` either side, as hole-free rings. Blender extrudes a ring's
    outline only, so a closed sidewalk around a block, buffered whole, would pave the block it
    encloses; a line whose strip has a hole is drawn as its two halves instead."""
    strip = line.buffer(half_m, cap_style="flat")
    parts = getattr(strip, "geoms", [strip])
    if all(not part.interiors for part in parts if isinstance(part, Polygon)):
        return _rings(strip)
    if len(line.coords) < 3:
        return _rings(strip)
    middle = len(line.coords) // 2
    return (_strip(LineString(line.coords[:middle + 1]), half_m)
            + _strip(LineString(line.coords[middle:]), half_m))


def street_ends(ways: list[dict]) -> dict[tuple[str | None, int], list[dict]]:
    """Each carriageway way by its street and the node at each of its ends, and by (None, node)
    at every node it has - what beyond() and _Reader.transition() walk."""
    ends: dict[tuple[str | None, int], list[dict]] = defaultdict(list)
    for way in ways:
        ids, name = way.get("node_ids") or [], way["tags"].get("name")
        if way["tags"].get("highway") not in CARRIAGEWAY or len(ids) < 2:
            continue
        for node in dict.fromkeys(ids):
            ends[(None, node)].append(way)
        if name:
            ends[(name, ids[0])].append(way)
            ends[(name, ids[-1])].append(way)
    return ends


def beyond(way: dict, ends: dict[tuple[str | None, int], list[dict]],
           frame: LocalFrame) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """At each end of `way`, the point half a metre into the way that carries on from it, else
    None - what Stations spans the join with. That is the one other way of the same street ending
    there (through a junction), else the one other carriageway at a node no third one touches -
    a street that changes its name there, or two unnamed pieces of one."""
    name, ids = way["tags"].get("name"), way.get("node_ids") or []
    out = []
    for node in (ids[0], ids[-1]) if len(ids) >= 2 else ():
        others = [o for o in ends.get((name, node), []) if o["id"] != way["id"]] if name else []
        if len(others) != 1:
            others = [o for o in ends.get((None, node), []) if o["id"] != way["id"]]
            if len(others) != 1 or node not in (others[0]["node_ids"][0], others[0]["node_ids"][-1]):
                others = []
        other = frame.line(others[0]["coords_wgs84"]) if others else None
        if other is None:
            out.append(None)
            continue
        at = min(TANGENT_M, other.length)
        out.append(other.interpolate(at if others[0]["node_ids"][0] == node
                                     else other.length - at).coords[0])
    return (out[0], out[1]) if out else (None, None)


class Stations:
    """Cross-sections along a centreline every STATION_STEP_M: each one's point and left normal.
    `before` / `after` are points on the way the line continues from / into, where there is one:
    an end's normal then spans the join, so two ways of one street share their end cross-section
    and a split draws as no seam."""

    def __init__(self, line: LineString, before=None, after=None):
        length = line.length
        self.s = np.linspace(0.0, length, max(2, math.ceil(length / STATION_STEP_M) + 1))
        self.xy = np.array([line.interpolate(at).coords[0] for at in self.s])
        path = LineString([*([before] if before is not None else []), *line.coords,
                           *([after] if after is not None else [])])
        lead = math.dist(before, line.coords[0]) if before is not None else 0.0
        ahead = np.array([path.interpolate(min(lead + at + TANGENT_M, path.length)).coords[0] for at in self.s])
        behind = np.array([path.interpolate(max(lead + at - TANGENT_M, 0.0)).coords[0] for at in self.s])
        tangent = ahead - behind
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        self.left = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    def at(self, offsets: np.ndarray) -> np.ndarray:
        """The points at signed `offsets` (positive = left of the way's direction)."""
        return self.xy + self.left * offsets[:, None]

    def strips(self, inner: np.ndarray, outer: np.ndarray,
               part: slice = slice(None)) -> list[list[list[float]]]:
        """The band between two signed offset profiles (over the stations in `part`), as hole-free
        rings CHUNK_M long."""
        step = max(1, round(CHUNK_M / STATION_STEP_M))
        a, b, rings = self.at(inner)[part], self.at(outer)[part], []
        for i in range(0, len(a) - 1, step):
            j = min(i + step, len(a) - 1) + 1
            polygon = Polygon([*a[i:j].tolist(), *b[i:j][::-1].tolist()])
            rings += _rings(polygon if polygon.is_valid else polygon.buffer(0))
        return rings

    def line(self, offsets: np.ndarray, part: slice = slice(None)) -> list[list[float]]:
        return self.at(offsets)[part].tolist()


class _Reader:
    def __init__(self, frame: LocalFrame, areas: dict[str | None, list[Polygon]]):
        self.frame = frame
        # Each street's `area:highway` polygons, by name: the street as built, where it is mapped.
        self.areas = areas
        self._own: dict[str, BaseGeometry] = {}
        self.hatched: list[tuple[Polygon, str | None]] = []   # (area, street): hatch_all draws them
        # restriction areas hatched in a colour other than white, by HATCH_COLOURS colour
        self.hatched_coloured: dict[str, list[Polygon]] = defaultdict(list)
        # restriction areas filled solid (`pattern=solid`), by FILL_COLOURS colour
        self.filled: dict[str, list[Polygon]] = defaultdict(list)
        self.parking: list[dict] = []        # marked parking lanes, piece by piece: stall_all draws them
        self.parked: list[Polygon] = []      # every strip a car may stand in, marked or not
        # each way's travel lanes' outer edges, signed offsets (left positive) at its Stations
        self.lane_edges: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._way: dict = {}                 # the way being drawn
        # Each named street's centrelines and widest carriageway: what its hatching is struck along.
        self.street_lines: dict[str, list[LineString]] = defaultdict(list)
        self.street_width: dict[str, float] = defaultdict(float)
        self.street_mph: dict[str, float] = defaultdict(float)     # its highest posted `maxspeed`
        self.crosswalk_bands: list[BaseGeometry] = []   # kept clear of other paint (on_carriageway)
        self.turn_boxes: list[Polygon] = []              # likewise, bar their own markings
        self.stop_bars: list[BaseGeometry] = []          # what a lane's arrows are held clear of
        self.turns: list[dict] = []          # turn:lanes, piece by piece: markings_all draws them
        self.bike_lanes: list[dict] = []     # cycleway=lane, piece by piece: markings_all draws them
        # Per road way: (id, street, {channel: (first, end) index of its paint}, carried across)
        self.paint_of: list[tuple[int, str | None, dict[str, tuple[int, int]], bool]] = []
        # Per road way: its nodes, drawn surface and edges - what a crossing is painted on.
        self.road_nodes: dict[int, set[int]] = {}
        self.road_ways: dict[int, tuple[LineString, list[int], dict]] = {}   # signals() reads them
        self.ends: dict[tuple[str, int], list[dict]] = defaultdict(list)   # (street, end node) -> ways
        self._transition: tuple[dict, dict, float, float] | None = None   # this way's, if one
        self.road_surfaces: dict[int, list[Polygon]] = {}
        self.road_names: dict[int, str | None] = {}
        self.edges: dict[int, tuple[Stations, np.ndarray, np.ndarray]] = {}
        self.node_xy: dict[int, list[float]] = {}
        self.stats: Counter[str] = Counter()
        self.out: dict[str, list] = {
            "pavement": [], "pavement_concrete": [], "pavement_gravel": [], "pavement_dirt": [], "sidewalks": [], "sidewalks_asphalt": [], "buildings": [], "kerbs": [], "paved_surfaces": [],
            "surveyed_crossings": [], "bike_lane_surface_polygons": [], "bike_lane_edge_lines": [],
            "turn_box_surface_polygons": [], "turn_box_edge_lines": [], "bike_lane_symbol_polygons": [],
            "parking_edge_lines": [], "lane_lines": [], "bike_lane_contraflow_lines": [], "lane_narrowing_edge_lines": [],
            "lane_narrowing_hatch_lines": [], "lane_narrowing_hatch_wide_lines": [],
            **{f"{c}_hatch_{k}_lines": [] for c in HATCH_COLOURS for k in ("edge", "stroke")},
            **{f"{c}_fill_polygons": [] for c in FILL_COLOURS},
            "tree_points": [], "props": [],
            "cycle_crossing_surface_polygons": [], "cycle_crossing_edge_lines": [],
            "cycle_crossing_divider_lines": [], "parking_stall_divider_lines": [],
            "tactile_paving_polygons": [], "lane_arrow_polygons": []}

    def width(self, tags: dict, key: str, default: str) -> float:
        found = width_m(tags.get(key))
        if found is not None:
            self.stats[f"{default} width from OSM"] += 1
            return found
        self.stats[f"{default} width DEFAULTED"] += 1
        return DEFAULT_WIDTHS_M[default]

    def index_ends(self, ways: list[dict]) -> None:
        """Each named way by the street and the node at each of its ends - transition() walks it."""
        self.ends = street_ends(ways)

    def beyond(self, way: dict) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        return beyond(way, self.ends, self.frame)

    def transition(self, way: dict) -> tuple[dict, dict, float, float] | None:
        """For a `placement=transition` way (wiki Key:placement: the street's layout changes along
        it, from the way before to the way after): the tags of the first piece of the same street
        that is not a transition on each side - as seen from this way, its left and right swapped
        where the run turns round - how far along the whole transition run this way starts, and
        the run's length. Ways split inside one transition (at a driveway) are one run."""
        if way["tags"].get("placement") != "transition":
            return None
        name, ids = way["tags"].get("name"), way.get("node_ids") or []
        if not name or len(ids) < 2:
            return None
        here = self.frame.line(way["coords_wgs84"])
        ends, lengths = [], []
        for node in (ids[0], ids[-1]):
            current, at, flipped, walked = way, node, False, 0.0
            while True:
                others = [o for o in self.ends[(name, at)] if o["id"] != current["id"]]
                if len(others) != 1:
                    ends.append(None)
                    break
                other = others[0]
                o_ids = other["node_ids"]
                # turned round where both pieces start, or both end, at the shared node
                flipped ^= (o_ids[0] == at) == (current["node_ids"][0] == at)
                if other["tags"].get("placement") != "transition":
                    tags = other["tags"]
                    ends.append({_flip(k): v for k, v in tags.items()} | {"_reversed": "yes"}
                                if flipped else dict(tags))
                    break
                walked += self.frame.line(other["coords_wgs84"]).length
                current, at = other, (o_ids[-1] if o_ids[0] == at else o_ids[0])
            lengths.append(walked)
        if None in ends:
            self.stats["transitions with no plain piece at an end: drawn as tagged"] += 1
            return None
        return ends[0], ends[1], lengths[0], lengths[0] + here.length + lengths[1]

    def along(self, own: float, theirs, stations: Stations) -> np.ndarray:
        """`own` at every station - a way's layout is constant along it, as its tags are - except
        on a `placement=transition` way, where each value runs from the piece before's
        (`theirs(tags)`) to the piece after's on an S-curve along the transition run."""
        out = np.full(len(stations.s), own, dtype=float)
        if self._transition is None:
            return out
        before, after, start, run = self._transition
        a, b = theirs(before), theirs(after)
        a = own if a is None else a
        b = own if b is None else b
        if run > 0:
            out[:] = a + (b - a) * ease((start + stations.s) / run)
        return out

    def eased(self, tags: dict, key: str, default: str, stations: Stations) -> np.ndarray:
        """A width tag's value (else its default) at every station, eased into the neighbours'."""
        return self.along(self.width(tags, key, default), lambda t: width_m(t.get(key)), stations)

    def carriageway_width(self, tags: dict) -> float:
        width, source = carriageway_width_m(tags)
        self.stats[f"carriageway width {source}"] += 1
        return width

    def road(self, way: dict) -> None:
        tags = way["tags"]
        line = self.frame.line(way["coords_wgs84"])
        if line is None or tags.get("highway") not in CARRIAGEWAY or tags.get("area") == "yes":
            return
        if tags.get("service") in ("driveway", "parking_aisle"):
            return  # paved_surfaces draws these
        stations = Stations(line, *self.beyond(way))
        self._way = way
        self._transition = self.transition(way)
        nodes = way.get("node_ids") or []
        self.node_xy |= {node: self.frame.point(*c) for node, c in zip(nodes, way["coords_wgs84"],
                                                                       strict=False)}
        # The way runs down the middle of its carriageway (wiki Key:placement, the default), so
        # its edges are half its `width` either side, and every marking is laid from them.
        half = self.along(self.carriageway_width(tags) / 2,
                          lambda t: (width_m(t.get("width")) or width_m(t.get("width:carriageway")) or 0) / 2
                          or None, stations)
        edges = {"left": half, "right": half}
        surface = stations.strips(-half, half)
        if tags.get("name") not in self.areas or tags.get("highway") == "service":
            # where its street is mapped, the area is the pavement
            self.out[_pavement_key(tags)] += surface
        self.road_nodes[way["id"]] = set(nodes)
        self.road_ways[way["id"]] = (line, list(nodes), tags)
        self.road_names[way["id"]] = tags.get("name")
        if tags.get("name"):
            self.street_lines[tags["name"]].append(line)
            self.street_mph[tags["name"]] = max(self.street_mph[tags["name"]], _mph(tags.get("maxspeed")))
            self.street_width[tags["name"]] = max(self.street_width[tags["name"]],
                                                  carriageway_width_m(tags)[0])
        self.road_surfaces[way["id"]] = [Polygon(ring) for ring in surface]
        self.edges[way["id"]] = (stations, half, half)
        marks = {key: len(self.out[key]) for key in STREET_PAINT}
        centre = self.cross_section(tags, line, stations, edges)
        if has_centre_line(tags):
            self.stripe_centre(tags, centre)
        carried = any(k.startswith("cycleway:") and k.endswith(":crossing:markings") and v == "dashes"
                      for k, v in tags.items())
        self.paint_of.append((way["id"], tags.get("name"),
                              {key: (marks[key], len(self.out[key])) for key in STREET_PAINT}, carried))

    def cross_section(self, tags: dict, line: LineString, stations: Stations,
                      edges: dict[str, np.ndarray]) -> LineString:
        """Draw the way's side features and return its centre line.

        Without `width:lanes` each side's shoulder, cycleway, buffer and parking are laid from its
        edge inward, every lane shares what is left equally, and the centre line is where the
        backward lanes meet the forward ones. With every lane's width (wiki Key:width:lanes,
        left to right) the section is laid IN ORDER from the edge whose side carries a cycleway
        (else the left): its cycleway, buffer and parking, then the lanes, then the far side's
        parking lane against the last lane, and its shoulder from there out - so the shoulder's
        hatching starts where the section ends wherever the widths are eased (a transition), never
        over a stall. A far side with a cycleway of its own is laid from its own edge."""
        lanes = _int(tags.get("lanes")) or 0
        widths = [width_m(w) for w in (tags.get("width:lanes") or "").split("|")]
        backward = backward_lanes(tags, lanes)
        if len(widths) != lanes or None in widths:
            left, right = (self.kerbside(tags, stations, side, edges[side]) for side in ("left", "right"))
            self.lane_edges[self._way["id"]] = (left, -right)
            if lanes < 1:
                return line
            # Every lane shares the travel way equally - nothing says otherwise without
            # `width:lanes` - so a two-way way's centre line is where its backward lanes meet its
            # forward ones: off the way where more run one way than the other (a turn lane), or
            # where the two sides' kerbside features differ. Lane j's left edge, counting lanes left
            # to right from 0; the last lane's right edge is `every[lanes]`.
            oneway = tags.get("oneway") in ("yes", "-1")
            every = [left - j * (left + right) / lanes for j in range(lanes + 1)]
            self.lane_lines(tags, stations, lanes, backward,
                            {j: every[j] for j in range(1, lanes) if oneway or j != backward})
            self.turn_lanes(tags, stations, lanes, backward, every)
            if oneway or not 0 < backward < lanes:
                return line
            return LineString(stations.line(every[backward]))
        def lane(i: int):
            def theirs(t: dict) -> float | None:
                values = [width_m(v) for v in (t.get("width:lanes") or "").split("|")]
                values = values[::-1] if t.get("_reversed") else values
                return values[i] if len(values) == len(widths) and values[i] is not None else None
            return theirs
        widths = [self.along(w, lane(i), stations) for i, w in enumerate(widths)]
        anchor = "right" if has_cycleway(tags, "right") and not has_cycleway(tags, "left") else "left"
        far = "left" if anchor == "right" else "right"
        sign = 1 if anchor == "left" else -1
        inner = self.kerbside(tags, stations, anchor, edges[anchor])
        lane_edge = inner - sum(widths)
        self.lane_edges[self._way["id"]] = (inner, lane_edge) if anchor == "left" else (-lane_edge, -inner)
        outer = lane_edge
        if (tags.get(f"parking:{far}") or tags.get("parking:both")) == "lane":
            outer = lane_edge - self.eased(tags, f"parking:{far}:width", "parking", stations)
            self.out["parking_edge_lines"].append(stations.line(sign * lane_edge))
            self.stalls(tags, far, stations, sign * lane_edge, sign * outer)
        self.kerbside(tags, stations, far, edges[far], parking=False,
                      shoulder_from=None if has_cycleway(tags, far) else -outer)
        near = widths[:backward] if anchor == "left" else widths[backward:]
        leftmost = sign * inner + (sum(widths) if anchor == "right" else 0)
        lefts = [leftmost - sum(widths[:j]) for j in range(lanes + 1)]   # each lane's left edge
        self.lane_lines(tags, stations, lanes, backward,
                        {j: lefts[j] for j in range(1, lanes)
                         if tags.get("oneway") in ("yes", "-1") or j != backward})
        self.turn_lanes(tags, stations, lanes, backward, lefts)
        # Each side's lane edge, where nothing else marks it: a hatched shoulder's outline and a
        # parking lane's edge already do, and a track's buffer line does. That leaves a shoulder
        # whose hatching breaks where traffic crosses the kerb - the edge line is maintained across
        # a driveway (MUTCD 3B.11(09)), dotted as through a conflict area (3B.11(10)), so a solid
        # line is never mistaken for a parking lane's or a hatched area's edge.
        for side, edge_at in ((anchor, inner), (far, inner - sum(widths))):
            marked = (tags.get(f"shoulder:{side}:markings") == "hatched" or has_cycleway(tags, side)
                      or (tags.get(f"parking:{side}") or tags.get("parking:both")) == "lane")
            if not marked and (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
                self.out["lane_narrowing_edge_lines"] += _dotted(LineString(stations.line(sign * edge_at)))
                self.stats["lane edge lines carried across an opening (pieces)"] += 1
        self.stats["cross-section laid by width:lanes (ways)"] += 1
        return LineString(stations.line(sign * (inner - sum(near))))

    def kerbside(self, tags: dict, stations: Stations, side: str, edge: np.ndarray,
                 parking: bool = True, shoulder_from: np.ndarray | None = None) -> np.ndarray:
        """Draw one side's shoulder, cycleway, its buffer (with its posts) and, if `parking`, its
        parking lane, edge inward, and return the travel way's edge on that side. The shoulder's
        inner line is `shoulder_from` where given (cross_section's far side), else its width in."""
        sign = 1 if side == "left" else -1
        edge = self.street_parking(tags, side, stations, edge)
        if (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
            # A shoulder at the edge (wiki Key:shoulder), `shoulder:<side>:width` wide; hatched
            # where `shoulder:<side>:markings=hatched` - a project convention, as OSM has no tag
            # for painted hatching along a way.
            inner = (shoulder_from if shoulder_from is not None
                     else edge - self.eased(tags, f"shoulder:{side}:width", "shoulder", stations))
            if tags.get(f"shoulder:{side}:markings") == "hatched":
                # A shoulder runs to the edge of the street as built: from its inner line out to
                # wherever its own street's `area:highway` ends - the kerb - so whatever is left
                # outside every marking is hatched. Cast out a full carriageway width, which
                # reaches past any kerb of this street; without an area, to `width`'s edge.
                own = self.own_surface(tags.get("name"))
                reach = edge + carriageway_width_m(tags)[0] if own is not None else edge
                for ring in stations.strips(sign * inner, sign * reach):
                    band = Polygon(ring).buffer(0)
                    for part in _polygons(band.intersection(own) if own is not None else band):
                        self.hatched.append((part, tags.get("name")))
            edge = inner
        section = kerb_section(tags, side)
        parked = parking and section.parking_m > 0
        if parked and section.cycleway_m and not section.cycleway_at_kerb:
            edge, parked = self.parking_lane(tags, side, stations, edge), False
        if section.cycleway_m:
            inner = edge - self.eased(tags, f"cycleway:{side}:width", "cycleway", stations)
            green = tags.get(f"cycleway:{side}:surface:colour") == "green"
            crossed = tags.get(f"cycleway:{side}:crossing:markings") == "dashes"
            edge_line = LineString(stations.line(sign * inner))
            if crossed:
                # Where traffic crosses the track the green is skipped and the edge dotted, in
                # the dotted line's pattern (MUTCD IA-14, as cited).
                if green:
                    self.out["bike_lane_surface_polygons"] += _skip_bars(
                        LineString(stations.line(sign * (edge + inner) / 2)),
                        float(np.mean(edge - inner)) / 2)
                self.out["bike_lane_edge_lines"] += _dotted(edge_line)
            else:
                if green:
                    self.out["bike_lane_surface_polygons"] += stations.strips(sign * inner, sign * edge)
                self.out["bike_lane_edge_lines"] += _lines(edge_line)
            if tags.get(f"cycleway:{side}:oneway") == "no":
                self.out["bike_lane_contraflow_lines"] += _dashes(
                    LineString(stations.line(sign * (edge + inner) / 2)))
            if (tags.get(f"cycleway:{side}") or tags.get("cycleway:both") or tags.get("cycleway")) == "lane":
                self.bike_lane(tags, side, stations, sign * (edge + inner) / 2, float(np.min(edge - inner)))
            edge = inner
            buffer = width_m(tags.get(f"cycleway:{side}:buffer"))
            if buffer:
                buffer = self.along(buffer, lambda t: width_m(t.get(f"cycleway:{side}:buffer")), stations)
                if any(v in ("flex_post", "bollard") for k, v in tags.items()
                       if k.startswith(f"cycleway:{side}:separation")):
                    posts = LineString(stations.line(sign * (edge - buffer / 2)))
                    self.out["props"] += [
                        {"type": "bollard", "heading_deg": 0.0,
                         "position_m": list(posts.interpolate(at).coords[0])}
                        for at in np.arange(BOLLARD_SPACING_M / 2, posts.length, BOLLARD_SPACING_M)]
                edge = edge - buffer
                buffer_line = LineString(stations.line(sign * edge))
                self.out["bike_lane_edge_lines"] += _dotted(buffer_line) if crossed else _lines(buffer_line)
        if parked:
            edge = self.parking_lane(tags, side, stations, edge)
        return edge

    def parking_lane(self, tags: dict, side: str, stations: Stations, edge: np.ndarray) -> np.ndarray:
        """A parking lane from `edge` inward - as deep as parking_depth_m says - its inner edge
        line and its stalls; return its inner edge. An angled stall's lines stop at its length
        (stall_all), short of that edge, so the bay is open to the lane it is driven into from."""
        sign = 1 if side == "left" else -1
        depth = parking_depth_m(tags, side)
        width = (self.along(depth, lambda t: parking_depth_m(t, side), stations) if depth
                 else self.eased(tags, f"parking:{side}:width", "parking", stations))
        outer, edge = edge, edge - width
        self.out["parking_edge_lines"].append(stations.line(sign * edge))
        self.stalls(tags, side, stations, sign * edge, sign * outer)
        return edge

    def street_parking(self, tags: dict, side: str, stations: Stations, edge: np.ndarray) -> np.ndarray:
        """Street parking that is not a lane on the carriageway, by its position (wiki
        Street_parking): `street_side` bays paved beside the carriageway, `on_kerb` on the
        pavement beyond it, `half_on_kerb` half on each - as deep as parking_depth_m says. Return
        the carriageway's edge left for the rest of the section: half the depth in, for parking
        astride the kerb."""
        position = _parking(tags, side)
        if position not in OFF_CARRIAGEWAY:
            return edge
        sign = 1 if side == "left" else -1
        depth = parking_depth_m(tags, side)
        depth = (self.along(depth, lambda t: parking_depth_m(t, side), stations) if depth
                 else self.eased(tags, f"parking:{side}:width", "parking", stations))
        inside = depth / 2 if position == "half_on_kerb" else np.zeros_like(depth)
        rings = stations.strips(sign * edge, sign * (edge + depth - inside))
        if position == "street_side":
            surface = tags.get(f"parking:{side}:surface") or tags.get("parking:both:surface")
            self.out["paved_surfaces"] += [_paving(ring, {"surface": surface}) for ring in rings]
        else:
            self.out["sidewalks"] += rings
        self.stats[f"parking={position} (way sides)"] += 1
        if (tags.get(f"parking:{side}:markings") or tags.get("parking:both:markings")) == "yes":
            # Paint is kept on the carriageway (on_carriageway), so stalls off it are not drawn.
            self.stats[f"parking={position} markings: not drawn off the carriageway"] += 1
        return edge - inside

    def lane_lines(self, tags: dict, stations: Stations, lanes: int, backward: int,
                   bounds: dict[int, np.ndarray]) -> None:
        """A lane line where two lanes running the same way meet - lane j's left boundary at
        signed offset `bounds[j]`, lanes counted left to right - white (MUTCD 3B.06):
        broken, solid where either lane's `change:lanes` keeps traffic from crossing it, or with no
        `change:lanes`, between a turn-only lane (its `turn:lanes` excludes `through`) and one that
        is not."""
        if tags.get("lane_markings") == "no":
            return
        changes = lane_changes(tags, lanes, backward)
        turns = [_turn_only(v) for v in lane_turns(tags, lanes, backward)]
        for j, offset in bounds.items():
            line = LineString(stations.line(offset))
            solid = changes[j - 1] in NO_CHANGE_RIGHT or changes[j] in NO_CHANGE_LEFT
            if changes[j - 1] is None and changes[j] is None:
                # With no `change:lanes` to say: solid between a turn-only lane and one that is not
                # (MUTCD 3B.04(25), a through lane beside an added mandatory turn lane).
                solid = turns[j - 1] != turns[j]
            self.out["lane_lines"] += _lines(line) if solid else _dashes(line)
            self.stats[f"lane lines ({'solid' if solid else 'broken'}, pieces of way)"] += 1

    def bike_lane(self, tags: dict, side: str, stations: Stations, centre: np.ndarray,
                  width_m: float) -> None:
        """A `cycleway:<side>=lane` piece, kept for markings_all: its centre line in the direction
        its riders travel - the traffic's on that side of the way, unless `cycleway:<side>:oneway`
        says otherwise - and its width. A two-way lane (`:oneway=no`) gets its symbol, no arrow."""
        nodes = self._way.get("node_ids") or []
        if len(nodes) < 2:
            return
        oneway, own = tags.get("oneway"), tags.get(f"cycleway:{side}:oneway")
        along = oneway == "yes" or (oneway != "-1" and side == "right")
        along = not along if own == "-1" else along
        line = stations.line(centre)
        self.bike_lanes.append({
            "street": tags.get("name"), "id": self._way["id"], "values": ("bike lane", own == "no"),
            "first": nodes[0] if along else nodes[-1], "last": nodes[-1] if along else nodes[0],
            "centres": [line if along else line[::-1]], "widths": [width_m]})

    def turn_lanes(self, tags: dict, stations: Stations, lanes: int, backward: int,
                   bounds: list[np.ndarray]) -> None:
        """This way's `turn:lanes` (wiki Key:turn), kept for markings_all: in each direction that has
        them, its lanes as their drivers see them, left to right - each lane's indications, its
        centre line in the direction of travel and its width. `bounds[j]` is lane j's left edge,
        lanes counted left to right across the way, `bounds[lanes]` the last one's right edge.
        A oneway's are `turn:lanes`; a two-way way's `turn:lanes:forward` and `:backward`, the
        backward lanes listed from the point of view of their drivers, as the wiki has it."""
        nodes = self._way.get("node_ids") or []
        if len(nodes) < 2 or lanes < 1:
            return
        if tags.get("oneway") in ("yes", "-1"):
            groups = [("turn:lanes", list(range(lanes)), tags["oneway"] == "-1")]
        else:
            groups = [("turn:lanes:forward", list(range(backward, lanes)), False),
                      ("turn:lanes:backward", list(range(backward)), True)]
            if "turn:lanes" in tags:
                self.stats["turn:lanes on a two-way way, no :forward/:backward: not drawn"] += 1
        for key, here, against in groups:
            if key not in tags:
                continue
            values = tags[key].split("|")
            if len(values) != len(here):
                self.stats[f"{key} not one value per lane: not drawn"] += 1
                continue
            order = here[::-1] if against else here
            centres = [stations.line((bounds[i] + bounds[i + 1]) / 2) for i in order]
            self.turns.append({
                "street": tags.get("name"), "id": self._way["id"], "values": tuple(values),
                "first": nodes[-1] if against else nodes[0], "last": nodes[0] if against else nodes[-1],
                "centres": [c[::-1] if against else c for c in centres],
                "widths": [float(np.min(bounds[i] - bounds[i + 1])) for i in order]})

    def stalls(self, tags: dict, side: str, stations: Stations, one: np.ndarray,
               other: np.ndarray) -> None:
        """A marked parking lane on this way's piece, kept for stall_all: its two edges and its
        `parking:<side>:capacity`. Any lane a car may stand in, marked or not, is kept in `parked`."""
        markings = tags.get(f"parking:{side}:markings") or tags.get("parking:both:markings")
        restriction = tags.get(f"parking:{side}:restriction") or tags.get("parking:both:restriction")
        nodes = self._way.get("node_ids") or []
        if restriction not in PROHIBITIONS:
            self.parked += [Polygon(ring) for ring in stations.strips(one, other)]
        if markings != "yes" or restriction in PROHIBITIONS or len(nodes) < 2:
            return
        # Which way traffic passes this kerb, along the way: a diagonal stall leans with it.
        oneway = tags.get("oneway")
        ahead = -1 if oneway == "-1" or (oneway != "yes" and side == "left") else 1
        direction = tags.get(f"parking:{side}:direction") or tags.get("parking:both:direction") or "head_in"
        self.parking.append({"street": tags.get("name"), "side": side, "first": nodes[0],
                             "last": nodes[-1], "capacity": _int(tags.get(f"parking:{side}:capacity")),
                             "orientation": _orientation(tags, side),
                             "lean": ahead * (1 if direction == "head_in" else -1),
                             "one": stations.line(one), "other": stations.line(other)})

    def stall_all(self) -> None:
        """Every marked parking lane's stalls, run by run: its pieces - a way split for any other
        reason - joined end to end, and the run's capacity, summed (else as many standard stalls as
        it holds, stall_pitch_m), laid in equal stalls along it, a stall line at each end of each. Stalls are laid only where no painted area is - a `road_marking=restriction`
        of any colour, or a hatched shoulder - so the hatching's edge is where they start."""
        painted = unary_union([part for part, _name in self.hatched]
                              + [part for parts in self.hatched_coloured.values() for part in parts]
                              + [part for parts in self.filled.values() for part in parts])
        painted = painted.buffer(-SEARCH_STEP_M)     # a divider on its edge touches; it does not overlap
        shapely.prepare(painted)
        after = {(p["street"], p["side"], p["first"]): p for p in self.parking}
        has_before = {(p["street"], p["side"], p["last"]) for p in self.parking}
        for start in self.parking:
            if (start["street"], start["side"], start["first"]) in has_before:
                continue
            run, piece, seen = [], start, set()
            while piece is not None and id(piece) not in seen:
                seen.add(id(piece))
                run.append(piece)
                piece = after.get((piece["street"], piece["side"], piece["last"]))
            one = LineString([c for k, p in enumerate(run) for c in p["one"][(k > 0):]])
            other = LineString([c for k, p in enumerate(run) for c in p["other"][(k > 0):]])
            # An angled divider leans with its stall: its kerb end lies along the street from its
            # lane end by the lane's depth there / tan(angle) - with traffic for a head-in bay,
            # against it for a back-in one - and the stalls fit between the run's ends.
            angle = {"diagonal": DIAGONAL_DEG, "perpendicular": 90.0}.get(run[0]["orientation"])

            def lean(s: float, run: list = run, angle: float | None = angle,
                     one: LineString = one, other: LineString = other) -> float:
                if angle is None:
                    return 0.0
                return run[0]["lean"] * other.distance(one.interpolate(s)) / math.tan(math.radians(angle))

            def divider(s: float, one: LineString = one, other: LineString = other, lean=lean,
                        angled: bool = angle is not None) -> LineString:
                """The stall line at `s`: from the kerb toward the lane - an angled one the
                stall's length along it (stall.length), the lane beyond open; a parallel one
                across the lane."""
                lane = np.asarray(one.interpolate(s).coords[0])
                kerb = np.asarray(other.interpolate(other.project(Point(lane)) + lean(s)).coords[0])
                reach = float(np.linalg.norm(lane - kerb))
                if angled and reach > STALL_BODY_M:
                    lane = kerb + (lane - kerb) * STALL_BODY_M / reach
                return LineString([lane, kerb])

            first, last = max(0.0, -lean(0.0)), one.length - max(0.0, lean(one.length))
            if last <= first:
                continue
            # The stretches of the run whose dividers cross no painted area, searched along it.
            free = [(first, last)]
            lane_strip = Polygon([*one.coords, *other.coords[::-1]]).buffer(0)
            if painted.intersects(lane_strip):
                at = np.append(np.arange(first, last, SEARCH_STEP_M), last)
                clear = np.array([not painted.intersects(divider(float(s))) for s in at])
                edges = np.flatnonzero(np.diff(np.concatenate([[0], clear.astype(int), [0]])))
                free = [(float(at[a]), float(at[b - 1])) for a, b in zip(edges[::2], edges[1::2], strict=True)
                        if b - 1 > a]
                self.stats["marked parking beside a painted area: stalls kept off it (runs)"] += 1
            tagged = sum(p["capacity"] or 0 for p in run)
            if not tagged:
                self.stats["marked parking with no capacity: standard stalls (runs)"] += 1
            total = sum(b - a for a, b in free)
            for a, b in free:
                # A tagged `capacity` shared over the free stretches by length; else as many
                # standard stalls as each holds.
                capacity = (round(tagged * (b - a) / total) if tagged and total > 0
                            else int((b - a) // stall_pitch_m(run[0]["orientation"])))
                for k in range(capacity + 1 if capacity else 0):
                    line = divider(a + (b - a) * k / capacity)
                    self.out["parking_stall_divider_lines"].append([list(c) for c in line.coords])

    def runs(self, pieces: list[dict]) -> list[list[dict]]:
        """`pieces` joined end to end into runs: the same street and `values`, past nodes that are
        not junctions - no road but the two of them through it - so a run ends at a junction,
        which is how far the wiki's turn indication reaches ("from the first indication ... to the
        junction") and where a bike lane begins again (MUTCD 9C.04(03))."""
        through: dict[int, set[int]] = defaultdict(set)   # node -> the drawn roads through it
        for way_id, (_line, nodes, _tags) in self.road_ways.items():
            for node in nodes:
                through[node].add(way_id)
        starting: dict[tuple, list[dict]] = defaultdict(list)
        for piece in pieces:
            starting[(piece["street"], piece["values"], piece["first"])].append(piece)

        def following(piece: dict) -> dict | None:
            nexts = [p for p in starting[(piece["street"], piece["values"], piece["last"])]
                     if p["id"] != piece["id"]]
            if len(nexts) == 1 and through[piece["last"]] == {piece["id"], nexts[0]["id"]}:
                return nexts[0]
            return None

        continued = {id(nxt) for piece in pieces if (nxt := following(piece)) is not None}
        runs = []
        for start in pieces:
            if id(start) in continued:
                continue
            run: list[dict] = []
            current: dict | None = start
            while current is not None and all(current is not p for p in run):
                run.append(current)
                current = following(current)
            runs.append(run)
        return runs

    def markings_all(self) -> None:
        """Every lane's markings, run by run (runs), drawn after the crossings and stop lines that
        they are held clear of (lane_markings):
          turn:lanes   each lane's lane-use arrow (MUTCD 3B.20(21)), one at the run's upstream end,
                       its first indication, and one at its downstream end; where the run holds
                       only one, the upstream one (3B.20(22): a short lane may omit the other)
          cycleway=lane  the bike lane symbol and its arrow at the run's upstream end - the
                       beginning of the lane (MUTCD 9C.04(03))."""
        surfaces = [(way_id, polygon) for way_id, polygons in self.road_surfaces.items()
                    for polygon in polygons]
        tree = STRtree([polygon for _way_id, polygon in surfaces])

        def along(run: list[dict], k: int) -> LineString:
            return LineString([c for n, p in enumerate(run) for c in p["centres"][k][(n > 0):]])

        for run in self.runs(self.turns):
            ids = {p["id"] for p in run}
            for k, value in enumerate(run[0]["values"]):
                indications = {v for v in value.split(";") if v not in ("", "none")}
                if not indications:
                    continue
                glyph = lane_use_arrow(indications)
                if glyph is None:
                    self.stats[f"turn:lanes `{value}`: no arrow drawn for it (lanes)"] += 1
                    continue
                self.lane_markings(_polygons(glyph), along(run, k), min(p["widths"][k] for p in run),
                                   ids, tree, surfaces, "lane_arrow_polygons", "lane-use arrows",
                                   both_ends=True)
        for run in self.runs(self.bike_lanes):
            self.lane_markings(bike_lane_marking(arrow=not run[0]["values"][1]), along(run, 0),
                               min(p["widths"][0] for p in run), {p["id"] for p in run}, tree, surfaces,
                               "bike_lane_symbol_polygons", "bike lane symbols", both_ends=False)

    def lane_markings(self, parts: list[Polygon], lane: LineString, width_m: float, ids: set[int],
                      tree: STRtree, surfaces: list[tuple[int, Polygon]], channel: str, label: str,
                      both_ends: bool) -> None:
        """One lane's marking (markings_all): `parts` in metres, x along the lane from the tail and
        y to its left, laid along `lane`, its centre line in the direction of travel, `width_m`
        wide, and scaled down to that width where it is wider (MUTCD 3B.20(11)); all of it on the
        street as drawn, which the way's `width` can overstate where the kerbs close in. At the run's
        upstream end and, `both_ends`, its downstream end; each held arrow.clearance along the lane
        clear of anything across it - a crosswalk, a stop line, another road's surface (`ids` are
        the ways the lane runs on, whose own surface is no obstacle)."""
        min_x, min_y, max_x, max_y = unary_union(parts).bounds
        to_m = min(1.0, width_m / (max_y - min_y))
        if to_m < 1.0:
            self.stats[f"{label} scaled to their lane's width"] += 1
        length, half = (max_x - min_x) * to_m, (max_y - min_y) * to_m / 2
        if lane.length < length:
            self.stats[f"{label}: lanes too short for one"] += 1
            return
        reach = lane.buffer(half, cap_style="flat")
        across = [shape for shape in (*self.crosswalk_bands, *self.stop_bars) if shape.intersects(reach)]
        across += [surfaces[i][1] for i in tree.query(reach) if surfaces[i][0] not in ids]
        obstacle = unary_union(across) if across else None
        if obstacle is not None:
            shapely.prepare(obstacle)

        street = self.carriageway().buffer(SEAM_M)
        shapely.prepare(street)

        def clear(tail: float) -> bool:
            # All of it on the street as drawn - paint is kept on the street - and held clear.
            if not street.contains(_substring(lane, tail, tail + length).buffer(half, cap_style="flat")):
                return False
            if obstacle is None:
                return True
            under = _substring(lane, max(0.0, tail - TURN_ARROW_CLEAR_M),
                               min(lane.length, tail + length + TURN_ARROW_CLEAR_M))
            return not obstacle.intersects(under.buffer(half, cap_style="flat"))

        # Searched from each end of the run, starting AT it, so a marking with nothing to clear is
        # at the end exactly.
        steps = np.arange(0.0, lane.length - length + 1e-9, TURN_ARROW_STEP_M)
        upstream = next((t for t in steps if clear(t)), None)
        if upstream is None:
            self.stats[f"{label}: lanes with no room clear of the junction"] += 1
            return
        placed = [upstream]
        if both_ends:
            # A stop line across this lane - one crossing its centre line - is what the arrow
            # nearest the junction stands upstream of (MUTCD 3B.20(21)), not the gap beyond it.
            stop = min((lane.project(Point(c)) for bar in self.stop_bars if bar.intersects(lane)
                        for piece in _lines(bar.intersection(lane)) for c in piece), default=lane.length)
            last = min(lane.length, stop - TURN_ARROW_CLEAR_M) - length
            downstream = next((t for t in last - steps if t >= 0 and clear(t)), None)
            if downstream is not None and downstream >= upstream + length:
                placed.append(downstream)
            else:
                self.stats[f"{label}: one, a short lane (MUTCD 3B.20(22))"] += 1
        middle = (min_y + max_y) / 2
        for tail in placed:
            a = np.asarray(lane.interpolate(tail).coords[0])
            b = np.asarray(lane.interpolate(tail + length).coords[0])
            ahead = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            left = np.array([-ahead[1], ahead[0]])
            for part in parts:
                self.out[channel].append([list(a + (x - min_x) * to_m * ahead + (y - middle) * to_m * left)
                                          for x, y in part.exterior.coords])
            self.stats[label] += 1

    def cycle_crossing(self, way: dict) -> None:
        tags, line = way["tags"], self.frame.line(way["coords_wgs84"])
        if line is None:
            return
        # Its own channels; where it runs through a crosswalk, the crosswalk cuts through it
        # (on_carriageway).
        half = self.width(tags, "width", "cycleway") / 2
        dotted = tags.get("crossing:markings") == "dashes"
        if tags.get("surface:colour") == "green":
            self.out["cycle_crossing_surface_polygons"] += (_skip_bars(line, half) if dotted
                                                            else _strip(line, half))
        for offset in (half, -half):
            edge = line.offset_curve(offset)
            self.out["cycle_crossing_edge_lines"] += _dotted(edge) if dotted else _lines(edge)
        if tags.get("oneway") == "no":
            self.out["cycle_crossing_divider_lines"] += _dashes(line)

    def bollard(self, item: dict) -> None:
        """`barrier=bollard`: a post at a node; along a way (a row of them), a post at each end
        and as many between, evenly, as keep them at most BOLLARD_SPACING_M apart."""
        if not item.get("coords_wgs84"):
            if "lon" in item:
                self.out["props"].append({"type": "bollard", "heading_deg": 0.0,
                                          "position_m": list(self.frame.point(item["lon"], item["lat"]))})
            return
        line = self.frame.line(item["coords_wgs84"])
        if line is None:
            return
        count = max(2, math.ceil(line.length / BOLLARD_SPACING_M) + 1)
        self.out["props"] += [{"type": "bollard", "heading_deg": 0.0,
                               "position_m": list(line.interpolate(at).coords[0])}
                              for at in np.linspace(0.0, line.length, count)]
        self.stats["bollard rows (barrier=bollard ways)"] += 1

    def sidewalk(self, way: dict) -> None:
        """A sidewalk or paved footpath, at kerb height: concrete unless `surface=asphalt`."""
        line = self.frame.line(way["coords_wgs84"])
        if line is not None:
            half = self.width(way["tags"], "width", "sidewalk") / 2
            key = "sidewalks_asphalt" if way["tags"].get("surface") == "asphalt" else "sidewalks"
            self.out[key] += _strip(line, half)

    def turn_box(self, box: Polygon) -> None:
        """A two-stage turn box (`cycleway=two_stage_box`, MUTCD 9E.11): green all over (12), a
        solid white line on all four sides (07), and a bicycle symbol beside a THROUGH arrow (05,
        06 - through, for a two-way bikeway), both pointing along its length into the bikeway: the
        way the nearest of the track's green outside it lies. Drawn after the roads."""
        self.turn_boxes.append(box)
        self.out["turn_box_surface_polygons"] += _rings(box)
        self.out["turn_box_edge_lines"] += _lines(box.exterior)
        corners = np.asarray(box.minimum_rotated_rectangle.exterior.coords)[:3]
        sides = np.diff(corners, axis=0)
        along = sides[np.argmax(np.linalg.norm(sides, axis=1))]
        along = along / np.linalg.norm(along)
        centre = np.asarray(box.centroid.coords[0])
        green = unary_union([Polygon(r).buffer(0) for r in self.out["bike_lane_surface_polygons"]
                             if len(r) >= 4]).difference(box.buffer(STATION_STEP_M))
        if not green.is_empty:
            toward = np.asarray(nearest_points(box.centroid, green)[1].coords[0]) - centre
            along = along if float(np.dot(toward, along)) >= 0 else -along
        across = np.array([-along[1], along[0]])
        width = float(np.min(np.linalg.norm(sides, axis=1)))
        nose, half = SYMBOL_LENGTH_M / 2, SYMBOL_WIDTH_M / 2
        symbol, arrow = ([(a * nose, c * half) for a, c in si(f"symbol.{name}_outline")]
                         for name in ("bicycle", "arrow"))
        for outline, side in ((symbol, 1), (arrow, -1)):
            middle = centre + across * side * width / 4
            self.out["bike_lane_symbol_polygons"].append(
                [list(middle + a * along + c * across) for a, c in [*outline, outline[0]]])
        self.stats["two-stage turn boxes"] += 1

    def trail(self, way: dict) -> None:
        """An unpaved path or track, at grade: its ground (gravel or dirt) to its `width`."""
        line = self.frame.line(way["coords_wgs84"])
        if line is not None:
            default = "service" if way["tags"].get("highway") == "track" else "sidewalk"
            half = self.width(way["tags"], "width", default) / 2
            self.out["paved_surfaces"] += [_paving(ring, way["tags"]) for ring in _strip(line, half)]

    def crossing(self, way: dict) -> None:
        tags, line = way["tags"], self.frame.line(way["coords_wgs84"])
        markings = tags.get("crossing:markings") or (
            "zebra" if tags.get("crossing") in ("marked", "zebra", "uncontrolled") else None)
        if line is None or markings in (None, "no"):
            return
        half = self.width(tags, "width", "crossing") / 2
        # Painted on the road it crosses - the one it shares a node with - and nowhere else, so
        # at a junction its lines stop at that road's edge rather than running on into the next.
        crossed = [polygon for road_id, nodes in self.road_nodes.items()
                   if nodes & set(way.get("node_ids") or [])
                   for polygon in self.road_surfaces[road_id] + self.areas.get(self.road_names[road_id], [])]
        road = unary_union(crossed) if crossed else self.carriageway()
        # Painted straight across that road: the chord from where the crossing's way enters its
        # surface to where it leaves it - so a kerb ramp mapped at an angle at either end bends
        # nothing.
        at = sorted(line.project(Point(c)) for piece in _lines(line.intersection(road))
                    for c in (piece[0], piece[-1]))
        if not at or at[-1] - at[0] <= 0:
            return
        chord = LineString([line.interpolate(at[0]), line.interpolate(at[-1])])
        # Its band is the road's own asphalt under the bars: no other paint is laid in it.
        self.crosswalk_bands.append(chord.buffer(half, cap_style="flat").intersection(road))
        # The two lines along its edges: all of a `lines` / `dashes` / `dots` crossing, and a
        # `ladder`'s rails either side of its bars (wiki Key:crossing:markings).
        edges = [c for g in (chord.offset_curve(half).intersection(road),
                             chord.offset_curve(-half).intersection(road)) for c in _lines(g)]
        if markings in ("lines", "dashes", "dots"):
            self.out["surveyed_crossings"].append({"lines": edges})
            return
        if markings not in ("zebra", "ladder"):
            # `yes` (marked, kind unknown) and kinds not drawn here (`zebra:paired`, ...)
            self.stats[f"crossings crossing:markings={markings}: drawn as zebra"] += 1
        elif "crossing:markings" not in tags:
            self.stats[f"crossings crossing={tags.get('crossing')}, no crossing:markings: drawn as zebra"] += 1
        # zebra and its variants: bars square to the chord, one bar width apart, as many as fit,
        # centred on it so the gaps at either kerb are equal.
        count = int((chord.length + ZEBRA_BAR_M) // (2 * ZEBRA_BAR_M))
        start = (chord.length - (2 * count - 1) * ZEBRA_BAR_M) / 2
        bars = []
        for i in range(count):
            piece = _substring(chord, start + 2 * i * ZEBRA_BAR_M, start + (2 * i + 1) * ZEBRA_BAR_M)
            bars += [ring for bar in _strip(piece, half)
                     for ring in _rings(Polygon(bar).intersection(road))]
        self.out["surveyed_crossings"].append({"bars": bars, **({"lines": edges} if markings == "ladder" else {})})

    def own_surface(self, name: str | None) -> BaseGeometry | None:
        """A named street's own `area:highway` polygons, as one shape - not the junctions'."""
        if name is None or name not in self.areas:
            return None
        if name not in self._own:
            self._own[name] = unary_union(self.areas[name])
        return self._own[name]

    def carriageway(self) -> BaseGeometry:
        """The drawn street surface, as one shape. Built once, after the roads."""
        if not hasattr(self, "_carriageway"):
            self._carriageway = unary_union([part for key in PAVEMENT_KEYS
                                             for ring in self.out[key] if len(ring) >= 4
                                             for part in _polygons(Polygon(ring).buffer(0))])
            shapely.prepare(self._carriageway)
        return self._carriageway

    def off_other_streets(self) -> None:
        """Each way's own lines and green (STREET_PAINT) cut out of every other street's surface,
        so a street's markings stop where the street it meets begins - none across a junction or
        a side street's mouth (is_street: a driveway or alley mapped as `highway=service` cuts
        nothing - the street's lines run on past it). Not a piece tagged as carried across the conflict
        (`cycleway:<side>:crossing:markings=dashes`), whose dotted lines are drawn there on purpose.
        Another way of the same street is not another street."""
        surfaces = [(way_id, polygon) for way_id, polygons in self.road_surfaces.items()
                    if is_street(self.road_ways[way_id][2]) for polygon in polygons]
        if not surfaces:
            return
        tree = STRtree([polygon for _way_id, polygon in surfaces])
        replaced: dict[str, dict[int, list]] = defaultdict(dict)
        for way_id, street, spans, carried in self.paint_of:
            if carried:
                continue
            for key, (first, end) in spans.items():
                for i in range(first, end):
                    item = self.out[key][i]
                    shape = Polygon(item) if key.endswith("_polygons") else LineString(item)
                    if not shape.is_valid or shape.is_empty:
                        continue
                    others = [surfaces[k][1] for k in tree.query(shape)
                              if surfaces[k][0] != way_id
                              and (street is None or self.road_names[surfaces[k][0]] != street)]
                    if not others:
                        continue
                    left = shape.difference(unary_union(others))
                    replaced[key][i] = _rings(left) if key.endswith("_polygons") else _lines(left)
                    self.stats["markings cut where another street begins (pieces)"] += 1
        for key, parts in replaced.items():
            self.out[key] = [piece for i, item in enumerate(self.out[key])
                             for piece in parts.get(i, [item])]

    def on_carriageway(self) -> None:
        """Paint is on the street and off the crosswalks: every marking clipped to the drawn
        surface, so where the street as built is narrower than its `width` the paint stops at its
        kerb, and cut out of every crosswalk's band, which a crosswalk cuts through - asphalt
        under its bars, the bikeway's green and lines stopping either side."""
        road = self.carriageway().difference(unary_union(self.crosswalk_bands))
        shapely.prepare(road)
        # A flush median (`colour=yellow` restriction area) is outlined by its own yellow lines
        # (MUTCD 3B.24, as cited), which carry the centre line past it: no centre line across it.
        if self.hatched_coloured.get("yellow"):
            median = unary_union(self.hatched_coloured["yellow"])
            self.out["bike_lane_contraflow_lines"] = [
                part for line in self.out["bike_lane_contraflow_lines"] if len(line) >= 2
                for part in _lines(LineString(line).difference(median))]
        # Inside a turn box, nothing but the box's own line, symbol and arrow: the lines along
        # its sides stay (a hair inside them is cut), the track's divider stops at it.
        lined = road.difference(unary_union([box.buffer(-SEAM_M) for box in self.turn_boxes]))
        shapely.prepare(lined)
        for key in PAINT_LINES:
            clipped = [part for line in self.out[key] if len(line) >= 2
                       for part in _lines(LineString(line).intersection(lined))]
            self.out[key] = ([[part[0], part[-1]] for part in clipped]
                             if key in ("lane_narrowing_hatch_lines", "lane_narrowing_hatch_wide_lines",
                                        *(f"{c}_hatch_stroke_lines" for c in HATCH_COLOURS),
                                        "parking_stall_divider_lines")
                             else clipped)
        for key in ("bike_lane_surface_polygons", "cycle_crossing_surface_polygons",
                    "turn_box_surface_polygons", "bike_lane_symbol_polygons", "lane_arrow_polygons",
                    *(f"{c}_fill_polygons" for c in FILL_COLOURS)):
            self.out[key] = [ring for poly in self.out[key] if len(poly) >= 4
                             for ring in _rings(Polygon(poly).buffer(0).intersection(road))]
        self.out["turn_box_edge_lines"] = [part for line in self.out["turn_box_edge_lines"]
                                           for part in _lines(LineString(line).intersection(road))]
        self.out["props"] = [prop for prop in self.out["props"] if prop["type"] != "bollard"
                             or road.contains(Point(prop["position_m"]))]

    def stripe_centre(self, tags: dict, line: LineString) -> None:
        both = tags.get("overtaking")
        forward = tags.get("overtaking:forward", both)
        backward = tags.get("overtaking:backward", both)
        untagged = forward is None and backward is None
        self.stats["centre line: overtaking " + ("not tagged" if untagged else "from OSM") + " (pieces)"] += 1
        if "no" not in (forward, backward):
            self.out["bike_lane_contraflow_lines"] += _dashes(line)
            return
        # Forward traffic keeps right of the way, so its stripe is the one on the right.
        for allowed, offset in ((forward, -CENTRE_PAIR_OFFSET_M), (backward, CENTRE_PAIR_OFFSET_M)):
            stripe = line.offset_curve(offset)
            self.out["bike_lane_contraflow_lines"] += (_lines(stripe) if allowed == "no"
                                                      else _dashes(stripe))

    def stop_line(self, way: dict) -> None:
        line = self.frame.line(way["coords_wgs84"])
        if line is not None:
            bar = line.buffer(STOP_BAR_M / 2, cap_style="flat").intersection(self.carriageway())
            self.stop_bars.append(bar)
            self.out["surveyed_crossings"].append({"bars": _rings(bar)})

    def restriction(self, way: dict) -> None:
        """A `road_marking=restriction` area, hatched in its `colour` - one of HATCH_COLOURS, else
        white, the colour of a marking with none tagged - or with `pattern=solid`, filled in it,
        one of FILL_COLOURS."""
        line = self.frame.line(way["coords_wgs84"])
        if line is not None and len(line.coords) >= 4:
            # On a street - its centre inside that street's own surface - it is struck along that
            # street, so it hatches as one with the street's shoulders (hatch_all).
            polygons = _polygons(Polygon(line.coords).buffer(0))
            name = next((street for street, own in self.areas.items() if street and polygons
                         and any(a.contains(polygons[0].representative_point()) for a in own)), None)
            parts = [(part, name) for part in polygons]
            colour = way["tags"].get("colour")
            if way["tags"].get("pattern") == "solid" and colour in FILL_COLOURS:
                self.filled[colour] += [part for part, _name in parts]
            elif colour in HATCH_COLOURS:
                self.hatched_coloured[colour] += [part for part, _name in parts]
            else:
                if colour not in (None, "white"):
                    self.stats[f"restriction areas colour={colour}: no such paint, hatched white"] += 1
                self.hatched += parts

    def hatch_all(self) -> None:
        """Every hatched area - shoulders and `road_marking=restriction` - outlined as one shape,
        so a way split for any other reason does not split its hatching, and struck at 45 degrees
        to the street it is on: from points HATCH_SPACING_M * sqrt(2) apart along that street's
        centreline, its ways joined end to end, so the strokes run on unbroken across the splits
        and turn with the street. A restriction area on no street takes its own long axis."""
        merged = _seamless([part for part, _name in self.hatched])
        for area in _polygons(merged):
            self.out["lane_narrowing_edge_lines"] += _lines(area.exterior)
        by_street: dict[str | None, list[Polygon]] = defaultdict(list)
        for part, name in self.hatched:
            by_street[name if name in self.street_lines else None].append(part)
        for name, parts in by_street.items():
            area = _seamless(parts)
            if name is None:
                for piece in _polygons(area):
                    corners = np.asarray(piece.minimum_rotated_rectangle.exterior.coords)
                    sides = np.diff(corners[:3], axis=0)
                    long = sides[np.argmax(np.linalg.norm(sides, axis=1))]
                    middle = np.asarray(piece.centroid.coords[0])
                    reach = float(np.linalg.norm(long))
                    self.strokes(LineString([middle - long, middle + long]), piece, reach, wide=False)
                continue
            shapely.prepare(area)
            for line in getattr(linemerge(self.street_lines[name]), "geoms", None) or [
                    linemerge(self.street_lines[name])]:
                self.strokes(line, area, self.street_width[name], wide=self.street_mph[name] >= HATCH_WIDE_MPH)
        # Solid restriction areas: filled in their colour's channel, outlined in white.
        for colour, parts in self.filled.items():
            for piece in _polygons(_seamless(parts)):
                self.out[f"{colour}_fill_polygons"] += _rings(piece)
                self.out["lane_narrowing_edge_lines"] += _lines(piece.exterior)
        # Coloured restriction areas: their own outline and strokes, in their colour's channels.
        for colour, parts in self.hatched_coloured.items():
            for piece in _polygons(_seamless(parts)):
                self.out[f"{colour}_hatch_edge_lines"] += _lines(piece.exterior)
                corners = np.asarray(piece.minimum_rotated_rectangle.exterior.coords)
                sides = np.diff(corners[:3], axis=0)
                long = sides[np.argmax(np.linalg.norm(sides, axis=1))]
                middle = np.asarray(piece.centroid.coords[0])
                self.strokes(LineString([middle - long, middle + long]), piece,
                             float(np.linalg.norm(long)), wide=False, channel=f"{colour}_hatch_stroke_lines")

    def strokes(self, along: LineString, area: BaseGeometry, reach: float, wide: bool,
                channel: str | None = None) -> None:
        """Strokes at 45 degrees to `along`, from points HATCH_SPACING_M apart along it - the
        longitudinal spacing - `reach` either side, clipped to `area`; in the wide strokes'
        channel where the street is posted at HATCH_WIDE_MPH or more."""
        channel = channel or ("lane_narrowing_hatch_wide_lines" if wide else "lane_narrowing_hatch_lines")
        for at in np.arange(HATCH_SPACING_M / 2, along.length, HATCH_SPACING_M):
            p = np.asarray(along.interpolate(at).coords[0])
            ahead = np.asarray(along.interpolate(min(at + TANGENT_M, along.length)).coords[0])
            behind = np.asarray(along.interpolate(max(at - TANGENT_M, 0.0)).coords[0])
            t = (ahead - behind) / max(float(np.linalg.norm(ahead - behind)), 1e-9)
            angle = math.radians(HATCH_ANGLE_DEG)                  # off the street
            d = math.cos(angle) * t + math.sin(angle) * np.array([-t[1], t[0]])
            stroke = LineString([p - d * reach, p + d * reach])
            for piece in _lines(stroke.intersection(area)):
                self.out[channel].append([piece[0], piece[-1]])

    def kerb(self, way: dict) -> None:
        """A kerb at its `kerb:height`, else its `kerb=*` reveal - a lowered one tagged
        `wheelchair=yes` (wiki Key:wheelchair: passable in a wheelchair, which is what makes a
        curb ramp) flush with the road. An untagged one is not guessed at: src/osm_osc.py reports
        each a crossing meets, for mapping."""
        line = self.frame.line(way.get("coords_wgs84") or [])
        if line is None:
            return
        tags = way["tags"]
        height = width_m(tags.get("kerb:height"))
        if height is None and tags.get("kerb") == "lowered" and tags.get("wheelchair") == "yes":
            height = RAMP_HEIGHT_M
            self.stats["lowered kerbs flush: curb ramps (wheelchair=yes)"] += 1
        if height is None:
            height = KERB_HEIGHT_M.get(tags.get("kerb", "raised"), KERB_HEIGHT_M["raised"])
        self.out["kerbs"].append({"coords": _lines(line)[0], "height_m": height})

    def paved(self, way: dict, closed: bool) -> None:
        line = self.frame.line(way["coords_wgs84"])
        if line is None:
            return
        if closed and way["tags"].get("parking") == "lane":
            # A parking lane mapped as its own area is ON the carriageway: it is paint, its
            # outline marked where `markings=yes`, not a lot paved beside the road.
            if way["tags"].get("markings") == "yes":
                self.out["parking_edge_lines"] += _lines(Polygon(line.coords).buffer(0).exterior)
            return
        rings = (_rings(Polygon(line.coords).buffer(0)) if closed and len(line.coords) >= 4
                 else _strip(line, self.width(way["tags"], "width", "service") / 2))
        self.out["paved_surfaces"] += [_paving(ring, way["tags"]) for ring in rings]

    def sidewalks_off_the_street(self) -> None:
        """Sidewalks stop where the street, or a ramp's tactile pad, begins: they stand at kerb
        height, above both, so a sidewalk strip drawn over a junction's corner would cover the
        road. A sidewalk runs on across a driveway, a lot's mouth or an apron, as it is built -
        the slab stands above the paving and draws over it."""
        paving = unary_union([self.carriageway(), *(
            part for ring in self.out["tactile_paving_polygons"] if len(ring) >= 4
            for part in _polygons(Polygon(ring).buffer(0)))])
        def walks(key: str) -> BaseGeometry:
            return unary_union([part for ring in self.out[key] if len(ring) >= 4
                                for part in _polygons(Polygon(ring).buffer(0))])

        # Where an asphalt walk meets a concrete one, each keeps its own slab: no two overlap.
        asphalt = walks("sidewalks_asphalt").difference(paving)
        concrete = walks("sidewalks").difference(paving).difference(asphalt)
        for key, shape in (("sidewalks", concrete), ("sidewalks_asphalt", asphalt)):
            self.out[key] = [ring for part in _hole_free(shape) for ring in _rings(part)]

    def away(self, line: LineString) -> tuple[np.ndarray, np.ndarray] | None:
        """A kerb's middle, and the unit normal there pointing away from the nearest street."""
        streets = [street for lines in self.street_lines.values() for street in lines]
        if not streets:
            return None
        middle = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
        ahead = np.asarray(line.interpolate(min(line.length / 2 + TANGENT_M, line.length)).coords[0])
        behind = np.asarray(line.interpolate(max(line.length / 2 - TANGENT_M, 0.0)).coords[0])
        t = (ahead - behind) / max(float(np.linalg.norm(ahead - behind)), 1e-9)
        street = min(streets, key=lambda one: one.distance(Point(middle)))
        normal = np.array([-t[1], t[0]])
        if np.dot(normal, middle - np.asarray(street.interpolate(street.project(Point(middle))).coords[0])) < 0:
            normal = -normal
        return middle, normal

    def tactile(self, kerb: dict, crossings: list[dict]) -> None:
        """A curb ramp's detectable warning surface (`tactile_paving=yes`): a pad RAMP_WIDTH_M wide
        and TACTILE_DEPTH_M deep at the foot of each ramp, behind the kerb, away from the street.
        OSM maps where a ramp is - where a footway crossing meets the kerb - not how wide it is,
        so each pad is centred there at the ramp's minimum width (tactile.width), not spread
        along a lowered kerb that also carries the ramp's flares or the rest of the corner.
        - a kerb way: a pad along it about each point a crossing meets it, at a node they share or
          across it;
        - a kerb node on a crossing way: a pad square to that crossing, from the node away from the
          street, along the direction of travel.
        A tactile kerb no crossing meets says nothing of where its ramp is, so draws none."""
        if kerb["tags"].get("tactile_paving") != "yes":
            return
        half = RAMP_WIDTH_M / 2
        line = self.frame.line(kerb.get("coords_wgs84") or [])
        if line is None:                         # a kerb node: on the crossing whose ramp it is
            at = np.asarray(self.frame.point(kerb["lon"], kerb["lat"]))
            for crossing in crossings:
                ids = crossing.get("node_ids") or []
                path = self.frame.line(crossing.get("coords_wgs84") or [])
                if kerb["id"] not in ids or path is None or len(ids) != len(path.coords):
                    continue
                i = ids.index(kerb["id"])
                coords = np.asarray(path.coords)
                ends = [coords[j] for j in (i - 1, i + 1) if 0 <= j < len(coords)]
                streets = [one for lines in self.street_lines.values() for one in lines]
                if not ends or not streets:
                    continue
                # away from the street: the neighbouring vertex further from its nearest centreline
                far = max(ends, key=lambda c: min(one.distance(Point(c)) for one in streets))
                d = (far - at) / max(float(np.linalg.norm(far - at)), 1e-9)
                w = np.array([-d[1], d[0]])
                self.out["tactile_paving_polygons"].append(
                    [list(at - w * half), list(at + w * half), list(at + w * half + d * TACTILE_DEPTH_M),
                     list(at - w * half + d * TACTILE_DEPTH_M), list(at - w * half)])
                self.stats["tactile paving pads (kerb node on a crossing)"] += 1
                return
            self.stats["tactile paving on a kerb node no crossing passes: not drawn"] += 1
            return
        if (away := self.away(line)) is None:
            return
        middle, normal = away
        ahead = np.asarray(line.interpolate(min(line.length / 2 + TANGENT_M, line.length)).coords[0])
        left = np.cross(np.append(ahead - middle, 0.0), np.append(normal, 0.0))[2] > 0
        nodes = set(kerb.get("node_ids") or [])
        ramps = []
        for crossing in crossings:
            path = self.frame.line(crossing.get("coords_wgs84") or [])
            if path is None:
                continue
            shared = [c for n, c in zip(crossing.get("node_ids") or [], path.coords, strict=False) if n in nodes]
            meet = Point(shared[0]) if shared else path.intersection(line)
            if not meet.is_empty:
                ramps.append(line.project(meet if isinstance(meet, Point) else meet.centroid))
        if not ramps:
            self.stats["tactile paving on a kerb no crossing meets: not drawn"] += 1
            return
        for at in ramps:
            # Off the kerb's own vertices, not resampled stations: a corner ramp's curve would be cut.
            piece = _substring(line, max(at - half, 0.0), min(at + half, line.length))
            pad = piece.buffer(TACTILE_DEPTH_M if left else -TACTILE_DEPTH_M, single_sided=True,
                               cap_style="flat", join_style="round")
            self.out["tactile_paving_polygons"] += _rings(pad)
            self.stats["tactile paving pads (kerb way, at a crossing)"] += 1

    def mouths(self, layers: dict) -> None:
        """Each lowered kerb's driveway mouth, paved - a rendering of what OSM maps (the lowered
        kerb, the driveway, the sidewalk, the lot), not a surface OSM has: a `kerb=lowered` way is
        where traffic leaves the street, so what lies between it and where that traffic goes is
        paved, the whole kerb wide.
        - one a footway crossing meets is that crossing's ramp - a landing, not a mouth;
        - one a driveway or parking aisle crosses is paved out to the sidewalk the branch crosses
          nearest the kerb, through that sidewalk's band - as far as the branch itself is paved;
        - any other is paved out to the lot (`amenity=parking`) or `highway=*` area met first
          straight out behind it, away from its street, where no building or other kerb is met
          before it - through any sidewalk, as a driveway is.
        Built after the buildings, which it reads; the sidewalk draws over it."""
        branches = layers["driveways"] + layers["parking_aisles"]
        areas = [Polygon(line.coords).buffer(0).exterior
                 for way in layers["parking_lots"] + layers["highway_areas"]
                 if way["tags"].get("parking") != "lane"
                 and (line := self.frame.line(way["coords_wgs84"])) is not None and len(line.coords) >= 4]
        blockers = [Polygon(b["coords"]).exterior for b in self.out["buildings"] if len(b["coords"]) >= 4]
        kerb_lines = {way["id"]: line for way in layers["kerbs"]
                      if (line := self.frame.line(way.get("coords_wgs84") or [])) is not None}
        west, south, east, north = self.carriageway().bounds
        reach = math.hypot(east - west, north - south)

        def meets(kerb: dict, line: LineString, ways: list[dict]) -> list[LineString]:
            nodes = set(kerb.get("node_ids") or [])
            return [path for way in ways if (path := self.frame.line(way.get("coords_wgs84") or [])) is not None
                    and (nodes & set(way["node_ids"]) or path.intersects(line))]

        def first(ray: LineString, edges: list) -> float:
            start = Point(ray.coords[0])
            return min((start.distance(ray.intersection(e)) for e in edges if ray.intersects(e)), default=math.inf)

        for kerb in layers["kerbs"]:
            line = kerb_lines.get(kerb["id"])
            if line is None or kerb["tags"].get("kerb") not in ("lowered", "flush"):
                continue
            if meets(kerb, line, layers["crossings"]):
                continue
            back, more = None, []
            for path in meets(kerb, line, branches):
                met = [(path.intersection(walk).distance(line), walk, way) for way in layers["sidewalks"]
                       if (walk := self.frame.line(way["coords_wgs84"])) is not None and walk.intersects(path)]
                if met:
                    _d, walk, way = min(met, key=lambda m: m[0])
                    back = [walk.interpolate(walk.project(Point(c))).coords[0] for c in line.coords]
                    along = sorted(walk.project(Point(c)) for c in back)
                    more = [_substring(walk, along[0], along[-1]).buffer(
                        self.width(way["tags"], "width", "sidewalk") / 2, cap_style="flat")]
                    self.stats["driveway mouths: to the sidewalk the driveway crosses"] += 1
                    break
            if back is None:
                if (away := self.away(line)) is None:
                    continue
                middle, normal = away
                ray = LineString([middle, middle + normal * reach])
                between = first(ray, blockers + [o for kid, o in kerb_lines.items() if kid != kerb["id"]])
                met = [(d, edge) for edge in areas if (d := first(ray, [edge])) < between]
                if not met:
                    self.stats["lowered kerbs with nothing paved straight behind them"] += 1
                    continue
                _d, edge = min(met, key=lambda m: m[0])
                back = [edge.interpolate(edge.project(Point(c))).coords[0] for c in line.coords]
                self.stats["driveway mouths: to the lot behind"] += 1
            mouth = unary_union([Polygon([*line.coords, *back[::-1]]).buffer(0), *more])
            self.out["paved_surfaces"] += [{"coords": ring} for ring in _rings(mouth)]

    def off_the_street(self, at: np.ndarray, outward: np.ndarray, limit: float) -> np.ndarray | None:
        """The first point from `at` along `outward` (a unit vector) standing SIGNAL_CLEARANCE_M from
        the drawn street - None if there is none before `limit`."""
        road = self.carriageway()
        for d in np.arange(0.0, limit, SEARCH_STEP_M):
            if not road.contains(point := Point(at + outward * d)) and road.distance(point) >= SIGNAL_CLEARANCE_M:
                return at + outward * d
        return None

    def signals(self, controls: list[dict], crossings: list[dict]) -> None:
        """Signal hardware where OSM maps it. OSM maps a signalized junction (a
        `highway=traffic_signals` node) and how its heads are held (`support=*`), not where each
        pole stands, so the poles follow the rule confirmed against street view for Broad &
        Greenwood: one at each corner, on the corner of the leg whose left edge (looking out from
        the junction) forms it, its head facing back into the junction - the far-side signal for
        traffic arriving from across it. With `support=mast_arm` the head hangs at the end of an
        arm square to that leg, over its centreline; otherwise it is on the pole (`support=pole`).
        A corner is where two neighbouring legs' edges meet - two legs less than 180 degrees
        apart - and the pole stands SIGNAL_CLEARANCE_M behind where the street ends along the
        corner's bisector. Each end of a `crossing=traffic_signals` crossing gets a pedestrian
        signal facing across it to the other end, with a push button where `button_operated=yes`."""
        for node in controls:
            if node["tags"].get("highway") != "traffic_signals":
                continue
            here = np.asarray(self.frame.point(node["lon"], node["lat"]))
            legs = []                                # (outward unit vector, half carriageway width)
            for line, ids, tags in self.road_ways.values():
                coords = np.asarray(line.coords)
                if node["id"] not in ids or len(ids) != len(coords):
                    continue
                i = ids.index(node["id"])
                for j in (i - 1, i + 1):
                    if 0 <= j < len(coords) and (n := float(np.linalg.norm(coords[j] - coords[i]))) > 0:
                        legs.append(((coords[j] - coords[i]) / n, self.carriageway_width(tags) / 2))
            if len(legs) < 3:
                self.stats["traffic signals not at a junction: not drawn"] += 1
                continue
            legs.sort(key=lambda leg: math.atan2(leg[0][1], leg[0][0]))
            arm = node["tags"].get("support") == "mast_arm"
            if node["tags"].get("support") is None:
                self.stats["signalized junctions with no support tag: heads drawn on their poles"] += 1
            for k, (u_a, h_a) in enumerate(legs):
                u_b, h_b = legs[(k + 1) % len(legs)]
                turn = (math.atan2(u_b[1], u_b[0]) - math.atan2(u_a[1], u_a[0])) % (2 * math.pi)
                if not 0 < turn < math.pi:
                    continue                         # no corner between them
                left_a, right_b = np.array([-u_a[1], u_a[0]]), np.array([u_b[1], -u_b[0]])
                t, _r = np.linalg.solve(np.column_stack([u_a, -u_b]), right_b * h_b - left_a * h_a)
                corner = here + left_a * h_a + u_a * t
                outward = (corner - here) / max(float(np.linalg.norm(corner - here)), 1e-9)
                pole = self.off_the_street(corner, outward, h_a + h_b + SIGNAL_CLEARANCE_M * 4)
                if pole is None:
                    self.stats["signal poles with no corner clear of the street: not drawn"] += 1
                    continue
                self.out["props"].append({
                    "type": "traffic_signal_pole", "position_m": pole.tolist(),
                    "heading_deg": math.degrees(math.atan2(-u_a[1], -u_a[0])),
                    "arm_heading_deg": math.degrees(math.atan2(-left_a[1], -left_a[0])),
                    # out to the leg's centreline: the pole's distance from it, square to the leg
                    "arm_length_m": float(np.dot(pole - here, left_a)) if arm else 0.0})
                self.stats["traffic signal poles"] += 1
        for way in crossings:
            line = self.frame.line(way.get("coords_wgs84") or [])
            if line is None or way["tags"].get("crossing") != "traffic_signals":
                continue
            # From the crossing's middle out along it each way, to where it has left the street:
            # the post stands at its end of the crosswalk, not where the footway goes on to.
            middle = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
            for end in (np.asarray(line.coords[0]), np.asarray(line.coords[-1])):
                outward = (end - middle) / max(float(np.linalg.norm(end - middle)), 1e-9)
                post = self.off_the_street(middle, outward, line.length)
                if post is None:
                    continue
                facing = math.degrees(math.atan2(-outward[1], -outward[0]))
                self.out["props"].append({"type": "pedestrian_signal_head", "position_m": post.tolist(),
                                          "heading_deg": facing, "own_post": True})
                self.stats["pedestrian signal heads"] += 1
                if way["tags"].get("button_operated") == "yes":
                    self.out["props"].append({"type": "pedestrian_pushbutton", "position_m": post.tolist(),
                                              "heading_deg": facing})
                    self.stats["pedestrian push buttons"] += 1

    def building(self, way: dict) -> None:
        line = self.frame.line(way["coords_wgs84"])
        if line is None or len(line.coords) < 4:
            return
        height = way.get("height_m")
        self.stats["building height " + ("from OSM" if height else "DEFAULTED")] += 1
        for ring in _rings(Polygon(line.coords).buffer(0)):
            self.out["buildings"].append({"mesh": False, "coords": ring,
                                          "height_m": height or DEFAULT_BUILDING_HEIGHT_M})


def has_centre_line(tags: dict) -> bool:
    """Whether a way's centre line is painted: a two-way way with 2+ `lanes`, or with
    `overtaking[:forward|:backward]` or `lane_markings=yes` - overtaking is only regulated by a
    centre line - unless `lane_markings=no`."""
    if tags.get("oneway") == "yes" or tags.get("lane_markings") == "no":
        return False
    return ((_int(tags.get("lanes")) or 0) >= 2 or tags.get("lane_markings") == "yes"
            or any(key == "overtaking" or key.startswith("overtaking:") for key in tags))


def carriageway_width_m(tags: dict) -> tuple[float, str]:
    """A way's kerb-to-kerb width: `width` / `width:carriageway`, else a counted fallback."""
    found = width_m(tags.get("width")) or width_m(tags.get("width:carriageway"))
    if found is not None:
        return found, "from OSM"
    if tags.get("highway") == "service":
        return DEFAULT_WIDTHS_M["service"], "DEFAULTED (service)"
    lanes = _int(tags.get("lanes")) or (1 if tags.get("oneway") == "yes" else 2)
    # Everything the way's tags put on its carriageway, each as wide as _Reader.kerbside lays it:
    # `width` includes lane parking and never street-side parking (wiki Street_parking).
    section = lanes * DEFAULT_WIDTHS_M["lane"]
    for side in ("left", "right"):
        kerb = kerb_section(tags, side)
        section += kerb.shoulder_m + kerb.cycleway_m
        position = _parking(tags, side)
        if position in ("lane", "half_on_kerb"):
            depth = parking_depth_m(tags, side) or DEFAULT_WIDTHS_M["parking"]
            section += depth if position == "lane" else depth / 2
    return section, "DEFAULTED (its tagged section summed)"


def has_lane_lines(tags: dict) -> bool:
    """Whether a way's lane lines are painted (lane_lines): two or more of its `lanes` run the
    same way - every lane of a oneway - unless `lane_markings=no`."""
    if tags.get("lane_markings") == "no":
        return False
    lanes = _int(tags.get("lanes")) or 0
    if tags.get("oneway") in ("yes", "-1"):
        return lanes >= 2
    backward = backward_lanes(tags, lanes)
    return max(backward, lanes - backward) >= 2


def backward_lanes(tags: dict, lanes: int) -> int:
    """How many of a two-way way's `lanes` run against it: `lanes:backward`, else what
    `lanes:forward` leaves, else half."""
    backward, forward = _int(tags.get("lanes:backward")), _int(tags.get("lanes:forward"))
    if backward is not None:
        return backward
    return lanes - forward if forward is not None else lanes // 2


# `change:lanes` values that keep traffic in a lane from crossing to its right / its left
# (wiki Key:change; `only_left` is `not_right`, `only_right` is `not_left`).
NO_CHANGE_RIGHT = ("no", "not_right", "only_left")
NO_CHANGE_LEFT = ("no", "not_left", "only_right")
_SWAP_CHANGE = {"not_left": "not_right", "not_right": "not_left",
                "only_left": "only_right", "only_right": "only_left"}


def lane_changes(tags: dict, lanes: int, backward: int) -> list[str | None]:
    """Each lane's `change:lanes` value, lanes left to right across the way; None where untagged.
    A list is read only where it has one value per lane it covers. A oneway's is `change:lanes`;
    a two-way way's is `change:lanes:backward` then `change:lanes:forward`. Lanes run against
    the way (`:backward`, `oneway=-1`) are listed as their drivers see them, so they are reversed
    and their left and right swapped into the way's frame."""
    def values(key: str, count: int, against: bool) -> list[str | None]:
        found = (tags.get(key) or "").split("|")
        if len(found) != count:
            return [None] * count
        return [_SWAP_CHANGE.get(v, v) for v in reversed(found)] if against else found
    if tags.get("oneway") in ("yes", "-1"):
        return values("change:lanes", lanes, tags.get("oneway") == "-1")
    return (values("change:lanes:backward", backward, True)
            + values("change:lanes:forward", lanes - backward, False))


def lane_turns(tags: dict, lanes: int, backward: int) -> list[str | None]:
    """Each lane's `turn:lanes` value, lanes left to right across the way, as lane_changes reads
    `change:lanes`: a oneway's `turn:lanes`, a two-way way's `:backward` (reversed into the way's
    order) then `:forward`; None where a list is absent or not one value per lane."""
    def values(key: str, count: int, against: bool) -> list[str | None]:
        found: list[str | None] = list((tags.get(key) or "").split("|"))
        if len(found) != count:
            return [None] * count
        return found[::-1] if against else found
    if tags.get("oneway") in ("yes", "-1"):
        return values("turn:lanes", lanes, tags.get("oneway") == "-1")
    return (values("turn:lanes:backward", backward, True)
            + values("turn:lanes:forward", lanes - backward, False))


def _turn_only(value: str | None) -> bool:
    """A mandatory turn lane: indications, and `through` not among them (wiki Key:turn)."""
    indications = {v for v in (value or "").split(";") if v not in ("", "none")}
    return bool(indications) and "through" not in indications


def _parking(tags: dict, side: str) -> str | None:
    """`side`'s street parking position (wiki Street_parking)."""
    return tags.get(f"parking:{side}") or tags.get("parking:both")


def stall_pitch_m(orientation: str) -> float:
    """How much kerb one standard stall takes: a parallel stall's length; an angled stall's width
    across the kerb, W / sin(angle) - a diagonal bay at stall.diagonal_angle, a perpendicular one
    square to it."""
    if orientation == "parallel":
        return PARALLEL_STALL_M
    angle = DIAGONAL_DEG if orientation == "diagonal" else 90.0
    return STALL_WIDTH_M / math.sin(math.radians(angle))


def _orientation(tags: dict, side: str) -> str:
    return tags.get(f"parking:{side}:orientation") or tags.get("parking:both:orientation") or "parallel"


def parking_depth_m(tags: dict, side: str) -> float | None:
    """How deep a parking lane is off the kerb: its `parking:<side>:width`, else by orientation -
    a diagonal bay `L sin(a) + W cos(a)`, a perpendicular one the stall's length;
    None for a parallel lane, which takes DEFAULT_WIDTHS_M."""
    tagged = width_m(tags.get(f"parking:{side}:width") or tags.get("parking:both:width"))
    if tagged:
        return tagged
    angle = {"diagonal": DIAGONAL_DEG, "perpendicular": 90.0}.get(_orientation(tags, side))
    if angle is None:
        return None
    a = math.radians(angle)
    return STALL_BODY_M * math.sin(a) + STALL_WIDTH_M * math.cos(a)


def kerbward(tags: dict, side: str) -> str:
    """Which side of `side`'s cycle lane its kerb is on, as the lane is ridden (wiki Key:separation:
    left and right are the direction a oneway cycle lane is used in). A right lane rides with the
    way; a left lane rides with it on a oneway street, against it on a two-way one."""
    if side == "right":
        return "right"
    return "left" if tags.get("oneway") == "yes" and tags.get("cycleway:left:oneway") != "-1" else "right"


def parking_at_the_kerb(tags: dict, side: str) -> bool:
    """Whether `side`'s parking lies between its cycle lane and the kerb - the lane outside it -
    as `cycleway:<side>:traffic_mode:<kerb side>` says (Proposal:Separation: `parking` there).
    Where it says nothing: a painted `lane` runs outside the parked cars, as a bike lane beside
    parking does (Danny, 2026-10-08); a `track` is at the kerb, the parking outside protecting it."""
    mode = tags.get(f"cycleway:{side}:traffic_mode:{kerbward(tags, side)}")
    if mode is not None:
        return mode == "parking"
    return (tags.get(f"cycleway:{side}") or tags.get("cycleway:both") or tags.get("cycleway")) == "lane"


class KerbSection(NamedTuple):
    """What a way's tags lay on one side, in metres: the cycle lane and its buffer, the parking
    lane, the shoulder - each 0.0 where the side has none - and whether the cycle lane is the
    one at the kerb, with the parking inside it."""
    cycleway_m: float
    parking_m: float
    shoulder_m: float
    cycleway_at_kerb: bool


def kerb_section(tags: dict, side: str) -> KerbSection:
    """`side`'s section as `_Reader.kerbside` lays it, from the same lookups and defaults:
    DEFAULT_WIDTHS_M where a width is not tagged, parking_depth_m for the parking lane, and the
    cycle lane at the kerb unless parking_at_the_kerb puts the parking there."""
    cycle = has_cycleway(tags, side)
    return KerbSection(
        (width_m(tags.get(f"cycleway:{side}:width")) or DEFAULT_WIDTHS_M["cycleway"])
        + (width_m(tags.get(f"cycleway:{side}:buffer")) or 0.0) if cycle else 0.0,
        parking_depth_m(tags, side) or DEFAULT_WIDTHS_M["parking"] if _parking(tags, side) == "lane" else 0.0,
        width_m(tags.get(f"shoulder:{side}:width")) or DEFAULT_WIDTHS_M["shoulder"]
        if (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes" else 0.0,
        cycle and not parking_at_the_kerb(tags, side))


def _footpath(tags: dict) -> bool:
    """A paved way on foot that is not a sidewalk or a crossing (those have their own layers):
    `highway=footway` / `pedestrian` unless its `surface` is unpaved, and `highway=path` - a
    generic way, often a trail - only where its `surface` says it is paved."""
    if tags.get("footway") in ("sidewalk", "crossing") or _trail(tags):
        return False
    if tags.get("highway") == "path":
        return tags.get("surface") in ("asphalt", "concrete", "concrete:plates", "concrete:lanes")
    return tags.get("highway") in ("footway", "pedestrian")


def _trail(tags: dict) -> bool:
    """A way on foot or a track whose `surface` is unpaved: drawn as that ground, at grade."""
    return tags.get("highway") in ON_FOOT and SURFACE_MATERIAL.get(tags.get("surface")) in UNPAVED


def _pavement_key(tags: dict) -> str:
    """The output a street's surface goes to, by its material."""
    material = SURFACE_MATERIAL.get(tags.get("surface"), "asphalt")
    return "pavement" if material == "asphalt" else f"pavement_{material}"


def _paving(ring: list, tags: dict) -> dict:
    """A paved-ground ring, and its material where that is not asphalt."""
    material = SURFACE_MATERIAL.get(tags.get("surface"), "asphalt")
    return {"coords": ring} if material == "asphalt" else {"coords": ring, "surface": material}


def has_cycleway(tags: dict, side: str) -> bool:
    """Whether `side` has a cycle lane or track (`cycleway:<side>`, `:both` or the bare key)."""
    return (tags.get(f"cycleway:{side}") or tags.get("cycleway:both")
            or tags.get("cycleway")) in ("lane", "track")


def _seamless(parts: list[Polygon]) -> BaseGeometry:
    """`parts` as one area. Each way's band ends square to its own last segment, so where the next
    way turns the two miss by millimetres and a plain union keeps them apart - an outline across
    the hatching at every split. Closing by SEAM_M (mitred, so corners stay square) joins them."""
    return (unary_union(parts).buffer(SEAM_M, join_style="mitre")
            .buffer(-SEAM_M, join_style="mitre").buffer(0))


def _hole_free(geometry: BaseGeometry) -> list[Polygon]:
    """`geometry`'s polygons with no holes: one with a hole is cut in two across it, as often as
    it takes. Blender extrudes a ring's outline only, so a hole drawn as a ring would be filled."""
    out = []
    for part in _polygons(geometry):
        if not part.interiors:
            out.append(part)
            continue
        x = Polygon(part.interiors[0]).representative_point().x   # inside the hole, so the cut crosses it
        _west, south, _east, north = part.bounds
        halves = _polygons(split(part, LineString([(x, south - 1.0), (x, north + 1.0)])))
        out += [whole for half in halves for whole in _hole_free(half)] if len(halves) > 1 else [
            Polygon(part.exterior)]
    return out


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    return [part for part in getattr(geometry, "geoms", [geometry])
            if isinstance(part, Polygon) and not part.is_empty]


def _dashes(geometry: BaseGeometry, dash_m: float = DASH_M,
            cycle_m: float = DASH_CYCLE_M) -> list[list[list[float]]]:
    """A broken line: `dash_m` painted in every `cycle_m` along each part of `geometry`."""
    out = []
    for part in _lines(geometry):
        line = LineString(part)
        at = 0.0
        while at < line.length:
            out += _lines(_substring(line, at, min(at + dash_m, line.length)))
            at += cycle_m
    return out


def _dotted(geometry: BaseGeometry) -> list[list[list[float]]]:
    """A dotted line through a conflict area: CROSSBIKE_DASH_M dashes, CROSSBIKE_DASH_M gaps."""
    return _dashes(geometry, CROSSBIKE_DASH_M, 2 * CROSSBIKE_DASH_M)


def _skip_bars(line: LineString, half_m: float) -> list[list[list[float]]]:
    """Green skip-paint along `line`, `half_m` either side: bars in the dotted line's pattern."""
    return [ring for dash in _dotted(line) for ring in _strip(LineString(dash), half_m)]


def _through_arrow(length_m: float) -> Polygon:
    """A through arrow `length_m` long, in metres (arrow.through_outline: x along the lane from the
    tail, y to its left); longer than arrow.through_length only as the through part of a
    turn-and-through arrow, which carries the same head out along a longer shaft."""
    head = length_m - si("arrow.through_length")
    return Polygon([(x + head if x > 0 else x, y) for x, y in si("arrow.through_outline")])


def _left_arrow() -> Polygon:
    """A left turn arrow, in metres (arrow.left_outline): a shaft bending left between two arcs
    whose centres stand apart, so the bend widens into the head."""
    shape = FIGURES["arrow.left_outline"]
    half = shape.si(shape.extra["shaft_half_width"])
    (ox, oy, outer_r), (ix, iy, inner_r) = (shape.si(shape.extra[k]) for k in ("outer_bend", "inner_bend"))
    bend = np.linspace(0.0, math.pi / 2, 12)
    outer = [(ox + outer_r * math.sin(t), oy - outer_r * math.cos(t)) for t in bend]
    inner = [(ix + inner_r * math.sin(t), iy - inner_r * math.cos(t)) for t in bend[::-1]]
    return Polygon([(0.0, -half), *outer, *shape.si(shape.extra["head"]), *inner, (0.0, half)])


# The `turn:lanes` indications a lane-use arrow is drawn for (wiki Key:turn; MUTCD Figure 3B-24).
LANE_USE_ARROWS = ("left", "through", "right")


def lane_use_arrow(indications: set[str]) -> BaseGeometry | None:
    """The arrow painted in a lane with these `turn:lanes` indications, in metres, x along the lane
    from the tail and y to its left: a turn arrow for `left` and for `right`, a through arrow for
    `through`, on one shaft - Figure 3B-24's arrows, and its turn-and-through arrow where a lane has
    both. None for any other indication: MUTCD has no arrow for a slight, sharp, reverse or merge."""
    if not indications or not indications <= set(LANE_USE_ARROWS):
        return None
    parts = []
    if "through" in indications:
        parts.append(_through_arrow(si("arrow.through_length" if len(indications) == 1 else "arrow.turn_through_length")))
    if "left" in indications:
        parts.append(_left_arrow())
    if "right" in indications:
        parts.append(_mirror(_left_arrow(), 1.0, -1.0, origin=(0.0, 0.0)))
    return unary_union(parts)


def bike_lane_marking(arrow: bool = True) -> list[Polygon]:
    """A bike lane's marking (MUTCD 9C.04, Figure 9C-3), in metres, x along the lane from its
    upstream end and y to its left: the bicycle symbol (bike_lane.symbol_shape), and with `arrow`
    its through arrow beyond it. Pieces, not one shape: a wheel is a ring, which a single outline
    without holes cannot draw, so each is two half-rings."""
    length = si("bike_lane.symbol_length")
    shape = FIGURES["bike_lane.symbol_shape"]
    stroke = shape.value * length
    parts: list[Polygon] = []
    for x, y, along, across in shape.extra["wheels"]:
        ring = shapely.affinity.scale(Point(x * length, y * length).buffer(1.0), along * length, across * length)
        hole = shapely.affinity.scale(Point(x * length, y * length).buffer(1.0),
                                      along * length - stroke, across * length - stroke)
        band = ring.difference(hole)
        for half in (shapely.box(-1e3, -1e3, x * length, 1e3), shapely.box(x * length, -1e3, 1e3, 1e3)):
            parts += _polygons(band.intersection(half))
    for line in shape.extra["strokes"]:
        parts.append(LineString([(x * length, y * length) for x, y in line]).buffer(stroke / 2))
    x, y, r = shape.extra["head"]
    parts.append(Point(x * length, y * length).buffer(r * length))
    if arrow:
        tail = length + si("bike_lane.symbol_to_arrow")
        scale = si("bike_lane.arrow_length") / si("arrow.through_length")
        parts.append(Polygon([(tail + x * scale, y * scale) for x, y in si("arrow.through_outline")]))
    return parts


def _mph(value: str | None) -> float:
    """A `maxspeed` in mph: "25 mph", or a bare number, which OSM reads as km/h. 0 where unknown."""
    if not value:
        return 0.0
    number = re.match(r"\s*(\d+(?:\.\d+)?)", value)
    if number is None:
        return 0.0
    return float(number.group(1)) * (1.0 if "mph" in value else 0.621371)


def taper_rate(tags: dict) -> float:
    """How far a marking may shift sideways per unit length: the MUTCD taper, L = W*S^2/D at or
    below taper.speed_break, L = W*S above, at the street's `maxspeed` (else speed.default)."""
    mph = _mph(tags.get("maxspeed")) or si("speed.default")
    return si("taper.low_speed_divisor") / mph ** 2 if mph <= si("taper.speed_break") else 1.0 / mph


def ease(x: float | np.ndarray) -> float | np.ndarray:
    """0 to 1 on an S-curve as `x` runs 0 to 1 (half a cosine): a shift that starts and ends
    parallel to the street, with no corner at either end."""
    return (1.0 - np.cos(np.pi * np.clip(x, 0.0, 1.0))) / 2.0


def _flip(key: str) -> str:
    """A tag key seen from a way drawn the other way: its left is this one's right."""
    return re.sub(r"(left|right)", lambda m: "right" if m.group(1) == "left" else "left", key)


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).split(";")[0])
    except ValueError:
        return None


def _substring(line: LineString, start_m: float, end_m: float) -> LineString:
    from shapely.ops import substring

    return substring(line, start_m, end_m)


def read(area: str, change: OsmChange | None = None) -> tuple[dict, _Reader]:
    """`area`'s OSM layers with `change` applied, and the reader that has drawn them."""
    layers = osm_layers(area)
    if change is not None:
        layers = apply_change(layers, change)
    frame = LocalFrame(SNAPSHOT_AREAS[area])
    areas: dict[str | None, list[Polygon]] = defaultdict(list)
    mouths: list[dict] = []
    boxes: list[Polygon] = []
    pavement: dict[str, list] = defaultdict(list)
    for way in layers.get("road_areas", []):
        line = frame.line(way["coords_wgs84"])
        if line is not None and len(line.coords) >= 4:
            polygons = _polygons(Polygon(line.coords).buffer(0))
            if way["tags"].get("cycleway") == "two_stage_box":
                boxes += polygons          # paint on the street, drawn by _Reader.turn_box
                continue
            # A driveway's surface (`area:highway=service` + `service=driveway`) is paved ground
            # beside the street, not its carriageway.
            if way["tags"].get("area:highway") == "service" and way["tags"].get("service") == "driveway":
                mouths += [_paving(ring, way["tags"]) for p in polygons for ring in _rings(p)]
                continue
            areas[way["tags"].get("name")].extend(polygons)
            pavement[_pavement_key(way["tags"])] += [
                ring for p in polygons for ring in _rings(p)]
    reader = _Reader(frame, dict(areas))
    reader.out["paved_surfaces"] += mouths
    for key, rings in pavement.items():
        reader.out[key] += rings
    reader.index_ends(layers["roads"])
    for way in layers["roads"]:
        if way["tags"].get("highway") == "cycleway" and way["tags"].get("cycleway") == "crossing":
            reader.cycle_crossing(way)
        else:
            reader.road(way)
    for box in boxes:
        reader.turn_box(box)
    for way in layers["sidewalks"] + [w for w in layers["roads"] if _footpath(w["tags"])]:
        reader.sidewalk(way)
    for way in layers["roads"]:
        if _trail(way["tags"]):
            reader.trail(way)
    for way in layers["crossings"]:
        reader.crossing(way)
    for way in layers["stop_lines"]:
        reader.stop_line(way)
    for way in layers["road_markings"]:
        reader.restriction(way)
    for way in layers["kerbs"]:
        reader.kerb(way)
        reader.tactile(way, layers["crossings"])
    for way in layers["driveways"] + layers["parking_aisles"]:
        reader.paved(way, closed=False)
    for way in layers["parking_lots"] + layers["highway_areas"]:
        reader.paved(way, closed=True)
    for way in layers["buildings"]:
        reader.building(way)
    reader.mouths(layers)
    reader.sidewalks_off_the_street()
    reader.signals(layers["traffic_control"], layers["crossings"])
    for item in layers.get("bollards", []):
        reader.bollard(item)
    reader.out["tree_points"] = [frame.point(n["lon"], n["lat"]) for n in layers["street_furniture"]
                                 if n["tags"].get("natural") == "tree"]
    reader.hatch_all()
    reader.stall_all()
    reader.markings_all()
    reader.off_other_streets()
    reader.on_carriageway()
    return layers, reader


def osm_world(area: str, proposal: Path | None = None) -> dict:
    """The world JSON for `area`, with `proposal` (an .osc) applied if given."""
    _layers, reader = read(area, load_change(proposal) if proposal is not None else None)
    try:
        from src.render.theme import build_default_theme

        reader.out["theme"] = build_default_theme()
    except Exception as error:  # textures are dressing; a world without them still renders
        reader.stats[f"no textures ({type(error).__name__})"] += 1
    return {**reader.out, "stats": dict(reader.stats)}
