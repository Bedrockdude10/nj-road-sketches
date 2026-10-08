"""OSM, read and drawn: an area's cached OSM with a proposal (.osc) applied, as the world JSON
scripts/blender/blender_scene.py `--build` reads.

Everything drawn is something OSM maps, placed where OSM puts it:
  carriageway   a street's `area:highway` polygons where it has them - the street as built -
                else half each way's `width` either side of its centreline: the middle of its
                street's mapped kerbs where any are mapped across from each other
                (src/osm_osc.py:kerb_centre_offsets), else the way. Every marking below is laid from
                the way's `width` (the way runs down the middle of its carriageway), never from a
                kerb, and then kept on that surface: paint stops at the kerb
  bike lanes    `cycleway[:<side>]=lane|track`, a band of `cycleway:<side>:width` inside that
                edge - green where `cycleway:<side>:surface:colour=green`, in skip bars with
                dotted edges where `cycleway:<side>:crossing:markings=dashes` - then
                `cycleway:<side>:buffer`, with a bollard every BOLLARD_SPACING_M down its centre
                where `cycleway:<side>:separation:*=flex_post`;
                `cycleway:<side>:oneway=no` adds the dashed yellow divider down its middle
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
                equally unless `width:lanes` places them; broken, solid where `change:lanes`
                keeps traffic from crossing; none where `lane_markings=no`
  centre line   two-way ways with 2+ lanes, or with `overtaking*` or `lane_markings=yes` tagged
                (has_centre_line), unless `lane_markings=no` - so where it stops is
                where the way is split and tagged, never decided here. Double yellow where
                `overtaking[:forward|:backward]=no`, dashed where `=yes`. On the way itself, or
                where `width:lanes` puts it - see _Reader.cross_section
  cycle crossings `highway=cycleway` + `cycleway=crossing` ways: a band of their `width`, green
                where `surface:colour=green`; where `crossing:markings=dashes`, edged in
                CROSSBIKE_DASH_M dots and the green in skip bars of the same pattern; a dashed
                yellow divider where `oneway=no`
  sidewalks     `footway=sidewalk` ways, and paved footpaths (_footpath), buffered to their
                `width`, concrete unless `surface=asphalt`
  crossings     `footway=crossing` ways, painted as their `crossing:markings` says, on the
                carriageway only
  markings      `road_marking=stop_line` bars (STOP_BAR_M wide), `road_marking=restriction`
                hatched areas
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

import numpy as np
import pyproj
import shapely
from shapely import unary_union
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, nearest_points, split

from src.geometry.model.crs import NJ_STATE_PLANE_FT, WGS84
from src.sources.osm_change import OsmChange, apply_change, load_change
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers

FT_TO_M = 0.3048

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

# NOT OSM: what is drawn where a way carries no width tag. Counted every time it is used.
DEFAULT_WIDTHS_M = {"lane": 3.0, "service": 3.5, "cycleway": 1.5, "parking": 2.2,
                    "sidewalk": 6 * FT_TO_M, "crossing": 3.0,
                    "shoulder": 1.0}
DEFAULT_BUILDING_HEIGHT_M = 7.0

# Kerb reveals above the gutter for `kerb=*` (wiki Key:kerb), where `kerb:height` is absent - the
# NJDOT Roadway Design Manual and N.J.A.C. 16:47 figures in STANDARDS.md 6a (as cited): a 6 in
# vertical face where sidewalks are built, a mountable kerb at most 4 in, a driveway's depressed
# kerb 1.5 in. A lowered kerb that is a curb ramp is flush instead (RAMP_HEIGHT_M, _Reader.kerb).
IN_TO_M = 0.0254
KERB_HEIGHT_M = {"raised": 6 * IN_TO_M, "regular": 6 * IN_TO_M, "rolled": 4 * IN_TO_M,
                 "lowered": 1.5 * IN_TO_M, "flush": 0.0}
RAMP_HEIGHT_M = 0.0
# A detectable warning surface's depth in the direction of travel, across the ramp's full width
# (ADA 2010 Standards 705.1, as cited; STANDARDS.md 6a).
TACTILE_DEPTH_M = 24 * IN_TO_M
# A curb ramp's width, flares excluded - NJDOT's minimum (as cited; STANDARDS.md 6a). OSM maps
# where a ramp is, not how wide, so each detectable warning pad is this wide.
RAMP_WIDTH_M = 48 * IN_TO_M
# A traffic-control support's lateral clearance behind the face of the kerb (MUTCD, as cited;
# STANDARDS.md 6a): where a signal pole or pedestrian signal post stands, off the drawn street.
SIGNAL_CLEARANCE_M = 2 * FT_TO_M

STATION_STEP_M = 2.0     # spacing of the cross-sections a road's edges are sampled at
CHUNK_M = 50.0           # a road is drawn in pieces this long, so a looped road encloses nothing
# A centre line's stripes (MUTCD 3A.05-3A.06, as cited): 4-6 in wide, a double line's two
# stripes one stripe-width apart, a broken line 10 ft dashes in 40 ft cycles.
CENTRE_PAIR_OFFSET_M = 0.15
DASH_M, DASH_CYCLE_M = 3.0, 12.0
STOP_BAR_M = 2 * FT_TO_M
BOLLARD_SPACING_M = 8 * FT_TO_M   # flex posts down a buffer's centre, as the old design spaced them
# A conflict area's dotted line, and the green skip-paint that follows its pattern: 2 ft dashes,
# 2 ft gaps (MUTCD 3B.08 dotted lines, IA-14 green colored pavement - as cited).
CROSSBIKE_DASH_M = 2 * FT_TO_M
# An angled stall, N.J.A.C. 5:21-4.14 (STANDARDS.md 1a): 9 ft across, 18 ft along, in its own frame.
# OSM has no key for a diagonal stall's angle, so `orientation=diagonal` is drawn at one: 60 degrees
# from the kerb, the angle observed on Grand Central Ave (Danny, 2026-09-10) and 1a's worked row.
STALL_WIDTH_M, STALL_BODY_M, DIAGONAL_DEG = 9 * FT_TO_M, 18 * FT_TO_M, 60.0
# A parallel stall along the kerb (STANDARDS.md 3), where a marked lane gives no capacity.
PARALLEL_STALL_M = 20 * FT_TO_M
# A bicycle symbol's / arrow's painted footprint - schematic, as the old pipeline drew them
# (MUTCD 9E.11(05) names the markings, not a size).
SYMBOL_LENGTH_M, SYMBOL_WIDTH_M = 5.5 * FT_TO_M, 2.4 * FT_TO_M
ZEBRA_BAR_M = 0.5            # a continental crossing's bar width, and the gap between bars
# Diagonal crosshatch (MUTCD 3B, as cited; STANDARDS.md 6b): strokes 30-45 degrees to the lines
# they meet (45 here), spaced along the street by engineering judgment - 10-20 ft on low-speed
# urban streets, the dense end taken so a short stretch between driveways still shows strokes - and
# 8 in wide below 45 mph, 12 in at or above (by OSM `maxspeed`: the wide strokes' own channel).
HATCH_SPACING_M = 10 * FT_TO_M
HATCH_WIDE_MPH = 45
# The `colour`s a `road_marking=restriction` area is hatched in other than white, each in its own
# channels (`<colour>_hatch_edge_lines`, `<colour>_hatch_stroke_lines`) - the channel is what
# decides a stripe's colour in 3D (scripts/blender/blender_scene.py:PAINT_COLOUR_CHANNELS).
HATCH_COLOURS = ("yellow", "blue")
SEAM_M = 0.05                # hatching laid way by way leaves mm seams where consecutive ways turn
# The line channels that are paint, kept on the street surface (_Reader.on_carriageway).
PAINT_LINES = ("bike_lane_edge_lines", "parking_edge_lines", "bike_lane_contraflow_lines",
               "lane_narrowing_edge_lines", "lane_narrowing_hatch_lines", "lane_narrowing_hatch_wide_lines",
               *(f"{c}_hatch_{k}_lines" for c in HATCH_COLOURS for k in ("edge", "stroke")),
               "parking_stall_divider_lines",
               "cycle_crossing_edge_lines",
               "cycle_crossing_divider_lines", "lane_lines")
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
        ahead = np.array([path.interpolate(min(lead + at + 0.5, path.length)).coords[0] for at in self.s])
        behind = np.array([path.interpolate(max(lead + at - 0.5, 0.0)).coords[0] for at in self.s])
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
    def __init__(self, frame: LocalFrame, areas: dict[str | None, list[Polygon]],
                 centre_offsets: dict[int, float] | None = None):
        self.frame = frame
        # Each way's drawn centreline off the way itself: the middle of its street's mapped kerbs
        # (src/osm_osc.py:kerb_centre_offsets). A way absent here is drawn where OSM has it.
        self.centre_offsets = centre_offsets or {}
        # Each street's `area:highway` polygons, by name: the street as built, where it is mapped.
        self.areas = areas
        self._own: dict[str, BaseGeometry] = {}
        self.hatched: list[tuple[Polygon, str | None]] = []   # (area, street): hatch_all draws them
        # restriction areas hatched in a colour other than white, by HATCH_COLOURS colour
        self.hatched_coloured: dict[str, list[Polygon]] = defaultdict(list)
        self.parking: list[dict] = []        # marked parking lanes, piece by piece: stall_all draws them
        self._way: dict = {}                 # the way being drawn
        # Each named street's centrelines and widest carriageway: what its hatching is struck along.
        self.street_lines: dict[str, list[LineString]] = defaultdict(list)
        self.street_width: dict[str, float] = defaultdict(float)
        self.street_mph: dict[str, float] = defaultdict(float)     # its highest posted `maxspeed`
        self.crosswalk_bands: list[BaseGeometry] = []   # kept clear of other paint (on_carriageway)
        self.turn_boxes: list[Polygon] = []              # likewise, bar their own markings
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
            "tree_points": [], "props": [],
            "cycle_crossing_surface_polygons": [], "cycle_crossing_edge_lines": [],
            "cycle_crossing_divider_lines": [], "parking_stall_divider_lines": [],
            "tactile_paving_polygons": []}

    def width(self, tags: dict, key: str, default: str) -> float:
        found = width_m(tags.get(key))
        if found is not None:
            self.stats[f"{default} width from OSM"] += 1
            return found
        self.stats[f"{default} width DEFAULTED"] += 1
        return DEFAULT_WIDTHS_M[default]

    def index_ends(self, ways: list[dict]) -> None:
        """Each named way by the street and the node at each of its ends - transition() walks it."""
        for way in ways:
            ids, name = way.get("node_ids") or [], way["tags"].get("name")
            if name and len(ids) >= 2:
                self.ends[(name, ids[0])].append(way)
                self.ends[(name, ids[-1])].append(way)

    def beyond(self, way: dict) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """At each end, the point half a metre into the one way of the same street that carries
        on from it, else None - what Stations spans the join with."""
        name, ids = way["tags"].get("name"), way.get("node_ids") or []
        out = []
        for node in (ids[0], ids[-1]) if name and len(ids) >= 2 else ():
            others = [o for o in self.ends[(name, node)] if o["id"] != way["id"]]
            other = self.frame.line(others[0]["coords_wgs84"]) if len(others) == 1 else None
            if other is None:
                out.append(None)
                continue
            at = min(0.5, other.length)
            out.append(other.interpolate(at if others[0]["node_ids"][0] == node
                                         else other.length - at).coords[0])
        return (out[0], out[1]) if out else (None, None)

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
        shift = self.centre_offsets.get(way["id"])
        if shift is not None:
            # Drawn down the middle of its kerbs: every cross-section, and the line itself.
            stations.xy = stations.at(np.full(len(stations.s), shift))
            line = LineString(stations.xy)
            self.stats["ways centred between their street's kerbs"] += 1
        else:
            self.stats["ways drawn where OSM has them: no kerb pair mapped"] += 1
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
        centre = self.cross_section(tags, line, stations, edges)
        if has_centre_line(tags):
            self.stripe_centre(tags, centre)

    def cross_section(self, tags: dict, line: LineString, stations: Stations,
                      edges: dict[str, np.ndarray]) -> LineString:
        """Draw the way's side features and return its centre line.

        Without `width:lanes` each side's shoulder, cycleway, buffer and parking are laid from its
        edge inward, and the centre line is the way. With every lane's width (wiki Key:width:lanes,
        left to right) the section is laid IN ORDER from the edge whose side carries a cycleway
        (else the left): its cycleway, buffer and parking, then the lanes, then the far side's
        parking lane against the last lane; the far side's shoulder is laid from its own edge."""
        lanes = _int(tags.get("lanes")) or 0
        widths = [width_m(w) for w in (tags.get("width:lanes") or "").split("|")]
        backward = backward_lanes(tags, lanes)
        if len(widths) != lanes or None in widths:
            left, right = (self.kerbside(tags, stations, side, edges[side]) for side in ("left", "right"))
            # Each direction's lanes share its half of the travel way equally; a oneway's share
            # all of it. Lane j's left boundary, counting lanes left to right from 0.
            if tags.get("oneway") in ("yes", "-1"):
                bounds = {j: left - j * (left + right) / lanes for j in range(1, lanes)}
            else:
                forward = lanes - backward
                bounds = ({j: left - j * left / backward for j in range(1, backward)}
                          | {backward + j: -j * right / forward for j in range(1, forward)})
            self.lane_lines(tags, stations, lanes, backward, bounds)
            return line
        def lane(i: int):
            def theirs(t: dict) -> float | None:
                values = [width_m(v) for v in (t.get("width:lanes") or "").split("|")]
                values = values[::-1] if t.get("_reversed") else values
                return values[i] if len(values) == len(widths) and values[i] is not None else None
            return theirs
        widths = [self.along(w, lane(i), stations) for i, w in enumerate(widths)]
        anchor = "right" if _has_cycleway(tags, "right") and not _has_cycleway(tags, "left") else "left"
        far = "left" if anchor == "right" else "right"
        sign = 1 if anchor == "left" else -1
        inner = self.kerbside(tags, stations, anchor, edges[anchor])
        self.kerbside(tags, stations, far, edges[far], parking=False)
        near = widths[:backward] if anchor == "left" else widths[backward:]
        leftmost = sign * inner + (sum(widths) if anchor == "right" else 0)
        self.lane_lines(tags, stations, lanes, backward,
                        {j: leftmost - sum(widths[:j]) for j in range(1, lanes)
                         if tags.get("oneway") in ("yes", "-1") or j != backward})
        if (tags.get(f"parking:{far}") or tags.get("parking:both")) == "lane":
            lane_edge = inner - sum(widths)
            self.out["parking_edge_lines"].append(stations.line(sign * lane_edge))
            self.stalls(tags, far, stations, sign * lane_edge,
                        sign * (lane_edge - self.eased(tags, f"parking:{far}:width", "parking", stations)))
        # Each side's lane edge, where nothing else marks it: a hatched shoulder's outline and a
        # parking lane's edge already do, and a track's buffer line does. That leaves a shoulder
        # whose hatching breaks where traffic crosses the kerb - the edge line is maintained across
        # a driveway (MUTCD 3B.11(09)), dotted as through a conflict area (3B.11(10)), so a solid
        # line is never mistaken for a parking lane's or a hatched area's edge.
        for side, edge_at in ((anchor, inner), (far, inner - sum(widths))):
            marked = (tags.get(f"shoulder:{side}:markings") == "hatched" or _has_cycleway(tags, side)
                      or (tags.get(f"parking:{side}") or tags.get("parking:both")) == "lane")
            if not marked and (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
                self.out["lane_narrowing_edge_lines"] += _dotted(LineString(stations.line(sign * edge_at)))
                self.stats["lane edge lines carried across an opening (pieces)"] += 1
        self.stats["cross-section laid by width:lanes (ways)"] += 1
        return LineString(stations.line(sign * (inner - sum(near))))

    def kerbside(self, tags: dict, stations: Stations, side: str, edge: np.ndarray,
                 parking: bool = True) -> np.ndarray:
        """Draw one side's shoulder, cycleway, its buffer (with its posts) and, if `parking`, its
        parking lane, edge inward, and return the travel way's edge on that side."""
        sign = 1 if side == "left" else -1
        edge = self.street_parking(tags, side, stations, edge)
        if (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
            # A shoulder at the edge (wiki Key:shoulder), `shoulder:<side>:width` wide; hatched
            # where `shoulder:<side>:markings=hatched` - a project convention, as OSM has no tag
            # for painted hatching along a way.
            inner = edge - self.eased(tags, f"shoulder:{side}:width", "shoulder", stations)
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
        parked = parking and (tags.get(f"parking:{side}") or tags.get("parking:both")) == "lane"
        if parked and _has_cycleway(tags, side) and parking_at_the_kerb(tags, side):
            edge, parked = self.parking_lane(tags, side, stations, edge), False
        if _has_cycleway(tags, side):
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
            edge = inner
            buffer = width_m(tags.get(f"cycleway:{side}:buffer"))
            if buffer:
                buffer = self.along(buffer, lambda t: width_m(t.get(f"cycleway:{side}:buffer")), stations)
                if any(v == "flex_post" for k, v in tags.items()
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
        line and its stalls; return its inner edge."""
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
        signed offset `bounds[j]`, lanes counted left to right - white (MUTCD 3B.06, STANDARDS.md):
        broken, solid where either lane's `change:lanes` keeps traffic from crossing it."""
        if tags.get("lane_markings") == "no":
            return
        changes = lane_changes(tags, lanes, backward)
        for j, offset in bounds.items():
            line = LineString(stations.line(offset))
            solid = changes[j - 1] in NO_CHANGE_RIGHT or changes[j] in NO_CHANGE_LEFT
            self.out["lane_lines"] += _lines(line) if solid else _dashes(line)
            self.stats[f"lane lines ({'solid' if solid else 'broken'}, pieces of way)"] += 1

    def stalls(self, tags: dict, side: str, stations: Stations, one: np.ndarray,
               other: np.ndarray) -> None:
        """A marked parking lane on this way's piece, kept for stall_all: its two edges and its
        `parking:<side>:capacity`."""
        markings = tags.get(f"parking:{side}:markings") or tags.get("parking:both:markings")
        nodes = self._way.get("node_ids") or []
        if markings != "yes" or len(nodes) < 2:
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
        reason - joined end to end, and the run's capacity, summed, laid in equal stalls along it,
        a divider across the lane at each end of each."""
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
            capacity = sum(p["capacity"] or 0 for p in run)
            if not capacity:
                continue
            one = LineString([c for k, p in enumerate(run) for c in p["one"][(k > 0):]])
            other = LineString([c for k, p in enumerate(run) for c in p["other"][(k > 0):]])
            for k in range(capacity + 1):
                at = k / capacity
                self.out["parking_stall_divider_lines"].append(
                    [list(one.interpolate(at, normalized=True).coords[0]),
                     list(other.interpolate(at, normalized=True).coords[0])])

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
        symbol = [(nose, 0.0), (0.0, half), (0.0, half / 3), (-nose, half / 3), (-nose, -half / 3),
                  (0.0, -half / 3), (0.0, -half)]
        arrow = [(nose, 0.0), (nose * 0.1, half), (nose * 0.1, half / 4), (-nose, half / 4),
                 (-nose, -half / 4), (nose * 0.1, -half / 4), (nose * 0.1, -half)]
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
        if markings in ("lines", "dashes", "dots"):
            lines = [chord.offset_curve(half).intersection(road),
                     chord.offset_curve(-half).intersection(road)]
            self.out["surveyed_crossings"].append({"lines": [c for g in lines for c in _lines(g)]})
            return
        # zebra and its variants: bars square to the chord, one bar width apart, as many as fit,
        # centred on it so the gaps at either kerb are equal.
        count = int((chord.length + ZEBRA_BAR_M) // (2 * ZEBRA_BAR_M))
        start = (chord.length - (2 * count - 1) * ZEBRA_BAR_M) / 2
        bars = []
        for i in range(count):
            piece = _substring(chord, start + 2 * i * ZEBRA_BAR_M, start + (2 * i + 1) * ZEBRA_BAR_M)
            bars += [ring for bar in _strip(piece, half)
                     for ring in _rings(Polygon(bar).intersection(road))]
        self.out["surveyed_crossings"].append({"bars": bars})

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

    def on_carriageway(self) -> None:
        """Paint is on the street and off the crosswalks: every marking clipped to the drawn
        surface, so where the street as built is narrower than its `width` the paint stops at its
        kerb, and cut out of every crosswalk's band, which a crosswalk cuts through - asphalt
        under its bars, the bikeway's green and lines stopping either side."""
        road = self.carriageway().difference(unary_union(self.crosswalk_bands))
        shapely.prepare(road)
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
                    "turn_box_surface_polygons", "bike_lane_symbol_polygons"):
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
        if forward is None and backward is None:
            self.stats["centre line: overtaking not tagged (pieces)"] += 1
            self.out["bike_lane_contraflow_lines"] += _lines(line)
            return
        self.stats["centre line: overtaking from OSM (pieces)"] += 1
        if forward == backward == "yes":
            self.out["bike_lane_contraflow_lines"] += _dashes(line)
            return
        # Forward traffic keeps right of the way, so its stripe is the one on the right.
        for allowed, offset in ((forward, -CENTRE_PAIR_OFFSET_M), (backward, CENTRE_PAIR_OFFSET_M)):
            stripe = line.offset_curve(offset)
            self.out["bike_lane_contraflow_lines"] += (_dashes(stripe) if allowed == "yes"
                                                      else _lines(stripe))

    def stop_line(self, way: dict) -> None:
        line = self.frame.line(way["coords_wgs84"])
        if line is not None:
            bar = line.buffer(STOP_BAR_M / 2, cap_style="flat").intersection(self.carriageway())
            self.out["surveyed_crossings"].append({"bars": _rings(bar)})

    def restriction(self, way: dict) -> None:
        """A `road_marking=restriction` area, hatched in its `colour` - one of HATCH_COLOURS, else
        white, the colour of a marking with none tagged."""
        line = self.frame.line(way["coords_wgs84"])
        if line is not None and len(line.coords) >= 4:
            # On a street - its centre inside that street's own surface - it is struck along that
            # street, so it hatches as one with the street's shoulders (hatch_all).
            polygons = _polygons(Polygon(line.coords).buffer(0))
            name = next((street for street, own in self.areas.items() if street and polygons
                         and any(a.contains(polygons[0].representative_point()) for a in own)), None)
            parts = [(part, name) for part in polygons]
            colour = way["tags"].get("colour")
            if colour in HATCH_COLOURS:
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
            ahead = np.asarray(along.interpolate(min(at + 0.5, along.length)).coords[0])
            behind = np.asarray(along.interpolate(max(at - 0.5, 0.0)).coords[0])
            t = (ahead - behind) / max(float(np.linalg.norm(ahead - behind)), 1e-9)
            d = (t + np.array([-t[1], t[0]])) / math.sqrt(2)      # 45 degrees off the street
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
        ahead = np.asarray(line.interpolate(min(line.length / 2 + 0.5, line.length)).coords[0])
        behind = np.asarray(line.interpolate(max(line.length / 2 - 0.5, 0.0)).coords[0])
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
        so each pad is centred there at the ramp's minimum width (STANDARDS.md 6a), not spread
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
        ahead = np.asarray(line.interpolate(min(line.length / 2 + 0.5, line.length)).coords[0])
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
        for d in np.arange(0.0, limit, 0.05):
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
        if (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
            section += width_m(tags.get(f"shoulder:{side}:width")) or DEFAULT_WIDTHS_M["shoulder"]
        if _has_cycleway(tags, side):
            section += (width_m(tags.get(f"cycleway:{side}:width")) or DEFAULT_WIDTHS_M["cycleway"]) \
                + (width_m(tags.get(f"cycleway:{side}:buffer")) or 0.0)
        position = _parking(tags, side)
        if position in ("lane", "half_on_kerb"):
            depth = parking_depth_m(tags, side) or DEFAULT_WIDTHS_M["parking"]
            section += depth if position == "lane" else depth / 2
    return section, "DEFAULTED (its tagged section summed)"


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


def _parking(tags: dict, side: str) -> str | None:
    """`side`'s street parking position (wiki Street_parking)."""
    return tags.get(f"parking:{side}") or tags.get("parking:both")


def _orientation(tags: dict, side: str) -> str:
    return tags.get(f"parking:{side}:orientation") or tags.get("parking:both:orientation") or "parallel"


def parking_depth_m(tags: dict, side: str) -> float | None:
    """How deep a parking lane is off the kerb: its `parking:<side>:width`, else by orientation -
    a diagonal bay `L sin(a) + W cos(a)`, a perpendicular one the stall's length (STANDARDS.md 1a);
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
    as `cycleway:<side>:traffic_mode:<kerb side>=parking` says (Proposal:Separation); else the
    lane is at the kerb and the parking outside it, protecting it."""
    return tags.get(f"cycleway:{side}:traffic_mode:{kerbward(tags, side)}") == "parking"


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


def _has_cycleway(tags: dict, side: str) -> bool:
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


def _mph(value: str | None) -> float:
    """A `maxspeed` in mph: "25 mph", or a bare number, which OSM reads as km/h. 0 where unknown."""
    if not value:
        return 0.0
    number = re.match(r"\s*(\d+(?:\.\d+)?)", value)
    if number is None:
        return 0.0
    return float(number.group(1)) * (1.0 if "mph" in value else 0.621371)


# A street with no `maxspeed`: NJ's statutory limit in a residence or business district (N.J.S.A.
# 39:4-98, as cited; STANDARDS.md 6c) - the speed a lateral shift's taper is sized for.
DEFAULT_MPH = 25.0


def taper_rate(tags: dict) -> float:
    """How far a marking may shift sideways per unit length: the MUTCD taper, L = W*S^2/60 at
    40 mph and under, L = W*S above (as cited; STANDARDS.md 6c), at the street's `maxspeed`."""
    mph = _mph(tags.get("maxspeed")) or DEFAULT_MPH
    return 60.0 / mph ** 2 if mph <= 40 else 1.0 / mph


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
    # The kerb measurement shares the proposal writer's kerb matching; imported here because
    # src/osm_osc.py imports this module.
    from src.osm_osc import kerb_centre_offsets
    reader = _Reader(frame, dict(areas), kerb_centre_offsets(layers, frame, SNAPSHOT_AREAS[area]))
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
