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
                where `cycleway:<side>:separation:*=flex_post`;
                `cycleway:<side>:oneway=no` adds the dashed yellow divider down its middle
  shoulder      `shoulder:<side>=yes`, `shoulder:<side>:width` at the edge; where
                `shoulder:<side>:markings=hatched` (a project convention), hatched from its inner
                line out to its street's own `area:highway` edge - everything the kerb as built
                leaves outside the markings
  parking       `parking:<side>=lane`, an edge line `parking:<side>:width` inside that edge;
                where `parking:<side>:markings=yes`, its `parking:<side>:capacity` stalls marked,
                in equal stalls along each run of parked pieces joined end to end
  centre line   two-way ways with 2+ lanes, or with `overtaking*` or `lane_markings=yes` tagged
                (has_centre_line), unless `lane_markings=no` - so where it stops is
                where the way is split and tagged, never decided here. Double yellow where
                `overtaking[:forward|:backward]=no`, dashed where `=yes`. On the way itself, or
                where `width:lanes` puts it - see _Reader.cross_section
  cycle crossings `highway=cycleway` + `cycleway=crossing` ways: a band of their `width`, green
                where `surface:colour=green`; where `crossing:markings=dashes`, edged in
                CROSSBIKE_DASH_M dots and the green in skip bars of the same pattern; a dashed
                yellow divider where `oneway=no`
  sidewalks     `footway=sidewalk` ways, buffered to their `width`
  crossings     `footway=crossing` ways, painted as their `crossing:markings` says, on the
                carriageway only
  markings      `road_marking=stop_line` bars (STOP_BAR_M wide), `road_marking=restriction`
                hatched areas
  kerbs         `barrier=kerb` ways, at the height their `kerb` / `kerb:height` says
  paved ground  driveways, parking aisles, `amenity=parking` areas and `highway=*` areas
                (src/sources/osm_context.py:is_highway_area) - except `parking=lane`
                areas, which are on the carriageway and drawn as their marked outline - and
                a driveway's mouth across the whole lowered kerb it crosses
  buildings     their footprints, at `height` / `building:levels`

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
from shapely.ops import linemerge

from src.geometry.model.crs import NJ_STATE_PLANE_FT, WGS84
from src.sources.osm_change import OsmChange, apply_change, load_change
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers

FT_TO_M = 0.3048

# Vehicular highway values (wiki Key:highway, "Roads" and "Link roads"), plus non-driveway service.
CARRIAGEWAY = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
               "residential", "living_street", "service", "motorway_link", "trunk_link",
               "primary_link", "secondary_link", "tertiary_link", "busway"}

# NOT OSM: what is drawn where a way carries no width tag. Counted every time it is used.
DEFAULT_WIDTHS_M = {"lane": 3.0, "service": 3.5, "cycleway": 1.5, "parking": 2.2,
                    "sidewalk": 6 * FT_TO_M, "crossing": 3.0,
                    "shoulder": 1.0}
DEFAULT_BUILDING_HEIGHT_M = 7.0

# Kerb heights for `kerb=*` (wiki Key:kerb), used where `kerb:height` is absent.
KERB_HEIGHT_M = {"raised": 0.15, "regular": 0.15, "rolled": 0.08, "lowered": 0.03, "flush": 0.0}

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
ZEBRA_BAR_M = 0.5            # a continental crossing's bar width, and the gap between bars
HATCH_SPACING_M = 1.0        # spacing of the hatch strokes across a restriction area
SEAM_M = 0.05                # hatching laid way by way leaves mm seams where consecutive ways turn
# The line channels that are paint, kept on the street surface (_Reader.on_carriageway).
PAINT_LINES = ("bike_lane_edge_lines", "parking_edge_lines", "bike_lane_contraflow_lines",
               "lane_narrowing_edge_lines", "lane_narrowing_hatch_lines", "parking_stall_divider_lines",
               "cycle_crossing_edge_lines",
               "cycle_crossing_divider_lines")

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
    """Cross-sections along a centreline every STATION_STEP_M: each one's point and left normal."""

    def __init__(self, line: LineString):
        length = line.length
        self.s = np.linspace(0.0, length, max(2, math.ceil(length / STATION_STEP_M) + 1))
        self.xy = np.array([line.interpolate(at).coords[0] for at in self.s])
        ahead = np.array([line.interpolate(min(at + 0.5, length)).coords[0] for at in self.s])
        behind = np.array([line.interpolate(max(at - 0.5, 0.0)).coords[0] for at in self.s])
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
        self.parking: list[dict] = []        # marked parking lanes, piece by piece: stall_all draws them
        self._way: dict = {}                 # the way being drawn
        # Each named street's centrelines and widest carriageway: what its hatching is struck along.
        self.street_lines: dict[str, list[LineString]] = defaultdict(list)
        self.street_width: dict[str, float] = defaultdict(float)
        self.crosswalk_bands: list[BaseGeometry] = []   # kept clear of other paint (on_carriageway)
        # Per road way: its nodes, drawn surface and edges - what a crossing is painted on.
        self.road_nodes: dict[int, set[int]] = {}
        self.road_surfaces: dict[int, list[Polygon]] = {}
        self.road_names: dict[int, str | None] = {}
        self.edges: dict[int, tuple[Stations, np.ndarray, np.ndarray]] = {}
        self.node_xy: dict[int, list[float]] = {}
        self.stats: Counter[str] = Counter()
        self.out: dict[str, list] = {
            "pavement": [], "sidewalks": [], "buildings": [], "kerbs": [], "paved_surfaces": [],
            "surveyed_crossings": [], "bike_lane_surface_polygons": [], "bike_lane_edge_lines": [],
            "parking_edge_lines": [], "bike_lane_contraflow_lines": [], "lane_narrowing_edge_lines": [],
            "lane_narrowing_hatch_lines": [], "tree_points": [], "props": [],
            "cycle_crossing_surface_polygons": [], "cycle_crossing_edge_lines": [],
            "cycle_crossing_divider_lines": [], "parking_stall_divider_lines": []}

    def width(self, tags: dict, key: str, default: str) -> float:
        found = width_m(tags.get(key))
        if found is not None:
            self.stats[f"{default} width from OSM"] += 1
            return found
        self.stats[f"{default} width DEFAULTED"] += 1
        return DEFAULT_WIDTHS_M[default]

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
        stations = Stations(line)
        self._way = way
        nodes = way.get("node_ids") or []
        self.node_xy |= {node: self.frame.point(*c) for node, c in zip(nodes, way["coords_wgs84"],
                                                                       strict=False)}
        # The way runs down the middle of its carriageway (wiki Key:placement, the default), so
        # its edges are half its `width` either side, and every marking is laid from them.
        half = np.full(len(stations.s), self.carriageway_width(tags) / 2)
        edges = {"left": half, "right": half}
        surface = stations.strips(-half, half)
        if tags.get("name") not in self.areas or tags.get("highway") == "service":
            self.out["pavement"] += surface      # where its street is mapped, the area is the pavement
        self.road_nodes[way["id"]] = set(nodes)
        self.road_names[way["id"]] = tags.get("name")
        if tags.get("name"):
            self.street_lines[tags["name"]].append(line)
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
        if len(widths) != lanes or None in widths:
            for side in ("left", "right"):
                self.kerbside(tags, stations, side, edges[side])
            return line
        anchor = "right" if _has_cycleway(tags, "right") and not _has_cycleway(tags, "left") else "left"
        far = "left" if anchor == "right" else "right"
        sign = 1 if anchor == "left" else -1
        inner = self.kerbside(tags, stations, anchor, edges[anchor])
        self.kerbside(tags, stations, far, edges[far], parking=False)
        backward = _int(tags.get("lanes:backward")) or lanes // 2
        near = widths[:backward] if anchor == "left" else widths[backward:]
        if (tags.get(f"parking:{far}") or tags.get("parking:both")) == "lane":
            lane_edge = inner - sum(widths)
            self.out["parking_edge_lines"].append(stations.line(sign * lane_edge))
            self.stalls(tags, far, stations, sign * lane_edge,
                        sign * (lane_edge - self.width(tags, f"parking:{far}:width", "parking")))
        self.stats["cross-section laid by width:lanes (ways)"] += 1
        return LineString(stations.line(sign * (inner - sum(near))))

    def kerbside(self, tags: dict, stations: Stations, side: str, edge: np.ndarray,
                 parking: bool = True) -> np.ndarray:
        """Draw one side's shoulder, cycleway, its buffer (with its posts) and, if `parking`, its
        parking lane, edge inward, and return the travel way's edge on that side."""
        sign = 1 if side == "left" else -1
        if (tags.get(f"shoulder:{side}") or tags.get("shoulder:both")) == "yes":
            # A shoulder at the edge (wiki Key:shoulder), `shoulder:<side>:width` wide; hatched
            # where `shoulder:<side>:markings=hatched` - a project convention, as OSM has no tag
            # for painted hatching along a way.
            inner = edge - self.width(tags, f"shoulder:{side}:width", "shoulder")
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
        if _has_cycleway(tags, side):
            inner = edge - self.width(tags, f"cycleway:{side}:width", "cycleway")
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
        if parking and (tags.get(f"parking:{side}") or tags.get("parking:both")) == "lane":
            outer, edge = edge, edge - self.width(tags, f"parking:{side}:width", "parking")
            self.out["parking_edge_lines"].append(stations.line(sign * edge))
            self.stalls(tags, side, stations, sign * edge, sign * outer)
        return edge

    def stalls(self, tags: dict, side: str, stations: Stations, one: np.ndarray,
               other: np.ndarray) -> None:
        """A marked parking lane on this way's piece, kept for stall_all: its two edges and its
        `parking:<side>:capacity`."""
        if tags.get(f"parking:{side}:markings") != "yes" or _int(tags.get(f"parking:{side}:capacity")) is None:
            return
        nodes = self._way.get("node_ids") or []
        if len(nodes) < 2:
            return
        self.parking.append({"street": tags.get("name"), "side": side, "first": nodes[0],
                             "last": nodes[-1], "capacity": _int(tags.get(f"parking:{side}:capacity")),
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
            capacity = sum(p["capacity"] for p in run)
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

    def sidewalk(self, way: dict) -> None:
        line = self.frame.line(way["coords_wgs84"])
        if line is not None:
            half = self.width(way["tags"], "width", "sidewalk") / 2
            self.out["sidewalks"] += _strip(line, half)

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
            self._carriageway = unary_union([part for ring in self.out["pavement"] if len(ring) >= 4
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
        for key in PAINT_LINES:
            clipped = [part for line in self.out[key] if len(line) >= 2
                       for part in _lines(LineString(line).intersection(road))]
            self.out[key] = ([[part[0], part[-1]] for part in clipped]
                             if key in ("lane_narrowing_hatch_lines", "parking_stall_divider_lines")
                             else clipped)
        for key in ("bike_lane_surface_polygons", "cycle_crossing_surface_polygons"):
            self.out[key] = [ring for poly in self.out[key] if len(poly) >= 4
                             for ring in _rings(Polygon(poly).buffer(0).intersection(road))]
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
        line = self.frame.line(way["coords_wgs84"])
        if line is not None and len(line.coords) >= 4:
            self.hatched += [(part, None) for part in _polygons(Polygon(line.coords).buffer(0))]

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
                    self.strokes(LineString([middle - long, middle + long]), piece, reach)
                continue
            shapely.prepare(area)
            for line in getattr(linemerge(self.street_lines[name]), "geoms", None) or [
                    linemerge(self.street_lines[name])]:
                self.strokes(line, area, self.street_width[name])

    def strokes(self, along: LineString, area: BaseGeometry, reach: float) -> None:
        """Strokes at 45 degrees to `along`, from points HATCH_SPACING_M * sqrt(2) apart on it -
        so HATCH_SPACING_M apart square to themselves - `reach` either side, clipped to `area`."""
        step = HATCH_SPACING_M * math.sqrt(2)
        for at in np.arange(0.0, along.length, step):
            p = np.asarray(along.interpolate(at).coords[0])
            ahead = np.asarray(along.interpolate(min(at + 0.5, along.length)).coords[0])
            behind = np.asarray(along.interpolate(max(at - 0.5, 0.0)).coords[0])
            t = (ahead - behind) / max(float(np.linalg.norm(ahead - behind)), 1e-9)
            d = (t + np.array([-t[1], t[0]])) / math.sqrt(2)      # 45 degrees off the street
            stroke = LineString([p - d * reach, p + d * reach])
            for piece in _lines(stroke.intersection(area)):
                self.out["lane_narrowing_hatch_lines"].append([piece[0], piece[-1]])

    def kerb(self, way: dict) -> None:
        line = self.frame.line(way.get("coords_wgs84") or [])
        if line is None:
            return
        tags = way["tags"]
        height = width_m(tags.get("kerb:height"))
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
        self.out["paved_surfaces"] += [{"coords": ring} for ring in rings]

    def apron(self, kerb: dict, branches: list[dict], sidewalks: list[dict]) -> None:
        """A dropped kerb's driveway mouth, paved. A `kerb=lowered` way that a driveway or parking
        aisle crosses (at a shared node, or across it) is where that traffic leaves the street, so
        the mouth is the whole lowered kerb, not the branch's own `width`; it is paved out to the
        sidewalk the branch crosses nearest the kerb, through that sidewalk's band - as far as the
        branch itself is paved. A lowered kerb nothing drives across (a crossing's ramp) is not."""
        if kerb["tags"].get("kerb") not in ("lowered", "flush"):
            return
        line = self.frame.line(kerb.get("coords_wgs84") or [])
        if line is None:
            return
        nodes = set(kerb.get("node_ids") or [])
        for branch in branches:
            path = self.frame.line(branch["coords_wgs84"])
            if path is None or not (nodes & set(branch["node_ids"]) or path.intersects(line)):
                continue
            met = [(path.intersection(walk).distance(line), walk, way) for way in sidewalks
                   if (walk := self.frame.line(way["coords_wgs84"])) is not None and walk.intersects(path)]
            if not met:
                self.stats["lowered kerbs a driveway crosses, no sidewalk to pave to"] += 1
                continue
            _d, walk, way = min(met, key=lambda m: m[0])
            back = [walk.interpolate(walk.project(Point(c))).coords[0] for c in line.coords]
            along = sorted(walk.project(Point(c)) for c in back)
            mouth = unary_union([
                Polygon([*line.coords, *back[::-1]]).buffer(0),
                _substring(walk, along[0], along[-1]).buffer(
                    self.width(way["tags"], "width", "sidewalk") / 2, cap_style="flat")])
            self.out["paved_surfaces"] += [{"coords": ring} for ring in _rings(mouth)]
            self.stats["driveway mouths paved across their lowered kerb"] += 1
            return

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
    return lanes * DEFAULT_WIDTHS_M["lane"], "DEFAULTED (lanes x lane)"


def _has_cycleway(tags: dict, side: str) -> bool:
    return (tags.get(f"cycleway:{side}") or tags.get("cycleway:both")
            or tags.get("cycleway")) in ("lane", "track")


def _seamless(parts: list[Polygon]) -> BaseGeometry:
    """`parts` as one area. Each way's band ends square to its own last segment, so where the next
    way turns the two miss by millimetres and a plain union keeps them apart - an outline across
    the hatching at every split. Closing by SEAM_M (mitred, so corners stay square) joins them."""
    return (unary_union(parts).buffer(SEAM_M, join_style="mitre")
            .buffer(-SEAM_M, join_style="mitre").buffer(0))


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
    for way in layers.get("road_areas", []):
        line = frame.line(way["coords_wgs84"])
        if line is not None and len(line.coords) >= 4:
            areas[way["tags"].get("name")] += _polygons(Polygon(line.coords).buffer(0))
    reader = _Reader(frame, dict(areas))
    reader.out["pavement"] += [ring for polygons in areas.values() for p in polygons
                               for ring in _rings(p)]
    for way in layers["roads"]:
        if way["tags"].get("highway") == "cycleway" and way["tags"].get("cycleway") == "crossing":
            reader.cycle_crossing(way)
        else:
            reader.road(way)
    for way in layers["sidewalks"]:
        reader.sidewalk(way)
    for way in layers["crossings"]:
        reader.crossing(way)
    for way in layers["stop_lines"]:
        reader.stop_line(way)
    for way in layers["road_markings"]:
        reader.restriction(way)
    for way in layers["kerbs"]:
        reader.kerb(way)
    for way in layers["driveways"] + layers["parking_aisles"]:
        reader.paved(way, closed=False)
    for way in layers["parking_lots"] + layers["highway_areas"]:
        reader.paved(way, closed=True)
    for way in layers["kerbs"]:
        reader.apron(way, layers["driveways"] + layers["parking_aisles"], layers["sidewalks"])
    for way in layers["buildings"]:
        reader.building(way)
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
