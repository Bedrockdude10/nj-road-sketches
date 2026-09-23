"""Every named street in a municipality as a Corridor, from OSM alone.

`corridors_from_models` can only answer about streets we already modelled - it reaches into
`models` for topology, per-leg SRI, kerb ways and centre. Here the street comes first: a road is
the ways sharing a name, a junction is a node where a differently named way meets it. Hopewell
Borough has 111 such junctions against the 7 this project models.

`junctions`, `kerb_runs` and `sri_spans` are empty - they need a modelled junction or an NJDOT
match, so `station_of` raises and `sites` is empty. Nothing renders from this yet.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge

from src.geometry.context_roads import assumed_width_ft, is_carriageway
from src.geometry.intersection.municipality import municipal_boundary_ft
from src.geometry.model import station_offset_many
from src.geometry.network.corridor import Corridor, _street_name
from src.geometry.network.kerb import KerbRun, _traced_kerb_runs
from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import (DEFAULT_BUILDING_HEIGHT_M, METERS_PER_LEVEL,
                                     SNAPSHOT_AREAS, fetch_borough_osm)

Bbox = tuple[float, float, float, float]
NodeXY = dict[int, tuple[float, float]]

#: Shorter than this inside the boundary is a tail the clip left, not a street. Hopewell's
#: shortest real block is ~250 ft.
MIN_CORRIDOR_FT: float = 100.0

#: A node must land back within this of where it started to count as ON the corridor - otherwise
#: a parallel street's node projects onto this axis and reports a junction that is not there.
ON_AXIS_FT: float = 1.0


def _projected_nodes(nodes: dict[int, dict]) -> NodeXY:
    """Every node in state-plane feet, transformed in one call."""
    ids = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
    lons = np.fromiter((nodes[i]["lon"] for i in ids), dtype=float, count=len(ids))
    lats = np.fromiter((nodes[i]["lat"] for i in ids), dtype=float, count=len(ids))
    xs, ys = wgs84_to_state_plane.transform(lons, lats)
    return dict(zip(ids.tolist(), zip(xs.tolist(), ys.tolist())))


def _named_carriageways(snapshot: dict) -> list[tuple[str, dict]]:
    """(normalised name, way) per named carriageway. A corridor is keyed on its name, so an
    unnamed service road has nothing to be part of."""
    return [(_street_name(tags["name"]), way)
            for way in snapshot["ways"]
            if (tags := way.get("tags") or {}).get("name") and is_carriageway(tags)]


def _way_line(way: dict, xy: NodeXY) -> LineString | None:
    coords = [xy[nid] for nid in way.get("nodes", []) if nid in xy]
    return LineString(coords) if len(coords) >= 2 else None


def _pieces_of(geometry) -> list[LineString]:
    """The LineStrings in whatever a clip returned - one, several, or a Point where it grazed."""
    return [part for part in getattr(geometry, "geoms", [geometry])
            if isinstance(part, LineString) and not part.is_empty]


def _connected_runs(lines: list[LineString]) -> list[LineString]:
    """One LineString per connected run of a street. `linemerge` chains and refuses to bridge a
    real gap, which is the wanted distinction: stationing cannot cross a gap.

    EVERY run, not the longest - keeping one lost 0.457 mi of the borough's 9.271 mi, unevenly:
    Eaton Pl kept 341 ft of 1,135 and Prospect St 2,545 of 3,697.
    """
    return _pieces_of(linemerge(lines)) if lines else []


def _cross_street_ft(centerline: LineString, name: str, by_node: dict[int, set[str]],
                     xy: NodeXY) -> tuple[float, ...]:
    """Stations where another named street meets this one, modelled or not."""
    shared = [nid for nid, names in by_node.items() if name in names and len(names) > 1]
    if not shared:
        return ()
    points = np.asarray([xy[nid] for nid in shared], dtype=float)
    stations = shapely.line_locate_point(centerline, shapely.points(points))
    landed = shapely.get_coordinates(shapely.line_interpolate_point(centerline, stations))
    on_axis = np.hypot(*(landed - points).T) <= ON_AXIS_FT
    return tuple(np.unique(np.round(stations[on_axis], 2)).tolist())


def _snapshot_center(bbox: Bbox) -> tuple[float, float]:
    west, south, east, north = bbox
    return (west + east) / 2.0, (south + north) / 2.0


def _area_kerb_ways(snapshot: dict, xy: NodeXY) -> dict[int, tuple[LineString, dict]]:
    """{way id: (line in feet, tags)} for every traced kerb in the snapshot.

    Read straight from the snapshot rather than per-junction at a radius, which is what
    `_corridor_kerb_ways` must do. That radius is also a failure mode: ebroad_elm's 400 m window
    reaches past the borough bbox and raises SiteOutsideSnapshotError, so the site cannot take
    part in a corridor at all. An area has no centre to measure from, so the question disappears.
    """
    return {way["id"]: (line, way.get("tags") or {})
            for way in snapshot["ways"]
            if (way.get("tags") or {}).get("barrier") == "kerb"
            and (line := _way_line(way, xy)) is not None}


def _kerb_runs_for(centerline: LineString, cross_street_ft: tuple[float, ...],
                   kerb_ways: dict[int, tuple[LineString, dict]]) -> tuple[KerbRun, ...]:
    """This corridor's traced kerb, as one run per unbroken stretch.

    `_traced_kerb_runs` already takes a centreline, ways and node stations - no model - so the
    corridor's own crossings serve as the nodes whose corner returns suspend the heading test.
    Every run is KERB_FROM_TRACING: there is no modelled junction here to contribute the other
    kind, and saying so is the point of the provenance field.
    """
    return tuple(_traced_kerb_runs(centerline, kerb_ways, tuple(cross_street_ft)))


def area_corridors(area: str = "hopewell_borough",
                   snapshot: dict | None = None) -> list[Corridor]:
    """Every named street inside one municipality's boundary, as a Corridor each.

    Built from the whole snapshot, not a radius around a site: the largest radius that fits inside
    the declared area still left 12 of 111 junctions out.
    """
    bbox: Bbox = SNAPSHOT_AREAS[area]
    snapshot = snapshot if snapshot is not None else fetch_borough_osm(bbox=bbox)
    xy = _projected_nodes(snapshot["nodes"])

    found = municipal_boundary_ft(Point(*_snapshot_center(bbox)))
    if found is None:
        raise RuntimeError(f"no admin_level=8 boundary at the centre of {area!r} - its streets "
                           f"have nothing to clip to, and municipality is half of "
                           f"route_decision_for's key. Re-pull the snapshot with relations kept.")
    boundary_name: str
    boundary: Polygon
    boundary_name, boundary = found

    by_node: dict[int, set[str]] = {}
    lines_by_name: dict[str, list[LineString]] = {}
    width_by_name: dict[str, float] = {}
    for name, way in _named_carriageways(snapshot):
        line = _way_line(way, xy)
        if line is None or not line.intersects(boundary):
            continue
        lines_by_name.setdefault(name, []).append(line)
        # The WIDEST way's width, not the first or the mean: OSM splits a street at every change
        # and a short narrow segment would otherwise shrink the whole corridor's untraced asphalt.
        width_by_name[name] = max(width_by_name.get(name, 0.0),
                                  assumed_width_ft(way.get("tags", {})))
        for nid in way.get("nodes", []):
            if nid in xy:
                by_node.setdefault(nid, set()).add(name)

    kerb_ways = _area_kerb_ways(snapshot, xy)

    # Clipped to the boundary so `municipalities` is true of every foot: unclipped, Carter Rd put
    # 6,744 ft of Hopewell Township into a Borough document, and a route decision is keyed on town.
    pieces = [(name, piece)
              for name, lines in sorted(lines_by_name.items())
              for run in _connected_runs(lines)
              for piece in _pieces_of(run.intersection(boundary))
              if piece.length >= MIN_CORRIDOR_FT]

    corridors = []
    for name, piece in pieces:
        crossings = _cross_street_ft(piece, name, by_node, xy)
        corridors.append(Corridor(name=name, centerline=piece, junctions=(),
                                  nominal_width_ft=width_by_name[name],
                                  kerb_runs=_kerb_runs_for(piece, crossings, kerb_ways),
                                  municipalities=(boundary_name,),
                                  cross_street_ft=crossings))
    return corridors


def corridor_pavement(corridor: Corridor) -> Polygon | None:
    """This street's asphalt, walked between ITS OWN traced kerbs.

    The kerb runs on a Corridor were already claimed by it, so unlike `assign_kerbs_to_roads`
    there is no nearest-road contest left to run - the vertices are handed straight over. Where a
    side is untraced the surface falls back to `nominal_width_ft`, which is why an untraced street
    still draws as a street rather than as nothing.
    """
    from src.geometry.context_roads import kerb_points, roadway_surface

    points = kerb_points([run.line for run in corridor.kerb_runs])
    stations, offsets = (station_offset_many(corridor.centerline, points)
                         if len(points) else (np.empty(0), np.empty(0)))
    surface, _, _ = roadway_surface(corridor.centerline, stations, offsets,
                                    corridor.nominal_width_ft)
    return surface


def _closed_ring(way: dict, xy: NodeXY) -> Polygon | None:
    line = _way_line(way, xy)
    return Polygon(line.coords).buffer(0) if line is not None and len(line.coords) >= 4 else None


def _building_height_m(tags: dict) -> float:
    """OSM's own answer where it has one, else the flat default. NOT the assessor's, which is a
    per-parcel join a whole municipality does not justify - src/sources/assessor.py owns that."""
    try:
        return float(str(tags["height"]).split()[0])
    except (KeyError, ValueError, IndexError):
        pass
    try:
        return float(tags["building:levels"]) * METERS_PER_LEVEL
    except (KeyError, ValueError):
        return DEFAULT_BUILDING_HEIGHT_M


def area_context(area: str = "hopewell_borough", snapshot: dict | None = None) -> dict[str, list]:
    """The context layers - buildings, sidewalks, crossings - for a whole municipality, in feet.

    Read from the SAME snapshot `area_corridors` reads, not through `fetch_buildings` and
    friends: those take a centre and a radius, and the largest radius that fits inside the
    declared bbox is smaller than the borough, so the corners would lose their context. An area
    has no centre to measure from, which is the same reason `_area_kerb_ways` exists.

    Buildings are (polygon, height_m); crossings are SurveyedCrossings, so the markings they get
    are the surveyor's - `crossing_bars_ft` paints nothing on a crossing recorded as unmarked.
    """
    from src.geometry.surveyed import SurveyedCrossing, _markings_from_tags

    bbox: Bbox = SNAPSHOT_AREAS[area]
    snapshot = snapshot if snapshot is not None else fetch_borough_osm(bbox=bbox)
    xy = _projected_nodes(snapshot["nodes"])
    found = municipal_boundary_ft(Point(*_snapshot_center(bbox)))
    if found is None:
        raise RuntimeError(f"no admin_level=8 boundary at the centre of {area!r}")
    _, boundary = found

    out: dict[str, list] = {"buildings": [], "sidewalks": [], "crossings": []}
    for way in snapshot["ways"]:
        tags = way.get("tags") or {}
        if "building" in tags:
            ring = _closed_ring(way, xy)
            if ring is not None and not ring.is_empty and ring.intersects(boundary):
                out["buildings"].append((ring, _building_height_m(tags)))
        elif tags.get("footway") == "sidewalk":
            line = _way_line(way, xy)
            if line is not None and line.intersects(boundary):
                out["sidewalks"].append(line)
        elif tags.get("footway") == "crossing":
            line = _way_line(way, xy)
            if line is not None and line.intersects(boundary):
                out["crossings"].append(SurveyedCrossing(
                    geometry=line, markings=_markings_from_tags(tags), tags=tags,
                    leg=None, distance_ft=0.0))
    return out
