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

from src.geometry.context_roads import is_carriageway
from src.geometry.intersection.municipality import municipal_boundary_ft
from src.geometry.network.corridor import Corridor, _street_name
from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import SNAPSHOT_AREAS, fetch_borough_osm

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
    for name, way in _named_carriageways(snapshot):
        line = _way_line(way, xy)
        if line is None or not line.intersects(boundary):
            continue
        lines_by_name.setdefault(name, []).append(line)
        for nid in way.get("nodes", []):
            if nid in xy:
                by_node.setdefault(nid, set()).add(name)

    # Clipped to the boundary so `municipalities` is true of every foot: unclipped, Carter Rd put
    # 6,744 ft of Hopewell Township into a Borough document, and a route decision is keyed on town.
    return [Corridor(name=name, centerline=piece, junctions=(), kerb_runs=(),
                     municipalities=(boundary_name,),
                     cross_street_ft=_cross_street_ft(piece, name, by_node, xy))
            for name, lines in sorted(lines_by_name.items())
            for run in _connected_runs(lines)
            for piece in _pieces_of(run.intersection(boundary))
            if piece.length >= MIN_CORRIDOR_FT]
