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
from shapely.ops import linemerge, polygonize, unary_union

from src.geometry.context_roads import assumed_width_ft, is_carriageway
from src.geometry.intersection.municipality import municipal_boundary_ft
from src.geometry.model import station_offset_many
from src.geometry.network.corridor import Corridor, _street_name
from src.geometry.network.kerb import KerbRun, _traced_kerb_runs
from src.render.coords import wgs84_to_state_plane
from src.sources.observations import ELEMENT_FROM_OSM, apply_observations, load_observations
from src.sources.osm_context import (SNAPSHOT_AREAS, fetch_borough_osm, is_building,
                                     is_crossing_way, is_driveway, is_kerb, is_parking_aisle,
                                     is_parking_lot, is_road, is_sidewalk, is_stop_line)

Bbox = tuple[float, float, float, float]
NodeXY = dict[int, tuple[float, float]]

#: Shorter than this inside the boundary is a tail the clip left, not a street. Hopewell's
#: shortest real block is ~250 ft.
MIN_CORRIDOR_FT: float = 100.0

#: A node must land back within this of where it started to count as ON the corridor - otherwise
#: a parallel street's node projects onto this axis and reports a junction that is not there.
ON_AXIS_FT: float = 1.0

#: `access` values that say a way is not the public's to redesign - a corridor carries a route
#: decision, and a route decision presumes anyone may use the road. `private` and `customers` say
#: so explicitly (Eaton Court: "Private service road on private property"). `permissive` and
#: `yes` are still open to the public, so they stay in. `unknown` asserts nothing, so it defaults
#: open too - an absent statement is not a restriction. This is NOT a carriageway gate: the way is
#: still real asphalt (`is_carriageway` in src/geometry/context_roads.py still says yes, and
#: `_context_roadways_ft` still draws it), only ineligible to become a Corridor.
NOT_PUBLIC_ACCESS = frozenset({"private", "customers"})


def _projected_nodes(nodes: dict[int, dict]) -> NodeXY:
    """Every node in state-plane feet, transformed in one call."""
    ids = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
    lons = np.fromiter((nodes[i]["lon"] for i in ids), dtype=float, count=len(ids))
    lats = np.fromiter((nodes[i]["lat"] for i in ids), dtype=float, count=len(ids))
    xs, ys = wgs84_to_state_plane.transform(lons, lats)
    return dict(zip(ids.tolist(), zip(xs.tolist(), ys.tolist())))


def _named_carriageways(snapshot: dict) -> list[tuple[str, dict]]:
    """(normalised name, way) per named carriageway. A corridor is keyed on its name, so an
    unnamed service road has nothing to be part of.

    Gated on `access` HERE, not in `is_carriageway`: `is_carriageway` is also what
    `_context_roadways_ft` (src/geometry/intersection/paved.py) uses to draw context roadways, so
    excluding a private way there would erase real asphalt from the picture. A private road is
    still asphalt; it is just not a corridor - nothing about it is the borough's to redesign, so
    it gets no route decision and no `street`/`pavement` row, but it stays in the document as its
    own `osm_roads` row and still gets drawn as context. See `NOT_PUBLIC_ACCESS` above.
    """
    return [(_street_name(tags["name"]), way)
            for way in snapshot["ways"]
            if (tags := way.get("tags") or {}).get("name") and is_carriageway(tags)
            and tags.get("access") not in NOT_PUBLIC_ACCESS]


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


#: {layer: (predicate on tags, how the geometry is built)}. THE SAME PREDICATES THE FETCHERS USE
#: - one definition, two readers, which is the whole reason they were extracted to module level in
#: src/sources/osm_context.py. Order matters only where a way matches twice: a parking lot is also
#: an area, and a driveway is also a road, so the specific layers are tested before `roads`.
#:
#: A LABEL, NOT A GATE. A way matching none of these is not nothing - `area_context` files it
#: under "other", geometry and tags verbatim. An allowlist here would be lossy by construction,
#: and this project's SSOT claim is that nothing tagged is dropped, only left unlabelled for
#: whichever reader has no use for that kind yet. Sidewalks are labelled despite being derivable
#: at a junction (`build_sidewalk_pieces` widens the corner ring) because a crop has no corner
#: ring: the surveyed footway is the only sidewalk a window has.
_AREA_LAYERS: tuple[tuple[str, object, str], ...] = (
    ("buildings", is_building, "ring"),
    ("parking_lots", is_parking_lot, "ring"),
    ("crossings", is_crossing_way, "line"),
    ("sidewalks", is_sidewalk, "line"),
    ("kerb_ways", is_kerb, "line"),
    ("driveways", is_driveway, "line"),
    ("parking_aisles", is_parking_aisle, "line"),
    ("stop_lines", is_stop_line, "line"),
    ("roads", is_road, "line"),
)


def _other_geometry(way: dict, xy: NodeXY) -> LineString | Polygon | None:
    """The geometry OSM's own vertex count and closure imply for a way no `_AREA_LAYERS`
    predicate claims: a ring where it closes on itself with enough points to bound an area, a
    line otherwise - including where the "ring" is degenerate enough that `buffer(0)` empties
    it, so a line is still tried rather than giving up. Only a way with fewer than two resolved
    vertices carries no geometry at all, and `area_context` counts those rather than dropping
    them without a word.
    """
    coords = [xy[nid] for nid in way.get("nodes", []) if nid in xy]
    if len(coords) >= 4 and coords[0] == coords[-1]:
        ring = Polygon(coords).buffer(0)
        if not ring.is_empty:
            return ring
    return LineString(coords) if len(coords) >= 2 else None


def _multipolygon_rows(snapshot: dict, xy: NodeXY, boundary: Polygon) -> tuple[list[dict], int]:
    """One row per resolvable `type=multipolygon` relation, tagged with the RELATION's own tags.

    THE TAGS LIVE ON THE RELATION, NOT THE MEMBERS - "Hopewell Boro Park" is a named
    leisure=park whose two member ways carry no tags of their own, and two natural=wood plus one
    landuse=farmland relations have members tagged only source=NJ2002LULC. No sweep over way
    tags can ever recover what only the relation says, which is the whole reason relations are
    read at all.

    `polygonize` reconstructs a ring from however many arcs a member way was split into, exactly
    as `_connected_runs` reconstructs a corridor from OSM's own splits. A relation whose outer
    members don't close all the way round - here, because a member way sits outside this
    snapshot's bbox and was never downloaded - yields no ring and is counted as unresolved, never
    closed for it (the same refusal `fetch_municipality_containing` makes for the admin ring).
    """
    ways = {w["id"]: w for w in snapshot["ways"]}
    rows: list[dict] = []
    unclosed = 0
    for relation in snapshot.get("relations", []):
        tags = relation.get("tags") or {}
        if tags.get("type") != "multipolygon":
            continue
        outer_lines: list[LineString] = []
        inner_lines: list[LineString] = []
        for member in relation.get("members", []):
            if member.get("type") != "way":
                continue
            way = ways.get(member["ref"])
            line = _way_line(way, xy) if way is not None else None
            if line is not None:
                (inner_lines if member.get("role") == "inner" else outer_lines).append(line)
        outers = list(polygonize(outer_lines))
        if not outers:
            unclosed += 1
            continue
        polygon = unary_union(outers)
        inners = list(polygonize(inner_lines)) if inner_lines else []
        if inners:
            polygon = polygon.difference(unary_union(inners))
        if polygon.is_empty or not polygon.intersects(boundary):
            continue
        rows.append({"geometry": polygon, "tags": tags, "id": relation.get("id")})
    return rows, unclosed


def area_context(area: str = "hopewell_borough", snapshot: dict | None = None) -> dict[str, list]:
    """Every OSM element inside a municipality's boundary, AS OSM RECORDS IT - lossless from OSM.

    A way lands under whichever `_AREA_LAYERS` label claims its tags, or "other" if none do:
    the predicate table is a LABEL, never a GATE, so a `landuse=residential` parcel or a
    `waterway=stream` is filed rather than skipped for being a kind no renderer has asked for
    yet. Every TAGGED node inside the boundary is carried the same way, role undecided - so a
    node that is ALSO a member of a carried way still gets its own row, because a bare id in
    that way's `node_ids` cannot carry the node's own tags. `relations` holds every resolvable `type=multipolygon` relation;
    see `_multipolygon_rows` for why that one needs its own reader.

    Each entry is `{"geometry": ..., "tags": {...}}` plus `node_ids` where the topology is the
    point - geometry in feet, tags verbatim. NOTHING IS TRANSLATED HERE, which is the whole
    design: a building's height and a crossing's markings are already functions of its tags
    (`height_from_tags`, `_markings_from_tags`), so storing either alongside the tags makes a
    second copy free to disagree with the first. The reader derives what it needs, once.

    Read from the SAME snapshot `area_corridors` reads, not through `fetch_buildings` and
    friends: those take a centre and a radius, and the largest radius that fits inside the
    declared bbox is smaller than the borough, so the corners would lose their context. An area
    has no centre to measure from, which is the same reason `_area_kerb_ways` exists.

    THE ONE MERGE POINT for observations/<area>.yaml (src/sources/observations.py): every
    field observation is additive OSM tags on an OSM element, applied to the snapshot here,
    before anything below reads it - so a slice and a site built from the same area can never
    see different tags, and no reader downstream needs to know observations exist at all.
    """
    bbox: Bbox = SNAPSHOT_AREAS[area]
    snapshot = snapshot if snapshot is not None else fetch_borough_osm(bbox=bbox)
    snapshot = apply_observations(snapshot, load_observations(area))
    xy = _projected_nodes(snapshot["nodes"])
    found = municipal_boundary_ft(Point(*_snapshot_center(bbox)))
    if found is None:
        raise RuntimeError(f"no admin_level=8 boundary at the centre of {area!r}")
    _, boundary = found

    out: dict[str, list] = {layer: [] for layer, _p, _g in _AREA_LAYERS}
    out["nodes"] = []
    out["other"] = []
    out["relations"] = []
    unbuildable = 0
    for way in snapshot["ways"]:
        tags = way.get("tags") or {}
        if not tags:
            continue   # no tags, no geometry-only reason to keep - see _AREA_LAYERS
        match = next(((layer, kind) for layer, predicate, kind in _AREA_LAYERS if predicate(tags)),
                     None)
        if match is not None:
            layer, kind = match
            geometry = _closed_ring(way, xy) if kind == "ring" else _way_line(way, xy)
        else:
            layer = "other"
            geometry = _other_geometry(way, xy)
            if geometry is None:
                unbuildable += 1
        if geometry is not None and not geometry.is_empty and geometry.intersects(boundary):
            # The node ids are carried because one rule reads them: a tactile pad is placed at a
            # node SHARED by a crossing way and a tactile_paving kerb way, so the topology is the
            # observation and the geometry alone cannot express it.
            out[layer].append({"geometry": geometry, "tags": tags, "id": way.get("id"),
                               "node_ids": tuple(way.get("nodes") or ()),
                               "provenance": way.get("provenance", ELEMENT_FROM_OSM)})
    # EVERY tagged node, with no role written down beside it: which of the three a node plays is
    # a function of its tags (`is_traffic_control`, `is_street_furniture`, `is_kerb`), and a
    # column repeating that is the second copy this module exists to avoid - the reader derives
    # it, as render_slice.slice_context does. Its own row even where it is a member of a carried
    # way, because a bare id in that way's `node_ids` cannot carry the node's tags.
    out["nodes"] = [{"geometry": point, "tags": tags, "id": node_id,
                     "provenance": node.get("provenance", ELEMENT_FROM_OSM)}
                    for node_id, node in snapshot["nodes"].items()
                    if (tags := node.get("tags") or {})
                    and (point := Point(xy[node_id])).intersects(boundary)]
    out["relations"], unclosed = _multipolygon_rows(snapshot, xy, boundary)
    if unbuildable:
        print(f"  area_context: {unbuildable} tagged way(s) in {area!r} have fewer than two "
              f"resolved vertices and carry no geometry")
    if unclosed:
        print(f"  area_context: {unclosed} multipolygon relation(s) in {area!r} have outer "
              f"rings that don't close (a member way outside this snapshot) and are not in "
              f"the document")
    return out
