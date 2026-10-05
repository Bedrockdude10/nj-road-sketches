"""OSM's street network as osmnx builds it: edges between junctions, nodes that ARE the junctions.

This replaces this project's own junction detection. OSM already says where streets meet - a node
two ways share - and osmnx turns a set of ways into exactly that graph: each edge is a stretch of
one way between two endpoints, in the way's direction, carrying the way id; each node knows every
edge that starts or ends at it. Nothing here snaps, rounds or matches geometry.

A JUNCTION is a node where differently named streets meet (`area.py`'s rule): a node two ways of
one street share is OSM splitting the way for a tag change, not a junction.
"""
from __future__ import annotations

from itertools import pairwise

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

from src.geometry.context_roads import is_carriageway
from src.geometry.intersection.paved import to_state_plane
from src.geometry.network.corridor import _street_name

#: `access` values that make a way nobody's to redesign - the same set area.py excludes.
NOT_PUBLIC_ACCESS = frozenset({"private", "customers"})


class StreetEdge:
    """One edge of the street graph: a stretch of one OSM way between two graph endpoints."""

    __slots__ = ("line", "name", "tags", "u", "v", "way_id")

    def __init__(self, u: int, v: int, way_id: int, line: LineString, tags: dict, name: str):
        self.u, self.v, self.way_id, self.line, self.tags, self.name = u, v, way_id, line, tags, name


def is_street(tags: dict) -> bool:
    """A named public carriageway - the ways a junction is made of."""
    return bool(tags.get("name")) and is_carriageway(tags) \
        and tags.get("access") not in NOT_PUBLIC_ACCESS


def street_graph(roads: list[dict]) -> nx.MultiDiGraph:
    """osmnx's simplified graph of `roads` (osm_layers' shape), in state-plane feet.

    One directed edge per consecutive node pair IN THE WAY'S OWN ORDER - OSM's direction, not
    osmnx's routing duplicates - then `ox.simplify_graph` keeps only true endpoints, and
    `edge_attrs_differ=["osmid"]` keeps every way boundary so each edge is one way and carries
    that way's tags whole (a restriction over part of a street is OSM splitting the way).
    """
    graph = nx.MultiDiGraph(crs="EPSG:3424")
    for road in roads:
        tags = road.get("tags") or {}
        ids = road.get("node_ids") or []
        if not is_street(tags) or len(ids) < 2 or len(ids) != len(road["coords_wgs84"]):
            continue
        for node_id, (x, y) in zip(ids, to_state_plane(road["coords_wgs84"])):
            graph.add_node(node_id, x=x, y=y)
        for u, v in pairwise(ids):
            if u != v:
                graph.add_edge(u, v, osmid=road["id"])
    return ox.simplify_graph(graph, edge_attrs_differ=["osmid"], remove_rings=False)


def street_edges(roads: list[dict]) -> tuple[list[StreetEdge], set[int], dict[int, tuple[float, float]]]:
    """(edges, junction node ids, node xy) for `roads`."""
    graph = street_graph(roads)
    tags_by_way = {road["id"]: road.get("tags") or {} for road in roads}
    xy = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}
    edges = []
    for u, v, data in graph.edges(data=True):
        way_id = data["osmid"]
        line = data.get("geometry") or LineString([xy[u], xy[v]])
        tags = tags_by_way[way_id]
        edges.append(StreetEdge(u, v, way_id, line, tags, _street_name(tags["name"])))
    names_at: dict[int, set[str]] = {}
    for edge in edges:
        names_at.setdefault(edge.u, set()).add(edge.name)
        names_at.setdefault(edge.v, set()).add(edge.name)
    junctions = {node for node, names in names_at.items() if len(names) > 1}
    return edges, junctions, xy
