"""A proposal is an osmChange: what the street would carry, as the edits OSM itself would take.

proposals/<area>/<scenario>.osc is a real osmChange file (OSM API 0.6) - JOSM reads it - so a
proposal is stated in OSM's own terms and nothing else: tags on a way, a way split where its
tags change (wiki Street parking: "The roadway needs to be split up where any of the properties
changes"), a new way such as a road_marking=restriction area. `apply_change` applies one to the
layers `osm_layers` returns, routing every created or modified way through the same predicates
`osm_layers` sorts OSM by, so a proposed scenario is drawn by exactly the code that draws the
street as OSM records it. Nothing downstream knows an element came from a proposal.

Supported: <create> node (a vertex, or with tags a feature of its own) and way, <modify> node
(moved, or retagged) and way, <delete> way. Every created or modified way
carries a `note` (OSM's Key:note) saying where it came from - a proposal with no provenance is
refused, as an observation with no `source` is.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROPOSALS_DIR = Path(__file__).resolve().parents[2] / "proposals"


@dataclass(frozen=True)
class NewNode:
    """A node the change creates (negative id, tags only where the node is a feature) or moves
    (an existing node's positive id, with its FULL tag set, as an osmChange modify carries)."""
    id: int
    lon: float
    lat: float
    tags: dict[str, str]


@dataclass(frozen=True)
class WayChange:
    """One way the change creates, modifies or deletes. `tags` is the FULL tag set, as osmChange
    has it; `node_ids` is empty for a delete."""
    id: int
    action: Literal["create", "modify", "delete"]
    node_ids: tuple[int, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class OsmChange:
    nodes: tuple[NewNode, ...]
    ways: tuple[WayChange, ...]


def proposal_path(area: str, scenario: str) -> Path:
    return PROPOSALS_DIR / area / f"{scenario}.osc"


_ACTIONS = ("create", "modify", "delete")


def _canonical(change: OsmChange) -> OsmChange:
    """Created nodes -1, -2..., then moved ones by id ascending; creates by id descending, then
    modifies, then deletes, by id ascending - so a change written and loaded again is the same."""
    def order(way: WayChange) -> tuple[int, int]:
        return _ACTIONS.index(way.action), -way.id if way.action == "create" else way.id

    return OsmChange(tuple(sorted(change.nodes, key=lambda node: (node.id > 0, abs(node.id)))),
                     tuple(sorted(change.ways, key=order)))


def _attr(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"<{element.tag} id={element.get('id')}> has no {name}")
    return value


def _id(element: ET.Element) -> int:
    return int(_attr(element, "id"))


def _tags(element: ET.Element) -> dict[str, str]:
    return {_attr(tag, "k"): _attr(tag, "v") for tag in element.findall("tag")}


def _way(element: ET.Element, action: str) -> WayChange:
    way_id = _id(element)
    if (way_id < 0) != (action == "create"):
        raise ValueError(f"{action} way/{way_id}: a created element has a negative id, an "
                         f"existing one a positive id")
    if action == "delete":
        return WayChange(way_id, "delete", (), {})
    node_ids, tags = tuple(int(_attr(nd, "ref")) for nd in element.findall("nd")), _tags(element)
    if len(node_ids) < 2:
        raise ValueError(f"{action} way/{way_id}: a way has at least 2 nodes")
    if not tags.get("note"):
        raise ValueError(f"{action} way/{way_id}: no note saying where the proposal came from")
    return WayChange(way_id, "modify" if action == "modify" else "create", node_ids, tags)


def load_change(path: Path) -> OsmChange:
    """The change at `path` in canonical order, or an empty change where there is no file."""
    if not path.exists():
        return OsmChange((), ())
    root = ET.parse(path).getroot()
    if root.tag != "osmChange":
        raise ValueError(f"{path}: root is <{root.tag}>, not <osmChange>")
    nodes: list[NewNode] = []
    ways: list[WayChange] = []
    for section in root:
        if section.tag not in _ACTIONS:
            raise ValueError(f"{path}: unsupported <{section.tag}>")
        for element in section:
            if element.tag == "node" and section.tag in ("create", "modify"):
                if (_id(element) < 0) != (section.tag == "create"):
                    raise ValueError(f"{section.tag} node/{_id(element)}: a created node has a negative "
                                     f"id, a moved one its existing positive id")
                nodes.append(NewNode(_id(element), float(_attr(element, "lon")),
                                     float(_attr(element, "lat")), _tags(element)))
            elif element.tag == "way":
                ways.append(_way(element, section.tag))
            else:
                raise ValueError(f"{path}: unsupported <{element.tag}> in <{section.tag}>")
    return _canonical(OsmChange(tuple(nodes), tuple(ways)))


def _tag_children(parent: ET.Element, tags: dict[str, str]) -> None:
    for key in sorted(tags):
        ET.SubElement(parent, "tag", k=key, v=tags[key])


def write_change(change: OsmChange, path: Path) -> None:
    """`change` as an osmChange file, in canonical order."""
    change = _canonical(change)
    root = ET.Element("osmChange", version="0.6", generator="nj-road-sketches")
    for action in _ACTIONS:
        nodes = [node for node in change.nodes
                 if (action == "create" and node.id < 0) or (action == "modify" and node.id > 0)]
        ways = [way for way in change.ways if way.action == action]
        if not nodes and not ways:
            continue
        section = ET.SubElement(root, action)
        for node in nodes:
            _tag_children(ET.SubElement(section, "node", id=str(node.id), lat=f"{node.lat:.7f}",
                                        lon=f"{node.lon:.7f}"), node.tags)
        for way in ways:
            element = ET.SubElement(section, "way", id=str(way.id))
            for node_id in way.node_ids:
                ET.SubElement(element, "nd", ref=str(node_id))
            _tag_children(element, way.tags)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def apply_change(layers: dict[str, list[dict]], change: OsmChange) -> dict[str, list[dict]]:
    """`layers` (osm_layers' shape) with `change` applied. Pure: the input is never mutated."""
    from src.sources.osm_context import WAY_LAYERS, height_from_tags

    out = {name: list(items) for name, items in layers.items()}
    # A way clipped at the snapshot's edge has fewer coords than node ids, so it places nothing.
    coords = {node_id: list(coord) for items in out.values() for way in items
              if isinstance(way, dict) and way.get("node_ids")
              and len(way["node_ids"]) == len(way.get("coords_wgs84") or ())
              for node_id, coord in zip(way["node_ids"], way["coords_wgs84"], strict=True)}
    coords |= {node.id: [node.lon, node.lat] for node in change.nodes}
    # A moved node moves everything that uses it - every way through it the change does not itself
    # rewrite, and the node itself where a layer holds it (a kerb, a signal) - so nothing joined to
    # a re-centred street is left behind at the old spot.
    from src.sources.osm_context import NODE_LAYERS

    def route_node(node: NewNode) -> None:
        """A node feature into each node layer its tags belong in, as osm_layers sorts OSM's."""
        entry = {"lon": node.lon, "lat": node.lat, "tags": dict(node.tags), "id": node.id}
        for name, predicate in NODE_LAYERS:
            if predicate(node.tags):
                mixed = any(isinstance(i, dict) and "coords_wgs84" in i for i in out.get(name, []))
                out.setdefault(name, []).append({"coords_wgs84": None, **entry} if mixed else entry)

    moved = {node.id: node for node in change.nodes if node.id > 0}
    if moved:

        def placed(item):
            if not isinstance(item, dict):
                return item
            ids = item.get("node_ids") or []
            if ids and len(ids) == len(item.get("coords_wgs84") or ()) and moved.keys() & set(ids):
                return {**item, "coords_wgs84": [coords[node_id] for node_id in ids]}
            return item

        def a_node(item) -> bool:
            return isinstance(item, dict) and "lon" in item and not item.get("coords_wgs84")

        # A modified node is its whole new self, as a modified way is: dropped from every node
        # layer, then routed by its new tags - a crossing node given `crossing:kerb_extension`, or
        # a node that only now becomes `highway=crossing`, lands where OSM's own would.
        out = {name: [placed(item) for item in items if not (a_node(item) and item["id"] in moved)]
               for name, items in out.items()}
        for node in moved.values():
            route_node(node)
    # A created node with tags is a feature in its own right - a bollard, say - not only a vertex.
    for node in change.nodes:
        if node.id < 0 and node.tags:
            route_node(node)

    for change_way in _canonical(change).ways:
        if change_way.action != "create":
            # A modify is the way's whole new self, as in osmChange: drop the old one everywhere
            # and route the new one, since its new tags may sort it into other layers.
            held = sum(isinstance(way, dict) and way.get("id") == change_way.id
                       for items in out.values() for way in items)
            if not held:
                raise ValueError(f"{change_way.action} way/{change_way.id}: not in this area")
            out = {name: [way for way in items
                          if not isinstance(way, dict) or way.get("id") != change_way.id]
                   for name, items in out.items()}
        if change_way.action == "delete":
            continue
        missing = [node_id for node_id in change_way.node_ids if node_id not in coords]
        if missing:
            raise ValueError(f"{change_way.action} way/{change_way.id}: nodes {missing} are not "
                             f"in this area or the change")
        way = {"coords_wgs84": [coords[node_id] for node_id in change_way.node_ids],
               "tags": dict(change_way.tags), "id": change_way.id,
               "node_ids": list(change_way.node_ids)}
        for name, predicate, min_coords in WAY_LAYERS:
            if predicate(change_way.tags) and len(change_way.node_ids) >= min_coords:
                entry = dict(way)
                if name == "buildings":
                    recorded = height_from_tags(change_way.tags)
                    entry["height_m"], entry["height_source"] = recorded if recorded else (None, None)
                out.setdefault(name, []).append(entry)
    return out
