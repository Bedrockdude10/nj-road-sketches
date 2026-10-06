"""A proposal is OSM: tags on the street's own OSM ways, and the new elements it would add.

proposals/<area>/<scenario>.yaml has the observations schema (src/sources/observations.py) - an
element, the tags, and a required `source` - plus the two things an observation cannot say, in
osmChange's own terms:

  * A NEW WAY. OSM's convention for an element not yet uploaded is a negative id (JOSM,
    osmChange). `way/-1` with `coords` ([lon, lat] per node, closed) is a new area; `way/-1`
    with `nodes` (existing OSM node ids) is a new way along nodes OSM already has, and it carries
    its full tag set, as a created way does.
  * A SPLIT. Wiki Street parking: "The roadway needs to be split up where any of the properties
    changes". In osmChange a split is the existing way restated with its shortened `nodes` plus
    a new way over the rest, sharing the node it was split at.

The merged
layers are handed to the model before it is built, so a proposed scenario is drawn by exactly
the code that draws the street as OSM records it. Nothing downstream knows a tag or a way came
from a proposal: it reads OSM.

New ways are limited to what a reader exists for - road_marking=restriction areas today - so a
proposed element nothing would draw is refused rather than silently dropped.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from src.config import load_config
from src.sources.observations import Sourced
from src.sources.osm_context import is_restriction_marking, is_road

PROPOSALS_DIR = Path(__file__).resolve().parents[2] / "proposals"

_EXISTING_WAY = re.compile(r"^way/(\d+)$")
_NEW_WAY = re.compile(r"^way/(-\d+)$")

#: {osm_layers name: predicate} for the new AREAS (`coords`) a proposal may add.
NEW_WAY_LAYERS = {"road_markings": is_restriction_marking}


class ProposalEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    element: str
    source: Sourced
    tags: dict[str, str] = {}
    #: [lon, lat] per node, first == last. Only on a new area.
    coords: list[tuple[float, float]] | None = None
    #: OSM node ids, in order: a new way along existing nodes, or an existing way's node list
    #: after a split.
    nodes: list[int] | None = None

    @model_validator(mode="after")
    def _shape(self) -> ProposalEntry:
        if self.nodes is not None and len(self.nodes) < 2:
            raise ValueError(f"{self.element}: a way needs at least 2 nodes")
        if _EXISTING_WAY.match(self.element):
            if self.coords is not None:
                raise ValueError(f"{self.element} already exists in OSM; only a new way "
                                 f"(negative id) carries coords")
            if not self.tags and self.nodes is None:
                raise ValueError(f"{self.element}: no tags and no nodes proposes nothing")
        elif _NEW_WAY.match(self.element):
            if not self.tags:
                raise ValueError(f"{self.element} is a new way and needs its tags")
            if (self.coords is None) == (self.nodes is None):
                raise ValueError(f"{self.element} is a new way: give it `coords` (a new area) "
                                 f"or `nodes` (a way along existing nodes), not both or neither")
            if self.coords is not None and (len(self.coords) < 4
                                            or self.coords[0] != self.coords[-1]):
                raise ValueError(f"{self.element} is a new way and needs closed coords "
                                 f"(at least 4 [lon, lat] points, first == last)")
        else:
            raise ValueError(f"proposal element {self.element!r}: only 'way/<id>' (an OSM way) "
                             f"or 'way/-<n>' (a new one) is proposed on")
        return self

    @property
    def way_id(self) -> int:
        return int(self.element.partition("/")[2])

    @property
    def is_new(self) -> bool:
        return self.way_id < 0


class _ProposalFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[ProposalEntry] = []


def proposal_path(area: str, scenario: str) -> Path:
    return PROPOSALS_DIR / area / f"{scenario}.yaml"


def load_proposal(area: str, scenario: str) -> list[ProposalEntry]:
    """Every change `scenario` proposes in `area`, or [] where it proposes none."""
    path = proposal_path(area, scenario)
    if not path.exists():
        return []
    try:
        return _ProposalFile.model_validate(load_config(path) or {}).observations
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
                 for err in e.errors()]
        raise ValueError(f"{len(lines)} problem(s) in {path}:\n" + "\n".join(lines)) from e


def with_proposal(osm: dict, entries: list[ProposalEntry]) -> dict:
    """`osm` (osm_layers' shape) with each proposed way's tags merged over OSM's own, each split
    way's node list replaced, and each new way added to the layer its tags put it in.

    A new dict and new entries: the layers handed in are shared with every other scenario. An
    entry naming a way the area does not carry is an error - a proposal for a street that is not
    in the drawing would otherwise vanish without a word - and so is a new way no layer reads,
    or one along a node no road in the area has.
    """
    if not entries:
        return osm
    roads_in = osm.get("roads") or []
    node_at = {node_id: coord for road in roads_in
               if len(road.get("node_ids") or []) == len(road.get("coords_wgs84") or [])
               for node_id, coord in zip(road.get("node_ids") or [], road.get("coords_wgs84") or [])}

    def along(entry: ProposalEntry) -> list:
        missing = [n for n in entry.nodes if n not in node_at]
        if missing:
            raise ValueError(f"proposal {entry.element} runs along node(s) no road in the area "
                             f"has: {missing}")
        return [list(node_at[n]) for n in entry.nodes]

    changes: dict[int, dict] = {}
    renoded: dict[int, ProposalEntry] = {}
    added: dict[str, list[dict]] = {}
    for entry in entries:
        if not entry.is_new:
            changes.setdefault(entry.way_id, {}).update(entry.tags)
            if entry.nodes is not None:
                renoded[entry.way_id] = entry
            continue
        if entry.nodes is not None:
            if not is_road(entry.tags):
                raise ValueError(f"proposal adds {entry.element} along existing nodes, tagged "
                                 f"{entry.tags} - only a road way is proposed that way")
            added.setdefault("roads", []).append({"coords_wgs84": along(entry),
                                                  "tags": dict(entry.tags), "id": entry.way_id,
                                                  "node_ids": list(entry.nodes)})
            continue
        layer = next((name for name, predicate in NEW_WAY_LAYERS.items()
                      if predicate(entry.tags)), None)
        if layer is None:
            raise ValueError(f"proposal adds {entry.element} tagged {entry.tags}, which no layer "
                             f"reads ({', '.join(NEW_WAY_LAYERS)}) - it would not be drawn")
        added.setdefault(layer, []).append({"coords_wgs84": [list(c) for c in entry.coords],
                                            "tags": dict(entry.tags), "id": entry.way_id,
                                            "node_ids": []})
    roads = []
    for road_in in roads_in:
        way_id, road = road_in.get("id"), road_in
        if way_id in changes:
            road = {**road, "tags": {**(road.get("tags") or {}), **changes.pop(way_id)}}
        if way_id in renoded:
            entry = renoded.pop(way_id)
            road = {**road, "node_ids": list(entry.nodes), "coords_wgs84": along(entry)}
        roads.append(road)
    if changes:
        raise ValueError(f"proposal names way(s) the area does not carry: {sorted(changes)}")
    return {**osm, "roads": [*roads, *added.pop("roads", [])],
            **{layer: [*(osm.get(layer) or []), *items] for layer, items in added.items()}}
