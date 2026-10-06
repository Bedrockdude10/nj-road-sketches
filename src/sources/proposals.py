"""A proposal is OSM: tags on the street's own OSM ways, and the new elements it would add.

proposals/<area>/<scenario>.yaml has the observations schema (src/sources/observations.py) - an
element, the tags, and a required `source` - plus one thing an observation cannot say: a NEW
WAY. OSM's own convention for an element not yet uploaded is a negative id (JOSM, osmChange),
so `way/-1` with `coords` ([lon, lat] per node, closed) is a way the proposal adds. The merged
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
from src.sources.osm_context import is_restriction_marking

PROPOSALS_DIR = Path(__file__).resolve().parents[2] / "proposals"

_EXISTING_WAY = re.compile(r"^way/(\d+)$")
_NEW_WAY = re.compile(r"^way/(-\d+)$")

#: {osm_layers name: predicate} for the new ways a proposal may add.
NEW_WAY_LAYERS = {"road_markings": is_restriction_marking}


class ProposalEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    element: str
    tags: dict[str, str]
    source: Sourced
    #: [lon, lat] per node, first == last. Only on a new way.
    coords: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def _shape(self) -> ProposalEntry:
        if not self.tags:
            raise ValueError("a proposal entry with no tags proposes nothing")
        if _EXISTING_WAY.match(self.element):
            if self.coords is not None:
                raise ValueError(f"{self.element} already exists in OSM; only a new way "
                                 f"(negative id) carries coords")
        elif _NEW_WAY.match(self.element):
            if not self.coords or len(self.coords) < 4 or self.coords[0] != self.coords[-1]:
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
    """`osm` (osm_layers' shape) with each proposed way's tags merged over OSM's own, and each
    new way added to the layer its tags put it in.

    A new dict and new entries: the layers handed in are shared with every other scenario. An
    entry naming a way the area does not carry is an error - a proposal for a street that is not
    in the drawing would otherwise vanish without a word - and so is a new way no layer reads.
    """
    if not entries:
        return osm
    changes: dict[int, dict] = {}
    added: dict[str, list[dict]] = {}
    for entry in entries:
        if not entry.is_new:
            changes.setdefault(entry.way_id, {}).update(entry.tags)
            continue
        layer = next((name for name, predicate in NEW_WAY_LAYERS.items()
                      if predicate(entry.tags)), None)
        if layer is None:
            raise ValueError(f"proposal adds {entry.element} tagged {entry.tags}, which no layer "
                             f"reads ({', '.join(NEW_WAY_LAYERS)}) - it would not be drawn")
        added.setdefault(layer, []).append({"coords_wgs84": [list(c) for c in entry.coords],
                                            "tags": dict(entry.tags), "id": entry.way_id,
                                            "node_ids": []})
    roads = [{**road, "tags": {**(road.get("tags") or {}), **changes.pop(road["id"])}}
             if road.get("id") in changes else road
             for road in osm.get("roads") or []]
    if changes:
        raise ValueError(f"proposal names way(s) the area does not carry: {sorted(changes)}")
    return {**osm, "roads": roads,
            **{layer: [*(osm.get(layer) or []), *items] for layer, items in added.items()}}
