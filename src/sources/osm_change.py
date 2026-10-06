"""A proposal is an osmChange: what the street would carry, as the edits OSM itself would take.

proposals/<area>/<scenario>.osc is a real osmChange file (OSM API 0.6) - JOSM reads it - so a
proposal is stated in OSM's own terms and nothing else: tags on a way, a way split where its
tags change (wiki Street parking: "The roadway needs to be split up where any of the properties
changes"), a new way such as a road_marking=restriction area. `apply_change` applies one to the
layers `osm_layers` returns, routing every created or modified way through the same predicates
`osm_layers` sorts OSM by, so a proposed scenario is drawn by exactly the code that draws the
street as OSM records it. Nothing downstream knows an element came from a proposal.

Supported: <create> node and way, <modify> way, <delete> way. Every created or modified way
carries a `note` (OSM's Key:note) saying where it came from - a proposal with no provenance is
refused, as an observation with no `source` is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PROPOSALS_DIR = Path(__file__).resolve().parents[2] / "proposals"


@dataclass(frozen=True)
class NewNode:
    """A node the change creates: negative id, and tags only where the node is a feature."""
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


def load_change(path: Path) -> OsmChange:
    """The change at `path` in canonical order, or an empty change where there is no file."""
    raise NotImplementedError("Phase 1 limb A")


def write_change(change: OsmChange, path: Path) -> None:
    """`change` as an osmChange file, in canonical order."""
    raise NotImplementedError("Phase 1 limb A")


def apply_change(layers: dict[str, list[dict]], change: OsmChange) -> dict[str, list[dict]]:
    """`layers` (osm_layers' shape) with `change` applied. Pure: the input is never mutated."""
    raise NotImplementedError("Phase 1 limb A")
