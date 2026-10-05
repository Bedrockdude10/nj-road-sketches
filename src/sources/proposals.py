"""A proposal is OSM tags: what the street would carry, written onto its own OSM ways.

proposals/<area>/<scenario>.yaml has the same schema as observations/<area>.yaml - an element,
the tags, and a required `source` - and is merged into the road ways before the model is built,
so a proposed scenario is drawn by exactly the code that draws the street as OSM records it.
Nothing downstream knows a tag came from a proposal: it reads the tag.
"""
from __future__ import annotations

from pathlib import Path

from src.sources.observations import ObservationEntry, load_observations

PROPOSALS_DIR = Path(__file__).resolve().parents[2] / "proposals"


def proposal_path(area: str, scenario: str) -> Path:
    return PROPOSALS_DIR / area / f"{scenario}.yaml"


def load_proposal(area: str, scenario: str) -> list[ObservationEntry]:
    """Every tag change `scenario` proposes in `area`, or [] where it proposes none."""
    return load_observations(area, path=proposal_path(area, scenario))


def with_proposal(osm: dict, entries: list[ObservationEntry]) -> dict:
    """`osm` (osm_layers' shape) with each proposed way's tags merged over OSM's own.

    A new dict and new road entries: the layers handed in are shared with every other scenario.
    An entry naming a way the area does not carry is an error - a proposal for a street that is
    not in the drawing would otherwise vanish without a word.
    """
    if not entries:
        return osm
    changes: dict[int, dict] = {}
    for entry in entries:
        kind, _, ident = entry.element.partition("/")
        if kind != "way" or not ident.lstrip("-").isdigit():
            raise ValueError(f"proposal element {entry.element!r}: only road ways are proposed on")
        changes.setdefault(int(ident), {}).update(entry.tags)
    roads = [{**road, "tags": {**(road.get("tags") or {}), **changes.pop(road["id"])}}
             if road.get("id") in changes else road
             for road in osm.get("roads") or []]
    if changes:
        raise ValueError(f"proposal names way(s) the area does not carry: {sorted(changes)}")
    return {**osm, "roads": roads}
