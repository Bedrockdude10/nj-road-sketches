"""Field observations, expressed as OSM tags added to OSM elements.

Some facts about the street are things nobody has mapped yet - a crossing that is actually
painted but untagged, a signal mast on a corner - and they used to live in per-site YAML that
only the site-model path could read: princeton_eprospect has 8 OSM crossings, ALL untagged,
while its config marks 3 legs as having painted crosswalks; columbia_princeton is 8 of 10
untagged against 4 config-marked legs. A site rendered correctly and the slice of the same
junction (src/geometry/network/area.py) rendered wrong, purely because the observation lived
somewhere the network path never reads.

AN OBSERVATION IS AN OSM TAG ON AN OSM ELEMENT. `crossing:markings` on a crossing way is
already read by `resolve_crosswalk_style`, `SurveyedCrossing.is_marked` and
`OSM_MARKINGS_TO_STYLE` (src/render/crosswalks.py, src/geometry/surveyed.py) - express the
observation in OSM's own vocabulary and every existing reader picks it up for free. This
module adds no reader: it only lets observations/<area>.yaml add tags to the snapshot
`area_context` assembles, at the one merge point in `area_context` itself, so 2D and 3D see
the same document by construction rather than by two call sites agreeing to stay in step.

ADDITIVE ONLY (rule 1). An observation may add a key an element does not already carry. The
same key with the same value is accepted but redundant, and reported so the file can be
pruned (rule 2). The same key with a DIFFERENT value is refused (rule 3): OSM is the
authoritative source here, and a silent override would make the merged document untrustworthy
in a way no test could later catch. `source` is required on every entry (rule 4) - an
observation with no provenance is not an observation. A synthetic element (one OSM does not
have at all) is tagged `provenance=ELEMENT_FROM_OBSERVATION` so it stays distinguishable from
a real OSM element once it reaches a row in the document (rule 5).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, model_validator

from src.config import load_config

OBSERVATIONS_DIR = Path(__file__).resolve().parents[2] / "observations"

#: Where a tag in the merged document came from - carried through to the row `area_context`
#: emits (rule 5). Absent on an element `apply_observations` never touched, which the read
#: site treats as `ELEMENT_FROM_OSM` via `.get(..., ELEMENT_FROM_OSM)`.
ELEMENT_FROM_OSM = "osm"
ELEMENT_FROM_OBSERVATION = "field_observation"

_EXISTING_REF = re.compile(r"^(node|way|relation)/(\d+)$")
#: The only synthetic kind this module builds - see the pinned format above. A bare "way" or
#: "relation" raises rather than half-working, since nothing here can lay out their geometry.
_SYNTHETIC_KIND = "node"

Sourced = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Strict(BaseModel):
    """extra="forbid" so a mistyped key is a validation error, not a silently ignored field -
    the same reasoning as src/site_schema.py's own Strict."""
    model_config = ConfigDict(extra="forbid")


class ObservationEntry(_Strict):
    """One entry of observations/<area>.yaml, validated but not yet merged into a snapshot."""
    element: str
    at: tuple[float, float] | None = None
    tags: dict[str, str]
    source: Sourced

    @model_validator(mode="after")
    def _shape(self) -> ObservationEntry:
        ref = _EXISTING_REF.match(self.element)
        if ref is not None:
            if self.at is not None:
                raise ValueError(
                    f"element {self.element!r} names an OSM element that already exists; "
                    f"'at' only applies to a synthetic element OSM does not have")
        elif self.element == _SYNTHETIC_KIND:
            if self.at is None:
                raise ValueError(
                    f"element: {_SYNTHETIC_KIND!r} is synthetic and needs 'at: [lon, lat]' "
                    f"to say where it is")
        else:
            raise ValueError(
                f"element {self.element!r} is neither 'kind/id' of an existing OSM element "
                f"(e.g. 'way/11647647') nor the one synthetic kind this module implements "
                f"({_SYNTHETIC_KIND!r})")
        if not self.tags:
            raise ValueError("an observation with no tags adds nothing")
        return self

    @property
    def kind(self) -> str:
        ref = _EXISTING_REF.match(self.element)
        return ref.group(1) if ref else self.element

    @property
    def element_id(self) -> int | None:
        ref = _EXISTING_REF.match(self.element)
        return int(ref.group(2)) if ref else None

    @property
    def is_synthetic(self) -> bool:
        return self.element_id is None


class _ObservationsFile(_Strict):
    observations: list[ObservationEntry] = []


class ObservationError(ValueError):
    """An observations file that doesn't parse, an entry with no source, an entry naming an
    element that isn't in the snapshot, or a tag whose value disagrees with OSM's own."""


def load_observations(area: str, path: Path | None = None) -> list[ObservationEntry]:
    """Every observation for `area`, or [] if observations/<area>.yaml doesn't exist or is
    empty - most areas have no field observations yet, and an absent file is not a malformed
    one."""
    file_path = path if path is not None else OBSERVATIONS_DIR / f"{area}.yaml"
    if not file_path.exists():
        return []
    try:
        return _ObservationsFile.model_validate(load_config(file_path) or {}).observations
    except ValidationError as e:
        lines = [f"  {'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
                 for err in e.errors()]
        raise ObservationError(
            f"{len(lines)} problem(s) in {file_path}:\n" + "\n".join(lines)) from e


def _merge_tags(existing: dict, added: dict, ref: str, source: str) -> dict:
    """`existing` with `added` merged in additively (rule 1). Same key/same value is accepted
    and reported as redundant (rule 2); same key/different value raises (rule 3) - the single
    most important behaviour here, because it is what keeps OSM authoritative rather than
    silently overridable.
    """
    merged = dict(existing)
    for key, value in added.items():
        prior = existing.get(key)
        if prior is None:
            merged[key] = value
        elif prior == value:
            print(f"  observations: {ref} already has {key}={value!r} - this observation is "
                  f"redundant and can be pruned ({source})")
        else:
            raise ObservationError(
                f"observation for {ref} sets {key}={value!r}, but OSM already has "
                f"{key}={prior!r} there. OSM is authoritative and may not be silently "
                f"overridden ({source}).")
    return merged


def apply_observations(snapshot: dict, observations: list[ObservationEntry]) -> dict:
    """`snapshot`, with every observation's tags merged in additively and every synthetic
    element added.

    Returns a NEW snapshot - `nodes`, `ways` and `relations` are shallow-copied before
    anything already in them is touched, so the snapshot `fetch_borough_osm` cached
    (src/sources/osm_context.py's `_MEMO`) is never mutated in place; a second area sharing the
    same cached snapshot must not see the first area's observations. With no observations this
    returns `snapshot` itself unchanged, so an empty observations/<area>.yaml makes this a
    no-op and the document byte-identical to one built before this module existed.
    """
    if not observations:
        return snapshot
    nodes = dict(snapshot["nodes"])
    ways = list(snapshot["ways"])
    relations = list(snapshot.get("relations", []))
    ways_by_id = {way["id"]: i for i, way in enumerate(ways)}
    relations_by_id = {rel["id"]: i for i, rel in enumerate(relations)}
    next_synthetic_id = -1

    for obs in observations:
        if obs.is_synthetic:
            lon, lat = obs.at
            nodes[next_synthetic_id] = {"id": next_synthetic_id, "lon": lon, "lat": lat,
                                        "tags": dict(obs.tags),
                                        "provenance": ELEMENT_FROM_OBSERVATION}
            next_synthetic_id -= 1
            continue

        ref = obs.element
        if obs.kind == "node":
            if obs.element_id not in nodes:
                raise ObservationError(f"observation references {ref}, which is not in this "
                                       f"snapshot ({obs.source})")
            node = nodes[obs.element_id]
            nodes[obs.element_id] = {
                **node, "tags": _merge_tags(node.get("tags") or {}, obs.tags, ref, obs.source)}
        elif obs.kind == "way":
            idx = ways_by_id.get(obs.element_id)
            if idx is None:
                raise ObservationError(f"observation references {ref}, which is not in this "
                                       f"snapshot ({obs.source})")
            way = ways[idx]
            ways[idx] = {
                **way, "tags": _merge_tags(way.get("tags") or {}, obs.tags, ref, obs.source)}
        else:  # relation
            idx = relations_by_id.get(obs.element_id)
            if idx is None:
                raise ObservationError(f"observation references {ref}, which is not in this "
                                       f"snapshot ({obs.source})")
            relation = relations[idx]
            relations[idx] = {
                **relation,
                "tags": _merge_tags(relation.get("tags") or {}, obs.tags, ref, obs.source)}

    return {**snapshot, "nodes": nodes, "ways": ways, "relations": relations}
