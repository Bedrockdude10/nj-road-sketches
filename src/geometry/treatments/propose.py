"""A design, written as the osmChange that would make OSM say it.

The route ladders (BROAD_ST_TWO_WAY_BIKEWAY) DECIDE a section per approach; this states each
decision in OSM's vocabulary - cycleway and parking tags on the ways, the way split where a kerb's
parking changes, a road_marking=restriction area for each hatched kerb - built from the design's
own numbers and never from paint. From then on the file is the proposal and the drawing reads it
like any other OSM (src/sources/osm_change.py).
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from src.geometry.model import Leg
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.state import DesignState
    from src.sources.osm_change import OsmChange


def to_the_inch(depth_ft: float) -> int:
    """A stall depth in whole inches, rounded DOWN so the stall is never deeper than sized."""
    raise NotImplementedError("Phase 1 limb B")


def leg_span(leg: Leg, aligned: bool, node_ids: Sequence[int]) -> tuple[int, int] | None:
    """(first, last) index into the way's node list that this leg covers."""
    raise NotImplementedError("Phase 1 limb B")


def way_pieces(node_ids: Sequence[int], spans: Sequence[tuple[int, int, Hashable]]
               ) -> list[tuple[list[int], Hashable]]:
    """The way cut wherever the answer changes from one leg to the next."""
    raise NotImplementedError("Phase 1 limb B")


def hatched_zone_ft(state: DesignState, leg_name: str, side: str,
                    zone: LaneNarrowing) -> Polygon | None:
    """The hatched kerbside zone `zone` states on this kerb, between the junction mouths."""
    raise NotImplementedError("Phase 1 limb B")


def proposal_from_design(model, state: DesignState, osm: dict[str, list[dict]],
                         note: str) -> OsmChange:
    """The osmChange that makes OSM carry `state`'s bikeway, far-kerb parking and hatching."""
    raise NotImplementedError("Phase 1 limb B")
