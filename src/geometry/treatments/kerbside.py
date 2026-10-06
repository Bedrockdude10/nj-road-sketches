"""How much of one kerb's width the design paints - the one answer every lane check reads."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.geometry.treatments.state import DesignState


def kerbside_paint_ft(state: DesignState, leg_name: str, side: str) -> float:
    """The kerbside width painted on one leg side, in the NOMINAL frame travel_lane_width_ft
    subtracts in. Parking, else lane narrowing, else road_marking=restriction areas."""
    raise NotImplementedError("Phase 1 limb C")
