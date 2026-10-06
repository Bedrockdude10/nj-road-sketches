"""How much of one kerb's width the design paints - the one answer every lane check reads."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.treatments.lanes import LaneNarrowing
from src.geometry.treatments.parking import MarkedParking
from src.geometry.treatments.road_markings import restriction_painted_ft

if TYPE_CHECKING:
    from src.geometry.treatments.state import DesignState


def kerbside_paint_ft(state: DesignState, leg_name: str, side: str) -> float:
    """The kerbside width painted on one leg side, in the NOMINAL frame travel_lane_width_ft
    subtracts in. Parking, else lane narrowing, else road_marking=restriction areas."""
    parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
    if parking is not None:
        return parking.depth_ft + parking.curb_offset_ft
    narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
    if narrowing is not None and Side(side) in narrowing.sides:
        return narrowing.stripe_width_ft
    return restriction_painted_ft(state, leg_name, side)
