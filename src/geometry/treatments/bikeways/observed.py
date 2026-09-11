"""The bike lane that is ALREADY PAINTED, read off OSM.

Every other applier in this package proposes something. This one records: `cycleway:left` and
`cycleway:right` say a lane is on the ground today, and until this existed nothing in `src/`
read either key - the only mention of "cycleway" anywhere was in the list of tags that are NOT a
carriageway. So NJ 35 NB, which has carried `cycleway:right=lane` since the way was drawn, was
rendered as a street with no bikeway on it, and every proposal for that junction was drawn and
described as ADDING one. It is a relocation of an existing treatment, and a drawing that cannot
say so is making a claim about the street that is false in the direction that flatters it.

Beside the parking appliers in spirit and in this package by file, for the reason
apply_observed_parking gives: turning an observation into treatments is the same job whichever
source the observation came from, and a site's scenarios.py is the wrong home for it because
EVERY scenario of a junction has to draw the same street - including the one the pipeline labels
"Existing Conditions" and builds without asking a site anything.
"""
from src.geometry.targets import LegSide
from src.geometry.intersection.junction import CYCLEWAY_IS_A_MARKED_LANE
from src.geometry.treatments.base import (ANGLED_STALL_WIDTH_FT, observed_bay,
                                          traffic_runs_outward)
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.place import AddBikeLane
from src.geometry.treatments.bikeways.sections import AASHTO_MIN_BIKE_LANE_FT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.geometry.intersection.junction import IntersectionModel


#: How wide to draw a lane OSM says is there but does not measure - `cycleway:<side>:width` is
#: absent on every way this project touches.
#:
#: AASHTO's DESIGN width, not the 4 ft floor, because this is the case AASHTO asks 5 ft for: a
#: lane running against a kerb and gutter or a parking lane, where the gutter pan is not ridable.
#: An existing lane is overwhelmingly likelier to have been striped to the design figure than to
#: the floor, and the floor would understate the street.
#:
#: AN ASSUMPTION AND REPORTED AS ONE - see the note apply_osm_bike_lanes leaves on the design,
#: which names the tag that would replace it. It is also the one figure here a mapper can settle
#: from an aerial in about a minute, which is why the note names the tag rather than asking for a
#: survey.
ASSUMED_BIKE_LANE_FT = AASHTO_MIN_BIKE_LANE_FT


def _lane_spans(model: "IntersectionModel", leg_name: str, side: str) -> list[tuple]:
    """[(start_ft, end_ft, width_ft|None)] where OSM records a MARKED LANE on this kerb."""
    return [(start_ft, end_ft, widths.get(side))
            for start_ft, end_ft, values, widths, _way in model.cycleway_spans(leg_name)
            if values.get(side) in CYCLEWAY_IS_A_MARKED_LANE]


def _reach_ft(spans: list[tuple], leg_name: str, model: "IntersectionModel") -> float | None:
    """How far out the lane runs, or None for "as far as the kerb is traced".

    A LANE MAPPED OVER PART OF A LEG IS A LANE OVER PART OF A LEG, the same care
    kerb_may_hold_parking takes with a restriction over part of a kerb (.claude/SKILLS.md 7):
    OSM splits a way where a tag changes, so a bikeway that ends mid-block is recorded as two
    spans and reading either one as the whole leg is a false statement in one direction or the
    other. None only where the tagged spans cover every span the leg has.
    """
    tagged = sorted((start_ft, end_ft) for start_ft, end_ft, _w in spans)
    covered_to_ft = 0.0
    for start_ft, end_ft in tagged:
        if start_ft > covered_to_ft + 1e-6:
            break                       # a gap: the lane stops at the last contiguous end
        covered_to_ft = max(covered_to_ft, end_ft)
    leg_end_ft = max((end_ft for _s, end_ft, _v, _w, _i in model.cycleway_spans(leg_name)),
                     default=0.0)
    return None if covered_to_ft >= leg_end_ft - 1e-6 else covered_to_ft


def apply_osm_bike_lanes(state: DesignState, model: "IntersectionModel") -> DesignState:
    """Mark the bike lane OSM says each kerb already carries.

    THE BAY GOES IN THE SAME SECTION. Where the kerb also carries observed parking, the stalls
    are `parking_ft` on this treatment rather than a MarkedParking beside it: one rigid section
    places all of a rigid cross-section (.claude/SKILLS.md 0a), and two treatments on one
    leg-side paint over each other. apply_observed_parking already skips a leg-side an AddBikeLane
    has claimed, so the order is this first and the parking after - which is the order
    existing_conditions and every builder here use.

    PINNED TO ITS OWN KERB, because a lane that is already painted fits by construction and the
    question is only where the traffic ends up. Unpinned, the section is measured from a target
    lane width off the alignment and NJ 35 NB's east kerb - a 20.09 ft bay plus a 5 ft lane -
    comes out 4.07 ft too wide for its own half of the street and is REFUSED, which would report
    a lane that exists as one that does not fit.

    "Unless otherwise specified", like both parking appliers: a leg-side that already carries a
    bikeway is left alone, so a proposal may relocate or rebuild this lane by applying its own
    first without having to say which kerb OSM put one on.
    """
    for leg_name, leg in state.legs.items():
        bay = observed_bay(model, leg_name)
        for side in ("left", "right"):
            spans = _lane_spans(model, leg_name, side)
            if not spans:
                continue        # absent OR "no" - and `no` is a positive statement, not a lane
            if any(t.target == LegSide(leg_name, side) for t in state.treatments_of(AddBikeLane)):
                continue
            mapped_ft = [w for _s, _e, w in spans if w is not None]
            width_ft = min(mapped_ft) if mapped_ft else ASSUMED_BIKE_LANE_FT
            state = state.apply(AddBikeLane(
                LegSide(leg_name, side),
                width_ft=width_ft,
                # NO BUFFER, and that is what a conventional bike lane IS - a single line against
                # the travel lane (BikeLane.__post_init__ says so where it refuses a buffer too
                # narrow to hold its own two lines). OSM has a `cycleway:<side>:buffer` key for
                # the buffered kind and no way here carries it.
                buffer_ft=0.0,
                parking_ft=bay.depth_ft if bay and side in bay.sides else 0.0,
                parking_angle_deg=bay.angle_deg if bay and side in bay.sides else None,
                parking_stall_width_ft=(bay.stall_width_ft if bay and side in bay.sides
                                        else ANGLED_STALL_WIDTH_FT),
                to_ft=_reach_ft(spans, leg_name, model),
                pin_to_kerb=True,
                runs_outward=traffic_runs_outward(state, leg, side),
                observed=True))
            if not mapped_ft:
                state.notes.append(
                    f"apply_osm_bike_lanes({leg_name}, {side}): OSM records a bike lane on this "
                    f"kerb (cycleway:{side}=lane) but no width, so it is DRAWN AT "
                    f"{ASSUMED_BIKE_LANE_FT:.0f} ft - AASHTO's design width, assumed. Tag "
                    f"cycleway:{side}:width to replace the assumption with a measurement.")
    return state
