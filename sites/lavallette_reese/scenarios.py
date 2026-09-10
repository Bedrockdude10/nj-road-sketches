"""Treatment proposals for NJ 35 NB (Grand Central Ave) & Reese Ave, Lavallette Borough.

THE STREET THIS DRAWS ON IS ONE CARRIAGEWAY OF A ONE-WAY PAIR, and that is the fact every
proposal here turns on. Grand Central Ave carries NJ 35 NORTHBOUND ONLY (OSM `oneway=yes`,
`lanes=2`, NJDOT "South to North"); the southbound carriageway is Anna O Hankins Blvd, 464.6 ft
west, and is a different junction and a different site. Three consequences:

  * its two travel lanes run the SAME way, so there is no centre stripe to hold - what divides
    them is a broken WHITE lane line (`centerline_style: single_white_dashed`), and a yellow one
    would tell a driver the next lane runs at them;
  * "the right-hand side" means the EAST kerb on both legs, because northbound traffic runs
    outward along grand_central_ave_north and INWARD along grand_central_ave_south - which is
    why the side is asked for with side_facing(leg, "east") and never written down per leg;
  * the street is enormously wide for what it carries: 69.55-69.91 ft between traced kerbs.

WHAT THE STREET SPENDS THAT WIDTH ON TODAY IS ANGLED PARKING, and every proposal here keeps it.
60 degrees against BOTH kerbs, observed in the field and declared in config.yaml because OSM
carries no parking tag on this street at all (see legs.*.existing_parking and
apply_observed_parking). At 60 degrees a 9x18 ft stall is a 20.09 ft bay taking 10.39 ft of kerb,
so:

    two bays          40.18 ft        the parking, unchanged in every scenario
    two travel lanes  29.37 ft        14.69 ft each TODAY - the whole surplus
                      -------
                      69.55 ft

THE BIKE LANE COMES OUT OF THE TRAVEL LANES, NOT OUT OF THE PARKING. That is the entire
arithmetic of these proposals and it was got wrong first time round, by charging the bikeway
against the 34.78 ft half of the street its kerb sits in and reporting that it did not fit. It
does: two 14.69 ft lanes hold 7.37 ft more than two 11 ft lanes need, which is a 5 ft lane and a
2 ft buffer. But that surplus is all on ONE side of the alignment, because only one kerb gets a
bike lane, so the section has to be pinned to its own kerb and the travel way allowed to sit
asymmetrically - AddBikeLane.pin_to_kerb, and BikeLane.near_half_ft under it. Nothing here
narrows a stall, and nothing here removes a space.
"""
from src.geometry.targets import LegSide
from src.geometry.treatments import (AddBikeLane, AddBikeLaneBollards,
                                     all_crosswalks_continental, apply_observed_parking,
                                     DesignState, MarkedParking, ProtectDaylightZone,
                                     traffic_runs_outward)
from src.geometry.model import angled_stall_depth_ft, side_facing

# THE DIRECTION NJ 35 RUNS IS NOT DECLARED HERE. It used to be, as a CARRIAGEWAY_RUNS = "north"
# constant this file's two builders each translated per leg, and that is the wrong home for it
# twice over: which way a street runs is a fact about the street rather than a decision a
# proposal makes (.claude/SKILLS.md section 5), and the "Existing Conditions" state the pipeline
# builds without asking a site anything then leaned half the bays the wrong way. It is now
# `legs.<leg>.traffic_heads_toward: north` in config.yaml, read through traffic_runs_outward.

#: The bikeway's own cross-section. BOTH proposals below use the same one, so that the only
#: difference between them is the ordering across the road.
#:
#: 5 ft is AASHTO's design width for an exclusive lane - not the 4 ft floor, and not negotiable
#: here, because on either ordering this lane runs against either a kerb or a parking lane and
#: that is the case AASHTO asks 5 ft for.
#:
#: THE BUFFER IS AT ITS FLOOR AND THE STREET DECIDED THAT, not a preference. What the bikeway
#: has to fit into is the travel lanes' surplus over target width, and that surplus is what the
#: observed parking leaves:
#:
#:     69.57 ft   between the traced kerbs, grand_central_ave_north at its narrowest
#:    -40.18 ft   two 60-degree bays, unchanged
#:     -------
#:     29.39 ft   for traffic today: 14.69 ft per lane
#:    -22.00 ft   two lanes at TARGET_LANE_WIDTH_FT
#:     -------
#:      7.39 ft   the whole budget for a bike lane, its buffer and the stripe between them
#:
#: A 5 ft lane and the 0.82 ft stripe bounding it spend 5.82 of that, leaving 1.57 ft - under
#: min_bike_lane_buffer_ft, which is 1.64 ft because a buffer has to hold the two stripes that
#: bound it. So the section is 0.07 ft over the budget at the narrowest buffer that is a buffer
#: at all, and that 0.07 ft comes off the travel lanes rather than off the parking.
#:
#: THE PROJECT'S STANDARD 2 FT BUFFER DOES NOT FIT, and it is worth recording what it costs
#: rather than leaving the floor looking like a default: at 2 ft the section is 27.91 ft, the
#: travel way shifts 4.06 ft, and the west kerb's lane comes out at 10.78 ft - which
#: TravelLanesKeepTheirWidth fails, fatally and correctly. At the floor the shift is 3.70 ft and
#: both lanes hold 11 ft or better. The buffer is the only piece in the section with anywhere to
#: give: the lane is at AASHTO's design width, the stripe is a stripe, and the bays are what is
#: on the ground.
BIKE_LANE_FT = 5.0


def _buffer_ft() -> float:
    """The narrowest strip that is still a buffer - one wide enough to hold its own two lines.

    Called rather than written as 1.64, so that it tracks LANE_EDGE_LINE_WIDTH_FT: this section
    has 0.07 ft of slack, and a figure copied here would stop being the floor the moment the
    stripe width moved and would fail as a travel lane 0.8 ft too narrow rather than as a buffer.
    """
    from src.geometry.treatments import min_bike_lane_buffer_ft

    return min_bike_lane_buffer_ft()

#: The observed bay, restated here only as the two numbers a treatment needs from it. The ANGLE
#: and the fact that both kerbs carry it are the observation and live in config.yaml; the stall
#: behind it is the project's standard, so the DEPTH is derived and never typed in.
PARKING_ANGLE_DEG = 60.0


def _bay_depth_ft() -> float:
    """The observed bay's depth across the street, from the project's standard angled stall.

    A function, not a constant, so the figure comes from angled_stall_depth_ft every time rather
    than from a number someone rounded once: the second term of that formula is the one that
    gets dropped, and dropping it understates this bay by 4.50 ft - the width of a bike lane, on
    the street where whether a bike lane fits is the question.
    """
    from src.geometry.treatments.base import ANGLED_STALL_LENGTH_FT, ANGLED_STALL_WIDTH_FT

    return angled_stall_depth_ft(ANGLED_STALL_LENGTH_FT, ANGLED_STALL_WIDTH_FT,
                                 PARKING_ANGLE_DEG)


def _daylight_every_parked_kerb(state: DesignState) -> DesignState:
    """Bollards in every daylight zone - the paint-and-post curb extension.

    NOT AddCurbExtension, which moves the kerb itself and needs concrete. The daylight zone is
    the corner ground R.S. 39:4-138 already keeps clear of parked cars; standing posts in it
    makes that legible and stops a car overhanging the crossing, which is what a curb extension
    is FOR, at the cost of paint and posts rather than of drainage and a contract.

    EVERY KERB THAT HOLDS PARKED CARS, WHICHEVER TREATMENT PUT THEM THERE. Asking only for
    MarkedParking missed exactly the kerbs these proposals are about: where the bikeway's own
    section carries the stalls, they belong to AddBikeLane and not to a separate parking
    treatment, so the two kerbs the whole proposal is about were the only ones drawn without a
    curb extension at their corners.
    """
    for treatment in (*state.treatments_of(MarkedParking), *state.treatments_of(AddBikeLane)):
        if isinstance(treatment, AddBikeLane) and not treatment.parking_ft:
            continue
        state = state.apply(ProtectDaylightZone(treatment.target, kind="bollards"))
    return state


# THERE IS NO build_existing_conditions HERE, AND THERE WAS. It applied the observed bays to the
# Phase 2 baseline and changed nothing else, because a DesignState knows the kerbs, the corner
# fillets and the restrictions while parking is a treatment - so every "Existing Conditions"
# panel drew this street as 70 ft of bare asphalt, which is the first thing a reviewer checks
# and a false statement about the street.
#
# It was the wrong fix, and the tell was that it fixed one panel: a SCENARIO is something the
# pipeline renders on request, and the panel labelled "Existing Conditions" is built by the
# pipeline itself, three times over (phase3_treatments, phase4_render_3d,
# phase4_export_geometry). So the sheets went on showing bare asphalt beside a separate,
# identical sheet that showed the parking. `existing_conditions(model)` in
# src/geometry/treatments/parking.py is the one home for it now, it is what those three scripts
# label, and it needs nothing from a site - which is what forced the lean of the bays out of
# this file and into config.yaml.
#
# Reese Ave has no observation recorded either way, and absent is not "no parking": it is
# unrecorded, so nothing is drawn there and this comment is where that is said out loud.


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Crossings brought up to continental over the existing street - the reference the rest
    move from.

    NO complete_centerlines. It adds a double yellow to any leg with no centre stripe today, on
    the reasoning that "nothing marks the middle of the road" - which is true here and is not a
    defect: Grand Central Ave is ONE-WAY, so its two lanes run the same way and a yellow
    centreline between them would be telling a driver there is opposing traffic. Reese Ave is
    two-way but its `centerline_style: none` is INFERRED rather than observed (see config.yaml),
    and inventing paint off an inference is worse than drawing what is there.

    THE LANE LINE BETWEEN GRAND CENTRAL'S TWO NORTHBOUND LANES IS DRAWN, and it is not a centre
    stripe: `single_white_dashed`, added to VALID_CENTERLINE_STYLES because the vocabulary held
    only centre stripes and a one-way multi-lane carriageway has no centre to stripe. It is
    INFERRED from OSM `lanes=2` rather than observed - see config.yaml, where that is said at
    length - so it is the one marking on this sheet a street-view check could still remove.
    """
    if model is None:
        return baseline
    return all_crosswalks_continental(apply_observed_parking(baseline, model))


def _bikeway_on_the_east_kerb(baseline: DesignState, model, section) -> DesignState:
    """Both proposals, differing only in `section` - the class that decides the ORDERING.

    ONE BUILDER, because the two proposals are one design drawn two ways round and writing them
    twice is how they would come to differ in something that is not the ordering. Everything
    else is shared: the same 5 ft lane and 2 ft buffer, the same observed bay on both kerbs, the
    same posts, the same paint-and-post curb extensions, the same pinning to the east kerb.

    THE WEST KERB GETS ITS OBSERVED BAY AND NOTHING ELSE - in particular NOT
    hold_travel_lane_at_target, which is what an earlier version of this called there. That
    function holds a lane at target and spends the surplus on parking, and the surplus here is
    already spent: it marked its own 8 ft PARALLEL bay on that kerb, which then sat in
    apply_observed_parking's way, so the west kerb of a street with 60-degree parking on both
    sides was drawn with 8 ft parallel stalls. Measured: 22.00 ft pitch and 0.00 ft of skew
    against the east kerb's correct 10.39 and 11.60.

    Reese Ave IS left alone, and the reason is the tracing rather than the street: its kerb is
    traced around the corner returns and stops, so only stations 35-40 (east) and 45-50 (west)
    have kerb on both sides at once. hold_travel_lane_at_target can promise an 11 ft lane over
    about 15 ft of a 130 ft leg and carries the rest as a FacilityRefusal, which
    travel_lane_over_target then reports - correctly - as a 17.19 ft lane on a leg this design
    claims to have restriped. Restriping a leg we cannot measure is the thing that check exists
    to stop. Trace Reese's kerbs and this goes away.
    """
    state = all_crosswalks_continental(baseline)
    for leg_name, leg in model.legs.items():
        if not leg_name.startswith("grand_central"):
            continue
        east = side_facing(leg, "east")
        state = state.apply(section(
            LegSide(leg_name, east),
            width_ft=BIKE_LANE_FT,
            buffer_ft=_buffer_ft(),
            # THE OBSERVED BAY, carried by the bikeway's own section rather than by a separate
            # MarkedParking on the same kerb: one section places all of a rigid cross-section
            # (SKILLS 0a), and two treatments on one leg-side paint over each other.
            parking_ft=_bay_depth_ft(),
            parking_angle_deg=PARKING_ANGLE_DEG,
            # Pinned, which is what lets the 7.37 ft of travel-lane surplus reach this kerb -
            # see the module docstring. Without it this section is 4.07 ft too wide for its own
            # half of the street and is refused.
            pin_to_kerb=True,
            # Asked of the street rather than written down per leg, for the same reason the
            # side is: both legs' bearings point outward, so the southern approach's riders and
            # drivers travel INWARD and only the compass knows it.
            runs_outward=traffic_runs_outward(model, leg, east)))
        state = state.apply(AddBikeLaneBollards(LegSide(leg_name, east)))
    # AFTER the bikeways, not before: a kerb the bikeway's own section already carries stalls on
    # must not get a second parking treatment painting over them, and this skips exactly the
    # leg-sides that have an AddBikeLane on them.
    state = apply_observed_parking(state, model)
    return _daylight_every_parked_kerb(state)


def build_bike_lane_inboard(baseline: DesignState, model=None) -> DesignState:
    """The bikeway BETWEEN the travel lanes and the parking, with the bay still against the kerb.

    Across the east kerb, outward from the alignment: travel lane, 2 ft buffer, 5 ft bike lane,
    then the 60-degree bay to the kerb. The bay does not move at all - this is the street as it
    is with a bike lane inserted into the travel lanes' surplus, which is the cheapest thing
    that can be built here and the one that asks nothing of the parking.

    THE TRADE AGAINST build_bike_lane_at_the_kerb IS THE REVERSING MOVEMENT. A driver leaving a
    front-in 60-degree bay reverses across whatever is inboard of it, and on this ordering that
    is the bike lane. Back-in angled parking removes the conflict and is the usual answer where
    a bikeway runs beside an angled bay; this project has no back-in marking yet, so the
    conflict is named here rather than drawn away.
    """
    if model is None:
        return baseline
    return _bikeway_on_the_east_kerb(baseline, model, AddBikeLane)


# THE OTHER ORDERING IS NOT DRAWN YET, and it is deliberately absent rather than half-built.
# Putting the bike lane AGAINST THE KERB with the 60-degree bay floated off it is the same four
# pieces in a different order and the identical 27.55 ft of section, so it fits exactly as well -
# and it buys something real: a driver reversing out of a front-in bay never crosses the bikeway,
# because they reverse toward the travel lane instead.
#
# It is not here because AddKerbsideBikeLane does not survive being pinned. Built, it reports 25
# fatal markings_collide violations on this junction - its bike_lane_edge_line painted 100% over
# its own bike_buffer_fill, and every stall divider drawn inside that buffer - because that
# class's parking_curb_offset_ft and buffer band are derived on the assumption that the section
# starts a target lane width from the alignment. A scenario that fails the invariants is worse
# than an absent one: phase3 still writes the PNG, so it would ship a picture nobody could
# export in 3D. Fix those two offsets against the pinned travel edge and this becomes six lines.
