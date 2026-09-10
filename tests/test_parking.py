"""hold_travel_lane_at_target: does the kerbside leftover actually reach the traced kerb."""
import contextlib
import io

import pytest

from tests.conftest import needs_source_data


@needs_source_data
def test_a_kerb_that_narrows_only_at_its_tail_still_keeps_the_lane_over_the_reach():
    """A whole-leg minimum let one narrow tail veto a kerb that has room almost everywhere.

    w_broad_st_southwest's left kerb, at ROAD_SKETCHES_FRAME_SCALE=3.0 - the scale the corridor's own
    checked-in renders actually use, confirmed by matching output/wbroad_louellen's committed
    frame.radius_m against every candidate scale rather than assuming it (tests/conftest.py's
    WIDE_FRAME_SCALE=2.5 is the suite's own invariant-sweep convention and is a DIFFERENT number)
    - holds an 11 ft travel lane for 336 of its 390 traced feet and pinches inside it only over
    the last stretch. Judged by narrowest_half_width_ft over the WHOLE leg - what
    hold_travel_lane_at_target used before _lane_target_reach_ft existed - that one pinch reads
    back as "the street has nothing spare" and NOTHING is drawn on the other 336 ft either, which
    is exactly the bug SKILLS 0a and 0b describe for the two-way section: a single station
    deciding a whole approach.

    Asked per station instead, the lane (or marking) has to reach as far as the kerb actually
    holds it and refuse the tail by name - not run the whole leg (which would draw past where
    the kerb narrows) and not silently draw nothing (which is what shipped and drew no hatching
    on this kerb at all).
    """
    from src.geometry.intersection import load_intersection_model
    from src.geometry.model import curb_station_span, side_facing
    from src.geometry.targets import LegSide, LegTarget
    from src.geometry.treatments.corridor import BROAD_ST_TWO_WAY_BIKEWAY
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.parking import MarkedParking
    from src.geometry.treatments.state import DesignState
    from src.render.frame import FRAME_SCALE_ENV

    LEG, SIDE = "w_broad_st_southwest", "left"
    CORRIDOR_RENDER_SCALE = "3.0"     # see the docstring: this is not WIDE_FRAME_SCALE

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(FRAME_SCALE_ENV, CORRIDOR_RENDER_SCALE)
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_intersection_model(site="wbroad_louellen")
    finally:
        monkey.undo()

    span = curb_station_span(model.legs[LEG], SIDE)
    assert span is not None, f"{LEG} {SIDE} has no traced kerb - this test is pinning nothing"

    # NOT a bare call: hold_travel_lane_at_target runs on the FAR kerb, after the corridor's
    # own AddTwoWayBikeLane has already gone on the near one (side_facing(leg, "north"), here
    # "right") and set the divider shift the far kerb's own room is measured against. Calling
    # it on a treatment-free state skips that shift and does not reproduce the real defect.
    near_side = side_facing(model.legs[LEG], BROAD_ST_TWO_WAY_BIKEWAY.side)
    assert near_side != SIDE, (
        f"{LEG}'s {BROAD_ST_TWO_WAY_BIKEWAY.side}-facing kerb is {SIDE!r}, same as the far "
        f"kerb under test - the fixture assumption behind this test no longer holds")
    with contextlib.redirect_stdout(io.StringIO()):
        state = BROAD_ST_TWO_WAY_BIKEWAY._place_on(
            DesignState.from_model(model), LEG, near_side, quiet=True)

    lane = state.treatment_for(LaneNarrowing, LegTarget(LEG))
    parking = state.treatment_for(MarkedParking, LegSide(LEG, SIDE))
    treatment = lane if lane is not None else parking
    assert treatment is not None, (
        f"nothing at all was marked on {LEG} {SIDE} - the whole-leg minimum is still vetoing "
        f"most of this kerb that has room")

    assert treatment.end_ft is not None, (
        f"the kerb narrows inside this leg's own tail (traced {span[0]:.1f}-{span[1]:.1f} ft), "
        f"so the treatment should stop short of the leg's end and refuse the rest rather than "
        f"either running the whole leg or (as it did before this fix) refusing all of it")
    assert float(span[0]) < treatment.end_ft < float(span[1]), (
        f"end_ft={treatment.end_ft} is not inside the traced kerb "
        f"{span[0]:.1f}-{span[1]:.1f} ft")

    refusals = state.refusals_on(LEG, SIDE)
    assert len(refusals) == 1, f"expected exactly one tail refusal, got {refusals}"
    tail = refusals[0]
    assert tail.start_ft == pytest.approx(treatment.end_ft), (
        f"the refusal starts at {tail.start_ft:.1f} ft but the paint stopped at "
        f"{treatment.end_ft:.1f} ft - a gap between them is kerb with neither paint nor a "
        f"reason on it")
    assert tail.end_ft >= float(span[1]) - 1e-6, (
        f"the refusal covers to {tail.end_ft:.1f} ft but the kerb is traced to {span[1]:.1f} ft")
    assert tail.narrowest_ft is not None and tail.narrowest_ft > 0
    assert f"{tail.narrowest_ft:.2f}" in tail.reason, (
        f"the reason should quote the width that stopped it: {tail.reason}")


# --------------------------------------------------------------------------
# Which way an angled bay leans - traffic_runs_outward and apply_observed_parking
# --------------------------------------------------------------------------

def a_one_way_pair_of_legs():
    """The two Grand Central Ave approaches, near enough: one street, both legs NORTHBOUND.

    A junction's legs both point OUTWARD from the centre by construction, so a bearing of 6.6
    and one of 186.7 deg are the two halves of one straight street - and northbound traffic runs
    outward along the first and INWARD along the second. That is the whole difficulty this
    fixture exists to reproduce, and no leg's own frame can see it.
    """
    from shapely.geometry import LineString

    from src.geometry.model import Leg

    return {"north": Leg(name="north", centerline=LineString([(0, 0), (0, 190)]),
                          curb_to_curb_ft=70.0),
            "south": Leg(name="south", centerline=LineString([(0, 0), (0, -190)]),
                          curb_to_curb_ft=70.0)}


def a_model_double(legs, **leg_keys):
    """Just the `config["legs"]` apply_observed_parking and traffic_runs_outward read.

    A double rather than a real site because the site this is about (lavallette_reese) is
    outside the committed test fixture's extent - see tests/fixtures/data - so a suite run
    cannot build it.
    """
    import types

    observed = {"sides": ["left", "right"], "angle_deg": 60,
                "source": "the test's own assertion"}
    return types.SimpleNamespace(
        config={"legs": {name: {"existing_parking": observed, **leg_keys} for name in legs}})


def test_a_one_way_street_leans_both_its_bays_the_way_THE_TRAFFIC_runs():
    """`traffic_heads_toward: north` on both legs, so all four bays point north.

    Which means OUTWARD on the northern approach and INWARD on the southern one, with the skew
    sign flipping between them - and the two kerbs of each leg agreeing, where the two-way rule
    would have leaned them at each other. A bay leaning against the traffic can only be entered
    by reversing into the travel lane, so this is not a drafting nicety.
    """
    from src.geometry.treatments import DesignState, MarkedParking, apply_observed_parking

    legs = a_one_way_pair_of_legs()
    model = a_model_double(legs, traffic_heads_toward="north")
    state = apply_observed_parking(DesignState(legs=legs, corner_fillets={}), model)

    leans = {(t.target.leg, t.target.side): t.runs_outward
             for t in state.treatments_of(MarkedParking)}
    assert leans == {("north", "left"): True, ("north", "right"): True,
                     ("south", "left"): False, ("south", "right"): False}
    skews = {t.target.leg: t.skew_ft for t in state.treatments_of(MarkedParking)}
    assert skews["north"] == pytest.approx(9.0, abs=1e-4)
    assert skews["south"] == pytest.approx(-9.0, abs=1e-4)


def test_a_two_way_street_still_leans_its_bays_by_the_SIDE():
    """No `traffic_heads_toward`, so the ordinary rule holds: right kerb outbound, left inbound.

    Pinned beside the test above so a fix for the one-way case cannot quietly become the rule
    for every street. This is what four of this repo's five sites are.
    """
    from src.geometry.treatments import DesignState, MarkedParking, apply_observed_parking

    legs = a_one_way_pair_of_legs()
    state = apply_observed_parking(DesignState(legs=legs, corner_fillets={}),
                                    a_model_double(legs))

    leans = {(t.target.leg, t.target.side): t.runs_outward
             for t in state.treatments_of(MarkedParking)}
    assert leans == {("north", "left"): False, ("north", "right"): True,
                     ("south", "left"): False, ("south", "right"): True}


def test_the_direction_of_travel_is_asked_of_the_model_not_of_the_caller():
    """traffic_runs_outward directly, because THREE pipeline scripts ask it without a site.

    phase3_treatments, phase4_render_3d and phase4_export_geometry each build the panel labelled
    "Existing Conditions" themselves, and none of them imports a site's scenarios.py. While the
    lean lived in sites/lavallette_reese/scenarios.py as a module constant, those three got
    apply_observed_parking's old two-way default and drew half the bays mirrored.
    """
    from src.geometry.treatments import traffic_runs_outward

    legs = a_one_way_pair_of_legs()
    one_way = a_model_double(legs, traffic_heads_toward="north")
    two_way = a_model_double(legs)

    assert [traffic_runs_outward(one_way, legs[n], s)
            for n in ("north", "south") for s in ("left", "right")] == [True, True, False, False]
    assert [traffic_runs_outward(two_way, legs[n], s)
            for n in ("north", "south") for s in ("left", "right")] == [False, True, False, True]


@needs_source_data
def test_existing_conditions_is_the_untreated_street_where_nothing_was_recorded(site_models):
    """The helper is a NO-OP on every site that declares no `existing_parking`.

    It is what the pipeline now labels "Existing Conditions" everywhere, so if it added anything
    of its own the four Mercer County sites' before/after sheets would all have moved - which is
    Danny's churn test, and the reason the assertion is on the treatments and the notes rather
    than on a picture.
    """
    from src.geometry.treatments import DesignState, existing_conditions

    for site, model in sorted(site_models.items()):
        declared = [name for name, cfg in model.config.get("legs", {}).items()
                    if cfg.get("existing_parking")]
        if declared:
            continue    # a site with an observation SHOULD differ - that is the point
        bare = DesignState.from_model(model)
        assert existing_conditions(model).treatments == bare.treatments, site
        assert existing_conditions(model).notes == bare.notes, site
