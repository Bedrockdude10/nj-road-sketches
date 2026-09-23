"""Treatment proposals for this site.

THE CORRIDOR'S EASTERN TERMINUS, and the honest one: unlike the west end, the kerb here is
traced right up to the borough line 133 ft east of the junction, so nothing on this sheet is
extrapolated. The section, the side and the rung ladder are BROAD_ST_TWO_WAY_BIKEWAY's.

The two Elm St legs are `estimated` widths with no traced kerb at all. They carry no facility,
so that weakness does not reach the bikeway - but it does reach their crossings and their
corner returns, and anything read off them is a placeholder. See config.yaml.
"""
from src.geometry.treatments import (all_crosswalks_continental, apply_osm_parking,
    complete_centerlines, DesignState, osm_derived_baseline)


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given: the street as OSM says
    it is used, with nothing proposed."""
    return osm_derived_baseline(baseline, model)


def build_proposal_two_way_bike_lane(baseline: DesignState, model=None) -> DesignState:
    """The borough's two-way bikeway, ended at the east borough line.

    Same treatment as the west terminus and deliberately not a different design: a rider who
    has learned how the west end works should find the east end working the same way. The
    ending goes on the EAST approach, which is the one that runs out of borough.
    """
    from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY

    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=("n_elm_st_north", "s_elm_st_south"))
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    return BROAD_ST_TWO_WAY_BIKEWAY.apply_to(state, model)
