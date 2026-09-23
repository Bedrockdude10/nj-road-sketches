"""Treatment proposals for this site.

THE CORRIDOR'S WESTERN TERMINUS. Nothing here is a new design - the section, the side and the
rung ladder are all BROAD_ST_TWO_WAY_BIKEWAY's, declared once in src/geometry/treatments. What
this site adds is the one thing no other Broad St sheet can draw: what happens at the end,
where the facility meets the borough line and a rider on the far kerb has to get across.

Every width on the W Broad legs is osm_derived from Danny's traced kerbs; the two Lanning legs
are traced round the corner only. See config.yaml.
"""
from src.geometry.treatments import (all_crosswalks_continental, apply_osm_parking,
    complete_centerlines, DesignState, osm_derived_baseline)


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given: the street as OSM says
    it is used, with nothing proposed. Same as every other Hopewell site."""
    return osm_derived_baseline(baseline, model)


def build_proposal_two_way_bike_lane(baseline: DesignState, model=None) -> DesignState:
    """The borough's two-way bikeway, run southwest to where the borough ends - and ENDED there.

    The southwest approach is drawn 2,307.5 ft, all the way to the jurisdictional line, which is
    what makes this sheet the corridor's western terminus rather than one more junction on it.
    The far 2,070 ft of that is extrapolated from the leg's bearing and its nominal 33 ft; the
    config's `source` on that leg is the whole argument for why that is defensible here and
    nowhere else, and reading it is not optional before quoting anything off this drawing.

    EndTheBikeway is what stops the facility being a protected lane that simply stops. It puts
    a two-stage bicycle turn box at the far end (MUTCD 9E.11, whose Figure 9E-11 is titled for
    this exact case), shared-lane markings past the merge, and the W9-5 / R9-23 signing.
    """
    from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY

    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model,
                              legs=("n_lanning_ave_north", "s_lanning_ave_south"))
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    return BROAD_ST_TWO_WAY_BIKEWAY.apply_to(state, model)
