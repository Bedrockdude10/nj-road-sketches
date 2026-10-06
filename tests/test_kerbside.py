"""One answer to "how much of this kerb is painted" (src/geometry/treatments/kerbside.py)."""
import contextlib
import io
import re
from pathlib import Path

import pytest

from src.checks import SceneContext, TravelLanesHoldTheTarget, TravelLanesKeepTheirWidth
from src.geometry.targets import LegSide, LegTarget
from src.geometry.treatments import (DesignState, LaneNarrowing, MarkedParking, RestrictionMarking,
                                     divider_shift_toward_ft, kerbside_paint_ft)
from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane
from tests.conftest import synthetic_leg

KERB_FT = 22.0
LENGTH_FT = 200.0
STRAIGHT = [(0.0, KERB_FT), (LENGTH_FT, KERB_FT)]
LEG = "east"


def _state(*treatments):
    leg = synthetic_leg(LEG, LENGTH_FT, 2 * KERB_FT, STRAIGHT, STRAIGHT)
    with contextlib.redirect_stdout(io.StringIO()):
        return DesignState(legs={LEG: leg}, corner_fillets={}).apply(*treatments)


def _area(inner_ft: float) -> RestrictionMarking:
    """A road_marking=restriction area on the left kerb, reaching in to `inner_ft`."""
    ring = ((0.0, inner_ft), (LENGTH_FT, inner_ft), (LENGTH_FT, KERB_FT), (0.0, KERB_FT),
            (0.0, inner_ft))
    return RestrictionMarking(LegSide(LEG, "left"), rings_ft=(ring,), osm_ids=(-1,))


@pytest.mark.parametrize("treatments, expected_ft", [
    ([MarkedParking(LegSide(LEG, "left"), depth_ft=8.0, curb_offset_ft=1.0)], 9.0),
    ([LaneNarrowing(LegTarget(LEG), stripe_width_ft=5.0, sides=("left",))], 5.0),
    ([_area(16.0)], KERB_FT - 16.0),
    ([], 0.0),
    ([LaneNarrowing(LegTarget(LEG), stripe_width_ft=5.0, sides=("left",)),
      MarkedParking(LegSide(LEG, "left"), depth_ft=8.0)], 8.0),        # parking wins
])
def test_kerbside_paint_precedence(treatments, expected_ft):
    assert kerbside_paint_ft(_state(*treatments), LEG, "left") == pytest.approx(expected_ft)


def _beside_a_two_way_track(lane_ft: float) -> DesignState:
    """A two-way track on the right, and on the left an area leaving a `lane_ft` travel lane."""
    track = AddTwoWayBikeLane(LegSide(LEG, "right"), width_ft=10.0, buffer_ft=3.0)
    shift_ft = divider_shift_toward_ft(_state(track), LEG, "left")
    return _state(track, _area(lane_ft + shift_ft))


def _kinds(check, state):
    return {v.check for v in check().run(SceneContext(state=state))}


def test_an_area_that_holds_the_lane_at_target_passes():
    assert "travel_lane_over_target" not in _kinds(TravelLanesHoldTheTarget,
                                                   _beside_a_two_way_track(11.0))


def test_an_area_that_leaves_the_lane_wide_is_over_target():
    assert "travel_lane_over_target" in _kinds(TravelLanesHoldTheTarget,
                                               _beside_a_two_way_track(14.0))


def test_an_area_that_leaves_the_lane_narrow_is_too_narrow():
    assert "travel_lane_too_narrow" in _kinds(TravelLanesKeepTheirWidth,
                                              _beside_a_two_way_track(9.0))


def test_the_checks_hold_no_copy_of_the_kerbside_sum():
    source = Path("src/checks.py").read_text()
    assert not re.search(r"\.stripe_width_ft\b", source)
    assert "depth_ft + " not in source
