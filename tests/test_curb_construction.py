"""Two ways a leg's curbs came out as something that is not two edges of one street.

1. GEOS 3.13's offset_curve SPLITS a smooth line into touching pieces, so a Leg's curb was a
   MultiLineString and build_corner_fillets died in `substring`.
2. A traced kerb pair that does not form a cross-section (one kerb crossing the alignment)
   measured a leg NEGATIVE width, and the export died on an empty np.arange.
"""
import contextlib
import io

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from src.geometry.model import Leg, point_at, station_offset_many
from src.geometry.model.leg_frame import offset_curb_line

# A real shape, measured: twelve 5 ft segments then one 295 ft segment (what _blend_onto leaves
# when a through street is blended over its first 60 ft), 355 ft in all, turning at most 0.35 deg
# at any vertex, in state-plane feet near W Broad St. offset_curve(-19.93) splits it into
# [5.0 ft, 349.96 ft] with a gap of 0.0 ft between the pieces.
BLENDED_CENTERLINE = LineString([
    (417160.0, 565384.0), (417165.0, 565383.984), (417170.0, 565383.987),
    (417175.0, 565383.988), (417180.0, 565383.997), (417185.0, 565383.999),
    (417190.0, 565384.002), (417195.0, 565383.999), (417200.0, 565384.002),
    (417205.0, 565383.975), (417210.0, 565383.961), (417215.0, 565383.942),
    (417220.0, 565383.931), (417514.998, 565382.843)])
SPLITTING_OFFSET_FT = -19.93


def test_the_measured_centerline_still_splits_in_geos():
    """The premise of the tests below. If this skips, GEOS fixed offset_curve and the repro needs
    a new shape - the fix itself is harmless either way."""
    raw = BLENDED_CENTERLINE.offset_curve(SPLITTING_OFFSET_FT)
    if not isinstance(raw, MultiLineString):
        pytest.skip("this GEOS no longer splits the measured centreline")
    assert [round(part.length, 1) for part in raw.geoms] == [5.0, 350.0]
    assert raw.geoms[0].distance(raw.geoms[1]) == 0.0


@pytest.mark.parametrize("sign", [1, -1])
def test_a_leg_curb_is_one_linestring_where_offset_curve_splits(sign):
    leg = Leg("blended", BLENDED_CENTERLINE, 2 * abs(SPLITTING_OFFSET_FT))
    curb = leg.left_curb if sign > 0 else leg.right_curb
    assert isinstance(curb, LineString), f"curb came out {curb.geom_type}"
    assert not curb.is_empty


def test_the_joined_curb_is_the_offset_curve_and_runs_the_centerlines_way():
    """Joined, not re-derived: every piece offset_curve returned is still on the line, in the
    centreline's direction, and it sits the full offset off the centreline throughout."""
    raw = BLENDED_CENTERLINE.offset_curve(SPLITTING_OFFSET_FT)
    curb = offset_curb_line(BLENDED_CENTERLINE, SPLITTING_OFFSET_FT)
    pieces = raw.geoms if isinstance(raw, MultiLineString) else [raw]
    assert curb.length == pytest.approx(sum(part.length for part in pieces), abs=1e-6)
    for part in pieces:
        assert curb.hausdorff_distance(part) <= curb.length    # sanity: same neighbourhood
        assert all(curb.distance(Point(xy)) < 1e-6 for xy in part.coords)
    assert Point(curb.coords[0]).distance(Point(BLENDED_CENTERLINE.coords[0])) < \
        Point(curb.coords[-1]).distance(Point(BLENDED_CENTERLINE.coords[0]))
    _stations, offsets = station_offset_many(BLENDED_CENTERLINE, np.asarray(curb.coords))
    assert offsets == pytest.approx(SPLITTING_OFFSET_FT, abs=0.1)


def test_a_line_that_never_split_comes_back_exactly_as_offset_curve_drew_it():
    """Most legs: bent, long, nothing wrong with them. They must not move - not by a hair and not
    in vertex order."""
    bent = LineString([(0, 0), (60, 0), (110, 14), (160, 40), (230, 44)])
    for offset in (7.0, -7.0, 15.5, -15.5):
        raw = bent.offset_curve(offset)
        assert isinstance(raw, LineString)
        assert np.array_equal(np.asarray(offset_curb_line(bent, offset).coords),
                              np.asarray(raw.coords))


def _straight_through_pair():
    """A 643 ft leg and its opposite number, so both of the first's sides count as running
    straight through the junction and are carried back to it."""
    return {"a": Leg("a", LineString([(0, 0), (643, 0)]), 26.0),
            "b": Leg("b", LineString([(0, 0), (-300, 0)]), 26.0)}


def _ways_for(leg, left, right):
    return [(LineString([point_at(leg.centerline, s, o) for s, o in pts]), {}, i)
            for i, pts in enumerate((left, right))]


# A through street's curb is carried back to the junction by the slope its vertices show. This
# left kerb rises 0.08 ft per ft over 100 ft of tracing, so at the junction it is at -1 ft: on the
# wrong side of the alignment, with a perfectly ordinary right kerb beside it.
LEFT_KERB = [(100, 7.0), (150, 11.0), (200, 15.0)]
RIGHT_KERB = [(100, -12.0), (150, -12.0), (200, -12.0)]
# prospect_street_1's shape: the same kind of tracing, but 500 ft out on a 643 ft leg.
FAR_LEFT_KERB = [(500, 12.0), (520, 14.0), (560, 18.0)]
FAR_RIGHT_KERB = [(500, -12.0), (520, -12.2), (560, -12.5)]


def test_a_kerb_that_crosses_the_alignment_is_refused_not_clamped():
    from src.geometry.intersection.fitting import _apply_traced_curb_lines

    legs = _straight_through_pair()
    ways = _ways_for(legs["a"], LEFT_KERB, RIGHT_KERB)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        coverage = _apply_traced_curb_lines(legs, ways, Point(0, 0))
    leg = legs["a"]
    assert leg.traced_sides == set(), "a pair that is not a cross-section was kept"
    assert ("a", "left") not in coverage and ("a", "right") not in coverage
    assert "crosses it" in out.getvalue() and "Refused" in out.getvalue(), "refused silently"
    # What an untraced leg has: both curbs an offset from the alignment at its own width.
    for side, sign in (("left", 1), ("right", -1)):
        _s, offsets = station_offset_many(leg.centerline, np.asarray(getattr(leg, f"{side}_curb").coords))
        assert offsets == pytest.approx(sign * 13.0, abs=1e-6)


def test_the_width_fit_never_returns_a_width_at_or_below_zero():
    from src.geometry.intersection.fitting import _fit_legs_to_traced_kerbs

    legs = _straight_through_pair()
    ways = _ways_for(legs["a"], LEFT_KERB, RIGHT_KERB)
    with contextlib.redirect_stdout(io.StringIO()):
        _fit_legs_to_traced_kerbs(legs, ways, Point(0, 0), {})
    for name, leg in legs.items():
        assert leg.curb_to_curb_ft > 0, f"{name} came out {leg.curb_to_curb_ft:.1f} ft wide"
        for side in ("left", "right"):
            _s, offsets = station_offset_many(leg.centerline, np.asarray(getattr(leg, f"{side}_curb").coords))
            assert (offsets > 0).all() if side == "left" else (offsets < 0).all(), (
                f"{name}'s {side} curb is on the wrong side of its alignment")


def test_a_cross_section_with_the_kerbs_the_wrong_way_round_is_not_measured():
    from src.geometry.intersection.fitting import _traced_cross_section

    leg = Leg("a", LineString([(0, 0), (200, 0)]), 26.0)
    leg.left_curb = LineString([point_at(leg.centerline, s, -5.0) for s in (0, 100, 200)])
    leg.right_curb = LineString([point_at(leg.centerline, s, -20.0) for s in (0, 100, 200)])
    leg.traced_sides = {"left", "right"}
    assert _traced_cross_section(leg) is None


def test_a_through_streets_kerb_is_not_carried_further_back_than_it_was_traced():
    """prospect_street_1 was traced over 614-620 ft (left) and 570-672 ft (right) and both kerbs
    were run back to the junction on the slope those vertices show - 570 ft on a 100 ft baseline.
    That fabricated a 37.3 ft street where the vertices say 27, and a left kerb that crossed the
    centreline. The curb starts where it was traced."""
    from src.geometry.intersection.fitting import _apply_traced_curb_lines, _fit_legs_to_traced_kerbs

    legs = _straight_through_pair()
    ways = _ways_for(legs["a"], FAR_LEFT_KERB, FAR_RIGHT_KERB)
    with contextlib.redirect_stdout(io.StringIO()):
        _apply_traced_curb_lines(legs, ways, Point(0, 0))
    leg = legs["a"]
    assert leg.traced_sides == {"left", "right"}
    for side in ("left", "right"):
        stations, _o = station_offset_many(leg.centerline, np.asarray(getattr(leg, f"{side}_curb").coords))
        assert stations.min() >= 499.0, f"{side} kerb was carried back to station {stations.min():.0f}"
    with contextlib.redirect_stdout(io.StringIO()):
        _fit_legs_to_traced_kerbs(legs, ways, Point(0, 0), {})
    assert legs["a"].curb_to_curb_ft == pytest.approx(26.0, abs=2.5)
