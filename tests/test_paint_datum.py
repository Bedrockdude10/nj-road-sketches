"""Tests for paint datum computation (paint spec v2, Phase 1)."""
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from src.geometry.paint.datum import (Across, Along, At, Centre, Glyph, Kerb, KerbToKerb, Narrowest,
                                      Taper, centre_profile, kerb_profile, place, resolve)
from src.geometry.model.leg_frame import Leg
from src.geometry.model import station_offset_many
from src.geometry.markings import BUFFER_EDGE_LINE
from src.geometry.paint import PaintContext, PaintPiece


# Test helpers
def straight(c2c=40.0):
    """Create a straight horizontal leg for testing."""
    return Leg("t", LineString([(0, 0), (200, 0)]), curb_to_curb_ft=c2c)


def trace(leg, side, coords):
    """Attach a traced kerb to a leg."""
    setattr(leg, f"{side}_curb", LineString(coords))
    leg.traced_sides.add(side)


S = np.arange(0.0, 201.0, 10.0)  # 21 stations
SPAN = (0.0, 200.0)              # S, as the span an Along shape is placed over
STEP = 10.0


class TestKerbProfile:
    """Tests D1-D5: kerb_profile with various setups."""

    def test_D1_traced_kerbs_on_both_sides(self):
        """D1: both kerbs traced; offsets are left 20, right -20; source 'traced'."""
        leg = straight()
        trace(leg, "left", [(0, 20), (200, 20)])
        trace(leg, "right", [(0, -20), (200, -20)])

        result = kerb_profile(leg, "left", S)
        np.testing.assert_allclose(result.offsets_ft, [20.0] * 21, atol=0.01)
        assert all(s == "traced" for s in result.source)

        result_r = kerb_profile(leg, "right", S)
        np.testing.assert_allclose(result_r.offsets_ft, [-20.0] * 21, atol=0.01)
        assert all(s == "traced" for s in result_r.source)

    def test_D2_derived_curbs_use_nominal_width(self):
        """D2: straight(40) with no traced kerbs; offsets nominal ±20, source 'nominal'."""
        leg = straight()  # curb_to_curb_ft=40, no traced sides

        result = kerb_profile(leg, "left", S)
        np.testing.assert_allclose(result.offsets_ft, [20.0] * 21, atol=0.01)
        assert all(s == "nominal" for s in result.source)

    def test_D3_partial_traced_kerb_tapers_to_nominal(self):
        """D3: traced left [(50,20),(150,20)] on 60 ft width; its own 20 ft is carried past the span."""
        leg = straight(60.0)
        trace(leg, "left", [(50, 20), (150, 20)])

        result = kerb_profile(leg, "left", S)
        expected = [20] * 21
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source) == (["carried"] * 5 + ["traced"] * 11 + ["carried"] * 5)

    def test_D4_mirrored_kerb_from_opposite_side(self):
        """D4: D3 on the right; no state line, so one 60 ft width off the left kerb, tapering to nominal."""
        leg = straight(60.0)
        trace(leg, "left", [(50, 20), (150, 20)])

        result = kerb_profile(leg, "right", S)
        expected = [-30, -32, -34, -36, -38] + [-40] * 11 + [-38, -36, -34, -32, -30]
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source) == (["nominal"] * 5 + ["offset"] * 11 + ["nominal"] * 5)

    def test_D5_mirrored_from_state_centreline(self):
        """D5: right traced at -18; state centreline at +2; left is mirrored (2*2-(-18)=22)."""
        leg = straight()
        trace(leg, "right", [(0, -18), (200, -18)])
        leg.state_centreline = LineString([(0, 2), (200, 2)])

        result = kerb_profile(leg, "left", S)
        np.testing.assert_allclose(result.offsets_ft, [22.0] * 21, atol=0.01)
        assert all(s == "mirrored" for s in result.source)


class TestCentreProfile:
    """Tests D6-D9: centre_profile with mixed traced/state sources."""

    def test_D6_state_centreline_overrides_nominal(self):
        """D6: same as D5; centre_profile returns state centreline at +2."""
        leg = straight()
        trace(leg, "right", [(0, -18), (200, -18)])
        leg.state_centreline = LineString([(0, 2), (200, 2)])

        result = centre_profile(leg, S)
        np.testing.assert_allclose(result.offsets_ft, [2.0] * 21, atol=0.01)
        assert all(s == "state" for s in result.source)

    def test_D7_reversed_state_centreline_still_works(self):
        """D7: state_centreline reversed and longer [(250,4),(-50,4)]; still returns 4."""
        leg = straight()
        leg.state_centreline = LineString([(250, 4), (-50, 4)])

        result = centre_profile(leg, S)
        np.testing.assert_allclose(result.offsets_ft, [4.0] * 21, atol=0.01)
        assert all(s == "state" for s in result.source)

    def test_D8_mixed_sources_priority_kerbs_then_state(self):
        """D8: left traced 0..100, right traced 0..200, state at -3; kerbs 0..100, state 100..200."""
        leg = straight()
        trace(leg, "left", [(0, 20), (100, 20)])
        trace(leg, "right", [(0, -16), (200, -16)])
        leg.state_centreline = LineString([(0, -3), (200, -3)])

        result = centre_profile(leg, S)
        # Stations 0-100 (11 stations): kerbs, midpoint = (20-16)/2 = 2
        # Stations 110-200 (10 stations): state = -3, tapered 0.20 ft/ft off station 100's 2.0
        expected = [2.0] * 11 + [0.0, -2.0] + [-3.0] * 8
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source)[:11] == ["kerbs"] * 11
        assert all(s == "state" for s in result.source[11:])

    def test_D9_kerb_profile_with_mixed_sources(self):
        """D9: same setup as D8; left kerb has traced then mirrored."""
        leg = straight()
        trace(leg, "left", [(0, 20), (100, 20)])
        trace(leg, "right", [(0, -16), (200, -16)])
        leg.state_centreline = LineString([(0, 1), (200, 1)])

        result = kerb_profile(leg, "left", S)
        expected = [20.0] * 11 + [18.0] * 10
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source)[:11] == ["traced"] * 11
        assert all(s == "mirrored" for s in result.source[11:])


class TestResolve:
    """Test D10: resolve with missing data raises ValueError."""

    def test_D10_no_traced_no_nominal_raises_error(self):
        """D10: leg with no curb_to_curb_ft and nothing traced raises ValueError."""
        leg = straight(c2c=None)

        with pytest.raises(ValueError) as exc_info:
            kerb_profile(leg, "left", S)
        assert "t" in str(exc_info.value)


class TestPlace:
    """Tests D11-D16: place() function with various configurations."""

    def test_D11_centre_offset_maintains_width(self):
        """D11: sinusoidal left kerb; Centre(0) vs Centre(11) differ by exactly 11.00 ft."""
        leg = straight()
        # Sinusoidal left kerb
        x_vals = np.linspace(0, 200, 41)
        y_vals = 20 + 3 * np.sin(2 * np.pi * x_vals / 100)
        trace(leg, "left", [(x, y) for x, y in zip(x_vals, y_vals)])
        trace(leg, "right", [(0, -20), (200, -20)])

        placed_c0 = place(leg, "left", Along(SPAN, Centre(0), step_ft=STEP))
        placed_c11 = place(leg, "left", Along(SPAN, Centre(11), step_ft=STEP))

        # Sample both lines and compare offsets at matching stations
        line_c0 = placed_c0.geometry
        line_c11 = placed_c11.geometry

        s0, off0 = station_offset_many(leg.centerline, np.asarray(line_c0.coords))
        s1, off1 = station_offset_many(leg.centerline, np.asarray(line_c11.coords))

        # Find matching stations and verify the difference is 11.00
        # Both lines should have points at similar stations
        n_compared = 0
        for i, st in enumerate(s0):
            if not np.isnan(st):
                # Find nearest station in s1
                matches = np.isclose(s1, st, atol=0.1)
                assert matches.any(), f"No matching station in s1 for {st}"
                idx = np.where(matches)[0][0]
                diff = off1[idx] - off0[i]
                assert abs(diff - 11.0) < 0.01
                n_compared += 1
        assert n_compared == len(S), f"Only compared {n_compared} stations, expected {len(S)}"

    def test_D12_band_area_calculation(self):
        """D12: band between Kerb(0) and Centre(11); area from trapezoid integration."""
        leg = straight()
        x_vals = np.linspace(0, 200, 41)
        y_vals = 20 + 3 * np.sin(2 * np.pi * x_vals / 100)
        trace(leg, "left", [(x, y) for x, y in zip(x_vals, y_vals)])
        trace(leg, "right", [(0, -20), (200, -20)])

        placed = place(leg, "left", Along(SPAN, Kerb(0), Centre(11), step_ft=STEP))

        assert len(placed.pinched_stations) == 0
        assert placed.geometry is not None

        # Calculate expected area: width = kerb - (centre + 11)
        kerb_p = kerb_profile(leg, "left", S)
        centre_p = centre_profile(leg, S)
        width = kerb_p.offsets_ft - (centre_p.offsets_ft + 11)

        # Verify width is never negative
        assert (width >= 0).all()

        expected_area = np.trapezoid(width, S)
        actual_area = placed.geometry.area

        # Within 1% tolerance
        assert abs(actual_area - expected_area) / expected_area < 0.01

    def test_D13_pinched_band_truncates_geometry(self):
        """D13: traced left kerb [(0,10),(100,10),(120,14),(200,14)]; pinches before station 110."""
        leg = straight()
        trace(leg, "left", [(0, 10), (100, 10), (120, 14), (200, 14)])
        # No right kerb, no state: use nominal for right

        placed = place(leg, "left", Along(SPAN, Kerb(0), Centre(11), step_ft=STEP))

        # Pinched stations should be the first 11 (0-100)
        np.testing.assert_array_equal(placed.pinched_stations, S[:11])

        # Geometry is non-None but only covers the latter part
        assert placed.geometry is not None
        bounds = placed.geometry.bounds  # (minx, miny, maxx, maxy)
        assert abs(bounds[0] - 110) < 0.01  # x_min = 110
        assert abs(bounds[2] - 200) < 0.01  # x_max = 200
        assert bounds[1] >= 11 - 0.01  # y_min >= 11 (band not swapped into lane)

    def test_D14_kerb_to_kerb_polygon(self):
        """D14: KerbToKerb on D1 setup with S = [90, 100, 110]."""
        leg = straight()
        trace(leg, "left", [(0, 20), (200, 20)])
        trace(leg, "right", [(0, -20), (200, -20)])

        placed = place(leg, "left", Along((90.0, 110.0), KerbToKerb(), step_ft=STEP))

        assert placed.geometry is not None
        bounds = placed.geometry.bounds
        # bounds should be (90, -20, 110, 20)
        assert abs(bounds[0] - 90) < 0.01
        assert abs(bounds[1] - (-20)) < 0.01
        assert abs(bounds[2] - 110) < 0.01
        assert abs(bounds[3] - 20) < 0.01

        # Contains Point(100, 0)
        assert placed.geometry.contains(Point(100, 0))

        # Datum is all "traced"
        assert placed.datum == {"traced": 1.0}

    def test_D15_datum_sums_to_one(self):
        """D15: D8 setup; datum keys are {traced, mirrored, kerbs, state}; values sum to 1.0."""
        leg = straight()
        trace(leg, "left", [(0, 20), (100, 20)])
        trace(leg, "right", [(0, -16), (200, -16)])
        leg.state_centreline = LineString([(0, 1), (200, 1)])

        placed = place(leg, "left", Along(SPAN, Kerb(0), Centre(11), step_ft=STEP))

        # Check datum keys
        expected_keys = {"traced", "mirrored", "kerbs", "state"}
        assert set(placed.datum.keys()) == expected_keys

        # Check sum
        assert abs(sum(placed.datum.values()) - 1.0) < 1e-9

        # traced should be 11/21*0.5 (11 stations with traced kerb, 0.5 weight each)
        assert abs(placed.datum["traced"] - 11/21 * 0.5) < 1e-9

    def test_D16_paint_context_calls_paint_method(self):
        """D16: PaintContext.paint() returns non-empty list of PaintPiece with datum."""
        # Use simple straight leg like test_paint.py
        from src.geometry.model import Leg as LegClass
        from src.geometry.treatments.state import DesignState as DS

        leg = LegClass("test", LineString([(0, 0), (200, 0)]), curb_to_curb_ft=40.0)
        trace(leg, "left", [(0, 20), (200, 20)])
        trace(leg, "right", [(0, -20), (200, -20)])

        state = DS(legs={"test": leg}, corner_fillets={})
        ctx = PaintContext(state=state, crosswalk_offsets={}, center_ft=None)

        # Call paint with any PaintKind from test_paint.py (using BUFFER_EDGE_LINE as example)
        result = ctx.paint(BUFFER_EDGE_LINE, "test", "left", Along((20.0, 60.0), Kerb(0), Centre(11)))

        assert len(result) > 0
        assert all(isinstance(p, PaintPiece) for p in result)
        assert all(p.datum for p in result)  # non-empty datum
        assert all(abs(sum(p.datum.values()) - 1.0) < 1e-9 for p in result)


class TestNarrowest:
    """D17-D20: Narrowest(inset) - a kerbside zone of declared depth, held straight at the kerb's
    narrowest over the stations asked for, measured from the centre line."""

    PINCH_LEFT = [(0, 22), (100, 20), (200, 22)]

    def test_D17_symmetric_pinch_gives_a_straight_line_at_the_narrowest_less_the_inset(self):
        leg = straight()
        trace(leg, "left", self.PINCH_LEFT)
        trace(leg, "right", [(s, -o) for s, o in self.PINCH_LEFT])
        result = resolve(leg, "left", Narrowest(8.0), S)
        np.testing.assert_allclose(result.offsets_ft, [12.0] * 21, atol=0.01)
        assert all(s == "traced" for s in result.source)

    def test_D18_it_is_held_off_the_centre_not_the_alignment(self):
        """Right kerb straight at -22: the centre is (left - 22) / 2, so left - centre is
        21 + |s - 100| / 100, narrowest 21 at s = 100, and the line is centre + 13."""
        leg = straight()
        trace(leg, "left", self.PINCH_LEFT)
        trace(leg, "right", [(0, -22), (200, -22)])
        result = resolve(leg, "left", Narrowest(8.0), S)
        np.testing.assert_allclose(result.offsets_ft, 12.0 + np.abs(S - 100.0) / 100.0, atol=0.01)

    def test_D19_the_right_side_is_signed(self):
        leg = straight()
        trace(leg, "left", self.PINCH_LEFT)
        trace(leg, "right", [(s, -o) for s, o in self.PINCH_LEFT])
        result = resolve(leg, "right", Narrowest(8.0), S)
        np.testing.assert_allclose(result.offsets_ft, [-12.0] * 21, atol=0.01)

    def test_D20_a_placed_narrowest_line_is_straight_and_says_it_came_off_the_traced_kerb(self):
        leg = straight()
        trace(leg, "left", self.PINCH_LEFT)
        trace(leg, "right", [(s, -o) for s, o in self.PINCH_LEFT])
        placed = place(leg, "left", Along(SPAN, Narrowest(8.0), step_ft=STEP))
        _, offsets = station_offset_many(leg.centerline, np.asarray(placed.geometry.coords))
        np.testing.assert_allclose(offsets, 12.0, atol=0.01)
        assert placed.datum == {"traced": 1.0}


def kerbs_at_20():
    leg = straight()
    trace(leg, "left", [(0, 20), (200, 20)])
    trace(leg, "right", [(0, -20), (200, -20)])
    return leg


def frame(geometry, leg):
    return station_offset_many(leg.centerline, np.asarray(geometry.coords))


class TestShapes:
    """D21-D28: every shape paint() draws, placed off references. Kerbs traced at +/-20."""

    def test_D21_across_runs_from_the_outer_reference_to_the_inner_at_one_station(self):
        placed = place(kerbs_at_20(), "left", Across(100.0, Kerb(0), Kerb(8)))
        np.testing.assert_allclose(np.asarray(placed.geometry.coords), [(100, 20), (100, 12)], atol=0.01)
        assert placed.datum == {"traced": 1.0}

    def test_D22_a_skewed_across_puts_its_outer_end_downstream(self):
        placed = place(kerbs_at_20(), "left", Across(100.0, Kerb(0), Kerb(8), skew_ft=3.0))
        np.testing.assert_allclose(np.asarray(placed.geometry.coords), [(103, 20), (100, 12)], atol=0.01)

    def test_D23_across_on_the_right_is_signed(self):
        placed = place(kerbs_at_20(), "right", Across(100.0, Kerb(0), Kerb(8)))
        np.testing.assert_allclose(np.asarray(placed.geometry.coords), [(100, -20), (100, -12)], atol=0.01)

    def test_D24_a_taper_leaves_the_edge_tangent_at_its_anchor_and_meets_the_kerb_at_its_target(self):
        """The arc is src/geometry/model/stripes.py:_taper_arc_points, fed the edge's offset at the
        anchor: 16 points, (60, 15) tangent to the edge, out to (30, 20) on the kerb."""
        leg = kerbs_at_20()
        placed = place(leg, "left", Taper(60.0, 30.0, Narrowest(5.0)))
        s, o = frame(placed.geometry, leg)
        assert len(s) == 16
        np.testing.assert_allclose([s[0], o[0], s[-1], o[-1]], [60, 15, 30, 20], atol=0.01)
        np.testing.assert_allclose([s[1], o[1]], [57.963, 15.022], atol=0.01)
        assert placed.datum == {"traced": 1.0}

    def test_D25_a_taper_fill_is_the_zone_between_the_kerb_and_the_arc(self):
        leg = kerbs_at_20()
        placed = place(leg, "left", Taper(60.0, 30.0, Narrowest(5.0), fill=True))
        assert placed.geometry.geom_type == "Polygon"
        assert placed.geometry.contains(Point(45.0, 19.0))
        minx, miny, maxx, maxy = placed.geometry.bounds
        assert miny >= 15.0 - 0.01 and maxy <= 20.0 + 0.01
        assert minx >= 30.0 - 0.01 and maxx <= 60.0 + 0.01

    def test_D26_at_puts_one_point_per_station_on_the_reference(self):
        placed = place(kerbs_at_20(), "left", At((50.0, 60.0, 70.0), Kerb(2.5)))
        assert placed.geometry.geom_type == "MultiPoint"
        np.testing.assert_allclose([(p.x, p.y) for p in placed.geometry.geoms],
                                   [(50, 17.5), (60, 17.5), (70, 17.5)], atol=0.01)
        assert placed.datum == {"traced": 1.0}

    def test_D27_a_glyph_is_drawn_where_its_reference_resolves(self):
        """`draw(leg, side, station_ft, offset_ft)` builds the symbol; place() only decides where."""
        def draw(leg, side, station_ft, offset_ft):
            return Point(station_ft, offset_ft).buffer(1.0)

        placed = place(kerbs_at_20(), "left", Glyph(80.0, Centre(5.0), draw))
        assert placed.geometry.centroid.x == pytest.approx(80.0, abs=0.01)
        assert placed.geometry.centroid.y == pytest.approx(5.0, abs=0.01)
        assert placed.datum == {"kerbs": 1.0}

    def test_D28_posts_painted_through_the_context_carry_their_datum(self):
        from src.geometry.markings import BOLLARD
        from src.geometry.treatments.state import DesignState as DS

        leg = kerbs_at_20()
        ctx = PaintContext(state=DS(legs={"t": leg}, corner_fillets={}), crosswalk_offsets={},
                           center_ft=None)
        posts = ctx.paint(BOLLARD, "t", "left", At((50.0, 60.0, 70.0), Kerb(2.5)))
        assert len(posts) == 3
        for post, station in zip(posts, (50.0, 60.0, 70.0)):
            assert post.kind is BOLLARD and post.datum == {"traced": 1.0}
            assert (post.geometry.centroid.x, post.geometry.centroid.y) == pytest.approx((station, 17.5), abs=0.01)


def test_D29_narrowest_finds_the_pinch_between_the_stations_asked_for():
    """The pinch vertex at station 100 lies between the stations sampled (95, 105, ...): the
    narrowest is still the kerb's own, 20 ft, not the 20.1 ft the samples happen to land on."""
    leg = straight()
    pinch = [(0, 22), (100, 20), (200, 22)]
    trace(leg, "left", pinch)
    trace(leg, "right", [(s, -o) for s, o in pinch])
    stations = np.arange(5.0, 200.0, 10.0)
    result = resolve(leg, "left", Narrowest(8.0), stations)
    np.testing.assert_allclose(result.offsets_ft, 12.0, atol=0.01)
