"""Tests for paint datum computation (paint spec v2, Phase 1)."""
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from src.geometry.paint.datum import Kerb, Centre, KerbToKerb, kerb_profile, centre_profile, place
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
        """D3: traced left [(50,20),(150,20)] on 60 ft width; offsets taper from 30 to 20 to 30."""
        leg = straight(60.0)
        trace(leg, "left", [(50, 20), (150, 20)])

        result = kerb_profile(leg, "left", S)
        expected = [30, 28, 26, 24, 22] + [20] * 11 + [22, 24, 26, 28, 30]
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source) == (["nominal"] * 5 + ["traced"] * 11 + ["nominal"] * 5)

    def test_D4_mirrored_kerb_from_opposite_side(self):
        """D4: same as D3 on right; source is 'mirrored' where traced left."""
        leg = straight(60.0)
        trace(leg, "left", [(50, 20), (150, 20)])

        result = kerb_profile(leg, "right", S)
        expected = [-30, -28, -26, -24, -22] + [-20] * 11 + [-22, -24, -26, -28, -30]
        np.testing.assert_allclose(result.offsets_ft, expected, atol=0.01)
        assert list(result.source) == (["nominal"] * 5 + ["mirrored"] * 11 + ["nominal"] * 5)

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

        placed_c0 = place(leg, "left", S, Centre(0))
        placed_c11 = place(leg, "left", S, Centre(11))

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

        placed = place(leg, "left", S, Kerb(0), Centre(11))

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

        placed = place(leg, "left", S, Kerb(0), Centre(11))

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

        S_short = np.array([90.0, 100.0, 110.0])
        placed = place(leg, "left", S_short, KerbToKerb())

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

        placed = place(leg, "left", S, Kerb(0), Centre(11))

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
        result = ctx.paint(BUFFER_EDGE_LINE, "test", "left", (20, 60), Kerb(0), Centre(11))

        assert len(result) > 0
        assert all(isinstance(p, PaintPiece) for p in result)
        assert all(p.datum for p in result)  # non-empty datum
        assert all(abs(sum(p.datum.values()) - 1.0) < 1e-9 for p in result)
