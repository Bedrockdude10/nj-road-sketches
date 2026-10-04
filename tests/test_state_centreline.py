"""Tests for registered state centrelines (paint spec v2, Phase 1)."""
import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.intersection.state_centreline import register_line, registered_state_centrelines, attach_state_centrelines
from src.geometry.model.leg_frame import Leg
from tests.conftest import needs_source_data


# Kerbs for synthetic tests
K_LEFT = LineString([(0, 14), (200, 14)])
K_RIGHT = LineString([(0, -26), (200, -26)])
K = [K_LEFT, K_RIGHT]  # midpoint y = -6


class TestRegisterLine:
    """Tests S1-S6: register_line with synthetic geometry."""

    def test_S1_straight_line_registers_to_kerb_midpoint(self):
        """S1: register_line(LineString([(0,0),(200,0)]), K) moves every vertex to y = -6."""
        line = LineString([(0, 0), (200, 0)])
        L = line.length
        result = register_line(line, K)
        coords = np.asarray(result.coords)
        np.testing.assert_allclose(coords[:, 1], -6.0, atol=0.01)
        np.testing.assert_allclose(coords[:, 0], np.append(np.arange(0.0, L, 10.0), L), atol=0.01)

    def test_S2_reversed_line_registers_to_same_midpoint(self):
        """S2: register_line with reversed input [(200,0),(0,0)]."""
        line = LineString([(200, 0), (0, 0)])
        result = register_line(line, K)
        coords = np.asarray(result.coords)
        np.testing.assert_allclose(coords[:, 1], -6.0, atol=0.01)

    def test_S3_registration_holds_past_kerb_span(self):
        """S3: kerbs only x 0..100; line extends to 300; output held at -6.0 past x=100."""
        line = LineString([(0, 0), (300, 0)])
        K_short = [LineString([(0, 14), (100, 14)]), LineString([(0, -26), (100, -26)])]
        result = register_line(line, K_short)
        coords = np.asarray(result.coords)
        np.testing.assert_allclose(coords[:, 1], -6.0, atol=0.01)

    def test_S4_empty_kerbs_returns_identical_object(self):
        """S4: register_line(line, []) returns the same object."""
        line = LineString([(0, 0), (200, 0)])
        result = register_line(line, [])
        assert result is line

    def test_S5_single_kerb_returns_identical_object(self):
        """S5: only the +14 kerb; no opposite side to mirror."""
        line = LineString([(0, 0), (200, 0)])
        result = register_line(line, [K_LEFT])
        assert result is line

    def test_S6_kerbs_beyond_search_distance_return_identical_object(self):
        """S6: kerbs at y = ±50 ft (beyond 40 ft search distance)."""
        line = LineString([(0, 0), (200, 0)])
        far_kerbs = [LineString([(0, 50), (200, 50)]), LineString([(0, -50), (200, -50)])]
        result = register_line(line, far_kerbs)
        assert result is line


class TestAttachStateCentrelines:
    """Tests S7-S9: attach_state_centrelines with monkeypatched registry."""

    def test_S7_attach_state_centrelines_clips_and_sets_leg_state_centreline(self, monkeypatch):
        """S7: attach_state_centrelines with registered line; leg.state_centreline is set."""
        # Monkeypatch registered_state_centrelines to return a single registered line
        registered_line = LineString([(-50, -6), (250, -6)])
        monkeypatch.setattr(
            "src.geometry.intersection.state_centreline.registered_state_centrelines",
            lambda area: {"X": registered_line}
        )

        leg = Leg("test", LineString([(0, 0), (200, 0)]), curb_to_curb_ft=40.0)
        legs = {"test": leg}

        attach_state_centrelines(legs, "hopewell_borough")

        assert leg.state_centreline is not None
        assert abs(leg.state_centreline.length - 200.0) < 0.1
        coords = np.asarray(leg.state_centreline.coords)
        np.testing.assert_allclose(coords[:, 1], -6.0, atol=0.01)

    def test_S8_attach_returns_none_when_registered_line_too_short(self, monkeypatch):
        """S8: registered line runs only x 0..100 (< 0.8 * leg length); state_centreline is None."""
        short_line = LineString([(0, -6), (100, -6)])
        monkeypatch.setattr(
            "src.geometry.intersection.state_centreline.registered_state_centrelines",
            lambda area: {"X": short_line}
        )

        leg = Leg("test", LineString([(0, 0), (200, 0)]), curb_to_curb_ft=40.0)
        legs = {"test": leg}

        attach_state_centrelines(legs, "hopewell_borough")

        assert leg.state_centreline is None

    def test_S9_registered_state_centrelines_returns_empty_dict_on_FileNotFoundError(self, monkeypatch):
        """S9: load_road_network raises FileNotFoundError; cache is cleared before and after."""
        def raiser(*args, **kwargs):
            raise FileNotFoundError("data/ missing")

        monkeypatch.setattr(
            "src.geometry.intersection.state_centreline.load_road_network",
            raiser
        )

        # Clear cache before and after as per spec
        registered_state_centrelines.cache_clear()
        result = registered_state_centrelines("hopewell_borough")
        registered_state_centrelines.cache_clear()

        assert result == {}


@needs_source_data
def test_S10_load_intersection_model_sets_state_centrelines():
    """S10: load_intersection_model(site='broad_st_greenwood') sets state_centreline on real legs."""
    from src.geometry.intersection.load import load_intersection_model
    from src.geometry.treatments import DesignState

    model = load_intersection_model(site="broad_st_greenwood")

    # Check that broad_st_west and broad_st_east have state_centreline set
    assert model.legs["broad_st_west"].state_centreline is not None
    assert model.legs["broad_st_east"].state_centreline is not None

    # Also check via DesignState
    state = DesignState.from_model(model)
    assert state.legs["broad_st_west"].state_centreline is not None
    assert state.legs["broad_st_east"].state_centreline is not None
