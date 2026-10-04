"""Assumptions that held for a 130-170 ft leg stub at one junction and broke on the Hopewell world.

The world is designed once, node to node: 92 legs, some of them 1,400 ft long, one of them a
closed loop, several sharing a street name. Each test is one of the ways that went wrong, built
from a few coordinates so none needs the licensed data.
"""
import contextlib
import io

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from src.geometry.model import Leg, station_offset_many
from src.geometry.model.leg_frame import offset_curb_line
from src.geometry.model.traced_kerbs import assign_curb_points_to_legs
from src.geometry.network.slice_design import _legs_of, _traced_widths


def _loop(radius_ft: float = 40.0, n: int = 19) -> LineString:
    """A closed loop wound counter-clockwise, starting at (radius, 0) with a kink where it closes."""
    angles = np.linspace(0.0, 2 * np.pi, n + 1)
    angles[-1] = 0.0
    return LineString([(radius_ft * np.cos(a), radius_ft * np.sin(a)) for a in angles])


def _streets(rows: list[tuple[str, LineString]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name_": [name for name, _ in rows]},
                            geometry=[line for _, line in rows])


# --- a street is several features ----------------------------------------------------------


def test_each_feature_of_a_street_gets_the_width_of_its_own_pavement():
    """Keyed by name, one polygon's area was divided by the length of EVERY feature of that name:
    Eaton Place's 6,350 sq ft over 1,133 ft of its four pieces came out 5.6 ft wide."""
    short = LineString([(0, 0), (100, 0)])
    long = LineString([(0, 500), (300, 500)])
    streets = _streets([("Eaton Place", short), ("Eaton Place", long)])
    paved = _streets([("Eaton Place", short.buffer(10, cap_style="flat")),
                      ("Eaton Place", long.buffer(14, cap_style="flat"))])
    widths = _traced_widths(streets, paved)
    assert widths[0] == pytest.approx(20.0, abs=0.01)
    assert widths[1] == pytest.approx(28.0, abs=0.01)


def test_a_loops_pavement_is_found_by_its_middle_not_its_centroid():
    """A loop's centroid is the empty ground it encloses, outside its own asphalt."""
    ring = _loop()
    band = Polygon(ring).buffer(13).difference(Polygon(ring).buffer(-13))
    streets = _streets([("Eaton Place", LineString([(300, 0), (400, 0)])), ("Eaton Place", ring)])
    paved = _streets([("Eaton Place", LineString([(300, 0), (400, 0)]).buffer(13, cap_style="flat")),
                      ("Eaton Place", band)])
    widths = _traced_widths(streets, paved)
    assert widths[1] == pytest.approx(band.area / ring.length)
    assert widths[1] == pytest.approx(26.0, abs=0.5)


def test_same_named_features_become_distinct_legs():
    """Each feature restarted its numbering at the bare slug and overwrote the one before:
    99 approaches went in and 92 legs came out, the seven that vanished being whole streets."""
    rows = [("Prospect Street", LineString([(0, 0), (100, 0)])),
            ("Prospect Street", LineString([(0, 200), (100, 200)])),
            ("Prospect Street", LineString([(0, 400), (100, 400)]))]
    street_of: dict[str, str] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        legs = _legs_of(_streets(rows), {}, [], street_of)
    assert sorted(legs) == ["prospect_street", "prospect_street_1", "prospect_street_2"]
    assert {street_of[key] for key in legs} == {"Prospect Street"}
    assert sorted(leg.centerline.coords[0][1] for leg in legs.values()) == [0, 200, 400]


# --- a leg that closes on itself -----------------------------------------------------------


@pytest.mark.parametrize("sign", [1, -1])
def test_the_curbs_of_a_loop_are_closed_rings_inside_the_loops_own_stations(sign):
    """offset_curve treats a ring as an open line, so one of its curbs came back with a stub
    across the centreline and stations out to 287 ft on a 253 ft street."""
    ring = _loop() if sign > 0 else LineString(list(_loop().coords)[::-1])
    leg = Leg("loop", ring, 25.0)
    for side, expected_sign in (("left", 1), ("right", -1)):
        curb = getattr(leg, f"{side}_curb")
        assert curb.is_closed and curb.is_valid
        stations, offsets = station_offset_many(ring, np.asarray(curb.coords))
        assert stations.min() >= -1.0 and stations.max() <= ring.length + 1.0
        assert np.allclose(np.abs(offsets), 12.5, atol=0.8), (side, offsets.min(), offsets.max())
        assert np.all(np.sign(offsets) == expected_sign)
        assert Point(curb.coords[0]).distance(Point(ring.coords[0])) < 13.0


def test_a_loop_has_no_terminal_rays():
    """A point on the loop's outer kerb beside the closing kink was 2.6 ft from the end ray and
    12.5 ft from the street, so it was read as station 284 on a 253 ft loop."""
    ring = _loop()
    outside_the_kink = np.array([[48.0, 13.0]])
    stations, offsets = station_offset_many(ring, outside_the_kink)
    assert 0.0 <= stations[0] <= ring.length
    assert abs(abs(offsets[0]) - np.hypot(48.0 - 40.0, 13.0)) < 6.0

    straight = LineString([(0, 0), (100, 0)])
    assert station_offset_many(straight, np.array([[130.0, 5.0]]))[0][0] == pytest.approx(130.0)


def test_an_open_leg_still_has_its_rays():
    leg = LineString([(0, 0), (100, 0)])
    stations, _ = station_offset_many(leg, np.array([[-20.0, 3.0], [140.0, 3.0]]))
    assert stations[0] == pytest.approx(-20.0) and stations[1] == pytest.approx(140.0)


def test_offset_curb_line_on_an_open_line_is_unchanged():
    line = LineString([(0, 0), (100, 0)])
    assert list(offset_curb_line(line, 10.0).coords) == [(0.0, 10.0), (100.0, 10.0)]


# --- a leg that ends at the next junction --------------------------------------------------


def _long_leg_and_kerb_far_down_its_extension():
    leg = Leg("a", LineString([(0, 0), (200, 0)]), 30.0)
    # A kerb vertex 3,000 ft along the street's own extension, 15 ft out: exactly where the next
    # leg's street runs, and where the terminal ray of this one still reaches.
    near = LineString([(20, 15), (190, 15)])
    far = LineString([(2900, 15), (3100, 15)])
    return {"a": leg}, [near, far]


def test_a_stub_leg_still_claims_kerb_beyond_its_working_length():
    legs, lines = _long_leg_and_kerb_far_down_its_extension()
    claimed = assign_curb_points_to_legs(legs, lines)["a"]["left"]
    assert max(station for station, _ in claimed) > 2000.0


def test_a_node_to_node_leg_claims_nothing_past_its_far_end():
    """broad_street_1 (1,464.6 ft) claimed vertices out to station 4,829 and drew a 4,836 ft kerb
    whose narrowest half-width was 1.85 ft."""
    legs, lines = _long_leg_and_kerb_far_down_its_extension()
    claimed = assign_curb_points_to_legs(legs, lines, bounded=True)["a"]["left"]
    assert max(station for station, _ in claimed) <= 203.0
    assert len(claimed) == 2


# --- kerbside paint needs room by BOTH datums ----------------------------------------------


def test_a_leg_whose_nominal_lane_edge_is_inside_the_lane_is_left_unpainted():
    """Seminary Ave is 29.7 ft between its traced kerbs for 117 ft and 21.1 ft nominal over its
    414, so the traced kerb spares 3.9 ft while the nominal lane edge sits -0.43 ft out and the
    hatch was handed a negative width."""
    from src.geometry.treatments import DesignState
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.parking import MarkedParking, apply_osm_parking

    leg = Leg("seminary", LineString([(0, 0), (400, 0)]), 21.15)
    leg.left_curb = LineString([(0, 14.7), (400, 14.7)])
    leg.right_curb = LineString([(0, -14.9), (400, -14.9)])
    leg.traced_sides.update({"left", "right"})
    state = DesignState(legs={"seminary": leg}, corner_fillets={})
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        out = apply_osm_parking(state, model=None)
    assert not out.treatments_of(LaneNarrowing) and not out.treatments_of(MarkedParking)
    assert "no kerbside paint is marked" in printed.getvalue()
