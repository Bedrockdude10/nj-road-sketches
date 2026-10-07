"""A way's cross-section as src/osm_world.py reads it from tags: its default width, its lane lines
and where each `parking:<side>` position puts the parking.

Every fixture is one straight way running due east through the area's centre, so the left of the
way is north and a signed offset from the way is simply the drawn y coordinate.
"""
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.osm_world import (DASH_M, DEFAULT_WIDTHS_M, LocalFrame, _Reader, carriageway_width_m,
                           parking_depth_m)
from src.sources.osm_context import SNAPSHOT_AREAS

AREA = "hopewell_borough"
LENGTH_M = 100.0
LANE, BIKE, PARK = DEFAULT_WIDTHS_M["lane"], DEFAULT_WIDTHS_M["cycleway"], DEFAULT_WIDTHS_M["parking"]
ONEWAY = {"highway": "trunk", "oneway": "yes", "name": "Test Avenue"}
TWO_WAY = {"highway": "secondary", "name": "Test Avenue"}


def _draw(tags: dict) -> _Reader:
    frame = LocalFrame(SNAPSHOT_AREAS[AREA])
    way = {"id": 1, "node_ids": [1, 2], "tags": tags,
           "coords_wgs84": [list(frame.wgs84(-LENGTH_M / 2, 0.0)), list(frame.wgs84(LENGTH_M / 2, 0.0))]}
    reader = _Reader(frame, {})
    reader.index_ends([way])
    reader.road(way)
    return reader


def _offsets(lines: list) -> list[float]:
    """Each drawn line's offset from the way (its mean y), rounded, sorted, deduplicated."""
    return sorted({round(sum(p[1] for p in line) / len(line), 2) for line in lines})


def _y_span(rings: list) -> tuple[float, float]:
    shape = unary_union([Polygon(ring) for ring in rings])
    return round(shape.bounds[1], 2), round(shape.bounds[3], 2)


# --- the carriageway's width where OSM gives none: the section it is tagged with, summed ----------

@pytest.mark.parametrize("extra, expected", [
    ({}, 2 * LANE),
    ({"cycleway:right": "lane"}, 2 * LANE + BIKE),
    ({"cycleway:right": "lane", "cycleway:right:width": "2"}, 2 * LANE + 2.0),
    ({"cycleway:both": "lane", "cycleway:left:buffer": "0.5"}, 2 * LANE + 2 * BIKE + 0.5),
    ({"parking:both": "lane"}, 2 * LANE + 2 * PARK),
    ({"parking:right": "lane", "parking:right:orientation": "diagonal"},
     2 * LANE + parking_depth_m({"parking:right:orientation": "diagonal"}, "right")),
    ({"shoulder:left": "yes", "shoulder:left:width": "1.2"}, 2 * LANE + 1.2),
    # Street parking off the carriageway (wiki Street_parking: `width` never includes it)...
    ({"parking:both": "on_kerb"}, 2 * LANE),
    ({"parking:both": "street_side", "parking:both:orientation": "diagonal"}, 2 * LANE),
    # ...and parking spanning the kerb is half on it.
    ({"parking:right": "half_on_kerb"}, 2 * LANE + PARK / 2),
])
def test_an_untagged_width_is_the_tagged_section_summed(extra, expected):
    width, source = carriageway_width_m({**ONEWAY, "lanes": "2", **extra})
    assert width == pytest.approx(expected)
    assert source.startswith("DEFAULTED")


def test_a_tagged_width_wins_over_the_section():
    assert carriageway_width_m({**ONEWAY, "lanes": "2", "cycleway:right": "lane", "width": "9"}) \
        == (pytest.approx(9.0), "from OSM")


def test_a_service_way_keeps_its_own_default():
    assert carriageway_width_m({"highway": "service", "cycleway:right": "lane"})[0] \
        == pytest.approx(DEFAULT_WIDTHS_M["service"])


# --- lane lines: between adjacent lanes running the same way (MUTCD 3B.06, STANDARDS.md) --------

def test_a_oneway_two_lane_street_with_a_bike_lane_gets_one_lane_line_between_its_lanes():
    out = _draw({**ONEWAY, "lanes": "2", "cycleway:right": "lane"}).out
    half = (2 * LANE + BIKE) / 2
    travel_left, travel_right = half, -half + BIKE
    assert _offsets(out["lane_lines"]) == [round((travel_left + travel_right) / 2, 2)]
    # and the travel lanes are full lanes: the bike lane did not come out of them
    assert travel_left - travel_right == pytest.approx(2 * LANE)


def test_a_lane_line_is_broken_by_default():
    out = _draw({**ONEWAY, "lanes": "2"}).out
    lengths = [LineString(line).length for line in out["lane_lines"]]
    assert len(lengths) > 1
    assert max(lengths) == pytest.approx(DASH_M, abs=0.05)


def test_three_oneway_lanes_get_two_lane_lines():
    out = _draw({**ONEWAY, "lanes": "3"}).out
    assert _offsets(out["lane_lines"]) == [-1.5, 1.5]


@pytest.mark.parametrize("tags", [{**ONEWAY, "lanes": "1"}, {**ONEWAY}, {**TWO_WAY, "lanes": "2"}])
def test_no_two_lanes_running_the_same_way_means_no_lane_line(tags):
    assert _draw(tags).out["lane_lines"] == []


def test_a_four_lane_two_way_street_gets_a_lane_line_each_side_of_its_centre():
    out = _draw({**TWO_WAY, "lanes": "4"}).out
    assert _offsets(out["lane_lines"]) == [-3.0, 3.0]


def test_lanes_forward_and_backward_split_a_two_way_street():
    # 1 lane backward on the left half (0..4.5), 2 forward on the right half (-4.5..0)
    out = _draw({**TWO_WAY, "lanes": "3", "lanes:forward": "2", "lanes:backward": "1"}).out
    assert _offsets(out["lane_lines"]) == [-2.25]


@pytest.mark.parametrize("change", ["no|yes", "yes|no", "not_right|yes", "yes|not_left"])
def test_a_lane_change_restriction_makes_the_line_solid(change):
    out = _draw({**ONEWAY, "lanes": "2", "change:lanes": change}).out
    assert len(out["lane_lines"]) == 1
    assert LineString(out["lane_lines"][0]).length == pytest.approx(LENGTH_M, abs=0.1)


@pytest.mark.parametrize("change", ["yes|yes", "not_left|not_right", "yes|yes|yes"])
def test_a_lane_change_allowed_across_the_line_leaves_it_broken(change):
    # not_left on the leftmost lane and not_right on the rightmost restrict nothing between them;
    # a list that is not one value per lane is not read.
    out = _draw({**ONEWAY, "lanes": "2", "change:lanes": change}).out
    assert len(out["lane_lines"]) > 1


def test_width_lanes_places_the_lane_line_where_the_lanes_meet():
    # 8 m, bike lane on the right: right edge -4, bike lane to -2.5, then right to left 3.0 and 3.5
    out = _draw({**ONEWAY, "lanes": "2", "width": "8", "width:lanes": "3.5|3.0",
                 "cycleway:right": "lane"}).out
    assert _offsets(out["lane_lines"]) == [0.5]


def test_width_lanes_on_a_two_way_street_skips_the_centre():
    out = _draw({**TWO_WAY, "lanes": "4", "width": "12", "width:lanes": "3|3|3|3"}).out
    assert _offsets(out["lane_lines"]) == [-3.0, 3.0]


# --- street parking, by its position (wiki Street_parking, "Parking position") -----------------

def test_lane_parking_is_still_inside_the_carriageway():
    out = _draw({**ONEWAY, "lanes": "1", "width": "8", "parking:right": "lane"}).out
    assert _offsets(out["parking_edge_lines"]) == [round(-4 + PARK, 2)]
    assert out["sidewalks"] == [] and out["paved_surfaces"] == []


def test_on_kerb_parking_is_on_the_pavement_beyond_the_kerb():
    out = _draw({**ONEWAY, "lanes": "1", "width": "6", "parking:right": "on_kerb"}).out
    assert _y_span(out["sidewalks"]) == (round(-3 - PARK, 2), -3.0)
    assert out["parking_edge_lines"] == []


def test_diagonal_on_kerb_parking_is_as_deep_as_a_diagonal_bay():
    tags = {**ONEWAY, "lanes": "1", "width": "6", "parking:both": "on_kerb",
            "parking:both:orientation": "diagonal"}
    depth = parking_depth_m(tags, "left")
    out = _draw(tags).out
    shape = unary_union([Polygon(ring) for ring in out["sidewalks"]])
    assert round(shape.bounds[1], 2) == round(-3 - depth, 2)
    assert round(shape.bounds[3], 2) == round(3 + depth, 2)


def test_street_side_parking_is_paved_ground_beside_the_carriageway():
    tags = {**ONEWAY, "lanes": "1", "width": "6", "parking:left": "street_side",
            "parking:left:orientation": "diagonal"}
    out = _draw(tags).out
    assert _y_span(out["paved_surfaces"]) == (3.0, round(3 + parking_depth_m(tags, "left"), 2))
    assert out["sidewalks"] == []


def test_half_on_kerb_parking_spans_the_kerb_and_narrows_the_travel_way():
    out = _draw({**ONEWAY, "lanes": "2", "width": "8", "parking:right": "half_on_kerb"}).out
    assert _y_span(out["sidewalks"]) == (round(-4 - PARK / 2, 2), -4.0)
    # the travel way runs from 4 down to -4 + PARK/2, and its one lane line splits it
    assert _offsets(out["lane_lines"]) == [round((4 + (-4 + PARK / 2)) / 2, 2)]


def test_lane_markings_no_means_no_lane_line():
    assert _draw({**ONEWAY, "lanes": "2", "lane_markings": "no"}).out["lane_lines"] == []
