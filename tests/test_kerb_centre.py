"""The drawn centreline is the middle of a street's mapped kerbs, not where its way was digitised:
src/osm_osc.py:kerb_centre_offsets measures it, src/osm_world.py draws every way of the street
from it.

Fixtures are straight ways running due east, so left of a way is north and an offset is the drawn
y coordinate.
"""
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.osm_osc import kerb_centre_offsets
from src.osm_world import LocalFrame, _Reader
from src.sources.osm_context import SNAPSHOT_AREAS

AREA = "hopewell_borough"
FRAME = LocalFrame(SNAPSHOT_AREAS[AREA])
STREET = {"highway": "residential", "name": "Test Avenue"}


def _coords(*xy):
    return [list(FRAME.wgs84(x, y)) for x, y in xy]


def _way(way_id: int, y: float, x0: float, x1: float, tags: dict, nodes: list[int]) -> dict:
    return {"id": way_id, "node_ids": nodes, "tags": tags, "coords_wgs84": _coords((x0, y), (x1, y))}


def _kerb(way_id: int, y: float, x0: float = -40.0, x1: float = 40.0) -> dict:
    return {"id": way_id, "node_ids": [way_id * 10, way_id * 10 + 1],
            "tags": {"barrier": "kerb", "kerb": "raised"}, "coords_wgs84": _coords((x0, y), (x1, y))}


def _layers(roads: list[dict], kerbs: list[dict]) -> dict:
    return {"roads": roads, "kerbs": kerbs, "crossings": []}


def test_the_offset_is_where_the_kerbs_middle_lies_from_the_way():
    # kerbs 12.3 m left and 9.2 m right: their middle is 1.55 m left of the way
    layers = _layers([_way(1, 0.0, -50, 50, STREET, [1, 2])], [_kerb(91, 12.3), _kerb(92, -9.2)])
    assert kerb_centre_offsets(layers, FRAME, SNAPSHOT_AREAS[AREA])[1] == pytest.approx(1.55, abs=0.02)


def test_a_street_with_kerbs_on_one_side_only_is_not_moved():
    layers = _layers([_way(1, 0.0, -50, 50, STREET, [1, 2])], [_kerb(91, 12.3)])
    assert 1 not in kerb_centre_offsets(layers, FRAME, SNAPSHOT_AREAS[AREA])


def test_every_piece_of_a_street_takes_the_one_offset_its_kerbs_give():
    # Two pieces of one street; only the first has kerbs. The second moves with it, so the
    # street does not step where the pieces meet.
    roads = [_way(1, 0.0, -100, 0, STREET, [1, 2]), _way(2, 0.0, 0, 100, STREET, [2, 3])]
    layers = _layers(roads, [_kerb(91, 12.0, -90, -10), _kerb(92, -8.0, -90, -10)])
    offsets = kerb_centre_offsets(layers, FRAME, SNAPSHOT_AREAS[AREA])
    assert offsets[1] == pytest.approx(2.0, abs=0.02)
    assert offsets[2] == pytest.approx(2.0, abs=0.02)


def test_the_offset_is_the_median_over_the_kerb_pairs():
    # 60 m of kerbs whose middle is 1.0 m left, 20 m whose middle is 3.0 m left: the median is 1.0
    kerbs = [_kerb(91, 11.0, -40, 20), _kerb(92, -9.0, -40, 20),
             _kerb(93, 13.0, 20, 40), _kerb(94, -7.0, 20, 40)]
    layers = _layers([_way(1, 0.0, -50, 50, STREET, [1, 2])], kerbs)
    assert kerb_centre_offsets(layers, FRAME, SNAPSHOT_AREAS[AREA])[1] == pytest.approx(1.0, abs=0.02)


def test_another_streets_kerbs_do_not_move_it():
    # A parallel street 30 m north owns the kerbs beside it, so this street measures none.
    roads = [_way(1, 0.0, -50, 50, STREET, [1, 2]),
             _way(2, 30.0, -50, 50, {"highway": "residential", "name": "Other Street"}, [3, 4])]
    layers = _layers(roads, [_kerb(91, 34.0), _kerb(92, 26.0)])
    offsets = kerb_centre_offsets(layers, FRAME, SNAPSHOT_AREAS[AREA])
    assert 1 not in offsets
    assert offsets[2] == pytest.approx(0.0, abs=0.02)


def test_the_reader_draws_the_street_from_the_shifted_centreline():
    way = _way(1, 0.0, -50, 50, {**STREET, "width": "21", "oneway": "yes", "lanes": "2"}, [1, 2])
    reader = _Reader(FRAME, {}, centre_offsets={1: 1.5})
    reader.index_ends([way])
    reader.road(way)
    surface = unary_union([Polygon(ring) for ring in reader.out["pavement"]])
    assert (surface.bounds[1], surface.bounds[3]) == pytest.approx((1.5 - 10.5, 1.5 + 10.5), abs=0.01)
    # the lane line splits the shifted travel way
    ys = {round(sum(p[1] for p in line) / len(line), 2) for line in reader.out["lane_lines"]}
    assert ys == {1.5}


def test_the_reader_leaves_an_unmeasured_street_where_osm_has_it():
    way = _way(1, 0.0, -50, 50, {**STREET, "width": "10"}, [1, 2])
    reader = _Reader(FRAME, {})
    reader.index_ends([way])
    reader.road(way)
    surface = unary_union([Polygon(ring) for ring in reader.out["pavement"]])
    assert (surface.bounds[1], surface.bounds[3]) == pytest.approx((-5.0, 5.0), abs=0.01)
