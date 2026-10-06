"""A design written as an osmChange (src/geometry/treatments/propose.py) - from its numbers, not paint."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.model import Leg, station_offset_many
from src.geometry.targets import LegSide, LegTarget
from src.geometry.treatments import DesignState, LaneNarrowing, MarkedParking
from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane
from src.geometry.treatments.propose import (hatched_zone_ft, proposal_from_design, to_the_inch,
                                             way_pieces)
from tests.conftest import synthetic_leg

KERB_FT = 22.0
LENGTH_FT = 200.0
STRAIGHT = [(0.0, KERB_FT), (LENGTH_FT, KERB_FT)]
WAY = 7
BROAD = {"highway": "secondary", "name": "Broad Street"}


def _osm():
    return {"roads": [{"id": WAY, "node_ids": [1, 2, 3], "tags": dict(BROAD),
                       "coords_wgs84": [[-74.770, 40.39], [-74.769, 40.39], [-74.768, 40.39]]}]}


def _leg(name: str, aligned: bool) -> Leg:
    """A leg homed at junction node 2 of way 7: `west` runs against the way (2 -> 1), `east`
    with it (2 -> 3). Neither far end is a junction."""
    leg = synthetic_leg(name, LENGTH_FT, 2 * KERB_FT, STRAIGHT, STRAIGHT)
    if not aligned:
        leg = Leg(name, LineString([(0.0, 0.0), (-LENGTH_FT, 0.0)]), curb_to_curb_ft=2 * KERB_FT)
        leg.left_curb = LineString([(0.0, -KERB_FT), (-LENGTH_FT, -KERB_FT)])
        leg.right_curb = LineString([(0.0, KERB_FT), (-LENGTH_FT, KERB_FT)])
        leg.traced_sides = {"left", "right"}
    leg.start_node, leg.end_node, leg.osm_way_id = 2, None, WAY
    return leg


def _design(*extra):
    """Both legs carry the two-way track on the way's RIGHT: west's left, east's right."""
    legs = {"west": _leg("west", aligned=False), "east": _leg("east", aligned=True)}
    state = DesignState(legs=legs, corner_fillets={})
    with contextlib.redirect_stdout(io.StringIO()):
        state = state.apply(AddTwoWayBikeLane(LegSide("west", "left"), width_ft=10.0, buffer_ft=3.0),
                            AddTwoWayBikeLane(LegSide("east", "right"), width_ft=10.0,
                                              buffer_ft=3.0), *extra)
    model = SimpleNamespace(legs=state.legs, leg_osm_aligned={"west": False, "east": True},
                            leg_osm_tags={"west": dict(BROAD), "east": dict(BROAD)}, osm=_osm())
    return model, state


def test_the_hatched_zone_is_the_band_the_design_states():
    leg = synthetic_leg("east", LENGTH_FT, 2 * KERB_FT, STRAIGHT, STRAIGHT)
    state = DesignState(legs={"east": leg}, corner_fillets={})
    zone = LaneNarrowing(LegTarget("east"), lane_edge_ft=14.0, sides=("left",))
    polygon = hatched_zone_ft(state, "east", "left", zone)
    stations, offsets = station_offset_many(leg.centerline, np.asarray(polygon.exterior.coords))
    assert offsets.min() == pytest.approx(14.0, abs=0.01)
    assert offsets.max() == pytest.approx(KERB_FT, abs=0.01)
    assert stations.min() >= -0.01 and stations.max() <= LENGTH_FT + 0.01
    assert hatched_zone_ft(state, "east", "right", zone) is None
    line_only = LaneNarrowing(LegTarget("east"), lane_edge_ft=14.0, sides=("left",),
                              line_only=True)
    assert hatched_zone_ft(state, "east", "left", line_only) is None


def test_a_stall_is_written_to_the_inch_below():
    assert to_the_inch(7.69) == to_the_inch(7.70) == 92
    assert to_the_inch(8.0) == 96


def test_the_way_is_cut_only_where_the_answer_changes():
    nodes = [10, 11, 12, 13, 14, 15]
    spans = [(0, 1, "a"), (1, 2, "a"), (2, 4, "b"), (4, 5, "b")]
    assert way_pieces(nodes, spans) == [([10, 11, 12], "a"), ([12, 13, 14, 15], "b")]
    assert way_pieces(nodes, [(0, 2, "a"), (2, 5, "a")]) == [(nodes, "a")]


def test_the_way_splits_where_the_far_kerb_parking_starts():
    model, state = _design(MarkedParking(LegSide("east", "left"), depth_ft=7.69))
    change = proposal_from_design(model, state, _osm(), note="test")
    modify, = [w for w in change.ways if w.action == "modify"]
    create, = [w for w in change.ways if w.action == "create"]
    assert (modify.id, modify.node_ids) == (WAY, (1, 2))
    assert "parking:left" not in modify.tags
    assert create.id < 0 and create.node_ids == (2, 3)
    assert create.tags["parking:left"] == "lane"
    assert create.tags["parking:left:width"] == "7'8\""
    for way in (modify, create):
        assert way.tags["cycleway:right"] == "track"
        assert way.tags["note"]
    assert create.tags["name"] == "Broad Street"         # a created way carries its full tags


def test_a_hundredth_of_a_foot_does_not_split_a_way():
    model, state = _design(MarkedParking(LegSide("west", "right"), depth_ft=7.69),
                           MarkedParking(LegSide("east", "left"), depth_ft=7.70))
    change = proposal_from_design(model, state, _osm(), note="test")
    modify, = change.ways
    assert (modify.action, modify.node_ids) == ("modify", (1, 2, 3))
    assert modify.tags["parking:left:width"] == "7'8\""


def test_a_hatched_far_kerb_is_one_restriction_area_over_new_nodes():
    model, state = _design(LaneNarrowing(LegTarget("east"), lane_edge_ft=14.0, sides=("left",)))
    change = proposal_from_design(model, state, _osm(), note="test")
    area, = [w for w in change.ways if w.tags.get("road_marking") == "restriction"]
    assert area.action == "create" and area.node_ids[0] == area.node_ids[-1]
    created = {node.id for node in change.nodes}
    assert set(area.node_ids) <= created
    assert all(node.id < 0 for node in change.nodes)
    assert area.tags["note"] and area.tags["pattern"] == "chevron"


def test_the_proposal_is_never_read_off_paint():
    source = Path("src/geometry/treatments/propose.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not {m for m in imported
                if m.startswith("src.render") or m == "src.geometry.paint.context"}

