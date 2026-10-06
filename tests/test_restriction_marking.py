"""road_marking=restriction areas: which kerb they belong to, and how they are painted."""
from types import SimpleNamespace

import numpy as np
import pyproj
import pytest
from shapely.geometry import LineString
from shapely.ops import unary_union

from src.checks import MarkingsDoNotCollide, PaintInsideTheCurb, SceneContext
from src.geometry.markings import LANE_EDGE_LINE, LANE_NARROWING_FILL
from src.geometry.model import NJ_STATE_PLANE_FT, WGS84, Leg, station_offset_many
from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT as W
from src.geometry.targets import LegSide
from src.geometry.treatments import (DesignState, RestrictionMarking, apply_osm_road_markings,
                                     restriction_painted_ft)
from tests.test_paint_on_the_kerb import KERB_FT, LEG, build, street

_TO_WGS84 = pyproj.Transformer.from_crs(NJ_STATE_PLANE_FT, WGS84, always_xy=True)
HATCH = {"road_marking": "restriction", "pattern": "chevron"}
INNER_FT, START_FT, END_FT = 14.0, 60.0, 240.0


def _ring(x0, x1, y0, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))


def test_a_restriction_area_takes_the_kerb_it_lies_along():
    x0, y0 = 417000.0, 565000.0
    leg = Leg(name="east", centerline=LineString([(x0, y0), (x0 + 120, y0)]), curb_to_curb_ft=30.0)
    state = DesignState(legs={"east": leg}, corner_fillets={})
    ring = _ring(x0 + 20, x0 + 100, y0 + 9, y0 + 15)
    area = {"coords_wgs84": [list(_TO_WGS84.transform(x, y)) for x, y in ring], "tags": HATCH,
            "id": -1}
    state = apply_osm_road_markings(state, SimpleNamespace(osm={"road_markings": [area]}))
    assert state.treatment_for(RestrictionMarking, LegSide("east", "left")) is not None
    assert state.treatment_for(RestrictionMarking, LegSide("east", "right")) is None
    # Half the nominal 30 ft less the 9 ft the area reaches in to.
    assert restriction_painted_ft(state, "east", "left") == pytest.approx(6.0, abs=0.01)
    assert restriction_painted_ft(state, "east", "right") == 0.0


def _painted():
    leg = street(2 * KERB_FT)
    marking = RestrictionMarking(LegSide(LEG, "left"),
                                 rings_ft=(_ring(START_FT, END_FT, INNER_FT, KERB_FT),),
                                 osm_ids=(-1,))
    return leg, *build(leg, [marking])


def test_the_edge_line_faces_the_road_and_none_runs_along_the_kerb():
    leg, _state, paint = _painted()
    lines = [p.geometry for p in paint if p.kind == LANE_EDGE_LINE]
    assert lines
    assert min(line.distance(leg.left_curb) for line in lines) > W / 2
    inner = unary_union(lines).intersection(
        LineString([(START_FT, INNER_FT + W / 2), (END_FT, INNER_FT + W / 2)]).buffer(0.02))
    assert inner.length >= (END_FT - START_FT - W) - 0.1


def test_the_fill_reaches_the_kerb():
    leg, _state, paint = _painted()
    fills = [p.geometry for p in paint if p.kind == LANE_NARROWING_FILL]
    assert fills
    offsets = np.concatenate([station_offset_many(leg.centerline,
                                                  np.asarray(f.exterior.coords))[1]
                              for f in fills])
    assert offsets.max() == pytest.approx(KERB_FT, abs=0.05)


def test_a_restriction_area_draws_nothing_the_checks_refuse():
    _leg, state, paint = _painted()
    context = SceneContext(state=state, paint=tuple(paint))
    assert not PaintInsideTheCurb().run(context)
    assert not MarkingsDoNotCollide().run(context)
