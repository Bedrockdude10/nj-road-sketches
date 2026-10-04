"""Getting the TRACED KERB out of OSM and into state-plane feet.

One world, one projection. Every traced kerb in the area is available; WHICH of them a consumer
wants is that consumer's geometric question of the layer - kerb along a leg, kerb in a junction's
corner returns, kerb in the camera's view - and none of them is a circle that decides what exists."""


import numpy as np
from shapely.geometry import LineString, Point

from src.render.coords import wgs84_to_state_plane
from src.geometry.model import (
    CURB_POINT_BEHIND_TOLERANCE_FT,
    CURB_POINT_MAX_WIDTH_RATIO,
    station_offset_many,
)

# A corner return belonging to THIS junction is within this of its node.
KERB_NEAR_JUNCTION_FT = 80
# How far outside a leg's plausible half-width band a traced vertex may sit and still count as
# that leg's kerb, for deciding whether a whole kerb WAY is relevant to this junction.
KERB_ALONG_LEG_TOLERANCE_FT = 8.0


def _runs_along_a_leg(line: LineString, legs: dict) -> bool:
    """Whether any vertex of `line` sits where one of these legs' kerbs would be.

    The test for "is this OUR kerb", as opposed to "is this near the middle of the junction".
    KERB_NEAR_JUNCTION_FT is the latter, and it is right for fitting a corner radius - a
    return belonging to this junction is within 80 ft of its centre, and the area otherwise
    holds neighbouring junctions' returns, which produced a nonsense 7.9-30.2 ft radius spread
    at Columbia & Princeton.

    It is wrong for building the CURB LINES, which want kerb anywhere along a leg - a kerb at
    station 100 of a 130 ft leg is 100 ft from the centre and fails the near test. Losing those
    ways is INVISIBLE, because curb_line_from_points extrapolates to the working length: the
    outer half of the leg is then drawn from a bearing instead of from tracing that existed.
    """
    for leg in legs.values():
        if leg.curb_to_curb_ft is None:
            continue
        half_ft = leg.curb_to_curb_ft / 2
        reach_ft = half_ft * CURB_POINT_MAX_WIDTH_RATIO + KERB_ALONG_LEG_TOLERANCE_FT
        # The area holds every kerb there is, so a way nowhere near this leg is rejected on its
        # bounding box before it is stationed. The margin is the whole band a vertex may sit in:
        # beside by `reach_ft`, and behind the leg's start by CURB_POINT_BEHIND_TOLERANCE_FT.
        margin = reach_ft + CURB_POINT_BEHIND_TOLERANCE_FT
        x0, y0, x1, y1 = leg.centerline.bounds
        lx0, ly0, lx1, ly1 = line.bounds
        if lx1 < x0 - margin or lx0 > x1 + margin or ly1 < y0 - margin or ly0 > y1 + margin:
            continue
        stations, offsets = station_offset_many(leg.centerline,
                                                np.asarray(line.coords, dtype=float))
        along = ((stations > -CURB_POINT_BEHIND_TOLERANCE_FT)
                 & (stations < leg.centerline.length))
        beside = np.abs(offsets) < reach_ft
        if (along & beside).any():
            return True
    return False


def kerb_lines_with_tags_ft(osm: dict, *, legs: dict | None = None, near: Point | None = None
                             ) -> list[tuple[LineString, dict, int | None]]:
    """[(LineString, tags, way id)] for the traced kerbs of an area that answer one relation to
    the network - geometry plus what OSM says about each (kerb=lowered, tactile_paving=yes).

    `osm` is an area's layers (`IntersectionModel.osm`). THREE questions, and the caller must
    say which one: "is this kerb ours" has three answers and they are not interchangeable.

      * neither `legs` nor `near` - EVERY traced kerb in the area. The DRAWING's question: what
        exists is the world, and a view crops it to its own extent. Nothing here is a circle.
      * `legs` - _runs_along_a_leg: kerb anywhere along a leg, however far out, which is what a
        curb LINE wants. Traced ways well out along a leg fail the near test.
      * `near` (a junction node) - kerb within KERB_NEAR_JUNCTION_FT of it. The right test for
        fitting a corner RADIUS and for measuring a width: a return belonging to this junction is
        close to it, and anything looser drags in the neighbouring junctions' returns.

    The wide set is deliberately NOT fed to the fit - admitting those ways reshuffles the vertex
    contest at the acute, partly traced junction and leaves louellen_st_west with fewer usable
    kerbs: more data in, less data used. So the fit runs on the near set, and
    _extend_curbs_with_far_tracing rebuilds the curb lines from the wide set afterwards, once
    the widths are settled and extra ways can only lengthen a curb, never redefine one.
    See tests/test_leg_frame.py.

    A renderer taking the near set draws a cross of asphalt floating on grass, because a filter
    written to keep a neighbouring junction out of a CIRCLE FIT then decides what the drawing
    contains - hence no default, and `near` is keyword-only so it cannot be passed by accident.
    """
    if legs and near is not None:
        raise ValueError("A kerb is selected along legs OR near a junction, not both.")

    def relevant(line: LineString) -> bool:
        if legs:
            return _runs_along_a_leg(line, legs)
        if near is not None:
            return line.distance(near) <= KERB_NEAR_JUNCTION_FT
        return True

    return [(line, tags, way_id) for line, tags, way_id in _projected_kerbs(osm)
            if relevant(line)]


# Every traced kerb way of one area, projected into state-plane feet once. The projection is a
# fact about the area's ways, and they are read four times over per model load (the radius fit,
# the width fit, the far-tracing pass, the plan view) plus once per scenario. Keyed on the layer
# list's identity and served only while it is still that list, so a re-pulled snapshot cannot be
# served a projection of the old one.
_PROJECTED_KERBS: dict[int, tuple] = {}


def _projected_kerbs(osm: dict) -> list[tuple]:
    """[(LineString in feet, tags, way id)] for every traced kerb WAY of the area.

    The id is carried because a marking this project BREAKS for a kerb has to be traceable to
    the kerb that broke it - see src/geometry/kerbs.py:KerbOpening.citation.

    Lone `barrier=kerb` NODES are dropped: they carry no arc to fit and no line to draw.
    """
    kerbs = osm["kerbs"]
    cached = _PROJECTED_KERBS.get(id(kerbs))
    if cached is not None and cached[0] is kerbs:
        return cached[1]
    projected = []
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords:
            continue
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        projected.append((LineString(zip(xs, ys)), kerb.get("tags", {}), kerb.get("id")))
    _PROJECTED_KERBS[id(kerbs)] = (kerbs, projected)
    return projected


def _kerb_lines_ft(osm: dict, center_ft: Point) -> list[LineString]:
    """Traced OSM kerb ways near this junction, in state-plane feet.

    The near set of kerb_lines_with_tags_ft without the tags.

    Only kerbs within KERB_NEAR_JUNCTION_FT of the junction node are kept: a corner return sits
    within a few tens of feet of the junction it belongs to, and the area also holds NEIGHBOURING
    junctions' returns, which get assigned to this junction's corners and scatter the fitted
    radius (a nonsense 7.9-30.2 ft spread at Columbia & Princeton).
    """
    return [line for line, *_ in kerb_lines_with_tags_ft(osm, near=center_ft)]
