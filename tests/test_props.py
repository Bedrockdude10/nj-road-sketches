"""Sign placement along a leg, at the one station that has no direction: the far end.

A leg is only as long as its kerb is traced (sites/README.md, working_length_ft), and an OSM
stop node is placed where the stop line really is - so the two disagree whenever a junction is
traced less far than it is signed. At Princeton Ave & E Prospect St the East Prospect legs are
drawn 30 and 35 ft while the stop nodes sit 31 and 33 ft out, which is the first time in this
project a sign has been asked for at or past a centerline's end.
"""
import ast
import math
import pathlib

import pytest
from shapely.geometry import LineString

from src.geometry.model import Leg
from src.render.props import _leg_sign_position_ft
from tests.conftest import needs_source_data


def a_leg(length_ft=30.0):
    """A straight 30 ft leg running due east, with no traced kerb - the shape of an East
    Prospect St approach at princeton_eprospect."""
    return Leg(name="short_leg", centerline=LineString([(0, 0), (length_ft, 0)]),
               curb_to_curb_ft=30.5, traced_sides=set())


@pytest.mark.parametrize("offset_ft", [30.0, 31.0, 33.0, 100.0])
def test_a_sign_at_or_past_the_end_of_a_leg_still_has_a_heading(offset_ft):
    """The bug: interpolate() clamps to the endpoint, so a sign at or past the end took its
    direction from a point minus itself - a zero vector, normalised to NaN. The prop was then
    emitted with a NaN heading, which is not caught by any scene invariant (they check
    position, not rotation) and reaches Blender as an unrenderable rotation.
    """
    pos, heading = _leg_sign_position_ft(a_leg(), offset_ft, side="left")
    assert pos is not None
    assert not math.isnan(heading), f"heading is NaN for a sign {offset_ft} ft along a 30 ft leg"
    assert all(not math.isnan(c) for c in pos), f"position is NaN at {offset_ft} ft"


def test_a_sign_past_the_end_faces_the_same_way_as_one_just_inside():
    """A leg does not change direction at its last foot, so neither should the sign."""
    _, h_inside = _leg_sign_position_ft(a_leg(), 20.0, side="left")
    _, h_end = _leg_sign_position_ft(a_leg(), 30.0, side="left")
    assert h_inside == pytest.approx(h_end, abs=1e-6)


@needs_source_data
def test_the_school_service_road_opens_the_kerb_it_meets():
    """A vehicle entrance 46 ft from the junction centre is an opening, not part of the junction.

    Hopewell Elementary's service road (OSM way 845227293, highway=service, maxspeed=5 mph) meets
    Princeton Ave 46 ft south of the junction centre. It was excluded by cross_streets.py's
    JUNCTION_OWN_REACH_FT test, which asked how far along the LEG the meeting point was rather
    than whether the WAY is one of this junction's own arms - so the school's entrance was
    swallowed by a filter meant for the four legs that meet at the middle, and the lane-narrowing
    hatching was painted straight across the drive children are dropped off in.

    Its dropped kerb cannot rescue it either: the two kerb=lowered ways at its mouth carry no
    `wheelchair` tag, which src/geometry/kerbs.py:opens_the_kerb deliberately reads as
    "unspecified, does not open".
    """
    from src.geometry.intersection import load_intersection_model
    from src.geometry.treatments import DesignState

    model = load_intersection_model(site="princeton_eprospect")
    state = DesignState.from_model(model)
    south = [o for (leg, _side), openings in state.kerb_openings.items()
             for o in openings if leg == "princeton_ave_south"]
    away_from_the_junction = [o for o in south if o.start_ft > 30]
    assert away_from_the_junction, (
        "no kerb opening on princeton_ave_south beyond the junction mouth - the school service "
        f"road produced none. Openings found: {south}")


def _prop_types_emitted() -> set[str]:
    """Every prop type src/render/props.py can put in an export, read out of the source.

    An AST scan and not a hardcoded list, for the reason the list it replaces existed: a
    literal enumerated by hand goes stale the moment somebody adds a builder, and it goes
    stale SILENTLY - which is exactly the drift this test is here to catch. Reads the
    `"type": "<literal>"` key of every dict displayed in the module, so a new emitter is
    picked up by writing it and nothing else.
    """
    tree = ast.parse(pathlib.Path("src/render/props.py").read_text())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "type"
                    and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                out.add(value.value)
    return out


def _plan_view_types() -> set[str]:
    """Every prop type the plan view draws: the PROP_MARKERS table, plus the types
    `_draw_props` handles in a branch of its own.

    The second half matters - a tactile paving pad is deliberately NOT a marker, it is drawn
    at its true size and orientation because whether it spills into the roadway is the thing
    the plan sheet exists to make checkable. A guard that only knew the marker table would
    report that pad as missing and push somebody to add a dot beside it.
    """
    from src.render.plan_view import PROP_MARKERS

    tree = ast.parse(pathlib.Path("src/render/plan_view.py").read_text())
    special = {node.comparators[0].value
               for node in ast.walk(tree)
               if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
               and node.left.id == "kind" and len(node.comparators) == 1
               and isinstance(node.comparators[0], ast.Constant)
               and isinstance(node.comparators[0].value, str)}
    return set(PROP_MARKERS) | special


def _blender_builder_types() -> set[str]:
    """Every prop type scripts/blender/blender_props.py:add_prop has a builder for - the
    `ptype == "<literal>"` arms of its dispatch. Scanned rather than imported because the
    module is Blender-only: it does `import bpy` at the top and cannot be imported here."""
    tree = ast.parse(pathlib.Path("scripts/blender/blender_props.py").read_text())
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "ptype" and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)):
            out.add(node.comparators[0].value)
    return out


def test_every_prop_type_is_drawn_in_both_views():
    """A prop the placement layer can emit must be drawn in BOTH the plan view and the 3D
    render. Nothing may exist in one view only.

    This is not tidiness. The two views are the same claim about the same street, and a prop
    in one and not the other is a disagreement with no way to tell which view is wrong - the
    reader sees a sign in the render, looks for it on the plan sheet, and finds nothing. It
    had already happened twice by the time this test was written, in both directions:
    `yield_sign` had a plan marker and no Blender builder, and `school_zone_sign` had a
    builder and no plan marker, falling through to EXTRA_PROP_MARKER and drawing as the
    generic "some other prop" triangle.

    EXTRA_PROP_MARKER stays, because a site config's `props.extra` may legitimately name a
    type this repo has never heard of. What it may not do is absorb a type src/ emits itself.
    """
    emitted = _prop_types_emitted()
    assert emitted, "the AST scan found no prop types at all - it has stopped seeing the source"
    missing_2d = sorted(emitted - _plan_view_types())
    missing_3d = sorted(emitted - _blender_builder_types())
    assert not missing_2d, (f"emitted but not in plan_view.PROP_MARKERS (drawn as the generic "
                            f"EXTRA_PROP_MARKER): {missing_2d}")
    assert not missing_3d, (f"emitted but no builder in blender_props.add_prop (not drawn in "
                            f"3D at all): {missing_3d}")


def test_every_prop_type_is_drawn_distinguishably():
    """No two prop types may share an identical plan-view marker style.

    Presence is not enough. A bicycle warning sign and an OSM RRFB beacon both went in as a
    small gold diamond, which on the sheet is one symbol with two meanings and two legend rows
    claiming it - the reader has no way to tell which is which, and neither has the author.
    """
    from src.render.plan_view import PROP_MARKERS

    seen: dict[tuple, str] = {}
    clashes = []
    for kind, styles in PROP_MARKERS.items():
        key = tuple(tuple(sorted(style.items())) for style in styles)
        if key in seen:
            clashes.append(f"{seen[key]} and {kind}")
        seen[key] = kind
    assert not clashes, f"prop types drawn identically in plan: {clashes}"
