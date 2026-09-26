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
    """No two prop types may share an identical plan-view footprint AND colouring.

    Presence is not enough. A bicycle warning sign and an OSM RRFB beacon both went in as a
    small gold diamond, which on the sheet is one symbol with two meanings and two legend rows
    claiming it - the reader has no way to tell which is which, and neither has the author.

    Now that a prop is drawn at its real size, two of them genuinely ARE the same object: the
    NO TURN ON RED plate and the R9-23 turn-box plate are one builder in 3D. Colour is the only
    honest thing left to tell those apart, which is why the key is footprint AND style.
    """
    from src.render.plan_view import PROP_MARKERS

    seen: dict[tuple, str] = {}
    clashes = []
    for kind, pieces in PROP_MARKERS.items():
        key = tuple((footprint, tuple(sorted(style.items()))) for footprint, style in pieces)
        if key in seen:
            clashes.append(f"{seen[key]} and {kind}")
        seen[key] = kind
    assert not clashes, f"prop types drawn identically in plan: {clashes}"


def _blender_constants(module: str) -> dict:
    """The module-level literal constants of a scripts/blender module, read by AST.

    Read and not imported because those modules run under Blender's own bundled Python: they
    `import bpy` at the top and are forbidden from importing this package at all (.importlinter).
    Same mechanism, and the same reason, as
    tests/test_paint.py:test_blender_stroke_widths_match_the_channels.
    """
    source = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "blender" / module
    out = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue    # a COMPUTED constant (42 * 0.0254), not one of the sizes read here
    return out


# Every size the plan view draws a prop at, against the figure the 3D builder builds it from.
# src/render/props.py keeps feet and blender_props.py metres, each the unit its own side works
# in; a RADIUS is paired with a radius and a WIDTH with a width, because halving one on the way
# across is the mistake this pin is for.
PROP_DIMENSIONS_FT_TO_M = {
    "SIGN_POST_RADIUS_FT": "SIGN_POST_RADIUS_M",
    "SIGN_PLATE_THICKNESS_FT": "SIGN_PLATE_THICKNESS_M",
    "STOP_SIGN_PLATE_RADIUS_FT": "STOP_SIGN_PLATE_RADIUS_M",
    "YIELD_SIGN_PLATE_RADIUS_FT": "YIELD_SIGN_PLATE_RADIUS_M",
    "BIKE_WARNING_PLATE_RADIUS_FT": "BIKE_WARNING_PLATE_RADIUS_M",
    "SCHOOL_ZONE_PLATE_RADIUS_FT": "SCHOOL_ZONE_PLATE_RADIUS_M",
    "RECTANGULAR_PLATE_WIDTH_FT": "RECTANGULAR_PLATE_WIDTH_M",
    "RECTANGULAR_PLATE_THICKNESS_FT": "RECTANGULAR_PLATE_THICKNESS_M",
    "RRFB_PLATE_WIDTH_FT": "RRFB_PLATE_WIDTH_M",
    "RRFB_PLATE_THICKNESS_FT": "RRFB_PLATE_THICKNESS_M",
    "RRFB_POST_RADIUS_FT": "RRFB_POST_RADIUS_M",
    "PUSHBUTTON_POST_RADIUS_FT": "PUSHBUTTON_POST_RADIUS_M",
    "PUSHBUTTON_HOUSING_WIDTH_FT": "PUSHBUTTON_HOUSING_WIDTH_M",
    "PUSHBUTTON_HOUSING_DEPTH_FT": "PUSHBUTTON_HOUSING_DEPTH_M",
    "PED_SIGNAL_HEAD_WIDTH_FT": "PED_SIGNAL_HEAD_WIDTH_M",
    "TRAFFIC_SIGNAL_POLE_RADIUS_FT": "TRAFFIC_SIGNAL_POLE_RADIUS_M",
    "MAST_ARM_RADIUS_FT": "MAST_ARM_RADIUS_M",
    "VEHICLE_SIGNAL_HEAD_WIDTH_FT": "VEHICLE_SIGNAL_HEAD_WIDTH_M",
    "HYDRANT_RADIUS_FT": "HYDRANT_RADIUS_M",
    "BOLLARD_RADIUS_FT": "BOLLARD_RADIUS_M",
}


def test_prop_dimensions_match_the_3d_builders():
    """One object, one size, in both views.

    The plan view draws a prop at its real ground size so that how big a signal pole looks is a
    fact about the street rather than about the window - which only means anything if the size it
    draws is the size the render builds. The two tables are a copy by necessity (blender_props.py
    cannot be imported), so they are the pair that drifts silently: a plate widened in 3D and not
    here is a sign that overhangs the kerb in one view and clears it in the other, with nothing to
    say which is right.
    """
    from src.render import props
    from src.render.coords import FT_TO_M

    declared = _blender_constants("blender_props.py")
    wanted = set(PROP_DIMENSIONS_FT_TO_M.values()) | {"TACTILE_PAD_FALLBACK_M"}
    assert wanted <= set(declared), (
        f"blender_props.py no longer declares {sorted(wanted - set(declared))} - the pin has "
        f"nothing to read, which is not the same as the two sides agreeing")

    for ft_name, m_name in sorted(PROP_DIMENSIONS_FT_TO_M.items()):
        in_ft = getattr(props, ft_name)
        assert in_ft * FT_TO_M == pytest.approx(declared[m_name]), (
            f"{ft_name} is {in_ft:.4f} ft ({in_ft * FT_TO_M:.4f} m) in plan and {m_name} is "
            f"{declared[m_name]} m in 3D - one object drawn at two sizes")

    # The pad's fallback, whose own comment promises it is kept in step with these. Rounded to
    # millimetres there, so compared at that tolerance rather than pretended to be exact.
    depth_m, width_m = declared["TACTILE_PAD_FALLBACK_M"]
    pad_in_plan_m = (props.TACTILE_PAD_DEPTH_FT * FT_TO_M, props.TACTILE_PAD_WIDTH_FT * FT_TO_M)
    assert pad_in_plan_m == pytest.approx((depth_m, width_m), abs=5e-4)


def test_no_prop_is_drawn_at_a_size_nobody_declared():
    """Every figure in PROP_MARKERS traces back to a named constant in src/render/props.py.

    The table is where a bare literal would be easiest to type and hardest to see: a plate written
    as 2.0 instead of `2 * STOP_SIGN_PLATE_RADIUS_FT` draws plausibly today and stops following
    the 3D builder the moment that builder changes - which is exactly what the pin above would
    then fail to catch, because it would be pinning a constant nothing reads.
    """
    from src.render import props
    from src.render.plan_view import PROP_MARKERS

    # Rounded only at the comparison. Doubling an already-rounded radius is off by an ulp at the
    # ninth decimal and reports every diameter on the sheet as undeclared.
    declared = [value for name, value in vars(props).items()
                if name.endswith("_FT") and isinstance(value, int | float)]
    named = ({round(value, 9) for value in declared}
             | {round(2 * value, 9) for value in declared})     # a diameter off a radius
    undeclared = sorted({round(size, 9)
                         for pieces in PROP_MARKERS.values()
                         for footprint, _style in pieces if footprint is not None
                         for size in (footprint.across_ft, footprint.along_ft)}
                        - named)
    assert not undeclared, (
        f"plan-view prop footprint(s) at {undeclared} ft, which is no constant in "
        f"src/render/props.py or twice one - a size with no home cannot be pinned to the 3D")


def test_the_plan_view_draws_a_kerb_at_the_width_the_render_builds_it():
    """One kerb, one width, all three kinds.

    scripts/blender/blender_scene.py builds every traced kerb at KERB_WIDTH_M whatever OSM calls
    it; what tells raised from lowered from flush in 3D is HEIGHT (src/render/export.py's
    KERB_HEIGHT_M), which a plan cannot draw. So a plan view drawing three different WEIGHTS is
    drawing three kerbs that do not exist - and it drew the raised one twice as thick as the
    lowered one, which is a claim about width that the render makes about height.
    """
    from src.geometry.kerbs import KerbType
    from src.render.coords import FT_TO_M
    from src.render.plan_view import KERB_DASHES_FT, KERB_STYLE, KERB_WIDTH_FT

    declared = _blender_constants("blender_scene.py")
    assert "KERB_WIDTH_M" in declared, "blender_scene.py no longer declares KERB_WIDTH_M"
    in_plan_m = KERB_WIDTH_FT * FT_TO_M
    assert in_plan_m == pytest.approx(declared["KERB_WIDTH_M"]), (
        f"a kerb is {in_plan_m:.3f} m wide in plan and {declared['KERB_WIDTH_M']} m in 3D")
    assert set(KERB_DASHES_FT) == set(KerbType) == set(KERB_STYLE), (
        "a kerb type with no dash cadence or no style raises mid-build, several phases from here")
    sized_on_the_sheet = sorted(str(kerb) for kerb, style in KERB_STYLE.items()
                                if {"linewidth", "linestyle"} & set(style))
    assert not sized_on_the_sheet, (
        f"{sized_on_the_sheet} still carry a point weight or a matplotlib dash beside a body in "
        f"feet - two answers to how wide a kerb is, and the sheet wins the one you can see")


def test_no_hatched_zone_is_drawn_with_a_sheet_space_pattern():
    """A hatched zone's strokes are PAINT, so they are drawn as paint.

    matplotlib's `hatch` is a pattern on the sheet: its spacing and weight are fixed in points, so
    one buffer read as five hairlines on a corridor sheet and as a solid mass on a junction one,
    and neither was the PAINT_HATCH_SPACING_FT cadence the render lays on the same zone. The wash
    underneath is allowed to stay - no colour is applied to a real street, and it is what tells a
    daylight zone from a parking buffer - but the strokes are the marking.
    """
    from src.render.plan_view import PAINT_STYLE

    patterned = sorted(str(kind) for kind, style in PAINT_STYLE.items() if "hatch" in style)
    assert not patterned, (
        f"{patterned} are drawn with a sheet-space hatch pattern. Draw the strokes instead - "
        f"plan_view._hatch_strokes_ft, off the same hatch_lines_ft the 3D export serializes")
