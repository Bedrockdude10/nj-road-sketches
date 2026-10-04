"""The two views have to be pointed at the same ground.

The plan view and the 3D render are the same reconstruction drawn twice, so they are read side by
side - and each of them used to decide its own frame: a hardcoded 110 ft square on the junction
node in 2D, the pavement's own clipped extent in 3D. Nothing compared them, and on the four sites
they disagreed by 1.15-1.57x and by 6.5-12.5 ft of centre.
"""
import contextlib
import io
import json
import math
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point

from src.render.coords import FT_TO_M
from src.render.frame import LEG_REACH_TOLERANCE
from tests.conftest import SITES, needs_source_data

TOL_FT = 0.01


def _export(model, state, name, out_path):
    from src.render.export import export_scenario

    with contextlib.redirect_stdout(io.StringIO()):
        path = export_scenario(model, state, name, out_path, theme={})
    return json.loads(Path(path).read_text())


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_both_views_frame_the_same_ground(site, site_models, tmp_path):
    """The plan view's axes and the render's camera cover one square of ground, to the foot."""
    from src.geometry.treatments import DesignState
    from src.render.plan_view import plot_design_state

    model = site_models[site]
    state = DesignState.from_model(model)
    data = _export(model, state, "existing", tmp_path / f"{site}.json")

    fig, ax = plt.subplots()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            plot_design_state(ax, model, state, "existing", dimension_labels=False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    finally:
        plt.close(fig)

    plan_radius_ft = max(xmax - xmin, ymax - ymin) / 2
    # The render's frame is local metres about the junction; the plan view's is state plane feet.
    render_centre_ft = (model.center_ft.x + data["frame"]["center_m"][0] / FT_TO_M,
                        model.center_ft.y + data["frame"]["center_m"][1] / FT_TO_M)
    render_radius_ft = data["frame"]["radius_m"] / FT_TO_M

    assert plan_radius_ft == pytest.approx(render_radius_ft, abs=TOL_FT), (
        f"{site}: the plan view takes in {plan_radius_ft:.1f} ft where the render takes in "
        f"{render_radius_ft:.1f} ft ({plan_radius_ft / render_radius_ft:.2f}x), so the two "
        f"pictures do not show the same street")
    assert (xmin + xmax) / 2 == pytest.approx(render_centre_ft[0], abs=TOL_FT)
    assert (ymin + ymax) / 2 == pytest.approx(render_centre_ft[1], abs=TOL_FT)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_shared_frame_is_the_one_the_render_computed_for_itself(site, site_models, tmp_path):
    """Adopting one frame moved the plan view, not the camera.

    The definition (the modelled pavement's extent, clipped at the legs' reach, plus a margin) is
    the one blender_scene.py had already worked out; src/render/frame.py only moved it to the side
    of the boundary that can be tested. This recomputes it the way blender_scene.py's fallback
    still does, from the exported pavement, and it has to come out the same.
    """
    from src.geometry.treatments import DesignState

    model = site_models[site]
    data = _export(model, DesignState.from_model(model), "existing", tmp_path / f"{site}.json")

    rings = data["pavement"]
    xs = [x for ring in rings for x, _y in ring]
    ys = [y for ring in rings for _x, y in ring]
    reach = max((math.hypot(*leg["far_m"]) for leg in data["legs"]), default=0.0)
    framed = [(x, y) for x, y in zip(xs, ys)
              if not reach or math.hypot(x, y) <= reach * LEG_REACH_TOLERANCE]
    fx = [x for x, _y in framed] or xs
    fy = [y for _x, y in framed] or ys

    assert data["frame"]["center_m"][0] == pytest.approx((min(fx) + max(fx)) / 2, abs=0.01)
    assert data["frame"]["center_m"][1] == pytest.approx((min(fy) + max(fy)) / 2, abs=0.01)
    assert data["frame"]["radius_m"] == pytest.approx(
        max(max(fx) - min(fx), max(fy) - min(fy)) / 2 * 1.2, abs=0.01)


@needs_source_data
def test_a_proposal_frames_the_same_ground_as_its_baseline(site_models):
    """A before/after pair is only comparable if both panels cover the same square.

    The kerb-moving state is built HERE rather than borrowed from a site's scenario list. It
    used to be Broad & Greenwood's bulb-out proposal - the only scenario in the repo that moved
    a kerb - so deleting that proposal would have quietly left this test with nothing to prove:
    every remaining scenario is paint-only, and a frame cannot shift when no kerb does. The
    subject of this test is the frame, not any one proposal, so it makes its own.
    """
    from src.geometry.model import build_pavement_polygon
    from src.geometry.treatments import DesignState, bulb_out_corner_pair
    from src.render.plan_view import plot_design_state

    model = site_models["broad_st_greenwood"]
    baseline = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        proposed = baseline
        for leg_name in ("broad_st_east", "broad_st_west"):
            proposed = bulb_out_corner_pair(proposed, leg_name, extension_ft=8.0,
                                             crossing_ft=25.0)
    assert (build_pavement_polygon(proposed.corner_fillets).area
            < build_pavement_polygon(baseline.corner_fillets).area - 500), (
        "this state is supposed to take roadway into the corners; if it no longer does, it "
        "cannot exercise what this test is about")

    frames = []
    for state, label in ((baseline, "baseline"), (proposed, "proposed")):
        fig, ax = plt.subplots()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plot_design_state(ax, model, state, label, dimension_labels=False)
            frames.append((ax.get_xlim(), ax.get_ylim()))
        finally:
            plt.close(fig)
    assert [v for pair in frames[0] for v in pair] == pytest.approx(
        [v for pair in frames[1] for v in pair], abs=0.01), (
        "the two panels of a before/after pair frame different ground, so the drawing invites "
        "a comparison it does not support")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_centerline_paint_follows_the_road_in_both_views(site, site_models, tmp_path):
    """The double yellow is the same stripe in the plan view and in the render.

    The render used to be handed the leg's near and far points and draw a straight stripe between
    them - the CHORD. Ten of this project's twelve legs are straight 2-vertex lines, so it looked
    right nearly everywhere; on the two that are not, it is 3.98 ft out on broad_st_east and 7.58
    ft on louellen_st_west. That is most of a lane's width of asphalt moved from one side of the
    road to the other, and it put the centerline where the stop bar it meets is not.

    Checked as "every exported vertex lies on the leg's own centerline, half the gap away from
    it", which is what following the road means, and separately that the two views produce the
    same lines - the chord passes neither.
    """
    from src.geometry.treatments import DesignState
    from src.render.crosswalks import (DOUBLE_YELLOW_SEPARATION_FT, NARROW_LINE_WIDTH_M,
                                       centerline_paint_ft, centerline_start_ft)
    from src.render.scene import SceneGeometry

    model = site_models[site]
    state = DesignState.from_model(model)
    data = _export(model, state, "existing", tmp_path / f"{site}.json")
    with contextlib.redirect_stdout(io.StringIO()):
        scene = SceneGeometry.resolve(model, state)

    painted_legs = 0
    for exported in data["legs"]:
        leg = state.legs[exported["name"]]
        lines = exported["centerline_paint_m"]
        style = exported["centerline_style"]
        if style == "none":
            assert not lines, f"{exported['name']} is styled 'none' but paint was exported"
            continue
        if not lines:
            continue        # paint that starts past the end of the leg - see centerline_start_ft
        painted_legs += 1
        if style == "double_yellow":
            assert len(lines) == 2, f"a double yellow is two stripes, got {len(lines)}"
            # AND THEY HAVE TO CLEAR EACH OTHER. The offset check below cannot see this: its
            # tolerance is 0.35 ft, set so a vertex on a bending offset curve still reads as
            # following the road, and the whole separation is 0.82 ft - so a pair drawn at a
            # third of it passed. They were, for the life of this test: 0.10 m apart with a
            # 0.15 m stripe, overlapping into one line in every render. Measured between the
            # two stripes rather than from the alignment, because the gap is the thing that
            # makes a double yellow a double yellow, and it is exact.
            a, b = LineString(lines[0]), LineString(lines[1])
            apart_m = max(b.distance(Point(q)) for q in a.coords)
            assert apart_m == pytest.approx(DOUBLE_YELLOW_SEPARATION_FT * FT_TO_M, abs=0.005), (
                f"{exported['name']}: the two stripes are {apart_m:.3f} m apart centre to "
                f"centre and each is laid {NARROW_LINE_WIDTH_M} m wide - a double yellow needs "
                f"{DOUBLE_YELLOW_SEPARATION_FT * FT_TO_M:.3f} m to leave the gap between them")

        for line in lines:
            for x_m, y_m in line:
                point = Point(model.center_ft.x + x_m / FT_TO_M,
                              model.center_ft.y + y_m / FT_TO_M)
                off_ft = leg.centerline.distance(point)
                expected_ft = DOUBLE_YELLOW_SEPARATION_FT / 2 if style == "double_yellow" else 0.0
                assert off_ft == pytest.approx(expected_ft, abs=0.35), (
                    f"{exported['name']}: a centerline vertex sits {off_ft:.2f} ft off the "
                    f"leg's centerline where the paint should be {expected_ft:.2f} ft off it - "
                    f"the stripe is not following the road")

        # And the plan view draws these exact lines, from the same call.
        start_ft = centerline_start_ft(scene.crosswalk_offsets[exported["name"]].offset_ft,
                                        scene.stop_bar_offsets.get(exported["name"]),
                                        exported["name"] in scene.marked_crosswalks)
        drawn = centerline_paint_ft(leg, start_ft, style)
        assert len(drawn) == len(lines), (
            f"{exported['name']}: the plan view draws {len(drawn)} stripe(s) where the render "
            f"gets {len(lines)}")

    assert painted_legs, f"{site} exported no centerline paint at all, so this proves nothing"


def _legs_only_model(reach_ft: float):
    """A model with legs and no pavement ring - enough for junction_frame, and nothing else.

    The ring is what puts the frame's centre off the junction node, so leaving it out is not a
    simplification: coincident centres are the WORST case for a radius that has to reach a
    corner, and the one every four-legged junction here sits near (0.3-12.5 ft apart).
    """
    from types import SimpleNamespace

    legs = {
        name: SimpleNamespace(centerline=LineString([(0, 0), (dx * reach_ft, dy * reach_ft)]))
        for name, (dx, dy) in {"north": (0, 1), "south": (0, -1),
                               "east": (1, 0), "west": (-1, 0)}.items()
    }
    return SimpleNamespace(center_ft=Point(0, 0), legs=legs, corner_fillets={})


def test_the_far_corner_of_the_sheet_is_in_the_view():
    """A drawing covers its WINDOW, and the window is the square Frame.bounds_ft() describes.

    The plan view sets its axes straight from those bounds, so the farthest ground on the sheet
    is a CORNER, not a point on the inscribed circle. Membership in the view - what the coverage
    audit counts as "in the frame" - has to be that square: a feature in a corner is on the sheet
    (11.13% of its area at coincident centres is outside the inscribed circle), and one just past
    the edge is not.

    Used to be the covering radius a fetch needed to supply; nothing is fetched to a radius any
    more, so the property that survives is what the VIEW counts as shown.
    """
    from src.geometry.coverage import _in_frame
    from src.render.frame import junction_frame

    model = _legs_only_model(500.0)
    frame = junction_frame(model)
    xmin, xmax, ymin, ymax = frame.bounds_ft()
    inside, out_ft = _in_frame(model, frame)

    corner = Point(xmax - 1.0, ymax - 1.0)
    assert corner.distance(frame.center_ft) > frame.radius_ft, (
        "the probe is supposed to sit outside the inscribed circle, where only a square view "
        "can count it")
    assert inside(corner), "a feature in the far corner of the sheet is on the sheet"
    assert inside(LineString([(xmin + 1.0, ymin + 1.0), (xmin + 3.0, ymin + 1.0)]))
    assert not inside(Point(xmax + 1.0, frame.center_ft.y)), "ground past the edge is not shown"
    assert out_ft(corner) == pytest.approx(model.center_ft.distance(corner), abs=TOL_FT), (
        "distance is reported from the JUNCTION, which is what a reader means by 'N ft out'")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_sheet_is_drawn_over_ground_the_world_holds_at_every_scale(site, site_models,
                                                                       wide_site_models):
    """Every corner of the sheet lies inside the OSM area the model reads - at 1x and on the wide one.

    The world is the whole area and the sheet is a view onto it, so the only way a view can show
    ground the world knows nothing about is a frame that reaches outside the area (which reads as
    "nothing mapped there"). That is what the covering radius used to guarantee by fetching far
    enough; now it is a claim about the frame against the area, and the frame scale is what carries
    a corner out - NOTE it has to be set again here, the fixture restores the environment and the
    frame is read at DRAW time.
    """
    import os

    from shapely.geometry import Polygon

    from src.geometry.intersection import to_state_plane
    from src.render.frame import FRAME_SCALE_ENV, junction_frame
    from src.sources.osm_context import SNAPSHOT_AREAS
    from tests.conftest import WIDE_FRAME_SCALE

    west, south, east, north = SNAPSHOT_AREAS[site_models[site].osm_area]
    # Densified edges: the state-plane grid is not the lon/lat one, so a box is not a box.
    ring = ([(west + (east - west) * t / 20, south) for t in range(20)]
            + [(east, south + (north - south) * t / 20) for t in range(20)]
            + [(east - (east - west) * t / 20, north) for t in range(20)]
            + [(west, north - (north - south) * t / 20) for t in range(20)])
    area = Polygon(to_state_plane(ring))

    previous = os.environ.get(FRAME_SCALE_ENV)
    try:
        for scale, model in ((1.0, site_models[site]), (WIDE_FRAME_SCALE, wide_site_models[site])):
            os.environ[FRAME_SCALE_ENV] = str(scale)
            xmin, xmax, ymin, ymax = junction_frame(model).bounds_ft()
            for x in (xmin, xmax):
                for y in (ymin, ymax):
                    assert area.contains(Point(x, y)), (
                        f"{site} at {scale}x: the sheet's corner ({x:.0f}, {y:.0f}) is "
                        f"{area.exterior.distance(Point(x, y)):.0f} ft outside the "
                        f"{model.osm_area} area, so the view shows ground the world holds "
                        f"nothing for")
    finally:
        if previous is None:
            os.environ.pop(FRAME_SCALE_ENV, None)
        else:
            os.environ[FRAME_SCALE_ENV] = previous
