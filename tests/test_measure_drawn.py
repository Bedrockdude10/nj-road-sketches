"""What scripts/measure_drawn.py measures, which is what this repo diagnoses from.

The tool is the quantitative layer SKILLS.md 0a says answer every geometry complaint at, so a
wrong number here is worse than a wrong number in a render: it is a wrong number that arrives
wearing the authority of a measurement. `--gaps` spent its life reporting up to 17.54 ft of
separation between paint and kerb on paint drawn flush, because it reduced over a polygon's
VERTICES and a fill inherits its outer edge - and only its outer edge's sparse vertices - from
the traced kerb. These pin the measurement, not the printing.
"""
import contextlib
import io

import numpy as np
import pytest

from conftest import needs_source_data

CORNER_END_FT = 50.0
"""Past the corner return on both Greenwood kerbs, where the hatch runs beside its kerb.

The return itself IS bare - the kerb flares to 21.12 ft off the alignment by station 33 on the
east while the hatch starts at 44.78 - so a profile that reported nothing anywhere would have
stopped measuring rather than started being right.
"""


@pytest.fixture(scope="module")
def greenwood_two_way():
    from scripts.measure_drawn import build

    with contextlib.redirect_stdout(io.StringIO()):
        return build("broad_st_greenwood", "build_proposal_two_way_bike_lane")


@needs_source_data
def test_the_gap_profile_reads_flush_where_the_hatch_follows_its_kerb(greenwood_two_way):
    """Both Greenwood kerbs, past the corner, measured 0.04 ft from their hatch and reported 4.06.

    A LaneNarrowing fill is offset from the traced kerb, so its outer edge carries the kerb's
    vertices and nothing between them - 9 over 85 ft against 44 on the inner edge, which is a
    plain offset from the alignment. Every 10 ft bin from station 54.5 out therefore held inner
    vertices and no outer one, and the reduction returned the INNER edge at 11.82 ft: station 120
    reported 15.88 - 11.82 = 4.06 ft of separation on paint that touches.
    """
    from scripts.measure_drawn import gap_profile
    from src.geometry.targets import BOTH_SIDES

    for side in BOTH_SIDES:
        edges, gap, why = gap_profile(greenwood_two_way, "greenwood_ave_south", side, 10.0)
        assert why is None, f"{side.value}: {why}"
        along = np.isfinite(gap) & (edges >= CORNER_END_FT)
        assert along.any(), f"{side.value}: nothing measured past the corner"
        worst = int(np.nanargmax(np.where(along, gap, -np.inf)))
        assert gap[worst] <= 0.25, (
            f"greenwood_ave_south {side.value}: the hatch reads {gap[worst]:.2f} ft off its kerb "
            f"at station {edges[worst]:.0f}, and a perpendicular cut through the drawn fill "
            f"there puts it flush")


@needs_source_data
def test_the_gap_profile_still_sees_the_bare_corner_return(greenwood_two_way):
    """And the fix did not buy that by blinding the measurement.

    A check that cannot fail proves nothing, and one that reports nothing anywhere is the same
    thing wearing an all-clear. The corner return is real bare pavement - the kerb curves out to
    21.12 ft off the alignment by station 33 while the hatch is held back to 44.78 by the corner
    clearance - so the profile has to still say so.
    """
    from scripts.measure_drawn import gap_profile
    from src.geometry.targets import Side

    edges, gap, why = gap_profile(greenwood_two_way, "greenwood_ave_south", Side("left"), 10.0)
    assert why is None
    corner = np.isfinite(gap) & (edges < CORNER_END_FT)
    assert np.nanmax(gap[corner]) > 1.0, (
        "the corner return reads flush, so the profile has stopped measuring rather than "
        "started being right")


@needs_source_data
def test_the_reach_is_the_paint_and_not_its_nearest_vertex(greenwood_two_way):
    """surface_reach_at, station by station, against the vertex reduction it replaced.

    The direct form of the defect: between two outer-edge vertices the vertex maximum falls back
    onto the fill's inner edge, a fixed 11.82 ft from the alignment, while the surface is out at
    the kerb. Asserted as a spread rather than a single station because which bins are starved
    depends on where the OSM trace happens to carry a node.
    """
    from scripts.measure_drawn import piece_coords, surface_reach_at
    from src.geometry.model import station_offset_many
    from src.geometry.targets import Side

    leg = greenwood_two_way.model.legs["greenwood_ave_south"]
    fills = [p for p in greenwood_two_way.paint
             if p.leg == "greenwood_ave_south" and str(p.side) == "left"
             and p.kind.name == "lane_narrowing_fill"]
    assert fills, "no hatch on this kerb to measure"
    _stations, offsets = station_offset_many(leg.centerline, piece_coords(fills[0].geometry))
    outer = np.abs(offsets) > np.abs(offsets).mean()
    assert outer.sum() * 3 < (~outer).sum(), (
        f"the outer edge carries {int(outer.sum())} of {len(offsets)} vertices, so this leg no "
        f"longer exhibits the sparsity the reduction fell through")

    sample = np.arange(60.0, 130.0, 5.0)
    reach = surface_reach_at(greenwood_two_way, "greenwood_ave_south", Side("left"), sample)
    assert np.isfinite(reach).all(), "the cut missed a fill it crosses"
    assert reach.min() > np.abs(offsets[~outer]).max() + 1.0, (
        f"the reach bottoms out at {reach.min():.2f} ft, at or inside the fill's inner edge - "
        f"the measurement is still reading the wrong edge")


def a_dashed_leg(style="single_yellow_dashed"):
    """The same synthetic leg, but with a DASHED divider down it.

    Dashed rather than double, because those two are binned differently and only one of them
    had ever been measured: every centreline at every site in the committed clip is
    `double_yellow` or `none`, so the dashed path in report_lanes had no coverage at all.
    """
    built = an_undivided_leg()
    built.state.existing_centerline_styles["nb"] = style
    return built


def an_undivided_leg():
    """One 70 ft leg with a traced kerb and a parking bay, and NO centre stripe.

    Synthetic, and it has to be: the `centerline_style: none` branch is unreachable on every
    site in the committed clip - each scenario there ends up drawing a divider on every leg,
    which is why this defect survived to be found on a new site instead. lavallette_reese is
    the site that has it and its bbox is outside the fixture's clip.
    """
    from shapely.geometry import LineString

    from scripts.measure_drawn import Built
    from src.geometry.markings import PARKING_EDGE_LINE
    from src.geometry.model import Leg, point_at
    from src.geometry.paint import PaintPiece
    from src.geometry.targets import LegSide
    from src.geometry.treatments import DesignState, MarkedParking
    from src.render.crosswalks import CrosswalkOffset

    centre = LineString([(0, 0), (190, 0)])
    leg = Leg(name="nb", centerline=centre, curb_to_curb_ft=70.05)
    for side in ("left", "right"):
        sign = 1 if side == "left" else -1
        setattr(leg, f"{side}_curb",
                LineString([point_at(centre, station, sign * 34.9)
                            for station in (10.0, 190.0)]))
    # existing_centerline_styles, NOT a default: DesignState falls back to
    # DEFAULT_CENTERLINE_STYLE (single_yellow_dashed) for a leg it was told nothing about, so a
    # leg built with no style is a DASHED leg here and not a stripeless one.
    state = DesignState(legs={"nb": leg}, corner_fillets={},
                        existing_centerline_styles={"nb": "none"})
    for side in ("left", "right"):
        state = state.apply(MarkedParking(LegSide("nb", side), depth_ft=20.09,
                                          stall_length_ft=18.0, angle_deg=60.0, observed=True))

    class Scene:
        crosswalk_offsets = {"nb": CrosswalkOffset(30.0, "geometric_estimate")}
        stop_bar_offsets: dict = {}
        marked_crosswalks = ()

    # The bay's inner edge line, which is what bounds the travel way on each side - the one
    # piece of paint report_lanes needs to have something to measure out to. Its axis sits half
    # a stripe outboard of the lane, and _longitudinal_on takes the inner FACE, so the figure
    # the report prints is 15.30 - LANE_EDGE_LINE_WIDTH_FT / 2.
    paint = [PaintPiece(PARKING_EDGE_LINE,
                        LineString([point_at(centre, station, sign * 15.30)
                                    for station in (30.0, 190.0)]), "nb", side)
             for side, sign in (("left", 1), ("right", -1))]
    return Built(model=type("M", (), {"legs": {"nb": leg}})(), state=state,
                 scene=Scene(), paint=paint, crossings=None)


def test_an_undivided_carriageway_is_not_reported_as_a_lane_under_target():
    """The one verdict in this tool that is printed rather than computed, and it cried wolf.

    Where NOTHING divides the carriageway, report_lanes measures the alignment out to the
    bounding paint. That is a HALF-ROAD - the docstring says so - and comparing a half-road
    against an 11 ft LANE target is a category error that reads as a finding.

    It reads as a WRONG finding on a design that shifts the travel way. lavallette_reese's
    build_bike_lane_inboard pins a 27.55 ft section against the east kerb of a one-way pair, so
    the two halves come out 7.30 and 14.89 ft and the tool reported "16 of 16 bins, worst 7.30
    ft" against an 11 ft target on a street whose lanes are (7.30 + 14.89) / 2 = 11.10 ft, i.e.
    at target, on BOTH halves of BOTH legs.

    The widths stay printed, because half a road is a real measurement. What must not be printed
    is a verdict about a lane nobody drew: SKILLS.md retires a check that cries wolf as fast as
    one that cannot fail, and this one arrived wearing a measurement's authority.
    """
    from scripts.measure_drawn import report_lanes

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report_lanes(an_undivided_leg(), "nb", 10.0)
    rows = [line for line in out.getvalue().splitlines() if line.startswith("nb ")]
    assert len(rows) == 2, out.getvalue()
    for row in rows:
        assert row.split()[2] == "none", (
            f"the leg built for this test now carries a stripe, so it no longer covers the "
            f"branch under test: {row}")
        assert "half-road" in row, row
        assert "bins, worst" not in row, f"a lane verdict on an undivided carriageway: {row}"
        # AND THE WIDTH IS STILL THERE. Withholding the verdict must not become withholding the
        # measurement - a row with no number on it is the tool declining to answer.
        assert float(row.split()[3]) == pytest.approx(15.09, abs=0.3), row


def test_a_DASHED_divider_is_one_stripe_in_pieces_not_many_stripes():
    """report_lanes averaged per LineString, which is right for a double yellow and blind here.

    A double yellow is TWO parallel stripes, each one LineString running the leg, and averaging
    them per-LineString is what keeps an uneven vertex count in a bin from weighting one over
    the other. A DASHED line is ONE stripe cut into ~23 dashes, each also a LineString - so that
    same average took the mean of 23 arrays that are NaN almost everywhere, every bin came out
    NaN, and the report printed "nothing drawn in any bin along this side" for a leg with a
    divider painted down the middle of it.

    A quantitative tool that cannot see anything must say so rather than pass, and this one did
    say so - which is the only reason it was caught. It went unnoticed for as long as it did
    because no leg in the committed clip uses a dashed centreline, and single_yellow_dashed is
    this project's DEFAULT style.
    """
    from scripts.measure_drawn import report_lanes

    for style in ("single_yellow_dashed", "single_white_dashed"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report_lanes(a_dashed_leg(style), "nb", 10.0)
        rows = [line for line in out.getvalue().splitlines() if line.startswith("nb ")]
        assert len(rows) == 2, out.getvalue()
        for row in rows:
            assert "nothing drawn in any bin" not in row, (
                f"the divider is painted down this leg and the tool cannot see it: {row}")
            assert row.split()[2] == style, row
            # The same 15.09 ft the undivided case measures, because the dashes lie ON the
            # alignment here - what changed is whether the tool can find them, not where they are.
            assert float(row.split()[3]) == pytest.approx(15.09, abs=0.3), row


@needs_source_data
def test_a_striped_leg_still_gets_its_lane_verdict(greenwood_two_way):
    """The exemption is on `no stripe is drawn`, not on `this leg is inconvenient`.

    Broad St through the same junction carries a double yellow, so there IS a drawn divider, the
    figure IS a lane, and an under-target lane there is the finding this column exists for.
    Without this the change above could have silenced the tool everywhere and nothing would
    have said so.
    """
    from scripts.measure_drawn import report_lanes

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report_lanes(greenwood_two_way, None, 10.0)
    striped = [line for line in out.getvalue().splitlines() if "double_yellow" in line]
    assert striped, out.getvalue()
    for row in striped:
        assert "half-road" not in row, row
        assert "held in by" in row or "bins, worst" in row, row
