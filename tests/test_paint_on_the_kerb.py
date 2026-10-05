"""Phase 2: every marking is placed off the traced kerb, through PaintContext.paint().

The rule (Danny, 2026-10-04): paint comes off the kerbs OSM traced. The nominal width
(`curb_to_curb_ft`) and the bare alignment are fallbacks for where nothing was traced, never the
datum where something was. One chain decides which evidence places a kerb or the centre at each
station - src/geometry/model/kerb_chain.py - and every marking reaches it through paint().

These tests say so behaviourally, on one synthetic 300 ft street, so they hold whichever file a
marking is built in:

  * NOMINAL-BLIND. With both kerbs traced, the paint is identical whatever the nominal width
    says. Today a wrong nominal width moves parking and lane narrowing - and, at 68 ft against
    kerbs 44 ft apart, silently drops 14 of parking's 16 pieces.
  * THROUGH paint(). Every line and fill carries the `datum` paint() stamps on it, and with both
    kerbs traced that datum is the traced kerbs and the centre between them, nothing else.
  * RIGID AT THE NARROWEST. A section is one rigid piece sized where the kerb is tightest
    (SKILLS.md 0a/0b), so the travel-lane edge is straight and sits where it would on a straight
    kerb at the narrowest width. The kerbside zone - parking, buffer, hatch - absorbs the rest.
  * FALLBACK, NEVER SKIP. With nothing traced, the nominal width places the same paint.

Where a test needs a reference value it reads it off the same build on a straight street whose
nominal width AGREES with its kerbs, rather than restating a treatment's arithmetic here - see
SKILLS.md 0a on rebuilding a section from the constants you think it used.
"""
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from src.checks import (MarkingsDoNotCollide, PaintClearOfTheTravelLane, PaintInsideTheCurb,
                        SceneCheck, SceneContext)
from src.geometry.markings import (BIKE_LANE_EDGE_LINE, LANE_EDGE_LINE, PARKING_EDGE_LINE,
                                   PaintKind)
from src.geometry.model import CentreSource, KerbSource, Leg, station_offset_many
from src.geometry.paint import PaintPiece, curbside_paint_ft
from src.geometry.paint.datum import kerb_profile
from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.treatments import (DesignState, LaneNarrowing, LaneNarrowingBollards,
                                     MarkedParking)
from src.geometry.treatments.base import Treatment
from src.geometry.treatments.bikeways.bollards import AddBikeLaneBollards
from src.geometry.treatments.bikeways.place import (AddBikeLane, AddKerbsideBikeLane,
                                                    AddTwoWayBikeLane)
from src.geometry.treatments.crossings import RaiseCrossing
from src.geometry.treatments.parking import ParkingBufferBollards
from src.render.crosswalks import (DOUBLE_YELLOW_SEPARATION_FT, CrosswalkOffset,
                                   centerline_paint_ft, stop_bar_ends_ft, stop_bar_width_ft)
from tests.conftest import KerbVertices, needs_source_data, synthetic_leg

LEG = "east"
LENGTH_FT = 300.0
CROSSING_FT = 20.0
#: How closely two derivations of one placement must agree: well under a stroke width.
TOL_FT = 0.02

KERB_FT = 22.0
STRAIGHT: KerbVertices = [(0.0, KERB_FT), (LENGTH_FT, KERB_FT)]
#: Pinched to 20 ft at stations 75 and 225 and bulging to 24 ft at 150. Used on BOTH sides, so the
#: centre between the kerbs stays on the alignment and only the kerbside zones see the wobble.
WOBBLE: KerbVertices = [(0.0, 22.0), (50.0, 22.0), (75.0, 20.0), (100.0, 22.0), (150.0, 24.0),
                        (200.0, 22.0), (225.0, 20.0), (250.0, 22.0), (LENGTH_FT, 22.0)]
#: How far in the kerb paint FOLLOWS comes at its tightest - read off kerb_profile, which flattens
#: the V of a traced pinch (20.08 ft here, not the 20.00 vertex), so the test asks the same kerb the
#: paint is placed off rather than the raw tracing.
PINCH_FT = KERB_FT - float(np.min(kerb_profile(
    synthetic_leg("pinch", LENGTH_FT, 2 * KERB_FT, WOBBLE, WOBBLE), Side.LEFT,
    np.arange(0.0, LENGTH_FT + 0.5, 0.5)).offsets_ft))
#: The nominal width that agrees with STRAIGHT kerbs on both sides.
AGREEING_NOMINAL_FT = 2 * KERB_FT

KERB = LegSide(LEG, Side.LEFT)

#: One scenario per marking family. The comment names the file that owns its placement, which is
#: also how Phase 2 is split between agents.
CASES: dict[str, list[Treatment]] = {
    # src/geometry/treatments/parking.py
    "parking": [MarkedParking(KERB)],
    "parking_buffered": [MarkedParking(KERB, curb_offset_ft=3.0)],
    "parking_bollards": [MarkedParking(KERB, curb_offset_ft=3.0), ParkingBufferBollards(KERB)],
    # src/geometry/treatments/lanes.py (+ the src/geometry/model/stripes.py helpers it calls)
    "narrowing": [LaneNarrowing(LegTarget(LEG))],
    "narrowing_line": [LaneNarrowing(LegTarget(LEG), line_only=True)],
    "narrowing_bollards": [LaneNarrowing(LegTarget(LEG)), LaneNarrowingBollards(LegTarget(LEG))],
    # src/geometry/treatments/bikeways/place.py, bikeways/bollards.py
    "bike": [AddBikeLane(KERB, width_ft=5.0, buffer_ft=2.0)],
    "bike_kerbside": [AddKerbsideBikeLane(KERB, width_ft=5.0, buffer_ft=2.0)],
    "bike_bollards": [AddKerbsideBikeLane(KERB, width_ft=5.0, buffer_ft=2.0),
                      AddBikeLaneBollards(KERB)],
    "bike_two_way": [AddTwoWayBikeLane(KERB, width_ft=10.0, buffer_ft=3.0)],
}

#: The kind that is the travel lane's edge on the treated kerb, per family (its innermost piece),
#: and how far that edge moves when the kerb pinches in by PINCH_FT. Two rules, both "straight":
#:
#:   * A section that SETS THE TRAVEL LANE holds it at its width off the centre, so the edge does
#:     not move at all and its buffer takes the pinch - the bike sections.
#:   * A kerbside ZONE OF DECLARED DEPTH (8 ft of parking, a 5 ft hatch) is rigid at the narrowest
#:     kerb, so the edge moves in by the whole pinch and the zone is its full depth even there.
#:     Sized anywhere else, the zone is shallower than declared at the pinch.
LANE_EDGE: dict[str, tuple[PaintKind, float]] = {
    "parking": (PARKING_EDGE_LINE, PINCH_FT), "parking_buffered": (PARKING_EDGE_LINE, PINCH_FT),
    "narrowing": (LANE_EDGE_LINE, PINCH_FT), "narrowing_line": (LANE_EDGE_LINE, PINCH_FT),
    "bike": (BIKE_LANE_EDGE_LINE, 0.0), "bike_kerbside": (BIKE_LANE_EDGE_LINE, 0.0),
}

#: The only sources that may place paint on a leg traced on both sides along its whole length.
TRACED_SOURCES: set[str] = {KerbSource.TRACED, CentreSource.KERBS}

#: The scene checks the proposed borough fails today because paint is sized off the nominal width
#: (35 violations when this was written).
DATUM_CHECKS: tuple[type[SceneCheck], ...] = (PaintInsideTheCurb, PaintClearOfTheTravelLane,
                                              MarkingsDoNotCollide)


def street(nominal_ft: float | None, left: KerbVertices | None = STRAIGHT,
           right: KerbVertices | None = STRAIGHT, state_line: LineString | None = None) -> Leg:
    return synthetic_leg(LEG, LENGTH_FT, nominal_ft, left, right, state_line)


def build(leg: Leg, treatments: list[Treatment]) -> tuple[DesignState, list[PaintPiece]]:
    """(state, every piece of paint) for one leg with these treatments applied."""
    state = DesignState(legs={LEG: leg}, corner_fillets={})
    with contextlib.redirect_stdout(io.StringIO()):
        state = state.apply(*treatments)
        paint = curbside_paint_ft(state, {LEG: CrosswalkOffset(CROSSING_FT, "geometric_estimate")},
                                  None)
    return state, paint


def frame_of(geometry: BaseGeometry, leg: Leg) -> tuple[np.ndarray, np.ndarray]:
    """(stations, offsets) of a line's vertices or a polygon's exterior, in the leg's frame."""
    coords = geometry.exterior.coords if geometry.geom_type == "Polygon" else geometry.coords
    return station_offset_many(leg.centerline, np.asarray(coords, dtype=float))


def lane_edge_offsets(case: str, leg: Leg) -> np.ndarray:
    """Offsets of the travel-lane edge on the left: the innermost piece of its family's kind."""
    _, paint = build(leg, CASES[case])
    kind = LANE_EDGE[case][0]
    lines = [p for p in paint if p.side == Side.LEFT and p.kind is kind
             and p.geometry.geom_type == "LineString" and p.geometry.length > 50.0]
    assert lines, f"{case}: no {kind.name} longer than 50 ft on the left"
    return min((frame_of(p.geometry, leg)[1] for p in lines), key=lambda o: float(np.mean(o)))


def same_paint(a: list[PaintPiece], b: list[PaintPiece]) -> bool:
    return len(a) == len(b) and all(p.kind is q.kind and p.side == q.side
                                    and p.geometry.equals_exact(q.geometry, TOL_FT)
                                    for p, q in zip(a, b))


def sources(paint: list[PaintPiece]) -> set[str]:
    """Every datum source any piece was placed off."""
    return {source for p in paint for source in p.datum}


# --------------------------------------------------------------------------
# Nominal-blind
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nominal_ft", [68.0, 30.0], ids=["nominal_too_wide", "nominal_too_narrow"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_paint_does_not_depend_on_the_nominal_width_where_both_kerbs_are_traced(
        case: str, nominal_ft: float) -> None:
    """Kerbs traced 44 ft apart: the paint with an agreeing nominal width and with a wrong one is
    the same paint, piece for piece - posts included."""
    _, agreeing = build(street(AGREEING_NOMINAL_FT), CASES[case])
    _, wrong = build(street(nominal_ft), CASES[case])
    assert agreeing, f"{case} paints nothing even where the nominal width agrees"
    assert same_paint(agreeing, wrong), (
        f"{case}: {len(agreeing)} pieces with a {AGREEING_NOMINAL_FT:.0f} ft nominal width, "
        f"{len(wrong)} with {nominal_ft:.0f} ft - the paint moved with a fallback figure")


#: What each family paints today on the street whose nominal width AGREES with its kerbs - piece
#: count and the travel-lane edge's offset - measured off the code before Phase 2. Where nominal
#: and traced agree the old placement was right, so moving onto the kerb must not change it.
TODAY: dict[str, tuple[int, float | None]] = {
    "bike": (7, 11.41), "bike_bollards": (36, None), "bike_kerbside": (8, 11.41),
    "bike_two_way": (42, None), "narrowing": (4, 17.41), "narrowing_bollards": (64, None),
    "narrowing_line": (2, 17.41), "parking": (16, 14.41), "parking_bollards": (44, None),
    "parking_buffered": (18, 11.41),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_where_the_nominal_width_agrees_the_paint_is_what_it_was(case: str) -> None:
    pieces, edge_ft = TODAY[case]
    _, paint = build(street(AGREEING_NOMINAL_FT), CASES[case])
    assert len(paint) == pieces
    if edge_ft is not None:
        assert float(np.mean(lane_edge_offsets(case, street(AGREEING_NOMINAL_FT)))) == pytest.approx(
            edge_ft, abs=TOL_FT)


# --------------------------------------------------------------------------
# Through paint()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_every_piece_carries_the_datum_paint_stamps(case: str) -> None:
    """Placed through PaintContext.paint(), which records where each piece's stations came from -
    lines, fills, posts and symbols alike."""
    _, paint = build(street(AGREEING_NOMINAL_FT), CASES[case])
    placed = [p for p in paint if p.leg == LEG]
    assert placed
    missing = sorted({p.kind.name for p in placed if not p.datum})
    assert not missing, f"{case}: placed without paint(): {missing}"
    for p in placed:
        assert sum(p.datum.values()) == pytest.approx(1.0, abs=1e-9)


#: Paint that enters the list any way but PaintContext.paint(). Corner paint (hatching, aprons)
#: hangs off a corner fillet rather than one leg, and waits on a CornerZone shape - this set may
#: only shrink.
ENTERS_WITHOUT_PAINT_ALLOWED = {"src/geometry/treatments/corners.py"}
SIDE_DOORS = re.compile(r"\bctx\.add\(|\bctx\.emit\(|\bPaintPiece\(")


def test_nothing_enters_the_paint_list_except_through_paint() -> None:
    """ctx.paint() is the one door: it resolves the references and stamps the datum, so a piece
    added any other way has geometry nobody checked came off the kerb. PaintContext's own module
    and the PaintPiece type are where the door is, so they are not scanned."""
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = str(path.relative_to(root))
        if rel in {"src/geometry/paint/context.py", "src/geometry/paint/pieces.py"} | ENTERS_WITHOUT_PAINT_ALLOWED:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if SIDE_DOORS.search(line.split("#", 1)[0]):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert not offenders, f"{len(offenders)} piece(s) added without ctx.paint():\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("case", sorted(CASES))
def test_with_both_kerbs_traced_only_the_traced_kerbs_place_paint(case: str) -> None:
    _, paint = build(street(68.0, left=WOBBLE, right=WOBBLE), CASES[case])
    used = sources(paint)
    assert used, f"{case}: no datum recorded at all"
    assert used <= TRACED_SOURCES, f"{case}: placed off {sorted(used - TRACED_SOURCES)}"


# --------------------------------------------------------------------------
# Rigid at the narrowest; the kerbside zone absorbs the rest
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(LANE_EDGE))
def test_the_travel_lane_edge_is_straight_and_rigid_at_the_narrowest_kerb(case: str) -> None:
    """On a kerb pinching from 22 ft to 20 ft, the lane edge is the line a straight 22 ft kerb
    gets, moved in by what its family's rule says - at every station, not just at the pinch."""
    straight = lane_edge_offsets(case, street(AGREEING_NOMINAL_FT))
    wobbling = lane_edge_offsets(case, street(AGREEING_NOMINAL_FT, left=WOBBLE, right=WOBBLE))
    expected = float(np.mean(straight)) - LANE_EDGE[case][1]
    assert np.ptp(wobbling) <= TOL_FT, f"{case}: the lane edge wobbles {np.ptp(wobbling):.2f} ft"
    assert np.allclose(wobbling, expected, atol=TOL_FT), (
        f"{case}: lane edge at {np.mean(wobbling):.2f} ft, expected {expected:.2f} - see LANE_EDGE")


@pytest.mark.parametrize("case", sorted(CASES))
def test_on_a_wobbling_kerb_no_paint_crosses_it_and_a_fill_at_the_kerb_stays_there(case: str) -> None:
    """The kerbside zone absorbs the kerb's wobble: a fill that reaches the kerb anywhere reaches
    it all along its own span, rather than standing off it where the kerb bulges."""
    leg = street(AGREEING_NOMINAL_FT, left=WOBBLE, right=WOBBLE)
    state, paint = build(leg, CASES[case])
    assert not PaintInsideTheCurb().run(SceneContext(state=state, paint=tuple(paint)))
    kerb = leg.left_curb
    assert kerb is not None
    for p in paint:
        if p.side != Side.LEFT or p.kind.is_line or p.kind.is_object or p.geometry.is_empty:
            continue
        stations = frame_of(p.geometry, leg)[0]
        reaches = []
        for s in np.arange(np.ceil(stations.min()) + 2.0, stations.max() - 2.0, 5.0):
            on_kerb = kerb.interpolate(kerb.project(Point(s, 0.0)))
            reaches.append(p.geometry.buffer(0.05).contains(Point(on_kerb.x, on_kerb.y - 0.3)))
        assert all(reaches) or not any(reaches), (
            f"{case}: {p.kind.name} meets the kerb at some stations and stands off it at others")


# --------------------------------------------------------------------------
# Fallback, never skip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_with_nothing_traced_the_nominal_width_places_the_same_paint(case: str) -> None:
    """An agreeing nominal width and no tracing puts the kerbs where the traced street has them,
    so the paint is the same paint - and it says it came off the nominal width."""
    _, traced = build(street(AGREEING_NOMINAL_FT), CASES[case])
    _, untraced = build(street(AGREEING_NOMINAL_FT, left=None, right=None), CASES[case])
    assert same_paint(traced, untraced), f"{case}: the untraced street lost or moved paint"
    assert KerbSource.NOMINAL in sources(untraced), (
        f"{case}: untraced paint does not say it came off the nominal width")


# --------------------------------------------------------------------------
# The centre line (src/render/crosswalks.py)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["double_yellow", "single_yellow_dashed"])
def test_the_centre_stripe_runs_midway_between_the_traced_kerbs(style: str) -> None:
    """Kerbs 26 ft left and 18 ft right of the OSM line: the road's centre is 4 ft left of it."""
    left_ft, right_ft = 26.0, 18.0
    centre_ft = (left_ft - right_ft) / 2
    leg = street(AGREEING_NOMINAL_FT, left=[(0.0, left_ft), (LENGTH_FT, left_ft)],
                 right=[(0.0, right_ft), (LENGTH_FT, right_ft)])
    stripes = centerline_paint_ft(leg, CROSSING_FT, style)
    assert stripes
    means = sorted(float(np.mean(frame_of(s, leg)[1])) for s in stripes)
    if style == "double_yellow":
        half = DOUBLE_YELLOW_SEPARATION_FT / 2
        assert means == pytest.approx([centre_ft - half, centre_ft + half], abs=TOL_FT)
    else:
        assert means == pytest.approx([centre_ft] * len(means), abs=TOL_FT)


def test_with_one_kerb_traced_the_centre_stripe_follows_the_state_line() -> None:
    state_ft = 3.0
    leg = street(AGREEING_NOMINAL_FT, right=None,
                 state_line=LineString([(-50.0, state_ft), (LENGTH_FT + 50.0, state_ft)]))
    stripes = centerline_paint_ft(leg, CROSSING_FT, "single_yellow_dashed")
    assert stripes
    assert all(np.allclose(frame_of(s, leg)[1], state_ft, atol=TOL_FT) for s in stripes)


# --------------------------------------------------------------------------
# Stop bars (src/render/crosswalks.py), raised crossings (src/geometry/treatments/crossings.py)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["none", "narrowing", "parking", "bike"])
def test_a_stop_bar_reaches_the_traced_kerb_whatever_the_nominal_width(case: str) -> None:
    treatments = [] if case == "none" else CASES[case]
    agreeing, _ = build(street(AGREEING_NOMINAL_FT), treatments)
    wrong, _ = build(street(68.0), treatments)
    assert stop_bar_ends_ft(wrong, LEG) == pytest.approx(stop_bar_ends_ft(agreeing, LEG), abs=TOL_FT)
    assert stop_bar_width_ft(wrong, LEG) == pytest.approx(stop_bar_width_ft(agreeing, LEG), abs=TOL_FT)


@pytest.mark.parametrize("nominal_ft", [90.0, 40.0])
def test_parking_inside_a_kerbside_bike_lane_does_not_depend_on_the_nominal_width(
        nominal_ft: float) -> None:
    """The parking-protected section needs ~29 ft of half-width, so it gets a 30 ft street.
    bikeways/place.py sizes the parking inside it off the nominal half-width today."""
    wide_ft = 30.0
    kerb: KerbVertices = [(0.0, wide_ft), (LENGTH_FT, wide_ft)]
    treatments: list[Treatment] = [AddKerbsideBikeLane(KERB, width_ft=5.0, buffer_ft=3.0,
                                                       parking_ft=8.0)]
    _, agreeing = build(street(2 * wide_ft, left=kerb, right=kerb), treatments)
    _, wrong = build(street(nominal_ft, left=kerb, right=kerb), treatments)
    assert agreeing
    assert same_paint(agreeing, wrong)


@pytest.mark.parametrize("nominal_ft", [68.0, 30.0])
def test_a_raised_crossing_spans_the_traced_kerbs(nominal_ft: float) -> None:
    state, _ = build(street(nominal_ft), [])
    offsets = frame_of(RaiseCrossing(LegTarget(LEG)).polygon(state), state.legs[LEG])[1]
    assert offsets.min() == pytest.approx(-KERB_FT, abs=TOL_FT)
    assert offsets.max() == pytest.approx(KERB_FT, abs=TOL_FT)


# --------------------------------------------------------------------------
# The real borough
# --------------------------------------------------------------------------

@needs_source_data
def test_the_borough_proposal_paints_inside_its_kerbs() -> None:
    from scripts.render_slice import design_for, load_network
    from src.render.export import build_props
    from src.render.scene import SceneGeometry

    network = load_network("hopewell_borough")
    with contextlib.redirect_stdout(io.StringIO()):
        model, state, pavement = design_for(network, "hopewell_borough", "proposed")
        scene = SceneGeometry.resolve(model, state, pavement=pavement)
        props = build_props(model, state, scene.crosswalk_offsets, pavement=scene.pavement)
        paint, props = scene.build_paint_and_posts(props)
        context = scene.context(props, paint)
        bad = [v for check in DATUM_CHECKS for v in check().run(context)]
    assert not bad, f"{len(bad)} violation(s):\n  " + "\n  ".join(map(str, bad[:20]))
