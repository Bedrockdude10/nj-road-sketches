"""WHERE A TWO-WAY BIKEWAY STOPS, and how a rider gets on and off it there.

A bikeway on one kerb serves riders in both directions, so at each end of it HALF THE RIDERS ARE
ON THE WRONG SIDE OF THE STREET. That is not a detail of the ends - it is the whole problem of
the ends, and until this module existed `BROAD_ST_TWO_WAY_BIKEWAY` was a 4,524 ft facility with
no statement at either one. A drawing that shows a protected lane simply stopping shows riders
being tipped into traffic across an unmarked centreline.

MUTCD 11th ed. answers it, and Figure 9E-11 is titled for our exact case - "Two-Stage Turn Box
Location at an Intersection with a Two-Way Bikeway". Two markings, for the two user groups a
terminus has:

  * the rider JOINING the facility crosses the roadway and queues in a TWO-STAGE BICYCLE TURN
    BOX (9E.11), then proceeds straight into the lane. The box is green, bounded on all four
    sides by a solid white line (07), and carries a bicycle symbol and a THROUGH arrow (05, 06);
  * the rider CONTINUING past it merges into the travelled way and gets SHARED-LANE MARKINGS
    (9E.09(13), which names terminating a separated bikeway as the case they are for).

Every figure below is either read off the drawn paint or is a local clearance out of the manual.
NOTHING here is measured over the leg, because a leg's length is a rendering decision
(.claude/SKILLS.md section 0b) and a terminus that moved with the sheet would be a terminus in a
different place on every drawing.
"""
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Polygon

from src.geometry.markings import (BIKE_LANE_SYMBOL, BIKE_THROUGH_ARROW, SHARED_LANE_MARKING,
                                   TURN_BOX_EDGE_LINE, TURN_BOX_SURFACE)
from src.geometry.targets import LegSide
from src.geometry.treatments.base import Treatment
from src.geometry.treatments.bikeways.symbols import (SYMBOL_LENGTH_FT, SYMBOL_WIDTH_FT,
                                                      bike_symbol_polygon)

if TYPE_CHECKING:
    from src.geometry.intersection.junction import IntersectionModel
    from src.geometry.treatments.state import DesignState

# A BICYCLE'S LENGTH, which is what sizes the box. MUTCD 9E.11(10) gives no dimension at all - it
# asks for engineering judgment against three named factors (intersection geometry, keeping
# queued riders away from moving traffic, and peak-hour volume so the box does not overflow) - so
# the number below is OURS and is recorded as Modelled in STANDARDS.md section 7, not cited.
BICYCLE_LENGTH_FT = 6.0
# Two riders nose to tail. One is the minimum that holds anybody; three would put the back of the
# queue level with the travel lane it is meant to keep them out of, which is factor two.
TURN_BOX_QUEUE_BICYCLES = 2
TURN_BOX_LENGTH_FT = BICYCLE_LENGTH_FT * TURN_BOX_QUEUE_BICYCLES

# MUTCD 9E.09(09): sharrows spaced "not less than 50 feet or greater than 250 feet" away from
# intersections, and (10) the first one "no more than 50 feet" past one. The interval is the
# middle of the permitted band rather than either end of it - Modelled, see STANDARDS.md.
SHARROW_FIRST_FT = 50.0
SHARROW_INTERVAL_FT = 150.0
# MUTCD 9E.09(08): where there is no on-street parking and the outside lane is under 14 ft, the
# CENTRE of the marking sits at least 4 ft from the kerb face. (07) makes it 12 ft beside
# parallel parking - a door zone rather than a shy line, which is why the two figures are so far
# apart and why the parking one cannot be derived from the other.
SHARROW_CLEAR_OF_KERB_FT = 4.0
SHARROW_CLEAR_OF_PARKING_FT = 12.0
# 9E.09(03): not on a roadway posted 40 mph or more. Broad St is posted 25, so this refuses
# nothing here today - it is carried because the treatment is universal and the next corridor
# might not be (see [network-oriented build] in the memory: no per-junction special cases).
SHARROW_MAX_SPEED_MPH = 40


def lane_far_end_face(ctx, leg_name: str, side: str):
    """(station, inner offset, outer offset) where this kerb's bikeway ends AWAY from the node.

    The mirror of `through_junction.lane_end_face`, which takes the station nearest the node
    because it is joining two legs across a mouth. A terminus is the other end, and it is a
    separate function rather than a flag because the two answer different questions and a caller
    that picked the wrong one would get a plausible face at the wrong end of the street.

    READ OFF THE DRAWN PAINT, for the reason `lane_end_face` is: the lane's edges sit on one of
    two datums depending on whether the section hugs the kerb, and the far end is additionally
    bounded by `paint_stations` - how far the kerb is really traced - which no arithmetic over
    the section knows about. Rebuilt from the cross-section instead, the box would be placed
    against a lane end that is not where the lane ends.
    """
    from src.geometry.markings import BIKE_LANE_SURFACE_KINDS
    from src.geometry.model import station_offset_many

    centerline = ctx.state.legs[leg_name].centerline
    stations, offsets = [], []
    for piece in ctx.pieces:
        if piece.kind not in BIKE_LANE_SURFACE_KINDS or piece.leg != leg_name:
            continue
        if str(piece.side) != side or piece.geometry.geom_type != "Polygon":
            continue
        station, offset = station_offset_many(
            centerline, np.asarray(piece.geometry.exterior.coords, dtype=float))
        stations.append(station)
        offsets.append(offset)
    if not stations:
        return None
    stations, offsets = np.concatenate(stations), np.concatenate(offsets)
    end_ft = float(stations.max())
    # The same two-foot window and the same by-magnitude naming as lane_end_face: a lane cut by a
    # skewed crossing ends on a diagonal, so one station gives a vertex rather than a face, and
    # on a right-hand kerb both offsets are negative so a plain min/max names them backwards.
    near = offsets[stations >= end_ft - 2.0]
    if near.size < 2:
        return None
    inner_ft = float(near[np.argmin(np.abs(near))])
    outer_ft = float(near[np.argmax(np.abs(near))])
    return end_ft, inner_ft, outer_ft


def turn_box_span_ft(lane_end_ft: float, municipal_limit_ft: float | None) -> tuple:
    """(near, far) stations of the two-stage turn box at a terminus, in the leg's own frame.

    THE BOX GOES INSIDE THE LINE, NEVER PAST IT. A rider joining the facility queues on the far
    side of it and rides forward into the lane, so the box's natural place is beyond the lane's own
    end - and at a JURISDICTIONAL terminus, which is every terminus in this project, beyond the
    lane's end is beyond the boundary. Unclamped it drew 12 ft into Hopewell Township at W Broad &
    Lanning and 8.9 ft over the line at E Broad & Elm: paint outside the municipality that would
    lay it, which checks.PaintInsideTheMunicipality now refuses. Clamped, the box takes the last
    TURN_BOX_LENGTH_FT of borough street and the bikeway's own green runs under it, which
    markings.MAY_LIE_ON declares - 9E.11(12) puts green under all of the box either way.

    ONE HOME, because two things place against this box: `EndTheBikeway.paint` draws it, and the
    two-way lane's yellow divider has to STOP at it (bikeways/place.py). A queue box is where
    riders in both directions wait, so a line dividing the two directions through it is a
    statement about lanes that are not there - and the divider computed from its own end and the
    box from its own would be the two-derivations-of-one-fact this repo's SKILLS section 0a is
    about. The divider asks for the same span this draws.
    """
    far_ft = lane_end_ft + TURN_BOX_LENGTH_FT
    if municipal_limit_ft is not None:
        far_ft = min(far_ft, municipal_limit_ft)
    return far_ft - TURN_BOX_LENGTH_FT, far_ft


def terminus_box_span_ft(state: "DesignState", leg_name: str, side: str) -> tuple:
    """(near, far) stations of a terminus's turn box, read off the DESIGN rather than the paint.

    `EndTheBikeway.paint` places the box against the DRAWN lane face, which is the right datum
    for paint and is not one the signing layer can reach: src/render/props.py builds every sign
    from a DesignState with no PaintContext in hand. So this derives the same span from the
    design - the lane's own `to_ft`, or the end of the leg - and hands it to the same
    `turn_box_span_ft` under the same municipal limit.

    THE TWO AGREE WHEREVER THE BOUNDARY BINDS, which is every terminus in this project, because a
    terminus here IS the borough line: both come out at the limit to the foot. Where nothing
    binds, the lane runs to the end of what is drawn and so does this. What this may never become
    is a second copy of the 12 ft span or of the clamp - those stay in `turn_box_span_ft`, which
    is why this returns its result instead of arithmetic of its own.
    """
    from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane

    leg = state.legs.get(leg_name)
    lane_end_ft = leg.centerline.length if leg is not None else 0.0
    for lane in state.every_treatment(AddTwoWayBikeLane):
        if (lane.target.leg == leg_name and str(lane.target.side) == side
                and lane.to_ft is not None):
            lane_end_ft = min(lane_end_ft, lane.to_ft)
    return turn_box_span_ft(lane_end_ft, state.municipal_limits_ft.get(leg_name))


def _through_arrow_polygon(on, station_ft: float, centre_offset_ft: float, side: str,
                           forward: bool) -> Polygon:
    """The THROUGH arrow MUTCD 9E.11(06) requires in a two-way bikeway's turn box.

    A plain stemmed arrow, distinct from `bike_symbol_polygon`'s barbed one: the two are drawn
    side by side inside the box and a reader has to be able to tell the symbol from the arrow, or
    the box looks like it carries two of the same marking. Same schematic register as every other
    stencil here - this pipeline positions paint, it does not draw glyph art (see
    markings.BIKE_LANE_SYMBOL_POLYGONS).

    THROUGH and not a turn arrow, which is the one thing 9E.11(06) is explicit about: a turn
    arrow is the one-way-lane case. A rider in this box is not turning - they have already
    crossed, and what they do next is carry straight on into the bikeway.
    """
    from src.geometry.model import place_in_measured_frame

    sign = 1.0 if side == "left" else -1.0
    nose = SYMBOL_LENGTH_FT / 2 * (1.0 if forward else -1.0)
    tail, head = -nose, nose * 0.45
    half, stem = SYMBOL_WIDTH_FT / 2, SYMBOL_WIDTH_FT / 8
    outline = [(nose, 0.0), (head, half), (head, stem), (tail, stem),
               (tail, -stem), (head, -stem), (head, -half)]
    stations = np.array([station_ft + along for along, _across in outline])
    offsets = np.array([sign * (centre_offset_ft + across) for _along, across in outline])
    return Polygon([tuple(point) for point in
                    place_in_measured_frame(on.centerline, stations, offsets)])


def _sharrow_polygons(on, station_ft: float, centre_offset_ft: float, side: str,
                      forward: bool) -> list[Polygon]:
    """The shared-lane marking: the bicycle glyph with two chevrons ahead of it.

    THREE pieces rather than one, because that is what the marking is - MUTCD Figure 9E-9 shows a
    bicycle under a pair of chevrons, and a striper lays three stencils. Returned as a list so
    each reaches the renderers as its own polygon; merged into one shape the chevrons' gaps would
    close and the marking would read as a solid blob at sheet scale.

    The glyph is `bike_symbol_polygon`'s, deliberately: a sharrow's bicycle is the SAME bicycle as
    a bike lane's (9E.09(01) points at the same Standard Highway Signs plate), and drawing two
    different schematic bicycles would invent a distinction the street does not have. What makes
    it a sharrow is the chevrons.
    """
    from src.geometry.model import place_in_measured_frame

    sign = 1.0 if side == "left" else -1.0
    ahead = 1.0 if forward else -1.0
    out = [bike_symbol_polygon(on, side, station_ft, centre_offset_ft, forward)]
    half, thick = SYMBOL_WIDTH_FT / 2, SYMBOL_WIDTH_FT / 5
    for index in (1, 2):
        base = station_ft + ahead * (SYMBOL_LENGTH_FT / 2 + index * thick * 2.2)
        tip = base + ahead * thick * 1.8
        outline = [(tip, 0.0), (base, half), (base, half - thick), (tip - ahead * thick, 0.0),
                   (base, -(half - thick)), (base, -half)]
        stations = np.array([along for along, _across in outline])
        offsets = np.array([sign * (centre_offset_ft + across) for _along, across in outline])
        out.append(Polygon([tuple(point) for point in
                            place_in_measured_frame(on.centerline, stations, offsets)]))
    return out


@dataclass(frozen=True)
class EndTheBikeway(Treatment):
    """The terminus: a two-stage turn box onto the facility, and sharrows past it.

    APPLIED TO THE LEG-SIDE THE BIKEWAY IS ON, and it places nothing unless a bikeway is actually
    drawn there - a terminus is a statement about a facility, so with no facility it is a
    statement about nothing. It refuses rather than drawing an orphan box, because a green box
    with a bicycle in it beside a street with no bike lane is a worse drawing than a blank one.

    WHY BOTH MARKINGS AT ONE PLACE, when they serve opposite movements. A rider arriving at the
    end of the borough in the general travel lane wants ON: that is the box. A rider leaving the
    bikeway and carrying on out of the borough is back in mixed traffic the moment the green
    stops, and 9C.07(01) says what the sign there means - "bicycles will share or occupy the
    travel lane after merging". 9E.09(13) is the pavement half of the same sentence. Drawing only
    one of the two would answer half the riders at a place where both are standing.

    THE BOX GOES OUTBOARD OF THE LANE'S LAST FOOT, in line with it, and that is read off the
    drawn green (`lane_far_end_face`) rather than computed from the section. 9E.11(04)(A) puts it
    between the through movement and the parallel crosswalk; at a terminus the through movement
    IS the bikeway and the ground just past its end is that place.
    """
    paint_group: ClassVar[int] = 60      # after the lane, whose drawn end this is measured off
    target: LegSide

    #: Whether the kerb the sharrows sit beside carries parallel parking, which moves them from
    #: 4 ft off the kerb to 12 - MUTCD 9E.09(07) against (08). A door zone, not a shy line.
    #: Passed in rather than re-derived here: the design already decided it, and deriving it a
    #: second time is the two-derivations-of-one-fact defect .claude/SKILLS.md section 0 is about.
    beside_parking: bool = False
    #: The posted speed, for 9E.09(03)'s 40 mph ceiling on sharrows. None means the config did not
    #: say, and an unknown speed does not license the marking - it refuses.
    speed_limit_mph: int | None = None

    def describe(self) -> str:
        return (f"EndTheBikeway({self.target}): a two-stage bicycle turn box onto the facility "
                f"(MUTCD 9E.11) with shared-lane markings past its end (9E.09)")

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str | None:
        from src.geometry.treatments.bikeways.place import AddBikeLane

        leg, side = self.target.leg, str(self.target.side)
        carries = any(isinstance(t, AddBikeLane) and getattr(t.target, "leg", None) == leg
                      and str(getattr(t.target, "side", "")) == side
                      for t in state.treatments)
        if not carries:
            raise ValueError(
                f"EndTheBikeway on {leg} {side}, which carries no bike lane - there is no "
                f"facility here for a terminus to be the end OF. A two-stage turn box beside a "
                f"street with no bikeway tells a rider to cross to nothing. Apply this to the "
                f"leg-side the corridor's facility actually reaches (route_decision_for names "
                f"which street, and the section may have refused this leg - read state.notes).")
        return (f"queue box {TURN_BOX_LENGTH_FT:.0f} ft deep, sharrows "
                f"{SHARROW_CLEAR_OF_PARKING_FT if self.beside_parking else SHARROW_CLEAR_OF_KERB_FT:.0f}"
                f" ft off the kerb")

    def paint(self, ctx) -> None:
        from src.geometry.model import place_in_measured_frame

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        face = lane_far_end_face(ctx, leg_name, side)
        if face is None:
            # The lane was refused on this leg-side, or every foot of it was clipped away. Not an
            # error - apply_to already checked a lane was ASKED for, and a lane that asked and got
            # nothing is a finding the section records. Drawing a box against a lane that is not
            # there is the failure this guard exists for.
            return
        end_ft, inner_ft, outer_ft = face
        # THE BOX IS THE LANE'S OWN CROSS-SECTION CARRIED ON, so it cannot be a different width
        # from the lane it feeds. Four corners from the two offsets the face gave, which come back
        # ALREADY SIGNED in the leg's frame (see lane_far_end_face) - so they go into
        # place_in_measured_frame as they are, and only the stencils below, which are built in the
        # lane's own terms, need the side's sign put back on.
        # THE BOX GOES INSIDE THE LINE, NEVER PAST IT. A rider joining the facility queues on the
        # far side of it and rides forward into the lane, so the box's natural place is beyond the
        # lane's own end - and at a JURISDICTIONAL terminus, which is every terminus in this
        # project, beyond the lane's end is beyond the boundary. Unclamped this drew the box 12 ft
        # into Hopewell Township at W Broad & Lanning and 8.9 ft over the line at E Broad & Elm:
        # paint outside the municipality that would lay it. Clamped, the box takes the last
        # TURN_BOX_LENGTH_FT of borough street and the bikeway's green runs under it, which
        # src/geometry/markings.py:MAY_LIE_ON declares - 9E.11(12) puts green under all of the box.
        limit_ft = ctx.state.municipal_limits_ft.get(leg_name)
        near_ft, far_ft = turn_box_span_ft(end_ft, limit_ft)
        corners = [(near_ft, inner_ft), (far_ft, inner_ft), (far_ft, outer_ft), (near_ft, outer_ft)]
        placed = place_in_measured_frame(leg.centerline,
                                         np.array([s for s, _ in corners]),
                                         np.array([o for _, o in corners]))
        box = Polygon([tuple(point) for point in placed])
        if not box.is_valid or box.area <= 0:
            return
        ctx.add(TURN_BOX_SURFACE, box, leg_name, side)
        # 9E.11(07), "bounded on all sides by a solid white line" - so the boundary is the box's
        # own ring and not four separately-built stripes. One derivation, and it cannot come
        # adrift from the shape it bounds.
        ctx.add(TURN_BOX_EDGE_LINE, LineString(box.exterior.coords), leg_name, side)

        # 9E.11(05): at least one bicycle symbol AND at least one arrow. Both face INWARD - a
        # rider in this box has finished crossing and is about to ride into the bikeway, so the
        # direction they are pointed is toward the junction, which is decreasing station.
        # ON THE KERB HALF OF THE BOX, NOT DOWN ITS MIDDLE. The box now sits inside the bikeway
        # (see the clamp above), and the middle of a two-way bikeway is where its yellow divider
        # runs - stencils centred there came out with dashes painted across a through arrow and a
        # bicycle symbol, 1.5 sq ft of stripe on 100% of one dash, which MarkingsDoNotCollide
        # reported and was right to. The half to move them into is not a guess: 9E.11(10)'s second
        # factor for siting a box is keeping queued riders clear of moving traffic, and a rider who
        # has just crossed the street arrives at the kerb. 2.4 ft of stencil centred in a 4 ft half
        # clears the 0.49 ft divider by 0.8 ft.
        centre_ft = abs(inner_ft + 3 * outer_ft) / 4
        symbol_at = near_ft + TURN_BOX_LENGTH_FT * 0.72
        arrow_at = near_ft + TURN_BOX_LENGTH_FT * 0.26
        ctx.add(BIKE_LANE_SYMBOL,
                bike_symbol_polygon(leg, side, symbol_at, centre_ft, forward=False),
                leg_name, side)
        # DOWNSTREAM OF THE SYMBOL, which 9E.01(04) and 9E.07(11) both specify - and downstream
        # for a rider facing inward is the LOWER station, which is why the arrow's station is the
        # smaller of the two and not the larger.
        ctx.add(BIKE_THROUGH_ARROW,
                _through_arrow_polygon(leg, arrow_at, centre_ft, side, forward=False),
                leg_name, side)

        # THE SHARROWS, past the box, in the travelled way. Refused outright above 9E.09(03)'s
        # ceiling and refused on an unknown speed, because a marking whose own guidance turns on
        # a number nobody stated is a marking placed on an assumption.
        if self.speed_limit_mph is None or self.speed_limit_mph >= SHARROW_MAX_SPEED_MPH:
            return
        clear_ft = (SHARROW_CLEAR_OF_PARKING_FT if self.beside_parking
                    else SHARROW_CLEAR_OF_KERB_FT)
        # Measured from the KERB FACE, which is what 9E.09(07)-(08) say and is not the alignment:
        # the two differ by half the roadway and the manual's 4 ft would land in the opposite
        # travel lane read off the wrong datum. narrowest_half_width_ft is the traced-kerb answer
        # (.claude/SKILLS.md section 2) and falls back to the nominal where nothing is traced,
        # which is exactly the right behaviour on a leg drawn past the end of the tracing.
        from src.geometry.model import narrowest_half_width_ft

        half_ft = narrowest_half_width_ft(leg, side)
        if half_ft is None or half_ft <= clear_ft:
            return
        across_ft = half_ft - clear_ft
        station = far_ft + SHARROW_FIRST_FT
        # AND THE SHARROWS STOP AT THE LINE TOO. They mark the travelled way a rider continues
        # into past the facility's end, which at a jurisdictional terminus is somebody else's
        # travelled way. The drawn leg alone does not bound them: at 2.5x it reaches 195 ft past
        # the borough line, and that is exactly where the first one would land.
        reach_ft = leg.centerline.length if limit_ft is None else min(leg.centerline.length,
                                                                      limit_ft)
        while station + SYMBOL_LENGTH_FT <= reach_ft:
            for piece in _sharrow_polygons(leg, station, across_ft, side, forward=True):
                ctx.add(SHARED_LANE_MARKING, piece, leg_name, side)
            station += SHARROW_INTERVAL_FT


# MUTCD 9C.07(01) requires the W9-5 BIKE LANE ENDS sign "in advance of" the end of a bike lane
# and gives no distance - the manual leaves it to engineering judgment, so this is OURS and is
# recorded as Modelled in STANDARDS.md section 7. Two seconds of reading and reacting at 25 mph
# is ~74 ft; 100 ft rounds that up and still sits inside a short borough block.
#
# A LOCAL clearance measured back from the lane's own end, never a fraction of the leg - a leg
# is as long as the sheet is wide (.claude/SKILLS.md section 0b), and a sign that moved with the
# render frame would stand in a different place on every drawing of the same street.
BIKE_LANE_ENDS_ADVANCE_FT = 100.0

#: The plan/render prop types these signs are drawn as. There is no per-legend geometry at this
#: scale: a W-series plate is a yellow diamond and an R-series plate is a white rectangle, which
#: is what the render can honestly carry, so the MUTCD code lives in the prop's `note` - it is
#: what the plan sheet's label and the export's provenance string are read from.
WARNING_SIGN = "bike_warning_sign"
REGULATORY_SIGN = "bike_regulatory_sign"


def bikeway_sign_entries(state: "DesignState") -> list[dict]:
    """The MUTCD signing a two-way bikeway needs, as `props.extra`-shaped entries.

    Built here rather than in src/render/props.py because WHICH sign and WHY is a standards
    question and this module is where the standards rows for it live; props.py owns only WHERE
    on the leg a sign physically fits. Three signs, for three different readers:

      * W9-5 BIKE LANE ENDS (9C.07(01)), in advance of each terminus, for the rider about to
        run out of facility;
      * R9-23 series (9B.18(04)), at the two-stage turn box, for the rider arriving from the
        other kerb who has to be told the box is theirs and how it works;
      * W16-21P TWO-WAY BICYCLE CROSS TRAFFIC (9C.06(01)), on every crossroad approach, for the
        driver who is about to look ONE way for bicycles on a street that has them coming from
        both. 9C.06(04) makes it a plaque, mounted under the crossroad's existing STOP or YIELD;
        it is emitted as its own prop because this repo models a plate on a post and not a
        second plate bolted to one, and standing it at the same near-corner station as the stop
        sign is what keeps the two together.

    A crossroad is identified from the DESIGN and not from a street name: the legs carrying a
    two-way lane are the bikeway's own street here, and every other leg crosses it. The caveat
    is that a leg the section REFUSED reads as a crossroad and picks up a cross-traffic plaque
    it does not need - which is the harmless direction of that error, and visible on the sheet.
    """
    from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane

    entries = []
    for end in state.every_treatment(EndTheBikeway):
        leg, side = end.target.leg, str(end.target.side)
        # The BOX, not the leg. Measured off the leg's length these two signs stood where the
        # sheet happened to end - the R9-23 labelling a queue box 2,265 ft away at W Broad &
        # Lanning, and the W9-5 at a station that moves with the frame scale, which is the
        # extent-may-not-move rule in .claude/SKILLS.md section 0b.
        box_near_ft, _box_far_ft = terminus_box_span_ft(state, leg, side)
        entries.append({
            "leg": leg, "type": WARNING_SIGN, "side": side,
            "offset_ft": max(0.0, box_near_ft - BIKE_LANE_ENDS_ADVANCE_FT),
            "note": f"MUTCD W9-5 BIKE LANE ENDS, {BIKE_LANE_ENDS_ADVANCE_FT:.0f} ft in advance of "
                    f"the end of the two-way bikeway (9C.07(01); the advance distance is modelled, "
                    f"the manual gives none).",
        })
        entries.append({
            "leg": leg, "type": REGULATORY_SIGN, "side": side, "offset_ft": box_near_ft,
            "note": "MUTCD R9-23 series at the two-stage bicycle turn box (9B.18(04)), standing "
                    "at the box's near edge - a rider reaches the sign and the box together.",
        })

    bikeway_legs = {t.target.leg for t in state.every_treatment(AddTwoWayBikeLane)}
    if bikeway_legs:
        for leg in state.legs:
            if leg in bikeway_legs:
                continue
            entries.append({
                # NO `side`: which kerb an approaching driver's sign stands on is a placement
                # fact and props.py owns it (APPROACHING_DRIVER_RIGHT), the same one the STOP
                # this plaque hangs under already uses. Naming the side here would be a second
                # copy of it, free to drift from the sign it is supposed to sit beneath.
                "leg": leg, "type": WARNING_SIGN, "offset_ft": None,
                "note": "MUTCD W16-21P TWO-WAY BICYCLE CROSS TRAFFIC (9C.06(01),(04)), under this "
                        "approach's STOP: the bikeway on the crossing street carries riders in "
                        "both directions past this mouth.",
            })
    return entries
