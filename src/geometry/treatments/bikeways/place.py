"""THE TREATMENTS THAT PLACE A BIKEWAY, and all the paint one puts down.

`AddTwoWayBikeLane` subclasses `AddBikeLane` and they are one file for that reason: painting the
section is the same problem at different offsets, and the base class asks `isinstance(self,
AddTwoWayBikeLane)` where the two differ. Splitting an inheritance pair across modules buys a
smaller file and a lazy import, which is a worse trade than reading 500 lines.

THE PAINT COMES OUT THROUGH `PaintContext.emit`, never appended directly, so every piece is
clipped against the crossings and held inside the traced kerb by one code path.
"""
from dataclasses import dataclass, replace
from typing import ClassVar
import numpy as np
import shapely.ops
from src.geometry.targets import Side
from src.geometry.model import narrowest_half_width_ft
from src.geometry.treatments.base import (ANGLED_STALL_WIDTH_FT, LANE_WIDTH_SLACK_FT,
                                          TARGET_LANE_WIDTH_FT, Treatment)
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.sections import (KerbsideBikeLane,BikeLane, CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                                       MIN_TWO_WAY_BIKE_LANE_FT, NJDOT_TWO_WAY_OBJECTION,
                                                       TWO_WAY_BIKE_LANE_WIDTH_FT, TwoWayBikeLane, _feet)
from src.geometry.treatments.bikeways.fit import (far_kerb_surplus_ft,
                                                 governing_half_widths_ft,
                                                 travel_lane_divider_shift_ft)
from src.geometry.treatments.bikeways.symbols import (CONTRAFLOW_DASH_FT, CONTRAFLOW_GAP_FT,
                                                      SYMBOL_CLEAR_OF_DIVIDER_FT, SYMBOL_LENGTH_FT,
                                                      SYMBOL_WIDTH_FT,
                                                      bike_symbol_polygon, bike_symbol_stations_ft)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# How far behind the junction node a THROUGH-RUNNING kerb's paint starts, so the two legs' halves
# overlap and fuse instead of each stopping at its own station 0. Enough to cover the 1.28 ft seam
# at W Broad & Louellen with margin, and bounded by the tracing either way, so a kerb traced right
# up to the node overlaps by this much and one traced only from the node out overlaps not at all
# and is no worse off than before.
THROUGH_JUNCTION_OVERLAP_FT = 3.0


@dataclass(frozen=True)
class AddBikeLane(Treatment):
    """Mark an exclusive bike lane along one side of a leg. Paint only - no kerb moves.

    LaneNarrowing cannot express this. It paints a BUFFER: a hatched strip of spare
    asphalt between the travel lane and the kerb, saying "nothing belongs here". A bike lane
    says the opposite about the same ground - that a specific vehicle belongs in it - so it
    needs its own edge line on both sides and its own reserved width, and where it is
    parking-protected it also needs the parking lane to sit OUTSIDE it rather than against the
    kerb in the ordinary way.

    Refused rather than shrunk when the leg cannot hold the cross-section asked for. The point
    of the exercise is to find out which legs can take a bike lane, and a lane quietly narrowed
    to fit answers a different question - see AASHTO_MIN_BIKE_LANE_FT.

    Measured against the NARROWEST point of the traced kerb, not the nominal half-width, because
    a bike lane is a promise about a whole leg and the two figures differ by feet. broad_st_east
    is 52.0 ft nominal - 26.0 per side - and its kerbs come within 22.8 ft of the alignment
    somewhere along the traced run; a cross-section sized off the nominal number would be drawn
    over the kerb there. This is what turns "verify before promising it corridor-wide" from a
    caveat into a refusal.

    The cross-section itself (BikeLane) validates its own widths, and this validates the fit
    against the street - which needs the design, so it happens in apply_to rather than in
    __post_init__. Both refusals are ValueErrors carrying the measurement that caused them.
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 30
    paint_rank: ClassVar[int] = 0
    width_ft: float = 0.0
    buffer_ft: float = 0.0
    parking_ft: float = 0.0
    shy_ft: float = 0.0
    #: The last station this lane occupies, or None for "as far as the kerb is traced".
    #:
    #: A LANE IS A STRETCH OF STREET, NOT A WHOLE APPROACH, and leaving that implicit is what let
    #: one station deny a facility 390 ft long. The section is sized over exactly this span
    #: (narrowest_half_width_ft takes it), painted over exactly this span, and the posts stood
    #: over exactly this span, so the three cannot disagree about how far the promise runs -
    #: which is the failure BikewayReachesTheEndOfItsKerb was written for, where green stopped at
    #: 180 ft under posts and a centre stripe that both ran 425.
    #:
    #: THE NEAR END IS NOT A FIELD. Where a kerbside marking STARTS is already decided by four
    #: things that disagree (SKILLS 0a) and resolved in paint(); a fifth, carried on the
    #: treatment, would be a limiter with no measurement behind it. A lane that cannot reach the
    #: junction is not a shorter lane, it is a corridor that breaks there - see
    #: CorridorFacility._place_on, which refuses that case by name instead of moving the start.
    to_ft: float | None = None
    #: Whether the rider travels OUTWARD along the leg, away from the junction.
    #:
    #: True for every ordinary case and that is why it is the default: a with-traffic lane sits
    #: on the leg's right, and a leg's right side is the one traffic leaves the junction on. It
    #: is False on the far leg of a ONE-WAY street, where the whole carriageway runs one compass
    #: way, so the approach that points the other way carries its riders INWARD. Nothing in a
    #: leg's own frame can tell: both legs' bearings point outward by construction. Ask
    #: leg_heads_toward(leg, <the direction the street runs>) and pass the answer.
    #:
    #: It only turns the stencil round, which is precisely why it is worth carrying - the arrow
    #: is the one marking whose whole job is to tell a driver at the mouth which way the rider
    #: bearing down on them is coming from, so an arrow that is merely decorative is worse than
    #: none. NJ 35 NB drew its southern approach pointing at the oncoming rider.
    runs_outward: bool = True
    #: The angle this lane's marked parking leans off the kerb, or None for parallel parking.
    #: Passed to the cross-section, which is where it belongs - see BikeLane.parking_angle_deg.
    #: `parking_ft` remains the bay's DEPTH and at an angle that is not 8 ft any more:
    #: angled_stall_depth_ft computes it, and a bay measured in the field is declared as measured.
    parking_angle_deg: float | None = None
    #: The angled stall's width across the car - what the pitch along the kerb divides. Read only
    #: when `parking_angle_deg` is set.
    parking_stall_width_ft: float = ANGLED_STALL_WIDTH_FT
    #: Whether to pin this section to ITS OWN kerb and let the travel way take the remainder,
    #: rather than starting it a target lane width from the alignment.
    #:
    #: OPT-IN, and it has to be, because it is a real design decision about the STREET and not an
    #: accommodation this treatment may quietly make: it moves the traffic. Off, a section too
    #: wide for its half of the road is refused - which is correct wherever the other kerb is
    #: also spoken for, and wrong wherever the other kerb has surplus that only this side can
    #: use. NJ 35 NB is the second: 60-degree angled parking on both kerbs leaves 29.37 ft for
    #: two lanes, so the bike lane comes out of two 14.7 ft travel lanes and all of that surplus
    #: is on one side of the alignment. See BikeLane.near_half_ft, which is what this sets.
    pin_to_kerb: bool = False

    @property
    def lane(self) -> BikeLane:
        """The cross-section this treatment marks - validated on construction, and askable
        without a design, which is how every width in it is tested."""
        return BikeLane(width_ft=self.width_ft, buffer_ft=self.buffer_ft,
                         parking_ft=self.parking_ft, shy_ft=self.shy_ft,
                         parking_angle_deg=self.parking_angle_deg,
                         parking_stall_width_ft=self.parking_stall_width_ft)

    def section(self, state: "DesignState") -> BikeLane:
        """The cross-section to paint, given the design.

        A hook, because a two-way lane's section depends on BOTH of the leg's half-widths and so
        cannot be known without the state, while a one-way lane's is fixed at construction. Both
        views and every check read the section through here, so a subclass cannot end up
        validated against one cross-section and drawn at another.

        `pin_to_kerb` needs the same two half-widths a two-way lane does, and measures them
        through the same one home (governing_half_widths_ft) over the same span - dataclasses.
        replace rather than a constructor call, so a SUBCLASS's section stays its own class.
        """
        if not self.pin_to_kerb:
            return self.lane
        leg = state.legs[self.target.leg]
        near_ft, far_ft = governing_half_widths_ft(leg, str(self.target.side), to_ft=self.to_ft)
        return replace(self.lane, near_half_ft=near_ft, far_half_ft=far_ft)

    def __post_init__(self):
        self.lane   # noqa: B018 - evaluated for its exception: raises for a lane under AASHTO's minimum

    def describe(self) -> str:
        # leg, side rather than str(target): a note is meant to read as the constructor call
        # that produced it, so someone reading a render's provenance can paste it back.
        #
        # A DECIMAL WHERE THERE IS ONE. Rounded to whole feet this reported E Broad's narrowed
        # protected lane as "4 ft lane" when it is 4.49 - understating a width by half a foot in
        # the one line a reader would check it against.
        return (f"AddBikeLane({self.target.leg}, {self.target.side}): {_feet(self.width_ft)} ft lane"
                + (f", {_feet(self.buffer_ft)} ft buffer" if self.buffer_ft else "")
                + (f", parking-protected behind {self.parking_ft:.0f} ft of marked parking"
                   if self.parking_ft
                   else f", {self.shy_ft:.1f} ft shy of the kerb" if self.shy_ft else "")
                + self._extent())

    def _extent(self) -> str:
        """The span in the note, and SILENT where there is none.

        A shortened facility is the one thing about a bikeway a reader cannot see is deliberate -
        that is the whole argument of BikewayReachesTheEndOfItsKerb - so the note has to say it.
        Silent at the default so a whole-leg lane's provenance line does not gain a clause
        restating "all of it".
        """
        return "" if self.to_ft is None else f", ending at station {self.to_ft:.0f} ft"

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        leg = state.legs[self.target.leg]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {self.target.leg!r} has no width - nothing to fit a bike lane into.")
        if self.pin_to_kerb:
            # PINNED, so "does this fit its own kerb" is not the question - it fits by
            # construction, since the section IS measured from that kerb. What can still fail is
            # what is left for traffic, and BikeLane raises on exactly that when the section is
            # built, carrying its own measurement. Reraised untouched.
            lane = self.section(state)
            travel_way_ft = lane.near_half_ft + lane.far_half_ft - lane.section_ft
            other = Side(str(self.target.side)).other
            # "TO THE FAR KERB", not "for traffic". What is left over here is everything
            # between this section and the opposite kerb, and on a street with parking over
            # there too that is not the travel way - it was reported as 42.02 ft of traffic
            # lane on a street whose two lanes measure 11 ft each, because the far kerb's own
            # 20.09 ft bay is a separate treatment this section cannot see.
            return (f". Pinned to the {self.target.side} kerb, spending {lane.section_ft:.2f} ft "
                    f"of this leg's {lane.near_half_ft + lane.far_half_ft:.2f} ft between kerbs "
                    f"at its narrowest, leaving {travel_way_ft:.2f} ft to the {other} kerb for "
                    f"the travel lanes and whatever that kerb is given, with the travel way "
                    f"shifted {lane.divider_shift_ft():.2f} ft toward it.")
        lane = self.lane
        available_ft = narrowest_half_width_ft(leg, str(self.target.side), to_ft=self.to_ft)
        if lane.total_ft > available_ft + LANE_WIDTH_SLACK_FT:
            raise ValueError(
                f"{self.target.leg} {self.target.side} comes within {available_ft:.2f} ft of the "
                f"centerline at its narrowest traced point ({leg.curb_to_curb_ft / 2:.2f} ft "
                f"nominal), and this cross-section needs {lane.total_ft:.2f} ft "
                f"({TARGET_LANE_WIDTH_FT:.0f} travel + {self.buffer_ft:.1f} buffer + "
                f"{self.width_ft:.1f} bike + {self.parking_ft:.1f} parking + {self.shy_ft:.1f} "
                f"shy). Short by {lane.total_ft - available_ft:.2f} ft.")
        spare_ft = available_ft - lane.total_ft
        return (f". Uses {lane.total_ft:.1f} of the {available_ft:.1f} ft this leg has at its "
                f"narrowest" + (f", {spare_ft:.1f} ft spare." if spare_ft > 0.05 else "."))


    def paint(self, ctx) -> None:
        """An edge line each side of the lane, so it reads as a lane rather than as the spare
        asphalt a lane-narrowing buffer marks; the buffer beside it, and the parking outside it,
        hatched and ticked with the machinery already here."""
        from src.geometry.markings import (BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE,
                                           BIKE_LANE_SURFACE, BIKE_LANE_SYMBOL, BUFFER_EDGE_LINE,
                                           BUFFER_FILL, DAYLIGHT_EDGE_LINE, DAYLIGHT_FILL,
                                           STALL_DIVIDER)
        from src.geometry.model import (band_from_offsets, curbside_strip_polygon, inset_line_ft,
                                        kerb_inset_offsets, kerb_parallel_line_ft,
                                        kerb_referenced_band_polygon, lane_narrowing_polygons_ft,
                                        offset_band_polygon, paint_stations,
                                        parking_stall_lines_ft, stall_lane_runs_ft)
        from src.geometry.daylighting import merged_no_parking_spans_ft, no_parking_zones_ft
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, MIN_LINE_LENGTH_FT, _one,
                                        end_against_crossing, parking_runs)

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        lane = self.section(ctx.state)
        at = ctx.anchors(leg_name, side, inner_offset_ft=(
            lane.kerbside_inner_offset_ft(leg.curb_to_curb_ft / 2)))
        # A bike lane RUNS INTO its crossing and is cut by it, like every other kerbside zone
        # here - a real one carries on to the crossing and often across it. Stopping it at the
        # corner clearance instead left the buffer 5.5 ft short of the crossing, which
        # test_curbside_paint_ends_against_its_crossing reads as hatching that gave up early.
        through = (leg_name, side) in ctx.straight_through
        if through:
            # BEHIND THE NODE, not up to it. Each leg's paint is built in its own frame, so two
            # halves that both stop at their own station 0 stop just shy of each other - 1.28 ft
            # of hole at W Broad & Louellen, in the middle of a lane whose whole point is running
            # continuously through the junction. Starting behind it makes the two overlap, and the
            # overlap is deduped by shares_a_kerb below, which is the same mechanism that keeps two
            # zones on one through kerb from double-painting. Honoured only as far as the kerb is
            # really traced there - see model.paint_stations.
            start_ft, beyond_ft = -THROUGH_JUNCTION_OVERLAP_FT, None
        elif leg_name in ctx.marked:
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        bounds = lane.offsets_from_centerline_ft()
        kerb = lane.offsets_from_kerb_ft()
        # THE FLOOR IS CAPPED AT THE ROOM THERE IS, and this is the ONLY thing the kerb is
        # allowed to do to the section. floor_ft says "never come inward of the design", which is
        # sound only while the design fits: the test that granted this section measures the TRAVEL
        # WAY, so a section can be granted whose outer stripe sits outside the near kerb - 24.16 ft
        # of section against a kerb 20.32 ft out on W Broad's southwest approach. Held at that
        # floor the lane is drawn over the kerb; capped here it follows the kerb inward instead.
        #
        # WHAT IT MAY NOT DO IS END THE FACILITY. There was a second guard here that stopped every
        # band at the first station where the section no longer fit, on the reasoning that a
        # section is only known to fit over the span it was sized on. It amputated two legs:
        # broad_st_east carried green for 180 ft of a 425 ft leg, under 42 flex posts and a centre
        # stripe that both ran the full length, because the sizing span was 1x and the drawing was
        # 2.5x. Sizing over the whole drawn leg is what actually fixes that (see
        # narrowest_half_width_ft) - a facility that fits the street it is drawn on needs no stop
        # station, and one that does not fit is a design decision for the rung ladder to take, not
        # a length for this method to trim. BikewayReachesTheEndOfItsKerb is the check that says so.
        room_ft = narrowest_half_width_ft(leg, side, max(start_ft, 0.0), self.to_ft)
        floor = {key: None if offset_ft is None else min(offset_ft, room_ft)
                 for key, offset_ft in bounds.items()}
        # Every stripe at its own CENTRE, which BikeLane has already offset half a stripe out
        # from the face it marks - so the travel lane keeps its 11 ft and the bike lane keeps
        # its own width, and the paint comes out of the buffer between them. Getting this wrong
        # is not subtle: an edge line centred on the mark leaves a 10.59 ft lane, which
        # PaintClearOfTheTravelLane reports on every vertex.
        #
        # WHICH DATUM EACH BOUNDARY IS MEASURED FROM. The travel lane's edge always comes off the
        # alignment, so the lane holds TARGET_LANE_WIDTH_FT whatever the kerb does. The lane's own
        # two edges come off the KERB where this section hugs it (see BikeLane.hugs_kerb) and off
        # the alignment where it does not - one branch, so a section cannot end up with its green
        # on one datum and its stripes on the other.
        def lane_edge_line(key, from_ft, to_ft=None):
            if lane.hugs_kerb:
                # floor_ft is this stripe's own designed offset, so it follows the kerb OUTWARD
                # and never comes in tighter than the section - see kerb_inset_offsets.
                return (kerb_parallel_line_ft(leg, side, kerb[key], from_ft, to_ft,
                                               floor_ft=floor[key])
                        if kerb[key] is not None else None)
            return (inset_line_ft(leg, side, bounds[key], from_ft, to_ft,
                                   keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)
                    if bounds[key] is not None else None)

        def lane_surface(from_ft, to_ft=None):
            if lane.hugs_kerb:
                return kerb_referenced_band_polygon(leg, side, kerb["bike_outer_ft"],
                                                     lane.width_ft, from_ft, to_ft,
                                                     floor_ft=floor["bike_outer_ft"])
            return offset_band_polygon(leg, side, bounds["bike_inner_ft"], bounds["bike_outer_ft"],
                                        from_ft, to_ft,
                                        keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)

        # THE LANE'S OWN FOOTPRINT, REGISTERED BEFORE ANYTHING ON THIS KERB IS PAINTED. Every
        # marking that crosses an entrance breaks at the stations this shape gives, so the green
        # marks land between the white ones instead of each being dashed along its own length and
        # drifting out of phase. The surface is canonical because the lines are its edges.
        #
        # What then happens at each entrance is markings.AT_AN_OPENING's answer, not this
        # treatment's: the lane's lines and its green go dotted across, the hatched buffer beside
        # it sweeps away from the mouth instead. This method used to re-lay each of those marks
        # itself, in a loop that only the bikeways had - which is why a lane's markings were
        # carried across a driveway and nothing else's ever could be.
        surface = lane_surface(start_ft, self.to_ft)
        ctx.dash_phase(leg_name, side, surface)
        ctx.add(BIKE_LANE_EDGE_LINE,
                 inset_line_ft(leg, side, bounds["inner_line_ft"], start_ft, self.to_ft,
                                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                 leg_name, side, beyond_ft, shares_a_kerb=through)
        # buffer_inner_line_ft is None on BikeLane, whose buffer is against the travel lane and
        # so is already bounded by inner_line_ft above; KerbsideBikeLane's buffer sits out beside
        # the parking and needs its own stripe on that side. lane_edge_line returns None for a
        # None offset, so the loop is the same loop.
        for key in ("buffer_inner_line_ft", "buffer_outer_line_ft", "outer_line_ft"):
            ctx.add(BIKE_LANE_EDGE_LINE, lane_edge_line(key, start_ft, self.to_ft), leg_name,
                     side, beyond_ft,
                     shares_a_kerb=through)
        # THE LANE'S OWN ASPHALT, PAINTED GREEN - between the two edge stripes, i.e. exactly the
        # width a rider gets. Bounded by the stripes' faces rather than their centres, so the
        # green stops where the white starts instead of running under it; MarkingsDoNotCollide
        # would report the overlap if it did, since a colour covers ground like a hatch does.
        #
        # offset_band_polygon, because the lane's own two offsets are what define it. Built as a
        # difference of two kerb-referenced strips instead, the green ran 6.6 ft past its outer
        # stripe wherever the kerb is unmapped - see that function.
        #
        # Through ctx.add like every other marking, NOT ctx.add_surface: a surface is built
        # ground that everything else is cut around (seal_surfaces), and colouring the lane must
        # not cut the lane's own edge lines - or the buffer hatching beside it - back out.
        ctx.add(BIKE_LANE_SURFACE, surface, leg_name, side, beyond_ft, shares_a_kerb=through)
        # THE BIKE LANE SYMBOL (MUTCD Fig 9E-1). NACTO asks for one after every driveway and
        # intersection and at least every 500 ft; both halves of that rule live in
        # bike_symbol_stations_ft, so this leg, the corridor strip and the 3D export all call for
        # the same symbols. The stations come from state.kerb_openings, which is where a mouth's
        # start and end actually are - the PaintContext knows openings as GROUND, which is the
        # right shape for cutting paint and the wrong one for measuring 15 ft past a mouth.
        mouths = tuple((o.start_ft, o.end_ft)
                       for o in ctx.state.kerb_openings.get((leg_name, side), ()))
        # THE SAME DATUM THE LANE ITSELF IS ON, and getting this wrong put symbols in the buffer.
        # A two-way lane hugs the kerb (BikeLane.hugs_kerb), so its edges are insets FROM THE KERB
        # and its position at a station is wherever the kerb is there. Measuring the symbol's
        # centre off the alignment instead places it where the lane would be if it did not hug -
        # which at Broad & Greenwood was 5 sq ft inside bike_buffer_fill, and the collision check
        # said so. Centreline-referenced only for the sections that are.
        def lane_centre_at(station_ft: float) -> float | None:
            centre_ft = (bounds["bike_inner_ft"] + bounds["bike_outer_ft"]) / 2
            if not lane.hugs_kerb:
                return centre_ft
            # THROUGH kerb_inset_offsets, which is the one home for "this many feet in from the
            # traced kerb" - and the same call the contraflow divider's axis is built from, with
            # the same floor. Rebuilt here as abs(raw kerb) - half instead, it read the RAW
            # tracing where the divider reads the TAPERED one, and the two disagreed by 0.87 ft
            # on broad_st_east: the divider ran 0.20 sq ft through the corner of a stencil that
            # was supposed to sit clear of it. Two derivations of the lane's centre, in agreement
            # with each other nowhere.
            at = kerb_inset_offsets(
                leg, side, np.array([station_ft]),
                (kerb["bike_inner_ft"] + kerb["bike_outer_ft"]) / 2, floor_ft=centre_ft)
            if at is None or not np.isfinite(at[0]):
                return None
            return abs(float(at[0]))
        # WHERE THE LANE IS ACTUALLY DRAWN, and that is not start_ft. start_ft is what this
        # treatment ASKED for; paint_stations then bounds the ask by the stations where the kerb is
        # traced, and on W Broad's southwest approach the two are 22 ft apart - the ask is station
        # 32, the tracing starts at 54.4. Everything between is the crossbike box, which has its
        # own dotted edges and its own carried centre stripe on the extension's straight axis. Run
        # from the ask and two stencils were placed in there off THIS leg's kerb-following centre,
        # 0.8 ft from the stripe actually drawn there, and 0.08 sq ft of one went under it.
        #
        # NOT beyond_ft for the far end either. beyond_ft is a clipping THRESHOLD - the station
        # past which a piece is discarded for lying behind a crossing - and reading it as the run's
        # end gave 6 ft "runs" at Broad & Greenwood, inside which no symbol interval could ever
        # land. Three different quantities that are all stations.
        drawn = paint_stations(leg, side, start_ft, self.to_ft)
        for station_ft in (() if drawn is None else
                           bike_symbol_stations_ft(float(drawn[0]), float(drawn[-1]), mouths)):
            centre_ft = lane_centre_at(station_ft)
            if centre_ft is None:
                continue
            # THE LANE MOVES UNDER THE STENCIL, so a pair is spread from the extremes of its centre
            # over the footprint the symbol covers, and not from its value at the middle station. A
            # stencil is rigid - one offset for all of its vertices - while the divider it has to
            # clear follows the traced kerb continuously, and on W Broad's approach that kerb swings
            # 0.70 ft across the 5.5 ft a symbol is long. Read once at the centre, the stencil was
            # placed on a stripe that had moved out from under it by more than the whole clearance.
            over_footprint = [across for across in
                              (lane_centre_at(station_ft - SYMBOL_LENGTH_FT / 2), centre_ft,
                               lane_centre_at(station_ft + SYMBOL_LENGTH_FT / 2))
                              if across is not None]
            inner_ft, outer_ft = min(over_footprint), max(over_footprint)
            # A two-way lane's two halves face opposite ways, so each gets its own symbol: that is
            # what tells a driver at a mouth which direction the rider bearing down on them is
            # coming from, and it is the reason the symbol is worth drawing rather than decorative.
            faces = ((True, False) if isinstance(self, AddTwoWayBikeLane)
                     else (self.runs_outward,))
            for index, forward in enumerate(faces):
                # Half a symbol plus the divider's own clearance either side of centre. Bounded
                # by the lane's own sixth so a narrow lane does not push them into the edge
                # stripes, and floored at half a symbol plus that clearance so a wide one does not
                # let the pair touch - which is what a plain sixth did on a 10 ft lane, overlapping
                # them by 2 sq ft.
                room_ft = max(lane.width_ft / 2 - SYMBOL_WIDTH_FT / 2, 0.0)
                spread_ft = max(SYMBOL_WIDTH_FT / 2 + SYMBOL_CLEAR_OF_DIVIDER_FT,
                                lane.width_ft / 6)
                spread_ft = min(spread_ft, room_ft)
                if len(faces) == 1:
                    across_ft = centre_ft
                else:
                    # Clear of the stripe wherever it runs under this stencil, but never further
                    # out than the lane's own edge over that same footprint: a symbol shoved off
                    # the lane to dodge its centre stripe has traded one collision for another.
                    # Where the kerb runs straight the two terms are equal and this is the old
                    # centre +/- spread exactly, which is why a straight leg's symbols do not move.
                    across_ft = (min(outer_ft + spread_ft, inner_ft + room_ft) if index
                                 else max(inner_ft - spread_ft, outer_ft - room_ft))
                # NOT shares_a_kerb, unlike the lane it sits on. That flag subtracts ground the
                # adjoining leg has already painted, which is right for a zone running through the
                # node and fatal for a symbol: the lane's own green is registered first, so a
                # symbol lying inside it was subtracted to nothing and 0 of them reached either
                # renderer. A symbol is a discrete mark at a station, not a run that two legs
                # could each paint half of.
                ctx.add(BIKE_LANE_SYMBOL,
                        bike_symbol_polygon(leg, side, station_ft, across_ft, forward),
                        leg_name, side, beyond_ft)
        if lane.buffer_ft:
            # The hatched buffer, between the two lines that bound it rather than under them.
            # lane_narrowing_polygons_ft measures its stripe inward from the kerb-to-kerb half,
            # so the depth is the distance from the kerb to the buffer's inner FACE, and the
            # zone is then cut back to the buffer's outer face.
            # THE BUFFER IS THE MIXED BAND, and it is the piece that makes the whole split work:
            # its inner face is the travel lane's edge, measured from the alignment so the lane
            # holds its target, and its outer face is the bike lane's inner face, measured from the
            # kerb so the lane hugs it. Everything the street does between those two - the 8 ft
            # convergence at W Broad's junction throat - ends up here, widening the separation
            # exactly where the turning conflicts are, which is where a designer would want it.
            inner_face_ft = bounds["travel_lane_edge_ft"] + LANE_EDGE_LINE_WIDTH_FT
            fill = None
            if lane.hugs_kerb:
                stations = paint_stations(leg, side, start_ft, self.to_ft)
                if stations is not None:
                    outer = kerb_inset_offsets(
                        leg, side, stations, kerb["bike_inner_ft"] + LANE_EDGE_LINE_WIDTH_FT,
                        floor_ft=bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT)
                    if outer is not None:
                        # Never inside the travel lane's edge: where the kerb comes in far enough
                        # that the buffer would have negative width it pinches to nothing, rather
                        # than reaching back across the stripe that bounds it.
                        fill = band_from_offsets(leg, side, stations,
                                                  np.full(stations.shape, inner_face_ft),
                                                  np.maximum(outer, inner_face_ft))
            else:
                fill = _one(lane_narrowing_polygons_ft(
                    leg, leg.curb_to_curb_ft / 2 - inner_face_ft,
                    start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,),
                    end_ft=self.to_ft))
                beyond = curbside_strip_polygon(
                    leg, side, bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT, start_ft,
                    self.to_ft)
                if fill is not None and beyond is not None:
                    fill = fill.difference(beyond)
            # Deduped against the other half of the same kerb like the lane itself, or the two
            # legs' buffers overlap through the node now that both reach behind it - 18 sq ft of
            # it at Louellen, which markings_collide reported.
            ctx.rim(ctx.add(BIKE_BUFFER_FILL, fill, leg_name, side, beyond_ft,
                             shares_a_kerb=through), BIKE_LANE_EDGE_LINE)
        if lane.parking_ft:
            # Parking-protected: the stalls sit OUTSIDE the bike lane, between it and the kerb,
            # which is what shields the lane. Ticked at the standard stall length over the runs
            # where parking is legal, exactly as a kerbside parking lane would be.
            #
            # NOT over the legal span directly - the same fix MarkedParking.paint applies and
            # for the same reason (see there): STALL_DIVIDER is STOPPED at every opening, so a
            # grid laid over ground the treatment has not yet been cut against draws a stall
            # with an open-ended tick at a driveway or a crossing, or one straddling it outright.
            # ctx.open_runs asks the real cut ahead of laying anything.
            #
            # NOT beyond_the_tracing: a stall proposes paint on the physical kerb, so it may
            # reach no further than the kerb is actually surveyed - see the note in
            # MarkedParking.paint for the disagreement asking past that bound produces.
            # ASKED OF THE SECTION, not recomputed from the kerb. The two orderings put the
            # stalls at opposite ends of the section - outermost for BikeLane, inboard of the
            # lane for KerbsideBikeLane - and a caller that derives one of them here is the
            # second derivation of a placed offset that SKILLS 0a is a list of. BikeLane's
            # method returns exactly the arithmetic that used to be on these three lines.
            half = leg.curb_to_curb_ft / 2
            inner_off, outer_off = lane.parking_band_from_centerline_ft(half)
            # THE CORNER END OF THAT SAME BAND, hatched, because parking is forbidden there.
            # parking_runs below starts at the first station a stall may legally go, so without
            # this the band from the corner to that station is bare asphalt - and on the swapped
            # ordering that bare strip sits between the travel lane and a bike lane which does
            # carry on to the junction, so it reads as somewhere to pull in rather than as the
            # setback it is. This is the paint half of the paint-and-posts curb extension;
            # ProtectDaylightZone stands the posts in the same span.
            #
            # Same channels as MarkedParking.paint's own daylighting, since it is the same
            # statute (R.S. 39:4-138) drawn on a different cross-section, and the plan view and
            # the 3D both already know how to draw them.
            #
            # NOT beyond_the_tracing, which MarkedParking needs and this does not: that zone runs
            # the full depth to the kerb, so past the tracing it has no outer edge, while this
            # band is bounded by two offsets from the alignment and is the same width whether the
            # kerb is traced there or not.
            for zone_start_ft, zone_end_ft in merged_no_parking_spans_ft(
                    no_parking_zones_ft(ctx.state, leg_name, side,
                                        ctx.crosswalk_offsets, ctx.props)):
                zone_start_ft = max(zone_start_ft, start_ft)
                if self.to_ft is not None:
                    zone_end_ft = min(zone_end_ft, self.to_ft)
                if zone_end_ft - zone_start_ft < MIN_LINE_LENGTH_FT:
                    continue
                zone = offset_band_polygon(leg, side, inner_off, outer_off,
                                           zone_start_ft, zone_end_ft)
                ctx.rim(ctx.add(DAYLIGHT_FILL, zone, leg_name, side), DAYLIGHT_EDGE_LINE)
            for run_start_ft, run_end_ft in parking_runs(ctx.state, leg_name, side,
                                                          ctx.crosswalk_offsets, ctx.props):
                band = offset_band_polygon(leg, side, inner_off, outer_off,
                                           max(run_start_ft, start_ft),
                                           run_end_ft if self.to_ft is None
                                           else min(run_end_ft, self.to_ft))
                open_runs = ctx.open_runs(leg_name, side, STALL_DIVIDER, band) if band else []
                # THE PITCH, NOT THE STALL LENGTH, and asked of the section. They are the same
                # number while the parking is parallel and they are not once it is angled: a 9 ft
                # stall at 60 degrees occupies 10.39 ft of kerb, so dividing the run by 22 lays
                # half as many stalls as the bay holds and puts every tick in the wrong place.
                pitch_ft = lane.parking_pitch_ft()
                for lo, hi in stall_lane_runs_ft(open_runs, pitch_ft,
                                                  keep_inside_ft=MIN_LINE_LENGTH_FT):
                    # THE LINE'S DEPTH, NOT THE BAY'S - see parking_line_depth_ft. Drawn to
                    # the full bay depth the divider meets this section's own outer edge line
                    # and the two read as one boundary painted across every stall opening.
                    for divider in parking_stall_lines_ft(
                            leg, side, lane.parking_line_depth_ft(), pitch_ft, lo, hi,
                            curb_offset_ft=lane.parking_curb_offset_ft(half),
                            # SIGNED BY WHICH WAY TRAFFIC RUNS - the same runs_outward the bike
                            # symbol's heading comes off, because a bay leans the way a driver
                            # turns into it and that is the direction of travel, not of the leg.
                            skew_ft=lane.parking_skew_ft(self.runs_outward)):
                        ctx.add(STALL_DIVIDER, divider, leg_name, side)
        # NOT `else`. The question is whether anything is LEFT OVER against the kerb, and that is
        # not the same as whether the section carried parking - KerbsideBikeLane carries parking
        # AND leaves the kerbside strip over, because its parking is inboard of the lane. Asked as
        # `else` the swapped section drew stalls and then left its kerbside spare as bare asphalt.
        if not (lane.parking_ft and lane.parking_is_outermost):
            # The leftover between the lane's outer stripe and the kerb, hatched. A bike lane is
            # a standard width and the street's spare asphalt is not part of it - the same
            # accounting an 8 ft parking stall gets, where the remainder becomes the kerb buffer
            # rather than a wider stall. Without this the lane read as reaching the kerb, which
            # is what made the drawn lanes look far wider than they are.
            #
            # Rimmed, like every other hatched zone here. The plan view outlines a fill polygon
            # for free, so this zone read as finished in 2D while the 3D render - which gets
            # only the hatch strokes and the lines actually painted - had its strokes stopping
            # in mid-air where the crossing cut them. See PaintContext.rim.
            # Now a CONSTANT shy_ft against the kerb rather than whatever the street had spare,
            # because the lane's outer edge follows the kerb instead of standing off the narrowest
            # point. That is the whole visible fix: this zone used to be the wedge, 0.87 ft of
            # hatching at one end of W Broad's lane and 8.68 ft at the other. Where shy_ft is 0
            # there is nothing to hatch and the lane meets its own edge stripe at the kerb.
            # Where the lane hugs the kerb this is a CONSTANT shy_ft against it rather than
            # whatever the street had spare, which is the whole visible fix: it used to be the
            # wedge - 0.87 ft of hatching at one end of W Broad's lane and 8.68 ft at the other.
            # With shy_ft at 0 there is nothing to hatch and the lane meets its own edge stripe.
            hatch = (kerb_referenced_band_polygon(leg, side, 0.0, lane.shy_ft, start_ft,
                                                   self.to_ft)
                     if lane.shy_ft else None) if lane.hugs_kerb else _one(
                lane_narrowing_polygons_ft(leg, leg.curb_to_curb_ft / 2 - bounds["outer_ft"],
                                            start_left_ft=start_ft, start_right_ft=start_ft,
                                            sides=(side,), end_ft=self.to_ft))
            # WHICH BUFFER IT IS. On a kerb with nothing outside the lane this leftover is the
            # bikeway's own separation from the kerb, so it is drawn in the BIKE buffer's channel
            # and reads as one protected corridor - lane with separation either side - instead of
            # a lane that stops short of the kerb beside an unexplained hatch. Where the section
            # puts parking out there, or the leg carries MarkedParking on this kerb, the leftover
            # belongs to THAT marking and keeps the parking buffer's channel; routing it to the
            # bike buffer painted 100 sq ft of broad_st_west's parking buffer twice, which
            # markings_collide reported.
            from src.geometry.targets import LegSide
            from src.geometry.treatments.parking import MarkedParking

            kerbside_is_the_bikeway_s = (
                not (lane.parking_ft and lane.parking_is_outermost)
                and ctx.state.treatment_for(MarkedParking, LegSide(leg_name, side)) is None)
            kind, edge = ((BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE) if kerbside_is_the_bikeway_s
                          else (BUFFER_FILL, BUFFER_EDGE_LINE))
            ctx.rim(ctx.add(kind, hatch, leg_name, side, beyond_ft,
                             shares_a_kerb=through), edge)



@dataclass(frozen=True)
class AddKerbsideBikeLane(AddBikeLane):
    """Mark a parking-protected bike lane: the lane against the kerb, the parking outboard of it.

    Everything AddBikeLane does, on KerbsideBikeLane's ordering instead of BikeLane's - the fit
    test, the refusal rather than the quiet narrowing, the stall ticks over the runs where
    parking is legal, and the hatched leftover. Only the cross-section differs, which is why this
    overrides `lane` and nothing else: `section()` reads `self.lane`, and every view and check
    reads the section.

    WHAT TO PAIR IT WITH. AddBikeLaneBollards stands its posts in the buffer, which on this
    ordering is the door zone between the parked cars and the rider - the side that needs them
    here. There is no need for a separate MarkedParking on the same kerb: the stalls are part of
    this section and drawn by it, and adding one would put a second parking lane on ground this
    treatment has already spent (DesignState.apply has no duplicate guard).
    """

    @property
    def lane(self) -> KerbsideBikeLane:
        return KerbsideBikeLane(width_ft=self.width_ft, buffer_ft=self.buffer_ft,
                                 parking_ft=self.parking_ft, shy_ft=self.shy_ft)

    def describe(self) -> str:
        return (f"AddKerbsideBikeLane({self.target.leg}, {self.target.side}): "
                f"{_feet(self.width_ft)} ft lane against the kerb"
                + (f", {_feet(self.buffer_ft)} ft door-zone buffer" if self.buffer_ft else "")
                + (f", protected by {self.parking_ft:.0f} ft of marked parking outboard"
                   if self.parking_ft else "")
                + (f", {self.shy_ft:.1f} ft shy of the kerb" if self.shy_ft else ""))

@dataclass(frozen=True)
class AddTwoWayBikeLane(AddBikeLane):
    """A bidirectional bike lane along ONE side of a leg, with the travel lanes shifted off it.

    Subclasses AddBikeLane because everything about painting the section is the same problem -
    two edge stripes, a hatched buffer, the green surface, the dotted extension across every
    driveway, all cut around the crossing bands. Only two things differ, and both are additions
    rather than changes: the section starts further out (TwoWayBikeLane resolves that), and the
    lane carries a yellow centre stripe because it holds opposing riders.

    THE SIDE IS A CORRIDOR DECISION, NOT A PER-JUNCTION ONE. A two-way lane that changes sides
    mid-corridor makes riders cross the street to stay on it, so the side is chosen once for the
    whole route from how many streets cut each kerb - see sites/*/scenarios.py for Broad St's.
    """
    paint_group: ClassVar[int] = 30
    paint_rank: ClassVar[int] = 0
    constrained: bool = False

    def __post_init__(self):
        """The width floor, checked without a street.

        NOT by constructing a TwoWayBikeLane - that also checks the fit against two half-widths
        this treatment does not carry, and feeding it invented ones to get at the width check
        would be a validation passing on made-up geometry. A 6 ft two-way lane is wrong before
        any street is consulted, so that part is checked here and the fit is checked in apply_to
        where the real half-widths exist.
        """
        floor = (CONSTRAINED_TWO_WAY_BIKE_LANE_FT if self.constrained
                 else MIN_TWO_WAY_BIKE_LANE_FT)
        if self.width_ft < floor:
            raise ValueError(
                f"A {self.width_ft:.2f} ft two-way bike lane is under NACTO's {floor:.0f} ft "
                f"{'constrained-conditions' if self.constrained else 'minimum'} width "
                f"({TWO_WAY_BIKE_LANE_WIDTH_FT:.0f} ft is the width to design to).")

    def section(self, state: "DesignState") -> TwoWayBikeLane:
        """The section as this leg's own kerbs make it, over the span this lane occupies.

        THE TWO KERBS ARE MEASURED SEPARATELY AND OVER DIFFERENT STRETCHES, which is the whole
        of governing_half_widths_ft's docstring and not restated here - it is the one home, and
        corridor_paint and CorridorBikeway._reach_on measure through it too. Note only that
        pairing each side's own minimum reads NARROWER than the narrowest real cross-section (by
        2.44 ft on w_broad_st_northeast) and that this is the exact figure rather than a
        conservative one: with a constant-width section on one datum, the near kerb's slack is
        drawn as hatching on the near kerb and never reaches the travel way.

        Raises through TwoWayBikeLane when the leg cannot hold two travel lanes beside it.
        """
        leg = state.legs[self.target.leg]
        side = Side(str(self.target.side))
        near_ft, far_ft = governing_half_widths_ft(leg, str(side), to_ft=self.to_ft)
        return TwoWayBikeLane(
            width_ft=self.width_ft, buffer_ft=self.buffer_ft, constrained=self.constrained,
            near_half_ft=near_ft, far_half_ft=far_ft)

    def describe(self) -> str:
        return (f"AddTwoWayBikeLane({self.target.leg}, {self.target.side}): "
                f"{_feet(self.width_ft)} ft two-way lane"
                + (f", {_feet(self.buffer_ft)} ft buffer" if self.buffer_ft else "")
                + self._extent())

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        leg = state.legs[self.target.leg]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {self.target.leg!r} has no width - nothing to fit a lane into.")
        # Building the section IS the fit check: TwoWayBikeLane refuses one that leaves less than
        # two travel lanes, and it does so carrying the measurement. Reraised untouched.
        section = self.section(state)
        shift_ft = travel_lane_divider_shift_ft(section)
        surplus_ft = far_kerb_surplus_ft(section)
        other = Side(str(self.target.side)).other
        # The ACTUAL lane width, not half the travel way. Those are the same number only on a leg
        # too narrow to hold the target, and reporting the equal-split figure on a leg that holds
        # 11 ft lanes plus a stall's worth of surplus described a design nobody drew.
        lane_ft = (TARGET_LANE_WIDTH_FT if surplus_ft >= 0
                   else (section.near_half_ft + section.far_half_ft - section.section_ft) / 2)
        note = (f". Spends {section.section_ft:.2f} ft of this leg's "
                f"{section.near_half_ft + section.far_half_ft:.2f} ft between kerbs, leaving two "
                f"{lane_ft:.2f} ft travel lanes with the centreline shifted {shift_ft:.2f} ft "
                f"toward the {other} kerb")
        if surplus_ft >= 0:
            note += f", and {surplus_ft:.2f} ft spare against that kerb"
        else:
            note += (f" - under the {TARGET_LANE_WIDTH_FT:.0f} ft target by "
                     f"{TARGET_LANE_WIDTH_FT - lane_ft:.2f} ft, which is this leg's width rather "
                     f"than a choice, so the travel way is split equally")
        return (note + ". The NJDOT alignment does not move; every station and crossing frame is "
                "measured from it as before. " + NJDOT_TWO_WAY_OBJECTION)

    def paint(self, ctx) -> None:
        """The one-way section's markings, plus the yellow stripe down the middle of the lane."""
        from src.geometry.markings import BIKE_CONTRAFLOW_DIVIDER
        from src.geometry.model import inset_line_ft, kerb_parallel_line_ft

        # Everything AddBikeLane paints, at this section's own offsets. Reached through the
        # resolved lane, so the stripes land where the shifted section actually is.
        section = self.section(ctx.state)
        super().paint(ctx)

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        bounds = section.offsets_from_centerline_ft()
        centre_ft = (bounds["bike_inner_ft"] + bounds["bike_outer_ft"]) / 2
        # Measured from the KERB, like the lane's own two edges - see BikeLane.offsets_from_kerb_ft.
        # Left on the alignment it stayed put while the lane moved onto the kerb, and on
        # broad_st_east's right kerb the divider ended up running along the lane's edge stripe for
        # 1.2 ft, which MarkingsDoNotCollide reported and was right to: a lane's centre stripe that
        # is not down the lane's centre is not a centre stripe.
        from_kerb = section.offsets_from_kerb_ft()
        centre_from_kerb_ft = (from_kerb["bike_inner_ft"] + from_kerb["bike_outer_ft"]) / 2
        at = ctx.anchors(leg_name, side, inner_offset_ft=centre_ft)
        through = (leg_name, side) in ctx.straight_through
        if through:
            # BEHIND THE NODE, not up to it. Each leg's paint is built in its own frame, so two
            # halves that both stop at their own station 0 stop just shy of each other - 1.28 ft
            # of hole at W Broad & Louellen, in the middle of a lane whose whole point is running
            # continuously through the junction. Starting behind it makes the two overlap, and the
            # overlap is deduped by shares_a_kerb below, which is the same mechanism that keeps two
            # zones on one through kerb from double-painting. Honoured only as far as the kerb is
            # really traced there - see model.paint_stations.
            start_ft, beyond_ft = -THROUGH_JUNCTION_OVERLAP_FT, None
        elif leg_name in ctx.marked:
            from src.geometry.paint import end_against_crossing
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        # BROKEN, not continuous - passing is permitted in a two-way bikeway where sight
        # distance allows, and MUTCD's yellow-broken is what says so. Cut into dashes here
        # rather than left to a line style, for the reason every other dashed marking in this
        # project is: a style is a 2D property and the 3D render gets geometry, so a continuous
        # line with a dashed style renders solid.
        axis = (kerb_parallel_line_ft(leg, side, centre_from_kerb_ft, start_ft, self.to_ft,
                                       floor_ft=centre_ft)
                 if section.hugs_kerb
                 else inset_line_ft(leg, side, centre_ft, start_ft, self.to_ft))
        if axis is None or axis.is_empty:
            return
        # AND IT CARRIES THROUGH EVERY DRIVEWAY, like the lane's other markings.
        #
        # MUTCD 11th ed. §9E.04 Option 02 permits a bicycle lane to be continued through a
        # driveway with solid or dotted longitudinal lines, and §9E.06 Guidance 15 says lane
        # extension markings SHOULD be used to extend a buffer-separated bicycle lane across
        # intersections and driveways. NACTO's Urban Bikeway Design Guide is more specific for
        # this facility: contraflow and bidirectional protected lanes must continue through
        # intersections and driveways, with a DOTTED YELLOW CENTRELINE along the lane and through
        # the crossings. See STANDARDS.md §4.
        #
        # This stripe used to simply stop at each driveway - 22 dashes on a kerb with two of them
        # against 30 on a kerb with none - while the edge lines continued dotted and the green
        # carried across. Three answers to one conflict point, and this was the one that belonged
        # to nobody: it was an omission, not a design.
        #
        # Its row in markings.AT_AN_OPENING says CARRIED, so `add` does not cut it at an entrance
        # at all and the cadence cannot break phase across one. Being already a broken line, it
        # needs no separate dotted pattern - the standard's "dotted extension" is what it already
        # looks like. It used to be cut and the part inside re-laid as an exact complement, which
        # is the same statement made twice and in two places.
        period_ft = CONTRAFLOW_DASH_FT + CONTRAFLOW_GAP_FT
        at_ft = 0.0
        while at_ft + CONTRAFLOW_DASH_FT <= axis.length:
            dash = shapely.ops.substring(axis, at_ft, at_ft + CONTRAFLOW_DASH_FT)
            if dash.geom_type == "LineString" and dash.length > 0:
                ctx.add(BIKE_CONTRAFLOW_DIVIDER, dash, leg_name, side, beyond_ft)
            at_ft += period_ft
