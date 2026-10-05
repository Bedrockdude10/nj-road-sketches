"""Lane narrowing, and the flex posts that hold it.

Narrowing is the one treatment applied at every site, on every leg, in every scenario - it is
the traffic-calming baseline the rest is layered onto."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar


from src.geometry.targets import BOTH_SIDES, Side

if TYPE_CHECKING:                       # DesignState is layered above this module;
    from src.geometry.treatments.state import DesignState   # the annotation is a string
    from src.geometry.intersection.junction import IntersectionModel
from src.geometry.treatments.base import (BOLLARD_DEFAULT_SPACING_FT,
                                          LANE_NARROWING_DEFAULT_STRIPE_FT, Treatment)



@dataclass(frozen=True)
class LaneNarrowing(Treatment):
    """Paint-only visual lane narrowing: a striped buffer/shoulder along one or both kerbs of a
    leg. Zero curb/pavement change - the lowest-cost alternative to a real curb extension.

    line_only=True paints the delineating line (straight run + corner taper) with no chevron
    fill: a bare-minimum lane-width marking, useful both for measuring against the plan view
    without hatch density affecting the read and as a low-cost treatment in its own right.

    `sides` defaults to both. Pass a single side when the OTHER side's edge is already owned by
    another treatment - MarkedParking delineates its own side, and this then adds only the
    matching plain line opposite.

    end_ft trims the buffer to a station short of the leg's own end, closing it with the same
    square-cut line a leg with no crossing gets at its start - for the caller that measured the
    kerb per station and found room only part of the way (see
    parking.hold_travel_lane_at_target), rather than one that wants the whole leg drawn.
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 10
    paint_rank: ClassVar[int] = 0
    stripe_width_ft: float = LANE_NARROWING_DEFAULT_STRIPE_FT
    line_only: bool = False
    sides: tuple = BOTH_SIDES
    end_ft: float | None = None

    def __post_init__(self):
        if self.stripe_width_ft <= 0:
            raise ValueError(f"A lane-narrowing buffer needs a width; got "
                             f"stripe_width_ft={self.stripe_width_ft}.")
        object.__setattr__(self, "sides", tuple(Side(side) for side in self.sides))
        if not self.sides:
            raise ValueError("A lane narrowing with no sides paints nothing - pass at least one.")

    def describe(self) -> str:
        end = f", end_ft={self.end_ft:.1f}" if self.end_ft is not None else ""
        return (f"LaneNarrowing({self.target}, stripe_width_ft={self.stripe_width_ft}, "
                f"line_only={self.line_only}, sides={tuple(str(s) for s in self.sides)}{end})")

    def paint(self, ctx) -> None:
        """An edge line, a hatched buffer, and a taper back to the kerb. line_only legs get the
        boundary lines without the fill."""
        from src.geometry.markings import (LANE_EDGE_LINE, LANE_NARROWING_FILL, TAPER_FILL,
                                           TAPER_LINE, ZONE_END_LINE)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, end_against_crossing, tapers_cleanly)
        from src.geometry.paint.datum import Along, Across, Taper, Kerb, Narrowest

        leg_name = self.target.leg
        leg = ctx.state.legs[leg_name]
        stripe_width_ft = self.stripe_width_ft
        fill = not self.line_only
        for side in (str(s) for s in self.sides):
            at = ctx.anchors(leg_name, side, inner_offset_ft=-stripe_width_ft)
            # A crossing is something to end against: run into it and let it cut the end. Only
            # where there is none does the paint have to resolve itself back to the kerb, and
            # only then is a taper the right way to do it.
            if (leg_name, side) in ctx.straight_through:
                # One unbroken kerb with no corner return at either end: run from the junction
                # NODE and let any crossing cut it, keeping BOTH halves. Tested before the
                # marked/unmarked split because it applies to both. Discarding the junction-side
                # half leaves the kerb bare between the crossing and the node, with no corner
                # there to justify it.
                start_ft, beyond_ft, curved = 0.0, None, False
            elif leg_name in ctx.marked:
                start_ft, beyond_ft = end_against_crossing(at)
                curved = False
            else:
                curved = tapers_cleanly(stripe_width_ft, at)
                start_ft, beyond_ft = (at.anchor_ft if curved else at.target_ft), None

            end_station = self.end_ft if self.end_ft is not None else leg.centerline.length

            # Edge line: centre half a stroke inside the zone
            edge_line_inset_ft = stripe_width_ft - LANE_EDGE_LINE_WIDTH_FT / 2
            ctx.paint(LANE_EDGE_LINE, leg_name, side,
                      Along((start_ft, end_station), Narrowest(edge_line_inset_ft)),
                      beyond_ft=beyond_ft)

            if curved:
                ctx.paint(TAPER_LINE, leg_name, side,
                          Taper(at.anchor_ft, at.target_ft, Narrowest(edge_line_inset_ft), fill=False))

            if fill:
                # Hatch fill
                ctx.rim(ctx.paint(LANE_NARROWING_FILL, leg_name, side,
                                  Along((start_ft, end_station), Kerb(0), Narrowest(stripe_width_ft)),
                                  beyond_ft=beyond_ft,
                                  shares_a_kerb=(leg_name, side) in ctx.straight_through), LANE_EDGE_LINE)

                if curved:
                    ctx.paint(TAPER_FILL, leg_name, side,
                              Taper(at.anchor_ft, at.target_ft, Narrowest(stripe_width_ft), fill=True))

                elif leg_name not in ctx.marked and (leg_name, side) not in ctx.straight_through:
                    # Only where the kerb does NOT run straight through. On one that does, the
                    # zone continues into the adjoining leg's zone rather than ending at the
                    # node, and closing it off draws a line across the hatching mid-intersection.
                    ctx.paint(ZONE_END_LINE, leg_name, side,
                              Across(start_ft, Kerb(0), Narrowest(stripe_width_ft)))

                if self.end_ft is not None:
                    # The OTHER end: not the junction, the station where the room this buffer
                    # was sized on ran out - see LaneNarrowing.end_ft. Same closing line, cut at
                    # the far station instead of the near one, independent of whether the near
                    # end tapers, ends at a crossing, or runs into the intersection.
                    ctx.paint(ZONE_END_LINE, leg_name, side,
                              Across(self.end_ft, Kerb(0), Narrowest(stripe_width_ft)))


@dataclass(frozen=True)
class LaneNarrowingBollards(Treatment):
    """Flex-post delineators down the centre of a leg's LaneNarrowing buffer - a firmer
    escalation of that treatment, still with no curb/pavement change. Requires LaneNarrowing on
    this leg: the lateral placement is derived from that buffer's own stripe_width_ft rather
    than specified again here."""
    paint_group: ClassVar[int] = 10
    paint_rank: ClassVar[int] = 1
    spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT

    def __post_init__(self):
        if self.spacing_ft <= 0:
            raise ValueError(f"Posts need a spacing; got spacing_ft={self.spacing_ft}.")

    def describe(self) -> str:
        return f"LaneNarrowingBollards({self.target}, spacing_ft={self.spacing_ft})"

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> None:
        if state.treatment_for(LaneNarrowing, self.target) is None:
            raise KeyError(f"{self.target} has no lane-narrowing buffer - apply LaneNarrowing "
                            f"first. A row of posts is placed inside a buffer, so its lateral "
                            f"position comes from that buffer's own width.")

    def paint(self, ctx) -> None:
        """Down the centre of the buffer LaneNarrowing paints, on both sides it narrowed. The
        offset comes from that buffer's own stripe_width_ft - a post placed off a separately
        guessed offset stands somewhere the buffer is not."""
        from src.geometry.markings import BOLLARD
        from src.geometry.model import leg_clearance_ft
        from src.geometry.paint.datum import At, Kerb

        leg_name = self.target.leg
        leg = ctx.state.legs[leg_name]
        narrowing = ctx.state.treatment_for(LaneNarrowing, self.target)
        stripe_width_ft = narrowing.stripe_width_ft

        clearance_ft = leg_clearance_ft(leg_name, ctx.state.legs, ctx.state.corner_fillets)

        # Build stations list for bollards on each side
        for side in (str(s) for s in narrowing.sides):
            stations = []
            station = clearance_ft
            while station <= leg.centerline.length - clearance_ft:
                stations.append(station)
                station += self.spacing_ft

            if stations:
                ctx.paint(BOLLARD, leg_name, side,
                          At(tuple(stations), Kerb(stripe_width_ft / 2)))
