"""Hatched areas OSM maps as `road_marking=restriction` - drawn where the way is.

Wiki Key:road_marking (approved): `restriction` is an AREAL marking, "Neutral areas or
restriction markings, like gore chevron or no-parking markings", mapped as a closed way. The
way is the painted area, as a `road_marking=stop_line` way is the painted bar, so nothing here
sizes anything: the outline is painted half a stripe inside the way and the chevron fill inside
that. A kerb's hatching beside a travel lane is one of these, existing or proposed alike - a
proposal adds it as a new way (src/sources/osm_change.py) and this reads it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from shapely.geometry import LineString, Polygon

from src.geometry.model import station_offset_many
from src.geometry.targets import LegSide
from src.geometry.treatments.base import Treatment
from src.geometry.treatments.state import DesignState

if TYPE_CHECKING:
    from src.geometry.intersection.junction import IntersectionModel


@dataclass(frozen=True)
class RestrictionMarking(Treatment):
    """Every `road_marking=restriction` area on one kerb of one leg, in state-plane feet."""
    paint_group: ClassVar[int] = 10
    paint_rank: ClassVar[int] = 0
    rings_ft: tuple = ()
    osm_ids: tuple = ()

    def __post_init__(self):
        if not self.rings_ft:
            raise ValueError("A restriction marking with no area paints nothing.")

    def describe(self) -> str:
        ids = ", ".join(f"way/{i}" for i in self.osm_ids)
        return f"RestrictionMarking({self.target}, {len(self.rings_ft)} area(s): {ids})"

    def polygons(self) -> list[Polygon]:
        return [Polygon(ring) for ring in self.rings_ft]

    def inner_offset_ft(self, leg) -> float | None:
        """How close to the alignment the areas reach, over the leg's own stations - where the
        travel lane beside them ends. None where no vertex stations onto the leg."""
        points = np.asarray([xy for ring in self.rings_ft for xy in ring], dtype=float)
        stations, offsets = station_offset_many(leg.centerline, points)
        within = (stations >= 0.0) & (stations <= leg.centerline.length)
        return float(np.abs(offsets[within]).min()) if within.any() else None

    def paint(self, ctx) -> None:
        """The way's outline, its centre half a stripe inside the way, and the chevron fill
        inside the outline - so the painted marking covers the mapped area and no more."""
        from src.geometry.markings import LANE_EDGE_LINE, LANE_NARROWING_FILL
        from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT

        leg_name, side = self.target.leg, str(self.target.side)
        for area in self.polygons():
            outline = area.buffer(-LANE_EDGE_LINE_WIDTH_FT / 2, join_style=2)
            for part in getattr(outline, "geoms", [outline]):
                if part.geom_type == "Polygon" and not part.is_empty:
                    ctx.add(LANE_EDGE_LINE, LineString(part.exterior.coords), leg_name, side)
            fill = area.buffer(-LANE_EDGE_LINE_WIDTH_FT, join_style=2)
            if not fill.is_empty:
                ctx.add(LANE_NARROWING_FILL, fill, leg_name, side)


def restriction_painted_ft(state: DesignState, leg_name: str, side: str) -> float:
    """The kerbside width a leg-side's restriction areas take, in the NOMINAL frame the lane
    checks subtract in (travel_lane_width_ft): half the nominal width less how close to the
    alignment the areas reach. 0.0 where the kerb has none."""
    marking = state.treatment_for(RestrictionMarking, LegSide(leg_name, side))
    leg = state.legs[leg_name]
    if marking is None or leg.curb_to_curb_ft is None:
        return 0.0
    inner_ft = marking.inner_offset_ft(leg)
    return 0.0 if inner_ft is None else max(leg.curb_to_curb_ft / 2 - inner_ft, 0.0)


def apply_osm_road_markings(state: DesignState, model: IntersectionModel) -> DesignState:
    """Put every `road_marking=restriction` area OSM carries on the kerb it lies along.

    WHICH KERB is a geometric question about the area, asked of every leg: the one whose own
    stations reach the area's representative point, nearest to its alignment, with the side the
    sign of the offset. An area no leg reaches is noted and not drawn - it belongs to a street
    this drawing does not model.
    """
    from src.geometry.intersection.paved import to_state_plane

    areas = (getattr(model, "osm", None) or {}).get("road_markings") or []
    by_kerb: dict[tuple[str, str], list] = {}
    for area in areas:
        ring = tuple(to_state_plane(area["coords_wgs84"]))
        point = Polygon(ring).representative_point()
        best = None
        for leg_name, leg in state.legs.items():
            (station,), (offset,) = station_offset_many(leg.centerline,
                                                        np.asarray([point.coords[0]]))
            if 0.0 <= station <= leg.centerline.length and (best is None
                                                            or abs(offset) < abs(best[1])):
                best = (leg_name, float(offset))
        if best is None:
            state.notes.append(f"apply_osm_road_markings: way/{area.get('id')} lies along no "
                               f"modelled leg - not drawn.")
            continue
        side = "left" if best[1] > 0 else "right"
        by_kerb.setdefault((best[0], side), []).append((ring, area.get("id")))
    for (leg_name, side), found in sorted(by_kerb.items()):
        state = state.apply(RestrictionMarking(LegSide(leg_name, side),
                                               rings_ft=tuple(ring for ring, _ in found),
                                               osm_ids=tuple(i for _, i in found)))
    return state
