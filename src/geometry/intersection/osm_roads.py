"""Tying this project's legs to somebody else's linework: NJDOT's SRI centrelines and OSM's
ways.

Both matches are by geometry rather than by name, and both can legitimately fail - a leg with no
match is a fact to report, not an error - so the thresholds that decide a match are constants here
rather than magic numbers at the call site."""


import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point

from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_roads
from src.geometry.model import (
    Alignment,
    leg_bearing_deg,
    line_direction,
    station_offset_many,
)
from src.geometry.intersection.junction import RoadSpan



def _piece_bearing_deg(piece: LineString) -> float:
    """Which way one split piece points, from its junction end outward.

    THE ONE DERIVATION IS leg_frame.leg_bearing_deg, reached through the Alignment it reads - a
    piece IS a centreline that starts at the junction, which is all that function asks for. It
    was written out a second time here, as a compass bearing from the resolved node to the
    piece's far end, and the two agree exactly: `_snap_to_center` puts every piece's first
    vertex on that node before it gets here (measured at 0.000 ft over all 34 leg-pieces at the
    nine sites, at frame scales 1x, 2.5x and 3x), and a chord's bearing is translation-invariant
    besides.

    Measuring the piece's OWN chord is what makes one call independent of one junction. A window
    onto the borough document holds several, so a bearing taken from the window's centre would
    be a fact about where the crop was taken rather than about the street.
    """
    return leg_bearing_deg(Alignment(centerline=piece))


def _declared_bearing_deg(legs_cfg: dict | None, name: str) -> float | None:
    """The bearing config.yaml declares for this leg, or None where it declares none.

    CONFIG IS AN OVERRIDE AND IT WINS BY BEING PRESENT - DesignState.centerline_style's
    precedence, for the same reason: a human's statement outranks what was derived, and the way
    to say "nothing stated" is to be absent rather than to carry a sentinel that has to be
    recognised at every reader.

    None is therefore not a default bearing and must not become one. Which HALF of a road a leg
    is cannot be measured off the road - both halves are there - so an absent declaration is
    answered by the assignment being forced, or by refusing; see _assign_leg_pieces.
    """
    return (legs_cfg or {}).get(name, {}).get("bearing_deg")


def _bearing_diff(a: float, b: float) -> float:
    """Smallest angular difference between two compass bearings, in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


# How far a road network centerline may sit from the resolved intersection node before
# the snap below is worth reporting. Sub-foot gaps are digitizing noise; anything larger
# is a real disagreement between the two sources and worth seeing in the phase output.
SNAP_REPORT_THRESHOLD_FT = 2.0
ROAD_CONTEXT_RADIUS_M = 130


def _snap_distance_ft(line: LineString, center_ft: Point) -> float:
    """Perpendicular distance from the resolved intersection node to a road centerline."""
    return line.distance(center_ft)


def _snap_to_center(piece, center_ft: Point):
    """Translate a leg centerline so it starts at the resolved intersection node.

    The intersection LOCATION comes from OSM (a shared junction node, cross-checked against
    the NJDOT SLD milepost - see data_loader.geocode_intersection), and so does every piece
    of context placed against it: the surveyed pedestrian crossings, buildings, and mapped
    footways. The leg CENTERLINES come from NJDOT's SRI linear-referencing layer. On state
    and county routes the two disagree systematically (up to 16 ft): an SRI line is a
    linear-referencing alignment, not a surveyed physical centerline.

    TRADE-OFF: the entire piece translates, so its far end moves off NJDOT's alignment by
    the same amount. Accuracy at the intersection beats accuracy at the far end of a leg
    that is only there for context. The offset is reported when it exceeds
    SNAP_REPORT_THRESHOLD_FT. Bearings, lengths and widths are unchanged; only position moves.
    """
    x0, y0 = piece.coords[0]
    return affinity.translate(piece, xoff=center_ft.x - x0, yoff=center_ft.y - y0)


def _assign_leg_pieces(pieces: list, leg_names: list[str], legs_cfg: dict | None = None,
                        center_ft: Point | None = None,
                        sri: str = "?") -> dict[str, object]:
    """
    Match centerline pieces (all sharing one SRI, split at the intersection) to
    the leg names that reference that SRI, by nearest compass bearing.
    Generalizes to any number of pieces per SRI (2 for a through road, 1 for a
    dead-end/stub leg) and any intersection shape - nothing here assumes a
    4-way or perpendicular roads.

    A PIECE'S BEARING IS MEASURED, A LEG NAME'S IS DECLARED, and that asymmetry is the whole
    of this function. `_piece_bearing_deg` reads which way a piece points off the piece; a
    NAME is a string, so which half of a road it means is the one thing here that no geometry
    answers - both halves are on the ground and they differ only in what they are called.
    `bearing_deg` in config.yaml is that label, it is consulted through
    `_declared_bearing_deg`, and it wins by being present.

    WHERE IT IS ABSENT THE ASSIGNMENT HAS TO BE FORCED, or this refuses. One piece and one
    name is forced - there is nothing to tell apart - and that is the case a caller with no
    config has: a window onto the borough document, whose legs carry a street name and nothing
    else. Two unlabelled names on a through road is not forced, and picking by piece order
    there is a coin flip that draws an approach's whole treatment on the wrong side of the
    junction, so it raises with the two bearings that settle it.

    `center_ft` is accepted and no longer read. It was the datum the bearings were taken from,
    which tied one call to one junction; a piece's own chord is the same number at a
    configured site (see _piece_bearing_deg) and the right one in a window holding several.

    THE COUNTS HAVE TO MATCH, and when they do not it is a config error worth naming.
    A road network splits an SRI at the junction into as many pieces as there are
    approaches on it; the config says how many legs it has there. A disagreement means
    one of two mistakes, and both used to be silent or near-silent:

      MORE PIECES THAN LEGS - the road runs through the junction but the config only
      declares one side of it. This raised `min() iterable argument is empty` from the
      matcher below, naming neither the SRI nor the config. It is an easy mistake where
      NJDOT's name for an SRI describes only part of what it covers: at NJ 31 & W
      Delaware Ave, NJDOT carries the whole of Delaware Ave under one SRI called
      "E DELAWARE AVE", so the obvious reading puts the west leg on the neighbouring
      "PENNINGTON-TITUSVILLE RD" SRI - whose segment begins 334 ft west and never
      reaches this junction.

      MORE LEGS THAN PIECES - a leg declared on an SRI that has no approach for it here.
      That one never raised at all: the leg was simply absent from the returned dict, so
      it got no centerline, no curb line, no crossing and no mention.

    The leftover piece's bearing is reported because it IS the `bearing_deg` the missing
    leg needs, so the message contains the fix rather than just the diagnosis.
    """
    bearings = [_piece_bearing_deg(piece) for piece in pieces]
    if len(pieces) != len(leg_names):
        declared = ", ".join(
            f"{name} ({stated:.1f} deg)"
            if (stated := _declared_bearing_deg(legs_cfg, name)) is not None
            else f"{name} (no bearing_deg declared)"
            for name in leg_names)
        raise ValueError(
            f"SRI {sri} splits into {len(pieces)} piece(s) at this junction "
            f"(bearings {', '.join(f'{b:.1f}' for b in bearings)} deg) but the config declares "
            f"{len(leg_names)} leg(s) on it: {declared}. "
            + ("Add the missing leg(s) to config.yaml with the unmatched bearing(s) above - a "
               "road that runs THROUGH the junction has two approaches on one SRI, and NJDOT's "
               "name for an SRI may describe only part of what it covers."
               if len(pieces) > len(leg_names) else
               "Remove the extra leg(s), or move them to the SRI that actually carries them - "
               "a leg on an SRI with no piece here is drawn as nothing at all.")
        )

    # BEST PAIR FIRST, across every (declared leg, piece) at once, rather than walking the
    # pieces and giving each its nearest name. Those agree wherever the declarations are
    # unambiguous - measured over the nine sites, every leg beats the runner-up by 71-90 deg -
    # but the per-piece walk can STRAND a name: the first piece it looks at takes the only
    # declared candidate left, and a name that would have been a better fit for it is handed a
    # piece pointing the other way. It could not bite while every leg was required to declare
    # a bearing, because then the two orders coincide; it bites as soon as some legs declare
    # one and some do not, which is what making the declaration optional allows.
    claimed_by: dict[int, str] = {}
    for _apart, index, name in sorted(
            (_bearing_diff(bearings[index], stated), index, name)
            for name in leg_names
            if (stated := _declared_bearing_deg(legs_cfg, name)) is not None
            for index in range(len(pieces))):
        if index in claimed_by or name in claimed_by.values():
            continue
        claimed_by[index] = name

    spare_names = [name for name in leg_names if name not in claimed_by.values()]
    spare_pieces = [index for index in range(len(pieces)) if index not in claimed_by]
    if len(spare_names) > 1:
        raise ValueError(
            f"SRI {sri} carries {len(spare_names)} legs that declare no bearing_deg "
            f"({', '.join(spare_names)}), and {len(spare_pieces)} of its pieces are unclaimed "
            f"(bearings {', '.join(f'{bearings[i]:.1f}' for i in spare_pieces)} deg). Which "
            f"half of a road a leg is cannot be read off the road - both halves are there - so "
            f"give those legs a `bearing_deg` in config.yaml, using the bearings above. Only a "
            f"leg left with a single piece needs no declaration.")
    if spare_names:
        claimed_by[spare_pieces[0]] = spare_names[0]

    # Keyed in PIECE order, as the per-piece walk left it: a Leg is built per entry by the
    # caller, so this dict's order is `IntersectionModel.legs`' order.
    return {claimed_by[index]: piece for index, piece in enumerate(pieces)}


# A leg is matched to the OSM way whose geometry it lies along: within this far of the
# leg's own centerline, and pointing the same way.
ROAD_MATCH_MAX_OFFSET_FT = 40.0
ROAD_MATCH_MAX_ANGLE_DEG = 30.0
# ...and it has to be a CARRIAGEWAY. Geometry alone is not enough: a parking aisle running
# 0.5 ft from a street at 0.2 deg to it wins the nearest-way tie and carries no tags.
ROAD_MATCH_HIGHWAY_CLASSES = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
})


def _match_legs_to_osm_roads(legs: dict, center_wgs84: Point, center_ft: Point,
                              roads: list[dict] | None = None) -> dict:
    """{leg name: (tags, aligned)} for the OSM highway way each leg runs along.

    `aligned` is True when the way is drawn in the same direction the leg points outward.
    It decides whether OSM's left/right mean the leg's left/right or the reverse.

    Matched on geometry rather than on the street name in config.yaml: names disagree
    between sources ("W Broad St" vs "West Broad Street"), and a leg is a piece of a
    specific way, not of a name.

    A caller may SUPPLY the ways, and a window onto the borough document must - this is the one
    call that carries OSM's operational tags onto the legs, and skipping it is not a missing
    layer but a missing STATEMENT: 5 of the 7 named ways through Broad & Greenwood are
    `overtaking=no`, and without this every one of them drew a dashed single yellow, which says
    on the sheet that passing is permitted where the survey says it is not. `parking:left` and
    `parking:right` travel the same road (see IntersectionModel.parking_restriction_spans).
    """
    if roads is None:
        try:
            roads = fetch_roads(center_wgs84, radius_m=ROAD_CONTEXT_RADIUS_M)
        except Exception as e:   # operational tags are an enhancement, not a dependency
            print(f"  NOTE: couldn't read OSM road tags ({type(e).__name__}); centerline styles "
                  f"fall back to the site config.")
            return {}

    # Projected once, outside the leg loop. Each candidate way was re-transformed for every
    # leg, so a 4-leg junction did the same coordinate transform four times per way.
    carriageways = [
        (road, LineString(zip(*wgs84_to_state_plane.transform(
            [c[0] for c in road["coords_wgs84"]], [c[1] for c in road["coords_wgs84"]]))))
        for road in roads
        if road["tags"].get("highway") in ROAD_MATCH_HIGHWAY_CLASSES
    ]

    out: dict[str, list[RoadSpan]] = {}
    for name, leg in legs.items():
        leg_dir = line_direction(leg.centerline)
        spans = []
        for road, line in carriageways:
            along = np.dot(line_direction(line), leg_dir)
            angle = np.degrees(np.arccos(np.clip(abs(along), -1, 1)))
            if angle > ROAD_MATCH_MAX_ANGLE_DEG:
                continue
            # The stretch of THIS leg the way actually covers, in the leg's own frame. A way is
            # matched on lying along the leg over that stretch, not on being nearest the leg's
            # midpoint - see RoadSpan for why the difference discarded a real restriction.
            stations, offsets = station_offset_many(leg.centerline,
                                                    np.asarray(line.coords, dtype=float))
            lo = max(float(stations.min()), 0.0)
            hi = min(float(stations.max()), leg.centerline.length)
            if hi - lo < MIN_ROAD_SPAN_FT:
                continue
            # Measured over the part that overlaps, so a way running alongside for miles is not
            # judged by how far away its far end wanders.
            covering = (stations >= -MIN_ROAD_SPAN_FT) & (stations <= leg.centerline.length
                                                          + MIN_ROAD_SPAN_FT)
            if not covering.any():
                continue
            if float(np.abs(offsets[covering]).min()) > ROAD_MATCH_MAX_OFFSET_FT:
                continue
            spans.append(RoadSpan(start_ft=lo, end_ft=hi, tags=road["tags"],
                                   aligned=bool(along >= 0), way_id=road.get("id")))
        if spans:
            out[name] = sorted(spans, key=lambda span: span.start_ft)
    return out


# Below this a way barely touches a leg - usually the cross street clipping the junction node -
# and its tags describe a different street.
MIN_ROAD_SPAN_FT = 5.0
