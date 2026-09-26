"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from src.metrics import Comparison, SceneMetrics, marked_stall_runs
from src.geometry.model import hatch_lines_ft, inset_point_at_station, trimmed_curb_lines
from src.geometry.intersection import (IntersectionModel, drawn_kerb_radius_ft,
                                       kerb_lines_with_tags_ft)
from src.geometry.kerbs import KerbType
from src.geometry.treatments import (CENTERLINE_IS_WHITE, DesignState, RaiseCrossing,
                                     RefugeIsland)
from src.provenance import PLOT_STYLE, built_width_provenance
from src.geometry import markings
from src.geometry.paint import RimCause, stroke_width_ft
from src.geometry.markings import require_every_kind
# The 3D render's own figures, read rather than re-stated: how wide it builds a footway, and the
# spacing, angle and phase its hatch strokes are laid on. src/render/export.py is where a 2D/3D
# disagreement about any of them would otherwise start.
from src.render.export import (HATCH_ANGLE_DEG, PAINT_HATCH_SPACING_FT, SIDEWALK_WIDTH_FT,
                               _leg_heading_deg)
from src.render.props import (BIKE_WARNING_PLATE_RADIUS_FT, BOLLARD_RADIUS_FT, DRAWN_BY_PAINT,
                               HYDRANT_RADIUS_FT, MAST_ARM_RADIUS_FT, PED_SIGNAL_HEAD_WIDTH_FT,
                               PUSHBUTTON_HOUSING_DEPTH_FT, PUSHBUTTON_HOUSING_WIDTH_FT,
                               PUSHBUTTON_POST_RADIUS_FT, RECTANGULAR_PLATE_THICKNESS_FT,
                               RECTANGULAR_PLATE_WIDTH_FT, RRFB_PLATE_THICKNESS_FT,
                               RRFB_PLATE_WIDTH_FT, RRFB_POST_RADIUS_FT,
                               SCHOOL_ZONE_PLATE_RADIUS_FT, SIGN_PLATE_THICKNESS_FT,
                               SIGN_POST_RADIUS_FT, STOP_SIGN_PLATE_RADIUS_FT,
                               TACTILE_PAD_DEPTH_FT, TACTILE_PAD_WIDTH_FT,
                               TRAFFIC_SIGNAL_POLE_RADIUS_FT, VEHICLE_SIGNAL_HEAD_WIDTH_FT,
                               YIELD_SIGN_PLATE_RADIUS_FT, build_props, pad_polygon,
                               signalization_conflicts)
from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.render.crosswalks import (CENTERLINE_STRIPE_WIDTH_FT,
                                   TRANSVERSE_LINE_WIDTH_FT, centerline_paint_ft,
                                   centerline_start_ft)
from src.render.frame import frame_covering_radius_m, junction_frame
from src.render.labels import LabelPlacer, ft_per_point
from src.render.scene import SceneGeometry
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_sidewalks,
                                     fetch_street_furniture, fetch_traffic_control)

# DOES NOT MATCH THE 3D, and the comment here claimed it did: it named a TACTILE_PAD_RED that
# blender_props.py has not had since the pads went yellow (TACTILE_PAD_YELLOW, because at this
# camera height a brick-red pad disappeared into the sidewalk). So one detectable warning surface
# is drawn in two colours, and the sheet and the render disagree about what it is. Left red here
# rather than changed in passing - it moves every plan view and is a decision, not a typo - but
# recorded as the disagreement it is rather than as the agreement it was written up as.
TACTILE_PAD_COLOR = "#8c1f14"

TRAFFIC_CONTROL_RADIUS_M = 60  # matches src/render/export.py
BUILDING_CONTEXT_RADIUS_M = 130  # matches src/render/export.py - the FLOOR under the frame's own reach
                                  # within, so a leg's crosswalk_offset here matches what the 3D export computes


def sidewalk_lines_ft(sidewalks: list[dict] | None) -> list[LineString]:
    """Fetched OSM sidewalk ways -> state-plane LineStrings."""
    lines = []
    for walk in sidewalks or []:
        coords = walk["coords_wgs84"]
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        lines.append(LineString(zip(xs, ys)))
    return lines


def _leg_heading(leg, along_ft: float | None = None) -> tuple[float, float]:
    """The direction this leg runs AWAY from the junction, at `along_ft` along it.

    A label's escape direction, and local on purpose: a leg that bends would send a label off
    the road in one place and into the opposing lane in another if one heading served the whole
    leg. See .claude/SKILLS.md 0b on quantities taken over a whole leg.
    """
    line = leg.centerline
    at = line.length * 0.5 if along_ft is None else along_ft
    a = line.interpolate(max(0.0, at - 5.0))
    b = line.interpolate(min(line.length, at + 5.0))
    return (b.x - a.x, b.y - a.y)


def _side_normal(leg, side: str, along_ft: float) -> tuple[float, float]:
    """The direction out across the `side` kerb: what a kerbside label follows to leave the road.

    Rotated off the alignment rather than aimed at the traced kerb, because a traced kerb wanders
    up to 3 ft over a block and the label would swing with it.
    """
    hx, hy = _leg_heading(leg, along_ft)
    sign = 1.0 if str(side) == "left" else -1.0
    return (-hy * sign, hx * sign)


def _at_real_width(kind, geometries: list, style: dict) -> tuple[list, dict]:
    """`geometries` given the body the paint really has, and `style` with the stroke dropped.

    A STRIPE IS A THING ON THE GROUND, NOT A WEIGHT ON A SHEET. Stroking it at a fixed
    `linewidth` makes its drawn width a function of the sheet: 1.5 pt is ~4 ft of ground on a
    630 ft site sheet and ~1.5 ft on a 300 ft one, so the same 0.82 ft edge line reads five
    times too wide on one and twice on the other, and half a stripe of error looks identical
    to none. src/checks.py:MarkingsDoNotCollide says the same thing from the other side, and
    buffers by this same stroke_width_ft to compare a line with anything else.

    IN FEET RATHER THAN CONVERTED TO POINTS, because a point conversion needs the axes' limits
    and plot_design_state sets those LAST, after every marking is drawn - labels.ft_per_point
    read mid-build returns a plausible wrong number rather than its 0.0 sentinel, which is why
    labels are queued and flushed at the end instead. A buffered polygon needs no frame: it is
    the same real width at any DPI, any --frame-scale, any window.

    Flat caps and mitred joins, matching checks.py - a striper's paint ends square.

    NO ESCAPE FOR A COSMETIC DASH, deliberately. This carried one at first, for a kind whose
    `linestyle` broke a continuous line - and the only kind that had one, the contraflow
    divider, does not have continuous geometry: place.py cuts it into real 3 ft dashes with 5 ft
    gaps, so matplotlib was sub-dividing dashes that were already there AND the exemption was
    costing it its width. Breaks belong in the geometry here (see BIKE_LANE_DOTTED_EXTENSION's
    note), so a style that needs a dash pattern is a marking built wrong, and it should be
    visibly solid rather than quietly let through.
    """
    width_ft = stroke_width_ft(kind)
    if width_ft is None or kind.covers_area:
        return geometries, style
    return ([g.buffer(width_ft / 2, cap_style=2, join_style=2) for g in geometries],
            {key: value for key, value in style.items() if key != "linewidth"})


def _hatch_keep_off(paint) -> dict:
    """Where hatching must not run, per leg-side: half a spacing off every OPENING rim.

    The second half of _hatch_strokes_ft - see the trap there.
    """
    rims: dict = {}
    for piece in paint:
        if piece.rim is RimCause.OPENING:
            rims.setdefault((piece.leg, piece.side), []).append(piece.geometry)
    return {key: unary_union(geometries).buffer(PAINT_HATCH_SPACING_FT / 2)
            for key, geometries in rims.items()}


def _hatch_strokes_ft(pieces, state, center_ft, keep_off: dict) -> list:
    """The strokes a hatched zone is actually painted with, as bodies at their real width.

    A `hatch="//"` IS A PATTERN ON THE SHEET. Its spacing and its stroke weight are points, so one
    buffer read as five hairlines on a corridor sheet and as solid paint on a junction one, and
    neither was the 8 ft cadence the render lays - the reader could not count the strokes, measure
    the gap, or see that a zone too narrow to hatch got hatched anyway. These are the strokes:
    hatch_lines_ft, the function src/render/export.py:paint_channels_local_m serializes for
    Blender, at the same spacing, the same angle rule and the same phase origin.

    THE ANGLE RULE, THE PHASE AND THE KEEP-OFF ARE A SECOND COPY of that function's `_hatch`,
    which is a closure inside it and cannot be imported. Three rules that have to agree or the
    sheet shows strokes the street will not have; they are written here in its order and named
    against it. One home for them is the fix and it needs src/render/export.py.
    """
    strokes = []
    for piece in pieces:
        angle_deg = (_leg_heading_deg(state.legs[piece.leg]) + 45
                     if piece.leg in state.legs else HATCH_ANGLE_DEG)
        lines = hatch_lines_ft(piece.geometry, spacing_ft=PAINT_HATCH_SPACING_FT,
                               angle_deg=angle_deg, phase_origin=(center_ft.x, center_ft.y))
        rims = keep_off.get((piece.leg, piece.side))
        if rims is not None:
            # WHOLE STROKES ONLY, as in the export: a stroke that clears the rim is painted and
            # one that does not is absent, never truncated into a stray mark at the kerb.
            lines = [line for line in lines if not line.intersects(rims)]
        strokes += lines
    width_ft = stroke_width_ft(pieces[0].kind)
    return [line.buffer(width_ft / 2, cap_style=2, join_style=2) for line in strokes]


def _draw(ax, geometries, boundary=None, **style) -> None:
    """Draw a group of same-styled geometries as ONE matplotlib collection.

    One collection PER STYLE, never per geometry: the cost of adding a matplotlib collection
    grows with how many are already on the axes, and a proposal builds ~170 of them - measured
    at 2.44 s against 0.016 s, the single largest cost in a 2D build.

    `boundary` is the kwargs for outlining filled shapes, drawn from the same series, so a
    fill and its outline stay one pair rather than two independent draws.
    """
    geometries = [g for g in geometries if g is not None and not g.is_empty]
    if not geometries:
        return
    series = gpd.GeoSeries(geometries)
    series.plot(ax=ax, **style)
    if boundary is not None:
        series.boundary.plot(ax=ax, **boundary)


def _scatter_groups(ax, points_by_style: dict) -> None:
    """One ax.scatter per marker style rather than one per prop, for the same reason.

    ONLY FOR A PROP WHOSE SIZE NOBODY HAS STATED - a streetlight, or a type a site config named
    that this repo has never heard of. `s` is an area in POINTS squared, so anything drawn this
    way is a symbol on the sheet rather than an object on the ground, and it changes size with
    the window. Everything with a dimension goes through _draw_discs or a footprint polygon.
    """
    for style, points in points_by_style.items():
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(xs, ys, **dict(style))


def _draw_discs(ax, discs: list, **style) -> None:
    """Round props - a signal pole, a sign post, a flex post - as circles of their REAL diameter.

    `discs` is [(point, diameter_ft)], one collection for the lot. An EllipseCollection in
    units="xy" is matplotlib's only circle sized in DATA rather than in points, which is the whole
    reason it is used here: a 0.66 ft pole is 0.66 ft of ground at any --frame-scale, any DPI. It
    also keeps one offset per prop, so what was drawn can still be counted, and it is one artist
    for every post on the sheet rather than N buffered polygons.
    """
    discs = [(point, size) for point, size in discs if size > 0]
    if not discs:
        return
    sizes = [size for _point, size in discs]
    ax.add_collection(EllipseCollection(
        widths=sizes, heights=sizes, angles=0, units="xy",
        offsets=[point for point, _size in discs], offset_transform=ax.transData,
        facecolors=style.get("color", "none"), edgecolors=style.get("edgecolor", "none"),
        linewidths=style.get("linewidth", 0), alpha=style.get("alpha"),
        zorder=style.get("zorder", 7)))


# One colour for a flex-post wherever it is drawn from - the treatment layer's own bollard
# pieces, the daylight-zone props, and legend_handles(). Named so a test can count markers of
# this colour rather than trusting that the dispatch has a branch for them at all.
BOLLARD_PLAN_COLOR = "darkorange"

@dataclass(frozen=True)
class PropFootprint:
    """What one piece of a prop covers on the ground, in feet.

    `across_ft` runs perpendicular to the prop's heading and `along_ft` parallel to it - the way
    the 3D builders scale a plate, which is thin along the axis it faces and wide across it, so a
    sign's long axis on the sheet is the direction it faces. A DISC sets the two equal and says
    so: a pole has no facing, and drawing one as a square would invent an orientation.
    """
    across_ft: float
    along_ft: float
    disc: bool = False


def _disc(diameter_ft: float) -> PropFootprint:
    return PropFootprint(diameter_ft, diameter_ft, disc=True)


# A sign post, in the grey the 3D paints it (blender_props.SIGN_POST_GRAY). One style for every
# post on the sheet, so they are one collection however many sign types stand on it.
POST_STYLE = dict(color="#59595e", zorder=7)
SIGN_POST = (_disc(2 * SIGN_POST_RADIUS_FT), POST_STYLE)

# How each prop type is drawn in plan: its real footprint on the ground, and the colours that
# tell it from the next one. Data rather than an if/elif chain, so every prop of a type draws in
# one call and a new type is a row here rather than a branch that can be forgotten.
#
# AT THE SIZE THE 3D RENDER BUILDS IT - the dimensions block in src/render/props.py, mirrored
# from the bpy calls that build these - and never at a fixed marker size. matplotlib's `s` is an
# area in POINTS squared, so `s=44` drew a stop sign 6.6 pt wide whatever the sheet showed: about
# 2 ft of ground on a junction plan and 8 ft on a corridor one. At that size the drawing cannot
# answer the question a plan view exists for - does this pole stand in the bike lane - and the
# tactile pad, drawn true-size all along, is why that question IS answerable for pads.
#
# What a plan sees of a SIGN is its PLATE: the post is 0.26 ft across and the plate up to 2.5 ft,
# so the plate is the object on the sheet. The post is drawn too, and is the smaller of the two.
PROP_MARKERS = {
    "traffic_signal_pole":    ((_disc(2 * TRAFFIC_SIGNAL_POLE_RADIUS_FT),
                                dict(color="limegreen", edgecolor="black", linewidth=0.6,
                                     zorder=7)),),
    "pedestrian_signal_head": ((PropFootprint(PED_SIGNAL_HEAD_WIDTH_FT, PED_SIGNAL_HEAD_WIDTH_FT),
                                dict(color="limegreen", edgecolor="black", linewidth=0.6,
                                     zorder=7)),),
    "stop_sign":              (SIGN_POST,
                               (PropFootprint(2 * STOP_SIGN_PLATE_RADIUS_FT,
                                              SIGN_PLATE_THICKNESS_FT),
                                dict(color="red", edgecolor="white", linewidth=0.6, zorder=7))),
    "pedestrian_pushbutton":  ((_disc(2 * PUSHBUTTON_POST_RADIUS_FT), POST_STYLE),
                               (PropFootprint(PUSHBUTTON_HOUSING_WIDTH_FT,
                                              PUSHBUTTON_HOUSING_DEPTH_FT),
                                dict(color="gold", edgecolor="black", linewidth=0.5, zorder=8))),
    "rrfb":                   ((_disc(2 * RRFB_POST_RADIUS_FT), POST_STYLE),
                               (PropFootprint(RRFB_PLATE_WIDTH_FT, RRFB_PLATE_THICKNESS_FT),
                                dict(color="gold", edgecolor="black", linewidth=0.6, zorder=8))),
    "fire_hydrant":           ((_disc(2 * HYDRANT_RADIUS_FT),
                                dict(color="firebrick", zorder=7)),),
    "yield_sign":             (SIGN_POST,
                               (PropFootprint(2 * YIELD_SIGN_PLATE_RADIUS_FT,
                                              SIGN_PLATE_THICKNESS_FT),
                                dict(color="white", edgecolor="red", linewidth=1.2, zorder=7))),
    "no_turn_on_red_sign":    (SIGN_POST,
                               (PropFootprint(RECTANGULAR_PLATE_WIDTH_FT,
                                              RECTANGULAR_PLATE_THICKNESS_FT),
                                dict(color="white", edgecolor="red", linewidth=1.2, zorder=7))),
    # MUTCD W-series warning plates at a bikeway terminus: W9-5 BIKE LANE ENDS, and the
    # W16-21P TWO-WAY BICYCLE CROSS TRAFFIC plaque under a crossroad's STOP. The plate is the
    # widest of the sign family, which is what the reader has to tell it by now that the shapes
    # are footprints rather than glyphs - two prop types drawn identically are two things the
    # reader cannot tell apart, and
    # tests/test_props.py:test_every_prop_type_is_drawn_distinguishably holds that down.
    "bike_warning_sign":      (SIGN_POST,
                               (PropFootprint(2 * BIKE_WARNING_PLATE_RADIUS_FT,
                                              SIGN_PLATE_THICKNESS_FT),
                                dict(color="gold", edgecolor="black", linewidth=1.2, zorder=7))),
    # The R9-23 series regulatory plate at the two-stage turn box (MUTCD 9B.18). White plate,
    # black edge - the NTOR sign is the other white rectangle and is edged RED. They are the SAME
    # PLATE in 3D (add_bike_regulatory_sign is add_no_turn_on_red_sign), so colour is the only
    # honest way a true-size plan tells them apart.
    "bike_regulatory_sign":   (SIGN_POST,
                               (PropFootprint(RECTANGULAR_PLATE_WIDTH_FT,
                                              RECTANGULAR_PLATE_THICKNESS_FT),
                                dict(color="white", edgecolor="black", linewidth=1.2, zorder=7))),
    # Drawn in 3D since the school-zone builder was added and never here, so a relocated
    # school zone sign was invisible in plan. Fluorescent yellow-green, as built.
    "school_zone_sign":       (SIGN_POST,
                               (PropFootprint(2 * SCHOOL_ZONE_PLATE_RADIUS_FT,
                                              SIGN_PLATE_THICKNESS_FT),
                                dict(color="greenyellow", edgecolor="black", linewidth=0.6,
                                     zorder=7))),
    # THE ONE PROP WITH NO SIZE ANYWHERE. add_streetlight imports an external Poly Haven glTF and
    # the procedural fallback beside it says in its own docstring that it is not what ships, so
    # there is no figure to draw a lamp at - see src/render/props.py. A fixed-size marker, and
    # the only symbol left on this sheet that changes with the window.
    "streetlight":            ((None, dict(color="dimgrey", marker="*", s=34, zorder=6)),),
    "bollard":                ((_disc(2 * BOLLARD_RADIUS_FT),
                                dict(color=BOLLARD_PLAN_COLOR, edgecolor="black", linewidth=0.4,
                                     zorder=7)),),
}
# EVERY KERB IS THE SAME WIDTH, because every kerb is built at KERB_WIDTH_M in
# scripts/blender/blender_scene.py - raised, lowered and flush alike. What tells them apart in 3D
# is HEIGHT (src/render/export.py:KERB_HEIGHT_M: 0.20 / 0.07 / 0.055 m), and a plan cannot draw a
# height. So the width here is the real one and the DASH PATTERN is this view's way of saying the
# thing a plan cannot show - which is worth saying, because a LOWERED kerb is where a vehicle
# crosses and the kerbside markings break over one and not the other, so a reader looking at a
# gap in a bike lane needs to see the driveway that caused it. UNKNOWN is drawn distinctly rather
# than as raised, because "nobody said" is not "raised"; all 95 kerbs mapped here are tagged.
KERB_WIDTH_FT = 0.15 / FT_TO_M
# (on, off, on, off, ...) in FEET, repeated along the kerb, or None for a kerb drawn whole.
# In feet rather than as a matplotlib linestyle for the same reason the width is: a pattern in
# points is a statement about the sheet, and at --frame-scale 2.5 the old "--" said something
# different about the same kerb. A cadence is a drawing convention, not a measurement - the
# width above is the measurement - but a convention still has to hold still.
KERB_DASHES_FT = {
    KerbType.RAISED:  None,
    KerbType.LOWERED: (4.0, 2.5),
    KerbType.FLUSH:   (1.0, 2.0),
    KerbType.UNKNOWN: (4.0, 2.0, 1.0, 2.0),
}
KERB_STYLE = {
    KerbType.RAISED:  dict(color="black", zorder=6),
    KerbType.LOWERED: dict(color="black", zorder=6),
    KerbType.FLUSH:   dict(color="black", zorder=6),
    KerbType.UNKNOWN: dict(color="dimgrey", zorder=6),
}


def _cadence_pieces(line: LineString, cadence: tuple | None) -> list[LineString]:
    """`line` cut into an (on, off, ...) cadence in feet, or whole if there is none.

    CUT, not styled. A kerb drawn at its real width is a POLYGON, and a polygon has no linestyle -
    matplotlib would run the dashes round its rim instead of along it. Same answer the contraflow
    divider already gets: a break the drawing means belongs in the geometry.
    """
    if not cadence:
        return [line]
    pieces, at, step = [], 0.0, 0
    while at < line.length:
        run = cadence[step % len(cadence)]
        if step % 2 == 0:
            piece = substring(line, at, min(at + run, line.length))
            if piece.geom_type == "LineString" and piece.length > 0:
                pieces.append(piece)
        at += run
        step += 1
    return pieces


# A driveway is PAVING, so it is drawn as paving: a filled strip under everything else, in a
# browner grey than the roadway so it reads as private access rather than carriageway. Not a
# centreline - this drawing already carries three kinds of those, and the thing a driveway
# exists to explain (the gap in the kerbside markings at its mouth) has to read as a surface.
PAVED_STYLE = dict(color="#8a7a68", alpha=0.55, zorder=2)
PAVED_EDGE = dict(color="#5d5044", linewidth=0.8, zorder=2)
# A surveyed outline gets a solid edge and a widened one a dashed edge, because the difference is
# the same one this drawing already makes for a curb line: an extent somebody traced against an
# extent this project inferred. A parking lot is mapped as an area; a driveway and an aisle are
# centrelines widened by an assumed number (PavedSurface.width_ft).
PAVED_EDGE_ASSUMED = dict(PAVED_EDGE, linestyle="--")


def _draw_paved_surfaces(ax, paved) -> None:
    """The junction's driveways, parking aisles and parking lots, off the MODEL - already
    projected and already turned into the polygon both views draw, so the plan view and the 3D
    render cannot disagree about where any of it is or how wide it is. See
    src/geometry/intersection/junction.py:PavedSurface."""
    for surveyed, edge in ((True, PAVED_EDGE), (False, PAVED_EDGE_ASSUMED)):
        _draw(ax, [p.surface for p in paved or ()
                   if p.surface is not None and p.extent_is_surveyed == surveyed],
              boundary=edge, **PAVED_STYLE)


def _draw_kerbs(ax, kerb_lines) -> None:
    """The traced kerbs, grouped by what OSM says each one is.

    One collection per kerb type rather than per line, the same reason _draw groups everything
    else. Grouped by TYPE and not drawn uniformly because a dropped kerb is why a marking stops:
    src/geometry/paint/openings.py:kerb_opening_bands breaks the kerbside paint over exactly these, and a
    reader looking at a gap in a bike lane needs to see the driveway that caused it.
    """
    by_type: dict = {}
    for line, tags, _way_id in kerb_lines:
        by_type.setdefault(KerbType.from_tags(tags), []).append(line)
    for kerb, lines in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        _draw(ax, [piece.buffer(KERB_WIDTH_FT / 2, cap_style=2, join_style=2)
                   for line in lines
                   for piece in _cadence_pieces(line, KERB_DASHES_FT[kerb])],
              **KERB_STYLE[kerb])


# Site- or scenario-specific extras (school zone signs, RRFB relocations, ...) have no
# dedicated marker: they are whatever a config or a proposal named.
EXTRA_PROP_MARKER = dict(color="darkgoldenrod", marker="^", s=30, zorder=7)

# How much ground a label has to leave clear around a prop, in POINTS, converted to feet once the
# limits are set (src/render/labels.py:ft_per_point). DELIBERATELY A SHEET QUANTITY, and the only
# one left here: the props themselves are now drawn at their real size, and their real size is
# 0.3-2.5 ft - smaller than the letters of the label that would sit on them. Keeping a label off
# 0.33 ft of flex post is keeping it off nothing, so what is reserved is the reader's-eye space
# around the prop rather than the prop.
PROP_MARKER_RADIUS_PT = 3.2

# How each marking is drawn in plan. Styling is a real per-marking choice - what colour says
# "this asphalt is spare" versus "parking here is illegal" is a judgement, not something
# derivable - so this table is written by hand. require_every_kind is what makes forgetting an
# entry impossible: a marking declared in src/geometry/markings.py with no style here raises on
# import, rather than being silently absent from the plan view while the 3D render draws it. An
# OBJECT is exempt - a flex post is drawn as a post, not as paint (see BOLLARD_PLAN_COLOR).
#
# NO `hatch` ON A FILL ANY MORE. A hatched zone's strokes are real paint, so they are drawn as
# real paint by _hatch_strokes_ft, in the colour PAINT_FILL_EDGE gives the wash. The translucent
# WASH stays and is a convention: no colour is applied to a street, and it is what lets the
# reader (and the legend) tell a daylight zone from a parking buffer at a glance.
#
# EVERY `linewidth` BELOW IS DEAD. _at_real_width buffers a stroked marking to the body it really
# has and drops the key before drawing - and it drops it for all eleven: measured, every LINE
# kind here has a stroke width (0.492 or 0.820 ft) and none of them covers_area, which are the
# two conditions. They are still written down only because
# tests/test_paint.py:test_every_fill_colour_has_an_outline_colour tells a fill from a line by
# whether the style HAS a linewidth key, so deleting them turns eleven lines into eleven fills
# with no outline colour and reds a test. Delete them in the commit that teaches that test to
# ask the marking (`kind.covers_area`) instead, which is what this file already does.
PAINT_STYLE = require_every_kind({
    markings.LANE_NARROWING_FILL: dict(color="gold", alpha=0.5, zorder=3),
    markings.TAPER_FILL:          dict(color="gold", alpha=0.5, zorder=3),
    markings.BUFFER_FILL:         dict(color="gold", alpha=0.5, zorder=3),
    # The statutory no-parking zone at the corner (R.S. 39:4-138). Drawn in a distinct
    # colour from the ordinary buffer hatch because it is a different claim: not "this
    # asphalt is spare", but "parking here is illegal and this proposal marks it".
    markings.DAYLIGHT_FILL:       dict(color="orangered", alpha=0.40, zorder=3),
    markings.CORNER_HATCH_FILL:   dict(color="gold", alpha=0.5, zorder=3),
    markings.APRON:               dict(color="peru", alpha=0.6, zorder=3),
    # A green bike lane's asphalt. Under the stripes' zorder so the white edge lines read on
    # top of it, exactly as they do on the street and in the render.
    markings.BIKE_LANE_SURFACE:   dict(color="mediumseagreen", alpha=0.45, zorder=2),
    # THE SAME LANE WITH NO GREEN ON IT - an existing conventional bike lane. Drawn, because a
    # reader of a 2D sheet has to see where the lane runs; drawn GREY and faint, because the
    # legend's green swatch says "green surface" and means it. The 3D render draws nothing here
    # at all (markings.NOT_DRAWN_IN_3D): on asphalt, an unpainted lane IS the asphalt, and the
    # white stripes and the symbol beside it are what make it a bike lane.
    markings.BIKE_LANE_UNCOLOURED_SURFACE: dict(color="slategrey", alpha=0.18, zorder=2),
    markings.LANE_EDGE_LINE:      dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.TAPER_LINE:          dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.BUFFER_EDGE_LINE:    dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.DAYLIGHT_EDGE_LINE:  dict(color="orangered", linewidth=1.5, zorder=3),
    # The square end of a zone with no crossing to be cut by and no room to taper.
    markings.ZONE_END_LINE:       dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.PARKING_EDGE_LINE:   dict(color="steelblue", linewidth=1.5, zorder=3),
    # THE SAME STRIPE AS THE LINE ABOVE, and drawn a different colour on purpose: gold is this
    # sheet's real yellow paint (see the centreline), so the reader can see that the west kerb of
    # a one-way street carries a yellow line and the east kerb a white one. MUTCD 3B.09 P3.
    markings.LEFT_EDGE_LINE:      dict(color="gold", linewidth=1.5, zorder=3),
    markings.STALL_DIVIDER:       dict(color="steelblue", linewidth=1, zorder=3),
    # An exclusive bike lane. Green, because that is what a bike lane is coloured on a real
    # street and in every other agency's drawings - and a colour of its own is the point: this
    # is the one treatment here that says a vehicle BELONGS in the strip, where the gold
    # hatching says nothing does.
    markings.BIKE_LANE_EDGE_LINE: dict(color="seagreen", linewidth=1.6, zorder=3),
    # Drawn identically to the continuous line, because it IS that line: the breaks are in the
    # geometry rather than in a dash pattern, so what the plan view draws here is a row of short
    # stripes - the same row the render extrudes. Styling it differently would say the paint is a
    # different colour across a driveway, which it is not.
    markings.BIKE_LANE_DOTTED_EXTENSION: dict(color="seagreen", linewidth=1.6, zorder=3),
    markings.BIKE_BUFFER_FILL:    dict(color="mediumseagreen", alpha=0.35, zorder=3),
    # A two-way lane's centre stripe. Yellow and dashed, the same as the roadway's own
    # centreline and for the same reason - it divides opposing traffic. Drawn above the green
    # surface it sits on (zorder 4, over the surface's 2) or the fill hides it.
    # NO `linestyle` HERE: place.py already cuts this into 3 ft dashes with 5 ft gaps, so a dash
    # pattern on top sub-divided dashes that were there - and cost the stripe its real width,
    # since _at_real_width cannot give a body to something matplotlib is going to break up.
    markings.BIKE_CONTRAFLOW_DIVIDER: dict(color="goldenrod", linewidth=1.3, zorder=4),
    # The BIKE LANE symbol, white on the green like the real marking, and above the surface it
    # sits on for the same reason the contraflow stripe is.
    markings.BIKE_LANE_SYMBOL:    dict(color="white", alpha=0.95, zorder=4),
    # The arrow beside it (MUTCD 9E.01(04), 9E.11(06)). Drawn exactly as the symbol is, because
    # it is the same paint - the two are one instruction to a striper and reading them in two
    # colours would invent a distinction the street does not have.
    markings.BIKE_THROUGH_ARROW:  dict(color="white", alpha=0.95, zorder=4),
    # THE TWO-STAGE TURN BOX. Green like the lane, because 9E.11(11)-(12) makes it the same
    # coloured pavement and a reader has to see at a glance that the box and the bikeway are one
    # facility - but a shade darker and more opaque than BIKE_LANE_SURFACE, because it is a place
    # to STAND rather than to ride through, and the sheet has to be able to say which is which.
    markings.TURN_BOX_SURFACE:    dict(color="mediumseagreen", alpha=0.55, zorder=2),
    # 9E.11(07)'s solid white line, on all four sides. Above the green it bounds, like every
    # other stripe over a coloured surface here.
    markings.TURN_BOX_EDGE_LINE:  dict(color="white", linewidth=1.8, zorder=4),
    # THE SHARROW, and the one marking on this sheet that must be seen NOT to be on green:
    # 9E.09(05) forbids a green background under it. Drawn white on whatever the road already is,
    # which past the end of the facility is bare asphalt.
    markings.SHARED_LANE_MARKING: dict(color="white", alpha=0.95, zorder=4),
}, "plan_view.PAINT_STYLE")
# Outline colour for each filled zone's own fill colour. White outlines white: a symbol is a
# SOLID glyph, not a hatched zone that needs a rim to read as bounded, so giving it a contrasting
# edge would draw a border no striper paints.
#
# NOT a lookup with a fallback. A fill colour missing from here used to raise KeyError deep in
# the plan build, several phases after the style was written, which is the same
# forgot-one-of-the-six-places failure require_every_kind exists to stop - so
# tests/test_paint.py:test_every_fill_colour_has_an_outline_colour reads both tables and fails
# on import instead. A .get(colour, colour) default would have made the miss invisible, which
# is worse: the zone would simply lose its rim.
PAINT_FILL_EDGE = {"gold": "goldenrod", "peru": "saddlebrown", "orangered": "orangered",
                   "mediumseagreen": "seagreen", "white": "white", "slategrey": "slategrey"}


def _draw_props(ax, model: IntersectionModel, state: DesignState, crosswalk_offsets: dict,
                 traffic_control: list[dict] | None, street_furniture: list[dict] | None,
                 crossings: list[dict] | None, labels: LabelPlacer, dimension_labels: bool,
                 pavement=None, kerb_ways: list[dict] | None = None):
    """Draw the street furniture the 3D render will build - signals above all.

    This calls the SAME src/render/props.py:build_props the export does, so the plan view shows
    exactly the hardware Blender will place - a signal appearing here that nobody proposed is a
    visible error rather than a surprise three phases later. Three of these four junctions are
    signalized and one is not.

    And at the size Blender will place it at: every prop here is its real footprint on the ground
    (PROP_MARKERS), so the question this drawing is read to answer - does this pole stand in the
    bike lane, does that pad reach the roadway - is answered by the drawing rather than by the
    size of the dot somebody chose.

    Bollards tagged DRAWN_BY_PAINT are skipped because the treatment layer already drew them
    (LaneNarrowingBollards, ParkingBufferBollards emit their own paint pieces). Bollards WITHOUT
    that tag - ProtectDaylightZone's posts - exist only as props, so skipping every bollard shows
    none in plan while the render of the same scenario shows thirteen.
    """
    # Every traced kerb inside the frame, which is the same set the 3D export writes - see
    # src/geometry/intersection/kerb_sources.py:kerb_lines_with_tags_ft on why the drawing test is not the
    # corner-fit's near set, and src/render/export.py for the matching call.
    kerb_lines = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                          radius_ft=drawn_kerb_radius_ft(), kerbs=kerb_ways)
    props = build_props(model, state, crosswalk_offsets, model.center_ft, traffic_control,
                         street_furniture, crossings,
                         kerb_ways if kerb_ways is not None
                         else fetch_kerbs(model.center_wgs84, radius_m=120),
                         pavement=pavement)
    _draw_paved_surfaces(ax, model.paved_surfaces)
    _draw_kerbs(ax, kerb_lines)

    # Grouped by style, then drawn once per group. See _draw / _draw_discs / _scatter_groups.
    discs_by_style: dict[tuple, list] = {}
    footprints_by_style: dict[tuple, list] = {}
    marker_points: dict[tuple, list] = {}
    pads, arms, heads = [], [], []
    signal_count = 0
    for prop in props:
        kind = prop["type"]
        if kind == "bollard" and prop.get(DRAWN_BY_PAINT):
            continue
        x, y = prop["position_ft"]
        heading = prop["heading_deg"]
        if kind == "tactile_paving_pad":
            # Its size arrives WITH the prop, because the step-back that keeps it off the roadway
            # is half its depth - see src/render/props.py:pad_polygon.
            pads.append(pad_polygon(x, y, heading,
                                      depth_ft=prop.get("pad_depth_ft", TACTILE_PAD_DEPTH_FT),
                                      width_ft=prop.get("pad_width_ft", TACTILE_PAD_WIDTH_FT)))
            continue
        if kind == "traffic_signal_pole":
            signal_count += 1
            # The mast arm is the part that reaches out over the roadway, and its length
            # is derived from a real leg width - worth seeing in plan, since it's the
            # most visually dominant thing in the 3D render. AT ITS REAL THICKNESS too: the
            # arm is 0.33 ft of steel and a 1.6 pt stroke drew it four times that on a site
            # sheet, so the one prop already drawn to a real LENGTH was not drawn to scale.
            arm_deg, arm_ft = prop.get("arm_heading_deg"), prop.get("arm_length_ft")
            if arm_deg is not None and arm_ft:
                end = (x + np.cos(np.radians(arm_deg)) * arm_ft,
                       y + np.sin(np.radians(arm_deg)) * arm_ft)
                arms.append(LineString([(x, y), end]).buffer(MAST_ARM_RADIUS_FT, cap_style=2,
                                                              join_style=2))
                # The vehicle head hangs off the far end of that arm in 3D
                # (add_traffic_signal_pole -> add_vehicle_signal_head) and had nothing in plan,
                # so the sheet showed a bare stick where the render shows the signal itself.
                heads.append(pad_polygon(end[0], end[1], heading,
                                          depth_ft=VEHICLE_SIGNAL_HEAD_WIDTH_FT,
                                          width_ft=VEHICLE_SIGNAL_HEAD_WIDTH_FT))
        for footprint, style in PROP_MARKERS.get(kind, ((None, EXTRA_PROP_MARKER),)):
            key = tuple(sorted(style.items()))
            if footprint is None:
                marker_points.setdefault(key, []).append((x, y))
            elif footprint.disc:
                discs_by_style.setdefault(key, []).append(((x, y), footprint.across_ft))
            else:
                footprints_by_style.setdefault(key, []).append(
                    pad_polygon(x, y, heading, depth_ft=footprint.along_ft,
                                width_ft=footprint.across_ft))

    _draw(ax, arms, color="black", zorder=7)
    _draw(ax, heads, color="#151515", zorder=7)
    _draw(ax, pads, color=TACTILE_PAD_COLOR, alpha=0.85, zorder=8,
          boundary=dict(color="black", linewidth=0.5, zorder=8))
    for style, footprints in footprints_by_style.items():
        _draw(ax, footprints, **dict(style))
    for style, discs in discs_by_style.items():
        _draw_discs(ax, discs, **dict(style))
    _scatter_groups(ax, marker_points)

    if dimension_labels:
        control = (f"SIGNALIZED - {signal_count} signal pole(s)" if signal_count
                    else "NOT signalized - stop/yield control")
        labels.caption(control, (0.5, 0.005), fontsize=8, fontweight="bold",
                       color="black" if signal_count else "dimgrey",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    # Handed to the invariant pass rather than rebuilt there: build_props is the most
    # expensive thing in the plan view, and checking a DIFFERENT set of props from the one
    # drawn would defeat the point of checking at all.
    return props


def _draw_surveyed_crossings(ax, crossings: list[dict] | None):
    """Draw each OSM crossing way as surveyed - its real endpoints, length and skew.

    THE REFERENCE LINE, not the paint (_draw_unmodelled_crossings below draws that). A crossing
    way runs sidewalk-centreline to sidewalk-centreline, so it is consistently 8-24 ft longer than
    the roadway it spans; behind the kerb-to-kerb band the two can be compared by eye, and where
    the band sticks out past the ends of this line the leg's configured width is too big. That
    comparison is why this overlay exists.
    """
    lines = sidewalk_lines_ft(crossings)
    _draw(ax, lines, color="darkviolet", linewidth=1.0, linestyle=":", alpha=0.8, zorder=5)
    ends = [line.coords[i] for line in lines for i in (0, -1)]
    if ends:
        ax.scatter(*zip(*ends), color="darkviolet", s=8, marker="o", zorder=5)


def _draw_unmodelled_crossings(ax, scene):
    """Paint every surveyed crossing that belongs to NO modelled leg, from its own traced way.

    See docs/network-renderer-plan.md. Only `leg is None`: the four this junction models are
    drawn by _draw_crosswalks from the leg's own band, including whatever a proposal restyles them
    to, and drawing both would put two crossings 1.44-2.73 ft apart on the same ground.

    Styled by SceneGeometry.surveyed_crossing_markings, NOT from each way's tags here, so a
    proposal that repaints every crossing continental reaches these too and an unrecorded crossing
    still gets nothing rather than invented transverse lines. Drawn in the same white as the
    modelled crosswalks: on the ground they are the same paint.
    """
    drawn = scene.surveyed_crossing_markings()
    bars = [bar for _c, crossing_bars, _l in drawn for bar in crossing_bars]
    lines = [line for _c, _b, crossing_lines in drawn for line in crossing_lines]
    if bars:
        _draw(ax, bars, facecolor="white", edgecolor="0.35", linewidth=0.4, zorder=6)
    if lines:
        # AT THE WIDTH THE 3D RENDER LAYS THEM AT - see _at_real_width. The dark rim goes round
        # the body rather than under a wider stroke: white on grey asphalt at a 431 ft frame is
        # otherwise nearly invisible, and a stroke drawn "slightly wider" than a cosmetic one is
        # a width with no meaning on the ground.
        _draw(ax, [line.buffer(TRANSVERSE_LINE_WIDTH_FT / 2, cap_style=2, join_style=2)
                   for line in lines],
              facecolor="white", edgecolor="0.35", linewidth=0.4, zorder=6)


def _draw_crosswalks(ax, scene: SceneGeometry, labels: LabelPlacer, dimension_labels: bool):
    """Draw each leg's crosswalk and stop bar exactly where the 3D export puts them.

    Gating mirrors scripts/blender/blender_scene.py precisely - a crosswalk is painted
    only on a leg listed in the config's `intersection.existing_marked_crosswalks` and
    not already carrying a raised crossing (which is drawn separately). Legs failing
    that gate still get their resolved offset drawn as a thin outline, because "this
    leg has no marked crossing" is itself a finding worth seeing in the reconstruction,
    and because a wrong offset on an unmarked leg is exactly the kind of latent error
    that only surfaces later when a proposal marks it.

    Surveyed (OSM) and estimated positions are drawn differently, following the same
    convention this view already uses for confirmed vs. estimated curb widths.

    Both footprints come from `scene` and are never rebuilt here - see src/render/scene.py.
    Rebuilding them draws them somewhere neither the render nor the checks agree with, which is
    the one failure this view exists to catch rather than commit.
    """
    state = scene.state
    raised = {t.target.leg for t in state.treatments_of(RaiseCrossing)}
    # Grouped by how each band gets drawn, which turns on two independent facts: whether it is
    # painted at all, and whether its position is surveyed or estimated.
    painted_surveyed, painted_estimated, unpainted = [], [], []

    for leg_name in state.legs:
        offset = scene.crosswalk_offsets[leg_name]
        painted = leg_name in scene.marked_crosswalks and leg_name not in raised
        band = scene.crosswalk_bands[leg_name]
        if not painted:
            unpainted.append(band)
        elif offset.is_surveyed:
            painted_surveyed.append(band)
        else:
            painted_estimated.append(band)

        if dimension_labels:
            centroid = band.centroid
            note = "" if painted else "\n(unmarked)"
            edge = "darkviolet" if offset.is_surveyed else "crimson"
            labels.dimension(
                f"{offset.offset_ft:.0f} ft\n{'OSM' if offset.is_surveyed else 'est.'}{note}",
                (centroid.x, centroid.y), toward=_leg_heading(state.legs[leg_name]),
                fontsize=5.5, pad=0.12, color=edge if painted else "grey", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    for bands, edge, dash in ((painted_surveyed, "darkviolet", "-"),
                              (painted_estimated, "crimson", "--")):
        _draw(ax, bands, color="white", alpha=0.95, zorder=4,
              boundary=dict(color=edge, linewidth=1.4, linestyle=dash, zorder=4))
    # Unpainted legs get the outline only - "this leg has no marked crossing" is itself a
    # finding, and a wrong offset on one is a latent error until a proposal marks it.
    if unpainted:
        gpd.GeoSeries(unpainted).boundary.plot(ax=ax, color="grey", linewidth=0.7,
                                                linestyle=":", zorder=4)

    # Stop bars, on the same terms the export uses: signalized sites only, which is what
    # leaves scene.stop_bar_bands empty everywhere else. A bar covers only the ENTERING half
    # of the roadway - a driver stops in their own lanes, never across the opposing ones -
    # and inherits the crossing's surveyed skew, being painted parallel to it.
    _draw(ax, scene.stop_bar_bands.values(), color="dimgrey", alpha=0.9, zorder=4)


@dataclass(frozen=True)
class PlotResult:
    """What one drawn panel reports back: what failed, and what the design achieves.

    Both, from one call, because both are read off the SAME resolved scene the panel drew
    (src/render/scene.py). A caller that wanted the outcome numbers would otherwise have to
    resolve the crossings, the paint and the props a second time to measure them - which is
    how the plan view, the export and the invariants came to be checking three different sets
    of geometry before SceneGeometry existed.
    """
    violations: list
    metrics: SceneMetrics


def draw_change_panel(fig, before: SceneMetrics, after: SceneMetrics) -> Comparison:
    """The summary block beside a before/after pair: what the proposal actually changes.

    Every other number on this drawing is an INPUT - measured street width, stall length, corner
    radius - saying what is built. This says what it accomplishes, which is what the drawing is
    shown in order to argue.

    Drawn in figure coordinates outside the axes, so it extends the saved image under
    bbox_inches="tight" rather than covering geometry - the same trick the legend already
    uses below the panels.
    """
    comparison = Comparison.of(before, after)
    fig.text(1.01, 0.5, comparison.panel_text(), ha="left", va="center", family="monospace",
             fontsize=8.5, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#444444", alpha=0.97))
    return comparison


def plot_design_state(ax, model: IntersectionModel, state: DesignState, title: str, dimension_labels: bool = True,
                       crossings: list[dict] | None = None, sidewalks: list[dict] | None = None,
                       traffic_control: list[dict] | None = None, street_furniture: list[dict] | None = None,
                       pavement=None, kerb_ways: list[dict] | None = None, frame=None,
                       stop_lines: list[dict] | None = None):
    """Every OSM layer may be SUPPLIED rather than fetched, exactly as export_scenario takes them,
    so the two views cannot be drawn from different data. See that function for why a crop has to
    supply them; `pavement` likewise overrides the ring built from corner fillets it has none of.
    """
    if sidewalks is None:
        try:
            sidewalks = fetch_sidewalks(
                model.center_wgs84,
                radius_m=frame_covering_radius_m(model, BUILDING_CONTEXT_RADIUS_M))
        except RuntimeError as e:
            print(f"  WARNING: could not fetch OSM sidewalks ({e}) - drawn without them.")
            sidewalks = []

    if traffic_control is None:
        try:
            traffic_control = fetch_traffic_control(model.center_wgs84, radius_m=TRAFFIC_CONTROL_RADIUS_M)
        except RuntimeError as e:
            print(f"  WARNING: could not fetch OSM traffic control ({e}) - falling back to guesses.")
            traffic_control = []
    if street_furniture is None:
        try:
            street_furniture = fetch_street_furniture(
                model.center_wgs84,
                radius_m=frame_covering_radius_m(model, BUILDING_CONTEXT_RADIUS_M))
        except RuntimeError:
            street_furniture = []
    for note in signalization_conflicts(model, traffic_control):
        print(f"  NOTE: {note}")

    # Always resolved, not just when a treatment needs them: the crosswalks ARE the subject of
    # this project, so a plan view that omits them is not a reconstruction the 3D render can be
    # checked against - a mis-matched OSM crossing (src/render/crosswalks.py:_match_crossings_to_legs)
    # is then visible only in the render. fetch_crossings is disk-cached per (center, radius).
    if crossings is None:
        try:
            crossings = fetch_crossings(
                model.center_wgs84,
                radius_m=frame_covering_radius_m(model, BUILDING_CONTEXT_RADIUS_M))
        except RuntimeError as e:
            # Overpass unreachable and nothing cached. Don't fail the whole plan view for
            # context data - fall back to the geometric estimate, which is drawn in a
            # visibly different style and labeled as such, so it can't be mistaken for
            # surveyed placement.
            print(f"  WARNING: could not fetch OSM crossings ({e}) - crosswalk positions "
                  f"shown are geometric estimates, not surveyed.")
            crossings = []
    # Once, for the whole figure: the pavement, every crossing and stop bar footprint, and the
    # offsets/skews everything else is measured from. See src/render/scene.py.
    scene = SceneGeometry.resolve(model, state, crossings, stop_lines=stop_lines,
                                   pavement=pavement, kerb_ways=kerb_ways)
    pavement = scene.pavement
    # Queued, not drawn: every label below is sized in points and has to be placed in feet, and
    # the conversion is a fact about axes limits this function sets last. See src/render/labels.py.
    # Always built, whatever `dimension_labels` says: that flag decides whether the DIMENSIONS
    # are asked for, but the panel captions are drawn either way and still stand on ground.
    labels = LabelPlacer()

    # Empty when the site declares no parcels layer. geopandas warns rather than raising on an
    # empty plot, so this is about keeping a parcel-less site's render quiet, not about crashing.
    if not model.parcels.empty:
        model.parcels.boundary.plot(ax=ax, color="tan", linewidth=0.6, zorder=1)
    if not model.corner_parcels.empty:
        model.corner_parcels.boundary.plot(ax=ax, color="saddlebrown", linewidth=1.5, zorder=1)

    _draw(ax, [pavement], color="#d9d9d9", zorder=2)

    # Real OSM sidewalk footways, drawn behind everything else. These are what the crossing ways
    # actually connect to, and they bound where the curb can possibly be
    # (src/geometry/model/context.py:sidewalk_span_ft) - so having them on the plot is what makes
    # an over-wide leg visible instead of merely arguable.
    #
    # AS A BAND, SIDEWALK_WIDTH_FT wide, because a footway is a strip of ground and a 1.0 pt line
    # was a strip whose width was the window's. TWO CAVEATS, both live:
    #   * OSM maps a footway as a CENTRELINE and almost never tags a width, so the 6 ft is
    #     assumed. Hence the dashed edge - the same thing this sheet already says about a
    #     driveway widened from a centreline (PAVED_EDGE_ASSUMED).
    #   * It is NOT the band the 3D render builds. That one is build_sidewalk_pieces, widened
    #     from the traced KERB rather than from the footway layer, so where OSM's footway does not
    #     run parallel to the kerb the two are in different places. Both are real; neither is a
    #     copy of the other, and conflating them would draw one and label it the other.
    _draw(ax, [line.buffer(SIDEWALK_WIDTH_FT / 2, cap_style=2, join_style=2)
               for line in sidewalk_lines_ft(sidewalks)],
          color="steelblue", alpha=0.16, zorder=2,
          boundary=dict(color="steelblue", linewidth=0.8, linestyle=(0, (4, 2)), alpha=0.65,
                        zorder=2))

    # Curb lines as the corners trim them. The raw lines overshoot into the junction on
    # purpose (fillet material), so drawing them raw would draw curb across the middle of
    # the intersection - marking a curb that isn't there and isn't in the 3D render.
    # Grouped by provenance tier, since that is what decides the style.
    curbs_by_leg = trimmed_curb_lines(state.legs, state.corner_fillets)
    curbs_by_tier: dict[str, list] = {}
    for name, leg in state.legs.items():
        tier = built_width_provenance(leg, model.config["legs"][name])
        curbs_by_tier.setdefault(tier, []).extend(curbs_by_leg[name].values())
        if dimension_labels:
            along_ft = min(leg.centerline.length * 0.85, leg.centerline.length - 5)
            mid = leg.centerline.interpolate(along_ft)
            labels.dimension(f"{leg.curb_to_curb_ft:.1f} ft", (mid.x, mid.y), fontsize=7,
                             toward=_leg_heading(leg, along_ft), color=PLOT_STYLE[tier]["color"],
                             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    # STILL A LINE ON THE SHEET, deliberately, where the TRACED kerb beside it is now a 0.49 ft
    # body. This is not a kerb: it is the config's claim about where one is, half a nominal width
    # off the alignment, and PLOT_STYLE draws it in three weights and dashes to say which tier of
    # evidence that claim rests on. Given a real body all three would read alike and the claim
    # would be gone - the same reason the crossing reference line and the centreline datum keep
    # theirs. Where this and the traced kerb disagree is the finding.
    for tier, curbs in curbs_by_tier.items():
        _draw(ax, curbs, linewidth=2, zorder=3, **PLOT_STYLE[tier])

    arcs = []
    for _corner, pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        arcs.append(pieces["arc"])
        # `is not None` as well as `in`: a corner that is not a corner - two legs of one street
        # running through the junction - has no radius, and reaching for one crashed the build.
        if dimension_labels and pieces.get("radius_ft") is not None:
            mid = pieces["arc"].interpolate(0.5, normalized=True)
            # Pushed outward through the corner, away from the junction centre: behind the kerb
            # return is the one direction from an arc that is nobody's carriageway.
            labels.dimension(f"R={pieces['radius_ft']:.0f} ft", (mid.x, mid.y), fontsize=7,
                             toward=(mid.x - model.center_ft.x, mid.y - model.center_ft.y),
                             color="darkorange", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # AT THE KERB WIDTH, because a corner fillet IS a kerb: src/render/export.py writes each arc
    # into the kerb list as RAISED, and blender_scene.py builds it at KERB_WIDTH_M like any other.
    # Kept darkorange rather than black, which is what says this one is solved geometry and not a
    # traced way - the radius label beside it is the other half of that.
    _draw(ax, [arc.buffer(KERB_WIDTH_FT / 2, cap_style=2, join_style=2) for arc in arcs],
          color="darkorange", zorder=4)

    # Asked of the treatments, which build their polygon against this design - see
    # src/render/export.py's note on why these two are not materialised onto the state.
    islands = [(island, island.polygon(state)) for island in state.treatments_of(RefugeIsland)]
    _draw(ax, [polygon for _island, polygon in islands],
          color="seagreen", alpha=0.6, zorder=5,
          boundary=dict(color="darkgreen", linewidth=1, zorder=5))
    if dimension_labels:
        for island, polygon in islands:
            c = polygon.centroid
            labels.dimension(f"refuge\n{island.width_ft:.0f} ft", (c.x, c.y), fontsize=6.5,
                             color="darkgreen", fontweight="bold")

    raised_bands = [t.polygon(state) for t in state.treatments_of(RaiseCrossing)]
    # THE ONE HATCH LEFT, and it is not paint. A raised crossing is BUILT GROUND - the band is
    # already drawn at its real extent, and what the hatch says is that the ground is higher,
    # which is the same thing KERB_DASHES_FT says about a kerb and the same thing a plan cannot
    # draw. There are no strokes on it to lay at a real width; inventing some would paint a ramp.
    _draw(ax, raised_bands, color="slateblue", alpha=0.35, hatch="//", zorder=2,
          boundary=dict(color="slateblue", linewidth=1, zorder=2))
    if dimension_labels:
        for poly in raised_bands:
            c = poly.centroid
            labels.dimension("raised\ncrossing", (c.x, c.y), fontsize=6.5, color="indigo",
                             fontweight="bold")

    _draw_surveyed_crossings(ax, crossings)
    # Before the modelled crosswalks, so where the two ever overlap the modelled one wins
    # the pixel - and the four this junction models are drawn by _draw_crosswalks below.
    _draw_unmodelled_crossings(ax, scene)
    _draw_crosswalks(ax, scene, labels, dimension_labels)
    props = _draw_props(ax, model, state, scene.crosswalk_offsets, traffic_control,
                         street_furniture, crossings, labels, dimension_labels, pavement,
                         kerb_ways)

    # Every painted marking comes from src/geometry/paint/ - the same builder the 3D export
    # draws from and src/checks.py inspects. Never assembled here in parallel; the two copies
    # drifted on where a parking buffer's taper starts and on whether taper fill is cut around a
    # crossing.
    #
    # props comes back extended with the bollards the paint places (see
    # SceneGeometry.build_paint_and_posts). Nothing more is drawn for them here - the loop
    # below already draws them from the paint - but the invariant pass has to see the same
    # props the export will, or the check that they reach the 3D render can only fail there.
    paint, props = scene.build_paint_and_posts(props)

    # All pieces of one kind in one collection - see _draw.
    by_kind: dict[markings.PaintKind, list] = {}
    bollards = []
    for piece in paint:
        # A flex post is an object, not paint, and is drawn as a post below. Asked of the
        # marking rather than matched against its name - see markings.Role.
        if piece.kind.is_object:
            bollards.append(piece.geometry.centroid)
        else:
            by_kind.setdefault(piece.kind, []).append(piece)
    keep_off_rims = _hatch_keep_off(paint)
    for kind, pieces in by_kind.items():
        style = PAINT_STYLE[kind]
        # A zone that covers ground gets its outline drawn too; a line has no boundary. Asked
        # of the marking, not of the geometry: a bollard is stored as a degenerate polygon, so
        # the geometry test answered "fill" for something that is neither.
        edge = (dict(color=PAINT_FILL_EDGE[style["color"]], linewidth=1, zorder=3)
                if kind.covers_area else None)
        geometries, body_style = _at_real_width(kind, [piece.geometry for piece in pieces], style)
        _draw(ax, geometries, boundary=edge, **body_style)
        if kind.is_fill:
            # The paint inside the wash, at the spacing and width the render lays it - see
            # _hatch_strokes_ft. In the zone's own rim colour, because on the ground the rim and
            # the strokes are one striper's pass.
            _draw(ax, _hatch_strokes_ft(pieces, state, model.center_ft, keep_off_rims),
                  color=PAINT_FILL_EDGE[style["color"]], zorder=3)
    if bollards:
        # The same flex post the props list stands in a daylight zone, at the same real diameter.
        # These come from the treatment layer's own paint (see _draw_props); drawing the two at
        # different sizes would say they are different objects.
        _draw_discs(ax, [((point.x, point.y), 2 * BOLLARD_RADIUS_FT) for point in bollards],
                    color=BOLLARD_PLAN_COLOR, zorder=6)

    if dimension_labels:
        _label_paint(labels, state, paint, scene.kerb_openings)
        _label_parking_legality(labels, state)

    ax.scatter([model.center_ft.x], [model.center_ft.y], color="blue", zorder=6, s=40)

    _draw_centerlines(ax, scene)

    violations = _mark_violations(ax, scene, props, paint, labels)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    # The frame the 3D render is pointed at as well, measured from the model rather than from
    # this DesignState so a before/after pair shares one frame - see src/render/frame.py.
    xmin, xmax, ymin, ymax = (frame if frame is not None else junction_frame(model)).bounds_ft()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Only now: a label's size in feet is a fact about the limits on the line above, and what it
    # must not cover is the design drawn between here and the top of this function. Everything
    # the 3D render builds goes into keep_off - a marking hidden under a call-out is a marking
    # this view failed to report, which is the one thing it exists to do. Thin paint is left out
    # deliberately: a stripe reads through a translucent box, and keeping clear of every edge
    # line leaves a dense junction nowhere to put a dimension.
    keep_off = ([piece.geometry for piece in paint if piece.kind.covers_area]
                + [polygon for _island, polygon in islands] + raised_bands
                + [Point(prop["position_ft"]).buffer(PROP_MARKER_RADIUS_PT * ft_per_point(ax))
                   for prop in props])
    label_violations = labels.flush(ax, unary_union(keep_off))
    for violation in label_violations:
        print(f"  LABEL PLACEMENT: {violation}")
    violations = violations + label_violations
    ax.set_xlabel("Feet (EPSG:3424)")
    return PlotResult(violations=violations, metrics=scene.metrics(paint))


def _draw_centerlines(ax, scene: SceneGeometry):
    """The leg centerline (the measurement datum) and the painted centerline on top of it.

    The datum matters because every width in this drawing - the 11 ft lane, the 8 ft stall, the
    depth of a hatched zone - is an OFFSET FROM IT, so without it there is nothing to check those
    offsets against by eye. The painted centerline matters because the render draws one
    (blender_scene.py, from the same DesignState.centerline_style this reads).

    The paint starts where src/render/crosswalks.py:centerline_start_ft says, which is at the
    stop bar - the same rule the export uses, not a second copy of it.
    """
    state = scene.state
    bodies: dict[str, list] = {}
    for leg_name, leg in state.legs.items():
        # The datum: thin, grey, dotted, the full length of the leg. Deliberately
        # unobtrusive - it is a construction line, not a marking on the road.
        ax.plot(*leg.centerline.xy, color="#3b6ea5", lw=0.9, ls=(0, (7, 3, 1, 3)), alpha=0.9,
                zorder=4)

        style = state.centerline_style(leg_name)
        if style == "none" or leg_name not in scene.crosswalk_offsets:
            continue
        start_ft = centerline_start_ft(scene.crosswalk_offsets[leg_name].offset_ft,
                                        scene.stop_bar_offsets.get(leg_name),
                                        leg_name in scene.marked_crosswalks)
        # The stripes themselves come from src/render/crosswalks.py, so this view and the render
        # draw the same paint. Drawn solid whatever the style, because a dashed style arrives as
        # separate dash segments rather than as one line with a pattern on it.
        # A two-way bike lane on one side pushes the travel lanes off the alignment, so the
        # divider between them moves with them. None on every leg of every other scenario.
        shift = state.travel_lane_divider_shift(leg_name)
        shift_ft, shift_side = shift if shift is not None else (0.0, None)
        # WHITE WHERE THE LANES RUN THE SAME WAY, gold where they oppose - off the style, not
        # hardcoded here, because the 3D render makes the same choice from the same table and
        # the two used to hardcode it separately (SKILLS.md section 3: the channel decides the
        # colour, and this line is the one marking that has no channel).
        colour = "white" if style in CENTERLINE_IS_WHITE else "gold"
        # AT ITS REAL WIDTH, like every other marking - see _at_real_width. This one has no
        # channel to carry the figure, so it comes from the constant the two stripes of a
        # double yellow are separated by: without a body, a 0.82 ft separation drawn as two
        # 1.2 pt strokes is one stripe on any sheet wider than ~500 ft.
        bodies.setdefault(colour, []).extend(
            line.buffer(CENTERLINE_STRIPE_WIDTH_FT / 2, cap_style=2, join_style=2)
            for line in centerline_paint_ft(leg, start_ft, style, shift_ft, shift_side))
    for colour, painted in bodies.items():
        _draw(ax, painted, color=colour, zorder=4)


# What OSM says about kerbside parking, and what that produced. Colour is the OSM statement
# alone, so a kerb the surveyor tagged and a kerb nobody has tagged never look the same.
PARKING_LEGALITY_COLOR = {"restricted": "#b3261e", "allowed": "#1b7f3b", "untagged": "#6b6b6b"}



def _label_parking_legality(labels: LabelPlacer, state: DesignState):
    """Per side of every leg: the OSM parking tag, and what the design did with it.

    Without this the drawing cannot answer the question it most often provokes - "why is that
    kerb hatched?" - because three different situations produce identical hatching: OSM says
    no parking, OSM says parking is fine but the road has less than one stall's width spare,
    and nobody has tagged it at all. The first is a restriction being marked; the other two
    are this design's own arithmetic. Only the tag distinguishes them, so the tag is drawn.

    Read per STRETCH of kerb, not per leg. A restriction covering only the approach is how OSM
    records "no parking for the first 100 ft", and a label reduced to one value per kerb reports
    only one end of it - East Broad reads "parking OK" over a kerb whose first 80 ft are tagged
    no_parking. See src/geometry/treatments/parking.py:RestrictionSummary.

    KEYED, not written on the road. This is prose - a sentence about a kerb - and a sentence is
    the one thing that cannot be sized to the street: at --frame-scale 2.5 these two lines
    covered 235 ft of ground, five Broad Sts, and buried a fifth of the bike lane surface the 3D
    render shows whole. The number goes where the kerb is; the sentence goes in the block.
    """
    from src.geometry.model import curb_point_at_station
    from src.geometry.targets import BOTH_SIDES, LegSide, LegTarget
    from src.geometry.treatments import (AddBikeLane, LaneNarrowing, MarkedParking,
                                         MIN_MARKED_PARKING_DEPTH_FT, TARGET_LANE_WIDTH_FT,
                                         restriction_summary, kerbside_allowance_ft)

    for leg_name, leg in state.legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
        for side in BOTH_SIDES:
            # The same measurement whichever mechanism decided this kerb actually used, so the
            # label cannot say "7.5 ft spare" about a kerb the code sized a treatment for at 5.0.
            # hold_travel_lane_at_target subtracts its own divider shift from the NOMINAL
            # half-width - a different datum from kerbside_allowance_ft's traced kerb - and
            # records that figure on the state precisely so this label can read it back rather
            # than recompute the wrong one, which is what put "6.3 ft spare" beside a kerb that
            # mechanism had already refused for having none. See target_lane_room's docstring.
            room_ft = state.target_lane_room(leg_name, side)
            allowance_ft = room_ft if room_ft is not None else kerbside_allowance_ft(leg, side)
            kerb = LegSide(leg_name, side)
            bike_lane = state.treatment_for(AddBikeLane, kerb)
            at = restriction_summary(state, leg_name, side, leg.centerline.length)
            if at.restricted_throughout:
                kind, says = "restricted", at.worst_value
            elif at.restricted_in_part:
                # The case a single value cannot express. Named on the drawing with the stretch
                # it covers, because "which 80 ft" is the whole content of the fact.
                kind = "restricted"
                says = at.describe().replace("OSM says ", "").replace("'", "")
            elif at.stated_ft > 0:
                kind, says = "allowed", "none (parking OK)"
            else:
                kind, says = "untagged", "untagged"

            if bike_lane is not None:
                drew = f"bike lane, {bike_lane.width_ft:.0f} ft"
            elif state.treatment_for(MarkedParking, kerb) is not None:
                # Naming the carve-out matters on a partly-restricted kerb: "stalls" alone, next
                # to a label saying no_parking over the first 80 ft, reads as a contradiction
                # rather than as the two facts it is.
                drew = ("stalls beyond it" if at.restricted_in_part else "stalls")
            elif narrowing is not None and side in narrowing.sides:
                # Three reasons a kerb ends up hatched, and the label may only claim the one
                # that applies. Attributing every unrestricted hatched kerb to insufficient width
                # produces "only 15.0 ft spare, under a 8 ft stall" on Broad St - 15 is not under
                # 8; the real reason there is a borough ordinance the geometry cannot see, so the
                # label stops asserting and reports what it can.
                if kind == "restricted":
                    drew = "hatched"
                elif allowance_ft < MIN_MARKED_PARKING_DEPTH_FT:
                    drew = (f"hatched: only {allowance_ft:.1f} ft spare, under a "
                            f"{MIN_MARKED_PARKING_DEPTH_FT:.0f} ft stall")
                else:
                    drew = (f"hatched by this proposal, though {allowance_ft:.1f} ft is spare "
                            f"- see the scenario for why")
            elif allowance_ft < 0:
                # target_lane_room can go negative - a corridor's own divider shift outgrew the
                # nominal half-width it was subtracted from - and "-0.6 ft spare" reads as a
                # typo where "0.6 ft short" reads as the deficit it is.
                drew = (f"nothing: {-allowance_ft:.1f} ft short of an "
                        f"{TARGET_LANE_WIDTH_FT:.0f} ft lane")
            else:
                drew = (f"nothing: {allowance_ft:.1f} ft spare beside an "
                        f"{TARGET_LANE_WIDTH_FT:.0f} ft lane")

            along_ft = leg.centerline.length * 0.42
            point = curb_point_at_station(leg, side, along_ft)
            if point is None:
                continue
            outward = 1 if side == "left" else -1
            here = inset_point_at_station(leg, along_ft,
                                           outward * (abs(_offset_of(leg, point)) + 9.0))
            labels.note(f"{leg_name} {side} kerb", f"OSM parking: {says} -> {drew}",
                        (here[0], here[1]), toward=_side_normal(leg, side, along_ft),
                        colour=PARKING_LEGALITY_COLOR[kind])


def _offset_of(leg, point):
    import numpy as np

    from src.geometry.model import station_offset_many

    _stations, offsets = station_offset_many(leg.centerline, np.asarray([point], dtype=float))
    return float(offsets[0])


def _label_paint(labels: LabelPlacer, state: DesignState, paint, openings=None):
    """Dimension labels for the curbside paint: what each treatment actually measures.

    One lane label PER SIDE narrowed, offset into that lane - not a single label on the
    centerline, which reads as "this road is one 11 ft lane" rather than what is there
    (a leg is not always narrowed on both sides).
    """
    from src.geometry.treatments import (LaneNarrowing, divider_shift_toward_ft,
                                          travel_lane_width_ft)

    for narrowing in state.treatments_of(LaneNarrowing):
        leg_name = narrowing.target.leg
        leg = state.legs[leg_name]
        along_ft = min(leg.centerline.length * 0.6, leg.centerline.length - 5)
        for side in narrowing.sides:
            sign = side.sign
            # THE LANE IS MEASURED FROM THE DIVIDER, not from the alignment. The two coincide
            # until a two-way bike lane shifts the travel lanes off it, and then the label reads
            # 9.6 ft beside a lane the geometry built at 11.00.
            lane_ft = travel_lane_width_ft(state, leg_name, str(side), narrowing.stripe_width_ft)
            # And the label belongs IN that lane, so its position takes the shift too.
            shift_ft = divider_shift_toward_ft(state, leg_name, str(side))
            at = leg.centerline.offset_curve(sign * (shift_ft + lane_ft / 2)).interpolate(along_ft)
            # Nudged ALONG the lane, not across it: a travel lane is the same width for its whole
            # length, so sliding the label up the leg keeps it measuring the thing it names.
            labels.dimension(f"lane {lane_ft:.1f} ft", (at.x, at.y), fontsize=6.5,
                             toward=_leg_heading(leg, along_ft), color="goldenrod",
                             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # One label per RUN of stalls, not per side: a hydrant mid-block splits a kerb into two
    # separate runs, and a single label would be counting stalls that are not in one place. A
    # DRIVEWAY splits one too, and this label could not see that while it read the edge line -
    # which is CARRIED straight across one. src/metrics.py:marked_stall_runs is the whole rule,
    # shared with the summary panel so the count beside a run and the count in the panel cannot
    # be two arithmetics OR two lengths.
    for piece, run, parking, n_stalls in marked_stall_runs(paint, state, openings):
        mid = run.interpolate(0.5, normalized=True)
        # Pushed out over the kerb, off the stalls it counts, with a leader back to them: their
        # depth is the label's own subject, and a reader cannot check it against paint the label
        # is standing on.
        leg = state.legs[piece.leg]
        labels.dimension(f"parking\n{n_stalls} stalls ({parking.depth_ft:.0f} ft)", (mid.x, mid.y),
                         toward=_side_normal(leg, piece.side, leg.centerline.project(mid)),
                         fontsize=6, color="steelblue", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def _mark_violations(ax, scene: SceneGeometry, props, paint, labels: LabelPlacer):
    """Run the scene invariants and draw whatever failed, right where it failed.

    The plan view reports rather than raises, and the phase script asserts after saving -
    so a failure always arrives with a picture of itself. Reading "tactile pad 40% in the
    roadway at (419160, 566742)" next to a red ring around that exact pad is the difference
    between one round trip and several.

    Checked against `scene`, so this validates the geometry the figure above actually drew and
    not a third set neither view uses.
    """
    violations = scene.check(props, paint)

    located = [v for v in violations if v.where]
    if located:
        xs, ys = zip(*(v.where for v in located))
        ax.scatter(xs, ys, s=260, facecolors="none", edgecolors="red", linewidths=2.0, zorder=10)
        labels.caption(f"{len(violations)} INVARIANT FAILURE(S) - see console", (0.5, 0.975),
                       va="top", fontsize=9, fontweight="bold", color="red",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.95))
    for violation in violations:
        label = "INVARIANT FAILED" if violation.fatal else "SOURCE CONFLICT"
        print(f"  {label}: {violation}")
    return violations


def legend_handles():
    return [
        Line2D([0], [0], color="black", lw=2, label="Curb line - FIELD-MEASURED width"),
        Line2D([0], [0], color="darkviolet", lw=2, ls="-.", label="Curb line - OSM-derived width"),
        Line2D([0], [0], color="crimson", lw=2, ls="--", label="Curb line - estimated width"),
        # ONE WEIGHT FOR BOTH, because both kerbs are one width on the ground (KERB_WIDTH_FT).
        # What the dashes say is a HEIGHT, which is what a plan cannot draw - see KERB_DASHES_FT.
        Line2D([0], [0], color="black", lw=1.8, label="Traced kerb - RAISED (OSM kerb=raised)"),
        Line2D([0], [0], color="black", lw=1.8, ls="--",
               label="Traced kerb - LOWERED (same width, dropped): a vehicle crosses, so the "
                     "paint opens"),
        Patch(facecolor="#8a7a68", alpha=0.55, edgecolor="#5d5044",
               label="Parking lot / street traced BOTH sides - outline as surveyed"),
        Patch(facecolor="#8a7a68", alpha=0.55, edgecolor="#5d5044", linestyle="--",
               label="Driveway, aisle, street part-traced - width DRAWN is assumed"),
        Patch(facecolor="steelblue", alpha=0.16, edgecolor="steelblue", linestyle="--",
               label="OSM sidewalk - mapped as a centreline, drawn 6 ft wide (ASSUMED)"),
        Line2D([0], [0], color="#3b6ea5", lw=0.9, ls=(0,(7,3,1,3)), label="Leg centerline (widths measured from this)"),
        Line2D([0], [0], color="gold", lw=1.2, label="Centerline paint (double yellow / dashed)"),
        Line2D([0], [0], color="white", lw=1.6,
                label="Lane line - broken WHITE, lanes running the same way"),
        Line2D([0], [0], color="gold", lw=1.5,
                label="Left edge line - solid YELLOW, left edge of a one-way roadway"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["restricted"],
               label="OSM: parking restricted"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["allowed"],
               label="OSM: parking allowed"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["untagged"],
               label="OSM: parking untagged"),
        Line2D([0], [0], color="darkviolet", lw=1, ls=":", label="OSM crossing way (as surveyed)"),
        # A curb extension's tightened face is drawn as this same arc - it IS a corner fillet,
        # solved at the radius the extension presents (src/geometry/treatments/).
        Line2D([0], [0], color="darkorange", lw=2.5,
                label="Corner fillet / curb extension face (radius labeled)"),
        Line2D([0], [0], color="seagreen", lw=6, alpha=0.6, label="Pedestrian refuge island"),
        Line2D([0], [0], color="slateblue", lw=6, alpha=0.35, label="Raised crossing"),
        Patch(facecolor="gold", alpha=0.5, edgecolor="goldenrod",
               label="Lane narrowing / corner hatching - wash, with the real strokes on it"),
        Line2D([0], [0], color="goldenrod", lw=1.5, label="Lane narrowing - line only (no chevron fill)"),
        Patch(facecolor="orangered", alpha=0.40, edgecolor="orangered",
               label="Daylighting - no parking (R.S. 39:4-138)"),
        Patch(facecolor="peru", alpha=0.6, edgecolor="saddlebrown", label="Mountable apron"),
        # One row for both kinds: the dotted extension is the same paint, and the dashes are in
        # the geometry rather than in the line style, so a second swatch would look identical.
        Line2D([0], [0], color="seagreen", lw=1.6,
               label="Bike lane - edge lines (dotted across a driveway)"),
        Patch(facecolor="mediumseagreen", alpha=0.45, edgecolor="seagreen",
               label="Bike lane - green surface"),
        Patch(facecolor="slategrey", alpha=0.18, edgecolor="slategrey",
               label="Bike lane - EXISTING, unpainted asphalt"),
        Patch(facecolor="mediumseagreen", alpha=0.35, edgecolor="seagreen",
               label="Bike lane buffer - the hatched one; the lane itself is unstriped green"),
        Line2D([0], [0], color="goldenrod", lw=1.3, ls="--",
               label="Two-way bike lane - contraflow divider (MUTCD yellow)"),
        # The two ends of the facility (MUTCD 9E.11, 9E.09 - STANDARDS.md section 2). The box is
        # where a rider crosses ONTO a bikeway running up the far kerb; the sharrow is what the
        # travelled way carries once the bikeway has stopped. They are the two halves of one
        # answer, so they sit together.
        Patch(facecolor="seagreen", alpha=0.55, edgecolor="white",
               label="Two-stage bike turn box - cross here to join the bikeway (MUTCD 9E.11)"),
        Patch(facecolor="white", edgecolor="white",
               label="BIKE LANE symbol, through arrow, and shared-lane marking (sharrow)"),
        Line2D([0], [0], marker="o", color=BOLLARD_PLAN_COLOR, lw=0, label="Bollard"),
        Line2D([0], [0], color="steelblue", lw=1.5, label="Marked parking lane + stalls"),
        Patch(facecolor="white", edgecolor="darkviolet", label="Crosswalk - OSM-surveyed position"),
        Patch(facecolor="white", edgecolor="crimson", ls="--", label="Crosswalk - estimated position"),
        Line2D([0], [0], color="grey", lw=0.7, ls=":", label="Unmarked leg (no crosswalk today)"),
        # "entering half" until Lavallette, where a one-way carriageway has no half to stop at -
        # the bar spans the whole roadway there. See stop_bar_ends_ft, which is what decides.
        Line2D([0], [0], color="dimgrey", lw=3, label="Stop bar (entering lanes only)"),
        Line2D([0], [0], marker="o", color="limegreen", markeredgecolor="black", lw=0,
                label="Traffic signal pole + mast arm"),
        Line2D([0], [0], marker="s", color="limegreen", markeredgecolor="black", lw=0,
                label="Pedestrian signal head"),
        Line2D([0], [0], marker="H", color="red", lw=0, label="Stop sign (unsignalized)"),
        Line2D([0], [0], marker="s", color="white", markeredgecolor="red", lw=0, label="No turn on red"),
        Line2D([0], [0], marker="*", color="dimgrey", lw=0, label="Streetlight"),
        Line2D([0], [0], marker="P", color="gold", markeredgecolor="black", lw=0,
                label="Pedestrian pushbutton (OSM)"),
        Patch(facecolor=TACTILE_PAD_COLOR, edgecolor="black", label="Tactile paving / curb ramp (OSM)"),
        Line2D([0], [0], marker="D", color="gold", markeredgecolor="black", lw=0, label="RRFB beacon (OSM)"),
        Line2D([0], [0], marker="d", color="gold", markeredgecolor="black", lw=0, markersize=9,
                label="Bicycle warning sign (MUTCD W9-5 / W16-21P)"),
        Line2D([0], [0], marker="s", color="white", markeredgecolor="black", lw=0,
                label="Turn box regulatory sign (MUTCD R9-23)"),
        Line2D([0], [0], marker="p", color="greenyellow", markeredgecolor="black", lw=0,
                label="School zone sign"),
        Line2D([0], [0], marker="v", color="white", markeredgecolor="red", lw=0,
                label="Yield sign (unsignalized)"),
        Line2D([0], [0], marker="P", color="firebrick", lw=0, label="Fire hydrant (OSM)"),
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
