"""The one piece of ground both views are pointed at.

The plan view and the 3D render are the same reconstruction drawn twice, so a reader comparing
them is entitled to assume both pictures cover the same street. That only holds if ONE number
decides the framing, so it is computed here and both views take it: `plan_view.plot_design_state`
sets its axis limits from it, and `export_scenario` writes it into the JSON as `frame` for
`blender_scene.py` to frame its camera on rather than recomputing an extent of its own.

WHAT THE FRAME IS. The extent of the pavement this project modelled, plus a small margin. Two
consequences worth stating:

  * Vertices past a leg's own far end are dropped. The pavement ring is stitched from traced OSM
    `barrier=kerb` ways, which do not stop where our leg does - at E Broad & Princeton one kerb
    runs 425 ft from the junction off a 130 ft leg because the mapper drew the whole block as one
    way. A leg's far end is the edge of what was modelled; kerb beyond it is street, not junction.
  * It is measured from the MODEL, not from a DesignState. A curb extension moves the kerb, so
    framing on resolved pavement would frame a proposal differently from its own baseline - and
    these are published as before/after pairs, where a frame that shifts between the panels is
    exactly what makes two pictures incomparable.

The 3D camera is a tilted perspective camera, so it necessarily sees ground beyond the frame:
what matches between the views is the subject and the radius, not the outline of the visible
region. The 2D axes are square; the render is 4:3.
"""
import math
import os
from dataclasses import dataclass

from shapely.geometry import Point

from src.geometry.model import build_pavement_polygon
from src.render.coords import FT_TO_M
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# How much wider than the modelled pavement the frame is drawn. Tight on purpose: this is a
# drawing of one junction, and the paint and signage detail is the subject.
FRAME_MARGIN = 1.2

# A deliberate zoom-out, for a picture whose subject is longer than one junction (a corridor
# treatment down a street). It scales the RADIUS only, so the frame stays centred on the same
# ground and both views widen together - a knob that moved only one of them would undo the reason
# this module exists. An environment variable because it must reach two call sites several layers
# down (plot_design_state and export_scenario); same problem and answer as ROAD_SKETCHES_RENDER_SCALE,
# and scripts/phase4_render_3d.py sets both from flags.
FRAME_SCALE_ENV = "ROAD_SKETCHES_FRAME_SCALE"


def frame_scale() -> float:
    """The zoom-out multiplier, or 1.0. Anything unparseable falls back with a warning rather than
    failing a batch of renders over an environment variable."""
    raw = os.environ.get(FRAME_SCALE_ENV, "1")
    try:
        scale = float(raw)
    except ValueError:
        print(f"  WARNING: {FRAME_SCALE_ENV}={raw!r} is not a number - framing at 1x.")
        return 1.0
    if scale <= 0:
        print(f"  WARNING: {FRAME_SCALE_ENV}={raw!r} is not positive - framing at 1x.")
        return 1.0
    return scale


# HOW FAR DOWN THE STREET THIS DRAWING ACTUALLY GOES, in feet, already frame-scaled. The other
# half of the answer FRAME_SCALE_ENV gives, and the half that was missing.
#
# The frame scale lengthens legs and widens the context fetch by the same factor, so a site whose
# legs are all `leg_working_length_ft` keeps the two in step at every zoom. A PER-LEG
# `working_length_ft` does not go through the frame scale: it lengthens ONE leg and tells nothing
# else. wbroad_lanning's southwest leg is 2,307.5 ft to the borough line, so the street was drawn
# 703 m while roads, kerbs, driveways and crossings were all fetched at 130 m - 573 m of corridor
# through an empty field, with Louellen St and W Prospect St simply absent from the picture.
#
# Set by load_intersection_model before it fetches anything, for the same reason the frame scale
# is an environment variable: it has to reach call sites several layers down that hold a centre
# point and nothing else (drawn_kerb_radius_ft holds not even that, and MUST return the radius the
# kerbs were really fetched at or the two renderers draw different sets). Set UNCONDITIONALLY on
# every load, so a site with no override resets it instead of inheriting the last site's reach.
_drawn_reach_ft = 0.0


def set_drawn_reach_ft(reach_ft: float) -> None:
    """Record the longest leg the model being built will draw. See `_drawn_reach_ft`."""
    global _drawn_reach_ft
    _drawn_reach_ft = max(0.0, float(reach_ft))


def context_radius_m(base_m: float) -> float:
    """How far out to pull CONTEXT - buildings, roads, parking - for the frame in force.

    The frame scale widens what the camera takes in; it must widen what there IS to take in by
    the same factor, or a zoom-out just adds bare ground around a street with sharply cut ends.

    Scaled off the base radius rather than off the frame's own radius on purpose: the frame is
    measured FROM the model and the context is fetched to BUILD the model, so reading one from
    the other is circular. A flat multiple of the constant each layer already uses is not, and at
    1x returns exactly the radius that layer used before, so an unscaled render is unchanged.

    AND NEVER LESS THAN THE STREET IS DRAWN. The drawn reach is not circular either - it is the
    configured leg length times the frame scale, known before a single feature is fetched. It
    binds only where a leg is longer than the base radius, which is no site whose legs are the
    130 ft default (39.6 m, well inside every base here) and is exactly the case the base radius
    cannot see. See `_drawn_reach_ft`.
    """
    return max(base_m * frame_scale(), _drawn_reach_ft * FT_TO_M)


def frame_covering_radius_m(model: "IntersectionModel", base_m: float) -> float:
    """Enough to cover the FRAME, for a layer whose extent is radial rather than along a street.

    THE DIFFERENCE FROM context_radius_m, which is easy to get wrong and expensive when you do.
    That one multiplies a base radius by the frame scale, right for a layer that follows the
    street: kerbs and roads run along a leg, so a wider frame needs more of their length.

    Buildings and crossings instead fill the picture, and the picture is a circle of known radius,
    so what they need is the frame's own radius rather than base times zoom. The two diverge fast
    - at Broad & Greenwood at 2.5x the frame reaches 131.4 m while context_radius_m(130) asks for
    325 m - and over-fetching is not free, because every building is meshed and decimated.

    Floored at `base_m`, so at 1x no existing render moves. The 10% margin covers the difference
    between a circular fetch and the square-ish ground the camera actually sees.

    AND MEASURED FROM THE JUNCTION NODE, WHICH IS NOT THE FRAME'S CENTRE. Every fetch is a circle
    around `model.center_wgs84`; the frame is centred on the modelled pavement. Those coincide
    only while the legs are about the same length - 0.3 to 3.8 m apart at the seven junctions
    here - and not at all on a corridor: wbroad_lanning's frame centre sits 306 m down W Broad,
    so a radius equal to the frame's own reached 444 m and left the far half of its own picture
    with no buildings and no crossings in it. The radius a circle at the node needs to cover a
    circle at the frame centre is the distance between them PLUS the frame's radius.
    """
    frame = junction_frame(model)
    reach_ft = frame.center_ft.distance(model.center_ft) + frame.radius_ft
    return max(base_m, reach_ft * FT_TO_M * 1.1)

# How far past a leg's far end a pavement vertex may still count as part of this junction. The
# corner fillets trim the curbs a little past the leg's own end, so a hard cut at the leg length
# would drop legitimate ring vertices.
LEG_REACH_TOLERANCE = 1.05


@dataclass(frozen=True)
class Frame:
    """Where both views point, and how much they take in. Feet, in the site's state plane."""
    center_ft: Point
    radius_ft: float

    def bounds_ft(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) - a square, which is what a plan view's axes want."""
        return (self.center_ft.x - self.radius_ft, self.center_ft.x + self.radius_ft,
                self.center_ft.y - self.radius_ft, self.center_ft.y + self.radius_ft)

    def as_local_m(self, center_ft: Point) -> dict:
        """The frame in the exported JSON's local-metre coordinates, for the 3D render."""
        return {"center_m": [(self.center_ft.x - center_ft.x) * FT_TO_M,
                            (self.center_ft.y - center_ft.y) * FT_TO_M],
                "radius_m": self.radius_ft * FT_TO_M}


def leg_reach_ft(model: "IntersectionModel") -> float:
    """How far from the junction the SURVEYED street extends, along its longest leg.

    The site's configured working length, not the built centerline, because the frame scale also
    carries the legs out (src/geometry/intersection/) and the two would compound: longer legs make
    a longer pavement ring, this measures the ring, and the scale multiplies a second time, so a
    2.5x frame comes out 6.2x.

    Measured by TRUNCATING each centerline to its surveyed length and taking the far end, not by
    reading the configured number off directly. The two differ - a centerline does not start
    exactly at the junction node and is not always straight, so w_broad_louellen's 130 ft leg
    reaches 136.4 ft - and truncating reproduces the pre-scale value exactly at 1x, where taking
    the number would shift that site's frame.

    Falls back to the built centerline for a model with no configured lengths recorded, which is
    every synthetic model in the tests.
    """
    from shapely.ops import substring

    surveyed = getattr(model, "surveyed_leg_lengths", None) or {}
    reaches = []
    for name, leg in model.legs.items():
        line = leg.centerline
        length = surveyed.get(name)
        if length is not None and line.length > length:
            line = substring(line, 0.0, length)
        reaches.append(model.center_ft.distance(Point(line.coords[-1])))
    return max(reaches, default=0.0)


def junction_frame(model: "IntersectionModel") -> Frame:
    """The one frame both views draw, from the baseline pavement of `model`.

    Falls back to the leg reach itself where there is no pavement ring to measure - an
    unclosable ring is reported by check_pavement_ring rather than being fatal here, and a
    render with no frame at all would be worse than one framed on the legs.
    """
    reach_ft = leg_reach_ft(model)
    try:
        pavement = build_pavement_polygon(model.corner_fillets)
    except Exception:
        pavement = None
    xs, ys = [], []
    if pavement is not None and not pavement.is_empty:
        for x, y in pavement.exterior.coords:
            if not reach_ft or math.hypot(x - model.center_ft.x,
                                          y - model.center_ft.y) <= reach_ft * LEG_REACH_TOLERANCE:
                xs.append(x)
                ys.append(y)
    scale = frame_scale()
    if not xs:
        return Frame(model.center_ft, max(reach_ft, 1.0) * FRAME_MARGIN * scale)
    center = Point((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    radius = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * FRAME_MARGIN
    return Frame(center, radius * scale)
