"""
Phase 4: headless Blender scene builder + renderer for one or more geometry
exports produced by scripts/phase4_export_geometry.py (or phase4_render_3d.py).

Not run with the project's normal Python - invoke via Blender's own
interpreter, which has no network access / requests / this project's venv.
Every real asset (textures, the streetlight model) is fetched beforehand in
the venv (src/render/theme.py) and passed in as local file paths via the
JSON - this script only ever reads files, never fetches them. Accepts any
number of <geometry.json> <output.png> pairs, all rendered in one Blender
process (each launch has ~1-1.5s of fixed startup overhead - paying it once
for N renders instead of N times is the single biggest lever for reducing
total render time):

  blender --background --python scripts/blender/blender_scene.py -- \\
      output/geometry_existing.json output/phase4_render_existing.png \\
      output/geometry_proposed.json output/phase4_render_proposed.png

THREE MODES, one entry point (see parse_args):

  <geometry.json> <out.png> [...]     the original: build ONE site's scene, point the camera at the
                                      JSON's `frame`, render, repeat. Unchanged.
  --build <world.json> --save <w.blend>
                                      build the scene from one geometry JSON with NO camera, and save
                                      it. The JSON need not carry a `frame` - a world has none.
  --open <w.blend> --camera <cameras.json> --out-dir <dir>
                                      open a saved world and render one PNG per camera spec,
                                      `{"name", "center_m": [x, y], "radius_m"}` in the world's own
                                      local-metre frame. A site render is a named camera in a world.

This file is the entry point + top-level scene assembly only - the actual
geometry-building code is split across sibling modules in this same
directory (plain local imports work fine under Blender's bundled Python, no
venv needed):
  blender_materials.py   flat-color and PBR-textured material builders
  blender_geometry.py    generic mesh helpers (extrude a ring, stripe rects)
  blender_crosswalks.py  the 3 painted crosswalk styles + dashed centerlines
  blender_props.py       street furniture: streetlights, signage, traffic
                          signals, trees - one builder function per prop type
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import bpy
import mathutils


def _standards():
    """src/standards.py, loaded by path: it imports nothing but the standard library (an import
    contract keeps it so), so Blender's own Python runs it, and every figure here is standards.toml's."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "src" / "standards.py"
    spec = importlib.util.spec_from_file_location("standards", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["standards"] = module          # dataclasses resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


si = _standards().si

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling blender_*.py imports below
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/blender/blender_scene.py -> repo root

from blender_crosswalks import (
    add_crosswalk, add_dashed_centerline, add_double_yellow_centerline, add_stop_bar,
)
from blender_geometry import (MeshBatch, build_merged_meshes, extrude_polygon,
                              line_ring, polyline_rings)
from blender_materials import make_material, make_textured_material
from blender_props import (
    PED_SIGNAL_HOUSING_DARK, SIGNAL_HOUSING_DARK, SIGN_POST_GRAY,
    add_prop, add_tree_instances, build_tree_proxy, import_gltf_template,
)

random.seed(7)  # stable building color assignment across existing/proposed renders

PAVEMENT_HEIGHT_M = si("render.pavement_height")
# A sidewalk is at kerb height: the road's top plus a raised kerb's reveal (kerb.raised). The
# reader cuts the street and the tactile pads out of the sidewalks, so standing
# above them hides neither; across a driveway or apron the sidewalk is built on, over the paving.
RAISED_KERB_M = si("kerb.raised")
SIDEWALK_HEIGHT_M = PAVEMENT_HEIGHT_M + RAISED_KERB_M
# crosswalks/centerlines/stop bars (add_crosswalk*/add_dashed_centerline/add_double_yellow_centerline/
# add_stop_bar) sit at blender_crosswalks.py:EXISTING_MARKING_Z_BASE (0.06) with thickness
# EXISTING_MARKING_THICKNESS_M (0.01) - this is their real top, i.e. EXISTING_MARKING_Z_BASE +
# EXISTING_MARKING_THICKNESS_M. Kept as its own constant here (rather than importing the two above)
# since this file only needs the single derived "top" value to stack the next layer above it.
EXISTING_MARKING_HEIGHT_M = si("render.existing_marking_height")
# The new paint-only overlay markings (lane narrowing, corner hatching, mountable apron) sit on top
# of EXISTING_MARKING_HEIGHT_M + this gap, NOT exactly at either that or PAVEMENT_HEIGHT_M - two
# surfaces at the exact same height are coincident/coplanar, which renders as flickering z-fighting
# (confirmed by an isolated test: a marking placed with zero gap above the pavement rendered as a
# visibly tessellated mess even as a flat, zero-height plane, ruling out "thin geometry aliasing" as
# the cause; a lane-narrowing stripe overlapping a crosswalk's footprint needed the SAME fix again
# relative to the crosswalk's own top height, not just the pavement's). ~1cm of clearance is
# imperceptible at this render's scale but enough to give the depth buffer an unambiguous answer.
#
# Separately, EXISTING_MARKING_Z_BASE=0 (the crosswalk/centerline/stop-bar layer's OLD z_base) had
# its own bug even though its 0.06 top height was never coincident with anything: z_base=0 meant its
# bottom fully overlapped the pavement's own 0-0.05 volume rather than sitting on top of it. That,
# combined with this camera's near/far clip range being far wider than the scene needed (see
# setup_camera_and_light) and so starving the depth buffer of precision at this camera's distance,
# produced a torn/tessellated look on thin, elongated shapes like a crosswalk line - confirmed by an
# isolated test. Fixed by both lifting z_base to sit flush on the pavement's top (see
# blender_crosswalks.py:EXISTING_MARKING_Z_BASE) and tightening the camera's clip range.
MARKING_CLEARANCE_M = si("render.marking_clearance")
# How thick a painted marking is built. Was add_paint_line's own `height_m=0.01` default, which is
# where every batched marking's thickness came from before the draw block stopped going through it -
# named here so the value is stated rather than inherited from a keyword default two modules away.
# Paint has no meaningful thickness; this exists only to give the depth buffer something to order.
PAINT_HEIGHT_M = si("render.paint_height")

BUILDING_PALETTE = [
    (0.62, 0.42, 0.35),  # brick red
    (0.82, 0.78, 0.68),  # cream siding
    (0.55, 0.55, 0.58),  # gray
    (0.70, 0.62, 0.48),  # tan
    (0.45, 0.38, 0.32),  # dark brown
]


class Job:
    """What this process was asked to do. `mode` is "sites", "build" or "cameras"."""

    def __init__(self, mode: str, **fields):
        self.mode = mode
        self.__dict__.update(fields)


# THE TEXTURE TIER A WORLD IS BUILT AT, as the suffix of the theme keys that name it (`asphalt_far`
# is the 2k set, `asphalt_near` the 4k one - src/render/theme.py). ONE tier for every surface in a
# world, because the near/far split was by distance from a junction centre and a world has none; see
# texture_keys below for the measured justification.
TEXTURE_TIERS = {"2k": "far", "4k": "near"}
DEFAULT_TEXTURE_RES = "2k"


def parse_args() -> Job:
    argv = sys.argv
    usage = ("Usage: blender --background --python blender_scene.py -- "
             "<geometry.json> <output.png> [...]\n"
             "   or: ... -- --build <world.json> --save <world.blend> [--texture-res 2k|4k]\n"
             "   or: ... -- --open <world.blend> --camera <cameras.json> --out-dir <dir>")
    if "--" not in argv:
        raise SystemExit(usage)
    args = argv[argv.index("--") + 1:]
    if args and not args[0].startswith("--"):
        if len(args) < 2 or len(args) % 2 != 0:
            raise SystemExit("Need pairs of <geometry.json> <output.png>")
        return Job("sites", pairs=[(Path(args[i]), Path(args[i + 1])) for i in range(0, len(args), 2)])
    parser = argparse.ArgumentParser(prog="blender_scene.py", usage=usage)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--open", type=Path, dest="open_")
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--texture-res", choices=sorted(TEXTURE_TIERS), default=DEFAULT_TEXTURE_RES)
    ns = parser.parse_args(args)
    if ns.build and not ns.open_:
        if not ns.save:
            raise SystemExit("--build needs --save <world.blend>\n" + usage)
        return Job("build", geometry=ns.build, blend=ns.save, texture_res=ns.texture_res)
    if ns.open_ and not ns.build:
        if not (ns.camera and ns.out_dir):
            raise SystemExit("--open needs --camera <cameras.json> and --out-dir <dir>\n" + usage)
        return Job("cameras", blend=ns.open_, cameras=ns.camera, out_dir=ns.out_dir)
    raise SystemExit("Give exactly one of --build or --open\n" + usage)


# The keys whose ABSENCE from a geometry file would produce a picture that is wrong rather than
# incomplete, so a file without them is refused instead of rendered. Every one of them is read
# below through `.get(..., [])`, which is right for a scenario that legitimately has none of
# something - and indistinguishable from a file too old to carry the key at all. That is not a
# cosmetic difference: 39 of the 65 exports committed at the time predated these four, so one drew
# a street with no kerbs, no driveways or parking aprons and no surveyed crossings, and with
# `frame` gone this script computed a camera extent of its own from the pavement, which
# src/render/export.py says in as many words it must not do. Nothing warned; the render looked
# fine. "A picture that shows a marked crosswalk as bare asphalt is not a conservative
# simplification - it is a false statement about the street, made to an audience deciding whether
# to build something" (docs/network-renderer-plan.md).
#
# NOT the whole 37-key schema, which lives in src/render/export.py and cannot be imported here
# (see .importlinter: this file runs in Blender's interpreter). These four are what this RENDERER
# needs in order not to lie. Nothing checks the committed files against that full schema: the
# exporter writes every key unconditionally, so only a file committed BEFORE a key existed can
# lack one, and this guard is what stops such a file being rendered rather than reported.
REQUIRED_KEYS = ("frame", "kerbs", "paved_surfaces", "surveyed_crossings")
# A WORLD HAS NO FRAME. A frame is "the ground one site's pictures are pointed at", and a world is
# every site's ground at once - its cameras name their own centre and radius. Everything else in
# REQUIRED_KEYS is about the street and applies to a world unchanged.
WORLD_REQUIRED_KEYS = tuple(k for k in REQUIRED_KEYS if k != "frame")


def load_geometry(path: Path, required: tuple = REQUIRED_KEYS) -> dict:
    with open(path) as f:
        data = json.load(f)
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(
            f"{path}: geometry export is stale - no {', '.join(missing)}. Rendering it would "
            f"silently draw a street without them. Re-export with scripts/build_all.py.")
    return data


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

# How far past a leg's own far end a pavement vertex may still sit and be treated as part of
# this junction. A traced kerb's last vertex lands a foot or two beyond the leg it bounds, and
# the corner fillets' trimmed curbs run to their tangent points rather than to the centerline's
# end - so this absorbs that, and nothing like the 3x overshoot a kerb drawn down the whole
# block produces. See build_scene's framing note.
LEG_REACH_TOLERANCE = 1.05

# How wide a kerb is built. A real kerb's top face is about 6 in; this only has to read as an
# edge at the camera distance, and the height is what carries the raised/lowered distinction.
KERB_WIDTH_M = si("kerb.top_width")

# One stripe of a centerline, matching add_double_yellow_centerline's own width_m - MUTCD's ~6 in.
# The two lines of a double yellow arrive already offset from each other, so this is the width of
# each, not of the pair.
CENTERLINE_WIDTH_M = si("line.width")

# A surveyed TRANSVERSE crossing's two lines. Named rather than left inline at the call below, so
# test_blender_stroke_widths_match_the_channels can read it: an unnamed literal is a width nothing
# on the src/ side can be held against, and the plan view drew these at a cosmetic 1.6 pt for as
# long as it went unnamed. Mirrors markings.EDGE_LINE_WIDTH_M.
SURVEYED_CROSSING_LINE_WIDTH_M = si("crosswalk.line_width")

# WHICH PAINT CHANNELS ARE SAMPLED POLYLINES, and which are honestly two-point segments. Declared
# as data rather than left implicit in the loops below, because the distinction is load-bearing and a
# test guards it (tests/test_paint.py).
#
# A SAMPLED POLYLINE MUST BE WALKED SEGMENT BY SEGMENT. These follow the traced kerb on a 2 ft
# station grid, so drawing the chord between the first and last vertex is not the line: it deviated
# 0.7 ft on Broad St's daylight zone, which pulled the painted edge inside the 11 ft lane it marks
# and lifted it off the hatching it bounds. The value beside each is its stripe width in metres - a
# drawn-scale choice, not a standard: a solid edge line reads at 0.25 m here, a taper at 0.15 m.
SAMPLED_POLYLINE_CHANNELS = (
    ("lane_narrowing_edge_lines", si("line.width")),
    ("yellow_hatch_edge_lines", si("line.width")),
    ("blue_hatch_edge_lines", si("line.width")),
    ("lane_narrowing_taper_lines", si("line.width")),
    ("parking_edge_lines", si("line.width")),
    ("left_edge_lines", si("line.width")),
    ("parking_buffer_edge_lines", si("line.width")),
    ("parking_buffer_taper_lines", si("line.width")),
    ("bike_lane_edge_lines", si("line.width")),
    # MUTCD 9E.11(07): a solid white line on all four sides of the two-stage turn box.
    ("turn_box_edge_lines", si("line.width")),
)
# Two-point strokes: a hatch stroke runs edge to edge of its zone and a stall tick lies across the
# kerbside strip. Only their two ends exist, so the chord IS the line.
TWO_POINT_CHANNELS = (
    "lane_narrowing_hatch_lines", "lane_narrowing_hatch_wide_lines", "yellow_hatch_stroke_lines",
    "blue_hatch_stroke_lines",
    "corner_hatching_lines",
    "parking_stall_divider_lines",
    "parking_buffer_hatch_lines", "bike_lane_hatch_lines",
)
TWO_POINT_WIDTH_M = si("line.width")
# Diagonal crosshatch strokes at MUTCD's widths (hatch.stroke_width*, as cited): 8 in below 45 mph,
# 12 in at or above - src/osm_world.py sorts each street's strokes by its OSM `maxspeed`.
TWO_POINT_WIDTHS_M = {**dict.fromkeys(("lane_narrowing_hatch_lines", "yellow_hatch_stroke_lines",
                                        "blue_hatch_stroke_lines"), si("hatch.stroke_width")),
                      "lane_narrowing_hatch_wide_lines": si("hatch.stroke_width_wide")}
# THE CHANNELS DRAWN IN THE YELLOW MATERIAL. Every other paint channel is white, so this is the
# whole of what makes a stripe yellow at this end - which is why a yellow marking gets its own
# channel upstream rather than sharing an edge-line one (src/geometry/markings.py). Two entries,
# and they are yellow for the same reason in two different places: `left_edge_lines` is the left
# edge of a ONE-WAY roadway (MUTCD 3B.09 P3) and `bike_lane_contraflow_lines` divides riders
# going opposite ways. Yellow means "do not cross to the other side of this".
YELLOW_CHANNELS = ("left_edge_lines", "bike_lane_contraflow_lines", "yellow_hatch_edge_lines",
                   "yellow_hatch_stroke_lines")
# A `colour=blue` restriction area's outline and strokes (src/osm_world.py:HATCH_COLOURS).
BLUE_CHANNELS = ("blue_hatch_edge_lines", "blue_hatch_stroke_lines")
# The centerline styles drawn in the WHITE marking material rather than the yellow one - a
# broken lane line between two lanes running the same way. Blender runs under its own bundled
# Python and cannot import src, so this mirrors src/geometry/treatments/base.py:
# CENTERLINE_IS_WHITE the same way SAMPLED_POLYLINE_CHANNELS mirrors the channel widths, and is
# pinned to it by a test for the same reason.
CENTERLINE_STYLES_WHITE = ("single_white_dashed",)



def _marking_frame(leg: dict, near, u, n, prefix: str, fallback_offset_m: float):
    """(centre, u, n) for a marking, from the geometry JSON where it says.

    `u` is the leg's near->far CHORD, and stepping an offset along it is only the same point
    crosswalk_axes picks while the centerline is straight. Two of these legs are not:
    broad_st_east kinks 4.5 deg 43.1 ft out where NJDOT rounds the corner, and
    louellen_st_west 29.4 deg 15.4 ft out. On broad_st_east the chord rotated the crosswalk
    bars 4.54 deg off the plan view's and drove them through 12.6 ft of paint the plan view
    had cleared - a 2D/3D disagreement of exactly the kind the shared geometry is supposed to
    make impossible. src/render/export.py resolves the frame now; this is the fallback for a
    geometry file written before it did.

    Module level, taking the leg and its frame as arguments, rather than a closure inside
    build_scene's per-leg loop: a closure reads the loop variables as they are AT CALL TIME,
    which is only the intended leg while every call stays in the iteration that defined it.
    """
    centre_m = leg.get(f"{prefix}_centre_m")
    axis = leg.get(f"{prefix}_axis")
    if centre_m is None or axis is None:
        return near + u * fallback_offset_m, u, n
    axis_u = mathutils.Vector((axis[0], axis[1], 0.0))
    return (mathutils.Vector((*centre_m, 0.0)), axis_u,
            mathutils.Vector((-axis_u.y, axis_u.x, 0.0)))


def resolve_theme_paths(theme: dict) -> dict:
    """`theme` with every asset path resolved against THIS checkout's root.

    src/render/theme.py writes them repo-relative (`output/.textures/...`) so a geometry file
    does not carry the absolute paths of the machine that exported it - the 65 committed exports
    each held 19 of those, naming a directory that exists on one laptop. Joining them here is the
    other half of that; an ALREADY-ABSOLUTE path is passed through, which is what a geometry file
    written before that change contains.

    Missing files are still not an error - make_textured_material and import_gltf_template each
    fall back - so this only has to name the right place to look.
    """
    resolved = {}
    for key, value in theme.items():
        if isinstance(value, dict):
            resolved[key] = {k: _under_repo(v) for k, v in value.items()}
        else:
            resolved[key] = _under_repo(value)
    return resolved


def _under_repo(path):
    if not isinstance(path, str) or os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)


_phase_clock = [time.perf_counter()]


def phase(name: str) -> None:
    """Print how long the build phase that just FINISHED took. A world is built once and takes
    minutes at borough scale, so where the time goes has to be on the log rather than guessed."""
    now = time.perf_counter()
    print(f"  [build] {name}: {now - _phase_clock[0]:.2f}s")
    _phase_clock[0] = now


def surface_rings(data: dict, key: str):
    """(near, far) ring lists for `pavement` / `sidewalks`.

    The exporter used to split each by distance from the junction centre (`pavement_near`,
    `pavement_far`); a world has no junction centre, so it writes ONE list under the bare key. Read
    that when present, else the pair - and a file with the single list returns it as `near` with
    nothing in `far`, which is what lets the caller treat both shapes alike.
    """
    if key in data:
        return list(data[key]), []
    return list(data.get(f"{key}_near", [])), list(data.get(f"{key}_far", []))


def world_extent(data: dict):
    """(minx, miny, maxx, maxy) over every layer that has a position, in the file's metres."""
    xs, ys = [], []

    def take(points):
        for p in points:
            xs.append(p[0])
            ys.append(p[1])

    for key in ("pavement", "pavement_near", "pavement_far", "pavement_concrete",
                "pavement_gravel", "pavement_dirt",
                "sidewalks", "sidewalks_near", "sidewalks_far", "sidewalks_asphalt"):
        for ring in data.get(key, []):
            take(ring)
    for b in data.get("buildings", []):
        take(b["vertices_m"] if b["mesh"] else b["coords"])
    for kerb in data.get("kerbs", []):
        take(kerb.get("coords") or [])
    for drive in data.get("paved_surfaces", data.get("driveways", [])):
        take(drive.get("coords") or [])
    for parcel in data.get("corner_parcels", []):
        take(parcel["coords"])
    if not xs:
        return (-50.0, -50.0, 50.0, 50.0)
    return (min(xs), min(ys), max(xs), max(ys))


def max_building_height(data: dict) -> float:
    """The tallest building's top, in metres - what a camera's near clip has to leave room for."""
    tops = [max(v[2] for v in b["vertices_m"]) if b["mesh"] else b.get("height_m", 0.0)
            for b in data.get("buildings", []) if (b["vertices_m"] if b["mesh"] else b["coords"])]
    return max(tops, default=0.0)


def _textures(theme: dict, stem: str, tier: str):
    """The texture paths for `stem` at `tier` ("near" 4k / "far" 2k), else whichever tier exists.

    A geometry file carries both tiers (src/render/theme.py), so the fallback only matters for one
    written with a single tier - and then the render should use that rather than flat colour.
    """
    return theme.get(f"{stem}_{tier}") or theme.get(f"{stem}_far") or theme.get(f"{stem}_near")


class Built:
    """What build_scene made, for the camera to be pointed at.

    `cx, cy, scene_radius, view` are only meaningful when the geometry carried a `frame`, or for a
    legacy site file - a world has no single frame, and its cameras bring their own.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)


def build_scene(data: dict, world: bool = False, texture_res: str = DEFAULT_TEXTURE_RES) -> Built:
    """Build every object in `data`. `world=True` is the build-once mode: no frame is read or
    required, the ground is sized off the geometry's own extent, and every textured surface uses ONE
    texture tier (`texture_res`).

    TEXTURES. The old scene textured pavement/sidewalks at 4k inside a "near zone" around the
    junction and 2k outside it - a distance from ONE junction centre, which a world has none of.
    Rather than invent a distance-from-camera rule (a per-camera material swap at render time), the
    world uses one tier everywhere and the tier is 2k. Why that is safe is arithmetic and then a
    measurement: a texture tiles every 2 m (extrude_polygon's uv_tile_m), so 2k is 1 mm per texel
    and 4k is 0.5 mm, while one pixel of a 1920 px render covers ~9 cm of ground at a 105 m frame
    radius (2 cm at the 4x render scale) - the extra texels are minified away by the mipmaps
    either way. The memory saving is measured in the scripts/phase4_render_3d.py handback, not
    asserted here. `--texture-res 4k` is kept for anyone who wants to see for themselves.
    A file that already carries single `pavement` / `sidewalks` lists (the world exporter) gets the
    same one-tier treatment whichever mode reads it, since it has no near/far to split on.
    """
    theme = resolve_theme_paths(data.get("theme") or {})
    tier = TEXTURE_TIERS[texture_res]
    uniform_pavement = world or "pavement" in data
    uniform_sidewalks = world or "sidewalks" in data

    def tex(name, stem, tier_, fallback, roughness):
        return make_textured_material(name, _textures(theme, stem, tier_), fallback, roughness)

    asphalt_rgb, concrete_rgb = (0.07, 0.07, 0.08), (0.72, 0.71, 0.67)
    if uniform_pavement:
        asphalt_near = asphalt_far = tex("Asphalt", "asphalt", tier, asphalt_rgb, 0.95)
    else:
        asphalt_near = tex("AsphaltNear", "asphalt", "near", asphalt_rgb, 0.95)
        asphalt_far = tex("AsphaltFar", "asphalt", "far", asphalt_rgb, 0.95)
    if uniform_sidewalks:
        concrete_near = concrete_far = tex("Concrete", "concrete", tier, concrete_rgb, 0.85)
    else:
        concrete_near = tex("ConcreteNear", "concrete", "near", concrete_rgb, 0.85)
        concrete_far = tex("ConcreteFar", "concrete", "far", concrete_rgb, 0.85)
    apron_mat = tex("Apron", "apron", tier if world else "near", (0.65, 0.6, 0.55), 0.8)
    # Unpaved ground, by the surface the reader names (src/osm_world.py:SURFACE_MATERIAL).
    gravel_mat = tex("Gravel", "gravel", tier if world else "far", (0.46, 0.43, 0.38), 0.95)
    dirt_mat = tex("Dirt", "dirt", tier if world else "far", (0.38, 0.30, 0.22), 1.0)
    lot = make_material("Lot", (0.55, 0.6, 0.48), roughness=0.9)
    grass = make_material("Grass", (0.3, 0.48, 0.24), roughness=1.0)
    refuge_mat = make_material("Refuge", (0.22, 0.5, 0.26), roughness=0.8)
    crossing_mat = make_material("RaisedCrossing", (0.68, 0.58, 0.48), roughness=0.8)
    marking_mat = make_material("Marking", (0.9, 0.9, 0.88), roughness=0.4)
    kerb_mat = make_material("Kerb", (0.62, 0.61, 0.58), roughness=0.8)
    # A green bike lane's surface colour. Matches plan_view.py's "mediumseagreen" closely
    # enough that the two views read as the same treatment - the plan view draws it
    # semi-transparent over grey paper, this over black asphalt, so they cannot be identical
    # numbers. Rough like the asphalt it is painted on rather than glossy like fresh stripes.
    bike_surface_mat = make_material("BikeLaneSurface", (0.13, 0.45, 0.28), roughness=0.85)
    centerline_mat = make_material("Centerline", (0.85, 0.7, 0.15), roughness=0.4)
    blue_paint_mat = make_material("BluePaint", (0.1, 0.25, 0.7), roughness=0.4)
    # A painted curb extension's fill (`colour=tan`), rough like the asphalt it is on. NYC DOT lays
    # pedestrian space in epoxy gravel (Standard Specifications 6.44 CST) and calls an interim curb
    # extension's colour "truffle paint" (Permanent Planter Guidelines, 2019) without publishing a
    # measured value, so this RGB is matched by eye - not a standard.
    tan_paint_mat = make_material("TanPaint", (0.72, 0.6, 0.42), roughness=0.85)
    building_mats = [make_material(f"Building{i}", c, roughness=0.75) for i, c in enumerate(BUILDING_PALETTE)]
    pole_mat = make_material("Pole", SIGN_POST_GRAY, roughness=0.5)
    trunk_mat = make_material("TreeTrunk", (0.32, 0.22, 0.15), roughness=0.9)
    foliage_mat = make_material("TreeFoliage", (0.16, 0.4, 0.14), roughness=0.85)
    signal_housing_mat = make_material("SignalHousing", SIGNAL_HOUSING_DARK, roughness=0.4)
    ped_signal_housing_mat = make_material("PedSignalHousing", PED_SIGNAL_HOUSING_DARK, roughness=0.4)

    pavement_near_rings, pavement_far_rings = surface_rings(data, "pavement")
    sidewalk_near_rings, sidewalk_far_rings = surface_rings(data, "sidewalks")
    all_pavement = pavement_near_rings + pavement_far_rings
    pavement_x = [x for ring in all_pavement for x, y in ring]
    pavement_y = [y for ring in all_pavement for x, y in ring]
    # WHERE THE CAMERA POINTS IS RESOLVED IN src/render/frame.py AND CARRIED IN THE JSON, so this
    # render and the plan view frame the same ground. Computing it here as well is what let the
    # two views drift: the plan view framed a hardcoded 110 ft square on the junction node while
    # this framed the pavement's own extent, and on the four sites the two disagreed by 1.15-1.57x
    # and by 6.5-12.5 ft of centre. The block below is the fallback for a geometry file written
    # before `frame` existed, and it is also the definition src/render/frame.py implements - the
    # same pair of numbers, computed once on the side that can be tested.
    #
    # Frame the camera on the intersection itself (the actual subject), not the
    # full building-context radius - buildings are background dressing and are
    # fine to crop at the frame edges.
    #
    # MEASURED AGAINST THE MODELLED LEGS, not against every pavement vertex. The pavement ring
    # is stitched from the corner fillets' trimmed curbs, and those curbs are TRACED OSM
    # barrier=kerb ways, which do not stop where our leg does: at E Broad & Princeton,
    # e_broad_st_west's left kerb runs 425 ft from the junction off a 130 ft leg, because the
    # mapper drew one continuous kerb down the block. That single vertex made the pavement
    # bounding box 168 x 78 m, put its centre 41 m down that leg and framed the camera at a
    # 100.6 m radius against ~53 m at the other three sites - a render zoomed nearly two-fold
    # out and not even pointed at the junction.
    #
    # A leg's own far end IS the edge of what this project modelled, so anything past it is
    # kerb running on down the street rather than part of this junction. All four sites have a
    # few such vertices (1, 4, 6 and 4 of them); dropping them tightens every render and
    # centres all four, rather than special-casing the one site where it had become glaring.
    framed_x = framed_y = None
    cx = cy = scene_radius = 0.0
    if not world:
        leg_reach = max((math.hypot(*leg["far_m"]) for leg in data.get("legs", [])), default=0.0)
        framed = [(x, y) for x, y in zip(pavement_x, pavement_y)
                  if not leg_reach or math.hypot(x, y) <= leg_reach * LEG_REACH_TOLERANCE]
        framed_x = [x for x, _y in framed] or pavement_x
        framed_y = [y for _x, y in framed] or pavement_y
        cx, cy = (min(framed_x) + max(framed_x)) / 2, (min(framed_y) + max(framed_y)) / 2
        pavement_radius = max(max(framed_x) - min(framed_x), max(framed_y) - min(framed_y)) / 2
        scene_radius = pavement_radius * 1.2  # tight enough to actually read paint markings/signage detail
        frame = data.get("frame")
        if frame:
            cx, cy = frame["center_m"]
            scene_radius = frame["radius_m"]


    # The GROUND still covers everything, framed or not: a plane that stopped at the framed
    # extent would leave the far end of an over-long kerb standing over blank space.
    def building_xy(b):
        return [(v[0], v[1]) for v in (b["vertices_m"] if b["mesh"] else b["coords"])]

    if world:
        # A WORLD'S GROUND IS SIZED OFF ITS OWN EXTENT: every layer that has a position, because a
        # world is whatever the exporter put in it and nothing here knows which layer is widest. The
        # same 1.25x-of-the-larger-span rule as a site's (context_radius * 2.5 across a diameter), so a
        # kerb running off the pavement's edge still has grass under it. A CAMERA near the edge needs
        # more than that - ground_for_camera grows the plane to the camera's own 4x frame, in memory,
        # at render time.
        extent = world_extent(data)
        wminx, wminy, wmaxx, wmaxy = extent
        ground_w = max((wmaxx - wminx) * 1.25, 100.0)
        ground_h = max((wmaxy - wminy) * 1.25, 100.0)
        gcx, gcy = (wminx + wmaxx) / 2, (wminy + wmaxy) / 2
        ground_size = max(ground_w, ground_h)
        cx, cy = gcx, gcy
    else:
        all_x = pavement_x + [x for b in data.get("buildings", []) for x, _y in building_xy(b)]
        all_y = pavement_y + [y for b in data.get("buildings", []) for _x, y in building_xy(b)]
        context_radius = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2
        # AND AT LEAST FOUR TIMES THE FRAME, because the camera can be asked to pull back further than
        # the context reaches (src/render/frame.py's ROAD_SKETCHES_FRAME_SCALE, for a picture whose subject
        # is longer than one junction). On a wide frame the ground ran out inside the shot and the
        # horizon showed the plane's own edge with sky under it - the buildings and pavement had all
        # been drawn correctly on a groundsheet too small for the view.
        ground_size = max(context_radius * 2.5, scene_radius * 4, 100)
        ground_w = ground_h = ground_size
        gcx, gcy = cx, cy
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(gcx, gcy, -0.03))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (ground_w, ground_h, 1.0)
    ground.data.materials.append(grass)

    phase("materials + ground")
    for parcel in data.get("corner_parcels", []):
        extrude_polygon(f"parcel_{parcel['name']}", parcel["coords"], 0.0, lot)

    phase("parcels")
    # ONE OBJECT PER COLOUR, not per building: a borough has thousands, and each used to cost an
    # edit-mode round trip that grows with the scene (blender_geometry.build_merged_meshes).
    meshed = [[] for _ in building_mats]
    extruded = [MeshBatch(f"buildings_extruded_{k}", m) for k, m in enumerate(building_mats)]
    for i, b in enumerate(data.get("buildings", [])):
        k = i % len(building_mats)
        if b["mesh"]:
            meshed[k].append((b["vertices_m"], b["faces"]))
        else:
            extruded[k].add_prism(b["coords"], b["height_m"])
    for k, parts in enumerate(meshed):
        build_merged_meshes(f"buildings_{k}", parts, building_mats[k])
    for batch in extruded:
        batch.build(uv_tile_m=2.0)

    phase("buildings")
    # ONE MESH PER MATERIAL for the slabs, not one object per ring: a world holds thousands of
    # pieces and a scene's cost scales with its object count (blender_geometry.MeshBatch).
    for name, rings, height, mat in (
            ("pavement_near", pavement_near_rings, PAVEMENT_HEIGHT_M, asphalt_near),
            ("pavement_far", pavement_far_rings, PAVEMENT_HEIGHT_M, asphalt_far),
            # streets by their `surface` (src/osm_world.py:SURFACE_MATERIAL), at road height
            ("pavement_concrete", data.get("pavement_concrete", []), PAVEMENT_HEIGHT_M, concrete_far),
            ("pavement_gravel", data.get("pavement_gravel", []), PAVEMENT_HEIGHT_M, gravel_mat),
            ("pavement_dirt", data.get("pavement_dirt", []), PAVEMENT_HEIGHT_M, dirt_mat),
            ("sidewalk_near", sidewalk_near_rings, SIDEWALK_HEIGHT_M, concrete_near),
            ("sidewalk_far", sidewalk_far_rings, SIDEWALK_HEIGHT_M, concrete_far),
            # `surface=asphalt` sidewalks (src/osm_world.py:ASPHALT_SURFACES), at kerb height
            ("sidewalk_asphalt", data.get("sidewalks_asphalt", []), SIDEWALK_HEIGHT_M, asphalt_far)):
        batch = MeshBatch(name, mat)
        for ring in rings:
            batch.add_prism(ring, height)
        batch.build(uv_tile_m=2.0)

    phase("pavement + sidewalks")
    # The paved ground beside the carriageway - driveways, parking aisles and parking lots - as
    # the POLYGON src/ built, the same one the plan view fills, so the two views cannot disagree
    # about where any of it is. All one asphalt, which is what they are; `kind` names each object
    # so a scene is readable in the outliner. `driveways` is the pre-parking key, kept as the
    # fallback for a geometry file written before this one.
    # Extruded to the pavement's own height so it reads as connected paving where it meets the
    # road. A driveway running off past the modelled legs is drawn where it really is; that it
    # ends in grass is our road model stopping, not the driveway being wrong.
    # One mesh, not one object each: a world carries every driveway in the borough.
    # Asphalt unless the reader names another surface (src/osm_world.py:_paving).
    paved = {"asphalt": MeshBatch("paved_surfaces", asphalt_far),
             "concrete": MeshBatch("paved_surfaces_concrete", concrete_far),
             "gravel": MeshBatch("paved_surfaces_gravel", gravel_mat),
             "dirt": MeshBatch("paved_surfaces_dirt", dirt_mat)}
    for drive in data.get("paved_surfaces", data.get("driveways", [])):
        coords = drive.get("coords") or []
        if len(coords) >= 3:
            paved[drive.get("surface", "asphalt")].add_prism(coords, PAVEMENT_HEIGHT_M)
    for batch in paved.values():
        batch.build(uv_tile_m=2.0)

    # The traced kerbs, at the height their OSM kerb= tag calls for (src/render/export.py:
    # KERB_HEIGHT_M). There was no kerb in this scene before - the road slab simply met the
    # concrete band - so a 6 in stood-up kerb and a driveway's dropped kerb looked the same, and
    # the kerbside markings that now BREAK over a dropped kerb had nothing visible to break for.
    #
    # polyline_rings, not the chord between the endpoints: a kerb is a band of constant width
    # following a sampled line, and the chord would cut every corner the tracing turns. One mesh per
    # distinct HEIGHT (there are three: raised, lowered, flush) rather than one object per segment -
    # a borough's kerbs are tens of thousands of segments.
    phase("paved surfaces")
    kerb_batches = {}
    for kerb in data.get("kerbs", []):
        coords = kerb.get("coords") or []
        if len(coords) < 2:
            continue
        # `kerb:height` is the kerb's face ABOVE THE ROAD (wiki Key:kerb:height), so it stands on
        # the road slab's top: built from the ground, a lowered kerb's 3 cm ended 2 cm inside the
        # 5 cm slab and drew nothing. A flush kerb keeps MARKING_CLEARANCE_M, not to be coplanar.
        height = kerb.get("height_m", RAISED_KERB_M)
        batch = kerb_batches.setdefault(height, MeshBatch(f"kerbs_{height:g}m", kerb_mat))
        for ring in polyline_rings(coords, KERB_WIDTH_M):
            batch.add_prism(ring, PAVEMENT_HEIGHT_M + max(height, MARKING_CLEARANCE_M), z_base=0.0)
    for batch in kerb_batches.values():
        batch.build()

    # Paint-only / no-curb-change proposal treatments (src/geometry/treatments/:
    # add_lane_narrowing / add_corner_hatching / add_mountable_apron) - sit
    # above BOTH the pavement and the existing crosswalk/centerline markings
    # they can overlap (a stripe runs the whole leg, crossing the crosswalk),
    # with a small MARKING_CLEARANCE_M gap either way (see docstring above).
    phase("kerbs")
    marking_z = EXISTING_MARKING_HEIGHT_M + MARKING_CLEARANCE_M
    # A lane-narrowing buffer is a solid edge line (the new lane's real edge)
    # plus diagonal hatching filling the buffer beyond it - a real gore/chevron
    # marking, not a solid filled block of paint (which at this render's scale
    # was visually indistinguishable from a sidewalk/apron - see export.py).
    # The edge line is dead straight along the main run (a simple 2-point
    # line) but curves where it tapers into the corner (a many-point sampled
    # arc, see export.py/lane_narrowing_taper_ft) - add_paint_line only ever
    # draws a single straight chord between whatever two points it's given,
    # so a curved line needs add_paint_polyline instead (drawing every
    # consecutive segment) or it silently collapses into a straight diagonal.
    # add_paint_polyline, not add_paint_line(line[0], line[-1]): these are sampled polylines
    # that follow the traced kerb, and drawing the chord between their endpoints throws away
    # every vertex in between. It deviated 0.7 ft on Broad St's daylight zone - enough to put
    # the painted lane edge inside the 11 ft it is supposed to mark, and to pull the line off
    # the hatching it is supposed to bound. The hatch strokes and stall ticks below really are
    # two-point segments, so add_paint_line is right for those.
    # EVERY WHITE MARKING IN ONE MESH, and the yellow ones in another. What this replaced was a
    # loop per channel calling add_paint_polyline, which called add_paint_line per SEGMENT, which
    # built an object each: a 130 ft lane edge sampled every 2 ft became 64 objects, its channel
    # became 1,209, and a wide render's scene held 3,753 objects for 842 JSON items. Batched, it is
    # two. See blender_geometry.MeshBatch, and src/render/plan_view.py:_draw for the same argument
    # winning the same 156x in matplotlib.
    #
    # Grouped by MATERIAL rather than by channel, because that is the only thing a merge has to
    # respect - a mesh carries one material, and every white marking shares one. The channel
    # distinctions above it (which line is a lane edge, which a stall tick) are decided upstream in
    # src/geometry/markings.py and are already spent by the time the geometry arrives here.
    #
    # POLYLINES, NOT CHORDS. polyline_rings walks every segment, so a sampled arc still curves; the
    # old add_paint_line(line[0], line[-1]) shortcut deviated 0.7 ft on Broad St's daylight zone,
    # enough to pull the painted lane edge inside the 11 ft it marks.
    white = MeshBatch("paint_white", marking_mat)
    yellow = MeshBatch("paint_yellow", centerline_mat)
    blue = MeshBatch("paint_blue", blue_paint_mat)

    def paint(key: str) -> MeshBatch:
        return yellow if key in YELLOW_CHANNELS else blue if key in BLUE_CHANNELS else white
    # (channel, stripe width) - the two widths are a drawn-scale choice, not a standard: a solid
    # edge line reads at 0.25 m here and a hatch stroke at 0.15 m.
    for key, width in SAMPLED_POLYLINE_CHANNELS:
        batch = paint(key)
        for line in data.get(key, []):
            for ring in polyline_rings(line, width):
                batch.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    # The hatch strokes and stall ticks really are two-point segments, so only their ends matter.
    for key in TWO_POINT_CHANNELS:
        for line in data.get(key, []):
            ring = line_ring(line[0], line[-1], TWO_POINT_WIDTHS_M.get(key, TWO_POINT_WIDTH_M))
            if ring is not None:
                paint(key).add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    # A TWO-WAY LANE'S CENTRE STRIPE IS YELLOW, and the channel is what decides that - see
    # YELLOW_CHANNELS. Its own loop rather than a row in SAMPLED_POLYLINE_CHANNELS because it is
    # laid at CENTERLINE_WIDTH_M, not at an edge line's width. Already cut into dashes upstream.
    for line in data.get("bike_lane_contraflow_lines", []):
        for ring in polyline_rings(line, CENTERLINE_WIDTH_M):
            yellow.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    # A lane line between two lanes running the same way: white, the centre stripe's normal width
    # (MUTCD 3B.06(05)). Already cut into dashes upstream where it is broken.
    for line in data.get("lane_lines", []):
        for ring in polyline_rings(line, CENTERLINE_WIDTH_M):
            white.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    white.build()
    yellow.build()
    blue.build()

    for i, ring in enumerate(data.get("corner_apron_polygons", [])):
        extrude_polygon(f"corner_apron_{i}", ring, 0.01, apron_mat, z_base=marking_z)
    # The bike lane's own asphalt, painted green. UNDER the stripe layer by one clearance gap, so the
    # white edge lines sit on top of the green the way they do on a real street - and so the two never
    # end up coplanar, which is the z-fighting this file's header is mostly about. Half a clearance
    # thick, so its TOP stays below the stripe layer's base rather than landing exactly on it.
    green = MeshBatch("bike_lane_surface", bike_surface_mat)
    for ring in data.get("bike_lane_surface_polygons", []):
        green.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z - MARKING_CLEARANCE_M)
    green.build()
    # AND NOTHING AT ALL for `bike_lane_uncoloured_surface_polygons`, which is the same footprint
    # with no green on it - an EXISTING conventional bike lane, whose surface is the road's own
    # asphalt. The carriageway already renders as asphalt, so the honest 3D drawing of it is the
    # white edge lines and the symbol, which travel in their own channels above. This is a
    # DECISION and not an omission: markings.NOT_DRAWN_IN_3D declares it, and a test compares
    # that declaration against the channels this file draws - because a channel quietly missing
    # from here is the one seam in the project nothing else can see (README).

    # The BIKE LANE symbol, white on the green. AT the stripe layer rather than half a clearance
    # below it like the green is, because the symbol is paint applied ON the coloured surface -
    # same height as the edge lines, which is what stops it z-fighting with the green it sits on.
    symbols = MeshBatch("bike_lane_symbol", marking_mat)
    for ring in data.get("bike_lane_symbol_polygons", []):
        symbols.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z)
    symbols.build()

    # THE TWO-STAGE TURN BOX (MUTCD 9E.11), where a rider crosses onto a bikeway on the far kerb.
    # Same green material and the same height as the lane surface, because 9E.11(12) makes it the
    # same coloured pavement - and at the same z for the same reason, so the white line round it
    # and the symbol and arrow inside it sit on top rather than z-fighting with it.
    box = MeshBatch("turn_box_surface", bike_surface_mat)
    for ring in data.get("turn_box_surface_polygons", []):
        box.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z - MARKING_CLEARANCE_M)
    box.build()

    # A PAINTED CURB EXTENSION's tan fill (src/osm_world.py:FILL_COLOURS), under its white outline
    # and posts at the same layer as the green surfaces.
    tan = MeshBatch("tan_fill", tan_paint_mat)
    for ring in data.get("tan_fill_polygons", []):
        tan.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z - MARKING_CLEARANCE_M)
    tan.build()

    # THE SHARROW (MUTCD 9E.09), downstream of where the facility ends. White paint on the road's
    # own asphalt, at the stripe layer - and note what is NOT here: 9E.09(05) forbids green
    # pavement as its background, so this batch reads a channel of its own and no green is ever
    # laid under it. See markings.py, where the same rule is stated as an absence from MAY_LIE_ON.
    sharrows = MeshBatch("shared_lane_marking", marking_mat)
    for ring in data.get("shared_lane_symbol_polygons", []):
        sharrows.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z)
    sharrows.build()

    # LANE-USE ARROWS (MUTCD 3B.20), from each lane's `turn:lanes` upstream: white (3B.20(03)), on
    # the road's own asphalt, at the stripe layer like the sharrow. Placed and shaped upstream.
    arrows = MeshBatch("lane_arrows", marking_mat)
    for ring in data.get("lane_arrow_polygons", []):
        arrows.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z)
    arrows.build()

    # EVERY SURVEYED CROSSING IN THE PICTURE, drawn from its own traced way rather than rebuilt
    # from a leg. This is the network-renderer change (docs/network-renderer-plan.md): a crossing
    # used to reach the render only by matching one of the modelled junction's legs, so at Broad &
    # Greenwood framed 2.5x, 6 of the 10 OSM crossings inside the frame were dropped - three of
    # them tagged crossing:markings=zebra, and Blackwell & Broad rendered as bare asphalt where its
    # crosswalks are traced. A render that shows a marked crosswalk as unmarked is a false claim
    # about the street, which is the one thing these drawings cannot afford.
    #
    # Alongside `kerbs` and `paved_surfaces` rather than in the paint channels, and for the same
    # reason those two are: this is SURVEYED CONTEXT, not a treatment this project proposes. It is
    # already in the ground's coordinates, it belongs to no leg, and nothing should be cut around it.
    #
    # The style comes from the crossing's own tags upstream - zebra becomes bars, `lines` becomes
    # two transverse lines, and a crossing with nothing recorded contributes neither. Blender
    # derives nothing here, which is the rule on this side of the boundary.
    phase("paint channels")
    surveyed = MeshBatch("surveyed_crossings", marking_mat)
    for crossing in data.get("surveyed_crossings", []):
        for ring in crossing.get("bars", []):
            surveyed.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z)
        for line in crossing.get("lines", []):
            for ring in polyline_rings(line, SURVEYED_CROSSING_LINE_WIDTH_M):
                surveyed.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    surveyed.build()

    # A curb ramp's detectable warning surface (`tactile_paving=yes`, ADA 705.1's 24 in pad): dark
    # red, at the ramp's flush foot - the road's top plus the clearance a flush kerb keeps. The
    # reader cuts each pad out of the sidewalk, so the slab above it hides none of it.
    tactile_mat = make_material("TactilePaving", (0.38, 0.06, 0.05), roughness=0.9)
    tactile = MeshBatch("tactile_paving", tactile_mat)
    for ring in data.get("tactile_paving_polygons", []):
        if len(ring) >= 3:
            tactile.add_prism(ring, PAVEMENT_HEIGHT_M + MARKING_CLEARANCE_M, z_base=0.0)
    tactile.build()

    # A CYCLE CROSSING THROUGH A JUNCTION, at the same layers as the bikeway it carries on: green
    # half a clearance under the stripes, its dotted edges and divider at the stripe layer. It
    # never meets a crosswalk's bars - the reader cuts every crosswalk's band out of other paint.
    crossing_green = MeshBatch("cycle_crossing_surface", bike_surface_mat)
    for ring in data.get("cycle_crossing_surface_polygons", []):
        crossing_green.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z - MARKING_CLEARANCE_M)
    crossing_green.build()
    crossing_white = MeshBatch("cycle_crossing_edges", marking_mat)
    for line in data.get("cycle_crossing_edge_lines", []):
        for ring in polyline_rings(line, si("line.width")):
            crossing_white.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    crossing_white.build()
    crossing_yellow = MeshBatch("cycle_crossing_divider", centerline_mat)
    for line in data.get("cycle_crossing_divider_lines", []):
        for ring in polyline_rings(line, CENTERLINE_WIDTH_M):
            crossing_yellow.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    crossing_yellow.build()

    for island in data.get("refuge_islands", []):
        extrude_polygon(f"refuge_{island['name']}", island["coords"], island.get("height_m", 0.15), refuge_mat)

    for crossing in data.get("raised_crossings", []):
        extrude_polygon(
            f"crossing_{crossing['name']}", crossing["coords"], crossing.get("height_m", 0.10), crossing_mat
        )

    raised_leg_names = {c["name"] for c in data.get("raised_crossings", [])}
    # Only draw a painted crosswalk where one is actually confirmed to exist
    # (config: intersection.existing_marked_crosswalks) - don't assume every
    # approach is marked just because it's a signalized 4-way.
    marked_leg_names = set(data.get("existing_marked_crosswalks", []))
    # Depth comes from src/render/crosswalks.py:CROSSWALK_DEPTH_M via the JSON, so the 2D plan
    # view (src/render/plan_view.py) and this render draw an identically-sized crosswalk.
    crosswalk_depth_m = data.get("crosswalk_depth_m", 1.829)  # 6 ft; see blender_crosswalks.py
    stop_bar_curb_clearance_m = data.get("stop_bar_curb_clearance_m", 0.5)
    phase("surveyed crossings + islands")
    centerline_yellow = MeshBatch("centerline_yellow", centerline_mat)
    centerline_white = MeshBatch("centerline_white", marking_mat)
    for leg in data.get("legs", []):
        near = mathutils.Vector((*leg["near_m"], 0.0))
        far = mathutils.Vector((*leg["far_m"], 0.0))
        direction = far - near
        if direction.length < 1e-3:
            continue
        u = direction / direction.length
        n = mathutils.Vector((-u.y, u.x, 0))
        offset_m = leg.get("crosswalk_offset_m", 3.0)

        if leg["name"] in marked_leg_names and leg["name"] not in raised_leg_names:
            style = leg.get("crosswalk_style", "lines")
            cw_centre, cw_u, cw_n = _marking_frame(leg, near, u, n, "crosswalk", offset_m)
            add_crosswalk(f"crosswalk_{leg['name']}", cw_centre, cw_u, cw_n, leg["width_m"],
                           marking_mat,
                           offset_m=0.0, style=style, depth_m=crosswalk_depth_m,
                           skew_deg=leg.get("crosswalk_skew_deg", 0.0),
                           reach_left_m=leg.get("crosswalk_reach_left_m"),
                           reach_right_m=leg.get("crosswalk_reach_right_m"),
                           n_stripes=leg.get("crosswalk_bar_count"))
        stop_bar_offset_m = leg.get("stop_bar_offset_m")
        if stop_bar_offset_m is not None:
            stop_bar_width_m = leg.get("stop_bar_width_m") or leg["width_m"]
            # A stop bar is painted parallel to the crosswalk ahead of it, so it takes
            # the same surveyed skew.
            sb_centre, sb_u, sb_n = _marking_frame(leg, near, u, n, "stop_bar", stop_bar_offset_m)
            add_stop_bar(f"stop_bar_{leg['name']}", sb_centre, sb_u, sb_n, stop_bar_width_m,
                         marking_mat,
                         offset_m=0.0, skew_deg=leg.get("crosswalk_skew_deg", 0.0),
                         curb_clearance_m=stop_bar_curb_clearance_m,
                         span_m=leg.get("stop_bar_span_m"),
                         lateral_offset_m=leg.get("stop_bar_lateral_offset_m"))
        # Real per-leg fact (confirmed via street-view, see src/geometry/treatments/
        # DEFAULT_CENTERLINE_STYLE) - some legs get no centerline paint at all, so this
        # is NOT drawn unconditionally the way it used to be.
        centerline_style = leg.get("centerline_style", "single_yellow_dashed")
        # Where the paint starts is decided upstream (src/render/crosswalks.py:
        # centerline_start_ft) so the "stop at the stop bar" rule is testable and can't
        # drift from the bar this same scene draws. Falls back to the old fixed gap past
        # the crosswalk for geometry written before that field existed.
        centerline_start_m = leg.get("centerline_start_m", offset_m + 2)
        # THE STRIPES COME DOWN AS GEOMETRY, following the leg's real centerline - already
        # offset into the two lines of a double yellow, already cut into the segments of a
        # dashed one (src/render/crosswalks.py:centerline_paint_ft). The two calls below build
        # a stripe between near and far instead, which is the leg's CHORD: up to 3.98 ft off
        # the centerline on broad_st_east and 7.58 ft on louellen_st_west, so the double yellow
        # missed the stop bar it is supposed to meet and the lanes either side of it came out
        # different widths. They remain as the fallback for a geometry file written before this
        # key, and they are the reason nothing may derive a marking's shape on this side of the
        # boundary: the plan view had it right the whole time and nothing could compare them.
        painted = leg.get("centerline_paint_m")
        if painted is not None:
            # YELLOW SEPARATES OPPOSING DIRECTIONS, WHITE SEPARATES LANES GOING THE SAME WAY
            # (MUTCD 11th ed. 3B.01 P1 and 3B.06 P1). The material used to be centerline_mat
            # unconditionally, which is why a one-way carriageway's lane line could not be drawn
            # at all: its geometry is this same line down the middle of the road, and coming out
            # yellow it would have told a driver the next lane runs at them. CENTERLINE_STYLES_
            # WHITE mirrors treatments.CENTERLINE_IS_WHITE, which Blender cannot import - pinned
            # by tests/test_paint.py:test_blender_centerline_colours_match_the_styles.
            line_mat = marking_mat if centerline_style in CENTERLINE_STYLES_WHITE else centerline_mat
            # BATCHED, one mesh per material across every leg: a centreline is a sampled polyline
            # cut into dashes, so it is thousands of segments, and as an object each it was 14,112
            # of the 18,807 objects in a synthetic 36-junction world. Same ring, same height and
            # base as add_paint_polyline's defaults, so the picture does not move.
            batch = centerline_white if line_mat is marking_mat else centerline_yellow
            for line in painted:
                for ring in polyline_rings(line, CENTERLINE_WIDTH_M):
                    batch.add_prism(ring, PAINT_HEIGHT_M, z_base=EXISTING_MARKING_HEIGHT_M - PAINT_HEIGHT_M)
        elif centerline_style == "double_yellow":
            add_double_yellow_centerline(f"centerline_{leg['name']}", near, far, centerline_mat,
                                          start_m=centerline_start_m)
        elif centerline_style == "single_yellow_dashed":
            add_dashed_centerline(f"centerline_{leg['name']}", near, far, centerline_mat,
                                   start_m=centerline_start_m)

    # Props: real streetlight model (or procedural fallback) at each corner,
    # procedural signage incl. traffic signals (no CC0 source available - see
    # blender_props.py / README.md). Placement is decided upstream by
    # src/render/props.py; add_prop() just dispatches each exported prop dict to its
    # builder.
    centerline_yellow.build()
    centerline_white.build()
    phase("legs (crosswalks, stop bars, centrelines)")
    streetlight_template = import_gltf_template(theme.get("streetlight_gltf"), "streetlight_template")
    for i, prop in enumerate(data.get("props", [])):
        add_prop(f"{prop['type']}_{i}", prop, streetlight_template, pole_mat,
                 signal_housing_mat, ped_signal_housing_mat)

    # Trees: one shared low-poly mesh, geometry-nodes-instanced along the
    # sidewalk bands (not one mesh copy per tree).
    tree_points = data.get("tree_points", [])
    if tree_points:
        tree_template = build_tree_proxy(trunk_mat, foliage_mat)
        add_tree_instances("street_trees", tree_points, tree_template)

    phase("props + trees")
    view = None if world else corridor_view(framed_x, framed_y, cx, cy)
    return Built(cx=cx, cy=cy, scene_radius=scene_radius, ground_size=ground_size, view=view,
                 ground_rect=(gcx - ground_w / 2, gcy - ground_h / 2, gcx + ground_w / 2, gcy + ground_h / 2),
                 max_height_m=max_building_height(data), ground=ground)


# A SUBJECT LONGER THAN THIS CANNOT BE FRAMED AS A SQUARE. The camera below looks from due
# south and sizes itself off half the LARGER span, which fits a junction about as wide as it is
# tall. W Broad & Lanning is not one: its southwest leg runs 2,307 ft to the borough line, 23x
# the other three, and squaring that extent laid the whole corridor across the diagonal of a
# 4:3 frame as a thread on an otherwise empty field.
#
# THE THRESHOLD IS SET FROM THE MEASUREMENT, not from taste, because a render that moves is a
# render somebody has to re-check (.claude/SKILLS.md: adding a site must not move existing
# sites). The seven junctions already in output/ measure 0.81, 1.02, 1.03, 1.03, 2.17, 2.18 and
# 3.90 on this ratio; this corridor measures 9.62. 6.0 sits in that gap, so no existing render
# changes and princeton_eprospect - the one T-junction that is genuinely long and thin at 3.90,
# and which the square frame still draws legibly - keeps the camera it has.
ELONGATED_ASPECT = 6.0


def oriented_extent(xs, ys, cx, cy):
    """((ux, uy) along the cloud's long axis, half-length along it, half-width across it).

    The closed-form principal axis of the 2x2 covariance. The half-extents are the largest
    absolute projection on each axis and NOT a standard deviation, because a camera has to
    contain the subject rather than describe it.
    """
    dx = [x - cx for x in xs]
    dy = [y - cy for y in ys]
    n = max(len(dx), 1)
    sxx = sum(a * a for a in dx) / n
    syy = sum(b * b for b in dy) / n
    sxy = sum(a * b for a, b in zip(dx, dy)) / n
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)
    along = max((abs(a * ux + b * uy) for a, b in zip(dx, dy)), default=0.0)
    across = max((abs(-a * uy + b * ux) for a, b in zip(dx, dy)), default=0.0)
    return (ux, uy), along, across


# THE CAMERA'S OWN FRAMING, as multiples of scene_radius, named because corridor_view has to
# divide by what they imply. The lens and the 36 mm horizontal sensor fit are Blender's.
# A little air past each end of the corridor, so the far kerb and the borough line are inside
# the sheet rather than cut by it - `along` is the extreme projected vertex, not a margin.
CORRIDOR_END_MARGIN = 1.1
CAMERA_DIST_MULTIPLE = 1.6
CAMERA_HEIGHT_MULTIPLE = 2.3
CAMERA_LENS_MM = 32.0
CAMERA_SENSOR_MM = 36.0


def lateral_coverage_multiple() -> float:
    """How many scene_radii of ground the frame spans side to side, at the target."""
    eye_to_target = math.hypot(CAMERA_DIST_MULTIPLE, CAMERA_HEIGHT_MULTIPLE)
    return eye_to_target * (CAMERA_SENSOR_MM / 2) / CAMERA_LENS_MM


def corridor_view(xs, ys, cx, cy):
    """(unit vector the camera should stand on, radius to frame) for an elongated subject.

    None for anything that fits a square, which is every junction here. THE VIEWING ANGLE IS
    UNCHANGED - same 3/4 oblique, same elevation, same distance-to-radius ratio as every other
    render. All this does is turn the camera about the vertical axis so the corridor lies ACROSS
    the frame instead of on its diagonal, and size the radius off the long half-extent rather
    than off half the larger axis-aligned span. A different angle for one site would be a second
    way of drawing the same thing.
    """
    (ux, uy), along, across = oriented_extent(xs, ys, cx, cy)
    if across <= 0 or along / across < ELONGATED_ASPECT:
        return None
    # Perpendicular to the corridor, and of the two perpendiculars the one that keeps the camera
    # as close to the others' due-south station as it can be.
    px, py = -uy, ux
    if py > 0:
        px, py = -px, -py
    # AND A RADIUS THAT FILLS THE FRAME. Everywhere else scene_radius is half the larger span and
    # the lens then shows about 1.6x that, which is the margin a junction wants around it. Here
    # that margin is 700 ft of empty field off each end of the corridor, so the radius is divided
    # back out by the camera's own lateral coverage instead of being taken raw.
    return (px, py), along * CORRIDOR_END_MARGIN / lateral_coverage_multiple(), across


def clip_range(cam, eye_z: float, ground_rect, max_height_m: float):
    """(clip_start, clip_end) for `cam`, from where it stands and what is in the world.

    Both ends are DEPTHS ALONG THE VIEW AXIS, which is what a clip plane is - not distances from
    the eye. The camera's four corner rays are intersected with the ground plane: the farthest hit is
    the farthest anything in the picture can be (a building stands ABOVE the ground and so meets the
    ray sooner), and the nearest hit, lowered by the tallest building's share of the camera's
    height, is the nearest anything can be (a rooftop on the bottom ray is (1 - H/h) of the way to the
    ground).

    Why this and not the old `dist + height + ground_size`: that bound grows with the GROUND, and a
    world's ground is a borough. The depth buffer's precision is set by far/near, and the comment in
    add_camera records what a too-wide range does to a thin crosswalk line at 50-100 m. With a
    frustum-derived range the answer does not depend on how big the world is, which is the property
    a world needs: a camera in a 2 km world and one in a 200 m site that look at the same ground get
    the same clip range.

    The ground rectangle still caps the far end, because a world that ends inside the view has
    nothing beyond its edge to draw. A ray that never reaches the ground (a camera pitched up) falls
    back to that cap alone.
    """
    rot = cam.rotation_euler.to_matrix()
    fwd = rot @ mathutils.Vector((0.0, 0.0, -1.0))
    half_w = CAMERA_SENSOR_MM / 2
    half_h = half_w * 3 / 4  # 4:3 - a letterboxed corridor sheet is a strict subset of this
    depths = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            ray = rot @ mathutils.Vector((sx * half_w, sy * half_h, -CAMERA_LENS_MM))
            if ray.z >= -1e-6:
                depths = []
                break
            # the ray scaled to touch the ground, then its component along the view axis
            depths.append((ray * (eye_z / -ray.z)).dot(fwd))
        else:
            continue
        break
    eye = cam.location
    rect_far = max((mathutils.Vector((x, y, 0.0)) - eye).dot(fwd)
                   for x in (ground_rect[0], ground_rect[2]) for y in (ground_rect[1], ground_rect[3]))
    if not depths:
        return CLIP_NEAR_FLOOR_M, max(rect_far, 10.0)
    far = min(max(depths), rect_far) * CLIP_FAR_MARGIN
    low = max(0.0, 1.0 - max_height_m / eye_z) if eye_z > 0 else 0.0
    near = max(min(depths) * low * CLIP_NEAR_MARGIN, CLIP_NEAR_FLOOR_M)
    return near, max(far, near * 2)


# Safety factors on the frustum-derived range: a few percent so a rooftop or a tree crown at the very
# edge of the picture is not shaved, and so a ray bound by rounding is not exactly on a plane.
CLIP_FAR_MARGIN = 1.05
CLIP_NEAR_MARGIN = 0.9
CLIP_NEAR_FLOOR_M = 1.0


def add_camera(cx: float, cy: float, scene_radius: float, ground_rect, max_height_m: float, view=None):
    """The camera for one picture, at the same angle, lens and distance as ever - only the clip range
    is derived (clip_range). Returns (camera object, its height, the radius it really framed)."""
    if view is not None:
        # SAME ANGLE, TURNED IN PLAN - see corridor_view. The two multipliers below are the ones
        # every other render uses; only the compass direction the camera stands in changes.
        (px, py), scene_radius, _across = view
    else:
        px, py = 0.0, -1.0
    dist = scene_radius * CAMERA_DIST_MULTIPLE
    height = scene_radius * CAMERA_HEIGHT_MULTIPLE
    eye = (cx + px * dist, cy + py * dist, height)
    bpy.ops.object.camera_add(location=eye)
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam
    direction = mathutils.Vector((cx, cy, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = CAMERA_LENS_MM
    # DEPTH-BUFFER PRECISION, which is why the clip range is tight and derived rather than left at
    # Blender's default (0.1 - 1000 m). That is enormously wider than a scene needs and starves the
    # depth buffer at the ~50-100 m this camera stands at - confirmed by an isolated test: thin, long
    # ground markings (crosswalk lines) rendered as a torn/tessellated mess with the default range and
    # perfectly solid once it was tightened, with shadow settings held constant (so a camera
    # depth-buffer problem, not a shadow one, despite looking like the z-fighting/shadow-acne bugs
    # documented elsewhere in this file/README). It matters MORE in a world, where the ground a
    # bound like `dist + height + ground_size` would reach is a borough wide - hence clip_range.
    cam.data.clip_start, cam.data.clip_end = clip_range(cam, height, ground_rect, max_height_m)
    return cam, height, scene_radius


def add_sun_and_sky(cx: float, cy: float, scene_radius: float, height: float):
    bpy.ops.object.light_add(type="SUN", location=(cx + scene_radius * 0.3, cy - scene_radius * 0.3, height))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 2.2
    sun.data.angle = 0.2  # soften shadow edges slightly
    sun.rotation_euler = (0.85, 0.15, 0.75)
    set_sky()
    return sun


def set_sky():
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.68, 0.82, 1.0)
        bg.inputs["Strength"].default_value = 0.6


def setup_camera_and_light(cx: float, cy: float, scene_radius: float, ground_rect, max_height_m: float,
                           view=None):
    cam, height, scene_radius = add_camera(cx, cy, scene_radius, ground_rect, max_height_m, view)
    add_sun_and_sky(cx, cy, scene_radius, height)
    return cam


# The render's own resolution, which --dpi does NOT control: that knob is matplotlib's and
# reaches only the 2D plan views. Setting --dpi 300 and expecting sharper renders is the
# obvious mistake and somebody made it, so there is now a knob for this too - a whole-number
# multiple of the base size, from ROAD_SKETCHES_RENDER_SCALE (scripts/build_all.py --render-scale).
# A multiplier rather than a width/height pair keeps the camera framing and the 4:3 aspect
# fixed, so scale 2 is the same picture with four times the pixels, not a different crop.
BASE_RESOLUTION = (1920, 1440)
# AND A WIDER SHEET FOR A CORRIDOR, which is the other half of framing one (see corridor_view).
# Turning the camera puts W Broad across the frame instead of down its diagonal, but a 4:3 sheet
# then spends two thirds of its height on empty field either side of a 108 ft-wide street. The
# CAMERA IS NOT TOUCHED - same elevation, same lens, same distance; the render is simply cropped
# to the band the subject occupies. Horizontal coverage is 2x the corridor's half-length, and
# this asks for 3x its half-width vertically - so the cross streets, the junction's buildings and
# the kerb either side all stay in shot.
CORRIDOR_SHEET_MARGIN = 3.0
# Not past this, because a strip thinner than 16:9 stops reading as a picture of a place and the
# 3D view has nothing left to show above the ground plane.
CORRIDOR_SHEET_MAX_ASPECT = 4.0
RENDER_SCALE_ENV = "ROAD_SKETCHES_RENDER_SCALE"
# Its pre-rename name, refused rather than ignored - a stale HOPEWELL_RENDER_SCALE=2 would
# render at 1 and say nothing, and an hour of Blender is a slow way to find that out. This
# module runs under Blender's own Python and cannot import src, so the check is duplicated
# here deliberately; src/sources/data_loader.py:RENAMED_ENV covers everything that can.
if "HOPEWELL_RENDER_SCALE" in os.environ:
    raise SystemExit("HOPEWELL_RENDER_SCALE was renamed to ROAD_SKETCHES_RENDER_SCALE and is "
                     "no longer read - this render would silently come out at scale 1.")


def render_scale() -> int:
    """The resolution multiplier, clamped to something a machine can actually finish.

    Scale 4 is 7680x5760, which is where EEVEE's memory use starts to matter alongside the
    ~11 GB a scene already costs (see phase4_render_3d.BLENDER_PEAK_RAM_GB) - past that the
    OOM killer arrives and Blender says nothing about why, so the cap is kinder than the
    crash. Anything unparseable falls back to 1 with a warning rather than failing a batch
    of renders over an environment variable.
    """
    raw = os.environ.get(RENDER_SCALE_ENV, "1")
    try:
        scale = int(raw)
    except ValueError:
        print(f"WARNING: {RENDER_SCALE_ENV}={raw!r} is not an integer - rendering at 1x.")
        return 1
    if scale < 1 or scale > 4:
        print(f"WARNING: {RENDER_SCALE_ENV}={scale} is outside 1-4 - clamping.")
    return max(1, min(scale, 4))


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64  # visually indistinguishable from 128 for this flat-shaded scene, ~30% faster
    scale = render_scale()
    if scale != 1:
        print(f"Rendering at {BASE_RESOLUTION[0] * scale}x{BASE_RESOLUTION[1] * scale} ({scale}x)")


def set_sheet(view=None):
    """The output resolution, which is per-scene because a corridor gets a letterbox.

    Sits apart from configure_render because everything there is scene-independent and is set
    once for a whole batch; this is the one render setting that is not.
    """
    scene = bpy.context.scene
    scale = render_scale()
    scene.render.resolution_x = BASE_RESOLUTION[0] * scale
    scene.render.resolution_y = BASE_RESOLUTION[1] * scale
    if view is None:
        return
    _direction, along, across = view
    aspect = min(along / (CORRIDOR_SHEET_MARGIN * across), CORRIDOR_SHEET_MAX_ASPECT)
    if aspect > BASE_RESOLUTION[0] / BASE_RESOLUTION[1]:
        scene.render.resolution_y = int(round(scene.render.resolution_x / aspect))


def render(output_path: Path):
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def disable_undo():
    """Stop Blender snapshotting the scene after every operator.

    Undo exists for a person clicking in the UI. In a headless batch there is nobody to undo for,
    and the cost is not small: every `bpy.ops` call pushes a snapshot of the scene, so the price of
    an operator grows with the scene it is called in. This build still uses operators for the
    buildings' face merge and for the props, which run late, when the scene is at its largest.

    Set once at startup rather than per render, because it is a preference and not scene state.
    """
    bpy.context.preferences.edit.use_global_undo = False


def render_sites(pairs):
    """The original mode: one JSON, one scene, one camera at its `frame`, one PNG - per pair."""
    for geometry_path, output_path in pairs:
        data = load_geometry(geometry_path)
        clear_scene()
        built = build_scene(data)
        setup_camera_and_light(built.cx, built.cy, built.scene_radius, built.ground_rect,
                               built.max_height_m, built.view)
        set_sheet(built.view)
        render(output_path)
        print(f"RENDER_DONE: {output_path}")


# The scene custom property a built world carries for camera mode to read, so a .blend is
# self-describing and cameras.json needs nothing but a centre and a radius.
WORLD_PROP = "nj_world"


def pack_images():
    """Embed every file-backed image in the .blend, so the saved world does not depend on
    output/.textures staying where it was when the world was built. Packed JPEGs stay compressed,
    so this costs the files' own size and nothing is decoded until a render uses it."""
    packed = 0
    for img in bpy.data.images:
        if img.source == "FILE" and not img.packed_file:
            try:
                img.pack()
                packed += 1
            except RuntimeError as e:
                print(f"  WARNING: could not pack {img.filepath!r} ({e}) - the .blend will look for it on disk")
    return packed


def build_world(job):
    """--build: the whole scene from one geometry JSON, no camera, saved as a .blend."""
    started = time.perf_counter()
    data = load_geometry(job.geometry, WORLD_REQUIRED_KEYS)
    clear_scene()
    built = build_scene(data, world=True, texture_res=job.texture_res)
    set_sky()  # the sky is the world's, and a camera render re-sets the same values
    bpy.context.scene[WORLD_PROP] = {
        "ground_rect": list(built.ground_rect),
        "max_height_m": built.max_height_m,
        "texture_res": job.texture_res,
    }
    packed = pack_images()
    built_s = time.perf_counter() - started
    job.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(job.blend), compress=True)
    print(f"WORLD_SAVED: {job.blend} ({len(bpy.data.objects)} objects, {packed} images packed, "
          f"built in {built_s:.1f}s, saved in {time.perf_counter() - started - built_s:.1f}s)")


def load_cameras(path: Path) -> list[dict]:
    """The camera specs, validated. A bad spec fails here, before a world is opened and a render
    is spent - and a name is a filename, so nothing that could escape --out-dir is allowed."""
    with open(path) as f:
        specs = json.load(f)
    if isinstance(specs, dict):
        specs = specs.get("cameras", [specs])
    seen = set()
    for spec in specs:
        name = spec.get("name")
        if (not isinstance(name, str) or not name or name.startswith(".")
                or not all(c.isalnum() or c in "_.-" for c in name)):
            raise SystemExit(f"{path}: camera name {name!r} must be non-empty and use only "
                             f"letters, digits, '_', '-' and '.'")
        if name in seen:
            raise SystemExit(f"{path}: duplicate camera name {name!r}")
        seen.add(name)
        center, radius = spec.get("center_m"), spec.get("radius_m")
        if not (isinstance(center, (list, tuple)) and len(center) == 2
                and all(isinstance(v, (int, float)) for v in center)):
            raise SystemExit(f"{path}: camera {name!r} needs center_m: [x, y] in metres")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise SystemExit(f"{path}: camera {name!r} needs a positive radius_m")
    return specs


class PavementIndex:
    """The pavement's top-face boundary, read back out of the saved world's own meshes.

    Camera mode has no geometry JSON - only the .blend - yet corridor_view needs the pavement's
    extent near the camera, to decide whether the subject is a corridor and turn the camera across it
    (see corridor_view). So the ground truth is the mesh. Its edges are kept as segments and SAMPLED
    inside the camera's disc at query time, because a world's pavement can be one long ring whose
    vertices are all outside the disc while its edge runs straight through it.
    """

    def __init__(self):
        self.segments = []   # (x1, y1, x2, y2)
        top = PAVEMENT_HEIGHT_M * 0.5
        for obj in bpy.data.objects:
            if obj.type != "MESH" or not obj.name.startswith("pavement"):
                continue
            mesh = obj.data
            n = len(mesh.vertices)
            co = [0.0] * (3 * n)
            mesh.vertices.foreach_get("co", co)
            ev = [0] * (2 * len(mesh.edges))
            mesh.edges.foreach_get("vertices", ev)
            for i in range(0, len(ev), 2):
                a, b = ev[i], ev[i + 1]
                if co[3 * a + 2] > top and co[3 * b + 2] > top:
                    self.segments.append((co[3 * a], co[3 * a + 1], co[3 * b], co[3 * b + 1]))

    def within(self, cx: float, cy: float, reach: float):
        """(xs, ys) of pavement-edge points within `reach` of (cx, cy)."""
        xs, ys = [], []
        step = max(reach / 40.0, 0.5)
        for x1, y1, x2, y2 in self.segments:
            if (max(x1, x2) < cx - reach or min(x1, x2) > cx + reach
                    or max(y1, y2) < cy - reach or min(y1, y2) > cy + reach):
                continue
            n = max(1, int(math.hypot(x2 - x1, y2 - y1) / step))
            for k in range(n + 1):
                t = k / n
                x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                if math.hypot(x - cx, y - cy) <= reach:
                    xs.append(x)
                    ys.append(y)
        return xs, ys


# How far from a camera's centre, in its own radii, the pavement is read to decide whether the
# subject is a corridor. A frame's radius is 1.2x its pavement's half-span (src/render/frame.py), so
# every vertex a site's own frame framed lies inside 1.2*sqrt(2)/1.2 ~ 1.2 radii of its centre.
CORRIDOR_PROBE_RADII = 1.2


def ground_for_camera(ground, world_rect, cx: float, cy: float, scene_radius: float):
    """Grow the ground plane, in memory, to cover the world AND this camera's 4x frame.

    A site's ground was always at least four frame-radii across (build_scene), because a camera that
    pulls back further than the context reaches shows the plane's own edge with sky beneath it. In a
    world the plane covers the geometry; a camera near the world's edge needs the same four radii
    beyond it. Returns the rectangle, which also bounds the clip range.
    """
    half = scene_radius * 2.0
    minx, miny = min(world_rect[0], cx - half), min(world_rect[1], cy - half)
    maxx, maxy = max(world_rect[2], cx + half), max(world_rect[3], cy + half)
    ground.location = ((minx + maxx) / 2, (miny + maxy) / 2, ground.location.z)
    ground.scale = (maxx - minx, maxy - miny, 1.0)
    return (minx, miny, maxx, maxy)


def render_cameras(job):
    """--open: one PNG per camera spec, in a world that was built once."""
    specs = load_cameras(job.cameras)
    opened = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=str(job.blend))
    meta = bpy.context.scene.get(WORLD_PROP)
    if meta is None or "Ground" not in bpy.data.objects:
        raise SystemExit(f"{job.blend} was not built by `--build` (no '{WORLD_PROP}' on its scene) - "
                         f"cannot place cameras in it.")
    world_rect = tuple(meta["ground_rect"])
    max_height_m = float(meta["max_height_m"])
    ground = bpy.data.objects["Ground"]
    disable_undo()
    configure_render()
    pavement = PavementIndex()
    print(f"WORLD_OPENED: {job.blend} ({len(bpy.data.objects)} objects, "
          f"{time.perf_counter() - opened:.1f}s incl. pavement index of {len(pavement.segments)} edges)")
    job.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        cx, cy = spec["center_m"]
        radius = float(spec["radius_m"])
        xs, ys = pavement.within(cx, cy, radius * CORRIDOR_PROBE_RADII)
        view = corridor_view(xs, ys, cx, cy) if xs else None
        framed = view[1] if view is not None else radius
        ground_rect = ground_for_camera(ground, world_rect, cx, cy, max(radius, framed))
        started = time.perf_counter()
        cam, height, used_radius = add_camera(cx, cy, radius, ground_rect, max_height_m, view)
        sun = add_sun_and_sky(cx, cy, used_radius, height)
        set_sheet(view)
        output_path = job.out_dir / f"{spec['name']}.png"
        render(output_path)
        print(f"RENDER_DONE: {output_path} ({time.perf_counter() - started:.1f}s, "
              f"clip {cam.data.clip_start:.1f}-{cam.data.clip_end:.1f} m"
              f"{', corridor view' if view is not None else ''})")
        for obj in (cam, sun):
            data_block = obj.data
            kind = obj.type
            bpy.data.objects.remove(obj, do_unlink=True)
            (bpy.data.cameras if kind == "CAMERA" else bpy.data.lights).remove(data_block)


def main():
    job = parse_args()
    if job.mode == "cameras":
        render_cameras(job)
        return
    disable_undo()
    configure_render()  # render settings are scene-independent - set once
    if job.mode == "build":
        build_world(job)
    else:
        render_sites(job.pairs)


if __name__ == "__main__":
    main()
