"""Headless Blender: build a world (src/osm_world.py's JSON) once, then render cameras into it.

  blender --background --python scripts/blender/blender_scene.py -- --build <world.json> --save <w.blend>
  blender --background --python scripts/blender/blender_scene.py -- --open <w.blend> \\
      --camera <cameras.json> --out-dir <dir>

Runs under Blender's own Python, which has none of this project's venv: every asset is fetched
beforehand (src/render/theme.py) and arrives as a path in the JSON, and every figure is
standards.toml's, read through src/standards.py (standard library only). Nothing is derived here.
The world says what each channel is - its `paint` and `surfaces` tables (src/osm_world.py) - and
this draws each by it.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import bpy
import mathutils


def _standards():
    """src/standards.py, loaded by path: it imports nothing but the standard library (an import
    contract keeps it so), so Blender's own Python runs it."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "src" / "standards.py"
    spec = importlib.util.spec_from_file_location("standards", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["standards"] = module          # dataclasses resolves its module through sys.modules
    spec.loader.exec_module(module)
    return module


si = _standards().si

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the sibling blender_*.py modules
REPO_ROOT = Path(__file__).resolve().parents[2]

from blender_geometry import MeshBatch, line_ring, polyline_rings
from blender_materials import make_material, make_textured_material
from blender_props import PED_SIGNAL_HOUSING_DARK, SIGN_POST_GRAY, SIGNAL_HOUSING_DARK, add_prop, add_tree_instances, build_tree_proxy

random.seed(7)

# The levels everything stands at. Two surfaces at one height z-fight, so each layer of paint is a
# clearance above the one under it, and the camera's clip range is tight (clip_range).
ROAD_M = si("render.pavement_height")
KERB_M = ROAD_M + si("kerb.raised")                     # a sidewalk's top: a raised kerb's reveal up
CLEARANCE_M = si("render.marking_clearance")
PAINT_Z = si("render.existing_marking_height") + CLEARANCE_M
PAINT_M = si("render.paint_height")
LEVEL_M = {"road": ROAD_M, "kerb": KERB_M}
KERB_WIDTH_M = si("kerb.top_width")
UV_TILE_M = 2.0

# (colour, roughness) by name: the world's paint and surfaces name these. Matched by eye - not a
# standard. Tan is NYC's painted curb extension ("truffle", NYC DOT Permanent Planter Guidelines
# 2019, no measured value published).
PAINTS = {"white": ((0.9, 0.9, 0.88), 0.4), "yellow": ((0.85, 0.7, 0.15), 0.4),
          "blue": ((0.1, 0.25, 0.7), 0.4), "green": ((0.13, 0.45, 0.28), 0.85),
          "tan": ((0.72, 0.6, 0.42), 0.85)}
GROUNDS = {"asphalt": ((0.07, 0.07, 0.08), 0.95), "concrete": ((0.72, 0.71, 0.67), 0.85),
           "gravel": ((0.46, 0.43, 0.38), 0.95), "dirt": ((0.38, 0.30, 0.22), 1.0)}
BUILDING_PALETTE = [(0.62, 0.42, 0.35), (0.82, 0.78, 0.68), (0.55, 0.55, 0.58), (0.70, 0.62, 0.48),
                    (0.45, 0.38, 0.32)]


_CLOCK = [time.perf_counter()]


def phase(name: str) -> None:
    """How long the build phase that just finished took."""
    now = time.perf_counter()
    print(f"  [build] {name}: {now - _CLOCK[0]:.2f}s")
    _CLOCK[0] = now


def world_extent(data: dict) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) over every surface, building, kerb and paved area."""
    points = [p for key in data["surfaces"] for ring in data.get(key, []) for p in ring]
    points += [p for b in data.get("buildings", []) for p in b["coords"]]
    points += [p for item in data.get("kerbs", []) + data.get("paved_surfaces", []) for p in item["coords"]]
    if not points:
        return (-50.0, -50.0, 50.0, 50.0)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def build_scene(data: dict) -> tuple[tuple, float]:
    """Every object in the world; its ground's rectangle and its tallest building."""
    theme = {k: {m: str(REPO_ROOT / p) for m, p in v.items()} if v else None
             for k, v in (data.get("theme") or {}).items()}
    grounds = {name: make_textured_material(name.title(), theme.get(name), rgb, rough)
               for name, (rgb, rough) in GROUNDS.items()}
    paints = {name: make_material(f"Paint_{name}", rgb, rough) for name, (rgb, rough) in PAINTS.items()}

    west, south, east, north = world_extent(data)
    width, height = max((east - west) * 1.25, 100.0), max((north - south) * 1.25, 100.0)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=((west + east) / 2, (south + north) / 2, -0.03))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (width, height, 1.0)
    ground.data.materials.append(make_material("Grass", (0.3, 0.48, 0.24), 1.0))
    rect = ((west + east - width) / 2, (south + north - height) / 2,
            (west + east + width) / 2, (south + north + height) / 2)

    batches = [MeshBatch(f"buildings_{k}", make_material(f"Building{k}", c, 0.75))
               for k, c in enumerate(BUILDING_PALETTE)]
    for i, b in enumerate(data.get("buildings", [])):
        batches[i % len(batches)].add_prism(b["coords"], b["height_m"])
    for batch in batches:
        batch.build(uv_tile_m=UV_TILE_M)
    phase("buildings")

    surfaces = {name: MeshBatch(f"surface_{name}", mat) for name, mat in grounds.items()}
    for key, (material, level) in data["surfaces"].items():
        for ring in data.get(key, []):
            surfaces[material].add_prism(ring, LEVEL_M[level])
    for paved in data.get("paved_surfaces", []):
        if len(paved["coords"]) >= 3:
            surfaces[paved.get("surface", "asphalt")].add_prism(paved["coords"], ROAD_M)
    for batch in surfaces.values():
        batch.build(uv_tile_m=UV_TILE_M)
    # A kerb stands on the road's top by its `kerb:height` (wiki Key:kerb:height); a flush one a
    # clearance proud, not to be coplanar.
    kerbs = MeshBatch("kerbs", make_material("Kerb", (0.62, 0.61, 0.58), 0.8))
    for kerb in data.get("kerbs", []):
        for ring in polyline_rings(kerb["coords"], KERB_WIDTH_M):
            kerbs.add_prism(ring, ROAD_M + max(kerb["height_m"], CLEARANCE_M))
    kerbs.build()
    tactile = MeshBatch("tactile_paving", make_material("TactilePaving", (0.38, 0.06, 0.05), 0.9))
    for ring in data.get("tactile_paving_polygons", []):
        tactile.add_prism(ring, ROAD_M + CLEARANCE_M)
    tactile.build()
    phase("surfaces")

    # Paint, by the world's table: a line along each polyline, a stroke between a line's ends, a
    # fill under the stripes, a mark on them.
    paint = {name: MeshBatch(f"paint_{name}", mat) for name, mat in paints.items()}
    for key, (kind, colour, width) in data["paint"].items():
        batch = paint[colour]
        for item in data.get(key, []):
            if kind == "line":
                for ring in polyline_rings(item, si(width)):
                    batch.add_prism(ring, PAINT_M, PAINT_Z)
            elif kind == "stroke" and (ring := line_ring(item[0], item[-1], si(width))) is not None:
                batch.add_prism(ring, PAINT_M, PAINT_Z)
            elif kind == "fill":
                batch.add_prism(item, CLEARANCE_M / 2, PAINT_Z - CLEARANCE_M)
            elif kind == "mark":
                batch.add_prism(item, CLEARANCE_M / 2, PAINT_Z)
    for crossing in data.get("surveyed_crossings", []):
        for ring in crossing.get("bars", []):
            paint["white"].add_prism(ring, CLEARANCE_M / 2, PAINT_Z)
        for line in crossing.get("lines", []):
            for ring in polyline_rings(line, si("crosswalk.line_width")):
                paint["white"].add_prism(ring, PAINT_M, PAINT_Z)
    for batch in paint.values():
        batch.build()
    phase("paint")

    pole_mat = make_material("Pole", SIGN_POST_GRAY, 0.5)
    housing_mat = make_material("SignalHousing", SIGNAL_HOUSING_DARK, 0.4)
    ped_housing_mat = make_material("PedSignalHousing", PED_SIGNAL_HOUSING_DARK, 0.4)
    for i, prop in enumerate(data.get("props", [])):
        add_prop(f"{prop['type']}_{i}", prop, pole_mat, housing_mat, ped_housing_mat)
    if data.get("tree_points"):
        tree = build_tree_proxy(make_material("TreeTrunk", (0.32, 0.22, 0.15), 0.9),
                                make_material("TreeFoliage", (0.16, 0.4, 0.14), 0.85))
        add_tree_instances("street_trees", data["tree_points"], tree)
    phase("props + trees")
    tallest = max((b["height_m"] for b in data.get("buildings", [])), default=0.0)
    return rect, tallest


# The camera: due south of its centre, `CAMERA_DIST` radii out and `CAMERA_HEIGHT` up.
CAMERA_DIST, CAMERA_HEIGHT, CAMERA_LENS_MM, CAMERA_SENSOR_MM = 1.6, 2.3, 32.0, 36.0
CLIP_FAR_MARGIN, CLIP_NEAR_MARGIN, CLIP_NEAR_FLOOR_M = 1.05, 0.9, 1.0
RESOLUTION = (1920, 1440)


def clip_range(cam, eye_z: float, ground_rect, tallest_m: float) -> tuple[float, float]:
    """(clip_start, clip_end), depths along the view axis from where the frustum's corner rays
    meet the ground - the depth buffer's precision is far/near, and a thin crosswalk line tears at
    Blender's default 0.1-1000 m. The far end is capped by the ground; the near end lowered by the
    tallest building's share of the camera's height."""
    rot = cam.rotation_euler.to_matrix()
    fwd = rot @ mathutils.Vector((0.0, 0.0, -1.0))
    half_w = CAMERA_SENSOR_MM / 2
    depths = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            ray = rot @ mathutils.Vector((sx * half_w, sy * half_w * 3 / 4, -CAMERA_LENS_MM))
            if ray.z >= -1e-6:
                depths = []
                break
            depths.append((ray * (eye_z / -ray.z)).dot(fwd))
        else:
            continue
        break
    rect_far = max((mathutils.Vector((x, y, 0.0)) - cam.location).dot(fwd)
                   for x in (ground_rect[0], ground_rect[2]) for y in (ground_rect[1], ground_rect[3]))
    if not depths:
        return CLIP_NEAR_FLOOR_M, max(rect_far, 10.0)
    far = min(max(depths), rect_far) * CLIP_FAR_MARGIN
    low = max(0.0, 1.0 - tallest_m / eye_z) if eye_z > 0 else 0.0
    near = max(min(depths) * low * CLIP_NEAR_MARGIN, CLIP_NEAR_FLOOR_M)
    return near, max(far, near * 2)


def render_cameras(job) -> None:
    """--open: one PNG per camera, in a world built once. A camera's ground is grown, in memory,
    to cover its own frame four radii across, so its horizon never shows the plane's edge."""
    cameras = json.loads(job.cameras.read_text())
    bpy.ops.wm.open_mainfile(filepath=str(job.blend))
    meta = bpy.context.scene["nj_world"]
    ground = bpy.data.objects["Ground"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    bpy.context.preferences.edit.use_global_undo = False
    job.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in cameras:
        (cx, cy), radius = spec["center_m"], float(spec["radius_m"])
        world = meta["ground_rect"]
        rect = (min(world[0], cx - 2 * radius), min(world[1], cy - 2 * radius),
                max(world[2], cx + 2 * radius), max(world[3], cy + 2 * radius))
        ground.location = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, ground.location.z)
        ground.scale = (rect[2] - rect[0], rect[3] - rect[1], 1.0)
        started = time.perf_counter()
        eye_z = radius * CAMERA_HEIGHT
        bpy.ops.object.camera_add(location=(cx, cy - radius * CAMERA_DIST, eye_z))
        cam = bpy.context.active_object
        scene.camera = cam
        cam.rotation_euler = (mathutils.Vector((cx, cy, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = CAMERA_LENS_MM
        cam.data.clip_start, cam.data.clip_end = clip_range(cam, eye_z, rect, float(meta["tallest_m"]))
        bpy.ops.object.light_add(type="SUN", location=(cx + radius * 0.3, cy - radius * 0.3, eye_z))
        sun = bpy.context.active_object
        sun.data.energy, sun.data.angle, sun.rotation_euler = 2.2, 0.2, (0.85, 0.15, 0.75)
        out = job.out_dir / f"{spec['name']}.png"
        scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        print(f"RENDER_DONE: {out} ({time.perf_counter() - started:.1f}s)")
        for obj, blocks in ((cam, bpy.data.cameras), (sun, bpy.data.lights)):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            blocks.remove(data)


def build_world(job) -> None:
    """--build: the whole scene from one world JSON, no camera, its images packed, saved."""
    started = time.perf_counter()
    data = json.loads(job.geometry.read_text())
    bpy.context.preferences.edit.use_global_undo = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    rect, tallest = build_scene(data)
    bpy.context.scene.world.use_nodes = True
    sky = bpy.context.scene.world.node_tree.nodes.get("Background")
    if sky:
        sky.inputs["Color"].default_value = (0.55, 0.68, 0.82, 1.0)
        sky.inputs["Strength"].default_value = 0.6
    bpy.context.scene["nj_world"] = {"ground_rect": list(rect), "tallest_m": tallest}
    for image in bpy.data.images:          # so the .blend does not depend on output/.textures
        if image.source == "FILE" and not image.packed_file:
            try:
                image.pack()
            except RuntimeError as e:
                print(f"  WARNING: could not pack {image.filepath!r} ({e})")
    job.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(job.blend), compress=True)
    print(f"WORLD_SAVED: {job.blend} ({len(bpy.data.objects)} objects, {time.perf_counter() - started:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="blender_scene.py")
    parser.add_argument("--build", type=Path, dest="geometry")
    parser.add_argument("--save", type=Path, dest="blend")
    parser.add_argument("--open", type=Path)
    parser.add_argument("--camera", type=Path, dest="cameras")
    parser.add_argument("--out-dir", type=Path)
    job = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if job.geometry and job.blend:
        build_world(job)
    elif job.open and job.cameras and job.out_dir:
        job.blend = job.open
        render_cameras(job)
    else:
        raise SystemExit("give --build <world.json> --save <w.blend>, or --open <w.blend> "
                         "--camera <cameras.json> --out-dir <dir>")


if __name__ == "__main__":
    main()
