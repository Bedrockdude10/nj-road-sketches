"""The props a world carries (src/osm_world.py emits them): traffic signal poles with their
vehicle heads, pedestrian signal heads and push buttons, flex posts, and trees instanced from one
low-poly mesh. Procedural, at real colours and sizes - no CC0 model of any was found. Imported by
blender_scene.py, under Blender's own Python."""
import math

import bpy

import blender_materials
import blender_prims as prims
from blender_materials import make_retroreflective_material

# Colours and sizes: real ones where a standard or a product gives them, matched by eye otherwise.
SIGN_POST_GRAY = (0.35, 0.35, 0.37)
SIGNAL_HOUSING_DARK = (0.08, 0.08, 0.08)
PED_SIGNAL_HOUSING_DARK = (0.1, 0.1, 0.1)
VEHICLE_SIGNAL_LENS_COLORS = [(0.85, 0.05, 0.05), (0.85, 0.65, 0.05), (0.05, 0.55, 0.15)]  # top down
TRAFFIC_SIGNAL_POLE_HEIGHT_M, TRAFFIC_SIGNAL_POLE_RADIUS_M, MAST_ARM_RADIUS_M = 5.5, 0.1, 0.05
VEHICLE_SIGNAL_HEAD_WIDTH_M, VEHICLE_SIGNAL_HEAD_HEIGHT_M = 0.32, 0.85
PED_SIGNAL_MOUNT_HEIGHT_M, PED_SIGNAL_POST_RADIUS_M = 2.3, 0.05
PED_SIGNAL_HEAD_WIDTH_M, PED_SIGNAL_HEAD_HEIGHT_M = 0.28, 0.32
# A flex post: white (Danny, 2026-10-07), 42 in tall (a common daylighting height), banded with
# retroreflective tape as MUTCD bands a tubular marker - the top band near the top, gaps no wider
# than a band - and the bands proud of the post, as a wrapped sleeve is, so they read as rings.
BOLLARD_POST_WHITE, BOLLARD_REFLECTIVE_WHITE = (0.92, 0.92, 0.90), (0.96, 0.96, 0.94)
BOLLARD_HEIGHT_M, BOLLARD_RADIUS_M = 42 * 0.0254, 0.05
BOLLARD_BAND_HEIGHT_M, BOLLARD_BAND_GAP_M, BOLLARD_TOP_TO_FIRST_BAND_M = 3 * 0.0254, 4 * 0.0254, 2 * 0.0254
BOLLARD_BAND_COUNT, BOLLARD_BAND_RADIUS_SCALE = 3, 1.18
# An accessible push button, at its 42-48 in mounting height (MUTCD/PROWAG).
PUSHBUTTON_POST_HEIGHT_M, PUSHBUTTON_POST_RADIUS_M = 1.2, 0.04
PUSHBUTTON_HOUSING_YELLOW = (0.85, 0.72, 0.08)
PUSHBUTTON_HOUSING_DEPTH_M, PUSHBUTTON_HOUSING_WIDTH_M, PUSHBUTTON_HOUSING_HEIGHT_M = 0.06, 0.13, 0.2


def add_vehicle_signal_head(name: str, position: tuple, heading_deg: float, housing_mat):
    """Procedural 3-section vehicle signal head: a dark housing box with 3
    stacked red/yellow/green lenses on the face pointed at `heading_deg` -
    real MUTCD color/layout."""
    x, y, z = position
    face = math.radians(heading_deg)
    fx, fy = math.cos(face), math.sin(face)

    housing = prims.add_cube(size=1.0, location=(x, y, z))
    housing.name = f"{name}_housing"
    # square cross-section - housing orientation doesn't matter visually
    housing.scale = (VEHICLE_SIGNAL_HEAD_WIDTH_M, VEHICLE_SIGNAL_HEAD_WIDTH_M,
                     VEHICLE_SIGNAL_HEAD_HEIGHT_M)
    housing.data.materials.append(housing_mat)

    for i, color in enumerate(VEHICLE_SIGNAL_LENS_COLORS):
        lens_pos = (x + fx * 0.17, y + fy * 0.17, z + 0.24 - i * 0.24)
        lens = prims.add_cylinder(radius=0.09, depth=0.03, vertices=16, location=lens_pos)
        lens.name = f"{name}_lens_{i}"
        lens.rotation_euler = (math.radians(90), 0, face)  # same flat-disc-facing-heading trick as sign plates
        lens.data.materials.append(blender_materials.shared_material(f"{name}_lens_{i}_mat", color, roughness=0.3))
    return housing


def add_traffic_signal_pole(name: str, position: tuple, head_facing_deg: float, pole_mat, housing_mat,
                            arm_heading_deg: float, arm_length_m: float):
    """A signal pole (src/osm_world.py:_Reader.signals): with `arm_length_m` a mast arm out along
    `arm_heading_deg` - square to the leg it serves, so not opposite the head - and the vehicle
    head at its end facing `head_facing_deg`; with none (`support=pole`) the head on the pole."""
    x, y = position
    pole = prims.add_cylinder(
radius=TRAFFIC_SIGNAL_POLE_RADIUS_M, depth=TRAFFIC_SIGNAL_POLE_HEIGHT_M,
location=(x, y, TRAFFIC_SIGNAL_POLE_HEIGHT_M / 2)
)
    pole.name = f"{name}_pole"
    pole.data.materials.append(pole_mat)

    arm_z = TRAFFIC_SIGNAL_POLE_HEIGHT_M - 0.4
    if arm_length_m <= 0:
        # Pole-mounted (OSM support=pole): the head on the pole's face, no arm.
        face = math.radians(head_facing_deg)
        offset = TRAFFIC_SIGNAL_POLE_RADIUS_M + VEHICLE_SIGNAL_HEAD_WIDTH_M / 2
        add_vehicle_signal_head(f"{name}_head", (x + math.cos(face) * offset, y + math.sin(face) * offset,
                                                 arm_z - 0.2), head_facing_deg, housing_mat)
        return pole
    arm_dir = math.radians(arm_heading_deg)
    dx, dy = math.cos(arm_dir), math.sin(arm_dir)
    arm_center = (x + dx * arm_length_m / 2, y + dy * arm_length_m / 2, arm_z)
    arm = prims.add_cylinder(radius=MAST_ARM_RADIUS_M, depth=arm_length_m,
                              location=arm_center)
    arm.name = f"{name}_arm"
    arm.rotation_euler = (0, math.radians(90), arm_dir)  # lay the cylinder flat, then point it along arm_dir
    arm.data.materials.append(pole_mat)

    head_pos = (x + dx * arm_length_m, y + dy * arm_length_m, arm_z - 0.2)
    add_vehicle_signal_head(f"{name}_head", head_pos, head_facing_deg, housing_mat)
    return pole


def add_pedestrian_signal_head(name: str, position: tuple, heading_deg: float, own_post: bool,
                                housing_mat, post_mat):
    """A pedestrian signal head at its mounting height - on a short post of its own where
    `own_post`."""
    x, y = position
    if own_post:
        post = prims.add_cylinder(
  radius=PED_SIGNAL_POST_RADIUS_M, depth=PED_SIGNAL_MOUNT_HEIGHT_M,
  location=(x, y, PED_SIGNAL_MOUNT_HEIGHT_M / 2)
)
        post.name = f"{name}_post"
        post.data.materials.append(post_mat)

    head = prims.add_cube(size=1.0, location=(x, y, PED_SIGNAL_MOUNT_HEIGHT_M))
    head.name = f"{name}_head"
    head.scale = (PED_SIGNAL_HEAD_WIDTH_M, PED_SIGNAL_HEAD_WIDTH_M, PED_SIGNAL_HEAD_HEIGHT_M)
    head.rotation_euler = (0, 0, math.radians(heading_deg))
    head.data.materials.append(housing_mat)
    return head


def bollard_band_centres_m() -> list[float]:
    """Height of each hi-vis band's centre above the ground, top band first.

    Stepped down from the TOP of the post, not up from the ground: the top band's placement
    is the one the banding pattern actually specifies (close to the top, so the post reads
    when only its tip clears an obstruction), and a post of a different height should move
    the whole stack with its top rather than leave a gap up there.

    Bands that would fall below the ground are dropped, so shortening the post shortens the
    stack instead of burying tape in the asphalt.
    """
    pitch = BOLLARD_BAND_HEIGHT_M + BOLLARD_BAND_GAP_M
    first_centre = BOLLARD_HEIGHT_M - BOLLARD_TOP_TO_FIRST_BAND_M - BOLLARD_BAND_HEIGHT_M / 2
    centres = [first_centre - i * pitch for i in range(BOLLARD_BAND_COUNT)]
    return [z for z in centres if z - BOLLARD_BAND_HEIGHT_M / 2 > 0]


def add_bollard(name: str, position: tuple):
    """A plastic flex post: a white post banded with white retroreflective tape. The bands are
    emissive (make_retroreflective_material) because they are retroreflectors:
    a diffuse white ring in this scene's single-soft-sun lighting renders mid-grey and
    vanishes into the post, which is the opposite of what the object is for.
    """
    x, y = position
    post = prims.add_cylinder(radius=BOLLARD_RADIUS_M, depth=BOLLARD_HEIGHT_M,
                               location=(x, y, BOLLARD_HEIGHT_M / 2))
    post.name = f"{name}_post"
    post.data.materials.append(blender_materials.shared_material(f"{name}_post_mat", BOLLARD_POST_WHITE, roughness=0.5))

    band_mat = make_retroreflective_material(f"{name}_band_mat", BOLLARD_REFLECTIVE_WHITE)
    for i, band_z in enumerate(bollard_band_centres_m()):
        band = prims.add_cylinder(radius=BOLLARD_RADIUS_M * BOLLARD_BAND_RADIUS_SCALE,
                                   depth=BOLLARD_BAND_HEIGHT_M, location=(x, y, band_z))
        band.name = f"{name}_band_{i}"
        band.data.materials.append(band_mat)
    return post


def add_pedestrian_pushbutton(name: str, position: tuple, heading_deg: float, post_mat):
    """An accessible push button (`button_operated=yes`): a short post with a yellow housing
    facing the waiting pedestrian."""
    x, y = position
    post = prims.add_cylinder(radius=PUSHBUTTON_POST_RADIUS_M, depth=PUSHBUTTON_POST_HEIGHT_M,
                               location=(x, y, PUSHBUTTON_POST_HEIGHT_M / 2))
    post.name = f"{name}_post"
    post.data.materials.append(post_mat)

    face = math.radians(heading_deg)
    fx, fy = math.cos(face), math.sin(face)
    housing = prims.add_cube(size=1.0,
                              location=(x + fx * 0.05, y + fy * 0.05, PUSHBUTTON_POST_HEIGHT_M - 0.1))
    housing.name = f"{name}_housing"
    housing.scale = (PUSHBUTTON_HOUSING_DEPTH_M, PUSHBUTTON_HOUSING_WIDTH_M,
                     PUSHBUTTON_HOUSING_HEIGHT_M)
    housing.rotation_euler = (0, 0, face)
    housing.data.materials.append(blender_materials.shared_material(f"{name}_housing_mat", PUSHBUTTON_HOUSING_YELLOW, roughness=0.4))
    return post


def add_prop(name: str, prop: dict, pole_mat, signal_housing_mat, ped_signal_housing_mat):
    """One prop dict of the world, by its `type`."""
    pos, heading, ptype = prop["position_m"], prop["heading_deg"], prop["type"]
    if ptype == "traffic_signal_pole":
        add_traffic_signal_pole(name, pos, heading, pole_mat, signal_housing_mat,
                                arm_heading_deg=prop["arm_heading_deg"], arm_length_m=prop["arm_length_m"])
    elif ptype == "pedestrian_signal_head":
        add_pedestrian_signal_head(name, pos, heading, prop.get("own_post", False), ped_signal_housing_mat, pole_mat)
    elif ptype == "bollard":
        add_bollard(name, pos)
    elif ptype == "pedestrian_pushbutton":
        add_pedestrian_pushbutton(name, pos, heading, pole_mat)
    else:
        print(f"WARNING: no Blender builder for prop type {ptype!r} ({name}) - not drawn.")


def build_tree_proxy(trunk_mat, foliage_mat):
    """One low-poly tree (a cone on a cylinder), the template every `natural=tree` instances."""
    trunk = prims.add_cylinder(radius=0.15, depth=2.0, vertices=6, location=(0, 0, 1.0))
    trunk.name = "tree_trunk"
    trunk.data.materials.append(trunk_mat)

    foliage = prims.add_cone(radius1=1.3, depth=3.0, vertices=8, location=(0, 0, 3.3))
    foliage.name = "tree_foliage"
    foliage.data.materials.append(foliage_mat)

    bpy.ops.object.select_all(action="DESELECT")
    trunk.select_set(True)
    foliage.select_set(True)
    bpy.context.view_layer.objects.active = trunk
    bpy.ops.object.join()
    tree = bpy.context.view_layer.objects.active
    tree.name = "tree_proxy_template"
    tree.hide_render = True
    tree.hide_set(True)
    return tree


def add_tree_instances(name: str, points: list, tree_template):
    """Geometry-nodes point instancing: ONE tree mesh's data is shared across
    every point (Instance on Points), not copied per-tree - the actual
    performance requirement behind 'not individual mesh copies'."""
    if not points or tree_template is None:
        return None

    mesh = bpy.data.meshes.new(f"{name}_points")
    mesh.from_pydata([(x, y, 0.0) for x, y in points], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    node_group = bpy.data.node_groups.new(f"{name}_GN", "GeometryNodeTree")
    node_group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = node_group.nodes
    links = node_group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = tree_template

    links.new(group_input.outputs["Geometry"], instance_on_points.inputs["Points"])
    links.new(object_info.outputs["Geometry"], instance_on_points.inputs["Instance"])
    links.new(instance_on_points.outputs["Instances"], group_output.inputs["Geometry"])

    modifier = obj.modifiers.new(name=f"{name}_GN", type="NODES")
    modifier.node_group = node_group
    return obj
