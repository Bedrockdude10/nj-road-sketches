"""Primitive meshes WITHOUT `bpy.ops.mesh.primitive_*_add`. Imported by blender_props.py - runs under
Blender's bundled Python.

WHY THIS EXISTS. Every `bpy.ops.mesh.primitive_*_add` call does work proportional to the number of
objects ALREADY in the scene - it deselects all of them and re-syncs the view layer. Measured on
this Blender (4.3), adding a unit cube: 3.7 ms each at 500 objects, 15.6 ms at 1,000, 32 ms at
1,500, 51 ms at 2,000 - and 0.08-0.11 ms flat for the same cube built from bmesh and linked
directly, at every size up to 4,000. A prop is three to eight primitives, so building a borough's
streetlights, signals, hydrants and tactile pads through the operator layer is quadratic: four
junctions' worth of props (100 props, 2,400 objects) took 41 s, which is 98% of that scene's
build, and a whole borough extrapolates to hours. One junction never saw it (blender_geometry.py
already learned the same lesson for markings and slabs).

These take the operator's own keyword names so a call site changes by one line, and build the same
mesh: the bmesh operators are what the `bpy.ops` primitives call underneath. Each returns the new
object, linked into the current collection, with no selection or active-object side effect - the
operators' only OTHER effect, which nothing here relied on.
"""
import bpy


def _new_bmesh():
    """(bmesh module, a fresh BMesh). `bmesh` is imported HERE and not at module level because
    tests/test_paint.py imports blender_props outside Blender with `bpy` stubbed, and there is no
    `bmesh` to stub - the import has to wait until something is actually built."""
    import bmesh

    return bmesh, bmesh.new()


def _object_from(bm, name: str, location, rotation):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    bpy.context.collection.objects.link(obj)
    return obj


def add_cylinder(radius: float = 1.0, depth: float = 2.0, vertices: int = 32,
                 location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """`primitive_cylinder_add`: an n-gon prism along Z, centred, both ends capped."""
    bmesh, bm = _new_bmesh()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=vertices,
                          radius1=radius, radius2=radius, depth=depth)
    return _object_from(bm, "Cylinder", location, rotation)


def add_cone(radius1: float = 1.0, radius2: float = 0.0, depth: float = 2.0, vertices: int = 32,
             location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """`primitive_cone_add`."""
    bmesh, bm = _new_bmesh()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=vertices,
                          radius1=radius1, radius2=radius2, depth=depth)
    return _object_from(bm, "Cone", location, rotation)


def add_cube(size: float = 2.0, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """`primitive_cube_add`: an axis-aligned cube of edge `size`, centred."""
    bmesh, bm = _new_bmesh()
    bmesh.ops.create_cube(bm, size=size)
    return _object_from(bm, "Cube", location, rotation)


def add_uv_sphere(radius: float = 1.0, segments: int = 32, ring_count: int = 16,
                  location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """`primitive_uv_sphere_add`."""
    bmesh, bm = _new_bmesh()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=ring_count, radius=radius)
    return _object_from(bm, "Sphere", location, rotation)
