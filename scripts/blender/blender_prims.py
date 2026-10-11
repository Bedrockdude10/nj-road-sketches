"""Primitive meshes from bmesh, not `bpy.ops.mesh.primitive_*_add`, whose cost grows with every
object already in the scene (3.7 ms a cube at 500 objects, 51 ms at 2,000, against 0.1 ms flat
this way). They take the operators' own keyword names. Imported by blender_props.py."""
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
