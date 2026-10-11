"""Meshes built as arithmetic and handed to Blender once: a prism off a ring is closed-form, and a
planar UV is x / tile, y / tile. Every `bpy.ops` call pushes an undo snapshot and costs time in
proportion to the scene, so nothing here uses one; and a scene's cost scales with its OBJECT count,
so `MeshBatch` builds one mesh per material. Imported by blender_scene.py, under Blender's Python."""
import math

import bpy


def prism(coords_2d, height: float, z_base: float = 0.0, first_index: int = 0):
    """(vertices, faces) for a closed 2D ring extruded to `height`, starting at vertex index n.

    `first_index` is what makes this batchable: the faces come back numbered from that offset, so a
    caller accumulating many prisms into one mesh passes the running vertex count and concatenates.

    A zero `height` gives the flat cap alone, which is what almost every marking is - paint has no
    thickness worth modelling, and the pieces that do (kerbs, buildings) pass a real height.

    Winding is preserved from the ring, and the top cap is reversed so both caps face outward. That
    matters under flat shading with backface culling off: a cap wound the wrong way reads as a hole.
    """
    pts = list(coords_2d)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return [], []
    n = len(pts)
    if height <= 0:
        verts = [(x, y, z_base) for x, y in pts]
        return verts, [tuple(range(first_index, first_index + n))]
    verts = [(x, y, z_base) for x, y in pts] + [(x, y, z_base + height) for x, y in pts]
    bottom = tuple(range(first_index, first_index + n))
    top = tuple(reversed(range(first_index + n, first_index + 2 * n)))
    walls = [(first_index + i, first_index + (i + 1) % n,
              first_index + n + (i + 1) % n, first_index + n + i) for i in range(n)]
    return verts, [bottom, top, *walls]


class MeshBatch:
    """Many pieces of one material, accumulated in Python and built as a single mesh.

    THE POINT IS THE OBJECT COUNT, not the vertex count. A scene's cost in Blender scales with how
    many OBJECTS it holds - depsgraph nodes, draw calls, and per-object overhead in EEVEE - far more
    than with how many triangles those objects contain. A wide render's ~900 markings are perhaps
    12,000 triangles between them, which is nothing; as 900 separate objects they are the whole cost
    of the render.

    Nothing is lost by merging. These pieces share a material, never move independently, and are
    never selected by name - they exist to be looked at. The one thing a caller gives up is a
    per-piece object name in the outliner, which no part of this pipeline reads.
    """

    def __init__(self, name: str, material):
        self.name = name
        self.material = material
        self.verts = []
        self.faces = []

    def add_prism(self, coords_2d, height: float, z_base: float = 0.0) -> None:
        verts, faces = prism(coords_2d, height, z_base, first_index=len(self.verts))
        self.verts += verts
        self.faces += faces

    def build(self, uv_tile_m: float | None = None):
        """One object for everything added, or None if nothing was. UVs only where asked for."""
        if not self.faces:
            return None
        mesh = bpy.data.meshes.new(self.name)
        mesh.from_pydata(self.verts, [], self.faces)
        mesh.update()
        if not mesh.polygons:
            return None
        if uv_tile_m:
            planar_uvs(mesh, uv_tile_m)
        obj = bpy.data.objects.new(self.name, mesh)
        if self.material is not None:
            obj.data.materials.append(self.material)
        bpy.context.collection.objects.link(obj)
        return obj


def planar_uvs(mesh, tile_size_m: float = 2.0) -> None:
    """Project UVs from above at a fixed real-world tile size, computed rather than operated.

    What `bpy.ops.uv.cube_project` was being used for, and all it was doing for a top-down scene:
    every face here is either horizontal or a short vertical wall, so the projection that keeps a
    tiled texture at a consistent physical scale is just the world x,y divided by the tile size. A
    2 ft pavement tile stays 2 ft whether it is under a 500 ft slab or a 3 ft sidewalk wedge.

    Written straight into the loop layer with foreach_set, so this is one buffer copy rather than an
    edit-mode round trip per object. CALL IT ONLY FOR TEXTURED MATERIALS - the old helper's own
    docstring admitted it was "harmless no-op-ish for flat-color materials", which meant a full mode
    switch per object to compute coordinates nothing would ever sample.
    """
    uv_layer = mesh.uv_layers.new(name="UVMap")
    coords = [0.0] * (len(mesh.loops) * 2)
    positions = mesh.vertices
    for i, loop in enumerate(mesh.loops):
        x, y, _z = positions[loop.vertex_index].co
        coords[2 * i] = x / tile_size_m
        coords[2 * i + 1] = y / tile_size_m
    uv_layer.data.foreach_set("uv", coords)


def line_ring(p1, p2, width_m: float):
    """The four corners of a stripe of `width_m` between two points, or None if they coincide."""
    (x1, y1), (x2, y2) = p1[:2], p2[:2]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy * width_m / 2, ux * width_m / 2
    return [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]


def polyline_rings(points, width_m: float):
    """One stripe ring per SEGMENT of a polyline: a mitred ribbon spikes where a line turns
    sharply, and butt joints overlap invisibly in flat paint."""
    rings = []
    for i in range(len(points) - 1):
        ring = line_ring(points[i], points[i + 1], width_m)
        if ring is not None:
            rings.append(ring)
    return rings
