"""The textures a world is built with, fetched (src/render/assets.py) in the project's own Python,
since Blender's has no network: {material: {map: path}} for every ground material
scripts/blender/blender_scene.py draws (src/osm_world.py:SURFACES), each path repo-relative so a
world JSON names no one machine's files, and None where an asset is unavailable - Blender then
falls back to a flat colour."""
from functools import lru_cache
from pathlib import Path

from src.render.assets import REPO_ROOT, fetch_polyhaven_texture

# Poly Haven slugs (CC0), each with Diffuse / Rough / nor_gl maps.
SLUGS = {"asphalt": "asphalt_01", "concrete": "pavement_02", "gravel": "gravel_road", "dirt": "park_dirt"}
RESOLUTION = "2k"   # a texture tiles every 2 m, so 2k is 1 mm a texel: past what a render resolves


@lru_cache(maxsize=1)
def build_default_theme() -> dict:
    out: dict[str, dict[str, str] | None] = {}
    for material, slug in SLUGS.items():
        paths = fetch_polyhaven_texture(slug, resolution=RESOLUTION)
        out[material] = None if paths is None else {
            k: Path(v).resolve().relative_to(REPO_ROOT).as_posix() for k, v in paths.items()}
    return out
