"""Fetch and disk-cache CC0 textures from Poly Haven's public API (api.polyhaven.com), under this
project's User-Agent (Poly Haven's ToS asks for a unique one). Returns None on failure rather than
raising: a missing texture never fails a render, which falls back to a flat colour."""
from pathlib import Path

import requests

from src.sources.overpass import USER_AGENT

POLYHAVEN_API = "https://api.polyhaven.com"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # src/render/assets.py -> repo root
CACHE_DIR = REPO_ROOT / "output" / ".textures"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 30


def _get_json(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


# One manifest per slug, per process: /files/<slug> covers every resolution of an asset, but
# this module asks for one resolution at a time, so without this the same manifest is refetched
# per resolution.
_manifests: dict[str, dict | None] = {}


def _manifest(slug: str) -> dict | None:
    if slug not in _manifests:
        _manifests[slug] = _get_json(f"{POLYHAVEN_API}/files/{slug}")
    return _manifests[slug]


def _texture_dest(slug: str, resolution: str, map_name: str) -> Path:
    """Where a texture map lands on disk. Fully determined by (slug, resolution, map) - no
    manifest needed - which is what lets a warm cache skip the network entirely."""
    return CACHE_DIR / slug / resolution / f"{slug}_{map_name}_{resolution}.jpg"


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except requests.exceptions.RequestException:
        return False


def fetch_polyhaven_texture(slug: str, resolution: str = "2k",
                             maps: tuple[str, ...] = ("Diffuse", "Rough", "nor_gl")) -> dict[str, Path] | None:
    """Download (or reuse cached) Diffuse/Roughness/Normal jpgs for a Poly Haven
    texture at the given resolution. Returns {"Diffuse": path, "Rough": path,
    "nor_gl": path} or None if the asset/network is unavailable.

    Checks the disk BEFORE the network, so a warm cache makes no HTTPS request at all and 3D
    rendering works offline.
    """
    dests = {map_name: _texture_dest(slug, resolution, map_name) for map_name in maps}
    if all(dest.exists() for dest in dests.values()):
        return dests

    manifest = _manifest(slug)
    if manifest is None:
        return None

    out = {}
    for map_name, dest in dests.items():
        try:
            file_info = manifest[map_name][resolution]["jpg"]
        except KeyError:
            return None
        if not _download(file_info["url"], dest):
            return None
        out[map_name] = dest
    return out
