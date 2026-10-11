"""OSM, read once per area: the one way into OpenStreetMap for this project.

`osm_layers(area, change)` is every layer the pipeline reads - kerbs, crossings, roads,
buildings, signals, furniture - over the WHOLE downloaded area, with an osmChange applied. There is no centre and no radius: what exists in a drawing is never a
question about where its camera stands. Each area (areas.yaml) is one download, cached."""
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import requests
import yaml

from src.sources.osm_change import OsmChange, apply_change
from src.sources.overpass import USER_AGENT, OfflineCacheMiss, query_overpass
from src.standards import si

# Where fetched OSM responses are cached. Overridable so the test suite can point at a
# committed fixture set and run hermetically (ROAD_SKETCHES_OFFLINE).
CACHE_DIR = Path(os.environ.get(
    "ROAD_SKETCHES_OSM_CACHE", Path(__file__).resolve().parents[2] / "output" / ".cache"))
REFRESH_ENV = "ROAD_SKETCHES_REFRESH_OSM"
OSM_API_MAP = "https://api.openstreetmap.org/api/0.6/map.json"
# The OSM API refuses a /map call bigger than this, and a bbox typed with a sign or a digit
# wrong is usually enormous - so the check that catches the typo is the API's own limit.
MAX_SNAPSHOT_SQ_DEG = 0.25
AREAS_FILE = Path(__file__).resolve().parents[2] / "areas.yaml"

_SNAPSHOTS: dict[str, dict] = {}                   # cache file -> its parsed snapshot
_LAYERS: dict[str, tuple[dict, dict[str, list]]] = {}  # area -> (the snapshot read, its layers)
_REFRESHED: set[str] = set()           # cache files this process re-pulled, which it need not again


def refresh_requested() -> bool:
    """Whether this process was told to re-pull from the OSM API rather than read the cache - never
    while ROAD_SKETCHES_OFFLINE is set, which the test suite runs under."""
    return bool(os.environ.get(REFRESH_ENV)) and not os.environ.get("ROAD_SKETCHES_OFFLINE")


def _load_areas(path: Path = AREAS_FILE) -> dict[str, dict]:
    """{area: {"bbox": (west, south, east, north), "cameras": {name: node}}}, validated: a reversed
    or oversized bbox fails at download, hours of tracing later, as "nothing mapped here"."""
    areas = {}
    for name, entry in (yaml.safe_load(path.read_text()) or {}).items():
        west, south, east, north = (float(v) for v in entry["bbox"])
        if not (east > west and north > south):
            raise ValueError(f"{path.name}: {name}'s bbox does not run west->east and south->north")
        if (east - west) * (north - south) > MAX_SNAPSHOT_SQ_DEG:
            raise ValueError(f"{path.name}: {name} is over the OSM API's {MAX_SNAPSHOT_SQ_DEG} sq deg "
                             "limit for one /map call: split it")
        areas[name] = {"bbox": (west, south, east, north), "cameras": dict(entry.get("cameras") or {})}
    return areas


AREAS = _load_areas()
SNAPSHOT_AREAS: dict[str, tuple[float, float, float, float]] = {k: v["bbox"] for k, v in AREAS.items()}


def _snapshot_path(bbox: tuple) -> Path:
    key = hashlib.sha1(f"borough,v1,{tuple(bbox)}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"borough_{key}.json"


def _download_snapshot(bbox: tuple) -> list[dict]:
    """One whole area from the OSM API - the live database, so a kerb traced a minute ago is
    there - falling back to Overpass."""
    if os.environ.get("ROAD_SKETCHES_OFFLINE"):
        raise OfflineCacheMiss(
            "ROAD_SKETCHES_OFFLINE is set and the snapshot for this area is not in the fixture "
            "cache. Refresh it with: cp output/.cache/borough_*.json tests/fixtures/osm_cache/")
    west, south, east, north = bbox
    try:
        resp = requests.get(f"{OSM_API_MAP}?bbox={west},{south},{east},{north}",
                            headers={"User-Agent": USER_AGENT}, timeout=(5, 120))
        resp.raise_for_status()
        return resp.json()["elements"]
    except requests.exceptions.RequestException as e:
        print(f"  OSM API unavailable ({type(e).__name__}); falling back to Overpass")
        # The same shape /map.json returns, so nothing downstream knows which answered.
        return query_overpass(f"""
        [out:json][timeout:180];
        ( node({south},{west},{north},{east});
          way({south},{west},{north},{east}); );
        out body;
        """)["elements"]


def fetch_borough_osm(bbox: tuple) -> dict:
    """{"nodes": {id: element}, "ways": [element], "relations": [element]} for one area. Raises if
    any way references a node that is not present: a gap is a truncated download."""
    cache_path = _snapshot_path(bbox)
    key = str(cache_path)
    if not cache_path.exists() or (refresh_requested() and key not in _REFRESHED):
        raw = _download_snapshot(bbox)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw))
        _REFRESHED.add(key)
        _SNAPSHOTS.pop(key, None)
    if key not in _SNAPSHOTS:
        raw = json.loads(cache_path.read_text())
        nodes = {el["id"]: el for el in raw if el["type"] == "node"}
        ways = [el for el in raw if el["type"] == "way"]
        if any(nid not in nodes for way in ways for nid in way.get("nodes", [])):
            raise RuntimeError(f"a way's nodes are missing from {cache_path}: the download is "
                               "truncated - delete it and re-pull")
        _SNAPSHOTS[key] = {"nodes": nodes, "ways": ways,
                           "relations": [el for el in raw if el["type"] == "relation"]}
    return _SNAPSHOTS[key]


# WHICH OSM ELEMENTS A RENDER READS, as predicates on tags.
def is_building(tags: dict) -> bool:
    """`building=no` is OSM saying a way is NOT a building - West Broad St carries it."""
    return tags.get("building", "no") != "no"


def is_crossing_way(tags: dict) -> bool:
    return tags.get("footway") == "crossing"


def is_sidewalk(tags: dict) -> bool:
    return tags.get("footway") == "sidewalk"


def is_driveway(tags: dict) -> bool:
    return tags.get("highway") == "service" and tags.get("service") == "driveway"


def is_parking_aisle(tags: dict) -> bool:
    return tags.get("highway") == "service" and tags.get("service") == "parking_aisle"


def is_parking_lot(tags: dict) -> bool:
    return tags.get("amenity") == "parking"


def is_stop_line(tags: dict) -> bool:
    return tags.get("road_marking") == "stop_line"


def is_restriction_marking(tags: dict) -> bool:
    """A hatched area on the carriageway - wiki Key:road_marking, `restriction` ("Neutral areas or
    restriction markings, like gore chevron or no-parking markings"), or wiki
    Tag:area:highway=prohibited - mapped as a closed way that IS the painted area."""
    return tags.get("road_marking") == "restriction" or tags.get("area:highway") == "prohibited"


def is_road_area(tags: dict) -> bool:
    """`area:highway=*` - a road's surface mapped as its outline - but not `prohibited`, which is
    hatching on it (is_restriction_marking)."""
    return "area:highway" in tags and tags["area:highway"] != "prohibited"


# `highway=*` values that are always an area, never a line (wiki Key:highway, "areas").
AREA_HIGHWAYS = frozenset({"services", "rest_area"})


def is_highway_area(tags: dict) -> bool:
    """A `highway=*` mapped as a surface - `services` / `rest_area`, or any with `area=yes`."""
    return tags.get("highway") in AREA_HIGHWAYS or ("highway" in tags and tags.get("area") == "yes")


def is_road(tags: dict) -> bool:
    return "highway" in tags and not is_highway_area(tags)


def is_traffic_control(tags: dict) -> bool:
    """`highway=traffic_signals` / `stop` / `give_way` / `crossing` - a crossing node is where OSM
    records what a crossing way does not (tactile_paving, button_operated, crossing:kerb_extension)."""
    return tags.get("highway") in ("traffic_signals", "stop", "give_way", "crossing")


def is_street_furniture(tags: dict) -> bool:
    return (tags.get("highway") == "street_lamp" or tags.get("emergency") == "fire_hydrant"
            or tags.get("natural") == "tree")


def is_bollard(tags: dict) -> bool:
    """`barrier=bollard` - a post on a node, a row of them on a way (wiki Tag:barrier=bollard)."""
    return tags.get("barrier") == "bollard"


def is_kerb(tags: dict) -> bool:
    return tags.get("barrier") == "kerb"


# THE ONE TABLE of which layer an OSM element lands in: (layer, predicate, least coords) for
# ways, (layer, predicate) for nodes.
WAY_LAYERS: tuple[tuple[str, Callable[[dict], bool], int], ...] = (
    ("buildings", is_building, 3), ("crossings", is_crossing_way, 2),
    ("sidewalks", is_sidewalk, 2), ("driveways", is_driveway, 2),
    ("parking_aisles", is_parking_aisle, 2), ("parking_lots", is_parking_lot, 4),
    ("kerbs", is_kerb, 2), ("roads", is_road, 2), ("stop_lines", is_stop_line, 2),
    ("road_markings", is_restriction_marking, 4), ("road_areas", is_road_area, 4),
    ("highway_areas", is_highway_area, 4), ("bollards", is_bollard, 2))
NODE_LAYERS: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    ("traffic_control", is_traffic_control), ("street_furniture", is_street_furniture),
    ("kerbs", is_kerb), ("bollards", is_bollard))
# Layers that are paint on the carriageway, not something built.
PAINT_LAYERS = ("stop_lines", "road_markings")


def height_from_tags(tags: dict) -> float | None:
    """A building's height in metres as its mapper recorded it - `height`, else `building:levels`
    at building.level_height a level - else None, which is not the default height."""
    if tags.get("height"):
        try:
            return float("".join(c for c in tags["height"] if c.isdigit() or c == "."))
        except ValueError:
            pass
    if tags.get("building:levels"):
        try:
            return float(tags["building:levels"]) * si("building.level_height")
        except ValueError:
            pass
    return None


class UnknownAreaError(KeyError):
    """An area name that areas.yaml does not declare."""


def osm_layers(area: str, change: OsmChange | None = None) -> dict[str, list]:
    """Every OSM layer this project reads over the whole area, with `change` (an .osc) applied to
    OSM before it is sorted into layers - so a proposed street is read by exactly what reads OSM's
    own."""
    if area not in SNAPSHOT_AREAS:
        raise UnknownAreaError(f"no OSM area named {area!r} - declared: {', '.join(SNAPSHOT_AREAS)} "
                               f"({AREAS_FILE.name})")
    raw = fetch_borough_osm(SNAPSHOT_AREAS[area])
    cached = _LAYERS.get(area)
    if change is None and cached is not None and cached[0] is raw:
        return cached[1]
    layers = layers_of(apply_change(raw, change) if change is not None else raw)
    if change is None:
        _LAYERS[area] = (raw, layers)
    return layers


def layers_of(snapshot: dict) -> dict[str, list]:
    """A snapshot (fetch_borough_osm's shape) sorted into layers, by WAY_LAYERS and NODE_LAYERS."""
    nodes = snapshot["nodes"]
    ways = [(way, [(nodes[nid]["lon"], nodes[nid]["lat"]) for nid in way.get("nodes", [])])
            for way in snapshot["ways"]]
    layers: dict[str, list] = {
        name: [{"coords_wgs84": coords, "tags": way.get("tags") or {}, "id": way["id"],
                "node_ids": way.get("nodes", [])}
               for way, coords in ways if predicate(way.get("tags") or {}) and len(coords) >= least]
        for name, predicate, least in WAY_LAYERS}
    for name, predicate in NODE_LAYERS:
        # A kerb or bollard is mapped as a way or as a node, so its layer holds both; a node has
        # no coords.
        found = [{"lon": n["lon"], "lat": n["lat"], "tags": n.get("tags") or {}, "id": n["id"]}
                 for n in nodes.values() if predicate(n.get("tags") or {})]
        layers[name] = ([*layers[name], *({"coords_wgs84": None, **node} for node in found)]
                        if name in layers else found)
    for building in layers["buildings"]:
        building["height_m"] = height_from_tags(building["tags"])
    return layers
