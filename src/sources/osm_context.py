"""OSM, read once per area: the one way into OpenStreetMap for this project.

`osm_layers(area)` returns every layer the pipeline reads - kerbs, crossings, roads, buildings,
signals, furniture, boundaries - over the WHOLE downloaded area, with field observations merged.
There is no centre and no radius: what exists in a drawing is never a question about where its
camera stands. Snapshots are downloaded per sites/osm_areas.yaml area and cached."""
import hashlib
import json
import os
import time
from pathlib import Path

from shapely.geometry import Point, Polygon

import requests

from src.sources.data_loader import OVERPASS_USER_AGENT, query_overpass

DEFAULT_BUILDING_HEIGHT_M = 7.0  # ~2 stories, typical for small-borough Main St buildings
METERS_PER_LEVEL = 3.0
# Where fetched OSM responses are cached. Overridable so the test suite can point at a
# committed fixture set and run hermetically - see tests/conftest.py and ROAD_SKETCHES_OFFLINE.
CACHE_DIR = Path(os.environ.get(
    "ROAD_SKETCHES_OSM_CACHE",
    Path(__file__).resolve().parent.parent.parent / "output" / ".cache"))

REFRESH_ENV = "ROAD_SKETCHES_REFRESH_OSM"

# Second-level cache: the raw borough snapshot and its parsed form. A batch build asks for
# the same junction's kerbs ~27 times over; re-parsing the same JSON each time is pure waste.
_MEMO: dict[str, list] = {}

# Cache files this process fetched and wrote itself, and which a refresh therefore does not
# need to pull again.
_REFRESHED: set[str] = set()

# What this process actually got each OSM layer from: cache file -> mtime, or None if pulled
# fresh. The staleness report (cache_summary) describes these files, not an independently
# derived key that would drift the moment a query gains a "v3".
_CACHE_READS: dict[Path, float | None] = {}

_warned: set[str] = set()


def refresh_requested() -> bool:
    """True when this process was told to ignore the cache and re-pull from Overpass.

    Refusing while ROAD_SKETCHES_OFFLINE is set is not politeness: the test suite runs against
    the committed fixture cache, and honouring a stray refresh there would turn every fetch
    into an OfflineCacheMiss.
    """
    if not os.environ.get(REFRESH_ENV):
        return False
    if os.environ.get("ROAD_SKETCHES_OFFLINE"):
        _warn_once(f"{REFRESH_ENV} ignored: ROAD_SKETCHES_OFFLINE is set, so the cached responses "
                   f"in {CACHE_DIR} are all this process is allowed to see.")
        return False
    return True


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"  {message}")


# Every layer is a view over the one borough snapshot. The old per-layer disk cache
# built a (kind, centre, radius) filename per layer; existing files in output/.cache
# are simply unread; the committed fixtures the test suite needs are borough_*.json.


def _cache_hit(cache_path: Path) -> bool:
    """Whether `cache_path` may be served instead of going to Overpass, recording the read."""
    if not cache_path.exists():
        return False
    if refresh_requested() and str(cache_path) not in _REFRESHED:
        return False
    # setdefault, not assignment: once a layer has been re-pulled this run its entry is
    # None ("fresh"), and the subsequent memo-backed reads of the file we just wrote must
    # not overwrite that with an age of zero seconds.
    _CACHE_READS.setdefault(cache_path, cache_path.stat().st_mtime)
    return True


def _write_cache(cache_path: Path, data: list) -> None:
    """Persist a freshly fetched layer, keeping both cache layers in step."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    _MEMO[str(cache_path)] = data
    _REFRESHED.add(str(cache_path))
    _CACHE_READS[cache_path] = None


def _memoized(cache_path: Path, build):
    """Return the parsed response for `cache_path`, from memory if it's already been read."""
    key = str(cache_path)
    if key not in _MEMO:
        _MEMO[key] = build()
    return _MEMO[key]


def cache_summary() -> str:
    """One line saying how old the OSM data this process just used is.

    An age printed on every build turns a silent trap (traced kerb, re-ran, no change
    because the disk cache never expires) into a number you can look at.
    """
    if not _CACHE_READS:
        return "OSM cache: no OSM layers were read"
    ages = [time.time() - mtime for mtime in _CACHE_READS.values() if mtime is not None]
    fresh = sum(1 for mtime in _CACHE_READS.values() if mtime is None)
    if not ages:
        if refresh_requested():
            return f"OSM cache: re-pulled all {fresh} layer(s) from Overpass"
        return f"OSM cache: nothing was cached - pulled {fresh} layer(s) fresh from Overpass"
    line = f"OSM cache: {len(ages)} layer(s), oldest {_humanize_age(max(ages))} old"
    if fresh:
        return f"{line}; {fresh} pulled fresh from Overpass"
    return f"{line} (--refresh-osm to re-pull)"


def _humanize_age(seconds: float) -> str:
    for unit, size in (("day", 86400.0), ("hour", 3600.0), ("minute", 60.0)):
        if seconds >= size:
            count = round(seconds / size)
            return f"{count} {unit}" + ("s" if count != 1 else "")
    return "less than a minute"


# ---------------------------------------------------------------------------
# The borough snapshot: one request, every layer.
# ---------------------------------------------------------------------------
#
# All six layers are VIEWS over a single download of the whole borough (2.86 MB, 1.25 s from
# api.openstreetmap.org), not six bbox queries each. The main OSM API is the live database,
# not a replica, so a kerb traced a minute ago is there. Overpass stays as fallback.
OSM_API_MAP = "https://api.openstreetmap.org/api/0.6/map.json"

# ONE AREA PER TOWN, not one bbox. Each is downloaded and cached separately, and a site is
# served from whichever area fully contains its context window. A new area costs one download
# and disturbs nothing already cached (the cache key is a hash of the bbox, so re-keying
# would re-download every existing site and orphan committed fixtures).
#
# THE AREAS THEMSELVES ARE NOT DECLARED HERE. They are a list of towns - the same kind of fact
# as which junctions this project studies - so they live in sites/osm_areas.yaml and porting to
# a new municipality touches no module. Sites in NO area are refused loudly rather than
# silently returning nothing; see assert_within_snapshot.
SNAPSHOT_AREAS_FILE = Path(__file__).resolve().parents[2] / "sites" / "osm_areas.yaml"

# The OSM API refuses a /map call bigger than this, and a bbox typed with a sign or a digit
# wrong is usually enormous - so the check that catches the typo is the API's own limit.
MAX_SNAPSHOT_SQ_DEG = 0.25


def _load_snapshot_areas(path: Path = SNAPSHOT_AREAS_FILE) -> dict:
    """{town: (west, south, east, north)} from sites/osm_areas.yaml, validated on load.

    Validated because every way a bbox can be wrong reads downstream as "nothing mapped here":
    an inverted or degenerate rectangle contains no site (so every site is refused with a
    message about the site), and one over the API's size limit fails at download time, hours of
    tracing later. Both are cheap to catch at the only moment the tuple is read.
    """
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    areas = {}
    for name, bbox in raw.items():
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError(f"{path.name}: {name} must be [west, south, east, north], got {bbox!r}")
        west, south, east, north = (float(v) for v in bbox)
        if not (east > west and north > south):
            raise ValueError(
                f"{path.name}: {name} is not a rectangle running west->east and south->north "
                f"({west}, {south}, {east}, {north}). A reversed bbox contains no site, so every "
                f"site in this town would be refused as though it were somewhere else.")
        if (east - west) * (north - south) > MAX_SNAPSHOT_SQ_DEG:
            raise ValueError(
                f"{path.name}: {name} is {(east - west) * (north - south):.3f} sq deg, over the "
                f"OSM API's {MAX_SNAPSHOT_SQ_DEG} sq deg limit for one /map call - it would be "
                f"refused at download. Split the town into smaller areas.")
        areas[name] = (west, south, east, north)
    return areas


SNAPSHOT_AREAS: dict[str, tuple[float, float, float, float]] = _load_snapshot_areas()

# The Hopewell bbox, still reachable by name. Kept because the cache key is a hash of this exact
# tuple and the committed fixture is named after it - see test_hopewells_cache_key_is_unchanged.
BOROUGH_BBOX = SNAPSHOT_AREAS["hopewell_borough"]


def _snapshot_path(bbox: tuple | None = None) -> Path:
    # The key format is unchanged, so Hopewell's cache file and the committed fixture keep
    # the name they already have.
    bbox = BOROUGH_BBOX if bbox is None else bbox
    key = hashlib.sha1(f"borough,v1,{tuple(bbox)}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"borough_{key}.json"


def _download_snapshot(bbox: tuple | None = None) -> list[dict]:
    """One whole snapshot area from the OSM API, falling back to Overpass."""
    if os.environ.get("ROAD_SKETCHES_OFFLINE"):
        from src.sources.data_loader import OfflineCacheMiss
        raise OfflineCacheMiss(
            "ROAD_SKETCHES_OFFLINE is set and the snapshot for this area is not in the fixture "
            "cache. Refresh it with: cp output/.cache/borough_*.json tests/fixtures/osm_cache/")

    west, south, east, north = BOROUGH_BBOX if bbox is None else bbox
    try:
        resp = requests.get(f"{OSM_API_MAP}?bbox={west},{south},{east},{north}",
                            headers={"User-Agent": OVERPASS_USER_AGENT}, timeout=(5, 120))
        resp.raise_for_status()
        return resp.json()["elements"]
    except requests.exceptions.RequestException as e:
        print(f"  OSM API unavailable ({type(e).__name__}); falling back to Overpass")
        # `out meta` for ways + `>` to pull their nodes: the same shape /map.json returns,
        # so everything downstream is indifferent to which source answered.
        return query_overpass(f"""
        [out:json][timeout:180];
        ( node({south},{west},{north},{east});
          way({south},{west},{north},{east}); );
        out body;
        """)["elements"]


def fetch_borough_osm(use_cache: bool = True, bbox: tuple | None = None) -> dict:
    """{"nodes": {id: element}, "ways": [element], "relations": [element]} for one snapshot area.

    `bbox` selects the area; None means Hopewell Borough. Raises if any way references a
    node that is not present - a gap means a truncated download, and half a kerb is worse
    than no kerb.

    RELATIONS ARE KEPT, and the only one read so far is the municipal boundary. A boundary is
    the one fact in this project nobody can trace off the pavement: an admin_level=8 relation
    carries the tags and its member ways carry the geometry, so dropping relations threw away
    where the corridor ENDS while keeping every foot of street past it. See
    `fetch_municipal_boundary`. The download already contained them - /map.json returns
    relations whose members are in the bbox - so this costs nothing but a key.
    """
    cache_path = _snapshot_path(bbox)
    if use_cache and _cache_hit(cache_path):
        raw = _memoized(cache_path, lambda: json.loads(cache_path.read_text()))
    else:
        raw = _download_snapshot(bbox)
        if use_cache:
            _write_cache(cache_path, raw)

    key = f"parsed:{cache_path}"
    if key not in _MEMO:
        nodes = {el["id"]: el for el in raw if el["type"] == "node"}
        ways = [el for el in raw if el["type"] == "way"]
        relations = [el for el in raw if el["type"] == "relation"]
        dangling = sum(1 for w in ways for nid in w.get("nodes", []) if nid not in nodes)
        if dangling:
            raise RuntimeError(
                f"{dangling} way node reference(s) in the snapshot don't resolve - the "
                f"download is truncated. Delete {cache_path} and re-pull; do not build geometry "
                f"from it, the ways would come out with missing vertices.")
        _MEMO[key] = {"nodes": nodes, "ways": ways, "relations": relations}
    return _MEMO[key]


# One resolved layer per (layer, centre, radius). Each entry stores the snapshot it was built
# from and is only served while that is still the snapshot in hand - the identity test is what
# makes a re-pull reach the render without an invalidation call someone could forget.
_LAYER_VIEWS: dict[tuple, tuple] = {}


def _in_bbox(bbox, lon: float, lat: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def _way_coords(snapshot: dict, way: dict) -> list[tuple[float, float]]:
    nodes = snapshot["nodes"]
    return [(nodes[nid]["lon"], nodes[nid]["lat"]) for nid in way.get("nodes", [])]


# WHICH OSM ELEMENTS A RENDER READS, as predicates on tags rather than as query strings.
# There are two ways into the same data - `_ways_near`/`_nodes_near` at a radius for a junction,
# and a walk over the whole snapshot for an area (src/geometry/network/area.py) - and a junction
# that drew a driveway while the area document did not carry one would be these two lists
# disagreeing, not OSM changing. One definition, both readers.
#
# THE NODE PREDICATES WERE EXTRACTED AND THE WAY PREDICATES WERE NOT, and that asymmetry is what
# the area document was missing: `area_context` matched buildings, crossings and kerbs by hand and
# had no idea the other six layers existed, so a slice rendered 0 paved surfaces where the same
# junction as a site rendered 33. A layer a fetcher knows about and the document does not is a
# layer the network path cannot draw.
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
    """A hatched area on the carriageway - wiki Key:road_marking, `restriction`: "Neutral areas
    or restriction markings, like gore chevron or no-parking markings". Mapped as a closed way;
    the way IS the painted area, as a stop_line way is the painted bar."""
    return tags.get("road_marking") == "restriction"


def is_road(tags: dict) -> bool:
    return "highway" in tags


def is_traffic_control(tags: dict) -> bool:
    """highway=traffic_signals / stop / give_way / crossing.

    Crossing NODES are control because that is where OSM records the pedestrian-facing detail
    that lives on the node rather than the way (tactile_paving, button_operated, crossing:island).
    """
    return tags.get("highway") in ("traffic_signals", "stop", "give_way", "crossing")


def is_street_furniture(tags: dict) -> bool:
    """highway=street_lamp, emergency=fire_hydrant, natural=tree.

    STREET LAMPS ARE NOT MAPPED at any of this project's four sites. This exists so a place
    where they ARE mapped gets real pole positions rather than a derived one-per-corner
    placement, and so the absence is reported rather than papered over.
    """
    return (tags.get("highway") == "street_lamp" or tags.get("emergency") == "fire_hydrant"
            or tags.get("natural") == "tree")


def is_kerb(tags: dict) -> bool:
    """barrier=kerb - the most direct geometry this project can get: a traced kerb IS the curb."""
    return tags.get("barrier") == "kerb"


def height_from_tags(tags: dict) -> tuple[float, str] | None:
    """(height in metres, which tag said so) if a mapper recorded one, else None.

    None rather than DEFAULT_BUILDING_HEIGHT_M: "nobody said" is a different answer from
    "7 m", and the caller looks elsewhere (src/sources/assessor.py). Returning the default
    here made every building the same height.
    """
    if tags.get("height"):
        try:
            return float("".join(c for c in tags["height"] if c.isdigit() or c == ".")), "osm_height"
        except ValueError:
            pass
    if tags.get("building:levels"):
        try:
            return float(tags["building:levels"]) * METERS_PER_LEVEL, "osm_levels"
        except ValueError:
            pass
    return None


class UnknownAreaError(KeyError):
    """An area name that sites/osm_areas.yaml does not declare."""


#: Every layer `osm_layers` returns. The names are the ones consumers already know them by.
OSM_LAYERS = ("buildings", "crossings", "sidewalks", "driveways", "parking_aisles",
              "parking_lots", "traffic_control", "street_furniture", "kerbs", "roads",
              "stop_lines", "road_markings", "municipalities")

_AREA_LAYERS_MEMO: dict[str, tuple] = {}


def osm_layers(area: str) -> dict[str, list]:
    """Every OSM layer this project reads, over the WHOLE snapshot area - the one way into OSM.

    THERE IS NO CENTRE AND NO RADIUS. A drawing is a view onto a world, and what exists in that
    world is not a question about where the camera stands: a circle about a junction decided
    which driveways opened a kerb, which signals governed a node and which town a street was in,
    and each of those answers moved with the circle. A consumer that needs "the ones that
    belong to this leg / node / view" asks that geometric question of the whole layer.

    Field observations (observations/<area>.yaml) are merged here, so every consumer - a site,
    a slice, the borough - reads one set of tags.

    Same element shapes the per-layer readers have always returned, so a consumer swaps its
    source and nothing else. `municipalities` is [(name, ring)] for every admin_level=8 ring
    that closes inside the area; `municipality_containing` answers "which town is this in".
    """
    from src.sources.observations import apply_observations, load_observations

    try:
        bbox = SNAPSHOT_AREAS[area]
    except KeyError:
        raise UnknownAreaError(
            f"no OSM area named {area!r} - declared areas: {', '.join(SNAPSHOT_AREAS)}. "
            f"Add it to {SNAPSHOT_AREAS_FILE.parent.name}/{SNAPSHOT_AREAS_FILE.name}.") from None
    raw = fetch_borough_osm(bbox=bbox)
    cached = _AREA_LAYERS_MEMO.get(area)
    if cached is not None and cached[0] is raw:
        return cached[1]
    snapshot = apply_observations(raw, load_observations(area))
    ways = [(way, _way_coords(snapshot, way)) for way in snapshot["ways"]]
    nodes = list(snapshot["nodes"].values())

    def ways_where(predicate, min_coords: int, **extra):
        return [{"coords_wgs84": coords, "tags": way.get("tags") or {}, "id": way["id"],
                 "node_ids": way.get("nodes", []), **{k: f(way) for k, f in extra.items()}}
                for way, coords in ways
                if predicate(way.get("tags") or {}) and len(coords) >= min_coords]

    def nodes_where(predicate):
        return [{"lon": n["lon"], "lat": n["lat"], "tags": n.get("tags") or {}, "id": n["id"]}
                for n in nodes if predicate(n.get("tags") or {})]

    buildings = ways_where(is_building, 3)
    for building in buildings:
        recorded = height_from_tags(building["tags"])
        building["height_m"], building["height_source"] = recorded if recorded else (None, None)
    kerbs = ways_where(is_kerb, 2)
    kerbs += [{"coords_wgs84": None, **n} for n in nodes_where(is_kerb)]

    layers = {
        "buildings": buildings,
        "crossings": ways_where(is_crossing_way, 2),
        "sidewalks": ways_where(is_sidewalk, 2),
        "driveways": ways_where(is_driveway, 2),
        "parking_aisles": ways_where(is_parking_aisle, 2),
        "parking_lots": ways_where(is_parking_lot, 4),
        "traffic_control": nodes_where(is_traffic_control),
        "street_furniture": nodes_where(is_street_furniture),
        "kerbs": kerbs,
        "roads": ways_where(is_road, 2),
        "stop_lines": ways_where(is_stop_line, 2),
        "road_markings": ways_where(is_restriction_marking, 4),
        "municipalities": _closed_municipal_rings(snapshot),
    }
    _AREA_LAYERS_MEMO[area] = (raw, layers)
    return layers


def _closed_municipal_rings(snapshot: dict) -> list[tuple[str | None, list]]:
    """[(qualified name, ring)] for every admin_level=8 outer ring that closes in this snapshot.

    A RING THAT DOES NOT CLOSE IS NOT A MUNICIPALITY: a town bigger than the snapshot has its
    ring clipped to a straight edge down the bbox, which a leg can cross anywhere. Dropped
    rather than closed for it - see `municipality_containing`.
    """
    nodes, ways = snapshot["nodes"], {w["id"]: w for w in snapshot["ways"]}
    rings = []
    for relation in snapshot.get("relations", []):
        tags = relation.get("tags") or {}
        if tags.get("boundary") != "administrative" or tags.get("admin_level") != "8":
            continue
        for member in relation.get("members", []):
            if member.get("type") != "way" or member.get("role") not in ("outer", ""):
                continue
            way = ways.get(member["ref"])
            if way is None:
                continue
            ring = [(nodes[nid]["lon"], nodes[nid]["lat"])
                    for nid in way.get("nodes", []) if nid in nodes]
            if len(ring) >= 4 and ring[0] == ring[-1]:
                rings.append((_qualified_municipality(tags), ring))
    return rings


def municipality_containing(layers: dict, point_wgs84: Point) -> tuple | None:
    """(name, ring) of the municipality this point stands in, from an area's own layers, or None.

    BY CONTAINMENT, NOT BY NAME: OSM calls the borough "Hopewell" and its neighbour on this
    corridor "Hopewell Township", so a name match returns the wrong town or none. None where no
    closed ring holds the point - the same answer as a street that never leaves town.
    """
    for name, ring in layers.get("municipalities", ()):
        if Polygon(ring).contains(point_wgs84):
            return name, ring
    return None


def _qualified_municipality(tags: dict) -> str | None:
    """"Hopewell" + border_type=borough -> "Hopewell Borough", the form every config uses.

    OSM names the relation for the place and puts the kind in `border_type`, so the bare name is
    ambiguous exactly where this project needs it not to be: Hopewell Borough and Hopewell
    Township share a corridor, and route_decision_for is keyed on (street, town). Returning
    "Hopewell" matched neither, so a borough-wide document found 0 of 42 streets carrying the
    decision that is written for them.

    The suffix is only appended when it is not already there - the township relation is named
    "Hopewell Township" outright, and "Hopewell Township Township" matches nothing either.
    """
    name = (tags.get("name") or "").strip()
    kind = (tags.get("border_type") or "").strip()
    if not name or not kind or name.lower().endswith(kind.lower()):
        return name or None
    return f"{name} {kind.title()}"
