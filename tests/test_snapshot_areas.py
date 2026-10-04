"""Which downloaded OSM area a site is a view onto.

The snapshot was ONE bbox covering Hopewell Borough, and a site outside it was refused with
"widen BOROUGH_BBOX and delete the cached snapshot". Taking that advice for a site in the
next town along would have re-keyed the only snapshot there is: every existing site would
re-download, the committed fixture in tests/fixtures/osm_cache/ would no longer match the
cache key, and 236 tests would start missing it. It would also pull the several miles of
farmland between the two boroughs to reach a junction 3 miles away.

So the areas are plural, each cached and keyed separately. What these tests hold down is
that adding one cannot disturb the others - particularly Hopewell's cache key, which the
committed fixture is named after.
"""
import pytest
from shapely.geometry import Point

from src.sources.osm_context import (SNAPSHOT_AREAS, UnknownAreaError, _load_snapshot_areas,
                                      _snapshot_path, osm_layers)

HOPEWELL_BBOX = (-74.7760, 40.3830, -74.7500, 40.3970)
BROAD_AND_GREENWOOD = Point(-74.7614, 40.3893)
NJ31_AND_W_DELAWARE = Point(-74.7989728, 40.3272925)


def test_hopewells_cache_key_is_unchanged():
    """The committed fixture is named for this hash. If it moves, the offline suite stops
    finding the snapshot and every junction test skips - silently green, testing nothing."""
    assert SNAPSHOT_AREAS["hopewell_borough"] == HOPEWELL_BBOX
    assert _snapshot_path(HOPEWELL_BBOX).name == "borough_33409013af7cbb1a.json"


def _contains(bbox, point):
    west, south, east, north = bbox
    return west <= point.x <= east and south <= point.y <= north


def test_each_borough_is_served_from_its_own_area():
    """Which area a site belongs to is no longer found by searching for the one that holds a
    circle about it: the config names it. So what has to hold is that the name is the right
    one - the area it names is the area the junction stands in."""
    assert _contains(SNAPSHOT_AREAS["hopewell_borough"], BROAD_AND_GREENWOOD)
    assert not _contains(SNAPSHOT_AREAS["hopewell_borough"], NJ31_AND_W_DELAWARE)
    assert _contains(SNAPSHOT_AREAS["pennington_borough"], NJ31_AND_W_DELAWARE)
    assert not _contains(SNAPSHOT_AREAS["pennington_borough"], BROAD_AND_GREENWOOD)


def test_the_two_areas_have_different_cache_keys():
    paths = {_snapshot_path(bbox).name for bbox in SNAPSHOT_AREAS.values()}
    assert len(paths) == len(SNAPSHOT_AREAS), "two areas sharing a cache file would overwrite"


def test_an_area_nobody_declared_is_refused_and_told_which_areas_exist():
    """A site naming an area that does not exist - Trenton's, say, a real place and not one this
    project has a snapshot of - must raise and say what the choices are, not read as an empty
    world. The error comes from the one way into OSM, so it holds for a site, a slice and the
    borough alike."""
    with pytest.raises(UnknownAreaError) as raised:
        osm_layers("trenton")
    message = str(raised.value)
    assert "hopewell_borough" in message and "pennington_borough" in message, message
    assert "osm_areas.yaml" in message, "the message must name what to edit"


def _margin_m(point, bbox):
    """Metres from a point to the nearest edge of a bbox, to a few percent at this latitude."""
    west, south, east, north = bbox
    return min((point.x - west) * 85_000.0, (east - point.x) * 85_000.0,
               (point.y - south) * 111_000.0, (north - point.y) * 111_000.0)


def test_a_site_near_its_areas_edge_is_caught_not_half_served():
    """THE FAILURE THE OLD WINDOW GUARD EXISTED FOR, re-expressed. Data stops at an area's edge,
    and a junction drawn beside it shows the elements that happen to be inside and NOTHING for
    the rest - which looks like geometry rather than like an error. There is no context window
    to test against any more, so the guard moves to the site: its centre must sit well inside
    the area it names.

    250 m is the largest radius any site was ever read with, so it is the least reach a drawing
    has been shown to want past its centre. The tightest margin today is 289 m (ebroad_elm).
    Not a probe at Hopewell's west edge: hopewell_wbroad_west overlaps it, so a point there is
    genuinely served by a neighbour, and "it is refused" would be a false claim.
    """
    from src.site import list_sites, load_site_config

    for site in list_sites():
        intersection = load_site_config(site)["intersection"]
        centre = Point(*intersection["center_wgs84"])
        margin = _margin_m(centre, SNAPSHOT_AREAS[intersection["osm_area"]])
        assert margin >= 250, (
            f"{site} is {margin:.0f} m from the edge of {intersection['osm_area']}: everything "
            f"past it is silently absent from the drawing")


def test_the_margin_check_sees_a_point_at_the_edge():
    """Verified to fail, which a check that has only ever passed does not prove: a point 20 m
    inside Hopewell's north edge is under the 250 m the test above demands."""
    near_north = Point(-74.7614, HOPEWELL_BBOX[3] - 0.0002)
    assert _margin_m(near_north, HOPEWELL_BBOX) < 250
    assert _margin_m(BROAD_AND_GREENWOOD, HOPEWELL_BBOX) >= 250


# --- the areas are declared in sites/osm_areas.yaml, so the file is a boundary too ---------

def _areas_file(tmp_path, body: str):
    path = tmp_path / "osm_areas.yaml"
    path.write_text(body)
    return path


def test_the_declared_areas_are_the_ones_in_the_yaml():
    """The list of towns is data beside the sites, not a literal in src/ - porting this project
    to another municipality is one entry here and one download."""
    import yaml

    from src.sources.osm_context import SNAPSHOT_AREAS_FILE

    declared = yaml.safe_load(SNAPSHOT_AREAS_FILE.read_text())
    assert {name: tuple(bbox) for name, bbox in declared.items()} == SNAPSHOT_AREAS


def test_a_reversed_bbox_is_refused_at_load(tmp_path):
    """A bbox typed south-north or east-west the wrong way round CONTAINS NOTHING, so every site
    in that town is refused with a message about the site - which sends the reader to look at a
    junction that is fine. Caught where the tuple is read instead."""
    path = _areas_file(tmp_path, "lavallette: [-74.0600, 39.9800, -74.0700, 39.9700]\n")
    with pytest.raises(ValueError) as raised:
        _load_snapshot_areas(path)
    assert "lavallette" in str(raised.value)


def test_an_area_over_the_apis_limit_is_refused_at_load(tmp_path):
    """The other way a bbox goes wrong: a digit dropped makes it enormous, and the OSM API
    refuses a /map call over 0.25 sq deg - at download time, which is the far end of the
    afternoon someone spent tracing kerbs."""
    path = _areas_file(tmp_path, "whole_state: [-75.6, 38.9, -73.9, 41.4]\n")
    with pytest.raises(ValueError) as raised:
        _load_snapshot_areas(path)
    assert "sq deg" in str(raised.value)


def test_every_site_this_project_models_falls_inside_the_area_it_names():
    """The porting check: a new site is a name in config.yaml and an entry in osm_areas.yaml, and
    nothing but this ties the two. A typo reads downstream as UnknownAreaError at build time, and
    an area that does not hold the junction reads as an empty street - so say both for the whole
    set at once."""
    from src.site import list_sites, load_site_config

    for site in list_sites():
        intersection = load_site_config(site)["intersection"]
        area = intersection["osm_area"]
        assert area in SNAPSHOT_AREAS, f"{site}: {area!r} is not in sites/osm_areas.yaml"
        assert _margin_m(Point(*intersection["center_wgs84"]), SNAPSHOT_AREAS[area]) > 0, (
            f"{site} stands outside {area} {SNAPSHOT_AREAS[area]}")
