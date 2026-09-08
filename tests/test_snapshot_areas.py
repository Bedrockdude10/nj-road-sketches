"""Which downloaded OSM snapshot a site is served from.

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

from src.sources.osm_context import (SNAPSHOT_AREAS, SiteOutsideSnapshotError,
                                      _area_for, _load_snapshot_areas, _snapshot_path,
                                      assert_within_snapshot)

HOPEWELL_BBOX = (-74.7760, 40.3830, -74.7500, 40.3970)
BROAD_AND_GREENWOOD = Point(-74.7614, 40.3893)
NJ31_AND_W_DELAWARE = Point(-74.7989728, 40.3272925)


def test_hopewells_cache_key_is_unchanged():
    """The committed fixture is named for this hash. If it moves, the offline suite stops
    finding the snapshot and every junction test skips - silently green, testing nothing."""
    assert SNAPSHOT_AREAS["hopewell_borough"] == HOPEWELL_BBOX
    assert _snapshot_path(HOPEWELL_BBOX).name == "borough_33409013af7cbb1a.json"


def test_each_borough_is_served_from_its_own_area():
    assert _area_for(BROAD_AND_GREENWOOD, 130) == SNAPSHOT_AREAS["hopewell_borough"]
    assert _area_for(NJ31_AND_W_DELAWARE, 130) == SNAPSHOT_AREAS["pennington_borough"]


def test_the_two_areas_have_different_cache_keys():
    paths = {_snapshot_path(bbox).name for bbox in SNAPSHOT_AREAS.values()}
    assert len(paths) == len(SNAPSHOT_AREAS), "two areas sharing a cache file would overwrite"


def test_a_site_in_no_area_is_refused_and_told_which_areas_exist():
    """Trenton - a real place, and not one this project has a snapshot of."""
    with pytest.raises(SiteOutsideSnapshotError) as raised:
        assert_within_snapshot(Point(-74.7429, 40.2206), 130)
    message = str(raised.value)
    assert "hopewell_borough" in message and "pennington_borough" in message, message
    assert "SNAPSHOT_AREAS" in message, "the message must name what to edit"


def test_a_window_straddling_an_edge_is_refused_not_half_served():
    """The failure this whole guard exists for: a context window partly outside its area
    returns the elements that happen to be inside and NOTHING for the rest, which looks
    like geometry rather than like an error. A point just inside Hopewell's west edge with
    a radius that reaches past it must raise, not quietly return half a junction."""
    west_edge = Point(HOPEWELL_BBOX[0] + 0.0002, 40.3900)
    with pytest.raises(SiteOutsideSnapshotError):
        assert_within_snapshot(west_edge, 130)


def test_a_site_well_inside_an_area_passes():
    assert_within_snapshot(BROAD_AND_GREENWOOD, 130)
    assert_within_snapshot(NJ31_AND_W_DELAWARE, 250)


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


def test_every_site_this_project_models_falls_inside_an_area():
    """The porting check: a new site whose town has no snapshot area is refused at build time
    with a message about the area, and this says the same thing for the whole set at once."""
    from shapely.geometry import Point as _Point

    from src.site import list_sites, load_site_config

    for site in list_sites():
        config = load_site_config(site)
        lon, lat = config["intersection"]["center_wgs84"]
        assert_within_snapshot(_Point(lon, lat), config["intersection"]["clip_radius_m"])
