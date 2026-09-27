"""What we keep, and what we throw away, from what the surveyor traced.

The most expensive class of bug in this project isn't a wrong calculation - it's ground
truth being silently discarded on the way in, so everything downstream computes a careful
answer from a guess. These tests guard the intake.
"""
import json
import re

import pytest
from shapely.geometry import LineString, Point

from src.geometry.intersection.osm_roads import _assign_leg_pieces
from src.sources import osm_context
from src.sources.data_loader import OfflineCacheMiss, query_overpass


def a_snapshot(ways=(), nodes=()):
    """A fake borough snapshot in the shape fetch_borough_osm returns.

    Node coordinates sit inside the bbox the tests query, so the bbox filter keeps them and
    the assertions are about intake rules rather than geography.
    """
    node_table, way_list = {}, []
    next_id = 100
    for tags in nodes:
        node_table[next_id] = {"type": "node", "id": next_id, "lon": LON, "lat": LAT, "tags": tags}
        next_id += 1
    for way_id, vertex_count, tags in ways:
        refs = []
        for i in range(vertex_count):
            node_table[next_id] = {"type": "node", "id": next_id,
                                    "lon": LON + i * 1e-5, "lat": LAT + i * 1e-5, "tags": {}}
            refs.append(next_id)
            next_id += 1
        way_list.append({"type": "way", "id": way_id, "nodes": refs, "tags": tags})
    return {"nodes": node_table, "ways": way_list}


LON, LAT = -74.7600, 40.3890   # inside BOROUGH_BBOX
CENTRE = Point(LON, LAT)


def test_two_vertex_kerb_ways_are_kept(monkeypatch):
    """A straight run of kerb is two points, and straight runs are most of what gets traced.

    The old `len(geom) < 3` rule - really a circle-fitting precondition applied at the wrong
    layer - dropped 12 of the 23 traced ways at Columbia & Princeton and at E Broad &
    Princeton, and 5 of 12 at W Broad & Louellen. Those sides then fell back to centerline
    offsets on legs the surveyor had actually traced.
    """
    snapshot = a_snapshot(ways=[(1, 2, {"barrier": "kerb"}), (2, 3, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    kerbs = osm_context.fetch_kerbs(CENTRE, radius_m=120)
    assert len(kerbs) == 2, "the 2-vertex way must survive"
    assert min(len(k["coords_wgs84"]) for k in kerbs) == 2


def test_a_one_vertex_way_is_still_dropped(monkeypatch):
    """One point is not a line - there is no kerb to follow."""
    snapshot = a_snapshot(ways=[(1, 1, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    assert osm_context.fetch_kerbs(CENTRE, radius_m=120) == []


def test_kerb_node_ids_are_kept(monkeypatch):
    """Node ids are how a kerb is matched to the crossing it serves - one lowered kerb
    serving two crossings is distinguishable from two separate ramps only through these."""
    snapshot = a_snapshot(ways=[(7, 2, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    assert osm_context.fetch_kerbs(CENTRE, radius_m=120)[0]["node_ids"]


def test_kerb_tags_are_kept_whatever_their_value(monkeypatch):
    """kerb=lowered is a corner RAMP - the corner return itself. Filtering to kerb=raised
    dropped whole traced corners in favour of a fitted guess."""
    snapshot = a_snapshot(ways=[
        (1, 2, {"barrier": "kerb", "kerb": "raised"}),
        (2, 2, {"barrier": "kerb", "kerb": "lowered", "tactile_paving": "yes"}),
    ])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    kerbs = osm_context.fetch_kerbs(CENTRE, radius_m=120)
    assert {k["tags"].get("kerb") for k in kerbs} == {"raised", "lowered"}


def test_a_dangling_node_reference_is_refused(monkeypatch):
    """A truncated download must not become geometry with missing vertices.

    The OSM API completes ways whose nodes fall outside the bbox, so an unresolvable
    reference means the snapshot is incomplete - and half a kerb looks like a real kerb.
    """
    raw = [{"type": "node", "id": 1, "lon": LON, "lat": LAT, "tags": {}},
           {"type": "way", "id": 9, "nodes": [1, 999], "tags": {"barrier": "kerb"}}]
    monkeypatch.setattr(osm_context, "_cache_hit", lambda p: False)
    monkeypatch.setattr(osm_context, "_download_snapshot", lambda bbox=None: raw)
    monkeypatch.setattr(osm_context, "_write_cache", lambda p, d: None)
    osm_context._MEMO.clear()
    with pytest.raises(RuntimeError, match="don't resolve"):
        osm_context.fetch_borough_osm()
    osm_context._MEMO.clear()


def test_a_site_outside_the_snapshot_is_refused():
    """Reaching outside the downloaded bbox returns NOTHING, silently - which is exactly
    how ground truth disappears in this project. It must raise instead."""
    with pytest.raises(osm_context.SiteOutsideSnapshotError):
        osm_context.assert_within_snapshot(Point(-75.5, 40.0), radius_m=130)


def test_the_test_suite_cannot_reach_the_network():
    """conftest sets ROAD_SKETCHES_OFFLINE. A cache miss must fail loudly, not fetch.

    A test that silently depends on Overpass depends on its uptime AND its current
    replication state - two consecutive live fetches of one junction returned 4 tactile
    pads and then 0 during a single editing session.
    """
    with pytest.raises(OfflineCacheMiss):
        query_overpass("[out:json];node(1);out;")


def test_the_fixture_cache_is_present_and_readable():
    """The snapshot the rest of the suite runs against."""
    files = list(osm_context.CACHE_DIR.glob("*.json"))
    assert files, f"no OSM fixtures in {osm_context.CACHE_DIR}"
    for path in files:
        json.loads(path.read_text())


def test_a_pre_rename_environment_variable_is_refused(monkeypatch):
    """The switches this project reads were HOPEWELL_-prefixed while it was one town's study.

    An unread environment variable is not an error - it is a run that does the OTHER thing and
    looks fine: HOPEWELL_OFFLINE ignored is a run that goes to the network, HOPEWELL_DATA_DIR
    ignored is a run against the full county download instead of the clip. So a stale name
    raises and names its replacement.
    """
    from src.sources.data_loader import RENAMED_ENV, refuse_renamed_env

    for old, new in RENAMED_ENV.items():
        with pytest.raises(RuntimeError) as raised:
            refuse_renamed_env({old: "1"})
        assert new in str(raised.value), f"{old} must say what to use instead"
    refuse_renamed_env({"ROAD_SKETCHES_OFFLINE": "1"})      # the new names are fine


# --- The OTHER intake: which piece of somebody's linework is which leg --------------------
#
# NJDOT's SRI centreline arrives as one line, gets split at the junction, and the pieces have
# to be labelled with the names the design speaks in. `bearing_deg` in config.yaml is how a
# human labels them, and it is the only thing here that was never measured - so these pin what
# happens when it is ABSENT, which is a window onto the borough document (its legs carry a
# street name and nothing else) and any future caller that is not a configured site.
#
# Measured before this was written, over all 34 leg-pieces at the nine sites and at frame
# scales 1x/2.5x/3x: the declared bearing and the piece's own chord differ by up to 9.63 deg
# (broad_st_greenwood/broad_st_east, 57.3 declared against 66.9 drawn at 3x), while the two
# pieces of one SRI are 143-180 deg apart. The declaration therefore has 71-90 deg of slack in
# every case, which is why no assignment turns on the difference - and why a declaration that
# is merely approximate must still win, rather than being second-guessed by the geometry.

NODE = Point(0.0, 0.0)
NORTH_PIECE = LineString([(0, 0), (0, 130)])
SOUTH_PIECE = LineString([(0, 0), (0, -130)])
EAST_PIECE = LineString([(0, 0), (130, 0)])
WEST_PIECE = LineString([(0, 0), (-130, 0)])


def test_one_piece_and_one_leg_needs_no_declared_bearing():
    """A stub approach is the assignment that geometry settles on its own.

    One piece, one name: there is nothing for a bearing to tell apart, so requiring one made a
    caller with no config - a crop of the borough document, whose legs carry a street name and
    nothing else - fail with a bare KeyError on the leg's own name.
    """
    assigned = _assign_leg_pieces([EAST_PIECE], ["cross_east"], {}, NODE)
    assert assigned["cross_east"] is EAST_PIECE


def test_a_declared_bearing_decides_even_where_the_geometry_disagrees():
    """Config is an override and it wins BY BEING PRESENT - src/geometry/treatments/state.py's
    centerline_style is the model.

    Stated with a config that is deliberately back to front, because that is the only way to
    show the declaration is read at all: on every real site the declared bearing and the drawn
    one agree to within 9.63 deg against 143-180 deg of separation, so a matcher that quietly
    ignored the config would pass every site and this repo would find out from a render.
    """
    back_to_front = {"main_north": {"bearing_deg": 180.0},
                     "main_south": {"bearing_deg": 0.0}}
    assigned = _assign_leg_pieces([NORTH_PIECE, SOUTH_PIECE], ["main_north", "main_south"],
                                   back_to_front, NODE)
    assert assigned["main_north"] is SOUTH_PIECE
    assert assigned["main_south"] is NORTH_PIECE


def test_two_undeclared_legs_on_one_road_are_refused_with_the_bearings_to_type():
    """Which half of a through street is which is the one thing the geometry cannot answer.

    Both pieces exist and neither carries a name; picking by anything available - piece order,
    whichever the splitter emitted first - is a coin flip that draws a whole approach's
    treatment on the wrong side of the junction. So it refuses, and the refusal carries the two
    numbers that resolve it, the same way the count-mismatch message does.
    """
    with pytest.raises(ValueError) as raised:
        _assign_leg_pieces([NORTH_PIECE, SOUTH_PIECE], ["main_north", "main_south"], {}, NODE,
                            sri="00000518__")
    message = str(raised.value)
    assert "00000518__" in message
    assert "main_north" in message and "main_south" in message, message
    assert "bearing_deg" in message, message
    assert re.search(r"\b0\.0\b", message) and re.search(r"\b180\.0\b", message), message


def test_a_missing_bearing_does_not_swallow_the_count_mismatch():
    """The more-pieces-than-legs report used to be reached through `legs_cfg[n]['bearing_deg']`,
    so a config with no bearing raised KeyError from inside the message instead - naming neither
    the SRI nor the count, which is the whole failure that message exists to replace."""
    with pytest.raises(ValueError) as raised:
        _assign_leg_pieces([EAST_PIECE, WEST_PIECE], ["cross_east"], {}, NODE,
                            sri="11081029__")
    message = str(raised.value)
    assert "11081029__" in message
    assert "2 piece" in message, message
    assert "cross_east" in message, message


def test_a_piece_is_measured_along_its_own_chord_not_from_the_junction_passed_in():
    """One derivation of "which way does this point", and it is leg_frame.leg_bearing_deg.

    The piece's OWN chord, not a line from the centre handed to this call. They coincide at a
    configured site - every piece is snapped onto the resolved node first, measured at 0.000 ft
    over all 34 leg-pieces - so nothing here moves. They stop coinciding the moment one call
    covers more than one junction, which is what a window onto the document is: measured from
    the window's centre these two pieces read 57.0 and 123.0 deg, and they run due north and
    due south.
    """
    up = LineString([(200, 0), (200, 130)])
    down = LineString([(200, 0), (200, -130)])
    with pytest.raises(ValueError) as raised:
        _assign_leg_pieces([up, down], ["cross_east"], {"cross_east": {"bearing_deg": 90.0}},
                            NODE, sri="00000518__")
    message = str(raised.value)
    assert re.search(r"\b0\.0\b", message) and re.search(r"\b180\.0\b", message), message
    assert "57.0" not in message and "123.0" not in message, message
