"""A proposal is an osmChange (src/sources/osm_change.py): load, write, and apply to the layers."""
import copy
import xml.etree.ElementTree as ET

import pytest

from src.sources.osm_change import (NewNode, OsmChange, WayChange, apply_change, load_change,
                                    write_change)

BROAD = {"highway": "secondary", "name": "Broad Street"}
NOTE = {"note": "for a test"}


def _layers():
    return {"roads": [{"id": 7, "node_ids": [1, 2, 3], "tags": dict(BROAD),
                       "coords_wgs84": [[-74.770, 40.39], [-74.769, 40.39], [-74.768, 40.39]]}],
            "road_markings": [], "driveways": []}


def _osc(tmp_path, body: str):
    path = tmp_path / "change.osc"
    path.write_text(f"<?xml version='1.0' encoding='utf-8'?>\n"
                    f"<osmChange version=\"0.6\" generator=\"test\">{body}</osmChange>")
    return path


def _square_nodes(first_id: int = -1):
    corners = [(-74.7700, 40.3900), (-74.7699, 40.3900), (-74.7699, 40.3901), (-74.7700, 40.3901)]
    return tuple(NewNode(first_id - i, lon, lat, {}) for i, (lon, lat) in enumerate(corners))


def test_a_modify_replaces_the_tags_and_the_node_list():
    change = OsmChange((), (WayChange(7, "modify", (1, 2), {**BROAD, "lanes": "2", **NOTE}),))
    road, = apply_change(_layers(), change)["roads"]
    assert road["node_ids"] == [1, 2]
    assert road["coords_wgs84"] == [[-74.770, 40.39], [-74.769, 40.39]]
    assert road["tags"] == {**BROAD, "lanes": "2", **NOTE}


def test_a_created_restriction_area_lands_in_road_markings_only():
    nodes = _square_nodes(-1)
    ring = (*(n.id for n in nodes), nodes[0].id)
    tags = {"road_marking": "restriction", "pattern": "chevron", **NOTE}
    out = apply_change(_layers(), OsmChange(nodes, (WayChange(-5, "create", ring, tags),)))
    assert [w["id"] for w in out["road_markings"]] == [-5]
    assert out["road_markings"][0]["coords_wgs84"][0] == [-74.7700, 40.3900]
    assert [w["id"] for w in out["roads"]] == [7]


def test_a_created_driveway_lands_in_driveways_and_roads():
    tags = {"highway": "service", "service": "driveway", **NOTE}
    out = apply_change(_layers(), OsmChange((), (WayChange(-1, "create", (2, 3), tags),)))
    assert [w["id"] for w in out["driveways"]] == [-1]
    assert sorted(w["id"] for w in out["roads"]) == [-1, 7]


def test_a_delete_removes_the_way_from_every_layer():
    out = apply_change(_layers(), OsmChange((), (WayChange(7, "delete", (), {}),)))
    assert out["roads"] == []


def test_the_layers_handed_in_are_not_changed():
    layers = _layers()
    before = copy.deepcopy(layers)
    apply_change(layers, OsmChange((), (WayChange(7, "modify", (1, 2), {**BROAD, **NOTE}),
                                        WayChange(-1, "create", (2, 3), {**BROAD, **NOTE}))))
    assert layers == before


@pytest.mark.parametrize("way", [
    WayChange(-1, "create", (2, 99), {**BROAD, **NOTE}),      # a node no way in the area has
    WayChange(99, "modify", (1, 2), {**BROAD, **NOTE}),       # a way the area does not carry
    WayChange(99, "delete", (), {}),
])
def test_a_change_that_cannot_apply_is_refused(way):
    with pytest.raises(ValueError):
        apply_change(_layers(), OsmChange((), (way,)))


@pytest.mark.parametrize("body", [
    '<create><way id="-1"><nd ref="1"/><nd ref="2"/><tag k="highway" v="service"/></way></create>',
    '<create><relation id="-1"><tag k="type" v="route"/></relation></create>',
    '<modify><node id="-1" lat="40.39" lon="-74.77"/></modify>',
    '<create><node id="1" lat="40.39" lon="-74.77"/></create>',
    '<create><way id="5"><nd ref="1"/><nd ref="2"/><tag k="note" v="x"/></way></create>',
    '<modify><way id="-3"><nd ref="1"/><nd ref="2"/><tag k="note" v="x"/></way></modify>',
    '<create><way id="-1"><nd ref="1"/><tag k="note" v="x"/></way></create>',
])
def test_a_file_outside_what_is_supported_is_refused(tmp_path, body):
    with pytest.raises(ValueError):
        load_change(_osc(tmp_path, body))


def test_a_moved_node_is_a_modify_that_keeps_its_tags_and_moves_what_uses_it(tmp_path):
    """The proposal re-centres a street on its kerbs by moving its existing nodes (positive ids):
    each goes under <modify> with its full tags, loads back unchanged, and moves every way that
    shares it."""
    moved = NewNode(2, -74.7690, 40.3901, {"highway": "crossing"})
    change = OsmChange((moved,), ())
    path = tmp_path / "moved.osc"
    write_change(change, path)
    sections = {section.tag: section for section in ET.parse(path).getroot()}
    assert list(sections) == ["modify"]
    node = sections["modify"].find("node")
    assert (node.get("id"), node.find("tag").get("v")) == ("2", "crossing")
    assert load_change(path) == change
    road = apply_change(_layers(), change)["roads"][0]
    assert road["coords_wgs84"][1] == [-74.7690, 40.3901]


def test_a_change_written_and_loaded_is_the_same_change(tmp_path):
    nodes = _square_nodes(-1)
    ring = (*(n.id for n in nodes), nodes[0].id)
    change = OsmChange(nodes, (
        WayChange(-5, "create", ring, {"road_marking": "restriction", **NOTE}),
        WayChange(-6, "create", (2, 3), {**BROAD, **NOTE}),
        WayChange(7, "modify", (1, 2), {**BROAD, **NOTE}),
        WayChange(8, "delete", (), {})))
    # Canonical order: nodes -1, -2...; creates by id descending, then modifies, then deletes.
    path = tmp_path / "out" / "change.osc"
    write_change(change, path)
    assert load_change(path) == change
    root = ET.parse(path).getroot()
    assert root.tag == "osmChange" and root.get("version") == "0.6"


def test_no_file_is_no_change(tmp_path):
    assert load_change(tmp_path / "absent.osc") == OsmChange((), ())


def test_a_split_is_two_ways_sharing_the_cut_node():
    out = apply_change(_layers(), OsmChange((), (
        WayChange(7, "modify", (1, 2), {**BROAD, **NOTE}),
        WayChange(-1, "create", (2, 3), {**BROAD, "parking:left": "lane", **NOTE}))))
    first, second = sorted(out["roads"], key=lambda way: way["node_ids"])
    assert (first["id"], first["node_ids"]) == (7, [1, 2])
    assert (second["id"], second["node_ids"]) == (-1, [2, 3])
    assert first["coords_wgs84"][-1] == second["coords_wgs84"][0]


def test_one_table_of_way_layers():
    from src.sources.osm_context import NODE_LAYERS, OSM_LAYERS, WAY_LAYERS

    assert {name for name, _predicate, _n in WAY_LAYERS} <= set(OSM_LAYERS)
    assert {name for name, _predicate in NODE_LAYERS} <= set(OSM_LAYERS)
