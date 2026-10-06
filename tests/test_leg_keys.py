"""World leg keys come from OSM's own identity, so a split elsewhere renames nothing."""
from src.geometry.network.slice_design import leg_key
from src.geometry.network.street_graph import street_edges


def _road(way_id, node_ids, name, lons, lat):
    return {"id": way_id, "node_ids": node_ids, "tags": {"highway": "secondary", "name": name},
            "coords_wgs84": [[lon, lat] for lon in lons]}


def _roads(split: bool):
    """Broad St crossing Main St at node 3, and again Elm St at node 6."""
    broad = ([_road(1, [1, 2, 3], "Broad Street", [-74.770, -74.769, -74.768], 40.39),
              _road(2, [3, 4, 5, 6], "Broad Street", [-74.768, -74.767, -74.766, -74.765], 40.39)]
             if not split else
             [_road(1, [1, 2, 3], "Broad Street", [-74.770, -74.769, -74.768], 40.39),
              _road(2, [3, 4], "Broad Street", [-74.768, -74.767], 40.39),
              _road(-1, [4, 5, 6], "Broad Street", [-74.767, -74.766, -74.765], 40.39)])
    return [*broad,
            _road(10, [11, 3, 12], "Main Street", [-74.768] * 3, 40.38)
            | {"coords_wgs84": [[-74.768, 40.389], [-74.768, 40.39], [-74.768, 40.391]]},
            _road(20, [21, 6, 22], "Elm Street", [-74.765] * 3, 40.38)
            | {"coords_wgs84": [[-74.765, 40.389], [-74.765, 40.39], [-74.765, 40.391]]}]


def _keys(roads):
    edges, _junctions, _xy = street_edges(roads)
    return {leg_key(e.name, e.u, e.v) for e in edges}


def test_a_key_is_the_street_and_its_two_nodes():
    assert leg_key("Broad Street", 9, 4) == "broad_street_4_9"
    assert leg_key("Broad Street", 4, 9) == leg_key("Broad Street", 9, 4)


def test_splitting_a_way_renames_no_other_leg():
    before, after = _keys(_roads(split=False)), _keys(_roads(split=True))
    # The split stretch 3-6 becomes 3-4 and 4-6; every other leg keeps its key.
    assert before - {"broad_street_3_6"} <= after


def test_keys_are_unique_and_do_not_depend_on_order():
    roads = _roads(split=False)
    edges, _j, _xy = street_edges(roads)
    keys = [leg_key(e.name, e.u, e.v) for e in edges]
    assert len(keys) == len(set(keys))
    assert set(keys) == _keys(list(reversed(roads)))
