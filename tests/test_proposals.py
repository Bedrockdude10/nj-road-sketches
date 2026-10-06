"""A proposal is OSM: tags on existing ways, and new ways with OSM's negative ids."""
from types import SimpleNamespace

import pyproj
import pytest
from shapely.geometry import LineString

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84, Leg
from src.geometry.targets import LegSide
from src.geometry.treatments import (DesignState, RestrictionMarking, apply_osm_road_markings,
                                     restriction_painted_ft)
from src.sources.proposals import ProposalEntry, with_proposal

_TO_WGS84 = pyproj.Transformer.from_crs(NJ_STATE_PLANE_FT, WGS84, always_xy=True)
HATCH = {"road_marking": "restriction", "pattern": "chevron"}


def _wgs84(ring_ft):
    return [list(_TO_WGS84.transform(x, y)) for x, y in ring_ft]


def _ring(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_a_new_way_needs_a_closed_ring():
    with pytest.raises(ValueError, match="closed coords"):
        ProposalEntry(element="way/-1", tags=HATCH, source="x",
                      coords=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def test_an_existing_way_carries_no_coords():
    with pytest.raises(ValueError, match="already exists"):
        ProposalEntry(element="way/5", tags=HATCH, source="x",
                      coords=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])


def test_a_new_restriction_way_lands_in_road_markings():
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    osm = {"roads": [{"id": 7, "tags": {"highway": "residential"}}], "road_markings": []}
    merged = with_proposal(osm, [
        ProposalEntry(element="way/-1", tags=HATCH, source="x", coords=ring),
        ProposalEntry(element="way/7", tags={"cycleway:right": "track"}, source="x")])
    assert [m["id"] for m in merged["road_markings"]] == [-1]
    assert merged["roads"][0]["tags"]["cycleway:right"] == "track"
    assert osm["road_markings"] == []          # the shared layers are not mutated


def test_a_new_way_nothing_reads_is_refused():
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    with pytest.raises(ValueError, match="no layer"):
        with_proposal({"roads": []}, [ProposalEntry(element="way/-1", tags={"amenity": "bench"},
                                                    source="x", coords=ring)])


def test_a_restriction_area_takes_the_kerb_it_lies_along():
    x0, y0 = 417000.0, 565000.0
    leg = Leg(name="east", centerline=LineString([(x0, y0), (x0 + 120, y0)]), curb_to_curb_ft=30.0)
    state = DesignState(legs={"east": leg}, corner_fillets={})
    area = {"coords_wgs84": _wgs84(_ring(x0 + 20, x0 + 100, y0 + 9, y0 + 15)), "tags": HATCH,
            "id": -1}
    state = apply_osm_road_markings(state, SimpleNamespace(osm={"road_markings": [area]}))
    assert state.treatment_for(RestrictionMarking, LegSide("east", "left")) is not None
    assert state.treatment_for(RestrictionMarking, LegSide("east", "right")) is None
    # Half the nominal 30 ft less the 9 ft the area reaches in to.
    assert restriction_painted_ft(state, "east", "left") == pytest.approx(6.0, abs=0.01)
    assert restriction_painted_ft(state, "east", "right") == 0.0


def _street(way_id, node_ids, lons, name="Broad Street"):
    return {"id": way_id, "node_ids": node_ids, "tags": {"highway": "secondary", "name": name},
            "coords_wgs84": [[lon, 40.39] for lon in lons]}


def test_a_split_restates_the_way_and_adds_the_rest_along_its_own_nodes():
    from src.geometry.network.street_graph import street_edges

    osm = {"roads": [_street(7, [1, 2, 3], [-74.770, -74.769, -74.768])]}
    merged = with_proposal(osm, [
        ProposalEntry(element="way/7", nodes=[1, 2], source="x"),
        ProposalEntry(element="way/-1", nodes=[2, 3], source="x",
                      tags={"highway": "secondary", "name": "Broad Street",
                            "parking:left": "lane"})])
    by_id = {road["id"]: road for road in merged["roads"]}
    assert by_id[7]["node_ids"] == [1, 2]
    assert by_id[-1]["coords_wgs84"] == [[-74.769, 40.39], [-74.768, 40.39]]
    # osmnx keeps the two ways as two edges, each with its own tags, and node 2 - one street on
    # both sides - is no junction.
    edges, junctions, _xy = street_edges(merged["roads"])
    assert {edge.way_id for edge in edges} == {7, -1}
    assert junctions == set()


def test_a_way_along_a_node_the_area_lacks_is_refused():
    osm = {"roads": [_street(7, [1, 2], [-74.770, -74.769])]}
    with pytest.raises(ValueError, match="no road in the area"):
        with_proposal(osm, [ProposalEntry(element="way/-1", nodes=[2, 99], source="x",
                                          tags={"highway": "secondary"})])


def test_the_way_is_cut_only_where_the_answer_changes():
    from scripts.propose_bikeway_tags import _pieces

    nodes = [10, 11, 12, 13, 14, 15]
    spans = [(0, 1, "a"), (1, 2, "a"), (2, 4, "b"), (4, 5, "b")]
    assert _pieces(nodes, spans) == [([10, 11, 12], "a"), ([12, 13, 14, 15], "b")]
    assert _pieces(nodes, [(0, 2, "a"), (2, 5, "a")]) == [(nodes, "a")]
