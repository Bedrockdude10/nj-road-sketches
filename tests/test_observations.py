"""The observation carrier: additive OSM tags on OSM elements.

An observation is an OSM tag on an OSM element (src/sources/observations.py's module
docstring), so these tests never check a renderer or a style table - only that the merge is
additive, refuses a real conflict, requires a source, and that an empty file changes nothing.
"""
import pytest
from pydantic import ValidationError

from src.geometry.network.area import area_context
from src.sources.observations import (ELEMENT_FROM_OBSERVATION, ELEMENT_FROM_OSM,
                                      ObservationEntry, ObservationError, apply_observations,
                                      load_observations)

# Broad & Greenwood's own centre (sites/broad_st_greenwood/config.yaml) - real, inside
# hopewell_borough's admin boundary, so a synthetic way built from it survives area_context's
# boundary-intersection gate without needing a live OSM fetch.
IN_BOROUGH = (-74.7619598, 40.389179)


def _crossing_snapshot() -> dict:
    lon, lat = IN_BOROUGH
    return {
        "nodes": {
            1: {"id": 1, "lon": lon, "lat": lat, "tags": {}},
            2: {"id": 2, "lon": lon + 0.0002, "lat": lat, "tags": {}},
        },
        "ways": [{"id": 11647647, "nodes": [1, 2],
                  "tags": {"highway": "footway", "footway": "crossing"}}],
        "relations": [],
    }


def test_load_observations_missing_file_returns_nothing(tmp_path):
    assert load_observations("no_such_area", path=tmp_path / "absent.yaml") == []


def test_an_observation_with_no_source_is_refused(tmp_path):
    path = tmp_path / "area.yaml"
    path.write_text("observations:\n  - element: way/1\n    tags:\n      foo: bar\n")
    with pytest.raises(ObservationError, match="source"):
        load_observations("area", path=path)


def test_a_synthetic_element_without_at_is_refused():
    with pytest.raises(ValidationError, match="synthetic"):
        ObservationEntry(element="node", tags={"highway": "traffic_signals"}, source="x")


def test_an_existing_reference_with_at_is_refused():
    with pytest.raises(ValidationError, match="already exists"):
        ObservationEntry(element="way/1", at=(0.0, 0.0), tags={"foo": "bar"}, source="x")


def test_an_unrecognised_element_form_is_refused():
    with pytest.raises(ValidationError, match="neither"):
        ObservationEntry(element="banana", tags={"foo": "bar"}, source="x")


def test_empty_observations_is_a_byte_identical_noop():
    """The acceptance bar: no observations, no change. `apply_observations` returns the exact
    snapshot object it was given rather than a reconstructed lookalike."""
    snapshot = _crossing_snapshot()
    assert apply_observations(snapshot, []) is snapshot


def test_additive_tag_is_merged_onto_the_existing_element():
    obs = ObservationEntry(element="way/11647647", tags={"crossing:markings": "lines"},
                           source="street-view review")
    merged = apply_observations(_crossing_snapshot(), [obs])
    way = next(w for w in merged["ways"] if w["id"] == 11647647)
    assert way["tags"]["crossing:markings"] == "lines"
    assert way["tags"]["footway"] == "crossing", "additive - the original tags must survive"


def test_same_key_same_value_is_accepted_as_redundant(capsys):
    """Verified this fires: an identical restatement must not raise, and must be reported so
    the file can be pruned (rule 2) - not silently swallowed."""
    obs = ObservationEntry(element="way/11647647", tags={"footway": "crossing"}, source="x")
    merged = apply_observations(_crossing_snapshot(), [obs])
    way = next(w for w in merged["ways"] if w["id"] == 11647647)
    assert way["tags"]["footway"] == "crossing"
    assert "redundant" in capsys.readouterr().out


def test_a_conflicting_value_is_refused():
    """THE MOST IMPORTANT BEHAVIOUR (rule 3): verified to actually fire, per SKILLS.md #4 - a
    check that has never raised pins nothing. Stashing the guard and re-running this would be
    the equivalent proof; this is the always-on version of that."""
    obs = ObservationEntry(element="way/11647647", tags={"footway": "sidewalk"},
                           source="a mistaken observation")
    with pytest.raises(ObservationError, match="authoritative"):
        apply_observations(_crossing_snapshot(), [obs])


def test_an_observation_naming_an_element_not_in_the_snapshot_is_refused():
    obs = ObservationEntry(element="way/999999", tags={"foo": "bar"}, source="x")
    with pytest.raises(ObservationError, match="not in this snapshot"):
        apply_observations(_crossing_snapshot(), [obs])


def test_a_synthetic_node_is_added_and_marked_as_observed():
    lon, lat = IN_BOROUGH
    obs = ObservationEntry(element="node", at=(lon, lat),
                           tags={"highway": "traffic_signals", "support": "mast_arm"},
                           source="street-view photo review, Rollo, 2026-07-02")
    merged = apply_observations(_crossing_snapshot(), [obs])
    new_ids = set(merged["nodes"]) - {1, 2}
    assert len(new_ids) == 1
    new_node = merged["nodes"][next(iter(new_ids))]
    assert new_node["tags"] == {"highway": "traffic_signals", "support": "mast_arm"}
    assert new_node["lon"] == lon and new_node["lat"] == lat
    assert new_node["provenance"] == ELEMENT_FROM_OBSERVATION
    # A real element apply_observations never touched carries no marker of its own; the reader
    # (area_context) is what supplies ELEMENT_FROM_OSM as the default for those.
    assert "provenance" not in merged["nodes"][1]


def test_area_context_applies_an_observation_at_its_one_merge_point(monkeypatch):
    """End to end through the real merge point: no reader was written for
    `crossing:markings` - `area_context`'s own crossing row picks it up because the tag landed
    on the way before anything downstream looked at it.
    """
    obs = ObservationEntry(element="way/11647647", tags={"crossing:markings": "lines"},
                           source="street-view review")
    monkeypatch.setattr("src.geometry.network.area.load_observations", lambda area: [obs])

    doc = area_context("hopewell_borough", snapshot=_crossing_snapshot())

    crossing = next(row for row in doc["crossings"] if row["id"] == 11647647)
    assert crossing["tags"]["crossing:markings"] == "lines"
    assert crossing["provenance"] == ELEMENT_FROM_OSM, "tags added, but still a real OSM way"


def test_area_context_marks_a_synthetic_node_as_observed_not_osm(monkeypatch):
    lon, lat = IN_BOROUGH
    obs = ObservationEntry(element="node", at=(lon, lat + 0.0001),
                           tags={"highway": "traffic_signals", "support": "mast_arm"},
                           source="street-view photo review, Rollo, 2026-07-02")
    monkeypatch.setattr("src.geometry.network.area.load_observations", lambda area: [obs])

    doc = area_context("hopewell_borough", snapshot=_crossing_snapshot())

    signals = [row for row in doc["nodes"] if row["tags"].get("highway") == "traffic_signals"]
    assert len(signals) == 1
    assert signals[0]["provenance"] == ELEMENT_FROM_OBSERVATION
