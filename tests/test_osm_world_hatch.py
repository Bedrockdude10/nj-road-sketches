"""A `road_marking=restriction` area is hatched in its `colour` (src/osm_world.py:HATCH_COLOURS)."""
import pytest

from src.osm_world import HATCH_COLOURS, LocalFrame, _Reader
from src.sources.osm_context import SNAPSHOT_AREAS

WHITE = ("lane_narrowing_edge_lines", "lane_narrowing_hatch_lines")


def _hatch(colour: str | None) -> _Reader:
    frame = LocalFrame(SNAPSHOT_AREAS["hopewell_borough"])
    square = [list(frame.wgs84(x, y)) for x, y in ((0, 0), (20, 0), (20, 6), (0, 6), (0, 0))]
    tags = {"road_marking": "restriction", "area": "yes", **({"colour": colour} if colour else {})}
    reader = _Reader(frame, {})
    reader.restriction({"id": 1, "node_ids": [1, 2, 3, 4, 1], "tags": tags, "coords_wgs84": square})
    reader.hatch_all()
    return reader


@pytest.mark.parametrize("colour", HATCH_COLOURS)
def test_a_coloured_area_is_outlined_and_struck_in_its_own_channels_only(colour):
    out = _hatch(colour).out
    assert out[f"{colour}_hatch_edge_lines"] and out[f"{colour}_hatch_stroke_lines"]
    assert not any(out[key] for key in WHITE)
    assert not any(out[f"{other}_hatch_{k}_lines"] for other in HATCH_COLOURS if other != colour
                   for k in ("edge", "stroke"))


@pytest.mark.parametrize("colour", [None, "white"])
def test_an_uncoloured_or_white_area_is_hatched_white(colour):
    out = _hatch(colour).out
    assert all(out[key] for key in WHITE)
    assert not any(out[f"{c}_hatch_{k}_lines"] for c in HATCH_COLOURS for k in ("edge", "stroke"))


def test_a_colour_with_no_paint_is_hatched_white_and_counted():
    reader = _hatch("purple")
    assert all(reader.out[key] for key in WHITE)
    assert reader.stats["restriction areas colour=purple: no such paint, hatched white"] == 1
