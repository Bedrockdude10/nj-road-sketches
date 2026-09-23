"""The borough document: every named street in a municipality, from OSM alone.

docs/network-model.md step 4. These pin the DOCUMENT, not a render. The claim under test is that
an OSM-sourced corridor lands where the model-sourced one lands - what makes the two
interchangeable and the per-site scenarios deletable.
"""
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

from src.geometry.intersection.municipality import municipal_boundary_ft
from src.geometry.network import corridors_from_models
from src.geometry.network.area import (MIN_CORRIDOR_FT, Corridor, _named_carriageways, _pieces_of,
                                       _projected_nodes, _snapshot_center, _way_line,
                                       area_corridors)
from src.sources.osm_context import SNAPSHOT_AREAS, fetch_borough_osm
from tests.conftest import needs_source_data

AREA = "hopewell_borough"

#: One lane width. The margin is thin on purpose - see test_the_osm_axis_agrees_with_the_modelled_one.
MAX_DATUM_GAP_FT: float = 10.0


@pytest.fixture(scope="module")
def borough() -> list[Corridor]:
    return area_corridors(AREA)


@pytest.fixture(scope="module")
def boundary() -> Polygon:
    return municipal_boundary_ft(Point(*_snapshot_center(SNAPSHOT_AREAS[AREA])))[1]


def test_the_document_holds_every_named_street_in_the_boundary(borough: list[Corridor],
                                                               boundary: Polygon) -> None:
    """Catches the longest-piece bug: keeping one run per street lost 0.457 mi silently, because
    every street still had A corridor."""
    snapshot = fetch_borough_osm(bbox=SNAPSHOT_AREAS[AREA])
    xy = _projected_nodes(snapshot["nodes"])
    available = sum(piece.length
                    for _name, way in _named_carriageways(snapshot)
                    if (line := _way_line(way, xy)) is not None
                    for piece in _pieces_of(line.intersection(boundary)))
    built = sum(corridor.length_ft for corridor in borough)

    assert built == pytest.approx(available, abs=len(borough) * MIN_CORRIDOR_FT), (
        f"the document carries {built:,.0f} ft of {available:,.0f} ft of named carriageway in "
        f"{AREA}: {available - built:,.0f} ft is in OSM and not in the document")


def test_no_corridor_runs_outside_its_own_municipality(borough: list[Corridor],
                                                       boundary: Polygon) -> None:
    for corridor in borough:
        outside = corridor.length_ft - corridor.centerline.intersection(boundary).length
        assert outside < 1.0, (
            f"{corridor.name} runs {outside:,.0f} ft outside {corridor.municipalities}, so a "
            f"route decision for this town would reach into the next one")


def test_a_street_osm_holds_in_pieces_stays_whole(borough: list[Corridor]) -> None:
    """Eaton Place is 1,135 ft in four OSM pieces; the longest alone reported 341 ft."""
    eaton = [corridor for corridor in borough if corridor.name == "Eaton Place"]
    assert len(eaton) > 1
    assert sum(corridor.length_ft for corridor in eaton) > 900.0


def test_every_crossing_lands_on_the_street_it_is_filed_under(borough: list[Corridor]) -> None:
    for corridor in borough:
        stations = np.asarray(corridor.cross_street_ft or [0.0])
        assert stations.min() >= 0.0 and stations.max() <= corridor.length_ft + 1.0, (
            f"{corridor.name} files a crossing off the end of its {corridor.length_ft:,.0f} ft")


@needs_source_data
def test_the_osm_axis_agrees_with_the_modelled_one(site_models: dict) -> None:
    """THE GATE. Deviation of the MODELLED axis from the OSM line - the modelled corridor is the
    shorter, so this asks whether OSM contains the axis we already drew.

    MEDIAN, not mean: Columbia Ave's modelled corridor opens 283.9 ft out and converges by station
    313 - its head extension bridging along NJDOT past where the model stops - putting the mean at
    28.9 ft on a corridor whose median is 1.8.

    The residual is a datum gap, not noise: Princeton Ave 0.9, Columbia 1.8, Broad St 7.5 (max
    11.3). A modelled corridor follows OSM through a junction it models and NJDOT where it
    bridges, so the figure tracks how much of a street is modelled. Broad St is worst because
    conftest SITES is four junctions, so most of its 6,868 ft is bridge; over five it was 2.8 ft.

    If this trips, read the per-street list before touching the bound - one street jumping is a
    corridor built wrong, all of them is the snapshot or the projection.
    """
    in_borough = {site: model for site, model in site_models.items()
                  if (model.config.get("intersection") or {}).get("municipality", "")
                  .lower().startswith("hopewell borough")}
    modelled: dict[str, Corridor] = {c.name: c for c in corridors_from_models(in_borough)}
    osm: dict[str, list[LineString]] = {}
    for corridor in area_corridors(AREA):
        osm.setdefault(corridor.name, []).append(corridor.centerline)

    compared: dict[str, float] = {}
    for name, corridor in modelled.items():
        if name not in osm:
            continue
        axis = corridor.centerline
        samples = shapely.line_interpolate_point(axis, np.linspace(0.0, axis.length, 200))
        # Nearest of this street's runs per sample: a street in pieces is still one street.
        compared[name] = float(np.median(
            np.min([shapely.distance(samples, run) for run in osm[name]], axis=0)))

    assert compared, (
        "no street is named the same by both builders. A leg's configured street name against "
        "OSM's `name` tag - _street_name does not reconcile 'Greenwood Ave' with 'Greenwood Avenue'")
    worst = max(compared, key=compared.get)
    assert compared[worst] <= MAX_DATUM_GAP_FT, (
        f"{worst}'s modelled axis sits a median {compared[worst]:.1f} ft off its OSM street. All: "
        + ", ".join(f"{n} {d:.1f} ft" for n, d in sorted(compared.items())))
