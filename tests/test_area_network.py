"""The borough document: every named street in a municipality, from OSM alone.

docs/network-model.md step 4. These pin the DOCUMENT, not a render. The claim under test is that
an OSM-sourced corridor lands where the model-sourced one lands - what makes the two
interchangeable and the per-site scenarios deletable.
"""
import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

from src.geometry.intersection.municipality import municipal_boundary_ft
from src.geometry.network import corridors_from_models
from src.geometry.treatments import CorridorCalming, route_decision_for
from src.geometry.treatments.corridor import CorridorFacility
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


def test_the_document_carries_the_decisions_written_for_its_streets(borough: list[Corridor]) -> None:
    """The point of the whole document: what is proposed on a street is a LOOKUP against the
    network, not a line in a site file.

    This found 0 of 42 before `_qualified_municipality`: OSM names the relation "Hopewell" and
    puts "borough" in border_type, while every decision is keyed on "Hopewell Borough".
    """
    decided = {corridor.name: route_decision_for(corridor.name, corridor.municipalities[0])
               for corridor in borough}
    carrying = {name: decision for name, decision in decided.items() if decision is not None}

    assert set(carrying) == {"Broad Street", "Princeton Avenue"}, (
        f"the borough has two route decisions and the document found {sorted(carrying)}")
    assert isinstance(carrying["Broad Street"], CorridorFacility)
    assert isinstance(carrying["Princeton Avenue"], CorridorCalming)


def test_kerb_runs_stay_inside_the_corridor_they_are_filed_on(borough: list[Corridor]) -> None:
    for corridor in borough:
        for run in corridor.kerb_runs:
            assert -1.0 <= run.start_ft <= run.end_ft <= corridor.length_ft + 1.0, (
                f"{corridor.name} has a {run.side} kerb run at {run.start_ft:,.0f}-"
                f"{run.end_ft:,.0f} ft, off its own {corridor.length_ft:,.0f} ft")
            assert run.is_traced, "an area corridor has no modelled junction to source kerb from"


def test_the_document_finds_the_kerb_that_is_actually_traced(borough: list[Corridor]) -> None:
    """Coverage is a LENGTH, not a run count. 11 of 75 runs are under 10 ft - corner-return
    fragments the heading test lets through near a crossing - and they carry 86 ft between them,
    while 35 runs over 100 ft carry 12,771 of the 14,515 ft. No length floor, because
    `_traced_kerb_runs` has none and a second rule here would make the two builders disagree.
    """
    by_name = {corridor.name: corridor for corridor in borough}
    broad = by_name["Broad Street"]
    traced_ft = sum(run.length_ft for run in broad.kerb_runs)

    assert traced_ft / (2 * broad.length_ft) > 0.5, (
        f"Broad St is traced end to end in OSM, but the document found {traced_ft:,.0f} ft of "
        f"kerb against {2 * broad.length_ft:,.0f} ft of kerbline")


@needs_source_data
def test_the_osm_kerb_agrees_with_the_modelled_kerb(site_models: dict) -> None:
    """Both builders must find the same surveyor's kerb - per side, since a street's two kerbs are
    traced independently and a swap would cancel in the total.

    Compared as TOTALS: the two use different station origins, so a span-by-span diff would
    measure the offset between the frames. Measured: Broad St 4,229/4,294 ft left and
    4,356/4,575 right, Columbia 295/295 and 248/232, Princeton 1,189/1,191 and 1,034/1,023. The
    OSM side runs slightly longer because it covers the whole in-borough street.
    """
    modelled = {c.name: c for c in corridors_from_models(site_models)}
    osm: dict[str, list[Corridor]] = {}
    for corridor in area_corridors(AREA):
        osm.setdefault(corridor.name, []).append(corridor)

    def traced_ft(runs, side: str) -> float:
        return sum(run.length_ft for run in runs if run.side == side and run.is_traced)

    for name in sorted(set(modelled) & set(osm)):
        for side in ("left", "right"):
            want = traced_ft(modelled[name].kerb_runs, side)
            got = sum(traced_ft(c.kerb_runs, side) for c in osm[name])
            if want < MIN_CORRIDOR_FT:
                continue
            assert got == pytest.approx(want, rel=0.15), (
                f"{name} {side}: the document finds {got:,.0f} ft of traced kerb where the "
                f"modelled corridor finds {want:,.0f} ft")


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


def test_the_network_exports_as_one_geojson(tmp_path) -> None:
    """The 2D layer's output for an AREA: `scripts/export_network.py`, which a map can open and a
    render can crop. Round-tripped rather than string-matched - a file that parses as GeoJSON but
    loses its CRS reads as a town off the coast of Africa.
    """
    from scripts.export_network import WGS84_EPSG, export_network

    path = export_network(AREA, tmp_path)
    back = gpd.read_file(path)

    assert back.crs.to_epsg() == WGS84_EPSG
    assert {"street", "kerb", "crossing", "bikeway", "bikeway_buffer", "edge_line",
            "bollard"} <= set(back["kind"])
    assert (back.geometry.is_valid | back.geometry.is_empty).all()

    streets = back[back["kind"] == "street"]
    decided = dict(zip(streets["name"], streets["decision"]))
    assert decided["Broad Street"] == "CorridorFacility"
    assert decided["Princeton Avenue"] == "CorridorCalming"
    assert streets[streets["name"] == "Broad Street"]["kerb_coverage"].iloc[0] > 0.5

    # THE DESIGN IS IN THE DOCUMENT, not only the geometry - what makes a render a slice of this
    # rather than a rebuild. Every bikeway run is on the street that carries the facility.
    bikeway = back[back["kind"] == "bikeway"]
    assert set(bikeway["name"]) == {"Broad Street"}
    assert (bikeway["end_ft"] - bikeway["start_ft"]).sum() > 3000.0
    assert (bikeway["compass_side"] == "north").all(), "CORRIDOR_SIDE is the north kerb"


def test_a_render_is_a_slice_of_the_document(tmp_path) -> None:
    """The end of the pipeline: a 2D sheet built from the GeoJSON alone - no site, no scenario,
    no IntersectionModel. If a marking is missing from the picture it is missing from the file.
    """
    from scripts.export_network import export_network
    from scripts.render_slice import draw, load_network, slice_around, _center_ft

    export_network(AREA, tmp_path)
    network = load_network(AREA, tmp_path)

    # Broad & Greenwood, the junction three site files describe between them.
    around = slice_around(network, _center_ft("-74.7619598,40.389179"), 320.0)
    kinds = set(around["kind"])
    assert {"street", "kerb", "bikeway", "bollard"} <= kinds, (
        f"a slice at the corridor's central junction should carry the facility; got {kinds}")

    out = draw(around, "test", tmp_path / "slice.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_a_slice_clips_rather_than_dropping_what_overhangs_it() -> None:
    """A 1,050 ft bikeway run whose centre is outside the window still crosses it. Filtering by
    centroid instead of clipping would draw a hole where the longest run should be."""
    from scripts.render_slice import load_network, slice_around, _center_ft

    network = load_network(AREA)
    tight = slice_around(network, _center_ft("-74.7619598,40.389179"), 150.0)
    assert not tight[tight["kind"] == "bikeway"].empty
    assert tight.total_bounds[2] - tight.total_bounds[0] <= 301.0
