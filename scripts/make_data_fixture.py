"""Clip data/ down to the few features the sites actually read, as a committed test fixture.

    .venv/bin/python scripts/make_data_fixture.py            # rebuild tests/fixtures/data
    .venv/bin/python scripts/make_data_fixture.py --pad-ft 3000 --out /tmp/clip

WHY. data/ is a 391 MB third-party download kept out of git, so every test that builds a real
junction - 333 of 707, including every geometry golden - skips in CI and in any checkout without
it. But a site reads only the road segments and parcels over its OSM AREA (sites/osm_areas.yaml) out
of NJDOT's 105,838 segments and the whole county's parcels: the areas the sites sit in are a few
square miles, a few megabytes. tests/fixtures/
osm_cache already established the pattern for Overpass; this is the same trick for the two
GIS layers and the tax list.

THE CLIP IS READ THROUGH THE PRODUCTION LOADERS. ROAD_SKETCHES_DATA_DIR re-roots `data/...` paths
(src/sources/data_loader.resolve_data_path) and nothing else changes: same pandera schemas, same
CRS checks, same indexed-sibling swap. A separate "test data" code path would be a second datum
for the same fact, which is the most expensive defect class this repo has.

TWO THINGS KEEP IT HONEST, because a clip's failure mode is a read that SUCCEEDS with fewer
features - a wrong measurement that looks right:

  * Every layer written here is read back and compared to the same bbox read of the source,
    attribute by attribute and geometry WKB by WKB (verify_identical, borrowed from
    convert_road_network.py). A clip that is not a byte-faithful subset is deleted, not kept.
  * FIXTURE.json records how far the clip reaches, and load_road_network/load_parcels raise
    FixtureExtentExceeded on a read that runs off the edge of it (or on an unbounded read).
    The reads are over each site's OSM AREA and nothing else - not ROAD_SKETCHES_FRAME_SCALE, which
    scales leg lengths - so a site in an area that is not in the clip is what makes the guard
    fire, and a second site in an area that is, is not.

tests/test_data_fixture.py closes the loop from the other side: it builds every site from the
clip AND from the real data/ and compares the resolved models. That test needs the download, so
it runs locally and skips in CI - the guard above is the part that works where data/ is absent.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
import pyogrio

from src.geometry.model import NJ_STATE_PLANE_FT
from src.site import list_sites, load_site_config
from src.sources.data_loader import (DATA_DIR, FIXTURE_MANIFEST_NAME, _resolve_indexed_path,
                                     _unpack_single_part, wgs84_box_in_state_plane)
from src.sources.osm_context import SNAPSHOT_AREAS

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT_DIR / "tests" / "fixtures" / "data"

# Slack past each area's box. The loaders read EXACTLY the area, so this only has to absorb float
# noise and a small edit to an area; a new area is a new box and a regenerate, which is correct.
DEFAULT_PAD_FT = 300.0

# Columns the parcel clip carries. NOT "all of them": LASTUPDATE is a shapefile Date, and OGR
# cannot round-trip a date through a shapefile as anything but a string ("created as String
# field, though DateTime requested"), so a clip that kept it would differ from the county file in
# a column nothing reads. Dropping it is what lets verify_identical stay an exact comparison
# rather than one with an exception carved into it. PAMS_PIN and MUN are what ParcelsSchema
# declares; BLOCK/LOT/QCODE are the assessor's own identifiers, kept because they are free.
PARCEL_COLUMNS = ("PAMS_PIN", "MUN", "BLOCK", "LOT", "QCODE")
# Same idea, and additionally a privacy one - see clip_tax_list.
TAX_COLUMNS = ("GIS_PIN", "BLDG_DESC")
FT_PER_DEG_LAT = 364000.0   # good to ~1% at this latitude; padding, not geometry


def site_reads(sites: list[str]) -> dict[str, dict]:
    """{site: {"area", "roads_wgs84", "parcels_nj_ft", "layers"}} - the box each site's loaders read.

    The boxes ARE the area (SNAPSHOT_AREAS[intersection.osm_area]), and the parcel box is made by
    the loaders' own wgs84_box_in_state_plane rather than re-derived here, because a fixture sized
    by a second copy of that arithmetic is exactly the kind of agreeing-but-wrong pair this repo
    keeps getting bitten by.
    """
    reads = {}
    for site in sites:
        config = load_site_config(site)
        area = config["intersection"]["osm_area"]
        bbox = tuple(float(v) for v in SNAPSHOT_AREAS[area])
        parcels = tuple(float(v) for v in wgs84_box_in_state_plane(bbox).total_bounds)
        reads[site] = {"area": area, "roads_wgs84": bbox, "parcels_nj_ft": parcels,
                       "layers": config.get("data_sources") or {}}
    return reads


def boxes_for(reads: dict[str, dict], layer: str, key: str, source: Path, pad_ft: float,
              degrees: bool) -> list[tuple]:
    """The padded, de-duplicated boxes of every site that reads `source` as `layer`.

    A LIST, not their union: two areas that are far apart (Hopewell and Lavallette are 60 miles
    apart) bound most of the state, and a clip of everything in between is not a fixture.
    """
    boxes: list[tuple] = []
    for read in reads.values():
        configured = read["layers"].get(layer)
        if configured and ROOT_DIR / configured == source:
            box_ = _pad(read[key], pad_ft, degrees=degrees)
            if box_ not in boxes:
                boxes.append(box_)
    return boxes


def read_boxes(source: Path, boxes: list, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    """The union of a bbox read per box, each feature once (by its FID in the source file)."""
    frames = [gpd.read_file(source, bbox=box_, columns=columns, fid_as_index=True) for box_ in boxes]
    merged = pd.concat(frames)
    return merged[~merged.index.duplicated()].reset_index(drop=True)


def _pad(bbox: tuple, pad_ft: float, degrees: bool) -> tuple:
    """Grow a bbox by pad_ft on all four sides.

    In degrees, both axes are padded by the LATITUDE figure. A longitude degree is shorter than
    a latitude one at 40.4 N (cos ~ 0.76), so that over-covers east-west - the safe direction,
    for a number whose whole job is slack.
    """
    pad = pad_ft / FT_PER_DEG_LAT if degrees else pad_ft
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def verify_identical(source_path: Path, boxes: list, written: Path, columns: list[str] | None = None) -> bool:
    """Compare the written clip against the same boxes' reads of the source, exactly.

    Attributes and geometry WKB, after _unpack_single_part on both sides - which is what the
    loaders apply on every read, so this checks the data as it will actually be consumed.
    """
    want = read_boxes(source_path, boxes, columns)
    got = gpd.read_file(written, bbox=None, columns=columns)
    if len(want) != len(got):
        print(f"  MISMATCH: {len(want)} features in the source bbox, {len(got)} written.")
        return False
    shared = [c for c in want.columns if c != "geometry"]
    if sorted(shared) != sorted(c for c in got.columns if c != "geometry"):
        print(f"  MISMATCH: columns differ - {sorted(shared)} vs "
              f"{sorted(c for c in got.columns if c != 'geometry')}")
        return False
    key = next((c for c in ("OBJECTID", "PAMS_PIN", "GIS_PIN") if c in shared), None)
    if key:
        want, got = want.sort_values(key).reset_index(drop=True), got.sort_values(key).reset_index(drop=True)
    for column in shared:
        if not want[column].astype("string").fillna("").equals(got[column].astype("string").fillna("")):
            print(f"  MISMATCH: column {column} differs.")
            return False
    if "geometry" in want.columns and want.geometry.notna().any():
        if want.crs is not None and got.crs is not None and want.crs != got.crs:
            print(f"  MISMATCH: CRS {want.crs.name} vs {got.crs.name}")
            return False
        a = want.geometry.map(_unpack_single_part).to_wkb()
        b = got.geometry.map(_unpack_single_part).to_wkb()
        if not (a == b).all():
            print("  MISMATCH: geometry WKB differs.")
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_geoseries(bbox: tuple):
    """A CRS-tagged box, which is how load_parcels queries the shapefile's own NAD83 variant."""
    return gpd.GeoSeries([gpd.GeoSeries.from_wkt([
        f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, "
        f"{bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"]).iloc[0]], crs=NJ_STATE_PLANE_FT)


def layer_sources(sites: list[str]) -> dict[str, list[Path]]:
    """{"road_network"/"parcels"/"tax_list": [distinct files these sites read]}.

    FROM THE CONFIGS, NOT FROM THREE FILENAMES WRITTEN HERE. Every site names its own layers
    (sites/README.md, `data_sources:`), and a site in another county names another county's
    parcels and MOD-IV rows - so a fixture built from hardcoded Mercer filenames would clip the
    wrong file for it and, worse, clip it SUCCESSFULLY: the sites it does cover keep passing
    while the new one silently reads a county it is not in. A list per layer, because two
    counties in one fixture is the normal case the moment a second one is added.
    """
    found: dict[str, list[Path]] = {"road_network": [], "parcels": [], "tax_list": []}
    for site in sites:
        sources = load_site_config(site).get("data_sources") or {}
        for layer, paths in found.items():
            configured = sources.get(layer)
            if not configured:
                continue                       # tax_list is optional - see src/site_schema.py
            path = ROOT_DIR / configured
            if path not in paths:
                paths.append(path)
    return found


def clip_roads(out: Path, boxes: list, source: Path) -> tuple[Path, Path, int]:
    """The roadway network, as FlatGeobuf - the format the loaders already prefer."""
    source = _resolve_indexed_path(source)
    target = out / (source.stem + ".fgb")
    roads = read_boxes(source, boxes)
    roads.to_file(target, driver="FlatGeobuf")
    return source, target, len(roads)


def clip_parcels(out: Path, boxes: list, source: Path) -> tuple[Path, Path, int]:
    """The parcels, as a shapefile with the source .prj copied over byte for byte.

    MercerCountyParcels.shp is a COMPOUND CRS whose WKT matches no EPSG code (see
    src/sources/schemas._horizontal), and a rewritten .prj is a different string for the same
    ground. Copying it keeps the fixture's CRS textually identical to the county's, so the
    boundary check tests the same thing here as it does against the download.
    """
    target = out / source.name
    parcels = read_boxes(source, boxes, list(PARCEL_COLUMNS))
    parcels.to_file(target, driver="ESRI Shapefile")
    shutil.copyfile(source.with_suffix(".prj"), target.with_suffix(".prj"))
    return source, target, len(parcels)


def clip_tax_list(out: Path, pins: set[str], source: Path) -> tuple[Path, Path, int]:
    """The MOD-IV rows for the clipped parcels, as a standalone .dbf.

    Only GIS_PIN and BLDG_DESC: those are the two columns assessor.py reads and TaxListSchema
    declares, and the other 100-odd carry owner names and sale prices that have no business in
    a public fixture. The schemas are strict=False on the column SET by design.
    """
    target = out / source.name
    rows = gpd.read_file(source, columns=list(TAX_COLUMNS))
    kept = pd.DataFrame(rows.drop(columns="geometry", errors="ignore"))
    kept = kept[kept["GIS_PIN"].astype("string").str.strip().isin(pins)]
    pyogrio.write_dataframe(kept, target, driver="ESRI Shapefile")
    return source, target, len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where to write the clip (default {DEFAULT_OUT.relative_to(ROOT_DIR)})")
    parser.add_argument("--pad-ft", type=float, default=DEFAULT_PAD_FT,
                        help=f"padding past the furthest read any site makes (default {DEFAULT_PAD_FT:.0f})")
    parser.add_argument("--site", action="append", choices=list_sites(),
                        help="only these sites (default: all of them)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"data/ is not here ({DATA_DIR}), and this script clips FROM it - it is how the "
              f"fixture gets built, not how it gets used. See README, section \"Data\".",
              file=sys.stderr)
        return 2

    sites = args.site or list_sites()
    reads = site_reads(sites)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Clipping {len(sites)} site(s) out of data/, padded {args.pad_ft:.0f} ft -> {out}")

    started = time.perf_counter()
    written: dict[str, dict] = {}
    sources = layer_sources(sites)
    # ONE PASS PER DISTINCT FILE, and the tax rows are filtered by the PINs of the parcels
    # actually clipped - so a two-county fixture keeps each county's rows against its own
    # parcels rather than crossing them.
    clipped: list[tuple[str, Path, object, Path, list[str] | None]] = []
    pins: set[str] = set()
    road_boxes: list[tuple] = []
    parcel_boxes: list[tuple] = []
    for source in sources["road_network"]:
        boxes = boxes_for(reads, "road_network", "roads_wgs84", source, args.pad_ft, degrees=True)
        road_boxes += [box_ for box_ in boxes if box_ not in road_boxes]
        road_source, road_target, n_roads = clip_roads(out, boxes, source)
        print(f"  roads:   {n_roads} features from {road_source.name} over {len(boxes)} area box(es)")
        clipped.append(("roads", road_source, boxes, road_target, None))
    for source in sources["parcels"]:
        boxes = boxes_for(reads, "parcels", "parcels_nj_ft", source, args.pad_ft, degrees=False)
        parcel_boxes += [box_ for box_ in boxes if box_ not in parcel_boxes]
        parcel_source, parcel_target, n_parcels = clip_parcels(out, [_as_geoseries(b) for b in boxes], source)
        print(f"  parcels: {n_parcels} polygons from {parcel_source.name} over {len(boxes)} area box(es)")
        clipped.append(("parcels", parcel_source, [_as_geoseries(b) for b in boxes], parcel_target,
                        list(PARCEL_COLUMNS)))
        pins |= {str(pin).strip() for pin in gpd.read_file(parcel_target)["PAMS_PIN"].dropna()}
    for source in sources["tax_list"]:
        tax_source, tax_target, n_tax = clip_tax_list(out, pins, source)
        print(f"  tax:     {n_tax} rows for {len(pins)} parcels from {tax_source.name}")
        clipped.append(("tax", tax_source, None, tax_target, list(TAX_COLUMNS)))

    print("Verifying each clip is a faithful subset of the source...")
    for name, source, bbox, target, columns in clipped:
        if name == "tax":
            continue      # a PIN filter, not a bbox clip - checked by the join test instead
        if not verify_identical(source, bbox, target, columns):
            for stem in target.parent.glob(target.stem + ".*"):
                stem.unlink()
            print(f"Verification FAILED for {target.name} - deleted it. data/ is untouched.",
                  file=sys.stderr)
            return 1
        print(f"  {target.name}: identical (attributes equal, geometry WKB equal).")

    for name, source, _bbox, target, _columns in clipped:
        written[target.name] = {"layer": name, "file": target.name,
                                "clipped_from": source.name,
                                "source_sha256": _sha256(source),
                                "columns": {"parcels": list(PARCEL_COLUMNS),
                                            "tax": list(TAX_COLUMNS)}.get(name, "all")}
    manifest = {
        "generated_by": "scripts/make_data_fixture.py",
        "why": "data/ is a large third-party download kept out of git; these are the features "
               "the configured sites actually read. Read through ROAD_SKETCHES_DATA_DIR.",
        "pad_ft": args.pad_ft,
        "sites": sites,
        # A LIST OF BOXES per layer (one per area), which the loader's guard accepts a read inside
        # any one of.
        "extents": {"roads": [list(b) for b in road_boxes], "parcels": [list(b) for b in parcel_boxes]},
        "extent_crs": {"roads": "EPSG:4326", "parcels": NJ_STATE_PLANE_FT},
        "per_site_reads": {site: {"area": r["area"], "roads_wgs84": r["roads_wgs84"],
                                  "parcels_nj_ft": r["parcels_nj_ft"]} for site, r in reads.items()},
        "layers": written,
    }
    (out / FIXTURE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"Wrote {total / 1e6:.2f} MB in {time.perf_counter() - started:.1f}s "
          f"(data/ is {sum(f.stat().st_size for f in DATA_DIR.iterdir() if f.is_file()) / 1e6:.0f} MB)")
    print(f"Use it with: ROAD_SKETCHES_DATA_DIR={out} .venv/bin/python -m pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
