#!/usr/bin/env python
"""Render a 3D still from a SLICE of a network GeoJSON. No site, no scenario, no model.

The 3D half of `render_slice.py`: the same crop of the same document, handed to the same
`blender_scene.py` the per-site pipeline drives. So a sheet and a still are two drawings of one
slice, and neither can show something the document does not hold.

    scripts/render_slice_3d.py --street "Broad Street"
    scripts/render_slice_3d.py --around=-74.76196,40.38918 --radius-ft 300 --name broad_greenwood

`blender_scene.py` reads a LOCAL METRE frame, so the slice's centre becomes the origin here -
see `scene_document`. That contract is the site exporter's (src/render/export.py); this module
only translates a document already built into it, and computes no geometry of its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.geometry import Point

from scripts.render_slice import (NETWORK_DIR, OUT_DIR, load_network, slice_around,
                                  slice_for_street, _center_ft)
from src.geometry.markings import KINDS
from src.geometry.paint import PaintPiece
from src.render.coords import pt_to_local_m, ring_to_local_m
from src.render.export import SIDEWALK_WIDTH_FT, paint_channels_local_m
from src.render.mesh_utils import build_decimated_building_mesh

FT_TO_M = 0.3048

#: A traced kerb with no OSM `kerb=` tag on it. RAISED rather than unknown: every run in the
#: document came off a `barrier=kerb` way, and drawing those flush would delete the one feature
#: a 3D still exists to show. src/render/export.py:KERB_HEIGHT_M is the tagged version.
UNTAGGED_KERB_HEIGHT_M = 0.20


def _rings(geom) -> list[list]:
    """Every exterior ring in a geometry - a clip can split one polygon into several."""
    parts = getattr(geom, "geoms", [geom])
    return [list(p.exterior.coords) for p in parts if p.geom_type == "Polygon"]


def _lines(geom) -> list[list]:
    parts = getattr(geom, "geoms", [geom])
    return [list(p.coords) for p in parts if p.geom_type == "LineString"]


def scene_document(features: gpd.GeoDataFrame, name: str) -> dict:
    """The slice as the local-metre document `blender_scene.py` consumes.

    The frame is the slice's own bounds, so the origin is wherever the reader cropped rather
    than a junction node - which is the whole point: a site here is a window, not a place the
    pipeline had to be told about in advance.
    """
    minx, miny, maxx, maxy = features.total_bounds
    center_ft = Point((minx + maxx) / 2, (miny + maxy) / 2)
    radius_m = max(maxx - minx, maxy - miny) / 2 * FT_TO_M

    from src.render.theme import build_default_theme

    doc: dict = {"name": name, "units": "meters", "theme": build_default_theme(), "notes": [],
                 "frame": {"center_m": [0.0, 0.0], "radius_m": round(float(radius_m), 6)},
                 # Both are required by blender_scene.REQUIRED_KEYS and both are legitimately
                 # empty here. The document holds no surveyed crossing GEOMETRY - only a centre
                 # station per cross street, and a band invented from that would be a guess at
                 # its skew - and `paved_surfaces` is driveways and parking aprons, which are a
                 # parcel fact this document does not carry. The CARRIAGEWAY is `pavement_near`.
                 "surveyed_crossings": [], "paved_surfaces": [],
                 "pavement_near": [], "kerbs": [], "props": [],
                 "buildings": [], "sidewalks_near": []}

    # ONE entry holding every bar in the slice, not one per crossing way. blender_scene draws
    # the bars and the lines and reads nothing else off the grouping, and the document files a
    # bar under the crossing it came from only as a `markings` tag - so regrouping here would be
    # rebuilding a structure to hand back something that does not look at it.
    crossing: dict = {"markings": "surveyed", "distance_m": 0.0, "bars": [], "lines": []}
    # Rebuilt as real PaintPieces and handed to the EXPORTER'S OWN serializer, rather than
    # mapped to channels here. A second table would be a second answer to "which channel draws
    # this", and the channel decides the colour and the builder - see SKILLS.md section 3.
    pieces: list[PaintPiece] = []

    for row in features.itertuples():
        kind, geom = row.kind, row.geometry
        if geom is None or geom.is_empty:
            continue
        if kind == "pavement":
            doc["pavement_near"] += [ring_to_local_m(r, center_ft) for r in _rings(geom)]
        elif kind == "kerb":
            doc["kerbs"] += [{"coords": ring_to_local_m(c, center_ft), "kerb": "raised",
                              "height_m": UNTAGGED_KERB_HEIGHT_M} for c in _lines(geom)]
        elif kind == "bollard":
            doc["props"].append({"type": "bollard", "heading_deg": 0.0, "drawn_by_paint": True,
                                 "position_ft": [geom.x, geom.y],
                                 "position_m": pt_to_local_m(geom.x, geom.y, center_ft)})
        elif kind == "building":
            mesh = build_decimated_building_mesh(geom, row.height_m / FT_TO_M)
            if mesh is not None:
                vertices, faces = mesh
                doc["buildings"].append(
                    {"mesh": True, "faces": faces, "height_source": "osm",
                     "vertices_m": [[*pt_to_local_m(x, y, center_ft)[:2], z * FT_TO_M]
                                    for x, y, z in vertices]})
        elif kind == "crossing_bar":
            crossing["bars"] += [ring_to_local_m(r, center_ft) for r in _rings(geom)]
        elif kind == "crossing_line":
            crossing["lines"] += [ring_to_local_m(c, center_ft) for c in _lines(geom)]
        elif kind == "sidewalk":
            doc["sidewalks_near"] += [ring_to_local_m(r, center_ft)
                                      for r in _rings(geom.buffer(SIDEWALK_WIDTH_FT / 2))]
        # A str and not merely present: geopandas fills the column with NaN on every row that
        # carries no paint, so `is not None` lets a street centreline through as a marking.
        elif isinstance(paint_kind := getattr(row, "paint_kind", None), str):
            pieces.append(PaintPiece(kind=KINDS[paint_kind], geometry=geom))
    doc.update(paint_channels_local_m(pieces, center_ft))
    if crossing["bars"] or crossing["lines"]:
        doc["surveyed_crossings"].append(crossing)
    return doc


def render(features: gpd.GeoDataFrame, name: str, out_dir: Path) -> Path:
    from scripts.phase4_render_3d import find_blender, render_all

    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_path, png = out_dir / f"{name}_3d.json", out_dir / f"{name}_3d.png"
    geometry_path.write_text(json.dumps(scene_document(features, name), indent=1))
    render_all(find_blender(), [(geometry_path, png)])
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--street", help="slice to one named street")
    parser.add_argument("--around", help="lon,lat to centre a square window on")
    parser.add_argument("--radius-ft", type=float, default=300.0)
    parser.add_argument("--name", help="output stem (default: derived from the slice)")
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    network = load_network(args.area, args.network_dir)
    if args.street:
        features, stem = slice_for_street(network, args.street), args.street.lower().replace(" ", "_")
    elif args.around:
        features = slice_around(network, _center_ft(args.around), args.radius_ft)
        stem = f"{args.around.replace(',', '_')}_{args.radius_ft:.0f}ft"
    else:
        features, stem = network, args.area

    if features.empty:
        raise SystemExit("that slice is empty - nothing to render")
    stem = args.name or stem
    counts = ", ".join(f"{n} {k}" for k, n in features["kind"].value_counts().items())
    print(f"{stem}: {len(features)} feature(s) - {counts}")
    print(f"wrote {render(features, stem, args.out_dir)}")


if __name__ == "__main__":
    main()
