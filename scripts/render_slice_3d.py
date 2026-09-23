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
from src.render.coords import pt_to_local_m, ring_to_local_m

FT_TO_M = 0.3048

#: A traced kerb with no OSM `kerb=` tag on it. RAISED rather than unknown: every run in the
#: document came off a `barrier=kerb` way, and drawing those flush would delete the one feature
#: a 3D still exists to show. src/render/export.py:KERB_HEIGHT_M is the tagged version.
UNTAGGED_KERB_HEIGHT_M = 0.20

#: Which `kind` in the document feeds which channel in the Blender contract. Polygons become
#: rings, lines become polylines; `bollard` and `pavement` are shaped differently and handled
#: in `scene_document`. A kind absent here is deliberately not in the 3D scene - `street` is the
#: centreline (an abstraction, not a thing on the ground) and `crossing` is a bare point with no
#: width or skew to draw, so drawing either would be an invention.
CHANNEL_OF: dict[str, str] = {
    "bikeway":        "bike_lane_surface_polygons",
    "bikeway_buffer": "bike_lane_uncoloured_surface_polygons",
    "edge_line":      "bike_lane_edge_lines",
}


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
                 "pavement_near": [], "kerbs": [], "props": []}
    for channel in CHANNEL_OF.values():
        doc[channel] = []

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
        elif (channel := CHANNEL_OF.get(kind)) is not None:
            shapes = _rings(geom) if channel.endswith("_polygons") else _lines(geom)
            doc[channel] += [ring_to_local_m(s, center_ft) for s in shapes]
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
