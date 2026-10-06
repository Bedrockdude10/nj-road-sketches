#!/usr/bin/env python
"""Write Broad St's two-way bikeway as OSM inline tags: proposals/<area>/two_way_bikeway.yaml.

    scripts/propose_bikeway_tags.py [--area hopewell_borough]

Runs the route's ladder (BROAD_ST_TWO_WAY_BIKEWAY) ONCE over the existing world and records, for
each OSM way an approach lies on, the section that approach landed on - in OSM's own schema:

    cycleway:<side>=track, cycleway:<side>:oneway=no, cycleway:<side>:width,
    cycleway:<side>:buffer, cycleway:<side>:separation:left=flex_post,
    parking:<side>:restriction=no_parking

and the far kerb's hatching as what OSM maps hatching as: a new closed way (negative id, OSM's
convention for an element not yet uploaded) tagged road_marking=restriction + pattern=chevron,
traced off the zone the ladder's design actually painted. Far-kerb PARKING is not written yet:
see proposed_tags.

`<side>` is the WAY's side, as OSM's always is. A way carrying several approaches gets the
narrowest section any of them took, because one way is one set of tags. From then on THE FILE IS
THE PROPOSAL: the two_way_bikeway scenario draws what it says (render_slice.SCENARIOS), so a width
is changed by editing the file, not by re-running this. Re-running overwrites it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from scripts.render_slice import load_network, slice_context
from src.geometry.network.slice_design import slice_design
from src.geometry.targets import Side
from src.geometry.treatments import (BROAD_ST_TWO_WAY_BIKEWAY, LaneNarrowing, MarkedParking,
                                     existing_conditions)
from src.geometry.treatments.bikeways import AddBikeLaneBollards, AddTwoWayBikeLane
from src.sources.proposals import proposal_path


def _feet(value_ft: float) -> str:
    """OSM's feet notation, so the width the ladder chose is the width written: `8'`."""
    return f"{value_ft:g}'"


#: The kinds a hatched kerbside zone is painted with (LaneNarrowing.paint): its fill, its taper,
#: and the lines that bound them. Their union is the area a mapper would trace.
_HATCH_KINDS = ("lane_narrowing_fill", "taper_fill", "lane_edge_line", "taper_line",
                "zone_end_line")


def _hatched_areas(model, state, network, carrying: set[str]) -> list[dict]:
    """[{element, tags, coords, source}] - each far-kerb hatched zone the ladder's design PAINTS,
    as one new road_marking=restriction way. Traced off the drawn paint rather than rebuilt from
    the treatment's numbers, so the area is the zone as drawn - cut at its crossings and swept
    round its openings - and redrawing it from the way reproduces it."""
    import pyproj
    from shapely.ops import unary_union

    from src.geometry.model import NJ_STATE_PLANE_FT, WGS84
    from src.geometry.network.slice_design import slice_pavement
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT, MIN_ZONE_AREA_SQ_FT
    from src.render.props import build_props
    from src.render.scene import SceneGeometry

    hatched = {(zone.target.leg, str(side)) for zone in state.treatments_of(LaneNarrowing)
               if zone.target.leg in carrying and not zone.line_only for side in zone.sides}
    if not hatched:
        return []
    pavement = slice_pavement(network, state.corner_fillets, state.legs, model.osm)
    with contextlib.redirect_stdout(io.StringIO()):
        scene = SceneGeometry.resolve(model, state, pavement=pavement)
        props = build_props(model, state, scene.crosswalk_offsets, pavement=scene.pavement)
        paint, _ = scene.build_paint_and_posts(props)
    to_wgs84 = pyproj.Transformer.from_crs(NJ_STATE_PLANE_FT, WGS84, always_xy=True)
    out, next_id = [], -1
    for leg_name, side in sorted(hatched):
        pieces = [p for p in paint if p.leg == leg_name and str(p.side) == side
                  and str(p.kind) in _HATCH_KINDS]
        shapes = [p.geometry if p.kind.covers_area
                  else p.geometry.buffer(LANE_EDGE_LINE_WIDTH_FT / 2, cap_style=2, join_style=2)
                  for p in pieces]
        if not shapes:
            continue
        # Closed by a hair and opened again, so the line laid along the fill's edge and the
        # fill itself read as one area rather than two that touch.
        union = unary_union(shapes).buffer(0.01, join_style=2).buffer(-0.01, join_style=2)
        for part in getattr(union, "geoms", [union]):
            if part.geom_type != "Polygon" or part.area < MIN_ZONE_AREA_SQ_FT:
                continue
            xs, ys = zip(*part.exterior.coords)
            lons, lats = to_wgs84.transform(xs, ys)
            out.append({"element": f"way/{next_id}",
                        "tags": {"road_marking": "restriction", "pattern": "chevron",
                                 "colour": "white"},
                        "coords": [[round(lon, 8), round(lat, 8)] for lon, lat in zip(lons, lats)],
                        "source": f"Proposal: the hatched far kerb of Broad St's two-way "
                                  f"bikeway, as BROAD_ST_TWO_WAY_BIKEWAY's ladder painted it on "
                                  f"{leg_name} {side} (scripts/propose_bikeway_tags.py). "
                                  f"Not a survey."})
            next_id -= 1
    return out


def proposed_tags(area: str) -> list[dict]:
    """[{element, tags, source}] - one entry per way side the ladder placed the bikeway on,
    and one new way per far-kerb hatched zone it painted."""
    network = load_network(area)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = slice_design(network, osm=slice_context(network), osm_area=area)
        state = BROAD_ST_TWO_WAY_BIKEWAY.apply_to(existing_conditions(model), model, quiet=True)
    posts = {(t.target.leg, str(t.target.side)) for t in state.treatments_of(AddBikeLaneBollards)}
    by_way: dict[tuple[int, str], dict] = {}
    for lane in state.treatments_of(AddTwoWayBikeLane):
        leg_name, side = lane.target.leg, str(lane.target.side)
        leg = model.legs[leg_name]
        aligned = model.leg_osm_aligned.get(leg_name, True)
        osm_side = side if aligned else str(Side(side).other)
        entry = by_way.setdefault((leg.osm_way_id, osm_side),
                                  {"width_ft": lane.width_ft, "buffer_ft": lane.buffer_ft,
                                   "posts": True, "legs": []})
        entry["width_ft"] = min(entry["width_ft"], lane.width_ft)
        entry["buffer_ft"] = min(entry["buffer_ft"], lane.buffer_ft)
        entry["posts"] = entry["posts"] and (leg_name, side) in posts
        entry["legs"].append(leg_name)
    carrying = {lane.target.leg for lane in state.treatments_of(AddTwoWayBikeLane)}
    out = [*_hatched_areas(model, state, network, carrying)]
    # FAR-KERB PARKING IS NOT WRITTEN. The ladder sizes it per leg, and one OSM way carries
    # several legs (way 27459436 carries broad_street_6..9): one parking tag for the way put a
    # 7.7 ft stall on legs where the ladder had hatched for want of room. The wiki's answer
    # (Street parking) is to split the way, or to map the lane as its own amenity=parking +
    # parking=lane area with parking:<side>=separate on the street - not chosen yet.
    stalls = sum(1 for zone in state.treatments_of(MarkedParking)
                 if zone.target.leg in carrying and not zone.curb_offset_ft)
    if stalls:
        print(f"  {stalls} far-kerb parking lane(s) the ladder chose are NOT written - the far "
              f"kerb's parking stays as OSM records it.")
    for (way_id, osm_side), entry in sorted(by_way.items()):
        key = f"cycleway:{osm_side}"
        tags = {key: "track", f"{key}:oneway": "no", f"{key}:width": _feet(entry["width_ft"]),
                f"{key}:buffer": _feet(entry["buffer_ft"]),
                f"parking:{osm_side}:restriction": "no_parking"}
        if entry["posts"]:
            tags[f"{key}:separation:left"] = "flex_post"
        out.append({"element": f"way/{way_id}", "tags": tags,
                    "source": f"Proposal: Broad St two-way protected bikeway, the rung "
                              f"BROAD_ST_TWO_WAY_BIKEWAY's ladder chose on "
                              f"{', '.join(sorted(entry['legs']))} "
                              f"(scripts/propose_bikeway_tags.py). Not a survey."})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    args = parser.parse_args()
    entries = proposed_tags(args.area)
    path = proposal_path(args.area, "two_way_bikeway")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Broad St's two-way protected bikeway, AS OSM TAGS on the ways it runs along.\n"
              "# Written by scripts/propose_bikeway_tags.py from the route's ladder; from then on\n"
              "# this file IS the proposal - edit a width here, not in code. Sides are each way's\n"
              "# own (OSM's convention); a negative id is a way the proposal adds.\n"
              "# Schema: src/sources/proposals.py.\n")
    path.write_text(header + yaml.safe_dump({"observations": entries}, sort_keys=False,
                                            allow_unicode=True, width=100))
    print(f"wrote {path} ({len(entries)} way side(s))")


if __name__ == "__main__":
    main()
