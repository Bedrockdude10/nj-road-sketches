#!/usr/bin/env python
"""Write Broad St's two-way bikeway as OSM inline tags: proposals/<area>/two_way_bikeway.yaml.

    scripts/propose_bikeway_tags.py [--area hopewell_borough]

Runs the route's ladder (BROAD_ST_TWO_WAY_BIKEWAY) ONCE over the existing world and records, for
each OSM way an approach lies on, the section that approach landed on - in OSM's own schema:

    cycleway:<side>=track, cycleway:<side>:oneway=no, cycleway:<side>:width,
    cycleway:<side>:buffer, cycleway:<side>:separation:left=flex_post,
    parking:<side>:restriction=no_parking

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
from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY, existing_conditions
from src.geometry.treatments.bikeways import AddBikeLaneBollards, AddTwoWayBikeLane
from src.sources.proposals import proposal_path


def _feet(value_ft: float) -> str:
    """OSM's feet notation, so the width the ladder chose is the width written: `8'`."""
    return f"{value_ft:g}'"


def proposed_tags(area: str) -> list[dict]:
    """[{element, tags, source}] - one entry per way side the ladder placed the bikeway on."""
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
    out = []
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
              "# own (OSM's convention). Schema: src/sources/observations.py.\n")
    path.write_text(header + yaml.safe_dump({"observations": entries}, sort_keys=False,
                                            allow_unicode=True, width=100))
    print(f"wrote {path} ({len(entries)} way side(s))")


if __name__ == "__main__":
    main()
