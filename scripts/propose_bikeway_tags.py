#!/usr/bin/env python
"""Write Broad St's two-way bikeway as an osmChange: proposals/<area>/two_way_bikeway.osc.

The design is written by src/geometry/treatments/propose.py:proposal_from_design; from then on
the file IS the proposal."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.render_slice import load_network, slice_context
from src.geometry.network.slice_design import slice_design
from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY, existing_conditions
from src.geometry.treatments.propose import proposal_from_design
from src.sources.osm_change import proposal_path, write_change

NOTE = ("Proposal: Broad St two-way protected bikeway, from BROAD_ST_TWO_WAY_BIKEWAY's ladder"
        " (scripts/propose_bikeway_tags.py). Not a survey.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    args = parser.parse_args()
    network = load_network(args.area)
    osm = slice_context(network)
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = slice_design(network, osm=osm, osm_area=args.area)
        state = BROAD_ST_TWO_WAY_BIKEWAY.apply_to(existing_conditions(model), model, quiet=True)
    change = proposal_from_design(model, state, osm, note=NOTE)
    path = proposal_path(args.area, "two_way_bikeway")
    write_change(change, path)
    splits = sum(1 for w in change.ways if w.action == "create" and "highway" in w.tags)
    print(f"wrote {path} ({len(change.ways)} ways, {len(change.nodes)} nodes, {splits} split pieces)")


if __name__ == "__main__":
    main()
