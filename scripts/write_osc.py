#!/usr/bin/env python
"""Write an area's osmChange files from the cached OSM pull (src/osm_osc.py says what each holds):

    proposals/<area>/existing.osc          where OSM's tags do not yet say what is painted
    proposals/<area>/two_way_bikeway.osc   existing.osc plus the Broad St two-way bikeway
    proposals/<area>/two_way_bikeway_daylighting.osc
                                           the same with a painted curb extension, edged in
                                           flexible posts, at each parked corner - so the
                                           statute's no-standing zone is 10 ft, not 25

    .venv/bin/python scripts/write_osc.py hopewell_borough

Each is a real osmChange: open it in JOSM over the area to check it. render_osm.py renders one
scenario per file.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.osm_osc import NO_STANDING_WITH_EXTENSION_FT, existing_markings, two_way_bikeway
from src.sources.osm_change import OsmChange, proposal_path, write_change


def describe(change: OsmChange) -> str:
    actions = Counter(w.action for w in change.ways)
    dark = sum(1 for w in change.ways if w.tags.get("lane_markings") == "no")
    hatched = sum(1 for w in change.ways
                  if "hatched" in (w.tags.get("shoulder:left:markings"), w.tags.get("shoulder:right:markings")))
    widths = sum(1 for w in change.ways if "width" in w.tags and "highway" in w.tags)
    return (f"{len(change.nodes)} new nodes, {actions['modify']} ways modified, "
            f"{actions['create']} created ({widths} with a width, {dark} tagged "
            f"lane_markings=no, {hatched} with a hatched shoulder)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("area")
    args = parser.parse_args()
    existing, found = existing_markings(args.area)
    write_change(existing, proposal_path(args.area, "existing"))
    print(f"existing.osc: {describe(existing)}")
    for key, count in sorted(found.items()):
        print(f"    {key}: {count}")
    bikeway, report = two_way_bikeway(args.area, existing)
    write_change(bikeway, proposal_path(args.area, "two_way_bikeway"))
    print(f"two_way_bikeway.osc: {describe(bikeway)}")
    for key, feet in sorted(report.items()):
        print(f"    Broad St, {key}: {feet}")
    extended, _found = existing_markings(args.area, no_standing_ft=NO_STANDING_WITH_EXTENSION_FT)
    daylit, report = two_way_bikeway(args.area, extended, kerb_extensions=True)
    write_change(daylit, proposal_path(args.area, "two_way_bikeway_daylighting"))
    print(f"two_way_bikeway_daylighting.osc: {describe(daylit)}")
    for key, feet in sorted(report.items()):
        print(f"    Broad St, {key}: {feet}")


if __name__ == "__main__":
    main()
