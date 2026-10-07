#!/usr/bin/env python
"""Pull OSM fresh and write an area's osmChange files (src/osm_osc.py says what each holds):

    proposals/<area>/existing.osc          where OSM's tags do not yet say what is painted
    proposals/<area>/two_way_bikeway.osc   existing.osc plus the Broad St two-way bikeway
    proposals/<area>/two_way_bikeway_daylighting.osc
                                           the same with a painted curb extension, edged in
                                           flexible posts, at each parked corner - so the
                                           statute's no-standing zone is 10 ft, not 25

    .venv/bin/python scripts/write_osc.py hopewell_borough            # re-pulls from Overpass
    .venv/bin/python scripts/write_osc.py hopewell_borough --cached   # the last pull

Then render with `render_osm.py --cached`, so the render reads the same pull. Each is a real osmChange: open it in JOSM over the area to check it. render_osm.py renders one
scenario per file.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.osm_osc import existing_markings, two_way_bikeway
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
    parser.add_argument("--cached", action="store_true", help="use the cached OSM pull")
    args = parser.parse_args()
    # Fresh OSM, as render_osm.py does: osm_context re-pulls on the first fetch while this is set.
    if not args.cached:
        os.environ["ROAD_SKETCHES_REFRESH_OSM"] = "1"
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
    extended, _found = existing_markings(args.area, kerb_extensions=True)
    daylit, report = two_way_bikeway(args.area, extended, kerb_extensions=True)
    write_change(daylit, proposal_path(args.area, "two_way_bikeway_daylighting"))
    print(f"two_way_bikeway_daylighting.osc: {describe(daylit)}")
    for key, feet in sorted(report.items()):
        print(f"    Broad St, {key}: {feet}")


if __name__ == "__main__":
    main()
