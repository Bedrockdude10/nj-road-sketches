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
    .venv/bin/python scripts/write_osc.py hopewell_borough --existing-only   # existing.osc alone

Every area gets an existing.osc; a proposal is rewritten only where the area already has its file.
render_osm.py writes them all from its own pull before it builds.

Then render with `render_osm.py --cached`, so the render reads the same pull. Each is a real osmChange: open it in JOSM over the area to check it. render_osm.py renders one
scenario per file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.osm_osc import write_changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("area")
    parser.add_argument("--cached", action="store_true", help="use the cached OSM pull")
    parser.add_argument("--existing-only", action="store_true",
                        help="write existing.osc alone, for an area with no proposal")
    args = parser.parse_args()
    # Fresh OSM, as render_osm.py does: osm_context re-pulls on the first fetch while this is set.
    if not args.cached:
        os.environ["ROAD_SKETCHES_REFRESH_OSM"] = "1"
    for name, report in write_changes(args.area, proposals=not args.existing_only).items():
        print(f"{name}.osc:")
        for key, count in sorted(report.items()):
            print(f"    {key}: {count}")

if __name__ == "__main__":
    main()
