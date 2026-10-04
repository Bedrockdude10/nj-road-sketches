#!/usr/bin/env python
"""Build one area's world once, save it, and draw every site as a view onto it.

    scripts/build_world.py                                  # hopewell_borough, every scenario
    scripts/build_world.py --scenario two_way_bikeway --no-3d

The world is the area's network document (scripts/export_network.py) designed ONCE per scenario:
every street, every kerb, every crossing, every building. Nothing about a site decides what
exists or what the design is. A site is only a camera - its 2D sheet is the world drawn inside
the site's frame, and its 3D still is a camera placed in the saved .blend at that same frame.

    output/world/<area>/<scenario>.json          the world's geometry, local metres
    output/world/<area>/<scenario>.blend         the built 3D world, textures packed
    output/world/<area>/<scenario>/<site>.png    3D still per site camera
    output/world/<area>/<scenario>/<site>_2d.png 2D sheet per site view
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # after matplotlib.use: the backend must be set first

from scripts.phase4_render_3d import build_world, find_blender, render_cameras
from scripts.render_slice import SCENARIOS, design_for, load_network, window_frame
from src.geometry.intersection.load import load_intersection_model
from src.render.export import export_scenario
from src.render.frame import Frame, junction_frame
from src.render.plan_view import plot_design_state
from src.site import list_sites, load_site_config

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "world"


def site_views(area: str) -> dict[str, Frame]:
    """{site: the frame its renders have always used} for every site that is a view onto `area`.

    The frame comes from the site's own junction - the same `junction_frame` a site render has
    always pointed at - so a site's picture of the world is framed exactly as its picture of
    itself was. Sites that fail to load are reported and skipped, not fatal: a view that cannot
    be placed does not stop the world being built.
    """
    views = {}
    for site in list_sites():
        if load_site_config(site)["intersection"].get("osm_area") != area:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                views[site] = junction_frame(load_intersection_model(site=site))
        except Exception as e:   # reported; the world does not depend on any one view
            print(f"  {site}: no view - {type(e).__name__}: {str(e)[:160]}")
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS),
                        help="repeatable; default is every scenario")
    parser.add_argument("--no-3d", action="store_true", help="skip Blender")
    parser.add_argument("--no-views", action="store_true", help="build the world only")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    network = load_network(args.area)
    streets = network[network["kind"].isin(["street", "pavement"])]
    views = {} if args.no_views else site_views(args.area)
    out = OUT_DIR / args.area
    out.mkdir(parents=True, exist_ok=True)
    blender = None if args.no_3d else find_blender()

    for scenario in args.scenario or sorted(SCENARIOS):
        t = time.perf_counter()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            model, state, pavement = design_for(network, args.area, scenario)
            world_json = export_scenario(model, state, f"{args.area}_{scenario}",
                                         out / f"{scenario}.json", pavement=pavement,
                                         frame=window_frame(streets))
        (out / f"{scenario}.log").write_text(log.getvalue())
        print(f"{scenario}: designed and exported in {time.perf_counter() - t:.1f}s "
              f"({len(state.legs)} legs, {world_json.stat().st_size / 1e6:.1f} MB)")

        scenario_dir = out / scenario
        scenario_dir.mkdir(exist_ok=True)
        for site, frame in views.items():
            fig, ax = plt.subplots(figsize=(11, 11))
            plot_design_state(ax, model, state, f"{site} - {scenario}", pavement=pavement,
                              frame=frame)
            fig.savefig(scenario_dir / f"{site}_2d.png", dpi=args.dpi, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)

        if blender is None:
            continue
        t = time.perf_counter()
        blend = build_world(blender, world_json, out / f"{scenario}.blend")
        print(f"{scenario}: built {blend.name} in {time.perf_counter() - t:.1f}s "
              f"({blend.stat().st_size / 1e6:.0f} MB)")
        if views:
            cameras = [{"name": site, **frame.as_local_m(model.center_ft)}
                       for site, frame in views.items()]
            t = time.perf_counter()
            render_cameras(blender, blend, cameras, scenario_dir)
            print(f"{scenario}: {len(cameras)} site cameras rendered in "
                  f"{time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
