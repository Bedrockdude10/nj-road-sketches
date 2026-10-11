#!/usr/bin/env python
"""Pull OSM fresh, draw it, render it in Blender: existing conditions and every proposal.

    .venv/bin/python scripts/render_osm.py hopewell_borough
    .venv/bin/python scripts/render_osm.py hopewell_borough --cached --jobs 4

Steps, in order:
  1. re-pull the area from the OSM API (skip with --cached)
  2. write proposals/<area>/*.osc from that pull (src/osm_osc.py:write_changes)
  3. write output/osm/<area>/<scenario>.json, one world per .osc (src/osm_world.py)
  4. blender --build each world into <scenario>.blend
  5. blender --open each .blend and render every camera areas.yaml gives the area - each a junction,
     by its OSM node - to <scenario>/<camera>.png

--jobs N runs up to N Blender processes at once. Blender is BLENDER_BIN, else `blender` on PATH,
else the default macOS install.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BLENDER_SCENE = REPO_ROOT / "scripts" / "blender" / "blender_scene.py"
MAC_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
OUTPUT_DIR = REPO_ROOT / "output" / "osm"


def find_blender() -> str:
    for candidate in (os.environ.get("BLENDER_BIN"), shutil.which("blender"), MAC_BLENDER):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Blender not found: set BLENDER_BIN, or put `blender` on PATH.")


def blender(binary: str, *args: str) -> None:
    started = time.perf_counter()
    result = subprocess.run([binary, "--background", "--python", str(BLENDER_SCENE), "--", *args],
                            capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if line.startswith(("WORLD_SAVED", "RENDER_DONE")):
            print(f"  {line}")
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise SystemExit(f"blender {' '.join(args)} failed ({result.returncode}):\n{tail}")
    print(f"  blender {args[0]} {Path(args[1]).name}: {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("area")
    parser.add_argument("--jobs", type=int, default=1, help="Blender processes at once")
    parser.add_argument("--cached", action="store_true", help="use the cached OSM pull")
    parser.add_argument("--no-render", action="store_true", help="write the .osc and world JSON only")
    args = parser.parse_args()
    if not args.cached:
        os.environ["ROAD_SKETCHES_REFRESH_OSM"] = "1"   # osm_context re-pulls on the first fetch
    from src.osm_osc import write_changes
    from src.osm_world import LocalFrame, osm_world
    from src.sources.osm_change import PROPOSALS_DIR
    from src.sources.osm_context import AREAS, fetch_borough_osm
    from src.standards import si

    if args.area not in AREAS:
        raise SystemExit(f"unknown area {args.area!r}; declared: {', '.join(sorted(AREAS))}")
    bbox = AREAS[args.area]["bbox"]
    write_changes(args.area)
    frame, nodes = LocalFrame(bbox), fetch_borough_osm(bbox)["nodes"]
    cameras = [{"name": name, "center_m": frame.point(nodes[node]["lon"], nodes[node]["lat"]),
                "radius_m": si("render.camera_radius")}
               for name, node in AREAS[args.area]["cameras"].items()]
    out = OUTPUT_DIR / args.area
    out.mkdir(parents=True, exist_ok=True)
    worlds = {}
    for proposal in sorted((PROPOSALS_DIR / args.area).glob("*.osc")):
        world = osm_world(args.area, proposal)
        stats = world.pop("stats")
        worlds[proposal.stem] = path = out / f"{proposal.stem}.json"
        path.write_text(json.dumps(world))
        print(f"{proposal.stem}: wrote {path.relative_to(REPO_ROOT)}")
        for key, count in sorted(stats.items()):
            print(f"    {count:5d}  {key}")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2))
    if args.no_render or not cameras:
        return

    binary = find_blender()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        blends = {scenario: out / f"{scenario}.blend" for scenario in worlds}
        list(pool.map(lambda s: blender(binary, "--build", str(worlds[s]), "--save", str(blends[s])), worlds))
        per_process = max(1, -(-len(cameras) * len(worlds) // max(1, args.jobs)))   # ceil
        renders = []
        for scenario, blend in blends.items():
            for i in range(0, len(cameras), per_process):
                chunk = out / f"cameras_{scenario}_{i}.json"
                chunk.write_text(json.dumps(cameras[i:i + per_process]))
                renders.append(("--open", str(blend), "--camera", str(chunk), "--out-dir", str(out / scenario)))
        list(pool.map(lambda job: blender(binary, *job), renders))
    print(f"renders in {out.relative_to(REPO_ROOT)}/<scenario>/")


if __name__ == "__main__":
    main()
