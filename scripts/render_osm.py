#!/usr/bin/env python
"""Pull OSM fresh, draw it, render it in Blender: existing conditions and every proposal.

    .venv/bin/python scripts/render_osm.py hopewell_borough
    .venv/bin/python scripts/render_osm.py hopewell_borough --jobs 4 \\
        --camera broad_greenwood,-74.7642,40.3889,60

Steps, in order:
  1. re-pull the area from Overpass (skip with --cached)
  2. write output/osm/<area>/<scenario>.json - one per proposals/<area>/*.osc, and `existing`
     from raw OSM where there is no existing.osc
  3. blender --build each JSON into <scenario>.blend
  4. blender --open each .blend and render every camera to <scenario>/<camera>.png

--jobs N runs up to N Blender processes at once, across scenarios and across cameras. Blender is
BLENDER_BIN, else `blender` on PATH, else the default macOS install.
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
PROPOSALS_DIR = REPO_ROOT / "proposals"
OUTPUT_DIR = REPO_ROOT / "output" / "osm"
DEFAULT_RADIUS_M = 150.0


def find_blender() -> str:
    for candidate in (os.environ.get("BLENDER_BIN"), shutil.which("blender"), MAC_BLENDER):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Blender not found: set BLENDER_BIN, or put `blender` on PATH.")


def blender(binary: str, *args: str) -> None:
    started = time.perf_counter()
    result = subprocess.run([binary, "--background", "--python", str(BLENDER_SCENE), "--", *args],
                            capture_output=True, text=True, check=False)
    done = [line for line in result.stdout.splitlines()
            if line.startswith(("WORLD_SAVED", "WORLD_OPENED", "RENDER_DONE"))]
    for line in done:
        print(f"  {line}")
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise SystemExit(f"blender {' '.join(args)} failed ({result.returncode}):\n{tail}")
    print(f"  blender {args[0]} {Path(args[1]).name}: {time.perf_counter() - started:.0f}s")


def parse_camera(spec: str, frame) -> dict:
    """`name,lon,lat[,radius_m]` -> a camera spec in the world's local metres."""
    name, lon, lat, *radius = spec.split(",")
    center = frame.point(float(lon), float(lat))
    return {"name": name, "center_m": center,
            "radius_m": float(radius[0]) if radius else DEFAULT_RADIUS_M}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("area")
    parser.add_argument("--jobs", type=int, default=1, help="Blender processes at once")
    parser.add_argument("--camera", action="append", default=[],
                        help="name,lon,lat[,radius_m]; repeatable. Default: the area's centre")
    parser.add_argument("--cached", action="store_true", help="use the cached OSM pull")
    parser.add_argument("--no-render", action="store_true", help="write the JSON only")
    args = parser.parse_args()

    # 1. Fresh OSM: osm_context re-pulls from Overpass on the first fetch while this is set.
    if not args.cached:
        os.environ["ROAD_SKETCHES_REFRESH_OSM"] = "1"
    from src.osm_world import LocalFrame, osm_world
    from src.sources.osm_context import SNAPSHOT_AREAS

    if args.area not in SNAPSHOT_AREAS:
        raise SystemExit(f"unknown area {args.area!r}; declared: {', '.join(sorted(SNAPSHOT_AREAS))}")
    frame = LocalFrame(SNAPSHOT_AREAS[args.area])
    cameras = ([parse_camera(spec, frame) for spec in args.camera]
               or [{"name": args.area, "center_m": [0.0, 0.0], "radius_m": DEFAULT_RADIUS_M}])

    # 2. One world JSON per scenario.
    out = OUTPUT_DIR / args.area
    out.mkdir(parents=True, exist_ok=True)
    # One scenario per .osc (scripts/write_osc.py). `existing` is raw OSM unless an existing.osc
    # says what OSM's own tags do not yet say.
    scenarios: dict[str, Path | None] = {"existing": None}
    scenarios |= {p.stem: p for p in sorted((PROPOSALS_DIR / args.area).glob("*.osc"))}
    worlds = {}
    for scenario, proposal in scenarios.items():
        world = osm_world(args.area, proposal)
        stats = world.pop("stats")
        path = out / f"{scenario}.json"
        path.write_text(json.dumps(world))
        worlds[scenario] = path
        print(f"{scenario}: wrote {path.relative_to(REPO_ROOT)}"
              + (f" (proposal {proposal.relative_to(REPO_ROOT)})" if proposal else ""))
        for key, count in sorted(stats.items()):
            print(f"    {count:5d}  {key}")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2))
    if args.no_render:
        return

    # 3. Build each world, then 4. render its cameras - N Blender processes at a time.
    binary = find_blender()
    jobs = max(1, args.jobs)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        blends = {scenario: out / f"{scenario}.blend" for scenario in worlds}
        list(pool.map(lambda s: blender(binary, "--build", str(worlds[s]), "--save", str(blends[s])),
                      worlds))
        renders = []
        per_process = max(1, -(-len(cameras) * len(worlds) // jobs))   # ceil
        for scenario, blend in blends.items():
            for i in range(0, len(cameras), per_process):
                chunk = out / f"cameras_{scenario}_{i}.json"
                chunk.write_text(json.dumps(cameras[i:i + per_process]))
                renders.append(("--open", str(blend), "--camera", str(chunk),
                                "--out-dir", str(out / scenario)))
        list(pool.map(lambda job: blender(binary, *job), renders))
    print(f"renders in {out.relative_to(REPO_ROOT)}/<scenario>/")


if __name__ == "__main__":
    main()
