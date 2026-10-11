#!/usr/bin/env python
"""What a change did, measured: every area's .osc, its report, and every scenario's drawn world,
saved before and after and compared - so a change is known by its numbers, not a render.

    .venv/bin/python scripts/verify.py save before        # on the code before the change
    .venv/bin/python scripts/verify.py save after         # on the code after it
    .venv/bin/python scripts/verify.py compare before after

`save` writes output/verify/<tag>/ from the cached pull (src/osm_osc.py:write_changes, into that
folder - proposals/ is not touched) and draws the worlds from the committed proposals/*.osc, so
the reader is compared on fixed input. `compare` prints each .osc identical or changed, each report
line that moved, and for each world channel how many pieces went and came.
"""
from __future__ import annotations

import filecmp
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "output" / "verify"


def save(tag: str) -> None:
    from src.osm_osc import write_changes
    from src.osm_world import osm_world
    from src.sources.osm_change import PROPOSALS_DIR
    from src.sources.osm_context import AREAS

    root = OUT / tag
    for area in AREAS:
        reports = write_changes(area, into=root / "osc")
        (root / "osc" / area / "reports.json").write_text(
            json.dumps({k: dict(v) for k, v in reports.items()}, indent=1, sort_keys=True, default=int))
        (root / "world" / area).mkdir(parents=True, exist_ok=True)
        for osc in sorted((PROPOSALS_DIR / area).glob("*.osc")):
            world = osm_world(area, osc)
            world.pop("theme", None)
            (root / "world" / area / f"{osc.stem}.json").write_text(json.dumps(world, sort_keys=True, default=float))
            print(f"{area}/{osc.stem}: saved", flush=True)


def compare(a: str, b: str) -> None:
    A, B = OUT / a, OUT / b
    for area in sorted(p.name for p in (A / "osc").iterdir()):
        for f in sorted((A / "osc" / area).glob("*.osc")):
            other = B / "osc" / area / f.name
            print(f"osc {area}/{f.name}:", "identical" if other.exists() and filecmp.cmp(f, other, shallow=False)
                  else "CHANGED")
        ra, rb = (json.loads((x / "osc" / area / "reports.json").read_text()) for x in (A, B))
        for name in sorted(set(ra) | set(rb)):
            for k in sorted(set(ra.get(name, {})) | set(rb.get(name, {}))):
                if ra.get(name, {}).get(k) != rb.get(name, {}).get(k):
                    print(f"   report {name}: {ra.get(name, {}).get(k, 0)} -> {rb.get(name, {}).get(k, 0)}  {k}")
        for f in sorted((A / "world" / area).glob("*.json")):
            wa, wb = json.loads(f.read_text()), json.loads((B / "world" / area / f.name).read_text())
            changed = [k for k in sorted(set(wa) | set(wb)) if wa.get(k) != wb.get(k)]
            print(f"world {area}/{f.stem}:", "identical" if not changed else "CHANGED")
            for k in changed:
                x, y = wa.get(k) or {}, wb.get(k) or {}
                if k == "stats":
                    for s in sorted(set(x) | set(y)):
                        if x.get(s) != y.get(s):
                            print(f"   stats {x.get(s, 0)} -> {y.get(s, 0)}  {s}")
                elif isinstance(x, list) and isinstance(y, list):
                    cx, cy = (Counter(json.dumps(i, sort_keys=True) for i in v) for v in (x, y))
                    print(f"   {k}: {len(x)} -> {len(y)} pieces; {sum((cx - cy).values())} gone, "
                          f"{sum((cy - cx).values())} new")
                else:
                    print(f"   {k}: changed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
