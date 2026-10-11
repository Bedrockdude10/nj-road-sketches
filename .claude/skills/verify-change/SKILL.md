---
name: verify-change
description: Verify a change to this repo's tags, .osc files, drawing or renders before calling it done. Use after editing anything under src/ or scripts/, or when asked whether a change is correct. Runs the measure/test/render loop in cost order.
---

# Verifying a change here

In order; each is cheaper than the next and catches what the earlier ones cannot.

## 0. Is it prose only?

```bash
.venv/bin/python scripts/check_prose_only.py --base <your-branch-point>
```

Exit 0 means no behaviour changed and the rest is not required.

## 1. Measure what it did

```bash
.venv/bin/python scripts/verify.py save before      # before the change (git stash, or first)
.venv/bin/python scripts/verify.py save after
.venv/bin/python scripts/verify.py compare before after
```

Every `.osc`, report line and world channel that moved. A change meant to move nothing shows every
`.osc` identical. Explain each one that moved - put the old rule back and watch the difference go.

## 2. The tests

```bash
./scripts/test.sh tests/test_<what you touched>.py   # while iterating
FULL_SUITE=1 ./scripts/test.sh                        # once, at the end
```

Never bare `pytest`. Includes lint, import contracts and mypy.

## 3. Render once, and open the PNGs

```bash
.venv/bin/python scripts/render_osm.py <area> --cached
```

Say so before running it - it takes minutes. Actually look at the pictures.
