# CLAUDE.md

A router, not a summary. README.md is the pipeline, standards.toml is every figure and OSM key the
code reads, and `.claude/SKILLS.md` (imported below) is what agents here get wrong.

## Answer at the quantitative layer

**Diagnose from numbers; render to confirm, and only at the end.** Everything drawn exists first as
numbers - the tags, the `.osc`, the world JSON - and a PNG is where you NOTICE a problem, never
where you diagnose one. The shape of a turn: **measure → change → measure → render once.**

## Running things

- **`scripts/verify.py save <tag>` / `compare <a> <b>`** - every area's `.osc`, its report and every
  drawn world, before and after a change: which `.osc` moved, which report line, which channel by
  how many pieces. Save before you change anything; compare after.
- **`./scripts/test.sh`, never bare `pytest`.** It pins the venv. Narrow it while iterating
  (`./scripts/test.sh tests/test_osm_change.py`); a hook refuses an unnarrowed run unless
  `FULL_SUITE=1`.
- **`scripts/whatis.py <symbol>`** before you write a second copy of anything: its signature,
  docstring and every call site.
- **`scripts/check_prose_only.py --base <rev>`** proves a diff changed only comments and
  docstrings.
- **Renders**: `scripts/render_osm.py <area>` - say so before running one; it takes minutes.

## Prose

One home per fact: keep the trap, the datum, the invariant and why not the obvious alternative;
cut the discovery story and anything the code already says. Every figure lives in standards.toml
- value, unit, source, status - and code reads it from there (`src/standards.py`), never as a
literal.

@.claude/SKILLS.md
