# CLAUDE.md

A router, not a summary. README.md is the architecture, STANDARDS.md is the figures register,
and `.claude/SKILLS.md` (imported below) is the list of what agents here actually get wrong.

## Answer at the quantitative layer

**Diagnose from numbers; render to confirm, and only at the end.** Everything this project
draws exists first as numbers, and both of the tools below print them. A render is cheap to
wait on; the cost is believing it.
A PNG is where you NOTICE a problem and never where you diagnose one: one session cropped
renders to three wrong diagnoses in a row on a single complaint, and two of the three were
about the wrong leg. See §0a of SKILLS.md, which is the same rule with the receipts.

So the shape of a turn here is: **measure → change → measure → verify → render once.** A
render inside the loop is a round trip that cannot answer the question you asked it.

## Running things

- **`scripts/measure_drawn.py <site> --scenario <s> --leg <leg> --all`.** The whole
  quantitative layer: what is drawn stationed against the centreline, plus `--section` (what
  the treatment thinks it placed vs the room the kerb gives), `--limiters` (all four things
  deciding where kerbside paint starts), `--gaps` (kerb minus outermost paint, station by
  station), `--lanes` (the drawn centre stripe to the innermost drawn marking, which is the only
  honest way to ask "is this lane 11 ft") and `--continuity` (is the facility one piece, how wide
  are the holes). Narrow with `--leg`/`--kind`; measure at the reader's `--frame-scale`, not at 1x.
- **`scripts/verify.py` before you report done.** One command for the whole loop: exports the
  working tree and `--base` side by side, diffs them, runs the suite, and reports failures as
  NEW / KNOWN / FIXED against a recorded baseline. Narrow it with `--no-tests` and `--site
  <site>`; its suite runs `-n auto`, as `test.sh` does. **NEW is the only number that says anything about your change** - this repo's suite is often red from
  work in flight, and re-deriving whose red it is by hand was costing more than the run.
- **`./scripts/test.sh`, never bare `pytest`.** The script pins the venv interpreter, so a
  wrong `python` on PATH cannot masquerade as a broken repo. It runs `-n auto`; pass `-n 0`
  when you need `-x`, `pdb`, or readable ordering.
- **The full suite runs once per change, at the end, through `verify.py`.** While iterating,
  name the test files or `-k` that can see the change. Re-running the whole suite after every
  edit was the largest tool cost here (~16 runs a session), so a PreToolUse hook
  (`.claude/hooks/no_full_suite_in_loop.sh`) refuses an unnarrowed run; `FULL_SUITE=1` gets
  past it when you mean it.
- **Never `rm` before re-exporting.** `export_all_scenarios.py` and `build_all.py` (2D or 3D)
  clear each built site's old `geometry_*.json` themselves. `rm` needs approval here, and waiting on it once stalled a
  session for hours while nobody was at the keyboard.
- `scripts/whatis.py <symbol>` before you write a second copy of anything: signature,
  the docstring's first line, and every call site. §1 of SKILLS.md is the list of facts that
  already have a home; this is how you find the ones that are not on it.
- `scripts/check_prose_only.py --base <rev>` proves a diff changed only comments and
  docstrings. Exit 0 means no behaviour moved and the suite is not required.

## What CI cannot see

`data/` is a 391 MB licensed download kept out of git, and every golden and whole-site test is
marked `needs_source_data` — those **skip**, not fail, when it is absent. A green tick on
`main` is green over the subset that does not need it, so geometry changes have to be verified
locally.

## Prose

One home per fact: keep the trap, the datum, the invariant and why not the obvious alternative;
cut the discovery story, the session archaeology, and anything the code already says. Published
figures live as rows in STANDARDS.md, not as inline comments.

@.claude/SKILLS.md
