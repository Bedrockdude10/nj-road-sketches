#!/usr/bin/env bash
# PreToolUse hook: refuse a full-suite run from inside the edit loop.
#
# Nothing in the suite is slow on its own - the cost is how often it runs. Measured over 30 sessions, Claude ran it whole 475 times, ~16 a session and 26 in
# one, which was the largest single tool cost here. The full suite belongs once, at the end,
# through scripts/verify.py (which runs it itself and is not affected by this hook).
#
# A run is let through if it names a test path, a node id, -k, or a last-failed/stepwise
# flag, or if the segment carries FULL_SUITE=1. Only Claude's tool calls pass through here;
# ./scripts/test.sh typed by hand is unchanged.
#
# Exit 2 is the contract Claude Code reads as "blocking feedback" - stderr goes back to the
# model. Any other failure (no jq, bad payload) exits 0 and stays out of the way.
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0
CMD="$(jq -r '.tool_input.command // empty' 2>/dev/null)" || exit 0
[[ -n "$CMD" ]] || exit 0

# Invocations: ./scripts/test.sh, scripts/test.sh, pytest, .venv/bin/pytest, python -m pytest,
# optionally behind `time`/`timeout N` and env assignments, and followed by nothing, a flag, a
# redirection or a path - so prose that merely starts with the word "pytest" is not a run.
RUNS='^(time[[:space:]]+|timeout[[:space:]]+[^[:space:]]+[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*([^[:space:]]*/)?(scripts/test\.sh|pytest|python3?[[:space:]]+-m[[:space:]]+pytest)([[:space:]]*$|[[:space:]]+(-|[0-9]*[<>]|[^[:space:]]*/|[^[:space:]]*\.py))'
NARROWED='[[:space:]][^[:space:]]*tests/[^[:space:]]+|::|[[:space:]]-k|--lf([[:space:]]|$)|--last-failed|--sw([[:space:]]|$)|--stepwise|--co([[:space:]]|$)|--collect-only|--help|--fixtures'

while IFS= read -r SEG; do
    SEG="${SEG#"${SEG%%[![:space:]]*}"}"
    [[ "$SEG" =~ $RUNS ]] || continue
    [[ "$SEG" == *FULL_SUITE=1* ]] && continue
    [[ "$SEG" =~ $NARROWED ]] && continue
    cat >&2 <<'EOF'
Full suite blocked inside the edit loop. While iterating, run the tests that can see your
change (./scripts/test.sh tests/test_x.py, or -k <expr>), and measure_drawn.py /
`scripts/verify.py --no-tests --site <site>` for geometry. The full suite runs ONCE,
at the end, through scripts/verify.py. If you genuinely need it now, prefix FULL_SUITE=1.
EOF
    exit 2
# Heredoc bodies and quoted strings are data (a file being written, a commit message, a grep
# pattern), not commands: stop at the first line that opens a heredoc, and drop quoted spans
# before splitting on the shell's separators.
done < <(printf '%s\n' "$CMD" | awk '{print} /<</{exit}' \
         | perl -0pe "s/'[^']*'//g; s/\"(?:\\\\.|[^\"\\\\])*\"//g; s/(&&|\|\||;|\|)/\n/g")
exit 0
