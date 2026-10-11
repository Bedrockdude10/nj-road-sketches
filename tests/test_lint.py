"""Static checks for the errors that only surface on a path nobody ran - a Blender script, a
proposal only one area has: ruff (ruff.toml) over every .py file, import-linter (.importlinter)
over the import graph, and mypy (mypy.ini) over its typed modules. A checker that cannot run is a
failure, not a pass: every one is pinned in requirements.txt, and this says how to install it."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["src", "scripts", "tests"]
RUFF = Path(sys.executable).parent / "ruff"
LINT_IMPORTS = Path(sys.executable).parent / "lint-imports"
MYPY = Path(sys.executable).parent / "mypy"

# The subset that is a guaranteed crash on whatever path reaches it, as opposed to the rest
# of ruff.toml's selection, which is code that works but shouldn't be written that way.
CRASHING = {
    "F821",   # undefined name
    "F822",   # undefined name in __all__
    "F823",   # local variable referenced before assignment
    "E999",   # syntax error - the file does not parse at all
}


def _ruff(*args) -> list[dict]:
    """Every ruff finding over TARGETS, as parsed JSON.

    Raises rather than returns on anything that isn't ruff working normally - see the module
    docstring for why a checker that can't run must not read as a clean checkout.
    """
    if not RUFF.exists():
        raise AssertionError(
            f"ruff is not installed in this interpreter's environment ({sys.executable}).\n"
            f"  expected: {RUFF}\n\n"
            "This test cannot pass without it - a lint guard that skips itself when the "
            "linter is missing is how the bug it exists to catch got shipped. Install the "
            "dev tooling:\n\n"
            "  .venv/bin/pip install -r requirements.txt"
        )
    result = subprocess.run([str(RUFF), "check", "--output-format=json", *args, *TARGETS],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    # 0 = clean, 1 = findings. Anything else (2 = bad config/arguments) means the result says
    # nothing about the code, so it must not be read as an absence of findings.
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"ruff could not run (exit {result.returncode}) - this is a tooling failure, not a "
            f"clean checkout:\n{result.stderr.strip()[:1000]}"
        )
    return json.loads(result.stdout or "[]")


def _format(findings: list[dict]) -> str:
    return "\n  ".join(
        f"{Path(f['filename']).relative_to(REPO_ROOT)}:{(f.get('location') or {}).get('row', '?')}: "
        f"{f['code'] or 'E999'} {f['message']}"
        for f in findings
    )


def test_no_undefined_names():
    """An undefined name is a guaranteed runtime crash on whatever path reaches it."""
    findings = [f for f in _ruff() if (f["code"] or "E999") in CRASHING]
    assert not findings, (
        "undefined name(s) - these WILL crash when that branch runs:\n  " + _format(findings))


def test_lint_clean():
    """Everything else ruff.toml selects. Each ignore in that file is an argued exception, so
    a finding here is a rule the project decided it wanted, firing on new code."""
    findings = [f for f in _ruff() if (f["code"] or "E999") not in CRASHING]
    assert not findings, (
        f"{len(findings)} lint finding(s) - fix, or argue the rule down in ruff.toml:\n  "
        + _format(findings))


def test_no_readme_section_shadows_a_module():
    """A README section titled with a src/ path duplicates the module's own docstring.

    The module docstring is the home — it lives beside the code someone is editing.
    A README section retelling the same story drifts from it and costs tokens on every
    file read.  This is the exact regression the prose cut was meant to prevent.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    shadowed = [
        line.strip()
        for line in readme.splitlines()
        if line.startswith("#") and "src/" in line
    ]
    assert not shadowed, (
        "README section title(s) contain a src/ path — the module docstring is the home:\n  "
        + "\n  ".join(shadowed)
    )


def test_import_contracts_hold():
    """The architectural rules in .importlinter - which one file's imports cannot show you.

    Each of the three is a rule the README already stated in prose, and each has the same shape:
    an import added in the wrong place works perfectly for whoever added it and breaks on a path
    they were not running. The blender one is the sharpest - `scripts/blender/*.py` execute in
    Blender's own interpreter, so importing anything from the venv is an error that first appears
    minutes into a 3D render, inside a subprocess.

    Runs the checker as a subprocess for the same reason the ruff tests do, and fails the same
    way if it cannot run: a contract nobody checked is not a contract that held.
    """
    if not LINT_IMPORTS.exists():
        raise AssertionError(
            f"import-linter is not installed in this interpreter's environment ({sys.executable}).\n"
            f"  expected: {LINT_IMPORTS}\n\n"
            "Install it with everything else:\n\n"
            "  .venv/bin/pip install -r requirements.txt"
        )
    result = subprocess.run([str(LINT_IMPORTS)], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    # Its own report is already the readable thing - contract by contract, with the offending
    # import chain and line numbers under each - so it is passed through rather than reformatted.
    report = result.stdout.strip() or result.stderr.strip()
    raise AssertionError(
        "import contract(s) broken - see .importlinter for what each rule is for:\n\n"
        + "\n".join(line for line in report.splitlines() if not line.startswith(("╔", "╚", "║", " ║", "  └", "      ╚")))
    )


def test_typed_modules_pass_mypy():
    """Every module listed in mypy.ini type-checks strictly: no unannotated function, no type error.

    The list in mypy.ini only grows - a module joins when it is written or substantially touched -
    so the typed part of the repo can widen without first paying for the whole of src/. Fails,
    rather than skips, when mypy is missing, for the same reason the ruff tests do.
    """
    if not MYPY.exists():
        raise AssertionError(
            f"mypy is not installed in this interpreter's environment ({sys.executable}).\n"
            f"  expected: {MYPY}\n\n"
            "Install it with everything else:\n\n"
            "  .venv/bin/pip install -r requirements.txt"
        )
    result = subprocess.run([str(MYPY)], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, "mypy (strict, modules in mypy.ini):\n" + result.stdout.strip()
