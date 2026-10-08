"""The project's figures and the OSM keys it reads, from standards.toml at the repo root - the one
place either is declared. `si(id)` is a figure in SI (metres, degrees, mph, counts); `FIGURES` and
`TAGS` are the whole schema. Loading refuses a schema that contradicts itself, so a figure cited
to nothing or a key drawn with a figure that does not exist never reaches the drawing.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PATH = Path(__file__).resolve().parents[1] / "standards.toml"
STATUSES = ("verified", "as_cited", "local", "modelled", "numerical")
TO_SI = {"m": 1.0, "ft": 0.3048, "in": 0.0254, "deg": 1.0, "mph": 1.0, "1": 1.0}
_FIELDS = {"value", "unit", "status", "source", "at", "note", "open"}


@dataclass(frozen=True)
class Figure:
    id: str
    value: Any                   # a number, or nested lists of numbers (an outline), in `unit`
    unit: str
    status: str
    source: str | None = None
    at: str | None = None
    note: str | None = None
    open: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # a shape's other numbers, in `unit`

    def si(self, value: Any = None) -> Any:
        """`value` (else the figure's own) converted from its unit to SI, element by element."""
        value = self.value if value is None else value
        if isinstance(value, list):
            return [self.si(v) for v in value]
        return float(value) * TO_SI[self.unit]


@dataclass(frozen=True)
class Tag:
    key: str
    convention: str              # "wiki" or "project"
    means: str
    wiki: str | None = None
    absent: str | None = None
    figures: tuple[str, ...] = ()


def _figures(table: dict, prefix: str = "") -> dict[str, Figure]:
    """Every table holding a `value`, by its dotted path."""
    found: dict[str, Figure] = {}
    for name, entry in table.items():
        path = f"{prefix}{name}"
        if not isinstance(entry, dict):
            continue
        if "value" not in entry:
            found |= _figures(entry, f"{path}.")
            continue
        found[path] = Figure(path, entry["value"], entry.get("unit", ""), entry.get("status", ""),
                             entry.get("source"), entry.get("at"), entry.get("note"), entry.get("open"),
                             {k: v for k, v in entry.items() if k not in _FIELDS})
    return found


def _problems(raw: dict, figures: dict[str, Figure], tags: dict[str, Tag]) -> list[str]:
    sources = raw.get("source", {})
    problems = []
    for f in figures.values():
        if f.unit not in TO_SI:
            problems.append(f"figure {f.id}: unit {f.unit!r} is not one of {sorted(TO_SI)}")
        if f.status not in STATUSES:
            problems.append(f"figure {f.id}: status {f.status!r} is not one of {STATUSES}")
        if f.status in ("verified", "as_cited") and f.source not in sources:
            problems.append(f"figure {f.id}: {f.status} but its source {f.source!r} is not declared")
        if f.status == "local" and not f.at:
            problems.append(f"figure {f.id}: local, but `at` does not say who set it and when")
    for t in tags.values():
        if t.convention not in ("wiki", "project"):
            problems.append(f"tag {t.key}: convention {t.convention!r} is neither wiki nor project")
        if t.convention == "wiki" and not t.wiki:
            problems.append(f"tag {t.key}: a wiki key with no wiki page")
        problems += [f"tag {t.key}: figure {f!r} is not declared" for f in t.figures if f not in figures]
    return problems


def load(path: Path = PATH) -> tuple[dict[str, Figure], dict[str, Tag]]:
    raw = tomllib.loads(path.read_text())
    figures = _figures(raw.get("figure", {}))
    tags = {key: Tag(key, entry.get("convention", ""), entry.get("means", ""), entry.get("wiki"),
                     entry.get("absent"), tuple(entry.get("figures", ())))
            for key, entry in raw.get("tag", {}).items()}
    problems = _problems(raw, figures, tags)
    if problems:
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(problems))
    return figures, tags


FIGURES, TAGS = load()


def si(figure_id: str) -> Any:
    """A figure's value in SI. A KeyError names a figure standards.toml does not declare."""
    return FIGURES[figure_id].si()


def in_unit(figure_id: str, unit: str) -> Any:
    """A figure in `unit` (one of TO_SI's) - for arithmetic done in that unit."""
    figure = FIGURES[figure_id]
    scale = TO_SI[figure.unit] / TO_SI[unit]
    return [v * scale for v in figure.value] if isinstance(figure.value, list) else float(figure.value) * scale
