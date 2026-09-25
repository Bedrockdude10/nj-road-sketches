#!/usr/bin/env python3
"""Draw the same ground in two windows and diff what each one drew.

A drawing is a projection of the document, so the window is allowed to decide HOW MUCH is
visible and nothing else. Crop the same junction at 300 ft and at 600 ft, put both back in
state-plane feet, and every piece that is not congruent inside the ground they share is a
defect - the crop reached the design.

WHY THIS AND NOT A LIST. Every violation of that rule found so far was found by eye, one render
at a time, and each cost a session: paint at a cosmetic point width, a section sized on the
narrowest pinch the sheet happened to expose, a prop scattered at a fixed marker size. This
asks the question once, of the data, and answers it for every layer at once - the same reason
the fetch tripwire exists rather than a list of fetches to check.

    scripts/compare_windows.py --around=-74.76196,40.38918 --radius-ft 300 --factor 2

MEASURED INSIDE A CORE, not inside the small window. A leg is cut where its window ends, so
paint near that edge is legitimately shorter in the smaller crop and comparing there reports
the crop rather than the defect. The core is the small window inset by `--margin-ft`.

A DISAGREEMENT REACHING INTO THE CORE IS STILL REAL EVEN THOUGH TRUNCATION CAUSED IT. A section
sized on `narrowest_half_width_ft` is measured over the whole drawn leg, so widening the sheet
finds a tighter pinch and re-sizes paint hundreds of feet away, well inside any core. That is
the defect, not an artefact of it - which is why the margin trims the edge and not the finding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapely.geometry import Point, box
from shapely.ops import unary_union

from src.geometry.markings import NARROW_LINE_WIDTH_M
from src.geometry.paint import stroke_width_ft
from src.render.coords import FT_TO_M
from src.render.crosswalks import (CENTERLINE_STRIPE_WIDTH_FT, centerline_paint_ft,
                                   centerline_start_ft)
from src.render.props import build_props
from src.render.scene import SceneGeometry
from scripts.render_slice import (_center_ft, design_for, load_network,
                                   slice_around)

#: How far inside the small window the comparison starts. One leg's worth of the corner
#: geometry a crop cuts through, so the trimmed end of a leg is outside the question.
DEFAULT_MARGIN_FT = 80.0

#: A disagreement smaller than this is arithmetic noise, not a defect: the two windows resolve
#: their centrelines from the same document but clip them at different stations, so a shared
#: piece can differ by a sliver at its ends.
NOISE_FRACTION = 0.005


def _drawn_layers(features, scenario: str) -> tuple[dict, list[dict], dict]:
    """(layer -> one geometry, props, leg -> scalars) for one window, in state-plane feet.

    Resolved through the same SceneGeometry and build_props the two renderers call, so this
    compares what is DRAWN rather than what is configured. A line is given the body it is
    painted at - paint.stroke_width_ft, the figure src/checks.py and the plan view both use -
    because a stripe half a width out of place is the defect being looked for, and comparing
    bare axes cannot see it.
    """
    model, state, pavement, context = design_for(features, scenario)
    scene = SceneGeometry.resolve(model, state, context["crossings"],
                                   stop_lines=context["stop_lines"], pavement=pavement,
                                   kerb_ways=context["kerb_ways"])
    props = build_props(model, state, scene.crosswalk_offsets, model.center_ft,
                         traffic_control=context["traffic_control"],
                         street_furniture=context["street_furniture"],
                         crossings=context["crossings"], kerb_ways=context["kerb_ways"],
                         pavement=pavement)
    paint, props = scene.build_paint_and_posts(props)

    by_layer: dict[str, list] = {}
    for piece in paint:
        body = _body(piece.geometry, None if piece.kind.covers_area
                     else stroke_width_ft(piece.kind))
        if body is not None:
            by_layer.setdefault(str(piece.kind), []).append(body)

    if scene.pavement is not None and not scene.pavement.is_empty:
        by_layer["pavement"] = [scene.pavement]

    # THE THREE MARKINGS THAT ARE NOT PaintPieces. A centerline, a crosswalk band and a stop bar
    # are resolved on the scene and drawn straight by each renderer, so the paint list above
    # cannot see them - and they are most of what an `existing` drawing contains. A differ blind
    # to the majority of the sheet would have reported "congruent" over two layers.
    for name, leg in state.legs.items():
        style = state.centerline_style(name)
        if style == "none" or name not in scene.crosswalk_offsets:
            continue
        start_ft = centerline_start_ft(scene.crosswalk_offsets[name].offset_ft,
                                        scene.stop_bar_offsets.get(name),
                                        name in scene.marked_crosswalks)
        shift = state.travel_lane_divider_shift(name)
        shift_ft, shift_side = shift if shift is not None else (0.0, None)
        by_layer.setdefault("centerline", []).extend(
            b for b in (_body(line, CENTERLINE_STRIPE_WIDTH_FT)
                        for line in centerline_paint_ft(leg, start_ft, style,
                                                         shift_ft, shift_side))
            if b is not None)
    by_layer["crosswalk_band"] = [scene.crosswalk_bands[name]
                                  for name in scene.marked_crosswalks
                                  if scene.crosswalk_bands.get(name) is not None]
    by_layer["stop_bar"] = [band for band in scene.stop_bar_bands.values() if band is not None]
    kerb_ft = NARROW_LINE_WIDTH_M / FT_TO_M
    by_layer["kerbs"] = [b for b in (_body(line, kerb_ft) for line in scene.drawn_kerbs)
                         if b is not None]
    for _name, crossing_bars, crossing_lines in scene.surveyed_crossing_markings():
        by_layer.setdefault("surveyed_crossing", []).extend(
            [b for b in (_body(g, kerb_ft) for g in crossing_lines) if b is not None]
            + list(crossing_bars))

    layers = {key: unary_union(geoms) for key, geoms in by_layer.items() if geoms}
    return layers, props, _leg_scalars(state)


def _body(geometry, width_ft: float | None):
    """A drawn piece as the ground it covers. Flat caps and mitred joins, as checks.py does."""
    if geometry is None or geometry.is_empty:
        return None
    if width_ft is None:
        return geometry
    return geometry.buffer(width_ft / 2, cap_style=2, join_style=2)


def _leg_scalars(state) -> dict:
    """The per-leg numbers a treatment decides on, keyed by the leg's midpoint.

    KEYED BY GROUND, NOT BY NAME. A window names its legs for its own slugs, so the same
    approach is `broad_street` in one crop and `broad_street_2` in another; only where the leg
    physically is survives the re-crop.
    """
    out = {}
    for name, leg in state.legs.items():
        mid = leg.centerline.interpolate(0.5, normalized=True)
        out[(round(mid.x), round(mid.y))] = {
            "name": name,
            "length_ft": round(leg.centerline.length, 1),
            "curb_to_curb_ft": round(float(leg.curb_to_curb_ft or 0.0), 2),
            "centerline_style": state.centerline_style(name),
            "narrowest_left_ft": _narrowest(leg, "left"),
            "narrowest_right_ft": _narrowest(leg, "right"),
        }
    return out


def _narrowest(leg, side: str):
    from src.geometry.model import narrowest_half_width_ft
    try:
        value = narrowest_half_width_ft(leg, side)
    except Exception:
        return None
    return None if value is None else round(float(value), 2)


def _compare_layers(small: dict, large: dict, core) -> list[tuple]:
    """Per layer: how much of the ground each window paints that the other does not."""
    rows = []
    for key in sorted(set(small) | set(large)):
        a = small.get(key)
        b = large.get(key)
        a = a.intersection(core) if a is not None else None
        b = b.intersection(core) if b is not None else None
        area_a = 0.0 if a is None else a.area
        area_b = 0.0 if b is None else b.area
        if not area_a and not area_b:
            continue
        only_a = area_a if b is None else a.difference(b).area
        only_b = area_b if a is None else b.difference(a).area
        denominator = max(area_a, area_b) or 1.0
        rows.append((key, area_a, area_b, only_a, only_b,
                     (only_a + only_b) / denominator))
    return sorted(rows, key=lambda row: -row[5])


def _compare_props(small: list, large: list, core, tolerance_ft: float = 3.0) -> list[str]:
    """Props that one window places and the other does not, or places somewhere else."""
    def inside(props):
        return [p for p in props if core.contains(Point(p["position_ft"]))]

    a, b = inside(small), inside(large)
    unmatched_b = list(b)
    findings = []
    for prop in a:
        here = Point(prop["position_ft"])
        near = [q for q in unmatched_b if here.distance(Point(q["position_ft"])) <= tolerance_ft]
        same = [q for q in near if q["type"] == prop["type"]]
        if not same:
            kinds = sorted({q["type"] for q in near})
            findings.append(f"  only in the small window: {prop['type']} at "
                            f"({here.x:.0f}, {here.y:.0f})"
                            + (f" - the large one has {kinds} there" if kinds else ""))
            continue
        unmatched_b.remove(same[0])
    for prop in unmatched_b:
        here = Point(prop["position_ft"])
        findings.append(f"  only in the large window: {prop['type']} at "
                        f"({here.x:.0f}, {here.y:.0f})")
    return findings


def _compare_legs(small: dict, large: dict, core, tolerance_ft: float = 40.0) -> list[str]:
    """Legs matched by where they are, then compared on the numbers a design reads."""
    findings = []
    for key, a in sorted(small.items()):
        here = Point(key)
        if not core.contains(here):
            continue
        near = sorted(large.items(), key=lambda kv: here.distance(Point(kv[0])))
        if not near or here.distance(Point(near[0][0])) > tolerance_ft:
            findings.append(f"  {a['name']}: no leg near ({key[0]}, {key[1]}) in the large "
                            f"window at all")
            continue
        b = near[0][1]
        for field in ("curb_to_curb_ft", "centerline_style",
                      "narrowest_left_ft", "narrowest_right_ft"):
            if a[field] != b[field]:
                findings.append(f"  {a['name']} vs {b['name']}: {field} "
                                f"{a[field]} -> {b[field]}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="hopewell_borough")
    parser.add_argument("--around", required=True, help="lon,lat to centre both windows on")
    parser.add_argument("--radius-ft", type=float, default=300.0, help="the SMALL window")
    parser.add_argument("--factor", type=float, default=2.0, help="how much bigger the other is")
    parser.add_argument("--margin-ft", type=float, default=DEFAULT_MARGIN_FT)
    parser.add_argument("--scenario", default="existing",
                        choices=("existing", "proposed", "two_way_bikeway"))
    args = parser.parse_args()

    network = load_network(args.area)
    centre = _center_ft(args.around)
    small_r, large_r = args.radius_ft, args.radius_ft * args.factor
    print(f"{args.scenario}: {small_r:.0f} ft window vs {large_r:.0f} ft, "
          f"compared inside {small_r - args.margin_ft:.0f} ft of centre\n")

    small = _drawn_layers(slice_around(network, centre, small_r), args.scenario)
    large = _drawn_layers(slice_around(network, centre, large_r), args.scenario)
    keep = small_r - args.margin_ft
    core = box(centre.x - keep, centre.y - keep, centre.x + keep, centre.y + keep)

    print(f"{'layer':38s} {'small ft2':>10s} {'large ft2':>10s} {'only S':>9s} "
          f"{'only L':>9s} {'disagree':>9s}")
    clean = True
    for key, area_a, area_b, only_a, only_b, fraction in _compare_layers(small[0], large[0],
                                                                         core):
        flag = " " if fraction <= NOISE_FRACTION else "*"
        if fraction > NOISE_FRACTION:
            clean = False
        print(f"{flag}{key:37s} {area_a:10.1f} {area_b:10.1f} {only_a:9.1f} "
              f"{only_b:9.1f} {fraction * 100:8.1f}%")

    for title, findings in (("props", _compare_props(small[1], large[1], core)),
                            ("legs", _compare_legs(small[2], large[2], core))):
        print(f"\n{title}:")
        if not findings:
            print("  congruent")
            continue
        clean = False
        for line in findings:
            print(line)

    print("\nthe window reached the drawing." if not clean
          else "\ncongruent: the window decided only how much was visible.")


if __name__ == "__main__":
    main()
