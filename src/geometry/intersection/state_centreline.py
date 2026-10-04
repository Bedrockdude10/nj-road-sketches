"""NJDOT centrelines registered to the traced kerbs: the centre chain's second source.

NJDOT's line surveys the route, not this kerb-to-kerb: on Route 518 it sits 9-14 ft south of the
traced kerbs' midpoint. Each line is moved, station by station, onto that midpoint wherever both
kerbs were traced, and held at the nearest correction beyond them. Computed once per area for
every line in it; nothing here knows about a site or leg until `attach_state_centrelines`.
"""
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from src.geometry.context_roads import (MAX_HALF_WIDTH_FT, assign_kerbs_to_roads, kerb_points,
                                        measured_edges)
from src.geometry.intersection.kerb_sources import kerb_lines_with_tags_ft
from src.geometry.model import NJ_STATE_PLANE_FT, point_at_many
from src.sources.data_loader import MissingSourceData, load_road_network
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers

if TYPE_CHECKING:
    from src.geometry.model import Leg


def _corrections(lines: list[LineString], kerbs: list[LineString]
                 ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Per line: (sample stations, calibrated mask, signed shift to the kerb midpoint), or None
    where no sample found a kerb on both sides. Each kerb vertex goes to its NEAREST line
    (assign_kerbs_to_roads), so two parallel routes cannot share a kerb. The shift is interpolated
    between calibrated samples and held at the end values beyond them."""
    out = []
    for line, (stations, offsets) in zip(lines, assign_kerbs_to_roads(lines, kerb_points(kerbs))):
        samples, left, right = measured_edges(line, stations, offsets)
        cal = np.isfinite(left) & np.isfinite(right)
        out.append(None if not cal.any() else
                   (samples, cal, np.interp(samples, samples[cal], (left[cal] + right[cal]) / 2)))
    return out


def register_lines(lines: list[LineString], kerbs: list[LineString]) -> list[LineString]:
    """Each line moved onto its kerbs' midpoint; a line nothing calibrates is returned as is."""
    return [line if c is None else LineString(point_at_many(line, c[0], c[2]))
            for line, c in zip(lines, _corrections(lines, kerbs))]


def register_line(line: LineString, kerbs: list[LineString]) -> LineString:
    """register_lines for one line."""
    return register_lines([line], kerbs)[0]


def _parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "LineString"]


def _area_lines(area: str) -> tuple[list[str], list[LineString], list[LineString]]:
    """(SRI per line, NJDOT lines, traced kerbs) for `area`."""
    net = load_road_network(bbox=SNAPSHOT_AREAS[area]).to_crs(NJ_STATE_PLANE_FT)
    rows = [(row["SRI"], part) for _, row in net.iterrows() for part in _parts(row.geometry)]
    kerbs = [line for line, _tags, _id in kerb_lines_with_tags_ft(osm_layers(area))]
    return [sri for sri, _ in rows], [line for _, line in rows], kerbs


@lru_cache(maxsize=1)
def registered_state_centrelines(area: str) -> dict[str, LineString | MultiLineString]:
    """{SRI: registered line} for every NJDOT line in `area`; {} where data/ is absent (CI)."""
    try:
        sris, lines, kerbs = _area_lines(area)
    except (MissingSourceData, FileNotFoundError):
        return {}
    grouped: dict[str, list[LineString]] = {}
    for sri, line in zip(sris, register_lines(lines, kerbs)):
        grouped.setdefault(sri, []).append(line)
    return {sri: ls[0] if len(ls) == 1 else MultiLineString(ls) for sri, ls in grouped.items()}


def attach_state_centrelines(legs: dict[str, "Leg"], area: str) -> None:
    """Attach registered state centrelines to legs.

    For each leg, finds the best-matching registered line and clips it to the leg.
    Sets leg.state_centreline if a sufficient match is found.
    """
    registered = registered_state_centrelines(area)

    for leg in legs.values():
        best_line = None
        best_length = 0.0

        for reg_line in registered.values():
            # Clip the registered line to the leg's centerline
            clipped = reg_line.intersection(
                leg.centerline.buffer(MAX_HALF_WIDTH_FT, cap_style="flat")
            )

            # Handle MultiLineString
            if clipped.geom_type == "MultiLineString":
                clipped = linemerge(clipped)

            # If still multi, keep the longest part
            if hasattr(clipped, 'geoms'):
                parts = [g for g in clipped.geoms if g.geom_type == "LineString"]
                if parts:
                    clipped = max(parts, key=lambda x: x.length)
                else:
                    continue

            if clipped.geom_type == "LineString" and clipped.length > best_length:
                best_line = clipped
                best_length = clipped.length

        # Set state_centreline if match is >= 0.8 * leg length
        if best_line is not None and best_length >= 0.8 * leg.centerline.length:
            leg.state_centreline = best_line
        else:
            leg.state_centreline = None


if __name__ == "__main__":
    # Per-SRI calibration report: samples with a traced kerb on both sides, and how far the NJDOT
    # line moves to reach their midpoint (+ = left of the line's own direction).
    sris, lines, kerbs = _area_lines("hopewell_borough")
    for sri, c in zip(sris, _corrections(lines, kerbs)):
        if c is None:
            print(f"{sri}: 0 calibrated (line used as-is)")
            continue
        samples, cal, corr = c
        print(f"{sri}: {int(cal.sum())}/{len(samples)} calibrated, median {np.median(corr[cal]):+.1f} "
              f"ft, range {corr[cal].min():+.1f}..{corr[cal].max():+.1f} ft")
