"""Registered state centrelines - Phase 1 of paint spec v2.

Register NJDOT centrelines to traced kerbs by calibrating lateral offsets.
"""
from functools import lru_cache

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union

from src.geometry.model import NJ_STATE_PLANE_FT
from src.sources.data_loader import load_road_network, MissingSourceData
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers
from src.geometry.intersection.kerb_sources import kerb_lines_with_tags_ft

STATE_CALIBRATION_STEP_FT = 10.0     # sample spacing along each NJDOT line
STATE_KERB_SEARCH_FT = 40.0          # how far each side to look for a traced kerb


def _corrections(line: LineString, kerbs: list[LineString]):
    """(points, normals, d, cal, corr) for `line` against `kerbs`, or None when nothing calibrates.

    d: stations every STATE_CALIBRATION_STEP_FT plus the end. cal: stations where a traced kerb was
    found within STATE_KERB_SEARCH_FT on BOTH sides. corr: the signed shift to the kerb midpoint
    (+ = left of the line's own direction), interpolated between calibrated stations and held
    at the end values beyond them.
    """
    if not kerbs:
        return None

    # Step 1: sample points along the line
    d = np.append(np.arange(0.0, line.length, STATE_CALIBRATION_STEP_FT), line.length)

    # Step 2: Points p_i and unit tangents t_i
    p_i = np.array([line.interpolate(d_val).coords[0] for d_val in d], dtype=float)

    # Tangent: use points ±1 ft along the line, or clamp to line bounds
    tangent_d_before = np.maximum(d - 1.0, 0.0)
    tangent_d_after = np.minimum(d + 1.0, line.length)
    p_before = np.array([line.interpolate(d_val).coords[0] for d_val in tangent_d_before], dtype=float)
    p_after = np.array([line.interpolate(d_val).coords[0] for d_val in tangent_d_after], dtype=float)
    tangent = p_after - p_before
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    tangent_norm = np.where(tangent_norm > 0, tangent_norm, 1.0)
    t_i = tangent / tangent_norm  # unit tangent (dx, dy)

    # Left normal: (-ty, tx)
    n_i = np.column_stack([-t_i[:, 1], t_i[:, 0]])

    # Step 3: Find kerb intersections
    wall = unary_union(kerbs)

    dl = np.full(len(d), np.nan, dtype=float)
    dr = np.full(len(d), np.nan, dtype=float)

    for i in range(len(d)):
        # Left ray: from p_i in the +n_i direction
        p_start = Point(p_i[i])
        p_end_left = Point(p_i[i] + STATE_KERB_SEARCH_FT * n_i[i])
        left_ray = LineString([p_start, p_end_left])
        left_intersection = left_ray.intersection(wall)
        if not left_intersection.is_empty:
            dl[i] = p_start.distance(left_intersection)

        # Right ray: from p_i in the -n_i direction
        p_end_right = Point(p_i[i] - STATE_KERB_SEARCH_FT * n_i[i])
        right_ray = LineString([p_start, p_end_right])
        right_intersection = right_ray.intersection(wall)
        if not right_intersection.is_empty:
            dr[i] = p_start.distance(right_intersection)

    # Step 4: Check if calibration is possible
    cal = np.isfinite(dl) & np.isfinite(dr)
    if not cal.any():
        return None

    # Step 5: Interpolate corrections
    # corr_cal is signed: + means kerb midpoint is LEFT of the line
    corr_cal = (dl[cal] - dr[cal]) / 2.0
    corr = np.interp(d, d[cal], corr_cal)
    return p_i, n_i, d, cal, corr


def register_line(line: LineString, kerbs: list[LineString]) -> LineString:
    """Register a line to traced kerbs by calibrating lateral offsets.

    Returns the same line object if no kerbs available or no calibration possible.
    Otherwise returns a new LineString with lateral offsets computed from kerb positions.
    Samples points at regular intervals, computes corrections, and returns all sampled points.
    """
    found = _corrections(line, kerbs)
    if found is None:
        return line
    p_i, n_i, _d, _cal, corr = found
    return LineString(p_i + corr[:, np.newaxis] * n_i)


@lru_cache(maxsize=1)
def registered_state_centrelines(area: str) -> dict[str, LineString]:
    """Load and register NJDOT centrelines to traced kerbs.

    Returns a memoized dict mapping SRI -> registered LineString.
    Returns empty dict if data/ is absent.
    """
    try:
        bbox = SNAPSHOT_AREAS[area]
    except KeyError:
        return {}

    try:
        net = load_road_network(bbox=bbox)
    except (MissingSourceData, FileNotFoundError):
        return {}

    net = net.to_crs(NJ_STATE_PLANE_FT)

    # Get kerbs from OSM
    osm = osm_layers(area)
    kerbs = [line for line, _tags, _id in kerb_lines_with_tags_ft(osm)]

    out = {}

    for _, row in net.iterrows():
        sri = row.get("SRI")
        geom = row.geometry
        if not sri or geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        registered = [register_line(g, kerbs) for g in parts if g.geom_type == "LineString"]
        if not registered:
            continue
        out[sri] = registered[0] if len(registered) == 1 else MultiLineString(registered)

    return out


def attach_state_centrelines(legs: dict, area: str) -> None:
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
                leg.centerline.buffer(STATE_KERB_SEARCH_FT, cap_style="flat")
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
    # Per-SRI calibration report: how many stations found a traced kerb on both sides, and how
    # far the NJDOT line moves to reach their midpoint (+ = left of the line's own direction).
    area = "hopewell_borough"
    net = load_road_network(bbox=SNAPSHOT_AREAS[area]).to_crs(NJ_STATE_PLANE_FT)
    kerbs = [line for line, _tags, _id in kerb_lines_with_tags_ft(osm_layers(area))]
    for _, row in net.iterrows():
        geom = row.geometry
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            found = _corrections(part, kerbs)
            if found is None:
                print(f"{row['SRI']}: 0 calibrated (line used as-is)")
                continue
            _p, _n, d, cal, corr = found
            c = corr[cal]
            print(f"{row['SRI']}: {int(cal.sum())}/{len(d)} calibrated, median {np.median(c):+.1f} ft, "
                  f"range {c.min():+.1f}..{c.max():+.1f} ft")
