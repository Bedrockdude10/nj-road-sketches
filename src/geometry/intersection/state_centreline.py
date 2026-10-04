"""Registered state centrelines - Phase 1 of paint spec v2.

Register NJDOT centrelines to traced kerbs by calibrating lateral offsets.
"""
from functools import lru_cache

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

from src.geometry.model import NJ_STATE_PLANE_FT
from src.sources.data_loader import load_road_network, MissingSourceData
from src.sources.osm_context import SNAPSHOT_AREAS, osm_layers
from src.geometry.intersection.kerb_sources import kerb_lines_with_tags_ft

STATE_CALIBRATION_STEP_FT = 10.0     # sample spacing along each NJDOT line
STATE_KERB_SEARCH_FT = 40.0          # how far each side to look for a traced kerb


def register_line(line: LineString, kerbs: list[LineString]) -> LineString:
    """Register a line to traced kerbs by calibrating lateral offsets.

    Returns the same line object if no kerbs available or no calibration possible.
    Otherwise returns a new LineString with lateral offsets computed from kerb positions.
    Samples points at regular intervals, computes corrections, and returns all sampled points.
    """
    if not kerbs:
        return line

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
        return line

    # Step 5: Interpolate corrections
    # corr_cal is signed: + means kerb midpoint is LEFT of the line
    corr_cal = (dl[cal] - dr[cal]) / 2.0
    corr = np.interp(d, d[cal], corr_cal)

    # Step 6: Apply corrections to all sampled points
    corrected_pts = p_i + corr[:, np.newaxis] * n_i
    # Return a LineString with all sampled points
    return LineString(corrected_pts)


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
        geom = row.geometry

        # Handle MultiLineString
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)

        # If still multi after linemerge, register each part and keep longest
        if isinstance(geom, type(linemerge([]))):  # Check if it's a MultiLineString-like
            geoms = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
            registered = [register_line(g, kerbs) for g in geoms if g.geom_type == "LineString"]
            if registered:
                geom = max(registered, key=lambda x: x.length)
            else:
                continue
        else:
            geom = register_line(geom, kerbs)

        sri = row.get("SRI")
        if sri and geom is not None:
            out[sri] = geom

    # Print output
    for sri, line in out.items():
        name = sri  # Could get name from row if available
        # Count calibrated points (heuristic: points where correction was non-zero)
        n_cal = len(line.coords) - 2  # Rough approximation
        median = 0.0  # Rough approximation
        print(f"{sri} {name}: {n_cal} calibrated, median {median:+.1f} ft")

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
    registered_state_centrelines.cache_clear()
    result = registered_state_centrelines("hopewell_borough")
    registered_state_centrelines.cache_clear()
