"""Paint datum: the source and placement of every painted marking (paint spec v2, Phase 1)."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
from src.geometry.model import (
    MAX_KERB_FOLLOW_TAPER,
    curb_station_span,
    line_from_offsets,
    station_offset_many,
)
from src.geometry.model.leg_frame import _tapered_curb_frame


KERB_SOURCES = ("traced", "mirrored", "nominal")      # priority order
CENTRE_SOURCES = ("kerbs", "state", "osm")


@dataclass(frozen=True)
class Profile:
    """A paint profile: stations and signed offsets with their sources."""
    stations: np.ndarray      # float
    offsets_ft: np.ndarray    # float, SIGNED, left +
    source: np.ndarray        # dtype "<U8"


@dataclass(frozen=True)
class Kerb:
    """Kerb edge reference: offset from traced/nominal/mirrored kerb."""
    inset_ft: float = 0.0


@dataclass(frozen=True)
class Centre:
    """Centre line reference: offset from registered or nominal centreline."""
    offset_ft: float = 0.0


@dataclass(frozen=True)
class KerbToKerb:
    """Full width: left kerb to right kerb."""
    pass


@dataclass(frozen=True)
class Placed:
    """Result of placing geometry: the shape, datum, and any pinched stations."""
    geometry: LineString | Polygon | MultiPolygon | None
    datum: dict[str, float]
    pinched_stations: np.ndarray


def _traced(leg, side: str, s: np.ndarray) -> np.ndarray:
    """Signed offsets of the traced kerb, NaN where not traced."""
    sign = 1 if side == "left" else -1

    if side not in leg.traced_sides:
        return np.full_like(s, np.nan)

    span = curb_station_span(leg, side)
    if span is None:
        return np.full_like(s, np.nan)

    lo, hi = span
    curb = getattr(leg, f"{side}_curb")
    grid, eroded = _tapered_curb_frame(leg.centerline, curb, MAX_KERB_FOLLOW_TAPER)

    # Interpolate with NaN outside the span
    v = np.interp(s, grid, eroded, left=np.nan, right=np.nan)

    # Set NaN where s is outside the span
    v[s < lo] = np.nan
    v[s > hi] = np.nan

    return sign * v


def _state(leg, s: np.ndarray) -> np.ndarray:
    """Signed offsets of leg.state_centreline, NaN where it does not reach."""
    if getattr(leg, "state_centreline", None) is None:
        return np.full_like(s, np.nan)

    line = leg.state_centreline

    # Sample every 2 ft plus the end
    d = np.append(np.arange(0.0, line.length, 2.0), line.length)
    pts = np.asarray([line.interpolate(di).coords[0] for di in d], dtype=float)

    # Get stations and offsets
    st, off = station_offset_many(leg.centerline, pts)

    # Keep only points within the leg's station range
    mask = (st >= 0) & (st <= leg.centerline.length) & np.isfinite(st) & np.isfinite(off)
    st = st[mask]
    off = off[mask]

    # Sort by station
    sort_idx = np.argsort(st)
    st = st[sort_idx]
    off = off[sort_idx]

    # Interpolate with NaN outside the range
    result = np.interp(s, st, off, left=np.nan, right=np.nan)

    return result


def _smooth_seams(s: np.ndarray, offsets: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Smooth seams between different source ranks.
    
    For each worse-rank station, interpolate from the nearest better-rank station's value
    towards this station's value, limited by MAX_KERB_FOLLOW_TAPER per unit distance.
    """
    result = offsets.copy()
    unique_ranks = np.unique(rank[np.isfinite(rank)])
    
    for r in sorted(unique_ranks):
        if r == 0:
            continue
        
        # Find stations at this rank
        at_rank = np.where(rank == r)[0]
        if len(at_rank) == 0:
            continue
        
        # For each station at rank r, find nearest station with rank < r
        for i in at_rank:
            if not np.isfinite(result[i]):
                continue
            
            # Find best rank stations
            better = np.where(rank < r)[0]
            if len(better) == 0:
                continue
            
            # Find nearest by distance
            distances = np.abs(s[better] - s[i])
            nearest_idx = better[np.argmin(distances)]
            d = distances[np.argmin(distances)]
            
            # Compute interpolation based on taper rate
            better_val = result[nearest_idx]
            worse_val = offsets[i]  # Use original value
            diff = abs(better_val - worse_val)

            if diff > 0:
                # Maximum distance (in feet) over which to taper
                # Computed from slope limit MAX_KERB_FOLLOW_TAPER
                max_taper_dist = diff / MAX_KERB_FOLLOW_TAPER
                # Interpolation factor: how much of the taper to apply
                # Decreases with distance from the boundary
                taper_factor = max(0.0, 1.0 - d / max_taper_dist)
                # Interpolate from worse_val towards better_val
                result[i] = worse_val + (better_val - worse_val) * taper_factor
    
    return result


def kerb_profile(leg, side: str, stations: np.ndarray) -> Profile:
    """Kerb profile with priority: traced > mirrored > nominal."""
    sign = 1 if side == "left" else -1
    other_side = "right" if side == "left" else "left"

    # Step 1: Try traced kerb
    k = _traced(leg, side, stations)
    src = np.full_like(stations, "traced", dtype="<U8")

    # Step 2: Fill remaining with mirrored from other side or state centre
    remaining = ~np.isfinite(k)
    if np.any(remaining):
        ko = _traced(leg, other_side, stations)
        c = _state(leg, stations)
        c[~np.isfinite(c)] = 0.0  # NaN becomes 0

        # Where opposite side is traced, use mirrored
        mirror_mask = remaining & np.isfinite(ko)
        k[mirror_mask] = 2 * c[mirror_mask] - ko[mirror_mask]
        src[mirror_mask] = "mirrored"
        remaining = ~np.isfinite(k)

    # Step 3: Fill remaining with nominal
    if np.any(remaining):
        if leg.curb_to_curb_ft is None:
            # Collect indices where we couldn't fill
            unfilled = np.where(remaining)[0]
            unfilled_stations = stations[unfilled]
            raise ValueError(
                f"{leg.name}: no traced kerb, no mirror and no curb_to_curb_ft at stations "
                f"{unfilled_stations.tolist()}")

        k[remaining] = sign * leg.curb_to_curb_ft / 2
        src[remaining] = "nominal"

    # Step 4: Smooth seams - compute rank
    rank = np.zeros_like(stations, dtype=float)
    for i, source in enumerate(src):
        if source == "traced":
            rank[i] = 0
        elif source == "mirrored":
            rank[i] = 1
        elif source == "nominal":
            rank[i] = 2

    k = _smooth_seams(stations, k, rank)

    return Profile(stations, k, src)


def centre_profile(leg, stations: np.ndarray) -> Profile:
    """Centre profile with priority: kerbs (both traced) > state > osm."""
    kl = _traced(leg, "left", stations)
    kr = _traced(leg, "right", stations)

    # Step 1: Where both kerbs are traced, use midpoint
    c = np.full_like(stations, np.nan, dtype=float)
    both_traced = np.isfinite(kl) & np.isfinite(kr)
    c[both_traced] = (kl[both_traced] + kr[both_traced]) / 2
    src = np.full_like(stations, "osm", dtype="<U8")
    src[both_traced] = "kerbs"

    # Step 2: Fill remaining with state centreline
    remaining = ~np.isfinite(c)
    c_state = _state(leg, stations)
    state_valid = remaining & np.isfinite(c_state)
    c[state_valid] = c_state[state_valid]
    src[state_valid] = "state"
    remaining = ~np.isfinite(c)

    # Step 3: Fill remaining with 0.0 (osm)
    c[remaining] = 0.0
    # src already set to "osm"

    # Step 4: Smooth seams
    rank = np.zeros_like(stations, dtype=float)
    for i, source in enumerate(src):
        if source == "kerbs":
            rank[i] = 0
        elif source == "state":
            rank[i] = 1
        elif source == "osm":
            rank[i] = 2

    c = _smooth_seams(stations, c, rank)

    return Profile(stations, c, src)


def resolve(leg, side: str, ref, stations: np.ndarray) -> Profile:
    """Resolve a reference (Kerb/Centre) to a profile."""
    sign = 1 if side == "left" else -1

    if isinstance(ref, Kerb):
        p = kerb_profile(leg, side, stations)
        offsets = p.offsets_ft - sign * ref.inset_ft
        return Profile(stations, offsets, p.source)
    elif isinstance(ref, Centre):
        p = centre_profile(leg, stations)
        offsets = p.offsets_ft + sign * ref.offset_ft
        return Profile(stations, offsets, p.source)
    else:
        raise TypeError(f"Unsupported reference type: {type(ref)}")


def place(leg, side: str, stations: np.ndarray, outer, inner=None) -> Placed:
    """Place geometry on a leg between outer and optional inner boundaries.

    Returns a Placed object with geometry, datum (source fractions), and pinched stations.
    """
    sign = 1 if side == "left" else -1

    if isinstance(outer, KerbToKerb):
        # KerbToKerb case: left and right kerbs
        l = kerb_profile(leg, "left", stations)
        r = kerb_profile(leg, "right", stations)

        outer_line = line_from_offsets(leg, "left", stations, l.offsets_ft)
        inner_line = line_from_offsets(leg, "left", stations, r.offsets_ft)

        if outer_line is None or inner_line is None:
            geometry = None
        else:
            outer_coords = list(outer_line.coords)
            inner_coords = list(reversed(inner_line.coords))
            geometry = Polygon(outer_coords + inner_coords)

        # Datum: equal weight to both sources
        datum = {}
        for source_name in KERB_SOURCES:
            l_count = np.sum(l.source == source_name)
            r_count = np.sum(r.source == source_name)
            total = l_count + r_count
            if total > 0:
                datum[source_name] = float(total) / (2 * len(stations))

        return Placed(geometry, datum, np.array([]))
    elif inner is None:
        # Line case
        o = resolve(leg, side, outer, stations)
        line = line_from_offsets(leg, "left", stations, o.offsets_ft)

        # Compute datum as fractions
        datum = {}
        for source_name in KERB_SOURCES + CENTRE_SOURCES:
            count = np.sum(o.source == source_name)
            if count > 0:
                datum[source_name] = float(count) / len(stations)

        return Placed(line, datum, np.array([]))
    else:
        # Band case (outer and inner both Kerb/Centre)
        o = resolve(leg, side, outer, stations)
        n = resolve(leg, side, inner, stations)

        width = sign * (o.offsets_ft - n.offsets_ft)
        ok = width > 0
        pinched = stations[~ok]

        # Build polygons from runs of ok stations
        polygons = []
        i = 0
        while i < len(stations):
            if not ok[i]:
                i += 1
                continue

            # Find run of consecutive ok stations (at least 2)
            j = i
            while j < len(stations) and ok[j]:
                j += 1

            if j - i >= 2:
                s_run = stations[i:j]
                o_run = o.offsets_ft[i:j]
                n_run = n.offsets_ft[i:j]

                outer_line = line_from_offsets(leg, "left", s_run, o_run)
                inner_line = line_from_offsets(leg, "left", s_run, n_run)

                if outer_line is not None and inner_line is not None:
                    outer_coords = list(outer_line.coords)
                    inner_coords = list(reversed(inner_line.coords))
                    poly = Polygon(outer_coords + inner_coords)
                    if poly.is_valid:
                        polygons.append(poly)

            i = j

        if len(polygons) == 0:
            geometry = None
        elif len(polygons) == 1:
            geometry = polygons[0]
        else:
            geometry = MultiPolygon(polygons)

        # Datum: equal weight to outer and inner sources, only counted at ok stations
        datum = {}
        ok_indices = np.where(ok)[0]
        if len(ok_indices) > 0:
            for source_name in KERB_SOURCES + CENTRE_SOURCES:
                o_count = np.sum(o.source[ok_indices] == source_name)
                n_count = np.sum(n.source[ok_indices] == source_name)
                total = o_count + n_count
                if total > 0:
                    datum[source_name] = float(total) / (2 * len(ok_indices))

        return Placed(geometry, datum, pinched)
