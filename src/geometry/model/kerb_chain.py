"""Where a kerb and the centre line are, station by station, from the best evidence at each.

THE ONE HOME of the source priority, read by paint (src.geometry.paint.datum) and by the road
surface (src.geometry.context_roads). Inputs are signed offsets, left-positive, NaN where that
evidence does not reach; how each was measured is the caller's business.

    kerb:   traced -> mirrored (2 x centre - the other side's traced kerb) -> nominal half-width
    centre: kerbs (midway, both traced) -> state (registered NJDOT line) -> osm (the way itself)

A worse source meets a better one on a MAX_KERB_FOLLOW_TAPER taper, never a step.
"""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.geometry.model.leg_frame import MAX_KERB_FOLLOW_TAPER
from src.geometry.targets import Side


class KerbSource(StrEnum):
    TRACED = "traced"
    MIRRORED = "mirrored"
    NOMINAL = "nominal"


class CentreSource(StrEnum):
    KERBS = "kerbs"
    STATE = "state"
    OSM = "osm"


@dataclass(frozen=True)
class Profile:
    stations: np.ndarray
    offsets_ft: np.ndarray      # signed, left-positive
    source: np.ndarray          # object array of KerbSource | CentreSource


def bridged(stations: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`values` with interior NaN gaps interpolated, NaN kept outside its finite span."""
    ok = np.isfinite(values)
    if not ok.any():
        return np.full(len(stations), np.nan)
    return np.interp(stations, stations[ok], values[ok], left=np.nan, right=np.nan)


def smooth_seams(stations: np.ndarray, offsets: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Clip each station to within MAX_KERB_FOLLOW_TAPER x distance of the nearest station with a
    better source, best ranks first. Rank-0 stations are never moved."""
    out = np.asarray(offsets, dtype=float).copy()
    for r in sorted({int(x) for x in np.unique(rank)} - {0}):
        better = np.flatnonzero(rank < r)
        if better.size == 0:
            continue
        for i in np.flatnonzero(rank == r):
            dist = np.abs(stations[better] - stations[i])
            k = int(np.argmin(dist))
            j, d = better[k], dist[k]
            out[i] = np.clip(out[i], out[j] - MAX_KERB_FOLLOW_TAPER * d,
                             out[j] + MAX_KERB_FOLLOW_TAPER * d)
    return out


def _first_finite(stations: np.ndarray, layers: list[np.ndarray], sources: list[StrEnum],
                  label: str) -> Profile:
    offsets = np.full(len(stations), np.nan)
    rank = np.full(len(stations), len(layers))
    for r, layer in enumerate(layers):
        take = np.isnan(offsets) & np.isfinite(layer)
        offsets[take], rank[take] = layer[take], r
    if np.isnan(offsets).any():
        raise ValueError(f"{label}: no traced kerb, no mirror and no nominal width at stations "
                         f"{stations[np.isnan(offsets)].tolist()}")
    return Profile(stations, smooth_seams(stations, offsets, rank),
                   np.array(sources, dtype=object)[rank])


def kerb_chain(stations: np.ndarray, side: Side, own: np.ndarray, other: np.ndarray,
               state: np.ndarray, nominal_half_ft: float | None, label: str) -> Profile:
    """One side's kerb. `nominal_half_ft` None raises where nothing else reaches."""
    centre = np.where(np.isfinite(state), state, 0.0)
    nominal = np.full(len(stations), np.nan if nominal_half_ft is None
                      else Side(side).sign * nominal_half_ft)
    return _first_finite(stations, [own, 2 * centre - other, nominal], list(KerbSource), label)


def centre_chain(stations: np.ndarray, left: np.ndarray, right: np.ndarray,
                 state: np.ndarray) -> Profile:
    """The centre line: midway between two traced kerbs, else the state line, else the way."""
    return _first_finite(stations, [(left + right) / 2, state, np.zeros(len(stations))],
                         list(CentreSource), "centre")
