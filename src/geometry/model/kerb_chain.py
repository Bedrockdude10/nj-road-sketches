"""Where a kerb and the centre line are, station by station, from the best evidence at each.

THE ONE HOME of the source priority, read by paint (src.geometry.paint.datum) and by the road
surface (src.geometry.context_roads). Inputs are signed offsets, left-positive, NaN where that
evidence does not reach; how each was measured is the caller's business.

    kerb:   traced -> mirrored about the registered state line (never about the OSM way, which
            can sit 10 ft off) -> one street width off the far kerb -> this side's own measured
            offset, carried past its traced end -> nominal half-width
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
    OFFSET = "offset"
    CARRIED = "carried"
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


@dataclass(frozen=True)
class StreetMeasures:
    """What the WHOLE street's tracing says, so a kerb past the traced span is placed from the
    street's own measurements and not from the stretch being drawn (which moves with the sheet)."""
    width_ft: float | None      # median kerb-to-kerb where both sides are traced
    left_ft: float | None       # median traced left offset, signed
    right_ft: float | None

    @classmethod
    def of(cls, left: np.ndarray, right: np.ndarray) -> "StreetMeasures":
        both = np.isfinite(left) & np.isfinite(right)

        def median(v: np.ndarray) -> float | None:
            return float(np.median(v)) if len(v) else None

        return cls(median(left[both] - right[both]), median(left[np.isfinite(left)]),
                   median(right[np.isfinite(right)]))

    def typical(self, side: Side) -> float | None:
        return self.left_ft if Side(side) is Side.LEFT else self.right_ft


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
               state: np.ndarray, street: StreetMeasures, nominal_half_ft: float | None,
               label: str) -> Profile:
    """One side's kerb. The far-kerb width is the street's measured width, else the nominal one;
    `nominal_half_ft` None raises where nothing else reaches."""
    side = Side(side)
    nan = np.full(len(stations), np.nan)
    width = street.width_ft if street.width_ft is not None else (
        None if nominal_half_ft is None else 2 * nominal_half_ft)
    typical = street.typical(side)
    layers = [own,
              2 * state - other,
              nan if width is None else other + side.sign * width,
              nan if typical is None else np.full(len(stations), typical),
              nan if nominal_half_ft is None else np.full(len(stations), side.sign * nominal_half_ft)]
    return _first_finite(stations, layers, list(KerbSource), label)


def centre_chain(stations: np.ndarray, left: np.ndarray, right: np.ndarray,
                 state: np.ndarray) -> Profile:
    """The centre line: midway between two traced kerbs, else the state line, else the way."""
    return _first_finite(stations, [(left + right) / 2, state, np.zeros(len(stations))],
                         list(CentreSource), "centre")
