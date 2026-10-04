"""paint(): every marking is placed off a Kerb or the Centre, resolved per station through the
chains in src.geometry.model.kerb_chain - the same ones the road surface is built from."""
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from src.geometry.model import (STRIP_SAMPLE_FT, Profile, band_from_offsets, centre_chain,
                                curb_station_span, kerb_chain, line_from_offsets,
                                nominal_half_ft, station_offset_many, tapered_curb_offsets)
from src.geometry.targets import Side

if TYPE_CHECKING:
    from src.geometry.model import Leg

__all__ = ["Centre", "Kerb", "KerbToKerb", "Placed", "Profile", "Ref", "centre_profile",
           "kerb_profile", "place", "resolve"]


@dataclass(frozen=True)
class Kerb:
    """`inset_ft` in from this side's kerb."""
    inset_ft: float = 0.0


@dataclass(frozen=True)
class Centre:
    """`offset_ft` toward this side from the centre line; a lane divider is Centre(TARGET_LANE_WIDTH_FT)."""
    offset_ft: float = 0.0


@dataclass(frozen=True)
class KerbToKerb:
    """The whole carriageway, left kerb to right kerb."""


Ref = Kerb | Centre


@dataclass(frozen=True)
class Placed:
    geometry: LineString | Polygon | MultiPolygon | None
    datum: dict[str, float]          # each source's share of the piece's stations
    pinched_stations: np.ndarray


def _traced(leg: "Leg", side: Side, s: np.ndarray) -> np.ndarray:
    """Signed offsets of this side's traced kerb, NaN outside its traced span."""
    span = curb_station_span(leg, side) if side in leg.traced_sides else None
    off = None if span is None else tapered_curb_offsets(leg, side, s, outside=np.nan)
    if off is None:
        return np.full(len(s), np.nan)
    return np.where((s >= span[0]) & (s <= span[1]), side.sign * off, np.nan)


def _state(leg: "Leg", s: np.ndarray) -> np.ndarray:
    """Signed offsets of the registered state centreline, NaN where it does not reach."""
    line = leg.state_centreline
    if line is None:
        return np.full(len(s), np.nan)
    d = np.append(np.arange(0.0, line.length, STRIP_SAMPLE_FT), line.length)
    st, off = station_offset_many(leg.centerline,
                                  shapely.get_coordinates(shapely.line_interpolate_point(line, d)))
    keep = np.isfinite(st) & np.isfinite(off) & (st >= 0) & (st <= leg.centerline.length)
    if not keep.any():
        return np.full(len(s), np.nan)
    order = np.argsort(st[keep])
    return np.interp(s, st[keep][order], off[keep][order], left=np.nan, right=np.nan)


def kerb_profile(leg: "Leg", side: Side | str, stations: np.ndarray) -> Profile:
    side = Side(side)
    return kerb_chain(stations, side, _traced(leg, side, stations),
                      _traced(leg, side.other, stations), _state(leg, stations),
                      nominal_half_ft(leg, default=None), leg.name)


def centre_profile(leg: "Leg", stations: np.ndarray) -> Profile:
    return centre_chain(stations, _traced(leg, Side.LEFT, stations),
                        _traced(leg, Side.RIGHT, stations), _state(leg, stations))


def resolve(leg: "Leg", side: Side | str, ref: Ref, stations: np.ndarray) -> Profile:
    side = Side(side)
    if isinstance(ref, Kerb):
        p = kerb_profile(leg, side, stations)
        return Profile(stations, p.offsets_ft - side.sign * ref.inset_ft, p.source)
    if isinstance(ref, Centre):
        p = centre_profile(leg, stations)
        return Profile(stations, p.offsets_ft + side.sign * ref.offset_ft, p.source)
    raise TypeError(f"not a paint reference: {ref!r}")


def _shares(*sources: np.ndarray) -> dict[str, float]:
    """Each source's share of the stations, every array weighted equally."""
    counts = Counter(str(v) for src in sources for v in src)
    total = sum(counts.values())
    return {k: n / total for k, n in counts.items()}


def _runs(ok: np.ndarray) -> list[slice]:
    """Maximal runs of True at least two stations long."""
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]


def place(leg: "Leg", side: Side | str, stations: np.ndarray, outer: Ref | KerbToKerb,
          inner: Ref | None = None) -> Placed:
    """A line (outer only), a band (outer and inner), or the carriageway (KerbToKerb). Offsets are
    signed left-positive, so they are laid out in the LEFT frame whatever `side` is."""
    none = np.array([])
    if isinstance(outer, KerbToKerb):
        lk, rk = kerb_profile(leg, Side.LEFT, stations), kerb_profile(leg, Side.RIGHT, stations)
        return Placed(band_from_offsets(leg, Side.LEFT, stations, rk.offsets_ft, lk.offsets_ft),
                      _shares(lk.source, rk.source), none)
    o = resolve(leg, side, outer, stations)
    if inner is None:
        return Placed(line_from_offsets(leg, Side.LEFT, stations, o.offsets_ft),
                      _shares(o.source), none)
    n = resolve(leg, side, inner, stations)
    ok = Side(side).sign * (o.offsets_ft - n.offsets_ft) > 0
    bands = [b for run in _runs(ok)
             if (b := band_from_offsets(leg, Side.LEFT, stations[run], n.offsets_ft[run],
                                        o.offsets_ft[run])) is not None]
    return Placed(unary_union(bands) if bands else None, _shares(o.source[ok], n.source[ok]),
                  stations[~ok])
