"""paint(): every marking is placed off a Kerb or the Centre, resolved per station through the
chains in src.geometry.model.kerb_chain - the same ones the road surface is built from."""
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

from src.geometry.model import (STRIP_SAMPLE_FT, Profile, StreetMeasures, band_from_offsets,
                                centre_chain, curb_station_span, kerb_chain,
                                line_from_offsets, nominal_half_ft, point_at_many,
                                station_offset_many, taper_arc_points, tapered_curb_offsets)
from src.geometry.targets import Side

if TYPE_CHECKING:
    from collections.abc import Callable
    from src.geometry.model import Leg

__all__ = ["Across", "Along", "At", "Centre", "Glyph", "Kerb", "KerbToKerb", "Narrowest", "Placed",
           "Profile", "Ref", "Shape", "Taper", "centre_profile", "kerb_profile", "place", "resolve"]


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


@dataclass(frozen=True)
class Narrowest:
    """`inset_ft` in from this side's kerb at its NARROWEST over the stations asked for, measured
    from the centre line and held there: a kerbside zone of declared depth, rigid at the pinch."""
    inset_ft: float = 0.0


Ref = Kerb | Centre | Narrowest


@dataclass(frozen=True)
class Along:
    """A line (outer only) or band (outer and inner) down the leg over `span`."""
    span: tuple[float, float]
    outer: Ref | KerbToKerb
    inner: Ref | None = None
    step_ft: float = 1.0


@dataclass(frozen=True)
class Across:
    """A line across the leg at `at_ft`, from `outer` to `inner`; `skew_ft` moves the OUTER end downstream."""
    at_ft: float
    outer: Ref
    inner: Ref
    skew_ft: float = 0.0


@dataclass(frozen=True)
class Taper:
    """The arc from `edge` at `anchor_ft`, tangent to it, out to the kerb at `target_ft`; `fill` gives the zone between it and the kerb."""
    anchor_ft: float
    target_ft: float
    edge: Ref
    fill: bool = False
    n_points: int = 16


@dataclass(frozen=True)
class At:
    """One point per station, on `ref`: posts."""
    stations: tuple[float, ...]
    ref: Ref


@dataclass(frozen=True)
class Glyph:
    """A symbol drawn by `draw(leg, side, station_ft, offset_ft)` where `ref` resolves at `at_ft`."""
    at_ft: float
    ref: Ref
    draw: "Callable[[Leg, Side, float, float], Polygon]"


Shape = Along | Across | Taper | At | Glyph


@dataclass(frozen=True)
class Placed:
    geometry: LineString | Polygon | MultiPolygon | None
    datum: dict[str, float]          # each source's share of the piece's stations
    pinched_stations: np.ndarray


def _traced(leg: "Leg", side: Side, s: np.ndarray) -> np.ndarray:
    """Signed offsets of this side's traced kerb, NaN outside its traced span."""
    span = curb_station_span(leg, side) if side in leg.traced_sides else None
    off = None if span is None else tapered_curb_offsets(leg, side, s, outside=np.nan)
    if span is None or off is None:
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


def _street(leg: "Leg") -> StreetMeasures:
    """The leg's whole tracing, sampled every STRIP_SAMPLE_FT, whatever stations are asked for."""
    grid = np.append(np.arange(0.0, leg.centerline.length, STRIP_SAMPLE_FT), leg.centerline.length)
    return StreetMeasures.of(_traced(leg, Side.LEFT, grid), _traced(leg, Side.RIGHT, grid))


def kerb_profile(leg: "Leg", side: Side | str, stations: np.ndarray) -> Profile:
    side = Side(side)
    return kerb_chain(stations, side, _traced(leg, side, stations),
                      _traced(leg, side.other, stations), _state(leg, stations), _street(leg),
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
    if isinstance(ref, Narrowest):
        # Measured at the stations asked for AND at every kerb vertex between them: a pinch is a
        # vertex, and samples that straddle it read the kerb wider than it is.
        lo, hi = float(np.min(stations)), float(np.max(stations))
        vertices = [station_offset_many(leg.centerline, np.asarray(curb.coords))[0]
                    for curb in (leg.left_curb, leg.right_curb) if curb is not None]
        probe = np.union1d(stations, [s for v in vertices for s in v if lo <= s <= hi])
        reach = float(np.min(side.sign * (kerb_profile(leg, side, probe).offsets_ft
                                          - centre_profile(leg, probe).offsets_ft)))
        k = kerb_profile(leg, side, stations)
        c = centre_profile(leg, stations)
        return Profile(stations, c.offsets_ft + side.sign * (reach - ref.inset_ft), k.source)
    raise TypeError(f"not a paint reference: {ref!r}")


def datum_shares(*sources: np.ndarray) -> dict[str, float]:
    """Each source's share of the stations, every array weighted equally."""
    counts = Counter(str(v) for src in sources for v in src)
    total = sum(counts.values())
    return {k: n / total for k, n in counts.items()}


def _runs(ok: np.ndarray) -> list[slice]:
    """Maximal runs of True at least two stations long."""
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 2]


def _along(leg: "Leg", side: Side | str, stations: np.ndarray, outer: Ref | KerbToKerb,
            inner: Ref | None = None) -> Placed:
    """A line (outer only), a band (outer and inner), or the carriageway (KerbToKerb)."""
    none = np.array([])
    if isinstance(outer, KerbToKerb):
        lk, rk = kerb_profile(leg, Side.LEFT, stations), kerb_profile(leg, Side.RIGHT, stations)
        return Placed(band_from_offsets(leg, Side.LEFT, stations, rk.offsets_ft, lk.offsets_ft),
                      datum_shares(lk.source, rk.source), none)
    o = resolve(leg, side, outer, stations)
    if inner is None:
        return Placed(line_from_offsets(leg, Side.LEFT, stations, o.offsets_ft),
                      datum_shares(o.source), none)
    n = resolve(leg, side, inner, stations)
    ok = Side(side).sign * (o.offsets_ft - n.offsets_ft) > 0
    bands = [b for run in _runs(ok)
             if (b := band_from_offsets(leg, Side.LEFT, stations[run], n.offsets_ft[run],
                                        o.offsets_ft[run])) is not None]
    return Placed(unary_union(bands) if bands else None, datum_shares(o.source[ok], n.source[ok]),
                  stations[~ok])


def place(leg: "Leg", side: Side | str, shape: Shape) -> Placed:
    """Place a shape off references, returning geometry and datum."""
    from src.geometry.model import curb_offsets_at_stations as curb_offs

    side = Side(side)

    if isinstance(shape, Along):
        s0, s1 = shape.span
        n = max(2, int(np.ceil((s1 - s0) / shape.step_ft)) + 1)
        stations = np.linspace(s0, s1, n)
        return _along(leg, side, stations, shape.outer, shape.inner)

    if isinstance(shape, Across):
        at_stations: np.ndarray = np.asarray([shape.at_ft], float)
        o: Profile = resolve(leg, side, shape.outer, at_stations)
        inner_p: Profile = resolve(leg, side, shape.inner, at_stations)
        geometry = LineString(point_at_many(leg.centerline, np.asarray([shape.at_ft + shape.skew_ft, shape.at_ft], float),
                                           np.asarray([o.offsets_ft[0], inner_p.offsets_ft[0]], float)))
        datum = datum_shares(o.source, inner_p.source)
        return Placed(geometry, datum, np.array([]))

    if isinstance(shape, Taper):
        anchor_stations = np.asarray([shape.anchor_ft], float)
        target_stations = np.asarray([shape.target_ft], float)
        e = resolve(leg, side, shape.edge, anchor_stations)

        def kerb_at(st: np.ndarray) -> np.ndarray:
            offs = curb_offs(leg, side, st)
            return np.abs(offs) if offs is not None else np.zeros(len(st))

        arc = taper_arc_points(leg, side, abs(e.offsets_ft[0]), shape.anchor_ft, shape.target_ft, kerb_at)
        if arc is None:
            return Placed(None, datum_shares(e.source), np.array([]))

        if not shape.fill:
            geometry = LineString(arc)
            datum = datum_shares(e.source, kerb_profile(leg, side, target_stations).source)
            return Placed(geometry, datum, np.array([]))

        # Fill: kerb run
        stations = np.linspace(shape.target_ft, shape.anchor_ft,
                              max(int(np.ceil((shape.anchor_ft - shape.target_ft) / STRIP_SAMPLE_FT)) + 1, 2))
        kerb_p = kerb_profile(leg, side, stations)
        kerb_pts = point_at_many(leg.centerline, stations, side.sign * kerb_p.offsets_ft)
        if kerb_pts is None or len(kerb_pts) == 0:
            return Placed(None, datum_shares(e.source), np.array([]))

        poly_pts = arc + list(kerb_pts[1:])
        poly = Polygon(poly_pts).buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)

        datum = datum_shares(e.source, kerb_p.source)
        return Placed(poly if poly.is_valid else None, datum, np.array([]))

    if isinstance(shape, At):
        p = resolve(leg, side, shape.ref, np.asarray(shape.stations, float))
        geometry = MultiPoint(point_at_many(leg.centerline, np.asarray(shape.stations, float), p.offsets_ft))
        datum = datum_shares(p.source)
        return Placed(geometry, datum, np.array([]))

    if isinstance(shape, Glyph):
        at_stations = np.asarray([shape.at_ft], float)
        p = resolve(leg, side, shape.ref, at_stations)
        geometry = shape.draw(leg, side, shape.at_ft, float(p.offsets_ft[0]))
        datum = datum_shares(p.source)
        return Placed(geometry, datum, np.array([]))

    raise TypeError(f"not a paint shape: {shape!r}")
