"""Where OSM says a crossing meets a kerb: the crossing way's own vertex that IS a kerb.

A `barrier=kerb` node on the footway, or a node a kerb way shares with the crossing. The point
is surveyed and so is the direction away from the road - along the crossing, away from its
middle - so both the tactile pad (src/render/props.py) and the road surface's edge
(src/geometry/network/slice_design.py:slice_pavement) are read off this one list.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from shapely.geometry import LineString, Point

from src.render.coords import wgs84_to_state_plane


class KerbPoint(NamedTuple):
    xy: tuple[float, float]
    outward: tuple[float, float]       # unit vector along the crossing, off the road
    kerb_tags: dict
    crossing_tags: dict
    kerb_line: LineString | None       # the kerb way through this vertex, None for a kerb node


def kerb_points(crossings: list[dict], kerbs: list[dict], roads: list[dict] = ()) -> list[KerbPoint]:
    """Every crossing vertex that is a kerb, deduplicated by position.

    OFF THE ROAD is read from OSM's topology: a crossing shares a node with the road way it
    crosses, so away from that node is away from the road. At a kerb WAY the pad goes square
    behind the kerb - its normal, on the far side from the road node - because the crossing's own
    last segment can run along the kerb (Broad & Greenwood's southeast corner ramp: 1.4 ft of
    crossing laid along the lowered kerb) and says nothing about which side is the road.
    """
    road_nodes = {n for road in roads for n in road.get("node_ids") or ()}
    node_tags: dict[tuple, dict] = {}
    way_of: dict[int, tuple[dict, LineString]] = {}
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords and kerb.get("lon") is not None:
            node_tags[(round(kerb["lon"], 7), round(kerb["lat"], 7))] = kerb.get("tags") or {}
        elif coords and len(coords) >= 2:
            line = LineString(zip(*wgs84_to_state_plane.transform([c[0] for c in coords],
                                                                   [c[1] for c in coords])))
            for node_id in kerb.get("node_ids") or []:
                way_of.setdefault(node_id, (kerb.get("tags") or {}, line))
    out, seen = [], set()
    for crossing in crossings:
        coords = crossing.get("coords_wgs84") or []
        if len(coords) < 2:
            continue
        ids = list(crossing.get("node_ids") or [None] * len(coords))
        line = LineString(zip(*wgs84_to_state_plane.transform([c[0] for c in coords],
                                                               [c[1] for c in coords])))
        found = []
        for (lon, lat), node_id, xy in zip(coords, ids, line.coords):
            tags, kerb_line = node_tags.get((round(lon, 7), round(lat, 7))), None
            if tags is None and node_id in way_of:
                tags, kerb_line = way_of[node_id]
            if tags is not None:
                found.append((line.project(Point(xy)), xy, tags, kerb_line))
        on_road = [line.project(Point(xy)) for node_id, xy in zip(ids, line.coords)
                   if node_id in road_nodes]
        for along, xy, tags, kerb_line in found:
            key = (round(xy[0], 2), round(xy[1], 2))
            if key in seen:
                continue
            anchor = line.interpolate(min(on_road, key=lambda st: abs(st - along)) if on_road
                                      else line.length / 2)
            away = np.array([xy[0] - anchor.x, xy[1] - anchor.y])
            if kerb_line is not None:
                at = kerb_line.project(Point(xy))
                a = kerb_line.interpolate(max(at - 0.5, 0.0))
                b = kerb_line.interpolate(min(at + 0.5, kerb_line.length))
                normal = np.array([-(b.y - a.y), b.x - a.x])
                if np.linalg.norm(normal) > 1e-9:
                    away = normal if float(normal @ away) >= 0 else -normal
            norm = float(np.linalg.norm(away))
            if norm < 1e-9:
                continue
            seen.add(key)
            out.append(KerbPoint((float(xy[0]), float(xy[1])),
                                 (float(away[0] / norm), float(away[1] / norm)),
                                 tags, crossing.get("tags") or {}, kerb_line))
    return out
