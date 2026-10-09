"""A corner's effective turning radius - NJDOT Design Manual - Roadway 6.4.3: turning radii are
designed on the "effective" radius, measured from the edge of the travel lane where a parking lane,
bike lane or shoulder lies outside it; a single circular arc joining the tangent edges serves a
simple intersection; and 30 ft (standards.toml `turn.effective_radius`) lets an occasional truck or
bus turn without much encroachment. A kerb protruding into the turning path endangers pedestrians
standing at it.

Each tangent edge is a travel lane's edge on the corner's side, straight, from where the street
leaves the junction along its direction there. For a radius the arc tangent to both is unique: its
centre is that radius in from each edge. The effective radius is the widest such arc, up to
EFFECTIVE_RADIUS_M - all a decision needs - that stays on the drawn carriageway and touches
nothing built in it (posts, parked cars).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from src.standards import si

STEP_M = si("numerical.turn_step")
EFFECTIVE_RADIUS_M = si("turn.effective_radius")


@dataclass(frozen=True)
class Edge:
    """A tangent edge: the travel lane's edge carried straight back to the junction (its point
    there), the street's direction out of the junction, and which side the corner is on (+1 its left)."""
    point: np.ndarray
    out: np.ndarray
    side: int

    @property
    def inward(self) -> np.ndarray:
        """The unit normal from the edge toward the corner."""
        return self.side * np.array([-self.out[1], self.out[0]])


def arc(approach: Edge, exit: Edge, radius: float) -> LineString | None:
    """The arc of `radius` tangent to both edges, from the approach's to the exit's; None where
    the edges are parallel or the arc would touch either behind the junction."""
    normals = np.stack([approach.inward, exit.inward])
    if abs(np.linalg.det(normals)) < 1e-9:
        return None
    centre = np.linalg.solve(normals, [radius + approach.inward @ approach.point,
                                       radius + exit.inward @ exit.point])
    a, b = centre - radius * approach.inward, centre - radius * exit.inward
    if (a - approach.point) @ approach.out < 0 or (b - exit.point) @ exit.out < 0:
        return None
    start, end = (math.atan2(*(p - centre)[::-1]) for p in (a, b))
    sweep = (start - end) % (2 * math.pi)                        # a right turn: clockwise, a to b
    angles = start - np.linspace(0.0, sweep, max(2, math.ceil(sweep * radius / STEP_M) + 1))
    return LineString(centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1))


def effective_radius(approach: Edge, exit: Edge, drivable: BaseGeometry, blocked: BaseGeometry) -> float | None:
    """The widest arc joining the edges, a turn_step apart up to EFFECTIVE_RADIUS_M, that stays
    within `drivable` and off `blocked` (both prepared); 0 where none does, None where no arc
    joins them at all - the streets run on into each other, no turn between them."""
    best = None
    for radius in np.arange(STEP_M, EFFECTIVE_RADIUS_M + STEP_M / 2, STEP_M):
        line = arc(approach, exit, radius)
        if line is None:
            continue
        best = best or 0.0
        if drivable.contains(line) and not blocked.intersects(line):
            best = float(radius)
    return best
