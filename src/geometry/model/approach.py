"""A leg seen from one of its ends: the junction-relative view of an OSM street segment.

A Leg is one edge of OSM's street graph, in the way's own direction, from `start_node` to
`end_node`. Everything junction-relative in this project - corners, crossings, stop bars, signs,
paint anchors - measures "this far out from the junction" as a station from a centreline's
start. An APPROACH is the leg oriented away from one of its junctions so that rule holds at
either end: the leg itself at its start, and the leg reversed (kerbs swapped) at its end.

Computed, never stored, so it cannot drift from the leg it is a view of.
"""
from __future__ import annotations

from enum import Enum

from shapely import reverse

from src.geometry.model.leg_frame import JUNCTION_AT_START, Leg

__all__ = [
    "JUNCTION_AT_START",
    "End",
    "LegTable",
    "approach_id",
    "approach_view",
    "approaches",
    "at_every_junction",
    "at_junction",
    "leg_of",
    "leg_side",
    "leg_station",
    "node_at",
    "reoriented",
    "split_approach_id",
]

END_SUFFIX = ":end"


class End(Enum):
    START = "start"
    END = "end"


def approach_id(leg_name: str, end: End) -> str:
    """The leg's own name at its start, `<leg>:end` at its end."""
    return leg_name if end is End.START else f"{leg_name}{END_SUFFIX}"


def split_approach_id(approach: str) -> tuple[str, End]:
    """(leg name, end) for an approach id."""
    if approach.endswith(END_SUFFIX):
        return approach[: -len(END_SUFFIX)], End.END
    return approach, End.START


def node_at(leg: Leg, end: End) -> int | None:
    return leg.start_node if end is End.START else leg.end_node


def reoriented(leg: Leg, name: str) -> Leg:
    """`leg` run the other way and renamed: centreline reversed, left and right kerbs swapped.

    Its own inverse, so an approach seen from its end and reoriented again is the leg.
    """
    view = Leg(name=name, centerline=reverse(leg.centerline), curb_to_curb_ft=leg.curb_to_curb_ft,
               start_node=leg.end_node, end_node=leg.start_node, osm_way_id=leg.osm_way_id)
    # Set after construction: __post_init__ rebuilds both kerbs as offsets whenever a width is
    # given, which would discard a traced kerb.
    view.left_curb = None if leg.right_curb is None else reverse(leg.right_curb)
    view.right_curb = None if leg.left_curb is None else reverse(leg.left_curb)
    view.traced_sides = {"left" if side == "right" else "right" for side in leg.traced_sides}
    view.width_provenance = leg.width_provenance
    view.state_centreline = (None if leg.state_centreline is None
                             else reverse(leg.state_centreline))
    return view


def approach_view(leg: Leg, end: End) -> Leg:
    """The leg oriented away from its junction at `end`, named by `approach_id`."""
    if end is End.START:
        return leg
    return reoriented(leg, approach_id(leg.name, end))


def leg_of(view: Leg, end: End, leg_name: str) -> Leg:
    """The leg an approach view was taken of - what to write back after fitting a view."""
    return view if end is End.START else reoriented(view, leg_name)


class LegTable(dict):
    """{leg name: Leg} that also answers for an approach id: `legs["broad_street_5:end"]` is that
    leg seen from its end. Iterating it yields legs only, so paint is still one solution per leg;
    looking up a corner's or a crossing's approach id finds the view it was built against.
    """

    def __missing__(self, key):
        leg_name, end = split_approach_id(str(key))
        if end is End.END and dict.__contains__(self, leg_name):
            leg = dict.__getitem__(self, leg_name)
            if leg.end_node is not None:
                return approach_view(leg, End.END)
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        if dict.__contains__(self, key):
            return True
        try:
            self[key]
        except KeyError:
            return False
        return True

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def approaches(legs: dict[str, Leg]) -> dict[str, Leg]:
    """{approach id: view} for every end of every leg that sits at a junction."""
    out: dict[str, Leg] = {}
    for name, leg in legs.items():
        for end in End:
            if node_at(leg, end) is not None:
                out[approach_id(name, end)] = approach_view(leg, end)
    return out


def at_junction(legs: dict[str, Leg]) -> dict[int, dict[str, Leg]]:
    """{junction node: {approach id: view}} - every street that starts OR ends at each node."""
    out: dict[int, dict[str, Leg]] = {}
    for approach, view in approaches(legs).items():
        out.setdefault(view.start_node, {})[approach] = view
    return out


def leg_station(leg: Leg, end: End, approach_station_ft: float) -> float:
    """A station measured out from the junction at `end`, in the leg's own frame."""
    if end is End.START:
        return approach_station_ft
    return leg.centerline.length - approach_station_ft


def leg_side(end: End, side: str) -> str:
    """The leg's side for an approach's `side` ("left"/"right"): swapped at the end."""
    if end is End.START:
        return side
    return {"left": "right", "right": "left"}.get(str(side), side)


def at_every_junction(state):
    """A shallow copy of a DesignState whose `legs` also hold each leg's END approach - what the
    junction-relative resolvers iterate, so a leg running junction to junction is placed at both.
    """
    import copy

    out = copy.copy(state)
    out.legs = LegTable(state.legs)
    for name, leg in state.legs.items():
        if leg.end_node is not None and not name.endswith(END_SUFFIX):
            dict.__setitem__(out.legs, approach_id(name, End.END), approach_view(leg, End.END))
    return out
