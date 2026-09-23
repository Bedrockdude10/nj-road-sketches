"""Where a leg leaves town, which is where this project's corridors end.

Every terminus drawn here is jurisdictional. The two-way bikeway on Broad St stops at the
Hopewell Borough line in both directions not because the street stops - it runs on into
Hopewell Township as CR 654 - but because nothing past the line is the borough's to build. So
"how far does this facility go" has a source, and the source is the boundary, not the drawing.

WHAT THIS EXISTS TO PREVENT. The terminus treatment used to read the end of the DRAWN bike lane
and place the two-stage turn box past it. On W Broad at Lanning the drawn end and the borough
line agree to 0.3 ft, because that leg's `working_length_ft` was configured to the line - so the
box came out 12 ft into the township and nothing could see it. They are not the same fact: a
leg's length is a rendering decision (.claude/SKILLS.md section 0b), and at 2.5x the same box
stood 195 ft over the line. An extent that moves with the sheet is the bug that section names.

THE STATION IS None WHERE A LEG NEVER LEAVES TOWN, which is the ordinary case - seven of the ten
legs across the two terminus sites. A caller must read None as "no limit on this leg", never as
zero: a limit of 0.0 ft would refuse the whole leg and the drawing would come out empty.
"""
from shapely.geometry import LineString, Point, Polygon

from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_municipality_containing


# The snapshot window the boundary is looked for in. Same base every layer that follows a street
# uses, and through the same `context_radius_m`, so a leg drawn to the line has the line in hand:
# W Broad's southwest leg reaches 2,307 ft and its window opens to 703 m to match. Larger would
# not help - a ring clipped by the snapshot AREA is dropped, not closed for (see
# fetch_municipality_containing) - and the areas are sized off this same constant.
BOUNDARY_CONTEXT_RADIUS_M = 130


def municipal_boundary_ft(center_wgs84: Point) -> tuple | None:
    """(name, boundary ring as a Polygon in state-plane feet) for this junction's municipality."""
    from src.render.frame import context_radius_m   # lazy: render layers above this one

    found = fetch_municipality_containing(center_wgs84,
                                          context_radius_m(BOUNDARY_CONTEXT_RADIUS_M))
    if found is None:
        return None
    name, ring = found
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in ring], [c[1] for c in ring])
    return name, Polygon(zip(xs, ys))


def municipal_limits_ft(center_wgs84: Point, legs: dict) -> dict:
    """{leg name: station in feet where that leg crosses out of the municipality}.

    Only legs that DO cross appear, so `.get(leg)` returns None for the rest - see the module
    docstring on why that distinction is load-bearing.

    THE FIRST CROSSING, not the nearest or the last. A leg leaving town has left it; if the line
    happens to cut a leg twice - a boundary that follows a stream back across a road - the
    facility ends at the first one, because the ground past it is not ours whatever happens
    after. Taking the max would carry a facility across somebody else's street to rejoin our own.
    """
    found = municipal_boundary_ft(center_wgs84)
    if found is None:
        return {}
    _name, boundary = found
    limits = {}
    for leg_name, leg in legs.items():
        line: LineString = leg.centerline
        crossings = line.intersection(boundary.exterior)
        if crossings.is_empty:
            continue
        points = list(crossings.geoms) if hasattr(crossings, "geoms") else [crossings]
        stations = [line.project(Point(p.coords[0]) if p.geom_type != "Point" else p)
                    for p in points]
        stations = [s for s in stations if s > 0.0]
        if stations:
            limits[leg_name] = min(stations)
    return limits
