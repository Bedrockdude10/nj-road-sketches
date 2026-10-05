"""A slice of the borough document, expressed as the (model, state) pair the renderers take.

THE POINT OF THIS MODULE IS THAT IT IS SMALL. `export_scenario` and `plot_design_state` read
only about fifteen attributes between them, and `Leg`, `DesignState` and `IntersectionModel`
require two, two and seven fields respectively - so a window onto OSM can be handed to the
renderers this project already has, rather than to a second pair written for the network path.

So a "site" stops being a configured place and becomes a crop: the legs are whatever named
streets cross the window, the treatments are whatever `route_decision_for` says about them, and
nothing here is per-site. The seven-field model carries no parcels and no surveyed lengths,
which is honest - the document does not hold them - and every consumer of those already guards
for empty.
"""
from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
from shapely import reverse
from shapely.geometry import LineString, Point, box

from src.geometry.cross_streets import cross_streets_ft
from src.geometry.intersection.fitting import (_centre_legs_on_traced_kerbs, _fit_legs_to_traced_kerbs,
                                                _join_through_legs, _widths_from_traced_kerbs)
from src.geometry.intersection.junction import IntersectionModel, RoadSpan
from src.geometry.intersection.kerb_sources import KERB_NEAR_JUNCTION_FT
from src.geometry.intersection.load import _build_corners
from src.geometry.intersection.state_centreline import attach_state_centrelines
from src.geometry.intersection.paved import _paved_surfaces_ft, to_state_plane
from src.geometry.model.traced_kerbs import corner_radii_from_kerbs
from src.geometry.model import Leg, NJ_STATE_PLANE_FT, build_pavement_polygon
from src.geometry.model.approach import End, LegTable, at_junction, split_approach_id
from src.geometry.network.street_graph import StreetEdge, street_edges
from src.geometry.treatments import DesignState

#: Shorter than this is a stub the crop left at the window edge, not an approach worth a leg.
MIN_APPROACH_FT: float = 20.0

def slice_pavement(features: gpd.GeoDataFrame, corner_fillets: dict | None = None,
                   legs: dict | None = None, osm: dict | None = None):
    """The asphalt in the window: the streets' traced pavement, ended at OSM's kerbs.

    The pavement is split along every traced kerb way, and a piece is road if an OSM street
    centreline runs through it. A piece a kerb cuts off - the square corner the street
    rectangles leave behind a corner kerb, or a bulb-out OSM traced into the street - has no
    street in it, so it is not road. No distance, depth or side rule decides it: the kerb is the
    edge because OSM says it is.

    Without `legs` (a site, one junction) the corner ring is used, as it always was.
    """
    from shapely.ops import polygonize, unary_union

    paved = list(features[features["kind"] == "pavement"].geometry)
    base = unary_union(paved) if paved else None
    if corner_fillets and legs is None:
        try:
            return build_pavement_polygon(corner_fillets)
        except ValueError:
            return base     # an unclosable ring is reported by check_pavement_ring, not here
    if base is None or not legs:
        return base
    kerbs = [line for line in features[features["kind"] == "kerb_way"].geometry
             if isinstance(line, LineString) and line.length > 0]
    if not kerbs:
        return base
    edges, _junctions, _xy = street_edges((osm or {}).get("roads") or [])
    streets = unary_union([edge.line for edge in edges] or [leg.centerline for leg in legs.values()])
    faces = polygonize(unary_union([base.boundary, *kerbs]))
    road = [face for face in faces
            if base.contains(face.representative_point()) and face.intersects(streets)
            and face.intersection(streets).length > 0]
    return unary_union(road) if road else base


def _traced_widths(streets: gpd.GeoDataFrame, paved: gpd.GeoDataFrame) -> dict[int, float]:
    """{street row: the width of ITS OWN asphalt}, pavement polygon area over centreline length.

    PER FEATURE, NOT PER NAME. A street is several features (Broad Street is a dozen, Eaton Place
    four) and each has a pavement polygon of its own, so keying by name kept ONE polygon's area
    and divided it by the length of every feature sharing the name: Eaton Place's 6,350 sq ft over
    1,133 ft of its four pieces read 5.6 ft wide, and every one of them was drawn that wide. A
    window holding one feature per name never saw it.

    A pavement belongs to the same-named street whose MIDDLE it contains - the midpoint of the
    line, not its centroid, which for a closed loop is the empty middle of the ring.
    """
    widths: dict[int, float] = {}
    for index, street in streets.iterrows():
        if street.geometry.length <= 0:
            continue
        middle = street.geometry.interpolate(0.5, normalized=True)
        own = paved[paved["name_"] == street["name_"]]
        if own.empty:
            continue
        pavement = own.geometry.iloc[int(own.geometry.distance(middle).argmin())]
        widths[index] = pavement.area / street.geometry.length
    return widths


def _home_frame(edge: StreetEdge, junctions: set[int]) -> tuple[LineString, int | None, int | None, bool]:
    """(centreline, start node, end node, aligned with the way) for one graph edge.

    Stored running away from its HOME junction - the way's start node where that is a junction,
    else its end node - because every junction-relative rule here measures out from a
    centreline's start. `aligned` records, from the graph and not from geometry, whether that is
    the way's own direction, which is what OSM's left/right tags are relative to. The far end is
    reached through `approach_view(leg, End.END)`.
    """
    start = edge.u if edge.u in junctions else None
    end = edge.v if edge.v in junctions else None
    if start is None and end is not None:
        return reverse(edge.line), end, None, False
    return edge.line, start, end, True


def _legs_of(edges: list[StreetEdge], junctions: set[int], streets: gpd.GeoDataFrame,
             width_by_street: dict[int, float], boundary, street_of: dict[str, str],
             way_of: dict[str, tuple[dict, bool, int]]) -> dict[str, Leg]:
    """One leg per edge of OSM's street graph, inside the area, in its home frame.

    The width is the document's own figure for the street the edge belongs to - the same-named
    `street` row under its middle - which is what corridor_pavement drew the asphalt to.
    """
    legs: dict[str, Leg] = {}
    taken: dict[str, int] = defaultdict(int)
    for edge in sorted(edges, key=lambda e: (e.name, e.way_id, e.u)):
        line, start, end, aligned = _home_frame(edge, junctions)
        if boundary is not None and not boundary.contains(line):
            clipped = line.intersection(boundary)
            parts = [p for p in getattr(clipped, "geoms", [clipped]) if isinstance(p, LineString)
                     and not p.is_empty]
            if not parts:
                continue
            home = Point(line.coords[0])
            touching = [p for p in parts if Point(p.coords[0]).distance(home) < 1e-6]
            line = touching[0] if touching and start is not None else max(parts, key=lambda p: p.length)
            if Point(line.coords[0]).distance(home) > 1e-6:
                start = None
            if Point(line.coords[-1]).distance(Point(edge.line.coords[-1 if aligned else 0])) > 1e-6:
                end = None
        if line.length <= MIN_APPROACH_FT:
            continue
        middle = line.interpolate(0.5, normalized=True)
        own = streets[streets["name_"] == edge.name]
        if own.empty:
            continue
        distances = own.geometry.distance(middle)
        if float(distances.min()) > 1.0:
            continue
        row = distances.idxmin()
        slug = edge.name.lower().replace(" ", "_")
        index = taken[slug]
        taken[slug] += 1
        key = slug if index == 0 else f"{slug}_{index}"
        # `Leg.name` IS THE KEY, as it is at a site; the street name lives only in
        # `config["legs"][key]["street_name"]`, which is where `legs_on_road` reads it. WITHOUT A
        # WIDTH A LEG IS NOT A STREET: every treatment sizes its section off curb_to_curb_ft.
        legs[key] = Leg(name=key, centerline=line, curb_to_curb_ft=width_by_street.get(row),
                        start_node=start, end_node=end, osm_way_id=edge.way_id)
        street_of[key] = edge.name
        way_of[key] = (edge.tags, aligned, edge.way_id)
    return legs


#: The placeholder every site config carries as `existing_corner_radius_ft`, for a corner whose
#: kerb nobody traced. A crop has no config to read it from and the number is not per-site - all
#: but one of them say 20 - so it lives here rather than being invented per window.
FALLBACK_CORNER_RADIUS_FT: float = 20.0


def _fit_to_traced_kerbs(legs: dict[str, Leg], groups: dict[int, dict[str, Leg]],
                          kerb_lines: list[LineString], kerb_ways: list[tuple],
                          xy: dict[int, tuple[float, float]]) -> None:
    """Put the window's legs through the SITE PATH's own kerb fitting, junction by junction.

    A crop skipped all of it, and the cost was on the sheet twice over. Every approach of a
    street carried ONE width - the street's whole pavement area over its whole length, so both
    Greenwood approaches read 31.4 ft where the site measures 26.5 and 31.2 - and no leg had a
    traced side at all, so every curb line was a plain offset from the alignment rather than the
    kerb somebody surveyed. Width is the datum kerbside paint is placed from
    (.claude/SKILLS.md section 2), so an over-wide leg puts its paint past its own kerb: 22.8% of
    all paint in this window landed outside the pavement, and 11.4% of the bikeway's did.
    Fitted, that is 1.3%, and the widths land within 0.1 ft of what the configured site measures.

    PER JUNCTION, because `center_ft` is what `_fit_legs_to_traced_kerbs` measures its near set
    against and a window holds several. Mutates the Legs in place as the site path does, in the
    site path's order and for its reasons: the widths first, so the assignment's ratio window
    keeps the kerbs the fit needs; then the fit; then the centring; then the through-street join.
    `_extend_curbs_with_far_tracing` is deliberately NOT here - it fetches at a radius about a
    junction centre, which is the one thing a window cannot ask for.
    """
    for node, group in groups.items():
        for name, width_ft in _widths_from_traced_kerbs(group, kerb_lines, {}).items():
            view = group[name]
            group[name] = Leg(name=view.name, centerline=view.centerline, curb_to_curb_ft=width_ft,
                              start_node=view.start_node, end_node=view.end_node,
                              osm_way_id=view.osm_way_id)
        _fit_legs_to_traced_kerbs(group, kerb_ways, Point(xy[node]), {}, bounded=True)
        _centre_legs_on_traced_kerbs(group)
        _join_through_legs(group)
        # Written back only where this junction is the leg's HOME (its start): the far end's view
        # was fitted so the kerb contest here saw every street at the node, but a leg has one
        # set of kerbs and its home junction measures them.
        for approach, view in group.items():
            leg_name, end = split_approach_id(approach)
            if end is End.START:
                legs[leg_name] = view


def _corners_of(groups: dict[int, dict[str, Leg]], kerb_lines: list[LineString],
                xy: dict[int, tuple[float, float]]) -> dict:
    """The corner geometry for EVERY junction in the window, through the site path's own builder.

    ONE GROUP AT A TIME, because `build_corner_fillets` sorts the legs it is given by compass
    bearing and fillets between each angularly-adjacent PAIR, wrapping around. That is right for
    a junction and wrong for a window: handed all six legs of Broad x Greenwood and Broad x
    Blackwell at once it pairs a Greenwood approach with a Blackwell one and rounds a corner
    between two streets that never meet.

    ONLY THE KERBS NEAR EACH JUNCTION, the site path's own `near` test: `assign_kerbs_to_corners`
    files every kerb it is handed under its two nearest legs however far away it is, so handed the
    whole borough it put a kerb on N Greenwood under Lafayette & Hamilton, 1,000 ft off, and drew
    that junction's corner arc across Greenwood's roadway.
    """
    corners: dict = {}
    for node, group in groups.items():
        at = Point(xy[node])
        near = [line for line in kerb_lines if line.distance(at) <= KERB_NEAR_JUNCTION_FT]
        radii, _notes = corner_radii_from_kerbs(group, near, FALLBACK_CORNER_RADIUS_FT)
        corners.update(_build_corners(group, FALLBACK_CORNER_RADIUS_FT, radii, near))
    return corners


def slice_design(features: gpd.GeoDataFrame, osm: dict | None = None,
                 osm_area: str | None = None) -> tuple[IntersectionModel, DesignState]:
    """The (model, state) for one slice, ready for export_scenario or plot_design_state.

    `features` is a slice of the document in state-plane feet - what render_slice.slice_around
    returns. EVERY junction in the window is modelled, not the one it happens to be centred on -
    a window is a piece of the network, and a road or a node inside it that the model does not
    carry is simply missing from the drawing.

    `osm` is the window's own context in the fetchers' shape (render_slice.slice_context), and
    it is what fills `paved_surfaces` - the driveways, parking and surrounding streets BOTH
    renderers read off the model. Without it a slice drew 0 of them where the same junction as a
    configured site drew 33, which is the whole of "everything in 3D must be in 2D" failing at
    the document rather than at the seam.
    """
    streets = features[features["kind"] == "street"].rename(columns={"name": "name_"})
    minx, miny, maxx, maxy = features.total_bounds
    center_ft = Point((minx + maxx) / 2, (miny + maxy) / 2)
    center_wgs84 = gpd.GeoSeries([center_ft], crs=NJ_STATE_PLANE_FT).to_crs(4326).iloc[0]
    # THE TRACED WIDTH, not OSM's `width` tag. Both are in the document and they disagree -
    # SKILLS.md section 2, the two datums - and here the choice is forced: the asphalt drawn under a slice
    # IS corridor_pavement's, which follows the traced kerb, so a prop placed off the nominal
    # half-width lands inside its own street's pavement and furniture_in_roadway fires. One kerb
    # for the drawing and the placement, which is the whole of that invariant's complaint.
    # `name_`, because itertuples gives every row a `.name` of its own - the index's.
    traced = _traced_widths(streets, features[features["kind"] == "pavement"]
                            .rename(columns={"name": "name_"}))
    # OSM's OWN TOPOLOGY: osmnx's graph of the area's streets (src/geometry/network/street_graph.py).
    # Each edge is a leg, each node where differently named streets meet is a junction, and a
    # leg's tags are its own way's - nothing is matched back onto it by geometry.
    edges, junctions, xy = street_edges((osm or {}).get("roads") or [])
    towns = features[features["kind"] == "municipality"].geometry
    boundary = towns.union_all() if len(towns) else None
    street_of: dict[str, str] = {}
    way_of: dict[str, tuple[dict, bool, int]] = {}
    legs = LegTable(_legs_of(edges, junctions, streets, traced, boundary, street_of, way_of))
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=NJ_STATE_PLANE_FT)

    # WHAT OSM SAYS ABOUT EACH LEG is its way's tags, whole: `overtaking=no` is a double yellow
    # (DesignState.from_model), `parking:left`/`parking:right` the kerbside restrictions
    # (parking_restriction_spans), read the right way round by `aligned`.
    spans = {name: [RoadSpan(start_ft=0.0, end_ft=legs[name].centerline.length, tags=tags,
                             aligned=aligned, way_id=way_id)]
             for name, (tags, aligned, way_id) in way_of.items()}
    # R.S. 39:4-138(e) applies at every cross street whether or not this slice is centred on one,
    # so a crop that never resolves it is making a false statement about the street, not merely
    # drawing less. Gated like `spans` and `paved_surfaces`: without `osm` there is nothing to
    # match against.
    cross_streets = (cross_streets_ft(osm, center_ft, legs)
                     if osm is not None else {})
    # THE CORNERS, per junction, off the window's own traced kerbs. Without them a crop had
    # `corner_fillets={}`, and that one empty dict is upstream of most of what a slice was
    # missing: build_pavement_polygon has no ring, build_sidewalk_pieces has no edge to widen,
    # the signal hardware has no corner to stand on, and every corner return and apron is absent.
    traced = [k for k in ((osm or {}).get("kerbs") or [])
              if len(k.get("coords_wgs84") or []) >= 2]
    kerb_lines = [LineString(to_state_plane(k["coords_wgs84"])) for k in traced]
    kerb_ways = [(line, k.get("tags") or {}, k.get("id")) for line, k in zip(kerb_lines, traced)]
    # Every street that starts OR ends at each node, each seen from that node.
    groups = at_junction(legs)
    # BEFORE the corners: the fit moves the very curb lines a fillet is trimmed against, and
    # `_centre_legs_on_traced_kerbs` bends the alignment every offset below is measured from.
    # Same order as the site path, for the same reason.
    if osm is not None:
        _fit_to_traced_kerbs(legs, groups, kerb_lines, kerb_ways, xy)
    corner_fillets = _corners_of(groups, kerb_lines, xy) if osm is not None else {}

    model = IntersectionModel(
        osm_area=osm_area,
        osm=osm or {},
        # `legs` is how legs_on_road tells which approaches are on a route, and therefore the
        # only reason a route decision reaches a crop at all. The street's own OSM name, so the
        # document and the decision are keyed on one string.
        config={"intersection": {},
                "legs": {slug: {"street_name": street_of[slug]} for slug in legs}
                | {f"{slug}:end": {"street_name": street_of[slug]} for slug in legs}},
        center_wgs84=center_wgs84,
        center_ft=center_ft,
        legs=legs,
        corner_fillets=corner_fillets,
        parcels=empty,
        corner_parcels=empty,
        # The minor carriageways, cut to the WINDOW rather than to a radius about its centre:
        # `reach` is the drawing's own extent, so asphalt reaches the corners of the sheet
        # instead of stopping on the circle inscribed in it. Cut around the junction rings AND
        # the traced pavement - see slice_pavement, which is the one place a window says what
        # its asphalt is.
        paved_surfaces=_paved_surfaces_ft(osm,
                                          pavement=slice_pavement(features, corner_fillets, legs, osm),
                                          reach=box(minx, miny, maxx, maxy))
        if osm is not None else (),
        leg_road_spans=spans,
        leg_osm_tags={name: tags for name, (tags, _aligned, _way) in way_of.items()},
        leg_osm_aligned={name: aligned for name, (_tags, aligned, _way) in way_of.items()},
        cross_streets=cross_streets,
    )
    # Attach registered state centrelines to legs if osm_area is provided
    if osm_area is not None:
        attach_state_centrelines(legs, osm_area)
    return model, DesignState(legs=legs, corner_fillets=corner_fillets)
