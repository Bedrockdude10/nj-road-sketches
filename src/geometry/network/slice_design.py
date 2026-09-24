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

from itertools import pairwise

import geopandas as gpd
from shapely import reverse
from shapely.geometry import LineString, Point, box
from shapely.ops import substring

from src.geometry.intersection.junction import IntersectionModel
from src.geometry.intersection.osm_roads import _match_legs_to_osm_roads
from src.geometry.intersection.paved import _paved_surfaces_ft
from src.geometry.model import Leg, NJ_STATE_PLANE_FT
from src.geometry.treatments import DesignState

#: Shorter than this is a stub the crop left at the window edge, not an approach worth a leg.
MIN_APPROACH_FT: float = 20.0

#: How far a junction node may sit off a street's centreline and still be ON it. The node is
#: interpolated along the OTHER street's centreline, so it misses this one by the angle between
#: them; generous, because the alternative is failing to cut and drawing a leg through a junction.
ON_STREET_FT: float = 30.0


def slice_pavement(features: gpd.GeoDataFrame):
    """The asphalt in the window, as one geometry. What a junction gets from its corner ring."""
    from shapely.ops import unary_union

    paved = list(features[features["kind"] == "pavement"].geometry)
    return unary_union(paved) if paved else None


def junction_nodes(features: gpd.GeoDataFrame) -> list[Point]:
    """Every junction in the window, deduplicated by position.

    The document records a junction once per street that meets there, so Broad x Greenwood is
    two `crossing` rows and one node. Rounded to the foot before deduping, because the two rows
    are interpolated along two different centrelines and land within an inch of each other
    rather than exactly on top.
    """
    points = features[features["kind"] == "crossing"].geometry
    return [Point(xy) for xy in {(round(p.x), round(p.y)) for p in points if p is not None}]


def _approaches(line: LineString, nodes: list[Point]) -> list[LineString]:
    """The street cut at EVERY junction on it, into the approaches that radiate from each.

    A Leg in this project is an approach measured outward from a node, and every placement that
    reads `offset_ft` - a near-corner sign, a stop bar, a crossing - means "this far out from the
    junction". Handed a whole street, station 0 is wherever the crop happened to cut it, so a
    sign "at the near corner" stands mid-block; that is what put a W16-21P in the carriageway.

    CUT AT ALL OF THEM, not at the window's centre. A window is a piece of the network, not a
    site that happens to be drawn wide: a 600 ft square of this borough holds four junction
    nodes, and modelling one of them is what left the other three's ramps and stop signs
    undrawn at 231-415 ft out. A node is only a junction for the approaches that touch it, so
    the interior cuts are junctions and the two outer ends are just where the crop fell.

    THE SEGMENTS TILE THE STREET; they never overlap. A stretch between two junctions is an
    approach to both, and emitting it twice - once from each end - is the obvious way to give
    both nodes their approach. It is also two derivations of one fact, which is this repo's
    oldest bug shape (SKILLS.md 0). Measured: the reversed copy of one 291 ft stretch of Broad
    came back with a DIFFERENT paint solution from the forward copy, shattered into 11 polygons
    by crosswalk cuts landing at different stations, and the scene check crashed on the
    multi-part geometry. One piece of asphalt gets one answer.

    The cost is real and worth naming: a middle segment has a junction at each end but only one
    station 0, so it is an approach for that end only. Junction-relative furniture on the other
    end comes from the NEXT segment out, which does start there.
    """
    on_line = sorted({line.project(node) for node in nodes if line.distance(node) <= ON_STREET_FT})
    cuts = [0.0, *(at for at in on_line if MIN_APPROACH_FT < at < line.length - MIN_APPROACH_FT),
            line.length]
    pieces = []
    for index, (a, b) in enumerate(pairwise(cuts)):
        segment = substring(line, a, b)
        if not isinstance(segment, LineString) or segment.length <= MIN_APPROACH_FT:
            continue
        # Reversed only where the junction is at the FAR end - the first segment, whose near end
        # is just where the crop fell. Every other segment already starts at one.
        pieces.append(reverse(segment) if index == 0 and len(cuts) > 2 else segment)
    return pieces


def _legs_of(streets: gpd.GeoDataFrame, width_by_name: dict[str, float],
             nodes: list[Point]) -> dict[str, Leg]:
    """One leg per APPROACH: each named street in the window, cut at every junction on it.

    The clipped centreline, so a leg is exactly as long as the drawing is wide. That is the
    frame-scale rule from the other end (SKILLS.md 0b): here the crop IS the extent, so a
    treatment applies to all of the street in the picture by construction.
    """
    legs: dict[str, Leg] = {}
    for row in streets.itertuples():
        pieces = [approach
                  for part in getattr(row.geometry, "geoms", [row.geometry])
                  if isinstance(part, LineString) and part.length > 0
                  for approach in _approaches(part, nodes)]
        for index, piece in enumerate(pieces):
            slug = str(row.name_).lower().replace(" ", "_")
            # WITHOUT A WIDTH A LEG IS NOT A STREET: every treatment sizes its section off
            # curb_to_curb_ft, and a leg missing it refuses the facility with "no width -
            # nothing to fit a lane into" rather than failing. The document's own figure, the
            # one corridor_pavement drew the asphalt to.
            legs[slug if index == 0 else f"{slug}_{index}"] = Leg(
                name=str(row.name_), centerline=piece,
                curb_to_curb_ft=width_by_name.get(str(row.name_)))
    return legs


def slice_design(features: gpd.GeoDataFrame, osm: dict | None = None
                 ) -> tuple[IntersectionModel, DesignState]:
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
    paved = features[features["kind"] == "pavement"].rename(columns={"name": "name_"})
    traced = {row.name_: row.geometry.area / length
              for row in paved.itertuples()
              if (length := streets[streets["name_"] == row.name_].geometry.length.sum()) > 0}
    legs = _legs_of(streets, traced, junction_nodes(features))
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=NJ_STATE_PLANE_FT)

    # WHAT OSM SAYS ABOUT EACH LEG, matched off the window's own road ways. Not a layer but a
    # STATEMENT, and the one call that carries it: `overtaking=no` is a double yellow
    # (DesignState.from_model), and `parking:left`/`parking:right` are the kerbside restrictions
    # (parking_restriction_spans). Without it every leg fell back to the repo's dashed default,
    # which claims on the sheet that passing is permitted - on five of the seven named ways
    # through Broad & Greenwood, where the survey says it is not.
    spans = _match_legs_to_osm_roads(legs, center_wgs84, center_ft,
                                     roads=(osm or {}).get("roads")) if osm is not None else {}
    dominant = {name: max(rows, key=lambda span: span.length_ft) for name, rows in spans.items()}

    model = IntersectionModel(
        # `legs` is how legs_on_road tells which approaches are on a route, and therefore the
        # only reason a route decision reaches a crop at all. The street's own OSM name, so the
        # document and the decision are keyed on one string.
        config={"intersection": {},
                "legs": {slug: {"street_name": leg.name} for slug, leg in legs.items()}},
        center_wgs84=center_wgs84,
        center_ft=center_ft,
        legs=legs,
        # No corner is modelled in a crop - there is no junction node to fillet around, and
        # inventing one would put a kerb return where OSM traced none.
        corner_fillets={},
        parcels=empty,
        corner_parcels=empty,
        # The minor carriageways, cut to the WINDOW rather than to a radius about its centre:
        # `reach` is the drawing's own extent, so asphalt reaches the corners of the sheet
        # instead of stopping on the circle inscribed in it. No `corner_fillets` to cut around -
        # a crop has none - so the traced pavement stands in as the measured geometry that wins.
        paved_surfaces=_paved_surfaces_ft(center_wgs84, osm=osm, pavement=slice_pavement(features),
                                          reach=box(minx, miny, maxx, maxy))
        if osm is not None else (),
        leg_road_spans=spans,
        leg_osm_tags={name: span.tags for name, span in dominant.items()},
        leg_osm_aligned={name: span.aligned for name, span in dominant.items()},
    )
    return model, DesignState(legs=legs, corner_fillets={})
