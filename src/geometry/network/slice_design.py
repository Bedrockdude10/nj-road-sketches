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

import geopandas as gpd
from shapely import reverse
from shapely.geometry import LineString, Point
from shapely.ops import substring

from src.geometry.intersection.junction import IntersectionModel
from src.geometry.model import Leg, NJ_STATE_PLANE_FT
from src.geometry.treatments import DesignState

#: Shorter than this is a stub the crop left at the window edge, not an approach worth a leg.
MIN_APPROACH_FT: float = 20.0


def slice_pavement(features: gpd.GeoDataFrame):
    """The asphalt in the window, as one geometry. What a junction gets from its corner ring."""
    from shapely.ops import unary_union

    paved = list(features[features["kind"] == "pavement"].geometry)
    return unary_union(paved) if paved else None


def _approaches(line: LineString, node: Point) -> list[LineString]:
    """The street, split at the junction into the approaches that RADIATE from it.

    A Leg in this project is an approach measured outward from a node, and every placement that
    reads `offset_ft` - a near-corner sign, a stop bar, a crossing - means "this far out from the
    junction". Handed a whole street instead, station 0 is wherever the crop happened to cut it,
    so a sign "at the near corner" stands mid-block. That is what put a W16-21P in the
    carriageway; it was never a bad placement rule, it was a leg that was not a leg.
    """
    at = line.project(node)
    back, ahead = substring(line, 0.0, at), substring(line, at, line.length)
    # Reversed so both run OUTWARD from the node, which is the direction a leg's stations count.
    return [piece for piece in (reverse(back), ahead)
            if isinstance(piece, LineString) and piece.length > MIN_APPROACH_FT]


def _legs_of(streets: gpd.GeoDataFrame, width_by_name: dict[str, float],
             node: Point) -> dict[str, Leg]:
    """One leg per APPROACH: each named street in the window, split at the junction.

    The clipped centreline, so a leg is exactly as long as the drawing is wide. That is the
    frame-scale rule from the other end (SKILLS.md 0b): here the crop IS the extent, so a
    treatment applies to all of the street in the picture by construction.
    """
    legs: dict[str, Leg] = {}
    for row in streets.itertuples():
        pieces = [approach
                  for part in getattr(row.geometry, "geoms", [row.geometry])
                  if isinstance(part, LineString) and part.length > 0
                  for approach in _approaches(part, node)]
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


def slice_design(features: gpd.GeoDataFrame) -> tuple[IntersectionModel, DesignState]:
    """The (model, state) for one slice, ready for export_scenario or plot_design_state.

    `features` is a slice of the document in state-plane feet - what render_slice.slice_around
    returns. The window's own centre is the node every leg radiates from: a crop has no OSM node,
    and centring the window on the junction you want drawn is what choosing the crop MEANS.
    """
    streets = features[features["kind"] == "street"].rename(columns={"name": "name_"})
    minx, miny, maxx, maxy = features.total_bounds
    center_ft = Point((minx + maxx) / 2, (miny + maxy) / 2)
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
    legs = _legs_of(streets, traced, center_ft)
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=NJ_STATE_PLANE_FT)

    model = IntersectionModel(
        # `legs` is how legs_on_road tells which approaches are on a route, and therefore the
        # only reason a route decision reaches a crop at all. The street's own OSM name, so the
        # document and the decision are keyed on one string.
        config={"intersection": {},
                "legs": {slug: {"street_name": leg.name} for slug, leg in legs.items()}},
        center_wgs84=gpd.GeoSeries([center_ft], crs=NJ_STATE_PLANE_FT).to_crs(4326).iloc[0],
        center_ft=center_ft,
        legs=legs,
        # No corner is modelled in a crop - there is no junction node to fillet around, and
        # inventing one would put a kerb return where OSM traced none.
        corner_fillets={},
        parcels=empty,
        corner_parcels=empty,
    )
    return model, DesignState(legs=legs, corner_fillets={})
