"""One resolved answer to "where is this scenario's paint", shared by every consumer.

Three consumers read that answer - the 2D plan view, the 3D export, and the scene invariants -
and the premise of the plan view is that it shows what the render will show. So all three must
look at the SAME geometry. Resolving once here and handing the result around is what makes that
claim structurally true instead of a convention four call sites have to keep remembering; each
site looked locally reasonable while they disagreed, and the disagreements were only visible
side by side.

Nothing here decides anything new - every value still comes from the same function in
src/render/crosswalks.py it always did. The only thing this module adds is that there is one
of each.
"""
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from src.geometry.model import build_pavement_polygon
from src.render.props import controls_at_junction
from src.render.crosswalks import (_match_crossings_to_legs, CROSSWALK_DEPTH_FT, crosswalk_bands_ft, crosswalk_reaches_ft,
                                   resolve_crosswalk_offsets, resolve_crosswalk_skews,
                                   resolve_stop_bar_offsets, stop_bar_bands_ft)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel
    from src.geometry.paint import PaintPiece
    from src.geometry.treatments.state import DesignState

def junction_is_signalized(model, leg_crossing_tags=()) -> bool:
    """Is this junction signalized? A site's own observation where it made one, else OSM's.

    THE PRECEDENCE IS `centerline_style`'s (src/geometry/treatments/state.py:from_model): config
    is an eyes-on observation and wins BY BEING PRESENT, not by being true - a site that looked
    and found no signal writes no `signals` block, so an absent key is silence rather than a
    denial, and OSM answers it. Read as the whole answer it was a branch a crop of the network
    always took the empty side of: a window has no config at all
    (src/geometry/network/slice_design.py), so Broad x Greenwood - signalized, with four traced
    stop bars across it - resolved as unsignalized in every drawing that was not a site.

    OSM SAYS IT TWO WAYS AND EITHER IS ENOUGH, because neither is reliably present: 4 of this
    project's 8 loadable junctions carry `crossing=traffic_signals` on their crossings and
    lavallette_reese's carries none, while a `highway=traffic_signals` node sits on all 4 of the
    signalized ones and none of the unsignalized ones.

    `leg_crossing_tags` is the tags of the crossings matched TO THIS MODEL'S OWN LEGS, which is
    what keeps the question about this junction: the crossings layer is the whole world's and the
    unfiltered tag picks up a neighbour's signal two junctions away (columbia_princeton sees 2,
    wbroad_lanning 1). The node half is asked the same way, of `controls_at_junction`, so a
    `highway=traffic_signals` node at the next junction down the street signalizes nothing here
    (columbia_princeton has one 127 m away).
    """
    if "signals" in getattr(model, "config", {}):
        return bool(model.config["signals"])
    return (any(tags.get("crossing") == "traffic_signals" for tags in leg_crossing_tags)
            or any(node["tags"].get("highway") == "traffic_signals"
                   for node in controls_at_junction(model, model.osm["traffic_control"])))


def _legs_that_may_derive_a_bar(model, matched: dict) -> frozenset | None:
    """The legs a bar may be INVENTED for: every leg of a site whose config records signals, and
    none anywhere else. OSM traces stop lines as `road_marking=stop_line`, and an approach it
    traced none across gets none - a signal node says the junction is signalized, not where a
    bar is painted.
    """
    if model.config.get("signals"):
        return None
    return frozenset()


@dataclass(frozen=True)
class SceneGeometry:
    """Every marking position one DesignState implies, resolved once and shared.

    Frozen on purpose: a consumer that could adjust one field in passing is how the three
    views drift apart.

    `stop_bar_offsets` and `stop_bar_bands` carry every bar OSM has traced across an approach,
    signalized or not, plus one derived per untraced approach at a junction
    junction_is_signalized() calls signalized. So they are empty only where nobody traced a bar
    AND nothing says there is a signal.
    """
    model: object
    state: object
    pavement: Polygon | None
    # Legs that carry a PAINTED crossing today, not merely a resolved offset. Every leg gets
    # an offset (a proposal may mark a leg that has nothing today); only a marked one is
    # something other paint has to keep clear of.
    marked_crosswalks: frozenset
    crosswalk_offsets: dict          # leg -> CrosswalkOffset(offset_ft, source)
    crosswalk_skews: dict            # leg -> degrees off square, surveyed legs only
    crosswalk_reaches: dict          # leg -> (left_ft, right_ft) out to the real kerbs
    crosswalk_bands: dict            # leg -> the painted footprint
    stop_bar_offsets: dict           # leg -> station, signalized junctions only
    stop_bar_bands: dict             # leg -> the painted footprint
    # EVERY SURVEYED CROSSING, drawn from its own traced way - including the ones at junctions
    # this site does not model, which the per-leg fields above cannot reach at all (six of the ten
    # in Broad & Greenwood's 2.5x frame). Resolved here rather than per renderer so the coverage
    # check audits the crossings the export actually drew.
    surveyed_crossings: tuple = ()
    # The traced kerbs the crossings above are trimmed against, kept so a consumer that wants to
    # draw them does not resolve a second, possibly different set.
    drawn_kerbs: tuple = ()
    # {leg: the style OSM records for its matched crossing}. Resolved here because the matcher
    # runs here already - asking it again in a renderer is a second answer to one question.
    surveyed_crossing_styles: dict = field(default_factory=dict)

    @classmethod
    def resolve(cls, model: "IntersectionModel", state: "DesignState",
                 pavement=None) -> "SceneGeometry":
        """Resolve one scenario's marking geometry, from the OSM layers `model.osm` carries.

        The order below is a real dependency chain, which is the other reason this belongs in
        one place: the reaches need the pavement and the marked set, the bands need the
        reaches, and the stop bars need the crosswalk offsets.
        """
        if pavement is None:
            try:
                pavement = build_pavement_polygon(state.corner_fillets)
            except ValueError:
                pavement = None     # an unclosable ring is reported by check_pavement_ring
        # THE SURVEY AND THE FIELD OBSERVATION, reconciled rather than one chosen - which is what
        # SurveyedCrossing.is_marked's docstring asks any renderer to do. A config list is an
        # eyes-on observation and OSM's silence is a survey GAP, not a statement that a crossing
        # is bare; equally, a crossing OSM records as painted is painted whether or not a config
        # happens to name its leg. So: the union. `drawable_markings` is the one decision about
        # whether a surveyed crossing has paint this project draws, so this cannot disagree with
        # what coverage.py counts as covered.
        #
        # Without the OSM half a crop of the network painted NOTHING - it has no config at all -
        # and drew every approach of Broad x Greenwood as an unmarked dotted outline while the
        # same junction as a site drew four solid bands.
        from src.geometry.surveyed import drawable_markings  # local: geometry<->render cycle

        # JUNCTION-RELATIVE, SO OVER EVERY APPROACH: each leg from its start, and from its end
        # where a junction is there too. The resolvers below measure out from a centreline's start;
        # handed `at_every_junction` they place a crossing or stop bar at both ends of a leg that
        # runs junction to junction, keyed by approach id (src/geometry/model/approach.py).
        from src.geometry.model.approach import at_every_junction

        state = at_every_junction(state)
        crossings = model.osm["crossings"]
        matched = _match_crossings_to_legs(state.legs, crossings)
        marked = frozenset(model.config["intersection"].get("existing_marked_crosswalks", [])) | {
            leg for leg, (_a, _st, _sk, _l, tags) in matched.items() if drawable_markings(tags)}
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)
        # Two passes inside crosswalk_reaches_ft, so adjoining crossings at a shared corner
        # stop reaching for the same kerb. Passing these into crosswalk_bands_ft is what the
        # plan view's invariant pass used to skip.
        reaches = crosswalk_reaches_ft(state, offsets, skews, pavement, marked)
        bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_FT, pavement, reaches)

        # A TRACED STOP BAR IS A PAINTED STOP BAR, signalized or not - so THE LAYER IS NOT GATED
        # AT ALL. Only the DERIVATION, hanging a bar off a crosswalk offset for an approach
        # nobody traced, is a claim a junction has to earn; that gate moved into the resolver.
        # Which bar belongs to which approach is the resolver's own question, asked of each leg
        # (STOP_LINE_MAX_ALONG_FT, STOP_LINE_MAX_OFFSET_FT), so it takes the world's whole layer.
        stop_bar_offsets = resolve_stop_bar_offsets(
            state, offsets, model.osm["stop_lines"],
            derive_for_legs=_legs_that_may_derive_a_bar(model, matched))
        from src.geometry.intersection import kerb_lines_with_tags_ft
        from src.geometry.surveyed import surveyed_crossings_in_frame

        drawn_kerbs = tuple(line for line, _tags, _way_id in kerb_lines_with_tags_ft(
            osm=model.osm))
        return cls(
            model=model, state=state, pavement=pavement, marked_crosswalks=marked,
            crosswalk_offsets=offsets, crosswalk_skews=skews, crosswalk_reaches=reaches,
            crosswalk_bands=bands, stop_bar_offsets=stop_bar_offsets,
            stop_bar_bands=stop_bar_bands_ft(state, stop_bar_offsets, skews),
            # `crossings` is the same layer the per-leg offsets above came from, so the two
            # cannot disagree about which crossings exist - only about which of them belong to a leg.
            surveyed_crossings=tuple(surveyed_crossings_in_frame(model, crossings)),
            drawn_kerbs=drawn_kerbs,
            surveyed_crossing_styles={leg: style for leg, (_a, style, _s, _l, _t)
                                      in _match_crossings_to_legs(state.legs, crossings).items()},
        )

    @property
    def unmodelled_crossings(self) -> tuple:
        """The surveyed crossings belonging to NO modelled leg - the ones only this path can draw.

        The four a junction models are drawn from their leg's own band, including whatever a proposal
        restyles them to; drawing both would put two crossings 1.44-2.73 ft apart on one piece of
        ground. So every consumer wants this, not `surveyed_crossings`, and it is a property here
        rather than a filter each of them repeats.
        """
        return tuple(c for c in self.surveyed_crossings if c.leg is None)

    def surveyed_crossing_paint(self) -> list:
        """The bars and lines for every unmodelled surveyed crossing, trimmed to the carriageway.

        One list, so the 2D view, the 3D export and the coverage check draw and audit exactly the
        same geometry rather than three near-copies of it.

        IN THE STYLE THE DESIGN CALLS FOR: crossing_style_in decides, not the crossing's own OSM
        tag, or a scenario that restyles everything to continental leaves the ways tagged
        `crossing:markings=lines` drawn as two parallel lines. It is also what stops a policy
        painting a crossing nobody marked.
        """
        return [piece for _crossing, bars, lines in self.surveyed_crossing_markings()
                for piece in (*bars, *lines)]

    def surveyed_crossing_markings(self) -> list:
        """[(crossing, bars, lines)] for every unmodelled crossing the design draws something on.

        THE ONE RESOLUTION, and it had to become one before a marking policy could reach these
        at all: with three consumers each building the list off the raw drawers, styling one of
        them changed neither picture.

        Per crossing rather than flattened, because the two renderers genuinely need them apart:
        the plan view strokes a line and fills a bar differently, and the export writes them to
        separate JSON keys. What they must NOT do is decide the style, which is why that is
        resolved here and handed over.
        """
        from src.geometry.surveyed import crossing_bars_ft, crossing_lines_ft, crossing_style_in

        kerbs = list(self.drawn_kerbs)
        out = []
        for crossing in self.unmodelled_crossings:
            style = crossing_style_in(self.state, crossing)
            if style is None:
                continue        # unmarked or unrecorded - a policy may not invent paint here
            out.append((crossing, crossing_bars_ft(crossing, kerbs, style),
                         crossing_lines_ft(crossing, kerbs, style)))
        return out

    def centre_stripe_end_ft(self, leg_name: str) -> float:
        """Where a leg's centre stripe stops at its FAR end: where the stripe seen from that end
        would start (`centerline_start_ft` of its END approach), or the leg's end where no
        junction is there. One rule, read from both ends.
        """
        from src.geometry.model.approach import End, approach_id
        from src.render.crosswalks import centerline_start_ft

        leg = self.state.legs[leg_name]
        far = approach_id(leg_name, End.END)
        if leg.end_node is None or far not in self.crosswalk_offsets:
            return leg.centerline.length
        return leg.centerline.length - centerline_start_ft(
            self.crosswalk_offsets[far].offset_ft, self.stop_bar_offsets.get(far),
            far in self.marked_crosswalks)

    @property
    def unmodelled_stop_bars(self) -> tuple:
        """Every stop line OSM traces that no approach claims, drawn as itself.

        A road_marking=stop_line way IS the painted bar, so it is drawn wherever it is painted:
        the per-leg bars above can hold one bar per leg, measured from a leg's start, and a
        borough leg runs junction to junction - Broad St's approach into Greenwood ends 1,053 ft
        down its leg and was thrown away. Its own line at the bar's depth, trimmed to the road.
        """
        from shapely.geometry import LineString

        from src.geometry.intersection.paved import to_state_plane
        from src.render.crosswalks import STOP_BAR_PLAN_DEPTH_FT, match_stop_lines_to_legs

        lines = self.model.osm.get("stop_lines") or []
        claimed = list(match_stop_lines_to_legs(self.state.legs, lines).values())
        out = []
        for entry in lines:
            line = LineString(to_state_plane(entry["coords_wgs84"]))
            if any(line.equals_exact(c, 0.01) for c in claimed):
                continue
            bar = line.buffer(STOP_BAR_PLAN_DEPTH_FT / 2, cap_style="flat")
            if self.pavement is not None:
                bar = bar.intersection(self.pavement)
            out += [g for g in getattr(bar, "geoms", [bar])
                    if g.geom_type == "Polygon" and not g.is_empty]
        return tuple(out)

    @property
    def unmodelled_crossing_bands(self) -> tuple:
        """The footprints of the MARKED crossings at junctions this site does not model.

        What curbside_paint_ft has to keep its paint off, and what the scene invariants check it
        against - one definition, for the reason this class exists. `band_ft` is the traced way's
        own footprint, which is the same shape surveyed_crossing_paint() draws the bars and lines
        inside, so the paint is cut around exactly the ground the crossing is drawn on.

        MARKED ONLY. An unmarked crossing is a crossing nobody has painted (or one the surveyor
        recorded as unpainted - SurveyedCrossing.is_marked keeps the two apart), and reserving
        asphalt around paint that is not there would be inventing a marking to defer to. It is
        the same rule `marked_crosswalks` applies to this junction's own four.
        """
        return tuple(c.band_ft for c in self.unmodelled_crossings if c.is_marked)

    @property
    def kerb_openings(self):
        """Where every kerb in this scene opens for a vehicle (src/geometry/paint/openings.py).

        Here, and not inside the paint builder that used to own it, because two different
        questions are answered off these entrances and they were being answered off two
        constructions of them. The paint asks WHAT BREAKS at an opening; src/metrics.py asks HOW
        MANY STALLS the design gets, and a stall may not be marked across an entrance either. One
        property, for the reason this class exists.

        Rebuilt per access like `unmodelled_crossing_bands`, which is cheap and keeps this class
        frozen; what matters is that the RULE has one home, not the call.
        """
        from src.geometry.paint.openings import junction_mouths_ft, kerb_opening_bands

        return kerb_opening_bands(self.state, junction_mouths_ft(self.state, self.crosswalk_bands))

    def build_paint(self, props: list[dict] | None = None) -> list["PaintPiece"]:
        """Every painted marking this scenario puts down (src/geometry/paint/).

        Here rather than at each call site so the paint is always cut around the same bands the
        crossings are drawn from, and always told which crossings are marked - including those at
        junctions this site does not model.
        """
        from src.geometry.paint import curbside_paint_ft

        return curbside_paint_ft(self.state, self.crosswalk_offsets, self.model.center_ft,
                                  self.crosswalk_bands, props,
                                  marked_crosswalks=self.marked_crosswalks,
                                  crossings_elsewhere=self.unmodelled_crossing_bands,
                                  openings=self.kerb_openings)

    def build_paint_and_posts(self, props: list[dict]) -> tuple[list["PaintPiece"], list[dict]]:
        """The paint, and `props` extended with the posts only the paint knows the place of.

        The dependency runs both ways, which is why both come back from one call: the paint
        needs the props (a hydrant or a stop sign lengthens a daylight zone), and a bike
        lane's bollards need the paint (the row starts where the crossing stops reaching, a
        station resolved in the paint builder). Returning them together is what stops one
        renderer from having posts the other does not - see props.bollard_props_from_paint.
        """
        from src.render.props import bollard_props_from_paint

        paint = self.build_paint(props)
        return paint, props + bollard_props_from_paint(self.state, paint)

    def metrics(self, paint: list):
        """What this scenario achieves, measured off this resolution (src/metrics.py).

        Here for the same reason `context` is: the outcome numbers a summary panel reports
        have to be measured from the geometry the figure drew, not recomputed from the config
        it was built out of. A crossing distance re-derived from `leg.curb_to_curb_ft` would
        agree with the drawing on a symmetric leg and quietly disagree everywhere else.
        """
        from src.metrics import SceneMetrics

        return SceneMetrics.of(self.state, reaches=self.crosswalk_reaches,
                                offsets=self.crosswalk_offsets, skews=self.crosswalk_skews,
                                paint=paint, marked=self.marked_crosswalks,
                                surveyed_leg_lengths=getattr(self.model, "surveyed_leg_lengths", None),
                                openings=self.kerb_openings)

    def context(self, props: list[dict], paint: list):
        """This scene as the one object every invariant reads (src/checks.py:SceneContext).

        Built here because this class is already the single resolution of the geometry both
        renderers draw - so the invariants are checked against that same resolution rather than
        against whatever subset of it a call site remembered to pass.
        """
        from src.checks import SceneContext

        return SceneContext(model=self.model, state=self.state, pavement=self.pavement,
                             props=tuple(props), paint=tuple(paint),
                             crosswalk_bands=self.crosswalk_bands,
                             marked_crosswalks=self.marked_crosswalks,
                             crosswalk_offsets=self.crosswalk_offsets,
                             stop_bars=self.stop_bar_bands,
                             # The same tuple build_paint was cut against, not a second
                             # derivation of it - a check reading a different set from the one
                             # the paint avoided is the drift this class exists to prevent.
                             unmodelled_crossing_bands=self.unmodelled_crossing_bands)

    def check(self, props: list[dict], paint: list) -> list:
        """Every scene invariant, all violations, no raising (src/checks.py)."""
        from src.checks import check_scene

        return check_scene(self.context(props, paint))

    def report_coverage(self, props: list[dict], paint: list, frame) -> list:
        """Print, and return, the surveyed features inside the frame that the drawing does not draw.

        A NOTE RATHER THAN A FAILURE, deliberately. Kerb ramps and traffic control are PROPS
        placed per leg, so one at a junction the drawing does not model has nowhere to come from;
        raising on that would fail a render for a reason no scenario can fix, and a check that
        cannot go green is one people learn to ignore. See src/geometry/coverage.py.

        `frame` is the `Frame` the CALLER drew - a VIEW, a square - and the only honest thing to
        judge coverage against: this is an audit of the picture, not of the world, which holds
        features the picture never reaches. Derived from the model it is the leg reach plus a
        margin, and for a crop of the network that overshoots the window - 437 ft against a 300 ft
        half-width - so the report demanded features the drawing was never given.

        The features audited are the model's own layers (`model.osm`), the ones the drawing was
        built from.
        """
        from src.geometry.coverage import coverage_gaps, describe_coverage

        gaps = coverage_gaps(self.model, [*paint, *self.crosswalk_bands.values(), *props,
                                          *self.surveyed_crossing_paint()], frame, self.model.osm)
        if gaps:
            print(describe_coverage(gaps))
        return gaps

    def assert_valid(self, props: list[dict], paint: list, scenario: str = "") -> None:
        """Raise SceneInvariantError listing every violation, or return quietly."""
        from src.checks import assert_scene_valid

        assert_scene_valid(self.context(props, paint), scenario=scenario)
