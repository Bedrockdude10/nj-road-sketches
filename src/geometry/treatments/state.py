"""DesignState - the thing every treatment transforms.

A treatment returns a NEW state rather than mutating one, so scenarios stack without disturbing
the existing-conditions baseline. Kept in its own module because everything else in this package
imports it and it imports (almost) nothing back: the few concrete treatments it has to reach for
are imported inside the methods that use them, which is what keeps this module at the bottom of
the graph next to base."""
from copy import deepcopy
from dataclasses import dataclass, field


from src.geometry.context_roads import osm_maxspeed_mph
from src.geometry.cross_streets import cross_streets_from_model
from src.geometry.kerbs import kerb_openings_from_model
from src.geometry.model import leg_heads_toward
from src.geometry.targets import LegTarget, Side
from src.geometry.treatments.base import (DEFAULT_CENTERLINE_STYLE, Treatment,
                                          _parking_restrictions_from_model)
# The vocabulary of traffic_heads_toward itself (site_schema.Leg types the key with it), and
# already pinned equal to leg_frame._COMPASS_AXES by tests/test_site_schema.py - so this is the
# one place the four words live that is public. The import runs the harmless way round the
# config-path-stays-light contract: site_schema may not reach geometry, not the reverse.
from src.site_schema import VALID_TRAFFIC_DIRECTIONS
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel


# What OSM's `oneway` says about the direction of travel, as a flag on the WAY'S OWN
# digitisation direction: True where traffic runs along it, False where it runs against it.
# A TABLE RATHER THAN `== "yes"` BECAUSE OF `-1`, which is a real OSM value meaning the way is
# drawn backwards - and which appears on none of the 1,832 highway ways in the four downloaded
# snapshot areas, so nothing in this project's own data would ever catch it being read as `yes`.
# Everything absent here - `no`, unset, `reversible`, `alternating` - is NOT a one-way
# carriageway: a reversible one has no single direction, and leaning a parking bay or painting a
# yellow left edge line for the way it runs half the day is worse than drawing it as two-way.
_ONEWAY_ALONG_THE_WAY = {"yes": True, "true": True, "1": True, "-1": False}


def _osm_traffic_heads_toward(leg, tags: dict, aligned: bool) -> str | None:
    """The compass direction ALL traffic on `leg` runs, read off OSM, or None where the
    carriageway is not one-way.

    TWO FRAMES, AND THE TAG IS IN NEITHER OF THE ONES WANTED HERE. `oneway` is stated about the
    way's DIGITISATION direction; `aligned` is whether that direction is the leg's own outward
    one; the field wants a COMPASS. So `oneway=-1` on a way drawn backwards along its leg is
    traffic running outward, exactly as `oneway=yes` on a way drawn forwards - both flips, or
    neither.

    THE WORD IS CHOSEN BY ASKING THE FUNCTION THAT READS IT BACK. traffic_runs_outward answers
    `leg_heads_toward(leg, word)`, so picking the word that already gives the right answer there
    makes the seed and its consumer one derivation rather than two that agree until a leg bends.
    A leg is BLIND on the axis it runs perpendicular to - leg_heads_toward raises rather than
    answering from survey noise - so the words are tried until one of them can answer, and a leg
    running NNE is described on the axis it actually has a heading on.
    """
    along_the_way = _ONEWAY_ALONG_THE_WAY.get(tags.get("oneway"))
    if leg is None or along_the_way is None:
        return None
    runs_outward = along_the_way is bool(aligned)
    for compass in VALID_TRAFFIC_DIRECTIONS:
        try:
            if leg_heads_toward(leg, compass) is runs_outward:
                return compass
        except ValueError:
            continue        # blind on that axis; the other one answers
    return None


@dataclass(frozen=True)
class FacilityRefusal:
    """One stretch a facility does NOT cover, and the measurement that says so.

    A REFUSAL IS AN OUTPUT, NOT AN ERROR. A corridor plan that quietly stops at its hardest point
    is the plan nobody costed, so every refusal is carried out and drawn as a gap.

    HERE RATHER THAN IN corridor_paint, WHICH IS WHERE IT WAS. A refusal is a decision the design
    made - SKILLS 5's first row - and once a check has to distinguish "this kerb was measured and
    could not hold the section" from "this kerb was quietly skipped", the record has to be
    somewhere the design carries it. The alternative was a second dataclass of the same four
    fields at this layer, which is this repo's most reliable way of producing two answers to one
    question. Both renderers now build it: corridor_paint imports it from here.
    """
    start_ft: float
    end_ft: float
    reason: str
    narrowest_ft: float | None = None

    @property
    def length_ft(self) -> float:
        return self.end_ft - self.start_ft


@dataclass
class DesignState:
    """A mutable-by-copy snapshot of intersection geometry. Treatments clone the
    state, apply one change, and return the clone - so `state = bump_out(state, ...)`
    chains cleanly and the original scenario is never touched."""
    legs: dict
    corner_fillets: dict
    # leg name -> what is painted down that leg's middle TODAY, one of VALID_CENTERLINE_STYLES.
    # An OBSERVED FACT, not a treatment's parameter: seeded in from_model from config.yaml or
    # OSM's overtaking=no. What a PROPOSAL paints is a SetCenterlineStyle; ask
    # centerline_style() for the resolved answer or a proposal's change is invisible.
    existing_centerline_styles: dict = field(default_factory=dict)
    # leg name -> the compass direction ALL traffic on that leg runs, where the carriageway is
    # one-way; absent/None means two-way, which is the ordinary case. The same standing as the
    # styles above: an OBSERVED FACT seeded in from_model from OSM's `oneway`, or from
    # config.yaml where a site states one - not a treatment's parameter, so every scenario of a
    # junction gets the same answer, including the one the pipeline labels "Existing Conditions"
    # and builds without asking a site anything.
    #
    # HERE AND NOT ON Leg, though it is just as much a fact about the street, because a Leg is
    # REBUILT five times during load (fitting.py re-centres it on the traced kerbs, and each
    # rebuild carries its fields across by hand) - a sixth field would be dropped by whichever
    # of those the next change forgot, silently and only at a site whose kerbs are traced.
    # Read through traffic_runs_outward; nothing should dig it out of config a second time.
    traffic_heads_toward: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [KerbOpening]. Where OSM says the kerb is DROPPED for a
    # vehicle to cross - a driveway or yard entrance. Seeded in from_model from the traced kerbs'
    # kerb=lowered / kerb=flush tags; read by src/geometry/paint/ to break the kerbside
    # markings over it. See src/geometry/kerbs.py.
    kerb_openings: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [ParkingRestriction]. What OSM says about this kerb, per
    # STRETCH of it. Read by src/geometry/daylighting.py, which turns a prohibition into a
    # no-parking zone like any statutory one.
    parking_restrictions: dict = field(default_factory=dict)
    #: {leg name: [CrossStreet]} - every OTHER street a leg runs across. An observed fact like
    #: the two above. R.S. 39:4-138(e) applies at every intersection, not only the one the
    #: drawing is about, and a leg drawn 374 ft crosses several - see src/geometry/cross_streets.
    cross_streets: dict = field(default_factory=dict)
    #: {leg name: station in feet where the leg leaves this municipality}, absent where it does
    #: not. Observed like the three above, and the one thing on this state that bounds what may
    #: be BUILT rather than what is there - see src/geometry/intersection/municipality.py.
    municipal_limits_ft: dict = field(default_factory=dict)
    # Every Treatment applied to this design, in order (see apply) - the design as a list of
    # decisions. Every renderer reads its parameters from here, through treatment_for /
    # treatments_of / every_treatment, and provenance is written from it.
    treatments: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # (leg name, "left"|"right") -> [FacilityRefusal]. Stretches of kerb a corridor facility was
    # MEASURED against and could not take, written by CorridorFacility._place_on.
    #
    # NOT DERIVABLE FROM THE TREATMENTS, which is why it is a field. A treatment records what the
    # design DOES; these are spans where it deliberately does nothing, and the difference between
    # that and an oversight is invisible downstream - which is exactly what
    # BikewayReachesTheEndOfItsKerb has to tell apart before it can call a shortened facility a
    # defect. Printing the reason to stdout, as this used to, put it somewhere no check can read.
    facility_refusals: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> room beside a TARGET_LANE_WIDTH_FT lane, off the NOMINAL
    # half-width - the datum hold_travel_lane_at_target actually decides against, which is NOT
    # kerbside_allowance_ft's traced-kerb datum (see THE TWO DATUMS in that function's docstring).
    # Written so the plan view's kerb label can show the number that governed this kerb instead
    # of recomputing a different one and disagreeing with it - the same failure divider_ft's own
    # docstring already recounts once happening to the lane-width label.
    target_lane_room_ft: dict = field(default_factory=dict)
    # leg name -> the posted speed limit in mph, or None where neither OSM nor config states
    # one. An OBSERVED FACT like existing_centerline_styles above - resolved once in from_model,
    # by the same by-provenance precedence, so every consumer (EndTheBikeway's sharrow gate)
    # reads one answer instead of each re-deriving it. Ask through speed_limit_mph().
    speed_limits_mph: dict = field(default_factory=dict)

    def refuse(self, leg_name: str, side: str, refusal: FacilityRefusal) -> None:
        """Record that this kerb was measured over `refusal`'s span and cannot take the facility.

        Mutates, unlike everything else here, because a refusal is not a treatment: it is what the
        treatment layer learned while deciding, and threading a whole new state through
        `_place_on`'s loop for each rung it rules out would make the refusals a function of the
        order they were found in.
        """
        self.facility_refusals.setdefault((leg_name, str(side)), []).append(refusal)

    def refusals_on(self, leg_name: str, side: str) -> list:
        """Every stretch of this kerb the design measured and declined, in station order."""
        return sorted(self.facility_refusals.get((leg_name, str(side)), ()),
                      key=lambda refusal: refusal.start_ft)

    def record_target_lane_room(self, leg_name: str, side: str, room_ft: float) -> None:
        """hold_travel_lane_at_target's own room figure for this kerb - mutates, for the same
        reason refuse() does: this is what the treatment layer learned while deciding, not a
        treatment of its own."""
        self.target_lane_room_ft[(leg_name, str(side))] = room_ft

    def target_lane_room(self, leg_name: str, side: str) -> float | None:
        """The room hold_travel_lane_at_target found on this kerb, or None if it never ran here
        (e.g. a bike lane already owns this side, or apply_osm_parking decided it instead)."""
        return self.target_lane_room_ft.get((leg_name, str(side)))

    @classmethod
    def from_model(cls, model: "IntersectionModel") -> "DesignState":
        # PRECEDENCE IS BY PROVENANCE, not by which file the value came from (src/provenance.py).
        # A double yellow IS the no-passing marking, so OSM's overtaking=no is a direct statement
        # about the paint. An explicit config.yaml centerline_style is direct observation and
        # wins - but a config entry that merely repeats DEFAULT_CENTERLINE_STYLE is not an
        # observation, it is this repo's own placeholder, so it defers to the surveyed OSM tag.
        # Any other configured value (double_yellow, none) is a positive statement and wins.
        osm_tags = getattr(model, "leg_osm_tags", {})
        centerline_styles = {}
        for name, leg_cfg in model.config["legs"].items():
            configured = leg_cfg.get("centerline_style")
            no_passing = osm_tags.get(name, {}).get("overtaking") == "no"
            if configured is not None and configured != DEFAULT_CENTERLINE_STYLE:
                centerline_styles[name] = configured
            elif no_passing:
                centerline_styles[name] = "double_yellow"
                if configured is not None:
                    print(f"  NOTE: {name} is tagged overtaking=no in OSM - drawing a double "
                          f"yellow centerline, over the retained repo default in config.yaml. "
                          f"Set a non-default centerline_style there if you've observed "
                          f"otherwise.")
            else:
                centerline_styles[name] = DEFAULT_CENTERLINE_STYLE
        # SAME SHAPE, DIFFERENT FRAME. maxspeed is a direct per-leg statutory fact, same standing
        # as overtaking=no above, and it wins whenever it parses. But there is no repo-default
        # placeholder to catch here the way DEFAULT_CENTERLINE_STYLE does: config.yaml's
        # speed_limit_mph is a real number on every site (each one a transcription of the same
        # tag this now reads directly - e.g. sites/wbroad_lanning/config.yaml). What makes it
        # the weaker source is scope, not honesty - it is CORRIDOR-level, one number for a whole
        # config, while maxspeed is PER LEG, and a corridor can carry more than one posted speed
        # (nj31_wdelaware's config states 35 for NJ 31, over a W Delaware Ave leg OSM tags 25).
        # EndTheBikeway's sharrow gate is asked per leg-side, so the per-leg tag is the truer
        # frame and config is only the fallback for a leg OSM says nothing about.
        corridor_speed = (model.config.get("corridor") or {}).get("speed_limit_mph")
        speed_limits_mph = {}
        for name in model.config["legs"]:
            raw_speed = osm_tags.get(name, {}).get("maxspeed")
            observed_speed = osm_maxspeed_mph(raw_speed)
            if observed_speed is not None:
                speed_limits_mph[name] = observed_speed
                if corridor_speed is not None and corridor_speed != observed_speed:
                    print(f"  NOTE: {name} is tagged maxspeed={raw_speed!r} in OSM "
                          f"({observed_speed} mph) - using it over config.yaml's corridor-level "
                          f"speed_limit_mph={corridor_speed}, which is one number for the whole "
                          f"corridor and may not be what THIS leg is posted. Set a per-leg "
                          f"override if {corridor_speed} is the one that should govern here.")
            else:
                speed_limits_mph[name] = corridor_speed
        # THE THIRD FACT ON THE SAME PRECEDENCE, and the only one whose tag is not already in the
        # frame it is wanted in - see _osm_traffic_heads_toward for the two flips. `oneway=yes`
        # is a direct statement that the carriageway carries one direction, which is the whole of
        # what this field records, so reading it beats treating every unconfigured leg as
        # two-way: a SLICE has no config at all (src/geometry/network/slice_design.py builds
        # `{"legs": {slug: {"street_name": ...}}}`), so config as the sole source left
        # carriageway_is_one_way permanently false in every window.
        #
        # Config still wins where a site states one, as an observed centerline_style does: a
        # compass someone wrote down after looking at the street outranks a tag, and there is no
        # repo-default placeholder to discount here. Where the two disagree, say so - they cannot
        # both be right about which way a street runs.
        osm_aligned = getattr(model, "leg_osm_aligned", {})
        traffic_heads_toward = {}
        for name, leg_cfg in model.config["legs"].items():
            configured = leg_cfg.get("traffic_heads_toward")
            observed = _osm_traffic_heads_toward(model.legs.get(name), osm_tags.get(name, {}),
                                                 osm_aligned.get(name, True))
            traffic_heads_toward[name] = configured if configured is not None else observed
            if configured is not None and observed is not None and configured != observed:
                print(f"  NOTE: {name} is tagged oneway={osm_tags[name]['oneway']!r} in OSM, "
                      f"which runs its traffic {observed} along this leg - config.yaml says "
                      f"{configured} and wins. One of the two has the direction of a one-way "
                      f"street wrong; check the tag against the street.")
        return cls(legs=deepcopy(model.legs), corner_fillets=deepcopy(model.corner_fillets),
                   existing_centerline_styles=centerline_styles,
                   speed_limits_mph=speed_limits_mph,
                   traffic_heads_toward=traffic_heads_toward,
                   kerb_openings=kerb_openings_from_model(model),
                   parking_restrictions=_parking_restrictions_from_model(model),
                   cross_streets=cross_streets_from_model(model),
                   # getattr, like scene.py reads surveyed_leg_lengths: the synthetic models
                   # in tests are hand-built and carry only the fields their case is about.
                   municipal_limits_ft=dict(getattr(model, "municipal_limits_ft", None) or {}))

    def clone(self) -> "DesignState":
        return deepcopy(self)

    def centerline_style(self, leg_name: str) -> str:
        """What this design paints down `leg_name`'s middle: a proposal's choice, else what is
        there today.

        A SetCenterlineStyle outranks the observed fact from_model seeded. Both renderers go
        through here so they cannot disagree about which source won.
        """
        # Imported here, not at module scope: crossings sits ABOVE this module in the package's
        # layering (see __init__.py) and importing it up here would close the cycle.
        from src.geometry.treatments.crossings import SetCenterlineStyle

        treatment = self.treatment_for(SetCenterlineStyle, LegTarget(leg_name))
        if treatment is not None:
            return treatment.style
        return self.existing_centerline_styles.get(leg_name, DEFAULT_CENTERLINE_STYLE)

    def speed_limit_mph(self, leg_name: str) -> int | None:
        """The posted speed on this leg, or None if neither OSM nor config states one.

        None is a refusal, not a default: EndTheBikeway's SHARROW_MAX_SPEED_MPH gate already
        treats an unstated speed as licensing no sharrows, so this must not invent one. See
        from_model for where the answer is resolved and why OSM's per-leg tag outranks
        config.yaml's corridor-level figure.
        """
        return self.speed_limits_mph.get(leg_name)

    def travel_lane_divider_shift(self, leg_name: str) -> tuple[float, str] | None:
        """How far off the alignment this leg's centreline paint sits, and toward which side.

        None where nothing moved it: the alignment IS the divider unless a two-way bike lane on
        one side has pushed the travel lanes over. Returns distance AND side, because a distance
        with no side is half a fact (see Side.sign).

        Read by BOTH the plan view and the 3D export, which is why it lives here rather than in
        either: a shift one view honours and the other does not is a render whose two travel
        lanes are different widths.
        """
        # Same reason as centerline_style above - bikeways is layered above state.
        from src.geometry.treatments.bikeways import divider_shift_toward_ft

        # ASKED OF divider_shift_toward_ft AND OF NOTHING ELSE, which is the whole point: this
        # used to loop over AddTwoWayBikeLane by name first and consult that function only for
        # the legs it found, so a section pinned by a ONE-WAY lane shifted the travel way, the
        # checks measured the shift (they ask the function), and the PAINT stayed on the
        # alignment. NJ 35 NB is that case - a 5 ft lane behind a 20 ft angled bay, pinned - and
        # it came out with a lane line 3.70 ft off the middle of its own travel way, which is
        # two northbound lanes 3.70 ft different in width. The lookup was the second derivation
        # of one fact; see .claude/SKILLS.md section 2.
        #
        # CANONICAL FORM: a NON-NEGATIVE distance paired with the side it is actually on. The
        # sign is resolved here, once, rather than travelling alongside a side that can
        # contradict it - a consumer taking abs() of a signed shift draws the paint on the wrong
        # side of the alignment.
        #
        # The divider is NOT always on the far side. It is wherever a target-width lane from the
        # section's inner edge lands, and on a wide leg (broad_st_west) that is still short of
        # the alignment, i.e. the shift is toward the lane's own side.
        toward_left_ft = divider_shift_toward_ft(self, leg_name, Side.LEFT)
        if not toward_left_ft:
            return None    # nothing moved it - the alignment IS the divider
        if toward_left_ft > 0:
            return toward_left_ft, str(Side.LEFT)
        return -toward_left_ft, str(Side.RIGHT)

    def treatment_for(self, kind, target) -> Treatment | None:
        """The treatment of `kind` applied at `target`, or None if there is none.

        The last one applied wins: a design is a sequence of decisions and the later one is the
        decision, so two MarkedParking treatments on one kerb are one marked lane.

        This is how a treatment asks about ANOTHER treatment. A bollard row's precondition is "is
        there a buffered bike lane here", which is a question about a decision someone made - not
        about an entry under a key that anything, including a test, could have written.
        """
        found = None
        for treatment in self.treatments:
            if isinstance(treatment, kind) and treatment.target == target:
                found = treatment
        return found

    def treatments_of(self, kind) -> list[Treatment]:
        """Every treatment of `kind`, ONE PER TARGET (the last applied), sorted by target.

        Last-applied-wins per target, for the reason treatment_for gives. Painting both instead
        is what makes MarkingsDoNotCollide fire.

        SORTED BY TARGET rather than in application order, so what a consumer sees is a property
        of the design and not of the order a scenario builder's loops ran in - the props array is
        order-sensitive in the exported JSON.

        Where a treatment ACCUMULATES rather than replacing - ShiftCrosswalk, ExtraProp - this is
        the wrong question and every_treatment is the right one.
        """
        by_target = {}
        for treatment in self.treatments:
            if isinstance(treatment, kind):
                by_target[treatment.target] = treatment
        return [by_target[target] for target in sorted(by_target, key=str)]

    def every_treatment(self, kind, target=None) -> list[Treatment]:
        """Every treatment of `kind`, in application order, WITHOUT collapsing per target.

        For the two treatments that add up instead of replacing: ShiftCrosswalk shifts a crossing
        by a delta, ExtraProp puts one more sign on a leg. treatments_of would silently drop all
        but the last, which for a second RRFB on one leg is a prop that stops being drawn.
        """
        return [t for t in self.treatments
                if isinstance(t, kind) and (target is None or t.target == target)]

    def apply(self, *treatments: Treatment, model: "IntersectionModel" = None) -> "DesignState":
        """Apply treatments to a COPY of this design and return it.

        THE SINGLE WAY a treatment enters a design, so everything every treatment needs checked
        is checked here once: that the target exists at this junction (a leg-name typo must not
        silently do nothing), that a treatment declaring needs_model got one, and that
        state.notes records what was applied without each treatment remembering to append.

        Chains, so a scenario reads `state.apply(a).apply(b)` or `state.apply(a, b)`.
        """
        new_state = self.clone()
        for treatment in treatments:
            missing = treatment.target.missing_from(new_state)
            if missing:
                raise KeyError(f"{type(treatment).__name__} cannot be applied: {missing}")
            if treatment.needs_model and model is None:
                raise ValueError(
                    f"{type(treatment).__name__} needs the IntersectionModel - it reads geometry "
                    f"that the design does not carry. Pass model= (see src/site.py:run_scenario, "
                    f"which hands every scenario builder the model for exactly this).")
            measured = treatment.apply_to(new_state, model)
            new_state.treatments.append(treatment)
            new_state.notes.append(treatment.describe() + (measured or ""))
        return new_state
