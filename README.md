# nj-road-sketches

Street sketches of real New Jersey junctions, drawn from OpenStreetMap and nothing else: the
street as OSM maps it, and each proposal as an osmChange on top of it, rendered in Blender.
Three areas: Hopewell Borough, Pennington Borough and Lavallette.

**Everything is tags.** What exists is OSM's tags; what OSM does not yet say about the street as
it is goes in `proposals/<area>/existing.osc`; a proposal is another `.osc` built on that. Each is
a real osmChange - open it in JOSM over the area. The reader draws tags and decides nothing.

## Running it

```bash
.venv/bin/python scripts/render_osm.py hopewell_borough            # pull, write .osc, draw, render
.venv/bin/python scripts/render_osm.py hopewell_borough --cached   # from the last pull
.venv/bin/python scripts/write_osc.py hopewell_borough --cached    # the .osc files only
./scripts/test.sh                                                  # lint, contracts, tests
```

Renders land in `output/osm/<area>/<scenario>/<camera>.png`, one per camera `areas.yaml` gives
the area - each a junction, by its OSM node.

To know what a change did, measure it rather than render it:

```bash
.venv/bin/python scripts/verify.py save before    # before the change
.venv/bin/python scripts/verify.py save after     # after it
.venv/bin/python scripts/verify.py compare before after
```

## The pipeline

| step | where | what |
|---|---|---|
| OSM in | `src/sources/osm_context.py` | each area (`areas.yaml`) downloaded once from the OSM API and cached; an `.osc` applied to it (`src/sources/osm_change.py`); sorted into layers by tag |
| .osc out | `src/osm_osc.py` | `existing.osc` (widths, street areas, unpainted stretches, no-standing zones, streets re-centred on their kerbs) and each `proposals/<area>/<name>.toml` laid on it |
| drawn | `src/osm_world.py` | the world: every surface, kerb, building, marking and prop, from tags, on one street graph (OSMnx) |
| rendered | `scripts/blender/blender_scene.py` | the world built once into a `.blend`, then each camera rendered into it |

`standards.toml` holds every figure the code uses - value, unit, source, status - and every OSM
key the reader acts on, with what it means; `src/standards.py` loads and checks it. Each module's
docstring says how it works.

## A proposal

`proposals/<area>/<name>.toml` names the ways it changes by their tags and says what tags they
take - nothing in the code names a street. `[track]` lays a two-way protected bikeway on a
compass kerb with a road diet everywhere (`two_way_bikeway`); `[[retag]]` puts tags on the ways it
chooses and changes nothing else (`retagged`). `kerb_extensions = true` paints a curb extension at
each corner (`crossing:kerb_extension`).

## What OSM is missing here

Facts known on the ground that the drawing can only show once they are mapped in OSM:

- **Broad St & Greenwood Ave signal (node 104048876): `support=mast_arm`** - a full-width mast arm
  at all four corners (street-view review, Rollo, 2026-07-02). Untagged, the heads draw on poles.
- **South Greenwood Ave has no centre line painted** (street view): `lane_markings=no`.
- **Borough Schedule I: no parking 100 ft each way on both Broad St legs at Greenwood** - not
  tagged, so stalls are drawn there: `parking:<side>:restriction=no_parking` on those stretches.
