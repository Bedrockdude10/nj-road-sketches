# Working in this repo without reinventing it

Read this before changing what is drawn. It is not a summary of README.md: it is what agents here
get wrong, and where each answer already lives.

## 0. Two derivations of one fact

Every serious bug in this repo's history has one shape: the same fact computed in two places, each
right on its own terms, disagreeing with each other. Two, found and fixed by folding them into one:

| the two copies | what it cost |
|---|---|
| the carriageway's default width summed each side's section with `width or default`, so `shoulder:*:width=0'` read as absent and a 1 m default; the drawing read it as 0 | every side street with no `width` in the proposals had its lanes laid 1 m off centre, its curb extensions on phantom asphalt, and 16 corners reported narrowed that were not |
| a field observation added `support=mast_arm` to the snapshot, and the `.osc` - which re-centring wrote that node into - carried OSM's own tags | the mast arm was gone from every render, with nothing to say so |

So: **one home per fact, one method per question.** Before writing a lookup, find the one that
exists (`scripts/whatis.py <name>`); when two call sites make the same lookup, fold them into one
in the same change. And when something looks wrong, look for the second copy first.

## 1. Where each fact already lives

| you need | it is | in |
|---|---|---|
| a sided key: `<key>:<side>`, else `<key>:both` | `sided(tags, key, side, rest)` | `src/osm_world.py` |
| a side's cycleway, with the bare key | `cycleway`, `has_cycleway` | `src/osm_world.py` |
| a oneway, either direction | `is_oneway` | `src/osm_world.py` |
| a tagged width; a width or its default | `tagged_width`, `width_or_default` | `src/osm_world.py` |
| what a side's section lays, in metres | `kerb_section` | `src/osm_world.py` |
| how a crossing is painted | `crossing_markings` | `src/osm_world.py` |
| per-lane values (`turn:lanes`, `change:lanes`) | `per_lane`, `lane_widths` | `src/osm_world.py` |
| a line's direction at a station | `tangent` | `src/osm_world.py` |
| junctions, legs, corners, what carries a way on | `StreetGraph` - the ONE street graph (OSMnx) | `src/osm_world.py` |
| every road at a node, service roads too | `StreetGraph.spokes` | `src/osm_world.py` |
| every paint channel and surface, and how each is drawn | `PAINT`, `SURFACES` (the world JSON carries them to Blender) | `src/osm_world.py` |
| a world's kerbs, stop lines, crosswalks, junction mouths, no-standing datums | `_World` - read once | `src/osm_osc.py` |
| an area's bbox and its render cameras | `areas.yaml` | repo root |
| a figure | `si("...")` | `standards.toml` |

Nothing names a street or a junction: a proposal's `.toml` chooses ways by their tags.

## 2. Everything is tags

The reader (`src/osm_world.py`) draws tags and decides nothing; there is no reader rule, special
case or threshold that is not a tag's meaning. A treatment is tags or features written into an
`.osc` (`src/osm_osc.py`); look the tagging up on the OSM wiki, and where OSM has none, say it is a
project convention in standards.toml's `[tag]` table. A fact about the street that OSM is missing
goes into OSM (README lists the ones known), or - where it is measured from OSM, like a width -
into `existing.osc`. Never a second overlay beside OSM: that is how the mast arm was lost (§0).
Prefer deleting a rule to adding one, and never special-case one place.

## 3. Measure, then render

`scripts/verify.py save before` → change → `save after` → `compare before after`. It says which
`.osc` moved, which report line, and which world channel by how many pieces. A change meant to
move nothing must show every `.osc` identical; anything that moved is explained before it is
committed - by putting the old rule back and watching the difference go, not by a picture.

## 4. OSM is partial, in specific ways

- **A restriction over part of a kerb does not close it.** OSM splits the way; read each piece.
- **A driveway's side is read along its first segment**, not at the node it shares with the street,
  where the offset is float noise.
- **The cache is the pull.** `output/.cache/borough_*.json` is re-pulled only by `render_osm.py` or
  `write_osc.py` without `--cached`; a mapping fix reaches the drawing on the next pull.
