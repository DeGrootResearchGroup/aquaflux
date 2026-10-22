---
paths:
  - "aquaflux/octree/**"
  - "tests/unit/test_octree.py"
---

# Rules — `aquaflux/octree/` (the octree, and adaptive sampling of a field on it)

> **Provenance boundary (binding).** As with every rule file: what you read here informs your
> understanding, and none of it may reach the shipped surface. See the root `CLAUDE.md`
> **Comment Convention**.

**A generic package** (`tests/unit/test_layering.py::GENERIC_PACKAGES`): imports nothing of aquaflux but
itself and the neutral leaves (`ragged`). Two consumers: the mesher's hex core (`meshing/hex_core.py`,
refined by `meshing/near_surface.py::NearSurface`, `.claude/rules/meshing.md`) and the adaptive sampler
here (#726), whose field is a callable and whose region is a `solids.Solid` used only through
`signed_distance` and `intervals`. Radiation's side is `radiation.fluence_rate_at(scene, points)`; output
through `aquaflux.io.write_hexahedra`. Dated record: `.claude/notes/octree-log.md`.

## Layout

| module | what |
|---|---|
| `tree.py` | **`Octree`, the one octree** (`origin`, `unit`, `depth`, `shape`, `level`, `corner`): `lattice(lower, upper, size, depth=0)`, `refined` (children after the leaves kept, `CORNERS` order), `refined_by` + `RefinementCriterion`, `balanced`, `subset`, `leaf_at` (finest cells), `locate` (points), `cube_corners`, `point_level`, `keys`/`coordinates`, `extent`, `span_at`. `CORNERS` (corner `4i+2j+k` at `(i,j,k)`), `EDGES`, `BITS`. |
| `hierarchy.py` | `Hierarchy`: per-level sorted node keys and **surpluses**; the interpolant is `sum surplus * hat` (`at_units`), `with_values` adds nodes in ascending level. |
| `cuts.py` | `merged_intervals`, `crossing_points`, `exit_points`, `leaf_cuts` (`LeafCuts`, CSR per leaf): cut points on cube edges where the region is a `standoff` deep. One placement rule, `_cut_parameters`/`_at_depth`. |
| `ghosts.py` | `ghost_values`: values for nodes outside the region (edges to sampled neighbours + a lightly weighted linear extrapolation, `EXTRAPOLATION_WEIGHT`), bounded by the samples' range. |
| `region.py` | the region's `signed_distance` and `intervals`, `jax.jit`-compiled per power-of-two batch. |
| `sampling.py` | `sample_adaptively` → `SampledField` (callable; `hexahedra()`), `Linear`/`Logarithm`, `Round` per pass, `_Refinement` (the pass loop's state), `STANDOFF_FRACTION`. |

## The tree (binding)

- **One `Octree` for both consumers.** There were two (`meshing/octree.py` and this package's) holding the
  same data -- leaves as a level and a lower corner in integer units, children appended after the leaves
  kept -- with different operations. They were merged here (2026-10-22) because the sampler may not import
  `meshing`. A new octree operation goes on this class, whichever consumer needs it.
- **`depth` is the finest level, and splitting a leaf at it halves `unit`** and doubles every coordinate
  (how the mesher's tree grows). A caller that keeps keys across passes builds at its deepest level
  (`lattice(..., depth=d)`) and never splits past it, so `unit` and every key stay fixed: the sampler does.
- **Points are keyed by `Octree.keys`** (`BITS` = 21 per axis; the hex core keys half-unit points, so a tree
  spans at most `2**(BITS-1)` units per axis, checked in `_check_span`); `Octree.coordinates` is the one
  decoder.
- `balanced()` is the mesher's: the sampler's interpolant needs no balance (below).

## The sampler: binding decisions, and why

- **The interpolant is HIERARCHICAL (hat functions, surplus per node), not trilinear per leaf**, so it
  needs no 2:1 balance: continuous wherever a coarse cube meets finer ones, through every node, and a
  node's surplus is the coarser tree's miss there -- the refinement criterion the project owner proposed.
- **Misses are measured, not estimated**: each new sample (node, cut point) against the previous pass's
  interpolant, before it joins. A new leaf is split when a sample on its closure (new corners, new cut
  points on its 12 edges, its probe) missed by more than `tolerance`, or below `min_level`.
- **A leaf with a corner not sampled gets a PROBE at its centre** (where a standoff inside), missed
  against the interpolant the pass just built and then entered as a node of the level below: its
  unsampled corners are ghosts fitted to its cut points, so nothing else tests its interior. Probes share
  the pass's one `sample` call and are kept out of that pass's ghost fit and rebuild.
- **A node outside the region (or within the standoff) gets a GHOST, refitted whenever finer cubes appear
  around it, and the surpluses are rebuilt from every node's value each pass.** Sampled values never
  change. The extrapolation is a linear fit to samples within `_EXTRAPOLATION_REACH` (2) cubes, weighted by
  inverse squared distance (floored at `NEAREST`), about the weighted mean; the ghost is clipped to the
  samples' range widened by that range either side. Each part was a measured fix (log).
- **Nothing nearer the boundary than the standoff is sampled, by signed distance**: a node only at
  `signed_distance <= -standoff`; a cut point where the signed distance is `-standoff` on its edge
  (bisection, `_at_depth`), none where the edge is never that deep. A crossing within `_AT_THE_END` of a
  segment's end counts (a boundary along a lattice plane).
- **The standoff defaults to `STANDOFF_FRACTION` (1e-3) of the finest cube** (project owner, 2026-10-16): a
  fixed length does not travel between cases, and the region is the drawn shape -- a convex lamp's facets
  lie inside it, so any point of the drawn water clears them; only the boundary itself must be avoided (a
  node on it can lie in a facet's plane). A concave emitter's facets bulge past the drawing by up to the
  chord; there the caller passes a standoff beyond it. ⚠️ A tiny standoff pays only with refinement near
  the boundary: on a coarse uniform tree it makes the near field worse (log).

## Known limits

- A convex edge of the region aligned with the lattice (a box's edge) is not refined by misses: the cubes
  along it have few samples, all of which agree.
- A region entering a cube through a face without crossing an edge is seen only through `reached`'s
  signed-distance test, never by a cut point.
- A scene that exchanges light re-solves the exchange per pass (`fluence_rate_at`).
- Not wired into case files (the issue's "receivers: the field to tolerance X"); `background_grid.py` not
  ported onto it.
- Every sample is kept for the field's lifetime.
