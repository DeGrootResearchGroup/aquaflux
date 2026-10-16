---
paths:
  - "aquaflux/octree/**"
  - "tests/unit/test_octree.py"
---

# Rules — `aquaflux/octree/` (adaptive sampling of a field on an octree, #726)

> **Provenance boundary (binding).** As with every rule file: what you read here informs your
> understanding, and none of it may reach the shipped surface. See the root `CLAUDE.md`
> **Comment Convention**.

Sample a costly, mostly smooth field — the fluence rate, an independent sum over a lamp at each point
— on an octree refined only where the field needs it, and interpolate it to any points afterwards (a
flow mesh's cell centres, for a dose). Built for issue #726 (adaptive refinement of the radiation
background grid). **A generic package** (`tests/unit/test_layering.py::GENERIC_PACKAGES`): it imports
nothing of aquaflux but itself and the neutral leaves (`ragged`). The field is a callable
(`sample(points) -> values`), the region a `solids.Solid` used only through
`signed_distance` and `intervals` — duck-typed, so `octree` does not import `solids`. The radiation
side is one function, `radiation.fluence_rate_at(scene, points)`.

## Layout

| module | what |
|---|---|
| `tree.py` | `Octree`: root cells of edge `spacing` over a box, leaves as `(levels, low)` arrays on ONE integer lattice (the finest cells' corners), `split`, `locate`, `node_level`, `key`/`unkey`. `CORNERS` (binary order, corner `4i+2j+k`), `EDGES`. |
| `hierarchy.py` | `Hierarchy`: per-level sorted node keys and **surpluses**; the interpolant is `sum surplus * hat` (`at_lattice`), `with_values` adds nodes in ascending level. |
| `cuts.py` | `merged_intervals`, `crossing_points`, `exit_points`, `leaf_cuts` (`LeafCuts`, CSR per leaf): cut points on cell edges where the region is a `standoff` deep. One placement rule, `_cut_parameters`/`_at_depth`, shared by both point functions. |
| `ghosts.py` | `ghost_values`: values for new nodes OUTSIDE the region (edges to sampled neighbours + a lightly weighted linear extrapolation, `EXTRAPOLATION_WEIGHT`). |
| `region.py` | the region's two queries (`signed_distance`, `intervals`), `jax.jit`-compiled per power-of-two batch size (`SMALLEST_BATCH`..`LARGEST_BATCH`). |
| `sampling.py` | `sample_adaptively` → `SampledField` (callable; `hexahedra()`), `Linear`/`Logarithm` transforms, `Round` per pass, `_Refinement` (the pass loop's state). |

Output: `SampledField.hexahedra()` → `aquaflux.io.write_hexahedra(points, corners, path, point_fields=,
cell_fields=)` (in `io/vtk/writer.py`; corners in binary order, mapped to VTK's hexahedron order
there, `_BINARY_TO_VTK_HEXAHEDRON`; `vtu_parts` gained `point_fields`, written as `PointData`).

## Binding decisions, and why

- **The interpolant is HIERARCHICAL (hat functions, surplus per node), not trilinear per leaf with
  hanging-node rules — and so needs NO 2:1 balance.** It is continuous wherever a coarse cell meets finer
  ones (a sum of continuous hats), passes through every node, and a node's surplus IS the miss of the
  coarser tree at it — the refinement criterion the project owner proposed ("interpolate a level down
  from the level up; split where it misses"). ⚠️ This replaced the plan agreed in the issue thread
  (trilinear per leaf + 2:1 balance + hanging-node cases); the balance was only there to cap the
  hanging-node cases, which the hat basis does not have. Reported to the owner as a deviation.
- **Misses are measured, not estimated**: every pass compares each NEW sample (node or cut point)
  against the previous pass's interpolant, before adding it. A new leaf is split when a new sample on
  its closure (its new corners, the new cut points on its 12 edges) missed by more than `tolerance`, or
  when it is coarser than `min_level`. The first pass has nothing to predict, so `min_level=0` refines
  nothing.
- **A node outside the region gets a GHOST value entered in the hierarchy, RE-FITTED whenever finer
  cells appear around it, and the surpluses are REBUILT from every node's value each pass**
  (`_Refinement._set_ghosts`, `Hierarchy.empty(...).with_values(all)`). Sampled values never change; a
  ghost is an estimate. ⚠️ Four designs were built and measured before this one, each a trap:
  1. *Dry nodes at zero surplus, nothing else*: cells the boundary cuts interpolate toward whatever the
     coarse tree made the dry corners; near the sleeve, garbage.
  2. *Per-cut-cell correction / per-cut-cell trilinear with fitted corner ghosts*: **double counts or
     drops hanging nodes.** Finer neighbours' nodes on a cut cell's faces carry surpluses measured
     against the junk dry-corner values, so a correction fitted to the corners counts their reach twice
     (13 % at a point 2 mm inside a test box), and interpolating the cut cell from its own corners
     ignores them (sampled hanging nodes missed by up to 0.3 % — caught by
     `test_the_interpolant_passes_through_every_sampled_node_and_near_every_cut_point`).
  3. *Ghost fitted once, when the node first appears*: a coarse node's ghost comes from a coarse, wide
     extrapolation and every finer surplus is measured against it, so its error never refines away
     (−0.087 in ln at a node on the test vessel's end plane, misses stuck there to the deepest level).
  Rebuilding costs a `with_values` over every node per pass (all levels, 8 lookups each).
- **A node is SAMPLED only when it lies at least `standoff` inside the region** (`signed_distance <=
  -standoff`); nearer, it is a ghost. ⚠️ Measured on the Sozzi drawing: a lattice node exactly on the
  sleeve's true surface lies in the plane of the lamp's facets and the gather returned **3e-13 W/m²
  instead of ~1300**; its neighbours' misses were 3.5–7 in ln G and the refinement chased them. (An
  earlier `< 0` rule had the opposite failure on lattice-aligned planar faces — nodes on them never
  sampled and never cut; the standoff rule fixes both, with the at-the-end crossing below.)
- **A cut point is placed where the region's SIGNED DISTANCE is `-standoff`**, on its edge, between
  the crossing and the middle of the region the edge holds there (`cuts._at_depth`, 40 bisections), and
  none where that middle is not a standoff deep. So **nothing nearer the boundary than the standoff is
  sampled**, by normal distance, whatever angle the edge meets the boundary at — the same rule as for
  nodes. ⚠️ Two along-the-edge versions came first: (1) a standoff along the edge put oblique cut points
  much nearer the boundary, and an edge running ALONG it (a lattice line tangent to a cylinder) put one
  ON the surface (15 % error; a zero gather with a real lamp); (2) a clearance rule on top (half a
  standoff, then 1 %) either dropped informative oblique points or kept points 1 µm from a faceted lamp.
  **Why normal distance matters, measured** on the Sozzi lamp (25,876 facets, chord 5e-5, 4 mm):
  at a fixed distance from the sleeve the gathered field varies around it and along one facet by a
  relative std of ~2 % (range 4–9 %) within 0.1 mm, 0.7 % (2.6 %) at 0.5 mm, 0.46 % (1.7 %) at 1 mm,
  0.16 % (0.6 %) at 2 mm (`validation/sozzi_radiation/adaptive_grid.py` stage `ripple`). Samples in
  that layer carry the lamp's triangulation, and a refinement asked for less than it chases the facets:
  the first full Sozzi run (standoff 0.1 mm along the edge, tolerance 0.05) still had misses of ~0.5 at
  the deepest level, 73 % of them at cut points within 0.07 mm of the sleeve, and took 540k samples.
- **A crossing within `_AT_THE_END` (1e-9 of the segment) of its end is a crossing** — an edge ending
  on a boundary that lies along a lattice plane (the vessel's flat ends). Without it that whole layer
  had no cut points and refined to the deepest level along the plane. `exit_points` uses only the far
  end's point (the start is in the region).
- **The ghost's extrapolation**: a linear fit to the sampled nodes and cut points within
  `_EXTRAPOLATION_REACH` (2) cells of the ghost's scale, **weighted by inverse squared distance**
  (floored at `NEAREST` = 0.5 cell), **about the weighted mean**. ⚠️ Each part was a measured fix:
  uncentred minimum-norm least squares shrank the intercept (one diagonal neighbour v gave v/3: 2.3
  against 7.1) — `test_one_diagonal_sample_extrapolates_as_a_constant`; a reach of one cell (the hat)
  left a node at a convex edge with data in one plane, no gradient across it (misses stuck at 0.21; the
  linear-field test failed by 0.006); a reach of two unweighted let far samples of a curved field pull
  the fit (27 % at cut points); weighted, both pass. The hat-box query includes its edges
  (`r*(1+1e-9)`, Chebyshev): the outside neighbours' cut points lie exactly on it.
- **A ghost is BOUNDED to the range of the samples it was fitted to, widened by that range either side,
  and pulled toward the extrapolation at `EXTRAPOLATION_WEIGHT` = 0.1 (was 0.01).** ⚠️ Measured on the
  Sozzi tolerance-0.02 arm (1.2 M samples): without the bound, reference points within the 0.5 mm standoff
  had p99 error **358 %, max 2080 %** — a ghost inside the lamp, refitted at 0.5 mm from samples bunched
  together, extrapolated the facets' ripple into **5e17 W/m²**. With the bound and the stronger pull the
  same band is 4.1 % median / 14 % p99; the other bands moved by up to ~10 % of their own values (1-2 mm
  median 0.115 % → 0.126 %, p99 1.06 % → 1.17 %), samples 1,235,164 → 1,201,461. Pinned
  by `test_a_ghost_cannot_run_away_from_the_samples_it_was_fitted_to` and the closed form in
  `test_an_edge_bends_the_ghost_away_from_the_straight_line_guess`.
- **Mutation check, 2026-10-16** (26 one-line bugs over `octree/`, `io/vtk` and `fluence_rate_at`, each
  against `test_octree.py`, `test_vtk_hexahedra.py` and the scene test): 23 red. Three green, judged
  unreachable or indistinguishable and left: `locate`'s `found < 0` guard (leaves partition the lattice,
  one level hits); `exit_points` also taking the near end (its start is a sampled node, a standoff deep,
  never on the boundary); the extrapolation box open at reach 2 (samples exactly on its outer edge weigh
  1/8 and move nothing a test sees). A fourth green — cut-point misses ignored in the split decision —
  was a real gap, now pinned by `test_the_cut_points_misses_take_the_cells_beside_the_rod_to_the_deepest_level`
  (without them no cell beside the rod reaches the deepest level at tolerance 0.05).

## Known limits (recorded so silence is not read as coverage)

- **A convex edge of the region aligned with the lattice** (a box's edge) is not refined by misses: the
  cells along it have few samples, all of which agree. The worst unit-test point (2.6 %) sits there.
  Round vessels do not have it.
- **A region entering a cell through a face without crossing an edge** is seen only through the
  signed-distance test in `reached` (does the cell get split at all), never by a cut point.
- **The exchange is re-solved per pass** when the scene reflects (`fluence_rate_at` docstring): fine
  for black walls, wasteful with reflectors. Not built: a scene object that solves the exchange once
  and gathers many times.
- **Not wired into case files yet** (the issue's "receivers: the field to tolerance X").
- Every sample is kept for the field's lifetime; memory is the sample table plus per-level key arrays.

## Measured

See `.claude/rules/validation.md` → `sozzi_radiation/adaptive_grid.py` for the Sozzi comparison with
uniform grids, with its configuration.
