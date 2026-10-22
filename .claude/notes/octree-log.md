# Octree log — the dated record behind `.claude/rules/octree.md`

Reference-only (never auto-loads). Configuration for every Sozzi number here unless stated: Linux x86_64,
4 cores, jax 0.11.2, Python 3.13.16; lamp 25,876 facets (chord 5e-5, facet 4 mm), absorption 35.67 /m,
black walls; ln G floored at 1 W/m2; adaptive roots 16 mm, `min_level` 1, 5 levels;
`validation/sozzi_radiation/adaptive_grid.py`. The study's table is in that directory's README.

## The interpolant: four designs before the hierarchical one with refitted ghosts (2026-10-15/16)

1. *Dry nodes at zero surplus*: cells the boundary cuts interpolate toward whatever the coarse tree made
   the dry corners; near the sleeve, garbage.
2. *Per-cut-cell correction / per-cut-cell trilinear with fitted corner ghosts*: double counts or drops
   hanging nodes. Finer neighbours' nodes on a cut cell's faces carry surpluses measured against the junk
   dry-corner values, so a correction fitted to the corners counts their reach twice (13 % at a point
   2 mm inside a test box), and interpolating the cut cell from its own corners ignores them (sampled
   hanging nodes missed by up to 0.3 %).
3. *Ghost fitted once, when the node first appears*: its coarse, wide extrapolation is what every finer
   surplus is measured against, so its error never refines away (-0.087 in ln at a node on the test
   vessel's end plane, misses stuck there to the deepest level).
4. *Trilinear per leaf with 2:1 balance and hanging-node rules*: the plan agreed in the issue thread,
   replaced by the hat basis, which has no hanging-node cases to cap (reported to the owner as a
   deviation).

## Nodes and cut points near the boundary (2026-10-16)

- A lattice node exactly on the sleeve's true surface lies in the plane of the lamp's facets; the gather
  returned 3e-13 W/m2 instead of ~1300 and its neighbours missed by 3.5-7 in ln G. An earlier `< 0` rule
  had the opposite failure on lattice-aligned planar faces (never sampled, never cut).
- Along-the-edge cut points came first: a standoff along the edge put oblique cut points much nearer the
  boundary, and an edge running along it (a lattice line tangent to a cylinder) put one on the surface
  (15 % error); a clearance rule on top either dropped informative oblique points or kept points 1 um from
  a faceted lamp.
- Facet ripple (stage `ripple`): at a fixed distance from the sleeve the gathered field varies around it
  and along one facet by a relative std of ~2 % (range 4-9 %) within 0.1 mm, 0.7 % (2.6 %) at 0.5 mm,
  0.46 % (1.7 %) at 1 mm, 0.16 % (0.6 %) at 2 mm.
- Without the at-the-end crossing, a boundary along a lattice plane (the vessel's flat ends) had no cut
  points and refined to the deepest level along the plane.

## The ghost's extrapolation and its bound (2026-10-16)

- Uncentred minimum-norm least squares shrank the intercept (one diagonal neighbour v gave v/3: 2.3
  against 7.1). A reach of one cell left a node at a convex edge with data in one plane, no gradient
  across it (misses stuck at 0.21; the linear-field test failed by 0.006). A reach of two unweighted let
  far samples of a curved field pull the fit (27 % at cut points). Weighted by inverse squared distance,
  both pass.
- Unbounded, on the tolerance-0.02 arm at a 0.5 mm standoff (1.2M samples): reference points within the
  standoff had p99 error 358 %, max 2080 % -- a ghost inside the lamp, refitted from samples bunched
  together, extrapolated the facets' ripple into 5e17 W/m2. With the bound and `EXTRAPOLATION_WEIGHT`
  0.01 -> 0.1 the band was 4.1 % median / 14 % p99; other bands moved by up to ~10 % of their own values.

## Probes (2026-10-16)

Tolerance 0.05, standoff 0.1 mm, no probes: a strip of 4 mm leaves along the whole lamp at one angular
position, two corners 0.68 mm inside the lamp, never split, overshot by 10 % (G ~1090 vs ~990) from 0.5 to
2.5 mm off the sleeve -- every one of the worst points in the 0.5-1, 1-2 and 2-5 mm bands. A nearer
standoff made it worse (1-2 mm p99 3.3 % at 0.25 mm, 8.5 % at 0.1 mm): the cut point sits higher on the
convex profile, so the ghost is higher and the straight line through the cell overshoots more.

## The standoff (2026-10-16)

The first full run (0.1 mm along the edge, before depth-based cut points, the ghost bound and probes)
never converged: misses of ~0.5 at the deepest level, 73 % of them at cut points within 0.07 mm of the
sleeve, 540k samples. With probes, 1-2 mm median / p99: 0.5 mm standoff 0.63 / 4.0 % (236k samples);
0.25 mm 0.37 / 2.2 % (364k); 0.1 mm 0.34 / 4.2 % (494k); the default (1e-3 of the finest cell, 0.5 um)
0.16 / 0.82 % (1.09M, 233k of them sleeve cut points), and 0-0.5 mm 6.6 / 19 % -> 1.1 / 4.4 %. One more
level (0.25 mm cells) cost 2.2M samples and moved nothing beyond 0.5 mm. The deepest-level misses are real
field features (the x = 0 end wall meeting the sleeve: +25 % in 1 mm off the wall; the pipe junctions),
smooth in depth down to 0.5 um. On a uniform grid the same standoff makes the near field worse (4 mm,
1-2 mm median 2.0 % -> 4.0 %).

## Mutation passes

- 2026-10-16: 26 one-line bugs over `octree/`, `io/vtk` and `fluence_rate_at`: 23 red. Green and left:
  `locate`'s `found < 0` guard (leaves partition the lattice); `exit_points` also taking the near end (its
  start is a sampled node); the extrapolation box open at reach 2 (samples on its outer edge weigh 1/8).
  A fourth green -- cut-point misses ignored in the split decision -- was a real gap, now pinned.
- 2026-10-16, probes: dropping the probe's miss from the split decision, and leaving the last pass's
  probes out of the interpolant, each red (the second only with `depth=4`, whose last pass probes).
- 2026-10-22, the shared tree: `locate` without the upper-face clamp, `lattice` ignoring `depth`, and
  `point_level` skipping level 0 each red.
