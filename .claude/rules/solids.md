---
paths:
  - "aquaflux/solids/**"
---

# Rules — `aquaflux/solids/` (solid bodies, their set combinations, fluid regions)

A body answers whether a segment passes through it (`blocks`), whether a point is inside it (`contains`),
and declares whether those are pure array expressions (`traceable`). That contract is `Body`; everything
else in `solids/bodies.py` is an exact, formula-based implementation: analytic primitives, constructive
solid geometry (CSG) over them, and `Outside`, a vessel described by the fluid it holds. Mutation passes
and measurements: `.claude/notes/solids-log.md`.

**Placement.** Generic geometry, so its own package (it began as `radiation/occluders.py`; the CAD reader
in `aquaflux/io/cad/` must emit bodies without importing a physics package). `tests/unit/test_layering.py`
holds it to `solve/`'s rule: it imports nothing of aquaflux but itself and neutral leaves (`vectors`).
What a body does to light (transmittance, frozen vs live masks) and the Sozzi `Outside` comparison are
radiation's (`.claude/rules/radiation.md`).

## ⚠️ Form the cylinder's discriminant as `a(r^2 - h^2)`, never as `b^2 - a c`

One home: `_Tube`, sharing `_positive_definite_span` with `_Ball`, reached by every body with a round
side. The naive form cancels catastrophically exactly at grazing and reports zero for rays that pass
inside — light leaks through. Grazing is the ordinary geometry here (a sleeve sits at the radius of the
lamp facets it surrounds). `_Taper` cannot use it (a cone's quadratic is indefinite), which is why a
`Cone` with equal end radii is refused — use a `Cylinder`.

Hygiene, each mutation-pinned: clamp the hit to the segment (a body beyond the receiver does not occlude);
exclude a sliver next to the source sized as a fraction of the facet's own `sqrt(area)`, never an absolute
epsilon; clip the cylinder to its flat ends.

- **A point inside a body is refused at build** (facets and receivers both).
- **Do not represent a body twice**: an emitting sleeve must not also be an occluder (every ray would
  leave a facet lying on it; the emitter's convexity already makes the source-side clamp exact).

## Blocking geometry as CAD primitives: three layers, and why the middle one is intervals

| layer | promise | implementers |
|---|---|---|
| `Body` | answers `blocks` and `contains` | anything, including host code over triangles |
| `Solid` | its inside along a line is a bounded union of intervals | every body here |
| `ConvexSolid` | that union is a single interval | `HalfSpace`, `Sphere`, `Cylinder`, `Cone`, `Box` |

Blocking is "does any interval overlap the segment"; `Union` pools intervals, `Intersection` overlaps
them pairwise, `Difference` cuts one body's by another's — one implementation per operation, not per shape
pair. The interval count (`interval_count`) is static per body, so the mask is one array expression with
no data-dependent search.

A convex body is a list of inequalities: private `_Plane`, `_Tube`, `_Ball`, `_Taper` each give a signed
distance and an interval of one `f(x) <= 0` (cylinder = tube + 2 planes, box = 6 planes, cone = taper + 2
planes), so `contains` and the interval cannot disagree about a surface.

⚠️ **`Difference` takes a convex hole** (a multi-piece complement needs sorting/merging). Write
`A - (B + C)` as `Difference(Difference(A, B), C)`; each cut doubles the interval count. The constructor
refuses a multi-interval hole.

## The fluid as regions: `Outside`, and why it beats a hand-derived occluder

`Outside(chamber, inlet, riser)` is everything those regions are not; a segment is clear exactly when the
regions' pooled intervals cover it end to end. It replaces per-reactor closures like
`validation/sozzi_radiation/compare_fluence.py`'s `BranchOpenings` and handles any chamber-pipe chain.
Any `Solid` may be a region; convex regions keep it cheap (one interval each).

- ⚠️ **Neighbouring regions must overlap or touch — a gap reads as solid**, silently, as a shadow. Extend a
  pipe's cylinder into the chamber rather than stopping it at the curved surface.
- ⚠️ **The covering test examines each interval's far end as well as the segment's start, with a strict
  comparison** (the pool must continue past, not merely reach). Dropping either half reports a gap as
  covered; pinned by `test_a_gap_between_two_regions_reads_as_solid`. Do not sort intervals: a sort over a
  receivers-by-facets array materializes a mask-sized permutation, while the pairwise test fuses.
- ⚠️ **`Outside.blocks` applies `min_distance` at both ends**: a facet centroid used as a receiver sits on
  the wall, and a rounding leaves an uncovered sliver at `t = 1` — a shadow over the whole enclosure.
- **`contains` means "outside every region"** (a point embedded in the wall, refused at build), with a
  length `tolerance` because a cell centre a rounding outside a region is discretization. The tolerance
  applies to `contains` only; segment clearance has no slack.

## Clearance: a body vouching for a whole convex hull

`Body.clearance(position) -> (..., n_features)` and `Body.vouches(summary) -> (...)`. **Contract:** a set
is summarized by the column-wise max of `clearance` over its points (sets merge by max); where `vouches` is
true, no point of the set's convex hull is in the body. It answers "clear" or "don't know", never
"blocked". Radiation's `ShaftCulling` uses it to certify receiver-source tiles without visiting pairs.

- Default `vouches` is `any(summary < 0)` (columns are witnesses; zero columns never certify). Radiation's
  `TriangleBody` (a `Body`, not a `Solid`) overrides it with a bounding-box query over `[x, -x]` features.
  ⚠️ `Intersection` concatenates members' columns, valid only because every `Solid` keeps the default
  `vouches`; a `Solid` overriding it could not be composed this way.
- **`Body.vouches_tiles(first, second, rows, cols)`** answers many tiles (tile `t` merges
  `first[rows[t]]` with `second[cols[t]]`). Default: host numpy `maximum` + `vouches`. An override must give
  the same answers; `TriangleBody`'s compiled loop re-implements the max merge, held equal by
  `test_a_body_vouches_for_tiles_as_it_does_for_their_merged_summaries`. `culling._vouches` pads tile
  indices to a power of two for a `traceable` body.

| body | witnesses |
|---|---|
| `Body` (default) | none |
| radiation's `TriangleBody` | `[x, -x]` as a bounding box; vouches when it holds no occupied voxel |
| `Solid` | a separating plane along each of 26 fixed axes (3x3x3 stencil), placed by `support` |
| `ConvexSolid` | its own `_Plane` faces, then the 26 |
| `Intersection` | every member's, concatenated |
| `Difference` | the body's |
| `Outside` | one per `ConvexSolid` region: its `signed_distance` |

**`Solid.support(direction)`** = `max over the body of n·x`, an upper bound by contract (`inf` default).
Exact for `Sphere`, `Cylinder`/`Cone` (hull of end discs), `Box` (`c·n + Σ h_j |n·E_j|`, `E = inv(axes)` —
⚠️ the columns of the inverse, not the rows of `axes`; only a skewed box tells them apart). `Union` max,
`Intersection` min, `Difference` the body's; `HalfSpace` stays `inf`.

- **A curved inequality is never a witness** (outside a tube/ball is not convex). Only planes qualify.
- **A non-convex `Outside` region gives no witness** — pinned by
  `test_an_outside_vouches_for_points_in_one_convex_region_and_not_across_two`.
- **Margin is sized from the magnitudes compared** (`_CLEARANCE_MARGIN = 1e-10` × `Σ|x_i||n_i| + |bound|`
  for a plane, × `|x|` for a region distance), so near-touching hulls fall to the exact segment test.
- Soundness sweep `test_a_certified_hull_holds_no_point_of_the_body` samples throughout each hull, not
  only corners.

## `ConvexSolid.face_distances`

`face_distances(position) -> (..., n_faces)` is each constraint's signed distance; `signed_distance` is
their max. Radiation's refracted-path solve (`radiation/refraction.py`) holds a crossing to one face
because the max has kinks and its gradient is not a face normal at edges; a crossing whose other columns
are positive is on a face's extension and is reported absent. Pinned in `tests/unit/test_solids.py`.

## `Torus` (#504)

`Torus(centre, axis, ring_radius, tube_radius, *, arc_start=None, arc=None)`: a whole ring, or the sector
from `arc_start` through `arc` (radians, right-hand rule about `axis`, `0 < arc <= pi`) cut by two `_Plane`s
through the axis. A `Solid`, `interval_count == 2` always; `support` is the whole ring's (`c·n + R|n_perp| +
r|n|`), so the 26 witnesses apply. ⚠️ The angle is `arc`, not `sweep` (smoother passes, Principle 3.7). The
CAD reader still refuses toroidal faces. Measurements, defects and mutation pass: `.claude/notes/solids-log.md`.

- ⚠️ **The quartic is never solved by formula.** `_ring_spans` (module-level, `jax.jit`ed — eagerly its
  small operations compile per input shape) reparameterizes the line by arc length from its closest
  approach to the centre, so the quartic's derivative is a depressed cubic (`_cubic_roots`); its turning
  points cut the line into stretches holding at most one crossing. **Whether a stretch is inside is read
  from the tube's excess `(rho - R)^2 + z^2 - r^2` at the turning point, formed from vector differences** —
  the `_Tube` lesson; a closed-form quartic reads grazing lines as misses.
- **Crossings: `_bracketed_roots`**, Newton from the quadratic model about the nearest turning point (the
  middle two take the better of the models about both of theirs), `rtsafe`-safeguarded — bisect wherever
  Newton would leave the bracket or not halve the step before last — and frozen once the excess is within
  `_EXCESS_ROUNDING` units of rounding of `(R + r)^2`. `_TORUS_ROOT_STEPS = 10`. ⚠️ Freeze on the excess,
  never on a step size (a near-double root's step stalls at ~sqrt(eps)); never return "the smallest
  residual seen" (on a line skimming the wall the residual is tiny far from the root).
- **Reference: `tests/unit/torus_reference.py`** — the quartic from the stored floats as `Fraction`s, real
  roots by Sturm sequence. Grazing classification 0 wrong where `numpy.roots` gets dozens wrong
  (`validation/torus_tangency.py` reproduces both).

## Not built here, deliberately
- **STEP reader** — built in `aquaflux/io/cad/` (OCP), emitting these bodies; nothing of OpenCASCADE enters
  this package (`.claude/rules/io.md`). Each primitive's constructor keywords are its full parameterization.
- **Trimmed patches** — not supported; the CAD reader describes whole solids (a pipe as its full cylinder
  carried into the vessel) and refuses what it cannot. Fallback: `aquaflux.radiation.TriangleBody`, in
  radiation beside its grid since this package imports nothing outside itself.
