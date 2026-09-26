---
paths:
  - "aquaflux/solids/**"
---

# Rules — `aquaflux/solids/` (solid bodies, their set combinations, fluid regions)

A body answers two questions about geometry — does a straight segment pass through it (`blocks`),
is a point inside it (`contains`) — and declares whether those answers are a pure array expression
(`traceable`). That contract is `Body`; everything else here is an exact, formula-based
implementation of it: analytic primitives, constructive solid geometry (CSG) over them, and
`Outside`, which describes a vessel by the fluid it holds.

**Why this is its own package.** It was written as `radiation/occluders.py`, because the radiation
model's shadows were the first thing that needed it, and nothing in it names any radiation. By the
placement test in `CLAUDE.md` (Principle 3.6) that made it generic code in a physics package — and it
bit the moment a second consumer appeared: a computer-aided design (CAD) reader in `aquaflux/io/cad/`
had to emit these bodies, and `io` importing a physics package to describe a cylinder is the wrong
direction. `tests/unit/test_layering.py` now holds `solids/` to the same rule as `solve/`: it imports
nothing of aquaflux but itself and the neutral leaves (`vectors`).

**What stays in radiation, and why.** What a body *does* to light — its transmittance, frozen versus
live, and how a mask of `blocks` answers is built and stored — is radiation's, and so is the
measured Sozzi comparison of `Outside` against a hand-derived occluder. Those are in
`.claude/rules/radiation.md`. The shadowing-specific examples in the docstrings here (a lamp sleeve,
a reactor wall) are illustrations of the package's purpose, not dependencies.

## ⚠️ FORM THE CYLINDER'S DISCRIMINANT AS `a(r^2 - h^2)`, NEVER AS `b^2 - a c`

**One home: `_Tube` in `solids/bodies.py`**, shared with `_Ball` through `_positive_definite_span`,
and reached by every body with a round side — a cylinder, a lamp's spherical tip, a vessel
composed of either. ⚠️ **`_Taper` cannot have it**: a cone's quadratic is *indefinite*, so there
is no positive-definite form to build a closest-approach distance from, and a `Cone` whose
surface a ray must graze is better described as a `Cylinder` — which is why equal end radii are
refused rather than quietly accepted.

The naive form subtracts two nearly equal numbers exactly when a ray almost grazes the surface,
which at *any* precision throws away most of the significant digits. Measured over eighteen
near-tangential cases — three source distances, offsets a few parts in `1e9` to `1e13` either
side of the radius — the naive form misclassifies **eight**, and every one in the dangerous
direction: it reports a discriminant of exactly zero for a ray that does pass inside, so **light
leaks through a body that should stop it**. The reformulation, which builds the closest-approach
distance as a vector difference, gets all eighteen right.

This is the ordinary geometry here, not a corner case: a sleeve sits at essentially the radius of
the lamp facets it surrounds.

Three more pieces of hygiene, each pinned by a mutation: clamp the hit to the segment (a body
beyond the receiver does not occlude), exclude a sliver next to the source **sized as a fraction
of the facet's own `sqrt(area)`** rather than absolutely (one length scale in the module; a fixed
epsilon gives self-shadowing when small and light leaks when large), and clip the cylinder to its
flat ends.

**A point inside a body is refused at build.** It is embedded in the solid, not shadowed by it,
and nothing computed there means anything. Both facets and receivers are checked.

**Do not represent a body twice.** A sleeve that is already an emitting surface must not also be
an occluder: every ray would leave a facet lying exactly on an occluder, and the emitter's own
convexity already makes the source-side clamp an exact visibility test for it.

## BLOCKING GEOMETRY AS CAD PRIMITIVES: three layers, and why the middle one is intervals

`solids/bodies.py` is a primitive library, a constructive-solid-geometry algebra over it, and a way
to describe a vessel by the fluid it holds. The whole of it rests on **one property**, and
naming that property is what keeps it from being a zoo of special cases.

| layer | what it promises | who implements it |
|---|---|---|
| `Body` | answers `blocks` and `contains` | anything, including host code over triangles |
| `Solid` | its inside, along a line, is a **bounded union of intervals** in the line's parameter | every body here |
| `ConvexSolid` | that union is a **single** interval | `HalfSpace`, `Sphere`, `Cylinder`, `Cone`, `Box` |

**Why the interval is the right middle layer.** Blocking is then "does any interval overlap the
segment"; `Union` pools intervals, `Intersection` overlaps them pairwise, `Difference` cuts each
of one body's by another's — so there is one implementation of each operation rather than one
intersection routine per pair of shapes. The count is a static property of the body (`Union` of
three convex bodies has three), the same for every ray, so nothing here is a search with a
data-dependent length and the whole mask stays a single array expression.

**A convex body is a list of inequalities, and that is the deduplication that matters.** Private
`_Plane`, `_Tube`, `_Ball` and `_Taper` each supply two views of one `f(x) <= 0` — a signed
distance at a point and an interval along a line. A cylinder is a tube and two planes, a box is
six planes, a cone is a taper and two planes. So `contains` and the interval **cannot disagree
about where a surface is**: they are two readings of the same inequalities. And the
grazing-robust cylinder quadratic below has exactly one home, reached by every body with a round
side.

⚠️ **`Difference` takes a CONVEX hole**, because the complement of a hole in several pieces
cannot be written down without sorting and merging its intervals first. `A - (B + C)` is
`Difference(Difference(A, B), C)`; each cut doubles the interval count, which is the honest
price. The constructor refuses a multi-interval hole and says this.

## THE FLUID AS REGIONS: `Outside`, and why it beats a hand-derived occluder

A reactor wall is awkward to write as a solid and trivial to write as the *water it holds*.
`Outside(chamber, inlet, riser)` is everything those three regions are not, and a segment is
clear exactly when they **cover it end to end** — an interval-covering test over the pooled
intervals, with no opening identified and no shadow edge derived.

**This is strictly more general than the hand-derived alternative it replaces.** The
`BranchOpenings` closure in `validation/sozzi_radiation/compare_fluence.py` works out, for that
one reactor, where each pipe's opening is and what crossing it means. `Outside` needs three
cylinders and the same three lines describe a chamber-pipe-elbow chain of any length.

**Correctness does not need convexity; cheapness does.** Any `Solid` may be a region — the test
pools every region's intervals and asks whether together they leave a gap. A *convex* region
contributes one interval, which is why a chain of chambers and pipes costs a handful of
comparisons per ray. Convexity is also where the shortcut comes from: two points inside one
convex region have a clear segment between them by definition, with nothing to test at all.

⚠️ **NEIGHBOURING REGIONS MUST OVERLAP OR TOUCH — A GAP BETWEEN THEM READS AS SOLID**, and reads
that way silently, as a shadow rather than as an error: the reactor simply goes dark up that
pipe. A pipe standing on a chamber is described by extending its cylinder *into* the chamber, not
by stopping it at the chamber's surface, where a curved junction leaves slivers of neither
region. The overlap is inside the chamber anyway, so it adds nothing to the fluid.

⚠️ **The covering test examines each interval's FAR END, not just the segment's start, and the
comparison there is STRICT.** Coverage can only first fail at the start or at the right-hand end
of some interval, and at each of those the pool must *continue past*, not merely reach. Dropping
either half reports a gap as covered: with only the start examined, a gap beyond it is never
looked at; with a non-strict comparison, an interval's own far end is trivially satisfied by
that same interval and no gap is ever found. Both are pinned by
`test_a_gap_between_two_regions_reads_as_solid`. Sorting the intervals would be the textbook
covering sweep and is the wrong choice here — a sort over the last axis of a
receivers-by-facets array materializes a permutation the size of the whole mask, where the
pairwise test fuses into the reduction and forms nothing.

⚠️ **`Outside` needs the margin at BOTH ENDS of a segment, and that is the same defect that
once shipped for facet-to-facet rays.** A facet centroid used as a receiver sits exactly on the
wall the regions are bounded by, so a rounding puts it outside and leaves an infinitesimal
uncovered sliver at `t = 1`. Over an enclosure that is not a small error but a shadow
everywhere. `min_distance` is therefore applied at the far end as well as the near one.

**`contains` means "outside every region", which is the right answer for a surface that bounds
fluid** — a point there is embedded in the wall, and the build-time guard should refuse it. It
carries a `tolerance`, a length, because a mesh never lands exactly on the surface its cells were
snapped to and a cell centre a rounding outside a region is a discretization rather than a cell
in the metal. It applies to that test only: where a *segment* is clear is decided with no slack.

**Two mutation rounds over `solids/bodies.py`, 23 mutations, 22 red.** Each broke one line and the
suite was rerun (`PYTHONDONTWRITEBYTECODE=1`, per the bytecode trap recorded above). Red: both
halves of the covering test, the far-end margin, the taper's branch selection, the box's own
axes, the difference's second piece, the intersection's overlap, the grazing discriminant, the
segment's far end, the near-origin exclusion, the cylinder's end caps, the convex signed
distance, the plane's solid side, the union's pooling and its nearest-body distance, the
difference's hole sign, the tolerance, the covering result's sense, the cone's slope, the ball's
centre, and the unconditional compile.

⚠️ **ONE MUTATION WAS INVALID, AND A MUTATION THAT GOES RED FOR THE WRONG REASON IS WORSE THAN NO
MUTATION — IT READS AS COVERAGE.** Written as `0.0 * solid_side[0]` it multiplied an infinity and
produced a NaN rather than removing the cut, so what it tested was NaN propagation; its red said
nothing about the line it was aimed at, and taken at face value it would have retired a real
question as answered. **Check that a mutation's failure comes from the mechanism you intended**,
not merely that the suite went red — the rerun with a clean removal is what found the survivor
below.

**The one genuine survivor is DISMISSED, with a measurement rather than an argument**: the
taper's solid-nappe cut changes no answer reachable through `Cone`, because non-negative end
radii put the mirror nappe beyond an end cap — 0 disagreements in 160,000 rays over four cone
shapes (a true cone, a frustum whose apex is far outside it, a tilted narrowing one, a steep
one). It is kept so the inequality means on its own what it says, since on a bare taper the two
differ on 7.2% of the same rays. What is *not* redundant is the branch selection beside it, whose
mutation is red.

## CLEARANCE: a body vouching for a whole convex hull (#554, 2026-09-25)

`Body.clearance(position) -> (..., n_features)` and `Body.vouches(summary) -> (...)` are the third
question a body answers, after `blocks` and `contains`. **The contract: a set is summarized by the
column-wise max of `clearance` over its points (two sets merge by max), and where `vouches` reads true
on that summary, no point of the set's convex hull is in the body** — so no segment between two of the
positions meets it. That is what lets a consumer (radiation's `ShaftCulling`) certify a whole tile of
receiver-source pairs from two short rows without visiting a pair. It says "clear" or "don't know",
never "blocked".

**`vouches` was split out of `clearance` in #554 phase B**, because a triangle soup's certificate is a
*box query* — its features are `[x, -x]`, whose maxima are the set's bounding box, and it vouches when
that box overlaps no occupied voxel — which "some column negative" cannot express. The default
`vouches` is exactly that old rule (`any(summary < 0)`, never with zero columns), so every body in
this package reads its columns as **witnesses** as before and nothing here changed behaviour. ⚠️ A
composition concatenates its members' columns (`Intersection`), which is only meaningful because
every `Solid` uses the default `vouches`; a `Solid` that overrode it could not be composed this way.
(`TriangleBody` is a `Body`, not a `Solid`, so it never enters a combinator.)

| body | witnesses |
|---|---|
| `Body` (default) | none — zero columns, never certifies |
| radiation's `TriangleBody` (not in this package) | `[x, -x]` read as a bounding box, with its own `vouches`: no occupied voxel of a fine occupancy grid inside it |
| `Solid` | a separating plane along each of 26 fixed axes (the 3x3x3 stencil), placed by `support` |
| `ConvexSolid` | its own `_Plane` faces (beyond a face is outside the whole body, and a half-space is convex), then the 26 |
| `Intersection` | every member's witnesses concatenated (missing any member misses the intersection) |
| `Difference` | the body's (missing the body misses what is left of it) |
| `Outside` | one per **`ConvexSolid`** region: its `signed_distance` (the region is convex and is fluid) |

**`Solid.support(direction)`** is the support function `max over the body of n . x`, an **upper
bound** by contract, `inf` by default: exact for `Sphere` (`c.n + r|n|`), `Cylinder` and `Cone` (the
convex hull of the two end discs; a disc reaches `p.n + r sqrt(|n|^2 - (n.a)^2)`), `Box` (the furthest
corner, `c.n + sum_j h_j |n . E_j|` with `E = inv(axes)` — **the columns of the inverse, not the rows
of `axes`**, which only a skewed box distinguishes); `Union` takes the max, `Intersection` the min (a
bound), `Difference` the body's. `HalfSpace` stays `inf`: its face is its only witness.

- **A curved inequality is never a witness.** The outside of a tube or a ball is not convex, so a
  hull of points all outside a cylinder's round side can still cut through it. Only planes, and the
  support-placed planes, qualify.
- **A non-convex `Outside` region gives no witness.** A union of regions is not one: every point of a
  chamber-plus-pipe cloud is in the water and the hull still cuts the chamber roof —
  `test_an_outside_vouches_for_points_in_one_convex_region_and_not_across_two` pins exactly that
  wrong answer.
- **The margin is Shewchuk-style, sized from the magnitudes compared** (`_CLEARANCE_MARGIN = 1e-10`
  times `sum |x_i||n_i| + |bound|` for a plane, times `|x|` for a region distance), so a hull that
  touches a body or misses it by a rounding is left to the exact segment test. ⚠️ **Its first test was
  wrong, not the code**: a plane *through the origin* with normal `(0,0,1)` computes `n . x` exactly,
  so a 1e-15 gap there is real and certifying it is right. The test now puts the plane at `z = 1`,
  where a rounding exists, and a zero margin goes red there.
- **Mutation pass (10, all red):** union min for max; box read along `axes` for `inv(axes)`; the
  cylinder's axial reach dropped; `Outside` witnessing the union's distance; zero margin; faces
  dropped from `ConvexSolid`; `Intersection` using its first member only; `Difference` using the
  hole's; the cone's tip disc at the base end; the sphere's radius not scaled by `|n|`. The soundness
  sweep (`test_a_certified_hull_holds_no_point_of_the_body`, every fixture body) samples points
  **throughout** each certified hull, not only its corners, since a hull can pass through a body all
  its corners miss.

## What is NOT built here, and why each was left out rather than forgotten

- **Torus.** An elbow is a torus and the bent-duct case wants one, but a torus *tube is not
  convex*, so it does not fit the convex-region machinery, and its ray intersection is a quartic
  whose branchless solution is accurate enough only with care that is its own piece of work.
  `Solid.intervals` already returns `interval_count` intervals rather than one, so a torus
  reporting two slots in with no re-cut of the algebra. Approximating an elbow by a fan of convex
  wedges is available and is **not** the answer: it reintroduces exactly the faceting error
  primitives exist to remove.
- **A STEP reader — BUILT, but in `aquaflux/io/cad/`, not here (#505).** It reads through
  OpenCASCADE's Python binding OCP and emits these bodies; nothing of OpenCASCADE crosses into this
  package, which is what keeps it import-free and traceable. The primitives were already the
  neutral description such a loader emits: each is an `equinox.Module` whose constructor keywords
  are its full parameterization. See `.claude/rules/io.md`.
- **Trimmed patches.** A B-rep face is a bounded piece of an analytic surface cut by edge loops
  in parameter space, and nothing here tests a point against a trim. The CAD reader avoids needing
  it by describing whole solids — a pipe cut to fit its vessel becomes its full cylinder carried into
  the vessel, checked exact together with it — and refuses what that cannot describe. The fallback
  for such a solid is a triangle-backed body: `aquaflux.radiation.TriangleBody` (#510), which lives in
  radiation beside the grid it walks, since this package may import nothing outside itself.
