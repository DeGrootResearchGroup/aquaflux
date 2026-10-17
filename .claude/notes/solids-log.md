# Investigation log — `aquaflux/solids/`

> Reference-only, never auto-loads. Dated measurements, mutation passes and investigation detail
> behind `.claude/rules/solids.md`; the rule holds the current state.

## From `solids.md` — the cylinder discriminant

The naive form subtracts two nearly equal numbers exactly when a ray almost grazes the surface,
which at *any* precision throws away most of the significant digits. Measured over eighteen
near-tangential cases — three source distances, offsets a few parts in `1e9` to `1e13` either
side of the radius — the naive form misclassifies **eight**, and every one in the dangerous
direction: it reports a discriminant of exactly zero for a ray that does pass inside, so **light
leaks through a body that should stop it**. The reformulation, which builds the closest-approach
distance as a vector difference, gets all eighteen right.

## From `solids.md` — mutation rounds over `solids/bodies.py` (CSG / `Outside`)

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

## From `solids.md` — clearance (#554, 2026-09-25)

- ⚠️ **The margin's first test was wrong, not the code**: a plane *through the origin* with normal
  `(0,0,1)` computes `n . x` exactly, so a 1e-15 gap there is real and certifying it is right. The test
  now puts the plane at `z = 1`, where a rounding exists, and a zero margin goes red there.
- **Mutation pass (10, all red):** union min for max; box read along `axes` for `inv(axes)`; the
  cylinder's axial reach dropped; `Outside` witnessing the union's distance; zero margin; faces
  dropped from `ConvexSolid`; `Intersection` using its first member only; `Difference` using the
  hole's; the cone's tip disc at the base end; the sphere's radius not scaled by `|n|`. The soundness
  sweep (`test_a_certified_hull_holds_no_point_of_the_body`, every fixture body) samples points
  **throughout** each certified hull, not only its corners, since a hull can pass through a body all
  its corners miss.
