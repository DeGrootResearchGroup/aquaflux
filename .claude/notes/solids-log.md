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

## `Torus` (#504, 2026-10-16) — how its intervals are found, and what was measured

`Torus(centre, axis, ring_radius, tube_radius, *, arc_start=None, arc=None)` -- a whole ring, or the
sector swept from `arc_start` through `arc` (radians, right-hand rule about `axis`, `0 < arc <= pi`)
cut by two `_Plane`s through the axis. A `Solid`, not a `ConvexSolid`: `interval_count == 2` always
(a sector's wedge is convex for `arc <= pi`, so it cuts each tube interval to one). Refused: `tube >=
ring` (plain numbers only), `arc` without `arc_start` or vice versa, `arc > pi` (write a `Union` of two
sectors), `arc_start` along the axis. ⚠️ **The angle is `arc`, not the issue's `sweep`**: `sweep` names
smoother passes elsewhere (Principle 3.7). `support` is the whole ring's (`c.n + R|n_perp| + r|n|`),
so the 26 separating witnesses work; no face witnesses, `face_distances` absent (not convex). The CAD
reader still refuses toroidal faces (`io/cad/recognize.py`) -- recognizing them is a separate change.

**How the intervals are found** (`_ring_spans`, module-level and `jax.jit`ed: eagerly its few hundred
small operations compile per input shape, ~5 s each, against ~1.4 s once compiled as one program):
the line is reparameterized by arc length from its closest approach to the centre, so the quartic's
derivative is a depressed cubic; `_cubic_roots` (Cardano / trigonometric, two Newton polishes kept only
where they lower the residual, sorted, one real root repeated thrice) gives the turning points, which
cut the line into four stretches each holding at most one crossing. **Whether a stretch is inside is
read from the tube's excess `(rho - R)^2 + z^2 - r^2` at the turning point, formed from vector
differences** -- the `_Tube` lesson again, and the whole near-tangency story. Each crossing is placed by
`_bracketed_roots`: Newton from the quadratic model about the nearest turning point (for the two
middle crossings, the better of the models about both of its turning points), `rtsafe`-safeguarded --
bisect wherever Newton would leave the bracket **or not halve the step before last** -- and frozen once
the excess is within `_EXCESS_ROUNDING = 16` units of rounding of `(R + r)^2`. `_TORUS_ROOT_STEPS = 10`.

**Measured** (exact reference `tests/unit/torus_reference.py`: the quartic from the stored floats as
`Fraction`s, real roots by a Sturm sequence -- no new dependency; jax 0.11.2, CPU, x64, Linux x86_64,
4 cores, 2026-10-16):
- **Grazing classification: 0 wrong** of 168 fixed lines (7 tangencies, offsets 1e-9..1e-12 r either
  side, sources 1-100 ring diameters away) and 0 of 3,000 random ones (offsets 1e-12..1e-2 r).
  `numpy.roots` on the same quartics: **47 wrong of the 168, 24 of them leaks**, and **295 of the
  3,000, 147 leaks** (`validation/torus_tangency.py`, which these are from; a scratch probe of a
  slightly different fixed set had read 46 / 19 and is not the record).
- **Crossing positions**: ordinary lines to <= 5e-14 (random, inside-to-inside, inlet-to-bend);
  grazing lines to ~2-4e-9 as a length -- the conditioning of a near-double root, not convergence.
- **Against an exact elbow** (the walked reference of `validation/elbow_torus_need.py`, which lands
  with the measurement on its own branch, `elbow-torus-measurement`; inlet + `Torus` sector + outlet
  as `Outside`): **0 of 27.6M pairs** differ, both
  bends. Mask 47-50 s for 13.8M rays against 7-10 s for one cylinder region (the 32-64-piece chain
  cost 39-80 s); not optimized.

⚠️ **Four defects, each found by ONE disagreeing line among millions, each now pinned.** (1) Newton
rejected a step landing on the bracket's own end, so a converged root was bisected away -- accept the
closed bracket. (2) A line running a long way inside the tube gives the quartic a wide flat bottom;
from the turning point's model plain Newton crawled (0.05 of the line short after 8 steps). (3) A line
**skimming under the bend's inner wall** (a fraction of a millimetre for a long stretch) made plain
Newton bounce across the bracket -- the `rtsafe` halving test is what fixes (2) and (3); a start from
the bracket's far end also fixed (2) but was **measured dominated** by the halving test and deleted.
(4) The halving test then bisected away converged roots (a noise-sized step "fails to halve"; a
one-sided bracket is wide) -- hence the freeze at the excess's rounding floor, **not** at a step size:
a near-double root's step stalls at ~sqrt(eps), above any step tolerance. ⚠️ **"Return the point of
smallest residual" was tried and is wrong**: on a skimming line the residual is tiny far from the root.

**Tests** (`tests/unit/test_solids.py`): the two fixtures (`torus`, `torus sector`) join the shared
oracle (intervals vs sampling `contains`), support and certified-hull sweeps; plus exact-quartic ends
on three line families, the 168 grazing lines, random grazing lines' ends to 1e-7, the two skimming
lines from the elbow, closed-form chords (axis misses; across the middle `[R - r, R + r]` both sides;
tangent to the centre circle `sqrt((R+r)^2 - R^2)`), the sector's cuts, an elbow `Outside` against
sampling, refusals, compile-and-broadcast, a zero-length segment. **Mutation pass, 13, 11 red.**
**Dismissed, equivalent**: deciding from the quartic's value at the turning point (it is assembled from
the same vector-difference `rho`, so it is not the naive quartic), and the turning points' order (sorted
afterwards).
