---
paths:
  - "aquaflux/radiation/**"
---

# Rules — `aquaflux/radiation/` (ultraviolet fluence rate)

The package computes the **fluence rate** `G` — radiant power arriving at a point from every
direction — at cell centres, by a deterministic backward gather over emitting surface facets,
with Beer–Lambert attenuation, occlusion, and diffuse surface interreflection closed by a
linear solve. It is differentiable end to end, which is the part with no precedent in the
ultraviolet-reactor literature; the gather itself is that field's mainstream method (the
segment-summation family) generalized from a cylinder to arbitrary triangles.

## Built so far

| piece | state |
|---|---|
| `solid_angle.py` — the two geometric kernels, plus the signed form | **BUILT** |
| `stl.py` — ASCII and binary STL reading | **BUILT** |
| `surfaces.py` — the `Surfaces` value object | **BUILT** |
| `checks.py` — build-time geometry checks, including the cell-in-the-metal test | **BUILT** |
| `subdivide.py` — the width-over-distance refinement | **BUILT** |
| `profiles.py` — `Isotropic`, `Lambertian`, `CosinePower` | **BUILT** |
| `gather.py` — `direct_fluence_rate` and `direct_irradiance` | **BUILT** |
| `absorption.py` — `UniformAbsorption`, `VoxelAbsorption` | **BUILT** |
| `occluders.py` — `Cylinder`, `HalfSpace` | **BUILT** |
| `visibility.py` — the frozen shadow mask | **BUILT** |
| `triangles.py` — watertight ray-triangle intersection | **BUILT** |
| `transfer.py` — the frozen facet-to-facet geometry | **BUILT** |
| `quadrature.py` — symmetric triangle rules for the receiving facet | **BUILT** |
| `model.py` — the assembled model and the three public entry points | **BUILT** |
| `units.py` — lamp watts to exitance, ultraviolet transmittance to absorbance | **BUILT** |

| Beer–Lambert optical depth, voxel-grid traversal | Not yet built |


## ⚠️ THERE ARE TWO SOLID-ANGLE KERNELS AND THEY ARE NOT INTERCHANGEABLE

This is the defect that recurred through five drafts of the design, and it recurs because the
wrong answer is *clean* rather than noisy.

- `solid_angle` — the plain solid angle `Ω`. For **fluence rate**, which carries no receiver
  cosine because the receiver is a point in a volume, not a surface.
- `projected_solid_angle` — `∫cos θ dω`. For **irradiance** and for every surface-to-surface
  transfer factor, because an oblique surface intercepts less.

**There is a third NAME and it is not a third quantity.** `signed_solid_angle` is the same
integral as `solid_angle` with the sign of the vertex winding kept instead of discarded, and
`solid_angle` is now literally its magnitude. On a single triangle that sign is meaningless — it
records the order an exporter happened to write three vertices in — which is exactly why the
public kernel drops it. It means something only summed over a **consistently wound, closed**
surface, where it gives the winding number; see the enclosure section below. Do not reach for it
anywhere else.

No scalar converts the two quantities into one another: the obliquity varies across the emitter. Used in the
wrong place, `Ω/π` makes every row of a transfer matrix sum to **exactly 2.000000000000** at
every refinement, so the spectral radius of a reflection system becomes `2ρ` — the Neumann
series diverges above `ρ = 0.5` and `I − ρF` is indefinite at 0.9. That reads as a different
problem being solved correctly, which is why review after review passed it. Measured against
an independent view-factor implementation on a general-position pair, the error is 42% / 25% /
11% at separations of a quarter, a half and one emitter width, and **it does not shrink under
refinement** — non-convergence, not size, is what identifies a wrong kernel.

## Four implementation details that are load-bearing, each measured

1. **Every angle is `arctan2`, never `arccos` and never a sum of interior angles.** For the
   plain solid angle, summing interior angles is the small difference of large quantities for
   a small triangle — which is what a refined facet at a distance is. For the contour form,
   `arccos(u·u')` is ill-conditioned as its argument approaches one, which is what consecutive
   edges of a refined mesh look like: the enclosure sum comes out wrong by ~1.3e-8, **flat in
   refinement**, against 0–5.7e-16 for the tangent form. Flatness is the diagnostic — an
   accumulation would grow.
2. **The contour form is signed and the polygon must be clipped to the receiver's front
   half-space first.** A triangle straddling the plane returns a partially cancelled value and
   one behind it a negative one. The clip emits a fixed six-vertex loop in which a rejected
   candidate repeats its predecessor, so the shape is static and the repeats are zero-length
   edges contributing no angle.
3. **Take the magnitude.** The sign of either contour sum records only the order vertices
   happen to be stored in, which for an imported surface file is arbitrary. Orientation is the
   normal's job. The test fixture winds its faces inconsistently on purpose.
4. **Guard the square root *inside* the root, not after it.** `v / where(zero, 1, sqrt(sq))`
   keeps the forward value finite and still yields a **NaN gradient**, because reverse mode
   visits `d/dx sqrt(x)` at zero and the unselected branch's NaN propagates through the
   selection. Write `sqrt(where(zero, 1, sq))`. Zero-length edges are the *normal* case here,
   not a degenerate one — the clip manufactures three of them for every fully visible triangle
   — so this is on the main path, and the whole project is gradients.

## ⚠️ THE VERTEX-DEGENERACY GUARD IS FOR THE GRADIENT; THE VALUE NEEDS NO HELP

`solid_angle` ends with `where(any(degenerate), 0.0, omega)` for a receiver sitting exactly on a
vertex. **A mutation deleting that line passed the whole suite**, and the reason is instructive:
with one direction zeroed the numerator is zero and the denominator is `1 + b·c`, which for unit
vectors cannot be negative, so `arctan2(0, non-negative)` is zero and the forward value is right
either way. The test that existed read only the value.

The derivative is not right either way. Unguarded, `d/dp` of that zero comes back **(0, 0, −2)** —
finite, plausible, and fictitious. That is strictly worse than a NaN, which at least announces
itself, and the gather is differentiated with respect to vertex positions whenever a study moves a
lamp, so one degenerate pair anywhere in a scene contributes a spurious term to the whole
derivative. `test_the_vertex_guard_is_there_for_the_GRADIENT_the_value_needs_no_help` pins both
kernels. This is the same lesson as the `sqrt`-guard detail above, arriving from the other side:
there, guarding after the root left a NaN gradient; here, not guarding at all leaves a *clean* one.

## A convention callers must honour: `F_ii = 0`

A facet contains its own centroid, so the spherical image of its boundary is a great circle
and both kernels return a full hemisphere or sphere by oriented area. That is documented
behaviour, not a bug — but an assembled transfer matrix must zero its own diagonal, or every
row sum is exactly one too large. The masking belongs to whatever assembles the matrix; it is
deliberately **not** in the kernel, which knows nothing about matrices.

A receiver coplanar with a facet but *outside* it correctly contributes nothing, so coplanar
neighbours on the same wall need no special handling.

## Testing

`tests/unit/test_solid_angle.py`, with references in `tests/unit/radiation_references.py`.
Every reference comes from outside the kernels: two closed forms, plus recorded values from
**pyviewfactor** (MIT), an independent contour-integral implementation checked against the
closed form to 3.5e-16 before being trusted.

⚠️ **The oracle's values are recorded, not imported.** A test behind an optional import is
skipped silently and a skip is indistinguishable from a pass — the trap
`tests/unit/test_optional_dependency_skips.py` exists to police. Regenerate with
`pip install pyviewfactor` when the geometry changes; do not turn it into a dependency.

⚠️ **On-axis tests cannot see a missing receiver cosine**, because on axis the cosine is one.
Any new case must include an off-axis or oblique configuration — the perpendicular-squares
test sweeps obliquity across its whole range for exactly this reason.

All five mutations of the kernels were verified to fail the suite: `arctan2`→`arccos`, drop
the clip, drop either magnitude, and return the plain solid angle from the projected function.


## ⚠️ A ZERO-AREA FACET IS LEGAL — IT IS A POINT SOURCE

The guard a reader reaches for is the bug. `Surfaces` carries **emission** (W/m², areal) and
**radiant power** (W, point) as separate fields, and a point source is a degenerate triangle
with zero area carrying power. Rejecting degenerate triangles deletes every point source in
the set; dividing by the area to get a normal fills the geometry with NaN, which then reaches
every gradient that touches the surface rather than only the facet that caused it. The normal
is zero on such a facet, and `from_triangles` guards the division rather than the input.

**Power is also extensive, which is why `refine_for_receivers` refuses to carry it.** Emission
and reflectance are intensive and are inherited unchanged by a facet's children; power would
have to be divided among them. Refine the areal facets, then add the point sources.

## ⚠️ DETECT THE STL FORMAT BY FILE LENGTH, NOT BY THE LEADING KEYWORD

A binary STL's 80-byte header is arbitrary text and exporters have shipped headers beginning
with the word `solid`. A reader that sniffs the keyword then attempts an ASCII parse of binary
data — failing on unparsable numbers if you are lucky, and returning garbage from a stray
`vertex` byte sequence if you are not. A binary file's length is exactly
`84 + 50 × count` with the count read from its own header, which no ASCII file matches except
by coincidence. That is the test, and it is pinned by a fixture whose header starts with
`solid`.

Body names are load-bearing: they are how optical properties are assigned, so `per_facet`
raises on a name the file does not contain and on a body the mapping omits. A silently dropped
name leaves a lamp emitting nothing, which looks like a physics result.

## Winding is checked, and it is the most common real defect

A triangulated surface carries no orientation of its own — outwardness is inferred from vertex
order. Exporters, boolean operations and hand edits all produce files where some triangles
disagree with their neighbours, and those facets' normals point into the solid, where the
source-side visibility clamp discards them. **The surface emits less over that patch and
nothing anywhere reports an error.** `check_winding` raises; `winding_report` returns the
counts without raising.

The test is edge parity: every shared edge must be traversed in opposite directions by its two
triangles. Vertices are merged within a tolerance **relative to the model's extent** first,
because an STL repeats an edge's endpoints in each triangle as separately-rounded coordinates —
match them exactly and no edge is ever shared, so no conflict can ever be found and every edge
reads as a boundary.

Boundary edges (one triangle) are fine — open surfaces are legal. Non-manifold edges (three or
more) are counted but not fatal: a gather never has to decide which side of a surface it is on,
which is also why watertightness is **not** required here even though ray-tracing codes that
track a medium do require it.

## The refinement criterion, and why the distance is measured from the corner

The closed-form solid angle is exact at any apparent size, so refinement is not about the
geometry term. It is about what the model assumes constant across a facet: one emission, one
reflectance, one outgoing radiance. The criterion is lighting simulation's own — refine until
longest edge over distance to the nearest receiver falls below 0.25 (0.15 and 0.05 for accurate
work), and **report the realized distribution**, so a build states the accuracy it reached
rather than the one it was asked for.

Two details:

- **Distance is the minimum over the three vertices and the centroid, not the centroid alone.**
  A centroid distance is never smaller, so it can only refine less, and it under-refines exactly
  where a receiver sits closest to a facet — the worst case. The separating fixture is narrow
  (most geometries refine under both measures), so the test constructs one that straddles the
  threshold and asserts it still straddles it.
- **The four-way split's middle child must keep its siblings' winding.** Reversed, a quarter of
  every refined facet stops emitting, and `check_winding` then blames the input file.

Splitting at edge midpoints into four *similar* triangles preserves shape quality where
repeated bisection of one edge would not, and conserves area exactly.
## A profile is a normalized INTENSITY distribution, and it has two views

`integral over 4 pi of f = 1`, so a source of power `P` has intensity `P f(omega)`. Power is
carried separately, which is the illumination-design convention and is what lets one object
describe both a point source (`G = P f / r^2`) and a surface.

**Each profile supplies `intensity_fraction` and `radiance_per_exitance`, and the pair must
satisfy `radiance_per_exitance(c) * c == intensity_fraction(c)`.** The second is not derived
from the first at run time because the derivation divides by `c`, and Lambertian — the default,
and the distribution every reflected ray leaves by — is `0/0` at grazing. Written out, it is the
constant `1/pi` with nothing to cancel.

- `CosinePower(n)` is `(n+1) max(cos,0)^n / (2 pi)`. **The constant is over `2 pi`, not `pi`**,
  because the normalization is hemispherical; `n = 1` must reduce exactly to Lambertian's
  `cos/pi`, and that reduction is the test that catches the slip. `n < 1` is refused: the
  radiance would be unbounded at grazing, which no surface emitter is.
- **`Isotropic` is a point-source profile and raises from `radiance_per_exitance`.** A zero-area
  facet has no normal, so no directional distribution has anything to measure against.
  `check_profiles` refuses isotropic-on-areal and directional-on-point; the latter is silent
  otherwise, since a zero normal reads as a right angle and the source contributes nothing.

## The clamp is a GATE, not a factor — and the difference is a factor of two or a zero

A facet cannot illuminate what is behind it. Measured on a Lambertian tube at `d/R = 2`, where
the closed form is `G = (4B/pi) arcsin(R/d) = 2`:

| formulation | result |
|---|---|
| clamp as a gate (what is built) | **1.99978** |
| absolute value of the cosine | 3.99941 — exactly double |
| no clamp at all | 3.99941 — same, because the gate is what was removed |
| signed cosine as a *multiplier* | **1.5e-4** — the two sides cancel |

The last row is why the sign is a gate. A near-zero field looks like an empty scene rather than
like a physics error, so it is the failure mode least likely to be noticed.

A convex emitting body needs no occluder: the clamp *is* its visibility condition, exactly. The
cylinder case tests that and **does not test occlusion**.

## ⚠️ A SOURCE'S KIND IS A LABEL, NOT ITS AREA

`Surfaces.point_source_index` records which facets are point sources. It is static metadata and
everything that needs the distinction reads `Surfaces.is_point_source`; nothing infers it from
`area > 0`. The two agree when a set is built, which is exactly why conflating them is tempting.

**The area is a quantity a gradient flows through; the kind decides a code path.** Tie them
together and the knot shows up the first time someone differentiates with respect to vertex
positions — moving a lamp, which is the question a design study asks. The area becomes a traced
quantity, the host-side partition can no longer read it, and the gradient is not wrong but
*unbuildable*. Separated, it works: `dG/dz` for an areal facet and for a point source both match
a central difference to 1e-10, and the full per-vertex jacobian is finite and non-zero.

Note that `area` is never used numerically in the gather at all — the solid angle already
carries the area-over-`r^2` geometry, and point sources carry power. It was purely a
discriminator, which is what made the conflation invisible.

**Move a set with `Surfaces.with_geometry`, never by substituting `vertices` alone.** Centroid,
normal and area all derive from the vertices; `tree_at` on `vertices` leaves all three
describing the old shape, silently, because nothing downstream can tell a stale normal from a
fresh one. `with_geometry` recomputes them and carries the labels and optics across.

## ⚠️ INSIDE A TRACE, `jnp` STAGES EVERYTHING — EVEN ON CONCRETE INPUTS

A build-time validation written with `jnp` works perfectly until the object is first rebuilt
inside a traced function, and then fails on an input that *is* concrete:

```
jax.jit(lambda x: int(jnp.max(concrete_int_array)))   # ConcretizationTypeError
```

`jnp.max` on a concrete array returns a **tracer** when a trace is active, because every `jnp`
operation encountered during tracing is staged out regardless of its inputs. `np.max` on the
same array does not. So a range check on an index array — `solid_id`, `profile_index` — must be
written in numpy, and skipped outright when the array itself is traced. This cost an afternoon
to find because the isinstance check said "not a tracer" while the very next line disagreed.

## ⚠️ GEOMETRY IS CLOSED OVER, VALUES ARE PASSED

The gather partitions facets by angular distribution and by areal-versus-point **on the host**,
so each group's profile is a concrete object whose methods inline and the traced program holds
no branch on facet kind. That partition decides the program's *shape*, so `area` and
`profile_index` cannot themselves be traced. `jit(lambda s, p: direct_fluence_rate(s, p))` over a whole
`Surfaces` raises with an explanation; close over the set and substitute values through
`with_optics` instead. `RadiationModel` formalizes that boundary: it holds the frozen
geometry and every entry point takes the surface set again, reading only its optics.

Only `profile_index` is structural in this sense. **Vertices are not** — see the label rule
above — so a source's position is free to move under a gradient.

## What the analytic cases actually pin, and what they cannot

- **Sweep the distance.** One radius cannot separate an `r`-versus-`r^2` error from a
  `pi`-versus-`4 pi` one.
- **Sweep the angle.** At normal incidence a missing receiver cosine passes. The geometry for
  `E = G cos` is a *fixed radius with a tilted normal*; on a plane at fixed perpendicular
  distance the published result is `cos^3`, because the slant range grows too.
- **`E = G cos` is a single-source identity.** Two opposed sources make fluence rates add while
  irradiances oppose. There is a test asserting it fails, so nobody re-derives it as general.
- **The disc is the strongest single case**: `G = 2B[1 - h/sqrt(a^2+h^2)]` and
  `E = B a^2/(a^2+h^2)` separate the two kernels on one geometry, and the infinite limits give
  `G/E -> 2` as a measurable statement rather than a tolerance.
- **A summed line source is a midpoint rule, so assert the RATE.** It is second order — but only
  once the segment spacing is below the receiver's distance from the line. At a spacing of 0.08
  against a distance of 0.05 a doubling improves the error 54-fold, which is not a rate. This is
  why a lamp needs hundreds of segments to be accurate at its own sleeve and far fewer across
  the reactor.
- **Relative error is second order in facet width over distance; absolute error is fourth.**
  Same statement, and an easy one to quote by mistake — the first version of that test asserted
  the second-order ratio against absolute errors and failed.

## Optical depth: exact, not marched — and three details make it so

`UniformAbsorption` is `tau = a r`, which is the field's own standard treatment rather than a
simplification. `VoxelAbsorption` carries a graded coefficient and walks the grid.

**Walking the cells is not ray marching.** Marching takes fixed steps and holds the coefficient
constant across each, which systematically overestimates the surviving fraction by Jensen's
inequality and shrinks only as the step does. Walking the real boundaries finds every crossing,
leaving only the field's own representation error — and it deletes a convergence parameter,
since there is no step size to choose.

Three details, each of which cost a debugging round to find:

1. **⚠️ CUT ON THE SAMPLE PLANES, NOT THE CELL FACES.** They are half a cell apart. The
   interpolated field is a separate cubic between neighbouring samples, so a piece that
   straddles a sample plane straddles a kink and no quadrature rule saves it. Measured on a
   linear field with a known closed form: **0.43% wrong** cutting on faces, **exact** cutting on
   sample planes. That size of error is the worst kind — large enough to matter, small enough to
   read as discretization.
2. **⚠️ GENERATE CUTS ONLY AT PLANES THAT CARRY SAMPLES, AT BOTH ENDS.** Beyond the outermost
   sample the field is clamped and has no further kinks. Cutting on the infinite lattice spends
   the fixed step budget outside the grid, and a segment reaching well past the grid exhausts it
   and **silently loses its tail** — which reads as weaker absorption, not as a bug. Both ends
   matter: a segment starting far outside meets its first real plane a long way in.
3. **Simpson's rule per piece, and it is exact.** A trilinear field along a straight line is a
   cubic in the path parameter, and Simpson integrates cubics exactly. So the optical depth is
   the exact integral of the interpolated field, not a quadrature of it — verified to 1e-12
   against fine quadrature of the same interpolant, for segments inside, straddling, outside,
   axis-aligned, reversed and degenerate.

**Interpolate, never nearest-cell.** A nearest-cell lookup is piecewise constant in position, so
its derivative is zero almost everywhere and undefined on the faces — a staircase in a path the
module promises gradients through. Outside the samples the value is **clamped**, so a segment
straying past the edge attenuates like the water at the edge rather than like vacuum.

**No clamp on the optical depth.** In double precision `exp(-tau)` reaches zero near `tau = 745`,
where zero is correct and is what is returned. A clamp would buy nothing and would flatten
`dG/da` across a whole region.

**The absorbance grid need not match the flow mesh, and usually should not.** A segment's cost is
the fixed `nx + ny + nz + 1` steps, so a coarse absorbance grid over a fine flow mesh is cheaper
and no less accurate — absorbance varies far more smoothly than velocity.

⚠️ **Attenuation is taken along the facet CENTROID's path.** A facet wide enough for its far
corner to sit at a different optical depth is attenuated as though it were not. Same remedy as
for the emission assumption: the refinement criterion. On the Beer–Lambert slab a uniform disc is
**2% wrong** near the axis and no extra radius fixes it; refining against the probes brings it
under 2e-3.

## The slab case pins which exponential integral is which

A Lambertian wall through an absorbing medium gives `G = 2 M E_2(kappa x)` and
`E = 2 M E_3(kappa x)` — **not** `exp(-kappa x)`, which is the collimated result. Against
collimated, the *fluence-rate* ratios are 0.80 at `kappa x = 0.1` and 0.28 at 2; the *irradiance*
ratios at the same depths are 0.920 and 0.445. Quoting one set for the other is a factor of one
and a half at depth and both look plausible, so the test pins each to its quantity.

Sweep to `kappa x >= 2`. At small optical depth `exp(-t) ~ 1 - t` and the exponential integrals
sit close to it, so a shallow sweep cannot separate them.

## Occlusion splits into a frozen half and a live half

`blocked` — does this body lie across this segment — is a hard yes or no fixed by geometry,
computed once at build and stored. `transmittance` — how much it lets through — is a number
supplied at every call and differentiated with respect to.

```
surviving = product over bodies of [ 1 - blocked * (1 - transmittance) ]
```

**Freezing the mask costs nothing, because the frozen thing is a staircase.** Move a body by a
hair and nothing changes until a shadow edge sweeps past a receiver, then the answer jumps. So
`dG/d(occluder geometry)` is **exactly zero, by construction**, and the test says so as a
contract rather than discovering it. `dG/d(transmittance)` is exact and matches a finite
difference.

⚠️ **The mask is indexed by receiver**, so one built for one set of points and used with another
puts every shadow in the wrong place and raises nothing of its own. `Visibility` carries its
receivers and the gather checks them.

⚠️ **A mask supplied without transmittances defaults to OPAQUE.** Defaulting the other way makes
a forgotten argument look like a working occlusion model that happens to do nothing.

**Memory:** the mask is `(n_occluders, n_receivers, n_facets)` — a hundred million entries per
body at production size, stored as bytes. Packing to bits is the obvious eightfold saving if it
ever matters.

## ⚠️ FORM THE CYLINDER'S DISCRIMINANT AS `a(r^2 - h^2)`, NEVER AS `b^2 - a c`

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

## The emitting surface occludes too, and that half is opaque

`Visibility` keeps two kinds apart. **Analytic primitives** each carry their own transmittance,
so each needs its own layer. **The surface's own triangles** are the reactor's walls — opaque —
so they collapse into one layer with nothing to carry. That second kind is what lets a bent duct
shadow itself, which no primitive can express because the geometry doing the blocking *is* the
emitting surface. `self_occlusion=True` is the default: a surface that does not shadow itself is
the defect the module exists to fix, and a mask silently missing it looks exactly like one that
includes it.

**Exclusion is by index, never by tolerance.** Every ray leaves its facet's centroid, so the
facet is always hit at zero distance. Excluding its whole *solid* would be wrong — a bent duct is
exactly the case this is for, and there the blocking wall belongs to the same body as the emitter.
Edge-adjacent neighbours are handled by the same near-origin exclusion the primitives use.

## ⚠️ A RAY BETWEEN TWO FACETS NEEDS **TWO** EXCLUSIONS, AND THE MISSING ONE SHIPPED

`segment_is_cut` excluded only the **source** facet. The far end of a segment has no margin —
`offset_scale` guards the origin, nothing guards the target — so a ray aimed at a *facet centroid*
ends exactly in that facet's plane and the hit at `distance == 1` counted. `build_transfer`'s
receivers **are** the facet centroids, so at the shipped default `self_occlusion=True` every
mutually visible pair read as blocked: measured on a closed box, **120 of 132 off-diagonal pairs**,
and on two bare plates facing each other across empty space, all of them. A closed enclosure came
back with `B = M` — ten times too dark at `rho = 0.9`, and shaped like a field rather than an
error. `exclude` now takes `(n_rays,)` or `(n_rays, k)`, and `build_visibility` takes
`receiver_facet` for the case where each receiver sits on a facet.

**Three separate reasons nothing caught it, all worth keeping:**

- **Every transfer and radiosity fixture passed `self_occlusion=False`**, so the default was never
  executed by any test. The tests that *are* about self-occlusion all use receivers out in the
  volume, where a ray ends on nothing and the bug cannot arise. The one path with no coverage was
  the one every user gets.
- **⚠️ `row_sum_error` is blind to this by construction and cannot be made to see it.** The mask is
  applied live in `TransferMatrix.assemble`; `geometric` is the raw geometry. So the gate reads 1e-15 while
  the matrix it reports on is being zeroed downstream. The subsystem's strongest invariant does not
  cover its visibility at all — do not read a green row sum as evidence about the mask.
- The defect is **invisible in the sign of the answer**: less light everywhere is what an absorbing
  medium, a dirty lamp or a low reflectance also look like.

`test_a_closed_box_reaches_its_closed_form_AT_THE_DEFAULT_SETTINGS` is the regression, and it is
deliberately the one test in that file that does *not* switch the mask off.

**The mask must change no number on a convex enclosure, and that is asserted bit-identically.**
Re-measured after the fix on boxes of 12, 48 and 192 facets: row-sum error, reciprocity residual
and the solved `B` agree to the last bit with `self_occlusion` on and off. That is what licenses
every other measurement in the subsystem, all of which were taken with it off.

⚠️ **A mutation pass that restores files with `mv` can report a false RED.** `mv` preserves the
source's mtime and `.pyc` validation is mtime-and-size with **one-second** granularity, so a fast
mutate-test-restore cycle can leave the *mutated* bytecode in play for the next run. That produced
ten spurious failures here and cost an hour chasing a bug that was not in the code. Run mutation
passes with `PYTHONDONTWRITEBYTECODE=1`.

**The consistency check that validates both halves at once:** on a convex emitter the source-side
cosine clamp *is* the exact visibility test, so tracing the body's own triangles must change
nothing. Measured on a 4608-facet cylinder, the two answers are **bit-identical**, and both match
`G = (4B/pi) arcsin(R/d)`. Acne on facets adjacent to the source would show here.

## ⚠️ THE MEMORY OF ONE PASS IS THE WHOLE PERFORMANCE STORY

The ray-by-triangle intermediate is the cost of the intersection test, and throughput does not
degrade gracefully — it falls off a cliff. Measured on 2048 triangles, f64, 11 cores, 19 GB:

| intermediate | Mtest/s |
|---|---|
| 0.5 - 134 MB | **42 - 57** |
| 537 MB | **2.4** |

So `work_limit` (entries per pass) is the only knob that matters, and **both axes must be cut to
honour it**. Blocking only the triangles is not enough: the ray count is itself receivers times
facets, so it reaches the millions on its own and would blow the limit at a block size of one.
Getting this wrong cost 13.4 Mtest/s against 31.0 on the same build — and the naive fix, a
*larger* triangle block, made it **ten times worse**, which is the opposite of the usual
dispatch-bound instinct.

## Watertight intersection, and one piece of the published algorithm deliberately dropped

Woop, Benthin & Wald (*JCGT* 2(1), 2013) rather than Möller-Trumbore. Measured on a closed hull
with rays from inside aimed at every vertex and edge midpoint: **Möller-Trumbore leaks 6 of 268,
the watertight form leaks 0.** A leak is a pinhole through a closed surface — the ray escapes
because neither of the two triangles sharing the feature claims it.

⚠️ **The published swap of `kx`/`ky` when the chosen axis is negative is omitted on purpose, and
this was measured, not assumed.** It keeps the coordinate system right-handed; flipping handedness
negates `u`, `v`, `w` and the determinant *together*, and both places they are used here are
invariant to that — the inside test accepts all-non-negative or all-non-positive, and the distance
divides by the same determinant. Over 20000 rays against 300 triangles the swap changes no hit and
moves no distance by more than 0.0. It is needed when the determinant's sign drives back-face
culling; this module culls with the facet normal instead. It was carried at first, and its mutation
was the one that came back green — the right conclusion there was to delete the code, not to add a
test for behaviour that does not exist.

## The surface system: what is exact, what is not, and by how much

`(I - diag(rho) F) B = M + rho * ((F^M - F) M + H_external)`, solved matrix-free. **The bounce
count is not a parameter** — the inverse is the infinite bounce sum. Verified against an explicit
Neumann series: 200 terms agree to 1e-10, one term is wrong by more than 100%.

**Exact, and gate on these:**

- **Row sums are 1 to about 1e-15** at every refinement. That is what bounds `spec(rho F)` by the
  largest reflectance and makes the system well conditioned. `row_sum_error` reports it.
- **`B = M/(1-rho)` on a uniform closed box, to 1e-12, at rho = 0.9** — where a single bounce
  gives 1.9 against 10, so nothing that truncates can pass.
- **A Lambertian source's energy balances exactly** — total landing equals total leaving,
  1.000000 at every refinement.

⚠️ **NOT exact — say the number rather than the assumption. None of these converges under mesh
REFINEMENT; the first two converge in the receiver QUADRATURE, which is the knob that exists.**

- **Reciprocity** is the quadrature error on the receiving facet, and nothing else. Refinement
  does not touch it: shrinking a closed box's facets brings their neighbours proportionally
  closer, so the ratio the error depends on never changes and `reciprocity_residual` reads the
  same at 12, 48, 192 and 432 facets. It falls with the point count instead — **0.2421 / 0.0281 /
  0.0078 / 0.0045 at 1 / 3 / 6 / 12 points**, the same four numbers at every refinement. Six is
  the shipped default. Still a **diagnostic, not a gate**.
- **Global energy conservation follows reciprocity, and at one point per receiver it GETS WORSE
  as the mesh is refined.** On a small lamp in a large box (4 emitting facets of `inward_box(d)`,
  reflectance 0.9) absorbed/emitted measures 1.000000 / 0.999868 / **0.982769 / 0.975625 /
  0.973077** at 12 / 48 / 192 / 432 / 768 facets: 2.7% of the lamp's output lost and still
  growing, because the facets nearest the lamp close in on it while staying the same size
  relative to their separation. At six points the same scene gives 1.000000 / 1.000212 /
  1.000171 / 1.000120 / 1.000102, improving. Per column, `sum_i A_i F_ij = A_j` is violated by
  **8.8% / 1.5% / 0.38% / 0.18%** at 1 / 3 / 6 / 12 points.
  ⚠️ **The earlier record here — "1.000112 / 1.000083 / 1.000062 … conservative to one part in ten
  thousand" at the centroid rule — is DELETED, not corrected: it does not reproduce and its
  fixture was not written down.** The measurement above is the lamp-in-a-dark-box fixture, whose
  first two entries match it exactly and whose remainder does not. The likely reason is the one
  already on record two sections down: **a box in which every facet emits balances to 1.000000 at
  every mesh and every rule**, because each facet's error is its neighbour's and they cancel
  identically — so an all-emitting fixture cannot see this defect at all.
- **A non-Lambertian source's energy still does not balance, and the receiver quadrature does not
  help** — the error is source-side, its profile being evaluated at the single centroid-to-centroid
  direction, which must stay outside the frozen build to keep its parameters differentiable. At
  six points a cosine-power exponent of 8 gives 1.086 / 0.978 / 0.982 / 0.987 at 12 / 48 / 192 /
  432 facets against 1.145 / 0.970 / 0.970 / 0.977 at one point, and the two agree to three
  figures from three points upward. This one *does* shrink with refinement, since more directions
  get sampled.

⚠️ **`reciprocity_residual` is normalized by the LARGEST entry, not per pair.** Two facets of the
same flat wall transfer nothing and hold values around 1e-18; a per-pair relative measure turns
that rounding noise into a residual of 0.97 while those pairs carry, measured, 0.0000 of the total
transfer. The first version did exactly that and reported ~1.0 on a healthy matrix.

## ⚠️ `geometric[i, j]` IS THE FORM FACTOR *FROM* `i` *TO* `j`, SO THE AREA MULTIPLIES THE ROW

`build_transfer` evaluates `projected_solid_angle` over facet `j` from points on facet `i`, which
is `F_{i->j}`. It is used as the weight with which `j`'s radiosity lights `i` — and that is the
same number, not a second one reached through reciprocity: the irradiance at a point `x` is
`sum_j B_j F_{dA_x -> A_j}` by definition. Reciprocity pairs `A_i F_ij` with `A_j F_ji`.

**The shipped `reciprocity_residual` weighted the COLUMN, and no test could see it**, because
every fixture in the suite was a subdivided cube, where all facets share one area and the two
expressions are identical. Measured once a fixture had unequal areas: on a box stretched to
1x1x3, the row weighting reads 0.2124 / 0.0930 / 0.0318 / 0.0072 across the four rules while the
column weighting reads 0.9596 / 0.9179 / 0.8983 / 0.8912 — large and flat. On two squares of area
1 and 100 a hundred widths apart, 0.0022 against 0.9999. `stretched_box` in the tests exists for
this; reach for it whenever a claim involves an area.

## The receiver quadrature: why ONE side of the double integral is exact and the other is not

A transfer factor is a double area integral. The source half is closed form
(`projected_solid_angle`); the receiver half is quadrature (`quadrature.py`, six points by
default). That asymmetry is deliberate and is the better of the two available trades:

- **Source exact + receiver quadrature** — row sums exact to 1e-15 at every rule, reciprocity
  O(quadrature error). Each quadrature point sees a whole closed enclosure so its own row sums to
  one, and the weights sum to one.
- **Both by the same quadrature** — reciprocity exact *by construction* (the double sum is
  manifestly symmetric), row sums only approximate.

The row sum is what bounds `spec(rho F)` and what catches a wrong kernel, so it is the one to keep
exact. Do not "finish the job" by quadraturing the source too.

⚠️ **A FINER RULE COSTS FAR LESS THAN ITS POINT COUNT — 6 points is 1.15-1.72x, not 6x.** The build
is limited by moving geometry through memory, not by evaluating the kernel, and the extra points
reuse triangles already loaded. Median of five warm `build_transfer` calls on closed boxes, JAX
0.10.2, CPU, x64, macOS arm64, 11 cores, default `chunk_size=256`: at 192 / 768 / 1728 / 3072
facets the one-point build takes 0.18 / 0.54 / 1.32 / 2.54 s, and six points costs 1.15 / 1.31 /
1.37 / 1.72x of that (twelve points 1.09 / 1.60 / 2.03 / 3.63x). Wall clock on a shared desktop
carries ~20% spread, so read the shape and not two figures. **The first measurement of this said
6-13x and was wrong**: the probe vmapped all receivers and all quadrature points at once, so it
measured a 287 MB intermediate rather than the shipped path, which scans the quadrature points
inside each chunk precisely so `chunk_size` keeps meaning what it meant at one point per facet.

⚠️ **Polynomial degree does not order the rules by accuracy here.** The integrand goes like
`1/r^2` and is nearly singular between facets sharing an edge, which is where the error lives and
what a polynomial rule is worst at. The 7-point degree-5 rule measures **0.0169 against the
6-point degree-4 rule's 0.0078** — it spends nearly a quarter of its weight at the centroid — so
it is deleted from the catalogue rather than offered. The 4-point degree-3 rule is absent too: its
centroid weight is negative, which can drive a transfer factor below zero.

**Not integrated over the receiver, and necessarily so:** the centroid separation carrying
absorption, the source cosine carrying a non-Lambertian profile, and the occlusion mask. All three
multiply the geometric term elementwise so they can stay live and differentiable; moving them
inside the quadrature would put them back in the frozen `n^2` build. In a scene with partial
shadowing, or a medium absorbing appreciably over a facet's own width, those are the coarse
approximations — not the receiver rule. Tracked as
<https://github.com/DeGrootResearchGroup/aquaflux/issues/447>, which carries the measured
cosine-power numbers and says what has to be measured before the occlusion half is designed.

## ⚠️ THE STOPPING RULE IS CHOSEN, NOT DEFAULTED

A componentwise relative test asks every residual entry to fall below `rtol` times *its own*
right-hand side. Most facets do not emit — a lamp is a handful among walls — so most of that side
is exactly zero and the demand becomes unsatisfiable. Measured on a lamp-in-a-dark-box fixture:
stock `lx.GMRES(rtol=1e-10, atol=0.0)` **fails outright**, while the global relative test
converges in 3 restart cycles.

⚠️ What that does *not* show: the same stock solver with a **non-zero** `atol` agrees to 4e-14 on
the same scene. The componentwise rule is then quietly an absolute tolerance — wrong in a way that
scales with the problem rather than one that raises. Only the degenerate case is pinned.

**Every fixture that emits everywhere is the wrong shape for this class of question.** The
lamp-in-a-dark-box fixture exists because the all-emitting boxes could not see the defect at all.

## The frozen/live split, and the two ways it has been got wrong

Frozen at build: the projected solid angles, the source-side cosines, the centroid separations,
the occlusion mask — all `n^2`. Live per call: reflectance, emission, power, profile parameters,
occluder transmittance, absorption coefficient.

Both failures leave a finite, plausible number behind rather than a NaN or a zero:

- freezing the whole of `F` costs a few percent of `dG/da`;
- freezing the visibility inside the geometry term costs about two thirds of `dG/dt`.

The rule: **anything promised a gradient is computed OUTSIDE the frozen arrays**, as an
elementwise multiply against them. A uniform absorption coefficient goes through
`exp(-a * frozen_separation)` in closed form, so no geometry is revisited; any other `Absorption`
re-walks every pair on every call, which is correct and costs the `n^2` build again.

**The adjoint is an implicit solve, not the iteration replayed.** Pinned by varying the restart
length — 2 against 120 gives 47 cycles against 3 on the same problem — and asserting the
gradients agree to 1e-8. The step counts are asserted to differ, or the test compares a
configuration against itself.

## The public surface: `model.py`, and the four assembly steps that are easy to omit

`build_radiation_model(receivers, surfaces, occluders=..., settings=...)` freezes everything a
scene's *shape* decides — the `n^2` transfer, its facet-side shadow mask, and a second mask for
the receivers — and the three entry points read off it:

```
B, cycles = radiosity(model, surfaces)            # what each facet sends out   (n_facets,)
H, cycles = surface_irradiance(model, surfaces)   # what lands on each facet    (n_facets,)
G, cycles = fluence_rate(model, surfaces)         # the volume field            (n_receivers,)
```

Each returns the solver's restart-cycle count alongside its field: a field is not evidence of
anything until the solve behind it is known to have converged.

**It takes an array of receiver positions, not a `Mesh`.** Nothing here reads anything else from
one, and keeping the fence one-way is what stops radiation from growing a dependency on cells,
fluxes or residuals. Injecting `G` into a transport equation is the *consumer's* job, through the
transport package's own volume-source seam.

Four steps the assembly does that are each invisible when left out:

1. **Point sources are fed in as an arrival, not through the transfer matrix.** A zero-area facet
   has no area to emit from and no surface to receive on, so it is absent from `F` entirely. Omit
   the extra gather and a lamp-lit enclosure comes back dark — which is a field, not an error.
2. **The reflected part is re-gathered as LAMBERTIAN**, whatever the source emitted like, because
   that is what diffuse reflection means. Re-gathering it with the source's own distribution is
   wrong by a quarter to nearly a factor of two on a cosine-power-8 box, and destroys the
   uniformity the enclosure should have.
3. **The facets' own emission is subtracted before that second gather** — `outgoing - emission`,
   not `outgoing` — or every source radiates twice.
4. **Both shadow masks are built against the same bodies, in one call.** Built separately, the
   volume is lit through a sleeve the surface solve correctly treated as opaque.

**`RadiationSettings` fields are all `None` by default, and unset means ABSENT rather than
copied.** `_passed` drops them so each default stays written down once, beside its own reasoning.
A settings object carrying its own copy of a default silently keeps using the old number the day
the real one moves. Membership test: a setting whose reason can be stated without naming a lamp,
a wall or a medium belongs here; anything else is physics and belongs on `Surfaces` or an
`Absorption`.

## `G = 4B` — the closed form that pins the whole assembly at once

A closed Lambertian enclosure at uniform radiosity `B` has radiance `B/pi` in every direction, so
at **any** interior point `G = 4 pi (B/pi) = 4B`, with no dependence on position. Two fixtures use
it, and between them they cover every step above:

- uniform emission `M` and reflectance `rho`: `B = M/(1-rho)`, so `G = 4M/(1-rho)` — exact to
  1e-12 at `rho = 0`, 0.5 and 0.9. Drop the reflected gather and it returns `4M`, a tenth of the
  answer at 0.9.
- **cosine-power sources with zero emission, lit only by an `external_irradiance` `E`**: every
  watt present has been reflected once, so the enclosure is Lambertian again whatever its
  emitters are, `B = rho E/(1-rho)` and `G = 4B` still holds exactly. This is the fixture that
  catches step 2, and it is the only one that can: with Lambertian emitters the two distributions
  coincide and the bug is invisible.

**A point source in a dark box conserves energy only approximately, and the number is the
source-side one-point error again** — the lamp's direction to each wall facet is evaluated at
that facet's centroid. Absorbed over emitted, lamp at the centre of `inward_box(d)`, reflectance
0.9 on the walls: **1.413436 / 1.030131 / 1.010384 / 1.004639** at 12 / 48 / 192 / 432 wall
facets. It shrinks with refinement, unlike the reciprocity error, because refining samples more
directions. With `rho = 0` the same fixture gives `G = P/(4 pi r^2)` exactly.

⚠️ **Two mutations of `model.py` and `gather.py` survive and were dismissed rather than covered:**

- Removing `power=jnp.zeros(...)` from the reflected set does nothing, because the `Lambertian`
  profile imposed on the same object returns **zero** intensity along a point source's zero
  normal. The two guards overlap today; both stay, because the overlap is a property of
  `Lambertian` and not of the function, and the comment in `model.py` says so.
- Transposing `_chunked`'s reshape from `(n_chunks, chunk_size)` to `(chunk_size, n_chunks)` is
  inert: the padded array is flattened again in the same order whichever way it is factored, so
  only the chunk partition changes. What *is* covered is losing a receiver off the padded end,
  which needs a chunk size that does not divide the receiver count to show up at all.

## Documentation

The package **is** in `docs/conf.py`'s `PUBLIC_SUBPACKAGES`, with a `SUBPACKAGE_GROUPS` entry
keyed on the modules its names are *defined* in — `model` first, then the pieces it composes.
Listing a subpackage publishes the whole of its `__all__`, so that list is the editorial
decision. It was reviewed at this point and kept entire: every export is something a user can
legitimately reach for, including the two solid-angle kernels, whose warning that they are **not
interchangeable** is worth publishing rather than hiding.


## The winding number answers "is this cell inside the metal", and it is exact

`enclosure_winding(vertices, points)` sums the **signed** solid angle of every facet at each point
and divides by `4π`. On a closed, consistently wound surface that is `±1` inside and `0` outside,
the overall sign set by whether the file is wound outward or inward — so `check_points_outside`
tests the **magnitude** against 0.5, a threshold with nothing behind it to tune because the
quantity it cuts takes only two values.

Measured on a unit box at 12, 48 and 192 facets: `1.0` to the last bit at a point a thousandth of
a box-width from a wall, and `~1e-16` just outside. There is no near-field regime where it
degrades, which is what disqualifies the two obvious alternatives:

| test | why not |
|---|---|
| ray parity (odd crossings = inside) | only meaningful on a genuinely **closed** surface, and this package deliberately does not require watertightness. On an open one it returns a clean, wrong bit. |
| nearest-triangle signed distance | the pseudonormal problem — unreliable near edges and creases, which is what a reactor corner is made of. |
| **winding number** | exact, and on an open surface it lands *between* the two answers rather than guessing. |

⚠️ **An open surface reads ~0.45 just off a bare disc, and that is the feature.** Open surfaces are
legal here, so `check_points_outside` **warns** rather than raising when nothing is enclosed but
the largest winding is above 0.01: a clean pass from a surface with no inside establishes less than
it reads as, and refusing would reject a geometry the rest of the package accepts.

⚠️ **An inconsistently wound closed surface is NOT detected as such — it reads as an open one.** A
capped cylinder with one cap reversed measures **0.858 and 0.951 at two interior points**: not 1,
not 0, and *different at each point*, which is the tell, since a real winding number is constant
over a region. Run `check_winding` first; this check cannot substitute for it and does not try.

**Cost is `n_points × n_facets`**, the same product the receiver visibility build already pays, so
it is affordable but not free — which is why `build_radiation_model` does **not** call it. Analytic
occluder bodies do refuse interior points at the visibility build, because `Occluder.contains` is
O(1) per point; a triangle soup has no such shortcut. Wiring it into the model by default would
change a shipped behaviour and roughly double that build, so it is an explicit call.

## The two engineering-unit conversions, and the numbers that make them worth having

`units.py` exists because both conversions were being done by hand at every call site and both are
wrong in ways that produce a plausible field.

**`absorption_from_uvt(uvt)` — ultraviolet transmittance (%, through 1 cm) to a NAPIERIAN
coefficient per METRE.** That is what `exp(-a r)` and metres-based geometry want. Two other
conventions are in circulation and both are silently wrong here: a *decadic* coefficient (paired
with `10^(-A r)`) is smaller by `ln 10`, and a per-*centimetre* one by a hundred. A 95% UVT water
is **5.129 /m** napierian, **2.228 /m** decadic, **0.05129 /cm**.

⚠️ **A fraction passed as a percentage is refused, and it is the one mistake a range check cannot
catch** — `0.95` is a legal percentage. Read as written it gives **466 /m** against 95% UVT's
**5.13**, ninety times apart, and the field is dark rather than erroneous. The cost is that a
genuine sub-1% water cannot be expressed; that is outside the range ultraviolet reactors are built
for, and Beer's law inverts in one line by hand. Note the guard is over the **whole array**, since
a single fraction hidden among percentages is how this arrives.

**`lamp_exitance(surfaces, {"lamp": watts})` divides by the TRIANGULATION's area, not the shape's.**
`sum(M A)` over the body then equals its rating exactly at any refinement, because the same areas
appear on both sides. Dividing by the analytic `π d L` does not: an inscribed triangulation of a
0.0115 m × 0.4 m sleeve undershoots it by **2.55% / 0.64% / 0.16% / 0.04%** at 8 / 16 / 32 / 64
sectors, so a hand-computed exitance makes the model radiate that much less than the lamp — always
in the same direction, and reported nowhere. `Surfaces.area_by_solid` is the one home for the
per-body total, so a helper and a user's own report cannot disagree about how big the lamp is.

⚠️ **Which body gets the rating is a modelling decision no signature can make.** A lamp is rated at
its envelope; the geometry in a model is usually the quartz sleeve, which is larger. Both readings
are legal and they differ by the area ratio.
