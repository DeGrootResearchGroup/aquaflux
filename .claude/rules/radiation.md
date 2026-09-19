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
| `solid_angle.py` — the two geometric kernels | **BUILT** |
| `stl.py` — ASCII and binary STL reading | **BUILT** |
| `surfaces.py` — the `Surfaces` value object | **BUILT** |
| `checks.py` — build-time geometry checks | **BUILT** |
| `subdivide.py` — the width-over-distance refinement | **BUILT** |
| `profiles.py` — `Isotropic`, `Lambertian`, `CosinePower` | **BUILT** |
| `gather.py` — `fluence_rate` and `irradiance` | **BUILT** |
| `absorption.py` — `UniformAbsorption`, `VoxelAbsorption` | **BUILT** |
| `occluders.py` — `Cylinder`, `HalfSpace` | **BUILT** |
| `visibility.py` — the frozen shadow mask | **BUILT** |
| the radiosity system | Not yet built |
| Beer–Lambert optical depth, voxel-grid traversal | Not yet built |
| occlusion against the emitting geometry's own triangles | Not yet built |

## ⚠️ THERE ARE TWO SOLID-ANGLE KERNELS AND THEY ARE NOT INTERCHANGEABLE

This is the defect that recurred through five drafts of the design, and it recurs because the
wrong answer is *clean* rather than noisy.

- `solid_angle` — the plain solid angle `Ω`. For **fluence rate**, which carries no receiver
  cosine because the receiver is a point in a volume, not a surface.
- `projected_solid_angle` — `∫cos θ dω`. For **irradiance** and for every surface-to-surface
  transfer factor, because an oblique surface intercepts less.

No scalar converts one into the other: the obliquity varies across the emitter. Used in the
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
`profile_index` cannot themselves be traced. `jit(lambda s, p: fluence_rate(s, p))` over a whole
`Surfaces` raises with an explanation; close over the set and substitute values through
`with_optics` instead. This is the boundary the built model will formalize.

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

## Documentation

The package is **deliberately absent from `docs/conf.py`'s `PUBLIC_SUBPACKAGES`.** Listing a
subpackage publishes the whole of its `__all__`, and the user-facing surface — `fluence_rate`,
`radiosity`, `surface_irradiance` — does not exist yet. Adding it now would put two internal
geometry helpers on the published site and force a re-cut later. Add it in the change that
introduces the public entry points, together with a `SUBPACKAGE_GROUPS` entry keyed on the
modules those names are *defined* in.
