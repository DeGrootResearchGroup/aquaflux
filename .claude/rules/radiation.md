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
| `self_occlusion.py` — the `SelfOcclusion` strategies: ray cast, silhouette clip, none | **BUILT** |
| `silhouette.py` — the exact covered fraction of a source, and the conservative cone cull | **BUILT** |
| `clipping.py` — convex clipping with filtered (decidable) sign tests, shared with `solid_angle.py` | **BUILT** |
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
so they collapse into one layer, `Visibility.hidden_by_geometry` — a **fraction** of each source,
beside a boolean `overlapping` — rather than one per body. That second kind is what lets a bent
duct shadow itself, which no primitive can express because the geometry doing the blocking *is*
the emitting surface. **How** it is computed is an injected `SelfOcclusion` strategy
(`self_occlusion.py`): `RayCastOcclusion` (one ray per pair, a 0/1 fraction — **the default**, what
`self_occlusion=None` resolves to), `SilhouetteOcclusion` (the exact clipped fraction, below), or
`NoOcclusion`. ⚠️ **"Off" is `NoOcclusion()`, never `None`**: `RadiationSettings` drops `None`
fields so they fall through to defaults, so a `None` meaning "off" would silently switch the ray
mask back *on*. A surface that does not shadow itself is the defect the module exists to fix, and
a mask silently missing it looks exactly like one that includes it.

**A model builds TWO masks — facet to facet, and facet to volume receiver — and they are routed
separately** (`RadiationSettings.visibility_options()` for the first,
`receiver_visibility_options()` for the second). `receiver_occlusion` overrides the second; unset,
it follows `self_occlusion` wherever that strategy declares `serves_volume_receivers` (the ray test
and `NoOcclusion` do), so "off" and "ray test" still apply to both masks alike. A strategy that
cannot serve a point in the fluid (`SilhouetteOcclusion`) is *not passed on*, so the volume mask
reaches `build_visibility`'s own default — the ray test — rather than a second copy of it. ⚠️ **Until
this split, `build_radiation_model` RAISED with the silhouette strategy selected** (#471 shipped it
that way): one strategy went to both masks and the clip correctly refuses volume receivers. Nothing
caught it because every silhouette test called `build_transfer` or the strategy directly and none
built a model; `test_a_model_can_be_built_with_the_silhouette_strategy` does now. ⚠️ **So with the
silhouette selected, the fluence rate in the fluid is still all-or-nothing per pair** — only the
surface transfer gets the exact fraction. Teaching the clip the unprojected (volume) measure is
the real fix and is a separate piece of work.

**Exclusion is by index, never by tolerance.** Every ray leaves its facet's centroid, so the
facet is always hit at zero distance. Excluding its whole *solid* would be wrong — a bent duct is
exactly the case this is for, and there the blocking wall belongs to the same body as the emitter.
Edge-adjacent neighbours are handled by the same near-origin exclusion the primitives use.

## ⚠️ A RAY BETWEEN TWO FACETS NEEDS **TWO** EXCLUSIONS, AND THE MISSING ONE SHIPPED

`segment_is_cut` excluded only the **source** facet. The far end of a segment has no margin —
`offset_scale` guards the origin, nothing guards the target — so a ray aimed at a *facet centroid*
ends exactly in that facet's plane and the hit at `distance == 1` counted. `build_transfer`'s
receivers **are** the facet centroids, so at the shipped default (the ray mask on) every
mutually visible pair read as blocked: measured on a closed box, **120 of 132 off-diagonal pairs**,
and on two bare plates facing each other across empty space, all of them. A closed enclosure came
back with `B = M` — ten times too dark at `rho = 0.9`, and shaped like a field rather than an
error. `exclude` now takes `(n_rays,)` or `(n_rays, k)`, and `build_visibility` takes
`receiver_facet` for the case where each receiver sits on a facet.

**Three separate reasons nothing caught it, all worth keeping:**

- **Every transfer and radiosity fixture switched the mask off** (then `self_occlusion=False`, now
  `self_occlusion=NoOcclusion()`), so the default was never
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

## ⚠️ THE MEMORY OF ONE PASS WAS THE WHOLE PERFORMANCE STORY, BECAUSE THE PASS RAN EAGERLY

The ray-by-triangle intermediate used to be the cost of the intersection test, and throughput did
not degrade gracefully — it fell off a cliff. Measured on 2048 triangles, f64, 11 cores, 19 GB:

| intermediate | Mtest/s |
|---|---|
| 0.5 - 134 MB | **42 - 57** |
| 537 MB | **2.4** |

⚠️ **EVERY NUMBER IN THAT TABLE WAS A PROPERTY OF THE CALL SITE, NOT OF THE KERNEL** — and that is
the lesson to keep, because nothing in the profile said so. `segment_is_cut` was invoked from a
Python loop with no `jit` anywhere on the path, so each block materialized its intermediate, which
is exactly why the memory of one pass was the story. The per-block kernel is now `_block_is_cut`,
traced: the compiler fuses the edge test, the distance window and the exclusion straight into the
`any` reduction and never forms the array. **6.6-8.3x, bit-identical output** (100,000 rays x 200
triangles, f64, 11 cores, 19 GB, jax 0.10.2, 2026-09-20, with and without the `exclude` argument;
54.0 -> 362.4 and 57.8 -> 430.2 Mtest/s at `work_limit` 4M and 20M). Issue #462. ⚠️ Those are
**small-block** absolute rates, taken under the old rays-first call shape below; read the speedup,
not the throughputs. A whole build at 3184 facets now runs at ~420 Mtest/s.

**`work_limit` bounds rays x triangles per compiled call, and the call takes TRIANGLES FIRST**
(`_call_shape` in `triangles.py`): as many triangles as the bound allows — the whole set, for any
mesh this pass can afford — and as many rays as then fit. The triangle block is the reuse factor
(each ray's origin, direction, margin and exclusions are streamed once per call and tested against
every triangle in it), so the order is not cosmetic.

⚠️ **RAYS FIRST WAS THE SHIPPED ORDER, AND IT WAS THE WHOLE OF THE "THROUGHPUT FALLS WITH THE MESH"
FINDING BELOW.** A transfer build's ray count is receivers x facets, millions, so rays took the
entire budget and left a block of **one** triangle: every call streamed millions of rays to test a
single triangle. Same kernel, same rays, bit-identical output, 1532 triangles, jax 0.10.2, CPU, x64,
11 cores, 2026-09-21: **120.5 Mtest/s rays-first against 465.4 with the whole set per call at 3.2M
rays** (350.5 against 464.1 at 200k). Block sizes 16 / 64 / 256 / 1532 gave 333 / 434 / 447 / 465 —
monotone, so bigger is better all the way to the full set. `test_the_triangle_block_does_not_shrink_as_the_rays_grow`
pins the order; reverting it fails there and passes every correctness test, which is why the test
exists.

⚠️ **The record used to say the opposite — "a *larger* triangle block made it ten times worse" (13.4
against 31.0 Mtest/s).** That was measured when the pass ran **eagerly** and each call materialized
its rays x block intermediate, so a big block meant a big array. Tracing (#462) fused the
intermediate away and reversed the trade-off, and the blocking rule was never revisited. **A rule
tuned under one execution model is a hypothesis under the next.**

**The bound is still load-bearing for speed, not only memory — and now in the other direction.**
With triangles first, raising it only enlarges the ray chunk per call, and whole builds get
*slower*: 100M against the default 4M reads **156.1 against 443.8 Mtest/s at 832 facets** and
**202.6 against 463.5 at 1532** (same run as the ladder below). The ~30% a larger bound once seemed
to buy was a rays-first, small-block artifact. ⚠️ **Not measured: whether a bound *below* 4M is
faster still** — the trend suggests smaller ray chunks help, and nothing has checked where it
turns. Each distinct call shape compiles once, so an uneven division costs one extra small program
for the remainder, not one per block.

## Watertight intersection, and one piece of the published algorithm deliberately dropped

Woop, Benthin & Wald (*JCGT* 2(1), 2013) rather than Möller-Trumbore. Measured on a closed hull
with rays from inside aimed at every vertex and edge midpoint: **Möller-Trumbore leaks 6 of 268,
the watertight form leaks 0.** A leak is a pinhole through a closed surface — the ray escapes
because neither of the two triangles sharing the feature claims it.

⚠️ **THE WATERTIGHT GUARANTEE RESTS ON EXACT ARITHMETIC THAT A COMPILER IS FREE TO BREAK, AND IT
DID.** Two triangles sharing an edge evaluate the same edge function with the two operand pairs
swapped, and a ray through that edge is claimed by exactly one of them only while the two results
are **exact negatives**. Written as `a * b - c * d` that holds one operation at a time — floating
multiplication commutes bit for bit, so the two are `fl(P) - fl(Q)` and `fl(Q) - fl(P)`. It does
**not** hold once the expression is compiled: XLA contracts it into a fused multiply-add, which
keeps one product at full precision and rounds the other, so the two triangles keep *different*
products exact. Measured on 200,000 random operand quadruples, the plain difference and its swap
are not exact negatives on **a third** of them — and tracing the ray-test kernel on that form
reopened **6 of 268** leaks on the closed-hull fixture, which is precisely the Möller-Trumbore
failure count the watertight form exists to beat.

Three things about this are worth carrying to any other exact-arithmetic predicate here:

- **It is invisible in the source and invisible eagerly.** The shipped code was watertight only
  because nothing had traced it; the guarantee was an accident of the call site, exactly like the
  throughput above. A caller wrapping the build in `jit` would have silently lost it.
- **Neither an XLA flag nor `lax.optimization_barrier` prevents it.** `xla_allow_excess_precision`
  and `xla_cpu_enable_fast_math` change nothing in either direction, and the barrier does not
  survive compilation — `fma` is still in the compiled HLO and the violation count is unchanged to
  the last item. Do not reach for a flag; fix the arithmetic.
- **The fix is to average the expression with the negation of its own swap**
  (`_edge_function`): whatever the compiler does to `a - b` it does to `b - a` up to an exact sign,
  because IEEE subtraction is antisymmetric however its operands were formed. That is exactly
  antisymmetric compiled *and* bit-identical to the one-operation-at-a-time value, for two extra
  multiplies and a subtraction per edge — which does not show up against the edge test at all
  (6.6-8.3x still, measured with it in place). Pinned by
  `test_the_edge_function_survives_being_compiled`, which also asserts the plain difference still
  *fails*, so the fixture cannot quietly stop proving anything.

**How far the fix goes, swept rather than argued (`validation/radiation_watertight_sweep.py`).**
Seven bodies closed to the last bit — convex hulls at 24/40/60/90 points, closed drums at 16 and
48 sectors, and an L-prism with a reflex edge — with rays from inside aimed at every vertex, edge
midpoint and face centroid. **8,764 rays, 0 leaks, eager and traced.** The same sweep on the plain
difference leaks **124**, all of them only once compiled. The other two candidates in the module
were checked and are **not** hazards:

| site | exact cancellation needed? | measured |
|---|---|---|
| inside test `sign(u), sign(v), sign(w)` | yes — picks which triangle claims a ray | broke; fixed |
| the shear `ox - shear * oz` | no — the same numbers go in, so the same come out | 0 disagreements |
| contour form `jnp.cross` (tiling additivity) | no | 2e-16 traced and eager |

⚠️ **THE DISCRIMINATOR IS WHETHER THE CANCELLATION FEEDS A DISCRETE PREDICATE.** The contour form
is built on the same difference of products, but a last-bit change there moves an *angle* by a
last bit. The inside test feeds it to a sign test that decides which of two triangles claims a
ray, and a discrete predicate has no small errors — it is right or it is a pinhole. Apply this
test before assuming the next exact-arithmetic site here is safe or unsafe.

Three traps met while measuring this, each of which produced a confidently wrong answer first:

- **Aim points snapped to a tolerance make the sweep blind.** The first version rounded the
  vertices and midpoints to 12 decimals before deduplicating them, which moves them off the
  feature. Its control came back **clean**, which is the only reason the mistake surfaced — a
  tightness sweep with no control arm is worth nothing.
- **`cylinder_triangles` does not close**, so a capped one leaks 13 rays eagerly with any edge
  function. Its angles run `linspace(0, 2*pi, n+1)` and the last sector ends at `2*pi`, whose sine
  is `-2.4e-16` rather than zero. Those 13 were briefly read as residual FMA sensitivity. Build a
  closed body from one vertex table (`closed_prism` / `closed_drum`), not from trigonometry
  evaluated twice.
- **Swapping a module global does not invalidate a compiled version that read it.** `jit` keys its
  cache on the function object, so the second arm of an in-process A/B silently replays the
  first's program. Here that made the fixed arm reproduce the control's leak counts *exactly*,
  which is the only tell. `jax.clear_caches()` between arms, or separate processes.

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
elementwise multiply against them. ⚠️ **That rule is what makes the one-point factors one-point,
and it does not force the frozen side to be a single number.** A frozen array may hold whatever
*geometry* a live factor needs, so long as the live parameter stays outside it — which is how a
non-Lambertian profile's bias could be removed by freezing two moments of `log cos` per pair
instead of one cosine, with the exponent still live. Measured and decided against building, for
reasons recorded in the one-point-factors section below; the point here is that "live" constrains
what the frozen array may *depend on*, not how wide it may be. A uniform absorption coefficient goes through
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
## MEASURED: what the binary per-pair occlusion mask costs (issue #447 item 2)

`build_visibility` casts one ray per facet pair, centroid to centroid, so a half-shadowed pair
is recorded wholly blocked or wholly clear. With the source exact and the receiver on six points
this is the only all-or-nothing term left in the transfer. It is now measured;
`validation/radiation_partial_occlusion.py` is the instrument and re-runs in ~90 s.

**Configuration for every number below.** Two 2 m square plates facing each other across a 2 m
gap, emission 1, reflectance 0, self-occlusion off (`NoOcclusion()`), default six-point receiver quadrature,
an opaque `Cylinder` on the axis between them. Reference: the same plates at 36 quads per side
(5184 facets), area-averaged back onto the coarse patches — which *is* the coarse form factor,
not a finer answer to a different question. JAX 0.10.2, CPU, x64, macOS arm64, 2026-09-19.

**The control is what makes it a measurement.** With the cylinder removed the same comparison
reads **5.673e-07**. That is the instrument's floor — six quadrature points on one large
receiving triangle against six on each of many small ones — so everything above it is the mask.

⚠️ **A MAX COLUMN WAS RECORDED HERE AND IS WITHDRAWN (2026-09-20). The trap is worth more than
the numbers were.** The worst single transfer entry is *not determinable* on this problem at any
affordable resolution: two independent reference constructions at the fine mesh — 16 source pixels
against 64, and source pixelation against receiver quadrature — disagree by **max 0.12 and 0.25
respectively, flat in both mesh and pixel count**, which is the same order as the coarse-against-
reference max that was being quoted as an error bound. A per-pair maximum of a *discontinuous*
quantity is a lottery in where the shadow edge happens to fall, and it is a lottery for the
reference too. **Before quoting a maximum, check that your reference has one.** The mean is sound:
once aggregated onto coarse patches the two reference constructions agree to 0.0001-0.0008, far
below the numbers below, because averaging kills the per-pair lottery.

Mean error in the transfer, normalized by the largest reference entry:

| quads/plate | r=0.15 | r=0.30 | r=0.60 |
|---|---|---|---|
| 2x2 | 0.0828 | 0.0715 | 0.0339 |
| 3x3 | 0.0405 | 0.0310 | 0.0129 |
| 4x4 | 0.0196 | 0.0217 | 0.0184 |
| 6x6 | 0.0145 | 0.0048 | 0.0117 |
| 9x9 | 0.0128 | 0.0053 | 0.0069 |
| 12x12 | 0.0043 | 0.0052 | 0.0023 |

**The mean falls by roughly twenty-fold from 2x2 to 12x12.** What happens to the worst pair is
not measured and is not measurable here — see the withdrawal above. The *reasoning* that a pair
the shadow edge crosses is wrong by up to the whole of its own value at any resolution still
holds, and it is reasoning rather than measurement: refining changes how *many* pairs straddle
the edge, not how wrong a straddling one is. Treat it as the mechanism, not as a bound.

**A thin body is worse than a fat one**, by about a factor of two in the mean at every mesh
(0.0828 against 0.0339 at 2x2; 0.0043 against 0.0023 at 12x12). A rod narrower than a facet
either falls between two ray endpoints and vanishes, or lands on one and blocks the whole pair.
That is the lamp-sleeve and baffle regime, which is what the package is for.

**Row sums drift by up to 0.037** relative to the reference — a facet's total output misrouted
by up to 3.7 percentage points. Note the mask legitimately removes energy (an opaque body
absorbs it); this is the error in *how much*, not the removal itself.

**What a user reads.** Fluence rate on a line at z = 0.5 m, radius 0.30 m, against the 36-per-side
reference. ⚠️ **That reference carries about 0.5% of its own uncertainty and is not converging**
(max change 0.47% from 18 to 24 quads per side, 0.51% from 24 to 36), so the first two rows stand
at five to ten times the floor and **the third is at it**:

| quads/plate | worst, of the field | worst, of the shadow being modelled | mean, of the field |
|---|---|---|---|
| 2x2 | 4.44% | **58.4%** | 2.07% |
| 4x4 | 2.50% | **35.9%** | 1.17% |
| 8x8 | 0.88% *(at the floor)* | 12.5% *(at the floor)* | 0.34% |

⚠️ **Quote the second column when sizing a fix.** As a fraction of the field the error looks
like a few percent; as a fraction of *the shadow the occluder exists to cast* it is a third at a
perfectly ordinary mesh. Those are the same numbers, and the first framing is the one that makes
this look ignorable.

⚠️ **Two traps this measurement walked into, both worth keeping.**

- **The aggregation map was wrong and every ownership check passed.** The two triangles of a quad
  are *adjacent* in the facet order, not in two blocks; the wrong map still gave every coarse
  patch exactly two facets on the correct plate. Only the control caught it, reading **0.398**
  where it should read 5.7e-07. A control that should be ~0 is worth more than any assertion
  about the structure of the thing being measured.
- **The plate fixture half-fails silently.** The second plate must be reversed or its normal
  points away and the source clamp deletes every transfer into it — the matrix is half zeros, and
  the comparison still passes, being perfectly consistent about a quantity that no longer exists
  (1.5986 one way, exactly 0.0 the other). `test_the_two_plates_actually_face_each_other` guards it.

`tests/unit/test_radiation_partial_occlusion.py` carries the cheap half in the fast tier: the
control, that the mask error is orders above it, and both fixture guards. Six of seven mutations
go red; the survivor — dividing by the patch area in `area_average_onto` — is inert because a
coarse patch and its fine cover have the same total area, so the factor cancels on both sides of
a normalized comparison.


## MEASURED AND REJECTED: two candidate fixes for the occlusion mask, and what standard practice does

Both obvious treatments were built and measured before any was designed in. **Neither is worth
its cost**, and recording that is the point of this section — the alternative is someone
rediscovering it.

**What other codes do**, since the analogy drives the design:

| family | how it treats partial occlusion |
|---|---|
| **Hemicube** (Cohen & Greenberg 1985) | rasterize the scene onto a half-cube around the receiver; a z-buffer per pixel resolves occlusion. Fluent exposes it as `Resolution` (default 10), raised "to reduce aliasing" — a fixed raster has our failure mode at finer granularity, not a different one. |
| **Discrete-ordinates pixelation** (Fluent DO) | subdivide a *control angle* that straddles the receiving face's plane into pixels, classify each. Default **1x1** for grey-diffuse; **3x3** only for symmetry, periodic, specular or semi-transparent boundaries — pixelation is turned up where the discontinuity bites, not globally. |
| **Discontinuity meshing** (Heckbert 1992; Lischinski, Tampieri & Greenberg 1992) | put mesh boundaries **on** the shadow edges, built from source edges against occluder vertices, so no element straddles. The only family that attacks the worst element rather than the average. |
| **Uniform ray seeding** (OpenFOAM `createViewFactors`) | N rays per face over a hemisphere, explicitly trading accuracy for production-scale performance. |
| **The ultraviolet-reactor mainstream** (LSI, MPSS, MSSS, RAD-LSI, UVCalc3D) | **does not model shadowing at all.** So the binary mask here is already ahead of that field's standard practice. |

**The measurement.** Same fixture; reference `src64` at 18 quads per side, independent of every
treatment; noise floor carried alongside. Mean error, normalized by the largest reference entry:

| coarse | today (1 ray) | receiver quadrature (6 rays) | source pixels (4 rays) | source pixels (16 rays) | reference uncertainty |
|---|---|---|---|---|---|
| 2x2 | 0.0806 | **0.0341** | 0.0453 | 0.0457 | 0.0004 |
| 3x3 | 0.0334 | 0.0076 | 0.0146 | **0.0060** | 0.0001 |
| 6x6 | 0.0051 | **0.0020** | 0.0036 | 0.0022 | 0.0004 |
| 9x9 | 0.0047 | 0.0033 | **0.0024** | 0.0025 | 0.0008 |

⚠️ **SOURCE-SIDE PIXELATION DOES NOT BEAT RECEIVER QUADRATURE, AND THE REASON IS STRUCTURAL.**
Neither axis dominates: receiver sampling wins at 2x2, 3x3 and 6x6, source pixelation at 9x9, and
16 source pixels match 6 receiver points while costing 2.7x the rays. Quadrupling the pixels from
4 to 16 buys **nothing** at 2x2 (0.0453 to 0.0457) and nothing at 9x9.

The reason the borrowed technique does not transfer: **Fluent's control-angle overhang is a
discontinuity at the receiving face's own plane** — local, living entirely in the angular index,
so subdividing that index resolves it. **Ours is a remote occluder silhouette**, which lives in
neither index alone and cuts diagonally across the (receiver x source) product. Subdividing either
index catches only its projection, which is exactly what the table shows — each axis helps where
it happens to be the dominant variable and little elsewhere. Resolving a discontinuity in the
*product* needs 4D sampling (Monte Carlo) or an element boundary placed on the silhouette
(discontinuity meshing).

**Both treatments cap out at 2-4x on the mean**, for 4-16x the ray budget — the expensive half of
the build. Two useful negatives fall out for free: a fractional mask applied **outside** the
quadrature is indistinguishable from folding visibility **inside** it (so `transmittance` may stay
live and outside the frozen array, and no architecture change is needed for a future fix), and
storage is free either way — six points with fixed weights admit at most 2^6 distinct weighted
fractions, so a `uint8` bitmask is exact and costs the byte the bool already costs.

⚠️ **Solid angle is additive over a partition of the source**, so splitting a source facet leaves
`sum_k Omega_k == Omega` exactly. Source pixelation therefore costs nothing in the geometry term
and nothing in the row sums — only rays. That is a genuinely attractive property and it is *not*
enough to make the technique pay here.


## MEASURED AND DECIDED: the two OTHER one-point factors (issue #447 items 1 and 3)

`build_transfer` integrates the geometric term over the receiving facet, then multiplies it
elementwise by three factors evaluated at **one point per pair**, because all three are live and
differentiable and folding them into the quadrature would freeze them. The occlusion mask is the
third and is priced above. The other two are absorption and a non-Lambertian source's angular
profile, and both are now measured against a dense integral of the same quantity.
`validation/radiation_one_point_factors.py` is the instrument (~40 s);
`tests/unit/test_radiation_one_point_factors.py` carries the controls, the mechanism and the
orders in the fast tier.

**Configuration.** Two right triangles of unit area, so `w = sqrt(area) = 1` exactly and the
sweeps read directly against `a * w` and `w / d`; dense reference at 24 sub-triangles per edge
(converged — 12 and 36 agree to the figures quoted). JAX 0.10.2, CPU, x64, macOS arm64,
2026-09-20.

**Both controls read EXACTLY zero**: `a = 0` for absorption, and Lambertian `n = 1` for the
profile at every distance and angle tested. That is what makes the rest attributable to the
factor under test rather than to the sampling or the fixture.

### Absorption is FIRST order in `a * w` — and the obvious argument gives the wrong order

Relative error in a pair's transfer, axial pairs, against `a * w`:

| `a * w` | d=1w | d=2w | d=4w | d=8w |
|---|---|---|---|---|
| 0.03 | -0.44% | -0.33% | -0.19% | -0.10% |
| 0.10 | -1.46% | -1.08% | -0.64% | -0.34% |
| 0.30 | -4.27% | -3.18% | -1.90% | -1.01% |
| 1.00 | -13.04% | -9.90% | -6.08% | -3.29% |

⚠️ **THE FIRST VERSION OF THIS FINDING SAID SECOND ORDER, FROM REASONING THAT IS SOUND AND IS NOT
THE LEADING TERM.** `exp` is convex, so the average of `exp(-a r)` over a pair exceeds
`exp(-a <r>)` — true, second order, and swamped. The leading term is that the centroid separation
is not the separation the factor is actually averaged over, and that gap is *first* order in the
facet's extent. The bias is `-a * (<r> - r_centroid)` with `<r>` weighted by the pair's own
transfer kernel, and the prediction holds **at every geometry tested**, to four figures:

| d / w | `(<r> - r_c) / w` | measured slope |
|---|---|---|
| 1 | 0.1483 | -0.1481 |
| 2 | 0.1091 | -0.1090 |
| 4 | 0.0645 | -0.0644 |
| 8 | 0.0340 | -0.0340 |

For squarely facing pairs `<r> - r_centroid` falls like `w^2 / (4 d)`, so the bias is
`(a w) * (w / 4d)` — worst between **neighbours**, where `d ~ w` and it reaches about
**`0.15 * a * w`**.

⚠️ **A SECOND WRONG CLAIM CAME OUT OF THE SAME REASONING AND A TEST CAUGHT IT: `<r> - r_centroid`
IS NOT ALWAYS POSITIVE.** Jensen on the norm says the distance between two mean positions cannot
exceed the mean of the distances — true, and about the **kernel-weighted** mean positions, not
the geometric centroids the build stores. Slide a pair sideways and the `1 / r^2` weighting
concentrates on the facing near corners until the separation it effectively averages falls
*below* the centroid-to-centroid one, and the excess goes negative. Measured at `d = 1w`:
`+0.148` head-on, `+0.055` at half a width of offset, `-0.099` at one width, `-0.241` at two.
**That sign change is precisely the sign change in the bias** — one mechanism explains both the
magnitude and the flip, which the earlier hand-waving ("the geometry turns it the other way")
did not. Pinned over the whole offset family, including the negative arm, because a prediction
checked only on the axial corner is a prediction checked where it cannot fail. The rule of thumb now in
`build_transfer`'s docstring: at `a * w = 0.1` the error is under 2%; by `a * w = 1` it is past
10% and the closed form has stopped describing the scene. Water at 95% ultraviolet transmittance
absorbs at 5.13 /m, so 20 mm facets in it sit at `a * w = 0.1`. **This closes issue #447 item 3.**

### A non-Lambertian profile is SECOND order in the angular width

Bias against `(n - 1) * (w / d)^2`, collapsing onto a slope near -0.13 while that parameter is
small and saturating once it is not. Right-hand column is the two-moment split described below:

| d / w | n | bias | after two frozen moments |
|---|---|---|---|
| 1 | 2 | -8.29% | -0.012% |
| 1 | 8 | -39.70% | -3.316% |
| 1 | 16 | -59.27% | -23.363% |
| 2 | 2 | -2.90% | -0.001% |
| 2 | 8 | -17.41% | -0.213% |
| 2 | 16 | -31.26% | -1.850% |
| 4 | 8 | -5.50% | -0.006% |
| 8 | 8 | -1.48% | -0.000% |

### ⚠️ NEITHER BIAS HAS A FIXED SIGN, SO NEITHER IS A CORRECTION

Both run one way near the axis and the other way off it, **inside a single enclosure**. Absorption
at `a * w = 0.3`, `d = 1w`: -4.27% head-on, +3.28% at one facet width of lateral offset, +8.14% at
two. The profile at `n = 8`, `d = 1w`: -39.7% on the axis, -10.4% at 30°, **+46.5% at 45°** and
+368% at 75°. On the axis the centroid direction sits at the profile's *peak*, so the one-point
value is an extreme rather than an average; past about 35° the profile is convex across the
facet's angular span and it goes the other way.

Two consequences, and the second is the one that matters when sizing a mesh:

- **They partly cancel in a total and not at all in a local transfer.** This is why a global
  energy balance on a closed box reads 1-3% while individual pairs are wrong by tens of percent.
  Both numbers are real; they answer different questions, and the aggregate is much the more
  flattering. Size a mesh against the per-pair figure. (The same lesson as "quote the second
  column when sizing a fix", recorded above for occlusion.)
- **The huge grazing percentages are relative errors on nearly nothing.** `cos^7` at 75° is
  8e-5, so a +368% error there moves far less light than -39.7% on the axis. Do not read the
  grazing column as the dominant term; read it as the reason the sign is not fixed.

### ⚠️ A RATIO-SHAPED CONTROL IS BLIND TO ITS OWN WEIGHTING, AND ONLY MUTATION TESTING SHOWED IT

Every measurement above is a ratio — exact integral over closed form — and the sampling weight
appears on **both sides of it**. So the two controls that read *exactly* zero, and the mechanism
identity that holds to four figures, are all blind to that weight being the wrong weight. Dropping
the inverse-square from the pair kernel, or the cosine from the per-sample solid angle, leaves the
whole suite green: the controls still read exactly zero, because a constant factor cancels, and
the slope still equals the excess, because both are computed from the same wrong weight.

Caught by a nine-mutation pass, 3 of which went undetected on the first suite. A wrong weight would
have moved every absolute figure in this section in the same direction while every check designed
to catch exactly that kind of error stayed silent. The repair is to pin each weight against the
**shipped** kernel it mirrors — the per-sample solid angles against `solid_angle`, the pair kernel
against `geometric[i, j] * A_i`, agreeing to 2e-4 and 4.4e-4 at 24 subdivisions. The third
undetected mutation was the same shape of blindness in the fixtures rather than the maths: every
fixture put the source at the origin, where its centroid is the zero vector and
`centroid_r - centroid_s` cannot be told from `centroid_r`.

**The general form, worth carrying beyond this subsystem: a control that is a ratio proves the two
sides agree, not that either is right.** Where the quantity is an integral, pin the measure too.

### DECIDED (issue #447 item 1): documented, NOT built — and the split a fix would use

**Documented as a limitation.** Four grounds:

1. **It is exactly zero for the case that carries most of the light.** A Lambertian radiance is
   constant over direction, so evaluating it at the centroid direction is evaluating a constant —
   not an approximation at all, at any geometry or distance. The *reflected* component leaves
   Lambertian by assumption in every scene, so this can only ever touch the **emitted** transfer
   of a **non-Lambertian areal** source. Point and line sources are untouched as well: they have
   no extent, and the gather evaluates `intensity_fraction` at the true direction with the true
   `r^2`.
2. **Where it is not zero it is second order in the angular width**, so refinement reaches it —
   halving the mesh quarters it. The occlusion maximum, by contrast, refinement does not reach.
3. **It has no fixed sign**, so it does not accumulate across a scene the way a one-way bias would.
4. **It is the smaller problem next to item 2**, which buys 22x on a maximum no sampling treatment
   moves.

**The frozen/live split a fix would use, recorded because the issue asks for it either way.** The
route the issue anticipated — freeze a *set* of `Q` directions per pair and evaluate the profile
at each — multiplies the frozen `n^2` array by `Q`, and is not necessary. The profile is
`c^(n-1)` up to constants, so with `l = log c` the solid-angle-weighted average the transfer wants
is `<exp((n-1) l)>`, whose cumulant expansion is

    exp( (n-1)<l>  +  (n-1)^2 Var(l) / 2  +  ... )

Truncating after the variance needs **two** frozen numbers per pair where the build stores one
today — `source_cosine` becomes a mean and a variance of `log cos` — and leaves the exponent
entirely outside them, live and differentiable, which is the constraint that made the obvious
route expensive. Measured above: two to three orders off the bias in the practical regime,
degrading only where the bias is already large. ⚠️ **It is exact at `n = 1`**, both correction
terms vanishing with the error itself, so adding it cannot disturb the Lambertian path — the
reduction that pins the profile constants.

What building it would cost: one extra frozen `n^2` array (75 MB at 3072 facets, on ~225 MB the
frozen arrays already hold), and a build pass that samples each source facet to form the two
moments — which the **contour-form** solid angle does not currently do, so it is a new pass and
not a cheaper use of an existing one. That build cost, not the storage, is the real price.

## MEASURED: what the mask build costs in SECONDS — and why its throughput USED TO fall with the mesh

Every cost ratio in this subsystem divides by the mask build, and until now nothing recorded what
it costs on its own. `validation/radiation_mask_build_cost.py` is the instrument.

**Configuration.** Closed box plus a lamp sleeve down its axis, one self-occluding surface set,
receivers at the facet centroids with `receiver_facet` supplied — that is, the transfer build's
own case, where `n_receivers == n_facets == n_triangles` and the pass is **`n^3`**. Default
`work_limit`, median of three warm calls (two above 2000 facets). JAX 0.10.2, CPU, x64, macOS
arm64, 11 cores, 19 GB. "Before" is 2026-09-20 with the rays-first call shape; "after" is
2026-09-21 with triangles first (`_call_shape`), one run of the whole harness:

| facets | tests | before s | before Mtest/s | **after s** | **after Mtest/s** |
|---|---|---|---|---|---|
| 224 | 1.12e7 | 0.03 | 348.3 | 0.03 | 424.3 |
| 480 | 1.11e8 | 0.34 | 324.1 | 0.24 | 453.9 |
| 832 | 5.76e8 | 2.38 | 242.2 | 1.30 | 443.8 |
| 1532 | 3.60e9 | 31.46 | 114.3 | 7.76 | 463.5 |
| 2448 | 1.47e10 | 151.56 | 96.8 | 34.19 | 429.1 |
| 3184 | 3.23e10 | 431.23 | 74.9 | **76.83** | **420.2** |

**A realistic reactor mesh now costs about a minute, and the cube is the whole story** — doubling
the mesh is eight times the build. 3184 facets is 77 s, down from 7.2 min. ⚠️ A second "before" run
of the same scene read 335.5 s at 3184 facets (in the silhouette comparison, below), so the gain
there is **4.4-5.6x** depending on which before is taken; quote the smaller. The blocked-pair counts
of the geometry-independence check (463,936 and 174,912) are identical before and after, so the
whole build's answer is unchanged, not only the unit fixtures'.

**Throughput is now flat in the ray count** — holding the triangle count at 1532 and sweeping the
rays at a fixed test count per call:

| rays | before Mtest/s | after Mtest/s |
|---|---|---|
| 50,000 | 424.8 | 464.1 |
| 200,000 | 334.9 | 465.3 |
| 800,000 | 262.4 | 465.6 |
| 3,200,000 | 116.5 | 465.6 |

**Cause, found and fixed: the call shape, not the caller's broadcast.** The record left this open,
naming the caller's flattened outer product (~810 MB of broadcast origins and targets at 3184
facets) as the obvious suspect while noting the sweep ruled it out as the whole story. It was
`segment_is_cut` cutting the rays first and leaving a one-triangle block; see the `work_limit`
section above. The broadcast is still there and still costs memory, but with the call shape fixed
the whole build reaches the kernel's own throughput, so it is no longer where the time goes.

**The cost is geometry-independent, which is why one ladder settles it for every scene.** The
pass tests every ray against every triangle with no early exit, so what the rays *hit* cannot
change what it costs. Checked rather than asserted: the sleeve scene and an otherwise identical
scene with the sleeve moved outside the box differ by **2.65x in blocked pairs** (463,936 against
174,912) and by **1.11x in wall clock**, inside the ~20% spread this machine carries. A future
change that gave the pass an early exit would break this, and the check is what would notice.

## ANALYTIC OCCLUSION: BUILT as `SilhouetteOcclusion` — exact per blocker, once six defects were out

Neither sampling treatment above moves the worst pair, because both sample a step function. The
clip does not sample: it cuts the source's angular extent by the blocker's silhouette and takes the
covered share in closed form — the classical analytic form factor, Nishita and Nakamae (1983), then
Baum, Rushmeier and Winget (*Computer Graphics* 23(3), 1989). It is **selectable beside the ray
mask, not a replacement for it** (`RadiationSettings(self_occlusion=SilhouetteOcclusion())`), and
the ray mask stays the default: neither dominates, because the clip over-counts overlapping
silhouettes (below) and costs more.

**What makes it expressible as a traced program**, each property load-bearing:

- **The contour form is signed and additive over loops** (`_signed_loop_solid_angle`; the magnitude
  is taken only at the end of `projected_solid_angle`). A triangle split four ways sums to the whole
  at 0.00e+00, so the visible region is *whole minus covered* and is never constructed.
- **The covered region is convex, so its vertex count is static.** Cutting by one half-space adds
  at most one vertex; compaction is by a **rank** (the running survivor count), so widths run
  `4, 5, 6, 7, 8` — the blocker is first cut to the near side of the source's plane, which can
  make it a quadrilateral and adds the fourth edge.
- **Direction space**, not the source's plane: no perspective divide, no infinity for a straddler.
- **Every sign read is decidable** — `clipping.py`, below. Without it the answer depended on the
  compiler.

**Two measured facts make STL geometry work:** a tiling sums exactly (a blocker split 4/16/64 ways
matches the whole at 1e-16 — a triangulated surface *is* a tiling), and **only front-facing
triangles are summed** (on a closed 384-triangle tube, summing every triangle gives exactly 2.0 —
the far wall counted too; front-facing gives 1.33e-15 against dense truth).

⚠️ **EVERY RECEIVER MUST SIT ON A FACET.** The fraction is of a *projected* solid angle, which needs
the receiver's normal; a volume point has none, and the unprojected measure a volume gather uses is
a different quantity. `SilhouetteOcclusion.field` raises for `receiver_facet=None` and for any
`-1`, naming `RayCastOcclusion` — an error, because silently falling back inside the strategy
would be the plausible brighter field this subsystem keeps warning about. The model does the
routing instead, openly: see `receiver_occlusion` above.

⚠️ **WHERE IT OVER-COUNTS, STATED EXACTLY: the angular overlap between two front-facing
silhouettes.** Each blocker is clipped against the *source*, not against what is still unblocked,
so areas add: `0.54 + 0.54 = 1.08` where a ray's `OR` is idempotent (measured, two blockers in one
cone: 0.5415 each, true union 0.5421, sum 1.083). It errs **dark**, and only here:

| geometry | front-facing crossings | result |
|---|---|---|
| one sleeve, baffle or wall between two facets | 1 | **exact** |
| **bent duct / elbow** | 1 (leaves the fluid, re-enters) | **exact** |
| convex vessel, no internals | 0 | **exact**, nothing blocks |
| sleeves side by side, cones disjoint | >=2, no overlap | **exact** |
| **multi-lamp bundle, one sleeve behind another** | >=2, overlapping | **over-counts, errs dark** |
| serpentine channel, sight line across two walls | >=2 | over-counts |

The exact repair is progressive, depth-sorted clipping against the remaining unblocked region — the
hidden-surface algorithm, whose per-pair vertex count is not static. The field reports
`overlapping` (more than one blocker contributed, so areas were *added* and **may** be double
counted; a tiling of one wall adds without overlapping, so this is "possibly", honestly) and a
count of one proves a pair exact. Nearly free: it is a count in the pass that already runs.

### ⚠️ "IT IS EXACT" WAS MEASURED ON ISOLATED, NON-DEGENERATE PAIRS — ON A MESH, SIX DEFECTS HID THERE

The prototype's sweep called the method exact, and per blocker it now is — but on a real mesh
degenerate configurations are the **norm** (shared vertices and edges, coplanar triangles of one
wall, a receiver in the plane of every triangle on its facet), and every defect below produced a
**confident wrong number, not a failure**. Each is pinned by a test that was mutation-checked.

1. **No depth cut.** The clip has no notion of depth, so a blocker straddling the source's plane
   occluded with all of itself: **1.0000 against a sampled 0.6396**. Fixed by cutting the blocker to
   the near side of the source's supporting plane first (the extra clip stage).
2. **The cull's source-plane test omitted its offset**, testing a parallel plane through the
   receiver — which rejects blockers *in front of* the source. Not merely optimistic but
   unconservative, and it is what the prototype's **6.14% workload** figure measured; the correct
   figure was 16.3% (12.6% with front-facing only). The same bug was written a second time when
   building this — caught only because a blocker known to block fully came back rejected.
   `decidable_heights(..., through=)` is the one place the offset lives now.
3. **A degenerate depth cut inverted the no-op.** A cut to a sliver makes every edge plane zero, so
   no stage removes anything and the source reads *wholly covered*: **1.0 against a sampled 0.0**.
   Fixed by clamping the covered part to the blocker's own projected extent, which holds
   geometrically and is a no-op wherever the clip was already right.
4. **The cone cull dropped 4,941 real occluders** on a 480-facet reactor — degenerate axes, and caps
   at or past a right angle (a receiver near the triangle's plane). `angular_cone` flags those
   `unusable`, and **an unusable cone overlaps everything**. The cull's invariant is asserted
   exhaustively (`DROPPED == 0` over every pair of a closed box), not argued.
5. **Coplanar and edge-on blockers** have every height or the triple product mathematically zero;
   before the `in_front` / `edge_on` guards, 119 of 2304 pairs of a closed box moved with the chunk
   size alone, one by the whole of its value.
6. **A source seen edge-on reported a ratio of two roundings.** Its projected solid angle is dust
   (~2e-16), the covered part is dust too, and their quotient is anything in `[0, 1]`: **13
   coplanar sleeve pairs read 0.06 to 1.0**, where the truth is that they exchange no light.
   `_EXTENT_FLOOR` (1e-12 sr, far above the contour sum's dust of a few parts in 1e15 of pi).
   ⚠️ This one was latent — the old noisy clip happened to collapse such loops — and surfaced only
   when the filters below made the coplanar loop *correct*. **A fix that makes an intermediate right
   can expose a downstream quantity that was only ever right by accident.** A clean hand-built
   fixture lands on an exact 0.0, which the old `== 0` guard handled, so the test is a property sweep
   over the reactor's coplanar pairs rather than a placed triangle.

### ⚠️ THE SIGN TESTS: TWO FILTERS, EACH LOAD-BEARING, AND THE SMALL FIXTURE COULD NOT TELL

After the six, 4 of the 80-facet reactor's 6400 pairs still moved with the chunk size, by up to
0.21. The cause: a height zero in exact arithmetic computes as ~1e-19 of noise whose **sign is set by
how XLA fuses the multiplies and adds** — which changes with batch shape, with whether the code is
compiled at all, and **with the presence of an arithmetically dead term**: adding `0.0 * bound` to
the unfiltered return made a lost occluder reappear in every context. Without the filters, whether
a real shadow is found depends on incidental compiler decisions.

`clipping.py` holds two filters, each the first stage of a Shewchuk (1997) filtered predicate
(bound the rounding error from the magnitudes summed, `_SLACK = 16` units of roundoff; trust the
sign only outside it):

- **`decidable_heights`** snaps an undecidable height to **exactly zero** — and **no exact
  arithmetic stage is needed**, which is what makes it cheap: zero is a *correct and consistent*
  answer for a clip (the kept half-space is closed; an edge with a zero at one end does not cross,
  its crossing *is* that endpoint). A triangulation must pick a side; a clip need not.
- **`spanning_plane`** snaps a plane through two directions indistinguishable in direction to the
  zero vector, making that stage the no-op it should be. It replaced an exact-equality test on the
  two corners (`corner[k] == corner[k+1]`), which cannot see a crossing computed at the end of its
  edge as `a + 1.0 * (b - a)` — `b` to within a rounding, not bit for bit.

**Measured truth table** — sleeved box `inward_box(5)` + `closed_drum(16, radius=0.15,
half_height=0.3)` at its centre, **364 facets**, receivers at facet centroids, every blocker of the
worst pairs adjudicated against the brute-force sampler at 200k samples; JAX 0.10.2, CPU, x64,
macOS arm64, 11 cores, 2026-09-21:

| heights | repeat guard | chunk-invariant | wrong pairs | failure |
|---|---|---|---|---|
| raw | exact-equality mask | **no** | — | the starting point |
| filtered | exact-equality mask | yes | **25** | a triangle covering 0.37 of a source read as covering nothing |
| raw | `spanning_plane` | yes | **1** | a blocker covering 0.0020 read that eagerly and **0.0 under `jit`** |
| filtered | `spanning_plane` | yes | **0** | every blocker matches the sampler |

⚠️ **At 80 and at 156 facets all three filtered rows were identical** — same occluded-pair counts,
same sums, chunk-invariant. A fixture too small to contain the degeneracy says nothing about which
fix can go; this project nearly deleted the height filter as "dominated" on that evidence. **And
chunk invariance was the wrong acceptance test on its own**: every row but the first passes it,
because each compiled shape gets the *same* wrong answer. Only the sampler, and an eager-vs-`jit`
comparison, expose the wrong rows. Both are now tests on the 364-facet body
(`test_a_real_occluder_is_not_emptied_by_the_plane_through_a_near_repeated_corner`,
`test_a_real_occluder_reads_the_same_whether_or_not_the_clip_is_compiled`).

**Per call site** (clean mutations — never a `+ 0.0 * x` keep-alive, which perturbs the fusion under
test and here *rescued* the bug): the height filter is individually load-bearing only at the
**edge-plane** cuts of the source, where a sign selects which part of the source every later cut
sees. At the source-view and extent cuts the loop goes straight into the contour integral and its
edges never become planes, so a near-repeat adds a zero-length edge worth nothing; at the depth cut
the plane filter repairs what a flip produces. Those three are recorded as dismissed, not gaps, and
the filter is **kept uniform anyway**: stripping it would create a second, unfiltered way to make
heights, and uniformity is what protects the next call site by default.

⚠️ **The ray test's remedy — the antisymmetric edge function — is NOT needed here, and was deleted.**
Under `jit`, `jnp.cross(v, v)` is exactly zero (the nonzero value appears only eagerly), and with the
two filters in place swapping the antisymmetric cross for `jnp.cross` changed no bit on either
reactor; its test could not fail. The difference in kind: the ray test *consumes* the edge function's
sign directly, while here every sign consumed is a filtered height. Protecting the consumed predicate
is stronger than making each quantity that feeds it reproducible. (It stays load-bearing in
`triangles.py`, above.)

What remains between chunk sizes is **~1e-13**, identical between chunks 1000 and 262144 and between
3, 7 and 64 — two compiled behaviours of the continuous contour sum, not a sign flip. The chunk test
holds it to 1e-9.

**The shipped `projected_solid_angle` now clips through `clipping.py` too** — its private
Sutherland–Hodgman (6 slots, repeat-last) was a second copy of the same algorithm. It moved by at
most **8.9e-16 sr per pair and 1.8e-15 sr per row sum** (80- and 364-facet enclosures facet to facet,
and 200k random straddling triangles against the `a6ec6a2` version); not bit-identical (20-63% of
values equal), no test pinned the old bits — the same category as the `dot` rewrite in `CLAUDE.md`.

⚠️ **`validation/radiation_analytic_occlusion.py` still carries the PROTOTYPE clip** (its doubling
and rank arms): no depth cut, no filters, no extent clamp. Its arm-against-arm throughput comparison
stands; its "exact" conclusion does not transfer to a mesh, and it is not the production path.

**`dG/d(occluder geometry)` is still exactly zero** with either strategy: the fraction is computed
on the host and frozen, off the tape. A continuous fraction *could* make baffle and sleeve placement
differentiable, but only by putting the clip live on the tape — not built, and its cost would be the
whole build per gradient.

### Cost, MEASURED end to end — not projected

Both strategies timed by `validation/radiation_mask_build_cost.py` (`silhouette_ladder`) on the same
box-plus-sleeve reactor, same receivers (facet centroids), same run, median of warmed builds (three;
two above 2000 facets). JAX 0.10.2, CPU, x64, macOS arm64, 11 cores, 19 GB, nothing else running,
default `work_limit` and `work_chunk`. Two runs, and **the ratio moved because the denominator did**:
the ray mask's call shape was fixed between them (triangles first; see the `work_limit` section).

| facets | ray, rays-first (09-20) | ray, triangles-first (09-21) | silhouette (09-20 / 09-21) | ratio now |
|---|---|---|---|---|
| 224 | 0.04 | 0.03 | 1.85 / 1.78 | 66.7x |
| 480 | 0.36 | 0.24 | 10.75 / 8.17 | 33.4x |
| 832 | 7.10 | 1.30 | 63.58 / **27.70** | 21.3x |
| 1,532 | 30.80 | 7.76 | 115.65 / 108.98 | 14.0x |
| 2,448 | 154.81 | 34.55 | 390.38 / 375.10 | 10.9x |
| 3,184 | 335.50 | 75.30 | 647.67 / **638.77 (11 min)** | **8.5x** |

**The silhouette strategy costs about 11 minutes at 3184 facets, and 8.5x the ray mask.** It was
recorded as 1.9x for one day, against a ray mask that was four times slower than it had to be; that
figure is dead. The clip itself did not change between the runs and reproduces to 1-6% at every
rung but one. ⚠️ **At 832 facets it read 63.58 s one day and 27.70 s the next, 2.3x apart, and that
is unexplained** — far outside this machine's ~20% spread, and not seen at any other rung. Do not
quote the 832 clip figure until it has been re-run.

**The ratio still falls with the mesh**, for the reason it always did: the ray mask is a dense
`n^3`, while the clip's cone cull discards a share of candidates that grows as the mesh refines. It
now falls from 67x to 8.5x rather than to 1.9x, because the ray mask no longer degrades with it.

⚠️ **EVERY EARLIER COST FIGURE FOR THIS METHOD WAS WRONG, AND ALL IN THE SAME WAY — each timed or
projected one piece rather than the whole build.** The record's first headline was 15-20x (a ray
test timed at 200,000 items, four orders below a build); an early projection said ~1.5x (two wrong
factors that cancelled); the corrected projection said ~4x / 27 min (a 6.14% workload from the
unconservative cull, times a per-item rate extrapolated 600x); the cone cull's projection said
1.6x / 12 min (before bucketing, the guards and both filters); and the first measured answer, 1.9x /
11 min, divided by a ray mask that was itself four times slower than it needed to be. The clip's
11 minutes has held; every *ratio* has not. **A ratio inherits every defect in its denominator — time
the whole build, both arms, at the size the answer is for, and re-time the ratio whenever either arm
changes.**

**Power-of-two chunk padding** (`_bucket`): padding every chunk to `work_chunk` compiled one program
and clipped a quarter of a million pairs for a receiver with sixty candidates; padding to the next
power of two wastes at most half a chunk and compiles a couple of dozen shapes. It took the
silhouette tests from 110 s to 19 s and is in every figure above.
