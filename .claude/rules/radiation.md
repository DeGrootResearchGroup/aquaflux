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
| `coarsen.py` — coarsening a dense surface (a mesh patch) by edge collapse, under longest-edge, chord and angle bounds | **BUILT** |
| `profiles.py` — `Profile` (asked about a direction and a facet normal), `AxisymmetricProfile` (the angle alone: `Isotropic`, `Lambertian`, `CosinePower`) | **BUILT** |
| `photometry.py` — `read_ies`, `Photometry`, `PhotometricProfile`: a measured luminaire's IES LM-63 table over both angles (2026-09-28) | **BUILT** (direct gathers only; the transfer refuses it) |
| `gather.py` — `direct_fluence_rate` and `direct_irradiance` | **BUILT** |
| `absorption.py` — `UniformAbsorption`, `VoxelAbsorption` | **BUILT** |
| the solid bodies (`Body`, primitives, CSG, `Outside`) — **moved to `aquaflux/solids/`**, see `.claude/rules/solids.md` | **BUILT** |
| `visibility.py` — the frozen shadow mask | **BUILT** |
| `culling.py` — how the bodies' layer is decided: `ShaftCulling` (the default since 2026-09-26; tiles certified clear, refined coarse to fine, #554) or `EveryPair` (the reference) | **BUILT** ("clear" certificates only; analytic and triangle bodies) |
| `back_faces.py` — `BackFaces`: which volume receivers lie certainly behind which facets, pair by pair (the ray mask's cut, #526) or a tile at a time from a box (the bodies' layer skips such pairs, 2026-09-27) | **BUILT** |
| `lit_blocks.py` — the gather's layout: receivers in blocks of 8 along the curve, each against the facets its box is not proven to lie behind, in segments padded to a short width ladder (2026-09-27) | **BUILT** |
| `triangle_body.py` — `TriangleBody`, a `Body` of triangles over a `TriangleGrid`, with `contains` decided per piece (#510) | **BUILT** |
| `triangles.py` — watertight ray-triangle intersection | **BUILT** |
| `grid_walk.py` — the triangle grid's walk: each ray walked to its first hit by one compiled (Numba) loop | **BUILT** (the only walk since 2026-09-26) |
| `self_occlusion.py` — the `SelfOcclusion` strategies: ray cast, silhouette clip, none | **BUILT** |
| `silhouette.py` — the exact covered fraction of a source, the conservative cone cull, and the exact second-stage rejection (`covers_nothing`) | **BUILT** |
| `clusters.py` — `FacetClusters`, facets grouped along their Morton order for the silhouette's clustered cull | **BUILT** |
| `clipping.py` — convex clipping with filtered (decidable) sign tests, shared with `solid_angle.py` | **BUILT** |
| `transfer.py` — the frozen facet-to-facet geometry | **BUILT** |
| `quadrature.py` — symmetric triangle rules for the receiving facet | **BUILT** |
| `model.py` — the assembled model and the three public entry points | **BUILT** |
| `images.py` — `mirrored_fluence_rate` / `summed_mirrored_fluence_rate`: one specular bounce into the volume, each source's image seen through each mirror's aperture (#537 PR 2, 2026-10-05) | **BUILT** (exported and wired into the model by PR 3a, which adds `mirrored_irradiance` and `plane_exchange`; shadowed through `shadows=` since PR 3b) |
| `mirrors.py` — `Mirror` (a plane, its aperture facets, reflection and the image of a surface set) and `planar_mirrors` (a body's facets grouped by plane); the reflectance split on `Surfaces` (#537 PR 1, 2026-10-05) | **BUILT** (the model carries specular bodies since PR 3a, shadowed since PR 3b) |
| `mirror_visibility.py` — `MirrorVisibility` / `build_mirror_visibility` / `build_mirror_masks`: what stands across each path reflected in one mirror, both legs, per body counted 0/1/2 (#537 PR 3b) | **BUILT** |
| `refraction.py` — `Transparent` / `Media` (convex regions with an index and an absorption, nested; `Media.region_of_facets`), `fresnel_transmittance`, `Chain`, `solve_paths` -> `Paths`: the transmitted path between two points by a descent on the optical length (#604 step 2a, 2026-10-08); `straight_through`, the straight segment's factor for a mask (#604 step 2b-i) | **BUILT** |
| `refracted.py` — `refracted_fluence_rate` / `refracted_irradiance` (each triangle's solid angle on its corners' arrival directions) and `build_refracted_visibility` -> `RefractedVisibility` (every leg of the centroid's path; `receiver_facet` excluded on the last leg since 2b-i) (#604 step 2a) | **BUILT** (in the scene's direct light since 2b-i; not yet in the model or the transfer — 2b-ii) |
| `units.py` — lamp watts to exitance, ultraviolet transmittance to absorbance | **BUILT** |
| `scene.py` — `Scene` / `solve_scene` / `SceneSolution`: lamps' EMISSION kept out of the transfer (any profile, incl. IES) while their areal facets EXCHANGE reflected light (reflect by `diffuse_reflectance`, shadow, absorb; #604 step 1, 2026-10-07), reflecting surfaces, bodies, medium, `VolumeReceivers` and named `SurfaceReceivers`; one `settings` whose `two_sided` may name a body of the lamps or the reflectors (checked once against both, cut per build by `settings_for`, 2026-10-06); what a radiation case file builds (2026-10-05) | **BUILT** |

There is no separate optical-depth piece to build: the voxel-grid traversal is `VoxelAbsorption` in
`absorption.py`, exact along each segment (trilinear field, Simpson per cell).


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
   ⚠️ **It recurred three more times (#620, 2026-10-07), each with a correct forward value and a
   NaN gradient, and none was seen because no test differentiated in POSITION with any profile but
   Lambertian.** (a) `CosinePower.radiance_per_exitance_at` raised `max(cos, 0)` to `n - 1`, whose
   slope at a zero base is infinite for `n` in `[1, 2)` — every receiver behind a facet; now
   `profiles._forward_power` substitutes the base (`where(cos > 0, cos, 1)`) inside the power, for
   both views. (b) `UniformAbsorption` and `VoxelAbsorption` took `sqrt(dot(offset, offset))`
   unguarded — a receiver at a facet's own centroid in a medium (vacuum was fine); both now use
   `aquaflux.vectors.norm`, the zero-safe magnitude (moved there from `mesh/face.py`'s private
   `_safe_magnitude`, so the guard has one home). (c) `PhotometricProfile.angles` took
   `arctan2(0, 0)` on a facet's axis — every receiver straight below a ceiling lamp; the operands
   are substituted with `(0, 1)` where the in-plane part is zero. **The generalization: the same
   defect arrives through `**`, `arctan2`, `log` — any primitive with an infinite or `0/0` slope
   where the selection discards its value.** ⚠️ **A Lambertian fixture cannot see a NaN direction**:
   its radiance is a constant that never reads the direction, so `gather._emitter_direction`'s guard
   (receiver at a centroid) survived a mutation until the centroid test ran a `CosinePower` too.
   Pinned by `test_a_source_can_be_moved_under_a_gradient_whatever_it_emits_with` (every areal
   profile, a receiver behind a facet and one on another's axis, against a finite difference),
   `test_a_receiver_at_a_facet_s_own_centroid_takes_nothing_from_it_and_has_a_finite_gradient`
   (vacuum / uniform / voxel x Lambertian / cosine: field and point gradient equal the other facet's
   alone), `test_a_receiver_on_a_facet_s_axis_has_a_finite_position_gradient` (FD only ALONG the
   axis — the bilinear table in polar angles is a cone at the pole, so no other derivative exists
   there), `test_a_direction_behind_the_facet_has_a_zero_derivative_not_a_nan` and
   `test_a_segment_of_zero_length_has_zero_depth_and_a_finite_gradient`. Mutation-checked: each of
   the three fixes reverted, `norm` guarded after the root, only one `arctan2` operand substituted,
   and `_emitter_direction` unguarded — all red.

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

⚠️ **The same convention binds the GATHER at a receiver lying ON an emitting facet**, and there it is
honoured by `direct_irradiance(..., receiver_facet=)` leaving those facets out — not by the kernel and
not by the emitter's cosine gate, both of which read rounding noise there off an axis-aligned plane
(measured in "THE SCENE" below).

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

**Each profile supplies `intensity_fraction(direction, normal)` and
`radiance_per_exitance(direction, normal)`, and the pair must satisfy
`radiance_per_exitance * dot(d, n) == intensity_fraction`.** The axisymmetric ones implement them
from `intensity_fraction_at(c)` / `radiance_per_exitance_at(c)`, the forms the transfer calls on its
frozen `source_cosine`. The second is not derived
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

## A measured luminaire is a table over TWO angles, so a profile is asked about a DIRECTION (2026-09-28)

`photometry.py` reads an IES LM-63 file (Type C, `TILT=NONE`) — the photometry of the Ushio Care222 B1
module (`validation/ray_effects_room/ushio_b1.ies`, OSLUV-measured) is the first consumer. Its
intensity varies by up to **8 % of the peak** round the axis, so no function of `cos gamma` describes
it, and the `Profile` methods changed from `(cos_theta)` to `(direction, normal)`. What is load-bearing:

- **The frame is the facet's, and it must match of-optical-radiation's `iesEmitter`.** `gamma` is from
  the facet's outward normal; `h` from `up` towards `normal x up`. That is OOR's `e2 = fixtureAxis x
  fixtureUp`, with the normal playing `fixtureAxis` — so a ceiling window facing down with `up = +x`
  reads the same table direction as OOR's BC with `fixtureAxis (0 0 -1)`, `fixtureUp (1 0 0)`.
  `test_the_horizontal_angle_runs_towards_normal_cross_up` is the port of OOR's `iesHframeOrientation`
  test. `up`'s own component along the normal drops out exactly (the in-plane part is normal to the
  facet and `atan2` is scale-free), so it is neither projected nor normalized; a mutation removing a
  projection there is **equivalent**, not a gap. An `up` parallel to a facet's normal leaves `h`
  undefined: `Profile.refuse_normals`, asked by `check_profiles`.
- **Interpolation and symmetry are OOR's**: bilinear in degrees, `gamma` clamped (not extrapolated),
  quadrant / bilateral / full tables folded by LM-63's rules. The table is expanded round the circle
  once, at `Photometry.profile`, so the traced code has no symmetry branch. Pinned against an
  independent numpy reading, and cross-checked against the ray-effects reference's own parser to
  **1.4e-15 of the peak** over 20,000 random directions.
- **Normalization is closed form and over the FRONT HEMISPHERE** — `front_hemisphere_flux`, the exact
  integral of the bilinear table times `sin gamma` over `[0, 90]` degrees, clamped. That is what a flat
  emitter can radiate, and what OOR's BC normalizes over (its outgoing rays), so both emit exactly `P`.
  `Photometry.flux` is the table's own range and is the lamp power when the file is absolute: the B1
  file integrates to **118.826 mW** (a trapezoid estimate gave 118.6 and was quadrature error).
- **The radiance is floored at `MIN_COSINE = 1e-3`** (OOR's `IES_MIN_COS`), since a table with light
  at `gamma = 90` would otherwise need unbounded radiance along the facet's plane.
- ⚠️ **The measured table disagrees with itself at the pole**: `I(0, h)` is 121.38 to 121.77 mW/sr
  across `h`. A point on the axis reads `h = 0` (as OOR's `hDegFromDir_` does), and within a fraction of
  a degree of the axis two neighbouring facets read different rows — a 0.18 % effect that a test at
  `gamma = 0.1 deg` once read as a code error. It is the data.
- **The transfer refuses it** (`NotImplementedError`): it freezes one `source_cosine` per pair, and an
  azimuthal table needs the direction. Only the *emitted* transfer is affected — reflected light leaves
  Lambertian — so this matters only for a reflecting room lit by an IES lamp. **And even there it
  need not matter: keep the lamp out of the surface set.** Gather its direct irradiance on the
  reflecting facets with `direct_irradiance` (which takes any profile), pass that as
  `external_irradiance` to `radiosity` / `fluence_rate`, and the transfer only ever carries
  Lambertian reflected light. **This is now the library's `solve_scene` (`scene.py`, section at the end
  of this file)**, which a radiation case file builds; the script that first did it for a Care222-lit
  room (`aquaflux_reflecting.py`, deleted 2026-10-05; 2,000 squares of 20 cm) found the two routes to
  its slice field -- the library's `fluence_rate` and a direct gather of the facets' radiosity -- agree
  to 0.0, which is why the scene takes the second route alone. Average the lamp's
  irradiance over sub-points of each facet, not its centroid, or a shadow on a coarse facet is
  point-sampled. ⚠️ **A lamp that must reflect does NOT need the direction frozen**: since #604 step 1
  the scene keeps the lamp's EMISSION out of the transfer and puts its FACETS in, as reflectors with no
  emission (THE SCENE section, end of this file). Freezing the unit direction (3x the frozen array)
  would only be needed to carry a measured profile's *emitted* light through the transfer itself.
- **The gather passes the unit direction, not the cosine, at no measured cost**: 2,000 Lambertian facets
  x 40,000 receivers, fluence 0.549-0.557 s and irradiance 11.77-11.82 s against `main`'s 0.557 /
  11.82 s, checksums equal to 13 figures (CPU, x64, jax 0.10.2, macOS arm64, separate processes).

## SPECULAR REFLECTION: THE REFLECTANCE SPLIT AND THE MIRRORS (#537 PR 1, 2026-10-05)

The plan, agreed with the user and posted on #537 (comment of 2026-10-05): image sources seen through
**planar** mirror apertures, one bounce, in four PRs — (1) the optical split and the mirror geometry,
no result changes; (2) images in the volume gather, unshadowed; (3) the transfer's specular exchange
terms, the third gathered set and two-leg shadows, which is where specular becomes reachable; (4)
validation, cost and docs. **Decided with the user**: a mirror is **one plane, not one body**; curved
specular bodies (a lamp sleeve's exterior, a round vessel) are **out of #537 and moved to #604** — an
exact path solve on the analytic `solids` body, shared with refraction, because a faceted sleeve is
~100 planar strips (the Sozzi lamp's OpenCASCADE tessellation: 100 full-length strips, kept coplanar
by the CAD reader's cutting planes) and at one dense `n x n` transfer term per mirror that is ~45 GB,
and because a sleeve reflects mostly at its *inner* quartz-air face, with total internal reflection
above ~47 deg incidence in water, which no outer-surface `rho_s` represents (those two figures are
estimates from recorded numbers, not measurements). Store the incidence cosine per image pair in PR 3,
so an angle-dependent (Fresnel) `rho_s` is a later strategy. **Still open, to decide before PR 3**:
the transfer's storage — `rho_s` live per declared solid (planes summed into one array) or live per
facet with a sparse store.

**What PR 1 built.**
- `Surfaces.reflectance` is **gone**: it is `diffuse_reflectance`, beside a new `specular_reflectance`
  (default 0), both live leaves, on `from_triangles`, `with_optics`, `with_geometry`, `coarsen_surfaces`
  and `refine_for_receivers`. ⚠️ **There is no `reflectance=` keyword any more**; an old call fails with
  a `TypeError`. `_check_reflectances` refuses each outside `[0, 1]` and a facet whose two sum past one
  (concrete values only, numpy, skipped for tracers — the same rule as `in_range`), from both
  `from_triangles` and `with_optics` (which checks the pair as it stands after the replacement).
- **PR 1's model refused all specular reflectance; since PR 3a it is `transfer._specular_by_solid`
  (there is no `model._without_specular` any more)**, which keeps the two refusals that still apply:
  a non-zero value on a body not built as specular (`ValueError` now), and a **traced** `rho_s` on a
  model with no specular body (`TypeError`, whatever its value). ⚠️ **The traced refusal is
  load-bearing, and PR 1's first version did not have it**: it checked a traced value with
  `eqx.error_if`, which passed a zero and so let `jax.grad` with respect to `rho_s` return **0.0** —
  finite, plausible, and wrong, since a mirror sends light on. The direct gathers ignore both
  reflectances, as they always did the diffuse one.
- `vectors.reflect(vectors, normal)` is the one reflection formula (in the plane through the origin).
- `Profile.mirrored(normal)` is **abstract** on `Profile`: an `AxisymmetricProfile` returns itself; a
  `PhotometricProfile` reflects `up` **and flips `handedness`** (a new static field, default 1, read in
  `angles`). ⚠️ **Reflecting `up` alone is wrong**: a reflection reverses handedness, so the image's own
  `normal x up` is the *negated* reflection of the source's, and every table asymmetric in `h` reads
  turned the wrong way. Pinned on the `b1` fixture (`sin h` term) against an oblique plane.
- `Mirror(point, normal, facets)`: `reflect_points`, `reflect_directions`, and `image(surfaces)` —
  vertices reflected **and two corners swapped**, which together give the reflected outward normal
  (either alone points the image away from its viewer, and a dark-behind profile then sends nothing);
  profiles replaced by their `mirrored`; every optical value and label carried. `facets` is a numpy
  label array, like `profile_index`. `image` decides no visibility: which sources get an image (none
  in or behind the plane) is PR 2's.
- `planar_mirrors(surfaces, solids, *, tolerance=None)`: per named body, areal facets only, **largest
  facet first** seeds a plane; a facet joins when all three corners lie within `tolerance` of the seed's
  plane **and** it faces the same side (two faces of a thin sheet are two mirrors); the plane is then
  the area-weighted mean normal and centroid of its members. Default tolerance `1e-6` of the set's
  extent (`_RELATIVE_TOLERANCE`), above single-precision STL rounding (~6e-8) and below any real step.
  ⚠️ **Seeding by area is load-bearing**: a sliver in a wall can compute a normal ~89 deg off the wall;
  seeding from it splits the wall. ⚠️ **Not yet decided**: the plane-count threshold above which a
  declared specular body is refused (pointing at #604) — set in PR 4 from measured cost, enforced in PR 3.

**Tests**: `tests/unit/test_radiation_mirrors.py` (box = 6 mirrors of 8; a 12- and a 180-sector prism
is exactly one mirror per strip; float32 rounding keeps a plane and a 1e-3 step splits one; sheet
sides; per-body grouping and a misspelt name; point sources; the sliver seed; the image's direct field
at reflected points equals the source's at the points, 1e-12, over Lambertian, `CosinePower`, an
asymmetric `PhotometricProfile` and a point source; the image's carried properties and its involution),
plus the reflectance split, its validation and the model's refusals in the surface and model test files.
**Mutation pass (22, all red)**: winding not reversed, profiles not mirrored, handedness not flipped,
handedness ignored in `angles`, seeding by index, no orientation test, tolerance 1e-9 and 1e-2, bodies
grouped together, point sources grouped, an off-plane point, no sum check, no range check,
`with_optics` unchecked, `with_geometry` dropping `rho_s`, the concrete and the traced refusals each
removed, `reflect` with factor one, coarsening reading `rho_s` from `rho_d`, subdivision dropping it.

### PR 2: images in the volume gather (2026-10-05)

`images.py`, unshadowed, one bounce, not yet exported or reachable from the model. Per mirror, per
(receiver, source facet, aperture facet): `rho_s[aperture facet] x radiance (mirrored profile,
direction from the image centroid) x solid angle of the image inside the aperture facet's cone`, the
last being `silhouette.covered_by(view of the image, None, aperture facet) x |view.whole|` -- the aperture
is the blocker, and **its depth cut is what discards an image's part in front of the mirror**, so a
source straddling the mirror's plane needs no clipping of its own (pinned: a wall across the plane
equals its upper half alone). Point sources: the image point is credited to the **first** aperture
facet whose closed cone (filtered heights `>= 0`) holds the direction, so a line through a shared edge
counts once -- a symmetric lamp over a square mirror puts the reflection point exactly on its diagonal
(pinned). The medium is crossed on the **two real legs** (receiver -> crossing -> real source
centroid), which equals `exp(-a |x - c'|)` for a uniform medium and is the correct path for a graded
one (pinned against a closed-form leg integral of a linear field). Receivers and sources wholly on or
behind the plane are culled on the host when positions are concrete -- **cost only**: traced positions
skip the cull and give the same field (pinned), and the two cull mutations are dismissed as equivalent.
It reuses `gather`'s private `_areal_groups`, `_groups`, `_one_geometry`, `_emitter_direction` and
`silhouette._orientation`. ⚠️ **The clip stacks its candidates, so every operand must be broadcast to one
batch shape first** (`(receiver, image, aperture facet)`); left to broadcast it fails in `jnp.stack`.
Cost is dense in (receivers in front) x (sources in front) x (aperture facets) per mirror -- no cone
cull or `covers_nothing` yet; PR 4 measures it. Tests `tests/unit/test_radiation_images.py` (18):
point image closed form with and without a medium; aperture edge and outside; shared-edge diagonal;
per-facet reflectance; nothing behind; image wider than the mirror = the mirror's own solid angle
(`axial_rectangle_solid_angle`), image narrower = the whole image; an unbounded `rho_s = 1` mirror equals
the direct gather of `Mirror.image` over Lambertian, `CosinePower` and an asymmetric photometric table;
the straddling wall; linearity in the two facets' reflectances; the graded medium; summed sets; exact
derivatives in `rho_s` and emission; pass-size invariance; a foreign mirror refused; two mirrors add;
traced receivers. **Mutation pass (16, 14 red; the 2 green are the cost-only culls)**.

### PR 3a: the specular transfer and the model (2026-10-05)

**Decided with the user**: PR 3 split into **3a** (this: transfer + model, unshadowed scenes only) and
**3b** (two-leg shadows); `rho_s` **live per declared solid**; per-solid storage makes a multi-plane
solid's path length and source cosine **G-weighted means** per pair (exact for one plane); the
incidence cosine for a later Fresnel `rho_s` is **not stored yet** (one more weighted-mean array when
built -- this reverses PR 1's decision 3); a **graded medium is refused** with specular bodies.

- `build_radiation_model(..., specular=(names,))` and `build_transfer(..., specular=, max_mirror_planes=
  MAX_MIRROR_PLANES)`. `TransferMatrix` gains `specular_solids` (static), `mirrors` (every plane of
  every specular solid), `specular_geometric` / `specular_separation` / `specular_source_cosine`, each
  `(n_terms, n, n)` (per solid in 3a; per `(solid, crossing pattern)` since 3b -- see below):
  `images.plane_exchange` summed over the planes, and the two G-weighted means. `assemble` adds
  `rho_s[k] * G_k * exp(-a d_k)` (times the pattern's transmittance since 3b) to **both** `F` and `F^M` (the
  latter times the relative radiance at the mean cosine, through `_relative_radiance`, which the
  direct emitted term now shares).
- `transfer._specular_by_solid` reads one `rho_s` per solid from its first facet, refusing on concrete
  values a solid whose facets differ and a non-zero value on an undeclared solid; traced, it takes the
  first facet's (and refuses a traced value when no solid is specular).
- **Refused, not approximated** (`NotImplementedError`): a graded medium at `assemble`. `ValueError`
  past `MAX_MIRROR_PLANES` planes per solid -- 64 in 3a, provisional; **12 since PR 4**, from measured
  cost. (3a also refused occluders and self-occlusion with specular bodies; PR 3b lifted that.)
- The model's point-source arrivals add `images.mirrored_irradiance(..., point_sources_only=True)` at
  the facet centroids; the volume's mirrored field is gathered by the `ReceiverShadows` strategy since
  3b (it was a separate `summed_mirrored_fluence_rate` call in `fluence_rate` in 3a).
- ⚠️ The specular arrays' first axis was **per solid** in 3a; since 3b it is **per term** --
  `(solid, crossing pattern)`, listed in `specular_terms`. With nothing in the way there is still one
  term per solid, with an empty pattern, so `specular_geometric[0]` on an unshadowed one-solid transfer
  is unchanged.
- **Compile cost was the build's whole cost**: a fresh closure per plane (and per call in the gather)
  retraced everything, so a 12-facet box took 15-25 s to build and 2.9 s per repeated `fluence_rate`.
  `images._exchange_rows` (jitted, scanned over the quadrature points), `_areal_through` and
  `_points_through` (`eqx.filter_jit`, `_Path` now a pytree) are module level, so planes and calls
  sharing shapes share programs: rebuild 0.1 s, repeated `fluence_rate` 0.4 s (first build still
  ~10-13 s of compilation). Measured on a 4-core Linux container, jax 0.10.2, one run each.
- **Gates** (`tests/unit/test_radiation_specular_model.py`): the **symmetry plane** -- half a box closed
  by a `rho_s = 1` mirror against the whole box made by reflecting that half triangle for triangle --
  equal radiosity, facet irradiance and volume field to 1e-9 for Lambertian walls, under uniform
  absorption, with `CosinePower` emitters and with a point lamp; at the transfer level
  (`test_radiation_images.py`) the exchange equals the whole box's transfer to the other half to 5e-17.
  ⚠️ **`inward_box`'s right half is a TRANSLATE of its left, not a mirror image** (every quad cut on the
  same diagonal), so a symmetry check against it fails at ~3% and looks like a bug in the code; build
  the whole from the half by reflection. The **one-bounce enclosure** `B = M/(1 - rho_d(1 + rho_s))`,
  `G = 4B(1 + rho_s)` to 1e-12; reciprocity falling with the receiver rule; `rho_s` and the absorption
  coefficient against central differences; every refusal.

### PR 3b: shadows on both legs of a reflected path (2026-10-05)

**Decided with the user**: the transfer keeps the bodies' transmittance **live** by grouping each
specular solid's planes **by crossing pattern** (how many legs each body crosses, 0/1/2) -- one
`(n, n)` term per `(solid, pattern)` present, `prod tau_b ** legs_b` applied at `assemble` -- rather
than one array per plane (storage by plane count, what 3a avoided) or freezing `tau` on reflected paths
(severs the gradient). Under `SilhouetteOcclusion` the **reflected paths are rays** (direct pairs
keep the exact fraction): a two-leg path has no single source view to clip.

- `mirror_visibility.py`: `build_mirror_visibility(mirror, occluders, surfaces, points, *,
  receiver_facet, sources, self_occlusion, offset_scale, pair_limit)` -> `MirrorVisibility`
  (`crossings` uint8 `(n_occ, n_r, n_s)`, `hidden` bool or `None`, `receivers`, `sources` -- numpy
  facet labels, default `Mirror.sources_in_front`). One path per pair through the source **centroid**:
  meeting point where receiver -> centroid's image crosses the plane. `build_mirror_masks(mirrors, ...)`
  returns `None` when `shadows_reflected_paths` says nothing can shadow (no bodies **and**
  `NoOcclusion`; `None` strategy means the default, which shadows). `reflected_surviving` is
  `surviving_fraction(c >= 1) * surviving_fraction(c >= 2)`: the direct expression twice, and **finite
  derivative at tau = 0** where `tau ** 0` would give `0 * inf`.
- ⚠️ **The meeting point is LIFTED off the mirror** by `offset_scale * mean sqrt(aperture area)`
  along the normal, and **both legs end there**. A leg ending exactly in the plane reads as cut by the
  mirror triangle it lands on (a hit at distance 1 counts), so without the lift the mirror shadows every
  path reflected in it; lifting clears every coplanar triangle without listing the aperture as
  exclusions (which would be as wide as the aperture). Leg 1 runs source centroid -> lifted point with
  the source's own margin and excluding the source facet (the direct ray's convention); leg 2 lifted
  point -> receiver, excluding the receiver facet when there is one.
- `SelfOcclusion.segments_hidden(surfaces, origin, target, near, exclude)`: per-segment rays for legs
  that are no source-receiver product. Base = `segment_is_cut` (so `SilhouetteOcclusion` uses rays),
  `NoOcclusion` -> `None`, `RayCastOcclusion` -> its grid when set. **Body culling does not apply** to
  reflected paths -- every candidate path is tested (cost to measure in PR 4).
- Candidate paths: receivers strictly in front, sources with a part in front; a source whose
  **centroid** is not strictly in front (straddling the plane) is **recorded clear**, untested.
- Gathers: `mirrored_fluence_rate` / `summed_...` / `mirrored_irradiance` take `shadows=` (one mask
  per mirror, checked against the receivers) and `transmittance=` (refused without masks; default
  opaque). With masks, which receivers and sources a mirror involves is **read off the mask** (built
  from concrete geometry), so it is known when the sets' vertices are traced -- the streamed backward pass.
- Transfer: `_specular_exchange` builds the facet-receiver mask per plane (`receiver_facet=arange`),
  drops hidden pairs, and accumulates by pattern (`_by_pattern`: digits base 3, body 0 least
  significant). New fields `specular_terms` (static) and `point_shadows` (per-mirror masks restricted to
  point-source columns, for `model._point_source_irradiance`; `None` when unshadowed or no point source).
- Receivers: `FrozenShadows` holds `mirrors` and `mirror_visibility`; `StreamedShadows` builds each
  chunk's mirror masks inside the chunk's custom VJP through `streamed_fluence_rate(extra=...)` -- an
  object with `field(live, points)` -- so a gradient's memory stays one chunk. `fluence_rate` no longer
  gathers the mirrored field itself.
- **Gate**: the symmetry plane with a baffle of the walls' own triangles and a partly transmitting ball
  in the half, and both again (reflected) in the whole: radiosity and field to **1e-9**, Lambertian and
  point lamp, held and streamed. ⚠️ **The first baffle sat on the box's sixths and a reflected path
  between two wall centroids grazed its edge at exactly x = 0.35**: the half's leg and the whole's
  segment round that edge hit differently, a 0.8% mismatch that looked like a defect. Moved off the
  rational grid; the comparison is pair for pair identical (masks compared directly: crossings and
  hidden equal on all 26 x 26 cross pairs). Also: tau^2 through the transfer (pattern `(2, 1)`, two
  bodies with different tau), the transmittance derivative against central differences held and
  streamed, rays under `SilhouetteOcclusion` equal to `RayCastOcclusion`'s, and 16 mask unit tests.
- **Mutation pass, 22 breaks, 21 red.** Red: one factor instead of `tau ** legs`; either leg's body
  test or self test dropped; no lift; no receiver-facet exclusion; straddling sources tested; hidden
  pairs kept in the transfer; pattern digits reversed (caught only once the tau^2 test carried a second
  body of different transmittance); `assemble` ignoring the pattern; point arrivals, streamed or held
  mirrors unshadowed; either gather dropping the surviving fraction; transparent by default;
  `SilhouetteOcclusion` hiding nothing on legs; held transmittance dropped; the transfer's mask without
  `receiver_facet`. **Dismissed, equivalent**: no source-facet exclusion on leg 1 -- a segment meets its
  source's plane only at its origin, which the margin already discards (kept for the direct rays'
  convention). ⚠️ **Receivers behind the mirror being tested first passed**, because the fixture's
  receiver sat at the lamp's mirrored height, where the meeting point divides by zero and its NaN reads
  clear; moved to `z = -0.75`, it goes red. The gather never reads those rows, so the field is unchanged
  either way -- the test pins the documented "recorded clear".

### PR 4: what a mirror costs, the cone cull, and the plane limit (2026-10-06)

**Decided with the user**: the validation case is the analytic gates already in place --
**Hassanpour et al. (2023) is NOT a flat-mirror benchmark**: their reflector is **cylindrical** (lamps
outside the water channel) and their 10-47% specular/diffuse spread is a **discrete-ordinates
simulation**, not a measurement (the docstrings said "measure"; fixed). It moves to #604 as the curved
reflector case (full text read 2026-10-06, geometry and figures posted on #604: DO 6x6 only, no Monte
Carlo; R = 0.95 is the paper's own assumption, citing a vendor note; the 10-47% is diffuse-over-specular
LRV at 10 GPM as the lamps move 7.5 -> 15 cm from the channel). The cost case is the
Sozzi reactor from its drawing with the chamber's flat end plates specular, **without meshing**
(receivers uniform in the drawing's fluid -- per-item throughput does not depend on where they are).

- ⚠️ **COST IS SET BY THE APERTURE'S TRIANGLE COUNT, NOT THE PLANE COUNT.** Every image was clipped
  against every aperture facet: on the coarse scene (`specular_cost.py` defaults: chord 1e-3, lamp
  1,270 + wall 1,008 facets, 4 cores, jax 0.10.2, Linux) one plane of 28 triangles took ~50 min for the
  exchange (3e5 clip items/s) against 47 s for the whole direct transfer, and 72 s against 1.9 s for a
  400-receiver gather (39x). Two planes were already unusable; a plane limit could not fix it.
- **Survival share** (`aperture_cull_share.py`, same scene, 2,000 uniform volume points and the facet
  centroids): the silhouette clip's cone test keeps **11-12 / 28** (sliver triangles of a coarse disc
  fan, wide cones), **4 / 35**, **5-6 / 211** aperture triangles per image -- p99 23 at 211; 4-6% of
  images keep none; unusable cones under 0.13%. Survivors stay ~a handful however fine the plate.
- **The cull** (`images._screen` + `_culled_fractions`): the cone test per (receiver, image, facet);
  survivors clipped `_CLIP_BATCH = 4096` at a time in a `lax.while_loop` whose trip count follows the
  survivor count. A survivor's pair is found by **bisecting the running count of what each pair keeps**
  and its facet by its rank in the pair's row -- ⚠️ the first version took `jnp.nonzero` over every
  triple, which on 16 receivers x 2,700 images x 211 facets cost **0.25 s of a 0.45 s** culled clip:
  more than clipping the survivors. ⚠️ **A while_loop is not reverse-differentiable**, so the culled
  clip takes its geometry under `stop_gradient` and the live weights (reflectances) multiply
  `fraction @ weights` **outside** the loop. `cull` is on where the geometry is readable (`_mirrored`'s
  `readable`; always in `_exchange_rows`); the dense clip remains for a gather differentiated in
  position. The streamed model's chunk puts the build's concrete vertices back on its sets
  (`with_geometry`) so its backward pass, where every live leaf is traced, still culls.
- **The lit-block layout** (decided with the user: "port the direct culls"): the mirrored gather uses
  `lit_blocks.areal_layout` with a `BackFaces` of the *image* set, so receivers behind an image's plane
  are left out as in the direct gather, and the clip works on (receiver, image) pairs. ⚠️ Facet cones
  must be taken **once per receiver** in the block grid, not per pair: per pair made the 4,000-receiver
  gather slower (581 s against 450 s), per receiver 330 s.
- **THE INSIDE SKIP** (decided with the user: "build it in PR 4"). An image seen wholly inside the
  mirror needs no clip -- the mirror shows all of it, weighted by the facet the line to its centroid
  crosses (zero outside, or in a hole). Measured first (`aperture_inside_share.py`, 300 receivers,
  exact 2D classification): **98-99.5%** of seen images are inside at the 0.05 m plate, **90%** at
  0.01 m (211 triangles, 107 outline edges). `mirrors.outline` gives the edges of one facet only
  (rounded relative to the extent); `_Path.of` keeps them **only for one concrete reflectance on every
  facet** -- ⚠️ a traced one would credit an image's whole derivative to the facet its centroid line
  crosses (caught by the reflectance-gradient test: 12 of 32 facets wrong, sum right). `_screen` skips
  a pair when every corner is behind the plane **by 1e-9 of the coordinates** (`_behind`) and
  `_touches_outline` finds the image, projected into the plane along the lines from the receiver, apart
  from every outline edge by separating axes with the same margin.
  - ⚠️ **The outline CONE test came first and was too loose**: an image's circular cone overlaps edges
    the image never reaches -- it flagged 50.5% of pairs against an exact 8% partial (64 receivers,
    0.01 plate), so the gather went only 330 -> 246 s.
  - ⚠️ **The behind test needs the margin** because the plane is an area-weighted mean over its
    facets: the image of a facet IN the plane read heights of -3.3e-16, was skipped, and got 60% of its
    solid angle where the dense clip (correctly) gives 0. Found by comparing every probe pair with the
    dense clip -- 2 of 172,800.
  - ⚠️ **Write the separating-axis test corner by corner, the outline's edges the innermost axis**: the
    same arithmetic as reductions over trailing axes of 3 corners and 2 ends ran **16x slower** (0.29
    against 0.018 s on 16 receivers), the CPU-vectorization lesson of `silhouette`'s 2x2 tiles again.
- **MEASURED with all of the above** (`specular_cost.py`, `SOZZI_SPECULAR_PLATES=0.01`: chord 1e-3,
  lamp 1,270 + wall 1,008 + plates 422 facets, two planes of 211 aperture triangles, 4,000 receivers
  uniform in the drawing's fluid; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, nothing else running,
  one run, 2026-10-06): **exchange 333 s per plane** warm (601 s for the first, which pays the
  compilation) against **51.6 s for the whole direct transfer** -- ~6.5x per plane, from ~13 min;
  **mirrored gather 93.5 s for both planes** warm against **2.47 s direct** -- ~19x per plane, from
  ~90x (450 s) with the cull alone, 330 s with the lit-block layout, 246 s with the outline cone skip.
  A separate probe of the same gather read 86.0 s, and its field matched the pre-skip field to 1.1e-16
  of the peak. What is left is the clip of the survivors, ~4 us each; the reflected-path masks cost
  28-39 s per plane (two legs, no culling). ⚠️ The earlier "17x per plane" gather figure was from 400
  receivers and was deleted: at 4,000 the same code read ~90x.
- **`MAX_MIRROR_PLANES = 12`** (user's choice from 6 / 12 / 24 / 64): at ~19x the direct gather and
  ~6.5x the direct transfer per plane (above), twelve keeps a body's mirrored gather within ~230x the
  direct one; any box (6) passes; a 16-strip tube is refused. Pinned by
  `test_by_default_a_box_is_flat_enough_and_a_tessellated_tube_is_not` (limit 16 and 5 each red).
- Tests: culled == unculled (volume and oriented, `_CLIP_BATCH` monkeypatched to 7 so survivors take
  several passes, per-facet weights), to one rounding of the largest value (the unculled clip leaves
  last-bit noise on facets that show nothing); the cone test keeps every facet with a non-zero share and
  under 25% of a 200-triangle mirror; a test that keeps nothing gives nothing; reflectance gradients
  equal. The skip: an image crossing the plane is clipped however far inside it lies; one in the
  mirror's own plane is not skipped on a rounding of its height; the outline test flags every pair an
  independent classification calls crossing (whole and holed mirrors) and lets most wholly-inside ones
  through; only one concrete weight keeps an outline; the skip's answers stand with the clip stubbed
  to clip nothing. **Skip mutations, 13 of 15 red** (straddle, behind and its margin, skip-nothing,
  weight one, any weights, traced weights, all edges, the edge axis, the side test inverted, all for
  any, the rank off by one, bisect left). **Dismissed**: dropping the triangle-side axes (more pairs
  clipped, never a wrong skip), a non-orthogonal plane frame (touching is invariant under any linear
  map of the plane; only the 1e-9 margin's size moves); and, not run, clipping the skipped pairs anyway
  (their answer is overwritten, so only time could tell). **Cull mutations, 6/6 red** after two fixes: "clip everything" passed until the keep-nothing test,
  "padding unmasked" passed until the fixture put light through triple (0, 0, 0). **Not run, and
  cost-only by construction**: `cull=False` in `_mirrored` or `_exchange_rows`, and dropping the
  streamed chunk's geometry swap -- each only selects the unculled clip, which the equivalence test pins
  to the culled one, so the answers cannot differ; a test of them would have to time something.

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

⚠️ **It recurred in two more places, and between them they made the whole model untraceable (#522,
2026-09-24).** `Visibility.for_receivers` compared positions with `bool(jnp.all(points == ...))`, and
`with_optics(profiles=...)` — which the reflected pass of `fluence_rate` calls — stored the new index
through `jnp.asarray`, so the gather's partition received a tracer. So **no model entry point with a
mask could be jitted**, every call ran op by op, and the streamed path paid a fresh trace per pass. The
fixes are the rule above applied: the receiver check runs in numpy whenever both sets are concrete
(a model closed over by a compiled sweep still is, and is still checked) and goes by shape alone only
for traced points; and **`profile_index` is now stored as a numpy array** everywhere
(`from_triangles`, `with_optics`), as the label it is — like `point_source_index`. Pinned by
`test_the_field_compiles_with_the_model_closed_over`, `test_a_surface_set_rebuilt_inside_the_trace_still_compiles`
(`with_geometry` inside `jit` failed the same way) and `test_a_mask_is_still_checked_inside_a_compiled_function`.

## ⚠️ GEOMETRY IS CLOSED OVER, VALUES ARE PASSED

The gather partitions facets by angular distribution and by areal-versus-point **on the host**,
so each group's profile is a concrete object whose methods inline and the traced program holds
no branch on facet kind. That partition decides the program's *shape*, so `area` and
`profile_index` cannot themselves be traced. `jit(lambda s, p: direct_fluence_rate(s, p))` over a whole
`Surfaces` raises with an explanation; close over the set and substitute values through
`with_optics` instead (a `profile_index` that arrives traced this way is refused with the reason;
one built inside a trace from concrete input stays numpy and is fine). `RadiationModel` formalizes that boundary: it holds the frozen
geometry and every entry point takes the surface set again — for its optics, but its geometry is
read live too (the direct gather, the point-source arrivals), so it must be the build's; every call
checks (`RadiationModel.geometry`, below).

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
and no less accurate — absorbance varies far more smoothly than velocity. Since #528 that is its
cost in time and, under a gradient, in memory (a carry a step); its forward memory is flat in the
grid (see the #528 section).

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
body at production size, stored as bytes — plus the surface's own layer, which since #525 is
nothing under `NoOcclusion`, one byte a pair from the ray test, and a float (and a flag) only from
the silhouette clip; it was nine bytes a pair whatever produced it. Packing to bits is the obvious
eightfold saving if it ever matters. ⚠️ **At MESH scale it cannot be held at all**: 1.6M cells against 7,516 lamp facets
is **12 GB per body**, so a real case could not be built through the model. `direct_fluence_rate`
therefore also takes `occluders=` (and `self_occlusion=`) **instead of** a built mask: it then
builds each chunk's mask, gathers, and drops it, so peak memory is the chunk's (at most
`pair_limit` receiver-by-facet pairs, 1.5-1.9 GB of footprint at the 4M default — see the #509
section below) rather than the problem's. The two
are mutually exclusive and giving both raises — with both, the mask doing the work would silently
be the streamed one. **Streaming rebuilds the mask every call**, so a sweep over one frozen scene
still wants the model's frozen mask when it fits. The model streams too since #489 — see the
next section.

## A MODEL CAN STREAM ITS RECEIVER MASK — explicit, frozen by default (#489, 2026-09-24)

`RadiationSettings(stream_receiver_mask=True)` makes `build_radiation_model` keep the *recipe* for
the receiver mask instead of the mask: `RadiationModel.receiver_shadows` is a `ReceiverShadows`
strategy (`receiver_shadows.py`) — `FrozenShadows(visibility)` (the default; `.visibility` is the
mask, and there is **no `RadiationModel.receiver_visibility` any more**) or `StreamedShadows(geometry,
occluders, visibility_options)`. `fluence_rate` asks the strategy for the emitted and reflected fields
together, so **one mask per chunk serves both gathers** (building one per set doubled the dominant
cost). **Explicit, not automatic, by the user's decision (2026-09-24)**: which one to use trades
memory against repeated work, and only the caller knows which they can afford. ⚠️ **It is a memory
choice, not a differentiability one** — both paths give the same field and the same gradients.

**The gradient is bounded by the chunk too, and that is the non-obvious half.** Reverse mode holds
every chunk's receiver-by-facet intermediates until the backward pass — the whole problem's worth,
however small the chunks. `gather.streamed_fluence_rate` makes each chunk an `eqx.filter_custom_vjp`
that saves only its inputs and, in the backward pass, rebuilds its mask and recomputes its gather by
`eqx.filter_vjp`: a second mask build and a second gather per chunk, memory for one. It works
because the mask builds run **eagerly under `jax.grad`** — the geometry is concrete, so the host code
inside the rules is never traced (verified before building: host code in a custom VJP's forward and
backward rules gives the right gradient). ⚠️ **Not compilable with `jit`**, for the same reason; the
model's call path never was. `direct_fluence_rate(occluders=...)` now goes through the same function
with one set, so its gradients are bounded as well.

**Masks come from the BUILD-TIME geometry** (`StreamedShadows.geometry`), not the call's surface set:
under a gradient with respect to a vertex the call's vertices are traced and a mask cannot be built
from them, so building from the call's set would make the streamed model undifferentiable in
geometry where the frozen one is not — the shadows-frozen rule, kept. A receiver inside a body is
refused **at build** (`visibility.refuse_points_inside`, shared with `build_visibility`), not on the
first call, though no mask is built then.

Pinned in `tests/unit/test_radiation_streamed_model.py`: field and every gradient (emission,
reflectance, absorption coefficient, transmittance; one also against a finite difference) equal the
frozen model's; one mask build per chunk forward (`[4, 4, 3]`); forward then reverse-order rebuilds
under a gradient (`[4, 4, 3, 3, 4, 4]`); a vertex gradient equal to the frozen one; refusal at build.
Seven mutations, all red: no custom VJP, a mask per set, masks from the call's geometry, no refusal,
the setting ignored, a backward pass dropping the medium, the reflected set not gathered.

**MEASURED (2026-09-24): the whole Sozzi field through the public model, streamed**
(`validation/sozzi_radiation/model_at_mesh_scale.py`): all 1,635,909 cells, the case's
`lampWall.stl` (7,516 facets), the water as `cad.fluid(...)` read from the drawing,
`RadiationSettings(self_occlusion=NoOcclusion(), stream_receiver_mask=True)`, `UniformAbsorption(35.67)`,
black walls; jax 0.10.2, CPU, x64, macOS arm64, 11 cores, OCP 8.0.1 on CPython 3.13, default
`pair_limit` (4M). Against `compare_fluence.py`'s hand-chunked field (`G_aquaflux.npy`): relative
difference over the 1,285,221 lit cells **median 0, p99 1.8e-16, max 4.4e-16**; the 206,713 far pipe
cells (never compared before) agree to 6.8e-21 W/m² absolute. Build 65 s, field **1,170 s** — about
twice the hand-chunked 557 s, because `fluence_rate` also gathers the reflected set, which with black
walls carries nothing (its own docstring already says so: reflectance is traced, so there is nothing
to branch on). Solve: 3 restart cycles. ⚠️ Since #524 the two sets share one geometric pass
(`summed_fluence_rate`), so the reflected set costs a weighted sum rather than a second gather; the
1,170 s predates that and #522. **Re-measured 2026-09-26** (MESH SCALE under SHAFT CULLING below): the
pre-speed-up code `99c472c` reproduces it at 1,138 s / 10.94 GB, main `e12f214` takes 472 s / 5.57 GB
testing every pair and 314 s under the new default culling.

⚠️ **Peak memory footprint 11.15 GB, and it is the FIELD phase, not the build.** It was first
recorded here as "8.5 GB of it is the transfer build", which was an inference from a separately
measured 8.33 GB `build_transfer` peak and was **wrong**: #521 halved that build to 4.07 GB (model build
4.31 GB) and the same whole run then peaked at **11.07 GB**. The field agreed with the previous run to
every digit (the same median, p99, max and far-pipe figures), so this is a memory measurement and not a
change in the answer. The field phase grows **with each call**, not per pass: on 100,000 sampled cells
the footprint read 4.10 GB after the build, 5.92 after one `fluence_rate` and 7.16 after an identical
second call. **Most of it is JAX's compile and trace caches, not arrays**: `jax.clear_caches()` plus
`gc.collect()` between the two calls took the footprint from 5.91 to **2.74 GB** (the kept transfer
plus base), and the second call rebuilt ~2.9 GB (67 s per call, uncontended). Every call re-traces and
recompiled the per-chunk programs. **Fixed by #522** (merged): `streamed_fluence_rate` compiles its
per-chunk gather once per call, taking the live values and each chunk's mask as arguments. On the same
100,000 cells at `66501ac`, three identical calls settled at 4.36 / 4.83 / 4.57 GB (peaks 5.77 / 6.54
GB within a call, about one 4M-pair pass above), 40-42 s each against 67 s, field unchanged
(median 0, max 4.3e-16 against `G_aquaflux.npy`). The whole-run peak on the full mesh has not been
re-measured with both #521 and #522 in. ⚠️ **A peak on a platform whose allocator
keeps freed pages is attributed only by sampling between phases in one process** — subtracting a phase
measured in another process from a whole-run total is exactly how the wrong attribution got written.

## THE TRANSFER BUILD IS FILLED IN PLACE, NOT ASSEMBLED (#521, 2026-09-24)

`build_transfer` peaked at **8.33 GB** on the Sozzi lamp (7,516 facets; one `n x n` float array is
0.45 GB) and now peaks at **4.07 GB** (the whole model build: 8.54 → 4.31 GB). ⚠️ **The whole-field run's
peak did not move** (11.15 → 11.07 GB). It is set by the streamed field phase (see the #489 section
above), so this halves what a model *build* costs and not yet what a mesh-scale field costs. Measured with
`validation/sozzi_radiation/transfer_build_peak.py` — `build_transfer` alone in its own process under
`/usr/bin/time -l`, the water as the three hand-typed cylinders, `NoOcclusion()`, default
`chunk_size` 256, bodies tested by `EveryPair` (the default until 2026-09-26); jax 0.10.2, CPU, x64,
macOS arm64, 11 cores.

**Why the old one peaked at ~18 arrays.** ⚠️ **On this backend freed memory is not handed back**, so a
build's peak footprint is the *running total* of everything it ever formed, not what is alive at the
worst instant — measured by sampling the footprint (`proc_pid_rusage`, `phys_footprint`) between
steps with each step forced to complete: dropping every temporary at the end moved it by nothing.
Increments, in order: solid-angle rows collected in a list then concatenated and divided by `pi`
(+2.2 GB), the diagonal and point-source masks applied as two more whole-matrix passes (+1.0), an
`(n, n, 3)` offset array (+1.4), separation from it via its square and a safe-divisor copy (+1.8),
the source cosine (+0.9), then the facet shadow pass (+0.3, reusing freed pages). ⚠️ **Without forcing
each step to complete the attribution is wrong**: JAX dispatches asynchronously, and the first
decomposition blamed 4.5 GB on the shadow pass that was really the steps before it.

**What replaced it.** `_row_block` is one compiled pass computing, for `chunk_size` receiving facets
(`index`), the solid angles against the facets in `columns` only (scanned over the quadrature points
as before) scattered into rows of the full width, both masks by *global* index, and the offsets,
separations and cosines against every facet — so nothing larger than a block is formed. `_written`
puts each block's rows into three buffers of the final `n x n` size by a row scatter that **donates**
the buffers (in place on CPU; the first version's `dynamic_update_slice` measured 30 blocks into a
0.45 GB buffer at +0.03 GB, no warning). Buffers are never padded — trimming a padded buffer is a
whole-matrix copy (measured: +0.46 GB) — so the last block is filled out by **repeating its last
receiver**, which writes the same row twice. Inputs are `stop_gradient`ed at the start rather than
the outputs at the end, so a build under a trace (the frozen-geometry test builds under `jax.grad`)
still works and still differentiates to zero.

**THE BUILD SKIPS EVERY SENDING FACET THAT CANNOT REACH A BLOCK'S RECEIVERS: behind, IN the plane,
or provably negligible (2026-09-27; sharpened 2026-09-28).** `_columns_in_front` (a Numba loop,
`_any_in_front` → `_can_receive`, per block) keeps an areal facet only if, for some quadrature point
of some areal receiver in the block, the kernel could return more than a rounding of zero. It mirrors
the kernel's own clip: heights are snapped with `clipping._SLACK` exactly as `decidable_heights`
does, so the two agree on which vertices lie in the plane.
- **No vertex strictly in front** → what survives the clip lies in the receiver's plane, where every
  direction has zero obliquity: dropped. That covers "wholly behind" (the 2026-09-27 rule) and the
  far larger set **in** the plane — every triangle of a faceted cylinder's flat strip is coplanar
  with the rest of it, and the old strictly-behind test (`_BEHIND_MARGIN`, deleted — there is no such
  constant any more) kept and evaluated them all to produce zeros.
- ⚠️ **Unless the point lies on that in-plane part** (`_on_triangle` / `_on_segment` / `_near`): a
  point inside a coplanar triangle sees a hemisphere (the self-facet convention), one on an edge is
  on the contour. A valid mesh has neither; a degenerate facet's points need not, so kept.
- **A vertex in front** → kept unless `_projected_bound` (area × max height / d³, with `d` the
  centroid distance LESS the vertex-sphere radius; infinite when `d <= 0`) is below `_NEGLIGIBLE =
  2**-52` sr. ⚠️ **This is what the in-plane rule alone missed**: a strip's normals come from 2 mm
  edges at coordinates near 1, so they are good only to ~1e-13, and the strip's far members come out
  up to **5.9e-14 × distance** in front — beyond the snap — and the kernel returns ~1e-17 slivers
  for 500-770 of them per row. With the bound they drop; immediate neighbours (too near for it) stay.
- **Contract changed**: a skipped pair's value is zero **or under 2^-52**, not bit for bit the
  kernel's (which returns such pairs as dust). `test_a_facet_is_dropped_only_where...` now asserts
  `<= _NEGLIGIBLE`; its slivers (areas ~1e-18) send up to 9.0e-17 and are dropped by the bound.
- ⚠️ **Every quadrature point, NOT the centroid** (still binding): a centroid plane dropped 7 pairs of
  a 2,176-facet lamp worth up to **1.2e-5** — collinear-to-rounding triangles whose stored normal is
  noise (facets 2157, 2165, 2173 of `lamp_resolution.lamp(16, 64)`).
- **Blocks are cut along the Morton curve** (`aquaflux.morton.morton_order`, per-axis cells — cubic
  cells widened the lists, #574); lists padded on `lit_blocks.rounded_width`'s capped ladder (a
  power-of-two pad first reached 8192 columns on a 4,160-facet scene).

**MEASURED** (`validation/sozzi_radiation/transfer_build_split.py`: the lamp alone as in the Sozzi
model, default 6 points and 256 rows, a warm-up build then one timed build with each step waited on,
then a build evaluating every column as the reference; analytic 32 x 128 lamp, 8,704 facets; jax
0.10.2, Linux x86_64, 4 cores; one run per arm, 2026-09-28):

| | `main` `aae00b4` (strictly behind) | in-plane dropped | + negligible bound (shipped) |
|---|---|---|---|
| build | 87.1 s | 69.2 s | **46.6 s** |
| row blocks (kernel) | 74.2 s | 60.2 s | 36.7 s |
| host front test | 11.6 s | 7.8 s | 8.8 s |
| columns kept, share of every block's width | 26.9% | 21.8% | **12.8%** (median 391 of 8,704) |
| largest difference from every column | 0 | 1.1e-16 | 1.1e-16 |

**This lamp's transfer is zero except 32 rows**: the zero-area tip triangles at x = 0.810 (stored
normal = noise), whose points "see" neighbours at up to **|F| = 0.556**. The block holding them keeps
nearly every column (max 8,703) and must. Everything else is each row's immediate neighbours. ⚠️
**Not yet measured on the case's `lampWall.stl`** (7,516 facets, a different triangulation) or at
mesh scale; the 2026-09-27 shares below were taken under the strictly-behind rule and are history.
Earlier (strictly-behind) measurements: storage order 92.3 / 90.1% of `n^2`; 16 x 64 lamp (2,176)
17.26 s → 8.2 s at 256 rows, 4.48 s at 64; with a 32 x 30 chamber (4,160) 66.15 → 60.7 s at 256.
Tests: `test_a_facet_is_dropped_only_where_the_kernel_returns_nothing_at_every_point`,
`test_pairs_wholly_behind_the_receiver_are_skipped_and_nothing_changes` (whole build vs every column,
1e-15), `test_facets_in_the_receiver_s_own_plane_are_skipped_and_the_matrix_is_unchanged` (a plate
under a roof), `test_a_long_strip_off_the_origin_is_skipped_despite_its_normals_rounding`,
`test_the_bound_on_what_a_triangle_can_send_is_never_exceeded` (4,000 random triangles; the bound
without the radius term is exceeded by 214), and the case table
`test_an_in_plane_facet_is_kept_where_the_receiving_point_lies_on_it`. **Mutation pass (9, all red)**:
coplanar always kept, the edge / vertex / inside guards off, in-front vertices ignored, the snap far
too wide, no bound, the threshold at 1e-3, the radius term dropped.

**Remaining 4.07 GB** = 0.15 base + 1.36 kept + ~1 GB of one block's quadrature working set (2.49 GB
after the blocks) + ~1.5 GB for the facet shadow pass with the three-cylinder `Outside` at the 4M
default pair limit. Not bit-identical to the old build: at one block size the solid angles and
masks are identical and separation/cosine differ by ≤ 3.3e-16 (a compiled `dot`, see `CLAUDE.md`);
across block sizes same-wall pairs whose transfer is zero come back as ~1e-17 dust that moves with
the block shape, so tests compare to a rounding of the O(1) row sums. Speed is unchanged, measured
in one process with the two alternating: old 61.2 / 62.4 s, new 63.6 / 62.5 s for the solid-angle
pass (a cross-process single run had read 63 against 89 s — noise, and a reminder).

⚠️ **A point source is left out by its LABEL, and only a labelled facet with an area can show it.**
The kernel returns zero for a zero-area triangle, so a mask indexed wrongly passed every point-source
fixture; `test_a_point_source_is_left_out_by_its_label_not_by_its_area` labels a wall triangle.
Mutations: last block dropped, rows indexed locally, the point-source mask by local row, the
diagonal kept, the gradient not stopped at the inputs — all red; the donation removed is
green by design (memory only, pinned by the harness, not a test).

## The solid bodies moved to `aquaflux/solids/` (their record is `.claude/rules/solids.md`)

The analytic primitives, the constructive-solid-geometry (CSG) algebra, `Outside`, the
grazing-robust cylinder discriminant and the `Body` contract (`blocks`, `contains`, `traceable`) were
written here, in `radiation/occluders.py`, and none of them names any radiation: they are plain
geometry. They now live in `aquaflux/solids/bodies.py` and radiation imports them. ⚠️ **There is no
`aquaflux.radiation.occluders` module and no `Occluder` class any more** — the contract is
`aquaflux.solids.Body`, and `aquaflux.radiation` does not re-export the bodies. The word *occluder*
survives here as a role — a body passed in `occluders=` — not as a type.

## ⚠️ AN OCCLUDER DECLARES WHETHER IT CAN BE TRACED, AND COMPILING ONE THAT CANNOT IS A CRASH

`Body.traceable` (in `aquaflux/solids/bodies.py`; a `ClassVar`, defaulting to `False`) says whether
a body's answers are a pure array expression. `build_visibility` compiles the bodies that say they can be and calls the rest
directly.

Both directions are load-bearing, and they pull opposite ways:

- **A primitive wants compiling badly.** A body assembled from several inequalities, or a fluid
  of several regions, forms several receiver-by-facet intermediates. Evaluated eagerly each one
  is materialized — hundreds of megabytes at a production chunk — and the mask's cost becomes the
  cost of writing them rather than of the arithmetic. This is the same failure the ray-triangle
  test had before `_block_is_cut` was traced, one layer up.
- **A triangulated body cannot be traced at all.** It walks a grid on the host, dropping rays as
  they are settled — a search whose whole value is the work it skips, which tracing prices at the
  same rate as the work it does. Handed a tracer it raises `TracerArrayConversionError`.

⚠️ **Compiling the mask build unconditionally breaks the hybrid, which is the point of having
both kinds.** Measured directly: a host-side `Body` through `build_visibility` raises under a
blanket `filter_jit`. `test_a_host_side_blocker_and_a_primitive_stand_in_one_scene` puts one of
each in one scene and goes red if the compile is made unconditional. ⚠️ Note what was **not**
affected and was first reported as though it were: `RayCastOcclusion(grid=...)` is a
`SelfOcclusion`, reached through `strategy.field`, not through the occluder list — verified, not
assumed. The seam is the occluder list alone.

## MEASURED: the Sozzi reactor as three cylinders reproduces the hand-derived occluder EXACTLY (2026-09-23)

`validation/sozzi_radiation/primitive_occlusion.py`, both arms in one process on the same rays.
Configuration: `Outside(chamber, inlet, riser)` at the tutorial's dimensions (`R_BODY` 0.0445,
`X_BODY_END` 0.889, `R_PIPE` 0.00955, `X_RISER` 0.04765), against `BranchOpenings` from
`compare_fluence.py`; the case's own `lampWall.stl` (7,516 facets); **24,000 receivers drawn
uniformly from the meshed case's 1,635,909 cell centres** (19,437 chamber, 2,214 inlet, 2,433 riser);
exitance 696.42 W/m², `UniformAbsorption(35.67)`, `NoOcclusion()` for the surface's own triangles;
jax/jaxlib 0.10.2, CPU, x64, macOS arm64, 11 cores.

⚠️ **THE HAND-TYPED PIPES STOPPED 640 mm SHORT, AND THE SAMPLER THAT SHOULD HAVE NOTICED DROPPED
THE CELLS INSTEAD (found 2026-09-24, #489).** `primitive_occlusion.py` ended the inlet and riser at
x = 1.10 and z = 0.40 under a comment calling that "beyond the meshed case's own extent"; the mesh
runs to 1.739 and 0.894, the drawing's full 850 mm pipes. Its receiver sampler keeps only cells
*inside* the regions under test, so the 206,713 cells beyond them (13% of the mesh, every one a pipe
cell) were silently never sampled — a check filtering its population by the very thing it is
checking hides its own coverage gaps. Found when the whole mesh went through a model whose build
refuses any receiver outside the water. Both ends are now 1.75 / 0.90; every figure below is from the
corrected run unless it says otherwise. **Check a region's extent against the mesh's, not against a
comment.**

The timing table was measured on the earlier, truncated population (22,250 / 640 / 1,200); the
primitive arm is one branch-free expression whose per-ray cost does not depend on where the
receivers are, and it is kept as measured rather than re-quoted from a single corrected pass.

| | rays/s | 180M rays | repeat spread over 3 passes |
|---|---|---|---|
| `Outside` of three `Cylinder` primitives | **20.7M** | 8.72 s | 1.003x |
| `BranchOpenings`, hand-derived for this reactor | **28.8M** | 6.27 s | 1.033x |

- **0 of 180 million pairs masked differently**, and the fluence rate agrees to `0.0` relative at
  the median, the 99th percentile *and* the maximum — not "to rounding", bit for bit. The two
  describe the same ideal cylinders, so unlike the triangle comparison there is no faceting to
  explain a difference away, and a disagreement would have been a defect in one of them. The
  4,647 pipe receivers are where the mask does anything at all, and they carry 34.9M of those
  pairs, so the agreement is not an artifact of testing mostly-clear geometry. (It held on the
  truncated population too; it now also covers the far pipe cells that one never sampled.)
- **The general construction costs 1.39x the bespoke one**, which is the honest price of not being
  told where the openings are: three regions of three inequalities each plus the covering test,
  against two hand-written quadratics. ⚠️ **That figure needs the three passes to be worth
  quoting.** Single runs of the same pair on the same machine gave 1.67x and 1.48x; the arms are
  timed in one process so each is internally fair, but the *ratio* still moves by 20% between runs.
  Three consecutive passes repeating to 1.003x are what makes 1.39x a number rather than an
  impression.
- **The agreement survives a reformulation of the arm it is checked against**, which is worth more
  than the original check. #502 rewrote `BranchOpenings.blocks` as `crossing_ratio(...) > 1.0` —
  `sqrt(x^2+y^2)/R > 1` where it had compared squares — which is algebraically the same test and
  numerically a different one at the rim, exactly where a disagreement would live. Re-run against
  it: still 0 pairs. So `Outside` matches two independent spellings of the bespoke occluder, not
  one.
- **81.1% of pairs lie in one convex region** (81.0% chamber, 0.13% riser, 0% inlet — the lamp is
  in the chamber, so no facet is in the inlet). The 92.8% once recorded here was the truncated
  population's. ⚠️ **Testing every pair (`EveryPair`) does not skip those pairs**: it is
  one branch-free expression, so every pair pays the same handful of comparisons. What convexity
  buys is that the handful is all there is. **`ShaftCulling` (#554, below; the default since
  2026-09-26) is what skips them** — a whole tile at a time, from `Outside.clearance`.

⚠️ **THAT SHARE IS A PROPERTY OF WHERE THE RECEIVERS ARE** — the cell-centre population the field
is computed on, not a volume-uniform sample of the geometry, because the snapped mesh refines near
the walls and the lamp, which is where the pairs that are *not* in one region live. A volume-uniform
figure recorded here (97.1%) was taken on the same truncated pipes and is deleted, not corrected;
quote the cell-centre figure.

**Against the triangle grid**, which is the comparison the primitive path exists for. ⚠️ **A
ratio here needs THREE axes named before it means anything**: which analytic arm is the numerator
(the general construction or the bespoke closure), where the receivers are (pipe cells cost the
walk far more than chamber cells), and what grid the walk uses.

Only the primitive arm was measured here, at **20.7M rays/s** — and being one branch-free
expression it runs at that rate whatever the receivers are, which is what makes it comparable
against any grid configuration. The grid's corners are the **compiled walk's**, from
`grid_walk_direction.py` (300,640 cell-to-lamp segments per corner, one process, two alternating
passes, fastest kept; `bodyWall.stl`; jax 0.10.2, numba 0.67.0, CPU, x64, macOS arm64, 11 cores,
2026-09-27):

| grid configuration | grid rays/s | primitives are |
|---|---|---|
| pipe cells, near-cubic grid `(212, 11, 114)` (the default before #503) | 1,351,334 | **15x** |
| cells drawn uniformly from the mesh, near-cubic grid | 2,755,017 | **7.5x** |
| pipe cells, box-shaped default `(102, 102, 102)` | 2,648,106 | **7.8x** |
| cells drawn uniformly from the mesh, box-shaped default | 4,332,899 | **4.8x** |

⚠️ **The 146-732x once recorded here was the ARRAY walk's** (#502's matched square, 28,248-141,451
rays/s) and is deleted: the compiled walk (#571) moved the corners by ~30-50x (across runs), so the primitive
path's cost advantage *per ray* is now an order of magnitude, not three. What still separates them
is exactness at the rim and whole-field cost after culling (TRIANGULATED WALL AT MESH SCALE below).
Why the grid's rate depends on the receivers and the voxel shape is under THE WALK'S COST (#503).

⚠️ **What licensed reading #502's (array-walk) square was the within-process repeat, not any
closure.** Pass to pass its pipe corners repeated to 1.10x and its random corners to 1.01x.
⚠️ **A 2x2's "closure identity" is VACUOUS and was twice read as corroboration here** —
once in this section and once in #502's script. Both paths through a 2x2 are `D/A` with the middle
corner cancelling, so they agree for *any* four numbers, including four wrong ones. There is no
independent second path. A printed check that cannot fail is the measurement-script form of the
vacuous-test defect this file warns about for tests — with no test runner to notice it, and one
thing worse: **a tautology printed with an assertion's phrasing** (`must agree`, `check:`, a pair
of numbers and a tick) is more dangerous than the bare quantity would have been, because the
phrasing is what stops the reader asking what it could ever have shown.

⚠️ **Cross-run division is the remaining soft spot in the four ratios above**: the primitive rate
is from this harness and the grid rates from `grid_walk_direction.py`, and the same nominal measurement on
different days has come out well over 1.1x apart. So read the *column* of ratios as approximate and
the grid's internal comparisons as sharp. Running both arms in one process is what would fix it,
and is the reason to fold this harness into `grid_mask_check.py` as a third body rather than keep
it beside it — at which point the caveat is deleted rather than carried.

⚠️ **A WORKED EXAMPLE OF THE MISTAKE THIS WHOLE SECTION IS ABOUT, MADE WHILE WRITING IT** (on the
array walk, whose rates these are). The acceptance run put "pipe/default" at 38,711 and the square puts it at 28,248 — a factor of 1.37,
which is also the spread once quoted between repeats, and it was written up here as two
independent routes to one number and therefore as corroboration. **It is a product of two effects
that happens to land there.** The acceptance run's receivers were not the square's pipe corner:
they were 20,000 pipe cells *plus 4,000 chamber cells*, and chamber cells run near the random-cell
rate. Blending by ray share — time adds, so the effective rate is harmonic — predicts **31,661**
from composition alone, which is 1.12x of the pure corner; the remaining 1.22x is unexplained and
sits inside the run-to-run band. **Composition accounts for about a third of the gap and is
systematic and knowable; the rest is noise.** A number that matches something you already believe
is the least reliable kind of agreement: decompose it before reading it as a check.

⚠️ **Earlier per-axis figures from this comparison are DELETED, not corrected**: 3.4x, "under 2x",
1.33x and 1.70x were each computed by dividing numbers from different runs, and the smaller of them
are at or below the run-to-run variation that produced them — they could not have been resolved
however carefully they were divided.

**⚠️ THE DEFECT IS IN THE PAIRING, NOT THE DIVISION, WHICH IS WHY CARE DOES NOT FIX IT.** Every one
of those figures was computed correctly from numbers that were never comparable. Nothing at the
point of division could have caught it, because division is not where it went wrong — so the
remedy is structural, not behavioural: **run every arm in one process and report the repeat spread
beside the result.** The matched square is trustworthy because it removed the opportunity to pair
across runs, not because anyone was more careful inside it.

⚠️ **An extrapolation, flagged as one: ~595 s for the whole 1.6M-cell mesh** (against ~430 s for
the hand-derived arm). That is a per-ray rate from a 24,000-receiver run multiplied out by 68x,
which is exactly the shape of estimate that has been wrong before in this subsystem. Read it
beside the 557 s the entire field took with analytic occlusion as an order of magnitude, not as a
prediction, and do **not** subtract the two to infer what the gather alone costs.

⚠️ **The harness falls back to sampling the three cylinders when `work/case` is absent**, and
says which it used in its summary. Those runs are reproducible anywhere but are a different
receiver population, and its convex share is not the cell-centre one — do not mix the two.

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
it **follows `self_occlusion`**, so "off", "ray test" and "silhouette" apply to both masks alike, and
unset stays unset (both reach `build_visibility`'s own default, the ray test). Since #479 every
strategy serves a point in the fluid, so there is no longer a "can this strategy serve the volume"
question to route on: ⚠️ **there is no `serves_volume_receivers` any more** — it existed only so the
model could keep the silhouette clip, which then took only a *projected* share, away from volume
receivers, and it was deleted with that limitation. (Before #480 split the routing, one strategy went
to both masks and `build_radiation_model` RAISED with the silhouette selected; before #479 the split
kept the fluence rate in the fluid all-or-nothing per pair under the silhouette. Both are history.)
**Selecting the silhouette now clips every cell** — see the volume-receiver section under analytic
occlusion for what that costs; `receiver_occlusion=RayCastOcclusion()` keeps the clip between facets
only. `test_a_model_can_be_built_with_the_silhouette_strategy` pins that both masks hold fractions.

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
to buy was a rays-first, small-block artifact. **A bound *below* 4M is not faster either — measured
flat (#526):** `validation/radiation_receiver_ray_mask.py` with `RADIATION_WORK_LIMIT_SWEEP=1`,
400,000 rays against 1,532 triangles, two alternating passes, fastest kept, run alone (jax 0.10.2,
CPU, x64, macOS arm64, 11 cores, 2026-09-25): **449 / 435 / 422 / 431 / 439 / 428 Mtest/s** at
0.25 / 0.5 / 1 / 2 / 4 / 8M, answers identical. ⚠️ The review that filed #526 read 134 → 424
across the same ladder, on a machine running another session's tests; that slope was the
contention, and is not recorded as a property of the limit. Each distinct call shape compiles once;
since #526 a chunk shorter than the full one is padded to a power of two (`segment_is_cut`), so a
caller whose ray count changes every pass compiles a few programs, not one per pass.

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

## Four closed-form checks added after the fact, and what each one found (2026-09-21)

Written to close gaps against the design specification's analytic cases. **None found a defect in
a computed value**, which is worth stating rather than implying; two found things worth knowing,
and all four were mutation-checked (seven one-line mutations, each red on its own assertion).

- **Case 6 in absorbing water** (`test_an_infinite_line_in_an_absorbing_medium_gives_the_bickley_functions`):
  `G = P'/(2 pi r) Ki_1(a r)`, `E = P'/(2 pi r) Ki_2(a r)`. A summed line of isotropic points,
  spacing `r/50`, optical half-length 35, reproduces both to **<1e-11 at `a r` = 0.5 and 2** —
  far better than the nominal second order, because a midpoint sum of a smooth integrand decaying
  along an effectively infinite line converges spectrally. So the tolerance (1e-9) is set by the
  reference quadrature: `scipy.integrate.quad`'s default `epsabs` of 1.5e-8 would have been the weak
  link, and the test passes `epsabs=0, epsrel=1e-13`.
- **Case 8c, Walton's obstructed squares** (`test_an_obstructed_pair_converges_on_the_published_view_factor`):
  `F = 0.11562061`, the geometry confirmed independently to 0.1156206021 (blocker 0.5 x 0.5,
  centred, **0.75 from the first plate**; midway gives ~0.0995). Both strategies converge at second
  order in plate spacing. Errors at 6 / 12 / 24 plates a side, six-point receiver rule: silhouette
  with the blocker near the source **-1.6e-4 / -4.0e-5 / -1.0e-5**; one ray per pair **1.06e-3 /
  2.6e-4 / 6.4e-5**. ⚠️ With the blocker near the *receiver* the two strategies agree to 1e-16: the
  shadow is scaled by 4 and triangle centroids sit at thirds of a cell, so every shadow edge lands on
  a facet edge and every pair is wholly hidden or wholly clear. A fixture can make the silhouette
  look binary.
- **Case 11, the emission transfer** (`test_a_narrow_source_sends_its_emission_where_its_profile_says`):
  replaced a test that asserted only *not equal to Lambertian*. Against
  `M A f(theta_e) cos(theta_r)/r^2` summed per triangle pair, **1.8e-7 at width 1e-3, second order,
  identical at n = 1 and 50**. ⚠️ The reference must be per pair: under a `CosinePower(50)` beam
  the two receiving triangles of one square differ by 12% at width 1e-2, which a reference at the
  square's middle reads as a first-order error in the code. ⚠️ A profile carries its constant in
  **two** methods — the gather reads both, the transfer `radiance_per_exitance_at` — so a
  mutation of one is invisible to a test of the other path. The contract between them is pinned in
  `test_radiation_profiles.py`.
- **Conservation of light through the adjoint** (`test_the_adjoint_conserves_light_through_every_facet`):
  in a closed box of uniform `rho`, every row of `d(B, H, G)/dM` sums to `1/(1-rho)` (times 4 for
  `G`), to 1e-14 at `rho = 0.9` — a check on the adjoint with no finite difference. ⚠️ **The
  per-facet form is NOT exact**: `d(sum A H)/dM_k = A_k/(1-rho)` needs reciprocity, and with the
  receiver on quadrature and the source in closed form the rows are exact at the columns' expense.
  On `stretched_box(2)` the column sums are off by up to **4.8%** (reciprocity residual 3.2%),
  and that per-facet gradient misses by exactly that. Not an adjoint error; do not "fix" the test
  toward it.

⚠️ **FIXED BY DECLARATION: the silhouette strategy used to ignore a ONE-SIDED sheet from behind.**
It counts only blockers facing the receiver (`facing` in `SilhouetteOcclusion._per_triangle`), which
is exact on a closed, consistently wound surface — a blocked sight line enters it through exactly
one front face — and back faces are excluded because they would double the count. A lone
zero-thickness sheet, though, has the medium on both sides: seen from behind it hid nothing, and
Walton's pair with a one-sided blocker read the **unobstructed 0.1998** from one plate, silently.
The ray test is two-sided and never had the problem. Now `SilhouetteOcclusion(two_sided=(names,))`
counts the named bodies (by `Surfaces.solid_names`) from both sides; a misspelt name **raises**
(it would otherwise leak light with no error) — against the one surface set, or in a `Scene` against the
union of lamps and reflectors, each set then built with its own share (see THE SCENE); and any **open piece left undeclared is warned
about** at build, naming its bodies. With the blocker declared, the one-sided Walton pair converges
exactly as a two-sided copy did (−1.6e-4 / −4.0e-5 at 6 / 12 plates a side).

**Why a declaration and not topology** (decided by the user, 2026-09-21, over "automatic by
topology" and "refuse open surfaces"): `checks.open_facets` marks facets on a connected piece with
a free edge (pieces joined only across edges used exactly twice), and it is wrong both ways on
real input. **A sheet welded to the wall all the way round has no free edge** — each rim edge is
used three times, which neither joins nor opens it — so it reads closed (pinned:
`test_a_sheet_welded_in_all_the_way_round_reads_as_closed`, 8 non-manifold edges). **A duct
exported without its end caps reads as a sheet**, and counting its wall from both sides would count
every exit-and-re-entry twice. So topology drives only the *warning*; the counting follows the
declaration. Naming a closed body two-sided makes it block twice and errs dark — not detectable,
because a welded sheet looks closed too.

## MEASURED: aquaflux against discrete ordinates on the Sozzi reactor (2026-09-22)

Instrument: `validation/sozzi_radiation/` (`generate_dom_reference.py` runs of-optical-radiation's
DOM on its `uvReactorSozzi2006-DOM` tutorial in Docker; `compare_fluence.py` computes aquaflux's `G`
on the same 1,635,909-cell mesh and writes a VTU + plots). Full numbers and configuration in its
`README.md`; the headline, so it is findable from here:

⚠️ **These DOM figures are on `8 x 4` (64) and `16 x 8` (256) grids, which are not uniform**: of-optical-radiation's
`nTheta` spans the whole sphere, so those polar bins are twice the azimuthal ones. The uvmesh swap set was rerun on
`6 x 6` / `12 x 12` (2026-10-02) and the 64-direction dose shortfall at k = 0.5 fell from 12.4 % to 2.2 % at 72; the
snapped-mesh figures below were not rerun.


- **Whole-reactor volume-mean G agrees to 0.09%** (133.28 aquaflux, 133.16 DOM-256, 131.67 DOM-64).
- **DOM converges on aquaflux**: DOM/aquaflux p10-p90 over lit cells 0.36-1.65 at 64 directions,
  0.886-1.076 at 256. The spread is DOM's ray effect — at mid-lamp the true field is axisymmetric
  and aquaflux gives one curve per radius, while DOM scatters with angle. In the pipes DOM is wrong
  by orders of magnitude (only a narrow cone of directions reaches down a 19 mm pipe).
- **aquaflux's own discretization error** near the lamp (4 mm facets, against a 9x-refined lamp):
  median 0.44%, p99 4.1%.
- **Cost**: 557 s for all 1.6M cells, 11 cores; DOM 1035 s (64 directions) and 4193 s (256), one
  core in a Docker VM, 15 outer sweeps each — not a like-for-like timing.

⚠️ **How it stayed affordable, and why the stock build could not.** With black walls there is no
interreflection, so the field is the direct gather from the lamp alone (7,516 facets) — building a
model would have formed a 61k-facet transfer and a ~6e14-test wall mask. Visibility is exact and
O(receivers x facets): the fluid is three convex cylinders, so a pipe cell sees a lamp point exactly
when the segment passes that pipe's opening (a custom `Body` in the harness, checked against
brute-force sampling: 0 disagreements in 8000). And `build_visibility`'s dense
(occluders x receivers x facets) mask is ~12 GB per occluder at this size, so the harness gathers in
20k-receiver chunks. **A user-facing driver needs all three of these for a real reactor** — the
wall mask in particular has no general cheap form yet.

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

⚠️ **The surface set passed at call time must be the build's geometry, and every call now checks.**
`model.py`'s docstring used to say the call-time set's "geometry … is not consulted". **False**:
the direct gather and the point-source arrivals read its vertices, centroids and normals live,
while the transfer matrix and both masks are the build's — so a moved set gave a field lit from
the new position through shadows cast from the old one, with no error. `RadiationModel.geometry`
is now a SHA-256 of the vertices plus the point-source labels (the labels decide which facets
the transfer leaves out, so they are geometry too); `radiosity` checks it, and
`surface_irradiance` / `fluence_rate` reach it through `radiosity`. **Exact, not toleranced**:
`with_optics` carries the build's own vertex array, so a legitimate call matches bit for bit; a
1e-6 move is refused. ⚠️ **Traced vertices are REFUSED at every call too (`TypeError`, since
2026-09-29) — they used to pass unchecked, and the gradient they gave was wrong.** The docs promised
a lamp-position derivative "with the shadows frozen", but the facet-to-facet transfer is frozen as
well, so a traced lamp reached the direct gather only and the reflected light lost all dependence on
the position. Measured by an independent new-user review, confirmed by rerunning it: 10 W lamp in a
0.3 x 0.3 x 1 m duct, walls rho = 0.3, `NoOcclusion`, one receiver, lamp moved along z — traced
`jax.grad` **-3.54 W/m^2 per m**, central difference over *rebuilt* models **+4.08 / +4.04** at
h = 1e-3 / 1e-5; the traced value equalled the black-wall `direct_fluence_rate` gradient to 1e-15.
The owner chose refusal over documenting it or making the transfer live. A model may still be
**built** from traced vertices (its fingerprint is then `None`, which no concrete set matches;
`test_the_expensive_geometry_is_frozen` does this). Position derivatives are
`direct_fluence_rate` / `direct_irradiance`'s, FD-checked in `test_radiation_gather.py`. Pinned by
`test_traced_vertices_are_refused_because_the_transfer_would_not_follow_them` (all three entry
points) and `test_a_streamed_model_refuses_traced_vertices_as_a_held_one_does`; mutation-checked
(disabling the refusal turns all four red). Mutation-checked; a separate check at the top of `surface_irradiance` was
**dominated and deleted** — with a wrong-sized set `assemble` completes and `radiosity` refuses
with the same message, so the extra check changed nothing any test could see.

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
- `work.in_passes` (`_chunked` in `gather.py` until #528) no longer pads (#524): full chunks are sliced in place and a shorter remainder runs
  as a scan of one step (a bare call until 2026-09-26, see "PER-CALL COMPILES" below). Losing or duplicating the remainder needs a chunk size that does not divide
  the receiver count to show up at all, which is why `test_chunking_changes_nothing_about_the_answer`
  uses 37 receivers.

## Documentation

The package **is** in `docs/conf.py`'s `PUBLIC_SUBPACKAGES`, with a `SUBPACKAGE_GROUPS` entry
keyed on the modules its names are *defined* in — `model` first, then the pieces it composes.
Listing a subpackage publishes the whole of its `__all__`, so that list is the editorial
decision. It was reviewed at this point and kept entire: every export is something a user can
legitimately reach for, including the two solid-angle kernels, whose warning that they are **not
interchangeable** is worth publishing rather than hiding.

**UV disinfection of AIR is in scope as well as water** (project owner, 2026-09-29). Write the
medium as "the medium" / "the fluid" (water or air) in the guide, the CAD page and the package's
docstrings, and name water only where the content is genuinely water's: the UVT conversion
(`absorption_from_uvt` and its 5.129 /m), Bolton's quartz-sleeve error figures, and the
Sozzi & Taghipour (2006) case. Leaving `absorption` out is the non-absorbing medium, which the guide
offers for air at 254 nm.

**The user guide is `docs/radiation.md`** (in the Guide toctree beside `cad_geometry.md`, which it
links to rather than repeats: CAD import, the mesh-patch lamp, streaming and culling timings on the
Sozzi mesh live there). ⚠️ **It restates defaults and measurements by value**, so a change to any of
these makes it false and must update it in the same change: the six-point `receiver_quadrature`,
`refine_for_receivers`' `max_ratio=0.25`, the GMRES global relative `1e-10`, `RayCastOcclusion` as
the self-occlusion default and `ShaftCulling` as the culling default, opaque as the transmittance
default, GMRES restart 120, point sources Isotropic-only (`check_profiles`), the 2.6%-at-eight-sectors inscribed-area undershoot, 95% UVT = 5.129 /m, and the Sozzi
figures (0.09% volume mean; 0.36-1.65 / 0.886-1.076 at 64 / 256 directions, non-uniform 8 x 4 / 16 x 8 grids; 0.44% median lamp
discretization; dose mean 0.3%, log reduction within 1.1%, DOM-64 short by 23% at k = 0.5). Its
quick start — `G = 80` in a 6x6x6 box mesh's own patches at `M = 10, rho = 0.5` — was run
(2026-09-28, and again 2026-09-29 after it became the quick start) and holds to rounding.
**A new-user review (an independent agent with only the published pages and docstrings,
2026-09-29) drove a second pass**: the runnable example moved first, the STL-units trap
(`read_stl` does not scale; millimetre drawings are common), an Air disinfection section,
irradiance at arbitrary oriented points *with* reflection (two `direct_irradiance` gathers, one of
`B - M` as Lambertian — checked `E = B` in a glowing box), a point-source snippet, one-line call
forms for the diagnostics, the closed-enclosure qualifier on the row sums, and the lamp-position
derivative (now refused through the model; see "The surface set passed at call time"). Every new
snippet was run. Its **Theory** section writes out every equation the code
evaluates — the direct gather, both solid-angle kernels, `surviving_fraction`, the voxel walk,
`F^geo`/`F`/`F^M`, the eliminated system, the two-gather `G`, the adjoint — and was checked
(2026-09-29, 3x3x3 box of its own patches, CosinePower(4) lid so `F^M != F`, uniform `a = 2`)
against `radiosity` / `surface_irradiance` / `fluence_rate` to 1e-16 relative, and `F^geo` rows
against its stated six-point formula to 4e-17. A change to any of those formulas must update it.


## The winding number answers "is this cell inside the metal", and it is exact

`enclosure_winding(vertices, points)` sums the **signed** solid angle of every facet at each point
and divides by `4π`. On a closed, consistently wound surface that is `±1` inside and `0` outside,
the overall sign set by whether the file is wound outward or inward — so `check_points_outside`
tests the **magnitude** against 0.5, a threshold with nothing behind it to tune because the
quantity it cuts takes only two values.

Measured on a unit box at 12, 48 and 192 facets, at x = 0.999 (a thousandth of a box-width from a
wall): **87 / 0 / 0 ulp from `1` eagerly and 25 / 0 / 0 compiled**, and at most 2 ulp at two further
interior points (`validation/radiation_enclosure_winding.py`, 2026-09-26; ~2e-14 at worst). Outside,
the drum in the same harness read at most 2.1e-14 eagerly — and exactly `0.0` now outside a closed
piece's box, where it is no longer summed (next section). ⚠️ **The record used to say "`1.0` to the
last bit"; re-measured under #530 that does not hold even eagerly** on the 12-facet box — the claim
was unfalsifiable (no harness) and is replaced, not annotated. There is no near-field regime where
it degrades, which is what disqualifies the two obvious alternatives:

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
occluder bodies do refuse interior points at the visibility build, because `Body.contains` is
O(1) per point; a triangle soup has no such shortcut. Wiring it into the model by default would
change a shipped behaviour and roughly double that build, so it is an explicit call.

## THE WINDING CHECK IS COMPILED, AND A CLOSED PIECE IS SKIPPED OUTSIDE ITS BOX (#530, 2026-09-26)

**Compiled per pass.** `_summed_signed_solid_angle` is one `jax.jit` program per pass; the host loop
(`_signed_total`) cuts the points by `pair_limit` and pads a short last pass to a power of two
(`triangles.padded_length`, repeating its last point, answers dropped), so a scene compiles one
program per distinct piece size plus a few remainder shapes. Eagerly each kernel operation wrote a
whole pass of pairs to memory. The docstring note that this loop was "deliberately not the padded
scan … padding would be cost with nothing to buy" is gone with it: the padding is what bounds the
number of compiled shapes.

**A closed piece is summed only at points inside its own bounding box.** The surface is split into
`_pieces` (the same pieces `open_facets` uses — refactored out of it, one home), and
`_closed_pieces` flags a piece whose **own triangles walk every edge as often one way as the other**
— i.e. it is a 2-cycle, so its winding number is an integer, locally constant off the surface and 0
on the unbounded complement of its box. ⚠️ **This is deliberately stricter than the issue's wording
("`open_facets` reports closed and its winding is consistent")**: `open_facets` reads a sheet welded
in all the way round as closed, and the weld also splits the box into two halves that read closed
too — none of which bounds anything alone, so skipping them would give a wrong number
(`test_a_sheet_welded_into_a_closed_body_is_summed_everywhere`). The cycle test subsumes both
halves of the issue's condition and catches that case. Open, inconsistent and welded pieces are
summed everywhere, so the open-surface warning in `check_points_outside` is unaffected. A new
`tolerance=` keyword on `enclosure_winding` and `check_points_outside` sets the vertex merge the
pieces are found through; a piece closed only to within it is skipped as if closed exactly.

**MEASURED** (`validation/radiation_enclosure_winding.py`: `closed_drum(2000)`, 8,000 facets × 4,000
points uniform in a cube three drum-widths across, 3.7% inside the drum's box, default 4M
`pair_limit`, all arms in one process, warm-up then two alternating passes, fastest kept; jax 0.10.2,
CPU, x64, **Linux x86_64, 4 cores, 16 GB** (a cloud container, not the usual macOS machine), run
directly with output redirected — `run_case.sh` needs `vm_stat` — nothing else running, 2026-09-26):

| arm | fastest s | Mpair/s | spread | vs eager |
|---|---|---|---|---|
| eager per-pass loop (the old code) | 12.25 | 2.6 | 1.02x | 1.00x |
| compiled kernel, no skip | 1.49 | 21.5 | 1.06x | **8.2x** |
| shipped (compiled + skip) | 0.125 | 256 | 1.11x | **98x** |

The skip alone is 11.9x against a 27x ceiling (1 / 3.7%); the rest is the host topology pass and the
in-box points. Inside/outside identical across all three arms, max |winding difference| 2.1e-13, and
the 3,853 points outside the box read exactly `0.0` (eager's largest there: 2.1e-14). The issue's own
probe read 24 → 169 Mpair/s (~7x) for the compile on a contended macOS machine; the ratio agrees,
the rates are not comparable across the two machines.

Tests (`tests/unit/test_radiation_enclosure.py`), mutation-checked, 7 of 8 red: the cycle test
replaced by `open_facets` semantics, never skipping, the direction ignored, only the first closed
piece handled, `+=` for `=` where an open and a closed piece overlap, one box drawn round the whole
surface, and the padding repeating the wrong point. **Dismissed**: a strict rather than inclusive
box test — a point on a closed piece's box face is either on the surface (winding undefined) or
outside it (0 either way), so no input tells them apart; it stays inclusive because the surface can
lie on the box.

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
| **The ultraviolet-reactor summation models** (PSS, MPSS, LSI, MSSS; extended by RAD-LSI, UVCalc3D) | the basic models assume an **unobstructed** lamp-to-point path. RAD-LSI and UVCalc3D add shadowing (and reflection) by **neighbouring lamp sleeves** in a multi-lamp array, and Liu et al. (2005, *Water Research*) found both reproduce measured lamp shadowing. So lamp-sleeve shadowing is **established practice there, not our addition**; what the mask here adds is occlusion by arbitrary triangulated or CAD geometry (baffles, pipe openings, walls). How those codes treat a *partially* blocked source is not recorded here — do not infer it. |

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

**The cost is geometry-independent *without a grid*, which is why one ladder settles it for
every scene the default path runs.** The pass tests every ray against every triangle with no
early exit, so what the rays *hit* cannot change what it costs. Checked rather than asserted:
the sleeve scene and an otherwise identical scene with the sleeve moved outside the box differ
by **2.65x in blocked pairs** (463,936 against 174,912) and by **1.11x in wall clock**, inside
the ~20% spread this machine carries. ⚠️ **`RayCastOcclusion(grid=...)` breaks this, by design**
— it stops at the first blocker, so its cost depends on what the rays hit and no single ladder
transfers between scenes. Every figure in this section is the ungridded path.

## A MASK STORES ITS ANSWER AT THE NARROWEST TYPE, AND A PASS HOLDS INDICES (#525, 2026-09-25)

**`hidden_by_geometry` and `overlapping` are stored at what the strategy can say.** The surface's
own layer was a float64 fraction and a bool flag for every strategy, 9 B/pair on top of 1 B per
body, although a ray test only ever says 0 or 1, `NoOcclusion` only 0, and `overlapping` is only
ever set by the silhouette clip and read by nothing in the package. Now:

| strategy | `hidden_by_geometry` | `overlapping` |
|---|---|---|
| `NoOcclusion` | `None` | `None` |
| `RayCastOcclusion` | bool | `None` |
| `SilhouetteOcclusion` | float64 | bool |

`surviving_fraction` widens the layer (`jnp.asarray(..., dtype=float)`) and skips it when `None`;
the gather hands `in_passes` only the layers that exist, so the widening is per chunk and never the
size of the problem. **Bit-identical**: `1.0 - float(bool)` is the float it replaced. ⚠️ **Code that
reads `hidden_by_geometry` must allow `None` and a bool** — `np.asarray(None)` is an object array
whose `np.any` is `False`, which reads as "nothing hidden" and happens to be right, but
`.nbytes`, `.shape` and arithmetic are not. The transfer's own mask (#525 item 4) is built the same
way and comes out the same. `validation/sozzi_radiation/transfer_build_peak.py` skips a `None`
layer when summing what the transfer stores.

**The brute-force pass forms rays a chunk at a time** (#525 item 2).
`triangles.pairs_are_cut(receivers, sources, min_distance, vertices, receiver, source, target=)`
takes a pass's pairs as indices; its chunk callback gathers the endpoints, forms the segments through
`_segments` — the same eager operations `segment_is_cut` applies to a whole array, so every ray is
the same numbers — and builds the exclusions, for one compiled chunk's rays only. Both entry points
share `_cut_in_chunks`. ⚠️ **The rays are formed eagerly, not inside the compiled kernel, on
purpose**: the issue warned that compiling the margin's division could move its last bit, and a
margin decides a hit. The grid path still takes whole-pass endpoints and exclusions
(`self_occlusion._exclusions`), because its walk is host code that steps every ray.

Measured with `validation/radiation_mask_storage.py` (`radiation_receiver_ray_mask.py`'s 4,992-facet
annular reactor, one `Cylinder` just inside the sleeve as a body, each build in its own process
under `/usr/bin/time -l`; jax 0.10.2, CPU, x64, macOS arm64, 11 cores, run alone, two rounds
alternating before and after; "before" is #550's branch, `3735118`, 2026-09-25), every checksum
identical:

| arm | stored per pair | peak footprint | seconds |
|---|---|---|---|
| `NoOcclusion`, 40,000 receivers | 10 → **1** B | 4.21 → **2.40** GB | 1.7 / 1.4 → 1.2 / 1.1 |
| `RayCastOcclusion`, 2,000 receivers, no grid | 10 → **2** B | 1.50-1.52 → **1.01-1.02** GB | 109.2 / 96.9 → 107.8 / 98.7 |

The ray-cast time does not move: the two rounds spread 12% within each side. ⚠️ A first single
run read +8% for it and a footprint of 7.40 → 5.60 GB for `NoOcclusion`; the harness then checked
its answer with `mask.surviving`, which forms a float of the whole mask, so that footprint was the
check's, and the +8% was inside the spread. Both were discarded, and the checksum now reads the
stored arrays. The `NoOcclusion` row is the Sozzi case's own configuration: a held mask there is
now a byte a pair per body, which moves the receiver count at which a model has to stream, not
whether a 1.6M-cell mesh fits (12 GB per body still does not).

Tests, mutation-checked: `test_each_strategy_stores_its_answer_at_the_narrowest_type_that_holds_it`
(types, and the stored bytes exactly 1 and 2 per pair), `test_a_mask_held_as_bits_gives_the_field_it_gave_as_a_fraction`
(the bool mask against itself widened to float, `array_equal`),
`test_a_pass_forms_its_rays_a_chunk_at_a_time_not_all_at_once` (no call of `_segments` forms more
rays than one compiled chunk), `test_the_grid_walk_is_handed_one_pass_of_exclusions_at_a_time`.
Seven mutations, all red: the ray test storing a float, `NoOcclusion` storing zeros, the surface
layer dropped from the gather, the rays formed for the whole pass, the target facet not excluded
(an existing transfer test), a pass's pair indices misaligned, and the widening replaced by a
logical `not` -- which is the same number for a bool, so only the silhouette's half-hidden test
(`test_the_surviving_fraction_lets_a_half_hidden_pair_through_by_half`) can see it.

## THE RECEIVER RAY MASK CASTS ONLY FACING PAIRS, AND THE GRID TAKES INDICES (#526, 2026-09-25)

Three changes to `RayCastOcclusion` for receivers in the volume. The field is unchanged: every
checksum below is identical to 13 figures before and after.

**A pair whose source faces away from its receiver is not cast.** The gather weights a pair by
`radiance_per_exitance`, which for `Lambertian`, `CosinePower` and `PhotometricProfile` is exactly zero
for `cos <= 0`, so a ray there could only be multiplied by nothing. Now `Profile.dark_behind` (a
`ClassVar`, `False` on the base, `True` on those three) declares it, and `Surfaces.dark_behind` asks it of every areal facet's
profile. When that holds and `receiver_facet is None`, `field` drops the pairs whose receiver is
**certainly** behind the facet's plane — `BackFaces.behind` (`back_faces.py`, since 2026-09-27; it was
`self_occlusion._facing_away`, and there is no such function now), `clipping.decidable_heights` of the receiver above the plane
through the centroid, a height too near zero to trust snapped to zero and **kept**, so rounding can
never cull a pair the gather's own cosine would light. Point sources are never culled (zero normal, so
zero height; and excluded by label). The dropped pairs are recorded clear, so
`hidden_by_geometry` there means "blocked, where it matters", and the mask says so:
**`Visibility.clear_behind`** (and `OcclusionField.clear_behind`), static, `False` from every other path.
⚠️ **A gather through such a mask refuses a set whose areal profiles are not all dark behind**
(`_refuse_light_from_behind`, in `summed_fluence_rate` and `direct_irradiance`), because a profile
swapped in at call time would be taken at the mask's word. **Receivers on facets are always cast in
full**: the transfer's weights do not vanish behind a source (the issue measured `geometric == 0` on
only 3-8% of facet pairs), and a `receiver_facet` array — even all `-1` — switches the cull off, which
is how a test builds the full mask to compare against.

**The cull needed two things it did not ask for.** Casting only the kept pairs means each pass forms
its exclusions for its own rays, from the two indices, so **the whole-problem `(n_receivers,
n_facets[, 2])` exclusion array of #525 item 1 is gone** as a side effect; the rest of #525 is the
section above. And a pass's ray count now differs from every other pass's, so `segment_is_cut` pads a
chunk shorter than the full one to a power of two (repeating its last ray, answers dropped) — without
that, every pass compiles its own programs.

**The grid kernel was sent indices** (then worth ~1.9x, below). ⚠️ That array walk — `_Rays`,
`_indexed_pair_is_cut`, the per-step `reduceat` — is **deleted** (2026-09-26): the walk is one compiled
loop per ray now (THE COMPILED WALK, under GRID ACCELERATION), and none of those names exists.

**A stream prepares once.** `SelfOcclusion.prepared(surfaces)` (default: itself) does a strategy's
surface-only work once; `RayCastOcclusion.prepared` builds the grid, and `grid=` now also accepts a
built `TriangleGrid` (refused if it is of other triangles). `streamed_fluence_rate` prepares the
strategy and calls `refuse_points_inside` **once over all points**, then builds each pass's mask with
`visibility._unchecked_visibility` — `build_visibility` without the refusal, which it now wraps.
⚠️ **There is no per-pass refusal any more**; a test that counts mask builds patches
`gather._unchecked_visibility`, not `gather.build_visibility`.

Measured with `validation/radiation_receiver_ray_mask.py` (an annular reactor in triangles: the Sozzi
lamp's radius and length at 24 × 32, a dark sleeve at x = 0.03 m, the dark vessel wall at 48 × 32
facing in — 4,992 facets, all of them blockers; receivers uniform in the annulus;
`UniformAbsorption(35.67)`; warm median of three; jax 0.10.2, CPU, x64, macOS arm64, 11 cores, each
run alone — two earlier runs overlapped another session's fast gate and were discarded; "before" is
`4e0b632`, 2026-09-25):

| build | before | after |
|---|---|---|
| every triangle, 500 receivers | 29.8 s | **22.4 s** (1.33x) |
| grid, 4,000 receivers | 218.3 s | **86.4 s** (2.53x) |
| streamed, grid, 4,000 receivers in 20 passes | 273.5 s | **87.0 s** (3.14x) |

- **The cull is 1.33x here, not the issue's 2.7x, and the scene is why.** 75.6% of pairs face their
  receiver, because the wall — 62% of the facets — faces every receiver; the issue's 37% was a lamp
  alone. The brute-force row is almost exactly `1 / 0.756`, which is what the cull alone predicts.
  ⚠️ **Read the saving against the scene's facing share**, not as a constant.
- **The grid row's remaining ~1.9x is the index change.** It was 91k rays/s before; the issue's own
  probe read 70.7k → 145k with a sleeve in the way, on a busy machine.
- **Streaming now costs what holding costs** (87.0 against 86.4 s). Before it paid ~55 s over the held
  build for 20 passes of grid rebuilds and facet refusals.
- The stored `hidden_by_geometry` mean falls 0.43 → 0.19: those are the back-facing pairs the lamp's
  own body blocked, now recorded clear. It is the field that must not move, and it does not.

Tests, each mutation-checked (11 of 12 red): `test_a_pair_facing_away_is_not_cast_and_the_field_does_not_notice`
(grid and not; culled against a full mask built with `receiver_facet = -1`, fields `array_equal`, masks
different on back pairs — the cull off, and the cull on the wrong side, both fail),
`test_only_the_facing_pairs_are_cast` and the updated `test_the_ray_test_passes_are_bounded_in_pairs_not_receivers`
(ray counts equal the facing count), `test_a_receiver_in_a_facet_s_own_plane_is_cast_whatever_its_rounding_says`
(a raw sign test fails), `test_a_profile_that_lights_behind_itself_is_refused_through_a_mask_that_did_not_look`,
`test_a_point_source_is_never_left_untested`, `test_a_stream_builds_its_grid_and_refuses_its_scene_once_not_once_a_pass`
(counts refusals through both module bindings, so a per-pass refusal through the mask builder shows),
`test_a_grid_built_for_other_triangles_is_refused` and `test_ray_counts_that_vary_from_call_to_call_share_their_compiled_programs`
(the grid's own compiled-shape test went with the array walk).
Culling facet receivers too fails an existing transfer test. **Dismissed**: dropping the point-source
label from the cull is inert — a point source's zero normal already gives it zero heights — and the
label stays because a source's kind is read from its label, never inferred.

## ONE MORTON ORDERING, PER-AXIS CELLS — cubic cells measured and REJECTED (#574, 2026-09-28)

`aquaflux/morton.py::morton_order` is the only ordering of points in the package: shaft culling's
receiver groups and lamp-facet clusters (`_Curve`), the gather's chunks and `lit_blocks`, the transfer
build's row blocks, and `FacetClusters`. It is the former `culling.spatial_order` moved to a neutral leaf,
**bit for bit** (10 bits, each axis scaled to its own extent — checked equal on four point sets), and
`FacetClusters`' private 21-bit copy (`clusters._morton_keys`) is gone: its ordering moved from 21 to 10
bits, which only reorders centroids sharing a 1/1024 cell, and nothing measured it.

**#574 proposed cubic cells** (scale all axes by the longest extent), because per-axis cells on a lamp
0.8 m long and 2 cm across let a group of four facets reach half the lamp. **Built, measured at mesh
scale, and reverted.** Cubic groups ARE tighter in space — lamp groups of 32, radius median / max
0.070 / 0.404 m per-axis against 0.013 / 0.018 m cubic — but they wrap further AROUND the cylinder, so
their facets face many ways: median / 90th-percentile normal spread over groups of 32 is **8.8 / 38.5°
per-axis against 51.7 / 162°** cubic (`validation/sozzi_radiation/curve_compactness.py`, deterministic,
analytic 32 x 128 lamp). The "wholly behind these points" tests (`back_faces`, the transfer's
`_columns_in_front`, `lit_blocks`) want a shared facing direction far more than compactness, and
`Outside`'s one-region certificate did not need tighter lamp clusters (the whole lamp is in the chamber).

**Mesh scale** (`field_cost_breakdown.py`, the user's macOS arm64 run, 2026-09-28: 1,635,909 cells,
case `lampWall.stl` 7,516 facets, `Outside` from three hand-typed cylinders, library defaults, one run
each; main `8fc66ac` against cubic merged onto it, `73ebe0e`, local): undecided pairs **760.0M → 798.2M
(+5%)**, tested with padding 901.8M → 904.1M, transfer row blocks **51.8 → 57.9 s**, call less the
hidden-tile count 171.3 → 177.1 s, wholly-blocked 32x32 tiles 50.5% → 38.6% of undecided pairs. A
second cubic run (`020ae40`, which lacks #585/#586) gave the same counts and 57.9 s. On the sampled
scene (`body_culling.py`, Linux, 4 cores, 208.9M pairs) certified share moved 91.7 → 91.8% and time
within noise. **No gain anywhere, and a loss in the build — do not retry cubic cells as a fix for
elongated lamps.** Grouping by facing direction as well as position is a separate, open idea: #589.
Pinned: `tests/unit/test_morton.py`'s `test_a_run_around_a_long_thin_tube_faces_one_way` and
`test_stretching_one_axis_does_not_change_the_order` are both red under cubic cells.

## SHAFT CULLING: BUILT as `ShaftCulling` — tiles certified clear, THE DEFAULT since 2026-09-26 (#554)

`culling.py` holds the bodies' layer's strategy family, `BodyCulling.blocked(bodies, sources,
near, receivers, pair_limit, *, facing=None)` (`facing` since 2026-09-27, see BACK FACES below): **`EveryPair`** (the reference, and the default until 2026-09-26 — the old
`visibility._blocked_by` loop, moved here with `_compiled_blocks`, now `_body_blocks` per body) and
**`ShaftCulling(receiver_blocks=(32, 8, 2), source_clusters=(32, 8, 2))`** — coarse-to-fine
group-size ladders since #554 phase B (default chosen by measurement, below); ⚠️ **there is no `receiver_block` / `source_cluster` any more**, and the
first version's default was one level at 32 x 32. **An unset `body_culling` is `ShaftCulling()`**
(the one line is `_unchecked_visibility`'s, so a built mask, a streamed chunk and both of a model's
masks all get it; pinned by `test_an_unset_strategy_is_shaft_culling_for_a_built_mask_and_a_streamed_one`).
⚠️ **Under a trace `ShaftCulling.blocked` hands off to `EveryPair`** (any leaf of bodies, sources,
near or receivers a `jax.core.Tracer`): its grouping and certificates are numpy host work and raised
`TracerArrayConversionError` on a mask built inside `jax.grad` of a body's radius — found by
`test_the_gradient_with_respect_to_a_body_s_GEOMETRY_is_exactly_zero` when the default flipped, and
pinned by it (red without the hand-off). Same mask, zero derivative either way; the model's masks are
always built from concrete geometry, so the model path never takes the hand-off.
Chosen by `build_visibility(..., body_culling=)` or `RadiationSettings(body_culling=)` — pass
`EveryPair()` for the unculled reference — which feeds **both** masks a model builds (and
survives `receiver_occlusion` overriding the self-occlusion half). Streamed masks get it through the
same options dict, but then the **receivers'** grouping is per streamed chunk, over whatever order the
receivers arrive in. **The facets' side is formed once per call** (2026-09-26):
`BodyCulling.prepared(bodies, sources)` (default: itself) returns, for `ShaftCulling`, a copy carrying
`sources=_Groups` — the facets' curve order and every body's clearance summary of every cluster at
every level, formed by `_Groups.of` — and `streamed_fluence_rate` calls it beside
`self_occlusion.prepared`, through `culling_or_default` (the one home of "unset means
`ShaftCulling()`", also used by `_unchecked_visibility`). `_Groups` is a frozen `eq=False` dataclass of
host arrays, not a pytree, like `TriangleGrid`, so it rides through the chunk's custom VJP untouched.
⚠️ **A prepared strategy refuses other sources or other bodies** (`_Groups.serves`, by VALUE: the
sources by `array_equal`, the bodies leaf by leaf on the host after an identity shortcut — a body
comes back from the custom VJP as a new object, so identity alone refused the stream's own bodies;
`eqx.tree_equal` was avoided because it dispatches an eager device op per leaf, once per chunk). Measured
on the analytic Sozzi-like scene (24,000 sampled receivers, 8,704 facets, 53 chunks, `NoOcclusion`,
default ladder; jax 0.10.2, Linux x86_64, 4 cores, one process, three alternating passes): streamed
field median **21.96 → 20.67 s** (~6%), fields bit-identical — in line with the ~14 s of 317 s the
Sozzi whole-field breakdown (#564) charged to per-chunk facet clearance and summaries.
**The leftover tiles' points are gathered inside the compiled test** (2026-09-26,
`_compiled_tile_blocks`): a traceable body gets the points whole (converted to device arrays once per
`_test_tiles` call) and each tile's `rows`/`cols` indices, instead of every tile's points gathered and
copied on the host before the call; a host body (`TriangleBody`) gets its pairs' points gathered, as a
flat list (`_walked_tile_blocks`, below). The
write-back into the mask stays a numpy scatter — **scattering on the device inside the same program was
measured and was slower** (median 1.89 against 1.63 s on the same tiles, no buffer donation). Measured on
the leftover tiles of 4,000 sampled receivers x 8,704 facets at the default ladder (720,086 2x2 tiles; 4
cores, alternating): tile test median 1.86 → 1.41 s in one run, 1.63 → 1.52 s in another; whole streamed
field (24,000 receivers, 53 chunks, four alternating passes) median **27.37 → 26.53 s (~3%, inside this
container's spread — one pass reversed)**. The stand-in tests only ~9% of its pairs; on the Sozzi mesh
the leftover test is 59% of the call (#564), so the saving there should be larger — **not measured**.
Fields bit-identical. Mutation: a reversed tile order goes red; handing each pair a neighbouring
facet's `near` (1e-6 x sqrt(area)) survives — **dismissed**, the margin is too small to change an answer
on any fixture, as it was for the host path. #564's harness times `_compiled_tile_blocks` as `test
tiles / compiled test` and `_walked_tile_blocks` as `test tiles / walked test`.
⚠️ **A body's certificate is asked on batches padded to a power of two** (2026-09-27, `_vouches`, used by
`_vouched` and `_vouched_pairs`; since 2026-09-28 it pads the tiles' *indices* and asks
`Body.vouches_tiles`). `Body.vouches` is, by default, a few **eager** `jnp` operations, and an
eager operation compiles once per shape: on the analytic `Outside` a batch length never seen before cost
**111 ms** against **0.6 ms** for a repeated one (median of 20, 20,000 tiles, 3 features; jax 0.10.2,
Linux, 4 cores). The tiles asked about change with every pass and level — more so since #578 drops the
tiles behind their facets before asking — so nearly every call compiled. Padded by repeating the last
tile (answers dropped), the lengths come from a short ladder. **Found because #579 measured the
certificates at 53.5 s on the Sozzi mesh against #564's 14.3 s** while the tiles they were asked about
had fallen; `validation/sozzi_radiation/certificate_levels.py` counts calls, tiles asked and refused, and
seconds per level (and #578's behind-the-facets drops), at any commit. On the analytic stand-in (24,000
sampled receivers, 8,704 facets, default ladder, fresh process each, two runs per arm): certificates
**3.13–3.43 → 1.29–1.33 s**, level 1 alone 1.60–1.69 → 0.31–0.33 s, identical tile counts at every level.
**On the mesh** (`field_cost_breakdown.py`, THE CALL ON `8fc66ac` below): the cylinders' certificates
53.5 → **4.2 s** (main `bb573d5` + #581) and 5.1 s on `8fc66ac`, identical tile counts.
Pinned by `test_a_body_is_asked_to_vouch_only_at_a_few_batch_lengths` (red without the padding, and red
keeping the wrong end of the padded answers). **A body that is not `traceable` is asked unpadded** — see
the next paragraph. On the same stand-in the first call of #578's
`BackFaces.tiles_behind` costs ~2 s — its Numba loop compiling — once per process.
⚠️ **A host-answered body is handed no padding, and no pair behind its source** (2026-09-28,
`_test_tiles` / `_walked_tile_blocks`, `_vouches`). Both paddings above exist so that *compiled* work
reuses its programs. A body that is not `traceable` compiles nothing, and `TriangleBody`'s grid walk has
no notion of a segment already walked, so every repeated tile was walked for real. The Sozzi
triangulated-wall run (1,635,909 cells, main + #580 + #581) had **4,866,359,296** pairs tested against
**3,654,843,680** in undecided tiles: 25% of the walk was padding, and the walk was 1,148.7 s of a
1,465 s call. Now, for such a body, the behind test still runs on the padded batch (it is a jitted
`BackFaces.behind`, so an unpadded batch would compile per shape), and the walk is handed only the
batch's own pairs not behind their source, as one flat segment list; those behind are written clear, as
before. Its certificates are asked unpadded (`TriangleBody.vouches` is numpy). Traceable bodies are
unchanged. ⚠️ **This ties "not traceable" to "answers on the host"**, which is what `Body.traceable`'s
own comment says it means; a bespoke body written in eager `jnp` and left at the default would now
compile per batch shape — still correct, only slower. Measured on a stand-in (the generated 64 x 400
triangulated chamber of `triangle_culling.py` plus four triangulated 4 mm rods at radius 25 mm, the
lattice slab at 12 mm (9,471 receivers), the analytic 24 x 64 lamp (3,360 facets), `BackFaces` of the
lamp, default ladder; jax 0.10.2, Linux x86_64, 4 cores, two processes per arm, three passes each):
segments walked **8,388,608 → 6,303,619** — of main's, 1,659,372 were padding and 425,617 real pairs
behind their facet (6,729,236 real pairs undecided) — and warm passes **14.05–16.06 s → 7.42–9.12 s**,
masks bit-identical. The time fell more than the count; that was **not decomposed**. **On the mesh**
(THE CALL ON `8fc66ac` below): walked 4,866,359,296 → **3,458,668,522** pairs, the walk 1,148.7 →
**864.8 s** (1.33x; rate unchanged at ~4M pairs/s). Pinned by
`test_a_host_answered_body_walks_each_lit_pair_once_and_is_asked_about_no_padding`: red for walking
padding (a pair walked twice), for walking pairs behind their source, and for padding the certificates.
`field_cost_breakdown.py` now counts what each test is handed — `tested by a compiled body, padding
included`, `walked by a host-answered body`, and `behind their source, inside tiles a host-answered body
tested` — instead of re-deriving the padding from the batching arithmetic.
**The triangle body's certificate is one compiled loop, and never forms a tile's summary** (2026-09-28,
`TriangleBody.vouches_tiles` → `TriangleGrid.holds_any_in_unions`, `grid._unions_held`). The numpy
certificate built each batch's merged summaries (`np.maximum` over the tiles), then clipped, truncated
and looked them up in `occupied_below` a whole-array pass at a time, single-threaded. Now a
`numba.njit(parallel=True)` loop over the tiles takes the two groups' boxes, forms their bounding box in
registers, and makes the eight lookups. `holds_any` runs on the same loop (each box paired with itself),
so the truncation, the margin and the table's corners have one home, `_voxel_span` and `_unions_held`.
**Profiled first** on the stand-in above, widened to a 30 mm slab (21,648 receivers streamed in 19 chunks
of 1,190, `prepared` culling, back faces; second pass of a fresh process; jax 0.10.2, numba 0.67.0,
Linux x86_64, 4 cores): the certificates were **0.97 s** of a 17.4 s mask, of which `holds_any` was
**0.71 s** (~144 ns a tile over 4.93M tiles) and building the merged tiles ~0.2 s. Afterwards, two
processes per arm: certificates **0.97–0.99 → 0.05–0.08 s**, the whole refinement (`_undecided`)
**1.50 → 0.56–0.62 s**, masks bit-identical (and the earlier 12 mm probe's too). The pair tests are
~90% of this stand-in, so the whole mask moved only 16.1–17.4 → 15.4–15.8 s. On the mesh the
certificates were 125.2 s of 1,465 s, and on `8fc66ac` **42.7 s** of 1,098 s (THE CALL ON `8fc66ac`
below). Pinned by
`test_the_compiled_box_test_gives_the_answer_the_array_passes_give` (the box test against an
independent whole-array one, on boxes whose faces sit on, a rounding off, or far from voxel edges),
`test_a_union_of_two_boxes_is_tested_as_the_box_that_bounds_them` (some unions hold what neither box
does) and `test_a_body_vouches_for_tiles_as_it_does_for_their_merged_summaries`. Mutation-checked: no
low-side margin, a missing `+ 1` on the last voxel, a wrong table corner, a union that ignores one
box's low corner on one axis, `rows` passed for `cols`, padding a host body's certificates, and not
padding a traceable one's — each red.

**How.** Receivers and facet centroids are each ordered along a Morton curve (`aquaflux.morton.morton_order`, 10 bits
an axis, each axis scaled to its own extent — cubic cells were measured and rejected, #574) and padded to a whole number of the coarsest groups by repeating the last point (`_Curve`;
repetition changes neither a max-summary nor a written answer). Each size in a ladder divides the one
before, so a group at one level is a whole number of groups at the next, read off the same order. Per
body, each group is summarized by the column-wise max of `body.clearance`, and a tile is **certified
clear** where `body.vouches(max(receiver summary, source summary))` (the contract is in
`.claude/rules/solids.md`). **A tile refused at one level is split into its children at the next**
(`_undecided`; children made wholly of padding are dropped, not asked) and asked again; only tiles
still refused at the finest level are gathered into batches (`pair_limit // (block * cluster)` tiles)
and answered by the same `body.blocks` the reference uses — a traceable body compiled, on the batch
padded to a power of two with `padded_length`; a host body on the batch's own pairs not behind their
source — so the two agree **bit for bit** on every pair either tests. Host
orchestration, compiled leaf — the `TriangleGrid` lesson. `certified_pairs` is every pair less those
in the finest refused tiles, so it counts what refinement vouched for at any level. A body with no
features is tested everywhere.

**Scope.** Phase A (analytic bodies, one level) and phase B (triangle bodies via #510's
`TriangleBody`, plus refinement for every body) are built; both give "clear" certificates only. Not
built, tracked in #554: **C** "fully hidden" certificates, which would also let the gather skip dark
tiles; **D** distance-based level of detail for the gather, which changes answers and needs its own
error measurement. ⚠️ **The facet-to-facet ray mask (`RayCastOcclusion`, self-occlusion) is NOT
culled**: its shafts start on the wall they are tested against, so every tile touches the wall's own
triangles and no box certificate can vouch for one — that needs per-ray exclusions carried into the
certificate, and is not designed. The ordering is the neutral leaf `aquaflux/morton.py` since #574; there is no
`culling.spatial_order` any more.

**MEASURED** (`validation/sozzi_radiation/body_culling.py`, all arms in one process on one ray set,
warm-up then two alternating passes, fastest kept): `Outside(chamber, inlet, riser)` at the
tutorial's dimensions, the **analytic 32 x 128 lamp (8,704 facets)** and **24,000 receivers sampled
uniformly inside the three cylinders** — ⚠️ **the case mesh was absent, so this is NOT the cell-centre
population** the 81.1% above is measured on (deepest-inside region 22,034 chamber / 1,013 inlet / 953
riser); 208.9M pairs, 3.90% blocked; offset scale 1e-6; jax 0.10.2, CPU, x64, **Linux x86_64, 4 cores**
(a cloud container, not the usual macOS arm64 machine), run directly with output redirected (not
through `run_case.sh`, which needs `vm_stat`), 2026-09-25, uncommitted #554 tree on `0c3a78c`:

| arm | certified | fastest s | spread | vs every pair | mask |
|---|---|---|---|---|---|
| `EveryPair` | — | 42.63 | 1.01x | 1.00x | reference |
| 16 x 16 | 91.0% | 5.22 | 1.06x | 8.17x | identical |
| **32 x 32** (the phase-A default; one level) | **90.3%** | **4.58** | 1.01x | **9.30x** | identical |
| 64 x 64 | 89.1% | 5.49 | 1.08x | 7.77x | identical |
| 32 x 128 | 90.3% | 4.80 | 1.02x | 8.88x | identical |

- **The speed-up is close to its ceiling**: with 9.7% of pairs left to test, pure savings would be
  ~10.3x; 9.3x means the grouping, summaries and scatter cost ~10% of what remains. Group size barely
  matters between 16 and 128 — the certified share moves 89-91% — so the default was not tuned further.
- ⚠️ **Less on the mesh's cell centres, as predicted** (the mesh refines towards the walls and the lamp,
  where the uncertifiable pairs are; one-convex-region share 81.1% there against ~91% here). The
  whole-field figure is below, under MESH SCALE; it times the field, not the mask alone.
- The `EveryPair` rate here (4.9M pairs/s) is ~4x below the 20.7M recorded for the same test on the
  11-core machine; it is the within-run ratio that is the finding, not either rate.

**MESH SCALE, and why it became the default (2026-09-26).** `validation/sozzi_radiation/
model_at_mesh_scale.py` (its `SOZZI_CULLING` switch picks the arm), the public model on all 1,635,909
cell centres, the case's 7,516-facet `lampWall.stl`, hand-typed `Outside(chamber, inlet, riser)` (no
CAD kernel installed), `NoOcclusion`, `stream_receiver_mask=True`, black walls; one arm per process, in
sequence through `run_case.sh`, jax 0.10.2, CPU, x64, macOS arm64, 11 cores, 19 GB; main `e12f214`.
Field seconds: `EveryPair` **471.7**, `ShaftCulling()` (32, 8, 2) **313.5** (1.50x), (32, 8) **251.9**
(1.87x); fields identical to 4.4e-16; `EveryPair` and (32, 8) repeated at 471.5 and 251.4 in an
earlier run of the same day. The pre-speed-up code (`99c472c`) took **1138.4** s at the same settings,
so main's unculled path alone is 2.41x. The default flipped on this measurement: the mask cannot
change, the cost falls on every scene with open space, and a scene with no body pays nothing (no
bodies, no call). ⚠️ **The default ladder's third level still costs 1.24x on an analytic-only scene**
— the phase-B finding, now confirmed on the real population — and is kept for the ~2.7x it buys on a
triangle wall. A per-body ladder was investigated and deliberately **not built** — see "WHY THE
FINEST LEVEL COSTS AN ANALYTIC BODY" below. Full table in the Sozzi README.

**WHERE THE CALL'S TIME GOES under the new default (2026-09-26, `validation/sozzi_radiation/
field_cost_breakdown.py`)**, same configuration as MESH SCALE, main `392f935`; call 316.8 s
instrumented (313.5 plain), repeated to 0.3 s. **Masks 243.9 s (77%)**: pair tests of uncertified
tiles 186.7 (compiled test 141.8 in 695 calls + 44.8 host gathers/padding/scatter), certification 45.5
(lamp-facet clearance 12.6 — **recomputed identically every one of 3,076 chunks** —, receiver
clearance 12.1, `_vouched`/`_vouched_pairs` 14.3), curve order 3.2, mask alloc/convert ~8.5. **Gather
66.5 s (21%)**; radiosity 1.5. Build: `_row_blocks` 56.4 s, facet mask 0.2 s (100% certified).
**Certified 81.0% of 12.3G pairs = the one-convex-region share (81.1%)**, so clearance certificates are
at their ceiling here and only a "fully hidden" certificate (phase C) can cut the tested count; padding
adds only 9.6% (2.33G → 2.56G). ⚠️ **The leftover test runs at ~18M pairs/s on 2x2 tiles against ~30M
for `EveryPair`** (separate processes, approximate) — the concrete levers are a denser layout for the
leftover pairs plus batched host work (up to ~100 s; ⚠️ the "denser layout" half is refuted — see WHY THE
FINEST LEVEL COSTS AN ANALYTIC BODY) and caching the source side per call (~14 s, **built since** as
`BodyCulling.prepared`, see the SHAFT CULLING section — this table predates it; the harness now times
that work once under `call / field / prepare culling` and the receivers' side under
`bodies / receiver groups`). Full table in the Sozzi README.
**Phase C's ceiling, measured (2026-09-26, same harness and configuration, a second run; the other
pieces repeated to 0.4 s, call less the count 317.7 s).** Blocked pairs 1.185G = **9.6% of all, 50.8% of
the 2.33G in undecided tiles**; pairs in wholly blocked tiles along the strategy's curve: 32x32
985.7M, 8x8 1,121.9M, 2x2 1,173.6M = **42.3% / 48.1% / 50.3% of the undecided pairs**. So a "fully
hidden" certificate could at most halve what is tested — worth roughly 70-90 s of the call (half the
compiled test and its host work, plus the gather skipping dark pairs). The other half are clear pairs
that **today's** certificate cannot vouch for: it proves a tile clear only inside one convex region
(81.0% certified against the 81.1% one-region share), so what is left are pairs crossing between
regions — chamber to inlet or riser. ⚠️ **They are not beyond any certificate.** Whether a crossing
pair is clear or blocked depends on whether it passes through the **port** where the pipe meets the
chamber, so a port-plane certificate — the shaft crossing the port's plane wholly inside the opening
(and each side's part inside its own region) is clear, wholly outside it is dark — could decide
tiles of **both** kinds, with a ceiling of most of the ~187 s leftover-pair test rather than half of
it. Not designed or built; the record of what #554 phase C is worth. An upper bound on what
could be skipped, not a prediction of what a certificate would prove. (The count itself cost 73.1 s,
harness time, timed outside the bodies' layer.)

**THE SAME CALL AFTER #575 AND #578, MEASURED (2026-09-27, same harness and configuration, main
`2810eea`, one run, counting on; macOS arm64, 11 cores, jax 0.10.2, numba 0.67.0).** Call less the
count **210.6 s against 317.7** (1.51x, one run against one, a day apart). Masks less the count 147.4 s
(70%): pair tests 64.7 (compiled 51.4 in 604 calls), undecided-tile decisions 65.5 (certificates 53.5,
tiles behind 5.4), receiver groups 13.2 (clearance 12.2); lamp-facet clearance ~0 (formed once per call
since #575). Gather 50.2 s (areal segments 44.1, layout 5.9). Build 57.3 s, unchanged. **Undecided
pairs 760.0M = 6.2% of all, against 2.33G (18.9%)**; tested with padding 901.8M. Blocked 562.1M (4.6%) —
halved because pairs behind their facet are now recorded clear, not because geometry changed. Wholly
blocked tiles: 383.8M / 506.2M / 550.3M at 32x32 / 8x8 / 2x2 = **50.5% / 66.6% / 72.4% of the undecided
pairs** (was 42.3 / 48.1 / 50.3%). ⚠️ **So the mask is now split evenly between DECIDING tiles and
TESTING them** (65.5 against 64.7 s): the certificates, not the pair test, are half of what is left, and
phase C's whole target is now ~65 s of test. **Harness fix in the same change**: #578 gave
`ShaftCulling._test_tiles` a `facing` argument and the harness's counting wrapper still took the old
signature, so the harness crashed on `main` ("got multiple values for argument 'out'") — the
monkeypatched-private-method hazard: nothing in any test tier runs this harness. Table in the Sozzi README.

**THE CALL ON `8fc66ac` (#580-#586), MEASURED (2026-09-27, same harness, whole mesh, default culling,
counting on, both waters back to back; macOS arm64, 11 cores, jax 0.10.2, numba 0.67.0; "before" is
main `bb573d5` with #581 merged, a few hours earlier; one run each, so ~1.1x).** **Cylinders: 171.5 s
against 161.0 — no clear change**; pairs identical (760.0M undecided, 6.2%); pair tests 65.2 (compiled
52.9), undecided-tile decisions 17.3 (certificates 5.1, against 53.5 on `2810eea` — #580), receiver
groups 13.5, areal gather 50.8 (44.0 before, not separated from the spread). **Triangulated wall
(`bodyWall.stl`, walk grid 102³): 1,098.2 s against 1,465.1 (1.33x).** Walk **864.8 s** against 1,148.7
(79% of the call), host work around it 66.4, certificates **42.7** against 125.2 (#586), rest of the
decisions 39.6, gather 55.8, build 53.0. Pairs walked **3,458,668,522** against 4,866,359,296 with
padding (#585: a host body walks no padding and no pair behind its source; 196.2M behind written clear
unwalked); undecided unchanged at **3.65G = 29.7%**; blocked 568.9M either way. ⚠️ **The lever for a
triangle body is now CERTIFICATION, not the walk**: 29.7% of pairs undecided against the cylinders'
6.2%, walked at an unchanged ~4M pairs/s; a fully-hidden certificate reaches at most 15.1% of the
undecided pairs (2x2). Tables in the Sozzi README.

**Tests** (`tests/unit/test_radiation_culling.py`, each mutation-checked): bit equality with
`EveryPair` at group sizes 32x32, 7x5, 1x1, 64x3 on a scene where every one of four body kinds
(`Outside` chamber+pipe, `Box` baffle, `Sphere`, `Difference` ring) blocks some pairs and each has
certified tiles; certified ⊆ clear; an all-open chamber certified exactly `n_r x n_f` with padded
groups (a padding-counting `certified_pairs` goes red); one-tile batches with padding; a witness-less
eager body; the settings plumbing; Morton z-order on a cube's corners; locality; group padding.
**Mutation pass (9, 8 red):** tile min for max, group min for max, a transposed write-back, padding
counted, identity order, swapped bit interleave, testing the certified tiles instead of the rest,
padding with the first member. **Dismissed:** one extra tile per batch (`per_batch + 1`) — it changes
only how many tiles one compiled call holds, i.e. memory, never an answer.

### PHASE B (#554, 2026-09-26): refinement, and triangle bodies vouching through an occupancy grid

**Decided with the user before building** (three questions, all answered with the recommended
option): #510 goes in the same change; the triangle certificate is **bounding box + refinement**
(not box alone, not a k-DOP); and the clearance contract is **generalized** (`Body.vouches`, see
`solids.md`) rather than special-casing triangles inside the strategy.

**Why refinement, measured before building** (a scratch probe, since folded into the harness below):
the plain box certificate over the walk grid vouched for **44%** of chamber pairs at 32x32 groups on a
64 x 400 triangulated chamber, against **100%** for the exact convex-region test on the same pairs —
a diagonal shaft's box pokes out through a round wall whatever the voxel size (59% at ~3.6 mm and
still 59% at ~1.8 mm), and only smaller groups move it (65-81% at 4x4).

**The occupancy grid is not the walk grid.** `TriangleBody.occupancy` is a second `TriangleGrid` at
`_CLEARANCE_REFINEMENT = 4` times a **near-cubic** ten-a-voxel grid per axis
(`grid._near_cubic_resolution`, halved as a whole past `grid._MAX_VOXELS`), read only through
`holds_any` (a summed-volume table, `occupied_below`, eight lookups per box). A certificate wants
voxels small enough that one beside a wall is not occupied — on the test fixture the walk grid was
2x2x2 and vouched for **nothing** until the occupancy grid was split out. ⚠️ **It is sized from the
near-cubic grid, NOT from the walk grid, since #503 (2026-09-27)**: the walk's default voxels now take
the box's proportions (17 x 0.9 x 9.2 mm on the Sozzi wall), and a long voxel beside a wall is occupied
along its whole length. Deriving it from the walk grid as before would have changed which pairs are
certified along with the walk; derived this way, the Sozzi occupancy grid is the same (424, 22, 228) it
was, so certification is unchanged there. An explicit `resolution=` no longer moves it either
(`test_the_grid_a_body_vouches_from_is_near_cubic_whatever_the_walk_grid`).

**MEASURED** — `validation/sozzi_radiation/triangle_culling.py`: the chamber alone as a closed cylinder
of **51,328 triangles** (64 sides x 400 slices plus caps, wound to face the water — the case's
`bodyWall.stl` was absent, so **no pipe openings and no shadows**: 0.00% of pairs blocked); walk grid
(125, 12, 12), occupancy (500, 48, 48); analytic 24 x 64 lamp (3,360 facets); **5,476 receivers on a
2 mm lattice in a 6 mm slab at mid-chamber** (dense on purpose — a block of 32 is only compact where
receivers are, and a scattered sample of the same count would understate every arm); 18.4M pairs; one
process, warm-up then two alternating passes, fastest kept; jax 0.10.2, CPU, x64, Linux x86_64, 4
cores, run directly with output redirected, 2026-09-26, uncommitted phase-B tree on `3e54fe1`:

| arm | certified | fastest s | spread | vs every pair | mask |
|---|---|---|---|---|---|
| `EveryPair` (grid walk) | — | 227.4 | 1.06x | 1.00x | reference |
| 32x32 | 37.0% | 206.9 | 1.00x | 1.10x | identical |
| 32 → 8 | 70.9% | 123.3 | 1.03x | 1.84x | identical |
| **32 → 8 → 2** (the default) | **89.3%** | **45.8** | 1.00x | **4.96x** | identical |

And the analytic `Outside` harness (`body_culling.py`, same configuration as the phase-A table above,
run straight after, same machine) at the same ladders: 32x32 **7.84x** (90.3%), 32 → 8 **8.53x**
(91.3%), 32 → 8 → 2 **5.35x** (91.7%), all identical. ⚠️ **So the best ladder depends on the body's
per-pair cost**, and neither default is right for both: the third level saves ~2.7x on the triangle
walk and costs ~1.6x on the cheap analytic test, whose finest-level tile bookkeeping outweighs the few
pairs it vouches for. **The default is (32, 8, 2)**, chosen on absolute time — it saves minutes where
the triangle walk dominates (62 min for the whole Sozzi field since #571, TRIANGULATED WALL AT MESH
SCALE below) and costs ~3 s where the analytic mask already runs in seconds — and the docstring says
to stop at 8 for analytic-only scenes. **A per-body depth was investigated and deliberately NOT built**
(decided with the user, 2026-09-26, after the findings below; three mechanisms were offered — two ladders
keyed on `traceable`, a per-ray cost declaration on `Body`, a per-level cost model — and all were
declined). ⚠️ Every triangle figure is on a shadowless chamber:
on the real wall, pairs crossing a pipe opening can never be vouched for, so the share falls with the
pipe-cell fraction; **re-run with `work/case` present before quoting a mesh-scale figure**, and note
that both 32x32 figures here and in the phase-A table were ONE level.

**WHY THE FINEST LEVEL COSTS AN ANALYTIC BODY, AND WHY NO PER-BODY DEPTH (2026-09-26).** Same analytic
`Outside` scene and configuration as the phase-A table (24,000 sampled receivers, 8,704 facets), same
triangle chamber as above; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, scratch probes run directly on
`main` at `e12f214`, all masks identical to `EveryPair`. This container's timings vary by up to ~1.5x
between runs, so every comparison below is within one process, alternating arms.
- **The certificates are not the cost**: `_undecided` takes 0.06 s at (32, 8) and 0.43 s at
  (32, 8, 2). **The compiled pair test is**: inside real `blocked` calls it took 4.9 s at (32, 8) against
  7.4 s at (32, 8, 2) for about the same number of tested pairs (18.1M against 17.3M), and the third
  level vouches for only 0.4% more of all pairs.
- **The per-pair cost of the compiled test rises on tiny tiles.** 4M random pairs, one compiled
  `Outside` test, alternating shapes: median **278–309 ns/pair as (T, 2, 2) tiles**, against 203–229
  at 8x8 and 194–213 at 32x32 (two runs). XLA on CPU vectorizes poorly when the axes before the
  coordinate are tiny.
- ⚠️ **Two fixes tried and refuted, so do not re-try them without a new reason:** (a) testing the
  undecided pairs as a **flat pair list** (gathering both endpoints per pair) was slower at every
  ladder, 365–432 ns/pair against 213–340 for tiles; (b) putting the **tile index innermost**, shape
  (b, b, T), made no difference at 2x2 (median 298 against 293 ns/pair). A split of a tile test's time
  into gather / compiled / write-back put gather + write-back at only 13–43 ns/pair.
- **One ladder cannot serve both body kinds**:

  | ladder | analytic `Outside`, full build (median of 6) | triangle chamber (fastest of 2; certified) |
  |---|---|---|
  | (32, 8) | 6.1 s | 110.8 s (70.9%) |
  | (32, 8, 4) | **5.65 s** | 94.3 s (81.0%) |
  | (32, 8, 2) — the default | 8.7 s | **39.1 s** (89.3%) |

- **So the default stays (32, 8, 2) and nothing chooses per body.** The finest level costs the cheap
  analytic mask seconds (~2.6 s here; proportional to tested pairs on a mesh) and saves the triangle
  walk — which dominates a triangulated field — a factor of ~2.4 over the next best ladder. Every
  mechanism for choosing per body carries a cost (`traceable` standing in for "cheap per ray", a
  contract term only the culling reads, or a machine-measured cost constant) out of proportion to the
  seconds it would save. **At mesh scale the penalty is ~62 s of a 314 s whole field** (the 1.24x
  above: `ShaftCulling()` 313.5 s against (32, 8) 251.9 s, all 1,635,909 Sozzi cells, macOS arm64,
  11 cores) — seen by the user before this was decided. A caller whose scene holds analytic bodies
  alone passes `ShaftCulling(receiver_blocks=(32, 8), source_clusters=(32, 8))` and gets it back.

**`TriangleBody`** (`triangle_body.py`, #510; exported, in radiation beside the grid because
`solids/` may import nothing outside itself). `build(vertices, *, sheet=None, resolution=None,
clearance_resolution=None, tolerance=None)`. `blocks` broadcasts to the grid walk (`traceable=False`).
**`contains` is decided per connected piece** (`checks._surface_pieces`, split out of `open_facets`):
an open piece is a sheet with no inside; a closed piece is solid on the side its normals point away
from — the emitting-surface convention — so an outward-wound piece (positive signed volume) is a lump
and an inward-wound one is a vessel, solid outside it. One formula covers both and nesting:
**solid where `enclosure_winding(closed pieces) + n_inward > 0.5`** (outward winding reads +1 inside,
inward -1, measured). Points outside a closed piece's bounding box skip that piece's winding — done
by `enclosure_winding` itself since #530 (merged into this branch; `TriangleBody` had its own global-box
shortcut, deleted as a duplicate). `sheet=True` overrides to no inside (a welded sheet reads closed); `sheet=False` refuses
an open piece; a closed piece enclosing no volume (a triangle and its reverse) is refused rather than
guessed. `grid_mask_check.py` now uses `TriangleBody.build(wall, sheet=True)` in place of its local
`WallTriangles` (identical answers by construction: same grid, same all-False `contains`; **not
re-run here**, the case is absent). **Not done from #510's list**: the CAD reader's triangle fallback
(`io.md`).

**Tests** (`tests/unit/test_radiation_triangle_body.py`; culling ladders in
`test_radiation_culling.py`): the walk against `segment_is_cut` over broadcast pair shapes; `contains`
on water / sleeve / just-inside-the-wall / metal (the last outside the bounding box), and a drum wound
inward as a cavity; an open vessel as a sheet; both overrides; the flat refusal; `holds_any` sound
against points **sampled on the triangles** over 3,000 boxes (and >300 of them empty, so it is not
vacuous); `vouches` sound against the brute test on compact clouds; a `Box` primitive and a
`TriangleBody` in one scene, culled and not, identical; refinement vouching for more than one level
with the mask unchanged; bit equality at ladders (32,8) and (24,6,1)/(8,4,2). **Mutation pass (15, 13
red):** inward count dropped, the bounding-box shortcut forced (since deleted — see above), volume sign flipped, open pieces treated
as closed, flat refusal off, box corners swapped, features `[x, x]`, occupancy = walk grid, the `+ 1`
on the box's far index, one inclusion-exclusion sign, children not scaled, refinement skipped,
refused filter dropped. **Dismissed:** the box margin at zero (it guards a rounding no fixture can
reach; kept), and dropping the all-padding child filter (such tiles repeat real points, so testing
them gives correct duplicates and costs only work).

## BACK FACES: THE BODIES' LAYER SKIPS PAIRS BEHIND THEIR SOURCE (2026-09-27, from #565)

**Exact, no tolerance — chosen over #565's cluster approximation by the user.** #565 measured an
error-bounded cutoff at a ~10x oracle ceiling, but it needs an accuracy tolerance the user did not
want exposed (it is not dimensionless in any natural way). A pair whose volume receiver lies behind a
facet that is `dark_behind` carries exactly zero, so whether a body blocks it cannot change the field.
#526 already skipped such pairs in the **ray mask**; this does the same in the **bodies' layer**, which
is where the mesh-scale call spends its time (masks 77% of the 317 s Sozzi field, #564). Agreed order
with the user: the mask first, then the gather (built since, see BACK FACES IN THE GATHER below).

**How.** `BackFaces.of(surfaces)` holds the facets' centroids, normals and an `areal` label (point
sources have no back). `_unchecked_visibility` passes it as `facing=` to `BodyCulling.blocked` exactly
when #526's cut applies — `receiver_facet is None and surfaces.dark_behind` — and sets
`Visibility.clear_behind` when there are bodies (the gather's refusal of a non-dark-behind set, #526,
then covers this layer too). Both strategies record a pair behind its source **clear**, so they still
build the same mask: `EveryPair` tests and then clears (`& ~facing.every_pair`); `ShaftCulling` drops,
at every level of `_undecided` (after vouching at the coarsest, before it at the finer ones), each tile
`BackFaces.tiles_behind` proves wholly behind, and clears the pairs behind within the tiles it tests.
Under a trace the hand-off to `EveryPair` carries `facing` (it is in the tracer check's leaves).

**The box test is conservative by construction, not by tuning.** The highest point of a receiver box
above a facet's plane is the corner the normal points towards; a tile is proven only if that corner is
below the plane by `_BOX_SLACK` = 3 × `clipping._SLACK` units of roundoff of `Σ reach·|n|`, with `reach`
the box's farthest extent from the centroid per axis. That bounds every pair's own snapping allowance
(16 units of `Σ|x−c||n|`) plus both rounding errors, so a proven tile's every pair is behind by the
pair test too — verified, not only argued: `test_a_tile_is_proven_behind_only_where_every_one_of_its_pairs_is`.
⚠️ **The first box test was numpy and cost more than it saved**: 2.4 s of a 4.6 s build (fancy
indexing over ~4.3M finest tiles). It is one `numba.njit(parallel=True)` loop over tiles
(`_boxes_behind`), stopping at a tile's first unproven facet: 2.74 → 0.44 s for `_undecided`. The
grid walk's lesson again — a per-item loop whose value is what it skips wants compiled host code.

**MEASURED** (`validation/sozzi_radiation/backface_share.py`; analytic 32 × 128 lamp, 8,704 facets;
23,046 receivers uniform in the three cylinders **outside the lamp** — 954 inside it removed, since the
vessel body does not exclude them; the case mesh was absent, so this is NOT the cell-centre population;
`ShaftCulling()` at (32, 8, 2); jax 0.10.2, numba 0.67.0, CPU, x64, Linux x86_64, 4 cores, run directly
with output redirected, nothing else running, 2026-09-27):

- **Shares**: 64.9% of all pairs face away from their receiver (what the gather could skip); 8.6% of
  pairs are undecided by the certificates (what is tested), and of those **74.2% face away, 72.9% lie in
  finest tiles wholly facing away, 72.7% in tiles the box test proves** — so the box test reaches almost
  every pair a per-pair skip would, and the ceiling on the tested count is ~3.7x.
- **The bodies' layer, both arms in one process, warm-up then two alternating passes, fastest kept**:
  every pair asked **6.81 s**, pairs behind skipped **2.03 s** — **3.35x** (spreads 1.08x / 1.02x); an
  earlier run of the same harness read 7.22 / 2.01 s, 3.59x, with a 1.28x spread on the first arm. So
  ~3.4-3.6x against the ~3.7x ceiling: the certificates, curve and tile bookkeeping are what is left.
- 4,327,084 pairs the full mask blocks lie behind their source; the skipped mask is the full one with
  exactly those cleared (`array_equal` against `full & ~behind`).
- **Measured since at mesh scale**, on the cell-centre population (THE SAME CALL AFTER #575 AND #578,
  under WHERE THE CALL'S TIME GOES): undecided pairs 18.9% → 6.2% of all, pair tests 186.7 → 64.7 s,
  the whole call 317.7 → 210.6 s with #575 included.
- ⚠️ **On the STREAMED field it is only ~1.1x** (18.6-20.1 s against `main`'s 21.6-22.0 s, same scene):
  the mask is the smaller part of a streamed call, the gather the larger. See BACK FACES IN THE GATHER
  below for the whole-field figures and the split.

**Tests** (`tests/unit/test_radiation_culling.py`), mutation-checked, 10 of 11 red on the shipped code: both strategies at
three ladders equal to the full mask with the pairs behind cleared, and more pairs decided with
`facing` than without; the box test sound on 6,000 tiles with points in the facets' own planes (heights
that round either way) and proving >90% of tiles clearly behind; a built mask clear behind, its
`clear_behind` set, and the field bit-identical to the full mask's; a non-dark-behind profile refused
through it. Red: the box slack at zero, the corner inverted, the box built from one receiver, a proven facet breaking out as proven, `EveryPair` not clearing, tested tiles
not cleared, no tile drop at any level, none at the finer levels, `facing` not passed, `clear_behind`
not set. **Dismissed**: `reach` as the nearer rather than the farther extent — it shrinks a rounding
allowance, and a box whose highest corner lies within a few roundings of the plane while a far receiver
does not is not constructible cheaply; kept as `max` because the soundness argument needs it. Dropping
the `areal` check is inert (a point source's zero normal gives height 0, never proven), kept because a
source's kind is read from its label.

## BACK FACES IN THE GATHER: RECEIVERS IN BLOCKS, EACH AGAINST THE FACETS THAT CAN LIGHT IT (2026-09-27)

The second half of the agreed order (mask first, then the gather). **Exact, no tolerance**: what is
left out is exactly zero, so the field moves only by the order its terms are added in (≤ 5.4e-16
relative, measured below) — a rounding, not a bit-for-bit identity, and tests say so with `rtol`.

**The layout (`lit_blocks.py`).** `lit_segments(points, facets, facing, *, block=8, segments=None)`
orders the points along the Morton curve, cuts them into blocks of `BLOCK = 8`, and lists per block
the facets `BackFaces.lit_facets` does not prove it lies wholly behind (a two-pass Numba count/fill into
compressed-sparse-row lists, sharing `_box_behind` with `tiles_behind`). Blocks are sorted by list
length and cut into **segments**, each padded to `rounded_width` (four steps per doubling, capped at
the facet count) — so a program's shapes come from a short ladder. `LitSegment(rows, facets, valid)`:
a padded block row is `n_points` (the gather's scatter drops it); a padded list entry repeats the
list's last facet with `valid` False. **Without planes, or with traced points, a block is listed
against every facet as ONE shared row** (`LitSegment.shared`) — a copy per block would be ~R/8 × F
indices, gigabytes at mesh scale.

**The gather (`gather.py`).** `summed_fluence_rate(..., layout=None)` = `_point_fluence` (point
sources, dense over their columns, as before) + `_segment_fluence` per segment of each `_ArealGroup`.
A group is the areal facets that every set emits with one profile each (the sets may differ — a
model's reflected set is Lambertian); a group is laid out by its planes only when **every** set's
profile there is `dark_behind`, so there is no refusal path — a glowing group is simply listed in
full. `areal_layout(points, geometry, groups, *, segments=None)` builds it; mask layers are read by
`(row, facet)` gathers, and `surviving_fraction` now broadcasts over any gathering of pairs.
`_emitter_direction(centroid, receivers)` (it was `_emitter_cosine` until the profiles took a direction,
2026-09-28; there is no such function now) and `_transmittance(absorption, source, receivers)`
now take broadcast-ready arrays (the irradiance gather passes `[None]` / `[:, None]`); ⚠️ there is no
`_compiled_gather` any more.

**The streamed path.** Points are ordered along the curve once, then chunked, so each chunk is compact
(and the result is scattered back by the order). Each chunk's layout is built on the host from the
build-time geometry in `_chunk_total` with `_STREAM_SEGMENTS = 4` equal segments per group, and each
piece is a **module-level** `jit`: `_point_part` and `_segment_part(segment, labels=, group=,
pair_limit=)`. ⚠️ **The programs are cached ACROSS CALLS, keyed by `_Labels` — the non-floating part of
the live values, hashed by content (treedef + each leaf's dtype, shape and bytes)**, which is the
cross-call cache the PER-CALL COMPILES section recorded as not done. Without it a stream compiled its
~14 segment shapes every call (3.7 s of a 15 s call here), which is what hid the saving the first
time it was measured (1.03x).

⚠️ **THREE TRAPS, each of which read as "the skip saves nothing"**:
- **`jnp.asarray` inside a trace makes the points a tracer**, so a layout formed after the conversion
  lists every facet. `summed_fluence_rate` now lays out from the points **as given**, and
  `BackFaces.of` keeps concrete positions in numpy (`_kept`). Before the fix the held gather under
  `jit` read 4.3 s for both arms; after, 1.9 s. Pinned by
  `test_points_closed_over_by_a_compiled_function_are_still_laid_out_by_their_planes`.
- **Per-call recompilation** (above).
- **`lax.dynamic_slice` and `jnp.take`'s default `fill` mode each cost ~1.4x** over a clipped gather
  of the same rows, in a probe of the per-pair kernel alone; in the shipped gather `work._slice` as a
  clipped `take` was worth **1.16x without a mask, 1.25x with one** (12,000 receivers × 8,704 facets,
  warm under `jit`, fields identical) — committed on its own (`63db3e8`). Swapping `_slice` for a
  plain `take` measured nothing, because a plain `take` is `fill`.

**A pre-existing bug fixed on the way**: `direct_fluence_rate(..., occluders=...)` could not be
differentiated with respect to emission or a profile — it passed the call's own set as the shadow
geometry, whose traced optics reached the chunk's custom VJP as an argument tangent ("Unexpected
tangent"). It now passes `jax.lax.stop_gradient(surfaces)`; the model's streamed path never had the
problem (it passes its build-time geometry). Pinned by
`test_a_streamed_field_is_the_held_one_and_so_are_its_gradients`.

**MEASURED** (analytic 32 × 128 lamp, 8,704 facets, `UniformAbsorption(35.67)`, `NoOcclusion`, the
water `Outside(chamber, inlet, riser)`, receivers uniform in the three cylinders outside the lamp —
the case mesh absent, so NOT the cell-centre population; jax 0.10.2, numba 0.67.0, CPU, x64, Linux
x86_64, 4 cores, run directly, nothing else running, 2026-09-27):

- **Held gather alone** (12,000 receivers, warm under `jit`, one process alternating): every facet
  listed 4.1-4.5 s, facets behind left out **1.9-2.3 s** (~2.2x; the lists hold 0.47 of the pairs,
  padded to 0.53).
- **Whole streamed field** (`validation/sozzi_radiation/backface_gather.py`, 23,046 receivers, one
  process, warm-up then two alternating passes, fastest kept): every facet listed **13.89 s**, facets
  behind left out **10.69 s** (1.30x, spreads 1.01/1.02x); lists hold 0.421 of the pairs; field max
  relative difference 5.4e-16.
- **Against `main` (`d9f249b`)**, the same call in worktrees, repeat calls: `main` 21.6-22.0 s; the
  mask skip alone (`5fb1384`) 18.6-20.1 s; this tree **10.8-11.1 s — ~2.0x**. Fields: 5.4e-16 against
  `main`; the mask-skip tree bit-identical to `main`. Cross-process, so read the ratios.
- **Where the streamed call's time goes now** (`validation/sozzi_radiation/streamed_cost_split.py`,
  each piece blocked until ready, same scene and machine, two passes after a warm-up): at the default
  chunk (51 chunks of ~460 receivers) mask **3.8 s**, gather segments **8.6-9.0 s**, layout 0.4 s,
  13.4-13.5 s in all; as ONE streamed chunk with the traced chunk left at the default (a diagnostic,
  not a setting) mask 3.1 s, segments **5.9-6.2 s**, 9.4-9.8 s. Blocking each piece serializes what
  the unwrapped call overlaps (host mask work beside the previous chunk's device gather), which is why
  its totals exceed the 10.8-11.1 s above. The mask's chunking is worth under 1 s. ⚠️ **The gap
  between the two segment rows was read as padding and small calls (#577) and is neither**: it is the
  size of one traced step — see A TRACED STEP IS BOUNDED BY CACHE below. Four equal segments pad to
  0.53 and a layout cut by width pads to 0.47, which bought only ~10%.
- ⚠️ **A first profile said the opposite ("the mask is 10.6 s of 11.5") and was the async-dispatch
  trap**: the mask build's `np.asarray` of its inputs waited for the previous chunk's gather, still
  running, and was charged for it. A streamed pass sized separately from the gather's traced chunk was
  built on that reading and measured **no faster** (13.8-14.1 s at 16M and 64M pairs a chunk against
  13.7 s at 4M, more memory), so it was reverted rather than shipped. Raising `pair_limit` to make one
  chunk is not the same experiment: it enlarges the traced chunk too, and the segments then take 15 s.

**Tests** (`tests/unit/test_radiation_lit_blocks.py`, each field against a pair-by-pair numpy sum
that shares none of the layout): rows cover the points once and each list is exactly what the box
test leaves (and leaves out >20% of pairs); a full listing is one shared row; equal segments share a
block count and come sorted; the width ladder; the field with a body, a medium, two profile groups and
a point source at two pair limits; a glowing group listed in full; traced receivers give the eager
field; streamed = held, field and emission / absorption gradients; a stream compiles few segment
programs; the layout under `jit`. `test_the_streamed_passes_share_one_compiled_gather` now also
asserts a second call compiles nothing. The gradient-memory test moved to a 200-facet panel: the
layout keeps ~8.5 B per **receiver** (its index and result, scattered back), which on the old
two-facet panel was the whole figure — measured flat per pair at 200 facets (435 → 469 kB from 64 to
4,096 receivers).
**Mutation pass (14, 13 red):** facing ignored, padding counted valid, padded rows pointing at point 0,
blocks unsorted, the width ladder uncapped, the valid mask dropped from the sum, the scatter clipping
instead of dropping padding, a glowing group laid out by its planes, the streamed order not scattered
back, every set given the first set's profile (caught by `test_radiation_model.py`), the lit count
inverted, the attenuation dropped, and `_Labels` compared by identity. **Dismissed**: `_Labels` hashed by
identity — measured inert: JAX found the cached program for an equal key with a different hash (a same
scene traced once, a changed profile again, either way), so equality is what the cache rests on; the
content hash stays because a hash must agree with equality.

## A TRACED STEP IS BOUNDED BY CACHE, NOT ONLY BY MEMORY: `work.PASS_PAIRS` (#577, 2026-09-27)

`work.in_passes` forms at most **`PASS_PAIRS = 2^16`** receiver-by-facet pairs a step, however high
`pair_limit` is. `pair_limit` is still the memory bound (and still sets the streamed chunk and every
host pass); `PASS_PAIRS` is the traced loop's speed bound. Answers do not change — each receiver's
row is formed from its own inputs either way — and every checksum below agreed across bounds.
Reaches every `in_passes` consumer: the areal segments, the point sources, `direct_irradiance` and the
graded-medium transfer walk.

**Why.** The compiled gather body writes each per-pair intermediate out and reads it back. On this
machine the per-pair rate is ~23-26M pairs/s while a step forms up to ~230k pairs and falls to
~12-15M pairs/s past ~300k (one segment kernel, 2,400-facet lists, blocks of 8: 12 blocks 24.6M/s,
16 blocks 13.3M/s). **Pinned to one core the cliff is still there** (4 blocks 16.2M/s, 8+ blocks
~7.9M/s), so it is the working set against a core's cache (2 MB L2 each here), not threading.

⚠️ **#577 WAS FILED ON THE WRONG MECHANISM.** Its candidates (bucketed segments, a whole-stream plan,
batched dispatches) all target padding and dispatch size. With held masks, the shipped four equal
segments (0.529 of pairs, 204 calls) took 9.2-9.4 s, eight (0.550, 406 calls) 9.2-9.4 s, and a layout
cut where the rounded width changes (0.468, 213 calls, far more compilation) 8.3-8.7 s. Padding and
call count are worth ~10%; the step size is worth ~1.5x. Blocks per call only *looked* like the lever
because a call of 15 blocks was one oversized step.

⚠️ **TRAP: a patched bound in one process measures nothing after its first arm.** `_segment_part` is
cached across calls under `_Labels` and `pair_limit`, neither of which carries the bound, so a second
arm silently reuses the first one's program. A first probe read "no effect" (12.2-12.5 s for every
cap) this way. `validation/sozzi_radiation/pass_pairs.py` runs each bound in its own process
(`peak_footprint.run_forwarded`, so a `kill` of the parent stops the child).

**MEASURED** (`pass_pairs.py`: analytic 32 x 128 lamp, 8,704 facets, `UniformAbsorption(35.67)`,
`NoOcclusion`, `Outside(chamber, inlet, riser)`, 23,046 receivers uniform in the three cylinders
outside the lamp — the case mesh absent, so NOT the cell-centre population; streamed gradient over the
first 4,000 in emission; held gather 12,000 receivers against a mask built once; jax 0.10.2, numba
0.67.0, CPU, x64, Linux x86_64, 4 cores, nothing else running, two alternating sweeps, fastest kept,
spreads ≤ 1.11x, 2026-09-27):

| bound (pairs a step) | 32k | 64k | 131k | 262k | 1M | 4M (= before) |
|---|---|---|---|---|---|---|
| streamed field, s | 8.76 | **8.43** | 8.60 | 10.73 | 12.03 | 12.21 |
| streamed gradient, s | 8.37 | **7.93** | 8.99 | 9.81 | 12.21 | 12.07 |
| held gather, eager, s | 6.38 | 6.17 | 6.63 | 6.25 | 6.58 | 6.36 |
| held gather under `jit`, s | 2.22 | 2.12 | 2.14 | 1.89 | 2.21 | 2.53 |

A flat floor from 32k to 131k; 64k is the fastest or tied on every path. **~1.45x on the streamed
field and ~1.5x on its gradient; the held gather within its spread** — it gathers from a whole-problem
mask, whose random reads are out of cache at any step size. Separately (own probes, same machine):
`direct_irradiance` at 12,000 receivers 56 → 47 s; a `VoxelAbsorption` held gather (1,500 receivers,
12 x 8 x 8 grid) 23.3 → 23.5 s, i.e. none.

**On the Mac the step size barely matters, and 2^16 stands** (`pass_pairs.py` on main `8fc66ac`, Apple
M3 Pro — 11 cores, 5 performance + 6 efficiency, 16 MB L2 shared by the performance cluster, 18 GiB —
macOS arm64, jax 0.10.2, numba 0.67.0, CPU, x64, run through `run_case.sh` with nothing else running,
two alternating sweeps, fastest kept, 2026-09-28). ⚠️ **A different scene from the Linux table**: with
`work/case` present the harness takes the case's `lampWall.stl` (**7,516 facets**) and **23,985 cell
centres** of the meshed case (24,000 sampled outside the solid, 15 inside the lamp removed), so read the
two tables' shapes against each other, never their seconds. Checksums identical to the last digit at
every bound, in both sweeps, on every arm.

| bound (pairs a step) | 32k | 64k | 131k | 262k | 1M | 4M (= before) |
|---|---|---|---|---|---|---|
| streamed field, s (spread) | **2.50** (1.10x) | 2.59 (1.07x) | **2.50** (1.00x) | 2.51 (1.00x) | 2.54 (1.01x) | 2.55 (1.02x) |
| streamed gradient, s | 1.84 (1.08x) | 1.70 (1.18x) | 1.74 (1.00x) | **1.67** (1.01x) | 1.70 (1.00x) | 1.70 (1.03x) |
| held gather, eager, s | **1.56** (1.14x) | 1.72 (1.09x) | 1.95 (1.01x) | 2.01 (1.03x) | 1.65 (1.02x) | 1.67 (1.00x) |
| held gather under `jit`, s | 0.39 (1.34x) | 0.36 (1.01x) | 0.33 (1.03x) | 0.37 (1.00x) | 0.32 (1.04x) | **0.30** (1.02x) |

- **No cliff anywhere from 32k to 4M on the streamed paths**: the field spans 2.50-2.59 s and the
  gradient 1.67-1.84 s, and 64k sits within its own spread of the fastest on both. So #585's bound costs
  nothing here, and it buys nothing either: the ~1.45x of the Linux box does not transfer. A per-machine
  bound (from the cache size, or settable) would not help on this machine.
- **The one arm where a larger step is faster beyond the spread is the held gather under `jit`**: 0.30 s
  at 4M against 0.36 s at 64k, 1.2x, with spreads of 1.01-1.04x at both. It is 0.06 s here, and the held
  path is not what the model's streamed call runs.
- The eager held gather has a bump at 131k-262k (1.95-2.01 s, tight spreads) against 1.56-1.72 s either
  side; not explained, and not on the streamed path.
- **So the mesh-scale hint that led here was not the bound**: `field_cost_breakdown.py`'s areal segments
  read 47.1 s without #585 and 52.9 s with it (one run each), but on this sweep the whole bound range moves
  the streamed field by 4%. Read that 1.12x as the run-to-run spread of one run each, or as something in
  #585 other than the bound; not separated.

⚠️ **The transfer build does NOT take this bound, although it was built and agreed (2026-09-28).**
Capping `_row_blocks` at `receivers_per_step(chunk_size * n, n)` rows measured ~1.2x (245-247 s at
256 rows against 194-204 s at 4-8, 8,704-facet lamp, checksums identical) — but on the build before
#582, which since evaluates each block only against the facets in front of it, so a block forms
`rows x width` costly pairs, not `rows x n`. The cap and its numbers were withdrawn in the merge;
re-measure on the new build before capping it by any count.

**Pinned by** `test_a_step_forms_no_more_than_the_pass_bound_under_a_higher_limit`
(`test_radiation_work.py`: scan length 6 for 12 points at a bound of 4 pairs and a limit of 8, and a
lower limit still winning); reverting the `min(pair_limit, PASS_PAIRS)` turns it red.

## GRID ACCELERATION: BUILT as `TriangleGrid` — a compiled host walk, off by default

`RayCastOcclusion` tests every ray against every triangle, which a reactor puts out of reach:
1.6M cells, 7,516 lamp facets and 53,500 wall triangles is **6.6e14** intersections, weeks at
the measured 120-150 Mtest/s. `grid.py` registers each triangle in the voxels its bounding box (a rounding wider)
spans and walks each segment through them (Amanatides & Woo's 3D-DDA), testing only what those
voxels hold and stopping at the first blocker. Selected with `RayCastOcclusion(grid=True)`, an
integer, or a per-axis triple; **`False` is the default** — it changes cost, not answers, and
the answers are what the shipped path is trusted for.

**The specification's "~20 triangles tested per ray" was an assumption and is now measured**, on
the Sozzi reactor's own geometry (`validation/sozzi_radiation/ray_acceleration_probe.py`, 4,000
sampled rays from cell centres to lamp facets, 53,500 wall triangles from `body.stl`, jax 0.10.2,
CPU, x64, macOS arm64, 11 cores):

| grid | occupied voxels | triangles per occupied voxel (mean / max) | steps per ray (mean / max) | tested per ray, early exit (mean / p95) | steps to the first occupied voxel |
|---|---|---|---|---|---|
| 32³ | 1,896 | 60.7 / 303 | 16.8 / 48 | 380 / 1,142 | 41.9 |
| 64³ | 8,399 | 22.3 / 77 | 32.8 / 97 | 121 / 469 | 12.2 |
| 128³ | 40,681 | 9.7 / 36 | 65.4 / 183 | **52** / 200 | 4.8 |

So the assumption was optimistic by about 2.5x at the best resolution measured, and the method
survives anyway: 52 against 53,500 is ~1,000x fewer tests. Refining past 128³ trades the two
columns against each other — halving the triangles per voxel doubles the steps — which is where
the default's ~10 triangles per occupied voxel comes from.

⚠️ **A TRACED walk is not merely slower here, it is WORSE THAN THE BRUTE FORCE IT REPLACES, and
that is why this is host code.** Under `jax` the trip count and the per-voxel triangle count must
both be static, so every ray pays the longest walk against the fullest voxel whether or not it
finds anything: 183 steps x 36 triangles = **6,588 tests a ray** at 128³, against 53,500 for the
brute force — an 8x saving, not 1,000x, and none of the early exit. The mask is frozen and built
from geometry alone, so nothing about it has to be traceable. **Reach for a host implementation
whenever a structure's whole value is in the work it SKIPS** — tracing prices the work skipped at the
same rate as the work done. ⚠️ **And "host" must mean compiled host code, not numpy passes**: the array
walk that stood here priced every step instead (THE COMPILED WALK, below).

**One predicate, two implementations.** `_counts_as_hit` holds the window-and-exclusion rule
(`distance > near`, `distance <= 1.0` inclusive at the far end, triangle not excluded) for
`_block_is_cut`, with the geometry in `_watertight_hit`. ⚠️ **The grid walk carries its own copy of
both** (`grid_walk._cuts`), because a Numba loop cannot call a traced function — there is no
`_pair_is_cut` any more. The grid's tests compare the two against each other at every resolution, and a
change to one must be made to the other.

**Exactness is the test, not a tolerance.** `test_the_grid_answers_exactly_what_testing_every
_triangle_answers` compares against `segment_is_cut` at resolutions `None, 1, 3, 16, (2,7,5)`
for bit equality; a rectangular grid catches an axis transposed in the flattening, and one voxel
*is* the brute force. A drum test asserts no segment from inside a closed body reaches outside,
which is how an under-registered triangle leaks — as a bright spot in a field rather than an
error. At the strategy level, `RayCastOcclusion(grid=...)` is compared with the ungridded one
facet to facet on a closed drum.

**Mutation pass (5 mutations, 4 red).** Registering triangles by centroid instead of bounding box,
stopping the walk after the first voxel, dropping the exclusions, and sizing the grid per axis
rather than by extent all go red. ⚠️ **One survived and is DISMISSED, not a gap: forcing every ray
to start inside the grid.** Rays entering diagonally through a far face were constructed
deliberately and gave 0 disagreements, because the walk's stepping comes from the ray itself, so a
clipped start still covers the true path. The entry point saves steps; it does not decide
correctness. The separate early-out for a segment that misses the box entirely *is* load-bearing
and has its own test.

**⚠️ THREE DEFECTS SURVIVED THE FIRST ROUND OF TESTS AND ALL THREE SURFACED ON FIRST USE AT
SCALE — the pattern is worth more than the individual bugs.** Every fixture in the first suite ran
segments about one unit long, with a margin of 1e-6, on a few hundred triangles scattered through a
cube. That is one shape of input, repeated; the suite could not distinguish a correct
implementation from three broken ones.

1. **`min_distance` was read as a share of the segment, not as a length.** `segment_is_cut` takes
   it in length units and divides by the segment's length; the grid compared it against the
   parameter directly. On a unit-length segment with a 1e-6 margin the two are indistinguishable,
   which is every fixture that existed. It bites on a segment whose length is not one — a blocker
   halfway along a ten-unit segment with a margin of 3 is blocked by the correct reading and clear
   by the wrong one.
2. **The array walk's candidate expansion had no work limit**, and the first walk of the reactor wall
   with 1M rays was **killed for memory**, not slow. ⚠️ The compiled walk forms no (ray, triangle)
   array at all, so `blocks` takes no `work_limit` any more; the trap to keep is that any walk which
   *expands* candidates needs a bound on them.
3. **The default resolution was sized from the box's VOLUME, and blocking triangles are a
   SURFACE.** Voxels occupied by a sheet go as `area / size**2`, not `volume / size**3`, so sizing
   one voxel per ten triangles assumes a filled box. On the reactor wall — 53,500 triangles over
   ~0.3 m^2 inside a 1.74 x 0.09 x 0.94 m box — it built 5,350 voxels of which **298** were
   occupied, holding **217 triangles each** against the ten intended. Sizing from the triangles'
   own area gives 6,537 occupied voxels holding 18.4 each, near cubic to 0.5% — the rule until #503;
   the area still sets the count, and the voxels now take the box's shape. (Entries per voxel
   run about twice the target because a triangle registers in every voxel its bounding box spans;
   that is expected and the test's tolerance says so.)

⚠️ **A wall-clock-against-resolution table measured on the ARRAY walk stood here and is deleted**
(its best arm, 128³, moved 1.37x between two identical runs — the trap worth keeping: read the
spread before the ranking). The resolution question was re-asked on the compiled walk, and the
default rule changed because of it: THE WALK'S COST (#503), below.

**THE MASK IS EXACT ON A REAL REACTOR, AND THE ONLY DISAGREEMENT IS THE STL'S IDEA OF A
CIRCLE.** `validation/sozzi_radiation/grid_mask_check.py` runs the general method -- the vessel
wall as the 53,500 triangles `bodyWall.stl` actually holds, culled by the grid -- against the
hand-derived analytic occluder that the Sozzi comparison uses, which is exact because that fluid
is three convex cylinders. Configuration: 7,516 lamp facets, 24,000 receivers (20,000 of the
310,886 pipe cells, 4,000 chamber cells as a control), 180,384,000 rays, area-sized grid
(212, 11, 114), `UniformAbsorption(35.67)`, jax 0.10.2, CPU, x64, macOS arm64, 11 cores, 78 min.

| | pipes, 20,000 cells | chamber, 4,000 cells |
|---|---|---|
| pairs masked differently, per cell | 88.8 of 7,516 | **0** |
| relative difference in `G`, median / p99 / max | 1.2% / 8.9% / 54% | **0 / 0 / 0** |

**The chamber control is exactly zero**: the faceted wall blocks nothing where the ideal one
blocks nothing, on four thousand cells. And of the 1,776,306 pairs the two masks disagree on,
**99.993% cross the opening between 0.976 and 0.9997 of its radius** (median 0.989). An STL
describes a round pipe as an inscribed polygon -- `cos(pi/15) = 0.978` puts this one at about
fifteen sides -- and the sliver between that polygon and the circle is the entire disagreement.
Neither mask is wrong; they are given different geometry. This is what makes the case for
describing blocking geometry as primitives rather than triangles, where the circle *is* the
geometry.

⚠️ **Two things the median hides, both worth keeping.** The worst cell moves by **54%**: cells
deep in a pipe see a sliver of lamp, so a handful of facets carry the whole signal and one of
them switching is a large relative change on a small number. And 0.007% of the disputed pairs --
about 124 of them -- fall outside the rim band and are **not explained**; they are too few to
matter for a field and too specific to dismiss, so they are recorded rather than rounded away.

**Cost, measured in the same run, and it is the real argument.** The hand-derived analytic arm
took **6.9 s** against the grid's **4,659.8 s** on those same 180M rays — **675x**. ⚠️ **That is
the one arm-to-arm figure on record taken WITHIN a run**, both arms in one process on one ray
set, which is why it is kept when the table that used to stand here was deleted. It is also the
most favourable corner available: the *bespoke* arm, against the grid on this run's pipe-heavy
receivers, at the near-cubic grid that was then the default.

⚠️ **A four-cell table stood here and has been DELETED rather than corrected (2026-09-24).** One
of its cells was the 675x above; the other three divided numbers that were never measured
together, and every quantity in it has a better-measured equivalent elsewhere in this file — the
general arm and the bespoke arm under "MEASURED: the Sozzi reactor as three cylinders", the grid's
corners re-measured on the compiled walk under THE WALK'S COST (#503). Three specific things were wrong with it, and they are worth
knowing because each is a shape that recurs:

- **It gave the bespoke arm a second value.** 26.1M rays/s here against 28.8M from three
  controlled passes — 1.10x apart, same quantity, same file.
- **Its grid rates predate the square.** 38,711 and 65,647 against the square's 28,248 and
  79,963 for those corners, 1.37x and 1.22x apart.
- **Its column header named a population it did not measure.** 38,711 was labelled "pipe cells",
  but that run's receivers were 20,000 pipe **plus 4,000 chamber**, and composition alone accounts
  for 1.12x of the gap to the square's actual pipe corner.

⚠️ **The first attempt to fix it patched one row and left the other three defects greppable**,
which is the annotation failure this file warns about in general terms: a corrected cell beside
three uncorrected ones reads as a maintained table. Supersede by deleting.

⚠️ **#502's matched square (receiver population x resolution, four corners in one process) was
measured on the ARRAY walk and is deleted**, along with the "distance travelled through empty
voxels" mechanism it was read as confirming; on the compiled walk that mechanism does not hold
(THE WALK'S COST, #503, below). Two lessons it taught are kept where they generalize: a 2x2's
closure identity is vacuous (`CLAUDE.md`, "A test that cannot fail"), and every arm of a
comparison goes in one process, alternating passes, fastest kept. For scale in the other
direction, the *entire* field with analytic occlusion -- all 1,635,909 cells, gather arithmetic
included -- takes **557 s** (`run-20260922-125655.log`). The grid mask alone then extrapolated to
88 h (1.23e10 rays at 38,711 rays/s, array walk, every pair); **measured since, the whole field with
the triangulated wall takes 3,720.6 s against 313.7 s with the cylinders — 11.9x, not ~600x** (the
compiled walk and default culling; TRIANGULATED WALL AT MESH SCALE below). Primitives remain the
path a real reactor should take — exact at the rim as well as 12x cheaper — and the grid the
fallback for geometry that exists only as triangles (issue #501).

**What it does not fix: the RAY COUNT, which is the binding cost at mesh scale.** 1.6M cells
against 7,516 facets is 1.2e10 segments however cheaply each is answered. On the array walk the grid
made scenes up to a few times 1e8 rays practical; with the compiled walk and default culling the Sozzi
field's 1.2e10 take ~62 min (~3.3M pairs/s overall, measured below), so an hour-scale mesh is now
practical and the facet count (the lamp ladder, next) or a coarser shadow emitter is what goes further. ⚠️ **Since #554 phase
B there is a third way, which cuts the rays WALKED rather than the rays asked about**: a
`TriangleBody` under `ShaftCulling` vouches for whole tiles, 89% of pairs and 4.96x on a shadowless
triangulated chamber (the SHAFT CULLING section above) — on the real wall, less, by the pipe-cell share.

### THE COMPILED WALK: one Numba loop per ray, the only walk (2026-09-26)

**Why the array walk was slow, profiled before building** (a 51,200-triangle cylindrical wall, 0.05 m
by 1.6 m, 200,000 rays between interior points, a fifth aimed out; cProfile plus a counter on the
kernel; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores): **the traced triangle test was 6-47% of the walk
and the rest host bookkeeping** — about **150-180 ns per ray per voxel step** of numpy passes over the
live rays. That is why a finer grid was *slower* on it: default (10, 10, 161) → (32, 32, 512) cut the
tests per ray 79 → 10 and raised the steps 54 → 171, and 55k → 37k rays/s. **93-99% of the steps
landed in empty voxels** on that scene. So a survey comparison with production ray tracers (Embree:
compiled per-ray traversal, a BVH, 32-bit SIMD, threads) came down to the execution model first and
the data structure second — and the execution model is adoptable without importing a ray tracer.

**What it is.** `TriangleGrid.blocks` hands the live rays' DDA state (`_enters_grid`, `_walk_state`)
to `grid_walk.walk_to_first_hit`, a `numba.njit(parallel=True)` loop, one ray per `prange` iteration,
walked to its first hit or its end, the first axis winning a tie, capped at `sum(resolution) + 3` steps
(a DDA visits at most `nx + ny + nz - 2`). Each ray's axis permutation and shear (`_ray_frame`) is
formed once per ray, not per triangle. No pair array is formed, so **`blocks` takes no `work_limit`**,
and `RayCastOcclusion.work_limit` now bounds only the every-triangle path. **Numba is a core dependency**
(`numba>=0.61`, the first with Python 3.13 wheels), decided by the project owner over keeping it
optional; the array walk was deleted as dominated (it lost at every grid and scene measured).
⚠️ **The first call of a process compiles the loop, ~2 s.** Not cached to disk (`cache=True` would
write beside the installed package).

⚠️ **The intersection test is a SECOND IMPLEMENTATION of `_watertight_hit` + `_counts_as_hit`**,
because a Numba loop cannot call a traced JAX function and the dense `_block_is_cut` path still needs
the traced one. Every contract test in `test_radiation_grid.py` compares the walk against
`segment_is_cut`, which is what keeps them one predicate; a change to one must be made to the other.

**MEASURED, against the array walk before it was deleted** (`validation/radiation_grid_walk.py` at
`d9fc017`, where it timed both walks in one process on the same rays, warm, two alternating passes,
fastest kept; 200,000 rays; jax 0.10.2, numba 0.67.0, CPU, x64, **Linux x86_64, 4 cores** (a cloud
container), run directly with output redirected, nothing else running, 2026-09-26; the per-triangle ray
setup was still inside the triangle loop); grids are the default and 2x / 4x it per axis:

| scene | grid | array rays/s | compiled rays/s | speed-up |
|---|---|---|---|---|
| long thin vessel (51,200 triangles, 13.8% blocked) | default 10x10x161 | 61,616 | 1,041,835 | **16.9x** |
| | 20x20x322 | 55,098 | 1,186,686 | 21.5x |
| | 40x40x644 | 26,540 | 854,825 | 32.2x |
| annular reactor (sleeve + wall, 32,768 triangles, 63.6% blocked) | default 9x9x92 | 21,755 | 812,356 | **37.3x** |
| | 18x18x184 | 35,407 | 1,026,089 | 29.0x |
| | 36x36x368 | 41,407 | 1,021,659 | 24.7x |

Answers identical in every row. The first compiled call of a process also compiles the loop: **2.0 s**.
A scratch run of the vessel scene earlier the same day read 25.0 / 20.9 / 27.4 / 35.2x at 1 / 1.6 / 3.2 /
6.4x the default — the same band, two processes. On the Sozzi wall and the 11-core machine it has
since been measured: see TRIANGULATED WALL AT MESH SCALE below.

**Through the paths users call, `main` against this branch** (`main` at `392f935` in a worktree, then
the branch, back to back on the same 4-core container, two processes, so read the ratios; jax 0.10.2,
numba 0.67.0, 2026-09-26):

| path | `main` (array walk) | compiled walk | speed-up | answers |
|---|---|---|---|---|
| `RayCastOcclusion(grid=True)` volume mask: `radiation_receiver_ray_mask.py`'s 4,992-facet annular reactor, 500 of its receivers, fastest of two warm builds | 54.47 s | **2.49 s** | **21.9x** | same mask (SHA-256 of the packed bits, 463,063 blocked) |
| `TriangleBody` wall, `EveryPair`: `sozzi_radiation/triangle_culling.py` (51,328-triangle shadowless chamber, 3,360 lamp facets, 5,476 receivers, 18.4M pairs) | 211.9 s | **15.4 s** | **13.8x** | identical to its own reference |
| same, `ShaftCulling()` (32, 8, 2) | 43.8 s | **2.3 s** | **19.0x** | identical, 89.3% certified |

So the culled triangle body is now **~90x** the phase-B `EveryPair` figure recorded above (227.4 s) — the
two speed-ups compound, since certification cuts the rays and the walk cuts what each costs.

**Resolution, re-asked under the compiled walk** (`validation/radiation_grid_walk.py` on the branch,
same container, one process): the vessel peaks at 1-2x the default per axis (406k / 1.35M / 1.34M /
0.93M rays/s at 0.5 / 1 / 2 / 4x) and the annular reactor at 4x (302k / 762k / 922k / 1.01M) — so the
ten-triangles-a-voxel target, tuned under the array walk, is **not** confirmed or refuted by one scene
each, and is left as it is. Every row matched every triangle on its 5,000 checked rays. The per-ray
frame hoist moved neither scene outside the cross-process band (1.35M against the prototype's 1.04M on
the vessel, 0.76M against 0.81M on the reactor). ⚠️ **Those multiples were of the NEAR-CUBIC default, which #503 replaced** (the default is
now box-shaped; `radiation_grid_walk.py` times both rules side by side since).

**What is left, estimated rather than measured**: at ~1M rays/s on 4 cores a ray costs ~4 µs of one
core, and ~80 triangle tests at a few tens of ns each account for most of that at the default grid — so
empty-space skipping (#503) or a BVH now buys less than the per-step overhead suggested, and the
triangle test's own cost is the next thing to profile.

**Tests** (`test_radiation_grid.py`): the contract tests against `segment_is_cut`, plus rays aimed at
every vertex and edge midpoint of a closed drum (the watertight fixture), a segment ending in a
triangle's plane (far end inclusive), a hit exactly at the margin (near end exclusive, axis-aligned so
the distance is exact), and y-parallel segments added to `rays_through`. The work-limit and
compiled-shape tests went with the array walk; there is nothing of either left to pin.
**Mutation pass on `grid_walk.py` (13, 8 red; re-run on the final kernel over the grid and triangle-body
tests):** exclusions dropped, far end exclusive, near end inclusive, one-sided inside test, shear sign,
one ray's frame used for every ray, and the leading axis fixed at x — that last one
caught only once y-parallel segments were added to the fixture, every earlier one having a non-zero x. **Dismissed, each because it cannot change an answer:**
no break on a hit and dropping the `leaving <= 1` walk bound (both only walk further — hits past the
end fail the window anyway); the edge-on guard (`x / 0` gives `inf` or `nan`, which fails the window);
one step short of the cap (a DDA visits at most `nx + ny + nz - 2` voxels, so the cap has slack); ties
broken to the last axis (the tied crossing is a corner the ray touches at one point). ⚠️ **And the plain
edge function in place of the averaged one survived, even with `fastmath={"contract"}`** — Numba did not
fuse the products on this fixture. The averaged form stays, so the watertight guarantee does not rest on
what a compiler happens to do; that is the lesson the traced kernel already paid for.

**`NUMBA_NUM_THREADS=1`** is now set beside `OMP_NUM_THREADS` in `tools/fastgate.sh`'s parallel tier and
in CI's environment, so xdist workers do not each start a pool the size of the machine.

### THE TRIANGULATED WALL AT MESH SCALE, MEASURED (2026-09-26) — 62 min, not 88 h

Replaces the 88 h extrapolation (1.23e10 rays at 38,711 rays/s, 2026-09-23, array walk, `EveryPair`).
Configuration throughout: main `f96e923` unless stated, jax 0.10.2, numba 0.67.0, CPU, x64, macOS arm64,
11 cores; 7,516-facet `lampWall.stl`, `NoOcclusion`, streamed receiver mask, default culling; one run
per row through `run_case.sh`, nothing else heavy running. Full tables in the Sozzi README. ⚠️ **All
measured before #575** (`BodyCulling.prepared`, the in-program tile gather): neither reaches the
triangle wall's cost (facet clearance ~0 s for it per chunk, `TriangleBody` keeps the host gather), so
the 62 min should stand; the 313.7 s analytic row may have moved. ⚠️ **The triangle-wall rows are on the
OLD near-cubic walk grid**: re-measured after #575 on it at 3,705.2 s (so #575 indeed did not move it),
and on the box-shaped default since #503 the field takes **2,914.3 s (48.6 min)** — THE WALK'S COST below.

- **Whole field** (`model_at_mesh_scale.py`, `SOZZI_WATER=triangles` = `bodyWall.stl` as a
  `TriangleBody` sheet, all 1,635,909 cells): **3,720.6 s** field, 58.0 s build, **5.58 GB** peak; the
  cylinders on the same commit 313.7 s / 56.9 s / 6.03 GB (313.5 s before #562/#563/#571). Chamber
  cells match the analytic field to **4.4e-16** (1,277,672 lit); lit pipe cells 1.42% / 5.83% / 13.5%
  median / p99 / max (the inscribed-polygon rim, as `grid_mask_check.py` found).
- **`grid_mask_check.py` re-run**: 40,000 pipe + 4,000 chamber cells (the old run 20,000 pipe), 330.7M
  rays, triangulated arm 282.2 s = **1,171,912 rays/s (~30x)**, extrapolated 2.91 h; answers unchanged
  (89.2 pairs/pipe cell differ, `G` 1.20 / 8.90 / 54.1%, chamber 0, 99.995% within 10% of the rim).
  ⚠️ **Its extrapolation overstates the measured whole field ~3x** — its receivers are 91% pipe cells,
  the dearest rays. Quote the whole-field figure, never a pipe-heavy rate times the mesh.
- **#571 on the real wall** (`triangle_culling.py`, `bodyWall.stl`, 5,476-receiver slab, 3,360 facets,
  18.4M pairs, `a0b8e18` → `53074d2`): every pair 47.4 → 3.4 s (13.9x); 32x32 39.7 → 2.6; 32→8 28.3 →
  1.8; default 16.7 → 1.2 s (13.9x); certified 31.6 / 59.5 / 77.2%. ⚠️ **0.00% of that slab's pairs are
  blocked**, so it times the walk and checks no answer.
- **Where it goes** (`field_cost_breakdown.py`, every 16th cell): masks 98% of the call, the compiled
  walk of uncertified tiles 85%, certificates 6%, gather 1.6%. **Only 25.2% of pairs certified**
  (analytic cylinders: 81.0%) — a round wall's occupied voxels surround every shaft not near the axis;
  padding adds 20%; wholly blocked tiles hold at most 12.9% of the undecided pairs, so a "fully
  hidden" certificate buys little on this wall. ⚠️ **A strided sample understates culling** (blocks
  spread 16x wider): the whole mesh ran 3,720.6 s where 16x the sample's 284.3 s is 4,549 s. So the
  lever for triangle-only geometry is a certificate that vouches for more of the chamber (the convex
  region a triangulated vessel encloses, #568), or a cheaper walk (#572, #503) — neither built.

### THE WALK'S COST: TRIANGLES TESTED IN NARROW PASSAGES, AND THE VOXELS' SHAPE (#503, 2026-09-27)

#503 proposed, in order: walk each segment from whichever end is nearer its blocker, then skip empty
space (a distance field, a two-level grid, a BVH). **Both rest on premises the compiled walk does not
have**, measured with `validation/sozzi_radiation/grid_walk_direction.py`: `bodyWall.stl` (53,500
triangles), 7,516 lamp facets, 40 cells per population (outlet-pipe cells; chamber cells; cells drawn
uniformly from the mesh) x every facet = 300,640 segments per corner, all corners in one process, two
alternating warm passes, fastest kept; jax 0.10.2, numba 0.67.0, CPU, x64, macOS arm64, 11 cores.

- **Direction is worth nothing: 0.95-1.06x** cell->lamp against lamp->cell, over 15 grids and all
  three populations, answers identical (the reversed arm has no margin at the lamp end and changed no
  answer). The blocked segments were never the cost — they stop within ~2 occupied voxels either way.
  ⚠️ So the issue's "a large asymmetry" prediction, reasoned from the geometry, is refuted.
- **Empty space is not the cost.** A harness-side copy of the walk with counters (checked ray for ray
  against `TriangleGrid.blocks`) at the old near-cubic default `(212, 11, 114)`: a clear pipe-cell
  segment steps **106.5** voxels, **35.6** occupied, and tests **787** triangles (539 distinct); a
  chamber segment tests 70. The pipe is 9.55 mm in radius against 8.2 mm voxels, so a segment down a
  pipe tests the wall's triangles in every voxel. Empty-space skipping skips the cheap part;
  mailboxing (not re-testing a triangle) is worth ~1.5x at most. **Neither (2)-(4) of #503 is built.**
- **Refining near-cubic voxels is flat**: 1x-4x per axis took pipe-cell tests 444 -> 139 per segment
  and steps 64 -> 255 at a constant 1.27-1.48M segments/s — a compiled step costs about 1.6 triangle
  tests here, so the target of ~10 triangles a voxel (tuned when a step cost 150-180 ns of numpy) is
  neither better nor worse refined. ⚠️ That flatness is the evidence against "just lower the target".
- **The voxels' SHAPE moves it.** Long along the chamber, fine across (the y axis crosses both pipes):

| grid | voxel (mm) | pipe cells /s | any cell /s | chamber cells /s |
|---|---|---|---|---|
| near-cubic `(212, 11, 114)`, the old default | 8.2 x 8.1 x 8.2 | 1,351,334 | 2,755,017 | 4,494,977 |
| box-shaped `(102, 102, 102)`, **the new default** | 17.0 x 0.9 x 9.2 | **2,648,106** (1.96x) | **4,332,899** (1.57x) | 4,624,531 (1.03x) |
| `(64, 128, 128)`, the best tried | 27.2 x 0.7 x 7.3 | 3,911,954 (2.89x) | 4,659,643 (1.69x) | 4,040,810 |

**THE NEW DEFAULT (`grid._resolution`)**: equal voxel counts per axis — each voxel in the proportions
of the triangles' bounding box, edges capped at `_MAX_ASPECT = 32` to one another (the Sozzi box is
19:1) — and `_VOXEL_BUDGET = 4` times the voxels of the near-cubic grid at `_TARGET_PER_VOXEL`
(`_near_cubic`, still sized from the triangles' AREA). An axis that would get under one voxel gets one
and the others share the count (a flat surface is one voxel thick). **The derivation**: steps along an
axis go as the segment's travel there over the edge, so at a fixed voxel count steps are fewest with
edges proportional to that travel; a mask pairs every receiver with every facet, so segments spread
like the box. ⚠️ **Where it can mislead**: the box is the triangles', not the segments' — a flat floor
lit from above, a vessel off the axes (whose box is near-cubic, so the rule degrades toward the old
one). Speed only: the grid never decides what counts as a hit, and `resolution=` overrides it.
**Across scenes** (`validation/radiation_grid_walk.py`, which now times the near-cubic rule beside the
default; 200,000 rays, one process, two alternating passes; same machine, 2026-09-27; every arm matched
every triangle on its 5,000 checked rays):

| scene | new default | old near-cubic default | near-cubic at the new default's voxel count | new default at 0.5x / 2x per axis |
|---|---|---|---|---|
| long thin vessel (51,200 triangles) | 40³: 5,651,873 /s | 10x10x161: 4,248,758 (**1.33x**) | 16x16x255: 4,499,168 (**1.26x**) | 5,578,165 / 4,636,010 |
| annular reactor (32,768 triangles) | 31³: 4,582,308 | 9x9x92: 2,562,123 (**1.79x**) | 14x14x146: 3,276,703 (**1.40x**) | 3,723,699 / 4,641,821 |

So the default's budget sits on the annular reactor's plateau (2x per axis is 1% faster) and near the
vessel's (0.5x is 1% slower, 2x 18% slower), while on the Sozzi wall `(64, 128, 128)` — about twice the
default's voxels — was 1.48x faster still on pipe cells: `_VOXEL_BUDGET = 4` is a compromise, not an
optimum.

**The occupancy grid did NOT follow** (`TriangleBody.occupancy`, above): it stays 4x a near-cubic grid,
so certification on the Sozzi wall is unchanged and the mesh-scale run below isolates the walk.

**AT MESH SCALE, the grid alone** (`model_at_mesh_scale.py`, `SOZZI_WATER=triangles`, all 1,635,909
cells, default culling, the same commit for both arms (#503 branch on `d7899ac`, so #575 in), the old
grid through `SOZZI_GRID=212:11:114`, two processes back to back, 2026-09-27): field **3,705.2 s** on the
old near-cubic grid against **2,914.3 s** on the new default — **1.27x**, fields identical to every
printed digit (chamber max 4.4e-16 against the cylinders, pipes median 1.42% / max 13.5%), build 58.0 s
both, peak footprint 5.87 / 6.35 GB. The 3,720.6 s recorded under TRIANGULATED WALL AT MESH SCALE
(pre-#575) and this 3,705.2 s agree, so #575 did not move this call. ⚠️ **Why 1.27x and not the
1.57-1.96x the sampled segments showed**: the walk is ~85% of the call, but the segments it walks are
the ones certification leaves — pairs crossing between regions and pairs near the wall — not a uniform
sample. Chamber-cell segments gained only 1.03x, which is the likely reason; not decomposed.

**Tests** (`test_radiation_grid.py`, `test_radiation_triangle_body.py`):
`test_the_default_voxels_take_the_boxs_proportions_and_its_triangle_count` (replaces the near-cubic
test), `test_no_default_voxel_is_more_than_the_cap_longer_than_it_is_wide` (a 200:1 box and a flat
surface), `test_the_grid_a_body_vouches_from_is_near_cubic_whatever_the_walk_grid`. **Mutation pass
(5 mutations, all red)**: the near-cubic rule back, the aspect cap removed, the voxel budget at 1x, the
occupancy grid from the walk grid, and dropping the flat-axis refit — ⚠️ **that last one first SURVIVED**:
the flat test checked only the per-axis counts, and without the refit the flat axis's near-zero extent
blew the other two counts up to the voxel cap. It now also bounds the total against the budget.

### A FLAT SURFACE AND A SHEET ON A VOXEL PLANE: three misses fixed (2026-10-05)

Found from a `RuntimeWarning: invalid value encountered in cast` in `_walk_state` on a
`TriangleBody` of one planar patch (`patch_triangles(..., ["front"])` of a box mesh, `sheet=True`).
Asked: does `blocks` still answer what `segment_is_cut` answers on a flat triangle set? **At the
default resolution, yes — at an explicit one, no; and one miss was not about flatness at all.**

- **The flat axis was `np.finfo(float).tiny` thick**, so its voxels were ~2e-308. A crossing point
  a rounding off the plane is then a voxel index past the range of an integer (the warning; the
  cast's garbage was clipped into range). With **one** voxel through the thickness the clip always
  lands in the right one, so the default grid's answers were right. With **two or more** (an
  explicit `resolution`), the point lands in a voxel holding nothing: **110 of 48,000** seam-aimed
  segments read clear. Fixed in `_box_and_area`: a flat axis is `_FLAT_THICKNESS = 1e-6` of the
  widest extent thick (non-flat axes unchanged).
- **A triangle on an interior voxel plane, any box, any resolution.** Registration truncated each
  triangle's box with no margin, so a face lying on a voxel plane, or an edge along one, landed on
  one side only — and **the box padding pushes every boundary past the grid's middle to round
  DOWN**, so the triangle ending there is missed from the voxel above. Where the segment crosses that
  plane and an in-plane boundary at one point, the DDA's tie-break can step around the voxel holding
  the triangle the exact test credits. **23 of 36,000** segments through the middle of three
  stacked sheets (an even voxel count through their height). Fixed: registration widens each
  triangle's box by `_rounding_margin` (a billionth of the extent), the margin `holds_any` already
  used, now one helper for the padding, the registration and the box test. ⚠️ Both sides are
  load-bearing: widening only the low side passes every fixture except a placement far from the
  origin, which is why `SHEET_PLACEMENTS` has one.
- **`_enters_grid` admitted a segment with a zero direction component lying OUTSIDE the box's slab
  on that axis** (it read zero motion as "never leaves", not "never in"). Answers were right — the
  exact test found nothing — but those segments were walked for nothing, and on a flat box they were
  the warning's main source. Fixed; pinned by counting walked rays.
- **Not a grid defect, and the trap that hid all three**: a segment that **ends exactly in** the
  sheet, or crosses its **open rim**, is a knife edge where the compiled `segment_is_cut` and the
  Numba `_cuts` round differently — 8,525 such disagreements in the first probe, every one of them
  knife-edge, at a one-voxel grid too. Compare the grid against a one-voxel grid (same kernel) to
  separate voxel selection from kernel rounding, and keep segments off the rim and past the plane.
- ⚠️ **A plane at a large coordinate hides the flat defect**: with origins within ±1 of a sheet at
  `|z| >= 2`, the crossing point rounds exactly onto the plane (its error is under half an ulp of
  `z`), and aiming at `origin + 2(seam - origin)` crosses at exactly `t = 0.5`. The first fixtures
  did both and found nothing. Tests: `test_a_flat_sheet_answers_exactly_what_testing_every_triangle_answers`,
  `test_a_sheet_lying_on_a_voxel_plane_inside_a_thick_box_is_found_from_either_side`,
  `test_a_segment_parallel_to_an_axis_beside_the_grid_is_not_walked`. **Mutation pass (6, 5 red)**:
  no thickness, no registration margin, either side of it alone, the slab test reverted; the
  original file fails 19. **Dismissed and deleted**: centring the plane in its slab changed no answer.
  Configuration: jax 0.10.2, numba 0.67.0, CPU, x64, macOS arm64, `main` at `943389a`.

## HOW MANY FACETS AN EMITTER NEEDS — measured, because it sets the price of everything

The gather costs `n_receivers x n_facets` and the mask costs that again times what it tests, so
the emitter's facet count is the cost. The Sozzi comparison uses the tutorial's own
`lampWall.stl` — 7,516 facets at ~4 mm, a mesh made for `snappyHexMesh` to snap to, not a number
anyone chose for radiation. `validation/sozzi_radiation/lamp_resolution.py` measures what it buys
against an analytic lamp refined to 270,336 facets (128 sectors x 1,024 slices, 35.4397 W), on
8,000 sampled cells of the case mesh, fixed exitance 696.42 W/m², `UniformAbsorption(35.67)`,
error as `|G - G_ref| / G_ref`:

| lamp | facets | power W | near the lamp (<5 mm), median / p99 | rest of the chamber, median / p99 |
|---|---|---|---|---|
| the case's STL | 7,516 | 35.2596 | 1.95% / 7.28% | 1.08% / 2.63% |
| 8 x 16 | 288 | 34.4966 | 25.4% / 52.6% | 8.82% / 26.3% |
| 16 x 32 | 1,152 | 35.205 | 10.4% / 27.7% | 2.21% / 8.38% |
| 24 x 64 | 3,360 | 35.3374 | 4.08% / 13.4% | 0.84% / 2.28% |
| 32 x 128 | 8,704 | 35.3837 | 1.42% / 6.51% | 0.38% / 0.88% |
| 48 x 256 | 25,728 | 35.4169 | 0.43% / 2.77% | 0.15% / 0.35% |
| 64 x 512 | 67,584 | 35.4285 | 0.15% / 0.98% | 0.07% / 0.17% |

**Error falls about in proportion to the facet count, and the near-lamp band sets the
requirement** — a cell a millimetre away sees one facet subtend a large angle, and absorption is
evaluated once per facet along the centroid path, so the near-field is where a coarse emitter is
wrong. Away from the lamp the same lamp is 3-5x better. **A Lambertian emitter's solid angle is
exact at any distance**, so none of this is a solid-angle error: what a facet count buys is
absorption sampling and the inscribed area, nothing else.

**The STL is worse than its count suggests** — 1.95% at 7,516 against 1.42% at 8,704 — because
its triangles are irregular and its area is 0.5% under the true cylinder. Rescaling to equal
emitted power removes the area part and gives 1.45% / 0.58%, so roughly half the STL's
rest-of-chamber error is the inscribed-area deficit rather than the sampling.

**The lamp read from the CAD drawing is the cheap way to get it** (`CadModel.triangles`, 2026-09-24,
same reference and cells, table in `validation/sozzi_radiation/README.md`): every vertex on the true
surface, facets bounded in size along the lamp. At `chord=1e-4, facet_size=2.5e-3` it is **66,011
facets, 0.16% / 1.75% near the lamp and 0.04% elsewhere**, against the analytic 64 x 512's 67,584
facets, 0.15% / 0.98% and 0.07% — equal near the lamp at equal count, better away from it. ⚠️ The
near-lamp error follows the spacing ALONG the lamp (`facet_size`), not around it (`chord`): halving
the chord at a 5 mm facet size bought 0.74% → 0.60% for 63% more facets.

⚠️ **The mesh's own patch, raw, is the expensive way to get this — so it is coarsened first (#492,
next section).** The snapped `lampWall` patch carries **48,550** faces, **194,636** centre-fan
triangles (the body patch 329,028 faces); and it is **no more accurate than the STL** against the
true lamp (2.01% near it against 1.95%), because it is the STL shrunk again by snapping.

## THE MESH'S OWN PATCH AS THE EMITTER: exact, then coarsened (#492, 2026-09-25)

`aquaflux.mesh.patch_triangles(mesh, geometry, names)` gives the exact triangles (its record is in
`mesh.md`); `coarsen.py` makes them affordable. Agreed design, three decisions taken with the user
before building: **collapse only** (every surviving vertex is an input vertex, so no projection onto
the input surface is needed or made), **each body's power held** (exitance scaled by area before
over after, in `coarsen_surfaces`), and **both steps in one change as separate classes**.

- **Two bounds, and the size one is not `CadModel.triangles`' `facet_size`.** `max_edge` is a strict
  longest-edge bound; `facet_size` is "fits in a cube of that side", edges up to `sqrt(3)` times it.
  Same job, different quantity, so it has a different name — do not "unify" the two keywords.
  `chord` means what it means there, measured at the **input vertices** (exact to the input's own
  resolution, far below any chord worth asking for on a millimetre patch). `angle` (default 0.5 rad)
  bounds each input facet's normal against the coarse triangle its centroid is nearest, and is also
  the crease threshold.
- **⚠️ THE CHORD IS JUDGED PER INPUT VERTEX AGAINST THE PATCH *AND ITS UNCHANGED RING*, AND THE FIRST
  VERSION DID NEITHER.** It measured each input *facet* against *one* coarse triangle, so a small facet
  straddling two coarse triangles read as lying off the surface — on a perfect plane — and
  800 → 228 facets became 800 → 56 once fixed; a tube went 14,400 → 5,810 → 2,984. Measuring against
  the changed triangles alone still refused 2,189 collapses on a plane, because input facets overhang
  the patch a collapse rewrites. Only the rewritten triangles' covers move; the ring's only grow.
- **Features.** Rims, body interfaces, non-manifold edges and creases sharper than `angle` are
  feature lines; a feature vertex may only slide along its line, and is **pinned where the line turns
  by more than `angle`**. That corner rule was added after a mutation check: with the whole feature
  rule disabled no test failed, because every fixture's chord was tight enough to hold the rim by
  itself — and a rectangle's corner has exactly two feature edges, so it was *allowed* to slide.
  `test_with_a_loose_chord_the_outline_is_held_by_the_feature_rule_alone` pins both now.
- **Dismissed mutation, recorded so it is not re-chased: the link condition.** Disabling it changed
  no output on any fixture tried (spheres at 512 and 2,048 triangles down to a tetrahedron, a loose
  plane, a loose tube): the shape, fold and feature checks refuse the same collapses. It stays as
  the standard, cheap guard; no test pins it.
- **The quantization is structural.** A half-edge collapse cannot put a vertex anywhere new, so edges
  grow in jumps and the realized median edge sits at 0.7-0.9x `max_edge` (2.19 mm at 2.5, 3.18 at 4,
  4.37 at 6 on the Sozzi lamp). That is the price of every vertex staying on the input; a remeshing
  that moved vertices would reach the bound and need a projection — the option declined.
- **BATCHED, NOT ONE AT A TIME (#492 follow-up, 2026-09-25): ~4x, and linear in the input.** Each
  sweep takes every candidate that is best within **two edges** of it (`_local_minima`: a scatter-min
  spread twice over the vertex graph), checks all of them in one vectorized pass (`_check_patches`,
  shared by collapses and flips), and applies all that pass. Two edges is what the checks require: a
  collapse reads the triangles around its two vertices *and the ring around those*, so two chosen
  together must be three edges apart or one could validate against a triangle the other rewrites.
  Measured (`validation/sozzi_radiation/coarsen_speed.py`, nothing else running, jax 0.10.2, macOS
  arm64, 11 cores): `lampWall` 194,636 → 18,432 facets in **20.8 s**, `bodyWall` 1,322,096 → 116,989
  in **140 s**, both 4 mm / 1e-4 m, **0.106-0.107 ms per input triangle** at both sizes. The first,
  one-at-a-time version, same machine and inputs: **93 s and 555 s** (0.42-0.48 ms/triangle), facet
  counts within 1%. Four things made the difference, each measured, and three dead ends worth not
  retrying:
  - **Random order within 5% length bands, not a strict length order** (`_priority`). Most edges of a
    snapped patch are one length, ties fell to index order, and a candidate was a local minimum only at
    the leading edge of that order: **1,722 sweeps at ~60 collapses each** on a 24,054-triangle slab,
    slower than the serial version. Banded-random: 675 sweeps starting at ~320. This is Luby's
    randomized independent-set choice; it is seeded, so a coarsening is reproducible. It costs a
    little on a perfectly regular mesh (a regular tube keeps 8% more facets than the strict order) and
    under 1% on a real patch.
  - **Refusals last a round** (`new_round`), not until a neighbour changes. Clearing them around every
    applied collapse re-checked the same doomed candidates after each neighbour's collapse — half the
    time on the full lamp went to ~650 sweeps that applied nothing. A round repeats until it changes
    nothing, so a collapse a later change makes valid is still found, one round later.
  - **Screening**: once a sweep applies under `SCREEN_BELOW` (a quarter) of what it chose, the next
    sweep checks *every* remaining candidate — legal because checking only reads — refuses all that
    fail, and chooses among the rest. ⚠️ **Only together with round-long refusals**: with refusals
    cleared around each change, screening doubled the tail's cost (51.6 s against 27.2 s), because the
    screened survivors were un-refused and re-screened.
  - **The pair distance runs compiled** (`_compiled_pair_distance`, jax; pairs and candidate lists
    padded to powers of two so a few sizes are compiled). A sweep measures millions of point-triangle
    pairs, where numpy's pass per operation costs several times the arithmetic. Measured every pair
    exactly — a bounding-sphere prune in numpy was tried first and cost as much as it saved once the
    kernel was compiled, and dropping it also dropped a fallback search at body interfaces.
  - **Dead ends:** (i) the first batched version was *slower* than serial (8.0 against 4.7 s on a tube)
    until the priority fix; (ii) screening without round-long refusals (above); (iii) the inputs a
    batch covers are found by a mask over the owner array (`_inputs_of`), not by grouping all input
    facets by owner every sweep — that grouping was 16.5 s of a 58 s run.
  - **Dismissed mutation, recorded: a one-edge independence radius** (`range(2)` → `range(1)` in
    `_local_minima`) passes every test and, re-measured from scratch on the lamp slab at two bounds,
    kept every chord within its bound. The conflict it permits needs an input facet overhanging
    exactly where two patches meet, which is rare and small. Two edges stays because it is what the
    argument above requires, not because a test demands it; zero edges is caught (7 tests fail).
- **Measured on Sozzi (full table in `validation/sozzi_radiation/README.md`, `lamp_resolution.py`,
  batched coarsener),
  coarsening error against the exact patch:** 4 mm / 1e-4 m → 18,432 facets, **0.52% / 2.97%** near
  the lamp (median / p99) and 0.14% elsewhere; 2.5 mm → 40,083, 0.31% / 2.05%; 6 mm → 10,860,
  0.73% / 4.35%. Loosening the chord 1e-4 → 2.5e-4 at 4 mm saves 16% of the facets for +70% error
  near the lamp: **size along the lamp, keep the chord at 1e-4**, as for the drawing's lamp. The exact
  patch against the STL-built field: 0.30% / 3.65% near the lamp, 0.24% / 0.59% elsewhere.

## PER-CALL COMPILES: THE REMAINDER CHUNK AND THE RADIOSITY SOLVE (2026-09-26)

Found in a survey of what in the package still ran op by op (after #530), and fixed:

- **`work.in_passes` ran its short last chunk as a bare call.** The full chunks were one `lax.scan`,
  which is compiled as a whole even when nothing around it is; the remainder was `run(remainder)(start)`,
  a `jax.checkpoint` called outside any trace, which evaluates the body **one operation at a time**. A
  remainder can be nearly a full chunk. It is now a scan of one step (`scanned(remainder, 1, ...)`),
  so both are compiled and nothing is padded. Every held-mask gather and the graded-medium transfer
  walk go through it.
- **`model._solve` handed `solve_linear` a new operator closure on every call**, and `lx.linear_solve`
  (itself compiled) retraced and recompiled for each one — `eqx.Partial` over a module-level matvec was
  tried and only halved it, because `FunctionLinearOperator` still converts the closure per call. The
  solve is now `_interreflection(reflected, reflectance, source, solver)`, an `eqx.filter_jit` function
  that builds the operator inside itself, so it is traced once per matrix size and solver settings.
  Gradients are unchanged (the implicit adjoint is `lineax`'s, inside the compiled function), and the
  existing reflectance / emission gradient tests pass against finite differences.

**MEASURED** (`validation/radiation_per_call_compile.py`; "before" is `main` at `3e54fe1`, "after"
the working tree, the same harness run back to back in one sitting — **two processes**, so read the
large ratios and not the small ones; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, nothing else running,
2026-09-26):

| | before | after |
|---|---|---|
| gather, 3 full chunks + 488-receiver remainder (4,096 facets, 976 a pass): plain call | 1.78 s | **1.07 s** |
| same call inside `jax.jit` | 0.99 s | 0.78 s |
| `radiosity` repeat call, 768 facets | 0.51-0.54 s | **0.048 s** (11x) |
| `radiosity` repeat call, 3,072 facets | 1.02-1.33 s | **0.58-0.62 s** |

Answers identical in every arm. ⚠️ **A plain gather is still 1.3-1.4x the jit-wrapped one** (and 2.4x
when there is only one chunk): the scan's body is a new closure on every call, so the scan is traced and
compiled again each time. Removing that needs the gather's program cached **across** calls — the live
values as arguments and the labels that shape the program as a hashable key. ⚠️ **Done for the
STREAMED path since 2026-09-27** (`_Labels`, see BACK FACES IN THE GATHER); a plain held-mask call still
compiles per call. ⚠️ **`build_radiation_model` returns
before its asynchronous work finishes** (0.8 s to return, ~32 s more to finish at 3,072 facets, on this
machine), so the first thing timed after a build absorbs the rest of it; the harness waits on the model
first, and any first-call figure that did not is the build's.

Tests, each mutation-checked (reverting either fix turns its test red):
`test_every_chunk_runs_inside_a_scan_including_a_short_last_one` (`test_radiation_work.py`: scan lengths
`[3]`, `[3, 1]`, `[1]`, rows in order) and `test_a_second_solve_of_the_same_size_reuses_the_compiled_program`
(`test_radiation_model.py`: after `jax.clear_caches()`, two calls with different emission trace
`solve_linear` once).

## A PASS IS BOUNDED IN RECEIVER-BY-FACET PAIRS, NOT RECEIVERS (#509, 2026-09-24)

`pair_limit` (default `work.DEFAULT_PAIR_LIMIT`, 4,000,000) bounds every loop that visits
receivers: the gather's traced chunks (`direct_fluence_rate`, `direct_irradiance`), the streamed
path's per-pass mask, `build_visibility`'s body test, `RayCastOcclusion`'s passes and
`enclosure_winding`. **One helper, `work.receivers_per_pass(pair_limit, per_receiver)`, turns it into
a receiver count** — `checks.py` used to hand-write the same division. `RadiationSettings.
gather_chunk_size` is now `gather_pair_limit`. ⚠️ **There is no `chunk_size` on any of these any
more**: renamed rather than reinterpreted, so an old caller fails with a `TypeError` instead of
silently getting a different chunk. `work_limit` keeps its meaning — rays × triangles in the
intersection test — which is a different unit. **The transfer build's `chunk_size` (receiving facets,
default 256) was deliberately left**: it has the same shape of trap, but an `n^2` transfer at a
facet count where it would bite is unaffordable anyway.

**Why.** A receiver count left a pass's size to the facet count: the old default 4,096 receivers
against a 270,336-facet lamp is a ~9 GB pass, which killed the first run of `lamp_resolution.py`
silently (an out-of-memory kill has no traceback naming the setting), and refining the emitter is
exactly what a user does to improve accuracy.

⚠️ **A SECOND, WORSE WHOLE-PROBLEM ARRAY WAS FOUND AND REMOVED IN THE SAME CHANGE.** With no mask,
`_surviving_rows` returned `jnp.ones((n_receivers, n_facets))`, formed *before* any chunking — at a
mesh's 1.6M cells against a 66k-facet CAD lamp that is ~850 GB, which no chunk size bounds. It went
unnoticed because every study that hit it (the lamp ladder formed 17 GB of it) ran on macOS, which
compresses an array of ones to almost nothing. It is now an empty tuple and only the points are cut
into chunks; `test_a_scene_with_nothing_in_the_way_forms_no_array_the_size_of_the_problem` pins it.
(The helper is `_shadow_rows` since #524, which hands the chunks the mask's own layers rather than a
fraction formed whole — see the #524 entry below.)

**The default is measured, not copied** (`validation/radiation_gather_pair_limit.py`: analytic
Sozzi lamps of 8,704 / 67,584 / 270,336 facets, every point doing ~35M pairs, receivers uniform in
the chamber, `UniformAbsorption(35.67)`; each point in its own process for its peak memory
footprint, two alternating passes, fastest kept, repeat spread ≤ 1.1 except two points at 1.5 / 1.8;
jax 0.10.2, CPU, x64, macOS arm64, 11 cores, 2026-09-24):

| path | 1M pairs | 4M | 16M | 64M |
|---|---|---|---|---|
| gather alone, s (8.7k / 67.6k / 270k facets) | 0.33 / 0.38 / 0.40 | 0.32 / 0.61 / 0.41 | 0.39 / 0.77 / 0.83 | 0.38 / 0.63 / 0.65 |
| gather alone, peak footprint | 0.36-0.43 GB | 0.82-0.95 GB | 2.6-2.8 GB | 5.0-5.3 GB |
| streamed with `Outside`, s | 3.3 / 3.5 / 3.7 | 1.14 / 1.43 / 1.36 | 0.68 / 0.92 / 1.00 | 1.9 / 2.4 / 1.3 |
| streamed, peak footprint | 0.88-0.93 GB | 1.5-1.9 GB | 4.3-4.6 GB | 7.0-7.3 GB |

4M is as fast as any limit for the traced gather within the spread and keeps every path under 2 GB. ⚠️ **Since 2026-09-27 a traced step
forms at most `work.PASS_PAIRS` (2^16) whatever the limit**, so the "gather alone" row no longer
varies the traced step at all — see A TRACED STEP IS BOUNDED BY CACHE.
⚠️ **The streamed path would be 1.4-1.7x faster at 16M, at 2.3-2.8x the footprint**, and its cost at
small limits is *per-pass host overhead* (a mask build and a fresh gather call per pass), not
arithmetic — 1M is ~3x slower than 4M on it for the same pairs. That is the lever for #489: the
streamed pass and the traced chunk inside it want different sizes, and today one limit sets both.
Every checksum agreed across all 48 points: how the work is cut changes nothing about the answer.
⚠️ **This table predates #522**, which compiles `streamed_fluence_rate`'s per-chunk gather once per
call (`_compiled_gather` then — since 2026-09-27 the module-level `_point_part` / `_segment_part`, cached
across calls too: the live floats and the chunk's mask are arguments, only the labels that
shape the program are closed over) instead of re-tracing it eagerly every chunk.
`validation/radiation_compiled_gather.py` (4,096-facet analytic lamp, 3,944 receivers in 20 chunks,
one sleeve as a `Cylinder`, `NoOcclusion`, `UniformAbsorption(35.67)`, jax 0.10.2, CPU, x64, macOS
arm64, 11 cores, nothing else running, 2026-09-24; "before" is `f46c648`, i.e. with #520's custom
VJP) reads **1.54 s → 0.35 s** per warm call, checksum identical to 13 figures; the first call of a
process pays ~0.7 s of compilation. The mask build per chunk is unchanged. So the streamed rows, and
the 1M-vs-4M gap in particular, are stale until re-run; the gather-alone rows are unaffected.

**The same change stops the field phase GROWING call by call**, which was the Sozzi session's finding
under #489: identical `fluence_rate` calls on the streamed model went 5.91 → 7.16 GB, and a
`jax.clear_caches()` between them released 3.2 GB — compiled programs, not arrays, because every
eager chunk of every call traced and compiled afresh. The harness's footprint column (ten identical
streamed calls, `proc_pid_rusage` `ri_phys_footprint`) reads **0.58 → 1.18 GB, ~67 MB a call, at
`f46c648`, against 0.413 → 0.418 GB after**. Not yet re-measured on the Sozzi model itself.

**A GRADIENT WAS NOT BOUNDED BY THE LIMIT AT ALL, UNTIL THE SCAN BODY WAS CHECKPOINTED (#523).** The
limit bounded the forward pass only: a scan's reverse pass keeps every chunk's intermediates, so a
gradient's working memory grew in proportion to the receivers. Same harness and configuration,
compiled `temp_size_in_bytes` of `grad` in the emission at the default limit: **960 → 608 MB at
8.2e6 pairs and 3,839 → 608 MB at 1.3e8** — before, ~25 B per pair on top of a fixed base; about
2.6 TB at Sozzi scale (1.6M cells × 66k facets), and a review probe read 80-97 B per pair through
the model path. `_chunked` (now `work.in_passes`, #528) wraps the body in `jax.checkpoint(prevent_cse=False)`: flat in the
receiver count, forward unaffected; the review probe measured value and gradient bit-identical and
~1.67x the plain gradient's time (on a shared machine — not re-measured). `test_a_gradient_s_memory_is_bounded_by_the_pair_limit_not_the_receiver_count`
pins it on the compiled figure. The streamed path is bounded separately, and was first: each of
`streamed_fluence_rate`'s chunks is a custom VJP that rebuilds its mask on the way back (#520, see
the streamed-model section), so its tape holds a chunk's inputs rather than its mask rows. For a few
scalar parameters `jax.jacfwd` also stays at forward memory.

**ONE GEOMETRIC PASS FOR THE EMITTED AND REFLECTED FIELDS, AND NOTHING OF THE PROBLEM'S SIZE FORMED
PER CALL (#524).** Three changes to the per-call path, all answer-preserving:

- **`summed_fluence_rate(sets, ...)`** is the gather for several sets on one geometry: the solid
  angle, emitter cosine, attenuation (a whole voxel walk under `VoxelAbsorption`) and surviving
  fraction are formed once from the **first** set, and each set contributes only its radiance
  weights. Each set's terms are summed first and the sets added after, in order, so it equals
  adding `direct_fluence_rate` per set (`test_summing_sets_in_one_pass_is_summing_their_gathers`,
  1e-14). `direct_fluence_rate`, `FrozenShadows` and the streamed gather (then `_compiled_gather`) all go through
  it. ⚠️ Sets whose concrete vertices differ are **refused**, not silently gathered with the first
  set's geometry; traced vertices cannot be compared and are trusted here. That is reached only
  through the direct gathers now: the model refuses traced vertices before it gathers (see "The
  surface set passed at call time" above). ⚠️ A translation test on a uniformly emitting closed
  box reads zero to rounding — its interior field is uniform — so a "gradient is non-zero" check
  there proves nothing; emit unevenly.
- **The surviving fraction is formed per chunk** from the mask's own layers (`_shadow_rows` hands
  `blocked` and `hidden_by_geometry` to `_chunked` (now `work.in_passes`) with their receiver axes; `surviving_fraction` in
  `visibility.py` is the one expression, also behind `Visibility.surviving`). Before, it was a
  float64 array of the whole problem, 8 B/pair beyond the mask, formed eagerly per call — and
  `_chunked` then **copied** it into a padded array. `_chunked` now slices chunks in place
  (`lax.dynamic_slice_in_dim`) with the slice **inside** the checkpoint, so a gradient keeps a
  chunk's start index rather than the chunk; with the slice outside, the gradient-memory test caught
  24 B per receiver of saved chunks. Pinned by
  `test_a_mask_is_cut_into_chunks_rather_than_turned_into_a_fraction_first`.
- **`direct_irradiance(..., point_sources_only=True)`** gathers the point sources and never visits
  an areal facet; `_point_source_irradiance` used to zero the areal emission and pay a clipped
  projected solid angle for every facet pair. And `surface_irradiance` reuses `radiosity`'s
  assembly (`_solve`) instead of assembling the transfer and gathering the point sources a second
  time; the `(F^M - F) M` product is skipped when the two are one array (every areal source
  Lambertian).

Measured with `validation/radiation_fluence_rate_call.py` (4,096-facet analytic lamp, reflectance
0.3, 3,954 receivers, one sleeve as a `Cylinder`, `NoOcclusion`; jax 0.10.2, CPU, x64, macOS arm64,
11 cores, nothing else running, 2026-09-24; "before" is `66501ac`, i.e. #522 applied), warm calls,
every checksum identical before and after:

| medium | held mask | streamed mask |
|---|---|---|
| `UniformAbsorption(35.67)` | 0.55 → 0.37-0.42 s | 0.70 → 0.47 s |
| graded `VoxelAbsorption`, 12 × 12 × 16 | ~219 → ~170 s | ~200 → ~158 s |

⚠️ **The graded rows were dominated by what this did NOT touch**: `TransferMatrix.assemble` walked all
`n^2` facet pairs through the grid unchunked on every call — 16.8M pairs here against the gather's
16.2M — so the saved second receiver walk showed as ~1.3x rather than ~2x. Graded calls also spread
~15% call to call on this machine; read the ratio, not the seconds. #528 fixed the walk itself, and
the graded rows are now ~19-21 s (next section).

## THE VOXEL WALK CARRIES ITS TOTAL, AND THE FACET-TO-FACET WALK GOES IN PASSES (#528, 2026-09-25)

**The walk collected every piece before summing them.** `VoxelAbsorption.optical_depth` scanned its
fixed `nx + ny + nz + 1` steps returning each Simpson piece, so the scan stacked a
`(max_crossings, ...)` array before one sum — **8 B a pair per step** on top of a flat ~100 B, i.e.
~1 kB a pair on a 32³ grid, ~4 GB in one 4M-pair gather chunk, where `UniformAbsorption` is a few
bytes. Now the running total and the field at the piece's near end ride in the scan's **carry**: the
near end is the previous piece's far end, so each step does two lookups, not three, and nothing is
stacked. **The step is `jax.checkpoint`ed**, which is the other half and was not in the issue: under a
gradient a scan keeps its step's residuals for the way back, and those were every lookup's gather
indices and trilinear weights, **~500 B a pair per step, 47.9 kB a pair at 32³** (3 GB for 65,536
segments). Checkpointed, it keeps the carry and recomputes the lookups: ~40-60 B a pair per step.
The value is not bit-identical — a running total sums in a different order from one reduction — and
the harness checksums agree to 15 figures forward and 14 for the gradient.

**The facet-to-facet walk went through `work.in_passes`.** Under any non-uniform `Absorption`,
`TransferMatrix.assemble` walked all `n^2` centroid pairs at once, on every call. It now walks a block
of receiving facets at a time, bounded by `pair_limit` (a new keyword on `assemble`; `_solve` passes
the model's `gather_pair_limit`, whose docstring now says it bounds this too, since both are per-call
receiver-by-facet walks). `_chunked` moved from `gather.py` to `work.py` as **`in_passes`** to make
that possible, and its output may now carry a row per receiver as well as a value (`reshape(-1,
*shape[2:])`). ⚠️ **There is no `gather._chunked` or `gather._slice` any more**; the chunk-slicing
test patches `work._slice`. The issue's second complaint, that an eager `surface_irradiance`
assembled twice, was already gone with #524's `_solve`.

Measured with `validation/radiation_voxel_walk.py` (walk: 65,536 random segments in a random 32³
grid; assemble: a closed 10 cm `inward_box` at 972 and 2,028 facets, a random 12³ medium,
`NoOcclusion`; compiled `temp_size_in_bytes`, median of five warm calls; jax 0.10.2, CPU, x64, macOS
arm64, 11 cores; "before" is `4e0b632` in one run and "after" the working tree in two runs straight
after it, 2026-09-25; checksums identical to the last printed digit except the walk gradient's last
two):

| | before | after |
|---|---|---|
| walk forward, working per pair | 988 B | **104 B** |
| walk gradient, working per pair | 47,864 B | **3,512 B** |
| walk forward / gradient, s | 0.41 / 0.66 | 0.09-0.17 / 0.23-0.37 |
| assemble, 972 facets | 0.578 GB, 1.84 s | 0.197 GB, 0.42 s |
| assemble, 2,028 facets, default limit | 2.517 GB, 8.98 s | 0.849 GB, 1.81-1.86 s |
| assemble, 2,028 facets, `pair_limit=1_000_000` | — | **0.241 GB**, 1.90 s |

⚠️ **The default-limit rows are still ONE pass**: 2,028² is 4.1M pairs against the 4M default, so
what bounds them is the walk's per-pair cost (612 → 206 B), not the passes. The 1M row is what
shows the passes bounding the working set. The seconds are across runs (the rule on dividing across
runs applies; two after-runs are quoted as a range) and are approximate; the working memory is exact.

**Per-call, `validation/radiation_fluence_rate_call.py`** (the #524 configuration above, run alone,
2026-09-25): graded **held ~170 → 18.9-19.5 s, streamed ~158 → 20.3-21.2 s**, uniform unchanged
(0.41-0.49 / 0.45 s), every checksum identical to 13 figures against the #524 run. About 8x, and
most of it is the walk itself rather than the passes: the 4,096-facet assemble is 16.8M pairs, four
passes at the default limit.

Tests, each mutation-checked: `test_the_walk_s_working_memory_does_not_grow_with_the_number_of_cells_it_crosses`
(forward flat from 4³ to 32³ — `main`'s walk reads 317 → 989 B there and fails; reverse under 100 B
a step — dropping the checkpoint fails); `test_a_graded_medium_attenuates_each_pair_by_its_own_walk_however_the_pairs_are_cut`
(a pass of 7 rows on 54 facets, against all pairs walked at once — a dropped remainder and a pass
shifted by a row both fail) and `test_a_uniform_grid_walked_between_facets_is_the_closed_form` (the
walk against the frozen separations, which never walk: **nothing tested the non-uniform `assemble`
path before this**); `test_the_walk_between_facets_is_bounded_by_the_pair_limit_not_the_facet_count`
(compiled memory at 8 rows a pass under 0.2x all at once — ignoring the limit fails). Carrying the
wrong sample as the near end passes all four and fails six of the existing walk exactness tests.
**Not done, and not probed**: sharing one 8-corner lookup across a piece's three Simpson samples
(the issue's third idea; a piece lies in one interpolation cell, but it is not bit-identical at the
plane endpoints).

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

### VOLUME RECEIVERS: the share of the PLAIN solid angle (#479, 2026-09-25)

A receiver on a facet takes its share of the *projected* solid angle `∫ cos θ dω` (what an irradiance
weights by); **a point in the fluid, which has no normal, takes its share of the plain solid angle
`∫ dω`** (what the fluence-rate gather weights by, `radiance × solid_angle`). `receiver_normal=None` is
how a volume receiver says so, all the way down: `source_view`, `covered_by`, `covered_fraction` and
`may_occlude` accept it, and `SilhouetteOcclusion.field` treats a receiver with **no
`receiver_normal` and no facet** (`receiver_facet=None`, or `-1` in its row) as a volume point (it used to
raise for both). ⚠️ **"On no facet" is NOT "no normal"** — see ORIENTED RECEIVERS ON NO FACET below.
One choice, `silhouette._measure`, picks the integral for the whole, the covered part and the blocker-extent clamp together, so a share cannot be
taken of one measure against another.

- **ORIENTED RECEIVERS ON NO FACET (2026-10-06).** `receiver_facet` conflated two things: "this
  receiver lies on facet k of the SOURCE set (leave it out of the shadow test)" and "this receiver has a
  normal (measure by the projected solid angle)". A point on a reflecting wall gathering the lamps'
  light has a normal and lies on no lamp facet, so `solve_scene`'s lamp-on-reflector gather (and every
  `SurfaceReceivers` gather of a set it does not lie on) took its silhouette share by the **plain** solid
  angle — wrong for an irradiance, and silently, since only a partly hidden pair on an oblique line can
  tell. Now `build_visibility(..., receiver_normal=(n, 3))` says which way a receiver faces, independent
  of where it lies; it reaches `SelfOcclusion.field(surfaces, points, near, receiver_facet,
  receiver_normal)` (a **new positional parameter on every strategy**; only the silhouette reads it), is
  normalized and refused on a wrong shape or a zero / non-finite length (`visibility._unit_normals`), and
  **takes precedence** over the facet's normal (`self_occlusion._receiver_frames`). With a normal the
  silhouette now also accepts `(n, k)` facet rows and excludes every facet named (`np.isin`), so the
  scene no longer cuts rows to their nearest facet for it; without one, `k > 1` is still refused. The
  scene's `_irradiance` passes the points' normals on every gather. ⚠️ **`clear_behind` / `BackFaces` still
  key on `receiver_facet is None`, deliberately**: the cull is valid for any receiver gathered directly
  (volume or oriented — `direct_irradiance` weights a pair by the source's radiance towards it, zero
  behind a dark-behind source, and refuses a non-dark-behind set through such a mask); it is the
  facet-to-facet transfer, whose receivers are named in `receiver_facet`, that must cast in full. So the
  new argument does not touch the cull, and an oriented point on no facet keeps it. Under `RayCastOcclusion`
  nothing changes (a share of 0 or 1 is the same in every measure). Tests:
  `test_a_receiver_on_no_facet_that_faces_a_way_takes_its_share_of_the_projected_solid_angle` (the
  oblique `OBLIQUE_SOURCE` / `OVERHEAD_BLOCKER` pair as one surface set, 0.117 between the measures, each
  build against its own sampler), `test_a_receiver_on_several_facets_is_measured_about_the_normal_it_is_given`,
  the refusals, and in `test_radiation_scene.py`
  `test_a_partly_shadowed_lamp_lights_a_wall_by_the_share_of_its_projected_solid_angle` (reflector
  irradiance and a floor point, `E / E_unshadowed = 1 - sampled projected share` to 4e-3). **Mutation pass
  (9, 8 red)**: the scene dropping the normals, `_unchecked_visibility` passing `None`, `_receiver_frames`
  ignoring the normal, no normalization (caught by exact equality at 3x length — a share is a ratio, so
  that one moves only a rounding), no `k > 1` refusal, no shape check, no length check, volume rows sent
  to the oriented pipeline. **Dismissed, equivalent**: excluding only a row's first facet — every facet a
  receiver lies on passes through it, so as a source it is seen edge-on and as a blocker it is not front
  facing (or, declared two-sided, edge-on); `np.isin` stays for the ray test's semantics.
- **The clip is unchanged; only the integral differs.** A volume point has no front half-space, so
  `_in_view` pads the source (and the depth-cut blocker) with a repeated corner instead of clipping —
  the same widths `4 … 8`, so one clip serves both. The tangent-plane cull in `_per_triangle` is skipped;
  the cone and source-plane culls stay.
- **The unprojected integral is `solid_angle._signed_loop_area`**: a fan from the loop's first vertex,
  each term the `signed_solid_angle` closed form (`2·arctan2(a·(b×c), 1+a·b+a·c+b·c)`). Signed and
  additive, negates with the winding; repeated slots and an emptied (all-zero) loop contribute exactly
  nothing. Valid for a convex loop inside an open hemisphere, which every clip of a triangle seen from
  off its plane is. **Not** an angle-excess sum: a sliver keeps its digits — the pole/equator triangle
  of area exactly `φ` reads right to 1e-12 relative at `φ = 1e-9`.
- Tests, each mutation-checked: fan = triangle closed form from each starting corner; additivity over a
  4-way split and a quadrilateral; the thin loop; the sampler agreeing in both measures (the sampler in
  `radiation_references.py` takes `receiver_normal=None` for the plain weighting `|cos_s|/d²`); an
  **oblique** fixture where the two measures' shares differ by 0.117 (on-axis fixtures cannot tell them
  apart — the same trap as the missing receiver cosine); a volume receiver seeing behind any plane
  through it; field-level volume rows against the sampler on the sleeved box; a mixed `[facet, -1]`
  build matching each kind's own build **and** the surface row matching the *projected* sampler. That
  last assertion was added because **"every row volume" survived the suite without it**: the
  projected measure was pinned only through `covered_fraction`, never through `field()`. Dismissed, not
  covered: the extent clamp on the volume path (same logic as the surface path, which is covered).

**MEASURED** — `validation/radiation_volume_silhouette.py`, two-sleeve reactor (`two_sleeve_reactor`,
walls black so only the volume mask is judged, sleeves Lambertian `M = 1`), lattice water points clear
of walls and sleeves, union reference 4096 samples/pair (plain weighting, `segment_is_cut`, a gap real
past 3σ + 1e-3), jax 0.10.2, CPU, x64, macOS arm64, 11 cores, run alone under `run_case.sh`,
2026-09-25, uncommitted #479 working tree on `b2b792c`:

- **Per pair** (288 facets, 300 edge cells of 948 with a partly hidden sleeve facet, all 21,604 pairs
  either mask hides): clip exact within the reference on **all but 3** — those 3 over (the overlap
  over-count, worst +0.036), none under; mean |gap| **0.0003**. The ray mask: mean |gap| 0.0106,
  **worst 0.876** on one pair. 2,000 control pairs neither mask hides: the reference finds **no**
  missed shadow.
- **Fluence rate at the edge cells** against the reference field (same public `fluence_rate`, hidden
  shares injected): silhouette mean / p95 / worst |rel| **0.0005 / 0.0019 / 0.0046**; ray mask
  **0.0215 / 0.0764 / 0.1419**. Over all 2,552 points the ray mask sits 0.9% mean, 14.2% worst from the
  clip. So the worst cell improves ~30x; the mean ~40x.
- **Cost of the volume mask alone** (warm, `build_visibility`, silhouette vs ray without a grid):

| facets | receivers | clip s | ray s | clip ms/receiver | ratio | cull survivors/receiver | of n² |
|---|---|---|---|---|---|---|---|
| 288 | 464 | 6.0 | 0.4 | 12.9 | 14 | 10,183 | 12.3% |
| 288 | 1,632 | 20.8 | 0.7 | 12.7 | 30 | 9,824 | 11.8% |
| 560 | 464 | 12.1 | 0.8 | 26.2 | 15 | 22,727 | 7.3% |
| 560 | 1,632 | 40.8 | 1.5 | 25.0 | 27 | 21,433 | 6.8% |

  Linear in receivers (constant ms/receiver), **roughly linear in facets** rather than quadratic, because
  the cull's survival falls as the mesh refines (12% → 7% of `n²`). The per-receiver loop is a Python
  loop over compiled calls, so the fixed per-receiver overhead is in these figures. ⚠️ **All of this
  table predates #527** (the second stage, packing across receivers, threads) and has not been re-run;
  its survivors column counts the cone cull's survivors, and the harness's `_candidates` now also
  drops sources that subtend nothing, so a re-run reads a little lower there.
  ⚠️ **Extrapolation, flagged as one:** at that rate a 100k-cell field against ~560 facets is ~40 min,
  and it grows with facet count; the ray mask is 15-30x cheaper. So selecting the silhouette now makes
  a mesh-scale model expensive — `receiver_occlusion=RayCastOcclusion()` is the escape, and the Sozzi
  lamp (convex, `NoOcclusion`) is unaffected. Batching receivers per compiled call was the obvious lever
  and is built since #527 (packed chunks); the extrapolation above predates it.

⚠️ **WHERE IT OVER-COUNTS, STATED EXACTLY: the angular overlap between two front-facing
silhouettes.** Each blocker is clipped against the *source*, not against what is still unblocked,
so areas add: `0.54 + 0.54 = 1.08` where a ray's `OR` is idempotent (measured, two blockers in one
cone: 0.5415 each, true union 0.5421, sum 1.083). It errs **dark**, and only here:

| geometry | front-facing crossings | result |
|---|---|---|
| one sleeve, wall, or baffle (a sheet declared `two_sided`) between two facets | 1 | **exact** |
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

### MEASURED (#472): the over-count on a two-lamp reactor is rare and small — and `overlapping` is nearly useless

`validation/radiation_overlap_overcount.py`. Box plus **two sleeves side by side along x** (radius
0.1, half-height 0.3, at x = 0.35 and 0.65), so from the end walls one stands behind the other;
sleeves emit 1, walls reflect 0.5. Reference: for each (receiver, source) pair, 4096 samples over
the source weighted by `max(cos_r, 0) |cos_s| / d^2`, blocked if `segment_is_cut` finds ANY
triangle across the segment (the union, so nothing is counted twice); cross-checked against the
independent brute-force sampler on six pairs per mesh, every gap inside 3 sigma. A gap is real past
3 standard errors + 0.001. JAX 0.10.2, CPU, x64, macOS arm64, 2026-09-21.

| mesh | flagged pairs | real over-counts | mean over | worst over | one-contributor control | unhidden control |
|---|---|---|---|---|---|---|
| 288 facets, all flagged pairs | 22,376 | **335 (1.5%)** | 0.136 | **0.318** | 1,053 / 1,053 exact | 2,000 / 2,000 |
| 560 facets, random 20,000 | 20,000 | **192 (1.0%)** | 0.139 | **0.336** | 2,000 / 2,000 exact | 2,000 / 2,000 |

- **The error runs one way only**: zero under-counts in any group, as the mechanism predicts. A
  random 40 of the 335 re-checked at 65,536 samples all stayed over-counts (mean gap 0.131, 3 sigma
  at most 0.006).
- **It is rare because clipping at 1 makes the common case exact.** Most sleeve-behind-sleeve pairs
  are hidden *completely* by the nearer sleeve (15,277 of the 288-facet mesh's hidden pairs read 1),
  and a sum clipped at 1 is then right. It bites only where two sleeves each hide *part* of a source.
- **As a receiver sees it**: the share of its hemisphere wrongly reported dark is mean 0.12%, worst
  **1.6%** (288 facets; the 560-facet figure is from a subset and understates it).
- **On the field it is smaller than the ray mask's error by ~8x.** Wall irradiance against the field
  with every flagged pair corrected: silhouette mean 0.14% / worst **0.96%**, ray mask mean 1.1% /
  worst **5.8%** (288 facets). The "corrected" field carries the reference's own sampling noise.

⚠️ **`overlapping` IS 1-1.5% PRECISE ON A MESHED BODY — it flags nearly every hidden pair.** It
means "more than one blocker contributed", and a sleeve is a tiling, so any pair a sleeve hides is
covered by several of its triangles: 22,376 of 23,429 hidden pairs flagged, of which 98.5% are exact
tilings. As a detector for the over-count it says almost nothing, which the claim that "a count of
one proves a pair exact" does not reveal — that claim is true and rarely applicable. A precise
detector needs to know when two contributors can overlap: a tiling of one convex front-facing sheet
cannot, so "contributors from two or more connected front-facing sheets" is the candidate.

**DECIDED (#472): documented, not fixed — the precise detector and the correction are #477.** The
error is one-sided, rare, and on this geometry an order smaller than the ray mask's own; the flag's
imprecision is the more pressing defect, because it leaves a user no way to tell which pairs to
distrust.

⚠️ **A binomial error bar is ZERO at a share of exactly 0 or 1, however few samples carried the
weight** — the harness first reported a single-blocker pair as a 26-sigma over-count (reference
0.0000 +/- 0.0000 at 4096 samples; 0.036 +/- 0.03 at 65,536; the clip at 0.026, correct). It uses
the Agresti-Coull interval now. Any sampled reference whose weights concentrate needs the same.

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
default `work_limit` and the then-default `work_chunk` of 262,144, **before #527** (per-receiver padded
chunks, no second stage, one thread — see the next section for what changed and by how much). Two runs,
and **the ratio moved because the denominator did**:
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

**Power-of-two chunk padding** (`triangles.padded_length`, called `_bucket` while it lived in
`self_occlusion.py` alone), as it stood for the table above: each receiver's pairs were clipped in
their own chunks, and padding every chunk to `work_chunk` compiled one program and clipped a quarter
of a million pairs for a receiver with sixty candidates, so chunks were padded to the next power of
two instead. That took the silhouette tests from 110 s to 19 s — and still averaged 1.41x the real
pairs at 832 facets (the issue's probe), because "at most half a chunk" is the worst case and the
average is what costs. ⚠️ **Since #527 chunks are packed across receivers and only a build's last
chunk is padded** (next section); `padded_length` now shapes the cull's blocks and that last chunk.

### THE CLIP SEES ONLY PAIRS THAT MIGHT COVER SOMETHING, PACKED ACROSS RECEIVERS (#527, 2026-09-25)

The clip was ~96% of the silhouette build, and **almost every pair it clipped covered nothing**: the
cone cull bounds, it does not decide. Four changes, the answers unchanged to the chunk-shape rounding.

**1. A second, exact stage before the clip: `silhouette.covers_nothing(view, to_source, to_blocker)`**,
run on the cone cull's survivors. Four reasons, each a pair the clip would return zero for:

- the source subtends nothing (`SourceView.subtends()`, the one home of the `_EXTENT_FLOOR` test, read by
  `covered_by` too);
- nothing of the blocker strictly in front of the source's plane (`_in_front` on `_depth_heights`, the
  predicate `covered_by` zeroes a pair by — the cone cull's `beyond_source_plane` is a raw `< 0`, so a
  blocker touching or coplanar with the source plane, i.e. every neighbour on the source's own wall,
  got through it);
- the blocker seen edge-on (`_orientation`, shared with `covered_by`'s winding);
- a **face-plane separating axis** (`_outside_an_edge_plane`, both ways): all three corners of one
  triangle strictly outside one inward edge plane of the other's direction cone, filtered heights,
  strict `< 0`, skipped when that triangle is edge-on. The uncut triangles are used, which is
  conservative: the directions the clip measures are a subset of them.

The first three are `covered_by`'s own zeroing predicates, factored out so the two cannot drift; they
can disagree only within rounding of their own thresholds, where coverage is a few parts in 1e12.

**2. The source view is computed once per receiver** (in `_per_triangle`, for all `n` sources) and
gathered per pair, rather than rebuilt inside every clip chunk.

**3. Pairs are packed across receivers** (`_PairPipeline`): every pair carries its receiver row and its
source's view, so pairs of many receivers fill chunks of one size (`work_chunk`, now **32,768**, down from
262,144 — a chunk is now always full, so its size bounds memory and no longer trades against padding),
and only the last chunk of a build is padded (to a power of two). Reject and clip chunks run on
`threads` (default 4) Python threads, and the per-receiver cull runs `threads` receivers ahead; answers
are scattered in arrival order, so **the thread count changes no bit** (pinned, `array_equal`).
Surface and volume receivers go to separate pipelines, since they take different measures.

**4. The cull is bounded in pairs** (`pair_limit`, default `DEFAULT_PAIR_LIMIT`): `_per_triangle` reduces
the sources to those that subtend and the blockers to those that face, are far enough, and are not
wholly behind the receiver's tangent plane — index lists in `O(n)` — and the cull ran over them in
blocks of sources against all candidate blockers (a `_cull` step; there is no such method any more,
the clustered `_cluster_pairs` / `_member_pairs` replaced it in #555), so no call formed more than
`pair_limit` pairs. ⚠️ **This bounded the cull's memory, not its cost**: it was still
`O(|sources| x |blockers|)` per receiver, host `np.nonzero` included. #555 replaced the dense blocks
with a clustered cull — next section, and read it before expecting much from it.

**MEASURED** with `validation/radiation_mask_build_cost.py` (`silhouette_ladder` and the new
`second_stage`; `RADIATION_SILHOUETTE_ONLY=1` runs just those), same box-plus-sleeve reactor, receivers at
the facet centroids, median of three warm builds. ⚠️ **NOT the machine the table above was taken on**:
a Linux cloud container, 4 cores, 16 GB, jax 0.10.2, CPU, x64, nothing else running, 2026-09-25;
"before" is `main` at `0c3a78c` from a worktree, run between two "after" runs in one sitting. Read the
ratios, not the seconds, against the 11-core table above.

| facets | clip before | clip after (two runs) | speedup | ray mask (same runs) |
|---|---|---|---|---|
| 224 | 5.56 s | 1.22 / 1.33 s | 4.4x | 0.33-0.45 s |
| 480 | 24.78 s | 4.01 / 4.00 s | 6.2x | 3.25-4.01 s |
| 832 | 100.74 s | 11.78 / 11.95 s | 8.5x | 16.68-19.80 s |
| 1,532 | 384.05 s | 45.38 / 45.17 s | **8.5x** | 109.45-111.26 s |

- **On this machine the clip is now cheaper than the ray mask from 832 facets up** (0.41x at 1,532).
  Do not carry that to the 11-core machine without re-running there: the ray mask's share of cores
  differs between the two, and the old clip used only ~2 of the 4 cores here (load average ~2 through
  the "before" run), which is part of what the threads recover.
- **The second stage rejects 84.0 / 89.0 / 91.7 / 93.9% of the cone cull's survivors** at 224 / 480 /
  832 / 1,532 facets (30-32 sampled receivers each; the cull's survivors already exclude sources that
  subtend nothing), and **every rejected pair, clipped anyway, covers exactly 0.0** — worst pair and
  worst per-source sum both `0.00e+00`. The share grows with the mesh, as the issue predicted.
- **Answers**, every receiver at 224 / 364 / 480 facets and every 7th at 832, against `main`: max
  |difference| 1.4e-13 per pair and per row, **no `overlapping` flag changed**.
- **`work_chunk`**, at 832 facets, same container: 32,768 builds in 12.2 / 11.4 s against 16.5 / 16.1 s
  at 262,144, and one compiled clip call's XLA temp is 57 MB against 452 MB (the reject call 24 against
  189 MB), with `threads` of them in flight.

Tests, each mutation-checked: `test_the_second_stage_never_rejects_a_pair_that_covers_something`
(exhaustive over every pair of every receiver of the 80-facet reactor and every 13th of the 364-facet
one, surface and volume, and that it rejects over half the cone cull's survivors — flipping the edge
planes' orientation fails it), `test_the_second_stage_rejects_what_the_cones_cannot` and
`test_a_source_behind_a_surface_receiver_is_rejected_for_subtending_nothing` (one fixture per reason,
each kept by the cone cull and rejected by that reason **alone** — deleting any one reason fails its
fixture), `test_a_blocker_sharing_only_an_edge_of_the_source_s_directions_is_kept` (a non-strict
separation fails it), `test_how_the_pairs_are_scheduled_does_not_change_the_answer` (threads
bit-identical, the cull's tiling to 1e-12, surface and volume receivers in one build) and
`test_the_cull_forms_no_more_pairs_per_call_than_its_limit`. The existing chunk-size test catches a
dropped remainder; untrimmed cull padding fails the scheduling test. ⚠️ **The first fixture for "a
source that subtends nothing" was a source in the receiver's own plane, and deleting the subtends
clause left it green**: the source's plane then passes through the receiver, its oriented normal is
the zero vector, and the depth test rejects the pair too. The fixture that isolates the reason is a
source wholly behind a surface receiver with a blocker straddling the tangent plane. **Dismissed**:
dropping the edge-on guard inside the separating test changes no answer — a triangle seen edge-on is a
blocker `covers_nothing` already rejects, or a source that subtends nothing — and it stays because it
is what keeps the predicate honest on its own.

### THE CULL RUNS ON CLUSTERS FIRST (#555, 2026-09-26) — worth 1.24x at 3,184 facets, and why no more (a second level measured slower)

**Measured first, on #556's code** (`silhouette_stages`, serial, 4-core Linux container, jax 0.10.2):
the cull was **54%** of a 3,184-facet build (35.5% on device, 18.4% host compaction), 45% at 1,532;
the reject pass 22%, the clip 11%. So the ceiling for any cull change there was ~2.2x.

**What was built, as the issue wrote it.** `FacetClusters.build` (`clusters.py`) groups facets into
runs of `cluster_size` (default 32) along the Morton order of their centroids (`aquaflux.morton.morton_order`
since #574; before, a 21-bit copy of its own), with a bounding sphere
each. Per receiver, `_cluster_bounds` gives each cluster an **enclosing cone** per role
(`silhouette.enclosing_cone`: cap on the mean member axis reaching `angle(axis, member axis) +
member half-angle`, `arctan2` throughout; an unusable member or a cap reaching a right angle makes the
cluster unusable). `_cluster_pairs` rejects cluster pairs whose cones cannot overlap or whose blocker
sphere lies beyond every eligible source's plane in the source cluster; `_member_pairs` runs the
unchanged member test (cones, `beyond_source_plane`) over the survivors. Both stages are bounded by
`pair_limit`. **The kept set is exactly the dense cull's** — pinned set for set, each pair once, at
cluster sizes 1 / 5 / 32, surface and volume receivers
(`test_the_clustered_cull_keeps_exactly_what_testing_every_pair_keeps`). Pairs come out grouped by
cluster pair, **not sorted**: the sort cost 10.6 of ~54 ms per receiver, and order only moves a sum by a
rounding — measured 4.4e-16 at most against #556, no `overlapping` flag changed.

**MEASURED** (same container, both arms in one sitting, default settings, two warm builds each):

| facets | #556 | clustered | |
|---|---|---|---|
| 1,532 | 44.0-44.5 s | 40.1-42.4 s | ~1.05x (with the sort) |
| 2,448 | 113.8-115.3 s | 109.4-112.3 s | 1.03x |
| 3,184 | 225.5-229.2 s | 182.4-183.7 s | **1.24x** |

⚠️ **WHY SO LITTLE: 40% OF CLUSTER PAIRS SURVIVE.** At 3,184 facets (100 clusters, receivers every
97th facet): 3,839 of ~9,600 cluster pairs pass, so the member test still sees ~3.9M pairs against the
dense 6.7M, to keep ~101,600. A 32-facet patch seen from inside the enclosure subtends tens of degrees,
and a circular cap around an elongated patch overlaps many others. The device cull roughly halved
(133 → ~72 s serial at 3,184) and the host side grew (69 → ~89 s: `np.nonzero` over the batch masks,
15 ms a receiver, and — until removed — the sort).

**Measured and rejected, so they are not retried:**
- **Other cluster sizes**: 4 / 8 / 16 / 32 / 64 facets gave 67.2 / 61.4 / 48.0 / 54.2 / 63.7 ms a
  receiver for the whole cull (with the sort), against ~63 ms dense. Small clusters make the cluster
  stage quadratic in clusters; large ones loosen the bound. 16 was best by ~10% and was not adopted on
  one scene.
- **Compaction on the device**: `jnp.nonzero(size=...)` after a device-side count took **114.5 ms**
  against numpy's **14.8 ms**, transfer included, on a 4096 × 32 × 32 mask at 2.5% density. On CPU
  the host is the place for it.

**Re-measured after a container change, all arms in one sitting** (4 cores, but faster hardware than
the table above — so compare within a table, never across): `main` (#556's cull) **172.5 / 177.3 s**
at 3,184 facets against one level's **142.4 / 140.7 s** — the same 1.24x — and 86.4 / 87.6 against
90.7 / 88.1 s at 2,448, even.

⚠️ **A SECOND LEVEL WAS BUILT AND MEASURED, AND IT IS SLOWER — not shipped.** A `ClusterHierarchy` of
sizes (32, 8), each level cut from one Morton order so a 32-cluster is exactly four 8-clusters, with the
same pairwise cluster test at every level and only surviving pairs expanded. It kept exactly the dense
set (the exactness test passed at (32, 8) and (20, 5, 1)). The probe that motivated it held up: of the
3,839 surviving 32-cluster pairs per receiver, **29%** of their 8×8 children survived, so the member test
fell **3.93M → 1.15M** pairs (3.4x). But the whole cull went only **~41 → ~33 ms** a receiver serially
(1.24x, not the ~2x projected), and at the default 4 threads the build got **slower**: **158.5 / 158.6
and 159.9 / 155.1 s** at 3,184 facets against one level's 141-142 s, and 83-96 s (a wide spread) at
2,448. The profile blamed the round trips: each receiver makes twice as many small compiled calls, and
most of the cull's time is `numpy.asarray` waiting on their results; under four threads those
synchronizations contend with the reject and clip passes running beside them. **Fewer member tests are
not worth more dispatches.** Anyone retrying depth needs the levels fused into one compiled call per
receiver (or batched across receivers), not stacked as separate calls. Dropped as dominated.

Tests, mutation-checked (8 mutations, 7 red): dropping the member half-angle from the enclosing cap,
the sphere radius, or flipping the sphere test fails the exactness test; ignoring `pair_limit` in
either stage fails `test_the_cull_forms_no_more_pairs_per_call_than_its_limit` (now watching both
stages); losing the Morton order fails `test_clusters_are_compact_rather_than_arbitrary`.
**Dismissed**: not propagating a member's unusable flag to its cluster changes nothing observable —
every unusable member also has a cone cosine at or below the floor, so its own half-angle already pushes
the enclosing cap to a right angle and flags the cluster — and the guard stays as the explicit rule.


## THE SCENE: LAMPS KEPT OUT OF THE TRANSFER, AS A LIBRARY FUNCTION (2026-10-05)

`scene.py` makes the recipe `aquaflux_reflecting.py` hand-assembled (and this file documented under
"the transfer refuses it ... keep the lamp out of the surface set") the library's: `solve_scene(Scene(
lamps, reflectors, occluders, absorption, volume, surfaces, lamp_samples, settings))`. It is what a
radiation **case file** builds (`.claude/rules/case.md` → Radiation cases); the old scripts are deleted.

- **Steps**: `check_profiles(lamps)`; with reflectors, the lamps' `direct_irradiance` at
  `subtriangle_centroids(reflector vertices, lamp_samples)` (the `k^2` equal-area sub-triangle centroids,
  default `DEFAULT_LAMP_SAMPLES = 4`), averaged per facet, is the `external_irradiance` of ONE
  `surface_irradiance` solve on a model built with **zero receivers** (`np.zeros((0, 3))` works); the
  radiosity is then `emission + rho H` (reflectors must not emit — refused — so `rho H`), re-gathered as a
  Lambertian set. Volume: `G_direct` and `G_reflected` by `streamed_fluence_rate` when anything can shadow
  (a body, or self-occlusion not `NoOcclusion`), else `summed_fluence_rate`. Surfaces: `direct_irradiance`
  through a mask built per pass of `receivers_per_pass(pair_limit, n_facets x n_bodies)` points.
  `medium_absorbed_power = sum(a(x) G V)` via the new **`Absorption.sample(position)`** (abstract;
  `UniformAbsorption` broadcasts, `VoxelAbsorption` already had it).
- **Gate**: `test_a_lamp_kept_out_of_the_transfer_lights_the_box_as_the_model_does_with_it_inside` — a
  Lambertian lamp kept out equals the model with it inside to 1e-11 (fluence and radiosity) at ONE
  receiver point per facet and ONE lamp sample, where both evaluate the same projected solid angles; the
  reflected share is > 30 %, so the agreement is not the direct light's. Plus: `lamp_samples` 8 is within
  5 % of `lamp_samples` 1's error against the 12-point transfer (lamp light averaged, not sampled once),
  wall irradiance = a direct gather of the solved radiosity, pass size changes nothing (1e-13) with a
  sphere shadowing, medium power, sub-triangle centroids.
- ⚠️ **A point on a reflecting surface IS lit by its own facet unless that facet is left out of the
  gather, and the argument that it is not was measured only on axis-aligned walls (2026-10-07).** The
  first version masked it out, its test (an axis-aligned box) showed the mask did nothing, and the mask
  was dropped on the reasoning "in front -> behind the receiver half-space, behind -> behind the emitter,
  exactly in-plane -> emitter cosine 0". On an axis-aligned plane the in-plane heights round to exact
  zero and that holds. **Off one it does not**: the corners' heights round to ~1e-17 of either sign,
  `clipping.decidable_heights` cannot always snap them, the clip keeps the in-plane triangle and the
  contour integral returns the full hemisphere; and `Lambertian.radiance_per_exitance_at` gates on
  `cos > 0`, which rounding noise passes about half the time. Measured: `inward_box(3)` floor, rotated
  by Euler `(0.3, 0.7, 0.2)` and shifted, `emission=1`, `direct_irradiance` at
  `subtriangle_centroids(vertices, 4)` with each facet's own normal: **51-67 % of the points read
  `E = 1`** (the full exitance; which share depends on the exact rotation convention and shift); at the centroids 0 (the zero-direction
  guard in `_emitter_direction`), axis-aligned 0. Face-centre receivers (every radiation case file) sit
  on the shared apex of their centre-fan triangles and read 0 — by luck, not design.
  **FIX (decided 2026-10-07): exclude the receiver's own facets as SOURCES in the gather**, by
  `direct_irradiance(..., receiver_facet=)` — `(n_points,)` or `(n_points, k)` rows, `-1` padding, the
  same form `build_visibility` takes — and `scene._irradiance` passes `_own_facets`' rows to it on BOTH
  paths (unshadowed and per-pass masked). **Rejected: offsetting the points a hair in front of their
  facet** (what #604 step 1 did for lamp light on lamp facets, `1e-12` of the scene's size -- since
  DELETED, see #604 STEP 1 below). It is correct only while the offset clears `clipping._SLACK`'s band at that facet's size, so it
  silently couples a scene-scale constant to the clip's rounding tolerance; exclusion is exact (a flat
  facet sends nothing into its own plane), needs no length scale, and is the transfer's `F_ii = 0`
  convention applied in the gather. Pinned by `test_a_point_on_a_tilted_facet_is_lit_by_the_others_and_not_by_its_own_facet`
  and `test_every_facet_a_row_names_is_left_out_and_a_minus_one_names_none` (gather; the reference is
  the same points 1e-9 in front, unnamed) and `test_a_point_inside_a_tilted_reflecting_facet_is_not_lit_by_that_facet`
  (scene, both paths); each mutation-checked, `tests/unit/radiation_references.tilted` is the fixture.
  What ALSO needs the facet is the **ray test**:
  a ray from any other facet ends in the facet under the point and `RayCastOcclusion` counts that (far end
  inclusive) — the "two exclusions" defect below. So `SurfaceReceivers.reflector` names the body, and
  `_own_facets` passes **every** facet of it the point lies on (ties within 1e-12 of the body's extent)
  as `receiver_facet`. ⚠️ **Several, not one**: a mesh face centre is the shared apex of all its fan
  triangles, and a point on a shared edge lies on two; with only the nearest excluded, 4 of 16 floor
  points of a regular box (on quad diagonals) came back shadowed. **`receiver_facet` now takes
  `(n_receivers, k)` rows (`-1` padding)** in `build_visibility`, `RayCastOcclusion` (`_exclusions`, both
  walks) and `pairs_are_cut(target=)`; `SilhouetteOcclusion` takes them too since 2026-10-06, because the
  scene now passes every gather its points' normals (ORIENTED RECEIVERS ON NO FACET, above) — without a
  `receiver_normal` it still refuses rows naming more than one facet. Pinned by
  `test_a_point_on_a_reflecting_wall_is_not_shadowed_by_the_facet_it_lies_on` (ray test with the
  exclusion = no occlusion to 1e-12 on a convex box; without it, < half).
- **Measured at mesh scale** — see `validation/ray_effects_room/README.md` (the case files and their
  agreement with the scripts they replaced).
- ⚠️ **ONE `Scene.settings` SERVES SEVERAL SURFACE SETS, SO A BODY IT NAMES MAY BE ANY SET'S (2026-10-06,
  re-applied over #604 step 1 on 2026-10-08).** Every mask the scene builds has as its sources either the
  **lamps alone** (their light on the exchange's samples, on `SurfaceReceivers` and in the volume) or the
  **exchange** (`_Exchange.surfaces`: reflectors then lamp facets, both sets' names — its transfer under
  `self_occlusion`, **its zero-receiver mask under `receiver_occlusion`**, and the reflected gathers).
  `SilhouetteOcclusion` refuses a `two_sided` name its surface set lacks (the misspelling guard), so a
  sheet among the reflectors declared two-sided made every lamps-only build raise (before #604 step 1 the
  reverse too, through a reflectors-only model). **Decided with the project owner** over per-set
  settings (two objects whose other fields could drift, against "the two cannot be built differently")
  and over moving the refusal out of the strategy into every entry point (a new entry point forgetting it
  loses it silently): `Scene.__post_init__` runs `settings.check_bodies(both sets' solid_names)` once,
  and every build gets `scene.settings_for(sources)` = `settings.for_bodies(sources.solid_names)`, a copy
  with each occlusion choice cut to those sources' bodies (`SelfOcclusion.for_bodies`/`check_bodies`,
  identity/no-op by default; `SilhouetteOcclusion` filters `two_sided`). On the exchange the cut keeps
  every name; it is applied anyway so no build reads the settings uncut. The strategy keeps its strict
  refusal (`_either_side` calls `check_bodies`), so standalone `build_radiation_model`/`build_visibility`
  still catch a misspelling. (Lamps and reflectors may not share a body name since #604 step 1, so no
  name is ambiguous.) Pinned by `test_a_sheet_declared_two_sided_shades_from_behind_whichever_set_it_belongs_to`
  (a baffle among the lamps and a shelf among the reflectors, each facing away from the point it shades:
  declared → that point's part 0 to 1e-12, undeclared → equal to `NoOcclusion` to 1e-12, and the
  open-sheet warning names exactly the undeclared ones), `test_a_scene_refuses_a_sheet_that_neither_set_has`
  (both fields) and `test_settings_cut_down_to_a_set_keep_its_own_sheets_and_nothing_else_changes`.
  Mutation-checked on the merged code (2026-10-08), 10 of 11 red: no narrowing, narrowing to nothing,
  inverted or no-op filter, no union check, checking against the lamps only, `receiver_occlusion` left out
  of `_occlusions`, the lamps-only `_fluence` and `_irradiance` unnarrowed, and the standalone check
  dropped (caught by `test_a_misspelt_sheet_is_refused_rather_than_left_one_sided`). **Dismissed,
  equivalent**: the exchange's model built from the uncut settings — the exchange holds every body of
  both sets, so the cut keeps every name (it was red before #604 step 1, when that model held the
  reflectors alone).
- ⚠️ **A reflecting body does NOT block the lamps' direct light.** The lamps-only masks' self-occlusion is
  over the lamps' own triangles; the reflectors enter only the exchange. (The other half — a lamp-set body
  not blocking reflected light — was fixed by #604 step 1: the reflected gather is from the whole
  exchange.) Only `occluders` shade both. Harmless for an enclosure the points sit inside; wrong for a
  reflecting baffle standing between a lamp and a point. The sheet test above depends on it (its shelf
  leaves the lamps' light alone) — not decided, recorded so it is not mistaken for physics.

### #604 STEP 1: LAMPS REFLECT AND SHADOW IN THE EXCHANGE (2026-10-07)

**Agreed with the user** (2026-10-07; full plan on #604): #604 is sequenced by GENERAL capability, not by
lamp physics -- (1) emitters that also reflect, here; (2) transparent solids (index + absorption, exact
paths through analytic bodies); (3) the equivalent surface as a general baking step; (4) curved specular
reflectors; (5) path cuts. Two decisions for this step: **straight-through transmittance is deferred to
(2)** (a surface's own triangles stay one opaque layer -- `visibility.py`); and **lamp facets are ALWAYS
in the exchange** (default reflectance 0), which fixes the pass-through and costs transfer size.

- **What changed.** `_Exchange.of(scene)` is the reflectors' facets then every AREAL lamp facet, built by
  `from_triangles` with the lamp bodies' names after the reflectors', `point_sources=()`, no emission;
  `None` when there are no reflectors and no lamp reflects (nothing is ever sent back). `_solve_exchange`
  gathers the lamps' light on all of them (sub-triangle centroids, as before), solves once, and splits
  back. `SceneSolution` gains `lamp_irradiance` (`(n_lamp_facets,)`, zero on point sources) and
  `lamp_absorbed_power` (`sum (1 - rho) E A`); `radiosity` / `reflector_irradiance` stay the reflectors'
  (and are `None` without them). The reflected gather is from the whole exchange, so **a lamp's triangles
  now shadow light the walls reflect past it** -- before, a "black" lamp let it through (the scene's own
  docstring said black; nothing tested shadowing). `Scene` now refuses lamps and reflectors sharing a body
  name, and specular reflectance on either (the scene carries none). `SurfaceReceivers.reflector` may name
  a lamp body; `_own_facets` returns `None` for a body not in the set asked about, and the DIRECT gather
  now also leaves out the lamp facet a point lies on.
- ⚠️ **A POINT INSIDE ITS OWN FACET CAN READ THE FACET AS A WHOLE HEMISPHERE.** The first version
  sampled the lamps' light on lamp facets exactly in their planes, and on a 12-sector drum a lamp lit
  itself: absorbed over emitted **1.22** with black walls. Off an axis-aligned plane the corners' heights
  round to ~1e-17 either way; where they stay undecidable the kernel keeps the in-plane triangle and
  returns the hemisphere (the self-facet convention). Measured on one tilted 9-facet plane: **51%** of
  sub-triangle centroids read `E = M` from their own facet (centroids read 0 -- the zero-direction guard);
  axis-aligned, none. **Fixed (#615) by naming the facet as the points' own** (`own=` -> the gather's
  `receiver_facet`), which leaves it out as a source exactly; the points stay IN the facet's plane.
  ⚠️ **Never move them BEHIND the facet** (the emitter gate would also zero it, and that was the first
  fix): behind a lamp is outside the fluid, so a body holding the fluid (a `CadFluid`,
  `Outside(Difference(box, lamp))`) refuses them (pinned by
  `test_the_lamps_light_on_a_lamp_is_taken_on_the_fluid_s_side_of_it`).
- ⚠️ **There is no offset IN FRONT any more either -- `_IN_FRONT_OF_OWN_FACET` (1e-12 of the scene's
  size) was DELETED as redundant (#615, 2026-10-07).** It had two jobs. *Self-lighting* is done exactly
  by the exclusion above. *Keeping the samples where a fluid body admits them* never applied on its own:
  the samples lie in their facet's plane with the same rounding as the facet's centroid, which
  `refuse_points_inside` checks too, so a body that admits the facets admits their samples and one that
  refuses them refuses the scene first. Measured (2026-10-07, jax 0.10.2, CPU, x64, macOS arm64): a box
  lamp `half_sizes (0.08, 0.08, 0.2)` flush with `Outside(Difference(unit box, Box(..., axes=R.T)))` in
  `inward_box(3)`, `RayCastOcclusion`, 4 orientations (axis-aligned and three Euler tilts): at
  `tolerance=0` the scene is refused for 2-9 of its 12 facet centroids **with or without the offset**
  (tilted, 38-134 of 192 in-plane samples also read inside; axis-aligned, `0.08` is inexact in binary
  and 6 centroids sit at +1.1e-16); at `tolerance=1e-12` nothing is refused and offset 1e-12 vs 0 agree
  to <=6.1e-16 in `lamp_irradiance`. A 12-sector drum with an inscribed `Cylinder` (half-length 0.25 or
  0.24) gave the same verdict: refused by its cap centroids regardless, or no sample on the surface at
  all. Pinned by `test_a_tilted_lamp_flush_with_the_water_s_wall_is_lit_as_without_the_water` (the
  samples do round onto the lamp's side -- asserted -- and the water at 1e-12 changes nothing).
  ⚠️ `Box(axes=)` takes the axes as ROWS: a rotation `R` applied as `x @ R.T` needs `axes=R.T`; with
  `axes=R` the body is misplaced by ~0.1 and every refusal reads as a rounding problem.
- **The same hemisphere at the scene's SurfaceReceivers on a tilted reflecting facet is FIXED (#615,
  2026-10-07) by the same exclusion** -- see "THE SCENE" above for the measurement.
- **Energy, measured** (12-sector drum, r 0.12, 0.5 high, in `inward_box(3)`, walls rho 0.8, lamp rho
  0.4, default 6-point rule; jax 0.10.2, CPU, x64, Linux, 2026-10-07): walls + lamp absorb **1.014** of the
  emitted power under `RayCastOcclusion` (the one-ray-per-pair mask of the drum's shadow on a coarse mesh);
  **1.59 under `NoOcclusion`**, which is not a defect of the scene -- with nothing shadowing, wall light
  passes through the lamp AND is absorbed by it. A drum in a box is not convex; do not run it unshadowed.
- **Gates**: a lamp that emits and reflects, kept out, equals the model with it inside (fluence, lamp
  landing, wall landing, 1e-11, one point per facet); a black drum shadows reflected light exactly as the
  model with the drum inside it does (`RayCastOcclusion`, 1e-11; gathered from the walls alone, >5%
  brighter somewhere); lamps alone (two reflecting plates) equal the model; a point-source lamp lands
  zero; points on a lamp named by its body equal the same points 1e-6 off it (1e-4); the energy test
  above (2.5%). **Mutation pass, 13 breaks, 12 red**: lamps left out of the exchange, lamps reflect
  nothing, absorbed by rho not 1 - rho, no offset, offset behind (both measured under the since-deleted
  offset; "behind" survives as the fluid-side test), no exchange without reflectors, lamp
  receivers not excluded, landing from the wrong rows, shared names allowed, the case ignoring
  `Lamp.reflectance`, `unaccounted_power` keeping the lamps' share, the reflectance unchecked; the
  shadow test alone goes red under "lamps left out". A lamp sample's own facet in its shadow test was
  dismissed as inert under the offset; **with the samples back in the facet's plane it is load-bearing
  again** (a ray from another facet ends in it -- the "two exclusions" defect), and `own=` supplies it;
  `SilhouetteOcclusion` also reads it to measure a receiver by the projected solid angle. ⚠️ **Only a
  SECOND lamp exposes it**: a single convex drum cannot light itself, so every one-drum test stays green
  with it removed (all 50 scene + case-radiation tests did). Measured (2026-10-08, jax 0.10.2, CPU, x64,
  macOS arm64): two 12-sector drums (r 0.1, half-height 0.25, at x 0.3 and 0.7) in `inward_box(3)`,
  walls rho 0.6, lamps rho 0.4, `RayCastOcclusion`, default 4x4 samples -- removing it changes **every**
  lamp facet's `lamp_irradiance`, by up to **51%** (lamp absorbed 0.664 -> 0.550 W). Pinned by
  `test_one_lamp_s_light_on_another_lands_as_the_model_with_both_inside_says` (two 8-sector drums, the
  model with both inside as the reference, 1e-11; red with the exclusion removed for lamp samples only). ⚠️ **The reflector samples pass no `receiver_facet`
  (they lie on no LAMP facet), so under `SilhouetteOcclusion` their shares of the lamps are taken by the
  plain solid angle** -- pre-existing, and the API cannot express "a surface receiver on no source
  facet"; raised with the user as a follow-up.
- **Cost**: the transfer is `n^2` in reflector + lamp facets now. Not measured at mesh scale. The
  `ray_effects_room` cases' field cannot move (black flat window flush in the ceiling, `NoOcclusion`:
  a black facet sends nothing and shadows nothing); not re-run.
- **Case**: `Lamp.reflectance` (see `case.md`), results `lamp_absorbed_power`, subtracted from
  `unaccounted_power`.

### #604 STEP 2a: TRANSPARENT SOLIDS AND THE REFRACTED DIRECT GATHER (2026-10-08)

**Agreed with the user** (2026-10-08): a transparent solid is a convex analytic body with an index and
an absorption; the path between a source and a receiver crosses the surface of every region holding
exactly one of them, once, in a fixed order; the crossings come from Fermat's principle, solved per
path and differentiated by the implicit function theorem; the weight is the existing closed-form solid
angle evaluated on the **arrival directions of the paths to a source triangle's three corners**, times
the corners' mean of radiance x Fresnel x leg absorption, times `(n_receiver / n_source)^2`. Three
choices put to the user: a region holding **neither** end (a neighbour's sleeve) is crossed **straight**
(its absorption on the chord and the Fresnel loss at the straight chord's angles, no bending) with the
error bounded by the tracer and exact handling a later step; **reflected branches are deferred to step
4** (each Fresnel reflection is a loss); **library only**, case-file wiring a later small PR. Step 2 was
split by me into 2a (this: regions, Fresnel, the path solve, the refracted gather and its leg masks) and
2b (the model, the scene, the facet-to-facet exchange, the four-lamp tracer comparison, cost at scale)
-- the same scope, two PRs, said to the user at the time.

- **What is built.** `refraction.py`: `fresnel_transmittance` (unpolarized, zero past the critical
  angle), `Transparent` (a `ConvexSolid`, an index, an `Absorption`, regions `inside` it), `Media` (the
  surrounding index and absorption and the outermost regions; `nodes` depth first, `parents`,
  `region_of` refusing a point within `1e-9` of the scene's size of a surface, in a region but not its
  holder, or in two siblings), `Chain.between` (crossings out of every region round the source and into
  every one round the receiver; the medium of each leg; the regions each leg passes straight through),
  `solve_paths` -> `Paths` (departure, arrival, transmittance, valid, crossing points). `refracted.py`:
  `refracted_fluence_rate`, `refracted_irradiance`, `build_refracted_visibility` ->
  `RefractedVisibility` (wraps a `Visibility`: per body, receiver and facet, any leg of the path to the
  facet's CENTROID blocked; the surface's own triangles one ray per leg, the source facet excluded on
  the first leg). Pairs in one medium are the direct gather's and contribute nothing here. A point
  source in another medium than a receiver is refused (no area to spread through a surface).
  `solids.ConvexSolid.face_distances` is the one solids addition: each face's own signed distance, so
  a crossing is held to ONE smooth face whose gradient is its normal.
- **Exported from `aquaflux.radiation`** (`Media`, `Transparent`, `Chain`, `Paths`, `solve_paths`,
  `fresnel_transmittance`, `refracted_fluence_rate`, `refracted_irradiance`,
  `build_refracted_visibility`, `RefractedVisibility`; docs group "Transparent solids"), and **not in the
  model, the scene or the transfer -- that is 2b**; the package docstring's refraction limitation says so.
- **`tools/sibling_builders.py` pairs `build_refracted_visibility` with `build_visibility`** (both build a
  `Visibility`, six shared parameters), reviewed and left: `media` is the refracted path's own;
  `body_culling` certifies tiles of STRAIGHT segments, which a bent path is not; `receiver_facet` (receivers
  sitting on facets) is not needed by volume receivers and arrives with the exchange in 2b.
- **The corner mean is not the centroid**: at equal indices in a strongly absorbing medium (40 /m, the
  test drum of 24 x 8 facets) the refracted gather differs from the direct one by 1-8 %, both being
  second-order estimates of the absorption across a facet with different constants; at the far test
  point 1.3 % -> 0.25 % from 24 x 8 to 48 x 16 facets. Equal to 1e-12 without absorption.
- ⚠️ **NEWTON ON THE STATIONARITY CONDITIONS, STARTED FROM THE STRAIGHT LINE, RUNS AWAY NEAR AN
  INTERFACE -- the solve is a DESCENT ON THE OPTICAL LENGTH.** A source seen at grazing incidence
  through a surface close to the receiver has its true crossing far from where the straight line meets
  the surface (0.2 mm against 0.6 mm at 3 mm separation), and plain Newton diverged there; the path
  was reported absent and the facet dropped. On the sleeve this read as **-1.5 % to -2.1 % within
  1-2 mm of the sleeve**, converged under refinement (so it looked like physics), and a flat-interface
  quadrature with the receiver 0.2 mm off the plane read **-18 %**. A transmitted path minimizes the
  optical length among paths crossing each surface once, so the length is a merit function: each step
  is the Newton step on the Lagrangian (the optical length's curvature, written out, plus each face's
  times its multiplier), else tangential steepest descent, moved back onto the faces (`onto`: exact
  in one move for a plane, tube or ball) and taken at the longest of a ladder of halvings that
  shortens the path. Inside `_BASIN = 1e-6` of the separation a Newton step is taken whole (the length
  cannot resolve the fall), and convergence is the Newton step under `1e-10` of the separation --
  a gradient tolerance was unreachable on short legs, where the curvature makes the gradient large.
- ⚠️ **I ATTRIBUTED THAT -2 % TO PHYSICS FIRST, AND IT WAS NOT.** Paths to arc points far along the
  axis came back absent and I explained it by the axial direction cosine Snell conserves -- wrong:
  the air leg can run nearly axially, so those paths exist, and the globalized solve finds every one.
  The check that found it was the closed form, not the reasoning: **a dropped path is the solver's
  until a forward trace or a quadrature says otherwise.**
- ⚠️ **A `while_loop` INSIDE A `while_loop` UNDER `vmap` NEVER ENDS.** The line search was a nested
  loop; once one pair of a batch had finished its outer loop its inner loop stopped updating while
  its own condition stayed true, and the batch hung (reproducible with two sources). The line search
  is now a fixed ladder of 41 step sizes evaluated together.
- ⚠️ **`Chain.beside` must leave out the crossed regions.** A leg leaving a region starts on its
  surface; counted as passing straight through it, its chord came out a rounding long or zero and the
  straight-through Fresnel factor at grazing zeroed paths at random (a flat-interface quadrature read
  -13 to -16 %, not converging).
- **Compiled once per chain and shape** (`_solve_paths`, `eqx.filter_jit`): eager calls cost 25 s
  first and 4 s each after; compiled, 3.2 s to trace, 3.2 s to compile, ~1-2 ms a call. The optical
  length's gradient and curvature are written out and each body's faces are built once per path,
  which took tracing from 5.1 to 3.2 s (one sleeve chain, jax 0.10.2, CPU, Linux, 2026-10-08).
- **Checks** (`tests/unit/test_radiation_refraction.py`, `test_radiation_refracted.py`): Fresnel
  against the amplitude form at 60 angles, reciprocity and the critical angle; the innermost region
  and every refusal; chains (out of a sleeve, into one, sibling to sibling); paths against a
  general-purpose optimizer on the optical length (a flat interface incl. the grazing case; a skew path
  out of a sleeve); a crossing off the end of its face is no path; the implicit derivative against a
  finite difference (path and gather); a region beside the path at normal incidence = `(1 - R)^2`
  times its excess absorption; equal indices = the direct gather to 1e-12 (fluence and irradiance);
  **a flat interface against a quadrature over the receiver's directions** (both index orders, both
  media absorbing; and the irradiance), 2.5e-3 at 32 cells a side, second order: air->water -0.60,
  -0.15, -0.04 % at 16/32/64; glass->air -0.52, -0.13, -0.03 %; receiver 0.2 mm off the plane -0.46,
  -0.11, -0.03 % at 32/64/128; the mask follows the refracted leg and not the straight line.
- **Against the tracer** (`validation/sleeve_optics/check_refraction.py`, numbers in its README): one
  sleeved lamp, 95 % UVT, 40M rays -- the transmitted paths agree within 1.5 standard errors at every
  radius 12.5-60 mm; the reflected paths left out are worth < 0.5 %; the straight gather reads 12-25 %
  high. Gather 90 s for 106,496 arc facets and 7 points, compile included (4-core Linux; 271 s before
  the compile caching and the written-out derivatives).
- **Mutation pass, 25 breaks, 25 red after two tests were added** (`PYTHONDONTWRITEBYTECODE=1`, the tests
  aimed at each): Fresnel's half, total reflection transmitting, the outermost region winning, each of
  the three refusals, crossed regions counted beside a leg, the leg after leaving taken as the region,
  the wrong face, no boundary check, Fresnel indices swapped, no leg absorption, the straight-through
  exit Fresnel and excess absorption, no basin step, no line search, no implicit tangent, the index ratio
  inverted and dropped, the plain kernel for irradiance, the mask's first leg only, corners mis-shared.
  **First green, now red**: a single corner instead of the corner mean (symmetric fixtures cancelled it;
  `test_with_every_index_equal_the_corners_absorption_converges_on_the_centroid_s`) and any corner seen
  instead of all (`test_a_triangle_with_a_corner_that_has_no_path_carries_nothing`). **Dismissed**: the
  crossing-direction check -- the descent finds the shortest path, which never turns back at a surface,
  so no reachable input fails it; kept as a guard on what the solve returns.

### #604 STEP 2b-i: TRANSPARENT SOLIDS IN THE SCENE'S DIRECT LIGHT (2026-10-08)

**Agreed with the user** (2026-10-08, after 2a merged as #638): 2b in two PRs — **2b-i** (this: media in
the `Scene`, the lamps' direct light routed by medium, the straight-through factor, the four-lamp
black-wall tracer check) and **2b-ii** (the refracted facet-to-facet transfer, the model's `media=`, the
reflecting-wall tracer check, cost at Sozzi scale). For 2b-ii, decided: refracted transfer pairs keep the
**surrounding absorption live** (per corner, the water leg's length stored sparsely beside the frozen
solid angle, Fresnel and in-region absorption; indices and region absorptions frozen, a traced `Media`
refused by the model), and use the transfer's **own receiver quadrature** (six points), cost measured
before anything coarser is considered. ⚠️ **Deviation, said here so it is not mistaken for scope**: the
2b-i option text read "media in Scene and the model's direct and volume gathers"; the model's gathers
went to 2b-ii with its transfer, because a model with media but a straight transfer would carry lamp
light refracted and the walls' light not.

- **`Scene.media`** (`Media | None`). With it, `Scene.absorption` must be unset (refused: the medium
  would have two absorptions) and `Scene.medium` reads `media.absorption`. **Refused, `NotImplementedError`,
  when anything exchanges** (`_Exchange.of(scene) is not None`: reflectors, or a lamp that reflects) —
  the transfer has no refraction until 2b-ii.
- **Routing** (`scene._by_medium`): each point is lit by the sources in **its own** medium along straight
  lines, in `media.absorption_of(its medium)` (not the scene's — a point in a sleeve's air gap is lit
  through air), and by the sources in **other** media along refracted paths (`scene._refracted`, a mask
  per pass from `build_refracted_visibility`, then `refracted_fluence_rate` / `refracted_irradiance`).
  With media, the straight half always builds masks (streamed for the volume, per pass for oriented
  points), because the mask is what removes the cross-medium pairs.
- **`Visibility.through`** (float `(n_receivers, n_facets)` or `None`), built by
  `build_visibility(..., media=)` from `refraction.straight_through`: for a pair in one medium, the
  Fresnel losses where the straight segment enters and leaves each region it passes and those regions'
  absorption **in excess of the medium's own** (telescoped through nesting, as 2a's `_straight_through`);
  **zero** for a pair in different media (its light is the refracted gather's). Frozen. `surviving_fraction`
  multiplies it in; **the gathers now read a mask's layers by name** (`Visibility.layers()` ->
  `(kinds, (array, axis) pairs)`, `surviving_from_layers`; `gather._Layers` carries the kinds), not by
  position — the positional `blocked, *hidden` unpacking could not take a third layer. Built
  `None` when the media have no regions, so a scene without regions costs nothing new.
- `straight_through` evaluates pairs a pass at a time, `min(pair_limit, PASS_PAIRS)` padded to a power
  of two (`_through_pairs`, `eqx.filter_jit` over the beside tuple, compiled per medium). A medium with
  no region inside it is all ones without evaluating anything.
- **`build_refracted_visibility(..., receiver_facet=)`**: the last leg of a refracted path ends in the facet
  a point lies on, and the ray test read that facet as a blocker — so points on one lamp lit by another
  were dark. Excluded on the last leg as `build_visibility` excludes it on the one leg.
- `Media.region_of_facets(surfaces)` is the one home of "a facet lies in one medium" (moved from
  `refracted._facet_regions`; there is no such function any more).
- **Tests** (`test_radiation_scene_media.py`, and the straight-through test in
  `test_radiation_refraction.py`): equal indices and no absorption = the scene with no media, to 1e-10,
  fluence and irradiance, points in the water and in a gap, two lamps; a gap point takes its own lamp
  through the air's absorption, and the medium's absorbed power uses each point's own coefficient; a quartz
  ball centred on the line in water = the straight gather times `(1 - R)^2 exp(-Δμ 2a)` to 1e-12; points on
  a lamp named by its body = points 1e-7 in front of it (RayCast); the refusals; `straight_through` across a
  sleeve through its axis = the closed form (Fresnel at all four surfaces, the two excesses telescoped) to
  1e-12, cross-medium zero, misses one, independent of the pass size. **Mutation pass, 14 breaks, 14 red**
  (after adding a third facet so a pass-index slip is visible — with one column per medium
  `flat // n_columns` and `flat % n_rows` coincide): refracted fluence or irradiance dropped, the streamed
  or the held mask without media, one absorption for every medium, one coefficient for the medium's power,
  no last-leg exclusion, `through` ignored in `surviving_fraction` or in the segment gather, the beside
  regions not evaluated, cross-medium pairs ones, the pass misindexed, both refusals.
- **Four-lamp tracer check** (`validation/sleeve_optics/check_array.py`, table in its README; 50 mm square,
  95 % UVT, black wall, 212,992 facets, 20M rays a lamp, one run, Linux 4 cores, 2026-10-08): within the
  tracer's error where no light passed another sleeve; **2.8-3.5 % low (2.3-3.3 standard errors) where it
  did** — larger than the share that entered a sleeve (1.0-2.4 %), so not decomposed (straight lines
  meeting a neighbour's arc where bent paths miss is a candidate, unmeasured). Reflections left out:
  5.5-7.5 % with four lamps (under 0.5 % with one). aquaflux 977 s, compile included.

