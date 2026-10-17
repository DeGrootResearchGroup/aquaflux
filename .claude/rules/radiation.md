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
| `refraction.py` — `Transparent` / `Media` (convex regions with an index and an absorption, nested; `Media.region_of_facets`), `fresnel_transmittance`, `Chain`, `solve_paths` -> `Paths`: the transmitted path between two points by a descent on the optical length (#604 step 2a, 2026-10-08); `Chain.routes`, `straight_reach`, `_only_saddles` / `_shot` and `shooting.py` (a sleeve's wall alone, its saddle paths shot) (#604 step 2b-i) | **BUILT** |
| `refracted.py` — `refracted_fluence_rate` / `refracted_irradiance` (each triangle's solid angle on its corners' arrival directions) and `build_refracted_visibility` -> `RefractedVisibility` (every leg of the centroid's path; `receiver_facet` excluded on the last leg since 2b-i) (#604 step 2a) | **BUILT** (in the scene's direct light since 2b-i; not yet in the model or the transfer — 2b-ii) |
| `units.py` — lamp watts to exitance, ultraviolet transmittance to absorbance | **BUILT** |
| `scene.py` — `Scene` / `solve_scene` / `SceneSolution`: lamps' EMISSION kept out of the transfer (any profile, incl. IES) while their areal facets EXCHANGE reflected light (reflect by `diffuse_reflectance`, shadow, absorb; #604 step 1, 2026-10-07); every triangle of the scene shadows the direct AND the reflected light (`_Direct`, #622, 2026-10-10); reflecting surfaces, bodies, medium, `VolumeReceivers` and named `SurfaceReceivers`; two-sided sheets declared on each set and solved as two faces (TWO-SIDED SHEETS, #689); what a radiation case file builds (2026-10-05) | **BUILT** |

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
   one behind it a negative one. The clip (`clipping.clip_to_halfspace(..., 4)`) emits a fixed
   four-slot loop in which an empty slot repeats the last survivor, so the shape is static and
   the repeats are zero-length edges contributing no angle.
3. **Take the magnitude.** The sign of either contour sum records only the order vertices
   happen to be stored in, which for an imported surface file is arbitrary. Orientation is the
   normal's job. The test fixture winds its faces inconsistently on purpose.
4. **Guard the square root *inside* the root, not after it.** `v / where(zero, 1, sqrt(sq))`
   keeps the forward value finite and still yields a **NaN gradient**, because reverse mode
   visits `d/dx sqrt(x)` at zero and the unselected branch's NaN propagates through the
   selection. Write `sqrt(where(zero, 1, sq))`. Zero-length edges are the *normal* case here,
   not a degenerate one — the clip manufactures three of them for every fully visible triangle
   — so this is on the main path, and the whole project is gradients.
   ⚠️ **It recurred three more times (#620)** -- through `**` (`profiles._forward_power` now substitutes
   the base inside the power), an unguarded `sqrt(dot(...))` in both absorptions (now
   `aquaflux.vectors.norm`) and `arctan2(0, 0)` in `PhotometricProfile.angles` -- so the rule covers any
   primitive with an infinite or `0/0` slope where the selection discards its value. ⚠️ A Lambertian
   fixture cannot see a NaN direction. Fixes, tests and mutation pass: `.claude/notes/radiation-log.md` →
   "Four implementation details that are load-bearing, each measured".

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
The point-source **labels** are carried through `division.origin` (#632) — re-inferring them from
the refined areas relabelled a labelled facet with an area as areal — and a labelled facet wide
enough to be split is refused. **Both ends of an index are checked** (`surfaces._check_indices`,
used by `from_triangles` and `with_optics` alike): a `-1` in `solid_id` or `profile_index`
selected the last body or profile. An ASCII STL triangle before the first `solid` line is refused
for the same reason (it got `solid_id = -1`). A specular body with no areal facet is refused by
name, and a specular exchange whose every path is hidden is an empty `(0, n, n)` stack.

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
- **Moved to the log** (`.claude/notes/radiation-log.md` → "A measured luminaire is a table over TWO
  angles, so a profile is asked about a DIRECTION (2026-09-28)"): the half-turn-span symmetry fix (#621:
  symmetry is read off the horizontal SPAN, a quarter-turn span must start at 0, `h` is measured from the
  table's first angle), the pole's 2.2 % spread in `ushio_b1.ies` and the `_POLE_SPREAD = 5 %` warning
  (#632), and the unit-direction gather timing.
- **Normalization is closed form and over the FRONT HEMISPHERE** — `front_hemisphere_flux`, the exact
  integral of the bilinear table times `sin gamma` over `[0, 90]` degrees, clamped. That is what a flat
  emitter can radiate, and what OOR's BC normalizes over (its outgoing rays), so both emit exactly `P`.
  `Photometry.flux` is the table's own range and is the lamp power when the file is absolute: the B1
  file integrates to **118.826 mW** (a trapezoid estimate gave 118.6 and was quadrature error).
- **The radiance is floored at `MIN_COSINE = 1e-3`** (OOR's `IES_MIN_COS`), since a table with light
  at `gamma = 90` would otherwise need unbounded radiance along the facet's plane.
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

## SPECULAR REFLECTION: THE REFLECTANCE SPLIT AND THE MIRRORS (#537 PR 1, 2026-10-05)

**Current status of #537 (all four PRs merged; plan, records, tests and mutation passes in
`.claude/notes/radiation-log.md` → "SPECULAR REFLECTION: THE REFLECTANCE SPLIT AND THE MIRRORS (#537 PR
1, 2026-10-05)" and its PR 2 / 3a / 3b / 4 subsections).** Planar mirrors, one bounce; **a mirror is one
plane, not one body**; curved specular bodies (a sleeve's exterior, a round vessel) moved to #604.
`Surfaces.reflectance` is gone: `diffuse_reflectance` and `specular_reflectance` (default 0), both live
leaves, each in `[0, 1]` and summing to at most one (there is no `reflectance=` keyword).
`vectors.reflect` is the one reflection formula; `Profile.mirrored(normal)` is abstract (a
`PhotometricProfile` reflects `up` AND flips `handedness`); `Mirror.image` reflects the vertices and
swaps two corners; `planar_mirrors` groups a body's areal facets by plane, seeded largest facet first.
`images.py` gathers each source's image through each mirror's aperture facets (`silhouette.covered_by`,
the aperture as blocker; the medium crossed on the two real legs). `build_radiation_model(...,
specular=(names,))` stores the exchange per `(solid, crossing pattern)` term (`specular_terms`), `rho_s`
live per declared solid, `prod tau_b ** legs_b` applied at `assemble`; a graded medium is refused with
specular bodies, and a traced `rho_s` on a model with no specular body is refused (an `eqx.error_if` let
`grad` return 0.0). Reflected paths are shadowed on both legs (`mirror_visibility.py`) through a meeting
point LIFTED off the mirror; under `SilhouetteOcclusion` they are rays. Cost is set by the aperture's
triangle count, not the plane count: the clip is cone-culled (`images._screen`, survivors clipped
`_CLIP_BATCH` at a time in a `while_loop`, so the culled clip takes its geometry under `stop_gradient`),
laid out by `lit_blocks`, and an image wholly inside the mirror skips the clip (only with one concrete
reflectance on every facet). `MAX_MIRROR_PLANES = 12` per specular solid. ⚠️ `inward_box`'s right half is
a translate of its left, not a mirror image: build a symmetry fixture by reflection.

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

It recurred twice more (#522: `Visibility.for_receivers` and `with_optics(profiles=)` made every model
entry point with a mask untraceable); `profile_index` is now stored as numpy everywhere. Record:
`.claude/notes/radiation-log.md` → "⚠️ INSIDE A TRACE, `jnp` STAGES EVERYTHING — EVEN ON CONCRETE
INPUTS".

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

**Current status.** A gradient is bounded by the chunk too: `gather.streamed_fluence_rate` makes each
chunk an `eqx.filter_custom_vjp` that saves only its inputs and rebuilds its mask and gather in the
backward pass (the mask builds run eagerly under `jax.grad`, so this path is not compilable with `jit`).
Masks come from the BUILD-TIME geometry (`StreamedShadows.geometry`), and a receiver inside a body is
refused at build (`visibility.refuse_points_inside`). Mechanism, tests and the whole-Sozzi-field
measurement through the streamed public model: `.claude/notes/radiation-log.md` → "A MODEL CAN STREAM ITS
RECEIVER MASK — explicit, frozen by default (#489, 2026-09-24)". ⚠️ On an allocator that keeps freed
pages, attribute a peak only by sampling between phases in one process.

## THE TRANSFER BUILD IS FILLED IN PLACE, NOT ASSEMBLED (#521, 2026-09-24)

**Current status.** `build_transfer` fills three `n x n` buffers in place: `_row_block` computes
`chunk_size` receiving facets against the sending facets `_columns_in_front` keeps (a Numba per-block
test: a facet is dropped when no vertex lies strictly in front of any quadrature point -- every point,
never the centroid -- or when `_projected_bound` is under `_NEGLIGIBLE = 2**-52` sr), and `_written`
row-scatters each block into donated buffers; the last block repeats its last receiver rather than
padding, and inputs are `stop_gradient`ed at the start. Contract: a skipped pair's value is zero or under
2^-52. The peak-memory decomposition, the skip rule's detail, measurements and tests:
`.claude/notes/radiation-log.md` → "THE TRANSFER BUILD IS FILLED IN PLACE, NOT ASSEMBLED (#521,
2026-09-24)". ⚠️ On this backend freed memory is not handed back and JAX dispatches asynchronously:
attribute a build's peak by sampling between steps, each forced to complete.

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

Both directions are load-bearing: a primitive wants compiling (evaluated eagerly, each inequality
materializes a receiver-by-facet intermediate), and a triangulated body cannot be traced at all (a host
grid walk; handed a tracer it raises). ⚠️ Compiling the mask build unconditionally breaks a scene holding
both kinds (`test_a_host_side_blocker_and_a_primitive_stand_in_one_scene`); `RayCastOcclusion(grid=...)`
is a `SelfOcclusion`, not on the occluder list, and is unaffected. Record:
`.claude/notes/radiation-log.md` → "⚠️ AN OCCLUDER DECLARES WHETHER IT CAN BE TRACED, AND COMPILING ONE
THAT CANNOT IS A CRASH".

## MEASURED: the Sozzi reactor as three cylinders reproduces the hand-derived occluder EXACTLY (2026-09-23)

**Current status.** `Outside(chamber, inlet, riser)` of three `Cylinder`s masks 0 of 180M pairs
differently from the hand-derived `BranchOpenings` and gives a bit-identical field; the general
construction costs ~1.39x the bespoke one (three in-process passes). 81.1 % of the cell-centre pairs lie
in one convex region. Configuration, the 640 mm pipe truncation it exposed, the grid comparison and the
measurement traps (a 2x2's closure identity is vacuous; never pair numbers across runs; decompose an
agreement before reading it as a check): `.claude/notes/radiation-log.md` → "MEASURED: the Sozzi reactor
as three cylinders reproduces the hand-derived occluder EXACTLY (2026-09-23)". ⚠️ Check a region's extent
against the mesh's, not against a comment.

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

Why nothing caught it (every transfer fixture switched the mask off) and the convex-emitter consistency
check: `.claude/notes/radiation-log.md` → "⚠️ A RAY BETWEEN TWO FACETS NEEDS **TWO** EXCLUSIONS, AND THE
MISSING ONE SHIPPED". ⚠️ `row_sum_error` is blind to the visibility mask by construction (the mask is
applied in `TransferMatrix.assemble`): a green row sum is no evidence about the mask.

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


## ⚠️ THE MEMORY OF ONE PASS WAS THE WHOLE PERFORMANCE STORY, BECAUSE THE PASS RAN EAGERLY

**Current status.** The per-block kernel `_block_is_cut` is traced, so the rays-by-triangles intermediate
is fused away (6.6-8.3x, bit-identical, #462); history and tables: `.claude/notes/radiation-log.md` → "⚠️
THE MEMORY OF ONE PASS WAS THE WHOLE PERFORMANCE STORY, BECAUSE THE PASS RAN EAGERLY".

**`work_limit` bounds rays x triangles per compiled call, and the call takes TRIANGLES FIRST**
(`_call_shape` in `triangles.py`): as many triangles as the bound allows — the whole set, for any
mesh this pass can afford — and as many rays as then fit. The triangle block is the reuse factor
(each ray's origin, direction, margin and exclusions are streamed once per call and tested against
every triangle in it), so the order is not cosmetic.

Rays first was the shipped call order and was the whole of an apparent "throughput falls with the mesh"
finding; with triangles first, a `work_limit` above 4M makes whole builds slower and one below it is flat
(#526). Chunks shorter than the full one are padded to a power of two. ⚠️ A rule tuned under one
execution model is a hypothesis under the next. Record in the log, same heading.

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

The leak sweep (`validation/radiation_watertight_sweep.py`: 8,764 rays, 0 leaks eager and traced; 124
with the plain difference, all compiled; the shear and the contour form's cross product need no exact
cancellation) and three traps met measuring it (aim points snapped to a tolerance; `cylinder_triangles`
does not close -- build closed bodies from one vertex table; swapping a module global does not invalidate
a compiled version): `.claude/notes/radiation-log.md` → "Watertight intersection, and one piece of the
published algorithm deliberately dropped".

⚠️ **THE DISCRIMINATOR IS WHETHER THE CANCELLATION FEEDS A DISCRETE PREDICATE.** The contour form
is built on the same difference of products, but a last-bit change there moves an *angle* by a
last bit. The inside test feeds it to a sign test that decides which of two triangles claims a
ray, and a discrete predicate has no small errors — it is right or it is a pinhole. Apply this
test before assuming the next exact-arithmetic site here is safe or unsafe.


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
  **8.8% / 1.5% / 0.38% / 0.18%** at 1 / 3 / 6 / 12 points — **on cubes**. On `inward_box`
  stretched to 1 x 1 x 3 the same figure is **35-38% / 14-15% / 4.6-4.8% / 0.99%** (48 and 192
  facets, `NoOcclusion`; jax 0.10.2, CPU, x64, Linux x86_64, 4 cores, 2026-10-16, #631), about
  ten times larger, and `build_transfer`'s docstring states both.
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

A finer rule costs far less than its point count (six points 1.15-1.72x the one-point build):
`.claude/notes/radiation-log.md` → "The receiver quadrature: why ONE side of the double integral is exact
and the other is not".

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
The ray test is two-sided and never had the problem. The sheets are declared on the surface set,
`Surfaces.from_triangles(..., two_sided=(names,))` (since #689; there is no
`SilhouetteOcclusion.two_sided` any more), and the clip counts their facets from both sides
(`surfaces.is_sheet & (surfaces.twin < 0)`: on a set whose back faces are appended each face counts
from its own side); a misspelt name **raises** at `from_triangles` (it would otherwise leak light with
no error); and any **open piece left undeclared is warned about** at build, naming its bodies. With the blocker declared, the one-sided Walton pair converges
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

**Current status.** On the Sozzi reactor (1,635,909 cells) aquaflux's whole-reactor volume-mean `G`
agrees with of-optical-radiation's DOM-256 to 0.09 %, DOM converges on aquaflux as its directions
increase (the spread is DOM's ray effect), and aquaflux's own error near the lamp is median 0.44 % / p99
4.1 %. Instrument `validation/sozzi_radiation/`; numbers, the non-uniform DOM grids and why the harness
needed analytic visibility and chunking: `.claude/notes/radiation-log.md` → "MEASURED: aquaflux against
discrete ordinates on the Sozzi reactor (2026-09-22)". `docs/radiation.md` restates these figures
(Documentation, below).

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

⚠️ **The surface set passed at call time must be the build's geometry, and every call checks**
(`RadiationModel.geometry`, a SHA-256 of the vertices plus the point-source labels, checked in
`radiosity`; exact, not toleranced -- `with_optics` carries the build's own vertex array). **Traced
vertices are refused at every call** (`TypeError`, since 2026-09-29): the facet-to-facet transfer is
frozen too, so a traced lamp reached the direct gather only and its derivative was wrong (-3.54 against
+4.08 W/m² per m by a finite difference over rebuilt models). Position derivatives are
`direct_fluence_rate` / `direct_irradiance`'s; a model may still be BUILT from traced vertices (its
fingerprint is then `None`). Record and tests: `.claude/notes/radiation-log.md` → "The public surface:
`model.py`, and the four assembly steps that are easy to omit".

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

Two mutations of `model.py` / `gather.py` survive and were dismissed (the reflected set's zero power; the
chunk remainder, which needs a receiver count the chunk does not divide -- 37 in
`test_chunking_changes_nothing_about_the_answer`): `.claude/notes/radiation-log.md` → "`G = 4B` — the
closed form that pins the whole assembly at once".

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

**Current status.** One `jax.jit` program per pass (`_summed_signed_solid_angle`; passes cut by
`pair_limit`, a short last pass padded to a power of two), and a piece that is a 2-cycle
(`_closed_pieces`: its own triangles walk every edge as often one way as the other -- deliberately
stricter than `open_facets`' "closed") is summed only at points inside its own bounding box; `tolerance=`
on `enclosure_winding` / `check_points_outside` sets the vertex merge. ⚠️ A sheet welded into a closed
body is summed everywhere: its halves read closed to `open_facets` but bound nothing alone. Design,
measurement (98x the eager loop) and mutation pass: `.claude/notes/radiation-log.md` → "THE WINDING CHECK
IS COMPILED, AND A CLOSED PIECE IS SKIPPED OUTSIDE ITS BOX (#530, 2026-09-26)".

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

**Current status.** The one-ray-per-pair mask's error is measured
(`validation/radiation_partial_occlusion.py`, two plates and a rod): the mean transfer error falls
~20-fold from 2x2 to 12x12 quads per plate, a thin body is ~2x worse than a fat one, and as a fraction
of the shadow being modelled the worst field error is 58 % / 36 % / 12.5 % at 2x2 / 4x4 / 8x8. ⚠️ A
per-pair maximum of a discontinuous quantity has no determinable reference -- check that yours has one
before quoting a maximum. ⚠️ Size a fix against the error as a share of the shadow, not of the field.
Tables and traps: `.claude/notes/radiation-log.md` → "MEASURED: what the binary per-pair occlusion mask
costs (issue #447 item 2)".

## MEASURED AND REJECTED: two candidate fixes for the occlusion mask, and what standard practice does

**Current status: REFUTED, do not rebuild.** Receiver quadrature of visibility (6 rays) and source
pixelation (4 or 16 rays) both cap at 2-4x on the mean for 4-16x the rays and neither dominates, because
an occluder's silhouette cuts diagonally across the (receiver x source) product. What other codes do
(hemicube, discrete-ordinates pixelation, discontinuity meshing, ray seeding, the ultraviolet-reactor
summation models), the table and the reasoning: `.claude/notes/radiation-log.md` → "MEASURED AND
REJECTED: two candidate fixes for the occlusion mask, and what standard practice does". The built answer
is `SilhouetteOcclusion` (ANALYTIC OCCLUSION, below).

## MEASURED AND DECIDED: the two OTHER one-point factors (issue #447 items 1 and 3)

**Current status.** Absorption at the centroid separation is FIRST order in `a * w` (bias `-a (<r> -
r_centroid)`, worst between neighbours at ~`0.15 a w`); a non-Lambertian profile is SECOND order in the
angular width; neither bias has a fixed sign, so neither is a correction. `build_transfer`'s docstring
carries the rule of thumb. Instrument `validation/radiation_one_point_factors.py`, tests
`tests/unit/test_radiation_one_point_factors.py`; tables, the corrected first-order argument, and the
trap that a ratio-shaped control is blind to its own weighting: `.claude/notes/radiation-log.md` →
"MEASURED AND DECIDED: the two OTHER one-point factors (issue #447 items 1 and 3)". **Decided (#447 item
1): documented as a limitation, not built** -- the profile bias is exactly zero for Lambertian radiance
(all reflected light), second order where it is not, of no fixed sign, and smaller than the occlusion
problem; a fix would freeze two moments of `log cos` per pair with the exponent left live.

## MEASURED: what the mask build costs in SECONDS — and why its throughput USED TO fall with the mesh

**Current status.** The ungridded ray mask's cost is `n^3` on the transfer build's own case and runs at
~420-465 Mtest/s, flat in the ray count since the triangles-first call shape (3,184 facets: 77 s). Ladder
and the cause found: `.claude/notes/radiation-log.md` → "MEASURED: what the mask build costs in SECONDS —
and why its throughput USED TO fall with the mesh". Without a grid the pass has no early exit, so its
cost is independent of what the rays hit and one ladder settles every scene; `RayCastOcclusion(grid=...)`
breaks this by design.

## A MASK STORES ITS ANSWER AT THE NARROWEST TYPE, AND A PASS HOLDS INDICES (#525, 2026-09-25)

**Current status.** The surface's own layer is stored at what each strategy can say: `NoOcclusion` `None`
/ `None`, `RayCastOcclusion` bool / `None`, `SilhouetteOcclusion` float64 / bool (`hidden_by_geometry` /
`overlapping`); `surviving_fraction` widens per chunk. ⚠️ Code reading `hidden_by_geometry` must allow
`None` and a bool. The brute-force pass takes a pass's pairs as indices (`triangles.pairs_are_cut`) and
forms its rays eagerly a chunk at a time (`_cut_in_chunks`, shared with `segment_is_cut`) -- eagerly on
purpose, so a margin's last bit cannot move. Detail, measurement and mutation pass:
`.claude/notes/radiation-log.md` → "A MASK STORES ITS ANSWER AT THE NARROWEST TYPE, AND A PASS HOLDS
INDICES (#525, 2026-09-25)".

## THE RECEIVER RAY MASK CASTS ONLY FACING PAIRS, AND THE GRID TAKES INDICES (#526, 2026-09-25)

**Current status.** For volume receivers (`receiver_facet is None`) `RayCastOcclusion` does not cast a
pair whose receiver lies certainly behind a `dark_behind` facet (`Profile.dark_behind`,
`Surfaces.dark_behind`, `BackFaces.behind`; an undecidable height is kept and cast); such pairs are
recorded clear, the mask says so (`Visibility.clear_behind`), and a gather through it refuses a set whose
areal profiles are not all dark behind (`_refuse_light_from_behind`). Receivers on facets are always cast
in full. `SelfOcclusion.prepared(surfaces)` does surface-only work once per stream
(`RayCastOcclusion.prepared` builds the grid; `grid=` accepts a built `TriangleGrid`), and
`streamed_fluence_rate` refuses points inside once over all points, building each pass through
`visibility._unchecked_visibility`. ⚠️ A test counting mask builds patches
`gather._unchecked_visibility`. Detail, measurement and tests: `.claude/notes/radiation-log.md` → "THE
RECEIVER RAY MASK CASTS ONLY FACING PAIRS, AND THE GRID TAKES INDICES (#526, 2026-09-25)".

## ONE MORTON ORDERING, PER-AXIS CELLS — cubic cells measured and REJECTED (#574, 2026-09-28)

`aquaflux/morton.py::morton_order` is the only ordering of points in the package: shaft culling's
receiver groups and lamp-facet clusters (`_Curve`), the gather's chunks and `lit_blocks`, the transfer
build's row blocks, and `FacetClusters`. It is the former `culling.spatial_order` moved to a neutral leaf,
**bit for bit** (10 bits, each axis scaled to its own extent — checked equal on four point sets), and
`FacetClusters`' private 21-bit copy (`clusters._morton_keys`) is gone: its ordering moved from 21 to 10
bits, which only reorders centroids sharing a 1/1024 cell, and nothing measured it.

Cubic cells (#574) were built, measured at mesh scale (undecided pairs +5 %, transfer row blocks 51.8 →
57.9 s) and reverted: cubic groups are tighter but wrap round a long lamp, so their facets stop facing
one way. **Do not retry cubic cells as a fix for elongated lamps**; grouping by facing direction is #589.
Record: `.claude/notes/radiation-log.md` → "ONE MORTON ORDERING, PER-AXIS CELLS — cubic cells measured
and REJECTED (#574, 2026-09-28)".

## SHAFT CULLING: BUILT as `ShaftCulling` — tiles certified clear, THE DEFAULT since 2026-09-26 (#554)

**Current status.** `culling.py`: `BodyCulling.blocked(bodies, sources, near, receivers, pair_limit, *,
facing=None)`, with `EveryPair` (the reference) and **`ShaftCulling(receiver_blocks=(32, 8, 2),
source_clusters=(32, 8, 2))`, the default** -- an unset `body_culling` is `ShaftCulling()` through
`culling_or_default`, for a built mask, a streamed chunk and both of a model's masks. Under a trace
`ShaftCulling` hands off to `EveryPair`. `BodyCulling.prepared(bodies, sources)` forms the facets' side
once per call (`_Groups`, which refuses other sources or bodies by value). A traceable body is asked on
batches padded to a power of two (`_vouches`, `Body.vouches_tiles`); a host-answered (not `traceable`)
body is handed no padding and no pair behind its source, and `TriangleBody`'s certificate is one Numba
loop (`grid._unions_held`). Contract detail, tests, mutation passes and every measurement (9.3x on a
sampled scene; 1.50x on the whole Sozzi mesh, where it became the default on 2026-09-26; where the call's
time goes; phase C's ceiling): `.claude/notes/radiation-log.md` → "SHAFT CULLING: BUILT as `ShaftCulling`
— tiles certified clear, THE DEFAULT since 2026-09-26 (#554)". **How:** receivers and facet centroids are
ordered along the Morton curve and grouped at each ladder size (each size divides the one before), each
group summarized by the column-wise max of `body.clearance`; a tile is certified clear where
`body.vouches(...)` says so, a refused tile is split at the next level (`_undecided`), and the finest
refused tiles are answered by the same `body.blocks` the reference uses, so the two agree bit for bit.
Only "clear" certificates are built; "fully hidden" certificates (phase C) and level of detail (phase D)
are not (#554). ⚠️ The facet-to-facet ray mask (`RayCastOcclusion`) is NOT culled: its shafts start on
the wall they are tested against.

### PHASE B (#554, 2026-09-26): refinement, and triangle bodies vouching through an occupancy grid

**Current status.** `TriangleBody` (`triangle_body.py`, #510; in radiation beside the grid because
`solids/` imports nothing outside itself): `build(vertices, *, sheet=None, resolution=None,
clearance_resolution=None, tolerance=None)`; `blocks` is the grid walk (`traceable=False`); `contains` is
decided per connected piece -- an open piece has no inside, a closed piece is solid on the side its
normals point away from (solid where `enclosure_winding(closed pieces) + n_inward > 0.5`); `sheet=True`
forces no inside, `sheet=False` refuses an open piece, and a closed piece enclosing no volume is refused.
It vouches through its own occupancy grid (`TriangleBody.occupancy`: `_CLEARANCE_REFINEMENT = 4` times a
near-cubic grid, read through `holds_any`) -- sized from the near-cubic grid, NOT the walk grid, so the
walk's box-shaped default does not move certification. The default ladder (32, 8, 2) is chosen on
absolute time: its third level saves ~2.7x on a triangle walk and costs ~1.24x on an analytic-only mask
(~62 s of 314 s on the whole Sozzi field). A per-body depth was investigated and deliberately **not
built** (user, 2026-09-26); a scene of analytic bodies alone may pass `ShaftCulling(receiver_blocks=(32,
8), source_clusters=(32, 8))`. Two fixes for the finest level's cost were refuted -- testing a flat pair
list, and the tile index innermost -- do not retry them without a new reason. Decisions, measurements,
tests and mutation passes: `.claude/notes/radiation-log.md` → "PHASE B (#554, 2026-09-26): refinement,
and triangle bodies vouching through an occupancy grid".

## BACK FACES: THE BODIES' LAYER SKIPS PAIRS BEHIND THEIR SOURCE (2026-09-27, from #565)

**Current status.** Exact, no tolerance (chosen over #565's cluster approximation):
`BackFaces.of(surfaces)` is passed as `facing=` to `BodyCulling.blocked` when `receiver_facet is None and
surfaces.dark_behind`; both strategies record a pair behind its source clear and set
`Visibility.clear_behind`, and `ShaftCulling` drops the tiles `BackFaces.tiles_behind` proves wholly
behind at every level. The box test is conservative by construction (`_BOX_SLACK` = 3 x `clipping._SLACK`
units of roundoff) and is one Numba loop (`_boxes_behind`). Design, measurement and tests:
`.claude/notes/radiation-log.md` → "BACK FACES: THE BODIES' LAYER SKIPS PAIRS BEHIND THEIR SOURCE
(2026-09-27, from #565)".

## BACK FACES IN THE GATHER: RECEIVERS IN BLOCKS, EACH AGAINST THE FACETS THAT CAN LIGHT IT (2026-09-27)

**Current status.** Exact (the field moves by summation order only). `lit_blocks.lit_segments` orders the
points along the Morton curve in blocks of `BLOCK = 8`, lists per block the facets `BackFaces.lit_facets`
does not prove it lies behind, and cuts the sorted blocks into segments padded to `rounded_width`;
without planes, or with traced points, a block is listed against every facet as one shared row
(`LitSegment.shared`). `summed_fluence_rate(..., layout=None)` is `_point_fluence` plus
`_segment_fluence` per segment of each `_ArealGroup`, a group laid out by its planes only when every
set's profile there is `dark_behind`. The streamed path's programs (`_point_part`, `_segment_part`) are
module-level `jit`s cached across calls by `_Labels`, the content-hashed non-floating part of the live
values. ⚠️ Three traps that each read as "the skip saves nothing": `jnp.asarray` inside a trace makes the
points a tracer (lay out from the points as given); per-call recompilation; `lax.dynamic_slice` and
`jnp.take`'s default `fill` mode (`work._slice` is a clipped `take`). Record, measurements and tests:
`.claude/notes/radiation-log.md` → "BACK FACES IN THE GATHER: RECEIVERS IN BLOCKS, EACH AGAINST THE
FACETS THAT CAN LIGHT IT (2026-09-27)".

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

**Current status.** #577 was filed on the wrong mechanism: on a 4-core Linux box padding and call count
were worth ~10 % and the step size ~1.5x; on the 11-core Mac the bound barely matters (2.50-2.59 s from
32k to 4M) and 2^16 stands. The transfer build does NOT take this bound (a cap was built and withdrawn;
re-measure on the current build before capping it). ⚠️ A patched bound in one process measures nothing
after its first arm -- `_segment_part` is cached under `_Labels` and `pair_limit`, which do not carry it;
run each bound in its own process. Tables: `.claude/notes/radiation-log.md` → "A TRACED STEP IS BOUNDED
BY CACHE, NOT ONLY BY MEMORY: `work.PASS_PAIRS` (#577, 2026-09-27)".

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

The probe that sized the grid on the Sozzi wall (~52 triangles tested per ray at 128³ against 53,500; the
specification's ~20 was optimistic): `.claude/notes/radiation-log.md` → "GRID ACCELERATION: BUILT as
`TriangleGrid` — a compiled host walk, off by default".

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

The exactness tests (bit equality with `segment_is_cut` at several resolutions, a closed drum that must
not leak) and their mutation pass are recorded in the log, same heading.

**Current status.** Three defects survived the grid's first, single-shaped test suite (a margin read as a
share rather than a length; an unbounded candidate expansion; voxels sized from the box's volume rather
than the triangles' area). On the real reactor (`validation/sozzi_radiation/grid_mask_check.py`) the
triangulated wall's mask differs from the analytic cylinders only in the inscribed-polygon sliver at the
pipe rims, chamber cells identically; the whole Sozzi field with the triangulated wall costs ~10-12x the
cylinders'. Primitives remain the path a real reactor should take; the grid is the fallback for geometry
that exists only as triangles (#501). Record in the log, same heading. ⚠️ A suite whose fixtures are one
shape of input repeated cannot tell a correct implementation from broken ones.

### THE COMPILED WALK: one Numba loop per ray, the only walk (2026-09-26)


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


Resolution re-asked under the compiled walk, the remaining per-ray cost, and the walk's tests and
mutation pass (the averaged edge function stays although Numba did not fuse the products):
`.claude/notes/radiation-log.md` → "THE COMPILED WALK: one Numba loop per ray, the only walk
(2026-09-26)".

**`NUMBA_NUM_THREADS=1`** is now set beside `OMP_NUM_THREADS` in `tools/fastgate.sh`'s parallel tier and
in CI's environment, so xdist workers do not each start a pool the size of the machine.

### THE TRIANGULATED WALL AT MESH SCALE, MEASURED (2026-09-26) — 62 min, not 88 h

**Current status.** The whole Sozzi field with `bodyWall.stl` as a `TriangleBody` takes 2,914.3 s on the
box-shaped default grid (3,705-3,721 s on the old near-cubic one) against ~314 s with the analytic
cylinders; chamber cells agree to 4.4e-16 and pipe cells differ by the rim sliver. Only ~25 % of pairs
are certified on a round triangulated wall (81 % for the cylinders), so the lever is a certificate
covering more of the chamber (#568) or a cheaper walk. ⚠️ Never multiply a pipe-heavy rate by the mesh;
quote the whole-field figure. Record: `.claude/notes/radiation-log.md` → "THE TRIANGULATED WALL AT MESH
SCALE, MEASURED (2026-09-26) — 62 min, not 88 h".

### THE WALK'S COST: TRIANGLES TESTED IN NARROW PASSAGES, AND THE VOXELS' SHAPE (#503, 2026-09-27)

**Current status.** Measured for #503 (`grid_walk_direction.py`): walking from the nearer end is worth
nothing (0.95-1.06x); empty space is not the cost (a segment down a pipe tests the wall's triangles in
every voxel it crosses); refining near-cubic voxels is flat -- but the voxels' SHAPE moves it, which set
the default below. Empty-space skipping is not built. Tables: `.claude/notes/radiation-log.md` → "THE
WALK'S COST: TRIANGLES TESTED IN NARROW PASSAGES, AND THE VOXELS' SHAPE (#503, 2026-09-27)".

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
The default against the old rule across scenes (1.33x and 1.79x) and at mesh scale (1.27x on the whole
Sozzi field), and the tests and mutation pass, are in the log, same heading.

**The occupancy grid did NOT follow** (`TriangleBody.occupancy`, above): it stays 4x a near-cubic grid,
so certification on the Sozzi wall is unchanged and the mesh-scale run below isolates the walk.


### A FLAT SURFACE AND A SHEET ON A VOXEL PLANE: three misses fixed (2026-10-05)

**Current status.** Three misses fixed (2026-10-05) for a flat triangle set or a sheet on a voxel plane:
a flat axis is `_FLAT_THICKNESS = 1e-6` of the widest extent thick (`_box_and_area`); registration widens
each triangle's box on both sides by `_rounding_margin`, the one helper for the padding, the registration
and the box test; `_enters_grid` refuses a segment with a zero direction component outside the box's slab
on that axis. ⚠️ A segment ending exactly in a sheet, or crossing its open rim, is a knife edge where the
compiled `segment_is_cut` and the Numba `_cuts` round differently: compare the grid against a one-voxel
grid to separate voxel selection from kernel rounding. ⚠️ A plane at a large coordinate hides the flat
defect. Record and tests: `.claude/notes/radiation-log.md` → "A FLAT SURFACE AND A SHEET ON A VOXEL
PLANE: three misses fixed (2026-10-05)".

## HOW MANY FACETS AN EMITTER NEEDS — measured, because it sets the price of everything

**Current status.** The emitter's facet count sets the cost. Against an analytic lamp refined to 270,336
facets the error falls about in proportion to the facet count, and the near-lamp band (< 5 mm) sets the
requirement; the case's `lampWall.stl` (7,516 facets) is 1.95 % median near the lamp, worse than its
count suggests (irregular triangles, area 0.5 % under the cylinder). The lamp read from the CAD drawing
(`CadModel.triangles`, `chord=1e-4`, `facet_size=2.5e-3`: 66,011 facets, 0.16 % near the lamp) is the
cheap way to accuracy; the mesh's own patch, raw, is the expensive one, so it is coarsened (next
section). Table: `.claude/notes/radiation-log.md` → "HOW MANY FACETS AN EMITTER NEEDS — measured, because
it sets the price of everything".

## THE MESH'S OWN PATCH AS THE EMITTER: exact, then coarsened (#492, 2026-09-25)

`aquaflux.mesh.patch_triangles(mesh, geometry, names)` gives the exact triangles (its record is in
`mesh.md`); `coarsen.py` makes them affordable. Agreed design, three decisions taken with the user
before building: **collapse only** (every surviving vertex is an input vertex, so no projection onto
the input surface is needed or made), **each body's power held** (exitance scaled by area before
over after, in `coarsen_surfaces`), and **both steps in one change as separate classes**.

**Current status.** `coarsen.py` bounds: `max_edge` (a strict longest-edge bound -- not
`CadModel.triangles`' `facet_size`, a different quantity; do not unify the keywords), `chord` (judged per
input vertex against the rewritten patch AND its unchanged ring), `angle` (default 0.5 rad; also the
crease threshold). Feature lines (rims, body interfaces, non-manifold edges, creases) slide only along
themselves and are pinned where they turn by more than `angle`. Collapses are batched (`_local_minima`
within two edges, a banded-random `_priority`, refusals lasting a round, screening, a compiled pair
distance), ~0.1 ms per input triangle. On Sozzi, 4 mm / 1e-4 m → 18,432 facets, 0.52 % near the lamp:
size along the lamp, keep the chord at 1e-4. Design traps, dead ends and dismissed mutations (the link
condition; a one-edge independence radius): `.claude/notes/radiation-log.md` → "THE MESH'S OWN PATCH AS
THE EMITTER: exact, then coarsened (#492, 2026-09-25)".

## PER-CALL COMPILES: THE REMAINDER CHUNK AND THE RADIOSITY SOLVE (2026-09-26)

**Current status.** `work.in_passes` runs a short last chunk as a scan of one step, not as a bare
(eagerly evaluated) call; the interreflection solve is `model._interreflection`, an `eqx.filter_jit`
function that builds the operator inside itself, so it is traced once per matrix size and solver
settings. ⚠️ A plain held-mask gather still compiles per call (only the streamed path is cached across
calls). ⚠️ `build_radiation_model` returns before its asynchronous work finishes: wait on the model
before timing what follows. Measurement and tests: `.claude/notes/radiation-log.md` → "PER-CALL COMPILES:
THE REMAINDER CHUNK AND THE RADIOSITY SOLVE (2026-09-26)".

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

**Current status.** A receiver count left a pass's size to the facet count (a ~9 GB pass killed a run
silently); a whole-problem `ones((n_receivers, n_facets))` formed before chunking was removed in the same
change (pinned by `test_a_scene_with_nothing_in_the_way_forms_no_array_the_size_of_the_problem`). The 4M
default is measured (`validation/radiation_gather_pair_limit.py`). A gradient is bounded too:
`work.in_passes` wraps its scan body in `jax.checkpoint(prevent_cse=False)`, flat in the receiver count
(#523). #524: `summed_fluence_rate(sets, ...)` forms the geometric pass once from the first set and
weights it per set (sets with different concrete vertices are refused); the surviving fraction is formed
per chunk from the mask's own layers (`_shadow_rows`, sliced inside the checkpoint);
`direct_irradiance(..., point_sources_only=True)` visits no areal facet; `surface_irradiance` reuses
`radiosity`'s `_solve`. ⚠️ A translation test on a uniformly emitting closed box reads zero to rounding:
emit unevenly. Record and tables: `.claude/notes/radiation-log.md` → "A PASS IS BOUNDED IN
RECEIVER-BY-FACET PAIRS, NOT RECEIVERS (#509, 2026-09-24)".

## THE VOXEL WALK CARRIES ITS TOTAL, AND THE FACET-TO-FACET WALK GOES IN PASSES (#528, 2026-09-25)

**Current status.** `VoxelAbsorption.optical_depth` carries its running total and the near end's field in
the scan's carry (two lookups a step, nothing stacked) and checkpoints the step, so its working memory is
~100 B a pair forward and ~3.5 kB under a gradient at 32³; the value is not bit-identical (summation
order). Under a non-uniform `Absorption`, `TransferMatrix.assemble` walks the facet pairs through
`work.in_passes` (moved from `gather.py`; there is no `gather._chunked` or `gather._slice`), bounded by a
`pair_limit` keyword that `_solve` fills from `gather_pair_limit`. Measurement and tests:
`.claude/notes/radiation-log.md` → "THE VOXEL WALK CARRIES ITS TOTAL, AND THE FACET-TO-FACET WALK GOES IN
PASSES (#528, 2026-09-25)".

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

- **Current status.** `build_visibility(..., receiver_normal=(n, 3))` (2026-10-06) says which way a
  receiver faces, independent of where it lies (`receiver_facet`); it reaches every
  `SelfOcclusion.field(surfaces, points, near, receiver_facet, receiver_normal)` (only the silhouette
  reads it), is normalized and refused on a wrong shape or a zero / non-finite length, and takes
  precedence over the facet's normal. With a normal the silhouette accepts `(n, k)` facet rows; without
  one, `k > 1` is refused. `clear_behind` / `BackFaces` still key on `receiver_facet is None`,
  deliberately. A volume point pads the source with a repeated corner instead of clipping, and its
  unprojected integral is `solid_angle._signed_loop_area`. Contract, tests, and the volume-mask
  measurement (the clip ~30x better than the ray mask at the worst edge cell, at 15-30x its cost, all
  before #527): `.claude/notes/radiation-log.md` → "VOLUME RECEIVERS: the share of the PLAIN solid angle
  (#479, 2026-09-25)".

**Where it over-counts:** only where two front-facing silhouettes overlap in angle -- each blocker is
clipped against the source, not against what is still unblocked, so areas add (a multi-lamp bundle with
one sleeve behind another; a serpentine channel); it errs dark. One sleeve, wall or declared baffle
between two facets, a bent duct, a convex vessel and disjoint cones are exact. `overlapping` means only
"more than one blocker contributed". The exact repair, depth-sorted clipping against the remaining
region, has no static vertex count. **Current status (decided: documented, not fixed; the precise
detector and the correction are #477).** On a two-lamp reactor the over-count is rare (1.0-1.5 % of
flagged pairs), one-sided (dark) and on the field ~8x smaller than the ray mask's own error;
`overlapping` flags nearly every hidden pair on a meshed body, so it says almost nothing as a detector.
The six degenerate-geometry defects that "it is exact" hid on a mesh are each guarded and pinned: the
depth cut (the blocker cut to the near side of the source's plane), the cull's source-plane offset
(`decidable_heights(..., through=)`), the extent clamp, `angular_cone`'s `unusable` (an unusable cone
overlaps everything), the `in_front` / `edge_on` guards, and `_EXTENT_FLOOR` (1e-12 sr) for a source seen
edge-on. ⚠️ A binomial error bar is zero at a share of exactly 0 or 1: use Agresti-Coull. ⚠️ A fix that
makes an intermediate right can expose a downstream quantity that was only right by accident. Records:
`.claude/notes/radiation-log.md` → "MEASURED (#472): the over-count on a two-lamp reactor is rare and
small — and `overlapping` is nearly useless" and the "IT IS EXACT" subsection after it.

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

The measured truth table (364-facet sleeved box: only both filters together give 0 wrong pairs, and at 80
and 156 facets every filtered row was identical -- a fixture too small to contain the degeneracy says
nothing, and chunk invariance alone is the wrong acceptance test), the per-call-site mutation analysis,
why the ray test's antisymmetric edge function is not needed here, the move of `projected_solid_angle`
onto `clipping.py`, and that `validation/radiation_analytic_occlusion.py` still carries the PROTOTYPE
clip: `.claude/notes/radiation-log.md` → "⚠️ THE SIGN TESTS: TWO FILTERS, EACH LOAD-BEARING, AND THE
SMALL FIXTURE COULD NOT TELL".

**`dG/d(occluder geometry)` is still exactly zero** with either strategy: the fraction is computed
on the host and frozen, off the tape. A continuous fraction *could* make baffle and sleeve placement
differentiable, but only by putting the clip live on the tape — not built, and its cost would be the
whole build per gradient.

### Cost, MEASURED end to end — not projected

**Current status.** Measured on 11 cores before #527, the silhouette build cost ~11 min at 3,184 facets,
8.5x the ray mask; #527 and #555 have since cut it and it has not been re-measured there. ⚠️ A ratio
inherits every defect in its denominator: time the whole build, both arms, at the size the answer is for,
and re-time the ratio whenever either arm changes. Table and the history of wrong cost figures:
`.claude/notes/radiation-log.md` → "Cost, MEASURED end to end — not projected".

### THE CLIP SEES ONLY PAIRS THAT MIGHT COVER SOMETHING, PACKED ACROSS RECEIVERS (#527, 2026-09-25)

**Current status.** `silhouette.covers_nothing(view, to_source, to_blocker)` runs on the cone cull's
survivors and rejects a pair the clip would return zero for (the source subtends nothing; nothing of the
blocker strictly in front of the source's plane; the blocker edge-on; a face-plane separating axis) --
the first three are `covered_by`'s own zeroing predicates, factored out. The source view is computed once
per receiver; pairs are packed across receivers (`_PairPipeline`) into chunks of `work_chunk` = 32,768,
only a build's last chunk padded; reject and clip run on `threads` (default 4) Python threads and **the
thread count changes no bit**; the cull is bounded in pairs (`pair_limit`). Detail, measurement (8.5x at
1,532 facets, 4-core container) and tests: `.claude/notes/radiation-log.md` → "THE CLIP SEES ONLY PAIRS
THAT MIGHT COVER SOMETHING, PACKED ACROSS RECEIVERS (#527, 2026-09-25)".

### THE CULL RUNS ON CLUSTERS FIRST (#555, 2026-09-26) — worth 1.24x at 3,184 facets, and why no more (a second level measured slower)

**Current status.** `FacetClusters.build` (`clusters.py`) groups facets into runs of `cluster_size`
(default 32) along the Morton order, each with a bounding sphere; per receiver `_cluster_pairs` rejects
cluster pairs by enclosing cones (`silhouette.enclosing_cone`) and source planes, then `_member_pairs`
runs the member test on the survivors. The kept set is exactly the dense cull's (pairs not sorted). Worth
1.24x at 3,184 facets, because ~40 % of cluster pairs survive. Measured and rejected, so not retried:
other cluster sizes; compaction on the device; and a second cluster level (fewer member tests but more
compiled round trips, slower under four threads -- retrying depth needs the levels fused into one
compiled call per receiver). Record and tests: `.claude/notes/radiation-log.md` → "THE CULL RUNS ON
CLUSTERS FIRST (#555, 2026-09-26) — worth 1.24x at 3,184 facets, and why no more (a second level measured
slower)".


### MEASURED AND DECIDED (#474): merging coplanar triangles into convex patches is NOT built

**Current status: not built.** Merging coplanar triangles into convex patches would cut clip items
2.5-3.2x on a box or a faceted tube but edge-plane stages only 1.3-1.9x, and the clip is ~12 % of a
3,184-facet build after #527: a realistic saving of ~3-7 %. `overlapping` barely sharpens. Patches as
blockers in the cull and reject are not measured. Record: `.claude/notes/radiation-log.md` → "MEASURED
AND DECIDED (#474): merging coplanar triangles into convex patches is NOT built".


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
- **Current status.** A point on a reflecting surface IS lit by its own facet unless that facet is left
  out: off an axis-aligned plane the corner heights round either way, the clip keeps the in-plane
  triangle and the kernel returns the hemisphere. **Fix (2026-10-07): the receiver's own facets are
  excluded as SOURCES**, `direct_irradiance(..., receiver_facet=)` (`(n_points,)` or `(n_points, k)`
  rows, `-1` padding) -- the transfer's `F_ii = 0` in the gather; offsetting points in front was rejected
  (it couples a scene-scale constant to the clip's rounding tolerance). The ray test needs the facet too:
  `SurfaceReceivers.reflector` names the body and `_own_facets` passes **every** facet of it the point
  lies on (a face centre is the apex of all its fan triangles). Record and tests:
  `.claude/notes/radiation-log.md` → "THE SCENE: LAMPS KEPT OUT OF THE TRANSFER, AS A LIBRARY FUNCTION
  (2026-10-05)".
- **Measured at mesh scale** — see `validation/ray_effects_room/README.md` (the case files and their
  agreement with the scripts they replaced).
- ⚠️ **There is no `Scene.settings_for`, `RadiationSettings.for_bodies`/`check_bodies` or
  `SelfOcclusion.for_bodies`/`check_bodies` any more** (deleted with #689). They existed only to route a
  `SilhouetteOcclusion.two_sided` naming bodies of several surface sets to each set's builds; the
  declaration now lives on each `Surfaces` set, which every build carries with it, so nothing needs
  routing. `_joined` unions the two sets' declarations for the exchange and the direct set.
- **EVERY TRIANGLE OF THE SCENE SHADOWS THE DIRECT LIGHT TOO (#622, 2026-10-10).** Before, the
  lamps-only masks' self-occlusion was over the lamps' triangles alone and the reflectors entered only
  the exchange, so a reflecting wall cut the reflected light and let the lamps' direct light straight
  through (a 0.2 m Lambertian lamp, a 2 m plate rho 0.5 over it, a point over the plate: `G_direct`
  0.0125147 with the plate and without it, `RayCastOcclusion`, `b2c003e`). **Chosen with the user
  (option 1 of the issue) over documenting it and refusing an uncovered reflecting `Wall`.** The
  direct light is now gathered from `_Direct.of(scene).surfaces`: the reflectors' facets, dark
  (emission and power zero, a Lambertian profile appended to the lamps' catalogue), then **every** lamp
  facet in its own order with its optics, point sources kept (`_joined(scene, lamp_facets,
  emitting=True)`; the exchange is `_joined(..., emitting=False)` over the areal lamp facets -- one
  concatenation for both). ⚠️ **A shadow geometry separate from the sources is not expressible**: a
  mask's columns ARE the sources' facets (`streamed_fluence_rate` requires `sets` to share
  `shadow_geometry`; `build_visibility`'s self-occlusion is over `surfaces`), so the reflectors join
  the sources rather than lending their triangles to the lamps' mask. That is also what keeps the
  own-facet exclusion simple: `receiver_facet` indexes the sources, so `_own_facets(direct.surfaces,
  body, points)` finds a reflector body or a lamp body alike, and the exchange's samples name their
  facet by `_Direct.facets_of(exchange)` (reflector `j` -> `j`, lamp facet `k` -> `split + k`). The
  reflector samples now **need** it -- a ray from another facet ends in the sample's own facet -- and
  a part (reflectors / lamps) whose facets are all `-1` passes `own=None`, keeping the culls that key
  on `receiver_facet is None`. **The lamps alone** when there are no reflectors or self-occlusion is
  `NoOcclusion` (dark facets would add only zeros to every gather); with bodies but `NoOcclusion` too.
  ⚠️ Behaviour change: a `SurfaceReceivers` set on a reflector that does NOT name its body now has its
  DIRECT part shadowed by its own facet under the ray test, as its reflected part already was.
  Gate: `test_a_reflector_between_a_lamp_and_a_point_shadows_the_lamp_s_direct_light` (lamp, black
  plate, reflecting 3x3 ceiling, `RayCastOcclusion`, one sample: fluence, ceiling landing and ceiling
  points' irradiance equal the model with every facet in one set to 1e-11; >= 5 points wholly
  shadowed, >= 5 wholly lit). Plus the floor-points test now checks both parts. Cost: the direct pass
  is `n_points x (n_lamp + n_reflector)` facets wherever self-occlusion is on -- not measured at mesh
  scale. The `ray_effects_room` cases run `NoOcclusion`, so they take the lamps-alone path and cannot move.
  Also new on `SceneSolution`: `reflector_absorbed_power` (`sum (1 - rho) H A` over the reflectors,
  beside `lamp_absorbed_power`, so the books close from the solution alone) and `irradiance_absorbed`
  (`(1 - rho) E` per `SurfaceReceivers` set, by its `reflectance` -- the field the radiation package
  carried but never read; the case's `E_absorbed` and `absorbed_power` now read it).
  **Mutation pass (9 breaks run, 8 red)** (jax 0.10.2, CPU, x64, Linux, 2026-10-10): lamps gathered alone
  always, reflector samples unnamed, surface points unnamed, lamp samples unnamed, point sources dropped
  from the direct set (caught only by `test_a_point_source_under_a_reflector_is_shadowed_by_it_and_lights_the_rest_as_a_point`,
  `P / 4 pi r^2` beside a black plate, 0 over it), absorbed by rho, `irradiance_absorbed` from the direct
  part only, lamp body ids not offset past the reflectors'. **Dismissed, equivalent**: joining the
  reflectors under `NoOcclusion` too (adds zeros; the lamps-alone branch is cost only); and a
  Lambertian reflector profile replaced by the lamps' first (a zero emission weighs nothing; NOT run).
- **A FACET SENDS FROM ITS FRONT ONLY, IN THE TRANSFER AS IN THE GATHERS (#667, fixed 2026-10-12).**
  `projected_solid_angle` clips the sending triangle to the RECEIVER's half-space alone, and a
  Lambertian set's `assemble` applies no source cosine (`reflected, reflected`), so before the fix
  nothing gated the SENDER's side: a facet sent reflected light out of its back, while every gather
  (`Lambertian.radiance_per_exitance_at`, zero for `cos <= 0`) did not. Measured on `777156e` (jax
  0.10.2, CPU, x64, Linux): a 1 m plate at z = 1 facing down, emission 1, under a 1 m plate at z = 2
  facing down, `receiver_quadrature=1`, either occlusion: the upper plate landed **0.2248 W/m²**; in the
  #622 fixture with the plate at rho 0.6, the ceiling's solved landing was 0.01153 against 0.00972
  gathered at the same point. **The same leak was in the specular exchange** (`images._exchange_rows`,
  found while fixing this): a plate over a floor mirror and facing AWAY from it still sent through it
  (on `d838f90`: the mirror took 0.0946 W/m² from the plate's back, a receiver above gained 0.0337).
  - **The fix**: `_row_block` zeroes the term at each receiving QUADRATURE POINT on or behind the
    sending facet's plane, and `_exchange_rows` at each one on or behind the IMAGE's plane (the image
    faces as its source does after the reflection, so behind it is "the source faces away from the
    mirror"). Judged per point, not per receiving facet: a receiving facet can straddle the sender's
    plane. The side test is `clipping.point_heights` (filtered, an undecidable height is exactly zero
    and counts as not in front), now also what `back_faces._certainly_behind` uses -- one home.
  - ⚠️ **Invisible in every closed enclosure whose facets face into it** (no facet sees another's
    back), which is why no test saw it, and the "model with every facet in one set" reference the
    scene tests use carried the same leak, so it could not see it either. One existing test DID rely
    on it: `test_a_body_in_the_way_removes_the_transfer_across_it` had both "facing" plates wound +z, so
    its only exchange was the lower plate taking light from the upper one's back (sum 0.0 after the
    fix); its upper plate now faces down. Every other radiation fast-tier test (953) passed unchanged.
  - A zero-thickness sheet reflects from both faces once declared: TWO-SIDED SHEETS (#689) below.
  - Tests: `test_a_facet_sends_nothing_out_of_its_back` (both occlusions, front-to-front control),
    `test_a_receiver_across_the_sender_s_plane_takes_light_only_at_its_points_in_front` (a wall across a
    small sender's plane against a hand sum over the points above it, strictly between 0 and the
    ungated sum), `test_a_facet_facing_away_from_the_mirror_sends_nothing_through_it` (both
    orientations), and in `test_radiation_scene.py`
    `test_a_reflector_sends_its_light_out_of_its_front_in_the_solve_as_the_gathers_do` (the #622 fixture
    with a reflecting plate: solved ceiling landing = gathered ceiling irradiance, 1e-11). **Mutation
    pass (4, all red)**: no gate, the gate judged at the receiving facet's centroid (caught only by the
    straddling wall), the gate inverted, no gate on the mirror path.

### #604 STEP 1: LAMPS REFLECT AND SHADOW IN THE EXCHANGE (2026-10-07)

**Current status.** #604 is sequenced by general capability: (1) emitters that also reflect (done); (2)
transparent solids (below); (3) the equivalent surface as a baking step; (4) curved specular reflectors;
(5) path cuts. Lamp facets are ALWAYS in the exchange (`_Exchange.of(scene)`: the reflectors' facets then
every areal lamp facet, no emission, default reflectance 0), so a lamp's triangles shadow reflected
light; `SceneSolution` gains `lamp_irradiance` and `lamp_absorbed_power`; `Scene` refuses lamps and
reflectors sharing a body name, and any specular reflectance. A lamp sample's own facet is named (`own=`
→ `receiver_facet`) and the samples stay in the facet's plane -- never behind it (outside the fluid); the
in-front offset `_IN_FRONT_OF_OWN_FACET` was deleted as redundant (#615). ⚠️ Only a SECOND lamp exposes
the lamp samples' own-facet exclusion. ⚠️ `Box(axes=)` takes the axes as ROWS. ⚠️ A drum in a box is not
convex: do not run it unshadowed. Record, energy, gates and mutation pass:
`.claude/notes/radiation-log.md` → "#604 STEP 1: LAMPS REFLECT AND SHADOW IN THE EXCHANGE (2026-10-07)".
- **Case**: `Lamp.reflectance` (see `case.md`), results `lamp_absorbed_power`, subtracted from
  `unaccounted_power`.

### #604 STEP 2a: TRANSPARENT SOLIDS AND THE REFRACTED DIRECT GATHER (2026-10-08)

**Current status (merged as #638).** A transparent solid is a convex analytic body with an index and an
absorption (`refraction.Transparent`, nested in `Media`); a path crosses the surface of every region
holding exactly one of its ends, once, in a fixed order, its crossings solved by Fermat's principle and
differentiated implicitly (`solve_paths` → `Paths`, compiled once per chain and shape).
`refracted_fluence_rate` / `refracted_irradiance` weigh each triangle by the solid angle on its corners'
arrival directions times the corners' mean of radiance x Fresnel x leg absorption x `(n_receiver /
n_source)^2`; `build_refracted_visibility` → `RefractedVisibility` shadows every leg of the centroid's
path. Reflected branches are deferred to step 4. ⚠️ The solve is a DESCENT on the optical length (Newton
from the straight line runs away near an interface); a dropped path is the solver's until a forward trace
or a quadrature says otherwise. ⚠️ A `while_loop` inside a `while_loop` under `vmap` never ends (the line
search is a fixed ladder of 41 steps). ⚠️ `Chain.beside` must leave out the crossed regions. Record,
checks (a tracer, a quadrature, an optimizer) and mutation pass: `.claude/notes/radiation-log.md` → "#604
STEP 2a: TRANSPARENT SOLIDS AND THE REFRACTED DIRECT GATHER (2026-10-08)".

### #604 STEP 2b-i: TRANSPARENT SOLIDS IN THE SCENE'S DIRECT LIGHT (2026-10-08)

**Current status.** `Scene.media` routes each point's light by medium (`scene._by_medium`): sources in
its own medium along straight lines in that medium's absorption, the others along refracted paths
(`scene._refracted`). With media `Scene.absorption` must be unset, and a scene that exchanges
(reflectors, or a reflecting lamp) is refused until 2b-ii (the refracted transfer and the model's
`media=`). The gathers read a mask's layers by name (`Visibility.layers()`, `surviving_from_layers`).
There is no `Visibility.through` and no `refraction.straight_through` (replaced by the pass-through
routes, next). `build_refracted_visibility(..., receiver_facet=)` excludes the facet a point lies on on
the last leg; `Media.region_of_facets` is the one home of a facet's medium. Record and the four-lamp
tracer checks: `.claude/notes/radiation-log.md` → "#604 STEP 2b-i: TRANSPARENT SOLIDS IN THE SCENE'S
DIRECT LIGHT (2026-10-08)".

### #604 STEP 2b-i, CONTINUED: BENT ROUTES THROUGH A NEIGHBOUR — BUILT, NOT YET MEASURED (2026-10-09)

**Agreed with the user**: fix the shortfall now; **one** pass-through region per path; same-medium pairs
are bent too (no straight-through factor left anywhere).

**Current status.** `Chain.between(..., through=node)` / `Chain.routes`: a route per passable node beside
the base route; a path meeting a region it does not pass through is invalid. A passing route is solved
from four aims about the region's middle (`_starts`), deduplicated. The descent finds only minima, and
the paths round a sleeve's wall alone are saddles, so such a route is SHOT (`shooting.py`,
`refraction._shot`: unknowns the impact parameter and `axial_index` = `n u_z`, a grid scan --
`RadiationSettings.shooting_grid`, default `_ROWS` x `_COLUMNS` -- then Newton polish, continued from
reached corners to those the grid missed); a wall route that cannot be shot is skipped (`_only_saddles`).
A shot route's triangles are CUT to the part with a path (`refracted._clipped_field`,
`_reached_triangles`, `_cut_edges`) and a straddling facet is shadowed along a reached corner's path; the
mask and the gathers share one `_Reach` (`RefractedVisibility.reach`), and `check_shooting_grid` →
`ShootingGridCheck` diagnoses a short grid. Gap routes stay descended (shooting misses their steep
paths). A sound geometric cull for coaxial cylinders exists and is not built. ⚠️ `shooting.py` calls `n
u_z` `axial_index`, never `beta` (the vocabulary guard). Full record -- the saddle finding, cost, the
user's four-lamp runs, the backward gather, shooting, cut triangles, continuation, the grid check, tests
and mutation passes: `.claude/notes/radiation-log.md` → "#604 STEP 2b-i, CONTINUED: BENT ROUTES THROUGH A
NEIGHBOUR — BUILT, NOT YET MEASURED (2026-10-09)".

## TWO-SIDED SHEETS: A DECLARED SHEET IS TWO FACES IN THE SOLVE (#689, 2026-10-17)

**Decided with the user**: the declaration lives on `Surfaces` (`two_sided`, body names, static), not
on an occlusion strategy, because it is geometry; back faces are **appended** after the set's own
facets with a map (`back_of`), not separate `*_back` fields. Found first (posted on #689): supplying the
sheet twice by hand did not work, for two reasons, and both are what the build now handles by index.

- **`Surfaces.with_back_faces()`** appends each areal facet of a declared body with two corners
  swapped; centroid, area and the negated normal are **carried over, not recomputed**, so a face and its
  back are one triangle exactly. `back_of[k]` is the front of the `k`-th back face; `twin` maps both
  ways (`-1` elsewhere). Idempotent; point sources get no back face. **A sheet may not emit**: concrete
  emission or power on a declared facet is refused there (and `Scene` asks it of the lamps), because
  the direct gathers emit from front faces only. `_with_backs` is the unchecked builder, for a solved
  set whose emission is a radiosity. `with_geometry` on an expanded set moves the fronts and remakes
  the backs, keeping every face's own optics. `refine_for_receivers` and `coarsen_surfaces` carry the
  declaration.
- **Where it is expanded**: `build_radiation_model` and each model entry point (`radiosity`,
  `surface_irradiance`, `fluence_rate`), and `_Exchange.of` in the scene. `build_transfer` does NOT
  expand: it builds the set it is given (so the Walton blocker declared a sheet stays one facet,
  two-sided for blocking). Facet-indexed results are `n_facets + len(back_of)` long; the scene splits
  back faces to the set their front belongs to (`SceneSolution.reflector_back_of`, `lamp_back_of`).
- **The two defects, fixed by index, not by tolerance.** (a) A ray aimed at a face's centroid ends in its
  twin's triangle and the far end is inclusive, so the receiving end excludes the twin too
  (`build_transfer` passes `(n, 2)` rows `[i, twin(i)]`; mirror legs exclude a source's twin). The
  sending end needs nothing: the twin is at distance 0, inside the margin. (b) The six-point rule puts
  receiving points one ulp off the plane (z = 0.5 + 4.4e-16), `point_heights` does not snap a rounding
  of the POINT's construction, so a point inside its twin read a whole hemisphere and the #667 gate let
  it through for one orientation: rows of faces turned down summed to **1.3299**, their twins' to 1.
  `_row_block` now zeroes `F[i, twin(i)]` as it zeroes `F[i, i]`.
- **`SilhouetteOcclusion._receiver_frames` accepts rows naming several facets in one plane** (a face and
  its twin are parallel), facing the first's way; rows not in one plane still need `receiver_normal`.
- **MEASURED against a closed slab** (`tests/unit/test_radiation_two_sided.py`): `inward_box(4)` (96
  facets, emitting, rho 0.6) round a sheet of 8 triangles at z = 0.5137 (off the box's grid), against
  the same box round a slab 1e-5 thick whose top and bottom are the two faces; 30 interior points; jax
  0.10.2, CPU, x64, Linux, 2026-10-17. Silhouette clip: walls, both faces and `G` within **6e-6**.
  Ray test: walls 8e-5, fronts 9e-6, `G` 2e-4, back faces **1e-3** -- and the 1e-3 is the slab's: the
  differing pairs are walls straddling the sheet's plane whose centroid ray crosses the slab's top,
  which a slab's bottom does not actually look through. ⚠️ **On the box's own grid the ray test read 4%
  off**: 288 wall-wall rays ran exactly along the sheet's rim, which the slab's sides block and a sheet
  cannot -- a fixture degeneracy (the baffle lesson of #537 PR 3b again), not a defect. Undeclared, the
  field below the sheet is 2-6% off the slab under the clip (brighter: light passes through from
  behind).
- **Mutation pass (11, 10 red, then 11 after a test was added)**: the twin exchange not zeroed, the
  receiving end not told of the twin, the back face's normal not negated, a faced sheet still counted
  two-sided, the declaration ignored, the scene's exchange not expanded, the emission refusal, the
  parallel-rows check, the model build not expanding, the name check. ⚠️ **The back face's own-facet map
  in the scene (`_Direct.facets_of`) first SURVIVED**: the shelf fixture ran `NoOcclusion`, where a
  sample point's own triangle cannot shadow it; `test_the_lamps_light_on_a_back_face_is_not_shadowed_by_its_own_triangle`
  runs the ray test and catches it. **Dismissed, equivalent**: the mirror path's first leg ignoring the
  source's twin -- it lies at distance 0, inside the margin (the direct rays need no such exclusion).

