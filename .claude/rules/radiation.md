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

The dated record — measurements with their configurations, probe results, per-change records (#NNN),
test inventories, mutation passes — is in `.claude/notes/radiation-log.md`. A pointer below names the log
heading to search for; sections condensed on 2026-10-17 are there verbatim as `## [rule text] <heading>`.

## Built so far

| module | what it holds |
|---|---|
| `solid_angle.py` | the two geometric kernels, plus the signed form |
| `stl.py` | ASCII and binary STL reading |
| `surfaces.py` | the `Surfaces` value object (optics, labels, two-sided sheets) |
| `checks.py` | build-time geometry checks, including the cell-in-the-metal test |
| `subdivide.py` | the width-over-distance refinement |
| `coarsen.py` | coarsening a dense surface (a mesh patch) by edge collapse |
| `profiles.py` | `Profile` (asked about a direction and a facet normal); `AxisymmetricProfile`: `Isotropic`, `Lambertian`, `CosinePower` |
| `photometry.py` | `read_ies`, `Photometry`, `PhotometricProfile` — an IES LM-63 table over both angles (direct gathers only; the transfer refuses it) |
| `gather.py` | `direct_fluence_rate`, `direct_irradiance`, `summed_fluence_rate`, `streamed_fluence_rate` |
| `work.py` | `in_passes`, `receivers_per_pass`, `DEFAULT_PAIR_LIMIT`, `PASS_PAIRS` |
| `absorption.py` | `UniformAbsorption`, `VoxelAbsorption` (the voxel walk is the optical-depth piece) |
| `visibility.py` | the frozen shadow mask |
| `receiver_shadows.py` | `FrozenShadows` / `StreamedShadows` |
| `culling.py` | how the bodies' layer is decided: `ShaftCulling` (default) or `EveryPair` (the reference) |
| `back_faces.py` | `BackFaces`: which volume receivers lie certainly behind which facets |
| `lit_blocks.py` | the gather's layout: receivers in blocks of 8, each against the facets that can light it |
| `triangle_body.py` | `TriangleBody`, a `Body` of triangles over a `TriangleGrid` |
| `triangles.py` | watertight ray-triangle intersection |
| `grid.py`, `grid_walk.py` | `TriangleGrid` and its compiled (Numba) walk |
| `self_occlusion.py` | `SelfOcclusion`: `RayCastOcclusion` (default), `SilhouetteOcclusion`, `NoOcclusion` |
| `silhouette.py` | exact covered fraction, conservative cone cull, `covers_nothing` |
| `clusters.py` | `FacetClusters` for the silhouette's clustered cull |
| `clipping.py` | convex clipping with filtered sign tests, shared with `solid_angle.py` |
| `transfer.py` | the frozen facet-to-facet geometry |
| `quadrature.py` | symmetric triangle rules for the receiving facet |
| `model.py` | the assembled model and the three public entry points |
| `images.py`, `mirrors.py`, `mirror_visibility.py` | planar specular reflection, one bounce (#537) |
| `refraction.py`, `refracted.py`, `shooting.py` | transparent solids: paths by Fermat's principle, the refracted direct gather (#604 step 2) |
| `units.py` | lamp watts to exitance, ultraviolet transmittance to absorbance |
| `scene.py` | `Scene` / `solve_scene` / `SceneSolution` — what a radiation case file builds |

The solid bodies (`Body`, primitives, CSG, `Outside`) live in `aquaflux/solids/`
(`.claude/rules/solids.md`). There is no `aquaflux.radiation.occluders` module and no `Occluder` class;
*occluder* is a role (a body passed in `occluders=`), and `aquaflux.radiation` does not re-export bodies.

## ⚠️ THERE ARE TWO SOLID-ANGLE KERNELS AND THEY ARE NOT INTERCHANGEABLE

- `solid_angle` — the plain solid angle `Ω`, for **fluence rate** (a point in a volume has no receiver
  cosine).
- `projected_solid_angle` — `∫cos θ dω`, for **irradiance** and every surface-to-surface transfer factor.
- `signed_solid_angle` is `solid_angle` with the winding sign kept (`solid_angle` is its magnitude). The
  sign means something only summed over a consistently wound closed surface (the winding number); do not
  use it elsewhere.

No scalar converts one into the other. Using `Ω/π` in a transfer makes every row sum to exactly 2, so
`spec(ρF) = 2ρ` — the wrong answer is clean, which is why it recurred through five design drafts. A wrong
kernel is identified by non-convergence under refinement, not by error size.

## Load-bearing implementation details in the kernels

1. **Every angle is `arctan2`**, never `arccos` and never a sum of interior angles (both ill-conditioned
   for small or nearly collinear triangles; the `arccos` error is flat in refinement at ~1e-8).
2. **The contour form is signed; clip the polygon to the receiver's front half-space first**
   (`clipping.clip_to_halfspace(..., 4)`: a fixed four-slot loop, empty slots repeat the last survivor and
   contribute zero-length edges).
3. **Take the magnitude.** A contour sum's sign records only vertex storage order; orientation is the
   normal's job. Test fixtures wind faces inconsistently on purpose.
4. **Guard a square root inside the root**: `sqrt(where(zero, 1, sq))`, never `v / where(zero, 1,
   sqrt(sq))` — the latter has a NaN gradient. Zero-length edges are the normal case (the clip makes
   them). ⚠️ It applies to any primitive with an infinite or `0/0` slope where a selection discards its
   value: it recurred through `**` (`profiles._forward_power`), an unguarded `sqrt(dot(...))` in both
   absorptions (now `aquaflux.vectors.norm`), and `arctan2(0, 0)` in `PhotometricProfile.angles` (#620).
   ⚠️ A Lambertian fixture cannot see a NaN direction; differentiate in position with other profiles.
5. **The vertex-degeneracy guard in `solid_angle` (`where(any(degenerate), 0, omega)`) is for the
   gradient**: the value is right without it, but `d/dp` comes back a finite, fictitious `(0, 0, −2)`.
   Pinned by `test_the_vertex_guard_is_there_for_the_GRADIENT_the_value_needs_no_help`.

## A convention callers must honour: `F_ii = 0`

A facet contains its own centroid, so both kernels return a full hemisphere/sphere there. An assembled
transfer must zero its own diagonal (whoever assembles does it; the kernel knows nothing of matrices). A
coplanar receiver outside the facet correctly gets zero. The same convention binds the **gather** at a
receiver lying on an emitting facet: `direct_irradiance(..., receiver_facet=)` leaves those facets out
(the kernel and the cosine gate read rounding noise there; THE SCENE, below).

## Testing

`tests/unit/test_solid_angle.py`, references in `tests/unit/radiation_references.py`: two closed forms
plus recorded **pyviewfactor** values. ⚠️ The oracle's values are **recorded, not imported** (an optional
import skips silently); regenerate with `pip install pyviewfactor` when the geometry changes, never make it
a dependency. ⚠️ On-axis tests cannot see a missing receiver cosine — every new case needs an oblique
configuration. ⚠️ Fixtures of equal-area facets (subdivided cubes) cannot see a row/column weighting error,
and all-emitting boxes cannot see energy-conservation or stopping-rule defects: use `stretched_box` for
anything involving an area and the lamp-in-a-dark-box fixture for conservation. ⚠️ Run mutation passes with
`PYTHONDONTWRITEBYTECODE=1`: `mv`-restored files keep their mtime and stale mutated `.pyc` gives false reds.

## ⚠️ A ZERO-AREA FACET IS LEGAL — IT IS A POINT SOURCE

`Surfaces` carries **emission** (W/m², areal) and **radiant power** (W, point) separately; a point source
is a zero-area triangle carrying power. Rejecting degenerate triangles deletes every point source; dividing
by area for a normal fills the geometry with NaN. `from_triangles` guards the division (normal zero there).

- **Power is extensive, so `refine_for_receivers` refuses it**: refine the areal facets, then add point
  sources. Point-source labels go through `division.origin`; a labelled facet wide enough to split is
  refused.
- **Both ends of an index are checked** (`surfaces._check_indices`): a `-1` in `solid_id` or
  `profile_index` would select the last entry. An ASCII STL triangle before the first `solid` line is
  refused for the same reason.

## ⚠️ A SOURCE'S KIND IS A LABEL, NOT ITS AREA

`Surfaces.point_source_index` (static) decides the code path; read `Surfaces.is_point_source`, never
`area > 0`. Area is a traced quantity under a vertex-position gradient, so a host-side partition on it makes
the gradient unbuildable. (`area` is not used numerically in the gather at all.) **Move a set with
`Surfaces.with_geometry`**, never by `tree_at` on `vertices` (centroid, normal and area would stay stale).

## STL reading and winding

- ⚠️ **Detect the STL format by file length** (`84 + 50 × count` from its own header), never by a leading
  `solid` keyword — binary headers are arbitrary text and some begin with `solid`. Pinned by such a fixture.
- Body names assign optical properties, so `per_facet` raises on a name missing from the file and on a body
  the mapping omits (a dropped name leaves a lamp dark).
- **Winding is the most common real defect**: reversed facets point into the solid, the source-side gate
  discards them, and the surface silently emits less. `check_winding` raises, `winding_report` counts. Test
  is edge parity after merging vertices within a tolerance **relative to the model's extent** (STL repeats
  endpoints as separately rounded coordinates). Boundary edges are legal (open surfaces); non-manifold edges
  are counted, not fatal; watertightness is not required.

## The refinement criterion

Refinement is about what is assumed constant across a facet (emission, reflectance, outgoing radiance,
attenuation along the centroid's path), not the geometry term (exact). Refine until longest edge over
distance to the nearest receiver is below `max_ratio=0.25` (0.15 / 0.05 for accurate work) and **report the
realized distribution**. Distance is the minimum over the three vertices and the centroid (the centroid
alone under-refines exactly where a receiver is closest). The four-way split's middle child must keep its
siblings' winding.

## Profiles

A profile is a normalized **intensity** distribution (`∫4π f = 1`; power carried separately). Each supplies
`intensity_fraction(direction, normal)` and `radiance_per_exitance(direction, normal)` with
`radiance_per_exitance · (d·n) == intensity_fraction`. Axisymmetric ones implement both from
`intensity_fraction_at(c)` / `radiance_per_exitance_at(c)` (what the transfer calls on its frozen
`source_cosine`); the radiance form is written out, not derived, because the derivation divides by `c` and
Lambertian is `0/0` at grazing. ⚠️ A profile carries its constant in two methods (the gather reads both, the
transfer only `radiance_per_exitance_at`), so a test of one path cannot see a mutation of the other; the
contract is pinned in `test_radiation_profiles.py`.

- `CosinePower(n) = (n+1) max(cos,0)^n / (2π)` — over `2π` (hemispherical); `n = 1` must reduce to `cos/π`.
  `n < 1` is refused.
- `Isotropic` is point-source-only and raises from `radiance_per_exitance`. `check_profiles` refuses
  isotropic-on-areal and directional-on-point.

## A measured luminaire is a table over TWO angles, so a profile is asked about a DIRECTION (2026-09-28)

`photometry.py` reads IES LM-63 (Type C, `TILT=NONE`); first consumer the Ushio Care222 B1 module
(`validation/ray_effects_room/ushio_b1.ies`), which varies by up to 8 % of peak round the axis.

- **Frame matches of-optical-radiation's (OOR) `iesEmitter`**: `gamma` from the facet's outward normal, `h`
  from `up` towards `normal × up` (the normal plays OOR's `fixtureAxis`). Pinned by
  `test_the_horizontal_angle_runs_towards_normal_cross_up`. `up`'s normal component drops out exactly (no
  projection needed); `up` parallel to a normal is refused (`Profile.refuse_normals`, via `check_profiles`).
- **Interpolation and symmetry are OOR's**: bilinear in degrees, `gamma` clamped, symmetry folded by LM-63
  rules and read off the horizontal span; the table is expanded round the circle once at
  `Photometry.profile`, so the traced code has no symmetry branch. `_POLE_SPREAD = 5 %` warns on a
  non-uniform pole.
- **Normalized over the front hemisphere** (`front_hemisphere_flux`, exact for the bilinear table), as
  OOR's BC is, so both emit exactly `P`. `Photometry.flux` is the table's own range.
- Radiance floored at `MIN_COSINE = 1e-3` (OOR's `IES_MIN_COS`).
- **The transfer refuses it** (`NotImplementedError`): it freezes one `source_cosine` per pair. This does
  not stop an IES lamp lighting a reflecting room: `solve_scene` keeps lamps' emission out of the transfer
  (THE SCENE). Average a lamp's irradiance over sub-points of each facet, not its centroid.

Record (#621 span fix, #632 pole warning, timings): log → "A measured luminaire is a table over TWO angles…".

## SPECULAR REFLECTION: planar mirrors, one bounce (#537)

Planar mirrors, one bounce; **a mirror is one plane, not one body**; curved specular bodies are #604 step 4.
`Surfaces` carries `diffuse_reflectance` and `specular_reflectance` (default 0), both live, each in `[0, 1]`,
summing to at most one; there is no `reflectance=` keyword. `vectors.reflect` is the one reflection
formula; `Profile.mirrored(normal)` is abstract (a `PhotometricProfile` reflects `up` and flips
`handedness`); `Mirror.image` reflects vertices and swaps two corners; `planar_mirrors` groups a body's
areal facets by plane, largest facet first.

- `images.py` gathers each source's image through each mirror's aperture facets (`silhouette.covered_by`
  with the aperture as blocker; medium on both real legs). Cost is set by the aperture's triangle count:
  the clip is cone-culled (`images._screen`; survivors clipped `_CLIP_BATCH` at a time in a `while_loop`,
  so the culled clip takes geometry under `stop_gradient`), laid out by `lit_blocks`; an image wholly
  inside the mirror skips the clip (only with one concrete reflectance on every facet).
- `build_radiation_model(..., specular=(names,))` stores the exchange per `(solid, crossing pattern)`
  (`specular_terms`), `ρ_s` live per declared solid, `∏ τ_b^legs_b` applied at `assemble`.
  `MAX_MIRROR_PLANES = 12` per specular solid. Refused: a graded medium with specular bodies; a traced
  `ρ_s` on a model with no specular body (an `eqx.error_if` let `grad` return 0.0); a specular body with no
  areal facet. A specular exchange whose every path is hidden is an empty `(0, n, n)` stack.
- Reflected paths are shadowed on both legs (`mirror_visibility.py`) through a meeting point lifted off the
  mirror; under `SilhouetteOcclusion` they are rays.
- ⚠️ `inward_box`'s right half is a translate of its left, not a mirror image: build a symmetry fixture by
  reflection.

Plan, PR records, tests, mutation passes: log → "SPECULAR REFLECTION: THE REFLECTANCE SPLIT AND THE
MIRRORS (#537 PR 1, 2026-10-05)" and its PR subsections.

## The source-side clamp is a GATE, not a factor

A facet cannot light what is behind it; the sign of the source cosine gates the term. As a multiplier the
two sides cancel (a Lambertian tube read 1.5e-4 against 2), and `abs` or no clamp doubles it — a near-zero
field reads as an empty scene. A convex emitting body needs no occluder: the gate is its visibility; the
cylinder case does not test occlusion.

## ⚠️ INSIDE A TRACE, `jnp` STAGES EVERYTHING — EVEN ON CONCRETE INPUTS

`jnp.max(concrete_array)` returns a tracer under an active trace. Build-time checks on index arrays
(`solid_id`, `profile_index`) are written in numpy and skipped when the array itself is traced;
`profile_index` is stored as numpy everywhere (#522). Record: log, same heading.

## ⚠️ GEOMETRY IS CLOSED OVER, VALUES ARE PASSED

The gather partitions facets by profile and by areal/point **on the host**, so the program's shape depends
on `area`-kind labels and `profile_index`, which cannot be traced. `jit` over a whole `Surfaces` raises;
close over the set and substitute values through `with_optics` (a traced `profile_index` arriving that way
is refused). Vertices are not structural, so a source's position may move under a gradient (direct
gathers only — see the model's geometry check below).

## What the analytic cases pin

- Sweep the distance (separates `r` from `r²` and `π` from `4π`) and the angle (a missing receiver cosine
  passes at normal incidence; a fixed radius with tilted normal gives `E = G cos`, a plane at fixed distance
  gives `cos³`).
- `E = G cos` is a single-source identity (a test asserts it fails for two opposed sources).
- The disc (`G = 2B[1 − h/√(a²+h²)]`, `E = B a²/(a²+h²)`) separates the kernels on one geometry.
- A summed line source is a midpoint rule: assert the rate, and only once spacing is below the receiver's
  distance. Relative error is second order in width/distance; absolute is fourth.
- The Beer–Lambert slab: `G = 2M E₂(κx)`, `E = 2M E₃(κx)` — not `exp(−κx)` (collimated). Sweep to `κx ≥ 2`.
- An infinite line in absorbing water gives the Bickley functions `Ki₁`/`Ki₂`; the reference quadrature
  needs `epsabs=0, epsrel=1e-13` (scipy's default `epsabs` is the weak link).
- `G = 4B` in a closed Lambertian enclosure at uniform radiosity, anywhere inside (next-but-one section).

## Optical depth: exact, not marched

`UniformAbsorption` is `τ = a r`. `VoxelAbsorption` walks the trilinear field's real boundaries (no step
size), with three details each of which cost a debugging round:

1. ⚠️ **Cut on the sample planes, not the cell faces** (half a cell apart); a piece straddling a sample
   plane straddles a kink (0.43 % wrong on a linear field).
2. ⚠️ **Generate cuts only at planes that carry samples, at both ends**; otherwise a long segment exhausts
   the fixed step budget outside the grid and silently loses its tail.
3. **Simpson per piece is exact** (trilinear along a line is cubic).

Interpolate, never nearest-cell (zero gradient a.e.); outside the samples the value is clamped. No clamp
on `τ`. The absorbance grid need not match the flow mesh (cost is `nx + ny + nz + 1` steps a segment). The
walk carries its running total in the scan carry and checkpoints the step (#528: ~100 B a pair forward,
~3.5 kB under a gradient at 32³; not bit-identical to the old stacked form). ⚠️ Attenuation is taken along
the facet **centroid's** path; the remedy for a wide facet is the refinement criterion.

## Occlusion: a frozen mask and live transmittances

`surviving = ∏_bodies [1 − blocked · (1 − transmittance)]`: `blocked` is frozen at build, `transmittance`
is live. `dG/d(occluder geometry)` is **exactly zero by construction** (with either self-occlusion
strategy: the fraction is computed on the host, off the tape); `dG/d(transmittance)` is exact.

- ⚠️ The mask is indexed by receiver; `Visibility` carries its receivers and the gather checks them.
- ⚠️ A mask supplied without transmittances defaults to **opaque**.
- The surface's own layer is stored at the narrowest type each strategy can state (#525): `NoOcclusion`
  `None`/`None`, `RayCastOcclusion` bool/`None`, `SilhouetteOcclusion` float64/bool
  (`hidden_by_geometry` / `overlapping`); `surviving_fraction` widens per chunk. ⚠️ Code reading
  `hidden_by_geometry` must allow `None` and a bool. Gathers read a mask's layers by name
  (`Visibility.layers()`, `surviving_from_layers`).
- **At mesh scale a mask cannot be held** (1.6M cells × 7,516 facets is 12 GB per body).
  `direct_fluence_rate` therefore also takes `occluders=` (and `self_occlusion=`) **instead of** a built
  mask, building and dropping each chunk's mask; giving both raises. Streaming rebuilds every call.
- `Body.traceable` (a `ClassVar`, default `False`): `build_visibility` compiles traceable bodies and calls
  the rest directly. ⚠️ Compiling unconditionally crashes a scene holding a host-side (triangle) body
  (`test_a_host_side_blocker_and_a_primitive_stand_in_one_scene`).

### A model can stream its receiver mask — explicit, frozen by default (#489)

`RadiationSettings(stream_receiver_mask=True)` keeps the recipe: `RadiationModel.receiver_shadows` is
`FrozenShadows(visibility)` (default) or `StreamedShadows(geometry, occluders, visibility_options)`; there is
no `RadiationModel.receiver_visibility`. `fluence_rate` asks for the emitted and reflected fields together,
so one mask per chunk serves both gathers. **Explicit by the user's decision**: it trades memory against
repeated work, which only the caller can judge; it is a memory choice, not a differentiability one. A
gradient is chunk-bounded too (`gather.streamed_fluence_rate`: each chunk an `eqx.filter_custom_vjp`
rebuilding its mask in the backward pass; not `jit`-compilable). Masks come from the build-time geometry;
a receiver inside a body is refused at build (`visibility.refuse_points_inside`). Record: log →
"A MODEL CAN STREAM ITS RECEIVER MASK…".

## The emitting surface occludes too, and that half is opaque

`Visibility.hidden_by_geometry` is the surface's own triangles' layer (opaque walls, a **fraction** of
each source; beside a boolean `overlapping`). How it is computed is an injected `SelfOcclusion`:
`RayCastOcclusion` (one ray per pair, 0/1 — **the default**), `SilhouetteOcclusion` (exact clipped
fraction), `NoOcclusion`. ⚠️ **"Off" is `NoOcclusion()`, never `None`**: `RadiationSettings` drops `None`
fields, so `None` falls through to the ray-test default.

- **A model builds two masks** — facet to facet (`visibility_options()`) and facet to volume receiver
  (`receiver_visibility_options()`). `receiver_occlusion` overrides the second; unset, it follows
  `self_occlusion`. Selecting the silhouette clips every cell; `receiver_occlusion=RayCastOcclusion()`
  keeps the clip between facets only.
- **Exclusion is by index, never by tolerance**, and never by whole solid (a bent duct's blocking wall is
  the emitter's own body).
- **The facet-to-facet mask casts each pair once, and only where it is read (#529).** `build_transfer`
  builds it with `visibility.build_facet_visibility(occluders, surfaces, needed=transfer._read_pairs(...))`,
  which shares `build_visibility`'s private tail (`visibility._assembled`) and asks the strategy
  `SelfOcclusion.between_facets(surfaces, near, needed)` instead of `field`. The base method is `field`
  with the centroids as receivers (rows `Surfaces.own_faces`), ignoring `needed` — what
  `SilhouetteOcclusion` keeps, since its share is not symmetric. `RayCastOcclusion` casts one ray per
  unordered pair, written to both entries, **from the lower index, or from the point source when exactly
  one is one** (a point source has margin 0, so one on a wall is hit at the far end of a ray aimed at it
  but not by the ray leaving it); the areal mask is exactly symmetric. `needed` is `geometric != 0`
  (exactly) **or** an areal row against a point-source column, which the model's point-source arrivals
  read through this mask; a pair is cast if needed either way. ⚠️ **So `transfer.visibility`'s own layer
  is complete only where read**, `False` elsewhere (the `TransferMatrix` docstring says so); build
  `build_visibility(..., receiver_facet=surfaces.own_faces)` for the whole mask. `row_sum_error` and
  `reciprocity_residual` read `geometric` only, so they cannot see a mask change. ~2.0x ungridded from
  casting once, ~2.5x with the cull on a closed box (measured in the log, "THE FACET-TO-FACET MASK").
- ⚠️ **A ray between two facets needs two exclusions** — the source facet and the target facet (the far
  end is inclusive, so a ray aimed at a centroid hits its own target). `exclude` takes `(n_rays,)` or
  `(n_rays, k)`; `build_visibility` takes `receiver_facet`. Missing it read 120 of 132 visible pairs blocked
  on a closed box. Regression: `test_a_closed_box_reaches_its_closed_form_AT_THE_DEFAULT_SETTINGS` (the one
  transfer test that keeps the mask on). The mask must change no number on a convex enclosure (asserted
  bit-identically). ⚠️ `row_sum_error` is blind to the mask (applied in `TransferMatrix.assemble`).

## Watertight ray-triangle intersection

Woop, Benthin & Wald (*JCGT* 2013), not Möller–Trumbore (which leaked 6 of 268 on a closed hull).

- ⚠️ **The watertight guarantee needs the shared-edge function to be exactly antisymmetric, and XLA's
  fused multiply-add breaks `a*b − c*d` once compiled.** `_edge_function` averages the expression with
  the negation of its own swap (IEEE subtraction is antisymmetric however operands were formed). No XLA
  flag or `optimization_barrier` prevents the FMA — fix the arithmetic. Pinned by
  `test_the_edge_function_survives_being_compiled`, which also asserts the plain form still fails.
  Leak sweep: `validation/radiation_watertight_sweep.py`.
- **Discriminator for other exact-arithmetic sites**: does the cancellation feed a discrete predicate? A
  sign test that picks which triangle claims a ray has no small errors; an angle moved by a last bit does.
- The published `kx`/`ky` swap for a negative axis is omitted on purpose (measured: changes no hit; it
  matters only for determinant-sign back-face culling, and this module culls by facet normal).
- **`work_limit` bounds rays × triangles per compiled call, triangles first** (`_call_shape`): the triangle
  block is the reuse factor; chunks shorter than full are padded to a power of two. The per-block kernel
  `_block_is_cut` is traced (#462). Cost is `n³` on the transfer's own case (~420–465 Mtest/s, 3,184
  facets: 77 s) and independent of what rays hit; the grid (below) breaks that by design.
- One predicate, two implementations: `_counts_as_hit` (`distance > near`, `distance <= 1.0`, not
  excluded) with `_watertight_hit` for `_block_is_cut`, and a copy in `grid_walk._cuts` because a Numba
  loop cannot call a traced function. A change to one must be made to the other; the grid tests compare
  them at every resolution.

## The surface system

`(I − diag(ρ) F) B = M + ρ ((F^M − F) M + H_external)`, solved matrix-free by GMRES (restart 120, global
relative `1e-10`). **The bounce count is not a parameter** — the inverse is the infinite sum.

**Exact, and gate on these:** row sums 1 to ~1e-15 (bounds `spec(ρF)` by max `ρ`; `row_sum_error`);
`B = M/(1−ρ)` on a uniform closed box to 1e-12 at `ρ = 0.9`; a Lambertian source's energy balances.

**Not exact, and none converges under mesh refinement:**

- **Reciprocity** is the receiver quadrature's error and falls with point count, not refinement
  (`reciprocity_residual`, a diagnostic). Global energy conservation follows it and at one point per
  receiver gets worse with refinement on a lamp in a dark box. Per-column error is ~10x larger on a 1×1×3
  box than on a cube (`build_transfer`'s docstring states both).
- **A non-Lambertian source** does not balance (source-side: its profile is evaluated at one
  centroid-to-centroid direction to keep its parameters live); this one shrinks with refinement. A point
  source in a dark box likewise.
- ⚠️ `reciprocity_residual` is normalized by the **largest** entry, not per pair (per-pair turns 1e-18
  coplanar entries into a residual of ~1).
- ⚠️ **`geometric[i, j]` is `F_{i→j}`**, used as the weight with which `j`'s radiosity lights `i`, so
  reciprocity weights the **row** by `A_i`. A column weighting is invisible on equal-area fixtures.

Numbers: log → `[rule text] The surface system…` and `[rule text] ⚠️ \`geometric[i, j]\`…`.

### The receiver quadrature

The source half of the double integral is closed form, the receiver half is a quadrature (six points by
default). That keeps row sums exact; quadraturing both sides would make reciprocity exact instead. Do not
"finish the job" by quadraturing the source. Polynomial degree does not order the rules here (the
integrand is nearly singular between neighbours): the 7-point degree-5 rule is worse than the 6-point and is
deleted; the 4-point degree-3 rule has a negative centroid weight and is absent.

### The one-point factors (#447)

Three factors multiply the geometric term elementwise so they stay live: the centroid separation
(absorption), the source cosine (a non-Lambertian profile), and the occlusion mask. **Anything promised a
gradient is computed outside the frozen arrays**; a frozen array may hold whatever *geometry* a live factor
needs, so long as the live parameter stays outside it. A uniform absorption goes through
`exp(−a · frozen_separation)`; any other `Absorption` re-walks every pair on every call. Getting the split
wrong leaves a plausible number (freezing all of `F` loses a few % of `dG/da`; visibility inside the
geometry loses ~2/3 of `dG/dt`).

- Absorption at the centroid separation is first order in `a·w`; the profile bias is second order in
  angular width and zero for Lambertian (all reflected light); **neither has a fixed sign, so neither is a
  correction**. **Decided: documented as a limitation, not built** (a fix would freeze two moments of
  `log cos` per pair). Instrument `validation/radiation_one_point_factors.py`.
- The binary per-pair mask's error, as a fraction of the shadow, is large on coarse plates
  (`validation/radiation_partial_occlusion.py`). Receiver quadrature of visibility and source pixelation
  were **measured and refuted** (2–4x for 4–16x the rays; a silhouette cuts diagonally across receiver ×
  source). The built answer is `SilhouetteOcclusion`.

Records: log → "MEASURED: what the binary per-pair occlusion mask costs (issue #447 item 2)", "MEASURED AND
REJECTED: two candidate fixes…", "MEASURED AND DECIDED: the two OTHER one-point factors…".

### Stopping rule and adjoint

⚠️ **The stopping rule is chosen, not defaulted**: a componentwise relative test is unsatisfiable when most
facets do not emit (stock `lx.GMRES(rtol=1e-10, atol=0.0)` fails on a lamp in a dark box); the solve uses a
global relative test. **The adjoint is an implicit solve**, pinned by restart length 2 against 120 (47
cycles against 3) giving gradients equal to 1e-8, with the step counts asserted to differ.

## The public surface: `model.py`

`build_radiation_model(receivers, surfaces, occluders=..., settings=...)` freezes the `n²` transfer, its
facet-side mask and the receivers' mask:

```
B, cycles = radiosity(model, surfaces)            # what each facet sends out   (n_facets,)
H, cycles = surface_irradiance(model, surfaces)   # what lands on each facet    (n_facets,)
G, cycles = fluence_rate(model, surfaces)         # the volume field            (n_receivers,)
```

Each returns the solver's restart-cycle count. It takes receiver positions, not a `Mesh` (the fence is
one-way; injecting `G` into transport is the consumer's job).

Four assembly steps, each invisible when omitted:

1. Point sources arrive by an extra gather (they are absent from `F`).
2. The reflected part is re-gathered as **Lambertian**, whatever the source emitted like.
3. The facets' own emission is subtracted first (`outgoing − emission`).
4. Both masks are built against the same bodies in one call.

- **`RadiationSettings` fields are all `None` by default and unset means absent** (`_passed` drops them),
  so each default is written once beside its reason. A setting whose reason names no lamp, wall or medium
  belongs here; anything else belongs on `Surfaces` or an `Absorption`.
- ⚠️ **The surface set passed at call time must be the build's geometry, and every call checks**
  (`RadiationModel.geometry`, a SHA-256 of the vertices plus point-source labels; exact). **Traced vertices
  are refused at every call** (`TypeError`): the transfer is frozen, so a traced lamp reached only the
  direct gather and its derivative was wrong. Position derivatives are `direct_fluence_rate` /
  `direct_irradiance`'s; a model may still be built from traced vertices (fingerprint `None`).
- The interreflection solve is `model._interreflection`, an `eqx.filter_jit` building the operator inside,
  traced once per matrix size and solver settings. ⚠️ A plain held-mask gather still compiles per call
  (only the streamed path is cached). ⚠️ `build_radiation_model` returns before its asynchronous work
  finishes: wait on the model before timing what follows.

### `G = 4B` — the closed form that pins the whole assembly

A closed Lambertian enclosure at uniform `B` has `G = 4B` at every interior point. Uniform `M`, `ρ`:
`G = 4M/(1−ρ)` (dropping the reflected gather gives `4M`). **Cosine-power sources with zero emission lit
only by `external_irradiance`** still give `G = 4B` with `B = ρE/(1−ρ)` — the only fixture that catches
step 2. In a closed box of uniform `ρ`, every row of `d(B, H, G)/dM` sums to `1/(1−ρ)` (×4 for `G`): a
no-finite-difference adjoint check. ⚠️ The **per-facet** form `d(ΣAH)/dM_k = A_k/(1−ρ)` needs reciprocity
and is off by the reciprocity error on a stretched box — not an adjoint error; do not "fix" the test toward
it.

## The transfer build is filled in place (#521)

`build_transfer` fills three `n × n` buffers in place: `_row_block` computes `chunk_size` (default 256)
receiving facets against the sending facets `_columns_in_front` keeps (a Numba per-block test: dropped when
no vertex is strictly in front of any quadrature point, or `_projected_bound` is under `_NEGLIGIBLE =
2⁻⁵²`), and `_written` row-scatters into donated buffers; inputs are `stop_gradient`ed. A skipped pair's
value is zero or under `2⁻⁵²`. ⚠️ **A point source is left out by its label**, and only a labelled facet
with an area can show a wrong index (`test_a_point_source_is_left_out_by_its_label_not_by_its_area`).
⚠️ Freed memory is not handed back and JAX dispatches asynchronously: attribute a peak by sampling between
steps, each forced to complete. Record: log → "THE TRANSFER BUILD IS FILLED IN PLACE…".

**A facet sends from its front only** (#667): `_row_block` zeroes the term at each receiving quadrature
point on or behind the sending facet's plane, and `images._exchange_rows` at each point on or behind the
image's plane; judged per point (a receiver can straddle the plane), with `clipping.point_heights` (also
what `back_faces._certainly_behind` uses). ⚠️ Invisible in any closed enclosure whose facets face into it.

## Bounds on work: `pair_limit` and `PASS_PAIRS`

- **`pair_limit`** (default `work.DEFAULT_PAIR_LIMIT = 4,000,000` receiver-by-facet pairs) is the memory
  bound on every loop over receivers: the gather's chunks, the streamed mask, `build_visibility`'s body
  test, `RayCastOcclusion`'s passes, `enclosure_winding`, and the non-uniform-absorption facet-pair walk in
  `TransferMatrix.assemble` (via `gather_pair_limit`). `work.receivers_per_pass(pair_limit, per_receiver)`
  is the one conversion. There is no `chunk_size` on these (renamed, so an old caller fails loudly);
  `RadiationSettings.gather_pair_limit` replaced `gather_chunk_size`. `work_limit` (rays × triangles) is a
  different unit; the transfer's `chunk_size` (receiving facets) was deliberately left.
- `work.in_passes` wraps its scan body in `jax.checkpoint(prevent_cse=False)` (gradient memory flat in
  receivers), runs a short last chunk as a one-step scan, and forms at most **`PASS_PAIRS = 2¹⁶`** pairs a
  step however high `pair_limit` is — a cache bound (a step's working set against a core's L2), not a
  memory one. The transfer build does not take it. ⚠️ `_segment_part` is cached under `_Labels` and
  `pair_limit`, which do not carry `PASS_PAIRS`: measure each bound in its own process. Pinned by
  `test_a_step_forms_no_more_than_the_pass_bound_under_a_higher_limit`.
- `summed_fluence_rate(sets, ...)` forms the geometric pass once from the first set and weights it per set
  (different concrete vertices refused); the surviving fraction is formed per chunk (`_shadow_rows`);
  `direct_irradiance(..., point_sources_only=True)` visits no areal facet.
- ⚠️ Never form a whole-problem `(n_receivers, n_facets)` array before chunking (pinned by
  `test_a_scene_with_nothing_in_the_way_forms_no_array_the_size_of_the_problem`). ⚠️ A translation test
  on a uniformly emitting closed box reads zero to rounding: emit unevenly.

Records: log → "A PASS IS BOUNDED IN RECEIVER-BY-FACET PAIRS…", "A TRACED STEP IS BOUNDED BY CACHE…",
"PER-CALL COMPILES…", "THE VOXEL WALK CARRIES ITS TOTAL…".

## Back faces: pairs behind their source

- **The ray mask** (#526): for volume receivers (`receiver_facet is None`), `RayCastOcclusion` does not
  cast a pair whose receiver lies certainly behind a `dark_behind` facet (`Profile.dark_behind`,
  `Surfaces.dark_behind`, `BackFaces.behind`; an undecidable height is cast). Such pairs are recorded
  clear (`Visibility.clear_behind`), and a gather through that mask refuses a set whose areal profiles are
  not all dark behind (`_refuse_light_from_behind`). Receivers on facets are never cut by facing (the
  transfer's weights do not vanish behind a source); the transfer's mask is cut by what it reads (#529, above).
  `SelfOcclusion.prepared(surfaces)` does surface-only work once per stream; `streamed_fluence_rate`
  refuses points inside once and builds passes through `visibility._unchecked_visibility`. ⚠️ A test
  counting mask builds patches `gather._unchecked_visibility`.
- **The bodies' layer**: `BackFaces.of(surfaces)` is passed as `facing=` to `BodyCulling.blocked` when
  `receiver_facet is None and surfaces.dark_behind`; both strategies record such pairs clear, and
  `ShaftCulling` drops tiles `BackFaces.tiles_behind` proves wholly behind. Exact, conservative box test
  (`_BOX_SLACK` = 3 × `clipping._SLACK`), one Numba loop (`_boxes_behind`).
- **The gather** (exact up to summation order): `lit_blocks.lit_segments` orders points along the Morton
  curve in blocks of `BLOCK = 8`, lists per block the facets `BackFaces.lit_facets` does not prove it lies
  behind, and pads segments to `rounded_width`; without planes or with traced points a block is one shared
  row (`LitSegment.shared`). A group is laid out by its planes only when every set's profile there is
  `dark_behind`. The streamed path's programs (`_point_part`, `_segment_part`) are module-level `jit`s
  cached by `_Labels`. ⚠️ Three traps that each make the skip save nothing: `jnp.asarray` inside a trace
  (lay out from the points as given), per-call recompilation, and `lax.dynamic_slice` / `jnp.take`'s
  default `fill` mode (`work._slice` is a clipped `take`).

## Morton ordering

`aquaflux/morton.py::morton_order` is the only ordering of points in the package (shaft culling's groups,
the gather's chunks and `lit_blocks`, the transfer's row blocks, `FacetClusters`), 10 bits, **each axis
scaled to its own extent**. Cubic cells were measured and rejected (#574: groups wrap round a long lamp
and stop facing one way). **Do not retry cubic cells for elongated lamps**; grouping by facing direction is
#589. Record: log → "ONE MORTON ORDERING, PER-AXIS CELLS…".

## SHAFT CULLING: `ShaftCulling`, the default (#554)

`culling.py`: `BodyCulling.blocked(bodies, sources, near, receivers, pair_limit, *, facing=None)`, with
`EveryPair` (the reference) and **`ShaftCulling(receiver_blocks=(32, 8, 2), source_clusters=(32, 8, 2))`,
the default** (an unset `body_culling` resolves through `culling_or_default` for a built mask, a streamed
chunk and both of a model's masks). Under a trace `ShaftCulling` hands off to `EveryPair`.

- Receivers and facet centroids are grouped along the Morton curve at each ladder size (each divides the
  one before), each group summarized by the column-wise max of `body.clearance`; a tile is certified clear
  where `body.vouches(...)` says so, a refused tile splits at the next level, and the finest refused tiles
  are answered by the same `body.blocks` the reference uses, so the two agree bit for bit.
- `BodyCulling.prepared(bodies, sources)` forms the facets' side once (`_Groups`, refusing other inputs by
  value). A traceable body is asked on batches padded to a power of two (`_vouches`, `Body.vouches_tiles`);
  a host-answered body gets no padding and no pair behind its source.
- Only "clear" certificates exist; "fully hidden" (phase C) and level of detail (phase D) are not built.
  ⚠️ The facet-to-facet ray mask is **not** culled (its shafts start on the wall they test).
- The 3-level ladder is chosen on absolute time (its third level saves ~2.7x on a triangle walk and costs
  ~1.24x on an analytic-only mask); a scene of analytic bodies alone may pass `(32, 8)` / `(32, 8)`. A
  per-body depth was investigated and **not built** (user, 2026-09-26). Refuted, do not retry without a new
  reason: testing a flat pair list at the finest level; the tile index innermost.

Records: log → "SHAFT CULLING: BUILT as `ShaftCulling`…" and "PHASE B (#554, 2026-09-26)…".

## Triangle bodies and the grid walk

**`TriangleBody`** (`triangle_body.py`, #510; in radiation beside the grid because `solids/` imports
nothing outside itself): `build(vertices, *, sheet=None, resolution=None, clearance_resolution=None,
tolerance=None)`; `blocks` is the grid walk (`traceable=False`); `contains` is decided per connected piece
(open: no inside; closed: solid where `enclosure_winding(closed pieces) + n_inward > 0.5`); `sheet=True`
forces no inside, `sheet=False` refuses an open piece, a closed piece enclosing no volume is refused. It
vouches through its own occupancy grid (`TriangleBody.occupancy`, `_CLEARANCE_REFINEMENT = 4` × a
near-cubic grid, `holds_any`; certificate one Numba loop, `grid._unions_held`), sized from the near-cubic
grid, not the walk grid.

**`TriangleGrid`** (`grid.py`): triangles registered in the voxels their (rounding-widened) boxes span;
segments walked by 3D-DDA (Amanatides & Woo) to the first blocker. Selected by `RayCastOcclusion(grid=True
| int | triple | TriangleGrid)`; **off by default** (changes cost, not answers).

- ⚠️ **A traced walk is worse than brute force** (static trip count and voxel occupancy make every ray
  pay the worst case), so the walk is host code — and **compiled** host code:
  `grid_walk.walk_to_first_hit`, a `numba.njit(parallel=True)` loop, one ray per `prange` iteration, the
  first axis winning a tie, capped at `sum(resolution) + 3` steps, each ray's frame (`_ray_frame`) formed
  once. `blocks` takes no `work_limit`. Numba is a core dependency (project owner's decision). ⚠️ The first
  call of a process compiles the loop (~2 s); not cached to disk. `NUMBA_NUM_THREADS=1` is set beside
  `OMP_NUM_THREADS` in `tools/fastgate.sh`'s parallel tier and CI.
- **Default resolution (`grid._resolution`)**: equal voxel counts per axis, each voxel in the proportions of
  the triangles' bounding box (edges capped at `_MAX_ASPECT = 32` to one another), `_VOXEL_BUDGET = 4` × the
  voxels of the near-cubic grid at `_TARGET_PER_VOXEL` (`_near_cubic`, sized from the triangles' area). An
  axis under one voxel gets one. Reason: steps along an axis go as the segment's travel over the edge, and
  a mask's segments spread like the box. ⚠️ It can mislead where the box is not the segments' (a flat floor
  lit from above, an off-axis vessel); `resolution=` overrides.
- Flat sets and sheets on a voxel plane: a flat axis is `_FLAT_THICKNESS = 1e-6` of the widest extent
  thick; registration widens each box by `_rounding_margin` (one helper for padding, registration and the
  box test); `_enters_grid` refuses a segment with a zero direction component outside the box's slab.
  ⚠️ A segment ending exactly in a sheet, or crossing its open rim, is a knife edge where compiled
  `segment_is_cut` and Numba `_cuts` round differently: compare against a one-voxel grid to separate voxel
  selection from kernel rounding. ⚠️ A plane at a large coordinate hides the flat defect.
- Primitives remain the path for a real reactor: on Sozzi the triangulated wall costs ~10x the analytic
  cylinders (~25 % of pairs certified against 81 %); the lever is a certificate covering more of the chamber
  (#568) or a cheaper walk. Empty-space skipping is not built (walking from the nearer end is worth
  nothing; the cost is wall triangles tested in narrow passages). ⚠️ Quote whole-field figures, never a
  pipe-heavy rate multiplied by the mesh. ⚠️ A suite whose fixtures are one shape of input repeated cannot
  tell a correct implementation from broken ones (three grid defects survived exactly that).

Records: log → "GRID ACCELERATION…", "THE COMPILED WALK…", "THE TRIANGULATED WALL AT MESH SCALE…", "THE
WALK'S COST…", "A FLAT SURFACE AND A SHEET ON A VOXEL PLANE…", "PHASE B…".

## Emitters: how many facets, and the mesh's own patch

The emitter's facet count sets the price of everything; error falls about in proportion to it, and the
near-lamp band (< 5 mm) sets the requirement. The CAD route (`CadModel.triangles`, `chord=1e-4`,
`facet_size=2.5e-3`) is the cheap way to accuracy on Sozzi; the mesh's own patch raw is the expensive one,
so it is coarsened. Record: log → "HOW MANY FACETS AN EMITTER NEEDS…".

`aquaflux.mesh.patch_triangles(mesh, geometry, names)` gives the exact triangles (`mesh.md`); `coarsen.py`
makes them affordable. Agreed with the user (#492): **collapse only** (every surviving vertex is an input
vertex), **each body's power held** (exitance scaled by area before over after, `coarsen_surfaces`), both
steps as separate classes. Bounds: `max_edge` (a strict longest-edge bound — not `CadModel.triangles`'
`facet_size`, a different quantity; do not unify the keywords), `chord` (per input vertex against the
rewritten patch and its unchanged ring), `angle` (default 0.5 rad; also the crease threshold). Feature
lines slide only along themselves and are pinned where they turn by more than `angle`. Collapses are
batched (`_local_minima` within two edges). On Sozzi: size along the lamp, keep the chord at 1e-4. Record:
log → "THE MESH'S OWN PATCH AS THE EMITTER…".

## ANALYTIC OCCLUSION: `SilhouetteOcclusion`

The clip cuts the source's angular extent by each blocker's silhouette and takes the covered share in
closed form (Nishita & Nakamae 1983; Baum, Rushmeier & Winget 1989). **Selectable beside the ray mask, not
a replacement**: neither dominates (the clip over-counts overlaps and costs more).

What makes it a traced program: the contour form is signed and additive over loops
(`_signed_loop_solid_angle`, magnitude only at the end), so visible = whole − covered; the covered region is
convex, so its vertex count is static (compaction by rank; widths `4…8`; the blocker is first cut to the
near side of the source's plane); it works in direction space; and every sign read is decidable. A tiling
sums exactly, and **only front-facing triangles are summed** (`facing`) — exact on a closed consistently
wound surface.

- **Volume receivers take the share of the plain solid angle** (#479); facet receivers the projected one.
  `receiver_normal=None` means a volume point all the way down (`source_view`, `covered_by`,
  `covered_fraction`, `may_occlude`); `silhouette._measure` picks the integral for whole, covered part and
  extent clamp together. A volume point pads the source with a repeated corner instead of clipping; its
  unprojected integral is `solid_angle._signed_loop_area`.
- `build_visibility(..., receiver_normal=(n, 3))` says which way a receiver faces, independent of
  `receiver_facet`; it reaches every `SelfOcclusion.field(surfaces, points, near, receiver_facet,
  receiver_normal)` (only the silhouette reads it), is normalized, refused on a bad shape or zero /
  non-finite length, and wins over the facet's normal. ⚠️ "On no facet" is not "no normal".
  `_receiver_frames` accepts rows naming several facets in one plane (a face and its twin), facing the
  first's way; other multi-facet rows need `receiver_normal`. `clear_behind` / `BackFaces` still key on
  `receiver_facet is None`, deliberately.
- **Where it over-counts**: only where two front-facing silhouettes overlap in angle (each is clipped
  against the source, not against what is still unblocked); it errs dark. **Decided: documented, not
  fixed** (#477 holds the detector and correction); measured rare on a two-lamp reactor. `overlapping`
  means only "more than one blocker contributed" and flags nearly every hidden pair on a meshed body.
- Six degenerate-geometry guards, each pinned: the depth cut, the cull's source-plane offset
  (`decidable_heights(..., through=)`), the extent clamp, `angular_cone`'s `unusable`, the `in_front` /
  `edge_on` guards, `_EXTENT_FLOOR = 1e-12` sr. ⚠️ A binomial error bar is zero at a share of exactly 0 or
  1: use Agresti–Coull.
- **Sign tests** (`clipping.py`, Shewchuk-style filters, `_SLACK = 16` units of roundoff): a height zero in
  exact arithmetic computes as noise whose sign depends on XLA fusion. `decidable_heights` snaps an
  undecidable height to exactly zero (correct for a clip; no exact stage needed); `spanning_plane` snaps a
  plane through two indistinguishable directions to zero. Both are needed. ⚠️ Chunk invariance alone is the
  wrong acceptance test, and a fixture too small to contain the degeneracy says nothing.
  `validation/radiation_analytic_occlusion.py` still carries the prototype clip.
- **Pipeline** (#527, #555): `silhouette.covers_nothing(view, to_source, to_blocker)` rejects pairs the clip
  would zero; source views once per receiver; pairs packed across receivers (`_PairPipeline`) into
  `work_chunk = 32,768` chunks; reject and clip on `threads = 4` Python threads (**the thread count changes
  no bit**); the cull is bounded by `pair_limit`. `FacetClusters.build` groups facets into Morton runs of
  `cluster_size = 32` with bounding spheres; `_cluster_pairs` rejects by enclosing cones and source planes,
  `_member_pairs` tests survivors; the kept set equals the dense cull's. Refuted, not retried: other cluster
  sizes, on-device compaction, a second cluster level (needs levels fused into one compiled call).
  Merging coplanar triangles into convex patches is **not built** (#474: ~3–7 % realistic saving).
- ⚠️ A ratio inherits every defect in its denominator: time whole builds, both arms, at the size the answer
  is for. The silhouette build's cost has not been re-measured since #527/#555.

Records: log → the ANALYTIC OCCLUSION subsections ("VOLUME RECEIVERS…", "MEASURED (#472)…", "IT IS
EXACT…", "THE SIGN TESTS…", "Cost, MEASURED…", "THE CLIP SEES ONLY PAIRS…", "THE CULL RUNS ON CLUSTERS
FIRST…", "MEASURED AND DECIDED (#474)…").

## The winding number: "is this cell inside the metal"

`enclosure_winding(vertices, points)` sums signed solid angles over `4π`: `±1` inside a closed consistently
wound surface, `0` outside; `check_points_outside` tests the magnitude against 0.5. Chosen over ray parity
(needs a closed surface; returns a clean wrong bit on an open one) and nearest-triangle signed distance
(unreliable near creases). Accurate to ~2e-14 (`validation/radiation_enclosure_winding.py`).

- ⚠️ An open surface reads ~0.45 just off a bare disc; `check_points_outside` **warns** when nothing is
  enclosed but the largest winding exceeds 0.01, rather than raising.
- ⚠️ An inconsistently wound closed surface reads as an open one (values between 0 and 1, varying by
  point). Run `check_winding` first.
- Cost `n_points × n_facets`, so `build_radiation_model` does not call it (analytic bodies refuse interior
  points at the visibility build, cheaply). One `jax.jit` program per pass (`_summed_signed_solid_angle`);
  a 2-cycle piece (`_closed_pieces`, stricter than `open_facets`' "closed") is summed only inside its own
  box; `tolerance=` sets the vertex merge. ⚠️ A sheet welded into a closed body is summed everywhere (its
  halves read closed to `open_facets` but bound nothing alone). Record: log → "THE WINDING CHECK IS
  COMPILED…".

## The two engineering-unit conversions (`units.py`)

- **`absorption_from_uvt(uvt)`**: UVT (%, through 1 cm) to a **napierian coefficient per metre** (95 % UVT
  = 5.129 /m; decadic is smaller by `ln 10`, per-cm by 100). ⚠️ A fraction passed as a percentage is
  refused, checked over the whole array (`0.95` is a legal percentage and would give a dark field); the
  cost is that sub-1 % UVT cannot be expressed.
- **`lamp_exitance(surfaces, {"lamp": watts})`** divides by the **triangulation's** area
  (`Surfaces.area_by_solid`, the one home of per-body totals), so `Σ M A` equals the rating at any
  refinement; an analytic `π d L` undershoots by the inscribed-area deficit (2.6 % at eight sectors).
  ⚠️ Which body carries the rating (lamp envelope or sleeve) is a modelling decision.

## Validation against discrete ordinates (Sozzi reactor)

On the Sozzi reactor (1,635,909 cells) the whole-reactor volume-mean `G` agrees with of-optical-radiation's
DOM-256 to 0.09 %; DOM converges on aquaflux as directions increase; aquaflux's own error near the lamp is
median 0.44 %. `Outside(chamber, inlet, riser)` of three `Cylinder`s reproduces the hand-derived occluder
exactly. Instrument `validation/sozzi_radiation/`. ⚠️ Check a region's extent against the mesh's, not a
comment. Records: log → "MEASURED: aquaflux against discrete ordinates…" and "MEASURED: the Sozzi reactor
as three cylinders…".

## Documentation

The package is in `docs/conf.py`'s `PUBLIC_SUBPACKAGES` with a `SUBPACKAGE_GROUPS` entry (`model` first);
its whole `__all__` is published, reviewed and kept entire (including both kernels and their warning).

**UV disinfection of air is in scope as well as water** (project owner, 2026-09-29): write "the medium" /
"the fluid", naming water only where the content is water's (the UVT conversion, Bolton's quartz-sleeve
figures, Sozzi & Taghipour 2006). No `absorption` is the non-absorbing medium.

⚠️ **The user guide `docs/radiation.md` restates defaults and measurements by value**, so a change to any
of these must update it in the same change: six-point `receiver_quadrature`, `refine_for_receivers`'
`max_ratio=0.25`, GMRES global relative `1e-10` and restart 120, `RayCastOcclusion` and `ShaftCulling` as
defaults, opaque as the transmittance default, point sources Isotropic-only, the 2.6 %-at-eight-sectors
undershoot, 95 % UVT = 5.129 /m, and the Sozzi figures (0.09 % volume mean; 0.36–1.65 / 0.886–1.076 at
64 / 256 directions on non-uniform 8 × 4 / 16 × 8 grids; 0.44 % median near the lamp; dose mean 0.3 %, log
reduction within 1.1 %, DOM-64 short by 23 % at k = 0.5). Its quick start (`G = 80` in a 6×6×6 box at
`M = 10, ρ = 0.5`) and its **Theory** section (every equation the code evaluates) were checked against the
code; a change to those formulas must update it. Record: log → `[rule text] Documentation`.

## THE SCENE: LAMPS KEPT OUT OF THE TRANSFER, AS A LIBRARY FUNCTION

`solve_scene(Scene(lamps, reflectors, occluders, absorption, volume, surfaces, lamp_samples, settings))`
→ `SceneSolution`. It is what a radiation **case file** builds (`.claude/rules/case.md` → Radiation cases).

- **Steps**: `check_profiles(lamps)`; with anything to exchange, the lamps' `direct_irradiance` at
  `subtriangle_centroids(vertices, lamp_samples)` (`k²` equal-area sub-triangle centroids,
  `DEFAULT_LAMP_SAMPLES = 4`), averaged per facet, is the `external_irradiance` of one
  `surface_irradiance` solve on a model with **zero receivers** (`np.zeros((0, 3))`); radiosity is
  `ρH` (reflectors may not emit), re-gathered as a Lambertian set. Volume: `streamed_fluence_rate` when
  anything can shadow, else `summed_fluence_rate`. Surfaces: `direct_irradiance` through a mask built per
  pass of `receivers_per_pass(pair_limit, n_facets × n_bodies)` points. `medium_absorbed_power = Σ a(x) G V`
  through `Absorption.sample(position)`.
- **Lamps reflect and shadow in the exchange** (#604 step 1): the exchange (`_Exchange.of(scene)`) is the
  reflectors' facets then every areal lamp facet, with no emission (lamp reflectance default 0, case key
  `Lamp.reflectance`). `SceneSolution` has `lamp_irradiance`, `lamp_absorbed_power`,
  `reflector_absorbed_power` (`Σ (1−ρ) H A`) and `irradiance_absorbed` (`(1−ρ) E` per `SurfaceReceivers`
  set), so the books close from the solution alone. `Scene` refuses lamps and reflectors sharing a body
  name, and any specular reflectance. ⚠️ `Box(axes=)` takes the axes as rows. ⚠️ A drum in a box is not
  convex: do not run it unshadowed.
- **Every triangle of the scene shadows the direct light too** (#622; chosen with the user over documenting
  it or refusing an uncovered reflecting wall). The direct light is gathered from
  `_Direct.of(scene).surfaces`: the reflectors' facets dark (a Lambertian profile appended), then every lamp
  facet with its optics, point sources kept (`_joined(scene, lamp_facets, emitting=True)`; the exchange is
  `_joined(..., emitting=False)` over areal lamp facets). ⚠️ A shadow geometry separate from the sources is
  not expressible: a mask's columns are the sources' facets (`streamed_fluence_rate` requires `sets` to
  share `shadow_geometry`), so reflectors join the sources. With no reflectors, or self-occlusion
  `NoOcclusion`, the lamps are gathered alone (dark facets would add only zeros). The direct pass is then
  `n_points × (n_lamp + n_reflector)` facets — not measured at mesh scale.
- **Own-facet exclusion** (the gather's `F_ii = 0`): a point on a reflecting surface is otherwise lit by its
  own facet (corner heights round either way off an axis-aligned plane, and the kernel returns the
  hemisphere). Excluded as sources via `direct_irradiance(..., receiver_facet=)` (`(n_points,)` or
  `(n_points, k)`, `-1` padding); offsetting points in front was rejected (it couples a scene-scale constant
  to the clip's tolerance). `SurfaceReceivers.reflector` names the body and `_own_facets(direct.surfaces,
  body, points)` passes **every** facet of it the point lies on (a face centre is the apex of all its fan
  triangles). Exchange samples name their facet by `_Direct.facets_of(exchange)` (reflector `j` → `j`, lamp
  facet `k` → `split + k`); a part whose facets are all `-1` passes `own=None`, keeping the culls that key
  on `receiver_facet is None`. Lamp samples stay in their facet's plane, never behind it. ⚠️ Only a second
  lamp exposes the lamp samples' own-facet exclusion. ⚠️ A `SurfaceReceivers` set on a reflector that does
  not name its body has both parts shadowed by its own facet under the ray test.
- **Gates**: `test_a_lamp_kept_out_of_the_transfer_lights_the_box_as_the_model_does_with_it_inside`
  (1e-11, one receiver point per facet, one lamp sample, reflected share > 30 %);
  `test_a_reflector_between_a_lamp_and_a_point_shadows_the_lamp_s_direct_light`;
  `test_a_point_source_under_a_reflector_is_shadowed_by_it_and_lights_the_rest_as_a_point`;
  `test_a_reflector_sends_its_light_out_of_its_front_in_the_solve_as_the_gathers_do`. Mesh-scale agreement:
  `validation/ray_effects_room/README.md` (those cases run `NoOcclusion`).

Records: log → "THE SCENE: LAMPS KEPT OUT OF THE TRANSFER…", "#604 STEP 1…", and `[rule text] THE SCENE…`
(the #622 and #667 measurements and mutation passes).

### #604: sequencing, and transparent solids (step 2)

#604 is sequenced by capability: (1) emitters that also reflect (done); (2) transparent solids; (3) the
equivalent surface as a baking step; (4) curved specular reflectors (and reflected refraction branches);
(5) path cuts.

- **Step 2a (#638)**: a transparent solid is a convex analytic body with an index and an absorption
  (`refraction.Transparent`, nested in `Media`; `Media.region_of_facets` is the one home of a facet's
  medium). A path crosses the surface of every region holding exactly one of its ends, once, in a fixed
  order, solved by Fermat's principle and differentiated implicitly (`solve_paths` → `Paths`, compiled
  once per chain and shape). `refracted_fluence_rate` / `refracted_irradiance` weigh each triangle by the
  solid angle on its corners' arrival directions times the corners' mean of radiance × Fresnel × leg
  absorption × `(n_receiver/n_source)²`; `build_refracted_visibility` → `RefractedVisibility` shadows every
  leg of the centroid's path (`receiver_facet=` excluded on the last leg). ⚠️ The solve is a **descent** on
  the optical length (Newton from the straight line runs away near an interface); a dropped path is the
  solver's fault until a forward trace or quadrature says otherwise. ⚠️ A `while_loop` inside a
  `while_loop` under `vmap` never ends (the line search is a fixed 41-step ladder). ⚠️ `Chain.beside` must
  leave out the crossed regions.
- **Step 2b-i**: `Scene.media` routes each point's light by medium (`scene._by_medium`): same-medium sources
  along straight lines in that medium's absorption, others along refracted paths (`scene._refracted`).
  With media, `Scene.absorption` must be unset, and a scene that exchanges is refused until 2b-ii (the
  refracted transfer and the model's `media=`). There is no `Visibility.through` and no
  `refraction.straight_through`.
- **Bent routes through a neighbour** (agreed with the user: one pass-through region per path; same-medium
  pairs bent too; built, not yet measured): `Chain.between(..., through=node)` / `Chain.routes` give a
  route per passable node; a path meeting a region it does not pass through is invalid. A passing route is
  solved from four aims (`_starts`), deduplicated. Paths round a sleeve's wall alone are **saddles**, which
  the descent cannot find, so such a route is **shot** (`shooting.py`, `refraction._shot`: impact parameter
  and `axial_index = n u_z`, a grid scan sized by `RadiationSettings.shooting_grid`, then Newton polish,
  continued to corners the grid missed); an unshootable wall route is skipped (`_only_saddles`). A shot
  route's triangles are cut to the part with a path (`refracted._clipped_field`, `_reached_triangles`,
  `_cut_edges`); mask and gathers share one `_Reach` (`RefractedVisibility.reach`); `check_shooting_grid` →
  `ShootingGridCheck` diagnoses a short grid. Gap routes stay descended. A sound geometric cull for coaxial
  cylinders exists on paper and is not built. ⚠️ Call `n u_z` `axial_index`, never `beta` (the vocabulary
  guard).

Records: log → "#604 STEP 2a…", "#604 STEP 2b-i…", "#604 STEP 2b-i, CONTINUED…".

## TWO-SIDED SHEETS: a declared sheet is two faces in the solve (#689)

**The declaration** is `Surfaces.from_triangles(..., two_sided=(names,))` — on the surface set, static,
because it is geometry (decided by the user, 2026-09-21, over topology-driven and refusing open surfaces;
placement on `Surfaces` decided with #689). Topology is wrong both ways on real input: a sheet welded all
round has no free edge (`checks.open_facets` reads it closed, pinned by
`test_a_sheet_welded_in_all_the_way_round_reads_as_closed`), and a duct exported without end caps reads as a
sheet. So topology drives only a **warning** (an undeclared open piece is named at build); a misspelt name
**raises**. Naming a closed body two-sided makes it block twice and errs dark, undetectably.

- **Blocking**: the silhouette clip counts only facing blockers, so an undeclared one-sided sheet hid
  nothing from behind (Walton's obstructed pair read unobstructed). Declared facets count from both sides
  (`surfaces.is_sheet & (surfaces.twin < 0)`; on an expanded set each face counts from its own side). The
  ray test is two-sided anyway.
- **Reflecting**: `Surfaces.with_back_faces()` appends each areal facet of a declared body with two corners
  swapped; centroid, area and negated normal are **carried over, not recomputed**, so a face and its back
  are one triangle exactly. `back_of[k]` is the front of the `k`-th back face; `twin` maps both ways (`-1`
  elsewhere). Idempotent; point sources get none. **A sheet may not emit** (refused there, and `Scene` asks
  it of the lamps): the direct gathers emit from front faces only. `_with_backs` is the unchecked builder
  for a solved set whose emission is a radiosity. `with_geometry` on an expanded set moves the fronts and
  remakes the backs; `refine_for_receivers` and `coarsen_surfaces` carry the declaration.
- **Where it is expanded**: `build_radiation_model`, each model entry point, and `_Exchange.of` in the
  scene. `build_transfer` does **not** expand (it builds the set it is given). Facet-indexed results are
  `n_facets + len(back_of)` long; the scene splits back faces to their front's set
  (`SceneSolution.reflector_back_of`, `lamp_back_of`).
- **Two defects fixed by index, not tolerance**: (a) a ray aimed at a face's centroid ends in its twin, so
  the receiving end excludes the twin too (`Surfaces.own_faces`, `(n, 2)` rows `[i, twin(i)]`, the
  facet mask's and the mirror masks' `receiver_facet`, and the ray test's per-pair exclusion; mirror
  legs exclude a source's twin; the sending end needs nothing — the twin is inside the margin). (b) the
  six-point rule puts receiving points one ulp off the plane and `point_heights` does not snap a rounding
  of the point's construction, so `_row_block` zeroes `F[i, twin(i)]` as it zeroes `F[i, i]`.
- Measured against a closed slab 1e-5 thick (`tests/unit/test_radiation_two_sided.py`): the silhouette
  clip agrees to 6e-6, the ray test to 1e-3 (the slab's own difference). ⚠️ A sheet on the box's own grid
  made rays run exactly along its rim — a fixture degeneracy, not a defect. ⚠️ Own-facet exclusion on a
  back face is only visible under the ray test
  (`test_the_lamps_light_on_a_back_face_is_not_shadowed_by_its_own_triangle`).

Records: log → `[rule text] TWO-SIDED SHEETS…` and `[rule text] Four closed-form checks added after the
fact…` (Walton's pair, the Bickley line, the emission transfer, adjoint conservation).
