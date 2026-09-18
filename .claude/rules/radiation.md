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
| STL parsing, `Surfaces` | Not yet built |
| the chunked gather, point and line sources | Not yet built |
| the radiosity system | Not yet built |
| Beer–Lambert optical depth, voxel-grid traversal | Not yet built |
| occlusion against the surface triangles | Not yet built |

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

## Documentation

The package is **deliberately absent from `docs/conf.py`'s `PUBLIC_SUBPACKAGES`.** Listing a
subpackage publishes the whole of its `__all__`, and the user-facing surface — `fluence_rate`,
`radiosity`, `surface_irradiance` — does not exist yet. Adding it now would put two internal
geometry helpers on the published site and force a re-cut later. Add it in the change that
introduces the public entry points, together with a `SUBPACKAGE_GROUPS` entry keyed on the
modules those names are *defined* in.
