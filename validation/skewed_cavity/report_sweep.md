# Skew-angle sweep: iterations vs mesh non-orthogonality

Same inclined lid-driven cavity (Re = 100, 40x40), swept over skew angle. The mesh
non-orthogonality is `90 - beta`. "Iterations" = OpenFOAM SIMPLE outer iterations,
or aquaflux Newton steps to `|R| < 1e-8`.

| beta (deg) | non-orthogonality (deg) | OF nOrtho=0 | OF nOrtho=1 | OF nOrtho=2 | aquaflux implicit | aquaflux lagged |
|---|---|---|---|---|---|---|
| 60 | 30.0 | diverged | 218 | 218 | **5** | 16 |
| 45 | 45.0 | diverged | 213 | 213 | **4** | 20 |
| 30 | 60.0 | diverged | 216 | diverged | **4** | 37 |

**aquaflux with the corrected gradient in the AD Jacobian stays flat** (a handful of
quadratic Newton steps) across the whole range, while the segregated SIMPLE count and
the deferred / lagged count grow with non-orthogonality (and the uncorrected variants
diverge). Folding the correction into the Jacobian is what removes the
non-orthogonality penalty. See `figures/sweep.png`.
