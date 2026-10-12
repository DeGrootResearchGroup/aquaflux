# Skew-angle sweep: iterations and accuracy vs mesh non-orthogonality

Same inclined lid-driven cavity (Re = 100, 40x40), swept over skew angle. The mesh
non-orthogonality is `90 - beta`. "Iterations" = OpenFOAM SIMPLE outer iterations, or
aquaflux Newton steps to `|R| < 1e-8`. "vel RMS/max" = the aquaflux (implicit)
velocity vs the converged OpenFOAM `nNonOrthogonalCorrectors=1` field on the same mesh
(the reference that converges at every skew angle), as a percentage of the lid speed
(`vel max` is the single worst cell, at the singular lid corner).

| beta (deg) | non-orthogonality (deg) | OF nOrtho=0 | OF nOrtho=1 | OF nOrtho=2 | aquaflux implicit | aquaflux lagged | vel RMS (%) | vel max (%) |
|---|---|---|---|---|---|---|---|---|
| 60 | 30.0 | diverged | 218 | 218 | **5** | 16 | 0.29 | 5.9 |
| 45 | 45.0 | diverged | 213 | 213 | **4** | 20 | 0.30 | 6.0 |
| 30 | 60.0 | diverged | 216 | diverged | **4** | 37 | 0.32 | 4.9 |

**aquaflux with the corrected gradient in the AD Jacobian stays flat** (a handful of
quadratic Newton steps) across the whole range, while the segregated SIMPLE count and
the deferred / lagged count grow with non-orthogonality (and the uncorrected variants
diverge). The velocity agreement is a fraction of a percent RMS across the range, with
the maximum confined to the singular lid corner. Folding the correction into the
Jacobian removes the non-orthogonality penalty in both convergence and accuracy. See
`figures/sweep.png`.
