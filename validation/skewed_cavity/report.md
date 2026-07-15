# Skewed lid-driven cavity: aquaflux vs OpenFOAM

Inclined (parallelogram) lid-driven cavity, the Demirdzic-Lilek-Peric (1992)
non-orthogonal benchmark. **Both codes solve on the identical mesh**: OpenFOAM's
`blockMesh` builds it, `foamRun` solves on it, and aquaflux reads that same ASCII
`polyMesh` back in via `read_openfoam`.

## Setup

| | |
|---|---|
| Geometry | unit parallelogram, walls inclined at **beta = 45 deg** |
| Mesh | 40 x 40 = **1600 cells**, uniform |
| Mesh non-orthogonality (checkMesh) | **45.0 deg** (max = avg) |
| Reynolds number | Re = U L / nu = 1 x 1 / 0.01 = **100** (laminar) |
| Lid | top wall, U = (1, 0) |
| Discretisation (both) | first-order upwind advection; non-orthogonal *corrected* Laplacian |
| aquaflux linearisation | corrected Green-Gauss gradient folded into the residual; **Jacobian by AD** |
| OpenFOAM linearisation | segregated SIMPLE; non-orthogonal correction *deferred* (`nNonOrthogonalCorrectors`) |

## Result 1 -- correctness (same mesh, fields agree in the bulk)

The two independently-converged fields agree well through the interior; the
remaining difference is concentrated at the singular top lid corners (where the
moving lid meets a fixed wall and the velocity is discontinuous -- see the error
map in `figures/comparison.png`, black everywhere except the corner):

| metric | value |
|---|---|
| RMS \|U_aqua - U_OF\| / lid speed | 0.30 % |
| max \|U_aqua - U_OF\| / lid speed (**at the singular lid corner**) | 6.5 % |
| RMS \|p_aqua - p_OF\| (mean-removed) | 4.777e-03 |
| max \|p_aqua - p_OF\| (**at the singular lid corner**) | 1.473e-01 |
| centroid-mapping mismatch | 5.0e-12 (confirms the identical mesh) |

The ~1-2% bulk difference is the expected consequence of two independent solvers
(coupled Newton vs segregated SIMPLE; different Rhie-Chow and pressure handling)
with a *first-order upwind* convection term, whose numerical diffusion is
formulation-sensitive. It is not a discretisation-order mismatch: the scatter of
`U_x` (aquaflux vs OpenFOAM) lies on the diagonal, and the corner singularity --
not the mesh non-orthogonality -- sets the maximum.

## Result 2 -- convergence / the linearisation thesis

| solver | treatment of the non-orthogonal correction | nonlinear iterations to \|R\| < 1e-8 |
|---|---|---|
| **OpenFOAM SIMPLE, `nNonOrthogonalCorrectors 0`** | correction absent | **DIVERGES** (nan) |
| OpenFOAM SIMPLE, `nNonOrthogonalCorrectors 1` | 1 deferred sweep / outer iter | 212 outer iterations |
| OpenFOAM SIMPLE, `nNonOrthogonalCorrectors 2` | 2 deferred sweeps / outer iter | 212 outer iterations |
| **aquaflux, corrected gradient in the AD Jacobian** | fully implicit | **5 Newton steps** |
| aquaflux, gradient lagged (`stop_gradient`) | deferred correction | 28 steps |

Reading the table:

* **The uncorrected segregated solve diverges** on a 45-deg mesh -- the correction
  is not optional here.
* **aquaflux converges quadratically in 5 Newton steps** because the
  corrected-gradient reconstruction is inside the residual and AD places its full
  linearisation in the coupled Jacobian. Residual history:
  1.8e-01, 9.4e-03, 7.0e-04, 7.6e-06, 7.1e-09, 6.4e-11.
* **The same aquaflux solver, with the gradient hidden from the Jacobian**
  (`stop_gradient`, i.e. deferred correction), converges only linearly -- reproducing the
  classical lagged behaviour and isolating the linearisation as the cause,
  controlling for mesh, discretisation, and solver.

See `figures/comparison.png`.

*(aquaflux wall time: implicit 45s, lagged 183s, incl. JIT warm-up.)*
