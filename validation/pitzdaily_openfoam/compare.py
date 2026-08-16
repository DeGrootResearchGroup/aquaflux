"""pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST.

A **same-mesh, cell-for-cell** cross-code validation. The OpenFOAM pitzDailySteady tutorial (its RAS
model switched from the shipped kEpsilon to kOmegaSST) is run in the openfoam13 container by
``of_case/run_of.sh``, which writes the converged fields, the mesh, and the SIMPLE residual history to
``runs/kwsst/``. This script then reads that **same mesh** into aquaflux via ``read_openfoam`` and
solves the coupled RANS system on it, so the two solutions live on identical cells and are compared
directly (no interpolation between independent meshes, unlike ``validation/turbulent_channel_openfoam``,
whose cyclic mesh the reader cannot yet import).

aquaflux setup, as requested for this study:

* the **coupled** turbulent solver (:func:`aquaflux.turbulence.solve_coupled` -- one monolithic Newton
  on ``R(u, p, k, omega)``, globalized by pseudo-transient continuation);
* **hybrid initialization** (potential-flow velocity + Laplace-smoothed turbulence), which
  ``solve_coupled`` invokes automatically to self-start;
* **second-order upwind** momentum advection (:class:`aquaflux.discretization.LimitedUpwind` with the
  :class:`aquaflux.schemes.VenkatakrishnanLimiter` -- the upwind cell reconstructed to the face with
  its gradient, slope-limited so the reconstruction stays bounded). The stiff k/omega scalars use
  bounded first-order upwind: a second-order stencil there lets the coupled Newton step drive omega
  negative (a Newton-update, M-matrix effect the limiter does not prevent -- see ``solve_aquaflux``);
* **corrected Green-Gauss** gradients (:class:`aquaflux.schemes.CorrectedGreenGauss`, the
  skewness/non-orthogonality-corrected reconstruction -- the analogue of OpenFOAM's ``corrected``
  surface-normal / non-orthogonal treatment);
* **log-variable omega** (:class:`aquaflux.turbulence.LogScalars` on ``omega_transform``): ``omega =
  e^w`` stays strictly positive under any Newton step. Without it a direct-omega step drives omega
  negative once the recirculation forms, poisoning ``nu_t = k/omega`` while the residual stays finite
  (so the divergence guard never trips) -- the failure this case exposes and log-omega structurally
  removes.

The physics caveat this study documents: the pitzDaily mesh is a **wall-function** mesh (first-cell
``y+`` well above the viscous sublayer), whereas aquaflux's SST is **wall-resolving** (it fixes the
analytical sublayer ``omega`` at the wall-adjacent cell). The comparison therefore isolates the *outer*
flow -- the shear-layer development, the recirculation bubble, and the reattachment length -- where the
near-wall treatment matters least, and reports the near-wall fields as the expected point of departure.
The near-wall ``omega`` also differs because the two codes blend the viscous and log branches
differently: aquaflux uses ``sqrt(omega_vis**2 + omega_log**2)`` (the quadrature blend) while OpenFOAM's
default ``omegaWallFunction`` uses ``max(omega_vis, omega_log)`` -- a ~20% difference in the buffer layer
that is a blend-shape choice, not an error in either code.

**Reference caveat (binding -- do not skip):** the OpenFOAM *steady* (SIMPLE / ``ddtSchemes steadyState``)
run does **not** converge this case -- its ``omega`` field limit-cycles and *checkerboards* in the inlet
channel (adjacent cells oscillating between O(0.1) and O(1e8)), which is a non-physical, non-converged
field, not a valid solution. Comparing aquaflux's residual against such a field is meaningless (it will be
huge because the field is garbage, not because aquaflux is wrong). A stable steady solution *does* exist
and is recovered by a time-accurate transient (``pimpleFoam`` / an unsteady ``ddtSchemes``) run to a
statistically steady state; use a **transient-converged** OpenFOAM field as the comparison target, and
compare the outer-flow profiles (velocity, reattachment length) rather than the raw residual.

**Cost note (binding for whoever runs this):** the coupled log-omega solve on the full ~12k-cell mesh
is compute-heavy -- each Newton step is several minutes and the march is long, so a full run is a
matter of hours. omega-log is validated on a smaller channel (``tests/integration/test_coupled_rans``);
efficient large-mesh convergence (the reparametrized-block preconditioner scaling and the
globalization) is a known tuning follow-up. Track the **per-field relative** residuals when running --
the absolute ``||R||`` is dominated by omega's ~1e5 scale and is a misleading convergence metric.

Run (after ``run_of.sh``) from the repo root:
    python3 validation/pitzdaily_openfoam/compare.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so
# `python3 validation/pitzdaily_openfoam/compare.py` from the repo root cannot find `aquaflux` unless it
# is separately installed. Add the repo root explicitly so the documented invocation works against a
# plain checkout -- this case had no such bootstrap, so it could not be run through the case launcher
# at all, which is one reason it was left behind while the sibling case was developed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter
from aquaflux.solve import (
    CflResidualDualTimeControl,
    MarchLogger,
    RefreshPolicy,
    RetryPolicy,
    native_nodal_inverse,
    relative_residual_gmres,
)
from aquaflux.turbulence import (
    CoupledJacobianProbe,
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    solve_reynolds_continuation,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "kwsst"
# The comparison target is the TIME-ACCURATE run, not the steady one. The steady case does not
# converge on this geometry: it leaves an odd-even checkerboard in the inlet, with omega spanning
# 0.03 to 1.15e8 across adjacent cells. Ten of those cells alone carry the entire omega residual
# measured on that field, so anything calibrated against it is calibrated against numerical noise.
TRANSIENT = HERE / "of_transient" / "0.14"
FIGS = HERE / "figures"

# The pitzDaily operating point (0/ and constant/): U_in = 10 m/s, nu = 1e-5, k_in = 0.375,
# omega_in = 440.15. rho = 1 (incompressible kinematic).
RHO, NU = 1.0, 1e-5
U_IN, K_IN, OMEGA_IN = 10.0, 0.375, 440.15
WALLS = ["upperWall", "lowerWall"]
STEP_X, STEP_Y = 0.0, 0.0  # the step lip; the lower wall drops to y = -0.0254 for x > 0
# The coupled Newton march budget. This is a stiff, separating, high-Re case on a wall-function mesh
# (aquaflux's SST is wall-resolving), so it converges to an engineering tolerance rather than machine
# zero; the cap is generous so the march exits on the tolerance, not the count.
#: ⚠️ AN ABSOLUTE BAR, MATCHING THE SIBLING CASE: `rtol = 0` so the test is `|R| <= atol` outright.
#: A RELATIVE tolerance means each Reynolds rung targets a fraction of its OWN starting residual, so
#: the cheap anchor rung -- whose only job is to hand the next one a warm start -- is asked for a
#: harder solve than the target rung ever needs. Run briefly with `rtol = 1e-6` here, rung 1's target
#: came out at 8.7e-08 against the 1e-05 the sibling case asks of any rung.
MAX_STEPS = 150  # per continuation rung
RTOL, ATOL = 0.0, 1e-5

# ---------------------------------------------------------------------------------------------
# The march configuration.
#
# ⚠️ THIS CASE RAN FOR A LONG TIME ON A SINGLE-STEP PSEUDO-TRANSIENT MARCH WITH NO DUAL-TIME INNER
# LOOP, NO COURANT CONTROL, NO RETRY LADDER AND NO PER-STEP LOG. All of that was built and calibrated
# on the three-dimensional case and never carried back here, so this case could not benefit from any
# of it -- and, worse, a timing taken from it measured the globalization rather than whatever was
# being studied. Under the old configuration the cold march is a documented reachability crawl,
# needing on the order of eight hundred outer steps to develop the recirculation against a two
# hundred step cap: it could not converge however long it was left.
#
# The values below are the three-dimensional case's, because that is where each was measured. Two are
# load-bearing enough to name:
#
#   * `POSITIVITY_FLOOR` -- without it the step limiter's room is a purely RELATIVE quantity, so a
#     numerically dead cell ratchets the global step cap by a factor of a hundred per step until the
#     march is taking no step at all while every field still reads finite.
#   * `scaled_norm` -- the coupled Euclidean residual is very nearly all omega, so a march judged on
#     it cannot see the flow converge. The row-scaled measure judges every equation comparably.
# ---------------------------------------------------------------------------------------------

#: ⚠️ REYNOLDS CONTINUATION, BECAUSE A COLD SOLVE AT THE TARGET REYNOLDS NUMBER DOES NOT REACH THE
#: ROOT IN ANY REASONABLE NUMBER OF STEPS. Measured on this case without it: the march is perfectly
#: healthy -- full steps, line search never clipping -- and contracts a steady 2.7% per outer step,
#: which needs about 490 steps to reach the stopping tolerance against a 200-step cap. It is not a
#: solver in difficulty; it is a solver correctly integrating a long transient, which is precisely the
#: reachability problem continuation exists to short-circuit. Each rung starts from the previous one's
#: converged field, so the expensive target rung begins near its own root instead of at a cold start.
#:
#: `N_POINTS` is the number of INTERMEDIATE rungs: 2 gives Re/100, Re/10, target, matching the
#: three-dimensional case.
N_POINTS = int(os.environ.get("PITZ_N_POINTS", "2"))

#: The dual-time inner loop. `inner_tol` 1e-2 rather than a tighter value: measured on the
#: three-dimensional case, 1e-3 bought nothing over 1e-2 while costing a third of the march.
INNER_STEPS, INNER_TOL = 5, 1e-2

#: Buys the step limiter out of a numerically dead cell instead of letting one cell ratchet the
#: global step cap toward zero. ⚠️ Reachable only because this case uses `coupled_amg_continuation`:
#: it is a parameter of that builder ALONE, and the default, complete-LU and threshold-ILU builders
#: expose neither it nor the `step_limit` it would be set on.
K_POSITIVITY_FLOOR = 1e-8

#: The inexact-Newton stop per inner linear solve, in the row-scaled measure, and the Krylov restart.
#: `FORWARD_MAX_RESTARTS` bounds a single solve: past the retry threshold the attempt is going to be
#: discarded anyway, so running it to a stagnation is work thrown away. Strictly above the threshold,
#: because the march's test is `>`, and a cap landing exactly on it would accept a truncated direction
#: instead of escalating.
FORWARD_RTOL, FORWARD_RESTART = 0.3, 15
FORWARD_MAX_RESTARTS = 14

#: ⚠️ THE VALIDATED SMOOTHER BUNDLE, AND NONE OF IT IS OPTIONAL. These are the library defaults'
#: opposites, and each was measured on the sibling case at adjoint-grade tolerance:
#:   * ⚠️ `FILL_LEVELS` **1** HERE, WHERE THE SIBLING CASE USES 0 -- THE TWO RANK THIS OPPOSITELY, AND
#:     copying the sibling's value is what kept this case from taking a single step. At zero fill the
#:     level sweep is not a contraction on this leading block: it AMPLIFIES, and the four sweeps below
#:     compound it (one apply of the split reads 9.88e+05 at one sweep and 1.56e+31 at four). With fill
#:     1 the same operator takes ONE matvec to the march's stop.
#:     The discriminator is a pivot census, and it inverts between the cases: this block's ILU(0) has
#:     NEGATIVE pivots at every shift (27/25/9 of 36675) with min |pivot| some twenty times smaller
#:     than the sibling's, whose ILU(0) has none at any shift. There the fill produces the negative
#:     pivots and dropping it is the fix; here dropping it produces them and the fill is the fix.
#:   * `SWEEPS` 4 -- zero-fill is the weaker smoother, so extra sweeps pay more than they did for
#:     ILU(1) (390 -> 69 iterations at beta 0.01). The library default of 2 was tuned against ILU(1)
#:     and does not carry over.
#:   * `COARSE_EQ_LIMIT` 2000 -- the default coarsens to ~50 equations, whose direct solve captures
#:     only the crudest global mode, and the indefinite saddle's wall is exactly that global pressure
#:     coupling. `None` stalls at every low shift. Not optional.
#:   * `PC_BETA_FLOOR` 0.05 -- the V-cycle is built at `max(beta, floor)` while the march still solves
#:     at its own shift. The OPERATOR is untouched, so the converged root and the adjoint are
#:     unchanged, and the mismatch saturates instead of growing as the shift falls.
FILL_LEVELS, SWEEPS, COARSE_EQ_LIMIT, PC_BETA_FLOOR = 1, 4, 2000, 0.05

#: The field split: the `[u, v, p]` saddle and the `[k, omega]` transported pair get their own
#: hierarchies, because a saddle and an advection-diffusion-reaction pair coarsen differently. Measured
#: 31% faster end to end on the sibling case -- while taking MORE Krylov cycles, because two smaller
#: V-cycles plus one sparse coupling product apply far more cheaply than one six-field V-cycle.
FIELD_SPLIT = os.environ.get("PITZ_FIELD_SPLIT", "1") not in ("", "0")
TRAILING_SWEEPS = 1

#: Clip each cell's own correction rather than scaling the whole step by the worst cell. Off by
#: default and byte-identical off; it removes a failure mode rather than buying speed.
POSITIVITY_PROJECTION = os.environ.get("PITZ_K_POSITIVITY_PROJECTION", "") not in ("", "0")

#: ⚠️ THE WALL CONDITION ON `k`, AND IT IS A CHOICE OF PROBLEM RATHER THAN OF SOLVER. Turbulent
#: fluctuations vanish at a no-slip wall, so `k -> 0` and `Dirichlet(0)` is the textbook condition --
#: but it makes a DIFFERENT discrete problem from the zero-gradient one, with its own reattachment
#: length, so the two cannot be compared and a number from one is not a target for the other. The
#: sibling case runs zero-gradient, and this case is the same geometry, so it runs zero-gradient too.
_K_WALL_BCS = {"dirichlet": Dirichlet(0.0), "zerogradient": ZeroGradient()}
K_WALL = os.environ.get("PITZ_K_WALL", "zerogradient")
if K_WALL not in _K_WALL_BCS:
    raise SystemExit(f"PITZ_K_WALL={K_WALL!r} is not one of {sorted(_K_WALL_BCS)}")
K_WALL_BC = _K_WALL_BCS[K_WALL]

#: The trailing `[k, omega]` block's inverse: the differentiable-framework nodal hierarchy, which the
#: sibling case defaults to after a controlled pair measured it ahead of the host V-cycle (67 steps and
#: 2124 s against 72 and 2893, to the same reattachment length).
NATIVE_TRAILING = {"max_coarse": COARSE_EQ_LIMIT, "equilibrate": False}

#: ⚠️ THE PROBED JACOBIAN IS EXACT ONLY AT REACH 5 ON THIS MESH, AND AT REACH 3 ON THE SIBLING'S --
#: WITH IDENTICAL SCHEMES. The cause is the mesh, not the dimension, and it generalizes.
#:
#: `CorrectedGreenGauss` does not compute a gradient in one shot: it solves `A_g G = B phi` by
#: Richardson sweeps (four, by default), and each sweep extends the gradient's stencil by one ring, so
#: the residual reaches `sweeps + 1`. But that coupling is weighted entirely by the skewness offset
#: `D_g,ip = x_f - (x_P + g*d)`. Where it vanishes, `A_g` is diagonal, sweeps two onward add exactly
#: nothing, and the scheme degenerates to compact Green-Gauss at reach 1.
#:
#:      mesh        median skew   max skew   interior faces above 1e-10
#:      pitzDaily      2.2e-09     7.5e-02       20049 of 24170
#:      bfs3d          7.0e-15     1.9e-12           0 of 66368
#:
#: The sibling is a rectilinear blockMesh, skew-free to roundoff, so its sweeps are INERT; this mesh
#: has the slanted lower wall and the contraction, so they are not. Confirmed four ways, the cleanest
#: being that this case with `sweeps=1` floors at reach 3 exactly as the sibling does.
#:
#: ⚠️⚠️ SO `stencil_reach = 3` IS A PROPERTY OF SKEW-FREE MESHES, NOT OF THE DISCRETIZATION. Any case
#: on a genuinely skewed mesh needs `sweeps + 1`, in three dimensions as much as in two. The sibling
#: gets 3 for free and that is luck, not physics.
#:
#: ⚠️ REACH 5 IS NECESSARY AND NOT SUFFICIENT, AND THAT PAIRING IS THE WHOLE POINT. Measured against
#: the smoother fill beside it, on the leading block at beta = 2:
#:
#:      reach 3 + fill 1   fails  (300 matvecs, true residual 3.36)
#:      reach 5 + fill 0   fails  (300 matvecs, true residual 3.50)
#:      reach 5 + fill 1   ONE matvec to the march's stop, ten to 1.7e-09
#:
#: Neither alone is worth anything, which is exactly how a one-variable-at-a-time sweep misleads: reach
#: 5 was measured "step-for-step identical, 35% dearer, buys nothing" and reverted -- a correct
#: measurement of the wrong pair. Vary these two together or not at all.
#:
#: The error reach 3 leaves is ~2e-07 concentrated in the PRESSURE column, which enters the residual
#: only through gradients and so inherits the sweep-extended stencil undiluted. Because a colouring is
#: collision-free only for its own pattern, that is corruption of near entries rather than truncation.
STENCIL_REACH = int(os.environ.get("PITZ_STENCIL_REACH", "5"))

#: ⚠️ UNIFORM PROBING REACH, deliberately, where the sibling case shortens two columns. Its
#: `(3,3,3,3,2,2)` is a SIX-field layout and was measured on that mesh and those schemes; the analogous
#: five-field value here is unmeasured, and the record is emphatic that shortening the pressure column
#: diverged that case at step one. Uniform reach costs more probes and is always correct, so it is what
#: this case uses until someone measures the shortened one HERE.
COLUMN_REACH = None

#: A cost cap on the inner loop, so a doomed attempt is cut short rather than run to a stagnation.
CYCLE_BUDGET = 42

#: Grow the pseudo-timestep while the inner line search is comfortable; brake on a clipped step or a
#: rising residual. Without a control beta never ramps and the march cannot develop at all.
CONTROL = CflResidualDualTimeControl(
    beta_start=0.5, beta_min=0.005, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25
)

#: ⚠️ REFRESH THE FROZEN PRECONDITIONER, ON SOLVE COST, EXACTLY AS THE THREE-DIMENSIONAL CASE DOES.
#: Frozen at the cold reference state for a whole march, the preconditioner goes stale precisely as the
#: recirculation forms -- which is the one thing a cold march is for. The symptom is unmistakable once
#: known, and was observed here before this was wired: the solve cost climbs while the step length
#: stays healthy, the control lowers the shift, the now-expensive solve trips the retry ladder, the
#: ladder puts the shift back, and the march enters a limit cycle with the residual flat. Five steps at
#: 14-18 cycles a solve moved the residual from 3.404e-03 to 3.441e-03 -- backwards.
#:
#: The trigger is the COST itself: a solve that reaches this many restart cycles rebuilds the
#: preconditioner at the iterate it was handed, and the inner loop carries on rather than the step
#: being discarded. Capped at one refresh per step. Reacting to cost rather than predicting staleness
#: is deliberate: a diagnostic on the sibling case refuted every cheap STATIC predictor of a bad step,
#: so detect-then-react is the honest design.
REFRESH_ON_CYCLES = int(os.environ.get("PITZ_REFRESH_ON_CYCLES", "3"))

#: Redo a step whose solve was expensive, whose line search collapsed, or that diverged -- escalating
#: the shift first, and falling back to a tighter Krylov solve only for a divergence damping cannot fix.
RETRY = RetryPolicy(
    solver=relative_residual_gmres(1e-4, restart=40),
    on_cycles=10,
    on_alpha=0.01,
    beta_factor=2.0,
)


# --- OpenFOAM ascii internalField parsing (nonuniform scalar / vector list) ---
def _of_scalar(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    return np.array(re.findall(r"[-+]?\d[\d.eE+-]*", body)[: int(m.group(1))], dtype=float)


def _of_vector(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    tri = re.findall(r"\(([^)]*)\)", body)[: int(m.group(1))]
    return np.array([[float(v) for v in s.split()] for s in tri], dtype=float)


def read_openfoam_reference():
    """The OpenFOAM comparison fields + cell centres, keyed to their cell centroids.

    The **fields** come from the time-accurate run (:data:`TRANSIENT`), which reaches a statistically
    steady state with a well-defined reattachment (``x_r/h`` 7.74); the steady case's fields are not a
    valid target and are deliberately not read here (see the comment on :data:`TRANSIENT`). The **cell
    centres** still come from the steady run's ``Ccx``/``Ccy``: both cases are built from the same
    ``blockMeshDict``, so the mesh is byte-identical and the centres are geometry rather than a
    solution -- the transient case ships its written fields but no ``Cc*``.

    Returns
    -------
    dict
        ``centroid`` ``(n_cells, 2)``, ``U`` ``(n_cells, 2)``, and ``p``, ``k``, ``omega``, ``nut``
        each ``(n_cells,)``, in the mesh's own cell order.
    """
    ccx, ccy = _of_scalar(RUNS / "Ccx"), _of_scalar(RUNS / "Ccy")
    return dict(
        centroid=np.column_stack([ccx, ccy]),
        U=_of_vector(TRANSIENT / "U")[:, :2],
        p=_of_scalar(TRANSIENT / "p"),
        k=_of_scalar(TRANSIENT / "k"),
        omega=_of_scalar(TRANSIENT / "omega"),
        nut=_of_scalar(TRANSIENT / "nut"),
    )


def build_case(model=None):
    """Assemble the benchmark: mesh, momentum, turbulence and the coupled residual -- no solve.

    Split out from :func:`solve_aquaflux` so a solver study can re-solve at a saved state (a
    mid-march checkpoint, say) without re-marching to it, and without restating the case. The
    mesh import, boundary conditions, model constants and scheme choices *are* the definition of
    this benchmark; a second copy of them would drift from the one the validation figures use.

    Parameters
    ----------
    model : SSTModel, optional
        The SST constants to use. Defaults to :class:`~aquaflux.turbulence.SSTModel`. Passing a model
        that differs only in the near-wall omega blend (``wall_omega_exponent`` /
        ``wall_omega_viscous_coeff``) is how a wall-treatment study compares blend shapes on the same
        case -- e.g. a large exponent to reproduce the ``max(omega_vis, omega_log)`` blend.

    Returns
    -------
    dict
        ``coupled``, ``momentum``, ``turbulence`` and ``geom`` for the assembled case.
    """
    if model is None:
        model = SSTModel()
    mesh = read_openfoam(RUNS / "polyMesh")
    geom = mesh.geometry()
    # Corrected (non-orthogonal / skewness) Green-Gauss gradients. Its A_g^-1 apply is the default O(n)
    # matrix-free swept solve (fixed Richardson sweeps), not a nested GMRES: identical discretization,
    # but it avoids a nested Krylov solve (carrying its own implicit-diff tangent) inside every
    # coupled-residual evaluation, which otherwise dominates the monolithic Newton cost on this
    # ~12k-cell mesh (measured ~180x per residual eval here). The default sweep count is used: this mesh
    # is only mildly non-orthogonal (worst face angle ~6 degrees), so the swept solve reaches the
    # converged corrected-gradient to machine precision in the default few sweeps -- and the
    # reconstructed gradient, the coupled residual, and the reattachment length are all unchanged from a
    # much higher sweep count, so paying for more sweeps only enlarges the differentiated residual.
    grad = CorrectedGreenGauss()
    # Momentum advection: second-order upwind = Venkatakrishnan-limited linear upwind (the upwind cell
    # reconstructed to the face with its corrected-Green-Gauss gradient, slope-limited so the
    # reconstruction is monotonicity-bounded) -- the analogue of OpenFOAM's `Gauss linearUpwind`.
    momentum_upwind = LimitedUpwind(limiter=VenkatakrishnanLimiter())
    # Turbulence advection: first-order upwind on k and omega. The slope limiter bounds the advective
    # *face value*, but the negative-omega failure of second-order on the stiff omega equation is a
    # Newton-*update* overshoot at the cell centre, not a face-value one: first-order upwind makes the
    # omega transport operator diagonally dominant (an M-matrix) so the pseudo-transient-shifted Newton
    # step preserves positivity, whereas a second-order stencil -- even limited -- weakens that
    # dominance and lets the update drive omega < 0 (then nu_t = k/omega flips sign and poisons the
    # closure while the residual stays finite, so the divergence guard never trips). The structural fix
    # for second-order scalars is log-variable transport (omega = e^w), which is not built here.
    scalar_upwind = FirstOrderUpwind()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(jnp.asarray(RHO * NU)), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
            }
        ),
        advection_scheme=momentum_upwind,
    )
    # ⚠️ MATCHES THE SIBLING CASE'S LINEARIZATION. With the limiter left implicit the Jacobian
    # carries the k-production cap's own derivative, which is destabilizing; freezing it is the
    # Patankar treatment the sibling case runs. The library default is False, so omitting this
    # silently gave the two cases DIFFERENT Newton operators on the same physics.
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        explicit_production_limiter=True,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                "upperWall": K_WALL_BC,
                "lowerWall": K_WALL_BC,
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
            }
        ),
    )
    # Log-transform omega: omega = e^w stays strictly positive under any Newton step. On this stiff
    # separating case a direct-omega step drives omega negative once the recirculation forms (nu_t =
    # k/omega then flips sign and poisons the closure while the residual stays finite, so the divergence
    # guard never trips). k stays direct -- log(k) is ill-conditioned where k -> 0 at the walls.
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    # The monolithic Newton is globalized by the default pseudo-transient continuation: an a_P /
    # transport-diagonal shift that damps each step heavily far from the fixed point and ramps to
    # zero on the residual, recovering the exact steady Newton state at convergence.
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def solve_aquaflux(*, log_path=None, **solve_kwargs):
    """Solve the coupled RANS system on the imported OpenFOAM mesh; return fields + geometry.

    Parameters
    ----------
    **solve_kwargs
        Forwarded to :func:`~aquaflux.turbulence.solve_coupled`, overriding the defaults set here.
        This is the seam a solver study uses to instrument or reconfigure the march -- an ``on_step``
        observer, a ``refresh_trigger``, a different ``method``.
    """
    case = build_case()
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    # One line per outer step, flushed, to `log_path` or stdout. A march nobody can read until it
    # finishes costs its whole wall time to tell you something it knew in the third minute -- and a
    # crawling march is indistinguishable from a hung one without it.
    log_file = open(log_path, "w") if log_path is not None else sys.stdout
    logger = MarchLogger(
        log_file,
        fields=coupled_fields(coupled),
        detail=("inner", "fields", "pc"),
        rtol=RTOL,
        atol=ATOL,
    )
    # Every run states the configuration it was taken under, in its own log, before any result: a
    # number whose configuration is not written beside it cannot be re-adjudicated later, which is
    # worse than being wrong, because a wrong finding gets corrected and an unanchored one gets cited.
    logger.note("[configuration]")
    for _name, _value in (
        ("dual-time inner steps / tol", f"{INNER_STEPS} / {INNER_TOL}"),
        ("k positivity floor", K_POSITIVITY_FLOOR),
        ("inner forward rtol (row-scaled) / restart", f"{FORWARD_RTOL} / {FORWARD_RESTART}"),
        ("cycle budget", CYCLE_BUDGET),
        ("retry on cycles / alpha", f"{RETRY.on_cycles} / {RETRY.on_alpha}"),
        ("step control", type(CONTROL).__name__),
        ("Reynolds continuation points", N_POINTS),
        ("k wall BC", K_WALL),
        ("preconditioner refresh", f"on {REFRESH_ON_CYCLES} restart cycles (mid-step)"),
        ("smoother fill / sweeps / coarse limit", f"{FILL_LEVELS} / {SWEEPS} / {COARSE_EQ_LIMIT}"),
        ("preconditioner beta floor", PC_BETA_FLOOR),
        ("field split / trailing sweeps", f"{FIELD_SPLIT} / {TRAILING_SWEEPS}"),
        ("trailing inverse", "native nodal" if FIELD_SPLIT else "n/a"),
        ("probe stencil reach", STENCIL_REACH),
        ("probe column reach", COLUMN_REACH or "uniform"),
        ("forward restart / max restarts", f"{FORWARD_RESTART} / {FORWARD_MAX_RESTARTS}"),
        ("k positivity projection", POSITIVITY_PROJECTION),
        ("stop (rtol, atol)", f"{RTOL}, {ATOL}"),
    ):
        logger.note(f"  {_name}: {_value}")

    # The refresh hook, built ONCE and pointed at each rung in turn. Its scheduled cadences are
    # switched OFF so the cycle trigger REPLACES them rather than adding to them: as an addition the
    # trigger was measured break-even on the sibling case, as a replacement it was the largest saving
    # on that march. ⚠️ `beta_rel_change=None` does NOT switch the schedule off -- it removes the gate,
    # and a missing gate means "refresh every step". Off means a gate that exists and never fires.
    # Built once and shared by the engine and the refresh hook: the coloured-probe plan is the single
    # largest allocation this case makes, and building it twice doubles that for nothing.
    probe = CoupledJacobianProbe.build(
        coupled, stencil_reach=STENCIL_REACH, column_reach=COLUMN_REACH
    )
    refresh = amg_beta_tracking_refresh(
        coupled,
        probe=probe,
        beta_rel_change=float("inf"),
        refresh_every=10**9,
        materialize_drift=None,
        materialize_every=None,
        beta_floor=PC_BETA_FLOOR,
        observer=logger.on_refresh,
    )
    #: One preconditioner shared across rungs, handed back for the next: only the viscosity changes
    #: between them, so a rung needs the V-cycle FITTED to it, not a fresh object.
    shared_preconditioner: list = []

    def point_setup(companion, seed_state, point):
        """Configure each Reynolds rung, re-fitting the one preconditioner to it.

        Only the molecular viscosity changes between rungs, so a rung needs its own residual assembler
        and its own row scales -- both ordinary data -- but not its own V-cycle. It needs that V-cycle
        *fitted to it*, which is what `rebind` arranges, and which is a different thing from rebuilding
        the object (that would only cost a compilation).
        """
        logger.note(f"[{point.label}]")
        refresh.rebind(companion)
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            probe=probe,
            cycle_budget=CYCLE_BUDGET,
            forward_rtol=FORWARD_RTOL,
            forward_restart=FORWARD_RESTART,
            forward_max_restarts=FORWARD_MAX_RESTARTS,
            refresh_on_cycles=REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if REFRESH_ON_CYCLES else None,
            positivity_floor=K_POSITIVITY_FLOOR,
            positivity_projection=POSITIVITY_PROJECTION,
            preconditioner=shared_preconditioner[0] if shared_preconditioner else None,
            smoother_fill_levels=FILL_LEVELS,
            smoother_sweeps=SWEEPS,
            coarse_eq_limit=COARSE_EQ_LIMIT,
            field_split=FIELD_SPLIT,
            trailing_smoother_sweeps=TRAILING_SWEEPS,
            trailing_inverse=native_nodal_inverse(**NATIVE_TRAILING) if FIELD_SPLIT else None,
            inner_observer=logger.on_inner,
        )
        shared_preconditioner[:] = [engine.shift_policy.preconditioner]
        return dict(continuation=engine, refresh=RefreshPolicy(precondition_step=refresh))

    solve_options = (
        dict(
            max_steps=MAX_STEPS,
            rtol=RTOL,
            atol=ATOL,
            intermediate_rtol=None,  # every rung stops at the same ABSOLUTE bar
            intermediate_atol=ATOL,
            step_control=CONTROL,
            retry=RETRY,
            point_setup=point_setup,
            scaled_norm=True,  # rebuild the row scales each outer step
            on_checkpoint=logger.on_checkpoint,
            on_retry=logger.on_retry,
        )
        | solve_kwargs
    )
    try:
        flow, k, omega = solve_reynolds_continuation(coupled, N_POINTS, **solve_options)
    finally:
        if log_file is not sys.stdout:
            log_file.close()
    velocity, pressure = momentum.unpack(flow)
    nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    return dict(
        centroid=np.asarray(geom.cell.centroid),
        U=np.asarray(velocity),
        p=np.asarray(pressure),
        k=np.asarray(k),
        omega=np.asarray(omega),
        nut=np.asarray(nu_t),
    )


def reattachment_length(centroid, u_x):
    """Lower-wall reattachment length x_r/h behind the step (h = step height = 0.0254 m).

    Reads the sign of the wall-adjacent streamwise velocity along the lower wall downstream of the
    step: the recirculation bubble is where it is negative, and reattachment is the last such x.
    """
    h = 0.0254
    x, y = centroid[:, 0], centroid[:, 1]
    # The wall-adjacent row along the lower wall (y just above the -h floor), downstream of the step.
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    xs = x[band]
    us = u_x[band]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = np.where(us < 0)[0]
    if neg.size == 0:
        return 0.0
    return float(xs[neg[-1]] / h)


def main():
    if not (RUNS / "U").exists():
        raise SystemExit(f"OpenFOAM results not found in {RUNS}; run of_case/run_of.sh first.")
    of = read_openfoam_reference()
    print(
        f"OpenFOAM: {of['centroid'].shape[0]} cells, Ux in "
        f"[{of['U'][:, 0].min():.3f}, {of['U'][:, 0].max():.3f}]",
        flush=True,
    )

    t0 = time.time()
    aq = solve_aquaflux()
    print(
        f"aquaflux coupled solve: {time.time() - t0:.0f}s, "
        f"Ux in [{aq['U'][:, 0].min():.3f}, {aq['U'][:, 0].max():.3f}]",
        flush=True,
    )

    from scipy.spatial import cKDTree

    tree = cKDTree(of["centroid"])
    dist, idx = tree.query(aq["centroid"])
    assert float(dist.max()) < 1e-6, f"mesh mismatch: max centroid distance {dist.max()}"

    def rel_l2(a, b, scale):
        return float(np.sqrt(np.mean((a - b) ** 2)) / scale)

    metrics = dict(
        ux=rel_l2(aq["U"][:, 0], of["U"][idx, 0], U_IN),
        uy=rel_l2(aq["U"][:, 1], of["U"][idx, 1], U_IN),
        xr_aqua=reattachment_length(aq["centroid"], aq["U"][:, 0]),
        xr_of=reattachment_length(of["centroid"], of["U"][:, 0]),
        nut_peak_aqua=float(aq["nut"].max() / NU),
        nut_peak_of=float(of["nut"].max() / NU),
    )
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}", flush=True)

    _figure(of, aq, idx)
    _report(metrics)


def _figure(of, aq, idx):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x, y = aq["centroid"][:, 0], aq["centroid"][:, 1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    vmax = max(abs(of["U"][:, 0]).max(), abs(aq["U"][:, 0]).max())
    for a, data, title in (
        (ax[0], of["U"][idx, 0], "OpenFOAM $U_x$"),
        (ax[1], aq["U"][:, 0], "aquaflux $U_x$"),
    ):
        sc = a.scatter(x, y, c=data, s=3, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    # Reattachment: lower-wall streamwise velocity sign.
    h = 0.0254
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    order = np.argsort(x[band])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(
        x[band][order] / h,
        of["U"][idx][band][order][:, 0],
        "s-",
        ms=3,
        color="C1",
        label="OpenFOAM",
    )
    ax[2].plot(
        x[band][order] / h, aq["U"][band][order][:, 0], ".-", ms=4, color="C0", label="aquaflux"
    )
    ax[2].set_xlabel("$x/h$ behind the step")
    ax[2].set_ylabel("near-wall $U_x$")
    ax[2].set_title("Lower-wall recirculation (sign change = reattachment)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "comparison.png", dpi=130)
    print(f"wrote {FIGS / 'comparison.png'}", flush=True)


def _report(m):
    lines = [
        "# pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST",
        "",
        "The OpenFOAM `pitzDailySteady` tutorial -- its RAS model switched from the shipped `kEpsilon`",
        "to `kOmegaSST` -- run in OpenFOAM, then solved on the **same imported mesh** by aquaflux's",
        "coupled RANS solver (hybrid initialization, second-order upwind momentum advection, corrected",
        "Green-Gauss gradients). U_in = 10 m/s, nu = 1e-5 (Re ~ 25000 on the 25.4 mm inlet).",
        "",
        "## Results",
        "",
        "| quantity | aquaflux | OpenFOAM |",
        "|---|---|---|",
        f"| reattachment length x_r/h (lower wall) | {m['xr_aqua']:.2f} | {m['xr_of']:.2f} |",
        f"| peak nu_t/nu | {m['nut_peak_aqua']:.0f} | {m['nut_peak_of']:.0f} |",
        f"| rel. L2 U_x error (cell-for-cell) | {m['ux']:.3f} | -- |",
        f"| rel. L2 U_y error (cell-for-cell) | {m['uy']:.3f} | -- |",
        "",
        "See `figures/comparison.png`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image) -> runs/kwsst/",
        "cd validation/pitzdaily_openfoam",
        'docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh',
        "# 2. aquaflux coupled solve + comparison (from the repo root)",
        "cd ../..",
        "python3 validation/pitzdaily_openfoam/compare.py",
        "```",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
