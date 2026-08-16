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
    RetryPolicy,
    relative_residual_gmres,
)
from aquaflux.turbulence import (
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    coupled_fields,
    solve_coupled,
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
MAX_STEPS, RTOL = 200, 1e-6

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

#: The dual-time inner loop. `inner_tol` 1e-2 rather than a tighter value: measured on the
#: three-dimensional case, 1e-3 bought nothing over 1e-2 while costing a third of the march.
INNER_STEPS, INNER_TOL = 5, 1e-2

#: ⚠️ NOT REACHABLE ON THIS PATH, and recorded here because the gap is the finding rather than the
#: number. A floor buys the step limiter out of a numerically dead cell instead of letting one cell
#: ratchet the global step cap toward zero -- but `positivity_floor` is a parameter of
#: `coupled_amg_continuation` ALONE. The default builder this case uses (`coupled_continuation`), and
#: the complete-LU and threshold-ILU builders beside it, expose neither it nor the `step_limit` it
#: would be set on. So the safeguard the three-dimensional case depends on cannot be switched on here
#: without a library change, and a march that meets the ratchet has no way to escape it.
#: The value the three-dimensional case ships, for when that is fixed:
POSITIVITY_FLOOR_WANTED = 1e-8

#: The inexact-Newton stop per inner linear solve, in the row-scaled measure, and the Krylov restart.
FORWARD_RTOL, FORWARD_RESTART = 0.3, 15

#: A cost cap on the inner loop, so a doomed attempt is cut short rather than run to a stagnation.
CYCLE_BUDGET = 42

#: Grow the pseudo-timestep while the inner line search is comfortable; brake on a clipped step or a
#: rising residual. Without a control beta never ramps and the march cannot develop at all.
CONTROL = CflResidualDualTimeControl(
    beta_start=0.5, beta_min=0.005, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25
)

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
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
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
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                "upperWall": Dirichlet(0.0),
                "lowerWall": Dirichlet(0.0),
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
        detail=("inner", "fields"),
        rtol=RTOL,
        atol=0.0,
    )
    # Every run states the configuration it was taken under, in its own log, before any result: a
    # number whose configuration is not written beside it cannot be re-adjudicated later, which is
    # worse than being wrong, because a wrong finding gets corrected and an unanchored one gets cited.
    logger.note("[configuration]")
    for _name, _value in (
        ("dual-time inner steps / tol", f"{INNER_STEPS} / {INNER_TOL}"),
        ("k positivity floor", "UNREACHABLE on this builder (see POSITIVITY_FLOOR_WANTED)"),
        ("inner forward rtol (row-scaled) / restart", f"{FORWARD_RTOL} / {FORWARD_RESTART}"),
        ("cycle budget", CYCLE_BUDGET),
        ("retry on cycles / alpha", f"{RETRY.on_cycles} / {RETRY.on_alpha}"),
        ("step control", type(CONTROL).__name__),
        ("stop (rtol, atol)", f"{RTOL}, 0.0"),
    ):
        logger.note(f"  {_name}: {_value}")

    solve_options = (
        dict(
            max_steps=MAX_STEPS,
            rtol=RTOL,
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            step_control=CONTROL,
            retry=RETRY,
            scaled_norm=True,  # rebuild the row scales each outer step
            on_checkpoint=logger.on_checkpoint,
            on_retry=logger.on_retry,
        )
        | solve_kwargs
    )
    try:
        flow, k, omega = solve_coupled(coupled, **solve_options)
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
