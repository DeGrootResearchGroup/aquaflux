"""Finite-width 3D backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST.

The first **3D** same-mesh, cell-for-cell cross-code validation, the three-dimensional sibling of
``validation/pitzdaily_openfoam``. A backward-facing step of height ``h = 0.01 m`` (expansion ratio 2)
spanning ``4h`` between two **no-slip side walls** -- the finite span with viscous side walls makes the
flow genuinely three-dimensional (corner/secondary flow, spanwise-varying reattachment), not a 2D
extrusion. ``of_case/run_of.sh`` runs the OpenFOAM ``incompressibleFluid`` steady SIMPLE solver with
``kOmegaSST`` and writes the fields, the mesh and the 3D cell centres to ``runs/kwsst/``;
``of_transient/run_transient.sh`` runs a time-accurate reference to a statistically-steady state
(``runs/kwsst_transient/``). This script reads that **same mesh** into aquaflux via ``read_openfoam``
and solves the coupled RANS system on it, so the two solutions live on identical cells and are compared
directly (no interpolation between independent meshes).

aquaflux setup, mirroring the 2D case:

* the **coupled** turbulent solver (:func:`aquaflux.turbulence.solve_coupled` -- one monolithic Newton
  on ``R(u, p, k, omega)``, globalized by pseudo-transient continuation);
* **hybrid initialization** (potential-flow velocity + Laplace-smoothed turbulence), which self-starts
  the monolithic Newton;
* **second-order upwind** momentum advection (:class:`aquaflux.discretization.LimitedUpwind` with the
  :class:`aquaflux.schemes.VenkatakrishnanLimiter`); the stiff k/omega scalars use bounded first-order
  upwind;
* **corrected Green-Gauss** gradients (:class:`aquaflux.schemes.CorrectedGreenGauss`);
* **log-variable omega** (:class:`aquaflux.turbulence.LogScalars` on ``omega_transform``): ``omega =
  e^w`` stays strictly positive under any Newton step, the fix for a direct-omega step driving omega
  negative once recirculation forms.

**Preconditioner (the point of a 3D case):** the coupled Jacobian is preconditioned by an
**algebraic-multigrid V-cycle** (:func:`aquaflux.turbulence.coupled_amg_continuation`), not by a
factorization. The complete LU is exact but its fill is a memory wall in 3D (``O(n^{4/3})``), and even the
threshold-ILU's factorization of the distance-3 3D coupled Jacobian (hundreds of nonzeros per row) is
prohibitively slow to build; the V-cycle keeps the heavy fill on only the small coarsest grid (a direct-LU
coarse solve), so its memory stays bounded and its setup is seconds. This 3D case is the first exercise of
that scaling path.

The V-cycle is **field-split** (``field_split=True``): the ``[u, v, w, p]`` saddle and the ``[k, omega]``
transported scalars get their own hierarchies, with one triangle of the coupling between them retained
exactly from the assembled Jacobian rather than dropped. Measured 31% faster end to end here, to the same
reattachment length -- see :data:`FIELD_SPLIT`, which also records why the *cycle* count moves the other
way.

**Reference caveat (binding -- do not skip):** the OpenFOAM *steady* (SIMPLE) run does **not** fully
converge this case -- it limit-cycles at ~1e-3 residual on the separated 3D flow (though, unlike the 2D
pitzDaily case, the field is physical: no inlet checkerboard). Use a **transient-converged** field
(``runs/kwsst_transient/``) as the comparison target, and compare the outer-flow structure (velocity,
reattachment length) rather than the raw residual.

Run (after both OpenFOAM runs) from the repo root:
    python3 validation/bfs3d_openfoam/compare.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

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
    InnerIterateCheckpointer,
    MarchLogger,
    StateCheckpointer,
    combine_observers,
    relative_residual_gmres,
)
from aquaflux.turbulence import (
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    coupled_residuals,
    solve_reynolds_continuation,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "kwsst"  # steady run: the mesh + 3D cell centres (geometry)
TRANSIENT = HERE / "runs" / "kwsst_transient"  # the comparison fields (time-averaged / steady)
FIGS = HERE / "figures"

# The operating point (0.orig/ and constant/): U_in = 10 m/s, nu = 1e-5, h = 0.01 -> Re_h = 10000.
# k_in = 0.375 (5% turbulence intensity), omega_in = 1600 (length scale 0.07 H1). rho = 1.
RHO, NU = 1.0, 1e-5
U_IN, K_IN, OMEGA_IN = 10.0, 0.375, 1600.0
H = 0.01  # step height
WALLS = ["upperWall", "lowerWall", "sideWalls"]
# The coupled Newton march budget. Stiff, separating, 3D, high-Re case on a wall-function mesh; the cap
# is generous so the march exits on the tolerance, not the count.
# --- the solve configuration, each part measured on this case (see the README) -------------------
MAX_STEPS = 150  # per continuation rung
# ABSOLUTE stop on the row-scaled residual. That measure is already a fractional change per equation,
# so dividing it again by |R0| makes the bar a property of the initial guess -- and under continuation
# every rung re-bases |R0|, which let a later rung stop looser than an earlier one had already reached.
RTOL, ATOL = 0.0, 1e-5
N_POINTS = 2  # Reynolds continuation: anchor at Re/100, then Re/10, then the target
INNER_STEPS, INNER_TOL = 5, 1e-3
# Preconditioner bundle. ILU(1) DIVERGES at the low shifts this march's tail runs at (ground truth: 303
# negative pivots at beta = 0.02, zero for ILU(0)); zero fill converges at every shift tested and builds
# 3-4x faster. ILU(0) is the weaker smoother, so the extra sweeps pay more than they did for ILU(1).
# coarse=None stalls at every low shift. The beta floor is PRECONDITIONER-ONLY: the V-cycle is built at
# max(beta, floor) while the march solves at its own beta, so the root and the adjoint are unchanged.
FILL_LEVELS, SWEEPS, COARSE_EQ_LIMIT, PC_BETA_FLOOR = 0, 4, 2000, 0.05
# Split the preconditioner's hierarchy in two -- the [u,v,w,p] saddle and the [k,omega] transported
# scalars each get their own, with one triangle of the coupling between them retained exactly -- rather
# than putting all six fields through one hierarchy with one smoother. ON, because it is measured 31%
# faster end to end on this case at the identical configuration:
#
#                       monolithic     split
#     wall                  3140 s    2161 s   (-31%)
#     steps                     58        66
#     Krylov cycles            293       324   (+11%)
#     refresh          19 / 310 s   23 / 352 s
#     mid-span x_r/h         8.361     8.361   (identical)
#
# Read the cycle row before concluding anything from a cycle count: the split takes MORE cycles and is
# far faster, because two smaller V-cycles plus one sparse coupling product apply much more cheaply than
# one six-field V-cycle. Its mean cycles per inner solve is actually lower (1.49 vs 1.68) -- the higher
# total comes from more, cheaper steps. It also triggers ~4 more cost-driven refreshes (9.7% of inner
# solves cross the threshold against 9.0%), which hands back ~42 s of the ~980 s saved.
# `BFS3D_FIELD_SPLIT=0` restores the monolithic V-cycle for an A/B.
FIELD_SPLIT = os.environ.get("BFS3D_FIELD_SPLIT", "1") not in ("", "0")
CYCLE_BUDGET = 42  # summed per step: a cost cap, so summed is what it should cap
RETRY_ON_CYCLES = (
    10  # PER SOLVE: a summed trigger is ~6x more sensitive for a 5-inner step than a 1-inner one
)
# The forward GMRES restart length, and the reason it is worth varying: a restarted GMRES tests
# convergence only at restart boundaries, so a solve that needs three matrix-vector products still pays
# a full restart's worth. Cycle counts cannot see that -- such a solve reports one cycle either way --
# so shortening the restart reduces seconds per cycle while leaving every cycle-based measurement
# unchanged. `BFS3D_FORWARD_RESTART=4 ...` to try it.
#
# BOTH cost thresholds above are denominated in CYCLES, so they must scale with the restart or the
# experiment changes the march's control behaviour rather than just its cost: at a restart of 5 an
# unscaled `retry_on_cycles = 10` would fire after 50 matrix-vector products where it used to take 150.
# Scaling by the ratio keeps every bailout at the same matvec count, so the only variable is how much
# over-solving happens inside a cycle. Vary the restart through the builder's own `forward_restart`, NOT
# by passing a whole `forward_solver`: the builder's default also carries a loose row-scaled stop that a
# hand-built solver would silently replace, which measures something else entirely.
BASELINE_RESTART = 15  # the coupled AMG builder's own default
FORWARD_RESTART = int(os.environ.get("BFS3D_FORWARD_RESTART", str(BASELINE_RESTART)))
_RESTART_SCALE = BASELINE_RESTART / FORWARD_RESTART
RETRY_BETA_FACTOR = 2.0
# How many per-step states to retain. Three is enough to restart from, which is all a normal run needs.
# A PRECONDITIONER STUDY needs more: an easy operator does not discriminate between preconditioners, so a
# sweep has to run at the march's own HARD states (highest cycle count, clipped a_min, a retry flag) and
# those are mid-march. Set `BFS3D_CHECKPOINT_KEEP` high enough to cover the run (~1.1 MB per step) and the
# whole trajectory is kept: `BFS3D_CHECKPOINT_KEEP=80 python3 validation/bfs3d_openfoam/compare.py`.
CHECKPOINT_KEEP = int(os.environ.get("BFS3D_CHECKPOINT_KEEP", "3"))
# Save the INNER iterates whose linear solve reached this many restart cycles. Off by default (0), and
# the march is byte-identical with it off. It exists because a checkpoint is written at the END of a
# step, so it holds the state the next step begins from -- and this march's step-initial solves all cost
# at most 2 cycles while solves later in the inner loop reach 15. The hard operators are only reachable
# here: they cannot be replayed from a checkpoint afterwards, because a step's preconditioner and shift
# are a product of the refresh history rather than of the state.
INNER_DUMP_ABOVE = int(os.environ.get("BFS3D_INNER_DUMP_ABOVE", "0"))
# Refresh the preconditioner MID-STEP as soon as one solve reaches this many restart cycles, and switch
# the scheduled refreshes off. Off by default (0), which keeps the shipped schedule.
#
# The point is the swap, not the addition. Measured on the 3501 s march: the scheduled refreshes cost
# 742 s -- 21 % of the wall -- while 193 of 232 solves already took a single cycle, so most of that is
# maintaining a freshness nothing consumes. And no fixed cadence can be right, because the interval that
# matters is regime-dependent: one step of staleness is free at beta 0.333 and triples the cost at 0.029.
# Reacting to the cost itself adapts; a schedule cannot. Refreshing mid-step (rather than at the next
# step boundary) also keeps the inner loop's progress, where the current reaction -- abort and escalate
# beta -- throws away the work and the pseudo-timestep together.
REFRESH_ON_CYCLES = int(os.environ.get("BFS3D_REFRESH_ON_CYCLES", "0"))
CONTROL = CflResidualDualTimeControl(
    beta_start=0.5, beta_min=0.005, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25
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
    """The OpenFOAM comparison fields + 3D cell centres, keyed to their cell centroids.

    The **fields** come from the time-accurate run (:data:`TRANSIENT`); the steady case's limit-cycling
    fields are not a valid target and are deliberately not read here (see the module docstring). The
    **cell centres** come from the steady run's ``Ccx``/``Ccy``/``Ccz``: both cases are built from the
    same ``blockMeshDict``, so the mesh is identical and the centres are geometry rather than a solution.

    Returns
    -------
    dict
        ``centroid`` ``(n_cells, 3)``, ``U`` ``(n_cells, 3)``, and ``p``, ``k``, ``omega``, ``nut`` each
        ``(n_cells,)``, in the mesh's own cell order.
    """
    ccx, ccy, ccz = (_of_scalar(RUNS / f) for f in ("Ccx", "Ccy", "Ccz"))
    return dict(
        centroid=np.column_stack([ccx, ccy, ccz]),
        U=_of_vector(TRANSIENT / "U")[:, :3],
        p=_of_scalar(TRANSIENT / "p"),
        k=_of_scalar(TRANSIENT / "k"),
        omega=_of_scalar(TRANSIENT / "omega"),
        nut=_of_scalar(TRANSIENT / "nut"),
    )


def build_case(model=None):
    """Assemble the benchmark: mesh, momentum, turbulence and the coupled residual -- no solve.

    Split out from :func:`solve_aquaflux` so a solver study can re-solve at a saved state without
    re-marching to it, and without restating the case. The mesh import, boundary conditions, model
    constants and scheme choices *are* the definition of this benchmark.

    Parameters
    ----------
    model : SSTModel, optional
        The SST constants to use. Defaults to :class:`~aquaflux.turbulence.SSTModel`.

    Returns
    -------
    dict
        ``coupled``, ``momentum``, ``turbulence`` and ``geom`` for the assembled case.
    """
    if model is None:
        model = SSTModel()
    mesh = read_openfoam(RUNS / "polyMesh")
    geom = mesh.geometry()
    grad = CorrectedGreenGauss()
    # Second-order upwind momentum advection (Venkatakrishnan-limited linear upwind); first-order upwind
    # on the stiff k/omega scalars (a second-order stencil there lets the coupled Newton step drive omega
    # negative -- an M-matrix effect the limiter does not prevent).
    momentum_upwind = LimitedUpwind(limiter=VenkatakrishnanLimiter())
    scalar_upwind = FirstOrderUpwind()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
                "sideWalls": NoSlipWall(),
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
                "sideWalls": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
                "sideWalls": ZeroGradient(),
            }
        ),
    )
    # Log-transform omega (positive under any Newton step); k stays direct (log(k) is ill-conditioned
    # where k -> 0 at the walls).
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def solve_aquaflux(*, log_path=None, checkpoint_dir=None, **solve_kwargs):
    """Solve the coupled RANS system on the imported OpenFOAM mesh; return fields + geometry.

    **This is the full continuation solve the case's published result comes from** -- Reynolds
    continuation onto the target Reynolds number, the measured preconditioner bundle, the dual-time
    Courant control, and the beta-tracking refresh. The target Reynolds number is **not reachable
    without the ramp**: a direct cold start diverges on its first step (the residual grows by ~100
    orders of magnitude at a line-search factor already clipped to 0.002), so the ramp buys
    reachability, not merely a better seed.

    Preconditioned by the **field-split algebraic-multigrid V-cycle** -- the 3D coupled path, since the
    complete LU's fill is a memory wall past ~10^4 3D cells and the threshold-ILU's factorization is
    prohibitively slow to build on the distance-3 3D coupled Jacobian. Every non-default setting is a
    measurement rather than a preference; see the constants above and the README.

    Parameters
    ----------
    log_path : str or Path, optional
        Stream the march here, one framed block per step. Pass a path when running interactively -- a
        march of this length must be readable while it runs, not after it ends.
    checkpoint_dir : str or Path, optional
        Write a rolling state checkpoint per step here. Worth setting for any long run: without it a
        failure at the last step discards every step before it.
    **solve_kwargs
        Forwarded to :func:`~aquaflux.turbulence.solve_reynolds_continuation`, overriding the defaults.
    """
    case = build_case()
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    log_file = open(log_path, "w") if log_path is not None else None
    # Each rung rebuilds both the case (its viscosity is rescaled) and its continuation, so the
    # per-equation residual reporter has to be rebuilt with them. The logger holds one callable, which
    # defers to whichever reporter the rung currently being marched installed.
    rung_residuals: list = []

    def residuals(state):
        return rung_residuals[-1](state) if rung_residuals else {}

    logger = MarchLogger(
        log_file,
        metrics=reattachment_metrics(case),
        fields=coupled_fields(coupled),
        residuals=residuals,
        detail=("inner", "fields", "residuals", "pc"),
        rtol=RTOL,
        atol=ATOL,
    )
    checkpoints = (
        StateCheckpointer(checkpoint_dir, every=1, keep=CHECKPOINT_KEEP)
        if checkpoint_dir is not None
        else None
    )
    on_checkpoint = (
        logger.on_checkpoint
        if checkpoints is None
        else combine_observers(logger.on_checkpoint, checkpoints.on_checkpoint)
    )
    inner_dump = (
        InnerIterateCheckpointer(checkpoint_dir, above=INNER_DUMP_ABOVE)
        if INNER_DUMP_ABOVE and checkpoint_dir is not None
        else None
    )

    def point_setup(companion, seed_state, point):
        """Build each rung's preconditioner at ITS OWN viscosity and seed state.

        A single continuation built once cannot serve the ramp: the frozen operator has to be rebuilt
        per rung, and the beta-tracking refresh has to close over that rung's residual.
        """
        logger.note(f"[{point.label}]")
        # With the cycle trigger on, the scheduled cadences are switched OFF so it REPLACES them: as an
        # addition it is break-even, as a replacement it is the largest saving measured on this march.
        # CAREFUL: `beta_rel_change=None` does NOT switch the schedule off -- it removes the gate, and a
        # missing gate means "refresh every step". Switching it off means a gate that exists and never
        # fires again after its first (initialising) call, plus no materialize gates, so the refresh
        # branch resolves to `none`.
        scheduled = not REFRESH_ON_CYCLES
        refresh = amg_beta_tracking_refresh(
            companion,
            beta_rel_change=0.25 if scheduled else float("inf"),
            refresh_every=8 if scheduled else 10**9,
            materialize_drift=0.05 if scheduled else None,
            materialize_every=4 if scheduled else None,
            beta_floor=PC_BETA_FLOOR,
            observer=logger.on_refresh,
        )
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            smoother_fill_levels=FILL_LEVELS,
            smoother_sweeps=SWEEPS,
            coarse_eq_limit=COARSE_EQ_LIMIT,
            cycle_budget=round(CYCLE_BUDGET * _RESTART_SCALE),
            forward_restart=FORWARD_RESTART,
            inner_observer=combine_observers(
                logger.on_inner,
                *([inner_dump.on_inner] if inner_dump is not None else []),
            ),
            refresh_on_cycles=REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if REFRESH_ON_CYCLES else None,
            field_split=FIELD_SPLIT,
        )
        # Seeded with this rung's own starting state: the march equilibrates each step at the state it
        # begins from, so without the seed the rung's first step would be scaled at its end state and
        # its per-equation rows would not add up to the residual reported beside them.
        rung_residuals.append(coupled_residuals(companion, engine, seed_state))
        return dict(
            continuation=engine,
            precondition_step=refresh,
        )

    options = (
        dict(
            max_steps=MAX_STEPS,
            rtol=RTOL,
            atol=ATOL,
            intermediate_rtol=None,  # every rung stops at the same ABSOLUTE bar
            intermediate_atol=ATOL,
            step_control=CONTROL,
            point_setup=point_setup,
            scaled_norm=True,  # rebuild the row scales each outer step
            retry_solver=relative_residual_gmres(1e-4, restart=40),
            on_checkpoint=on_checkpoint,
            on_retry=logger.on_retry,
            retry_on_cycles=round(RETRY_ON_CYCLES * _RESTART_SCALE),
            retry_beta_factor=RETRY_BETA_FACTOR,
        )
        | solve_kwargs
    )
    try:
        flow, k, omega = solve_reynolds_continuation(coupled, N_POINTS, **options)
    finally:
        if log_file is not None:
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


def mid_span_slab(centroid, half_width=0.06):
    """The thin spanwise slab about ``z = W/2`` that the primary reattachment metric is measured in.

    The **full-span** reattachment length is not the primary bubble: the side walls carry their own
    corner separation that reattaches much further downstream, and because
    :func:`reattachment_length` takes the *last* reversed wall cell, those corner cells set the
    full-span number on their own. On the reference field the two outermost spanwise slabs read
    ``x_r/h = 10.28`` while all six interior slabs read ``7.24`` -- so a full-span number overstates
    the primary bubble by ~40% while being a perfectly good measurement of something else.

    Parameters
    ----------
    centroid : ndarray (n_cells, 3)
        Cell centroids.
    half_width : float
        Half-thickness of the slab as a fraction of the span. Default ``0.06``.

    Returns
    -------
    tuple(float, float)
        The ``(z_lo, z_hi)`` slab, ready for :func:`reattachment_length`'s ``z_slab``.
    """
    z = centroid[:, 2]
    width = z.max() - z.min()
    centre = z.min() + 0.5 * width
    return (centre - half_width * width, centre + half_width * width)


def reattachment_metrics(case):
    """A ``state -> {"xr/h": ..., "xr/h_full": ...}`` metrics callable for ``MarchLogger``.

    The march logger reports everything a solver step knows, but the quantity this case is actually
    steered by -- the reattachment length -- needs the case's own geometry and field layout. This is
    that seam, so a driver logging the march never re-derives the unpacking.

    Reports the **mid-span** length as ``xr/h``, the same measurement the run's final comparison makes,
    so the number a march is watched by and the number it is judged by cannot diverge. The full-span
    length rides alongside as ``xr/h_full`` because the gap between them *is* the corner separation
    (see :func:`mid_span_slab`) -- worth seeing, but not the primary bubble.

    Parameters
    ----------
    case : dict
        The assembled benchmark from :func:`build_case`.

    Returns
    -------
    callable
        ``state -> mapping``, mapping a packed coupled state to reattachment lengths in step heights.
    """
    coupled, momentum, geom = case["coupled"], case["momentum"], case["geom"]
    centroid = np.asarray(geom.cell.centroid)
    slab = mid_span_slab(centroid)

    def metrics(state):
        flow, _k, _omega = coupled.physical_fields(state)
        velocity, _pressure = momentum.unpack(flow)
        u_x = np.asarray(velocity)[:, 0]
        return {
            "xr/h": float(reattachment_length(centroid, u_x, z_slab=slab)),
            "xr/h_full": float(reattachment_length(centroid, u_x)),
        }

    return metrics


def reattachment_length(centroid, u_x, z_slab=None):
    """Lower-wall reattachment length x_r/h behind the step (h = step height).

    Reads the sign of the wall-adjacent streamwise velocity along the lower wall (y just above the -h
    floor) downstream of the step: the recirculation bubble is where it is negative, and reattachment is
    the last such x.

    Parameters
    ----------
    centroid : ndarray (n_cells, 3)
        Cell centroids.
    u_x : ndarray (n_cells,)
        Streamwise velocity.
    z_slab : tuple(float, float), optional
        Restrict to a spanwise slab ``z in [z_lo, z_hi]`` (e.g. a mid-span slice). ``None`` uses the full
        span, which averages the (spanwise-varying) bubble.
    """
    x, y, z = centroid[:, 0], centroid[:, 1], centroid[:, 2]
    band = (x > 1e-4) & (y < -H + 0.002) & (y > -H)
    if z_slab is not None:
        band &= (z >= z_slab[0]) & (z <= z_slab[1])
    xs, us = x[band], u_x[band]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = np.where(us < 0)[0]
    if neg.size == 0:
        return 0.0
    return float(xs[neg[-1]] / H)


def spanwise_reattachment(centroid, u_x, n_bins=8):
    """Reattachment length x_r/h in each of ``n_bins`` spanwise slabs -- the 3D structure the 2D case
    cannot show. Returns the per-slab (z_centre, x_r/h) so the writeup can report the mid-span value and
    the spanwise variation (corner vs centre)."""
    z = centroid[:, 2]
    z_lo, z_hi = z.min(), z.max()
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    out = []
    for i in range(n_bins):
        xr = reattachment_length(centroid, u_x, z_slab=(edges[i], edges[i + 1]))
        out.append((0.5 * (edges[i] + edges[i + 1]), xr))
    return out


def main():
    if not (RUNS / "polyMesh").exists():
        raise SystemExit(f"OpenFOAM mesh not found in {RUNS}; run of_case/run_of.sh first.")
    if not (TRANSIENT / "U").exists():
        raise SystemExit(
            f"Transient reference not found in {TRANSIENT}; run of_transient/run_transient.sh first."
        )
    of = read_openfoam_reference()
    print(
        f"OpenFOAM: {of['centroid'].shape[0]} cells, Ux in "
        f"[{of['U'][:, 0].min():.3f}, {of['U'][:, 0].max():.3f}], "
        f"Uz in [{of['U'][:, 2].min():.3f}, {of['U'][:, 2].max():.3f}] (3D)",
        flush=True,
    )

    t0 = time.time()
    aq = solve_aquaflux(log_path=HERE / "march.log", checkpoint_dir=HERE / "checkpoints")
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

    slab = mid_span_slab(aq["centroid"])

    metrics = dict(
        ux=rel_l2(aq["U"][:, 0], of["U"][idx, 0], U_IN),
        uy=rel_l2(aq["U"][:, 1], of["U"][idx, 1], U_IN),
        uz=rel_l2(aq["U"][:, 2], of["U"][idx, 2], U_IN),
        xr_aqua_mid=reattachment_length(aq["centroid"], aq["U"][:, 0], z_slab=slab),
        xr_of_mid=reattachment_length(of["centroid"], of["U"][:, 0], z_slab=slab),
        nut_peak_aqua=float(aq["nut"].max() / NU),
        nut_peak_of=float(of["nut"].max() / NU),
    )
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}", flush=True)
    print("  spanwise x_r/h (z, aquaflux, OpenFOAM):", flush=True)
    aq_span = spanwise_reattachment(aq["centroid"], aq["U"][:, 0])
    of_span = spanwise_reattachment(of["centroid"], of["U"][:, 0])
    for (zc_i, xa), (_, xo) in zip(aq_span, of_span, strict=True):
        print(f"    z={zc_i:.4f}  aqua {xa:.2f}  OF {xo:.2f}", flush=True)

    _figure(of, aq, idx, slab)
    _report(metrics, aq_span, of_span)


def _figure(of, aq, idx, slab):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = aq["centroid"][:, 0], aq["centroid"][:, 1], aq["centroid"][:, 2]
    # Mid-span slice for the 2D-style Ux comparison.
    mid = (z >= slab[0]) & (z <= slab[1])
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    vmax = max(abs(of["U"][:, 0]).max(), abs(aq["U"][:, 0]).max())
    for a, data, title in (
        (ax[0], of["U"][idx, 0][mid], "OpenFOAM $U_x$ (mid-span)"),
        (ax[1], aq["U"][:, 0][mid], "aquaflux $U_x$ (mid-span)"),
    ):
        sc = a.scatter(x[mid], y[mid], c=data, s=6, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    band = (x > 1e-4) & (y < -H + 0.002) & (y > -H) & mid
    order = np.argsort(x[band])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(
        x[band][order] / H,
        of["U"][idx][band][order][:, 0],
        "s-",
        ms=3,
        color="C1",
        label="OpenFOAM",
    )
    ax[2].plot(
        x[band][order] / H, aq["U"][band][order][:, 0], ".-", ms=4, color="C0", label="aquaflux"
    )
    ax[2].set_xlabel("$x/h$ behind the step")
    ax[2].set_ylabel("near-wall $U_x$ (mid-span)")
    ax[2].set_title("Lower-wall recirculation (sign change = reattachment)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "comparison.png", dpi=130)
    print(f"wrote {FIGS / 'comparison.png'}", flush=True)


def _report(m, aq_span, of_span):
    span_rows = "\n".join(
        f"| {za:.4f} | {xa:.2f} | {xo:.2f} |"
        for (za, xa), (_, xo) in zip(aq_span, of_span, strict=True)
    )
    lines = [
        "# 3D backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST",
        "",
        "A finite-width 3D backward-facing step (step height h = 0.01 m, expansion ratio 2, span 4h",
        "between no-slip side walls), solved by OpenFOAM `incompressibleFluid` (kOmegaSST) and, on the",
        "**same imported mesh**, by aquaflux's coupled RANS solver (hybrid initialization, second-order",
        "upwind momentum advection, corrected Green-Gauss gradients, log-omega, field-split",
        "algebraic-multigrid preconditioner). U_in = 10 m/s, nu = 1e-5 -> Re_h = 10000.",
        "",
        "## Results",
        "",
        "| quantity | aquaflux | OpenFOAM |",
        "|---|---|---|",
        f"| reattachment length x_r/h (mid-span) | {m['xr_aqua_mid']:.2f} | {m['xr_of_mid']:.2f} |",
        f"| peak nu_t/nu | {m['nut_peak_aqua']:.0f} | {m['nut_peak_of']:.0f} |",
        f"| rel. L2 U_x error (cell-for-cell) | {m['ux']:.3f} | -- |",
        f"| rel. L2 U_y error (cell-for-cell) | {m['uy']:.3f} | -- |",
        f"| rel. L2 U_z error (cell-for-cell) | {m['uz']:.3f} | -- |",
        "",
        "### Spanwise reattachment (the 3D structure)",
        "",
        "| z (m) | aquaflux x_r/h | OpenFOAM x_r/h |",
        "|---|---|---|",
        span_rows,
        "",
        "See `figures/comparison.png`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image)",
        "cd validation/bfs3d_openfoam",
        'docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh',
        'docker run --rm -v "$PWD":/work -w /work/of_transient openfoam13:latest bash run_transient.sh',
        "# 2. aquaflux coupled solve + comparison (from the repo root)",
        "cd ../..",
        "python3 validation/bfs3d_openfoam/compare.py",
        "```",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
