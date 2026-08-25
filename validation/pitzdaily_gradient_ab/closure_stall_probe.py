"""Why does ``SkewCorrectedGradient`` find no descent direction on pitzDaily's target rung?

``MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient(), fallback=None)`` clears the two
lower Reynolds rungs of ``run_ab.py`` at a full step, matching the ``OwnerGradient`` arm's residual to
three significant figures, and then finds no usable step on the **first step of the target rung**: the
line search collapses to the smallest rung of its ladder and the residual freezes. Three mechanisms
were proposed and refuted before this probe was written (the near-wall ``omega`` gradient, the
correction matrices' conditioning, and the closure's double correction on gradient-type patches), and
all three missed for the same reason: every one was measured at the cold initial condition or on a
single reconstruction, and the two closures are **identical** at every state either of those reaches.

What this probe found, and what it is kept to re-ask. The arms are identical at the target rung's
anchor too, and part at the **second inner Newton iteration** of that rung's first step -- so the
reported per-step ``alpha`` is a *minimum over inner iterations* and the anchor says nothing. At that
iterate the correction's ``log omega`` component at boundary-owning cells reaches order ten to
seventy, while the 11670 interior cells agree to three figures; since ``omega`` is transported in log
form that becomes many orders in the residual, and halving the step length only takes the square root
of the factor, so the ladder cannot recover a step worth taking.

⚠️ **Run more than two arms, and take the ranking only where a usable step exists.** Two arms of one
scheme cannot distinguish "the closure does this" from "this state is fragile"; four can, and
``PITZ_STALL_ARMS=owner,skew,gauss,betchen`` is what settled it -- the interior maximum is flat to 4 %
across all four while the boundary and corner maxima span thirtyfold, ordering exactly as the kept
step length does. Two wrong readings were written up on the way and both are worth avoiding: the first
attributed everything to the closure from a **single** shift, which is under-determined; the second
read a ``PITZ_STALL_BETAS`` sweep as refuting the closure entirely, on an alternation between two arms
that had **both already failed** (every ratio below a shift of 0.45 is 0.98--0.998, i.e. no usable
step in either). Ranking two dead configurations is not evidence.

It runs in three modes:

``capture``
    March the chosen arm and write each Reynolds rung's **seed state** to ``stall-seed-<n>.npz``,
    stopping the moment the target rung's seed exists. That state is the one input every question below
    needs, and it costs a rung-and-a-half of marching to obtain.

``march``
    Drive the real :class:`~aquaflux.solve.DualTimeStep` from that seed, one line per **inner**
    iteration, and write out the iterate the first clipped step was taken from. This is what located
    the failure; a per-step log cannot, because it reports the inner minimum.

``analyze``
    At one saved state (``PITZ_STALL_STATE``), with everything else held identical, compare the two
    closures on the four quantities that separate "the step is wrong" from "the step is right and the
    landscape is bad":

    1. the coupled residual per block, Euclidean and row-scaled;
    2. what the closure reports as a normal derivative on the **flow** block's boundary faces;
    3. the Jacobian action along a fixed random tangent;
    4. the shifted correction itself -- its **true** linear residual (did the solve converge?), its
       descent slope ``d/ds |G(phi + s delta)|``, and the whole line-search ladder profile.

The fourth is the decisive one, and it came out a landscape result: every linear solve converges (true
relative residual 1e-11 to 1e-4), the descent slope is negative and **the same in both arms**, and it
is the finite-step curvature that differs -- the full step multiplies the measure by 7.6e+03 under the
owner closure and 1.4e+23 under this one, entirely in the ``omega`` block.

⚠️ **One place this is not the march**: it rebuilds ``coupled_scaled_norm`` at the state it measures,
where a march holds the anchor's measure fixed across a step. That moves which rung the ladder keeps,
so do not quote this probe's ``alpha`` as the march's -- take those from ``march`` mode, which uses the
march's own step. Every per-block and per-cell figure is unaffected.

Usage
-----
    PITZ_STALL_CLOSURE=skew validation/run_case.sh validation/pitzdaily_gradient_ab/closure_stall_probe.py
    PITZ_STALL_MODE=analyze validation/run_case.sh validation/pitzdaily_gradient_ab/closure_stall_probe.py

``PITZ_STALL_CLOSURE`` selects the arm to capture with (``skew`` or ``owner``); the two rungs before the
failure are step-for-step alike, so either seed is usable and ``skew``'s is the exact one. ``analyze``
always compares both closures at whichever seed was captured.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[0] / "pitzdaily_openfoam"))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import run_ab  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    CorrectedGreenGauss,
    HessianCorrectedGradient,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.schemes.interpolation import non_orthogonal_correction  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    MarchLogger,
    RefreshPolicy,
    jacobi_smoothed_inverse,
    simple_smoothed_inverse,
    solve_linear,
)
from aquaflux.solve.implicit import backtracking_line_search  # noqa: E402
from aquaflux.turbulence import (  # noqa: E402
    CoupledJacobianProbe,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    omega_wall,
    production_and_limit,
    solve_coupled,
    solve_reynolds_continuation,
    strain_rate_magnitude,
)
from aquaflux.turbulence.coupled import coupled_scaled_norm, positive_k_limit  # noqa: E402
from aquaflux.vectors import dot  # noqa: E402

#: Every reconstruction this case can run, so the comparison is not confined to the two that differ
#: only in a closure. `gauss` and `betchen` both march this benchmark to the same answer, so including
#: them turns "which closure is worse" into "does the reconstruction decide this at all" -- a question
#: two arms of one scheme cannot answer.
#:
#: `fallback=None` on the two multiple-correction arms is what makes their closure GLOBAL. The class
#: default repairs per cell, which on this quadrilateral mesh finds nothing to repair and so silently
#: gives the owner reconstruction back -- i.e. the configuration under study is otherwise unreachable.
SCHEMES = {
    "owner": lambda: MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
    "skew": lambda: MultipleCorrectionGradient(
        boundary_closure=SkewCorrectedGradient(), fallback=None
    ),
    "gauss": CorrectedGreenGauss,
    "betchen": HessianCorrectedGradient,
}
CLOSURE = os.environ.get("PITZ_STALL_CLOSURE", "skew")


def multicorr(name: str):
    """The reconstruction an arm runs, by name."""
    return SCHEMES[name]()


SEED = HERE / f"stall-seed-{CLOSURE}.npz"

#: The block names of the coupled layout, for reporting. pitzDaily is 2D, so `[u, v, p, k, omega]`.
BLOCKS = ("u", "v", "p", "k", "omega")

#: Which arms a run drives, and which saved state it reads. Both exist so a question about one arm at
#: one state costs one arm at one state -- the marches here are minutes each.
ARMS = tuple(os.environ.get("PITZ_STALL_ARMS", "owner,skew").split(","))
#: The arm every cross-arm figure is quoted against. `owner` is the natural reference: it is the
#: closure this scheme ships and the one that marches the case.
BASELINE = os.environ.get("PITZ_STALL_BASELINE", "owner")
STATE = HERE / os.environ.get("PITZ_STALL_STATE", SEED.name)

#: The shifts to measure the step at: the case's own `beta_start` and the rungs the retry ladder
#: escalates through to its cap. The observed stall walks all of them, so measuring one would leave
#: "it just needed more damping" standing.
BETA_LADDER = tuple(
    float(v) for v in os.environ.get("PITZ_STALL_BETAS", "0.5,2.0,8.0,16.0").split(",")
)


class _Captured(Exception):
    """The target rung's seed is on disk; nothing after it is wanted."""


def capture() -> None:
    """March the ramp until the target rung's seed state exists, then stop.

    The seed is taken from ``point_setup``, which the continuation calls for every point with that
    point's **packed** coupled state -- so it is the solved-variable state the next rung starts from,
    not a physical-field bundle needing a transform. Reproducing it any other way would mean re-running
    the ramp, which is exactly the cost this exists to pay once.
    """
    scheme = multicorr(CLOSURE)
    print(f"capturing the target-rung seed with the {CLOSURE} closure -> {SEED.name}", flush=True)
    case = compare.build_case(gradient_scheme=scheme)
    coupled = case["coupled"]
    log_path = HERE / f"stall-capture-{CLOSURE}.log"
    log_file = open(log_path, "w")
    logger = MarchLogger(
        log_file,
        fields=coupled_fields(coupled),
        detail=("inner", "fields"),
        rtol=compare.RTOL,
        atol=compare.ATOL,
    )
    probe = CoupledJacobianProbe.build(
        coupled, stencil_reach=run_ab.REACH, column_reach=run_ab.COLUMN_REACH
    )
    refresh = amg_beta_tracking_refresh(
        coupled,
        probe=probe,
        beta_rel_change=float("inf"),
        refresh_every=10**9,
        materialize_drift=None,
        materialize_every=None,
        beta_floor=compare.PC_BETA_FLOOR,
        observer=logger.on_refresh,
    )
    shared: list = []

    def point_setup(companion, seed_state, point):
        logger.note(f"[{point.label}]")
        print(f"  reached {point.label}", flush=True)
        if point.is_target:
            np.savez(SEED, state=np.asarray(seed_state))
            raise _Captured
        refresh.rebind(companion)
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
            probe=probe,
            cycle_budget=compare.CYCLE_BUDGET,
            forward_rtol=compare.FORWARD_RTOL,
            forward_restart=compare.FORWARD_RESTART,
            forward_max_restarts=compare.FORWARD_MAX_RESTARTS,
            refresh_on_cycles=compare.REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if compare.REFRESH_ON_CYCLES else None,
            positivity_floor=compare.K_POSITIVITY_FLOOR,
            positivity_projection=compare.POSITIVITY_PROJECTION,
            preconditioner=shared[0] if shared else None,
            coarse_eq_limit=run_ab.SIMPLE_FLOW["max_coarse"],
            field_split=True,
            leading_inverse=simple_smoothed_inverse(**run_ab.SIMPLE_FLOW),
            trailing_inverse=jacobi_smoothed_inverse(**run_ab.JACOBI_TRAILING),
            inner_observer=logger.on_inner,
        )
        shared[:] = [engine.shift_policy.preconditioner]
        return dict(continuation=engine, refresh=RefreshPolicy(precondition_step=refresh))

    started = time.perf_counter()
    try:
        solve_reynolds_continuation(
            coupled,
            compare.N_POINTS,
            max_steps=compare.MAX_STEPS,
            rtol=compare.RTOL,
            atol=compare.ATOL,
            intermediate_rtol=None,
            intermediate_atol=compare.ATOL,
            step_control=compare.CONTROL,
            retry=compare.RETRY,
            point_setup=point_setup,
            scaled_norm=True,
            on_checkpoint=logger.on_checkpoint,
            on_retry=logger.on_retry,
        )
    except _Captured:
        pass
    finally:
        log_file.close()
    print(f"  wrote {SEED} in {time.perf_counter() - started:.1f} s", flush=True)


def _engine_at(coupled, state):
    """The march's own shifted step at ``state``, built exactly as the case builds it.

    Built through ``coupled_amg_continuation`` rather than assembled here, so the shift diagonal, the
    preconditioner and the relaxation schedule are the ones the march uses rather than a second set
    that could differ from them.
    """
    probe = CoupledJacobianProbe.build(
        coupled, stencil_reach=run_ab.REACH, column_reach=run_ab.COLUMN_REACH
    )
    return coupled_amg_continuation(
        coupled,
        state,
        inner_steps=compare.INNER_STEPS,
        inner_tol=compare.INNER_TOL,
        probe=probe,
        cycle_budget=compare.CYCLE_BUDGET,
        forward_rtol=compare.FORWARD_RTOL,
        forward_restart=compare.FORWARD_RESTART,
        forward_max_restarts=compare.FORWARD_MAX_RESTARTS,
        positivity_floor=compare.K_POSITIVITY_FLOOR,
        positivity_projection=compare.POSITIVITY_PROJECTION,
        coarse_eq_limit=run_ab.SIMPLE_FLOW["max_coarse"],
        field_split=True,
        leading_inverse=simple_smoothed_inverse(**run_ab.SIMPLE_FLOW),
        trailing_inverse=jacobi_smoothed_inverse(**run_ab.JACOBI_TRAILING),
    )


def _blocks(coupled, vector):
    """A coupled vector split into ``[u, v, p, k, omega]`` per-block arrays."""
    flow, k, omega = coupled.layout.unpack(vector)
    n, dim = coupled.layout.n_cells, coupled.layout.dim
    velocity, pressure = flow[: dim * n].reshape(n, dim), flow[dim * n :]
    return [velocity[:, i] for i in range(dim)] + [pressure, k, omega]


def _report_residual(name, coupled, state, measure) -> None:
    """The coupled residual at ``state``, per block, in both measures the march ever reads."""
    residual = coupled.residual(state)
    per_block = measure.per_block(residual)
    print(f"  {name:<6}", end="")
    for block in _blocks(coupled, residual):
        print(f" {float(jnp.linalg.norm(block)):>11.4e}", end="")
    print(f"  | scaled {float(measure(residual)):.4e}  per-block ", end="")
    print(" ".join(f"{float(v):.3e}" for v in per_block), flush=True)


def _flow_boundary_census(coupled, state) -> None:
    """What the skew closure reports as the normal derivative on the FLOW block's boundary faces.

    The same measurement ``validation/multiple_correction/boundary_closure_probe.py`` makes for the
    scalar transport equations, made here for pressure and velocity -- which that probe never
    reaches, and which take a different route to their boundary values. A scalar equation's
    reconstruction is fed its closures through
    :meth:`~aquaflux.discretization.ResidualAssembler.boundary_values`; the flow block's are
    :meth:`~aquaflux.flow.FlowBoundary.pressure_face` / ``velocity_face``, evaluated at the
    reconstructed gradient by its own two-pass fold. Both therefore carry the tangential
    non-orthogonal correction, and the reported normal derivative below is the *field's*.

    Read the ``max |bval - phi_P|`` column first. On a gradient-type patch it is the correction
    itself, so a **zero** there says the closures are being read at a zero gradient -- and every
    number to its right is then an artifact rather than a measurement, since the rise the closure
    divides by ``d.n`` is a term nothing added.
    """
    momentum, mesh, geometry = coupled.momentum, coupled.momentum.mesh, coupled.momentum.geometry
    fields = momentum.flow_fields(coupled.layout.unpack(state)[0])
    face_cells = mesh.face_cells
    owner, normal = np.asarray(face_cells.owner), geometry.face.normal
    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    along = np.asarray(dot(displacement, normal))

    dim = mesh.dim
    probed = [("p", fields.pressure, fields.boundary_pressure, fields.grad_pressure)]
    velocity = fields.velocity_fields
    probed += [
        (
            f"U{i}",
            velocity.velocity[:, i],
            velocity.boundary_velocity[:, i],
            velocity.gradient[:, i],
        )
        for i in range(dim)
    ]

    print(
        f"\n  {'field':<6} {'patch':<12} {'faces':>6} {'max |bval - phi_P|':>19} "
        f"{'med |d.n|':>11} {'med |rise/d.n|':>15} {'max':>11} {'med |grad phi|':>15}",
        flush=True,
    )
    for label, cell_values, boundary_values, gradient in probed:
        correction = np.asarray(
            non_orthogonal_correction(gradient[face_cells.owner], displacement, normal)
        )
        difference = np.asarray(boundary_values) - np.asarray(cell_values)[owner]
        rise = difference - correction
        magnitude = np.linalg.norm(np.asarray(gradient), axis=-1)
        for name in mesh.face_patches.names:
            faces = np.asarray(mesh.face_patches.indices(name))
            if faces.size == 0 or name == "interior":
                continue
            derivative = np.abs(rise[faces] / along[faces])
            print(
                f"  {label:<6} {name:<12} {faces.size:>6} "
                f"{np.abs(difference[faces]).max():>19.3e} "
                f"{np.median(np.abs(along[faces])):>11.3e} "
                f"{np.median(derivative):>15.3e} {derivative.max():>11.3e} "
                f"{np.median(magnitude[owner[faces]]):>15.3e}",
                flush=True,
            )


def _flow_gradient_difference(built, state) -> None:
    """The flow block's reconstructed gradients, skew against owner, resolved by wall proximity.

    The causal path the census above only implies: the pressure gradient is what Rhie--Chow and the
    momentum pressure force read, so a boundary artifact matters exactly insofar as it reaches this.
    Split by whether the cell owns a boundary face, since that is where the two closures can differ at
    all -- a difference spread over the interior would mean something else is going on.
    """
    fields = {
        name: coupled.momentum.flow_fields(coupled.layout.unpack(state)[0])
        for name, coupled in built.items()
    }
    mesh = built["owner"].momentum.mesh
    owner = np.asarray(mesh.face_cells.owner)
    interior = np.asarray(mesh.face_cells.interior)
    at_boundary = np.zeros(mesh.n_cells, dtype=bool)
    at_boundary[owner[~interior]] = True

    dim = mesh.dim
    probed = [("grad p", "grad_pressure", None)] + [
        (f"grad U{i}", "velocity_fields", i) for i in range(dim)
    ]
    print(
        f"\n  {'field':<9} {'region':<10} {'cells':>7} {'rel L2':>12} {'rel max':>12} "
        f"{'cells >1%':>10}",
        flush=True,
    )
    for label, attribute, component in probed:
        pair = []
        for name in ("owner", "skew"):
            value = getattr(fields[name], attribute)
            pair.append(np.asarray(value if component is None else value.gradient[:, component]))
        reference, other = pair
        for region, mask in (("boundary", at_boundary), ("interior", ~at_boundary)):
            a, b = other[mask], reference[mask]
            rows = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-300)
            print(
                f"  {label:<9} {region:<10} {int(mask.sum()):>7} "
                f"{np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300):>12.3e} "
                f"{np.abs(a - b).max() / max(np.abs(b).max(), 1e-300):>12.3e} "
                f"{int((rows > 0.01).sum()):>10}",
                flush=True,
            )


def _corrected_flow_pressure_gradient(built, state) -> None:
    """How far the flow block's CORRECTED boundary pressure moves its reconstruction.

    The flow closures take the owner's reconstructed gradient, so the boundary pressure the
    reconstruction and the fluxes read carries its own tangential non-orthogonal correction; the
    leading-order value (the closures at a **zero** gradient, which is what feeds the first pass) is
    built here by hand so the two can be compared at one state. On a gradient-type patch they differ
    by exactly ``non_orthogonal_correction(g_owner, d, n)``, and the census above is where that
    difference shows up as a reported normal derivative.

    ``chain = d(p_face)/d(p_owner)`` -- one on a gradient-type patch, zero on a prescribed one -- is
    read off the case's own closures by differentiating them rather than by declaring which is which,
    and reported to say how many faces the correction can reach at all.
    """
    momentum = built["skew"].momentum
    mesh, geometry = momentum.mesh, momentum.geometry
    face_cells = mesh.face_cells
    fields = {
        name: coupled.momentum.flow_fields(coupled.layout.unpack(state)[0])
        for name, coupled in built.items()
    }
    pressure = fields["skew"].pressure
    normal = geometry.face.normal

    def boundary_pressure(p, gradient):
        return momentum.boundary.apply(
            face_cells,
            jnp.zeros(mesh.n_faces),
            lambda bc, faces, owner: bc.pressure_face(
                p[owner],
                gradient[owner],
                geometry.face.centroid[faces] - geometry.cell.centroid[owner],
                normal[faces],
                geometry.face.centroid[faces],
            ),
        )

    zero_gradient = jnp.zeros((mesh.n_cells, mesh.dim))
    leading = boundary_pressure(pressure, zero_gradient)
    chain = jax.jvp(
        lambda p: boundary_pressure(p, zero_gradient), (pressure,), (jnp.ones_like(pressure),)
    )[1]
    # The leading-order arm the shipped two-pass fold replaced: the same first pass, but with the
    # closures never re-read at the gradient it produces.
    uncorrected = momentum.gradient_scheme.gradients(
        pressure, mesh, geometry, leading, boundary_chain=chain
    )
    print(
        f"  d(p_face)/d(p_owner) is 1 on {int(jnp.sum(chain > 0.5))} boundary faces "
        f"(gradient-type) and 0 on {int(jnp.sum(jnp.abs(chain) < 0.5)) - int(face_cells.interior.sum())}"
        " (prescribed)",
        flush=True,
    )
    corrected_bvals = fields["skew"].boundary_pressure
    rise = np.abs(np.asarray(corrected_bvals - leading))[~np.asarray(face_cells.interior)]
    print(
        f"  the correction the closures now carry: max {rise.max():.3e}, med {np.median(rise):.3e} "
        "over the boundary faces",
        flush=True,
    )
    owner_grad = np.asarray(fields["owner"].grad_pressure)
    print(f"  {'grad p arm':<28} {'rel L2 vs owner':>17} {'rel max':>12}", flush=True)
    for label, value in (
        ("skew, leading-order bvals", np.asarray(uncorrected)),
        ("skew, corrected bvals (shipped)", np.asarray(fields["skew"].grad_pressure)),
    ):
        print(
            f"  {label:<28} "
            f"{np.linalg.norm(value - owner_grad) / max(np.linalg.norm(owner_grad), 1e-300):>17.3e} "
            f"{np.abs(value - owner_grad).max() / max(np.abs(owner_grad).max(), 1e-300):>12.3e}",
            flush=True,
        )


def analyze() -> None:
    """Compare the two closures at the captured seed on everything a stalled step depends on."""
    if not STATE.exists():
        raise SystemExit(f"no saved state at {STATE}; run this without PITZ_STALL_MODE first")
    state = jnp.asarray(np.load(STATE)["state"])
    print(f"analysing {STATE.name} ({state.size} degrees of freedom)", flush=True)

    # Built once per arm and reused: the coloured Jacobian probe inside the engine is the single
    # largest allocation this case makes, so rebuilding it per section would dominate the probe.
    built, engines, measures = {}, {}, {}
    for name in ARMS:
        built[name] = compare.build_case(gradient_scheme=multicorr(name))["coupled"]
        engines[name] = _engine_at(built[name], state)
        measures[name] = coupled_scaled_norm(built[name], engines[name].shift_policy, state)

    # --- 1. the residual, per block ------------------------------------------------------------
    print("\n=== coupled residual at the seed (Euclidean per block) ===", flush=True)
    print(f"  {'arm':<6}" + "".join(f" {b:>11}" for b in BLOCKS), flush=True)
    for name, coupled in built.items():
        _report_residual(name, coupled, state, measures[name])

    # --- 2. what the closure reports on the FLOW block's boundary faces --------------------------
    print("\n=== skew closure on the flow block's boundary faces (at this state) ===", flush=True)
    _flow_boundary_census(built["skew"], state)
    print("\n=== the flow block's reconstructed gradients, skew against owner ===", flush=True)
    _flow_gradient_difference(built, state)
    print(
        "\n=== the flow block's pressure gradient with a CORRECTED boundary value ===", flush=True
    )
    _corrected_flow_pressure_gradient(built, state)

    # --- 3. the Jacobian action ----------------------------------------------------------------
    # One fixed tangent for both arms: a random direction reads every coupling, where the residual
    # alone reads only the state's own.
    tangent = jax.random.normal(jax.random.PRNGKey(0), state.shape, dtype=state.dtype)
    actions = {
        name: jax.jvp(coupled.residual, (state,), (tangent,))[1] for name, coupled in built.items()
    }
    print("\n=== Jacobian action along one fixed random tangent, per block ===", flush=True)
    base = BASELINE if BASELINE in built else next(iter(built))
    others = [name for name in built if name != base]
    header = "  ".join(f"{'rel diff ' + name:>18}" for name in others)
    print(f"  {'block':<8} {base:>14}  {header}", flush=True)
    columns = {name: _blocks(built[name], actions[name]) for name in built}
    for index, label in enumerate(BLOCKS):
        reference = np.asarray(columns[base][index])
        scale = max(float(np.linalg.norm(reference)), 1e-300)
        diffs = "  ".join(
            f"{float(np.linalg.norm(np.asarray(columns[name][index]) - reference)) / scale:>18.3e}"
            for name in others
        )
        print(f"  {label:<8} {float(np.linalg.norm(reference)):>14.4e}  {diffs}", flush=True)

    # --- 4. the shifted step, its linear residual, and its ladder ------------------------------
    print("\n=== the first shifted step of the target rung ===", flush=True)
    steps = {
        name: _report_step(name, coupled, state, engines[name], measures[name])
        for name, coupled in built.items()
    }

    # --- 5. which cells carry the difference ----------------------------------------------------
    print("\n=== the cells carrying the omega correction at the lowest shift ===", flush=True)
    _omega_hotspots(built["owner"], state, steps, BETA_LADDER[0])

    # --- 6. is that correction the omega equation's, or the k equation's? -----------------------
    print("\n=== the near-wall omega fixation rows ===", flush=True)
    _wall_fixation_identity(built, state, steps, BETA_LADDER[0])

    # --- 7. and why is the k correction that large? ---------------------------------------------
    print("\n=== the k row at those cells, term by term ===", flush=True)
    _k_row_anatomy(built, engines, state, steps, BETA_LADDER[0])

    # --- 8. does the Patankar treatment of that term fix it? -------------------------------------
    print("\n=== exact operator against a frozen-production operator ===", flush=True)
    _frozen_production_test(built, engines, state, steps, measures, BETA_LADDER[0])


def _report_step(name, coupled, state, engine, measure) -> None:
    """One arm's shifted correction: did the LINEAR solve converge, and does the direction descend?

    The two questions are deliberately answered together and in that order. A ladder with no admissible
    rung means one thing if the correction solves its own linear system and quite another if it does
    not, and every earlier account of this stall was written without knowing which.
    """
    residual = coupled.residual(state)
    reference = float(measure(residual))
    print(f"  [{name}] |R| scaled at the seed {reference:.4e}", flush=True)
    # Swept over the shift the march itself visits, not one value: the observed stall escalates beta
    # from its start to the ladder's 16.0 cap without ever finding a step, so a single beta could
    # only ever reproduce one rung of that and would leave the other explanations open.
    return {
        beta: _report_step_at(coupled, state, engine, measure, residual, reference, beta)
        for beta in BETA_LADDER
    }


def _report_step_at(coupled, state, engine, measure, residual, reference, beta) -> None:
    """One arm at one shift: the correction, its true linear residual, and its ladder."""
    term = engine.shift_policy.shift_term(state)
    shift = beta * term.diagonal
    preconditioner = term.make_preconditioner(jnp.asarray(beta))

    def operator(tangent):
        return jax.jvp(coupled.residual, (state,), (tangent,))[1] + shift * tangent

    delta, cycles = solve_linear(
        operator,
        -residual,
        solver=engine.default_solver(),
        preconditioner=preconditioner,
        throw=False,
    )
    linear = float(jnp.linalg.norm(operator(delta) + residual) / jnp.linalg.norm(residual))

    # G(p) = R(p) + beta d (p - state): at the anchor the transient term vanishes, so `reference` is
    # both the honest steady residual and the inner loop's own starting norm -- the same number the
    # march's line search is handed.
    def transient(p):
        return coupled.residual(p) + shift * (p - state)

    slope = float(jax.jvp(lambda p: measure(transient(p)), (state,), (delta,))[1])
    # The march caps every rung by the fraction-to-the-boundary limit that keeps `k` positive, so a
    # search run without it is not the march's search. An alpha of 0.001 can be the ladder finding no
    # admissible rung OR this cap binding, and those are different failures with different cures.
    limit = positive_k_limit(coupled, floor=compare.K_POSITIVITY_FLOOR)
    max_alpha = 1.0 if limit is None else limit(state, delta)
    searched = backtracking_line_search(
        transient,
        state,
        delta,
        jnp.asarray(reference),
        engine.line_search,
        norm=measure,
        max_alpha=max_alpha,
    )
    profile = " ".join(
        f"{float(measure(transient(state + 0.5**power * delta))) / reference:>8.4f}"
        for power in (0, 1, 2, 4, 6, 8, 10)
    )
    # Which equation the full step wrecks, and how far the correction moves each block to do it.
    # The descent slope is identical between the arms, so the overshoot is curvature -- and curvature
    # is a property of one block or of all of them, which is the difference between a closure defect
    # and a globalization one.
    at_full = measure.per_block(transient(state + delta))
    blocks = _blocks(coupled, delta)
    parts = " ".join(
        f"{label} {float(jnp.linalg.norm(block)):.2e}/{float(value):.2e}"
        for label, block, value in zip(BLOCKS, blocks, at_full, strict=True)
    )
    # `omega` is transported as `log omega`, so its correction is a LOG increment: an entry of 20 is a
    # factor of e**20. Where those entries sit decides whether the difference between the closures is a
    # boundary effect -- which is the only place the two can differ at all -- or something spread.
    mesh = coupled.momentum.mesh
    at_boundary = np.zeros(mesh.n_cells, dtype=bool)
    at_boundary[np.asarray(mesh.face_cells.owner)[~np.asarray(mesh.face_cells.interior)]] = True
    d_omega = np.abs(np.asarray(blocks[-1]))
    omega_note = (
        f"max |d log omega| {d_omega.max():.3e} (boundary cells {d_omega[at_boundary].max():.3e}, "
        f"interior {d_omega[~at_boundary].max():.3e}); "
        f"{int((d_omega > 5.0).sum())} cells above 5 "
        f"({int((d_omega[at_boundary] > 5.0).sum())} of them at a boundary)"
    )
    print(
        f"    beta {beta:>7.4g}  linear rel {linear:>9.3e} ({int(cycles):>2} cyc)  "
        f"|delta| {float(jnp.linalg.norm(delta)):>10.4e}  slope {slope:>+11.4e}  "
        f"k cap {float(max_alpha):>9.4g}  "
        f"kept alpha {float(searched.alpha):>8.4g} -> {float(searched.residual_norm) / reference:.6f}",
        flush=True,
    )
    print(f"      |G|/|G0| at alpha 1, 1/2, 1/4, 1/16, 1/64, 1/256, 1/1024: {profile}", flush=True)
    print(f"      per block |delta| / scaled |G| at alpha 1:  {parts}", flush=True)
    print(f"      {omega_note}", flush=True)
    return delta


#: How many outer steps of the target rung to drive in ``march`` mode, and the inner line-search
#: factor below which an iterate is written out. The stall reports its step's MINIMUM inner alpha, so
#: a step that opens at alpha = 1 and collapses later still reads as `alpha 0.001` in the log -- which
#: is why the anchor state says nothing and the collapsing inner iterate has to be caught here.
MARCH_STEPS = int(os.environ.get("PITZ_STALL_STEPS", "4"))
CLIPPED_ALPHA = 0.5


#: The cells the omega hotspot table names, ranked by the baseline arm. Reported in full by
#: :func:`_k_row_anatomy` so the chain can be read term by term at the cell that decides the step.
HOTSPOT_COUNT = 3


def frozen_production_residual(coupled):
    """``CoupledRANS.residual`` with ``nu_t``'s ``k``-dependence frozen inside the ``k`` equation.

    The Patankar treatment of the one term that drives the ``k`` row's Jacobian diagonal negative.
    ``nu_t = a_1 k / max(a_1 omega, S F_2)`` is proportional to ``k``, so the production ``nu_t S**2``
    is too, and subtracting a source that grows with its own variable puts a negative term on that
    row's diagonal -- measured at ``-1.2e-03`` to ``-1.9e-03`` against a pseudo-time shift of
    ``+2.3e-03``, so the effective diagonal is a near-cancellation and amplifies anything upstream of
    it.

    This rebuilds the coupled residual **line for line as the library assembles it**, changing exactly
    one thing: the eddy viscosity handed to the ``k`` equation is evaluated at a ``stop_gradient``-ed
    ``k``. Momentum and the ``omega`` equation keep the live closure, and ``omega`` is live in the
    frozen viscosity too, so nothing but that one derivative is removed.

    ⚠️ Intended as the **operator only** -- ``_shifted_solve``'s ``jacobian_fn``, which forms the
    Krylov matvec while the true residual still decides where the march lands. Used that way the root
    and the implicit-function-theorem adjoint are untouched. Do **not** substitute it for the residual
    itself: that is the adjoint hazard the production limiter's own flag is on record for.

    ⚠️ It also does not reach every ``k``-dependence of the production. The adaptive near-wall
    blend (``NearWallKClosure.production``) reads the solved ``k`` directly, and the Menter cap reads
    it unless ``explicit_production_limiter`` is set. This freezes the ``nu_t`` path, which is the one
    the diagonal measurement implicates.
    """

    def residual(state):
        flow, k, omega = coupled.physical_fields(state)
        velocity = coupled.momentum.velocity_fields(flow)
        closure = coupled.turbulence.closure_fields(velocity, k, omega)
        momentum = coupled.momentum.with_eddy_viscosity(
            closure.nu_t, coupled.turbulence.wall_face_eddy_viscosity(k)
        )
        fields = momentum.flow_fields(flow)
        flow_residual = momentum.residual_from_fields(fields)
        # The single change: the eddy viscosity the k equation sees carries no k derivative.
        frozen_nu_t = coupled.turbulence.eddy_viscosity(
            velocity.gradient, jax.lax.stop_gradient(k), omega
        )
        k_closure = eqx.tree_at(lambda c: c.nu_t, closure, frozen_nu_t)
        k_residual = coupled.turbulence.k_residual(fields.mdot, k_closure)(k)
        omega_residual = coupled.turbulence.omega_residual(
            fields.mdot, closure, coupled.omega_transform.fixation_row()
        )(omega)
        return coupled.layout.pack(flow_residual, k_residual, omega_residual)

    return residual


def _frozen_production_test(built, engines, state, steps, measures, beta) -> None:
    """Does freezing that one derivative make the ``k`` row's diagonal positive, and the step usable?

    Reports both arms twice -- exact operator against frozen-production operator -- on the quantities
    the chain runs through: the ``k`` row's own diagonal, the effective diagonal once the shift is
    added, the ``k`` correction, the ``log omega`` correction the wall fixation then carries, and the
    step the line search keeps.

    The **residual is identical in both columns by construction** (only its derivative differs), which
    is the control: a difference in the reported residual would mean the stand-in is not the same
    function and nothing else here would be readable.
    """
    print(
        f"  {'arm':<9} {'operator':<10} {'cell':>6} {'|R| scaled':>12} {'J_kk':>12} {'J_kk+beta d':>13} "
        f"{'max |dk/k|':>12} {'max |d log w|':>14} {'kept alpha':>11} {'|G|/|G0|':>10}",
        flush=True,
    )
    for name, coupled in built.items():
        engine = engines[name]
        measure = measures[name]
        residual = coupled.residual(state)
        reference = float(measure(residual))
        term = engine.shift_policy.shift_term(state)
        shift = beta * term.diagonal
        preconditioner = term.make_preconditioner(jnp.asarray(beta))
        wall = np.asarray(coupled.turbulence.wall_cells)
        k_wall = np.asarray(coupled.physical_fields(state)[1])[wall]
        # THE SAME cell `_k_row_anatomy` reports, ranked by the baseline arm's `log omega` correction,
        # so the two sections' `J_kk` columns are the same quantity. Ranking each arm by its own worst
        # cell instead would put a different row in each line, which is how a diagonal was first
        # reported here with the opposite sign to the one measured a section earlier.
        cell = int(np.argmax(np.abs(np.asarray(_blocks(coupled, steps[BASELINE][beta])[-1]))))
        unit = jnp.zeros_like(state).at[coupled.layout.flow_size + cell].set(1.0)

        for label, jacobian in (
            ("exact", coupled.residual),
            ("frozen", frozen_production_residual(coupled)),
        ):
            j_kk = float(_blocks(coupled, jax.jvp(jacobian, (state,), (unit,))[1])[-2][cell])
            damped = j_kk + float(_blocks(coupled, shift)[-2][cell])

            def operator(tangent, jacobian=jacobian, shift=shift):
                return jax.jvp(jacobian, (state,), (tangent,))[1] + shift * tangent

            delta, _ = solve_linear(
                operator,
                -residual,
                solver=engine.default_solver(),
                preconditioner=preconditioner,
                throw=False,
            )

            def transient(p, coupled=coupled, shift=shift):
                return coupled.residual(p) + shift * (p - state)

            limit = positive_k_limit(coupled, floor=compare.K_POSITIVITY_FLOOR)
            searched = backtracking_line_search(
                transient,
                state,
                delta,
                jnp.asarray(reference),
                engine.line_search,
                norm=measure,
                max_alpha=1.0 if limit is None else limit(state, delta),
            )
            d_k = np.asarray(_blocks(coupled, delta)[-2])[wall]
            d_omega = np.asarray(_blocks(coupled, delta)[-1])[wall]
            print(
                f"  {name:<9} {label:<10} {cell:>6} {reference:>12.4e} {j_kk:>12.4e} {damped:>13.4e} "
                f"{np.abs(d_k / k_wall).max():>12.4e} {np.abs(d_omega).max():>14.4e} "
                f"{float(searched.alpha):>11.4g} "
                f"{float(searched.residual_norm) / reference:>10.6f}",
                flush=True,
            )


def _k_row_anatomy(built, engines, state, steps, beta) -> None:
    """Term by term, why the ``k`` correction at a wall cell is thousands of times the local ``k``.

    The near-wall ``omega`` correction is measured to be the ``k`` correction handed through the wall
    fixation, so the question moves upstream: is ``dk`` large because the **velocity gradient** at
    those cells differs between reconstructions and feeds a different ``k`` production, or because the
    ``k`` row's own diagonal is too weak to damp whatever production it is given? Those are different
    faults with different cures, and every quantity that separates them is public:

    * ``|grad u|`` -- the reconstruction's own output, and the only thing that differs by construction.
    * the strain rate, raw and after the near-wall log-layer blend, since the blend can mask or
      amplify a gradient difference at exactly these cells.
    * ``nu_t`` and the ``k`` production, with the Menter cap alongside -- a capped production is
      insensitive to the strain rate, which would break the chain the hypothesis proposes.
    * the ``k`` residual, the ``k`` row's own Jacobian diagonal, and the pseudo-time shift on that row,
      whose ratio ``-R / (J_kk + beta d)`` is what a one-row Newton step would ask for.
    """
    reference = next(iter(built.values()))
    layout = reference.layout
    mesh = reference.momentum.mesh
    wall = np.asarray(reference.turbulence.wall_cells)
    # The cells the hotspot table names, so the two sections describe the same cells.
    ranked = np.argsort(-np.abs(np.asarray(_blocks(reference, steps[BASELINE][beta])[-1])))
    hotspots = [int(c) for c in ranked[:HOTSPOT_COUNT]]

    print(
        f"  {'arm':<9} {'cell':>6} {'|grad u|_F':>12} {'S raw':>11} {'S blended':>11} "
        f"{'nu_t':>11} {'P_k':>12} {'cap':>12} {'R_k':>12} {'J_kk':>11} {'beta*d_k':>11} "
        f"{'-R/(J+bd)':>12} {'actual dk':>12}",
        flush=True,
    )
    for name, coupled in built.items():
        flow = layout.unpack(state)[0]
        fields = coupled.momentum.flow_fields(flow)
        velocity = fields.velocity_fields
        _, k_field, omega_field = coupled.physical_fields(state)
        closure = coupled.turbulence.closure_fields(velocity, k_field, omega_field)
        raw_strain = np.asarray(strain_rate_magnitude(velocity.gradient))
        production, cap = production_and_limit(
            closure.nu_t, closure.strain_rate, closure.omega, k_field, coupled.turbulence.model
        )
        gradient_norm = np.linalg.norm(np.asarray(velocity.gradient), axis=(1, 2))
        residual_k = np.asarray(_blocks(coupled, coupled.residual(state))[-2])
        # The engine is the one already built for this arm -- rebuilding it here would re-pay the
        # coloured probe, which is the single largest allocation this case makes.
        shift = beta * np.asarray(
            _blocks(coupled, engines[name].shift_policy.shift_term(state).diagonal)[-2]
        )
        delta_k = np.asarray(_blocks(coupled, steps[name][beta])[-2])
        for cell in hotspots:
            # One directional derivative per cell: the tangent is a single 1 on that cell's k unknown,
            # so the k row of the response IS that row's own diagonal.
            tangent = jnp.zeros_like(state).at[layout.flow_size + cell].set(1.0)
            j_kk = float(
                _blocks(coupled, jax.jvp(coupled.residual, (state,), (tangent,))[1])[-2][cell]
            )
            damped = j_kk + shift[cell]
            print(
                f"  {name:<9} {cell:>6} {gradient_norm[cell]:>12.4e} {raw_strain[cell]:>11.4e} "
                f"{float(closure.strain_rate[cell]):>11.4e} {float(closure.nu_t[cell]):>11.4e} "
                f"{float(production[cell]):>12.4e} {float(cap[cell]):>12.4e} "
                f"{residual_k[cell]:>12.4e} {j_kk:>11.4e} {shift[cell]:>11.4e} "
                f"{-residual_k[cell] / damped if damped else float('nan'):>12.4e} "
                f"{delta_k[cell]:>12.4e}",
                flush=True,
            )
    wall_set = set(int(c) for c in wall)
    print(
        f"\n  the {HOTSPOT_COUNT} cells above are wall-fixed: "
        f"{[c in wall_set for c in hotspots]}; mesh has {mesh.n_cells} cells, "
        f"{len(wall)} wall-fixed",
        flush=True,
    )


def _wall_fixation_identity(built, state, steps, beta) -> None:
    """Is the wall cells' ``omega`` correction SLAVED to their ``k`` correction?

    The near-wall ``omega`` rows are not transport balances at all: they are replaced by the algebraic
    fixation ``log omega - log omega_wall(k)``, and ``omega_wall``'s log-layer branch carries
    ``sqrt(k)``. So that row's only two entries are a **1** on its own unknown and
    ``-d(log omega_target)/dk`` on ``k``, and the shift leaves such a row alone -- which makes the
    correction there an identity rather than something to be inferred:

        ``d log omega  =  -R_row  +  (d log omega_target / dk) * dk``

    with ``d log omega_target / dk`` of order ``1/(2k)``, since the branch goes as ``sqrt(k)``. If
    that identity holds, the wall cells' ``omega`` correction is not being *computed* by the omega
    equation at all -- it is whatever the ``k`` correction asks for, divided by twice a small ``k``.
    Reported per arm, so a reconstruction that perturbs ``k`` at a wall cell can be traced straight
    through to the exponent it produces.
    """
    reference = next(iter(built.values()))
    turbulence = reference.turbulence
    wall = np.asarray(turbulence.wall_cells)
    k_field = reference.physical_fields(state)[1]
    nu = turbulence.molecular_viscosity[turbulence.wall_cells]
    distance = turbulence.wall_distance[turbulence.wall_cells]

    def summed_log_target(values):
        return jnp.sum(jnp.log(omega_wall(nu, distance, values, turbulence.model)))

    # omega_wall is elementwise in k, so the gradient of the sum IS the per-cell derivative.
    chain = np.asarray(jax.grad(summed_log_target)(k_field[turbulence.wall_cells]))
    k_wall = np.asarray(k_field)[wall]

    print(
        f"  {len(wall)} wall-fixed cells; d(log omega_target)/dk median {np.median(chain):.3e}, "
        f"max {chain.max():.3e}; 1/(2k) median {np.median(1.0 / (2.0 * k_wall)):.3e}",
        flush=True,
    )
    print(
        f"  {'arm':<9} {'max |d log w|':>14} {'from the k term':>16} {'identity resid':>15} "
        f"{'max |dk/2k| at wall':>20}",
        flush=True,
    )
    for name, coupled in built.items():
        delta = steps[name][beta]
        residual = coupled.residual(state)
        row = np.asarray(_blocks(coupled, residual)[-1])[wall]
        d_omega = np.asarray(_blocks(coupled, delta)[-1])[wall]
        d_k = np.asarray(_blocks(coupled, delta)[-2])[wall]
        predicted = -row + chain * d_k
        scale = max(float(np.abs(d_omega).max()), 1e-300)
        print(
            f"  {name:<9} {np.abs(d_omega).max():>14.4e} {np.abs(chain * d_k).max():>16.4e} "
            f"{np.abs(predicted - d_omega).max() / scale:>15.3e} "
            f"{np.abs(d_k / (2.0 * k_wall)).max():>20.4e}",
            flush=True,
        )

    worst = int(np.argmax(np.abs(chain)))
    print(
        f"  the cell with the largest chain factor is wall entry {worst} (cell {int(wall[worst])}): "
        f"k = {k_wall[worst]:.4e}, d(log omega_target)/dk = {chain[worst]:.4e}",
        flush=True,
    )
    # Which SIGN the large k corrections carry, because the shipped limiter is one-sided:
    # `positive_block_limit` bounds only entries with `delta < 0` (the ones that could cross zero), so
    # an unbounded *increase* in a tiny near-wall `k` passes it untouched -- and the fixation row then
    # carries that increase straight into `log omega`.
    print(
        f"\n  {'arm':<9} {'min dk/k at wall':>18} {'max dk/k at wall':>18} "
        f"{'k cap fires?':>14} {'dk at the worst omega cell':>28}",
        flush=True,
    )
    for name, coupled in built.items():
        d_k = np.asarray(_blocks(coupled, steps[name][beta])[-2])[wall]
        d_omega = np.asarray(_blocks(coupled, steps[name][beta])[-1])[wall]
        ratio = d_k / k_wall
        hot = int(np.argmax(np.abs(d_omega)))
        decreasing = d_k < 0.0
        print(
            f"  {name:<9} {ratio.min():>18.4e} {ratio.max():>18.4e} "
            f"{('yes' if decreasing.any() else 'no'):>14} "
            f"{d_k[hot]:>+15.4e} (k {k_wall[hot]:.3e})",
            flush=True,
        )


def _omega_hotspots(coupled, state, steps, beta, count: int = 3) -> None:
    """The cells taking the largest ``log omega`` correction, both arms side by side.

    ``omega`` is transported as ``log omega``, so an entry of this correction is the logarithm of the
    factor that cell's ``omega`` is multiplied by. A handful of entries therefore decide the whole
    step's admissibility, and naming them is what turns "the omega block overshoots" into something a
    reader can go and look at.
    """
    mesh = coupled.momentum.mesh
    centroid = np.asarray(coupled.momentum.geometry.cell.centroid)
    omega = np.asarray(coupled.physical_fields(state)[2])
    owner = np.asarray(mesh.face_cells.owner)
    interior = np.asarray(mesh.face_cells.interior)
    patch_of: dict[int, set] = {}
    for name in mesh.face_patches.names:
        if name == "interior":
            continue
        for face in np.asarray(mesh.face_patches.indices(name)):
            patch_of.setdefault(int(owner[face]), set()).add(name)

    per_arm = {name: np.asarray(_blocks(coupled, step[beta])[-1]) for name, step in steps.items()}
    ranking = np.argsort(-np.abs(next(iter(per_arm.values()))))[:count]
    arms = list(per_arm)
    print(f"  at beta {beta:g}, ranked by the {arms[0]} arm's correction", flush=True)
    header = "  ".join(f"{'d log omega ' + name:>20}" for name in arms)
    print(
        f"  {'cell':>7} {'x':>10} {'y':>10} {'omega':>11} {'patches':<22} {header}",
        flush=True,
    )
    for cell in ranking:
        values = "  ".join(f"{float(per_arm[name][cell]):>20.4e}" for name in arms)
        patches = ",".join(sorted(patch_of.get(int(cell), {"-"})))
        print(
            f"  {int(cell):>7} {centroid[cell, 0]:>10.5f} {centroid[cell, 1]:>10.5f} "
            f"{omega[cell]:>11.4e} {patches:<22} {values}",
            flush=True,
        )
    # Split by how many boundary faces a cell owns. This is the discriminator: a closure that reads a
    # boundary value replaces the NORMAL component of that face's gradient, so a cell owning two
    # boundary faces has two independent normal directions replaced -- the mirror image of the owner
    # closure's documented corner failure, where a boundary face supplies no direction the cell did
    # not already have. If the difference were merely "near a boundary" it would not sort this way.
    faces_owned = np.bincount(owner[~interior], minlength=mesh.n_cells)
    heading = "  ".join(f"{'max |d log omega| ' + name:>26}" for name in arms)
    print(f"\n  {'boundary faces owned':<22} {'cells':>7}  {heading}", flush=True)
    for label, mask in (
        ("0 (interior)", faces_owned == 0),
        ("1", faces_owned == 1),
        ("2 or more (corners)", faces_owned >= 2),
    ):
        if not mask.any():
            continue
        values = "  ".join(f"{float(np.abs(per_arm[name][mask]).max()):>26.4e}" for name in arms)
        print(f"  {label:<22} {int(mask.sum()):>7}  {values}", flush=True)


def march_from_seed() -> None:
    """Drive the target rung from the captured seed, both arms, one line per inner iteration.

    Uses the real :class:`~aquaflux.solve.DualTimeStep` through
    :func:`~aquaflux.turbulence.solve_coupled` rather than a re-implementation of its inner loop, so
    what is measured is the march's own step and not a probe's idea of it. The iterate at the first
    clipped inner iteration is written out for :func:`analyze` to be pointed at.
    """
    if not SEED.exists():
        raise SystemExit(f"no captured seed at {SEED}; run this without PITZ_STALL_MODE first")
    seed = jnp.asarray(np.load(SEED)["state"])
    for name in ARMS:
        coupled = compare.build_case(gradient_scheme=multicorr(name))["coupled"]
        print(f"\n[{name}] target rung from the captured seed", flush=True)
        print(
            f"  {'step':>4} {'inner':>5} {'|G| before':>12} {'|G| after':>12} {'cyc':>4} "
            f"{'alpha':>9}",
            flush=True,
        )
        caught: list = []
        previous: list = [seed]
        step_index = [0]

        def on_inner(
            inner,
            before,
            after,
            cycles,
            alpha,
            candidate,
            name=name,
            caught=caught,
            at=step_index,
            previous=previous,
        ):
            print(
                f"  {at[0]:>4} {int(inner):>5} {float(before):>12.5e} "
                f"{float(after):>12.5e} {int(cycles):>4} {float(alpha):>9.4g}",
                flush=True,
            )
            if float(alpha) < CLIPPED_ALPHA and not caught:
                path = HERE / f"stall-iterate-{name}.npz"
                np.savez(path, state=np.asarray(previous[0]))
                caught.append(path)
                print(f"    wrote the iterate this step was taken FROM to {path.name}", flush=True)
            previous[0] = np.asarray(candidate)

        def on_step(report, at=step_index):
            at[0] = int(report.step)
            print(
                f"  step {int(report.step):>3} done: |R| {float(report.residual_norm):.5e} "
                f"alpha_min {float(report.alpha):.4g} shift {float(report.shift):.4g} "
                f"inner {int(report.inner_iterations)} cyc {int(report.cycles)}",
                flush=True,
            )

        probe = CoupledJacobianProbe.build(
            coupled, stencil_reach=run_ab.REACH, column_reach=run_ab.COLUMN_REACH
        )
        refresh = amg_beta_tracking_refresh(
            coupled,
            probe=probe,
            beta_rel_change=float("inf"),
            refresh_every=10**9,
            materialize_drift=None,
            materialize_every=None,
            beta_floor=compare.PC_BETA_FLOOR,
        )
        engine = coupled_amg_continuation(
            coupled,
            seed,
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
            probe=probe,
            cycle_budget=compare.CYCLE_BUDGET,
            forward_rtol=compare.FORWARD_RTOL,
            forward_restart=compare.FORWARD_RESTART,
            forward_max_restarts=compare.FORWARD_MAX_RESTARTS,
            refresh_on_cycles=compare.REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if compare.REFRESH_ON_CYCLES else None,
            positivity_floor=compare.K_POSITIVITY_FLOOR,
            positivity_projection=compare.POSITIVITY_PROJECTION,
            coarse_eq_limit=run_ab.SIMPLE_FLOW["max_coarse"],
            field_split=True,
            leading_inverse=simple_smoothed_inverse(**run_ab.SIMPLE_FLOW),
            trailing_inverse=jacobi_smoothed_inverse(**run_ab.JACOBI_TRAILING),
            inner_observer=on_inner,
        )
        try:
            solve_coupled(
                coupled,
                *coupled.physical_fields(seed),
                continuation=engine,
                refresh=RefreshPolicy(precondition_step=refresh),
                max_steps=MARCH_STEPS,
                rtol=compare.RTOL,
                atol=compare.ATOL,
                step_control=compare.CONTROL,
                retry=compare.RETRY,
                scaled_norm=True,
                on_step=on_step,
            )
        except Exception as exc:  # a step cap is how this probe stops; it is not a failure here
            print(f"  stopped: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}", flush=True)


def main() -> None:
    mode = os.environ.get("PITZ_STALL_MODE", "capture")
    if mode == "analyze":
        analyze()
    elif mode == "march":
        march_from_seed()
    else:
        capture()


if __name__ == "__main__":
    main()
