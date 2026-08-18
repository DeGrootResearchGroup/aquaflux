"""Is the exact complete LU worth carrying, at the 2D mesh size it is the preferred tool for?

The complete LU is the only thing in this package that still wants an optional PETSc build (for
UMFPACK), and it is confined to two dimensions and moderate meshes -- its fill is ``O(n log n)`` in
2D but ``O(n^{4/3})`` in 3D, where it was measured to exhaust memory at around ten thousand cells. So
the dependency is carried for one regime, and the question is what is actually lost by dropping it:
if a factorization that needs no such build marches this case about as fast, the exact LU is a
strategy to delete rather than a dependency to keep.

The arms are the exact LU against the PETSc-free alternative, because "is the LU worth keeping"
is a question about what would REPLACE it:

* ``lu``      -- the complete LU, re-factored every step. Frozen it mis-preconditions a ramping
                 shift badly (measured: 1 Krylov iteration at the matching shift against 474 two
                 doublings away), so a fair arm gives it the per-step refresh it is designed around.
* ``hostilu`` -- the field split whose blocks are a native hierarchy smoothed by this package's own
                 zero-fill factorization. No optional dependency at all.

(A threshold incomplete-LU arm, ``ilut``, used to run alongside these two; it was removed once the
family verdict settled that it was dominated by both the complete LU at 2D and the field-split
multigrid at 3D, with no case selecting it.)

**Judge on WALL CLOCK, not on Krylov cycles.** The arms have different per-application costs by
construction -- an exact factorization converges in one iteration and pays for it in the factor --
so a cycle count is not a cost proxy across them. Cycles are reported beside the clock because they
are contention-immune and the clock is not; where the two disagree, say so rather than picking one.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    MarchLogger,
    RefreshPolicy,
    host_ilu_inverse,
    native_nodal_inverse,
    restart_cycles,
)
from aquaflux.turbulence import (  # noqa: E402
    coupled_amg_continuation,
    coupled_lu_continuation,
    hybrid_initialize,
    lu_beta_tracking_refresh,
    solve_coupled,
)

#: Matches the case's own stopping test, so an arm is not flattered by a looser bar than its rivals.
#: ⚠️ ATOL TOO. The case's bar moved from a relative `RTOL` to an absolute `ATOL`; importing only
#: `RTOL` leaves the stopping test `|R| <= 0`, so every arm silently runs the full step cap and
#: the comparison measures the cap rather than the preconditioners.
MAX_STEPS, RTOL, ATOL = compare.MAX_STEPS, compare.RTOL, compare.ATOL

#: The march configuration is the CASE's, imported rather than restated. It used to be a second copy
#: here, which is how two files that must agree stop agreeing: the case is what a reader runs and what
#: every other study inherits, so an arm measured against a private copy of its settings is measuring
#: a configuration nobody else uses.
INNER_STEPS, INNER_TOL = compare.INNER_STEPS, compare.INNER_TOL
CYCLE_BUDGET = compare.CYCLE_BUDGET
FORWARD_RTOL, FORWARD_RESTART = compare.FORWARD_RTOL, compare.FORWARD_RESTART
RETRY = compare.RETRY
CONTROL = compare.CONTROL
DUAL_TIME = dict(inner_steps=INNER_STEPS, inner_tol=INNER_TOL)

#: The flow block's settings from the three-dimensional study, where they were measured. Carried here
#: unchanged rather than retuned: the point is whether the PETSc-free path is competitive as it
#: stands, and a per-case tuning pass would answer a different question.
HOST_FLOW = dict(
    sweeps=1,
    cycles=1,
    # ⚠️ ZERO, NOT the 0.25 the three-dimensional study ships, and this is a case property rather
    # than a tuning preference. The criterion keeps an edge whose magnitude is within `theta` of its
    # row's largest, so what it retains depends on how the operator's weight is spread. Measured on
    # this mesh: theta 0.25 keeps 5.5% of the cell graph's edges and yields aggregates of median size
    # ONE -- no coarse space at all, and the build then refuses because the "coarsest" level is the
    # fine one. At zero it keeps 47.9% and coarsens 7.24x with a median aggregate of six.
    strength_threshold=0.0,
    avoid_singletons=True,
    aggressive_levels=0,
    # ⚠️ `max_levels` is 10 here against the three-dimensional study's 5, and that is not tuning -- it
    # is the difference between coarsening far enough and not. A level cap binds BEFORE `max_coarse`
    # whenever it is reached first, and on this mesh five levels leave a coarsest grid of 10197
    # degrees of freedom, above the 8192 a dense inverse is allowed. Raising the cap lets
    # `max_coarse` be what stops the coarsening, which is the rule that transfers between meshes;
    # a level count is not.
    max_levels=10,
    max_coarse=500,
    prolongation_smoothing="none",
)

#: The trailing block's settings as the three-dimensional case ships them.
NATIVE_TRAILING = dict(
    cycles=1,
    sweeps=4,
    max_coarse=2000,
    aggressive_levels=1,
    prolongation_smoothing="none",
    spectral_damping=False,
    equilibrate=False,
)


def _require_finite(coupled, state) -> None:
    """Refuse to measure anything if the starting residual is not finite.

    A preconditioner study reads factorization failures as evidence about the factorization. That
    inference is only valid on a finite operator, and nothing downstream checks: a NaN residual
    produces a NaN Jacobian, and every arm then fails in a way that looks like a property of the
    method rather than of the input. Assert it once, here, where the message can say so.
    """
    residual = coupled.residual(state)
    if not bool(jnp.all(jnp.isfinite(residual))):
        raise SystemExit(
            "the starting residual is NOT finite, so no measurement below would mean anything. "
            "The usual cause is packing PHYSICAL fields with `pack_state` on a case that transports "
            "a transformed variable (here log-omega) -- use `state_from_physical`."
        )


def _arms(coupled, reference_state):
    """The two continuations, each with the refresh policy it is designed around."""

    def lu():
        return (
            coupled_lu_continuation(coupled, reference_state, **DUAL_TIME),
            # A complete LU is exact only for the operator it factored, and the shift ramps over a
            # march, so a frozen factor preconditions a system nothing is solving. Cheap to redo.
            lu_beta_tracking_refresh(coupled),
        )

    def hostilu():
        return (
            coupled_amg_continuation(
                coupled,
                reference_state,
                field_split=True,
                leading_inverse=host_ilu_inverse(**HOST_FLOW),
                trailing_inverse=native_nodal_inverse(**NATIVE_TRAILING),
                cycle_budget=CYCLE_BUDGET,
                forward_rtol=FORWARD_RTOL,
                forward_restart=FORWARD_RESTART,
                **DUAL_TIME,
            ),
            None,
        )

    return {"lu": lu, "hostilu": hostilu}


def run(name, build, coupled, start):
    """March one arm from the shared initial state, reporting cost and where it landed."""
    flow0, k0, omega0 = start
    continuation, precondition_step = build()
    steps: list = []
    began = time.perf_counter()

    def watch(report):
        """One line per outer step, flushed.

        A march nobody can read until it finishes costs its whole wall time to tell you something it
        knew in the third minute -- and on this case an arm may run to the step cap without
        converging, which is indistinguishable from a hang unless the steps are visible.
        """
        steps.append(report)
        print(
            f"  [{name}] step {len(steps):3d}  t {time.perf_counter() - began:7.1f}s"
            f"  cyc {int(report.cycles):3d}  |R| {float(report.residual_norm):.4e}"
            f"  alpha {float(report.alpha):.3f}",
            flush=True,
        )

    logger = MarchLogger(sys.stdout, rtol=RTOL, atol=0.0, detail=("inner",))
    logger.note(f"[arm] {name}")
    flow, k, omega = solve_coupled(
        coupled,
        flow0,
        k0,
        omega0,
        continuation=continuation,
        max_steps=MAX_STEPS,
        rtol=RTOL,
        atol=ATOL,
        # The three seams the three-dimensional case drives its march with. Without `step_control`
        # the shift never ramps and this case crawls; without `retry` a bad step can only be absorbed
        # rather than escalated out of.
        step_control=CONTROL,
        retry=RETRY,
        on_step=watch,
        on_checkpoint=logger.on_checkpoint,
        # ⚠️ THROUGH A RefreshPolicy. `solve_coupled` ends in `**continuation_kwargs`, so a bare
        # `precondition_step=` is SWALLOWED rather than rejected and the refresh never runs.
        **(
            {"refresh": RefreshPolicy(precondition_step=precondition_step)}
            if precondition_step
            else {}
        ),
    )
    wall = time.perf_counter() - began
    residual = float(jnp.linalg.norm(coupled.residual(coupled.state_from_physical(flow, k, omega))))
    cycles = sum(int(restart_cycles(r.cycles, max(int(r.inner_iterations), 1))) for r in steps)
    return dict(arm=name, steps=len(steps), cycles=cycles, wall=wall, residual=residual, flow=flow)


def main() -> None:
    case = compare.build_case()
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    start = hybrid_initialize(momentum, turbulence)
    # ⚠️ `state_from_physical`, NOT `pack_state`. `hybrid_initialize` returns PHYSICAL fields, while
    # `pack_state` takes the SOLVED variables -- and this case transports omega as `log(omega)`, so the
    # two differ by an exponential. Packing physical omega where a log is expected exponentiates ~1e5
    # and the residual is silently NaN: the state still reads finite, every factorization then fails
    # in its own idiom (out of memory, "exactly singular", "SVD did not converge"), and each failure
    # invites an interesting and completely wrong story about the method. Which is what happened here.
    reference_state = coupled.state_from_physical(*start)
    _require_finite(coupled, reference_state)

    arms = _arms(coupled, reference_state)
    # The blessed launcher runs a script with no arguments, so the arm list is read from the
    # environment as well -- the same convention the other cases use for their settings.
    wanted = (
        sys.argv[1:] or [a for a in os.environ.get("PITZ_ARMS", "").split(",") if a] or list(arms)
    )
    print(
        f"pitzDaily, {coupled.n_cells if hasattr(coupled, 'n_cells') else '?'} cells; "
        f"stop rtol {RTOL}, max {MAX_STEPS} steps",
        flush=True,
    )
    print(f"{'arm':10} {'steps':>6} {'cycles':>7} {'wall(s)':>9} {'|R|':>12}  x_r/h", flush=True)

    for name in wanted:
        try:
            out = run(name, arms[name], coupled, start)
        except Exception as exc:  # a failing arm is a result; do not lose the others
            print(f"{name:10} FAILED: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        velocity, _ = momentum.unpack(out["flow"])
        import numpy as np

        x_r = compare.reattachment_length(
            np.asarray(geom.cell.centroid), np.asarray(velocity)[:, 0]
        )
        print(
            f"{out['arm']:10} {out['steps']:6d} {out['cycles']:7d} {out['wall']:9.1f} "
            f"{out['residual']:12.3e}  {x_r:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
