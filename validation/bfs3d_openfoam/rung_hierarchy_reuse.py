"""Does a multigrid coarse space built at one continuation rung still precondition the next one?

The Reynolds continuation walks the molecular viscosity down a decade per rung, and each rung currently
builds a **new** preconditioner object. That rebuild is what forces the coupled solve to recompile at
every rung boundary: the preconditioner rides in a *static* field of the forward step, so a fresh object
is a fresh compilation-cache key. Reusing one object across the whole ramp and refreshing it in place
removes the recompile -- but a refresh deliberately **reuses the aggregation coarse space**
(``pc_gamg_reuse_interpolation``), and nothing has ever asked whether a coarse space chosen for the
operator at one viscosity is still a good one a decade lower.

That is the whole question this harness answers, and it is worth answering before building the change,
because a degraded coarse space would show up as higher Krylov cycle counts early in every rung and
would eat the saving.

**What is varied, and what is deliberately held fixed.** Every arm is measured on the *same* state, so
the only difference between them is the viscosity the hierarchy was built at. That is the isolation the
question needs: at a real rung boundary the state moves as well, and a probe that let both move could
not say which one mattered. The consequence to state with any result is that this is **not** a
reproduction of a rung boundary -- it is the viscosity half of one.

Arms, each judged on the **true** residual through GMRES (never the preconditioned norm, and never a
one-application contraction -- both have produced retracted verdicts on this operator):

* ``fresh``   -- built at this rung's own viscosity and state. What ``compare.py`` does today.
* ``carried`` -- built once at the anchor viscosity, then ``refresh_in_place``d down the ladder. What
  reusing one object across the ramp would do. Its coarse space dates from the anchor throughout, so by
  the target rung it is being asked to coarsen an operator two decades away from the one it was chosen
  for.

Run from the repo root (one heavy probe at a time -- this materializes a ~39M-nonzero Jacobian per arm):

    python3 validation/bfs3d_openfoam/rung_hierarchy_reuse.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    FieldSplitAmgPreconditioner,
    block_stencil_gather_map,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)

#: Adjoint-grade, far past the march's own 30 % inexact-Newton stop, so a degraded coarse space shows as
#: a cycle-count difference rather than as two arms tying inside one restart cycle.
RTOL = 1e-8
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=60)

#: The shift the V-cycle is built at, and the one the operator carries. They are the same here, which is
#: the pairing a rung's FIRST step meets: the control starts each rung at ``beta_start`` = 0.5, well
#: above the preconditioner's own 0.05 floor, so no mismatch is in play yet. A rung boundary is exactly
#: where this probe belongs, and it is not the low-shift tail -- say so with the result.
BETA = compare.CONTROL.beta_start

#: The viscosity ladder, anchor first, as ``compare.py``'s own schedule produces it.
SCALES = tuple(10.0**power for power in range(compare.N_POINTS, -1, -1))


def probes(companion, state):
    """The single- and batched-tangent Jacobian probes at this ``(companion, state)``.

    Built by a function rather than written as lambdas in the ladder loop, so each closes over its own
    arguments: a lambda defined in the loop body captures the loop *variable*, and every arm would then
    silently probe whichever rung the loop happened to end on.
    """
    return (
        lambda v: _jacobian_matvec(companion, state, v),
        lambda seeds: _batched_jacobian_matvec(companion, state, seeds),
    )


def build_preconditioner(companion, state, plan, structure, shift):
    """One field-split V-cycle, built exactly as ``coupled_amg_continuation`` builds it."""
    matvec, batched = probes(companion, state)
    return FieldSplitAmgPreconditioner.build(
        matvec,
        plan,
        shift,
        FieldGroups(
            n_cells=companion.layout.n_cells,
            n_leading_fields=companion.layout.dim + 1,
            n_trailing_fields=2,
        ),
        smoother_fill_levels=compare.FILL_LEVELS,
        smoother_sweeps=compare.SWEEPS,
        trailing_smoother_sweeps=compare.TRAILING_SWEEPS,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        trailing_options=compare.TRAILING_OPTIONS,
        trailing_inverse=compare.TRAILING_INVERSE,
        batched_matvec=batched,
        probe_batch_size=_PROBE_BATCH_SIZE,
        structure=structure,
    )


def measure(label, preconditioner, companion, state, rhs, shift, built):
    """Solve the rung's own shifted system with this preconditioner; report cycles and the TRUE residual.

    The reported cost is the cycle count on the true residual. A one-application contraction and the
    preconditioned norm are both invalid on this indefinite saddle -- each has ranked a preconditioner
    the wrong way round here -- so neither is computed.
    """

    def operator(v):
        return _jacobian_matvec(companion, state, v) + shift * v

    started = time.time()
    solution, raw = solve_linear(
        operator, rhs, SOLVER, preconditioner=preconditioner.matvec(), throw=False
    )
    true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
    cycles = restart_cycles(int(raw))
    print(
        f"    {label:<10} build {built:>5.0f}s  cycles {cycles:>4}  TRUE rel {true:.3e}  "
        f"solve {time.time() - started:>4.0f}s",
        flush=True,
    )
    return cycles, true


def main():
    case = compare.build_case()
    coupled = case["coupled"]
    # The probe plan and its gather map are mesh-fixed, so they are built once and shared by every arm --
    # the same sharing the production change makes. Building them per arm would be pure repeated cost.
    plan = _coupled_jacobian_plan(coupled, 3, compare.COLUMN_REACH)
    structure = block_stencil_gather_map(plan)

    # ONE state for every arm. The anchor's cold hybrid initialization, because it is the state the
    # anchor rung's hierarchy would genuinely be built at, and holding it fixed is what makes the
    # viscosity the only variable.
    anchor = coupled.with_scaled_molecular_viscosity(SCALES[0])
    state = anchor.state_from_physical(*hybrid_initialize(anchor.momentum, anchor.turbulence))

    print(
        f"\n3D backward-facing step, {coupled.layout.n_cells} cells, "
        f"{plan.n_fields} fields, shift beta = {BETA:g}\n"
        f"  ladder {' -> '.join(f'{s:g}x nu' for s in SCALES)}, one state throughout "
        "(the anchor's cold hybrid initialization)\n"
        f"  smoother ILU({compare.FILL_LEVELS}) x{compare.SWEEPS}, trailing x{compare.TRAILING_SWEEPS}"
        f" ({compare.TURBULENCE_INVERSE}), coarse limit {compare.COARSE_EQ_LIMIT}, "
        f"column reach {compare.COLUMN_REACH}\n"
        f"  judged on the TRUE residual through GMRES(restart 15) to rtol {RTOL:g}\n",
        flush=True,
    )

    carried, results = None, {}
    for scale in SCALES:
        companion = coupled.with_scaled_molecular_viscosity(scale)
        base = _coupled_shift_policy(companion, state, None)
        shift = _frozen_shift_diagonal(base, BETA, state)
        rhs = -companion.residual(state)
        print(f"  {scale:g}x nu   |R| = {float(jnp.linalg.norm(rhs)):.4e}", flush=True)

        if carried is None:
            started = time.time()
            carried = build_preconditioner(companion, state, plan, structure, shift)
            built = time.time() - started
        else:
            matvec, batched = probes(companion, state)
            started = time.time()
            carried.refresh_in_place(
                matvec,
                plan,
                shift,
                batched_matvec=batched,
                probe_batch_size=_PROBE_BATCH_SIZE,
                structure=structure,
            )
            built = time.time() - started
        results[scale] = {
            "carried": measure("carried", carried, companion, state, rhs, shift, built)
        }

        # The anchor's `fresh` arm IS the carried arm -- the hierarchy has only just been built there --
        # so measuring it twice would report a tautology as a control.
        if scale != SCALES[0]:
            fresh = None
            try:
                started = time.time()
                fresh = build_preconditioner(companion, state, plan, structure, shift)
                results[scale]["fresh"] = measure(
                    "fresh", fresh, companion, state, rhs, shift, time.time() - started
                )
            finally:
                if fresh is not None:
                    fresh.factors.destroy()
                del fresh
                gc.collect()

    print("\n  verdict (cycles, carried against fresh):", flush=True)
    for scale, arms in results.items():
        if "fresh" not in arms:
            print(f"    {scale:>6g}x nu   carried {arms['carried'][0]:>4}   (the build itself)")
            continue
        carried_cycles, fresh_cycles = arms["carried"][0], arms["fresh"][0]
        print(
            f"    {scale:>6g}x nu   carried {carried_cycles:>4}   fresh {fresh_cycles:>4}   "
            f"ratio {carried_cycles / max(fresh_cycles, 1):.2f}",
            flush=True,
        )
    if carried is not None:
        carried.factors.destroy()


if __name__ == "__main__":
    main()
