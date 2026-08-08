"""Capture a dual-time step's INNER iterates, and ask what made the expensive solves expensive.

A checkpoint is written at the end of a step, so it holds the state the *next* step begins from -- and a
step's first solve is its easy one, taken from a settled state with a freshly rebuilt preconditioner. On
the three-dimensional coupled march this was written against, **every one of the 70 step-initial solves
cost at most 2 restart cycles, while solves later in the inner loop reached 15.** A preconditioner study
that probes checkpoints therefore never meets a hard operator and reports that every candidate performs
identically -- which is what happened, and what this script exists to get past.

It re-runs ONE step from a checkpoint with an ``inner_observer`` that keeps each iterate, then probes the
most expensive one two ways:

* **matched** -- the preconditioner rebuilt at that iterate;
* **stale** -- the preconditioner built at the state the step started from, which is what the march
  actually had, since the preconditioner is frozen for the whole inner loop.

Those two separate the only two explanations for an expensive inner solve, and they call for opposite
fixes. If the matched pairing is cheap, the operator was never hard and the cost was **staleness** -- a
refresh-cadence question, and no smoother or coarse-space change will touch it. If the matched pairing is
also expensive, the operator at that iterate genuinely is hard, and that is the state a preconditioner
study should be run at.

**Faithfulness -- and this is the whole difficulty.** Building a continuation at a checkpoint produces a
*self-consistent* ``(state, shift, preconditioner)`` triple, and **the march never occupies that
configuration**: it freezes the shift and the preconditioner and carries them forward, so both lag the
state. A self-consistent rebuild is a different problem, and it does not fail the way the march fails --
measured here, a rebuild at the march's own state and beta descends cleanly at ``alpha = 1`` through five
inner iterations where the march stalled at ``alpha = 0`` and had to escalate beta.

So the probe pins two things that are easy to get wrong and fatal when wrong:

* ``amg_beta`` is set to the pairing the march's refresh maintains, ``max(beta, PC_BETA_FLOOR)``. The
  builder's own default is 2.0, which at a sub-floor beta is a mismatch of nearly two orders of
  magnitude and on its own turns a one-cycle solve into a seven-cycle one.
* the run **validates itself against ``march.log`` before reporting anything**, comparing the first inner
  iteration's cycle count and line-search factor with the recorded ones and refusing to continue if they
  disagree. Reproducing a march step is difficult enough that an unvalidated harness should be assumed to
  be measuring something else; the numbers above are what that looks like.

**Usage**::

    python3 validation/bfs3d_openfoam/inner_iterate_probe.py 49 0.0293 1 0.500 38

with the checkpoint index, the beta that step ran at, the cycle count and line-search factor of that
step's **first inner iteration**, and the index of the **Reynolds rung's first checkpoint**. The first
four come off ``march.log`` (the middle two so the run can check itself); the last is where the march
froze its shift policy, found by looking for the checkpoint whose per-rung ``step`` restarts at 0. For a
step that was redone, use the *pre-escalation* attempt throughout: that is the one that met the hard
operator.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    AmgVCycle,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.amg_preconditioner import ShiftedCellMajorOperator  # noqa: E402
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.turbulence import coupled_amg_continuation  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
    coupled_scaled_norm,
)

FLOOR = compare.PC_BETA_FLOOR


def capture_inner_iterates(coupled, state, beta, seed_state):
    """Run one dual-time step from ``state`` at ``beta``, returning every inner iterate it passed through.

    Beta is pinned by seeding the relaxation schedule at the reference residual: the switched-evolution
    rule returns ``beta0`` when the current and reference residuals agree, so passing the step's own
    residual as the reference makes the shift exactly ``beta0``. The preconditioner is frozen at the
    pairing the march's refresh maintains rather than at the builder's default.
    """
    records = []

    def observer(inner, g_before, g_after, cycles, alpha, iterate):
        records.append(
            dict(
                inner=int(inner),
                g_before=float(g_before),
                g_after=float(g_after),
                cycles=restart_cycles(int(cycles)),
                alpha=float(alpha),
                iterate=np.asarray(iterate),
            )
        )

    # Built at the RUNG SEED, then stepped from `state`: that is the arrangement the march is in, with
    # its shift policy frozen at the seed and lagging the state. Building at `state` instead gives a
    # self-consistent triple the march never occupies, and it does not fail the way the march fails.
    engine = coupled_amg_continuation(
        coupled,
        seed_state,
        beta0=beta,
        amg_beta=max(beta, FLOOR),  # the march's refresh pairing; the builder's own default is 2.0
        inner_steps=compare.INNER_STEPS,
        inner_tol=compare.INNER_TOL,
        smoother_fill_levels=compare.FILL_LEVELS,
        smoother_sweeps=compare.SWEEPS,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        cycle_budget=compare.CYCLE_BUDGET,
        inner_observer=observer,
    )
    reference_norm = engine.norm()(coupled.residual(state))
    result = engine.stepper()(coupled.residual, state, reference_norm, engine.default_solver())
    result.phi.block_until_ready()  # flush the ordered debug callbacks before reading `records`
    print(
        f"one step from the checkpoint at beta {beta}: {len(records)} inner iterations",
        flush=True,
    )
    for record in records:
        print(
            f"  inner {record['inner']}  cyc {record['cycles']:>3}  "
            f"|G| {record['g_before']:.4e} -> {record['g_after']:.4e}  alpha {record['alpha']:.3f}",
            flush=True,
        )
    return records


def solve_with(label, coupled, state, pc_state, beta, colouring, structure, n_fields):
    """Solve the shifted system at ``state`` with the preconditioner built at ``pc_state``."""
    base = _coupled_shift_policy(coupled, state, "twolevel")
    op_shift = _frozen_shift_diagonal(base, beta, state)
    pc_base = _coupled_shift_policy(coupled, pc_state, "twolevel")
    pc_shift = _frozen_shift_diagonal(pc_base, max(beta, FLOOR), pc_state)
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, pc_state, v),
        colouring,
        n_fields,
        lambda seeds: _batched_jacobian_matvec(coupled, pc_state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    indptr, indices, _ = structure
    cell_major, scale, perm = ShiftedCellMajorOperator(indptr, indices, n_fields).assemble(
        jacobian.data, pc_shift
    )
    pc = MonolithicAmgPreconditioner(
        AmgVCycle(
            cell_major,
            scale,
            perm,
            n_fields,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        )
    )

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    # The march's own inner loop solves G(p) = R(p) + beta d (p - reference); at the probed iterate the
    # right-hand side is that transient residual, not the steady one.
    rhs = -coupled.residual(state)
    started = time.time()
    # The march's OWN forward solver: 30 % in the row-scaled measure at restart 15. The coupled ILUT
    # path's solver is a different object (1 % in a plain 2-norm at restart 10) and reaching for it
    # measures a solve this march never performs.
    solver = relative_residual_gmres(
        0.3,
        norm=coupled_scaled_norm(coupled, base, state),
        restart=15,
        stagnation_iters=40,
        max_restarts=60,
    )
    x, raw = solve_linear(operator, rhs, solver, preconditioner=pc.matvec(), throw=False)
    true = float(jnp.linalg.norm(operator(x) - rhs) / jnp.linalg.norm(rhs))
    print(
        f"  {label:<34} cycles {restart_cycles(int(raw)):>3}  TRUE rel {true:.3e}  "
        f"{time.time() - started:>4.0f}s",
        flush=True,
    )
    pc.destroy()
    return restart_cycles(int(raw))


def check_against_the_march(records, expected_cycles, expected_alpha):
    """Refuse to go on unless the reproduced first inner iteration matches the march's recorded one.

    Reproducing a march step is easy to get subtly wrong -- the shift policy, the preconditioner pairing
    and the residual norm all have to line up -- and a harness that is off measures a configuration the
    march never ran while looking perfectly healthy. Comparing against the log costs nothing and is the
    difference between a result and a plausible number.
    """
    got_cycles, got_alpha = records[0]["cycles"], records[0]["alpha"]
    if got_cycles != expected_cycles or abs(got_alpha - expected_alpha) > 1e-3:
        raise SystemExit(
            f"harness does not reproduce the march: inner 0 gave {got_cycles} cycles at "
            f"alpha {got_alpha:.3f}, the log records {expected_cycles} at {expected_alpha:.3f}. "
            "Fix the reproduction before reading anything else from this run."
        )
    print(
        f"validated against march.log: inner 0 = {got_cycles} cycles at alpha {got_alpha:.3f}",
        flush=True,
    )


def main():
    index, beta = int(sys.argv[1]), float(sys.argv[2])
    expected_cycles, expected_alpha = int(sys.argv[3]), float(sys.argv[4])
    seed_index = int(sys.argv[5])  # the Reynolds rung's first checkpoint
    data = np.load(CASE / f"checkpoints/state-{index:05d}.npz")
    start = jnp.asarray(data["state"])
    print(
        f"state-{index:05d}: step {int(data['step'])}, |R| {float(data['residual_norm']):.3e}; "
        f"driving one step at beta {beta}",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    colouring = _coupled_jacobian_colouring(coupled, 3)
    structure = block_stencil_gather_map(colouring, n_fields)

    seed = jnp.asarray(np.load(CASE / f"checkpoints/state-{seed_index:05d}.npz")["state"])
    records = capture_inner_iterates(coupled, start, beta, seed)
    check_against_the_march(records, expected_cycles, expected_alpha)
    hardest = max(records, key=lambda r: r["cycles"])
    print(
        f"\nhardest inner iteration: inner {hardest['inner']} at {hardest['cycles']} cycles. "
        f"Probing that iterate against the step's start:",
        flush=True,
    )
    iterate = jnp.asarray(hardest["iterate"])
    # Stale FIRST: it is the pairing the march actually ran, so if only one of the two completes it
    # should be the one that describes what happened.
    solve_with(
        "stale PC (as the march had it)",
        coupled,
        iterate,
        start,
        beta,
        colouring,
        structure,
        n_fields,
    )
    solve_with(
        "matched PC (rebuilt at the iterate)",
        coupled,
        iterate,
        iterate,
        beta,
        colouring,
        structure,
        n_fields,
    )


if __name__ == "__main__":
    main()
