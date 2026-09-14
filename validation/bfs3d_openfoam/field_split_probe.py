"""The monolithic coupled preconditioner at the hard states of the 3D backward-facing-step march.

This harness began as a study of whether splitting the turbulence out of the coupled preconditioner's
hierarchy helps: giving the ``[u, v, w, p]`` saddle and the ``[k, omega]`` pair their own hierarchies while
keeping one triangle of the coupling between them. **Every field-split arm has been removed**, together
with the helpers that built them, because the library no longer builds a PETSc V-cycle on a split block
and no longer offers the turbulence-first ordering (issue #371); the host incomplete-LU smoothed hierarchy
some arms used is gone as well. Their recorded results remain in the project's records. What is left is
the monolithic arm -- one V-cycle over all six fields -- and its two Jacobi-class smoother variants, plus
the state table, materialization and solve machinery that other harnesses in this directory import.

**Where the headroom is, and is not.** At the states the march visits, a preconditioner rebuilt at the
iterate solves the forward system in one restart cycle *at the march's own loose stop*, so that pairing
cannot separate two candidates -- an easy operator is not a test, and a tie there is no information. Two
kinds of state restore the discrimination, and both are configurations something really solves:

* the **converged state at zero shift** -- the operator every gradient's transpose solve meets. Removing
  the pseudo-transient shift is what makes this operator hard, and the adjoint has no preconditioner floor
  to soften it, so this is where a better preconditioner has something real to win. Note it must be the
  *converged* state: stripping the shift off a mid-march iterate would measure an operator that nothing,
  forward or adjoint, ever solves. This is the discriminating state in the shipped ``STATES`` set.
* a **captured hard inner iterate**, at the shift the march solved it under, driven far past the march's
  own stop so the arms separate instead of all stopping at one cycle. The march's expensive solves are
  measured to be staleness rather than hard operators, so this is a comparison of quality at a matched
  preconditioner, not a reproduction of the march's cost. These are written only when a march is run with
  ``BFS3D_INNER_DUMP_ABOVE`` set, so a set captured without it has none, and adding one means re-running
  the march rather than reusing an older capture under a bundle whose defaults have since moved.

**Method** -- each of these has produced a verdict on this case that had to be retracted:

* the TRUE residual through GMRES, never a preconditioned norm, a one-apply contraction, or a spectral
  radius;
* a REAL right-hand side, the steady residual ``-R(state)``, never a random vector;
* the REAL shift diagonal ``beta * d``, not a uniform stand-in;
* one materialization per state, shared by every arm, so two arms can never differ for any reason but the
  options under test -- and so only one copy of a multi-gigabyte Jacobian is ever live;
* a **faithfulness gate**: where a restart-cycle count is on record for the shipped monolithic arm at a
  state, that arm must reproduce it or the run refuses to report. A state with nothing on record is
  gated only on the control converging at all;
* **states from ONE march, whose bundle is written down beside them** (see ``STATES``). The checkpoint
  names come from a per-run counter over a rolling buffer, so they carry no date and no configuration:
  a file from a march run before a default moved looks exactly like a current one and describes a
  different discrete problem.

**The one way this departs from the march, stated because it cannot be removed.** A dual-time step solves
for its own residual ``G = R + beta d (phi - phi_n)``, not for ``R``. At inner iteration 0 the two are
identical (``phi = phi_n``), which is why a sweep over end-of-step checkpoints is right to use ``R``; at a
captured inner iterate they are not, and on the hardest one they differ by a factor of some 200 (``|G|``
3.8e-03 against ``|R|`` 8.3e-01). Recovering ``G`` needs ``phi_n``, which the observer does not record. So
the operator, the state and the shift here are the march's and the right-hand side is not: the cycle count
is comparable to the record and is gated on, the achieved residual is not and is only reported. Since every
arm sees the identical right-hand side, the comparison *between* arms -- the point of the probe -- is
unaffected.

**Usage** -- one state per run, since each materializes a Jacobian of some gigabytes::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00067 > field_split.log 2>&1

A second argument builds the preconditioner at a **different** state from the operator: the march freezes
its preconditioner for a whole inner loop, and its expensive solves are measured to be that staleness
rather than hard operators. Two consecutive iterates reproduce exactly one iteration of it::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00066 state-00065

``--arms=key,key`` restricts the ladder (the control is always kept)::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00067 --arms=mono/cheb

Every smoother named here is a **fixed linear operator**, which the adjoint's transpose solve requires: a
Chebyshev smoother is a fixed polynomial once its eigenvalue bounds are estimated during setup, unlike a
GMRES-accelerated smoother, whose polynomial depends on the right-hand side.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    build_amg_vcycle,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
    coupled_scaled_norm,
)

#: Adjoint-grade, far past the march's own 30 % inexact-Newton stop, so arms separate rather than tie.
RTOL = 1e-8
#: A failing arm is identified by its true residual; letting one run to thousands of matrix-vector
#: products costs more than every healthy arm together.
#: ``BFS3D_PROBE_MAX_RESTARTS`` caps the restart cycles. **A failing arm costs 3-4x a converging one**
#: precisely because it runs the cap out -- measured, the Krylov solves are ~80 % of a run's wall and
#: the fixed setup is a constant ~95 s -- so an exploratory sweep is far cheaper at a low cap. What it
#: costs is comparability: the reported residual is wherever the arm had reached when the budget ran
#: out, so numbers from two different caps are NOT comparable and every run must carry its own
#: controls. A candidate that will beat the incumbent's 11 cycles shows it well inside 20.
MAX_RESTARTS = int(os.environ.get("BFS3D_PROBE_MAX_RESTARTS", "60"))
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=MAX_RESTARTS)


#: The states this runs on. All three come from ONE march, and which march is part of the measurement:
#: a 67-step Reynolds continuation converging to ``|R|`` 3.586e-06 at mid-span ``x_r/h`` 8.361 for 319
#: restart cycles, run under the shipped bundle -- field split, native trailing inverse,
#: ``zerogradient`` k wall, positivity floor 1e-08, ILU(0) x4 on the saddle, plain aggregation,
#: ``coarse_eq_limit`` 2000, column reach 3/3/3/3/2/2, forward restart 15, ``refresh_on_cycles`` 3.
#:
#: **A checkpoint set is only usable with the bundle it was written under**, which is why that list is
#: here rather than in a commit message. Several of those defaults moved within the last few days --
#: the trailing inverse, the wall closure and the positivity floor all changed -- and a state carried
#: over from before them is a different discrete problem, not an older measurement of this one. The
#: names are a per-run counter over a rolling buffer, so they carry no date and nothing complains.
#:
#: Nothing is on record for any of them from this probe, so the self-check reduces to the control
#: converging at all; the monolithic arm is in ``ARMS`` for that reason.
class _State(NamedTuple):
    """One probed operating point.

    ``march_beta`` and ``checkpoint_shift`` are **different quantities that happen to coincide** for
    every mid-march entry, which is why they were one field until the converged state needed them apart.
    ``march_beta`` is the shift to build the OPERATOR at -- the operating point under test.
    ``checkpoint_shift`` is the shift the checkpoint was WRITTEN at, and is only an identity check: the
    names come from a per-run counter over a rolling buffer, so a later march silently replaces a file
    under a name this table documents.

    They part at ``state-00069``. That entry is measured at ``march_beta = 0`` **on purpose** -- the
    adjoint's transpose solve meets the unshifted operator and has no preconditioner floor to soften it
    -- while the checkpoint itself was written mid-march at the shift that step ran under. Conflating
    the two made the identity check demand a shift of zero from a file that could never carry one, so
    the converged entry could not be loaded at all.
    """

    march_beta: float
    checkpoint_shift: float | None  # None for an inner iterate, which records no shift
    checkpoint_residual: float | None  # likewise; the stronger half of the identity fingerprint
    recorded: tuple[int, float] | None
    description: str


STATES = {
    # The ADJOINT's operating point, and the one discriminating state this set has. Every other entry
    # here is an end-of-step checkpoint, which is by construction the CHEAP solve of a step -- a settled
    # state met with a freshly refreshed preconditioner -- and on the shipped march those all cost one
    # or two restart cycles, so no arm can separate from another on them. Stripping the shift is what
    # makes this one hard, and it is not an artificial hardness: the transpose solve behind every
    # `jax.grad` meets exactly this operator, with no preconditioner floor to soften it.
    "state-00067": _State(
        0.0,
        0.0064,
        3.5860e-06,
        None,
        "the converged state, |R| 3.586e-06 -- the ADJOINT's operator, at zero shift",
    ),
    # STEP-INITIAL states, at their own shift. They cannot rank candidates, but they are what a march
    # actually pays for, so they say what an arm COSTS where the operator is easy -- the complement to
    # the hard state, which only says whether it survives at all.
    "state-00066": _State(
        0.0096,
        0.0096,
        2.6025e-05,
        None,
        "step 27 of the target rung, step-initial -- the cheap solve that is most of the march",
    ),
    "state-00065": _State(
        0.0144,
        0.0144,
        1.3046e-04,
        None,
        "step 26 of the target rung, step-initial -- a second cheap solve, higher shift",
    ),
}

#: Level-smoother recipes for the monolithic V-cycle, as PETSc options layered over the shipped bundle.
#: ``ilu0`` is the shipped default (an empty override). The other two are the Jacobi-class candidates a
#: traced multigrid could actually implement, since neither needs a sequential triangular solve.
SMOOTHERS = {
    "ilu0": {},
    "chebyshev": {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi"},
    "jacobi": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
    },
}

FLOOR = (
    compare.PC_BETA_FLOOR
)  # 0.05 -- the forward V-cycle is built here, the operator keeps its own beta


def load_state(name: str) -> jnp.ndarray:
    """The captured state, reporting what it was and REFUSING one that is not what :data:`STATES` says.

    The two checkpoint kinds carry different metadata -- an inner iterate knows its attempt and how its
    own solve went, an end-of-step checkpoint knows the step and its residual -- so each is reported on
    its own terms rather than through a lowest common denominator that would name neither.

    **The shift check is not defensive programming, it is the fix for a real silent failure.** The
    checkpointer keeps only the last few files and numbers them from a counter that restarts with each
    march, so a later run REPLACES ``state-000NN`` with a completely different state under the same name.
    That happened: a name documented here as the converged zero-shift state came back holding a mid-march
    iterate at shift 0.98 from an abandoned run. Nothing would have complained -- the probe would have
    paired an operator built at this table's shift with a state that never had it, and reported the
    result as a measurement at the documented operating point. Every step checkpoint records the shift it
    was written at, so the mismatch is free to detect; refuse rather than measure.
    """
    path = CASE / "checkpoints" / f"{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"{name}: no such checkpoint. These are a rolling buffer (`BFS3D_CHECKPOINT_KEEP`, default "
            f"3) and a later march will have rotated it away -- re-run the case to regenerate, raising "
            f"the keep count if a study needs the whole trajectory.\n  present: "
            f"{sorted(p.stem for p in path.parent.glob('*.npz'))}"
        )
    data = np.load(path)
    if "attempt" in data:
        detail = (
            f"attempt {int(data['attempt'])} inner {int(data['inner'])}, the march took "
            f"{int(data['cycles'])} cycles at alpha {float(data['alpha']):.2e}, "
            f"|G| {float(data['g_before']):.4e} -> {float(data['g_after']):.4e}"
        )
    else:
        entry = STATES.get(name)
        # Fingerprint on the SHIFT and the RESIDUAL together, and require both to be documented.
        #
        # The shift alone is a weak fingerprint precisely where this has to work: every end-of-step
        # checkpoint in a converged tail carries essentially the same shift, so a file replaced by a
        # DIFFERENT march's tail state passes a shift-only test while being a different state. The
        # residual separates them for free -- it is in the file already, and it moves by orders of
        # magnitude along a march where the shift moves by a few percent.
        #
        # `checkpoint_shift` is what the march WROTE the file at, which is not `march_beta`, the shift
        # the probe goes on to operate it at. They coincide everywhere except the converged state,
        # whose whole point is to be measured unshifted.
        for label, recorded, expected, tolerance in (
            ("shift", float(data["shift"]), entry.checkpoint_shift if entry else None, 0.02),
            (
                "residual",
                float(data["residual_norm"]),
                entry.checkpoint_residual if entry else None,
                0.05,
            ),
        ):
            if expected is None:
                # A documented step checkpoint with nothing to check against is not a state this probe
                # can stand behind: the guard exists because these files are silently replaced, and a
                # missing expectation is how it was previously switched off by accident.
                raise SystemExit(
                    f"{name}: the STATES entry documents no expected {label}, so this checkpoint's "
                    "identity cannot be verified. Fill it in from the march log that wrote the file."
                )
            # Loose on purpose: the table records these to about four figures, so an exact comparison
            # rejects a matching state. What this must catch is a REPLACED state, which differs by
            # orders of magnitude (a shift of 0.98 where 0.0064 was documented), not by rounding.
            if not np.isclose(recorded, expected, rtol=tolerance, atol=1e-12):
                raise SystemExit(
                    f"{name}: this checkpoint carries {label} {recorded:.6g}, but the STATES table "
                    f"describes it as {expected:.6g}. The file has been overwritten by a later march "
                    "(the names come from a per-run counter over a rolling buffer), so it is NOT the "
                    "state this entry documents. Re-run the case to regenerate it, or point the entry "
                    "at a checkpoint that still matches."
                )
        recorded = float(data["shift"])
        detail = (
            f"end of step {int(data['step'])}, |R| {float(data['residual_norm']):.4e}, "
            f"march shift {recorded:.4f}"
        )
    print(f"{name}: {detail}", flush=True)
    return jnp.asarray(data["state"])


def march_solver(coupled, policy, state):
    """The forward solver the coupled multigrid march actually runs, for the self-check arm.

    Not the coupled incomplete-LU path's solver, which is a different object -- 1 % in a plain 2-norm at
    restart 10 against this one's 30 % in a row-scaled norm at restart 15. Reaching for the wrong one is
    easy and it does not announce itself: at a state where both converge in a single cycle the check still
    passes and reports a validation it never performed.
    """
    return relative_residual_gmres(
        0.3,
        norm=coupled_scaled_norm(coupled, policy, state),
        restart=15,
        stagnation_iters=40,
        max_restarts=60,
    )


def materialize(coupled, state, plan, structure, n_fields) -> sp.csr_matrix:
    """The **unshifted** field-major Jacobian at this iterate -- the one expensive step, done once.

    Unshifted because the operating points below differ only in the diagonal they add, and re-running a
    several-hundred-probe coloured jvp for each of them would dominate the run.
    """
    started = time.time()
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    print(
        f"  materialized {jacobian.shape[0]} dofs, {jacobian.nnz / 1e6:.1f}M nnz "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    return jacobian


def monolithic(shifted, groups, n_fields, smoother):
    """The shipped arrangement: one V-cycle over all six fields. The control."""
    return MonolithicAmgPreconditioner(
        build_amg_vcycle(
            shifted,
            n_fields,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            extra_options=SMOOTHERS[smoother] or None,
        )
    )


#: ``(key, label, builder)``. The builder takes the shifted matrix rather than closing over it, so nothing
#: holds a reference to a multi-gigabyte operator the run wants to free between operating points.
ARMS = (
    # The control, and the two Jacobi-class smoothers applied to the whole six-field block.
    ("mono/ilu0", "monolithic, ILU(0)", lambda m, g, n: monolithic(m, g, n, "ilu0")),
    ("mono/cheb", "monolithic, Chebyshev", lambda m, g, n: monolithic(m, g, n, "chebyshev")),
    ("mono/jac", "monolithic, damped Jacobi", lambda m, g, n: monolithic(m, g, n, "jacobi")),
)


def run_arm(label, preconditioner, built, coupled, state, rhs, op_shift, solver):
    """Solve the REAL system with one already-built preconditioner; report cycles and the TRUE residual."""

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    # Time the two halves of a Krylov iteration SEPARATELY before attributing cost to either. Both the
    # incumbent and every candidate pay the same exact matrix-free Jacobian product; only the
    # preconditioner differs. So if the product is a large share of an iteration, the whole
    # preconditioner effort is bounded by the remainder, and no amount of work on it can close a gap
    # larger than that share allows. Nothing in this campaign has measured the split.
    if os.environ.get("BFS3D_PROBE_SPLIT"):
        apply_pc = preconditioner.matvec()
        probe = jnp.asarray(rhs)
        jax.block_until_ready(operator(probe))
        jax.block_until_ready(apply_pc(probe))
        started = time.time()
        for _ in range(5):
            jax.block_until_ready(operator(probe))
        jvp_each = (time.time() - started) / 5
        started = time.time()
        for _ in range(5):
            jax.block_until_ready(apply_pc(probe))
        pc_each = (time.time() - started) / 5
        print(
            f"      per-iteration split: jacobian product {jvp_each * 1e3:.0f} ms  |  "
            f"preconditioner {pc_each * 1e3:.0f} ms  |  preconditioner is "
            f"{100 * pc_each / max(jvp_each + pc_each, 1e-12):.0f}% of the pair",
            flush=True,
        )

    solving = time.time()
    solution, raw = solve_linear(
        operator, rhs, solver, preconditioner=preconditioner.matvec(), throw=False
    )
    true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
    cycles = restart_cycles(int(raw))
    print(
        f"    {label:<36} build {built:>5.0f}s  cycles {cycles:>4}  TRUE rel {true:.3e}  "
        f"solve {time.time() - solving:>4.0f}s",
        flush=True,
    )
    return cycles, true


def one_arm(label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, solver):
    """Build and run a single arm, surviving a failure so the arms queued behind it still run.

    A raise here -- a singular coarse solve, a zero pivot, a failed Chebyshev eigenvalue estimate -- is a
    result about that arm, and by the time it happens the remaining arms represent most of the run.
    """
    preconditioner = None
    try:
        started = time.time()
        preconditioner = build(shifted, groups, n_fields)
        return run_arm(
            label, preconditioner, time.time() - started, coupled, state, rhs, op_shift, solver
        )
    except Exception as failure:
        print(f"    {label:<36} FAILED  {type(failure).__name__}: {failure}", flush=True)
        return None
    finally:
        if preconditioner is not None:
            preconditioner.factors.destroy()
        del preconditioner
        gc.collect()


def self_check(name, recorded, shifted, groups, n_fields, coupled, state, rhs, op_shift, solver):
    """Reproduce the recorded control measurement, and refuse to go on if the CYCLE COUNT disagrees.

    Run at the **march's own** solver, because that is the configuration the recorded numbers were taken
    under: judging them against this study's far tighter stop would fail a faithful harness, since a solve
    that reaches 1e-07 in one cycle keeps going when asked for 1e-08.

    **Only the cycle count is gated, and the reason is a real difference this probe cannot remove.** The
    recorded numbers come from driving an actual dual-time step, whose right-hand side is the step's own
    residual ``G = R + beta d (phi - phi_n)``; this probe uses the steady residual ``R``. At inner
    iteration 0 the two coincide exactly (``phi = phi_n``), which is why a checkpoint-based sweep is right
    to use ``R`` -- but a captured inner iterate is precisely where they part, and on the hardest one they
    differ by a factor of some 200 (``|G|`` 3.8e-03 against ``|R|`` 8.3e-01). Reconstructing ``G`` would
    need ``phi_n``, the state the outer step began from, which the observer does not record.

    So the operator, state and shift are the march's; the right-hand side is not. The cycle count is
    comparable and is gated; the achieved residual is not comparable and is reported for the record
    rather than asserted against. Every arm sees the identical right-hand side, so the comparison
    *between* arms -- which is what this probe exists for -- is unaffected.
    """
    print("\n  -- self-check: the shipped preconditioner at the march's own solver", flush=True)
    measured = one_arm(
        "monolithic, ILU(0), march solver",
        ARMS[0][2],
        shifted,
        groups,
        n_fields,
        coupled,
        state,
        rhs,
        op_shift,
        solver,
    )
    if measured is None:
        raise SystemExit(f"SELF-CHECK FAILED for {name}: the control arm did not run at all.")
    cycles, true = measured
    if recorded is None:
        if not np.isfinite(true) or true > 1e-3:
            raise SystemExit(
                f"SELF-CHECK FAILED for {name}: the control did not converge (true relative residual "
                f"{true:.3e}), so nothing else measured here can be trusted."
            )
        print("    [no recorded value for this state; control converges, continuing]", flush=True)
        return
    expected_cycles, expected_true = recorded
    if cycles != expected_cycles:
        raise SystemExit(
            f"SELF-CHECK FAILED for {name}: the shipped preconditioner took {cycles} restart cycles "
            f"where {expected_cycles} is on record for this state and pairing. The harness is not "
            "solving the operator the record describes; fix that before reading any other row."
        )
    print(
        f"    [self-check passed on CYCLES: {cycles}, as recorded. The true residual reads {true:.3e} "
        f"against the recorded {expected_true:.1e}; these are not comparable, because the recorded run's "
        "right-hand side was the step's dual-time residual G and this one's is the steady residual R.]",
        flush=True,
    )


def study(coupled, state, rhs, shifted, op_shift, groups, n_fields, only=None):
    """Every arm at this state's pairing, at the study's own tight stop so the arms separate.

    ``only`` restricts to a subset of arm keys. Re-running the whole ladder to add one arm costs several
    minutes of arms whose answer is already on the log -- and the arms that FAIL are the expensive ones,
    since running to the restart cap is what failing means here. The control is always kept, because a
    subset without it cannot be compared against anything.
    """
    arms = ARMS
    missing = set(only or ()) - {key for key, _, _ in arms}
    if missing:
        raise SystemExit(f"unknown arm(s) {sorted(missing)}; known: {[key for key, _, _ in arms]}")
    selected = [a for a in arms if only is None or a[0] == arms[0][0] or a[0] in only]
    print(f"\n  -- study arms, GMRES to rtol {RTOL:.0e} on the TRUE residual", flush=True)
    return {
        key: one_arm(label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, SOLVER)
        for key, label, build in selected
    }


def _invocation() -> list[str]:
    """The command line, falling back to the environment when there is none.

    A probe of this size is a long solve on a shared machine, which means it belongs behind
    ``validation/run_case.sh`` -- the runner that refuses to start a second one, holds the machine
    awake, and writes a run-file saying what is running and under what settings. That runner takes a
    script and forwards the **environment**, not script arguments, so a probe configured only through
    ``sys.argv`` cannot be launched through it and has to be run bare, where two sessions can collide on
    a machine with room for one 2 GB Jacobian.

    So the state and the arm list are readable from ``BFS3D_PROBE_STATE`` / ``BFS3D_PROBE_PC_STATE`` /
    ``BFS3D_PROBE_ARMS`` as well, which is the same convention the case itself uses. Arguments win where
    both are given, so every existing invocation is unchanged.
    """
    if len(sys.argv) > 1:
        return sys.argv[1:]
    state = os.environ.get("BFS3D_PROBE_STATE")
    if not state:
        return []
    argv = [state]
    pc_state = os.environ.get("BFS3D_PROBE_PC_STATE")
    if pc_state:
        argv.append(pc_state)
    arms = os.environ.get("BFS3D_PROBE_ARMS")
    if arms:
        argv.append(f"--arms={arms}")
    return argv


def main():
    supplied = _invocation()
    argv = [a for a in supplied if not a.startswith("--arms=")]
    chosen = [a for a in supplied if a.startswith("--arms=")]
    only = tuple(chosen[-1].split("=", 1)[1].split(",")) if chosen else None
    if not 1 <= len(argv) <= 2 or argv[0] not in STATES:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}> [preconditioner state] "
            "[--arms=key,key]"
        )
    sys.argv = [sys.argv[0], *argv]
    name = sys.argv[1]
    entry = STATES[name]
    march_beta, recorded, description = entry.march_beta, entry.recorded, entry.description
    # `BFS3D_PROBE_BETA` builds the OPERATOR at a chosen shift on whichever state is loaded, which is the
    # only way to vary beta as an axis: every entry in `STATES` carries one fixed shift, and the two
    # shifted ones are step-initial checkpoints that cost a cycle or two for every arm and so cannot rank
    # anything. Holding the state fixed and moving the shift separates the shift from the state, where
    # switching entries confounds them. `checkpoint_shift` is a separate field and still checks identity
    # against the file, so this does not weaken the faithfulness gate.
    override = os.environ.get("BFS3D_PROBE_BETA")
    if override is not None:
        march_beta = float(override)
        recorded = None  # nothing is on record at a synthesized shift
        description = f"{description} -- OPERATOR SHIFT OVERRIDDEN to beta={march_beta}"
    # An optional SECOND state builds the preconditioner, while the operator and right-hand side stay at
    # the first. That is what the march actually does -- it freezes the preconditioner for a whole inner
    # loop -- and its expensive solves are measured to be this staleness rather than hard operators (15
    # cycles against 1 when matched), so it is the pairing with real headroom on this case. Passing two
    # consecutive inner iterates of one attempt reproduces exactly one inner iteration of staleness.
    pc_state_name = sys.argv[2] if len(sys.argv) == 3 else name
    if pc_state_name not in STATES:
        raise SystemExit(f"unknown preconditioner state {pc_state_name!r}")
    stale = pc_state_name != name
    if stale:
        recorded = None  # nothing is on record for a deliberately mismatched pairing
    # The V-cycle is built at the floor while the operator keeps the march's own beta -- the shipped
    # mismatch. At the converged state's zero shift there is no floor: the adjoint has none, and flooring
    # it here would measure a preconditioner the gradient path never uses.
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.n_fields
    # The monolithic arms read nothing from the grouping; it carries the degree-of-freedom count and the
    # cell count for the banner, and keeps the builder signature the importing harnesses share.
    groups = FieldGroups.split_before(coupled.layout, "k")
    print(
        f"{'=' * 100}\nmonolithic preconditioner: {n_fields} fields over {groups.n_cells} cells\n"
        "bundle: plain aggregation, "
        f"ILU({compare.FILL_LEVELS}) x{compare.SWEEPS} where not overridden, coarse_eq_limit "
        f"{compare.COARSE_EQ_LIMIT}, stencil reach 3, column reach "
        f"{'uniform' if compare.COLUMN_REACH is None else '/'.join(map(str, compare.COLUMN_REACH))}"
        f", GMRES restart 15, max restarts {MAX_RESTARTS}\n"
        f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 100}",
        flush=True,
    )
    state = load_state(name)
    print(f"  {description}", flush=True)

    # Probe each column at the reach the CASE uses, read from `compare` rather than restated here, so a
    # probe cannot measure a preconditioner built from a sparsity the march does not use. That is not a
    # hypothetical: this default has already moved twice, and both moves turned on the SPARSITY rather
    # than on any value. A shortened column writes its out-of-reach entries as exact zeros where a
    # uniform probe leaves the true value -- tiny, but nonzero -- and an assembly written as a sparse
    # product stores only entries whose result is nonzero, so it deletes those explicit zeros and hands
    # a zero-fill incomplete factorization a structurally weaker pattern for a numerically identical
    # matrix. Reading the case's value is what keeps this probe on the right side of that.
    plan = _coupled_jacobian_plan(coupled, 3, compare.COLUMN_REACH)
    structure = block_stencil_gather_map(plan)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    op_shift = _frozen_shift_diagonal(base, march_beta, state) if march_beta > 0 else 0.0
    # Report both norms, so the run records the one way it departs from the march: the march solved for
    # the step's dual-time residual G, this solves for the steady residual R, and on an inner iterate
    # those are different right-hand sides over the same operator.
    print(f"  right-hand side |R| {float(jnp.linalg.norm(rhs)):.4e}", flush=True)

    # The preconditioner is assembled at its own state, which is the same one unless a stale pairing was
    # asked for. Only one Jacobian is ever live: the operator side is applied matrix-free, by the exact
    # jvp at `state`, so the materialization is needed only for the preconditioner.
    if stale:
        print("  preconditioner built at a DIFFERENT state:", flush=True)
        pc_state = load_state(pc_state_name)
        pc_base = _coupled_shift_policy(coupled, pc_state, "twolevel")
    else:
        pc_state, pc_base = state, base
    jacobian = materialize(coupled, pc_state, plan, structure, n_fields)
    pc_shift = (
        _frozen_shift_diagonal(pc_base, pc_beta, pc_state)
        if pc_beta > 0
        else np.zeros(groups.n_dofs)
    )
    shifted = MonolithicAmgPreconditioner._shifted(jacobian, pc_shift)
    del jacobian
    gc.collect()

    self_check(
        name,
        recorded,
        shifted,
        groups,
        n_fields,
        coupled,
        state,
        rhs,
        op_shift,
        march_solver(coupled, base, state),
    )
    study(
        coupled,
        state,
        rhs,
        shifted,
        op_shift,
        groups,
        n_fields,
        only=only,
    )


if __name__ == "__main__":
    main()
