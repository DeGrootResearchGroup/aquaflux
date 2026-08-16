"""Compare coupled-AMG preconditioner variants at the march's OWN hard states.

A preconditioner study that runs at a converged state measures nothing: an easy operator does not
discriminate between candidates. The first sweep this script grew out of returned **6 cycles for every
arm** -- shipped, plain aggregation, and two strength thresholds -- and read as "no difference". The same
arms at a state entering one of the march's hard steps separated **22 vs 9 cycles**. That result
(``pc_gamg_agg_nsmooths = 0``) is now the shipped default and took ~16% off the march's Krylov cost.

So this script exists to make the hard-state comparison the easy thing to do, and it is kept in the
repository rather than in a scratch directory because the previous generation of preconditioner probes
here -- the Vanka smoother and a monolithic-AMG arm -- were scratchpad-only, are gone, and their
conclusions can no longer be re-adjudicated.

**Method (each of these has produced a retracted verdict on this case):**

* a REAL right-hand side -- the march's own ``-R(state)`` at a checkpoint, never a random vector;
* the REAL preconditioner pairing -- the operator at the march's own beta with the V-cycle built at
  ``max(beta, PC_BETA_FLOOR)``. Building it at the raw beta measures a configuration the floor exists to
  prevent, and reports non-convergence where the shipped pairing takes six cycles;
* the REAL shift diagonal ``beta * d``, not a uniform stand-in;
* judged on the TRUE residual through GMRES, never a preconditioned norm or a one-apply contraction;
* one preconditioner in memory at a time -- the Jacobian is a couple of gigabytes a copy.

**Usage.** The states come from checkpoints, so first run a march that keeps them::

    BFS3D_CHECKPOINT_KEEP=80 python3 validation/bfs3d_openfoam/compare.py

:func:`hard_states` then ranks those checkpoints by what the march recorded alongside each one -- the
cycles its **hardest single** linear solve took, and how far its line search was clipped -- and hands
back the hardest **from each side of the preconditioner's shift floor**, so which states get probed is
read off the march rather than chosen by intuition. The state *entering* a hard step is the checkpoint
written after the previous one, and the beta of the hard step itself is the one the preconditioner is
paired against. States can also be named directly::

    python3 validation/bfs3d_openfoam/preconditioner_sweep.py 49:0.0585 39:0.3333

Add or remove arms in :data:`ARMS`; each is a dict of PETSc options passed straight through to GAMG
through the V-cycle's ``extra_options`` seam. Record the smoother and aggregation alongside any
conclusion drawn from a run of this: both defaults have changed once already, and each change inverted
a previously recorded finding.

**Scope of what a run can conclude.** The V-cycle is always built at ``max(beta, PC_BETA_FLOOR)``, so
no arm here ever sees a preconditioner below the floor. At a state above the floor the pairing is the
matched one the march solves; below it, the operator and the V-cycle are deliberately mismatched by the
ratio ``FLOOR / beta``. Either way this measures **the shipped pairing**, which is the operationally
relevant configuration -- but it is not the same thing as a preconditioner built at a genuinely small
shift, so a claim of the form "at every beta" cannot be settled from here.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    AmgVCycle,
    ColumnProbePlan,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.amg_preconditioner import ShiftedCellMajorOperator  # noqa: E402
from aquaflux.solve import restart_cycles  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
    coupled_scaled_norm,
)


def march_solver(coupled, policy, state):
    """The forward solver the coupled AMG march actually runs, rebuilt for the self-check arm.

    Not the coupled *ILUT* path's solver, which is a different object -- 1 % in a plain 2-norm at
    restart 10 against this one's 30 % in a row-scaled norm at restart 15. Reaching for the wrong one is
    easy and it does not announce itself: at a state where both converge in a single cycle the check
    still passes, and reports a validation it did not perform.
    """
    return relative_residual_gmres(
        0.3,
        norm=coupled_scaled_norm(coupled, policy, state),
        restart=15,
        stagnation_iters=40,
        max_restarts=60,
    )


#: The study's own solver: far past the march's 1% inexact-Newton stop, so arms separate rather than
#: tie. `max_restarts` is deliberately modest -- a failing arm is identified by its true residual, and
#: letting one run to thousands of matrix-vector products costs more than every healthy arm together.
STUDY_SOLVER = relative_residual_gmres(1e-6, restart=15, stagnation_iters=40, max_restarts=60)


#: The variants to compare. ``{}`` is the shipped bundle; anything else overrides it through the
#: ``extra_options`` seam. Keep the shipped arm first as the control.
#:
#: The ladder is built to answer one question -- **is the level smoother, or the coarse space, what
#: limits this V-cycle?** The recorded verdict says the coarse space, but it was measured under
#: prolongator smoothing, which has since been found to degrade the coarse correction on this operator
#: (22 -> 9 cycles), so every smoother arm in that campaign was rescuing a coarse correction that was
#: broken for an unrelated reason. Two independent readings here:
#:
#: * **sweeps** -- more of the *shipped* smoother, costing nothing new. If cycles keep falling as the
#:   sweep count rises, the smoother is not yet saturated and cannot be the binding constraint; if they
#:   plateau, the error that survives is in the coarse space's blind spot.
#:   (A Vanka patch arm sat here once and is gone: measured against a working coarse space it stagnated
#:   on its own at a state where the shipped incomplete-LU converges in two cycles, so it was deleted
#:   rather than carried. The implementation is recoverable from git history if the question reopens.)
#:
#: A third arm attacks the coarse space directly, by **degrading** it: replacing the coarse direct LU
#: with a Jacobi sweep. That is the decisive control for the whole question, and it reads in both
#: directions. If degrading the coarse solve barely changes the cycle count, the coarse correction is
#: not load-bearing here -- and then a smoother plateau *cannot* be blamed on the coarse space, so a
#: tie between smoothers only means the new one is no better than incomplete-LU. If it wrecks the cycle count, the coarse
#: grid is doing real work and a plateau does point there. (Raising ``coarse_eq_limit`` instead would
#: measure nothing: the hierarchy is already two levels, so its single aggregation step has already
#: landed under the limit and a larger limit cannot make it stop any sooner. That is why the recorded
#: "K=8000 is identical to K=2000" reads as inert -- the arm was a no-op, not a null result.)
#:
#: The first arm is not a variant at all but a **self-check**: the shipped preconditioner driven by the
#: march's OWN forward solver, so its cycle count should reproduce what ``march.log`` recorded for that
#: step's first inner solve. It is worth one extra solve because the harness rebuilds the shift policy
#: at the probed state, while the march froze its policy at the Reynolds rung's seed state -- so the two
#: can silently be different operators, and a whole sweep can measure something the march never solved.
#: If this arm disagrees with the log by more than a cycle or two, stop and fix that before reading any
#: other row.
#: An arm may also change the Jacobian probe's **stencil reach** via a ``"reach"`` key, which is not a
#: PETSc option: it changes the sparsity the preconditioner is built from, so it needs its own plan
#: and materialization. Reach 2 roughly halves the probe (112 colours -> 60), and the probe is 79 % of a
#: refresh -- but it is measured to DIVERGE under smoothed aggregation (41 cycles at a true residual of
#: 1.9, where reach 3 reaches 1.5e-10 in six). Plain aggregation is the one condition under which it has
#: not been tried, and the same sweep that condemned it also showed a strength threshold going from NaN
#: under smoothed aggregation to 1.98e-12 under plain -- plain does rescue what smoothed breaks, which is
#: the whole reason this is worth one more run.
ARMS = [
    ("shipped reach 3 (ilu0 x4)", {}),
    ("reach 2, plain aggregation", {"reach": 2}),
    ("reach 2, plain agg, ilu0 x8", {"reach": 2, "mg_levels_ksp_max_it": 8}),
]
FLOOR = (
    compare.PC_BETA_FLOOR
)  # 0.05 -- the V-cycle is built here while the operator keeps MARCH_BETA


def hard_states(
    per_regime: int = 1, directory: Path | None = None
) -> list[tuple[Path, float, str]]:
    """The march's hardest steps on **each side of the preconditioner's shift floor**.

    Ranked by ``max_inner_cycles`` -- the most restart cycles any **single** linear solve of the step
    took -- and by how far the line search was clipped second. Ranking on the step's *summed* cycle
    count instead is a trap that has already been walked into once: the sum rewards a step that took
    many easy inner iterations over a step that took one genuinely hard solve, and on a real march the
    two orderings disagree completely. On the march this was written against, the summed count picked
    a step whose hardest solve was 6 cycles over one whose hardest was 15. Probing the first would have
    compared every preconditioner on an operator none of them find difficult, and reported the
    resulting tie as "no difference" -- which is exactly the null result this whole harness exists to
    avoid.

    The state handed back is the one *entering* the hard step (the checkpoint written after the
    previous step), paired with the beta the hard step itself ran at, because that pairing is what the
    march actually solved. Both are refused unless the two checkpoints really are consecutive steps of
    the same Reynolds rung: a gap (retention dropped a file) would pair a state with another step's
    beta, and a rung boundary would pair a state with an operator at a different Reynolds number.

    .. warning::
       **A step's checkpoint describes only its ACCEPTED attempt, and the hardest linear systems in a
       march are systematically in the rejected ones.** When a solve blows past ``retry.on_cycles`` the
       step is redone at an escalated beta, and it is the *pre-escalation* attempt that met the hard
       operator; the retry then succeeds easily, and that easy attempt is what the checkpoint records.
       On the march this was written against, the hardest solve anywhere was 15 cycles at beta 0.0293
       in a rejected attempt, whose step reports 3 cycles at beta 0.0585 — so ranking on checkpoints
       alone cannot see it. Until the escalated attempts are recorded too, read them out of
       ``march.log`` (the ``redo step N (attempt 2): cycles, beta -> ...`` lines and the per-inner
       tables above them) and pass the state and beta on the command line.

    Parameters
    ----------
    per_regime : int
        How many states to return from each side of the preconditioner's shift floor, hardest first.
        The two regimes are separated because the V-cycle is built at ``max(beta, FLOOR)``: above the
        floor it matches the operator, below it is deliberately mismatched, and cost alone tends to
        return only the Reynolds-continuation restarts at the top of the beta range.
    directory : Path or None
        Where the checkpoints are; defaults to the case's own ``checkpoints/``.

    Returns
    -------
    list of (Path, float, str)
        The state file, that step's beta, and a label naming the regime and why it was picked.

    Raises
    ------
    FileNotFoundError
        If fewer than two checkpoints exist -- a march run with the default retention keeps only the
        converged tail, where every preconditioner ties and the sweep measures nothing.
    KeyError
        If the checkpoints predate ``max_inner_cycles``. Rather than silently fall back to the summed
        count, which is the mistake above, this says so and asks for a fresh march.
    """
    directory = directory or CASE / "checkpoints"
    paths = sorted(directory.glob("state-*.npz"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"{directory} holds {len(paths)} checkpoint(s); the sweep needs the march's mid-run "
            "states. Re-run with BFS3D_CHECKPOINT_KEEP set high enough to cover the whole march."
        )
    records = [dict(np.load(path).items()) | {"path": path} for path in paths]
    if "max_inner_cycles" not in records[0]:
        raise KeyError(
            f"{directory} predates `max_inner_cycles` in the checkpoint, and the summed `cycles` is "
            "not a substitute -- it ranks many-easy-solves above one-hard-solve. Re-run the march, or "
            "name the states explicitly on the command line (see the module docstring)."
        )
    consecutive = [
        i
        for i in range(1, len(records))
        if int(records[i]["step"]) == int(records[i - 1]["step"]) + 1
    ]
    ranked = sorted(
        consecutive,
        key=lambda i: (-int(records[i]["max_inner_cycles"]), float(records[i]["alpha"])),
    )
    picked = []
    for regime, below_floor in (("below the shift floor", True), ("above the floor", False)):
        matching = [i for i in ranked if (float(records[i]["shift"]) < FLOOR) == below_floor]
        for i in matching[:per_regime]:
            record = records[i]
            picked.append(
                (
                    records[i - 1]["path"],
                    float(record["shift"]),
                    f"entering step {int(record['step'])}, {regime} "
                    f"({int(record['max_inner_cycles'])} cyc in its hardest solve, "
                    f"alpha {float(record['alpha']):.3f}, beta {float(record['shift']):.4f})",
                )
            )
    return picked


def load_state(path: Path):
    """The state field from a checkpoint, reported so the run records what it was measured on."""
    data = np.load(path)
    print(
        f"state {path.name}: step {int(data['step'])}, |R| {float(data['residual_norm']):.3e}, "
        f"march beta {float(data['shift']):.4f}",
        flush=True,
    )
    return jnp.asarray(data["state"])


def stencil(coupled, n_fields, reach, _cache={}):  # noqa: B006 - a deliberate per-process memo
    """The plan and gather map for one stencil reach, built once per reach."""
    if reach not in _cache:
        plan = _coupled_jacobian_plan(coupled, reach)
        _cache[reach] = (
            plan,
            block_stencil_gather_map(plan),
        )
    return _cache[reach]


def materialize(coupled, state, plan, structure, n_fields, pc_shift):
    """The equilibrated cell-major operator every arm at this state is built from.

    Done once per state rather than per arm, for two reasons. The coloured jvp probe is the dominant
    part of a build, and the shift-equilibrate-reorder that follows it allocates several temporaries
    the size of the Jacobian -- on this mesh a couple of gigabytes of transient per arm, for a
    bit-identical result. Sharing the assembly also makes it impossible for two arms to be compared on
    operators that differ for any reason other than the options under test.

    Returns the ``(matrix, scale, perm)`` triple :class:`AmgVCycle` takes directly, skipping
    ``build_amg_vcycle``'s internal equilibration. The matrix aliases the assembler's reused buffer,
    which is safe here because the V-cycle copies the arrays it keeps.
    """
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    indptr, indices, _ = structure
    assembler = ShiftedCellMajorOperator(indptr, indices, n_fields)
    return assembler.assemble(jacobian.data, pc_shift)


def arm(label, coupled, state, rhs, op_shift, assembled, n_fields, options, solver=None):
    """Build one V-cycle and solve the REAL system with it; report cycles and the TRUE residual."""
    cell_major, scale, perm = assembled
    t0 = time.time()
    pc = MonolithicAmgPreconditioner(
        AmgVCycle(
            cell_major,
            scale,
            perm,
            n_fields,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            extra_options=options or None,
        )
    )
    build_s = time.time() - t0

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    t1 = time.time()
    x, raw = solve_linear(
        operator, rhs, solver or STUDY_SOLVER, preconditioner=pc.matvec(), throw=False
    )
    true = float(jnp.linalg.norm(operator(x) - rhs) / jnp.linalg.norm(rhs))
    # The offset-corrected count, so a cycle here means what a cycle means in the march's log.
    cycles = restart_cycles(int(raw))
    print(
        f"  {label:<28} levels {pc.factors.levels} coarse {pc.factors.coarse_size:>5}  "
        f"build {build_s:>5.0f}s  cycles {cycles:>4}  "
        f"TRUE rel {true:.3e}  solve {time.time() - t1:>4.0f}s",
        flush=True,
    )
    pc.destroy()  # one preconditioner in memory at a time; the collector is too late for that
    del pc
    gc.collect()
    return cycles, true


def probe_state(coupled, state, march_beta, label, plan, structure, n_fields):
    """Run every arm at one state, reporting each and surviving any that fails."""
    base = _coupled_shift_policy(coupled, state, "twolevel")
    self_check = march_solver(coupled, base, state)
    op_shift = _frozen_shift_diagonal(base, march_beta, state)
    pc_shift = _frozen_shift_diagonal(base, max(march_beta, FLOOR), state)
    rhs = -coupled.residual(state)
    print(
        f"\n{'=' * 90}\n{label}\n  operator beta {march_beta}, V-cycle beta "
        f"{max(march_beta, FLOOR)}, real rhs |R| {float(jnp.linalg.norm(rhs)):.3e}",
        flush=True,
    )
    for name, options in ARMS:
        options = dict(options)
        plan, structure = stencil(coupled, n_fields, options.pop("reach", 3))
        assembled = materialize(coupled, state, plan, structure, n_fields, pc_shift)
        # A raise in one arm -- a singular patch, a coarse-solve zero pivot -- must not take the arms
        # queued behind it, which by then represent most of the run's elapsed time.
        try:
            arm(
                name,
                coupled,
                state,
                rhs,
                op_shift,
                assembled,
                n_fields,
                options,
                solver=self_check if name.startswith("SELF-CHECK") else None,
            )
        except Exception as failure:
            print(f"  {name:<28} FAILED  {type(failure).__name__}: {failure}", flush=True)
        del assembled
        gc.collect()


def main():
    """Probe the states named on the command line, or the hardest ones the checkpoints report.

    ``python3 preconditioner_sweep.py`` picks the states itself. ``python3 preconditioner_sweep.py
    49:0.0585 39:0.3333`` probes ``state-00049`` at beta 0.0585 and ``state-00039`` at beta 0.3333
    instead -- for checkpoints written before :func:`hard_states`' ranking key existed, or to re-probe
    a state a previous run flagged.
    """
    if len(sys.argv) > 1 and ":" in sys.argv[1]:
        states = [
            (
                CASE / f"checkpoints/state-{int(index):05d}.npz",
                float(beta),
                f"state-{int(index):05d} at beta {float(beta):.4f} (named on the command line)",
            )
            for index, beta in (argument.split(":") for argument in sys.argv[1:])
        ]
    else:
        states = hard_states(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    reach3 = _coupled_jacobian_plan(coupled, 3)
    struct3 = block_stencil_gather_map(ColumnProbePlan.uniform(reach3, n_fields))
    for path, march_beta, label in states:
        probe_state(coupled, load_state(path), march_beta, label, reach3, struct3, n_fields)


if __name__ == "__main__":
    main()
