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

then pick the hard steps out of ``march.log`` -- highest ``cyc``, smallest ``a_min``, any ``e``-flagged
retry -- and list them in :data:`HARD_STATES` as ``(checkpoint index entering the step, that step's beta,
label)``. The checkpoint entering step N is the one written after step N-1.

Add or remove arms in :data:`ARMS`; each is a dict of PETSc options passed straight through to GAMG via
``MonolithicAmgPreconditioner.build(extra_options=...)``. Record the smoother and aggregation alongside
any conclusion drawn from a run of this: both defaults have changed once already, and each change
inverted a previously recorded finding.
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
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)

RTOL = 1e-6  # well past the march's 1% inexact-Newton stop, so arms are separated rather than tied
# (state entering step N, the beta that step ran at). The converged tail tied every arm at 6 cycles, so
# these are the march's OWN hard steps: 51 is the retry whose line search collapsed to alpha = 0, and 29
# runs below the preconditioner floor, so the V-cycle is deliberately mismatched to the operator.
#: ``(checkpoint index, that step's beta, label)`` -- read off ``march.log``; see the module docstring.
HARD_STATES = [
    (50, 0.0520, "entering step 51 (19 cyc, alpha -> 0, retried)"),
    (28, 0.0154, "entering step 29 (13 cyc, alpha 0.200, below the shift floor)"),
]

#: The variants to compare. ``{}`` is the shipped bundle; anything else overrides it through the
#: ``extra_options`` seam. Keep the shipped arm first as the control.
ARMS = [
    ("shipped", {}),
    ("smoothed aggregation", {"pc_gamg_agg_nsmooths": 1}),
    ("thr=0.05 (5 levels)", {"pc_gamg_threshold": 0.05}),
]
FLOOR = (
    compare.PC_BETA_FLOOR
)  # 0.05 -- the V-cycle is built here while the operator keeps MARCH_BETA


def load_state(index):
    path = CASE / f"checkpoints/state-{index:05d}.npz"
    d = np.load(path)
    print(
        f"state {path.name}: step {int(d['step'])}, |R| {float(d['residual_norm']):.3e}, "
        f"march beta {float(d['shift']):.4f}",
        flush=True,
    )
    return jnp.asarray(d["state"])


def arm(label, coupled, state, rhs, op_shift, pc_shift, colouring, structure, n_fields, options):
    """Build one V-cycle and solve the REAL system with it; report cycles and the TRUE residual."""
    t0 = time.time()
    pc = MonolithicAmgPreconditioner.build(
        lambda v: _jacobian_matvec(coupled, state, v),
        colouring,
        n_fields,
        pc_shift,
        smoother_fill_levels=compare.FILL_LEVELS,
        smoother_sweeps=compare.SWEEPS,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        batched_matvec=lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        probe_batch_size=_PROBE_BATCH_SIZE,
        structure=structure,
        extra_options=options or None,
    )
    build_s = time.time() - t0

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    t1 = time.time()
    x, cycles = solve_linear(
        operator,
        rhs,
        relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=200),
        preconditioner=pc.matvec(),
        throw=False,
    )
    true = float(jnp.linalg.norm(operator(x) - rhs) / jnp.linalg.norm(rhs))
    levels = pc.factors._pc.getMGLevels()
    print(
        f"  {label:<28} levels {levels}  build {build_s:>5.0f}s  cycles {int(cycles):>4}  "
        f"TRUE rel {true:.3e}  solve {time.time() - t1:>4.0f}s",
        flush=True,
    )
    del pc
    gc.collect()
    return int(cycles), true


def main():
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    reach3 = _coupled_jacobian_colouring(coupled, 3)
    struct3 = block_stencil_gather_map(reach3, n_fields)
    for index, march_beta, label in HARD_STATES:
        state = load_state(index)
        base = _coupled_shift_policy(coupled, state, "twolevel")
        op_shift = _frozen_shift_diagonal(base, march_beta, state)
        pc_shift = _frozen_shift_diagonal(base, max(march_beta, FLOOR), state)
        rhs = -coupled.residual(state)
        print(
            f"\n{'=' * 78}\n{label}\n  operator beta {march_beta}, V-cycle beta "
            f"{max(march_beta, FLOOR)}, real rhs |R| {float(jnp.linalg.norm(rhs)):.3e}",
            flush=True,
        )
        for name, options in ARMS:
            arm(name, coupled, state, rhs, op_shift, pc_shift, reach3, struct3, n_fields, options)


if __name__ == "__main__":
    main()
