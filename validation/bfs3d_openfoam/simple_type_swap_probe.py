"""Does a SIMPLE-type block preconditioner win as a drop-in for bfs3d's shipped leading inverse?

Every prior measurement of the mass-scaled Schur on this case's real operator
(``field_split_probe.py``'s ``split msimple/ilu0`` family, and the first version of this probe)
tested the **lower block-triangular** composition -- one velocity solve, one Schur solve. That is
Klaij & Vuik (2013)'s MSIMPLE minus its closing velocity update, not their MSIMPLER: the pressure
prediction that distinguishes the ``R`` variants was not implemented at the time, and it is the axis
the paper measures the convergence benefit on. So the recorded ~8-9x cycle penalty was measured
against a method the name did not describe.

This probe changes exactly one thing per arm. Every arm shares the SAME materialized Jacobian, the
SAME shift, the SAME field-split wiring, and the SAME trailing inverse (``compare.TRAILING_INVERSE``,
i.e. ``NodalNativeInverse`` at the case's own settings) -- so only the leading (flow-saddle) inverse
differs:

* **shipped** -- ``compare.LEADING_INVERSE``, whatever the case ships as of this run.
* **msimple/<composition>** -- ``BlockPreconditioner.build(momentum, velocity="convection",
  schur_scaling="msimple", composition=..., strength_threshold=0.25)``, built from the real assembler
  and eddy viscosity at the probed state exactly as the shipped coupled shift policy does it, over
  every composition in ``COMPOSITIONS``. ``composition="simpler"`` is the paper's **MSIMPLER**;
  ``"triangular"`` is what every earlier measurement actually tested.

Method matches ``field_split_probe.py`` throughout: the TRUE residual through GMRES (never a
preconditioned norm), the REAL right-hand side ``-R(state)``, one materialization shared by every
arm, GMRES restart 15 to ``field_split_probe.RTOL`` on the true residual, capped at
``field_split_probe.MAX_RESTARTS`` restarts. Both the adjoint's operator (zero shift) and the march's
own shift are probed, because the two rank preconditioners differently on this case -- an incomplete
factorization gains a diagonal-dominance windfall from a shift that a multigrid does not.

Usage::

    python3 -u validation/bfs3d_openfoam/simple_type_swap_probe.py [checkpoint-name]

With no argument, probes the most recently written checkpoint in ``checkpoints/``.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE.parents[1]))
sys.path.insert(0, str(CASE))

import field_split_probe as fsp  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.flow.block_preconditioner import BlockPreconditioner  # noqa: E402

compare = fsp.compare


def _latest_checkpoint() -> str:
    files = sorted((CASE / "checkpoints").glob("*.npz"), key=lambda p: p.stat().st_mtime)
    ends = [p for p in files if "attempt" not in np.load(p).files]
    if not ends:
        raise SystemExit(f"no end-of-step checkpoint found under {CASE / 'checkpoints'}")
    return ends[-1].stem


def _load(name: str):
    data = np.load(CASE / "checkpoints" / f"{name}.npz")
    state = jnp.asarray(data["state"])
    shift = float(data["shift"])
    residual = float(data["residual_norm"])
    step = int(data["step"]) if "step" in data.files else -1
    print(f"{name}: end of step {step}, |R| {residual:.4e}, march shift {shift:.4f}", flush=True)
    return state, shift


#: Which SIMPLE-type compositions to sweep. ``"simpler"`` is the pressure-prediction variant, i.e.
#: the paper's MSIMPLER; ``"triangular"`` is what every measurement before this probe tested.
COMPOSITIONS = ("triangular", "simple", "simpler")


def _block_simple_build(coupled, pc_state, pc_beta, trailing_inverse, composition):
    """The `block_simple_arms` construction, verbatim, with the composition and trailing inverse set."""
    flow_ref, k_ref, omega_ref = coupled.physical_fields(pc_state)
    closure = coupled.turbulence.closure_fields(
        coupled.momentum.velocity_fields(flow_ref), k_ref, omega_ref
    )
    momentum = coupled.momentum.with_eddy_viscosity(closure.nu_t)
    flow = flow_ref
    n_flow = int(flow.shape[0])

    block = BlockPreconditioner.build(
        momentum,
        velocity="convection",
        reference_state=flow,
        schur_scaling="msimple",
        composition=composition,
        strength_threshold=0.25,
    )
    a_p = jax.lax.stop_gradient(block.frozen_momentum_diagonal(flow) * (1.0 + pc_beta))
    matvec = jax.jit(block.apply_at(flow, a_p))

    def build(shifted, groups, n_fields):
        return fsp.field_split(
            shifted,
            groups,
            n_fields,
            "ilu0",  # unused: leading_inverse below replaces the V-cycle wholesale
            "ilu0",  # unused: trailing_inverse below replaces it too
            flow_first=True,
            leading_inverse=lambda sub, n_sub: fsp.JaxNativeBlockInverse(matvec, n_flow),
            trailing_inverse=trailing_inverse,
        )

    return build


def _shipped_build(trailing_inverse):
    def build(shifted, groups, n_fields):
        return fsp.field_split(
            shifted,
            groups,
            n_fields,
            "ilu0",  # unused: leading_inverse below (compare.LEADING_INVERSE) replaces it
            "ilu0",  # unused: trailing_inverse below replaces it too
            flow_first=True,
            leading_inverse=compare.LEADING_INVERSE,
            trailing_inverse=trailing_inverse,
        )

    return build


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else _latest_checkpoint()
    state, march_beta = _load(name)

    print(
        f"bundle: field_split={compare.FIELD_SPLIT}, flow_inverse={compare.FLOW_INVERSE}, "
        f"turbulence_inverse={compare.TURBULENCE_INVERSE}, column_reach={compare.COLUMN_REACH}, "
        f"GMRES restart 15, rtol {fsp.RTOL:.0e}, max_restarts {fsp.MAX_RESTARTS}",
        flush=True,
    )

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = fsp.FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=2,
    )

    plan = fsp._coupled_jacobian_plan(coupled, 3, compare.COLUMN_REACH)
    structure = fsp.block_stencil_gather_map(plan)
    base = fsp._coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    print(f"right-hand side |R| {float(jnp.linalg.norm(rhs)):.4e}", flush=True)

    for beta_label, beta in (
        ("the converged/march state at ZERO SHIFT (the adjoint's operator)", 0.0),
        (f"the march's own shift beta={march_beta:.4f}", march_beta),
    ):
        print(f"\n{'=' * 100}\noperator at {beta_label}\n{'=' * 100}", flush=True)
        pc_beta = max(beta, fsp.FLOOR) if beta > 0 else 0.0
        op_shift = fsp._frozen_shift_diagonal(base, beta, state) if beta > 0 else 0.0

        t0 = time.time()
        jacobian = fsp.materialize(coupled, state, plan, structure, n_fields)
        pc_shift = (
            fsp._frozen_shift_diagonal(base, pc_beta, state)
            if pc_beta > 0
            else np.zeros(groups.n_dofs)
        )
        shifted = fsp.MonolithicAmgPreconditioner._shifted(jacobian, pc_shift)
        del jacobian
        gc.collect()
        print(f"  materialized in {time.time() - t0:.0f}s", flush=True)

        arms = [
            (
                f"shipped ({compare.FLOW_INVERSE} leading + native trailing)",
                _shipped_build(compare.TRAILING_INVERSE),
            )
        ]
        arms += [
            (
                f"msimple/{composition} leading + native trailing (all else shipped)",
                _block_simple_build(coupled, state, pc_beta, compare.TRAILING_INVERSE, composition),
            )
            for composition in COMPOSITIONS
        ]
        for label, build in arms:
            fsp.one_arm(
                label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, fsp.SOLVER
            )

        del shifted
        gc.collect()


if __name__ == "__main__":
    main()
