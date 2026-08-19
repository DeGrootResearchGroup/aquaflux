"""Does MSIMPLER win as a drop-in for the bfs3d case's actual shipped leading inverse?

Every prior measurement of ``schur_scaling="msimpler"`` on this case's real operator
(``field_split_probe.py``'s ``split msimpler/ilu0`` family) paired it with an ``ilu0`` trailing
inverse, which is not what ``bfs3d`` ships -- the case's ``TURBULENCE_INVERSE`` default is
``"native"`` (``compare.TRAILING_INVERSE = native_nodal_inverse(**compare.NATIVE_TRAILING)``). So
that comparison changed two things at once relative to the shipped bundle: the leading inverse AND
the trailing one.

This probe changes exactly one thing. Two arms, at the SAME materialized Jacobian, the SAME shift,
the SAME field-split wiring, and the SAME trailing inverse (``compare.TRAILING_INVERSE``, i.e.
``NodalNativeInverse`` at the case's own settings):

* **shipped** -- ``compare.LEADING_INVERSE`` (``host_ilu_inverse`` at the case's own settings, the
  actual default the case ships as of this run).
* **msimpler** -- the SAME construction ``field_split_probe.block_simple_arms`` uses for its
  ``msimpler`` arm (``BlockPreconditioner.build(momentum, velocity="convection",
  schur_scaling="msimpler", strength_threshold=0.25)``, built from the real assembler + eddy
  viscosity at the probed state, exactly as the shipped coupled shift policy does it), but paired
  with the shipped trailing inverse instead of ``ilu0``.

Method matches ``field_split_probe.py`` throughout: the TRUE residual through GMRES (never a
preconditioned norm), the REAL right-hand side ``-R(state)``, one materialization shared by both
arms, GMRES restart 15 to ``field_split_probe.RTOL`` on the true residual, capped at
``field_split_probe.MAX_RESTARTS`` restarts.

Usage::

    python3 -u validation/bfs3d_openfoam/msimpler_swap_probe.py [checkpoint-name]

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


def _msimpler_build(coupled, pc_state, pc_beta, trailing_inverse):
    """The `block_simple_arms` msimpler construction, verbatim, with the trailing inverse swapped."""
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
        schur_scaling="msimpler",
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

        for label, build in (
            (
                "shipped (hostilu leading + native trailing)",
                _shipped_build(compare.TRAILING_INVERSE),
            ),
            (
                "msimpler leading + native trailing (all else shipped)",
                _msimpler_build(coupled, state, pc_beta, compare.TRAILING_INVERSE),
            ),
        ):
            fsp.one_arm(
                label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, fsp.SOLVER
            )

        del shifted
        gc.collect()


if __name__ == "__main__":
    main()
