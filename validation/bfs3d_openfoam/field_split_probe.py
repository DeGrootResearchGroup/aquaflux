"""Does splitting the turbulence out of the coupled preconditioner's hierarchy help?

The coupled preconditioner puts all six fields -- the ``[u, v, w, p]`` saddle and the two transported
scalars ``k`` and ``omega`` -- through one multigrid hierarchy with one level smoother. This asks whether
giving the two groups their own hierarchies, while retaining one triangle of the coupling between them, is
better. It is **not** the block-*diagonal* arrangement that was tried and refuted: that one drops the
coupling, and the coupling is load-bearing. A triangle keeps half of it exactly.

**Where the headroom is, and is not.** At the states the march visits, a preconditioner rebuilt at the
iterate solves the forward system in one restart cycle, so the shipped forward pairing cannot separate two
candidates -- an easy operator is not a test, and a tie there is no information. Two operating points on the
same captured iterates do have room, and both are operationally real:

* **forward pairing at adjoint-grade tolerance** -- the same operator and shift the march solved, driven
  far past the march's own loose stop so the arms separate instead of all stopping at one cycle.
* **zero shift** -- the operator every gradient's transpose solve meets. Removing the pseudo-transient
  shift is what makes this operator hard, and no preconditioner floor rescues it, so this is where a
  better preconditioner has something to win. It is a real configuration, not a contrived one.

**The second question, which the same arms answer.** Every smoother the JAX-native multigrid in this
package has is Jacobi-class; the one thing PETSc supplies that it does not is the incomplete-LU sweep the
shipped bundle depends on -- which is also the least parallelizable component, being a sequential
triangular solve. So: is the ``[u, v, w, p]`` block alone smoothable by a Jacobi-class method even though
the six-field block is not? If it is, a field split is the route to a multigrid with no incomplete-LU
anywhere in it.

Two distinct obstructions could defeat a Jacobi-class smoother here, and the arms are chosen to tell them
apart, because they point at opposite conclusions:

* the ``omega`` **column is nearly empty** in the diagonal cell blocks -- perturbing ``omega`` in a cell
  barely changes that cell's own equations, since its influence travels by neighbour transport. A local
  smoother cannot see the field. If this is the obstruction, removing ``omega`` from the block fixes it.
* the ``[u, v, w, p]`` block is **indefinite** -- it is the saddle. Chebyshev acceleration assumes a
  bounded positive spectrum, which a saddle does not have. If *this* is the obstruction it survives the
  split untouched, because the split does not make the saddle definite.

A Chebyshev arm that fails on the four-field block as badly as on the six-field one indicts the
indefiniteness and closes the route; one that succeeds on four where it failed on six indicts ``omega``
and opens it.

**Method** -- each of these has produced a verdict on this case that had to be retracted:

* the TRUE residual through GMRES, never a preconditioned norm, a one-apply contraction, or a spectral
  radius;
* a REAL right-hand side, the march's own ``-R(state)`` at the captured iterate;
* the REAL shift diagonal ``beta * d``, not a uniform stand-in;
* one materialization per state, shared by every arm, so two arms can never differ for any reason but the
  options under test -- and so only one copy of a multi-gigabyte Jacobian is ever live;
* a **faithfulness gate**: the shipped monolithic arm must reproduce the cycle count already recorded for
  it at this iterate, or the run refuses to report. The captured iterates predate the current march log,
  so this replaces the usual join against it.

**Usage** -- one state per run, since each materializes a Jacobian of some gigabytes::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py inner-00050-03 > field_split.log 2>&1

Every smoother named here is a **fixed linear operator**, which the adjoint's transpose solve requires: a
Chebyshev smoother is a fixed polynomial once its eigenvalue bounds are estimated during setup, unlike a
GMRES-accelerated smoother, whose polynomial depends on the right-hand side.
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
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    build_amg_vcycle,
    build_block_triangular_field_split,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)

#: Adjoint-grade, far past the march's own 30 % inexact-Newton stop, so arms separate rather than tie.
RTOL = 1e-8
#: A failing arm is identified by its true residual; letting one run to thousands of matrix-vector
#: products costs more than every healthy arm together.
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=60)

#: ``name -> (operator beta, description)`` for the captured inner iterates. The shift is not in the file
#: -- the observer that wrote it is not told the shift -- and the march log that recorded it has since been
#: overwritten, so these are the two whose pairing is on record.
ITERATES = {
    "inner-00050-03": (
        0.0293,
        "the march's hardest solve: 15 cycles, line search collapsed to alpha 0",
    ),
    "inner-00040-03": (0.3333, "8 cycles at a healthy alpha = 1"),
}

#: What the shipped monolithic preconditioner is already recorded as doing at each iterate under the
#: forward pairing, used to gate the run. Cycles must match and the true residual must be within an order
#: of magnitude; a harness that cannot reproduce a recorded measurement is measuring something else.
RECORDED_FORWARD = {"inner-00050-03": (1, 6.6e-06), "inner-00040-03": (1, 1.7e-10)}

#: Level-smoother recipes, as PETSc options layered over the shipped bundle. ``ilu0`` is the shipped
#: default (an empty override). The other two are the Jacobi-class candidates a JAX-native multigrid could
#: actually implement, since neither needs a sequential triangular solve.
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


def load_iterate(name: str) -> jnp.ndarray:
    """The captured iterate, reporting what it was so the run records its own inputs."""
    data = np.load(CASE / "checkpoints" / f"{name}.npz")
    print(
        f"{name}: attempt {int(data['attempt'])} inner {int(data['inner'])}, the march took "
        f"{int(data['cycles'])} cycles at alpha {float(data['alpha']):.2e}, "
        f"|G| {float(data['g_before']):.4e} -> {float(data['g_after']):.4e}",
        flush=True,
    )
    return jnp.asarray(data["state"])


def materialize(coupled, state, colouring, structure, n_fields) -> sp.csr_matrix:
    """The **unshifted** field-major Jacobian at this iterate -- the one expensive step, done once.

    Unshifted because the operating points below differ only in the diagonal they add, and re-running a
    several-hundred-probe coloured jvp for each of them would dominate the run.
    """
    started = time.time()
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        colouring,
        n_fields,
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


def field_split(shifted, groups, n_fields, flow_smoother, turbulence_smoother, *, flow_first):
    """A hierarchy per field group, retaining one triangle of the coupling between them."""
    return MonolithicAmgPreconditioner(
        build_block_triangular_field_split(
            shifted,
            groups,
            flow_first=flow_first,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            leading_options=SMOOTHERS[flow_smoother] or None,
            trailing_options=SMOOTHERS[turbulence_smoother] or None,
        )
    )


#: ``(key, label, builder)``. The builder takes the shifted matrix rather than closing over it, so nothing
#: holds a reference to a multi-gigabyte operator the run wants to free between operating points.
ARMS = (
    # The control, and the two Jacobi-class smoothers applied to the SIX-field block -- the comparison
    # that says whether the split changes the answer for them.
    ("mono/ilu0", "monolithic, ILU(0)", lambda m, g, n: monolithic(m, g, n, "ilu0")),
    ("mono/cheb", "monolithic, Chebyshev", lambda m, g, n: monolithic(m, g, n, "chebyshev")),
    ("mono/jac", "monolithic, damped Jacobi", lambda m, g, n: monolithic(m, g, n, "jacobi")),
    # The split itself, both triangles, on the shipped smoother -- does ordering the coupling help at all?
    (
        "split flow/ilu0",
        "split flow-first, ILU(0) both",
        lambda m, g, n: field_split(m, g, n, "ilu0", "ilu0", flow_first=True),
    ),
    (
        "split turb/ilu0",
        "split turbulence-first, ILU(0) both",
        lambda m, g, n: field_split(m, g, n, "ilu0", "ilu0", flow_first=False),
    ),
    # The GPU question: a Jacobi-class smoother on the FOUR-field saddle, with the scalars kept on
    # incomplete-LU, then on both. If the four-field block is smoothable where six is not, these work.
    (
        "split flow/cheb",
        "split flow-first, Chebyshev on flow",
        lambda m, g, n: field_split(m, g, n, "chebyshev", "ilu0", flow_first=True),
    ),
    (
        "split flow/cheb both",
        "split flow-first, Chebyshev both",
        lambda m, g, n: field_split(m, g, n, "chebyshev", "chebyshev", flow_first=True),
    ),
)


def run_arm(label, preconditioner, built, coupled, state, rhs, op_shift):
    """Solve the REAL system with one already-built preconditioner; report cycles and the TRUE residual."""

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    solving = time.time()
    solution, raw = solve_linear(
        operator, rhs, SOLVER, preconditioner=preconditioner.matvec(), throw=False
    )
    true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
    cycles = restart_cycles(int(raw))
    print(
        f"    {label:<36} build {built:>5.0f}s  cycles {cycles:>4}  TRUE rel {true:.3e}  "
        f"solve {time.time() - solving:>4.0f}s",
        flush=True,
    )
    return cycles, true


def operating_point(name, coupled, state, rhs, jacobian, op_shift, pc_shift, groups, n_fields):
    """Every arm at one (operator shift, preconditioner shift) pairing, off the shared materialization."""
    print(f"\n  -- {name}", flush=True)
    shifted = MonolithicAmgPreconditioner._shifted(jacobian, pc_shift)
    results = {}
    for key, label, build in ARMS:
        preconditioner = None
        # A raise in one arm -- a singular coarse solve, a zero pivot, a failed eigenvalue estimate --
        # must not take the arms queued behind it, which by then represent most of the run's elapsed time.
        try:
            started = time.time()
            preconditioner = build(shifted, groups, n_fields)
            results[key] = run_arm(
                label, preconditioner, time.time() - started, coupled, state, rhs, op_shift
            )
        except Exception as failure:
            print(f"    {label:<36} FAILED  {type(failure).__name__}: {failure}", flush=True)
            results[key] = None
        finally:
            if preconditioner is not None:
                preconditioner.factors.destroy()
            del preconditioner
            gc.collect()
    del shifted
    gc.collect()
    return results


def gate(name, measured):
    """Refuse to report if the control does not reproduce what is already recorded for it."""
    expected_cycles, expected_true = RECORDED_FORWARD[name]
    if measured is None:
        raise SystemExit(
            f"FAITHFULNESS GATE FAILED for {name}: the control arm did not run at all."
        )
    cycles, true = measured
    if cycles != expected_cycles or not (expected_true / 10 <= true <= expected_true * 10):
        raise SystemExit(
            f"FAITHFULNESS GATE FAILED for {name}: the shipped monolithic arm gave {cycles} cycles at a "
            f"true relative residual of {true:.3e}, where {expected_cycles} cycles at ~{expected_true:.1e} "
            "is on record for this iterate and pairing. The harness is not solving the system the record "
            "describes; fix that before reading any other row."
        )
    print(
        f"    [gate passed: the control reproduces the recorded {expected_cycles} cycles]",
        flush=True,
    )


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ITERATES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{' | '.join(ITERATES)}>")
    name = sys.argv[1]
    march_beta, description = ITERATES[name]

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,  # u, v, w, p -- the saddle
        n_trailing_fields=2,  # k, omega -- the transported scalars
    )
    print(
        f"{'=' * 100}\nfield split: {groups.n_leading_fields} leading + {groups.n_trailing_fields} "
        f"trailing fields over {groups.n_cells} cells\nbundle: plain aggregation, "
        f"ILU({compare.FILL_LEVELS}) x{compare.SWEEPS} where not overridden, coarse_eq_limit "
        f"{compare.COARSE_EQ_LIMIT}, stencil reach 3, GMRES restart 15 to rtol {RTOL:.0e}\n{'=' * 100}",
        flush=True,
    )
    state = load_iterate(name)
    print(f"  {description}", flush=True)

    colouring = _coupled_jacobian_colouring(coupled, 3)
    structure = block_stencil_gather_map(colouring, n_fields)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    jacobian = materialize(coupled, state, colouring, structure, n_fields)

    # The shipped forward pairing, at adjoint-grade tolerance so the arms are not all tied at one cycle.
    # This is the pairing the faithfulness gate is recorded against.
    forward = operating_point(
        f"forward pairing: operator beta {march_beta}, V-cycle beta {max(march_beta, FLOOR)}",
        coupled,
        state,
        rhs,
        jacobian,
        _frozen_shift_diagonal(base, march_beta, state),
        _frozen_shift_diagonal(base, max(march_beta, FLOOR), state),
        groups,
        n_fields,
    )
    gate(name, forward["mono/ilu0"])

    # Zero shift: the operator every gradient's transpose solve meets, and the one place on this case with
    # genuine headroom left. No floor -- the adjoint has none.
    operating_point(
        "adjoint pairing: operator beta 0, V-cycle beta 0",
        coupled,
        state,
        rhs,
        jacobian,
        np.zeros(groups.n_dofs),
        np.zeros(groups.n_dofs),
        groups,
        n_fields,
    )


if __name__ == "__main__":
    main()
