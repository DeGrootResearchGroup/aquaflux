"""Does the shipped field split precondition this case at ZERO shift -- the adjoint's own operator?

This decides whether a monolithic factorization still has a regime. Both the threshold-ILU and the
complete LU are dominated on the forward march, so the only place either can still earn its keep is
the implicit-function-theorem adjoint, which meets the Jacobian with no pseudo-transient shift to make
it diagonally dominant. If the field split handles zero shift here, nothing selects those two and they
can go; if it does not, whichever survives is what every ``jax.grad`` on this case depends on.

**Why this case and this state.** A march never visits zero shift -- the continuation ramps it and the
preconditioner is additionally floored (``compare.PC_BETA_FLOOR``), so no march measurement speaks to
it. The transpose solve at the converged root has no such protection. The same question asked at a
case's cold self-start answers something else entirely: measured there, the coupled Jacobian is nearly
singular (smallest pivot ``1.3e-12`` against a matrix 1-norm of ``278``) and even a complete LU is not
an accurate inverse of it -- so a cold-state failure says nothing about the adjoint. This runs at a
**converged checkpoint** instead.

⚠️ **UNIFORM reach, NOT the case's shipped ``COLUMN_REACH``.** ``(3, 3, 3, 3, 2, 2)`` is sound for the
field split alone, because a flow-first split never applies ``dR_flow/dturb`` and so never touches the
shortened k/omega columns. A monolithic factorization applies them, and a short colouring does not
truncate a column -- it folds far couplings onto near entries, corrupting the matrix. Probing every arm
uniformly is what makes them comparable.

**Method** (each of these has produced a retracted verdict somewhere in this project):

* the REAL right-hand side, ``-R(state)``, never a random vector;
* judged on the TRUE residual, never a preconditioned norm;
* the TRANSPOSE as well as the forward apply, since the transpose is what the adjoint calls;
* a GATE on the loaded state -- its residual against this case's OWN self-start, in one norm -- so a
  state written by a differently configured case cannot be silently measured against this one;
* one preconditioner in memory at a time -- the Jacobian is a couple of gigabytes a copy.

⚠️ **The recorded result below was taken with the `petsc` leading inverse**, which was this case's
default when it was measured: the field split converged at zero shift in 116 applications forward
(`6.07e-09`) and 117 transposed (`2.67e-09`), and at its shipped floor in 105 / 104. The default has
since moved to `hostilu`, so a fresh run measures a **different arm** unless `BFS3D_FLOW_INVERSE=petsc`
is set. Whether the host V-cycle preconditions the zero-shift operator as well is UNMEASURED, and it is
the first thing to re-run here.

Usage -- the checkpoint arrives as ENVIRONMENT, because the blessed launcher runs a script with no
arguments and forwards none; argv works for a direct invocation::

    BFS3D_ZERO_SHIFT_STATE=checkpoints/state-000NN.npz \
        validation/run_case.sh validation/bfs3d_openfoam/zero_shift_adjoint.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux.turbulence.coupled as C  # noqa: E402
import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from aquaflux.solve import native_nodal_inverse  # noqa: E402
from aquaflux.solve.sparse_jacobian import (  # noqa: E402
    materialize_block_jacobian,
    shifted_jacobian,
)
from aquaflux.turbulence import coupled_amg_continuation, hybrid_initialize  # noqa: E402

#: Far past the march's inexact-Newton stop so arms separate rather than tie; modest in restarts
#: because a failing arm is identified by its true residual long before it would converge.
RTOL, RESTART, MAX_RESTARTS = 1e-8, 30, 20

#: The reach every arm is probed at. Uniform deliberately -- see the module docstring.
REACH = 3

#: How far below the case's own self-start a state must sit to count as converged FOR THIS CASE. A
#: configuration mismatch moves the residual by orders, so this separates the two cleanly.
GATE = 100.0


def load_state(coupled, seed, path: Path):
    """The checkpoint, gated against this case's OWN self-start in a common norm.

    A saved state is only numbers; nothing in it says which mesh, Reynolds number or scheme set wrote
    it, so "this state belongs to this case" has to be checked rather than assumed.

    ⚠️ **Do not gate against the checkpoint's recorded ``residual_norm``.** That number is whatever
    measure the march was steered by, and this case marches with ``scaled_norm=True`` -- a
    row-equilibrated norm, not a Euclidean one. Comparing the two rejects a perfectly good checkpoint:
    ``state-00069`` records ``2.64e-06`` and computes ``1.04e-03`` here, a factor of 395 that is
    entirely the change of measure. Comparing a state against the SELF-START in whichever single norm
    this function uses is immune to that, because both ends move together.
    """
    data = np.load(path)
    state = jnp.asarray(data["state"])
    here = float(jnp.linalg.norm(coupled.residual(state)))
    start = float(jnp.linalg.norm(coupled.residual(seed)))
    if not np.isfinite(here) or here * GATE > start:
        raise SystemExit(
            f"{path.name} leaves |R| {here:.4e} against this case's self-start {start:.4e} -- not "
            f"converged by a factor of {GATE:g}. Either it was written by a different configuration, "
            f"or it is a mid-march state; either way the operator below is not the adjoint's."
        )
    print(
        f"{path.name}: step {int(data['step'])}, march shift {float(data['shift']):.4f}, "
        f"alpha {float(data['alpha']):.3f}; euclidean |R| {here:.4e}, "
        f"{start / here:.0f}x below this case's self-start "
        f"(the checkpoint's own {float(data['residual_norm']):.4e} is the march's SCALED measure, "
        f"not comparable)",
        flush=True,
    )
    return state


def measure(label, factors, a, rhs, transpose):
    """Applications and the TRUE relative residual, for one preconditioner against one operator."""
    n = a.shape[0]
    operator = a.T if transpose else a
    applies = [0]

    def apply(v):
        applies[0] += 1
        return factors.apply(np.asarray(v, dtype=np.float64), transpose=transpose)

    m = spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    began = time.perf_counter()
    x, _ = spla.gmres(operator, rhs, M=m, rtol=RTOL, restart=RESTART, maxiter=MAX_RESTARTS)
    wall = time.perf_counter() - began
    true = float(np.linalg.norm(operator @ x - rhs) / np.linalg.norm(rhs))
    # Three outcomes, kept distinct: a converged arm, one still descending when its budget ran out,
    # and one that diverged. Collapsing the last two into "failed" is how a budget gets cited as a
    # property of the method.
    verdict = (
        "converged"
        if true <= 1e-6
        else "DIVERGED"
        if true >= 1.0
        else f"hit the {MAX_RESTARTS}-restart budget, still descending"
    )
    print(
        f"  {label:22} {'Mt' if transpose else 'M ':3} applies {applies[0]:5d}  "
        f"true |r|/|b| {true:.3e}  {wall:6.1f}s  {verdict}",
        flush=True,
    )
    return true


def main() -> None:
    named = os.environ.get("BFS3D_ZERO_SHIFT_STATE") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not named:
        raise SystemExit(
            "no state given. Set BFS3D_ZERO_SHIFT_STATE=<state-000NN.npz> (the launcher forwards no "
            "arguments), or pass the path directly when running this script yourself."
        )
    path = Path(named)

    case = compare.build_case()
    coupled = case["coupled"]
    seed = coupled.state_from_physical(*hybrid_initialize(case["momentum"], case["turbulence"]))
    state = load_state(coupled, seed, path)
    rhs = -np.asarray(coupled.residual(state), dtype=np.float64)

    plan = C._coupled_jacobian_plan(coupled, REACH, None)
    frozen = jnp.asarray(state)
    a = shifted_jacobian(
        materialize_block_jacobian(lambda v: C._jacobian_matvec(coupled, frozen, v), plan).tocsr(),
        np.zeros(int(state.shape[0])),
    )
    # ⚠️ The leading inverse is named because it MOVED under this harness: the case's default flipped
    # from `petsc` to `hostilu`, so a result recorded without it cannot be told apart from a result for
    # the other arm. Everything the arms below depend on is printed, so a log is self-describing.
    print(
        f"{a.shape[0]} dofs, uniform reach {REACH}, nnz {a.nnz / 1e6:.2f} M; zero shift; "
        f"gmres rtol {RTOL}, restart {RESTART}; "
        f"leading inverse {compare.FLOW_INVERSE}, trailing {compare.TURBULENCE_INVERSE}, "
        f"smoother fill {compare.FILL_LEVELS}, sweeps {compare.SWEEPS}",
        flush=True,
    )

    def field_split(beta):
        return coupled_amg_continuation(
            coupled,
            state,
            amg_beta=beta,
            stencil_reach=REACH,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            field_split=compare.FIELD_SPLIT,
            trailing_smoother_sweeps=compare.TRAILING_SWEEPS,
            leading_inverse=compare.LEADING_INVERSE,
            trailing_inverse=native_nodal_inverse(**compare.NATIVE_TRAILING),
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
        )

    # Built at zero shift, and at the shipped floor against the same zero-shift operator: the second
    # is what the shipped code would actually do, the first is what "zero shift" implies. Reporting
    # one as the other is how a floor gets read as a property of the method.
    arms = {
        "field split @ beta=0": lambda: field_split(0.0),
        f"field split @ floor {compare.PC_BETA_FLOOR}": lambda: field_split(compare.PC_BETA_FLOOR),
    }

    engine = None
    for label, build in arms.items():
        try:
            began = time.perf_counter()
            engine = build()
            factors = engine.shift_policy.preconditioner.factors
            print(f"{label}  (built in {time.perf_counter() - began:.1f}s)", flush=True)
            measure(label, factors, a, rhs, transpose=False)
            measure(label, factors, a, rhs, transpose=True)
        except Exception as exc:  # a failing arm is a result; do not lose the others
            print(f"{label}  BUILD FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        finally:
            engine = None
            gc.collect()


if __name__ == "__main__":
    main()
