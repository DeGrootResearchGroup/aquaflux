"""Does anything but a monolithic factorization solve this case at ZERO shift?

The question this settles is whether the threshold-ILU and the complete LU still have a regime.
Both are dominated on the forward march -- the field-split multigrid marches this case faster than
either -- so the only place a monolithic factorization can still earn its keep is the **adjoint**,
which is the one solve that meets the Jacobian with no pseudo-transient shift to make it diagonally
dominant. If the field split handles zero shift, nothing selects the other two and they can go; if it
does not, whichever survives here is what every ``jax.grad`` on this case depends on.

**Why zero shift is the whole question.** A march never visits it: the shift is what the continuation
ramps, and the preconditioner is additionally floored (``compare.PC_BETA_FLOOR``) so it is never even
built below that. The transpose solve at the converged root has no such protection -- the shift has
vanished by construction -- so zero shift is the adjoint's operating point and no march measurement
speaks to it.

**Method, following the sibling case's preconditioner sweep** (each of these has produced a retracted
verdict somewhere in this project):

* the REAL right-hand side, ``-R(state)``, never a random vector;
* the REAL pairing -- the field split is measured BOTH built at zero shift and built at its shipped
  floor against the zero-shift operator, because the second is what the shipped code would actually
  do and the first is what the name "zero shift" implies. They are different configurations and
  reporting one as the other is how a floor gets read as a property of the method;
* judged on the TRUE residual, never a preconditioned norm -- a preconditioned-norm "win" has been
  recorded on this project before and was an artifact;
* the TRANSPOSE as well as the forward apply, since the transpose is what the adjoint actually calls
  and a factorization that cannot supply it cheaply is not an adjoint preconditioner at all;
* the Jacobian probed at ``compare.STENCIL_REACH``, which is measured exact on this mesh -- at the
  shorter reach the colouring folds far couplings onto near entries and every arm would be ranked on
  a matrix that is not the Jacobian;
* one preconditioner in memory at a time.

⚠️ **The state is the initial one, and that is a HARD state rather than the adjoint's own.** No
converged checkpoint of this case is kept, so this runs at ``hybrid_initialize``. The adjoint's true
operator is the Jacobian at the converged root, which is the *easier* operator -- so read a success
here as strong evidence the adjoint is safe, and a failure here as inconclusive about the adjoint
rather than as proof it fails. Recording which it was is the point; a number without its state cannot
be re-adjudicated later.
"""

from __future__ import annotations

import gc
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
from aquaflux.turbulence import (  # noqa: E402
    coupled_amg_continuation,
    coupled_ilut_continuation,
    coupled_lu_continuation,
    hybrid_initialize,
)

#: Far past the march's inexact-Newton stop, so arms separate rather than tie, and modest in restarts
#: because a failing arm is identified by its true residual long before it would converge.
RTOL, RESTART, MAX_RESTARTS = 1e-8, 30, 20


def build_operator(coupled, state, reach):
    """The zero-shift Jacobian every arm is measured against, assembled once.

    Shared rather than rebuilt per arm: the coloured probe dominates the cost, and two arms compared
    on separately-assembled matrices are not compared on the same operator.
    """
    plan = C._coupled_jacobian_plan(coupled, reach, None)
    frozen = jnp.asarray(state)
    zero = np.zeros(int(state.shape[0]))
    return shifted_jacobian(
        materialize_block_jacobian(lambda v: C._jacobian_matvec(coupled, frozen, v), plan).tocsr(),
        zero,
    )


def measure(label, factors, a, rhs, transpose):
    """Restart cycles and the TRUE residual, for one preconditioner against one operator."""
    n = a.shape[0]
    operator = a.T if transpose else a
    applies = [0]

    def apply(v):
        applies[0] += 1
        return factors.apply(np.asarray(v, dtype=np.float64), transpose=transpose)

    m = spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    began = time.perf_counter()
    x, info = spla.gmres(operator, rhs, M=m, rtol=RTOL, restart=RESTART, maxiter=MAX_RESTARTS)
    wall = time.perf_counter() - began
    true = float(np.linalg.norm(operator @ x - rhs) / np.linalg.norm(rhs))
    verdict = "converged" if true <= 1e-6 else "FAILED"
    print(
        f"  {label:26} {'Mt' if transpose else 'M ':3} applies {applies[0]:5d}  "
        f"true |r|/|b| {true:.3e}  {wall:6.1f}s  info {info:3d}  {verdict}",
        flush=True,
    )
    return true


def field_split_arm(coupled, state, beta):
    """The preconditioner `compare.py` ships, built at `beta`."""
    return coupled_amg_continuation(
        coupled,
        state,
        amg_beta=beta,
        stencil_reach=compare.STENCIL_REACH,
        smoother_fill_levels=compare.FILL_LEVELS,
        smoother_sweeps=compare.SWEEPS,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        field_split=True,
        trailing_smoother_sweeps=compare.TRAILING_SWEEPS,
        # The case's own selection, imported rather than re-branched here: a second copy of that
        # `petsc | native | hostilu` choice is how two files that must agree stop agreeing.
        leading_inverse=compare.LEADING_INVERSE,
        trailing_inverse=native_nodal_inverse(**compare.NATIVE_TRAILING),
        inner_steps=compare.INNER_STEPS,
        inner_tol=compare.INNER_TOL,
    )


def main() -> None:
    case = compare.build_case()
    coupled = case["coupled"]
    state = coupled.state_from_physical(*hybrid_initialize(case["momentum"], case["turbulence"]))
    if not bool(jnp.all(jnp.isfinite(coupled.residual(state)))):
        raise SystemExit("starting residual not finite -- nothing below would mean anything")

    rhs = -np.asarray(coupled.residual(state), dtype=np.float64)
    a = build_operator(coupled, state, compare.STENCIL_REACH)
    print(
        f"pitzDaily, {a.shape[0]} dofs, reach {compare.STENCIL_REACH}, nnz {a.nnz / 1e6:.2f} M; "
        f"state = hybrid_initialize (NOT the converged root -- see the module docstring); "
        f"zero shift; gmres rtol {RTOL}, restart {RESTART}",
        flush=True,
    )

    arms = {
        "field split @ beta=0": lambda: field_split_arm(coupled, state, 0.0),
        f"field split @ floor {compare.PC_BETA_FLOOR}": lambda: field_split_arm(
            coupled, state, compare.PC_BETA_FLOOR
        ),
        "ILUT @ beta=0": lambda: coupled_ilut_continuation(
            coupled,
            state,
            ilut_beta=0.0,
            stencil_reach=compare.STENCIL_REACH,
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
        ),
        "complete LU @ beta=0": lambda: coupled_lu_continuation(
            coupled,
            state,
            lu_beta=0.0,
            stencil_reach=compare.STENCIL_REACH,
            backend="scipy",
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
        ),
    }

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
            engine = factors = None
            gc.collect()


if __name__ == "__main__":
    main()
