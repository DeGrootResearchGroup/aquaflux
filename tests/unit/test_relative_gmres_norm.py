"""Unit tests for the injected-norm relative forward-solve stop.

:func:`relative_residual_gmres` stops on a relative residual in an *injected* measure. The default
global 2-norm is dominated by whichever block has the largest right-hand side -- on the coupled saddle
the ``omega`` block, whose residual is orders above the flow -- so it halts once that block is resolved
and leaves the small-right-hand-side (flow) blocks blind. Passing the row-scaled march measure
(:class:`~aquaflux.solve.RowScaledNorm`) makes the stop see every block; a *loose* tolerance on that
measure keeps the solve cheap while never leaving the flow blind.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from aquaflux.solve import RowScaledNorm, relative_residual_gmres

# Restart well below the total dimension so GMRES stops on tolerance rather than reaching the exact
# solution within one Krylov cycle (only then does the stopping measure decide which blocks resolve).
_SOLVER_KW = dict(restart=20, stagnation_iters=200, max_restarts=60)


def _two_block_system():
    """A small easy block with a HUGE right-hand side, plus a large ill-conditioned block with an O(1)
    right-hand side -- the structure of the coupled saddle's ``omega`` vs flow split. The operator is
    diagonal, so its diagonal is the row-equilibration scale."""
    n0, n1 = 3, 200
    eig = jnp.asarray(np.geomspace(1.0, 1e4, n1))  # ill-conditioned: the big block needs many iters
    diagonal = jnp.concatenate([jnp.ones(n0), eig])

    def matvec(x):
        return diagonal * x

    b = jnp.concatenate([jnp.full(n0, 1e6), jnp.ones(n1)])  # block 0 huge RHS, block 1 O(1)
    return matvec, b, (n0, n1), diagonal


def _row_scaled_norm(b, sizes, diagonal):
    """The real :class:`RowScaledNorm`, built the way :func:`coupled_scaled_norm` builds it: row scale =
    the operator's own diagonal (each row's self-derivative), field scale = each block's equilibrated
    mean magnitude, so the two disparate-scale blocks are weighed comparably."""
    split = tuple(int(p) for p in np.cumsum(sizes)[:-1])
    equilibrated = jnp.abs(b) / diagonal
    field_scale = jnp.stack([jnp.mean(block) for block in jnp.split(equilibrated, split)])
    return RowScaledNorm(sizes=sizes, row_scale=diagonal, field_scale=field_scale)


def _block_norms(v, sizes):
    split = tuple(int(p) for p in np.cumsum(sizes)[:-1])
    return jnp.stack([jnp.linalg.norm(block) for block in jnp.split(v, split)])


def _solve(solver, matvec, b):
    operator = lx.FunctionLinearOperator(matvec, jax.ShapeDtypeStruct(b.shape, b.dtype))
    return lx.linear_solve(operator, b, solver=solver, throw=False).value


def test_default_two_norm_stop_leaves_the_row_scaled_measure_large():
    """The default (global 2-norm) stop resolves the huge-RHS block and halts, so the row-scaled measure
    -- which weighs every block comparably -- is left far above tolerance: the flow is blind. Passing no
    ``norm`` keeps this backward-compatible 2-norm behaviour."""
    matvec, b, sizes, diagonal = _two_block_system()
    norm = _row_scaled_norm(b, sizes, diagonal)
    x = _solve(relative_residual_gmres(1e-2, **_SOLVER_KW), matvec, b)
    rel = np.asarray(_block_norms(b - matvec(x), sizes) / _block_norms(b, sizes))
    assert rel[0] < 1e-2  # the loud block is resolved in the 2-norm
    assert float(norm(b - matvec(x)) / norm(b)) > 0.1  # but the row-scaled measure is left large


def test_injected_row_scaled_norm_stops_in_that_measure():
    """With ``norm=RowScaledNorm`` the solve stops when the *row-scaled* residual has fallen by ``rtol``
    -- so the flow block is driven down too, not left blind."""
    matvec, b, sizes, diagonal = _two_block_system()
    norm = _row_scaled_norm(b, sizes, diagonal)
    rtol = 1e-2
    x = _solve(relative_residual_gmres(rtol, norm=norm, **_SOLVER_KW), matvec, b)
    assert float(norm(b - matvec(x)) / norm(b)) <= 2 * rtol  # stopped in the injected measure


def test_loose_tolerance_resolves_the_flow_loosely_not_blindly():
    """A LOOSE tolerance on the row-scaled measure still *sees* the small block -- it falls to ~the loose
    tolerance rather than being left blind at ~1 as under the 2-norm."""
    matvec, b, sizes, diagonal = _two_block_system()
    norm = _row_scaled_norm(b, sizes, diagonal)
    x = _solve(relative_residual_gmres(2e-1, norm=norm, **_SOLVER_KW), matvec, b)
    rel = np.asarray(_block_norms(b - matvec(x), sizes) / _block_norms(b, sizes))
    assert rel[1] < 0.5  # the flow block is loosely resolved, not left blind at ~1


def test_injected_norm_is_jitmappable():
    """The stop lives inside a jitted march, so the whole solve must trace and run under ``jax.jit``."""
    matvec, b, sizes, diagonal = _two_block_system()
    solver = relative_residual_gmres(1e-2, norm=_row_scaled_norm(b, sizes, diagonal), **_SOLVER_KW)

    @jax.jit
    def solve(rhs):
        operator = lx.FunctionLinearOperator(matvec, jax.ShapeDtypeStruct(rhs.shape, rhs.dtype))
        return lx.linear_solve(operator, rhs, solver=solver, throw=False).value

    rel = np.asarray(_block_norms(b - matvec(solve(b)), sizes) / _block_norms(b, sizes))
    assert rel[1] < 5e-2  # the flow block is resolved under the row-scaled stop
