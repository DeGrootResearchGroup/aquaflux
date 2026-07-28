"""Unit tests for the linear-solve preconditioning seam.

Left preconditioning must be **transparent**: it accelerates the Krylov iteration but changes
neither the converged solution nor its gradient. (Its convergence acceleration is exercised on
the real coupled-flow system, where the unpreconditioned block solve genuinely scales badly —
a synthetic matrix is a poor and fragile proxy for that.)
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from aquaflux.solve import relative_residual_gmres, solve_linear


def _system(theta):
    """A well-posed dense system A(theta) x = b with a widely spread diagonal."""
    n = 40
    rng = np.random.default_rng(0)
    a = jnp.diag(jnp.asarray(np.logspace(0.0, 3.0, n))) + 0.05 * jnp.asarray(
        rng.standard_normal((n, n))
    )
    a = a + theta * jnp.eye(n)
    b = jnp.asarray(rng.standard_normal(n))
    return a, b


def _jacobi(a):
    inverse_diagonal = 1.0 / jax.lax.stop_gradient(jnp.diag(a))
    return lambda v: inverse_diagonal * v


def test_preconditioner_does_not_change_solution() -> None:
    a, b = _system(2.0)
    plain, _ = solve_linear(lambda v: a @ v, b)
    preconditioned, _ = solve_linear(lambda v: a @ v, b, preconditioner=_jacobi(a))
    assert jnp.allclose(plain, preconditioned, atol=1e-8)


def test_preconditioner_is_gradient_transparent() -> None:
    """The gradient of a functional of the solution is identical with and without the preconditioner."""

    def loss(theta, use_preconditioner):
        a, b = _system(theta)
        precond = _jacobi(a) if use_preconditioner else None
        return jnp.sum(solve_linear(lambda v: a @ v, b, preconditioner=precond)[0])

    plain = jax.grad(lambda t: loss(t, False))(2.0)
    preconditioned = jax.grad(lambda t: loss(t, True))(2.0)
    assert jnp.allclose(plain, preconditioned, atol=1e-8)


def test_solve_returns_the_solution_and_a_positive_cycle_count() -> None:
    """``solve_linear`` returns ``(x, cycles)``: the solution, and what the solve cost."""
    a, b = _system(2.0)
    value, cycles = solve_linear(lambda v: a @ v, b)
    assert jnp.allclose(value, jnp.linalg.solve(a, b), atol=1e-8)
    assert int(cycles) > 0
    # Pinned dtype: a caller carries this through a lax.while_loop, whose carry must be invariant.
    assert cycles.dtype == jnp.int32


def test_counted_solve_reports_zero_for_a_direct_solver() -> None:
    """A solver that reports no iteration count (a direct factorization) yields 0, not an error."""
    a, b = _system(2.0)
    _, cycles = solve_linear(lambda v: a @ v, b, solver=lx.AutoLinearSolver(well_posed=True))
    assert int(cycles) == 0


def test_the_cycle_count_falls_when_the_preconditioner_improves() -> None:
    """The count measures how hard the *preconditioned* system was -- the staleness signal.

    This is the property a mid-march preconditioner refresh triggers on: on a fixed system the count
    rises as the frozen preconditioner drifts from the operator, and falls when it matches it again.
    Here the same system is solved with and without a matching Jacobi preconditioner; the
    well-preconditioned solve must take strictly fewer iterations, or the count would be measuring
    nothing useful.
    """
    a, b = _system(2.0)
    _, plain = solve_linear(lambda v: a @ v, b)
    _, preconditioned = solve_linear(lambda v: a @ v, b, preconditioner=_jacobi(a))
    assert int(preconditioned) < int(plain)


def _poisson_with_zero_entries():
    """A 2D-Poisson operator and a right-hand side with many *exactly zero* entries.

    The zero entries are the point: they are what makes ``lineax``'s stock componentwise stopping
    test degenerate. There the per-entry scale ``atol + rtol*|b_i|`` collapses to ``atol`` alone, an
    absolute demand, so under the default max-norm those entries hold the whole solve to ``~atol`` and
    force it far past the relative tolerance actually requested -- the exact regime a coupled saddle
    system lands in (wall-fixation rows that start satisfied have a near-zero right-hand side).
    """
    m = 24
    lap = 2.0 * jnp.eye(m) - jnp.eye(m, k=1) - jnp.eye(m, k=-1)
    a = jnp.kron(lap, jnp.eye(m)) + jnp.kron(jnp.eye(m), lap)
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(m * m)).at[jnp.arange(0, m * m, 3)].set(0.0)
    return a, b


def _true_relative_residual(a, b, x):
    return float(jnp.linalg.norm(a @ x - b) / jnp.linalg.norm(b))


def test_stock_gmres_over_solves_when_the_right_hand_side_has_zero_entries() -> None:
    """The pathology `relative_residual_gmres` exists to fix, pinned so it cannot silently change.

    Stock GMRES asked for a *loose* relative tolerance drives the residual orders of magnitude tighter
    than requested, because the zero right-hand-side entries pin it to the absolute ``atol`` floor.
    """
    a, b = _poisson_with_zero_entries()
    solver = lx.GMRES(rtol=1e-2, atol=1e-10, restart=8, stagnation_iters=50)
    x, _ = solve_linear(lambda v: a @ v, b, solver=solver, throw=False)
    # Asked for 1e-2; the zero entries force it far past that toward the 1e-10 floor.
    assert _true_relative_residual(a, b, x) < 1e-6


def test_relative_residual_gmres_stops_near_its_target_despite_zero_entries() -> None:
    """The global 2-norm relative stop lands near its target and does not chase the absolute floor."""
    a, b = _poisson_with_zero_entries()
    for rtol in (1e-1, 1e-2, 1e-3):
        solver = relative_residual_gmres(rtol, restart=8, stagnation_iters=50, max_restarts=300)
        x, _ = solve_linear(lambda v: a @ v, b, solver=solver, throw=False)
        achieved = _true_relative_residual(a, b, x)
        assert achieved <= rtol  # the target is actually met ...
        assert achieved >= rtol * 1e-3  # ... and not massively overshot toward machine precision


def test_relative_residual_gmres_is_cheaper_and_adaptive() -> None:
    """It stops far sooner than the over-solving stock solve, and a tighter target costs more cycles."""
    a, b = _poisson_with_zero_entries()
    _, stock = solve_linear(
        lambda v: a @ v,
        b,
        solver=lx.GMRES(rtol=1e-2, atol=1e-10, restart=8, stagnation_iters=50),
        throw=False,
    )
    counts = [
        int(
            solve_linear(
                lambda v: a @ v,
                b,
                solver=relative_residual_gmres(
                    rtol, restart=8, stagnation_iters=50, max_restarts=300
                ),
                throw=False,
            )[1]
        )
        for rtol in (1e-1, 1e-2, 1e-3)
    ]
    assert counts[0] < int(stock)  # a loose relative stop is far cheaper than the over-solve
    assert counts[0] <= counts[1] <= counts[2]  # tightening the target costs monotonically more


def test_relative_residual_gmres_is_hashable_for_the_static_solver_slot() -> None:
    """It is carried as a static field on the forward step, so it must be hashable and comparable."""
    solver = relative_residual_gmres(1e-2)
    assert hash(solver) is not None
    assert solver == relative_residual_gmres(1e-2)
