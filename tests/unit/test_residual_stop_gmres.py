"""The residual-only GMRES stop: it meets its tolerance in its measure, and stops as soon as it does."""

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import (
    relative_residual_gmres,
    residual_stop_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.solve.linear import (
    in_progress_measure,
)

N = 120


def _problem(spread):
    """A diagonally preconditioned non-symmetric system and a weighted measure with a large spread."""
    rng = np.random.default_rng(3)
    a = np.eye(N) * 2.0 + rng.standard_normal((N, N)) * spread
    m = np.diag(1.0 / np.diag(a))
    weights = jnp.asarray(rng.uniform(0.1, 10.0, N))
    b = jnp.asarray(rng.standard_normal(N))
    return a, m, weights, b


def _solve(solver, a, m, b):
    return solve_linear(
        lambda v: jnp.asarray(a) @ v,
        b,
        solver=solver,
        preconditioner=lambda v: jnp.asarray(m) @ v,
        throw=False,
    )


@pytest.mark.parametrize("rtol", [0.3, 1e-2, 1e-7])
def test_it_meets_its_tolerance_in_its_own_measure_and_no_further(rtol):
    # A harder system, so a solve takes more than one restart cycle at the tight tolerance.
    a, m, weights, b = _problem(0.04)

    def norm(v):
        return jnp.linalg.norm(weights * v)

    counted = []
    solver = residual_stop_gmres(
        rtol, norm=norm, restart=6, max_restarts=40, on_solve=lambda i, c: counted.append((i, c))
    )
    x, cycles = _solve(solver, a, m, b)
    achieved = float(norm(b - jnp.asarray(a) @ x) / norm(b))
    assert achieved <= rtol
    applications, run = (int(v) for v in counted[-1])
    iterations = applications - run  # each cycle also applies the operator once for its residual
    # It stopped at the first iteration that met the tolerance: one fewer would not have.
    assert run == restart_cycles(int(cycles)) == -(-iterations // 6)
    if iterations > 1:
        cut = _cut_short(a, m, b, norm, iterations - 1, restart=6)
        assert cut > rtol


def _cut_short(a, m, b, norm, iterations, restart):
    """The residual GMRES(restart) reaches in exactly ``iterations`` iterations, by a dense reference."""
    op = a @ m
    x = np.zeros(N)
    r = np.asarray(b).copy()
    left = iterations
    while left:
        k = min(left, restart)
        basis = [r / np.linalg.norm(r)]
        h = np.zeros((k + 1, k))
        for j in range(k):
            w = op @ basis[j]
            for i in range(j + 1):
                h[i, j] = basis[i] @ w
                w = w - h[i, j] * basis[i]
            h[j + 1, j] = np.linalg.norm(w)
            basis.append(w / h[j + 1, j])
        e = np.zeros(k + 1)
        e[0] = np.linalg.norm(r)
        y = np.linalg.lstsq(h, e, rcond=None)[0]
        x = x + np.array(basis[:k]).T @ y
        r = np.asarray(b) - op @ x
        left -= k
    return float(norm(jnp.asarray(r)) / norm(b))


def test_it_does_less_work_than_the_lineax_stop_on_a_loose_tolerance():
    a, m, weights, b = _problem(0.04)

    def norm(v):
        return jnp.linalg.norm(weights * v)

    counted = []
    residual = residual_stop_gmres(
        0.3, norm=norm, restart=15, max_restarts=14, on_solve=lambda i, c: counted.append(int(i))
    )
    _solve(residual, a, m, b)
    _, raw = _solve(relative_residual_gmres(0.3, norm=norm, restart=15, max_restarts=14), a, m, b)
    lineax_applications = 1 + 16 * (int(raw) - 1)
    assert counted[-1] < 15 < lineax_applications


def test_it_follows_the_running_steps_measure_and_refuses_to_run_without_one():
    a, m, weights, b = _problem(0.04)
    unbound = residual_stop_gmres(0.3)
    with pytest.raises(TypeError, match="progress measure"):
        _solve(unbound, a, m, b)
    bound = in_progress_measure(unbound, lambda v: jnp.linalg.norm(weights * v))
    x, _ = _solve(bound, a, m, b)
    assert (
        float(jnp.linalg.norm(weights * (b - jnp.asarray(a) @ x)) / jnp.linalg.norm(weights * b))
        <= 0.3
    )
