"""The implicit-function-theorem adjoint attached to a root found by any means.

Every root here is found by a plain Python Newton loop on stopped inputs, so nothing about the
derivative can come from the iteration -- only from ``root_adjoint``.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.solve import TransposedPreconditioner, root_adjoint


def _cubic(x, theta):
    """x^3 + x - theta = 0; root x*(theta) with dx*/dtheta = 1/(3 x*^2 + 1)."""
    return x**3 + x - theta


def _eager_root(residual_fn, theta, x0):
    """A root found outside any transform: full Newton in a Python loop on a stopped ``theta``."""
    theta = jax.lax.stop_gradient(theta)
    x = x0
    for _ in range(60):
        r = residual_fn(x, theta)
        if float(jnp.linalg.norm(r)) < 1e-14:
            break
        jac = jax.jacfwd(lambda y: residual_fn(y, theta))(x)
        x = x - jnp.linalg.solve(jac, r)
    return x


_THETA = jnp.array([2.0, -5.0, 0.3])


def test_the_gradient_at_an_eagerly_found_root_matches_the_closed_form() -> None:
    root = _eager_root(_cubic, _THETA, jnp.zeros(3))

    grad = jax.grad(lambda t: jnp.sum(root_adjoint(_cubic, root, t)))(_THETA)

    assert jnp.allclose(grad, 1.0 / (3.0 * root**2 + 1.0), atol=1e-10)


def test_the_value_is_the_root_unchanged() -> None:
    root = _eager_root(_cubic, _THETA, jnp.zeros(3))
    assert jnp.array_equal(root_adjoint(_cubic, root, _THETA), root)


def test_a_derivative_the_root_already_carries_is_discarded() -> None:
    """The root's dependence on theta is the adjoint's to supply; whatever the root was built with is not.

    The root handed in here is numerically the root but carries a spurious derivative of 7 per unit
    theta. Passing that through instead of discarding it would add 7 to every component.
    """
    root = _eager_root(_cubic, _THETA, jnp.zeros(3))

    def objective(theta):
        tainted = root + 7.0 * (theta - jax.lax.stop_gradient(theta))
        return jnp.sum(root_adjoint(_cubic, tainted, theta))

    assert jnp.allclose(jax.grad(objective)(_THETA), 1.0 / (3.0 * root**2 + 1.0), atol=1e-10)


def test_it_differentiates_under_jit_and_vmap() -> None:
    root = _eager_root(_cubic, _THETA, jnp.zeros(3))
    expected = 1.0 / (3.0 * root**2 + 1.0)

    def grad_at(theta, found):
        return jax.grad(lambda t: jnp.sum(root_adjoint(_cubic, found, t)))(theta)

    assert jnp.allclose(jax.jit(grad_at)(_THETA, root), expected, atol=1e-10)

    thetas = jnp.stack([_THETA, _THETA + 1.0])
    roots = jnp.stack([root, _eager_root(_cubic, _THETA + 1.0, jnp.zeros(3))])
    assert jnp.allclose(jax.vmap(grad_at)(thetas, roots), 1.0 / (3.0 * roots**2 + 1.0), atol=1e-10)


class _Params(eqx.Module):
    source: jnp.ndarray
    stiffness: jnp.ndarray


def _coupled(x, params):
    """A coupled nonlinear system whose parameters are a module, not a flat array."""
    laplacian = jnp.roll(x, 1) + jnp.roll(x, -1) - 2.0 * x
    return params.stiffness * x**3 + x - 0.2 * laplacian - params.source


def test_a_module_of_parameters_receives_every_cotangent() -> None:
    """Checked against central differences on each leaf, since this system has no closed form."""
    params = _Params(jnp.array([0.4, -1.2, 2.0, 0.7]), jnp.array(1.5))

    def objective(p):
        found = _eager_root(_coupled, p, jnp.zeros(4))
        return jnp.sum(jnp.sin(root_adjoint(_coupled, found, p)))

    grad = eqx.filter_grad(objective)(params)

    h = 1e-6
    for i in range(4):
        bump = jnp.zeros(4).at[i].set(h)
        up = objective(_Params(params.source + bump, params.stiffness))
        down = objective(_Params(params.source - bump, params.stiffness))
        assert jnp.allclose(grad.source[i], (up - down) / (2 * h), atol=1e-7)
    up = objective(_Params(params.source, params.stiffness + h))
    down = objective(_Params(params.source, params.stiffness - h))
    assert jnp.allclose(grad.stiffness, (up - down) / (2 * h), atol=1e-7)


# Strongly non-symmetric, so a preconditioner applied without its transpose is far from inverting the
# transpose operator.
_A = jnp.array([[2.0, 5.0, 0.0], [0.0, 2.0, 5.0], [0.0, 0.0, 2.0]])


def _linear(x, theta):
    return _A @ x - theta


def _forward_inverse(state):
    del state
    return lambda r: jnp.linalg.solve(_A, r)


def _already_transposed(state):
    del state
    return lambda r: jnp.linalg.solve(_A.T, r)


@pytest.mark.parametrize(
    "preconditioner",
    [_forward_inverse, TransposedPreconditioner(_already_transposed)],
    ids=["transposed-here", "supplied-transposed"],
)
def test_the_adjoint_preconditioner_is_applied_as_the_transpose(preconditioner) -> None:
    """The transpose solve inverts ``A^T``, so its preconditioner must be ``M^T``, not ``M``.

    Both preconditioners here make ``M^T`` exactly ``A^{-T}``, which a one-vector GMRES solves in a
    single cycle. The solver is too weak for anything else: ``M`` applied untransposed, a supplied
    transpose transposed a second time, or no preconditioner at all each leave it short of tolerance.
    """
    theta = jnp.array([1.0, -2.0, 0.5])
    root = jnp.linalg.solve(_A, theta)
    weak = lx.GMRES(rtol=1e-12, atol=1e-12, restart=1, max_steps=4)

    grad = jax.grad(
        lambda t: jnp.sum(
            root_adjoint(
                _linear, root, t, adjoint_solver=weak, adjoint_preconditioner=preconditioner
            )
            * jnp.array([1.0, 2.0, 3.0])
        )
    )(theta)

    # dx/dtheta = A^{-1}, so the gradient of w . x is A^{-T} w.
    assert jnp.allclose(grad, jnp.linalg.solve(_A.T, jnp.array([1.0, 2.0, 3.0])), atol=1e-10)
