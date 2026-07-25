"""Unit tests for the FixedValueCells cell-value fixation."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.discretization import DifferenceRow, FixedValueCells, LogRatioRow


def test_replaces_only_the_fixed_rows() -> None:
    """The fixed rows become ``field - target``; every other row is untouched."""
    residual = jnp.array([10.0, 20.0, 30.0, 40.0])
    field = jnp.array([1.0, 2.0, 3.0, 4.0])
    fix = FixedValueCells(indices=jnp.array([1, 3]), values=jnp.array([5.0, 7.0]))
    out = fix.apply(residual, field)
    # row 1 -> 2 - 5 = -3; row 3 -> 4 - 7 = -3; rows 0, 2 unchanged.
    assert jnp.allclose(out, jnp.array([10.0, -3.0, 30.0, -3.0]))


def test_constraint_residual_vanishes_at_the_target() -> None:
    """When the field already equals the target, the fixed rows are zero (constraint satisfied)."""
    field = jnp.array([1.0, 2.0, 3.0])
    fix = FixedValueCells(indices=jnp.array([0, 2]), values=jnp.array([1.0, 3.0]))
    out = fix.apply(jnp.full(3, 9.0), field)
    assert jnp.allclose(out[jnp.array([0, 2])], 0.0)


def test_is_differentiable_in_the_target() -> None:
    field = jnp.array([1.0, 2.0, 3.0])

    def loss(values):
        fix = FixedValueCells(indices=jnp.array([0, 1]), values=values)
        return jnp.sum(fix.apply(jnp.zeros(3), field) ** 2)

    g = jax.grad(loss)(jnp.array([0.5, 0.5]))
    assert not bool(jnp.any(jnp.isnan(g)))


def test_difference_row_jacobian_scale_matches_ad_through_the_parametrization() -> None:
    """``d(phi - target)/d(w)`` is the chain factor itself, and agrees with automatic differentiation.

    Checked against AD of the row composed with ``phi = e**w`` so the analytical derivative cannot
    drift from the row it describes.
    """
    w = jnp.array([1.5, -0.5, 4.0])
    target = jnp.array([2.0, 0.3, 90.0])
    phi = jnp.exp(w)
    row = DifferenceRow()
    analytic = row.jacobian_scale(phi, chain=phi)
    ad = jax.grad(lambda ww: jnp.sum(row.row(jnp.exp(ww), target)))(w)
    assert jnp.allclose(analytic, ad)
    # The difference row inherits the exponential: its derivative is phi, not one.
    assert jnp.allclose(analytic, phi)


def test_log_ratio_row_jacobian_scale_is_one_for_a_log_solved_field() -> None:
    """``d(log(phi/target))/d(w) = 1`` when ``phi = e**w`` -- the property the row exists for.

    This is what separates the fixation rows from the transport rows of the same block, which carry
    ``d(phi)/d(w) = phi``: near a wall the two differ by orders of magnitude.
    """
    w = jnp.array([1.5, -0.5, 11.0])
    target = jnp.array([2.0, 0.3, 1.0e5])
    phi = jnp.exp(w)
    row = LogRatioRow()
    analytic = row.jacobian_scale(phi, chain=phi)
    ad = jax.grad(lambda ww: jnp.sum(row.row(jnp.exp(ww), target)))(w)
    assert jnp.allclose(analytic, ad)
    assert jnp.allclose(analytic, 1.0)


def test_log_ratio_row_jacobian_scale_matches_ad_for_a_directly_solved_field() -> None:
    """With ``phi`` itself the unknown the chain factor is one, so the derivative is ``1/phi``."""
    phi = jnp.array([2.0, 5.0, 100.0])
    target = jnp.array([1.0, 7.0, 20.0])
    row = LogRatioRow()
    analytic = row.jacobian_scale(phi, chain=jnp.ones_like(phi))
    ad = jax.grad(lambda p: jnp.sum(row.row(p, target)))(phi)
    assert jnp.allclose(analytic, ad)
    assert jnp.allclose(analytic, 1.0 / phi)
