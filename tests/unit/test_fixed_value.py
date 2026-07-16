"""Unit tests for the FixedValueCells cell-value fixation."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.discretization import FixedValueCells


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
