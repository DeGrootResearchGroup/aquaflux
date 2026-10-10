"""The preconditioner for a system bordered by one scalar unknown, on small dense matrices.

No mesh and no physics: an inner block ``J``, a border column ``a`` and row ``c``, and a stub border
that says where the scalar sits. Both ends are exercised, because the position is the border's to
decide and a preconditioner that assumed one end would be right for one of them only.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import FieldLayout, bordered_preconditioner

FIELDS = FieldLayout.cell_fields(3, phi=2)  # an inner block of six unknowns


class _Trailing:
    """The scalar after the inner block."""

    def split(self, fields, bordered):
        return bordered[: fields.size], bordered[fields.size]

    def join(self, fields, inner, border):
        return jnp.concatenate([inner, jnp.atleast_1d(border)])


class _Leading:
    """The scalar before the inner block."""

    def split(self, fields, bordered):
        return bordered[1:], bordered[0]

    def join(self, fields, inner, border):
        return jnp.concatenate([jnp.atleast_1d(border), inner])


def _system(seed: int = 0):
    """A well-conditioned inner block and a border pair whose Schur complement is far from zero."""
    rng = np.random.default_rng(seed)
    n = FIELDS.size
    inner = rng.standard_normal((n, n)) + 4.0 * np.eye(n)
    column = rng.standard_normal(n)
    row = rng.standard_normal(n)
    return jnp.asarray(inner), jnp.asarray(column), jnp.asarray(row)


def _bordered_matrix(border, inner, column, row) -> jnp.ndarray:
    """``[[J, a], [c^T, 0]]`` with the scalar where ``border`` puts it, assembled column by column."""

    def apply(v):
        w, beta = border.split(FIELDS, v)
        return border.join(FIELDS, inner @ w + beta * column, jnp.dot(row, w))

    size = FIELDS.size + 1
    return jnp.stack([apply(e) for e in jnp.eye(size)], axis=1)


@pytest.mark.parametrize("border", [_Trailing(), _Leading()], ids=["trailing", "leading"])
def test_an_exact_inner_inverse_gives_the_exact_bordered_inverse(border) -> None:
    inner, column, row = _system()
    seen = []

    def exact(state):
        seen.append(state)
        return lambda r: jnp.linalg.solve(inner, r)

    bordered_state = jnp.arange(FIELDS.size + 1.0)
    apply = bordered_preconditioner(exact, border, FIELDS, column, row)(bordered_state)
    matrix = _bordered_matrix(border, inner, column, row)
    v = jnp.asarray(np.random.default_rng(1).standard_normal(FIELDS.size + 1))
    np.testing.assert_allclose(apply(matrix @ v), v, rtol=0, atol=1e-12)
    # The inner factory is handed the inner part of the state, not the bordered vector.
    np.testing.assert_array_equal(seen[0], border.split(FIELDS, bordered_state)[0])


@pytest.mark.parametrize("border", [_Trailing(), _Leading()], ids=["trailing", "leading"])
def test_any_inner_preconditioner_meets_the_border_row_exactly(border) -> None:
    """``c^T dw = r_border`` holds for every ``M``: the scalar is eliminated, not approximated.

    With a deliberately poor inner inverse (the inverse diagonal) the inner rows are far from solved,
    but the border row of ``J_aug`` reads only ``c^T dw``, and the Schur step makes that exact.
    """
    inner, column, row = _system(seed=2)
    diagonal = jnp.diag(inner)
    apply = bordered_preconditioner(
        lambda _state: lambda r: r / diagonal, border, FIELDS, column, row
    )(jnp.zeros(FIELDS.size + 1))
    residual = jnp.asarray(np.random.default_rng(3).standard_normal(FIELDS.size + 1))
    dw, _ = border.split(FIELDS, apply(residual))
    _, r_border = border.split(FIELDS, residual)
    assert float(jnp.dot(row, dw)) == pytest.approx(float(r_border), abs=1e-12)
    # ...while the inner rows are not solved, so the check above is not passing on an exact inverse.
    matrix = _bordered_matrix(border, inner, column, row)
    assert float(jnp.linalg.norm(matrix @ apply(residual) - residual)) > 1e-2
