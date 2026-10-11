"""A preconditioner for a system bordered by one scalar unknown, built from one for the system inside.

A constraint held by a Lagrange multiplier borders a residual with one extra row and its state with one
extra unknown,

    J_aug = [[J,   a],
             [c^T, 0]],

where ``a`` is the column the multiplier enters the residual through and ``c`` the row the constraint
reads the state through. A preconditioner ``M ~ J^{-1}`` for the inner block does not apply to that
system as it stands, but it extends to one by eliminating the scalar through its 1x1 Schur complement
(constraint preconditioning):

    y      = M r_inner
    dbeta  = (c^T y - r_border) / (c^T M a)      # c^T M a approximates c^T J^{-1} a
    dw     = y - dbeta (M a)

One application costs one application of ``M`` plus a few inner products, and the result is exact when
``M = J^{-1}``. The vectors ``a`` and ``c`` may be assembled by hand: a preconditioner changes only how
quickly a Krylov solve converges, never the solution or its derivative.

Nothing here knows what the constraint is. Where the scalar sits in the bordered vector is the
:class:`ScalarBorder`'s to say, so the layout is decided in one place for the residual, the seed and the
preconditioner alike.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import jax.numpy as jnp

from .state import FieldLayout

__all__ = ["ScalarBorder", "bordered_preconditioner"]

_Matvec = Callable[[jnp.ndarray], jnp.ndarray]
_PreconditionerFactory = Callable[[jnp.ndarray], _Matvec]


class ScalarBorder(Protocol):
    """Where a single bordering unknown sits in a vector, given the layout of the block it borders."""

    def split(self, fields: FieldLayout, bordered: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The inner part, shape ``(fields.size,)``, and the scalar border entry of ``bordered``."""
        ...

    def join(self, fields: FieldLayout, inner: jnp.ndarray, border: jnp.ndarray) -> jnp.ndarray:
        """The bordered vector holding ``inner``, shape ``(fields.size,)``, and the scalar ``border``."""
        ...


def bordered_preconditioner(
    inner_preconditioner: _PreconditionerFactory,
    border: ScalarBorder,
    fields: FieldLayout,
    column: jnp.ndarray,
    row: jnp.ndarray,
) -> _PreconditionerFactory:
    """Extend a preconditioner for a block to the block bordered by one scalar unknown.

    Parameters
    ----------
    inner_preconditioner : callable
        Factory ``state -> (matvec ~ J^{-1})`` for the inner block, called with the inner part of the
        bordered state.
    border : ScalarBorder
        Where the scalar sits in a bordered vector; the same object the bordered residual is assembled
        with, so the preconditioner and the operator it approximates cannot disagree about the split.
    fields : FieldLayout
        The layout of the inner block ``M`` inverts.
    column, row : jnp.ndarray
        The border column ``a`` and row ``c``, shape ``(fields.size,)``.

    Returns
    -------
    callable
        Factory ``bordered_state -> (matvec ~ J_aug^{-1})`` for the bordered system.

    Notes
    -----
    With an exact inner inverse the result is the exact inverse of the bordered matrix, so a Krylov
    solve preconditioned by it converges in one iteration. With any inner preconditioner the result
    satisfies the border row exactly: the scalar is eliminated, not approximated.
    """

    def factory(bordered: jnp.ndarray) -> _Matvec:
        inner_matvec = inner_preconditioner(border.split(fields, bordered)[0])
        m_column = inner_matvec(column)  # M a
        schur = jnp.dot(row, m_column)  # c^T M a, approximating c^T J^{-1} a

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            inner, border_residual = border.split(fields, residual)
            y = inner_matvec(inner)  # M r_inner
            d_border = (jnp.dot(row, y) - border_residual) / schur
            return border.join(fields, y - d_border * m_column, d_border)

        return apply

    return factory
