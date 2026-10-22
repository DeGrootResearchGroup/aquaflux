"""Keep a materialized preconditioner's diagonal pivots away from zero.

A frozen inverse fitted to a shifted Jacobian inverts each row's diagonal block. Where a row's shifted
diagonal ``J_ii + s_i`` passes near zero -- a source term whose derivative cancels the row's transport and
the pseudo-time shift, say -- the inverse amplifies that row by orders of magnitude more than any other.
In double precision a Krylov solve copes, since the preconditioner is still one fixed linear map. In a
reduced precision it does not: the amplified entries sit beside ordinary ones in the same vector, so
rounding relative to the amplified ones destroys the ordinary ones, and the operator times the
preconditioner, which brings the amplified entries back to ordinary size by cancellation, keeps that
rounding as an error of order one near those rows. The solve then stalls at its cap, and the direction
it returns depends on rounding.

:class:`PivotFloor` changes only what the inverse is **fitted** to: a pivot too close to zero is replaced,
for the fit, by its row's own pseudo-time diagonal, keeping its sign. The operator the Krylov solve
applies, the residual and its root are untouched, so the converged state and its adjoint are unchanged;
the inverse simply preconditions those rows as if they were ordinary.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

__all__ = ["PivotFloor"]


@dataclasses.dataclass(frozen=True)
class PivotFloor:
    """Fit a materialized inverse with every near-zero diagonal pivot moved out to its row's scale.

    A row's pivot is its shifted diagonal ``p = J_ii + s_i``, and its scale is the base pseudo-time
    diagonal ``d_i`` the shift is formed from (the step's ``β scale(β) d``). A row with ``d_i > 0`` and
    ``|p| < threshold * d_i`` is fitted with the pivot ``sign(p) * d_i`` instead (a pivot of exactly zero
    taking ``+d_i``); every other row, and every row with no shift, is fitted as it is.

    The replacement is the row's own pseudo-time diagonal because that is the size of an ordinary pivot:
    for a transported quantity it is the row's transport coefficient, so a replaced row is preconditioned
    like its neighbours. Keeping the sign keeps the fit closest to the operator actually solved -- a
    negative pivot moved to the other side of zero is a larger change than one moved away from zero on its
    own side. Rows whose pivot is already of ordinary size, negative ones included, are left alone: only
    the ones near zero are amplified enough to need it.

    Attributes
    ----------
    threshold : float
        The fraction of its row's pseudo-time diagonal within which a pivot counts as near zero, in
        ``(0, 1)``. Below ``1``, a replaced pivot (of magnitude ``d_i``) always ends up outside the band
        it was taken from.

    Raises
    ------
    ValueError
        If ``threshold`` is not strictly between ``0`` and ``1``.

    Examples
    --------
    >>> import numpy as np
    >>> floor = PivotFloor(threshold=0.15)
    >>> pivot = np.array([1.0e-6, -1.0e-6, 5.0e-5, -7.0e-5, 0.0])
    >>> scale = np.array([2.0e-5, 2.0e-5, 2.0e-5, 2.0e-5, 0.0])
    >>> pivot + np.asarray(floor.correction(pivot, scale))
    array([ 2.e-05, -2.e-05,  5.e-05, -7.e-05,  0.e+00])
    """

    threshold: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                f"PivotFloor.threshold is a fraction of the row's pseudo-time diagonal and must be "
                f"strictly between 0 and 1, got {self.threshold!r}."
            )

    def correction(
        self, pivot: jnp.ndarray, scale: jnp.ndarray, *, rows: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """What to add to each pivot so the fit sees it floored, shape ``(n_dofs,)``.

        Computed on the device, beside the diagonal it corrects: the arrays may be host or device ones,
        and the result is a device array.

        Parameters
        ----------
        pivot : jnp.ndarray
            Each row's shifted diagonal ``J_ii + s_i`` as the fit would otherwise see it, shape
            ``(n_dofs,)``.
        scale : jnp.ndarray
            Each row's base pseudo-time diagonal ``d_i``, shape ``(n_dofs,)``; zero (or negative) on a row
            that carries no shift, which is never changed.
        rows : jnp.ndarray, optional
            Boolean, shape ``(n_dofs,)``: the rows that may be changed at all. ``None`` (default) allows
            every row.

        Returns
        -------
        jnp.ndarray
            ``sign(p) d - p`` on the rows replaced and ``0`` elsewhere, shape ``(n_dofs,)``.
        """
        return _correction(
            jnp.asarray(pivot, dtype=jnp.float64),
            jnp.asarray(scale, dtype=jnp.float64),
            None if rows is None else jnp.asarray(rows, dtype=bool),
            self.threshold,
        )


@jax.jit
def _correction(
    pivot: jnp.ndarray, scale: jnp.ndarray, rows: jnp.ndarray | None, threshold: float
) -> jnp.ndarray:
    """:meth:`PivotFloor.correction`, compiled (run eagerly, each operation is a dispatch of its own)."""
    # A row with no shift (d <= 0) can never fall inside the band, since |p| >= 0 >= threshold * d.
    replaced = jnp.abs(pivot) < threshold * scale
    if rows is not None:
        replaced &= rows
    side = jnp.where(pivot < 0.0, -1.0, 1.0)
    return jnp.where(replaced, side * scale - pivot, 0.0)
