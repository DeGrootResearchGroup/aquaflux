"""Strong cell-value fixation: replace a set of cells' residual rows with an algebraic constraint.

Most of a residual is the finite-volume balance, but a few cells sometimes carry a strong algebraic
constraint instead -- a reference pressure pinned in a closed domain (where the level is otherwise
free), or the near-wall specific dissipation rate fixed to its analytical value in a turbulence
model. :class:`FixedValueCells` replaces the residual of a chosen set of cells with a constraint that
vanishes exactly when ``phi[cell] == target[cell]``, so the solver drives those cells to the target
while every other cell keeps its balance. The target is a differentiable leaf, so a constraint value
that depends on parameters (a wall value formed from the viscosity and wall distance) is
differentiated like any other input.

**How the constraint is written is a first-class choice (:class:`FixationRow`), because it must match
the variable actually being solved for.** The difference ``phi - target`` is the natural form when the
solved unknown *is* ``phi``; it is a poor one when ``phi`` is a nonlinear function of the unknown,
because the row's linearization then inherits that nonlinearity. The canonical case is a field solved
in log form (``phi = e**w``): the difference row's Newton correction is ``dw = target/phi - 1``, which
is the *linearization of an exponential* and overshoots wildly when the ratio is far from one, whereas
the ratio row ``log(phi/target) = w - log(target)`` is **exactly linear in the unknown** and lands on
the constraint in a single full step from any ratio. Same root either way -- only the path and the
conditioning differ.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp


class FixationRow(Protocol):
    """How a value fixation is written as a residual row: a function vanishing at ``phi == target``.

    Structural interface only (a ``Protocol``). An implementation is a pure, elementwise map of the
    field and target values at the fixed cells to the residual entries there, so it traces inside the
    residual like any other term and is differentiated by automatic differentiation.
    """

    def row(self, phi: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """The residual entries at the fixed cells, vanishing exactly at ``phi == target``.

        Parameters
        ----------
        phi : jnp.ndarray
            The solved field's values at the fixed cells, shape ``(n_fixed,)``.
        target : jnp.ndarray
            The constraint values there, shape ``(n_fixed,)``.

        Returns
        -------
        jnp.ndarray
            The residual entries, shape ``(n_fixed,)``.
        """

    def jacobian_scale(self, phi: jnp.ndarray, chain: jnp.ndarray) -> jnp.ndarray:
        """``d(row)/d(unknown)`` at the fixed cells, given the field's own ``d(phi)/d(unknown)``.

        A block solved for a reparametrized unknown ``w`` (``phi = phi(w)``) has transport rows
        assembled in the physical field, so each carries the chain factor ``chain = d(phi)/d(w)``. A
        fixation row need not: it is written by this object, and may already be expressed in ``w``.
        So the two kinds of row can differ by orders of magnitude, and anything that scales the block
        per row -- a diagonal rescale of a frozen preconditioner built for the physical operator --
        must ask the row rather than assume the chain factor everywhere.

        Parameters
        ----------
        phi : jnp.ndarray
            The solved field's values at the fixed cells, shape ``(n_fixed,)``.
        chain : jnp.ndarray
            ``d(phi)/d(unknown)`` at those cells, shape ``(n_fixed,)`` (one for a field solved
            directly).

        Returns
        -------
        jnp.ndarray
            The row derivatives with respect to the solved unknown, shape ``(n_fixed,)``.
        """


class DifferenceRow(eqx.Module):
    """The plain difference ``phi - target`` -- the right form when the solved unknown is ``phi``.

    Linear in ``phi`` (unit derivative), so a full Newton step satisfies the constraint exactly in one
    iteration. The default.
    """

    def row(self, phi: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return phi - target

    def jacobian_scale(self, phi: jnp.ndarray, chain: jnp.ndarray) -> jnp.ndarray:
        """``d(phi - target)/d(unknown) = d(phi)/d(unknown)`` -- the chain factor itself."""
        del phi
        return chain


class LogRatioRow(eqx.Module):
    """The log ratio ``log(phi / target)`` -- the difference row's counterpart for a log-solved field.

    For a strictly positive field solved as ``phi = e**w`` this is ``w - log(target)``: **exactly
    linear in the solved unknown**, with unit derivative, so a full Newton step lands on the
    constraint in one iteration however far off ``phi`` starts. The plain difference row instead gives
    ``dw = target/phi - 1`` -- the linearization of an exponential -- which overshoots by ``e**(r-1)``
    against a target ratio ``r`` and forces the step length down. It also puts the row on the same
    scale as the solved variable rather than on the scale of ``phi`` itself, which for a field
    spanning orders of magnitude is the difference between a residual measure dominated by these rows
    and one that reflects the whole system.

    Both fields must be strictly positive, which is exactly the invariant the log parametrization
    maintains.
    """

    def row(self, phi: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(phi / target)

    def jacobian_scale(self, phi: jnp.ndarray, chain: jnp.ndarray) -> jnp.ndarray:
        """``d(log(phi/target))/d(unknown) = chain / phi`` -- exactly **one** for ``phi = e**w``.

        This is the whole point of the row under a log parametrization, and it is why a block-wide
        rescale by the chain factor is wrong here: the transport rows scale as ``phi``, these do not.
        """
        return chain / phi


class FixedValueCells(eqx.Module):
    """Replace the residual of a fixed set of cells with an algebraic constraint on ``phi``.

    Attributes
    ----------
    indices : jnp.ndarray
        The (distinct) cell indices whose residual rows are replaced, shape ``(n_fixed,)``.
    values : jnp.ndarray
        The target values for those cells, shape ``(n_fixed,)`` -- a differentiable leaf.
    row_form : FixationRow
        How the constraint is written (see the module docstring). Defaults to
        :class:`DifferenceRow` (``phi - target``); a field solved in log form should pass
        :class:`LogRatioRow` so the row is linear in the *solved* unknown.
    """

    indices: jnp.ndarray
    values: jnp.ndarray
    row_form: FixationRow = DifferenceRow()

    def apply(self, residual: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
        """Return ``residual`` with the fixed cells' rows replaced by the fixation row.

        Parameters
        ----------
        residual : jnp.ndarray
            The assembled cell residual, shape ``(n_cells,)``.
        field : jnp.ndarray
            The solved field whose fixed cells are constrained, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The residual with the fixed rows replaced, shape ``(n_cells,)``.
        """
        return residual.at[self.indices].set(self.row_form.row(field[self.indices], self.values))
