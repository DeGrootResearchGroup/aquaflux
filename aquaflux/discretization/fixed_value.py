"""Strong cell-value fixation: replace a set of cells' residual rows with ``phi - target``.

Most of a residual is the finite-volume balance, but a few cells sometimes carry a strong algebraic
constraint instead -- a reference pressure pinned in a closed domain (where the level is otherwise
free), or the near-wall specific dissipation rate fixed to its analytical value in a turbulence
model. :class:`FixedValueCells` replaces the residual of a chosen set of cells with the constraint
``phi[cell] - target[cell]``, so the solver drives those cells to the target while every other cell
keeps its balance. The target is a differentiable leaf, so a constraint value that depends on
parameters (a wall value formed from the viscosity and wall distance) is differentiated like any
other input.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class FixedValueCells(eqx.Module):
    """Replace the residual of a fixed set of cells with ``phi - target``.

    Attributes
    ----------
    indices : jnp.ndarray
        The (distinct) cell indices whose residual rows are replaced, shape ``(n_fixed,)``.
    values : jnp.ndarray
        The target values for those cells, shape ``(n_fixed,)`` -- a differentiable leaf.
    """

    indices: jnp.ndarray
    values: jnp.ndarray

    def apply(self, residual: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
        """Return ``residual`` with the fixed cells' rows replaced by ``field - values``.

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
        return residual.at[self.indices].set(field[self.indices] - self.values)
