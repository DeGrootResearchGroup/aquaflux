"""Flat-vector layout for the coupled block flow state.

The monolithic ``(velocity, pressure)`` unknown is stored as one flat vector laid out
``[vel_0, vel_1, ..., vel_{dim-1}, pressure]`` (each block ``n_cells`` long). The slice arithmetic
that packs/unpacks that layout is a distinct responsibility from residual assembly, so it lives in
this small, mesh-free :class:`BlockStateLayout` object rather than being open-coded (and repeated)
across the residual and its preconditioner. It is testable in isolation — ``BlockStateLayout(dim,
n_cells)`` — and reusable by any future coupled system (e.g. energy coupling).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class BlockStateLayout(eqx.Module):
    """Pack/unpack of the flat ``[vel_0..vel_{dim-1}, pressure]`` block state.

    Attributes
    ----------
    dim : int
        Number of velocity components (spatial dimension), static.
    n_cells : int
        Number of cells (each block's length), static.
    """

    dim: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    @property
    def size(self) -> int:
        """Length of the flat state vector, ``(dim + 1) * n_cells``."""
        return (self.dim + 1) * self.n_cells

    def unpack(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split the flat state into velocity ``(n_cells, dim)`` and pressure ``(n_cells,)``.

        Parameters
        ----------
        state : jnp.ndarray
            Flat state vector, shape ``((dim + 1) * n_cells,)``.

        Returns
        -------
        velocity, pressure : jnp.ndarray
            Velocity ``(n_cells, dim)`` and pressure ``(n_cells,)``.
        """
        n = self.n_cells
        velocity = state[: self.dim * n].reshape(self.dim, n).T
        return velocity, state[self.dim * n :]

    def pack(self, velocity: jnp.ndarray, pressure: jnp.ndarray) -> jnp.ndarray:
        """Assemble per-component velocity and pressure into the flat vector.

        Parameters
        ----------
        velocity : jnp.ndarray
            Per-component velocity, shape ``(n_cells, dim)``.
        pressure : jnp.ndarray
            Pressure, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            Flat state vector, shape ``((dim + 1) * n_cells,)``.
        """
        # Component-first layout [vel_0, ..., vel_{dim-1}, pressure], each block (n_cells,): the
        # transpose makes the reshape lay the components out contiguously.
        return jnp.concatenate([velocity.T.reshape(-1), pressure])

    def zeros(self) -> jnp.ndarray:
        """A zero flat state vector, shape ``((dim + 1) * n_cells,)``."""
        return jnp.zeros(self.size)
