"""Strain-rate magnitude of a velocity field — the scalar invariant turbulence models consume.

The mean strain-rate tensor is the symmetric part of the velocity gradient,
``S_ij = 1/2 (du_i/dx_j + du_j/dx_i)``, and its magnitude ``S = sqrt(2 S_ij S_ij)`` drives the
turbulence production and the eddy-viscosity limiter. It is a pure function of the velocity-gradient
tensor (reconstructed one component at a time by a gradient scheme), independent of any turbulence
model.
"""

from __future__ import annotations

import jax.numpy as jnp


def strain_rate_magnitude(velocity_gradient: jnp.ndarray) -> jnp.ndarray:
    """Strain-rate magnitude ``S = sqrt(2 S_ij S_ij)`` per cell, shape ``(n_cells,)``.

    Parameters
    ----------
    velocity_gradient : jnp.ndarray
        The velocity-gradient tensor per cell, shape ``(n_cells, dim, dim)``, with
        ``velocity_gradient[c, i, j] = d u_i / d x_j``. Only its symmetric part enters, so the
        transpose convention does not matter.

    Returns
    -------
    jnp.ndarray
        The strain-rate magnitude per cell, shape ``(n_cells,)``.
    """
    strain = 0.5 * (velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2))
    return jnp.sqrt(2.0 * jnp.sum(strain * strain, axis=(-2, -1)))
