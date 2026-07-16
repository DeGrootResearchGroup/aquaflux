"""Boundary values for the k-omega SST fields.

The k and omega transport equations use the generic scalar boundary closures -- a
:class:`~aquaflux.boundary.conditions.Dirichlet` value at an inlet or wall, a
:class:`~aquaflux.boundary.conditions.ZeroGradient` at an outlet -- so what is turbulence-specific
is only *what value* to impose. These helpers compute those values:

- :func:`omega_wall_value` -- the analytical near-wall omega, fixed in the wall-adjacent cell (via
  :class:`~aquaflux.discretization.fixed_value.FixedValueCells`), since a resolved viscous wall has
  no finite face value for omega.
- :func:`inlet_k` / :func:`inlet_omega` -- the free-stream k and omega from a turbulence intensity
  and a turbulent length scale (a Dirichlet value at the inlet). Wall k is simply zero.

All are pure functions of their inputs and the model constants, differentiable in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .sst import SSTModel


def omega_wall_value(nu: jnp.ndarray, d: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    """Near-wall omega ``60 nu / (beta_1 d**2)``, the value fixed in the wall-adjacent cell.

    Parameters
    ----------
    nu : jnp.ndarray
        Kinematic (molecular) viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_1``).

    Returns
    -------
    jnp.ndarray
        The near-wall omega per wall-adjacent cell, shape ``(n_wall,)``.
    """
    return 60.0 * nu / (model.beta_1 * d**2)


def inlet_k(velocity_magnitude: jnp.ndarray, intensity: float) -> jnp.ndarray:
    """Inlet turbulent kinetic energy from a turbulence intensity: ``k = 1.5 (U I)**2``.

    Parameters
    ----------
    velocity_magnitude : jnp.ndarray
        The inlet velocity magnitude ``U``, shape ``(n_inlet,)`` (or a scalar).
    intensity : float
        The turbulence intensity ``I`` (a fraction, e.g. 0.05 for 5%).

    Returns
    -------
    jnp.ndarray
        The inlet ``k``, matching the shape of ``velocity_magnitude``.
    """
    return 1.5 * (velocity_magnitude * intensity) ** 2


def inlet_omega(k: jnp.ndarray, length_scale: float, model: SSTModel) -> jnp.ndarray:
    """Inlet omega from a turbulent length scale: ``omega = sqrt(k) / (beta_star**0.25 L)``.

    Parameters
    ----------
    k : jnp.ndarray
        The inlet turbulent kinetic energy (see :func:`inlet_k`), shape ``(n_inlet,)`` (or scalar).
    length_scale : float
        The turbulent length scale ``L`` (e.g. a fraction of a characteristic dimension).
    model : SSTModel
        The model constants (reads ``beta_star``).

    Returns
    -------
    jnp.ndarray
        The inlet ``omega``, matching the shape of ``k``.
    """
    return jnp.sqrt(k) / (model.beta_star**0.25 * length_scale)
