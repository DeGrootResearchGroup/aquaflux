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
    """Near-wall omega ``6 nu / (beta_1 d**2)``, the value fixed in the wall-adjacent cell.

    This is the analytical viscous-sublayer solution evaluated **at the cell centroid**, which is
    where it is imposed. As ``k -> 0`` at a smooth wall the omega equation collapses to a balance of
    viscous diffusion against destruction, ``nu d2(omega)/dy2 = beta_1 omega**2``; substituting
    ``omega = A / y**2`` gives ``6 nu A = beta_1 A**2``, hence ``A = 6 nu / beta_1``. So
    ``omega(y) = 6 nu / (beta_1 y**2)`` solves the sublayer equation exactly (Wilcox).

    The larger ``60 nu / (beta_1 dy**2)`` seen in the literature is a different quantity: a wall
    *face* boundary value, ten times the asymptote, standing in for the singularity at ``y = 0``
    where the analytical solution diverges (Menter, 1994). It is not the value the solution takes at
    a finite distance, so imposing it at a cell centroid overshoots the near-wall omega by 10x --
    which suppresses the near-wall eddy viscosity and stiffens the omega equation.

    Parameters
    ----------
    nu : jnp.ndarray
        Kinematic (molecular) viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells (centroid to wall), shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_1``).

    Returns
    -------
    jnp.ndarray
        The near-wall omega per wall-adjacent cell, shape ``(n_wall,)``.
    """
    return 6.0 * nu / (model.beta_1 * d**2)


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


def equilibrium_k(friction_velocity: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    """Turbulent kinetic energy of equilibrium wall turbulence: ``k = u_tau**2 / sqrt(beta_star)``.

    In the log layer production balances dissipation, the shear stress is the wall value
    ``-u'v' = u_tau**2``, and the eddy-viscosity closure ties that to ``k`` through
    ``-u'v' = sqrt(beta_star) k``. Rearranged, ``k = u_tau**2 / sqrt(beta_star)`` (~``3.3 u_tau**2``
    for the standard ``beta_star = 0.09``).

    This is the counterpart to :func:`inlet_k` for a flow with **no inlet to read a level from** — a
    streamwise-periodic channel driven by a body force, where the friction velocity instead follows
    from the global force balance (:func:`~aquaflux.flow.scales.friction_velocity`). Being an
    equilibrium relation it is a level for the core flow, not a wall value; ``k`` still goes to zero at
    the wall through its boundary condition.

    Parameters
    ----------
    friction_velocity : jnp.ndarray
        The wall friction velocity ``u_tau``, any shape (or a scalar).
    model : SSTModel
        The model constants (reads ``beta_star``).

    Returns
    -------
    jnp.ndarray
        The equilibrium ``k``, matching the shape of ``friction_velocity``.
    """
    return friction_velocity**2 / jnp.sqrt(model.beta_star)


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
