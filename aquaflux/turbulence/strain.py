"""Strain-rate magnitude of a velocity field — the scalar invariant turbulence models consume.

The mean strain-rate tensor is the symmetric part of the velocity gradient,
``S_ij = 1/2 (du_i/dx_j + du_j/dx_i)``, and its magnitude ``S = sqrt(2 S_ij S_ij)`` drives the
turbulence production and the eddy-viscosity limiter. It is a pure function of the velocity-gradient
tensor (reconstructed one component at a time by a gradient scheme), independent of any turbulence
model.

``S`` is the Euclidean norm of the (symmetric) gradient, so as a function of the gradient tensor it
has a **cone point at zero** — exactly like ``|x| = sqrt(x^2)`` at the origin. The value is
continuous there, but the ``sqrt`` chain rule differentiates to ``dS = dq / (2 S)`` with
``q = 2 S_ij S_ij``, which is ``0/0 = NaN`` at ``S = 0``. A perfectly uniform velocity region has
``S = 0`` identically (zero gradient), so that NaN reaches the whole automatic-differentiation
Jacobian — a body-force-driven periodic channel starts from a uniform plug and hits it in every
interior cell. The magnitude is therefore taken through a **guarded ``sqrt``** (:func:`safe_sqrt`)
whose argument is clamped away from zero *only on the branch that is discarded there*, so the
derivative at exactly ``q = 0`` is the finite minimum-norm subgradient ``dS = 0`` while the value and
the derivative are bit-identical to a plain ``sqrt`` wherever ``q > 0``.

Returning ``0`` at ``S = 0`` is not merely NaN-avoidance — it is the correct local derivative for
every downstream consumer: production reads ``S^2`` (derivative ``2 S dS -> 0``), and the
eddy-viscosity limiter reads ``max(a1 omega, S f2)``, whose ``max`` selects the strain-independent
``a1 omega`` branch there. And ``S = 0`` never occurs at a converged (sheared) turbulent field, so
the exact adjoint at the fixed point is untouched.
"""

from __future__ import annotations

import jax.numpy as jnp


def safe_sqrt(argument: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(argument)`` with a finite (zero) derivative at ``argument = 0`` instead of ``inf``.

    ``d/dx sqrt(x) = 1 / (2 sqrt(x)) -> inf`` at ``x = 0``, so automatic differentiation of a plain
    ``sqrt`` at a zero argument yields a NaN that contaminates the whole Jacobian. The double
    ``where`` clamps the argument to ``1`` on the branch selected at ``x = 0`` (so the ``sqrt`` never
    differentiates at zero) while returning ``0`` for the value there; the primal and its derivative
    are unchanged wherever ``x > 0``.
    """
    positive = argument > 0.0
    clamped = jnp.where(positive, argument, jnp.ones_like(argument))
    return jnp.where(positive, jnp.sqrt(clamped), jnp.zeros_like(argument))


def strain_rate_magnitude(velocity_gradient: jnp.ndarray) -> jnp.ndarray:
    """Strain-rate magnitude ``S = sqrt(2 S_ij S_ij)`` per cell, shape ``(n_cells,)``.

    Differentiable at ``S = 0`` (via :func:`safe_sqrt`), where the plain ``sqrt`` chain rule gives
    ``0/0 = NaN``; the value and derivative are bit-identical to a plain ``sqrt`` wherever ``S > 0``
    (see the module docstring).

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
    return safe_sqrt(2.0 * jnp.sum(strain * strain, axis=(-2, -1)))
