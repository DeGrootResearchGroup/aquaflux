"""The k-omega SST turbulence model: its constants and the quantities derived directly from them.

Menter's shear-stress-transport (SST) model blends a k-omega treatment near the wall with a
k-epsilon treatment in the free stream through the function ``F1``, and limits the eddy viscosity in
adverse-pressure-gradient boundary layers through ``F2``. This module owns the model constants and
the closed-form quantities they define -- the blending functions, the constant blend, and the eddy
viscosity. The k and omega transport *source* terms are separate operators that consume this model.

Every constant is a model parameter carried on the object, so each is a differentiable leaf and a
sensitivity / calibration target. The subscript ``1`` denotes the inner (near-wall, k-omega) set and
``2`` the outer (free-stream, k-epsilon) set; a blended constant is ``phi = F1 phi_1 + (1-F1) phi_2``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot

# A positive floor on the cross-diffusion term CD, guarding the ``1 / CD`` inside ``F1``'s argument
# where the two gradients are near-orthogonal. Part of the SST definition of ``F1``.
_CROSS_DIFFUSION_FLOOR = 1e-10


class SSTModel(eqx.Module):
    """The k-omega SST constants and the blending / eddy-viscosity quantities they define.

    Attributes
    ----------
    a1 : float
        The eddy-viscosity limiter constant in ``nu_t = a1 k / max(a1 omega, S F2)``.
    beta_star : float
        The k-destruction constant (shared by both sets).
    alpha_1, beta_1, sigma_k1, sigma_omega1 : float
        The inner (near-wall, k-omega) constant set.
    alpha_2, beta_2, sigma_k2, sigma_omega2 : float
        The outer (free-stream, k-epsilon) constant set.
    kappa : float
        The von Karman constant, used by the log-layer near-wall relations (the adaptive wall
        treatment's log branch).
    e_wall : float
        The log-law wall roughness constant ``E`` in ``u+ = ln(E y+) / kappa`` (smooth wall), used
        by the adaptive momentum wall function.
    """

    a1: float = 0.31
    beta_star: float = 0.09
    alpha_1: float = 5.0 / 9.0
    beta_1: float = 3.0 / 40.0
    sigma_k1: float = 0.85
    sigma_omega1: float = 0.5
    alpha_2: float = 0.44
    beta_2: float = 0.0828
    sigma_k2: float = 1.0
    sigma_omega2: float = 0.856
    kappa: float = 0.41
    e_wall: float = 9.8

    @property
    def wall_y_star_lam(self) -> jnp.ndarray:
        """The laminar/log intersection ``y*_lam`` of the near-wall coordinate.

        The wall coordinate at which the viscous ``u+ = y+`` and log ``u+ = ln(E y+)/kappa``
        profiles cross, i.e. the root of ``y = ln(E y) / kappa``. Below it the wall function is in
        the viscous sublayer (zero extra eddy viscosity); above it, the log law. It depends only on
        :attr:`kappa` and :attr:`e_wall`, so a short fixed-point iteration (``y <- ln(E y)/kappa``,
        contractive here) converges it once.

        Kept in ``jnp`` (not a Python ``float``) so it stays valid when the model constants are
        traced — the SST constants are differentiable leaves, so a sensitivity path carries ``kappa``
        and ``e_wall`` as tracers.

        Returns
        -------
        jnp.ndarray
            The laminar/log intersection ``y*_lam`` (~11 for the standard constants), a scalar.
        """
        y = jnp.asarray(11.0)
        for _ in range(10):
            y = jnp.log(self.e_wall * y) / self.kappa
        return y

    def blend(self, f1: jnp.ndarray, inner: float, outer: float) -> jnp.ndarray:
        """Blend an inner/outer constant pair by ``F1``: ``f1 * inner + (1 - f1) * outer``.

        Parameters
        ----------
        f1 : jnp.ndarray
            The blending function per cell, shape ``(n_cells,)``.
        inner, outer : float
            The inner (k-omega) and outer (k-epsilon) constant of a blended pair.

        Returns
        -------
        jnp.ndarray
            The blended constant per cell, shape ``(n_cells,)``.
        """
        return f1 * inner + (1.0 - f1) * outer

    def f2(
        self, k: jnp.ndarray, omega: jnp.ndarray, nu: jnp.ndarray, d: jnp.ndarray
    ) -> jnp.ndarray:
        """The eddy-viscosity blending function ``F2 = tanh(arg2**2)``, shape ``(n_cells,)``.

        ``arg2 = max(2 sqrt(k) / (beta_star omega d), 500 nu / (d**2 omega))``.

        Parameters
        ----------
        k, omega : jnp.ndarray
            Turbulent kinetic energy and specific dissipation rate per cell, shape ``(n_cells,)``.
        nu : jnp.ndarray
            Kinematic (molecular) viscosity per cell, shape ``(n_cells,)``.
        d : jnp.ndarray
            Wall distance per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            ``F2`` per cell, shape ``(n_cells,)``.
        """
        arg2 = jnp.maximum(
            2.0 * jnp.sqrt(k) / (self.beta_star * omega * d),
            500.0 * nu / (d**2 * omega),
        )
        return jnp.tanh(arg2**2)

    def f1(
        self,
        k: jnp.ndarray,
        omega: jnp.ndarray,
        nu: jnp.ndarray,
        d: jnp.ndarray,
        grad_k: jnp.ndarray,
        grad_omega: jnp.ndarray,
    ) -> jnp.ndarray:
        """The model-blending function ``F1 = tanh(arg1**4)``, shape ``(n_cells,)``.

        ``arg1 = min(max(sqrt(k) / (beta_star omega d), 500 nu / (d**2 omega)),
        4 sigma_omega2 k / (CD d**2))`` with the cross-diffusion
        ``CD = max(2 sigma_omega2 (grad_k . grad_omega) / omega, 1e-10)``.

        Parameters
        ----------
        k, omega : jnp.ndarray
            Turbulent kinetic energy and specific dissipation rate per cell, shape ``(n_cells,)``.
        nu : jnp.ndarray
            Kinematic (molecular) viscosity per cell, shape ``(n_cells,)``.
        d : jnp.ndarray
            Wall distance per cell, shape ``(n_cells,)``.
        grad_k, grad_omega : jnp.ndarray
            Cell gradients of ``k`` and ``omega``, shape ``(n_cells, dim)``.

        Returns
        -------
        jnp.ndarray
            ``F1`` per cell, shape ``(n_cells,)`` (``-> 1`` near the wall, ``-> 0`` in the free
            stream).
        """
        cross_diffusion = jnp.maximum(
            2.0 * self.sigma_omega2 * dot(grad_k, grad_omega) / omega, _CROSS_DIFFUSION_FLOOR
        )
        arg1 = jnp.minimum(
            jnp.maximum(
                jnp.sqrt(k) / (self.beta_star * omega * d),
                500.0 * nu / (d**2 * omega),
            ),
            4.0 * self.sigma_omega2 * k / (cross_diffusion * d**2),
        )
        return jnp.tanh(arg1**4)

    def eddy_viscosity(
        self,
        k: jnp.ndarray,
        omega: jnp.ndarray,
        strain_rate: jnp.ndarray,
        nu: jnp.ndarray,
        d: jnp.ndarray,
    ) -> jnp.ndarray:
        """Kinematic eddy viscosity ``nu_t = a1 k / max(a1 omega, S F2)``, shape ``(n_cells,)``.

        The ``S F2`` branch is the shear-stress limiter that caps ``nu_t`` in adverse-pressure-
        gradient boundary layers; ``F2`` is evaluated from ``(k, omega, nu, d)``.

        Parameters
        ----------
        k, omega : jnp.ndarray
            Turbulent kinetic energy and specific dissipation rate per cell, shape ``(n_cells,)``.
        strain_rate : jnp.ndarray
            Strain-rate magnitude ``S`` per cell, shape ``(n_cells,)`` (see
            :func:`~aquaflux.turbulence.strain.strain_rate_magnitude`).
        nu : jnp.ndarray
            Kinematic (molecular) viscosity per cell, shape ``(n_cells,)``.
        d : jnp.ndarray
            Wall distance per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The kinematic eddy viscosity ``nu_t`` per cell, shape ``(n_cells,)``.
        """
        return self.a1 * k / jnp.maximum(self.a1 * omega, strain_rate * self.f2(k, omega, nu, d))
