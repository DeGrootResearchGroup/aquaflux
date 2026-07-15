"""Flux-limited second-order advection in the solve, and the limiter's AD linearization.

Extends the first-order upwind case to a limited linear-upwind reconstruction with the
Venkatakrishnan limiter. The limiter makes the residual **nonlinear** (``psi`` depends on the
field through stencil min/max and a rational function), so the solve needs multiple Newton
iterations and the implicit-function-theorem adjoint (:class:`ImplicitNewtonSolver`) to
differentiate the converged state.

The headline experiment: the conventional **deferred-correction** approach lags the limiter
(freezes ``psi`` from the previous iterate and adds the limited term as an explicit source),
which converges only linearly. Writing ``psi(phi)`` into the residual and letting AD linearize
it puts the limiter in the Jacobian and recovers **quadratic** Newton convergence — the "after"
to the lagged "before". The lagged behaviour is emulated with ``stop_gradient`` on the limiter.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    FirstOrderUpwind,
    LimitedUpwind,
    ResidualAssembler,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, Limiter, VenkatakrishnanLimiter
from aquaflux.solve import ImplicitNewtonSolver, newton_step

from tests.support.fields import face_mass_flux

U = 1.0


class _LaggedLimiter(Limiter):
    """Freeze the limiter in the Jacobian (reference-style deferred correction)."""

    inner: Limiter

    def limit(self, field, gradient, face_cells, geometry):
        return jax.lax.stop_gradient(self.inner.limit(field, gradient, face_cells, geometry))


def _exact(x: np.ndarray, pe: float) -> np.ndarray:
    return (np.exp(pe * x) - 1.0) / (np.exp(pe) - 1.0)


def _assembler(nx, gamma, scheme):
    mesh = structured_grid_2d(nx, 1, lx=1.0, ly=1.0 / nx, named_boundaries=True)
    geom = mesh.geometry()
    mdot = face_mass_flux(geom.face, jnp.array([U, 0.0]))
    assembler = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(gamma)}),
        (AdvectionFlux(mass_flux=mdot, scheme=scheme), DiffusionFlux()),
        BoundaryConditions(
            {
                "left": Dirichlet(0.0),
                "right": Dirichlet(1.0),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        gradient_scheme=CorrectedGreenGauss(),
    )
    return mesh, geom.cell, assembler


def _solve(nx, gamma, scheme):
    mesh, cell_geometry, assembler = _assembler(nx, gamma, scheme)
    phi = ImplicitNewtonSolver().solve(
        lambda p, a: a.residual(p), jnp.zeros(mesh.n_cells), assembler
    )
    return cell_geometry, assembler, phi


def test_limited_beats_first_order_on_smooth_profile() -> None:
    """On a resolved smooth profile the limited scheme is far more accurate than first order."""
    gamma = 0.2
    pe = U / gamma
    cell_geometry, _, phi_limited = _solve(
        80, gamma, LimitedUpwind(limiter=VenkatakrishnanLimiter(k=5.0))
    )
    _, _, phi_first = _solve(80, gamma, FirstOrderUpwind())
    x = np.asarray(cell_geometry.centroid)[:, 0]
    err_limited = np.sqrt(np.mean((np.asarray(phi_limited) - _exact(x, pe)) ** 2))
    err_first = np.sqrt(np.mean((np.asarray(phi_first) - _exact(x, pe)) ** 2))
    assert err_limited < 0.2 * err_first


def test_ad_linearized_limiter_converges_faster_than_lagged() -> None:
    """AD-linearizing the limiter recovers quadratic Newton convergence; the lagged (frozen)
    limiter converges only linearly and needs more steps."""

    def steps_to_converge(limiter):
        mesh, _, assembler = _assembler(40, 0.05, LimitedUpwind(limiter=limiter))
        phi = jnp.zeros(mesh.n_cells)
        for step in range(1, 13):
            phi = newton_step(assembler.residual, phi)
            if float(jnp.linalg.norm(assembler.residual(phi))) < 1e-9:
                return step
        return 99

    ad_steps = steps_to_converge(VenkatakrishnanLimiter(k=1.0))
    lagged_steps = steps_to_converge(_LaggedLimiter(inner=VenkatakrishnanLimiter(k=1.0)))
    assert ad_steps < lagged_steps


def test_limited_solve_differentiable_via_ift() -> None:
    """Reverse-mode gradient through the nonlinear limited solve (IFT) matches finite difference."""

    def objective(gamma):
        mesh, _, assembler = _assembler(
            40, gamma, LimitedUpwind(limiter=VenkatakrishnanLimiter(k=1.0))
        )
        phi = ImplicitNewtonSolver().solve(
            lambda p, a: a.residual(p), jnp.zeros(mesh.n_cells), assembler
        )
        return jnp.mean(phi)

    grad = float(jax.grad(objective)(0.05))
    fd = float((objective(0.05 + 1e-6) - objective(0.05 - 1e-6)) / 2e-6)
    assert np.isfinite(grad)
    assert abs(grad - fd) < 1e-4


@pytest.mark.validation
def test_limited_scheme_is_second_order() -> None:
    """The limited scheme converges at second order on a smooth profile (vs first for upwind)."""
    gamma = 0.2
    pe = U / gamma
    errors = []
    for nx in (20, 40, 80):
        cell_geometry, _, phi = _solve(
            nx, gamma, LimitedUpwind(limiter=VenkatakrishnanLimiter(k=5.0))
        )
        x = np.asarray(cell_geometry.centroid)[:, 0]
        errors.append(float(np.sqrt(np.mean((np.asarray(phi) - _exact(x, pe)) ** 2))))
    order = np.log2(errors[0] / errors[-1]) / np.log2(80 / 20)
    assert order > 1.8


@pytest.mark.validation
def test_limiter_reduces_overshoot_on_advected_step() -> None:
    """Advecting a top-hat: the limiter substantially reduces the over/undershoot of the
    unlimited second-order scheme (Venkatakrishnan is smooth, so it damps rather than strictly
    eliminates oscillation)."""

    def advect_tophat(limiter, nx=80, n_steps=40):
        from aquaflux.discretization import TransientTerm
        from aquaflux.solve import NewtonSolver

        mesh = structured_grid_2d(nx, 1, lx=1.0, ly=1.0 / nx, named_boundaries=True)
        geom = mesh.geometry()
        x = geom.cell.centroid[:, 0]
        phi0 = jnp.where((x > 0.2) & (x < 0.4), 1.0, 0.0)
        mdot = face_mass_flux(geom.face, jnp.array([U, 0.0]))
        assembler = ResidualAssembler.build(
            mesh,
            geom,
            PropertyModel({"diffusivity": Constant(1e-4)}),
            (
                AdvectionFlux(mass_flux=mdot, scheme=LimitedUpwind(limiter=limiter)),
                DiffusionFlux(),
            ),
            BoundaryConditions(
                {
                    "left": Dirichlet(0.0),
                    "right": ZeroGradient(),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
            gradient_scheme=CorrectedGreenGauss(),
            transient=TransientTerm(),
        )
        dt = 0.4 / n_steps
        solver = NewtonSolver(iterations=6)
        phi1 = solver.solve(
            lambda p: assembler.residual(p, phi_old=phi0, dt=dt, first_step=True), phi0
        )

        def step(carry, _):
            old, older = carry
            new = solver.solve(
                lambda p: assembler.residual(
                    p, phi_old=old, phi_older=older, dt=dt, first_step=False
                ),
                old,
            )
            return (new, old), None

        (phi, _), _ = jax.lax.scan(step, (phi1, phi0), None, length=n_steps - 1)
        return phi

    unlimited = advect_tophat(None)
    limited = advect_tophat(VenkatakrishnanLimiter(k=0.3))
    unlimited_overshoot = max(float(jnp.max(unlimited)) - 1.0, -float(jnp.min(unlimited)))
    limited_overshoot = max(float(jnp.max(limited)) - 1.0, -float(jnp.min(limited)))
    assert unlimited_overshoot > 0.05  # unlimited genuinely overshoots
    assert limited_overshoot < 0.7 * unlimited_overshoot  # limiter substantially damps it
