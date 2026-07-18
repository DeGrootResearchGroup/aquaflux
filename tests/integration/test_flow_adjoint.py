"""The coupled-flow IFT adjoint must match finite differences -- not merely be *finite*.

This is the regression the earlier differentiability tests missed: they only asserted that
``jax.grad`` returned a finite number, never that it was *correct*. With the Rhie--Chow momentum
diagonal ``a_P`` ``stop_gradient``-ed, the parameter adjoint was inconsistent with the residual
actually solved -- e.g. it reported a large spurious sensitivity of the Stokes velocity to
viscosity, which is provably zero. These tests differentiate a converged flow functional with
respect to viscosity and check against central finite differences of the same discrete solve.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import BlockPreconditioner, MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, SweptGradientSolve
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver
from aquaflux.solve.implicit import _INEXACT_FORWARD_SOLVER

from tests.support.meshes import perturbed_grid_2d

RHO = 1.0
_residual = lambda state, assembler: assembler.residual(state)  # noqa: E731


def _cavity(mu, n, advection):
    mesh = perturbed_grid_2d(n, n, perturb=0.15, named_boundaries=True)
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=16)),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=advection,
        pressure_pin=0,
    )


def _adjoint_and_fd(functional, mu0, *, advection, n, h):
    """(adjoint gradient, central finite difference) of ``functional`` w.r.t. viscosity.

    The block preconditioner is ``stop_gradient``-ed and only accelerates the Krylov solve, so it
    is built once from a reference state and reused across ``mu`` (building it inside the
    differentiated function would capture the ``mu`` tracer).
    """
    precond = BlockPreconditioner.build(_cavity(mu0, n, advection)).factory()
    solver = ImplicitNewtonSolver(
        max_steps=20, forward_step=DampedNewtonStep(preconditioner=precond)
    )

    def f(mu):
        assembler = _cavity(mu, n, advection)
        state = solver.solve(_residual, assembler.initial_state(), assembler)
        return functional(assembler, state)

    adjoint = float(jax.grad(f)(mu0))
    fd = float((f(mu0 + h) - f(mu0 - h)) / (2.0 * h))
    return adjoint, fd


def _mean_speed(assembler, state):
    velocity, _ = assembler.unpack(state)
    return jnp.mean(jnp.abs(velocity[:, 0]))


def _mean_abs_pressure(assembler, state):
    _, pressure = assembler.unpack(state)
    return jnp.mean(jnp.abs(pressure - jnp.mean(pressure)))


@pytest.mark.validation
def test_stokes_velocity_viscosity_adjoint_is_zero() -> None:
    """Stokes velocity is viscosity-independent, so d(mean|u_x|)/dmu = 0.

    The ``stop_gradient``-ed ``a_P`` used to report an O(0.1) spurious sensitivity here; the
    consistent adjoint matches the finite-difference zero to ~machine precision.
    """
    adjoint, fd = _adjoint_and_fd(_mean_speed, 0.02, advection=None, n=10, h=1e-6)
    assert abs(fd) < 1e-6  # the true (finite-difference) sensitivity is ~0
    assert abs(adjoint) < 1e-6  # the adjoint agrees (was O(0.1) before the a_P fix)


@pytest.mark.validation
def test_stokes_pressure_viscosity_adjoint_matches_fd() -> None:
    """Stokes pressure scales with viscosity -- a smooth, non-zero sensitivity the adjoint must
    reproduce to tight tolerance."""
    adjoint, fd = _adjoint_and_fd(_mean_abs_pressure, 0.02, advection=None, n=10, h=1e-6)
    assert abs(fd) > 1e-3  # a genuinely non-zero sensitivity
    assert abs(adjoint - fd) <= 1e-4 * abs(fd)


@pytest.mark.validation
def test_adjoint_preconditioner_is_a_drop_in() -> None:
    """The M^T-preconditioned adjoint returns the *same* gradient as the unpreconditioned adjoint.

    M^T (the transpose of the forward block preconditioner) only changes the adjoint Krylov path
    -- it makes the transpose solve mesh-independent -- so it must not perturb the gradient it
    computes."""
    n = 8

    def viscosity_grad(preconditioner):
        solver = ImplicitNewtonSolver(
            max_steps=20, forward_step=DampedNewtonStep(preconditioner=preconditioner)
        )

        def f(mu):
            assembler = _cavity(mu, n, FirstOrderUpwind())
            state = solver.solve(_residual, assembler.initial_state(), assembler)
            return _mean_speed(assembler, state)

        return float(jax.grad(f)(0.02))

    precond = BlockPreconditioner.build(_cavity(0.02, n, FirstOrderUpwind())).factory()
    g_preconditioned = viscosity_grad(precond)  # M forward + M^T adjoint
    g_unpreconditioned = viscosity_grad(None)  # unpreconditioned both
    assert abs(g_preconditioned - g_unpreconditioned) <= 1e-6 * abs(g_unpreconditioned)


@pytest.mark.validation
def test_inexact_newton_matches_tight_solve_with_fewer_matvecs() -> None:
    """Inexact Newton (the default loose forward linear solve) reaches the same converged state and
    the same viscosity gradient as a tight forward solve, while doing strictly fewer matvecs per
    Newton step. The forward tolerance is a convergence-rate knob, not an accuracy one: the outer
    loop drives the residual to the nonlinear tolerance regardless, and the adjoint stays tight.
    """
    n, mu0 = 12, 0.02
    precond = BlockPreconditioner.build(_cavity(mu0, n, FirstOrderUpwind())).factory()
    tight = lx.GMRES(rtol=1e-10, atol=1e-10)

    def converged_and_grad(forward_solver):
        solver = ImplicitNewtonSolver(
            max_steps=30,
            forward_step=DampedNewtonStep(preconditioner=precond),
            solver=forward_solver,
        )
        assembler = _cavity(mu0, n, FirstOrderUpwind())
        state = solver.solve(_residual, assembler.initial_state(), assembler)

        def f(mu):
            a = _cavity(mu, n, FirstOrderUpwind())
            return _mean_speed(a, solver.solve(_residual, a.initial_state(), a))

        return state, float(jax.grad(f)(mu0))

    state_inexact, grad_inexact = converged_and_grad(
        None
    )  # default: inexact forward, tight adjoint
    state_tight, grad_tight = converged_and_grad(tight)  # tight forward throughout
    # Same converged field (both driven to the nonlinear tolerance) and same gradient (tight adjoint).
    assert jnp.allclose(state_inexact, state_tight, atol=1e-7)
    assert abs(grad_inexact - grad_tight) <= 1e-6 * abs(grad_tight)

    # Strictly fewer matvecs: on the Jacobian at a representative mid-solve iterate, the default
    # inexact forward GMRES converges in fewer steps (hence matvecs) than the tight one.
    assembler = _cavity(mu0, n, FirstOrderUpwind())
    phi = ImplicitNewtonSolver(
        max_steps=2, forward_step=DampedNewtonStep(preconditioner=precond), solver=tight
    ).solve(_residual, assembler.initial_state(), assembler)
    apply_m = precond(phi)
    residual = assembler.residual(phi)
    operator = lx.FunctionLinearOperator(
        lambda v: apply_m(jax.jvp(assembler.residual, (phi,), (v,))[1]),
        jax.ShapeDtypeStruct(residual.shape, residual.dtype),
    )
    rhs = apply_m(-residual)
    steps_inexact = int(lx.linear_solve(operator, rhs, _INEXACT_FORWARD_SOLVER).stats["num_steps"])
    steps_tight = int(lx.linear_solve(operator, rhs, tight).stats["num_steps"])
    assert steps_inexact < steps_tight


@pytest.mark.validation
def test_navier_stokes_viscosity_adjoint_matches_fd() -> None:
    """The full nonlinear coupled adjoint (with convection) matches finite differences.

    First-order upwind is non-smooth at flow reversals, so the tolerance is looser than the
    smooth Stokes cases; it still pins the sign and magnitude that a broken adjoint would miss.
    """
    adjoint, fd = _adjoint_and_fd(_mean_speed, 0.02, advection=FirstOrderUpwind(), n=12, h=1e-4)
    assert abs(fd) > 1e-2  # a genuinely non-zero sensitivity
    assert abs(adjoint - fd) <= 2e-2 * abs(fd)
    assert np.sign(adjoint) == np.sign(fd)
