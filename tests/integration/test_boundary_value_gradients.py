"""Gradients of a converged solve with respect to the boundary values it was given.

The implicit solve differentiates the parameters it carries -- the assembler -- through the
implicit-function-theorem adjoint, which sees only floating array leaves. A boundary value that is
not one does not yield a wrong finite gradient: it raises, or it contributes nothing. So every check
here compares against a central finite difference and asserts a non-zero result, rather than merely
asserting the gradient is finite.

The problem is a small Stokes channel (no convection, so one Newton step converges) solved with a
direct linear solve, which keeps the forward and adjoint solves exact.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, DirichletField, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import RootSolver, assembler_residual

MESH = structured_grid_2d(8, 6, lx=3.0, ly=1.0, named_boundaries=True)
GEOMETRY = MESH.geometry()
DIRECT = lx.AutoLinearSolver(well_posed=True)
SOLVER = RootSolver(max_steps=4, linear_solver=DIRECT, adjoint_solver=DIRECT)
STEP = 1e-6


class _Parabola(eqx.Module):
    """A parabolic inlet profile on ``0 <= y <= 1`` whose peak speed is a field."""

    peak: jnp.ndarray

    def __call__(self, x):
        y = x[:, 1]
        u = self.peak * 4.0 * y * (1.0 - y)
        return jnp.stack([u, jnp.zeros_like(u)], axis=1)


class _LinearInY(eqx.Module):
    a: jnp.ndarray

    def __call__(self, x):
        return self.a * x[:, 1]


def _channel(inlet, *, top=None, outlet_pressure=0.0):
    return MomentumContinuity.build(
        MESH,
        GEOMETRY,
        PropertyModel(
            {"viscosity": Constant(jnp.asarray(0.1)), "density": Constant(jnp.asarray(1.0))}
        ),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": inlet,
                "right": PressureOutlet(pressure=outlet_pressure),
                "bottom": NoSlipWall(),
                "top": NoSlipWall() if top is None else top,
            }
        ),
    )


def _flow(assembler):
    return SOLVER.solve(assembler_residual, assembler.initial_state(), assembler)


def _mean_streamwise_speed(assembler):
    return jnp.mean(assembler.unpack(_flow(assembler))[0][:, 0])


def _mean_pressure(assembler):
    return jnp.mean(assembler.unpack(_flow(assembler))[1])


def _assert_matches_finite_difference(loss, x0):
    gradient = float(jax.grad(loss)(x0))
    finite_difference = float((loss(x0 + STEP) - loss(x0 - STEP)) / (2.0 * STEP))
    assert gradient != 0.0
    assert gradient == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_gradient_with_respect_to_a_constant_inlet_speed() -> None:
    _assert_matches_finite_difference(
        lambda u: _mean_streamwise_speed(_channel(VelocityInlet(velocity=jnp.stack([u, 0.0])))), 1.0
    )


def test_gradient_with_respect_to_an_inlet_profile_coefficient() -> None:
    _assert_matches_finite_difference(
        lambda peak: _mean_streamwise_speed(_channel(VelocityInlet(velocity=_Parabola(peak)))), 1.0
    )


def test_gradient_with_respect_to_a_moving_wall_speed() -> None:
    inlet = VelocityInlet(velocity=(1.0, 0.0))
    _assert_matches_finite_difference(
        lambda w: _mean_streamwise_speed(
            _channel(inlet, top=MovingWall(velocity=jnp.stack([w, 0.0])))
        ),
        0.5,
    )


def test_filter_grad_reaches_an_outlet_pressure_written_as_a_float_literal() -> None:
    """The trap a float-annotated field sets: a caller passing ``0.5`` rather than an array."""
    assembler = _channel(VelocityInlet(velocity=(1.0, 0.0)), outlet_pressure=0.5)
    gradient = eqx.filter_grad(_mean_pressure)(assembler)
    closure_gradient = float(gradient.boundary.conditions["right"].pressure)

    def shifted(p):
        return float(
            _mean_pressure(_channel(VelocityInlet(velocity=(1.0, 0.0)), outlet_pressure=p))
        )

    finite_difference = (shifted(0.5 + STEP) - shifted(0.5 - STEP)) / (2.0 * STEP)
    assert closure_gradient != 0.0
    assert closure_gradient == pytest.approx(finite_difference, rel=1e-6)


def test_gradient_with_respect_to_a_dirichlet_field_coefficient() -> None:
    properties = PropertyModel({"diffusivity": Constant(jnp.asarray(1.0))})

    def mean_value(a):
        assembler = ResidualAssembler.build(
            MESH,
            GEOMETRY,
            properties,
            (DiffusionFlux(),),
            BoundaryConditions(
                {
                    "left": DirichletField(field_fn=_LinearInY(a)),
                    "right": Dirichlet(0.0),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
        )
        phi = SOLVER.solve(assembler_residual, jnp.zeros(MESH.n_cells), assembler)
        return jnp.mean(phi)

    _assert_matches_finite_difference(mean_value, 2.0)


def test_a_plain_function_profile_still_solves_to_the_same_field() -> None:
    """Function profiles are held static rather than as leaves, so the forward solve is unchanged."""

    def uniform(x):
        return jnp.stack([jnp.ones(x.shape[0]), jnp.zeros(x.shape[0])], axis=1)

    from_function = _flow(_channel(VelocityInlet(velocity=uniform)))
    from_constant = _flow(_channel(VelocityInlet(velocity=(1.0, 0.0))))
    assert jnp.allclose(from_function, from_constant, atol=1e-12)


def test_changing_a_boundary_value_does_not_recompile() -> None:
    """As leaves, a new inlet speed or outlet pressure is a new value, not a new compilation key."""
    traces = []

    @eqx.filter_jit
    def residual(assembler, state):
        traces.append(None)
        return assembler.residual(state)

    state = _channel(VelocityInlet(velocity=(1.0, 0.0))).initial_state()
    for speed, pressure in ((1.0, 0.0), (2.0, 0.5), (3.0, -1.0)):
        residual(_channel(VelocityInlet(velocity=(speed, 0.0)), outlet_pressure=pressure), state)
    assert len(traces) == 1
