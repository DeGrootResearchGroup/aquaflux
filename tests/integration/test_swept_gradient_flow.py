"""The swept (fixed matrix-free Richardson) corrected gradient inside the coupled flow Newton.

On a non-orthogonal mesh the momentum non-orthogonal correction and the Rhie--Chow pressure
gradient are reconstructed by :class:`CorrectedGreenGauss` with a :class:`SweptGradientSolve`
strategy, whose constant operator ``A_g`` is inverted by a fixed number of matrix-free sweeps
rather than a nested iterative solve. This is the
scalable realization of absorbing the gradient into the flow system: it must (a) drive the coupled
residual to zero (the sweeps are accurate enough that Newton still converges), (b) reproduce the
solution of the exact iterative corrected gradient (a drop-in), and (c) remain differentiable
through the nonlinear solve.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, SweptGradientSolve
from aquaflux.solve import ImplicitNewtonSolver, newton_step

from tests.support.meshes import perturbed_grid_2d

RHO, MU, U_LID = 1.0, 0.02, 1.0


def _cavity(scheme, n=12, perturb=0.15, mu=MU):
    mesh = perturbed_grid_2d(n, n, lx=1.0, ly=1.0, perturb=perturb, named_boundaries=True)
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        scheme,
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )


def test_swept_gradient_cavity_converges() -> None:
    """Newton drives the skewed coupled residual to ~zero with the fixed-sweep gradient."""
    assembler = _cavity(CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=16)))
    phi = assembler.initial_state()
    residual_norm = None
    for _ in range(10):
        phi = newton_step(assembler.residual, phi)
        residual_norm = float(jnp.linalg.norm(assembler.residual(phi)))
        if residual_norm < 1e-8:
            break
    assert residual_norm < 1e-8


def test_swept_gradient_matches_iterative_solution() -> None:
    """The fixed-sweep gradient is a drop-in: it converges to the same flow field as the exact
    iteratively-solved corrected gradient."""
    swept = _cavity(CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=16)))
    iterative = _cavity(CorrectedGreenGauss())
    phi_s, phi_i = swept.initial_state(), iterative.initial_state()
    for _ in range(10):
        phi_s = newton_step(swept.residual, phi_s)
        phi_i = newton_step(iterative.residual, phi_i)
    assert jnp.allclose(phi_s, phi_i, atol=1e-7)


def test_swept_gradient_flow_is_differentiable() -> None:
    """Reverse-mode gradient through the nonlinear skewed cavity solve (IFT) is finite."""

    def mean_speed(mu):
        assembler = _cavity(CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=16)), mu=mu)
        state = ImplicitNewtonSolver(max_steps=30).solve(
            lambda s, a: a.residual(s), assembler.initial_state(), assembler
        )
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(MU))
    assert np.isfinite(grad)
