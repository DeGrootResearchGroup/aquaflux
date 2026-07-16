"""High-Reynolds open channel: the pseudo-transient continuation that lifts the Re ~ 100 floor.

The basic open channel (:mod:`tests.integration.test_channel`) solves at Re ~ 100 with a
line-searched Newton step, but stagnates as the Reynolds number rises: once convection dominates,
the block-SIMPLE preconditioner (built on the viscous operator) no longer approximates the Jacobian
and the inner GMRES stalls a fixed fraction of the way down, which the line search cannot recover.

:class:`~aquaflux.flow.PseudoTransientContinuation` fixes this by damping each Newton step with an
``a_P``-proportional diagonal shift that ramps to zero on the residual — restoring the
preconditioner's diagonal dominance while the shift is present, and recovering the exact steady
Newton step (and its converged state) as it vanishes. These tests drive the same channel setup to
genuinely convective Reynolds numbers, on both uniform and wall-graded meshes, and confirm the
converged solve stays differentiable.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    PseudoTransientContinuation,
    VelocityInlet,
)
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import ImplicitNewtonSolver

H, L, U_IN, RHO = 1.0, 4.0, 1.0, 1.0


def _channel(nx, ny, mu, *, wall_growth=1.0):
    """The plane channel of :mod:`tests.integration.test_channel`, at viscosity ``mu``.

    ``wall_growth > 1`` grades the wall-normal spacing (finest at both walls) so the near-wall cell
    resolves a boundary layer — the mesh a turbulent-channel validation needs.
    """
    y_nodes = graded_nodes(ny, H, wall_growth) if wall_growth != 1.0 else None
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )


def _reynolds(mu):
    return RHO * U_IN * H / mu


def _solve(assembler, *, continuation=None, max_steps=120, **kwargs):
    if continuation is None:
        continuation = PseudoTransientContinuation.build(assembler)
    solver = ImplicitNewtonSolver(max_steps=max_steps, continuation=continuation, **kwargs)
    return solver.solve(lambda s, a: a.residual(s), assembler.initial_state(), assembler)


@pytest.mark.parametrize(
    ("nx", "ny", "mu"),
    [(24, 16, 5e-3), (32, 24, 2e-3), (40, 32, 1e-3)],  # Re = 200, 500, 1000
)
def test_continuation_converges_at_high_reynolds(nx, ny, mu) -> None:
    """The continuation drives the convective channel to a converged flow well past the Re~100 floor."""
    assembler = _channel(nx, ny, mu)
    state = _solve(assembler)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8


def test_continuation_is_necessary_beyond_the_laminar_floor() -> None:
    """At Re = 200 the line-searched Newton solve stagnates; the continuation recovers convergence.

    This is the higher-Reynolds analogue of ``test_line_search_is_necessary``: there the line search
    was the missing globalization at Re ~ 100, here it is no longer enough and the pseudo-transient
    continuation is what makes the solve converge.
    """
    assembler = _channel(24, 16, 5e-3)  # Re = 200

    from aquaflux.flow import BlockPreconditioner

    line_searched = ImplicitNewtonSolver(
        max_steps=40, preconditioner=BlockPreconditioner.build(assembler).factory()
    )
    try:
        undamped = line_searched.solve(
            lambda s, a: a.residual(s), assembler.initial_state(), assembler
        )
        residual = float(jnp.linalg.norm(assembler.residual(undamped)))
        line_search_failed = (not np.isfinite(residual)) or residual > 1e-6
    except Exception:
        line_search_failed = True  # the inner GMRES stagnated on the convective operator
    assert line_search_failed

    converged = _solve(assembler)  # continuation on
    assert float(jnp.linalg.norm(assembler.residual(converged))) < 1e-8


def test_continuation_conserves_mass() -> None:
    """The high-Re converged flow conserves mass: the inlet delivers exactly what the outlet passes."""
    assembler = _channel(32, 24, 2e-3)  # Re = 500
    state = _solve(assembler)
    mdot = np.asarray(assembler.mass_flux(state))
    patches = assembler.mesh.face_patches
    inlet = mdot[np.asarray(patches.indices("left"))].sum()
    outlet = mdot[np.asarray(patches.indices("right"))].sum()
    assert inlet < 0.0 < outlet
    assert abs(inlet + outlet) < 1e-6  # global continuity
    assert abs(abs(inlet) - RHO * U_IN * H) < 1e-8  # inlet delivers rho U H exactly


def test_wall_graded_channel_resolves_and_converges() -> None:
    """A wall-graded mesh (fine near the walls) converges at Re = 1000 — the wall-resolved setup a
    turbulent boundary layer needs, which the scale-invariant ``a_P``-proportional shift handles."""
    assembler = _channel(40, 32, 1e-3, wall_growth=1.2)  # Re = 1000, graded
    state = _solve(assembler)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8

    # The near-wall cell is much finer than the core cell — the point of the grading.
    volume = np.asarray(assembler.geometry.cell.volume)
    wall_cell = volume.min()
    core_cell = volume.max()
    assert wall_cell < 0.5 * core_cell


def test_continuation_solve_is_differentiable() -> None:
    """Reverse-mode gradient through the continuation solve is finite and matches finite differences.

    The continuation (like the preconditioner) is built once outside ``jax.grad`` with concrete
    parameters and reused across ``mu``; its diagonal shift vanishes at convergence, so the IFT
    adjoint linearises the same steady residual it would without it.
    """
    continuation = PseudoTransientContinuation.build(_channel(24, 16, 5e-3))

    def mean_speed(mu):
        assembler = _channel(24, 16, mu)
        state = _solve(assembler, continuation=continuation)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(5e-3))
    assert np.isfinite(grad)

    step = 5e-5
    finite_difference = (float(mean_speed(5e-3 + step)) - float(mean_speed(5e-3 - step))) / (
        2 * step
    )
    assert abs(grad - finite_difference) < 1e-3 * abs(finite_difference)
