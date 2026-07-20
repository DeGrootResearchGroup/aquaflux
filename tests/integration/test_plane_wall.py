"""Gate A / Gate B: transient diffusion in a plane wall with convection, and its sensitivity.

The Milestone-0 acceptance case. A non-dimensional plane wall of unit half-thickness, initially
``theta = 1``, cools through a convective surface (Biot number ``Bi``) with a symmetry plane at
the centreline. It has a closed-form solution for both the temperature field ``theta`` and its
sensitivity ``d theta / d Bi``, so it is an *analytical* oracle for the whole differentiable
substrate — assembly, the transient BDF march, the Newton/linear solve, and the reverse-mode
gradient through all of it.

Governing (non-dimensional): ``d theta / d Fo = d^2 theta / d x^2`` on ``x in [0, 1]``,
``theta(x, 0) = 1``, symmetry ``d theta / d x = 0`` at ``x = 0``, convection
``-d theta / d x = Bi theta`` at ``x = 1``.

Analytical series (Fourier number ``Fo``):

    theta(x, Fo) = sum_n  C_n exp(-zeta_n^2 Fo) cos(zeta_n x),
    C_n = 4 sin(zeta_n) / (2 zeta_n + sin(2 zeta_n)),   zeta_n tan(zeta_n) = Bi.

- **Gate A** matches ``theta`` at ``Fo = 1`` to discretization error, second-order in space.
- **Gate B** matches ``d theta / d Bi`` (reverse/forward-mode AD through the solver) against the
  analytical sensitivity, to the *same* order as the primary field — the project's core claim.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Convective, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler, TransientTerm
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.solve import newton_step

# --- analytical oracle -----------------------------------------------------------------


def _eigenvalues(bi: float, n: int = 12) -> np.ndarray:
    """First ``n`` positive roots of ``zeta tan(zeta) = bi`` (one per ``(k pi, k pi + pi/2)``)."""
    roots = []
    for k in range(n):
        lo, hi = k * np.pi + 1e-12, k * np.pi + np.pi / 2 - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid * np.tan(mid) - bi > 0.0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))
    return np.array(roots)


def analytical_theta(x: np.ndarray, fo: float, bi: float, n: int = 12) -> np.ndarray:
    """Series solution ``theta(x, Fo)`` at the points ``x``."""
    z = _eigenvalues(bi, n)
    cn = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    modes = cn[None, :] * np.exp(-(z[None, :] ** 2) * fo) * np.cos(z[None, :] * x[:, None])
    return np.sum(modes, axis=1)


# --- the differentiable solver under test ----------------------------------------------


def _wall_mesh(nx: int):
    """A 1-D strip (nx cells across the half-wall, one cell deep) with named boundary sides."""
    mesh = structured_grid_2d(nx, 1, lx=1.0, ly=1.0 / nx, named_boundaries=True)
    return mesh, mesh.geometry()


def solve_wall(bi, mesh, geometry, fo_final=1.0, n_steps=400):
    """March the plane wall to ``Fo = fo_final`` and return the cell field (differentiable in bi)."""
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions(
            {
                "left": ZeroGradient(),  # symmetry plane at x = 0
                "right": Convective(h=bi, t_inf=0.0),  # convective surface at x = 1
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        transient=TransientTerm(),
    )
    dt = fo_final / n_steps
    # The per-step problem is linear, so one Newton correction is exact. newton_step is plain
    # traced operations, which is what lets the whole march be differentiated in forward mode
    # (jacfwd below) as well as reverse.
    solve = eqx.filter_jit(newton_step)
    phi0 = jnp.ones(mesh.n_cells)
    phi1 = solve(lambda p: assembler.residual(p, phi_old=phi0, dt=dt, first_step=True), phi0)

    def step(carry, _):
        old, older = carry
        new = solve(
            lambda p: assembler.residual(p, phi_old=old, phi_older=older, dt=dt, first_step=False),
            old,
        )
        return (new, old), None

    (phi_final, _), _ = jax.lax.scan(step, (phi1, phi0), None, length=n_steps - 1)
    return phi_final


def sensitivity_field(bi, mesh, geometry, n_steps=400):
    """Forward-mode AD sensitivity ``d theta / d Bi`` per cell at ``Fo = 1``."""
    return np.asarray(jax.jacfwd(lambda b: solve_wall(b, mesh, geometry, n_steps=n_steps))(bi))


# --- Gate A ----------------------------------------------------------------------------


def test_gate_a_primary_field_matches_analytical() -> None:
    """theta at Fo = 1 matches the analytical series to discretization error."""
    bi = 1.0
    mesh, geom = _wall_mesh(40)
    theta = np.asarray(solve_wall(bi, mesh, geom, n_steps=400))
    exact = analytical_theta(np.asarray(geom.cell.centroid)[:, 0], 1.0, bi)
    assert np.max(np.abs(theta - exact)) < 1e-3


def test_gate_a_one_newton_step_converges() -> None:
    """A single Newton step drives the per-step residual to ~machine zero (the problem is linear)."""
    bi = 1.0
    mesh, geom = _wall_mesh(20)
    assembler = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions(
            {
                "left": ZeroGradient(),
                "right": Convective(h=bi, t_inf=0.0),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        transient=TransientTerm(),
    )
    phi0 = jnp.ones(mesh.n_cells)

    def residual_fn(p):
        return assembler.residual(p, phi_old=phi0, dt=0.01, first_step=True)

    phi = eqx.filter_jit(newton_step)(residual_fn, phi0)
    assert float(jnp.linalg.norm(residual_fn(phi))) < 1e-10


# --- Gate B ----------------------------------------------------------------------------


def test_gate_b_sensitivity_matches_analytical() -> None:
    """d theta / d Bi through the solver (forward-mode AD) matches the analytical sensitivity."""
    bi = 1.0
    mesh, geom = _wall_mesh(40)
    x = np.asarray(geom.cell.centroid)[:, 0]

    sensitivity = sensitivity_field(bi, mesh, geom, n_steps=400)
    assert not np.any(np.isnan(sensitivity))

    step = 1e-5
    analytical = (analytical_theta(x, 1.0, bi + step) - analytical_theta(x, 1.0, bi - step)) / (
        2.0 * step
    )
    assert np.max(np.abs(sensitivity - analytical)) < 1e-3


def test_gate_b_gradient_flows_without_nans() -> None:
    """A scalar objective through the whole transient solve differentiates cleanly (jax.grad)."""
    mesh, geom = _wall_mesh(20)
    grad = jax.grad(lambda b: jnp.mean(solve_wall(b, mesh, geom, n_steps=100)))(1.0)
    assert np.isfinite(float(grad))
    assert float(grad) < 0.0  # a larger Biot number cools the wall faster -> lower mean theta


# --- order of accuracy (slower; the reproduction of the 2019 convergence study) ---------


@pytest.mark.validation
def test_gate_a_second_order_spatial_convergence() -> None:
    """Grid refinement gives second-order spatial convergence of theta at Fo = 1."""
    bi = 1.0
    errors = []
    for nx in (10, 20, 40):
        mesh, geom = _wall_mesh(nx)
        theta = np.asarray(solve_wall(bi, mesh, geom, n_steps=2000))
        exact = analytical_theta(np.asarray(geom.cell.centroid)[:, 0], 1.0, bi)
        errors.append(np.max(np.abs(theta - exact)))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.9


@pytest.mark.validation
def test_gate_b_sensitivity_second_order_convergence() -> None:
    """The sensitivity field converges at the same (second) order as the primary field."""
    bi = 1.0
    step = 1e-5
    errors = []
    for nx in (10, 20, 40):
        mesh, geom = _wall_mesh(nx)
        x = np.asarray(geom.cell.centroid)[:, 0]
        sens = sensitivity_field(bi, mesh, geom, n_steps=2000)
        analytical = (analytical_theta(x, 1.0, bi + step) - analytical_theta(x, 1.0, bi - step)) / (
            2.0 * step
        )
        errors.append(np.max(np.abs(sens - analytical)))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.8
