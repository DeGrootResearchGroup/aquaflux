"""Lid-driven cavity: the first genuinely nonlinear flow, validated against Ghia et al. (1982).

A unit square with three stationary no-slip walls and a lid (top) translating at ``u = U``. The
convective term ``u . grad u`` makes the coupled residual **nonlinear** (the mass flux and the
advected velocity both depend on the velocity), so Newton takes several steps and the
implicit-function-theorem adjoint differentiates the converged flow. It is a closed domain, so
the pressure is pinned at one cell.

Validated against the benchmark centreline velocities of Ghia, Ghia & Shin (1982) at ``Re = 100``:
``u`` along the vertical centreline and ``v`` along the horizontal centreline. On a uniform mesh
the near-lid layer is under-resolved (Ghia used a stretched 129x129 grid), so the tolerance is
looser near the lid; the primary-vortex extrema are matched closely.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import BlockPreconditioner, MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver, newton_step

RE, U_LID, RHO = 100.0, 1.0, 1.0
MU = RHO * U_LID / RE

# Ghia et al. (1982), Re=100: u on the vertical centreline, v on the horizontal centreline.
GHIA_Y = np.array([0.0625, 0.1016, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531])
GHIA_U = np.array([-0.0419, -0.0643, -0.1566, -0.2109, -0.2058, -0.1364, 0.0033, 0.2315, 0.6872])
GHIA_X = np.array([0.0625, 0.0938, 0.1563, 0.2266, 0.5, 0.8047, 0.8594, 0.9063, 0.9453])
GHIA_V = np.array([0.0923, 0.1232, 0.1608, 0.1751, 0.0545, -0.2453, -0.2245, -0.1691, -0.1031])


def _cavity(n, mu=MU, scheme=None):
    mesh = structured_grid_2d(n, n, lx=1.0, ly=1.0, named_boundaries=True)
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=scheme if scheme is not None else FirstOrderUpwind(),
        pressure_pin=0,
    )


def test_cavity_nonlinear_newton_converges() -> None:
    """Convection makes the residual nonlinear; Newton drives it to ~zero in a few steps."""
    assembler = _cavity(32)
    phi = assembler.initial_state()
    residual_norm = None
    for _ in range(8):
        phi = newton_step(assembler.residual, phi)
        residual_norm = float(jnp.linalg.norm(assembler.residual(phi)))
        if residual_norm < 1e-8:
            break
    assert residual_norm < 1e-8


def test_cavity_solve_is_differentiable() -> None:
    """Reverse-mode gradient through the nonlinear cavity solve **matches central finite
    differences** -- not merely finite. A broken (e.g. frozen-a_P) adjoint returns a finite but
    wrong value here; only an FD comparison catches it."""
    n = 20
    precond = BlockPreconditioner.build(_cavity(n)).factory()  # stop_gradient-ed; reuse across mu
    solver = ImplicitNewtonSolver(
        max_steps=30, forward_step=DampedNewtonStep(preconditioner=precond)
    )

    def mean_speed(mu):
        assembler = _cavity(n, mu=mu)
        state = solver.solve(lambda s, a: a.residual(s), assembler.initial_state(), assembler)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(MU))
    h = 1e-5  # MU = 0.01
    fd = float((mean_speed(MU + h) - mean_speed(MU - h)) / (2.0 * h))
    assert np.isfinite(grad)
    assert abs(fd) > 1e-2  # a genuinely non-zero sensitivity
    assert abs(grad - fd) <= 3e-2 * abs(fd)  # first-order upwind is non-smooth, so a loose tol


@pytest.mark.validation
def test_cavity_matches_ghia_centrelines() -> None:
    """Centreline velocities match the Ghia et al. (1982) Re=100 benchmark."""
    n = 48
    assembler = _cavity(n)
    phi = assembler.initial_state()
    for _ in range(10):
        phi = newton_step(assembler.residual, phi)
        if float(jnp.linalg.norm(assembler.residual(phi))) < 1e-9:
            break
    velocity, _ = assembler.unpack(phi)
    _, cell_geometry = assembler.mesh, assembler.geometry.cell
    xc = np.asarray(cell_geometry.centroid)[:, 0]
    yc = np.asarray(cell_geometry.centroid)[:, 1]
    u = np.asarray(velocity[:, 0])
    v = np.asarray(velocity[:, 1])

    # Extract the cell column/row nearest each centreline and interpolate to the Ghia stations.
    xcol = np.unique(xc)[np.argmin(np.abs(np.unique(xc) - 0.5))]
    col = np.abs(xc - xcol) < 1e-9
    order = np.argsort(yc[col])
    u_line = np.interp(GHIA_Y, yc[col][order], u[col][order])

    yrow = np.unique(yc)[np.argmin(np.abs(np.unique(yc) - 0.5))]
    row = np.abs(yc - yrow) < 1e-9
    order = np.argsort(xc[row])
    v_line = np.interp(GHIA_X, xc[row][order], v[row][order])

    # Primary-vortex extrema (robust benchmark features).
    assert abs(u_line.min() - (-0.211)) < 0.03
    assert abs(v_line.min() - (-0.245)) < 0.04
    assert abs(v_line.max() - 0.175) < 0.04
    # Full centreline agreement (looser near the under-resolved lid).
    assert np.max(np.abs(u_line - GHIA_U)) < 0.1
    assert np.max(np.abs(v_line - GHIA_V)) < 0.04
