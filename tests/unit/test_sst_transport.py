"""Tests for the k-omega SST transport-equation assembly.

The assembler is built on a small channel and each equation is solved with prescribed (frozen)
closure fields and no advection, checking that the equation is well-posed (the Newton residual
converges), the fields are finite, and the omega wall cells are fixed to the analytical value.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import VelocityFields
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import CorrectedGreenGauss
from aquaflux.solve import ImplicitNewtonSolver
from aquaflux.turbulence import (
    SSTClosureFields,
    SSTModel,
    SSTTurbulence,
    log_layer_shear_rate,
    omega_wall,
)

NU = 1e-3


def _turbulence():
    mesh = structured_grid_2d(6, 4, lx=3.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    turb = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CorrectedGreenGauss(),
        FirstOrderUpwind(),
        density=1.0,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.01),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(10.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return mesh, turb


def _closure(turb):
    n = turb.mesh.n_cells
    return SSTClosureFields(
        nu_t=jnp.full(n, 0.01),
        strain_rate=jnp.full(n, 1.0),
        f1=jnp.full(n, 0.5),
        grad_k=jnp.zeros((n, 2)),
        grad_omega=jnp.zeros((n, 2)),
        omega=jnp.full(n, 1.0),
        k=jnp.full(n, 1.0),
        wall_shear_rate=jnp.full(turb.wall_cells.shape, 1.0),
    )


def _velocity(mesh, gradient):
    """A kinematic bundle carrying a prescribed gradient and a quiescent (zero) velocity."""
    return VelocityFields(
        velocity=jnp.zeros((mesh.n_cells, mesh.dim)),
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),
        gradient=gradient,
    )


def test_build_identifies_the_wall_adjacent_cells() -> None:
    """The bottom and top rows each contribute their cells to the omega fixation set."""
    mesh, turb = _turbulence()
    assert turb.wall_cells.shape[0] == 2 * 6  # bottom row + top row of a 6x4 grid
    assert turb.wall_distance.shape == (mesh.n_cells,)


def test_k_equation_solves_to_a_finite_bounded_field() -> None:
    """The equation is well-posed and solvable; the residual converges and the field is bounded.

    Strict positivity is *not* guaranteed by the raw solve (AD-Newton has no discrete maximum
    principle) -- it is secured by the realizability floor the driver applies between sweeps -- so
    this checks only convergence, finiteness, and a sensible magnitude.
    """
    mesh, turb = _turbulence()
    residual = turb.k_residual(jnp.zeros(mesh.n_faces), _closure(turb))
    k = ImplicitNewtonSolver(max_steps=30).solve(
        lambda phi, _: residual(phi), jnp.full(mesh.n_cells, 0.01), None
    )
    assert float(jnp.linalg.norm(residual(k))) < 1e-8  # the equation is solvable
    assert not bool(jnp.any(jnp.isnan(k)))
    assert float(jnp.max(jnp.abs(k))) < 0.1  # bounded near the inlet magnitude, no blow-up


def test_omega_equation_fixes_the_wall_cells_to_the_adaptive_value() -> None:
    """The wall cells are fixed to the adaptive (blended) near-wall omega, reading the closure ``k``.

    With the frozen closure's ``k = 1`` the log branch dominates the viscous one here, so the fixed
    value is the blend :func:`~aquaflux.turbulence.omega_wall`, not the bare viscous fixation.
    """
    mesh, turb = _turbulence()
    closure = _closure(turb)
    residual = turb.omega_residual(jnp.zeros(mesh.n_faces), closure)
    omega = ImplicitNewtonSolver(max_steps=40).solve(
        lambda phi, _: residual(phi), jnp.full(mesh.n_cells, 10.0), None
    )
    assert float(jnp.linalg.norm(residual(omega))) < 1e-6
    expected = omega_wall(
        jnp.full(turb.wall_cells.shape[0], NU),
        turb.wall_distance[turb.wall_cells],
        closure.k[turb.wall_cells],
        SSTModel(),
    )
    assert jnp.allclose(omega[turb.wall_cells], expected)


def test_k_residual_is_differentiable_in_a_closure_field() -> None:
    """Gradient flows through the frozen eddy viscosity into the k residual, no NaNs."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k = jnp.full(n, 0.01)

    def loss(nu_t_scale):
        closure = _closure(turb)._replace(nu_t=nu_t_scale * jnp.full(n, 0.01))
        return jnp.sum(turb.k_residual(jnp.zeros(mesh.n_faces), closure)(k) ** 2)

    assert not bool(jnp.isnan(jax.grad(loss)(1.0)))


def _shear(n, gamma=2.0):
    """A uniform simple-shear velocity gradient (du_x/dy = gamma), so S = gamma."""
    return jnp.tile(jnp.array([[[0.0, gamma], [0.0, 0.0]]]), (n, 1, 1))


def test_eddy_viscosity_matches_the_model() -> None:
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k, omega = jnp.full(n, 0.01), jnp.full(n, 10.0)
    nu_t = turb.eddy_viscosity(_shear(n), k, omega)
    expected = SSTModel().eddy_viscosity(
        k, omega, jnp.full(n, 2.0), jnp.full(n, NU), turb.wall_distance
    )
    assert jnp.allclose(nu_t, expected)
    assert bool(jnp.all(nu_t > 0.0))


def test_closure_fields_are_well_formed() -> None:
    """The strain rate, blending function, gradients, and eddy viscosity are sensible."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k, omega = jnp.full(n, 0.01), jnp.full(n, 10.0)
    closure = turb.closure_fields(_velocity(mesh, _shear(n)), k, omega)
    interior = jnp.setdiff1d(jnp.arange(n), turb.wall_cells)
    assert jnp.allclose(
        closure.strain_rate[interior], 2.0
    )  # away from the wall, the reconstruction
    assert bool(jnp.all(closure.nu_t > 0.0))
    assert bool(jnp.all((closure.f1 >= 0.0) & (closure.f1 <= 1.0)))  # F1 = tanh(.) in [0, 1]
    assert closure.grad_k.shape == (n, mesh.dim)
    assert closure.grad_omega.shape == (n, mesh.dim)
    assert jnp.allclose(closure.omega, omega)


def test_eddy_viscosity_is_differentiable_in_k() -> None:
    mesh, turb = _turbulence()
    n = mesh.n_cells
    omega = jnp.full(n, 10.0)
    g = jax.grad(lambda k: jnp.sum(turb.eddy_viscosity(_shear(n), k, omega)))(jnp.full(n, 0.01))
    assert not bool(jnp.any(jnp.isnan(g)))


# --- the adaptive near-wall seams the closure fields carry --------------------------------------


def test_wall_shear_rate_is_the_wall_normal_velocity_difference_over_the_distance() -> None:
    """|U_P - U_wall| / d at the wall-adjacent cells, measured against the patch's own velocity."""
    mesh, turb = _turbulence()
    speed = 3.0
    velocity = jnp.zeros((mesh.n_cells, mesh.dim)).at[:, 0].set(speed)
    fields = VelocityFields(
        velocity=velocity,
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),  # stationary no-slip walls
        gradient=_shear(mesh.n_cells),
    )
    got = turb.wall_shear_rate(fields)
    assert got.shape == turb.wall_cells.shape
    assert jnp.allclose(got, speed / turb.wall_distance[turb.wall_cells])


def test_wall_shear_rate_is_relative_to_a_moving_wall() -> None:
    """A cell moving with its wall has no wall-normal shear -- the difference is the relative one."""
    mesh, turb = _turbulence()
    speed = 3.0
    velocity = jnp.zeros((mesh.n_cells, mesh.dim)).at[:, 0].set(speed)
    fields = VelocityFields(
        velocity=velocity,
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)).at[:, 0].set(speed),
        gradient=_shear(mesh.n_cells),
    )
    assert jnp.allclose(turb.wall_shear_rate(fields), 0.0)


def test_wall_shear_rate_has_a_finite_derivative_at_zero_velocity() -> None:
    """A quiescent field sits exactly on the cone point of the vector magnitude -- no NaN there."""
    mesh, turb = _turbulence()
    zeros = jnp.zeros((mesh.n_cells, mesh.dim))

    def total(velocity):
        fields = VelocityFields(
            velocity=velocity,
            boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),
            gradient=_shear(mesh.n_cells),
        )
        return jnp.sum(turb.wall_shear_rate(fields))

    assert bool(jnp.all(jnp.isfinite(total(zeros))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(zeros))))


def test_strain_rate_blends_the_wall_cells_onto_the_log_layer_shear() -> None:
    """Resolved (tiny k): the reconstruction, untouched. Log layer (large k): the log-law shear.

    Only the wall-adjacent cells move; the interior keeps the reconstructed strain either way.
    """
    mesh, turb = _turbulence()
    n = mesh.n_cells
    wall = turb.wall_cells
    interior = jnp.setdiff1d(jnp.arange(n), wall)
    d = turb.wall_distance[wall]

    resolved = turb.strain_rate(_shear(n), jnp.full(n, 1e-12))
    assert jnp.allclose(resolved, 2.0, rtol=1e-6)  # entirely inside the sublayer: no change

    k = jnp.full(n, 30.0)  # far into the log layer
    got = turb.strain_rate(_shear(n), k)
    assert jnp.allclose(got[interior], 2.0)
    assert jnp.allclose(got[wall], log_layer_shear_rate(d, k[wall], turb.model), rtol=1e-3)


def test_strain_rate_is_differentiable_in_k() -> None:
    """The blend is live in the coupled residual, so its k-derivative must be finite -- also at k = 0."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    for k in (jnp.zeros(n), jnp.full(n, 30.0)):
        g = jax.grad(lambda kk: jnp.sum(turb.strain_rate(_shear(n), kk)))(k)
        assert bool(jnp.all(jnp.isfinite(g)))
