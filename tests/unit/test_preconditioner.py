"""Unit tests for the SIMPLE-preconditioner numerics: the pressure Schur Laplacian and the fixed
damped-Jacobi inner solve. Both are tested in isolation from the Newton driver."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    damped_jacobi_solve,
    pressure_schur_laplacian,
)
from aquaflux.materials import Constant, MaterialModel
from aquaflux.schemes import CompactGreenGauss

from tests.support.meshes import perturbed_grid_2d


def _geometry(n, perturb=0.0):
    """A small closed-cavity assembler, only for its geometry (interp_factor, normal_distance)."""
    mesh = perturbed_grid_2d(n, n, perturb=perturb, named_boundaries=True)
    geom = mesh.geometry()
    walls = {side: NoSlipWall() for side in ("top", "bottom", "left", "right")}
    return MomentumContinuity.build(
        mesh,
        geom,
        MaterialModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions(walls),
    )


def test_schur_laplacian_is_conservative_and_spd() -> None:
    """It is an M-matrix Laplacian: constant pressure -> zero, positive diagonal, symmetric, PSD."""
    asm = _geometry(8, perturb=0.15)
    a_p = 2.0 + jnp.arange(asm.mesh.n_cells, dtype=float)  # arbitrary positive, non-uniform
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
    )
    n = asm.mesh.n_cells
    assert jnp.allclose(matvec(jnp.ones(n)), 0.0, atol=1e-12)  # constant in the null space
    assert bool(jnp.all(diagonal > 0.0))
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal(n))
    q = jnp.asarray(rng.standard_normal(n))
    assert float(jnp.abs(jnp.dot(p, matvec(q)) - jnp.dot(q, matvec(p)))) < 1e-10  # symmetric
    assert float(jnp.dot(p, matvec(p))) > 0.0  # positive semi-definite (definite off the constant)


def test_schur_laplacian_pin_row_is_identity() -> None:
    """A pinned cell's row is the identity: its diagonal is 1 and its matvec returns its own value."""
    asm = _geometry(6)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    assert float(diagonal[0]) == 1.0
    p = jnp.asarray(np.random.default_rng(1).standard_normal(asm.mesh.n_cells))
    assert float(matvec(p)[0]) == float(p[0])


def test_damped_jacobi_is_linear_in_rhs() -> None:
    """A fixed sweep count makes rhs -> x a linear operator (required for a plain-GMRES left PC)."""
    asm = _geometry(6)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    rng = np.random.default_rng(2)
    r1 = jnp.asarray(rng.standard_normal(asm.mesh.n_cells))
    r2 = jnp.asarray(rng.standard_normal(asm.mesh.n_cells))

    def solve(r):
        return damped_jacobi_solve(matvec, diagonal, r, sweeps=12, omega=0.7, pressure_pin=0)

    lhs = solve(2.5 * r1 - 1.5 * r2)
    rhs = 2.5 * solve(r1) - 1.5 * solve(r2)
    assert jnp.allclose(lhs, rhs, atol=1e-12)


def test_damped_jacobi_converges_toward_solution() -> None:
    """More sweeps drive the residual of the pinned Laplacian system down (it is a valid solver)."""
    asm = _geometry(8)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    x_true = jnp.asarray(np.random.default_rng(3).standard_normal(asm.mesh.n_cells))
    rhs = matvec(x_true)  # consistent RHS (pin row carries x_true[0])

    def residual_norm(sweeps):
        x = damped_jacobi_solve(matvec, diagonal, rhs, sweeps=sweeps, omega=0.7, pressure_pin=0)
        return float(jnp.linalg.norm(matvec(x) - rhs))

    assert residual_norm(40) < 0.3 * residual_norm(5)  # clearly decreasing with sweeps
