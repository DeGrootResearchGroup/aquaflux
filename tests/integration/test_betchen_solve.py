"""HessianCorrectedGradient inside the differentiable diffusion solve.

The second-order, Hessian-Schur-eliminated gradient scheme was validated physics-free (against
analytic gradients) in ``tests/unit/test_gradient.py``. Here it is exercised *end to end inside
the Newton solve*: the outer Newton's Jacobian-vector product differentiates through Betchen's
nested Schur solve (itself an inner ``A_HH`` solve) — a triply-nested linear solve — which must
converge in one step and stay differentiable on a skewed mesh.

The physics finding this pins down: in *pure diffusion* the cell gradient enters only through the
small non-orthogonal correction, so the solved **field** order is set by the operator's own
truncation (about second order) and is essentially the same with Betchen or ``CorrectedGreenGauss``.
Betchen's second-order advantage instead shows in the **reconstructed gradient / diffusive flux**
of the solved field, where ``CorrectedGreenGauss`` caps near first order on skewed grids — the
Green–Gauss accuracy ceiling. (This is why the gradient scheme matters far more for advection / Rhie-Chow,
where the gradient enters a face value at leading order.)

Test problem: the harmonic field ``phi* = exp(x) cos(y)`` (Laplace, no source) with the exact
value imposed on all boundaries (a :class:`DirichletField`), on a skewed grid.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, DirichletField
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.materials import Constant, MaterialModel
from aquaflux.schemes import CorrectedGreenGauss, HessianCorrectedGradient
from aquaflux.solve import NewtonSolver

from tests.support.meshes import perturbed_grid_2d


def _phi_exact(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(x[..., 0]) * jnp.cos(x[..., 1])


def _grad_exact(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [jnp.exp(x[..., 0]) * jnp.cos(x[..., 1]), -jnp.exp(x[..., 0]) * jnp.sin(x[..., 1])],
        axis=1,
    )


def _laplace(n, gradient_scheme, perturb=0.2, seed=1):
    mesh = perturbed_grid_2d(n, n, perturb=perturb, seed=seed, named_boundaries=True)
    geom = mesh.geometry()
    bc = DirichletField(field_fn=_phi_exact)
    assembler = ResidualAssembler.build(
        mesh,
        geom,
        MaterialModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({"left": bc, "right": bc, "bottom": bc, "top": bc}),
        gradient_scheme=gradient_scheme,
    )
    return mesh, geom.cell, assembler


def _interior_mask(mesh) -> np.ndarray:
    boundary_cells = set(
        np.asarray(mesh.face_cells.owner)[np.asarray(mesh.face_cells.neighbour) < 0].tolist()
    )
    return np.array([c not in boundary_cells for c in range(mesh.n_cells)])


def test_betchen_one_newton_step_on_skewed_mesh() -> None:
    """The nested Schur solve inside the residual still converges in a single Newton step."""
    mesh, _, assembler = _laplace(12, HessianCorrectedGradient())
    phi0 = jnp.zeros(mesh.n_cells)
    before = float(jnp.linalg.norm(assembler.residual(phi0)))
    phi = NewtonSolver(iterations=1).solve(assembler.residual, phi0)
    after = float(jnp.linalg.norm(assembler.residual(phi)))
    assert before > 1.0
    assert after < 1e-8


def test_betchen_solve_is_differentiable() -> None:
    """jax.grad flows through the triply-nested linear solve without NaNs."""
    mesh, _, _ = _laplace(8, HessianCorrectedGradient())
    geom = mesh.geometry()

    def objective(scale):
        bc = DirichletField(field_fn=lambda x: scale * _phi_exact(x))
        assembler = ResidualAssembler.build(
            mesh,
            geom,
            MaterialModel({"diffusivity": Constant(1.0)}),
            (DiffusionFlux(),),
            BoundaryConditions({"left": bc, "right": bc, "bottom": bc, "top": bc}),
            gradient_scheme=HessianCorrectedGradient(),
        )
        return jnp.sum(
            NewtonSolver(iterations=1).solve(assembler.residual, jnp.zeros(mesh.n_cells)) ** 2
        )

    grad = jax.grad(objective)(1.0)
    assert np.isfinite(float(grad))
    assert float(grad) > 0.0


@pytest.mark.validation
def test_betchen_field_converges_second_order_on_skewed() -> None:
    """The solved field is a valid ~2nd-order solution (operator-floor-limited)."""
    errors = []
    for n in (8, 16, 32):
        mesh, cell_geometry, assembler = _laplace(n, HessianCorrectedGradient())
        phi = NewtonSolver(iterations=1).solve(assembler.residual, jnp.zeros(mesh.n_cells))
        err = phi - _phi_exact(cell_geometry.centroid)
        errors.append(float(jnp.sqrt(jnp.mean(err[_interior_mask(mesh)] ** 2))))
    order = np.log2(errors[0] / errors[-1]) / np.log2(32 / 8)
    assert order > 1.7


@pytest.mark.validation
def test_betchen_flux_beats_green_gauss_on_skewed() -> None:
    """The reconstructed gradient (diffusive flux) of the solved field is markedly more accurate
    with Betchen than with corrected Green-Gauss, and converges at a higher order — the concrete
    benefit of the 2nd-order scheme in the solve."""

    def gradient_errors(scheme):
        errs = []
        for n in (8, 16, 32):
            mesh, cell_geometry, assembler = _laplace(n, scheme)
            phi = NewtonSolver(iterations=1).solve(assembler.residual, jnp.zeros(mesh.n_cells))
            g_err = assembler.gradient(phi) - _grad_exact(cell_geometry.centroid)
            per_cell = jnp.sqrt(jnp.sum(g_err**2, axis=1))
            errs.append(float(jnp.sqrt(jnp.mean(per_cell[_interior_mask(mesh)] ** 2))))
        return errs

    betchen = gradient_errors(HessianCorrectedGradient())
    green_gauss = gradient_errors(CorrectedGreenGauss())

    # Markedly more accurate at the finest resolution ...
    assert betchen[-1] < 0.6 * green_gauss[-1]
    # ... and a higher full-range order (Green-Gauss caps near first order on skewed).
    betchen_order = np.log2(betchen[0] / betchen[-1]) / np.log2(32 / 8)
    green_gauss_order = np.log2(green_gauss[0] / green_gauss[-1]) / np.log2(32 / 8)
    assert betchen_order > green_gauss_order + 0.3
    assert green_gauss_order < 1.2  # the Green–Gauss accuracy ceiling on skewed grids
