"""Gate C: AD-linearized non-orthogonal correction vs deferred correction on a skewed mesh.

The concrete demonstration of the project's linearization thesis. On a deliberately skewed grid
the diffusion operator carries a non-orthogonal correction that depends on the cell gradients.
Folding the gradient reconstruction (:class:`CorrectedGreenGauss`) into the residual and letting
automatic differentiation form the Jacobian puts that correction *in the matrix*, so Newton
converges the (linear) problem in **one step** and reproduces a linear field to machine precision
on a skewed mesh. The classical alternative — **deferred correction** — instead lags the
correction (computes the gradient from the previous iterate and holds it fixed in the linear
solve), which converges only linearly over several outer iterations.

The test case is the harmonic linear field ``phi* = 2x - 3y + 1`` on a 25%-skewed grid with the
exact field imposed weakly on every boundary (a :class:`DirichletField`). Because the boundary
values are then exact and gradient-independent, this isolates the *interior* non-orthogonal
correction — the object of the claim — from the separate boundary-gradient coupling.

The lagged (deferred-correction) behaviour is emulated by wrapping the gradient reconstruction in
``stop_gradient``: the residual value still uses the true gradient, but the Jacobian omits the
correction's dependence on the field — exactly the unlinearized deferred correction.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, DirichletField
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, GradientScheme
from aquaflux.solve import newton_step

from tests.support.meshes import perturbed_grid_2d


def _linear_field(x: jnp.ndarray) -> jnp.ndarray:
    """A harmonic linear field: exact discrete solution of the Laplace problem on any mesh."""
    return 2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0


class _LaggedGradient(GradientScheme):
    """Wrap a scheme so its gradient is used for the residual value but not its Jacobian.

    ``stop_gradient`` blocks the gradient's contribution to the linearization, reproducing the
    lagged / unlinearized non-orthogonal correction (deferred correction).
    """

    inner: GradientScheme

    def gradients(self, field, mesh, geometry, boundary_values):
        return jax.lax.stop_gradient(self.inner.gradients(field, mesh, geometry, boundary_values))


def _skewed_laplace(gradient_scheme, n=12, perturb=0.25, seed=1):
    """Assembler for steady Laplace on a skewed grid with the linear field imposed on all sides."""
    mesh = perturbed_grid_2d(n, n, perturb=perturb, seed=seed, named_boundaries=True)
    geom = mesh.geometry()
    bc = DirichletField(field_fn=_linear_field)
    assembler = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({"left": bc, "right": bc, "bottom": bc, "top": bc}),
        gradient_scheme=gradient_scheme,
    )
    return mesh, geom.cell, assembler


def test_gate_c_one_newton_step_on_skewed_mesh() -> None:
    """With the gradient folded into the residual, one Newton step converges to ~machine zero."""
    mesh, _, assembler = _skewed_laplace(CorrectedGreenGauss())
    phi0 = jnp.zeros(mesh.n_cells)
    before = float(jnp.linalg.norm(assembler.residual(phi0)))
    phi = eqx.filter_jit(newton_step)(assembler.residual, phi0)
    after = float(jnp.linalg.norm(assembler.residual(phi)))
    assert before > 1.0  # a genuinely non-trivial residual to begin with
    assert after < 1e-8  # driven to ~zero in a single step


def test_gate_c_linear_exact_on_skewed_mesh() -> None:
    """The consistently-linearized operator reproduces a linear field exactly on a skewed grid."""
    mesh, cell_geometry, assembler = _skewed_laplace(CorrectedGreenGauss())
    phi = eqx.filter_jit(newton_step)(assembler.residual, jnp.zeros(mesh.n_cells))
    exact = _linear_field(cell_geometry.centroid)
    assert float(jnp.max(jnp.abs(phi - exact))) < 1e-9


def test_gate_c_lagged_correction_needs_several_iterations() -> None:
    """The lagged (deferred-correction) scheme converges only linearly — the 'before' to our 'after'.

    One implicit step reaches ~machine zero; the lagged scheme still has a sizable residual after
    one step and needs several deferred-correction sweeps to catch up.
    """
    mesh, _, implicit = _skewed_laplace(CorrectedGreenGauss())
    _, _, lagged = _skewed_laplace(_LaggedGradient(inner=CorrectedGreenGauss()))
    phi0 = jnp.zeros(mesh.n_cells)

    implicit_after_one = float(
        jnp.linalg.norm(implicit.residual(eqx.filter_jit(newton_step)(implicit.residual, phi0)))
    )
    lagged_after_one = float(
        jnp.linalg.norm(lagged.residual(eqx.filter_jit(newton_step)(lagged.residual, phi0)))
    )
    assert implicit_after_one < 1e-8
    assert lagged_after_one > 1e-2  # far from converged after a single step

    # Deferred correction: sweep the lagged gradient until it finally converges.
    phi = phi0
    steps = 0
    for _ in range(20):
        phi = eqx.filter_jit(newton_step)(lagged.residual, phi)
        steps += 1
        if float(jnp.linalg.norm(lagged.residual(phi))) < 1e-8:
            break
    assert steps >= 4  # many sweeps, versus the single implicit step


def test_gate_c_residual_is_differentiable_on_skewed_mesh() -> None:
    """A solve on the skewed mesh differentiates cleanly through the nested gradient solve."""
    mesh, _, assembler = _skewed_laplace(CorrectedGreenGauss(), n=8)

    def objective(scale):
        bc = DirichletField(field_fn=lambda x: scale * _linear_field(x))
        scaled = ResidualAssembler.build(
            mesh,
            assembler.geometry,
            PropertyModel({"diffusivity": Constant(1.0)}),
            (DiffusionFlux(),),
            BoundaryConditions({"left": bc, "right": bc, "bottom": bc, "top": bc}),
            gradient_scheme=CorrectedGreenGauss(),
        )
        phi = eqx.filter_jit(newton_step)(scaled.residual, jnp.zeros(mesh.n_cells))
        return jnp.sum(phi**2)

    grad = jax.grad(objective)(1.0)
    assert np.isfinite(float(grad))
    assert float(grad) > 0.0  # scaling the boundary data up increases the interior magnitude
