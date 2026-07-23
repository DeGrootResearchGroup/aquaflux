"""Unit tests for the diffusion flux operator and the residual scatter engine.

The operator is tested in isolation on a hand-built two-cell context (no reader, no solve), then
its consistency is checked by recovering the Laplacian of an analytic field to second order on
an orthogonal grid. The scatter engine is tested with a stub flux operator so the gather ->
scatter plumbing is verified independently of any physics.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, ZeroGradient
from aquaflux.discretization import DiffusionFlux, FaceContext, FaceFluxOperator, ResidualAssembler
from aquaflux.mesh import (
    CellGeometry,
    FaceCellConnectivity,
    FaceGeometry,
    MeshGeometry,
    structured_grid_2d,
)
from aquaflux.properties import Constant, PropertyModel


def _single_face(
    phi_owner,
    phi_neighbour,
    *,
    grad=(0.0, 0.0),
    grad_neighbour=None,
    boundary_value=0.0,
    interior=True,
    face_centroid=(1.0, 0.0),
):
    """A one-face ``(field, FaceContext)`` with unit-spacing orthogonal geometry.

    Owner cell centroid ``(0.5, 0)``, neighbour ``(1.5, 0)``, face centroid ``(1, 0)``,
    ``n = x``, area ``0.5``. On a boundary face the neighbour is the sentinel ``-1`` (only the
    owner cell exists).
    """
    g_o = list(grad)
    g_n = g_o if grad_neighbour is None else list(grad_neighbour)
    n_cells = 2 if interior else 1
    neighbour = 1 if interior else -1
    field = jnp.array([phi_owner, phi_neighbour][:n_cells])
    gradient = jnp.array([g_o, g_n][:n_cells])
    geometry = MeshGeometry(
        face=FaceGeometry(
            area=jnp.array([0.5]),
            centroid=jnp.array([list(face_centroid)]),
            normal=jnp.array([[1.0, 0.0]]),
        ),
        cell=CellGeometry(
            volume=jnp.ones(n_cells),
            centroid=jnp.array([[0.5, 0.0], [1.5, 0.0]][:n_cells]),
        ),
    )
    context = FaceContext(
        face_cells=FaceCellConnectivity(jnp.array([0]), jnp.array([neighbour]), n_cells=n_cells),
        geometry=geometry,
        boundary_values=jnp.array([boundary_value]),
        gradient=gradient,
        properties={"diffusivity": jnp.ones(n_cells)},
    )
    return field, context


def test_interior_flux_orthogonal_matches_finite_difference() -> None:
    """Owner-outward flux of phi is -Gamma (phi_N - phi_P)/(d.n) * A (down-gradient); here
    d.n = 1, A = 0.5, so flux = -0.5 (phi_N - phi_P)."""
    field, context = _single_face(1.0, 3.0)
    flux = DiffusionFlux().face_flux(field, context)
    assert abs(float(flux[0]) + 0.5 * (3.0 - 1.0)) < 1e-13


def test_boundary_flux_one_sided() -> None:
    """Boundary face uses the weak value: -Gamma (phi_ip - phi_P)/(d.n) * A, d.n = 0.5, A = 0.5."""
    field, context = _single_face(1.0, 0.0, boundary_value=2.0, interior=False)
    flux = DiffusionFlux().face_flux(field, context)
    assert abs(float(flux[0]) + 1.0 * (2.0 - 1.0) / 0.5 * 0.5) < 1e-13


def test_boundary_coefficient_overrides_the_boundary_face_gamma() -> None:
    """On a boundary face the per-face coefficient replaces the owner-cell gamma (linear scaling)."""
    field, context = _single_face(1.0, 0.0, boundary_value=2.0, interior=False)
    default = DiffusionFlux().face_flux(field, context)  # gamma = 1
    overridden = DiffusionFlux(boundary_coefficient=jnp.array([3.0])).face_flux(field, context)
    assert jnp.allclose(overridden, 3.0 * default)


def test_boundary_coefficient_leaves_interior_faces_unchanged() -> None:
    """An interior face ignores the per-face boundary coefficient (only boundary faces are overridden)."""
    field, context = _single_face(1.0, 3.0)  # interior
    default = DiffusionFlux().face_flux(field, context)
    overridden = DiffusionFlux(boundary_coefficient=jnp.array([99.0])).face_flux(field, context)
    assert jnp.allclose(overridden, default)


def test_boundary_coefficient_is_differentiable() -> None:
    """Gradients flow through the boundary coefficient (state-dependent in a wall-function residual)."""

    def loss(gamma_b):
        field, context = _single_face(1.0, 0.0, boundary_value=2.0, interior=False)
        return jnp.sum(DiffusionFlux(boundary_coefficient=gamma_b).face_flux(field, context) ** 2)

    g = jax.grad(loss)(jnp.array([3.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_non_orthogonal_correction_enters_flux() -> None:
    """A face-tangential gradient contributes through the correction terms."""
    # Face centroid offset off the P-N line; distinct owner/neighbour gradients so the correction
    # (the difference corr_N - corr_P) does not cancel as it would for equal gradients.
    without = DiffusionFlux().face_flux(
        *_single_face(1.0, 3.0, grad=(0.0, 0.0), face_centroid=(1.0, 0.2))
    )
    withgrad = DiffusionFlux().face_flux(
        *_single_face(
            1.0, 3.0, grad=(0.0, 5.0), grad_neighbour=(0.0, 0.0), face_centroid=(1.0, 0.2)
        )
    )
    assert abs(float(without[0]) - float(withgrad[0])) > 1e-6


def test_flux_is_differentiable() -> None:
    """jax.grad flows through the operator without NaNs."""

    def loss(phi_n):
        field, context = _single_face(1.0, phi_n)
        return jnp.sum(DiffusionFlux().face_flux(field, context) ** 2)

    g = jax.grad(loss)(3.0)
    assert not bool(jnp.isnan(g))


class _StubFlux(FaceFluxOperator):
    """Returns a fixed flux on interior faces only — to probe the scatter engine."""

    value: float

    def face_flux(self, field, context):
        return jnp.where(context.face_cells.interior, self.value, 0.0)


def test_scatter_is_conservative_and_signed() -> None:
    """An interior face flux adds to the owner and subtracts from the neighbour (sum zero)."""
    mesh = structured_grid_2d(2, 1)
    geom = mesh.geometry()
    asm = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (_StubFlux(value=5.0),),
        BoundaryConditions({}),
    )
    residual = asm.residual(jnp.zeros(mesh.n_cells))  # steady: residual = net outward flux
    # The one interior face has owner cell 0, neighbour cell 1; +5 leaves the owner, enters nb.
    assert abs(float(residual[0]) - 5.0) < 1e-13  # owner: +5 out
    assert abs(float(residual[1]) + 5.0) < 1e-13  # neighbour: -5 (5 in)
    assert abs(float(jnp.sum(residual))) < 1e-12  # interior fluxes cancel globally


def test_operator_recovers_laplacian_second_order() -> None:
    """On an orthogonal grid, transport/volume approximates the Laplacian at 2nd order.

    For phi = cos(pi x) cos(pi y), the Laplacian is -2 pi^2 phi. The diffusion transport
    (with unit Gamma and zero-gradient on all boundaries) integrates that over each cell, so
    transport/V -> Laplacian; the interior error must fall like h^2.
    """

    def interior_laplacian_error(n):
        mesh = structured_grid_2d(n, n)
        geom = mesh.geometry()
        xc = geom.cell.centroid[:, 0]
        yc = geom.cell.centroid[:, 1]
        phi = jnp.cos(jnp.pi * xc) * jnp.cos(jnp.pi * yc)
        asm = ResidualAssembler.build(
            mesh,
            geom,
            PropertyModel({"diffusivity": Constant(1.0)}),
            (DiffusionFlux(),),
            BoundaryConditions({"boundary": ZeroGradient()}),
        )
        # Steady residual = net outward diffusive flux = -(diffusion into the cell).
        diffusion_in = -asm.residual(phi)
        laplacian_numeric = diffusion_in / geom.cell.volume
        laplacian_exact = -2.0 * jnp.pi**2 * phi
        boundary_cells = set(
            np.asarray(mesh.face_cells.owner)[np.asarray(mesh.face_cells.neighbour) < 0].tolist()
        )
        interior = np.array([c not in boundary_cells for c in range(mesh.n_cells)])
        err = laplacian_numeric - laplacian_exact
        return float(jnp.sqrt(jnp.mean(err[interior] ** 2)))

    e_coarse = interior_laplacian_error(16)
    e_fine = interior_laplacian_error(32)
    order = np.log2(e_coarse / e_fine)
    assert order > 1.8


def _skewed_diffusion_assembler():
    """A skewed-grid diffusion assembler with a gradient scheme (so the correction is nonzero)."""
    from aquaflux.schemes import CompactGreenGauss

    from tests.support.meshes import perturbed_grid_2d

    mesh = perturbed_grid_2d(8, 8, perturb=0.2, named_boundaries=True)
    geom = mesh.geometry()
    sides = ("left", "right", "bottom", "top")
    asm = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({s: ZeroGradient() for s in sides}),
        gradient_scheme=CompactGreenGauss(),
    )
    xc, yc = geom.cell.centroid[:, 0], geom.cell.centroid[:, 1]
    phi = jnp.sin(1.7 * xc) * jnp.cos(
        1.3 * yc
    )  # non-constant, so the reconstructed gradient is nonzero
    return asm, phi


def test_gradient_hook_is_identity_when_omitted() -> None:
    """Passing the identity as the gradient hook reproduces the default residual exactly."""
    asm, phi = _skewed_diffusion_assembler()
    assert jnp.allclose(asm.residual(phi), asm.residual(phi, gradient_hook=lambda g: g), atol=1e-14)


def test_gradient_hook_transforms_the_reconstructed_gradient() -> None:
    """The hook feeds the flux a modified gradient — scaling it changes the residual.

    The diffusion flux's non-orthogonal correction is what reads the cell gradient, so a hook that
    scales the gradient must change the residual on a skewed grid (where the correction is nonzero).
    """
    asm, phi = _skewed_diffusion_assembler()
    default = asm.residual(phi)
    scaled = asm.residual(phi, gradient_hook=lambda g: 2.0 * g)
    assert not jnp.allclose(default, scaled, atol=1e-10)
