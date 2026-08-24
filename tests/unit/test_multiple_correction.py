"""The multiple-correction reconstruction: a gradient and Hessian with no system to solve."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.schemes import (
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.schemes.gradient import expand_symmetric

from tests.support.meshes import (
    perturbed_grid_2d,
    perturbed_grid_3d,
    tetrahedral_grid_3d,
)

QUADRATIC_MESHES = [
    perturbed_grid_2d(6, 6, perturb=0.40, seed=2),
    perturbed_grid_3d(4, 4, 4, perturb=0.35, seed=4),
    tetrahedral_grid_3d(3, perturb=0.25, seed=6),
]
MESH_IDS = ["quadrilateral", "hexahedral", "tetrahedral"]


def _quadratic(mesh, seed=0):
    """A quadratic field with its analytic gradient and Hessian, sampled at cells and faces."""
    dim = mesh.dim
    rng = np.random.default_rng(seed)
    hessian = rng.standard_normal((dim, dim))
    hessian = hessian + hessian.T
    linear = rng.standard_normal(dim)
    geometry = mesh.geometry()
    cell = np.asarray(geometry.cell.centroid)
    face = np.asarray(geometry.face.centroid)

    def value(points):
        return 0.5 * np.einsum("ni,ij,nj->n", points, hessian, points) + points @ linear

    return {
        "geometry": geometry,
        "cell_values": jnp.asarray(value(cell)),
        "face_values": jnp.asarray(value(face)),
        "gradient": cell @ hessian + linear,
        "hessian": hessian,
    }


@pytest.mark.parametrize("mesh", QUADRATIC_MESHES, ids=MESH_IDS)
def test_the_reconstruction_is_exact_for_a_quadratic_field(mesh) -> None:
    """The scheme's whole reason to exist, on the mesh shapes that make it hard.

    Exactness rather than an order of accuracy: the operators are built to reproduce quadratics, so
    on a quadratic the error is roundoff at every mesh size. A tolerance loose enough to pass a
    merely second-order scheme would not test the property.
    """
    case = _quadratic(mesh)
    scheme = MultipleCorrectionGradient().bind(mesh, case["geometry"])
    gradient, packed = scheme.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    hessian = np.asarray(expand_symmetric(packed, mesh.dim))

    assert (
        np.abs(np.asarray(gradient) - case["gradient"]).max()
        < 1e-12 * np.abs(case["gradient"]).max()
    )
    assert np.abs(hessian - case["hessian"]).max() < 1e-11 * np.abs(case["hessian"]).max()


def test_the_owner_closure_fails_on_boundary_tetrahedra() -> None:
    """A limitation, pinned so it cannot be rediscovered as a mystery.

    :class:`OwnerGradient` satisfies the linear-exactness requirement and is exact on hexahedra, so
    nothing about it looks wrong. It nonetheless fails on tetrahedra: such a cell has four faces,
    and at a boundary one or two of them carry no direction the owner's own gradient has not already
    supplied, leaving six Hessian components underdetermined. The failure is silent -- a finite,
    plausible, wrong answer -- which is why it is asserted rather than left to a docstring.
    """
    mesh = tetrahedral_grid_3d(3, perturb=0.25, seed=6)
    case = _quadratic(mesh)

    owner = MultipleCorrectionGradient(boundary_closure=OwnerGradient()).bind(
        mesh, case["geometry"]
    )
    gradient, _ = owner.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    relative = (
        np.abs(np.asarray(gradient) - case["gradient"]).max() / np.abs(case["gradient"]).max()
    )
    assert relative > 1e-3, "owner closure unexpectedly exact on tetrahedra -- check the mesh"

    corrected = MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient()).bind(
        mesh, case["geometry"]
    )
    gradient, _ = corrected.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    assert (
        np.abs(np.asarray(gradient) - case["gradient"]).max()
        < 1e-12 * np.abs(case["gradient"]).max()
    )


def test_the_reconstruction_is_exactly_linear_in_the_field() -> None:
    """The property that keeps the tangent cheap, asserted exactly rather than to a tolerance.

    A fixed sequence of fixed linear maps is linear, so its tangent is the same sequence applied to
    the tangent -- no implicit-function solve, no unrolled iteration. That is what makes this
    scheme usable inside a differentiated flow solve, and it is checked by comparing the
    forward-mode tangent against the map evaluated *on* the tangent, which agree bit for bit only
    if the map really is linear and homogeneous.
    """
    mesh = perturbed_grid_3d(4, 4, 4, perturb=0.30, seed=5)
    geometry = mesh.geometry()
    scheme = MultipleCorrectionGradient().bind(mesh, geometry)
    zero_faces = jnp.zeros((mesh.n_faces,))
    field = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_cells,))
    tangent = jax.random.normal(jax.random.PRNGKey(1), (mesh.n_cells,))

    def reconstruct(phi):
        return scheme.gradients(phi, mesh, geometry, zero_faces)

    # Homogeneous as well as additive: a linear map sends zero to zero, and a scheme that
    # added a geometry-dependent constant would still pass the tangent check below.
    assert not np.any(np.asarray(reconstruct(jnp.zeros_like(field))))
    _, tangent_out = jax.jvp(reconstruct, (field,), (tangent,))
    assert np.array_equal(np.asarray(tangent_out), np.asarray(reconstruct(tangent)))


def test_the_gradient_is_differentiable_in_the_field() -> None:
    """Reverse mode flows and agrees with a finite difference.

    Finiteness is not the test -- a severed adjoint returns zeros, which are finite. This compares
    a real derivative against a real perturbation.
    """
    mesh = perturbed_grid_3d(4, 4, 4, perturb=0.25, seed=7)
    geometry = mesh.geometry()
    scheme = MultipleCorrectionGradient().bind(mesh, geometry)
    boundary = jnp.zeros((mesh.n_faces,))
    field = jax.random.normal(jax.random.PRNGKey(3), (mesh.n_cells,))

    def loss(phi):
        return jnp.sum(scheme.gradients(phi, mesh, geometry, boundary) ** 2)

    gradient = jax.grad(loss)(field)
    assert np.abs(np.asarray(gradient)).max() > 0.0

    # A CENTRAL difference: the forward one carries an O(step) truncation error, which on this
    # loss is ~2e-5 relative and would be measuring the differencing scheme rather than the
    # derivative.
    index, step = 11, 1e-6
    difference = (loss(field.at[index].add(step)) - loss(field.at[index].add(-step))) / (2.0 * step)
    assert np.isclose(float(gradient[index]), float(difference), rtol=1e-8)


def test_binding_changes_the_cost_and_not_the_answer() -> None:
    """A bound scheme returns the same reconstruction bit for bit."""
    mesh = perturbed_grid_2d(5, 5, perturb=0.25, seed=8)
    case = _quadratic(mesh, seed=2)
    plain = MultipleCorrectionGradient()
    bound = plain.bind(mesh, case["geometry"])
    assert bound.prepared is not None

    unbound_gradient, unbound_hessian = plain.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    bound_gradient, bound_hessian = bound.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    assert np.array_equal(np.asarray(unbound_gradient), np.asarray(bound_gradient))
    assert np.array_equal(np.asarray(unbound_hessian), np.asarray(bound_hessian))


def test_it_refuses_a_domain_decomposed_solve_rather_than_silently_misreconstructing() -> None:
    """The distributed path is unbuilt, and the refusal says so in the terms a reader needs."""
    mesh = perturbed_grid_2d(4, 4, perturb=0.2, seed=9)
    geometry = mesh.geometry()
    with pytest.raises(NotImplementedError, match="one ring per pass"):
        MultipleCorrectionGradient().gradients(
            jnp.zeros((mesh.n_cells,)),
            mesh,
            geometry,
            jnp.zeros((mesh.n_faces,)),
            operator_hook=lambda x: x,
        )
