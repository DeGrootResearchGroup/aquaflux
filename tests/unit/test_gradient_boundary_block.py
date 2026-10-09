"""Green--Gauss reconstructions that satisfy the boundary conditions they are reconstructed against.

A zero-gradient, Neumann or Robin face value is affine in its owner's gradient, ``phi_f0 + w . g``,
and a residual assembler hands the scheme ``phi_f0`` and ``w``. On a skewed boundary cell ``w`` is
non-zero, so a scheme summing against ``phi_f0`` alone contradicts the very field it reconstructs.
:func:`boundary_gradient_block` turns ``w`` into the per-cell block every Green--Gauss scheme moves
to its left-hand side; these tests pin that each scheme does, and that nothing moves where ``w`` is
zero.
"""

from __future__ import annotations

import dataclasses
import warnings

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import (
    BoundaryConditions,
    Convective,
    Dirichlet,
    Neumann,
    ZeroGradient,
)
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    ExactCellBlock,
    GmresGradientSolve,
    GradientScheme,
    InverseCellVolume,
    MultipleCorrectionGradient,
    SweptGradientSolve,
    boundary_gradient_block,
    contraction_rate,
)
from aquaflux.schemes.gradient import CellPreconditioner, InverseVolume, _small_inverse
from aquaflux.vectors import dot, scale

from tests.support.meshes import (
    columnwise_perturbed_grid_3d,
    perturbed_grid_2d,
    tetrahedral_grid_3d,
)

#: The diffusivity and Robin exchange coefficient the flux-type conditions run at. Neither is one, so
#: a face value that dropped either would be caught.
_DIFFUSIVITY, _EXCHANGE = 1.3, 2.5


class _Blind(GradientScheme):
    """``inner`` with the boundary gradient weight withheld: exactly the behaviour before it was used."""

    inner: GradientScheme

    def _reconstruct_gradient(self, field, mesh, geometry, boundary_values, **kwargs):
        kwargs["boundary_gradient_weight"] = None
        return self.inner._reconstruct_gradient(field, mesh, geometry, boundary_values, **kwargs)


class _PlainVolume(CellPreconditioner):
    """``1/V`` whatever the operator holds per cell -- the preconditioner before the block existed."""

    def build(self, terms):
        return InverseVolume(1.0 / terms.volume)


def _sheared_grid(nx, ny, ly, shear):
    """A grid of identical parallelograms: every interior face centroid lies on its P--N line.

    So a compact Green--Gauss sum is exact for a linear field in every cell owning no boundary face,
    and the only error left is at the boundary -- where the horizontal walls' face centroids sit a
    tangential ``shear * h_y / 2`` off their owners' normals. That isolates the term under test.
    """
    mesh = structured_grid_2d(nx, ny, 1.0, ly, named_boundaries=True)
    nodes = mesh.node_coords
    return eqx.tree_at(lambda m: m.node_coords, mesh, nodes.at[:, 0].add(shear * nodes[:, 1] / ly))


def _condition(kind, faces, geometry, gradient):
    """``kind`` on ``faces``, with the data the linear field ``x . gradient`` carries there."""
    centroid = geometry.face.centroid[faces]
    normal_derivative = geometry.face.normal[faces] @ gradient
    return {
        "Dirichlet": lambda: Dirichlet(value=centroid @ gradient),
        "ZeroGradient": lambda: ZeroGradient(),
        # The outward flux is -Gamma dphi/dn.
        "Neumann": lambda: Neumann(flux=-_DIFFUSIVITY * normal_derivative),
        # Gamma dphi/dn = h (Tinf - phi), so Tinf = phi + (Gamma / h) dphi/dn.
        "Convective": lambda: Convective(
            h=_EXCHANGE, t_inf=centroid @ gradient + _DIFFUSIVITY / _EXCHANGE * normal_derivative
        ),
    }[kind]()


def _assembler(mesh, conditions, scheme):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # an under-resolved sweep warns; these tests judge the answer
        return ResidualAssembler.build(
            mesh,
            mesh.geometry(),
            PropertyModel({"diffusivity": Constant(_DIFFUSIVITY)}),
            (DiffusionFlux(),),
            BoundaryConditions(conditions),
            gradient_scheme=scheme,
        )


def _linear_case(mesh, kind, walls, scheme):
    """``kind`` on ``walls``, the exact value elsewhere, and a linear field satisfying both.

    A zero-gradient field cannot vary normal to its walls, so it varies only along the first axis and
    the walls are the ones normal to the others; the flux-type conditions take any gradient.
    """
    geometry = mesh.geometry()
    gradient = jnp.array([1.7, -1.1, 0.6][: mesh.dim])
    if kind == "ZeroGradient":
        gradient = gradient.at[1:].set(0.0)
    conditions = {
        name: _condition(
            kind if name in walls else "Dirichlet",
            mesh.face_patches.indices(name),
            geometry,
            gradient,
        )
        for name in mesh.face_patches.names
        if name != "interior" and (name != "boundary" or name in walls)
    }
    return _assembler(mesh, conditions, scheme), geometry.cell.centroid @ gradient, gradient


def _error(assembler, field, gradient):
    result = assembler.gradient(field)
    return float(jnp.max(jnp.linalg.norm(result - gradient, axis=-1)) / jnp.linalg.norm(gradient))


#: Each scheme with a mesh on which it is exact for a linear field away from the boundary, so the
#: boundary is all that can be wrong. Corrected Green--Gauss is linear-exact on any mesh of planar
#: faces (solved exactly here, since a fixed sweep count is exact only to its own residual);
#: compact Green--Gauss only where the interior face centroids lie on their P--N lines.
_EXACT_CASES = [
    pytest.param(
        perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True),
        CorrectedGreenGauss(solver=GmresGradientSolve()),
        ("bottom", "top"),
        id="corrected-quadrilateral",
    ),
    pytest.param(
        columnwise_perturbed_grid_3d(4, 4, 3, perturb=0.3, seed=4, named_boundaries=True),
        CorrectedGreenGauss(solver=GmresGradientSolve()),
        ("bottom", "top"),
        id="corrected-hexahedral",
    ),
    pytest.param(
        _sheared_grid(8, 8, 1.0, 0.4),
        CompactGreenGauss(),
        ("bottom", "top"),
        id="compact-parallelogram",
    ),
]


@pytest.mark.parametrize("kind", ["ZeroGradient", "Neumann", "Convective"])
@pytest.mark.parametrize(("mesh", "scheme", "walls"), _EXACT_CASES)
def test_a_linear_field_satisfying_its_conditions_reconstructs_exactly(
    mesh, scheme, walls, kind
) -> None:
    """Exact to roundoff with the boundary block, and off by a non-vanishing margin without it.

    The second half is the control the issue asked for: without the weight -- the behaviour before
    these schemes read it -- the same field misses by 0.15--0.35 of its gradient on every case here,
    so exactness is the block's doing and not the fixture's.
    """
    assembler, field, gradient = _linear_case(mesh, kind, walls, scheme)
    assert _error(assembler, field, gradient) < 1e-12
    blind = dataclasses.replace(assembler, gradient_scheme=_Blind(assembler.gradient_scheme))
    assert _error(blind, field, gradient) > 1e-2


@pytest.mark.parametrize("kind", ["Neumann", "Convective"])
def test_tetrahedra_with_two_flux_faces_reconstruct_exactly(kind) -> None:
    """The whole boundary of a tetrahedral cube carries the condition, so its edge cells own two.

    Those are the cells whose interior faces span too few directions on their own: the conditions
    supply the missing one, through the block. Zero-gradient is absent only because no nonconstant
    linear field has a vanishing normal derivative on every face of a cube.
    """
    mesh = tetrahedral_grid_3d(3, perturb=0.25, seed=6)
    scheme = CorrectedGreenGauss(solver=GmresGradientSolve())
    assembler, field, gradient = _linear_case(mesh, kind, ("boundary",), scheme)
    assert _error(assembler, field, gradient) < 1e-12
    blind = dataclasses.replace(assembler, gradient_scheme=_Blind(assembler.gradient_scheme))
    assert _error(blind, field, gradient) > 1e-2


@pytest.mark.parametrize(
    "scheme",
    [
        CompactGreenGauss(),
        CorrectedGreenGauss(),
        CorrectedGreenGauss(preconditioner=ExactCellBlock()),
        MultipleCorrectionGradient(),
    ],
    ids=["compact", "corrected", "corrected-exact-block", "multiple-correction"],
)
def test_an_orthogonal_mesh_is_bit_identical_to_ignoring_the_weight(scheme) -> None:
    """Where every boundary face centroid lies on its owner's normal the weight is exactly zero.

    Then each scheme must return precisely what it returned before reading the weight -- including
    the corrected scheme, whose preconditioner becomes a per-cell block inverse of ``V I`` instead of
    a ``1/V`` scaling, and must not move a bit for it.
    """
    mesh = structured_grid_2d(8, 6, 1.0, 0.7, named_boundaries=True)
    assembler, _, _ = _linear_case(mesh, "Convective", ("bottom", "top", "left"), scheme)
    field = jax.random.normal(jax.random.PRNGKey(3), (mesh.n_cells,))
    blind = dataclasses.replace(assembler, gradient_scheme=_Blind(assembler.gradient_scheme))
    assert jnp.array_equal(assembler.gradient(field), blind.gradient(field))


@pytest.mark.parametrize(
    "scheme",
    [CompactGreenGauss(), CorrectedGreenGauss()],
    ids=["compact", "corrected"],
)
def test_the_reconstruction_stays_exactly_linear_in_the_field(scheme) -> None:
    """Under homogeneous conditions the map from field to gradient is linear, block included.

    So its tangent is the map itself, with no implicit-function solve: ``jvp`` at any point in any
    direction returns the reconstruction of that direction. Checked on a skewed mesh with
    zero-gradient walls, where the block is non-zero.
    """
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    conditions = {
        "left": Dirichlet(value=0.0),
        "right": ZeroGradient(),
        "bottom": ZeroGradient(),
        "top": ZeroGradient(),
    }
    assembler = _assembler(mesh, conditions, scheme)
    point = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_cells,))
    direction = jax.random.normal(jax.random.PRNGKey(1), (mesh.n_cells,))
    tangent = jax.jvp(assembler.gradient, (point,), (direction,))[1]
    assert float(jnp.max(jnp.abs(tangent - assembler.gradient(direction)))) < 1e-13


def test_both_cell_preconditioners_invert_the_boundary_block() -> None:
    """On parallelograms the operator IS its per-cell blocks, so one sweep must solve it exactly.

    Every interior skewness offset vanishes there, leaving ``A_g = V (I - B)``, and a single sweep
    returns ``P^-1 b`` -- exact precisely when the preconditioner is that block. The thin cells make
    ``B`` large (here up to 10) and the zero-gradient side walls give the corner cells two
    derivative-type faces, where ``B`` is not nilpotent. A ``1/V`` scaling, the control, is off by
    orders of magnitude after one sweep and still at ``6e-4`` after eight on the same mesh.
    """
    mesh = _sheared_grid(16, 16, 0.02, 0.4)
    conditions = {name: ZeroGradient() for name in ("left", "right", "bottom", "top")}
    geometry = mesh.geometry()
    centroid = geometry.cell.centroid
    field = jnp.sin(3.0 * centroid[:, 0]) + 7.0 * centroid[:, 1] ** 2
    exact = _assembler(mesh, conditions, CorrectedGreenGauss(solver=GmresGradientSolve()))
    reference = exact.gradient(field)

    def one_sweep(preconditioner):
        scheme = CorrectedGreenGauss(
            solver=SweptGradientSolve(sweeps=1, warn_tol=None), preconditioner=preconditioner
        )
        result = _assembler(mesh, conditions, scheme).gradient(field)
        return float(jnp.linalg.norm(result - reference) / jnp.linalg.norm(reference))

    assert one_sweep(InverseCellVolume()) < 1e-13
    assert one_sweep(ExactCellBlock()) < 1e-13
    assert one_sweep(_PlainVolume()) > 1e-2


def test_the_swept_contraction_rate_is_unchanged_by_the_block() -> None:
    """The block reaches only a cell's own gradient, so the sweep's asymptotic rate stays where it was.

    For a cell owning one such face ``B = s w^T / V`` with ``w`` tangential to that face, so
    ``B^2 = 0``: it adds no eigenvalue. The rate is set by the interior skewness coupling, before and
    after (measured 0.1817 against 0.1816 here) -- which is also why a sweep count calibrated on the
    geometry alone stays valid for the operator that carries the block.
    """
    mesh = perturbed_grid_2d(16, 16, perturb=0.3, seed=3, named_boundaries=True)
    conditions = {
        "left": Dirichlet(value=0.0),
        "right": Dirichlet(value=0.0),
        "bottom": ZeroGradient(),
        "top": ZeroGradient(),
    }
    assembler = _assembler(mesh, conditions, CorrectedGreenGauss())
    weight = assembler._build_time_boundary_linearization().gradient_weight
    geometry = mesh.geometry()
    before = contraction_rate(
        CorrectedGreenGauss.system(CorrectedGreenGauss.terms(mesh, geometry))
    ).rate
    after = contraction_rate(
        CorrectedGreenGauss.system(CorrectedGreenGauss.terms(mesh, geometry, weight))
    ).rate
    assert after < 1.01 * before


def test_the_block_is_the_boundary_faces_reading_their_owners_gradient() -> None:
    """``B_P g`` is what the boundary faces add to a Green--Gauss sum when their value moves by ``w.g``.

    Built from the definition rather than the formula: perturb every boundary face value by
    ``w_f . g_owner`` for a random ``g``, and the change in the per-volume sum is ``B_P g``. Interior
    entries of the weight are deliberately non-zero, since the block must ignore them.
    """
    mesh = perturbed_grid_2d(5, 4, perturb=0.3, seed=2)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    weight = jax.random.normal(jax.random.PRNGKey(5), (mesh.n_faces, mesh.dim))
    gradient = jax.random.normal(jax.random.PRNGKey(6), (mesh.n_cells, mesh.dim))
    face_value = jnp.where(face_cells.interior, 0.0, dot(weight, gradient[face_cells.owner]))
    area = scale(geometry.face.normal, geometry.face.area)
    added = face_cells.scatter_conservative(scale(area, face_value))
    expected = scale(added, 1.0 / geometry.cell.volume)
    block = boundary_gradient_block(weight, face_cells, geometry)
    np.testing.assert_allclose(
        np.asarray(jnp.einsum("nij,nj->ni", block, gradient)), np.asarray(expected), atol=1e-14
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_closed_form_inverse_matches_a_library_inverse_and_keeps_the_identity_exact(
    dim,
) -> None:
    """The adjugate inverse the reconstructions use in place of a batched library call.

    It must agree with ``jnp.linalg.inv`` on a general well-conditioned batch, and return the identity
    exactly for the identity -- that is what keeps every scheme bit-identical where no boundary
    condition reads the gradient.
    """
    matrix = jnp.eye(dim) + 0.3 * jax.random.normal(jax.random.PRNGKey(dim), (64, dim, dim))
    np.testing.assert_allclose(
        np.asarray(_small_inverse(matrix)), np.asarray(jnp.linalg.inv(matrix)), atol=1e-12
    )
    identity = jnp.broadcast_to(jnp.eye(dim), (8, dim, dim))
    assert jnp.array_equal(_small_inverse(identity), identity)
