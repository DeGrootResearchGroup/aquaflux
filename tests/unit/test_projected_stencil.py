"""`ProjectedStencilGradient`: compact weights, exact for quadratics, chosen to damp.

What each test pins, and the wrong answer it catches:

* exactness for a quadratic, on tetrahedra and on perturbed hexahedra, under prescribed values and
  under zero-gradient conditions -- the property the constrained construction exists for, and the one
  a wrong constraint row (a monomial's value where its normal derivative belongs, an unscaled
  coordinate) silently breaks;
* the weights stay bounded where the multiple-correction scheme's do not, which is what decides the
  sign of the Rhie--Chow damping;
* the blend moves the weights towards the reference and away from the minimum-norm ones, so a blend
  that was ignored would not pass;
* the minimum-norm end equals an independently built unweighted quadratic least-squares fit, which
  says what `blend = 0` is (the classical construction) and is the only test here sensitive to WHICH
  exact weights the fit lands on -- a 0.2 % departure from that end fails it and nothing else;
* the stencil reaches exactly `reach` face hops, so a Jacobian's sparsity is what the scheme claims;
* a Robin condition is refused rather than silently reconstructed from the wrong datum;
* an unbound scheme, a stale binding and the distributed hook each raise rather than return a wrong
  gradient;
* an imposed gradient survives the reconstruction.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.schemes import (
    BoundaryLinearization,
    ImposedGradient,
    MultipleCorrectionGradient,
    ProjectedStencilGradient,
    SkewCorrectedGradient,
)
from aquaflux.schemes.projected_stencil import build_stencil

from tests.support.meshes import perturbed_grid_3d, tetrahedral_grid_3d

HESSIAN = np.array([[1.3, 0.4, -0.2], [0.4, -0.7, 0.3], [-0.2, 0.3, 0.9]])
SLOPE = np.array([0.5, -1.1, 0.8])


def quadratic(points: np.ndarray) -> np.ndarray:
    return 0.5 * np.einsum("...i,ij,...j->...", points, HESSIAN, points) + points @ SLOPE


def quadratic_gradient(points: np.ndarray) -> np.ndarray:
    return points @ HESSIAN + SLOPE


@pytest.fixture(scope="module")
def tetrahedra():
    mesh = tetrahedral_grid_3d(3, perturb=0.25, seed=6)
    return mesh, mesh.geometry()


def _prescribed_values(mesh):
    """A linearization whose every boundary face carries a prescribed value."""
    return BoundaryLinearization(
        value_weight=jnp.zeros(mesh.n_faces),
        gradient_weight=jnp.zeros((mesh.n_faces, mesh.dim)),
    )


def _zero_gradient(mesh):
    """A linearization whose every boundary face extrapolates from its owner."""
    boundary = ~jnp.asarray(mesh.face_cells.interior)
    return BoundaryLinearization(
        value_weight=jnp.where(boundary, 1.0, 0.0),
        gradient_weight=jnp.zeros((mesh.n_faces, mesh.dim)),
    )


def _error(scheme, mesh, geometry, boundary_values, field, exact):
    gradient = scheme.gradients(jnp.asarray(field), mesh, geometry, jnp.asarray(boundary_values))
    return float(jnp.max(jnp.abs(gradient - exact))) / float(np.abs(exact).max())


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: tetrahedral_grid_3d(3, perturb=0.25, seed=6), id="tetrahedra"),
        pytest.param(lambda: perturbed_grid_3d(4, 4, 4, perturb=0.25, seed=2), id="hexahedra"),
    ],
)
def test_it_reconstructs_a_quadratic_exactly_under_prescribed_values(build) -> None:
    """The constrained construction's defining property, on both cell shapes.

    A tetrahedron has four face neighbours against the nine coefficients a quadratic needs in three
    dimensions, so nothing on the one-hop stencil could do this; the two-hop stencil is what makes it
    possible and the constraints are what make it happen.
    """
    mesh = build()
    geometry = mesh.geometry()
    centroid = np.asarray(geometry.cell.centroid)
    scheme = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    error = _error(
        scheme,
        mesh,
        geometry,
        quadratic(np.asarray(geometry.face.centroid)),
        quadratic(centroid),
        quadratic_gradient(centroid),
    )
    assert error < 1e-9, error


def test_it_reconstructs_a_quadratic_exactly_under_zero_gradient_conditions(tetrahedra) -> None:
    """A zero-gradient face's datum is its NORMAL DERIVATIVE, and the stencil is constrained with the
    monomials' normal derivatives there rather than their values.

    Constraining with the values instead would still reproduce a quadratic whose normal derivative
    happens to vanish at every wall; this quadratic's does not, so the two constructions differ here.
    The boundary values a residual assembler passes are evaluated at zero gradient, which for this
    condition means ``phi_owner + (d . n) dphi/dn`` -- what the scheme inverts to recover the datum.
    """
    mesh, geometry = tetrahedra
    centroid = np.asarray(geometry.cell.centroid)
    face_centroid = np.asarray(geometry.face.centroid)
    owner = np.asarray(mesh.face_cells.owner)
    normal = np.asarray(geometry.face.normal)
    along = np.sum((face_centroid - centroid[owner]) * normal, axis=-1)
    derivative = np.sum(quadratic_gradient(face_centroid) * normal, axis=-1)
    # what the assembler passes for an extrapolating condition, evaluated at zero gradient
    boundary_values = quadratic(centroid)[owner] + along * derivative

    scheme = ProjectedStencilGradient().bind(mesh, geometry, _zero_gradient(mesh))
    error = _error(
        scheme, mesh, geometry, boundary_values, quadratic(centroid), quadratic_gradient(centroid)
    )
    assert error < 1e-9, error


def test_its_weights_stay_bounded_where_the_multiple_correction_scheme_amplifies(
    tetrahedra,
) -> None:
    """The size of the weights is what decides the Rhie--Chow damping's sign, and exactness alone
    does not bound them: both schemes here are exact for quadratics on this mesh, and their largest
    weights differ by an order: on this fixture the multiple-correction scheme's worst weight is 145
    against this scheme's 7.9, and on the tetrahedral duct of
    ``validation/tetrahedral_gradient_ab`` the same comparison is ~2900 against ~1.2 (relative to the
    reference reconstruction's largest weight), where its reconstruction anti-damps and this one does
    not.
    """
    mesh, geometry = tetrahedra
    n, n_faces = mesh.n_cells, mesh.n_faces
    scheme = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    _, weights, face_weights, _, _ = scheme.prepared
    compact = max(float(jnp.abs(weights).max()), float(jnp.abs(face_weights).max()))

    multiple = MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(mesh, geometry)
    rows = jax.jacfwd(lambda phi: multiple.reconstruct(phi, mesh, geometry, jnp.zeros(n_faces))[0])(
        jnp.zeros(n)
    )
    assert compact < 0.1 * float(jnp.abs(rows).max())


def test_the_blend_moves_the_weights_between_the_two_ends(tetrahedra) -> None:
    """``blend`` is the whole choice this scheme makes within the exact weights, so it must move them.

    At ``0`` they are the minimum-norm exact weights and so have the smallest norm of any exact
    choice; raising it trades that for closeness to the reference, and every blend stays exact.
    """
    mesh, geometry = tetrahedra
    linearization = _prescribed_values(mesh)
    norms = []
    for blend in (0.0, 0.5, 1.0):
        scheme = ProjectedStencilGradient(blend=blend).bind(mesh, geometry, linearization)
        _, weights, _, _, _ = scheme.prepared
        norms.append(float(jnp.sum(weights**2)))
    assert norms[0] < norms[1] < norms[2]

    centroid = np.asarray(geometry.cell.centroid)
    for blend in (0.0, 1.0):
        scheme = ProjectedStencilGradient(blend=blend).bind(mesh, geometry, linearization)
        error = _error(
            scheme,
            mesh,
            geometry,
            quadratic(np.asarray(geometry.face.centroid)),
            quadratic(centroid),
            quadratic_gradient(centroid),
        )
        assert error < 1e-9, (blend, error)


def test_the_minimum_norm_end_is_an_unweighted_quadratic_least_squares_fit(tetrahedra) -> None:
    """At ``blend = 0`` this scheme IS the classical construction, and that is worth stating.

    Fit a quadratic to the stencil's data by unweighted least squares and differentiate it at the
    cell: with more stencil entries than monomials that fit is ``w = B (B^T B)^-1 e``, which is the
    same operator as the smallest weights satisfying ``B^T w = e``. So the minimum-norm end is not a
    new reconstruction, and the blend is the whole of what this scheme adds to it.

    Checked on a field that is NOT a quadratic, so both sides are wrong in the same way rather than
    exact for independent reasons -- which is the only version of this comparison that can fail. The
    stencil membership is taken from the scheme; what is built independently here is the fit, down to
    its own monomials in unscaled coordinates (a least-squares fit is invariant to that scaling, so
    agreeing across it is evidence rather than a shared convention).
    """
    mesh, geometry = tetrahedra
    centroid = np.asarray(geometry.cell.centroid)
    face_centroid = np.asarray(geometry.face.centroid)

    def smooth(points: np.ndarray) -> np.ndarray:
        return np.sin(1.3 * points[:, 0]) * np.exp(0.2 * points[:, 1]) + np.cos(0.7 * points[:, 2])

    field, boundary_values = smooth(centroid), smooth(face_centroid)
    scheme = ProjectedStencilGradient(blend=0.0).bind(mesh, geometry, _prescribed_values(mesh))
    reconstructed = np.asarray(
        scheme.gradients(jnp.asarray(field), mesh, geometry, jnp.asarray(boundary_values))
    )

    stencil = build_stencil(mesh, 2)
    cells, cell_used = np.asarray(stencil.cells), np.asarray(stencil.cell_used)
    faces, face_used = np.asarray(stencil.faces), np.asarray(stencil.face_used)
    fitted = np.zeros_like(reconstructed)
    for cell in range(mesh.n_cells):
        positions = np.concatenate(
            [centroid[cells[cell][cell_used[cell]]], face_centroid[faces[cell][face_used[cell]]]]
        )
        data = np.concatenate(
            [field[cells[cell][cell_used[cell]]], boundary_values[faces[cell][face_used[cell]]]]
        )
        d = positions - centroid[cell]
        monomials = np.stack(
            [
                np.ones(len(d)),
                d[:, 0],
                d[:, 1],
                d[:, 2],
                d[:, 0] ** 2,
                d[:, 1] ** 2,
                d[:, 2] ** 2,
                d[:, 0] * d[:, 1],
                d[:, 0] * d[:, 2],
                d[:, 1] * d[:, 2],
            ],
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(monomials, data, rcond=None)
        # Differentiated at the expansion point, where every quadratic term's derivative vanishes.
        fitted[cell] = coefficients[1:4]

    error = np.max(np.abs(reconstructed - fitted)) / np.max(np.abs(fitted))
    assert error < 1e-8, error


def test_the_stencil_reaches_exactly_the_hops_it_claims(tetrahedra) -> None:
    """A residual's Jacobian inherits this stencil, so its width is part of the scheme's contract."""
    mesh, _ = tetrahedra
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    interior = neighbour >= 0
    adjacent = [{p} for p in range(mesh.n_cells)]
    for a, b in zip(owner[interior], neighbour[interior], strict=True):
        adjacent[a].add(b)
        adjacent[b].add(a)

    for reach in (1, 2):
        expected = [set(part) for part in adjacent]
        for _ in range(reach - 1):
            expected = [set().union(*[adjacent[q] for q in part]) for part in expected]
        stencil = build_stencil(mesh, reach)
        cells, used = np.asarray(stencil.cells), np.asarray(stencil.cell_used)
        for p in (0, mesh.n_cells // 2, mesh.n_cells - 1):
            assert set(cells[p][used[p]]) == expected[p], (reach, p)


def test_a_robin_condition_is_refused_rather_than_read_as_the_wrong_datum(tetrahedra) -> None:
    """A convective or Robin face's value is part owner and part ambient, so neither a value nor a
    normal derivative is prescribed there and no stencil constraint expresses it. Reconstructing
    anyway would read the ambient-weighted value as if it were one of the two."""
    mesh, geometry = tetrahedra
    boundary = ~jnp.asarray(mesh.face_cells.interior)
    robin = BoundaryLinearization(
        value_weight=jnp.where(boundary, 0.4, 0.0),
        gradient_weight=jnp.zeros((mesh.n_faces, mesh.dim)),
    )
    with pytest.raises(ValueError, match="Robin or convective"):
        ProjectedStencilGradient().bind(mesh, geometry, robin)


def test_an_unbound_or_stale_scheme_refuses_to_reconstruct(tetrahedra) -> None:
    """The weights are the scheme; without the right ones it would return a plausible wrong gradient
    rather than fail, which is the failure this pair of guards exists to prevent."""
    mesh, geometry = tetrahedra
    field, values = jnp.zeros(mesh.n_cells), jnp.zeros(mesh.n_faces)
    with pytest.raises(ValueError, match="has not been bound"):
        ProjectedStencilGradient().gradients(field, mesh, geometry, values)

    other = tetrahedral_grid_3d(2, perturb=0.2, seed=3)
    bound = ProjectedStencilGradient().bind(other, other.geometry(), _prescribed_values(other))
    assert bound.prepared[1].shape[0] != mesh.n_cells
    with pytest.raises(ValueError, match="was bound to a geometry of"):
        bound.gradients(field, mesh, geometry, values)


def test_the_distributed_hook_is_refused_rather_than_ignored(tetrahedra) -> None:
    """A two-hop stencil reads cells a one-deep halo does not carry, so a silently-accepted hook
    would return a gradient built from stale ghost values."""
    mesh, geometry = tetrahedra
    scheme = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    with pytest.raises(NotImplementedError, match="two-deep halo"):
        scheme.gradients(
            jnp.zeros(mesh.n_cells),
            mesh,
            geometry,
            jnp.zeros(mesh.n_faces),
            operator_hook=lambda g: g,
        )


def test_binding_against_a_traced_mesh_is_refused_with_the_reason(tetrahedra) -> None:
    """Its stencil and weights are built from concrete connectivity, so a mesh that is traced -- an
    assembler built inside a residual evaluation, which is what the turbulence closure does for k and
    omega -- cannot bind. Without this the failure is a `TracerArrayConversionError` from inside
    numpy, naming a boolean array and nothing about what to do."""
    mesh, geometry = tetrahedra
    with pytest.raises(NotImplementedError, match="needs a concrete mesh"):
        jax.jit(lambda m: ProjectedStencilGradient().bind(m, geometry).prepared[1])(mesh)


def test_an_imposed_gradient_survives_the_reconstruction(tetrahedra) -> None:
    """The wall treatment of a turbulence closure imposes a gradient it knows; a scheme that dropped
    it would reconstruct those cells from the field instead, silently."""
    mesh, geometry = tetrahedra
    centroid = np.asarray(geometry.cell.centroid)
    scheme = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    cells = jnp.asarray([1, 4, 9])
    imposed_value = jnp.asarray([[2.0, -3.0, 0.5]] * 3)
    gradient = scheme.gradients(
        jnp.asarray(quadratic(centroid)),
        mesh,
        geometry,
        jnp.asarray(quadratic(np.asarray(geometry.face.centroid))),
        imposed=ImposedGradient(cells=cells, gradient=imposed_value),
    )
    np.testing.assert_allclose(np.asarray(gradient[cells]), np.asarray(imposed_value), atol=1e-12)


def test_it_is_differentiable_in_the_field(tetrahedra) -> None:
    """The reconstruction is a fixed linear map of the field, so a flow solve's Jacobian and adjoint
    both run through it; a finite difference pins that the map that differentiates is the one that
    reconstructs."""
    mesh, geometry = tetrahedra
    scheme = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    values = jnp.zeros(mesh.n_faces)
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal(mesh.n_cells))
    direction = jnp.asarray(rng.standard_normal(mesh.n_cells))

    def total(phi):
        return jnp.sum(scheme.gradients(phi, mesh, geometry, values) ** 2)

    analytic = float(jnp.vdot(jax.grad(total)(field), direction))
    step = 1e-6
    numeric = float(
        (total(field + step * direction) - total(field - step * direction)) / (2 * step)
    )
    assert abs(analytic - numeric) <= 1e-6 * max(abs(numeric), 1.0)


def test_binding_without_conditions_reads_every_boundary_face_as_prescribed(tetrahedra) -> None:
    """An assembler also builds a condition-free form, for an initializer with no field in view. That
    binding must still reconstruct -- and it is the prescribed-value one, which is what a caller
    passing face values means."""
    mesh, geometry = tetrahedra
    centroid = np.asarray(geometry.cell.centroid)
    free = ProjectedStencilGradient().bind(mesh, geometry)
    explicit = ProjectedStencilGradient().bind(mesh, geometry, _prescribed_values(mesh))
    for scheme in (free, explicit):
        error = _error(
            scheme,
            mesh,
            geometry,
            quadratic(np.asarray(geometry.face.centroid)),
            quadratic(centroid),
            quadratic_gradient(centroid),
        )
        assert error < 1e-9, error
    assert bool(eqx.tree_equal(free.prepared[1], explicit.prepared[1]))
