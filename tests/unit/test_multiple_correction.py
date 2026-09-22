"""The multiple-correction reconstruction: a gradient and Hessian with no system to solve."""

from __future__ import annotations

import dataclasses
import warnings

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import (
    BoundaryConditions,
    Convective,
    Dirichlet,
    DirichletField,
    Neumann,
    ZeroGradient,
)
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.flow import MomentumContinuity, MovingWall, PressureOutlet
from aquaflux.properties import Constant, Property, PropertyModel
from aquaflux.schemes import (
    CellwiseFallback,
    CompactGreenGauss,
    CorrectedGreenGauss,
    HessianCorrectedGradient,
    ImposedGradient,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
    multiple_correction,
)
from aquaflux.schemes.gradient import expand_symmetric
from aquaflux.schemes.interpolation import non_orthogonal_correction
from aquaflux.vectors import dot, scale

from tests.support.meshes import (
    perturbed_grid_2d,
    perturbed_grid_3d,
    tetrahedral_grid_3d,
)

#: Each mesh with the boundary closure that suits it. The default `OwnerGradient` never reads a
#: boundary value, which is what makes it usable on a real case, but a boundary TETRAHEDRON then has
#: too few independent face directions to determine six Hessian components -- so that mesh, and only
#: that mesh, is reconstructed with `SkewCorrectedGradient`. Pinned the other way round by
#: `test_the_owner_closure_fails_on_boundary_tetrahedra`.
QUADRATIC_MESHES = [
    perturbed_grid_2d(6, 6, perturb=0.40, seed=2),
    perturbed_grid_3d(4, 4, 4, perturb=0.35, seed=4),
    tetrahedral_grid_3d(3, perturb=0.25, seed=6),
]
MESH_IDS = ["quadrilateral", "hexahedral", "tetrahedral"]
QUADRATIC_CASES = list(
    zip(
        QUADRATIC_MESHES,
        [OwnerGradient(), OwnerGradient(), SkewCorrectedGradient()],
        strict=True,
    )
)


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


@pytest.mark.parametrize("mesh, closure", QUADRATIC_CASES, ids=MESH_IDS)
def test_the_reconstruction_is_exact_for_a_quadratic_field(mesh, closure) -> None:
    """The scheme's whole reason to exist, on the mesh shapes that make it hard.

    Exactness rather than an order of accuracy: the operators are built to reproduce quadratics, so
    on a quadratic the error is roundoff at every mesh size. A tolerance loose enough to pass a
    merely second-order scheme would not test the property.
    """
    case = _quadratic(mesh)
    scheme = MultipleCorrectionGradient(boundary_closure=closure).bind(mesh, case["geometry"])
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

    owner = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None).bind(
        mesh, case["geometry"]
    )
    gradient, _ = owner.reconstruct(
        case["cell_values"], mesh, case["geometry"], case["face_values"]
    )
    relative = (
        np.abs(np.asarray(gradient) - case["gradient"]).max() / np.abs(case["gradient"]).max()
    )
    assert _is_not_exact(relative), (
        "owner closure unexpectedly exact on tetrahedra -- check the mesh"
    )

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
    # The tolerance is set by the DIFFERENCE, not by the derivative: a central difference at this
    # step carries a cancellation error of order eps * |loss| / step, which is ~1e-7 relative here.
    # It is still four orders inside anything that would catch a wrong adjoint, and a severed one
    # returns zero, which the assertion above already rejects.
    index, step = 11, 1e-6
    difference = (loss(field.at[index].add(step)) - loss(field.at[index].add(-step))) / (2.0 * step)
    assert np.isclose(float(gradient[index]), float(difference), rtol=1e-6)


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


def _is_not_exact(error, tol=1e-3):
    """Did the reconstruction fail, allowing for a failure that is NaN rather than large?

    ⚠️ `error > tol` is the wrong test and it cost a CI run to learn. A singular correction is a
    large finite number under macOS Accelerate and NON-FINITE under the BLAS on the CI runners, and
    every comparison against NaN is False -- so `> tol` reports a broken reconstruction as fine.
    Negating the success condition catches both, which is what "not exact" means anyway.
    """
    return not (np.all(np.isfinite(error)) and np.all(np.asarray(error) < tol))


def _boundary_owner(mesh):
    """A cell that owns at least one boundary face, and the faces it owns there."""
    face_cells = mesh.face_cells
    boundary = np.where(~np.asarray(face_cells.interior))[0]
    owner = int(np.asarray(face_cells.owner)[boundary[0]])
    return owner, [f for f in boundary if int(np.asarray(face_cells.owner)[f]) == owner]


def test_an_imposed_gradient_is_returned_exactly_and_leaves_the_rest_reconstructed() -> None:
    """The contract a caller relies on: what was imposed comes back, and only that."""
    mesh = perturbed_grid_3d(4, 4, 4, perturb=0.30, seed=5)
    geometry = mesh.geometry()
    scheme = MultipleCorrectionGradient().bind(mesh, geometry)
    field = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_cells,))
    boundary = jnp.zeros((mesh.n_faces,))
    cells = jnp.array([0, 3, 7])
    imposed = ImposedGradient(
        cells, jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0], [4.0, 4.0, 4.0]])
    )

    gradient = scheme.gradients(field, mesh, geometry, boundary, imposed=imposed)
    assert np.array_equal(np.asarray(gradient[cells]), np.asarray(imposed.gradient))
    # Not merely overwritten at the end: the second-order defect correction is a defect of the
    # *reconstruction*, and subtracting it from an imposed value would corrupt what was imposed.


def test_an_imposed_gradient_reaches_the_hessian_rather_than_correcting_it_afterwards() -> None:
    """The reason this is an argument to the reconstruction and not a patch on its result.

    The scheme differentiates its own first estimate to build the Hessian, so an imposition that
    arrived after the reconstruction would leave that Hessian built on the value it replaces. The
    Hessian moving is the observable proof that it arrived in time -- a post-hoc patch could not
    move it at all.
    """
    mesh = perturbed_grid_3d(4, 4, 4, perturb=0.30, seed=5)
    geometry = mesh.geometry()
    scheme = MultipleCorrectionGradient().bind(mesh, geometry)
    field = jax.random.normal(jax.random.PRNGKey(2), (mesh.n_cells,))
    boundary = jnp.zeros((mesh.n_faces,))
    owner, _ = _boundary_owner(mesh)
    imposed = ImposedGradient(jnp.array([owner]), jnp.array([[5.0, -2.0, 1.0]]))

    _, plain_hessian = scheme.reconstruct(field, mesh, geometry, boundary)
    _, hessian = scheme.reconstruct(field, mesh, geometry, boundary, imposed=imposed)
    assert not np.array_equal(np.asarray(plain_hessian), np.asarray(hessian))


def test_imposing_nothing_is_byte_identical_to_never_having_been_asked() -> None:
    """The default path must not move, on the gradient or the Hessian."""
    mesh = perturbed_grid_2d(5, 5, perturb=0.25, seed=8)
    case = _quadratic(mesh, seed=2)
    scheme = MultipleCorrectionGradient().bind(mesh, case["geometry"])
    args = (case["cell_values"], mesh, case["geometry"], case["face_values"])

    for plain, given in zip(
        scheme.reconstruct(*args), scheme.reconstruct(*args, imposed=None), strict=True
    ):
        assert np.array_equal(np.asarray(plain), np.asarray(given))


def test_an_imposed_cells_boundary_faces_take_its_gradient_instead_of_the_closure() -> None:
    """A closure guesses what a boundary condition does not carry; an imposed gradient is not a guess.

    This is what unblocks :class:`SkewCorrectedGradient` on a wall ``omega``, whose boundary value is
    itself a zero-gradient closure rather than data -- differencing against it imposes a near-zero
    normal derivative exactly where the modelled profile diverges.
    """
    mesh = perturbed_grid_2d(4, 4, perturb=0.2, seed=9)
    face_cells = mesh.face_cells
    owner, owned_faces = _boundary_owner(mesh)
    imposed = ImposedGradient(jnp.array([owner]), jnp.array([[7.0, 9.0]]))

    closed = imposed.impose_on_faces(jnp.zeros((mesh.n_faces, mesh.dim)), face_cells)
    assert np.array_equal(np.asarray(closed[np.asarray(owned_faces)]), np.tile([7.0, 9.0], (2, 1)))
    untouched = np.array([f for f in range(mesh.n_faces) if f not in owned_faces])
    assert not np.any(np.asarray(closed)[untouched])


def test_an_imposed_reconstruction_is_affine_with_the_same_linear_part() -> None:
    """Imposition adds a constant; it does not make the tangent an implicit solve.

    The linearity test above pins the unimposed map. With something imposed the map is affine -- it
    no longer sends zero to zero -- but its tangent is still the same fixed sequence of fixed linear
    maps, which is the property a differentiated solve depends on. Evaluated by comparing the
    forward-mode tangent against the map run with the imposed values zeroed, which is exactly its
    linear part.
    """
    mesh = perturbed_grid_3d(4, 4, 4, perturb=0.30, seed=5)
    geometry = mesh.geometry()
    scheme = MultipleCorrectionGradient().bind(mesh, geometry)
    boundary = jnp.zeros((mesh.n_faces,))
    field = jax.random.normal(jax.random.PRNGKey(0), (mesh.n_cells,))
    tangent = jax.random.normal(jax.random.PRNGKey(1), (mesh.n_cells,))
    cells = jnp.array([0, 3, 7])
    values = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0], [4.0, 4.0, 4.0]])

    def reconstruct(phi, gradient):
        return scheme.gradients(
            phi, mesh, geometry, boundary, imposed=ImposedGradient(cells, gradient)
        )

    assert np.any(np.asarray(reconstruct(jnp.zeros_like(field), values)))  # affine, not linear
    _, tangent_out = jax.jvp(lambda phi: reconstruct(phi, values), (field,), (tangent,))
    assert np.array_equal(
        np.asarray(tangent_out), np.asarray(reconstruct(tangent, jnp.zeros_like(values)))
    )


def test_every_scheme_honours_an_imposed_gradient_whether_or_not_it_can_use_it_early() -> None:
    """The base class guarantees the contract, so a scheme with no internal consumer still honours it.

    :class:`~aquaflux.schemes.HessianCorrectedGradient` deliberately does *not* project the
    imposition onto its sweep's iterate -- that would change the fixed point rather than the path to
    it -- so this is the whole of what it does with one, and it must still do it.
    """
    mesh = perturbed_grid_2d(5, 5, perturb=0.25, seed=8)
    geometry = mesh.geometry()
    field = jax.random.normal(jax.random.PRNGKey(4), (mesh.n_cells,))
    boundary = jnp.zeros((mesh.n_faces,))
    cells = jnp.array([2, 6])
    imposed = ImposedGradient(cells, jnp.array([[1.5, -0.5], [0.25, 3.0]]))

    for scheme in (
        CompactGreenGauss(),
        CorrectedGreenGauss(),
        HessianCorrectedGradient().bind(mesh, geometry),
        MultipleCorrectionGradient().bind(mesh, geometry),
    ):
        plain = scheme.gradients(field, mesh, geometry, boundary)
        given = scheme.gradients(field, mesh, geometry, boundary, imposed=imposed)
        assert np.array_equal(np.asarray(given[cells]), np.asarray(imposed.gradient)), type(scheme)
        # Nothing else moves in a scheme that consumes no gradient of its own.
        if not isinstance(scheme, MultipleCorrectionGradient):
            assert np.array_equal(
                np.asarray(given.at[cells].set(plain[cells])), np.asarray(plain)
            ), type(scheme)


def test_a_field_satisfying_its_boundary_conditions_reconstructs_exactly() -> None:
    """Linear exactness on zero-gradient and Neumann patches, for BOTH closures, on a skewed mesh.

    In a finite-volume discretization a boundary condition holds at every iterate, through the face
    value it defines: ``phi_P + grad phi_P . (d - (d.n) n)`` on a zero-gradient face, less the
    prescribed flux on a Neumann one. That value depends on the gradient being reconstructed, and a
    residual assembler hands the scheme the value at zero gradient, which drops the tangential term.
    The perturbed grid keeps its boundary nodes on the box, so every boundary face stays axis-aligned
    while its owner's centroid is off the face normal -- exactly where that dropped term is nonzero.

    The field satisfies every condition it is given (zero normal derivative on the zero-gradient
    walls, the matching flux on the Neumann ones), so the exact gradient is the only right answer.
    Without the boundary-condition weight both closures miss it; with it both reproduce it to
    roundoff.
    """
    a = jnp.array([1.7, 0.0])  # zero normal derivative on the bottom and top walls
    b = jnp.array([1.7, -1.1])  # nonzero on every wall, so left/right carry a real flux
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    geometry = mesh.geometry()
    for gradient, boundary in (
        (
            a,
            {
                "left": ZeroGradient(),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            },
        ),
        (
            b,
            # q = -Gamma dphi/dn with Gamma = 1, i.e. -b.n on each wall.
            {
                "left": Neumann(flux=float(b[0])),
                "right": Neumann(flux=-float(b[0])),
                "bottom": Neumann(flux=float(b[1])),
                "top": Neumann(flux=-float(b[1])),
            },
        ),
    ):
        if gradient is a:
            # Zero-gradient is only this field's condition on the walls normal to y.
            boundary = {
                **boundary,
                "left": DirichletField(field_fn=lambda x: x @ a),
                "right": DirichletField(field_fn=lambda x: x @ a),
            }
        field = geometry.cell.centroid @ gradient
        for closure in (OwnerGradient(), SkewCorrectedGradient()):
            assembler = ResidualAssembler.build(
                mesh,
                geometry,
                PropertyModel({"diffusivity": Constant(1.0)}),
                (DiffusionFlux(),),
                BoundaryConditions(boundary),
                gradient_scheme=MultipleCorrectionGradient(boundary_closure=closure, fallback=None),
            )
            error = jnp.max(jnp.linalg.norm(assembler.gradient(field) - gradient, axis=-1))
            assert float(error) / float(jnp.linalg.norm(gradient)) < 1e-12, type(closure).__name__


def test_corner_tetrahedra_are_determined_by_their_boundary_conditions() -> None:
    """A tetrahedron owning two gradient-type faces has only two interior faces for three gradient
    components; its boundary conditions are what supply the missing direction.

    Reading those faces as the owner's own extrapolation instead left ``M1 - B`` singular on every such
    cell (``cond`` to ``4e18`` on a real mesh). Here the weight is the boundary condition's, and both
    the inverse and a field satisfying the conditions come out right. The weights and values are built
    by hand so the test reaches the scheme directly: zero-gradient on the walls normal to ``y`` and
    ``z``, the exact value on the walls normal to ``x``.
    """
    mesh = tetrahedral_grid_3d(3, perturb=0.25, seed=6)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    boundary = ~np.asarray(face_cells.interior)
    owned = _boundary_face_count(mesh)
    assert (owned >= 2).any()  # the fixture really has the cells this is about

    gradient = jnp.array([2.3, 0.0, 0.0])
    normal = np.asarray(geometry.face.normal)
    zero_gradient = boundary & (np.abs(normal[:, 0]) < 0.5)
    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    tangential = displacement - scale(geometry.face.normal, dot(displacement, geometry.face.normal))
    weight = jnp.where(jnp.asarray(zero_gradient)[:, None], tangential, 0.0)
    field = geometry.cell.centroid @ gradient
    exact_face = geometry.face.centroid @ gradient
    values = jnp.where(jnp.asarray(zero_gradient), field[face_cells.owner], exact_face)

    inverse = multiple_correction._boundary_condition_first_pass(
        MultipleCorrectionGradient().bind(mesh, geometry).prepared.m1, weight, face_cells, geometry
    )
    worst = np.max(np.abs(np.asarray(inverse)), axis=(1, 2))
    assert np.all(np.isfinite(worst)) and worst[owned >= 2].max() < 1e3

    scheme = MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient()).bind(
        mesh, geometry
    )
    result = scheme.reconstruct(field, mesh, geometry, values, boundary_gradient_weight=weight)[0]
    error = jnp.max(jnp.linalg.norm(result - gradient, axis=-1)) / jnp.linalg.norm(gradient)
    assert float(error) < 1e-10


def test_a_prescribed_patch_is_left_alone_by_the_boundary_condition_weight() -> None:
    """The control: where the boundary value does not depend on the gradient, nothing changes.

    The weight is zero on a prescribed patch, so the reconstruction is bit-identical to one built
    without the argument. Without this, a repair that overwrote every boundary value would pass the
    exactness tests above and quietly discard every Dirichlet datum in the problem.
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.25, seed=5)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    field = jax.random.normal(jax.random.PRNGKey(4), (mesh.n_cells,))
    values = jax.random.normal(jax.random.PRNGKey(5), (mesh.n_faces,))
    scheme = MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient()).bind(
        mesh, geometry
    )
    plain = scheme.reconstruct(field, mesh, geometry, values)[0]
    prescribed = scheme.reconstruct(
        field, mesh, geometry, values, boundary_gradient_weight=jnp.zeros((mesh.n_faces, mesh.dim))
    )[0]
    assert jnp.array_equal(prescribed, plain)
    # ...and a gradient-dependent patch genuinely moves, so the argument is not being ignored.
    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    derived = scheme.reconstruct(
        field,
        mesh,
        geometry,
        values,
        boundary_gradient_weight=jnp.where(face_cells.interior[:, None], 0.0, displacement),
    )[0]
    assert not jnp.allclose(derived, plain)


def test_the_second_pass_reads_the_geometrys_own_first_pass_operator() -> None:
    """The second pass must differentiate through the operator its corrections were probed with.

    ``M2`` and the gradient defect are built by applying the *geometry's* one-exact operator
    ``M1^-1`` to the quadratic probes. A boundary condition whose face value follows the owner's
    gradient moves that dependence onto the left-hand side of the FIRST pass, which then inverts
    ``M1 - B`` instead -- a different operator, and the right one there, because the dependence is a
    statement about the field's boundary values. Handing that boundary-condition operator to the
    second pass as well corrects an operator nobody evaluates.

    The wrong answer this catches: ``_one_exact(first, face_gradient, m1_inverse, ...)`` in place of
    ``prepared.m1_inverse`` in the second pass. That mutation passes every other test in this file.
    Measured here, it takes the Hessian error from 1.0e-2 to 6.8e-2 of ``|H|`` (and 1.8e-2 to 1.0e-1
    at 16x16, 1.3e-2 to 1.0e-1 at 32x32), so the threshold below clears both sides by about 2x.

    The scheme is bound **without** the weight here, which is the regime where the two operators
    differ, and so the one this mutation is visible in. The loss of quadratic exactness that binding
    blind causes, and its repair, are pinned separately by
    ``test_binding_against_a_boundary_condition_restores_quadratic_exactness``. With the boundary
    values prescribed (``B = 0``) the same field comes back exact either way, which the second
    assertion pins.
    """
    curvature = 1.7  # phi = curvature * y^2 / 2, so dphi/dx = 0 on every x-normal face
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    centroid, face_centroid = geometry.cell.centroid, geometry.face.centroid
    normal = geometry.face.normal

    field = 0.5 * curvature * centroid[:, 1] ** 2
    exact_hessian = jnp.tile(jnp.asarray([0.0, 0.0, curvature]), (mesh.n_cells, 1))
    displacement = face_centroid - centroid[face_cells.owner]
    tangential = displacement - scale(normal, dot(displacement, normal))
    # The zero-gradient patch: the walls whose normal is x, where this field's normal derivative is
    # exactly zero. Its face value is the owner's, carried along the tangential offset.
    follows_owner = (~face_cells.interior) & (jnp.abs(normal[:, 0]) > 0.5)
    prescribed = 0.5 * curvature * face_centroid[:, 1] ** 2

    def boundary_values_at(cell_gradient):
        follow = field[face_cells.owner] + dot(cell_gradient[face_cells.owner], tangential)
        return jnp.where(follows_owner, follow, prescribed)

    weight = jnp.where(follows_owner[:, None], tangential, 0.0)
    leading = boundary_values_at(jnp.zeros((mesh.n_cells, mesh.dim)))

    for closure in (OwnerGradient(), SkewCorrectedGradient()):
        scheme = MultipleCorrectionGradient(boundary_closure=closure, fallback=None).bind(
            mesh, geometry
        )
        _, hessian = scheme.reconstruct(
            field,
            mesh,
            geometry,
            leading,
            boundary_values_at=boundary_values_at,
            boundary_gradient_weight=weight,
        )
        error = float(jnp.max(jnp.abs(hessian - exact_hessian))) / curvature
        assert error < 3e-2, f"{type(closure).__name__}: {error:.3e}"

        # The control: prescribe every boundary value and the same field is exact, so the residue
        # above belongs to `M1 - B` and not to the quadratic reconstruction itself.
        _, exact = scheme.reconstruct(field, mesh, geometry, prescribed)
        assert float(jnp.max(jnp.abs(exact - exact_hessian))) < 1e-11 * curvature


def test_binding_against_a_boundary_condition_restores_quadratic_exactness() -> None:
    """Probe the corrections through the first pass the reconstruction will actually apply.

    Given ``boundary_gradient_weight`` the first pass inverts ``M1 - B`` rather than ``M1``. Probing
    the corrections on ``M1^-1`` and applying them to that leaves them correcting an operator nobody
    evaluates -- the mistake :func:`_build_corrections` is written to avoid -- and the scheme stops
    reproducing a quadratic: measured here at 2.0e-3 of the gradient, against 2e-15 when every
    boundary value is prescribed, so the residue is the condition's operator and not the
    reconstruction. Handing the same weight to ``bind`` recovers roundoff.

    Both halves of the repair are needed and this pins both. Probing the first pass through
    ``(M1 - B)^-1`` while still handing the closure the exact face value repairs
    :class:`OwnerGradient` and makes :class:`SkewCorrectedGradient` about five times *worse* (1.4e-1
    against 2.1e-3), because a value-reading closure is then itself corrected for an operator nobody
    evaluates. The closure has to get the condition re-evaluated at the probe's own first-pass
    gradient, which is what ``reconstruct`` gives it.
    """
    curvature = 1.7  # phi = curvature * y^2 / 2, so dphi/dx = 0 on every x-normal face
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    centroid, face_centroid = geometry.cell.centroid, geometry.face.centroid
    normal = geometry.face.normal

    field = 0.5 * curvature * centroid[:, 1] ** 2
    exact_gradient = jnp.stack([jnp.zeros(mesh.n_cells), curvature * centroid[:, 1]], axis=-1)
    displacement = face_centroid - centroid[face_cells.owner]
    tangential = displacement - scale(normal, dot(displacement, normal))
    follows_owner = (~face_cells.interior) & (jnp.abs(normal[:, 0]) > 0.5)
    prescribed = 0.5 * curvature * face_centroid[:, 1] ** 2

    def boundary_values_at(cell_gradient):
        follow = field[face_cells.owner] + dot(cell_gradient[face_cells.owner], tangential)
        return jnp.where(follows_owner, follow, prescribed)

    weight = jnp.where(follows_owner[:, None], tangential, 0.0)
    leading = boundary_values_at(jnp.zeros((mesh.n_cells, mesh.dim)))
    scale_of = float(jnp.max(jnp.linalg.norm(exact_gradient, axis=-1)))

    for closure in (OwnerGradient(), SkewCorrectedGradient()):
        unbound = MultipleCorrectionGradient(boundary_closure=closure, fallback=None)
        errors = {}
        for label, scheme in (
            ("blind", unbound.bind(mesh, geometry)),
            ("against the condition", unbound.bind(mesh, geometry, weight)),
        ):
            gradient, _ = scheme.reconstruct(
                field,
                mesh,
                geometry,
                leading,
                boundary_values_at=boundary_values_at,
                boundary_gradient_weight=weight,
            )
            errors[label] = float(jnp.max(jnp.linalg.norm(gradient - exact_gradient, axis=-1)))

        assert errors["against the condition"] < 1e-12 * scale_of, (
            f"{type(closure).__name__}: {errors['against the condition']:.3e}"
        )
        # And the blind binding is inexact by far more than roundoff, so the assertion above is
        # measuring the repair rather than a case that was never broken.
        assert errors["blind"] > 1e-6 * scale_of, f"{type(closure).__name__}: {errors['blind']:.3e}"


def _zero_gradient_quadratic_case():
    """A quadratic with a zero-gradient wall pair it satisfies exactly, and its assembler's inputs."""
    curvature = 1.7  # phi = curvature * y^2 / 2, so dphi/dx = 0 on every x-normal face
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    geometry = mesh.geometry()

    def quadratic(x):
        return 0.5 * curvature * x[..., 1] ** 2

    boundary = {
        "left": ZeroGradient(),
        "right": ZeroGradient(),
        "bottom": DirichletField(field_fn=quadratic),
        "top": DirichletField(field_fn=quadratic),
    }
    exact = jnp.stack([jnp.zeros(mesh.n_cells), curvature * geometry.cell.centroid[:, 1]], axis=-1)
    return mesh, geometry, quadratic, boundary, exact


def test_the_assembler_binds_the_scheme_against_its_boundary_conditions() -> None:
    """The end of the chain: a real assembler reproduces a quadratic on a gradient-type patch.

    ``ResidualAssembler`` hands the scheme a ``boundary_gradient_weight`` on every reconstruction, so
    its first pass inverts ``M1 - B``. Binding the scheme against the geometry alone leaves the
    corrections built for ``M1^-1``, and the assembler's gradient is then wrong by 2.0e-3 of the
    gradient on this case -- the wrong answer this catches, and what shipped until the weight reached
    ``bind``. Exactness here is not a property of the scheme alone; it is a property of the scheme
    being prepared against the conditions it will run under.
    """
    mesh, geometry, quadratic, boundary, exact = _zero_gradient_quadratic_case()
    field = quadratic(geometry.cell.centroid)
    scale_of = float(jnp.max(jnp.linalg.norm(exact, axis=-1)))

    for closure in (OwnerGradient(), SkewCorrectedGradient()):
        assembler = ResidualAssembler.build(
            mesh,
            geometry,
            PropertyModel({"diffusivity": Constant(1.0)}),
            (DiffusionFlux(),),
            BoundaryConditions(boundary),
            gradient_scheme=MultipleCorrectionGradient(boundary_closure=closure, fallback=None),
        )
        error = float(jnp.max(jnp.linalg.norm(assembler.gradient(field) - exact, axis=-1)))
        assert error < 1e-12 * scale_of, f"{type(closure).__name__}: {error:.3e}"


def test_the_boundary_gradient_weight_does_not_depend_on_the_state() -> None:
    """Why evaluating the weight once at build time is exact rather than an approximation.

    The weight is ``d(boundary value)/d(grad phi_owner)``, which for every shipped condition is a
    property of the condition and the geometry -- the tangential offset a face value carries, or zero
    where the value is prescribed. Nothing about it moves with the field or with the gradient it is
    evaluated at, so binding the scheme against it before any field exists cannot be stale.

    Catches a condition (or a future one) whose gradient dependence varies with the state, which
    would make that build-time binding silently wrong rather than merely approximate.
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.3, seed=5, named_boundaries=True)
    geometry = mesh.geometry()
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions(
            {
                "left": ZeroGradient(),
                "right": Neumann(flux=0.7),
                "bottom": Dirichlet(value=1.0),
                "top": Convective(h=2.0, t_inf=0.5),
            }
        ),
    )
    properties = assembler.properties.evaluate(mesh.cell_zones, {})
    key = jax.random.PRNGKey(3)
    reference = assembler._boundary_gradient_weight(
        jnp.zeros(mesh.n_cells), jnp.zeros((mesh.n_cells, mesh.dim)), properties
    )
    assert float(jnp.max(jnp.abs(reference))) > 1e-6  # not trivially zero everywhere

    for field in (
        jnp.zeros(mesh.n_cells),
        jax.random.normal(key, (mesh.n_cells,)),
        5.0 + jax.random.normal(key, (mesh.n_cells,)),
    ):
        for gradient in (
            jnp.zeros((mesh.n_cells, mesh.dim)),
            jax.random.normal(key, (mesh.n_cells, mesh.dim)),
        ):
            moved = assembler._boundary_gradient_weight(field, gradient, properties)
            assert np.array_equal(np.asarray(moved), np.asarray(reference))


def test_a_property_needing_a_field_asks_the_caller_for_the_weight() -> None:
    """A property that cannot be evaluated before a field exists cannot give a weight either.

    ``boundary_values`` falls back to a zero coefficient when the properties lack one, and a
    convective condition blends that coefficient into its face value -- so computing the weight
    against that fallback would hand ``bind`` a silently wrong operator, which is the defect binding
    against the conditions exists to remove. Refuse instead, and name the way out.
    """

    class NeedsAField(Property):
        """A property whose value is read from a field, as the interface allows."""

        def evaluate(self, cell_zones, fields):
            return fields["temperature"]

        def scaled(self, factor):
            return self

    mesh, geometry, _quadratic, boundary, _exact = _zero_gradient_quadratic_case()
    with pytest.raises(ValueError, match="boundary_gradient_weight"):
        ResidualAssembler.build(
            mesh,
            geometry,
            PropertyModel({"diffusivity": NeedsAField()}),
            (DiffusionFlux(),),
            BoundaryConditions(boundary),
            gradient_scheme=MultipleCorrectionGradient(fallback=None),
        )


def _flow_quadratic_case():
    """A flow assembler whose quadratic velocity and pressure satisfy its own closures exactly.

    Both fields have to satisfy their gradient-type patches **identically**, not merely at the wall:
    a zero-gradient closure builds its face value from the *owner* cell's gradient, so a quadratic
    whose normal derivative vanishes on the wall but not a cell behind it gives boundary data no
    reconstruction can reproduce (measured at 5.8e-1, unchanged by any binding). That is what fixes
    the geometry here. Pressure is prescribed on the two x-normal patches and zero-gradient on the
    y-normal walls, so it may vary only with ``x``; velocity is the reverse -- prescribed on the
    walls, extrapolated at both x-normal patches -- so it may vary only with ``y``. Each still
    carries genuine curvature, which is what a 1-exact reconstruction would miss.
    """
    pressure_of = lambda x: 1.7 * x**2 - 0.9 * x + 0.4  # noqa: E731
    profiles = ((1.3, -0.4, 0.2), (-0.7, 0.9, -0.3))  # u_i = a y^2 + b y + c

    def velocity_of(centroid):
        y = centroid[..., 1]
        return jnp.stack([a * y**2 + b * y + c for a, b, c in profiles], axis=-1)

    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=3, named_boundaries=True)
    geometry = mesh.geometry()
    inflow = geometry.face.centroid[mesh.face_patches.indices("left")]
    assembler = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(0.1), "density": Constant(1.0)}),
        BoundaryConditions(
            {
                "left": PressureOutlet(pressure=pressure_of(inflow[:, 0])),
                "right": PressureOutlet(pressure=pressure_of(1.0)),
                "bottom": MovingWall(velocity=velocity_of),
                "top": MovingWall(velocity=velocity_of),
            }
        ),
        gradient_scheme=MultipleCorrectionGradient(fallback=SkewCorrectedGradient()),
    )
    centroid = geometry.cell.centroid
    x, y = centroid[:, 0], centroid[:, 1]
    zero = jnp.zeros(mesh.n_cells)
    exact_pressure = jnp.stack([2 * 1.7 * x - 0.9, zero], axis=-1)
    exact_velocity = jnp.stack(
        [jnp.stack([zero, 2 * a * y + b], axis=-1) for a, b, _ in profiles], axis=1
    )
    return assembler, pressure_of(x), velocity_of(centroid), exact_pressure, exact_velocity


def _bound_blind(assembler: MomentumContinuity) -> MomentumContinuity:
    """``assembler`` with every field back on the geometry-only binding -- what shipped before."""
    return dataclasses.replace(
        assembler,
        velocity_gradient_schemes=(assembler.gradient_scheme,) * assembler.mesh.dim,
        pressure_gradient_scheme=assembler.gradient_scheme,
    )


def test_the_flow_assembler_binds_a_scheme_per_solved_field() -> None:
    """The coupled flow's four fields do not share one binding, and could not.

    ``MomentumContinuity`` reconstructs each velocity component and the pressure, and the patches
    treat them oppositely -- an outlet prescribes the pressure and leaves the velocity to
    extrapolate, a wall does the reverse. So each field's first pass inverts a *different*
    ``M1 - B``, and one binding cannot be right for all of them.

    Both halves are pinned: the per-field bindings reproduce these quadratics to roundoff, and the
    single geometry-only binding this replaced does not (4.4e-3 of the pressure gradient, 3.1e-3 of
    the velocity gradient on this case) -- the wrong answer the split catches.
    """
    assembler, pressure, velocity, exact_pressure, exact_velocity = _flow_quadratic_case()
    scale_of = float(jnp.max(jnp.abs(exact_velocity)))

    def errors(case):
        return (
            float(jnp.max(jnp.abs(case._pressure_gradient(pressure)[0] - exact_pressure))),
            float(jnp.max(jnp.abs(case._velocity_gradient(velocity)[0] - exact_velocity))),
        )

    per_field = errors(assembler)
    assert max(per_field) < 1e-12 * scale_of, f"per field: {per_field}"
    blind = errors(_bound_blind(assembler))
    assert min(blind) > 1e-4 * scale_of, f"geometry-only binding is already exact: {blind}"


def test_the_velocity_and_pressure_weights_differ_on_the_same_patch() -> None:
    """Why the split is needed at all, stated as the quantity the bindings are built from.

    Were the weights equal, ``dim + 1`` bindings would be ``dim + 1`` copies of one. They are not:
    on this case the velocity's weight is nonzero exactly on the two prescribed-pressure patches and
    the pressure's exactly on the two walls, so neither is a rescaling of the other.
    """
    assembler = _flow_quadratic_case()[0]
    velocity_weights = assembler._build_time_velocity_gradient_weights()
    pressure_weight = assembler._build_time_pressure_gradient_weight()
    assert len(velocity_weights) == assembler.mesh.dim

    live = {
        name: (
            float(jnp.max(jnp.abs(velocity_weights[0][faces]))),
            float(jnp.max(jnp.abs(pressure_weight[faces]))),
        )
        for name in ("left", "right", "bottom", "top")
        for faces in (assembler.mesh.face_patches.indices(name),)
    }
    for name in ("left", "right"):
        assert live[name][0] > 1e-6 and live[name][1] == 0.0, f"{name}: {live[name]}"
    for name in ("bottom", "top"):
        assert live[name][0] == 0.0 and live[name][1] > 1e-6, f"{name}: {live[name]}"


def test_the_flow_boundary_gradient_weights_do_not_depend_on_the_state() -> None:
    """Why evaluating the flow's weights once at build time is exact -- the twin of the scalar check.

    Every flow closure is affine in the gradient it is handed: a prescribed value ignores it, an
    extrapolating one adds the tangential offset ``grad . d_t``. So the derivative at rest is the
    derivative everywhere, and binding before any state exists cannot be stale. Catches a closure
    (or a future one) whose gradient dependence moves with the flow, which would make that
    build-time binding silently wrong rather than merely approximate.
    """
    assembler = _flow_quadratic_case()[0]
    mesh = assembler.mesh
    key = jax.random.PRNGKey(7)
    reference_velocity = assembler._build_time_velocity_gradient_weights()
    reference_pressure = assembler._build_time_pressure_gradient_weight()
    assert float(jnp.max(jnp.abs(reference_pressure))) > 1e-6  # not trivially zero everywhere
    assert float(jnp.max(jnp.abs(reference_velocity[0]))) > 1e-6

    for scale_of in (1.0, 5.0):
        velocity = scale_of * jax.random.normal(key, (mesh.n_cells, mesh.dim))
        pressure = scale_of * jax.random.normal(key, (mesh.n_cells,))
        grad_velocity = scale_of * jax.random.normal(key, (mesh.n_cells, mesh.dim, mesh.dim))
        grad_pressure = scale_of * jax.random.normal(key, (mesh.n_cells, mesh.dim))
        for component in range(mesh.dim):
            moved = assembler._velocity_boundary_gradient_weight(velocity, component, grad_velocity)
            assert np.array_equal(np.asarray(moved), np.asarray(reference_velocity[component]))
        moved = assembler._pressure_boundary_gradient_weight(pressure, grad_pressure)
        assert np.array_equal(np.asarray(moved), np.asarray(reference_pressure))


def _zero_gradient_tetrahedra():
    """The tetrahedral fixture with its whole boundary zero-gradient, and that condition's weight.

    18 of its 162 cells have two boundary faces, and those are the ones this condition leaves
    undetermined under either closure -- see the test below.
    """
    mesh = QUADRATIC_MESHES[2]
    geometry = mesh.geometry()
    weight = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({"boundary": ZeroGradient()}),
    )._build_time_boundary_gradient_weight()
    return mesh, geometry, weight


def _boundary_face_count(mesh) -> np.ndarray:
    """How many boundary faces each cell owns, shape ``(n_cells,)``."""
    face_cells = mesh.face_cells
    return np.bincount(
        np.asarray(face_cells.owner)[~np.asarray(face_cells.interior)], minlength=mesh.n_cells
    )


def _worst_correction(corrections):
    return np.max(np.abs(np.asarray(corrections.m2_inverse)), axis=(1, 2))


def test_a_non_finite_correction_counts_as_undetermined_at_any_mesh_size() -> None:
    """A singular inverse comes back NaN under some linear-algebra libraries, and must be repaired.

    Reducing to ``max|M2^-1|`` before comparing loses it: the CPU backend's max-reduction returns a
    NaN on a short array but ``-inf`` on a long one, which then passes as determined. The sizes
    bracket that switch, so the obvious reduce-then-compare version fails the long one.
    """
    for n_cells in (4, 1000):
        inverse = jnp.ones((n_cells, 6, 6))
        inverse = inverse.at[1].set(jnp.nan).at[2, 0, 3].set(jnp.nan).at[3, 5, 5].set(1e16)
        cells = multiple_correction._undetermined_cells(inverse)
        np.testing.assert_array_equal(np.asarray(cells), [1, 2, 3], err_msg=f"{n_cells} cells")


def test_cells_a_condition_leaves_undetermined_keep_their_geometry_only_correction() -> None:
    """The third repair tier: where no closure can determine a cell under the conditions.

    A zero-gradient face's value is the owner's own value carried along the tangential offset, so it
    tells a value-reading fallback nothing the owner's gradient did not. On a tetrahedron with two
    such faces neither closure then determines the Hessian, and binding against the condition left
    ``max|M2^-1|`` near 1e16 there -- on a real tetrahedral duct, 94 pressure cells, and a laminar
    march whose every linear solve then ran to its cycle cap without the residual moving.

    Pinned three ways, each of which a plausible wrong repair fails: no cell is left singular; the
    cells the condition leaves undetermined carry exactly the geometry-only correction (not some
    other well-conditioned matrix); and every other cell carries exactly the condition-aware one, so
    the repair is local and exactness is kept wherever the condition allows it.
    """
    mesh, geometry, weight = _zero_gradient_tetrahedra()
    closure = CellwiseFallback(
        multiple_correction._undetermined_cells(
            multiple_correction._probe_corrections(
                mesh, geometry, OwnerGradient(), weight
            ).m2_inverse
        ),
        OwnerGradient(),
        SkewCorrectedGradient(),
    )
    conditioned = multiple_correction._probe_corrections(mesh, geometry, closure, weight)
    geometric = multiple_correction._probe_corrections(mesh, geometry, closure, None)
    # Which cells are stuck is decided by the mesh, not by thresholding the singular inverse: there
    # ``max|M2^-1|`` is rounding noise that is ~1e16 under one BLAS and NaN under another, and a NaN
    # fails ``> limit`` -- which dropped two of the 18 on some CI runners and not others.
    stuck = _boundary_face_count(mesh) >= 2
    assert stuck.sum() == 18  # the fixture still exhibits the defect this repairs
    flagged = multiple_correction._undetermined_cells(conditioned.m2_inverse)
    assert flagged is not None
    np.testing.assert_array_equal(np.asarray(flagged), np.flatnonzero(stuck))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bound = MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(
            mesh, geometry, weight
        )
    assert _worst_correction(bound.prepared).max() < multiple_correction._UNDETERMINED_CORRECTION
    for name in ("m2_inverse", "gradient_defect"):
        held = np.asarray(getattr(bound.prepared, name))
        np.testing.assert_array_equal(held[stuck], np.asarray(getattr(geometric, name))[stuck])
        np.testing.assert_array_equal(held[~stuck], np.asarray(getattr(conditioned, name))[~stuck])


def test_a_repair_report_does_not_silence_a_later_graver_one() -> None:
    """Each repair warning is emitted once per process, and independently of the others.

    A scheme is bound once per field, and on a tetrahedral mesh the first binding reports a repair.
    When all three warnings shared one flag, that report swallowed the later one saying the
    correction had been left singular -- which is how a stalled march on a real duct printed nothing.
    """
    mesh, geometry, weight = _zero_gradient_tetrahedra()
    multiple_correction._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(mesh, geometry)
        MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(mesh, geometry, weight)
        MultipleCorrectionGradient(fallback=None).bind(mesh, geometry)
    messages = [str(w.message) for w in caught]
    assert sum("is used on those cells' boundary faces" in m for m in messages) == 1
    assert sum("once the boundary conditions are accounted for" in m for m in messages) == 1
    assert sum("no fallback closure was given" in m for m in messages) == 1


def test_a_differentiating_closure_gets_boundary_values_at_its_own_gradient() -> None:
    """The seam that makes a one-sided closure legal on a gradient-type patch.

    A residual assembler evaluates its boundary closures at *zero* gradient, so a gradient-type
    condition -- whose whole content is a correction -- comes back equal to the owner cell's value.
    A closure that then differences it subtracts a correction nothing added, and divides the residue
    by the wall-normal distance. Re-evaluating the closures at the reconstruction's own gradient
    returns the correction, and the difference collapses to the zero normal derivative the condition
    asserts.

    Read what the SCHEME handed the closure, through a recording closure, rather than what the test
    can recompute for itself: an earlier version of this test evaluated ``boundary_values_at`` on the
    returned gradient and measured the rise against that same gradient, which is zero by
    construction whatever ``reconstruct`` did with the argument. It passed against a ``reconstruct``
    that ignored ``boundary_values_at`` outright -- the one wrong answer it names.

    The Dirichlet arm is the control: a prescribed value does not depend on the gradient, so it must
    reach the closure unchanged. A fix that merely re-evaluated everything would flatten that one.
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.30, seed=11)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    boundary = np.where(~np.asarray(face_cells.interior))[0]
    field = jax.random.normal(jax.random.PRNGKey(12), (mesh.n_cells,))
    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    normal = geometry.face.normal

    seen: list[np.ndarray] = []

    class RecordingClosure(multiple_correction.GradientBoundaryClosure):
        """Records the boundary values it is handed, then defers to the closure under test."""

        reads_boundary_values = True

        def face_gradient(self, gradient, field, boundary_values, face_cells, geometry):
            seen.append(np.asarray(boundary_values))
            return SkewCorrectedGradient().face_gradient(
                gradient, field, boundary_values, face_cells, geometry
            )

    def zero_gradient(cell_gradient):
        """A zero-gradient patch: the owner value plus the correction the cell gradient supplies."""
        return field[face_cells.owner] + non_orthogonal_correction(
            cell_gradient[face_cells.owner], displacement, normal
        )

    scheme = MultipleCorrectionGradient(boundary_closure=RecordingClosure()).bind(mesh, geometry)
    leading = zero_gradient(jnp.zeros((mesh.n_cells, mesh.dim)))

    def values_reaching_the_closure(boundary_values, boundary_values_at):
        seen.clear()  # drop the probe-time calls `bind` already made
        gradient = scheme.reconstruct(
            field, mesh, geometry, boundary_values, boundary_values_at=boundary_values_at
        )[0]
        assert seen, "the closure was never called"
        return seen[-1], gradient

    given, reconstructed = values_reaching_the_closure(leading, zero_gradient)
    withheld, _ = values_reaching_the_closure(leading, None)

    # Withheld, the closure gets exactly what the caller passed. Given, it gets the condition
    # evaluated at the scheme's own gradient -- so it lands far closer to that than to the
    # zero-gradient values it was handed. (Not identical to either: the closure is called with the
    # FIRST pass's gradient, which the second-order correction then moves.)
    assert np.array_equal(withheld, np.asarray(leading))
    at_own_gradient = np.abs(given - np.asarray(zero_gradient(reconstructed)))[boundary].max()
    at_zero_gradient = np.abs(given - np.asarray(leading))[boundary].max()
    assert at_own_gradient < 0.2 * at_zero_gradient, f"{at_own_gradient:.3e} {at_zero_gradient:.3e}"

    # The control: a prescribed value does not depend on the gradient, so it must arrive unchanged
    # whether or not the scheme re-evaluates it.
    prescribed = jax.random.normal(jax.random.PRNGKey(13), (mesh.n_faces,))
    unchanged, _ = values_reaching_the_closure(prescribed, lambda _g: prescribed)
    assert np.array_equal(unchanged, np.asarray(prescribed))


@pytest.mark.parametrize(
    "label, mesh, closure, warns",
    [
        ("quadrilateral-owner", QUADRATIC_MESHES[0], OwnerGradient(), False),
        ("hexahedral-owner", QUADRATIC_MESHES[1], OwnerGradient(), False),
        ("tetrahedral-owner", QUADRATIC_MESHES[2], OwnerGradient(), True),
        ("tetrahedral-skew", QUADRATIC_MESHES[2], SkewCorrectedGradient(), False),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_it_warns_when_the_mesh_and_closure_leave_the_hessian_underdetermined(
    label, mesh, closure, warns
) -> None:
    """A silently-singular correction is the failure this scheme can produce without saying so.

    The reconstruction stops being exact for quadratics and starts amplifying, which looks like a
    solver problem rather than a closure one. All four combinations are checked together because
    only the pair discriminates: a detector that fired on every tetrahedral mesh, or on every owner
    closure, would be useless -- it is the combination that is broken.
    """
    multiple_correction._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultipleCorrectionGradient(boundary_closure=closure, fallback=None).bind(
            mesh, mesh.geometry()
        )
    fired = [w for w in caught if "underdetermined" in str(w.message)]
    assert bool(fired) is warns, label
    if warns:
        assert "no fallback closure was given" in str(fired[0].message)


def test_the_underdetermined_warning_is_emitted_once_per_process() -> None:
    """It reports a fixed property of the geometry, and a scheme is bound on every assembler."""
    mesh = QUADRATIC_MESHES[2]
    multiple_correction._WARNED.clear()
    scheme = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None)
    counts = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scheme.bind(mesh, mesh.geometry())
        counts.append(len([w for w in caught if "underdetermined" in str(w.message)]))
    assert counts == [1, 0]


def test_the_owner_closure_fails_only_where_a_cell_has_two_boundary_faces() -> None:
    """Which cells the owner closure cannot determine, resolved rather than asserted in the aggregate.

    "It fails on boundary tetrahedra" is too coarse, and predicts failure on meshes where there is
    none: a tetrahedron with ONE boundary face still has three informative faces and reconstructs a
    quadratic to roundoff. It is the corner and edge tets -- two or more boundary faces, hence two
    informative faces against six Hessian components -- that go singular. That distinction is what
    makes a real snappyHexMesh mesh usable under the default closure, so it is pinned here rather
    than left as a remark.
    """
    mesh = tetrahedral_grid_3d(3, perturb=0.25, seed=6)
    geometry = mesh.geometry()
    boundary_faces = _boundary_face_count(mesh)
    case = _quadratic(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheme = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None).bind(
            mesh, geometry
        )

    gradient = np.asarray(
        scheme.gradients(case["cell_values"], mesh, geometry, case["face_values"])
    )
    magnitude = np.linalg.norm(case["gradient"], axis=-1)
    relative = np.linalg.norm(gradient - case["gradient"], axis=-1) / magnitude

    determined = boundary_faces < 2
    assert (
        determined.sum() and (~determined).sum()
    )  # the mesh has both kinds, or this proves nothing
    assert np.all(np.isfinite(relative[determined])) and relative[determined].max() < 1e-12
    assert _is_not_exact(relative[~determined].min())


def test_the_repair_is_exact_where_it_fires_and_absent_where_it_need_not() -> None:
    """The point of repairing per cell rather than choosing a closure for the whole mesh.

    The two shipped closures fail in opposite regimes, so choosing one globally means accepting one
    of the two failures everywhere. Choosing per cell -- by measuring the correction, not by counting
    faces -- gives the accurate one where it is needed and leaves the rest of the mesh on a closure
    that reads no boundary value at all. `fallback` defaults to `None`, so the repair has to be asked
    for explicitly; the second assertion is the load-bearing one: on a mesh with nothing to repair,
    asking for it anyway must be *bit-identical* to not asking, or the repair is a change to every
    case rather than to the cases that need it.
    """
    tetrahedral, hexahedral = QUADRATIC_MESHES[2], QUADRATIC_MESHES[1]

    case = _quadratic(tetrahedral)
    geometry = case["geometry"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repaired = MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(
            tetrahedral, geometry
        )
        raw = MultipleCorrectionGradient().bind(tetrahedral, geometry)  # fallback=None, the default
    assert isinstance(repaired.prepared.closure, CellwiseFallback)
    args = (case["cell_values"], tetrahedral, geometry, case["face_values"])
    error = np.abs(np.asarray(repaired.gradients(*args)) - case["gradient"]).max()
    assert error < 1e-12 * np.abs(case["gradient"]).max()
    assert _is_not_exact(np.abs(np.asarray(raw.gradients(*args)) - case["gradient"]).max())

    # Nothing to repair: the repair must not have happened, and must cost the answer nothing.
    case = _quadratic(hexahedral)
    geometry = case["geometry"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        requested = MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(
            hexahedral, geometry
        )
        default = MultipleCorrectionGradient().bind(hexahedral, geometry)
    assert not isinstance(requested.prepared.closure, CellwiseFallback)
    args = (case["cell_values"], hexahedral, geometry, case["face_values"])
    assert np.array_equal(
        np.asarray(requested.gradients(*args)), np.asarray(default.gradients(*args))
    )


def test_the_repair_says_so_rather_than_silently_changing_the_closure() -> None:
    """Asking for the repair and getting a different closure on some cells has to be visible."""
    multiple_correction._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultipleCorrectionGradient(fallback=SkewCorrectedGradient()).bind(
            QUADRATIC_MESHES[2], QUADRATIC_MESHES[2].geometry()
        )
    fired = [w for w in caught if "underdetermined" in str(w.message)]
    assert len(fired) == 1
    message = str(fired[0].message)
    assert "OwnerGradient" in message and "SkewCorrectedGradient" in message
    assert "18 of 162" in message  # how many cells, so the reader can judge the scale


def test_the_default_leaves_undetermined_cells_unrepaired_and_names_the_opt_in() -> None:
    """`fallback` defaults to `None`: nothing is silently installed on the cells that need it.

    The closure that repairs those cells is measured (see `SkewCorrectedGradient`'s own docstring)
    to destabilize a coupled march at an unconverged iterate on exactly the cells it would be
    installed on, so it must be requested by a caller who has weighed that, not defaulted onto every
    mesh with a corner tetrahedron. The warning is what makes the unrepaired state visible and names
    the opt-in, rather than requiring a reader to already know it exists.
    """
    multiple_correction._WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheme = MultipleCorrectionGradient().bind(  # fallback=None, the default
            QUADRATIC_MESHES[2], QUADRATIC_MESHES[2].geometry()
        )
    assert not isinstance(scheme.prepared.closure, CellwiseFallback)
    fired = [w for w in caught if "underdetermined" in str(w.message)]
    assert len(fired) == 1
    message = str(fired[0].message)
    assert "fallback=SkewCorrectedGradient()" in message
    assert "18 of 162" in message
