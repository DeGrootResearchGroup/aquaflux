"""The multiple-correction reconstruction: a gradient and Hessian with no system to solve."""

from __future__ import annotations

import warnings

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.schemes import (
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
from aquaflux.vectors import dot

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


def test_a_differentiating_closure_gets_boundary_values_at_its_own_gradient() -> None:
    """The seam that makes a one-sided closure legal on a gradient-type patch.

    A residual assembler evaluates its boundary closures at *zero* gradient, so a gradient-type
    condition -- whose whole content is a correction -- comes back equal to the owner cell's value.
    A closure that then differences it subtracts a correction nothing added, and divides the residue
    by the wall-normal distance. Re-evaluating the closures at the reconstruction's own gradient
    returns the correction, and the difference collapses to the zero normal derivative the condition
    asserts.

    The Dirichlet arm is the control: a prescribed value does not depend on the gradient, so it must
    come back unchanged. A fix that merely suppressed the term would flatten that one too.
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.30, seed=11)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    scheme = MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient()).bind(
        mesh, geometry
    )
    owner = np.asarray(face_cells.owner)
    boundary = np.where(~np.asarray(face_cells.interior))[0]
    field = jax.random.normal(jax.random.PRNGKey(12), (mesh.n_cells,))

    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    normal = geometry.face.normal
    along = np.asarray(dot(displacement, normal))

    def zero_gradient(cell_gradient):
        """A zero-gradient patch: the owner value plus the correction the cell gradient supplies."""
        return field[face_cells.owner] + non_orthogonal_correction(
            cell_gradient[face_cells.owner], displacement, normal
        )

    def normal_derivative(boundary_values, boundary_values_at):
        gradient = scheme.reconstruct(
            field, mesh, geometry, boundary_values, boundary_values_at=boundary_values_at
        )[0]
        values = boundary_values if boundary_values_at is None else boundary_values_at(gradient)
        rise = (
            np.asarray(values)
            - np.asarray(field)[owner]
            - np.asarray(
                non_orthogonal_correction(gradient[face_cells.owner], displacement, normal)
            )
        )
        return np.abs(rise[boundary] / along[boundary]).max()

    leading = zero_gradient(jnp.zeros((mesh.n_cells, mesh.dim)))
    uncorrected = normal_derivative(leading, None)
    corrected = normal_derivative(leading, zero_gradient)
    assert corrected < 1e-8 * max(uncorrected, 1.0)  # collapses to roundoff
    assert uncorrected > 1.0  # and was not small to begin with

    # The control: a prescribed value does not depend on the gradient, so re-evaluating changes
    # nothing, and the closure's difference against it must survive untouched.
    prescribed = jax.random.normal(jax.random.PRNGKey(13), (mesh.n_faces,))
    assert np.isclose(
        normal_derivative(prescribed, None),
        normal_derivative(prescribed, lambda _g: prescribed),
        rtol=1e-12,
    )


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
    multiple_correction._CORRECTION_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultipleCorrectionGradient(boundary_closure=closure).bind(mesh, mesh.geometry())
    fired = [w for w in caught if "Hessian correction is singular" in str(w.message)]
    assert bool(fired) is warns, label
    if warns:
        assert "SkewCorrectedGradient" in str(fired[0].message)  # names the way out


def test_the_underdetermined_warning_is_emitted_once_per_process() -> None:
    """It reports a fixed property of the geometry, and a scheme is bound on every assembler."""
    mesh = QUADRATIC_MESHES[2]
    multiple_correction._CORRECTION_WARNED = False
    scheme = MultipleCorrectionGradient(boundary_closure=OwnerGradient())
    counts = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scheme.bind(mesh, mesh.geometry())
        counts.append(len([w for w in caught if "singular" in str(w.message)]))
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
    face_cells = mesh.face_cells
    boundary_faces = np.bincount(
        np.asarray(face_cells.owner)[~np.asarray(face_cells.interior)], minlength=mesh.n_cells
    )
    case = _quadratic(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheme = MultipleCorrectionGradient(boundary_closure=OwnerGradient()).bind(mesh, geometry)

    gradient = np.asarray(
        scheme.gradients(case["cell_values"], mesh, geometry, case["face_values"])
    )
    magnitude = np.linalg.norm(case["gradient"], axis=-1)
    relative = np.linalg.norm(gradient - case["gradient"], axis=-1) / magnitude

    determined = boundary_faces < 2
    assert (
        determined.sum() and (~determined).sum()
    )  # the mesh has both kinds, or this proves nothing
    assert relative[determined].max() < 1e-12
    assert relative[~determined].min() > 1e-3
