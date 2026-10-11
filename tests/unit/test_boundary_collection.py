"""Unit tests for the named boundary-condition collection (mesh-free, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.mesh import FaceCellConnectivity, FacePatches
from aquaflux.mesh.groups import PADDING_PATCH


def _topology() -> tuple[FaceCellConnectivity, FacePatches]:
    """A 5-face, 2-cell strip: face 0 interior (0-1), faces 1-4 boundary.

    Patches: ``left`` = {1}, ``right`` = {2, 3}; face 4 is a boundary face no named patch claims,
    so it sits in the automatic ``boundary`` patch.
    """
    owner = jnp.array([0, 0, 1, 1, 0])
    neighbour = jnp.array([1, -1, -1, -1, -1])
    face_cells = FaceCellConnectivity(owner=owner, neighbour=neighbour, n_cells=2)
    patches = FacePatches.from_dict(neighbour, {"left": [1], "right": [2, 3]})
    return face_cells, patches


def _resolved(conditions):
    face_cells, patches = _topology()
    return face_cells, BoundaryConditions(conditions).resolve(patches, face_cells)


def test_constructed_collection_is_unbound() -> None:
    """Constructing from a dict keeps the closures but binds no faces until resolve()."""
    bcs = BoundaryConditions({"left": 10.0, "right": 20.0})
    assert bcs.conditions == {"left": 10.0, "right": 20.0}
    assert bcs.faces is None


def test_resolve_binds_each_patch_to_its_face_indices() -> None:
    _, bcs = _resolved({"left": 10.0, "right": 20.0, "boundary": 0.0})
    np.testing.assert_array_equal(np.asarray(bcs.faces["left"]), [1])
    np.testing.assert_array_equal(np.asarray(bcs.faces["right"]), [2, 3])
    np.testing.assert_array_equal(np.asarray(bcs.faces["boundary"]), [4])


def test_resolve_rejects_unknown_patch_name() -> None:
    with pytest.raises(ValueError, match="no group named"):
        _resolved({"plasma": 1.0})


def test_resolve_names_every_uncovered_patch_not_just_the_first() -> None:
    """An omitted patch would otherwise keep a zero face value, so all of them are reported at once."""
    with pytest.raises(ValueError) as excinfo:
        _resolved({"left": 1.0})
    message = str(excinfo.value)
    assert "'right' (2 faces)" in message
    assert "'boundary' (1 face)" in message
    assert "'left'" not in message


def test_the_automatic_boundary_patch_is_reported_and_naming_it_covers_it() -> None:
    """Boundary faces no named patch claims are the most common omission, so they are not exempt."""
    with pytest.raises(ValueError, match=r"'boundary' \(1 face\)"):
        _resolved({"left": 1.0, "right": 2.0})
    _resolved({"left": 1.0, "right": 2.0, "boundary": 3.0})  # no raise


def test_interior_faces_are_never_reported_and_may_be_named() -> None:
    """Interior faces need no closure; naming the ``interior`` patch anyway stays legal."""
    with pytest.raises(ValueError) as excinfo:
        _resolved({"left": 1.0, "right": 2.0})
    assert "interior" not in str(excinfo.value)
    _resolved({"interior": 0.0, "left": 1.0, "right": 2.0, "boundary": 3.0})  # no raise


def test_padding_faces_are_never_reported() -> None:
    """A distributed partition's padding faces have no neighbour but carry no physics."""
    face_cells, _ = _topology()
    names = ("interior", "boundary", "left", "right", PADDING_PATCH)
    patches = FacePatches(label=jnp.array([0, 2, 3, 3, 4]), names=names)
    BoundaryConditions({"left": 1.0, "right": 2.0}).resolve(patches, face_cells)  # no raise


def test_apply_before_resolve_raises() -> None:
    face_cells, _ = _topology()
    bcs = BoundaryConditions({"left": 1.0})
    with pytest.raises(ValueError, match="resolve"):
        bcs.apply(face_cells, jnp.zeros(5), lambda bc, faces, owner: bc)


def test_apply_sets_patch_rows_and_leaves_interior_faces_at_init() -> None:
    """Patch faces get their closure's value; the interior face keeps init."""
    face_cells, bcs = _resolved({"left": 10.0, "right": 20.0, "boundary": 30.0})
    out = bcs.apply(
        face_cells,
        -jnp.ones(5),
        lambda bc, faces, owner: bc * jnp.ones(faces.shape[0]),
    )
    np.testing.assert_allclose(np.asarray(out), [-1.0, 10.0, 20.0, 20.0, 30.0])


def test_apply_gathers_each_patch_owner_cells() -> None:
    """The closure receives the patch's owner-cell indices (used to gather a cell field)."""
    face_cells, bcs = _resolved({"left": None, "right": None, "boundary": None})
    phi = jnp.array([3.0, 7.0])  # per-cell field
    out = bcs.apply(face_cells, jnp.zeros(5), lambda bc, faces, owner: phi[owner])
    # left face 1 owner=0 -> 3; right faces 2,3 owner=1 -> 7; boundary face 4 owner=0 -> 3
    np.testing.assert_allclose(np.asarray(out), [0.0, 3.0, 7.0, 7.0, 3.0])


def test_apply_supports_vector_valued_init() -> None:
    """The fold works for a rank-2 per-face array (the flow path's velocity closure)."""
    face_cells, bcs = _resolved({"left": 1.0, "right": 2.0, "boundary": 3.0})
    out = bcs.apply(
        face_cells,
        jnp.zeros((5, 2)),
        lambda bc, faces, owner: bc * jnp.ones((faces.shape[0], 2)),
    )
    expected = np.zeros((5, 2))
    expected[1] = 1.0
    expected[[2, 3]] = 2.0
    expected[4] = 3.0
    np.testing.assert_allclose(np.asarray(out), expected)


def test_boundary_parameter_is_a_differentiable_leaf() -> None:
    """A closure value flows as a pytree leaf, so gradients pass through resolve + apply."""
    face_cells, patches = _topology()

    def total(k):
        bcs = BoundaryConditions({"left": k, "right": 2.0, "boundary": 0.0}).resolve(
            patches, face_cells
        )
        out = bcs.apply(
            face_cells,
            jnp.zeros(5),
            lambda bc, faces, owner: bc * jnp.ones(faces.shape[0]),
        )
        return jnp.sum(out)

    assert float(jax.grad(total)(3.0)) == 1.0  # 'left' has a single face


def _interleaved() -> tuple[FaceCellConnectivity, BoundaryConditions]:
    """A 6-face, 2-cell strip whose patches interleave: ``a`` = {1, 4}, ``b`` = {2, 5}, ``c`` = {3}.

    Patches of unequal size whose faces are not contiguous, so a value written against another
    patch's face, or against the wrong face within its own patch, lands on a row that shows it.
    """
    owner = jnp.array([0, 0, 1, 0, 1, 1])
    neighbour = jnp.array([1, -1, -1, -1, -1, -1])
    face_cells = FaceCellConnectivity(owner=owner, neighbour=neighbour, n_cells=2)
    patches = FacePatches.from_dict(neighbour, {"a": [1, 4], "b": [2, 5], "c": [3]})
    bcs = BoundaryConditions({"a": 100.0, "b": 200.0, "c": 300.0}).resolve(patches, face_cells)
    return face_cells, bcs


def test_apply_writes_each_value_at_its_own_face_when_patches_interleave() -> None:
    """Every face gets its own patch's value for itself -- ``bc + face index`` -- and nothing else's."""
    face_cells, bcs = _interleaved()
    out = bcs.apply(face_cells, -jnp.ones(6), lambda bc, faces, owner: bc + faces)
    np.testing.assert_allclose(np.asarray(out), [-1.0, 101.0, 202.0, 303.0, 104.0, 205.0])


def test_apply_broadcasts_a_closure_value_over_its_patch() -> None:
    """A closure may return one value for its whole patch, scalar or per-component."""
    face_cells, bcs = _interleaved()
    out = bcs.apply(face_cells, jnp.zeros(6), lambda bc, faces, owner: bc)
    np.testing.assert_allclose(np.asarray(out), [0.0, 100.0, 200.0, 300.0, 100.0, 200.0])
    out = bcs.apply(face_cells, jnp.zeros((6, 2)), lambda bc, faces, owner: jnp.array([bc, -bc]))
    expected = np.array(
        [[0.0, 0.0], [100, -100], [200, -200], [300, -300], [100, -100], [200, -200]]
    )
    np.testing.assert_allclose(np.asarray(out), expected)


def test_apply_with_no_patches_returns_init() -> None:
    """A mesh with no boundary faces needs no closure, and the fold leaves ``init`` as it is."""
    neighbour = jnp.array([1])
    face_cells = FaceCellConnectivity(owner=jnp.array([0]), neighbour=neighbour, n_cells=2)
    bcs = BoundaryConditions({}).resolve(FacePatches.from_dict(neighbour, {}), face_cells)
    init = jnp.array([7.0])
    np.testing.assert_array_equal(np.asarray(bcs.apply(face_cells, init, None)), [7.0])


def test_apply_writes_every_patch_in_one_scatter() -> None:
    """The fold is one scatter whatever the patch count, not one per patch.

    One per patch would chain the writes -- each reading the previous one's result -- so they could
    not overlap, and the traced program would grow with every patch.
    """
    face_cells, bcs = _interleaved()
    jaxpr = jax.make_jaxpr(
        lambda init: bcs.apply(face_cells, init, lambda bc, faces, owner: bc + faces)
    )(jnp.zeros(6))
    assert str(jaxpr).count("= scatter[") == 1
