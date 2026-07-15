"""Unit tests for the named boundary-condition collection (mesh-free, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.mesh import FaceCellConnectivity, FacePatches


def _topology() -> tuple[FaceCellConnectivity, FacePatches]:
    """A 5-face, 2-cell strip: face 0 interior (0-1), faces 1-4 boundary.

    Patches: ``left`` = {1}, ``right`` = {2, 3}; face 4 is an unlisted boundary face.
    """
    owner = jnp.array([0, 0, 1, 1, 0])
    neighbour = jnp.array([1, -1, -1, -1, -1])
    face_cells = FaceCellConnectivity(owner=owner, neighbour=neighbour, n_cells=2)
    patches = FacePatches.from_dict(neighbour, {"left": [1], "right": [2, 3]})
    return face_cells, patches


def test_constructed_collection_is_unbound() -> None:
    """Constructing from a dict keeps the closures but binds no faces until resolve()."""
    bcs = BoundaryConditions({"left": 10.0, "right": 20.0})
    assert bcs.conditions == {"left": 10.0, "right": 20.0}
    assert bcs.faces is None


def test_resolve_binds_each_patch_to_its_face_indices() -> None:
    _, patches = _topology()
    bcs = BoundaryConditions({"left": 10.0, "right": 20.0}).resolve(patches)
    np.testing.assert_array_equal(np.asarray(bcs.faces["left"]), [1])
    np.testing.assert_array_equal(np.asarray(bcs.faces["right"]), [2, 3])


def test_resolve_rejects_unknown_patch_name() -> None:
    _, patches = _topology()
    with pytest.raises(ValueError, match="no group named"):
        BoundaryConditions({"plasma": 1.0}).resolve(patches)


def test_apply_before_resolve_raises() -> None:
    face_cells, _ = _topology()
    bcs = BoundaryConditions({"left": 1.0})
    with pytest.raises(ValueError, match="resolve"):
        bcs.apply(face_cells, jnp.zeros(5), lambda bc, faces, owner: bc)


def test_apply_sets_patch_rows_and_leaves_others_at_init() -> None:
    """Named-patch faces get the closure value; interior and unlisted-boundary faces keep init."""
    face_cells, patches = _topology()
    bcs = BoundaryConditions({"left": 10.0, "right": 20.0}).resolve(patches)
    out = bcs.apply(
        face_cells,
        jnp.zeros(5),
        lambda bc, faces, owner: bc * jnp.ones(faces.shape[0]),
    )
    np.testing.assert_allclose(np.asarray(out), [0.0, 10.0, 20.0, 20.0, 0.0])


def test_apply_gathers_each_patch_owner_cells() -> None:
    """The closure receives the patch's owner-cell indices (used to gather a cell field)."""
    face_cells, patches = _topology()
    bcs = BoundaryConditions({"left": None, "right": None}).resolve(patches)
    phi = jnp.array([3.0, 7.0])  # per-cell field
    out = bcs.apply(face_cells, jnp.zeros(5), lambda bc, faces, owner: phi[owner])
    # left face 1 owner=0 -> 3; right faces 2,3 owner=1 -> 7
    np.testing.assert_allclose(np.asarray(out), [0.0, 3.0, 7.0, 7.0, 0.0])


def test_apply_supports_vector_valued_init() -> None:
    """The fold works for a rank-2 per-face array (the flow path's velocity closure)."""
    face_cells, patches = _topology()
    bcs = BoundaryConditions({"left": 1.0, "right": 2.0}).resolve(patches)
    out = bcs.apply(
        face_cells,
        jnp.zeros((5, 2)),
        lambda bc, faces, owner: bc * jnp.ones((faces.shape[0], 2)),
    )
    expected = np.zeros((5, 2))
    expected[1] = 1.0
    expected[[2, 3]] = 2.0
    np.testing.assert_allclose(np.asarray(out), expected)


def test_boundary_parameter_is_a_differentiable_leaf() -> None:
    """A closure value flows as a pytree leaf, so gradients pass through resolve + apply."""
    face_cells, patches = _topology()

    def total(k):
        bcs = BoundaryConditions({"left": k, "right": 2.0}).resolve(patches)
        out = bcs.apply(
            face_cells,
            jnp.zeros(5),
            lambda bc, faces, owner: bc * jnp.ones(faces.shape[0]),
        )
        return jnp.sum(out)

    assert float(jax.grad(total)(3.0)) == 1.0  # 'left' has a single face


def test_empty_collection_leaves_init_untouched() -> None:
    face_cells, patches = _topology()
    bcs = BoundaryConditions({}).resolve(patches)
    init = jnp.arange(5.0)
    out = bcs.apply(face_cells, init, lambda bc, faces, owner: bc)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(init))
