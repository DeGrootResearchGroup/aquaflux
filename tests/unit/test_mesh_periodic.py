"""Two boundary patches joined into one periodic seam (``aquaflux.mesh.PeriodicJoin``)."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import Mesh, PeriodicJoin, structured_grid_3d

from tests.support.meshes import restated

NX, NY, NZ = 4, 3, 2
LX, LY, LZ = 2.0, 1.5, 1.0


def _box(**restatement) -> Mesh:
    """A 4 x 3 x 2 box with its six sides named, stated as ``restated`` is told to."""
    return restated(
        structured_grid_3d(NX, NY, NZ, lx=LX, ly=LY, lz=LZ, named_boundaries=True), **restatement
    )


def _shuffled() -> Mesh:
    """The box with its faces in a random order, some rings reversed and interior faces turned over."""
    mesh = _box()
    rng = np.random.default_rng(3)
    interior = np.asarray(mesh.face_cells.interior)
    return _box(
        order=rng.permutation(mesh.n_faces),
        reverse=rng.random(mesh.n_faces) < 0.5,
        swap=interior & (rng.random(mesh.n_faces) < 0.5),
    )


@pytest.mark.parametrize("mesh", [_box(), _shuffled()], ids=["as built", "restated"])
def test_a_join_makes_each_face_of_the_seam_a_face_between_neighbours_one_period_apart(mesh):
    """Across the seam the neighbour's image sits one cell along x, square to the face.

    On an orthogonal grid every interior face's owner-to-neighbour displacement is the face's normal
    times the spacing along it, so a seam face whose offset had the wrong sign or size (or paired the
    wrong twins) would show a displacement of ``LX - dx`` or one leaning off the normal.
    """
    joined = PeriodicJoin("right", "left").join(mesh)
    geometry = joined.geometry()
    fc = joined.face_cells
    interior = np.asarray(fc.interior)
    cc = np.asarray(geometry.cell.centroid)
    owner = np.asarray(fc.owner)[interior]
    displacement = np.asarray(fc.neighbour_centroid(geometry.cell.centroid))[interior] - cc[owner]
    normal = np.asarray(geometry.face.normal)[interior]
    spacing = np.abs(normal) @ np.array([LX / NX, LY / NY, LZ / NZ])
    np.testing.assert_allclose(displacement, normal * spacing[:, None], atol=1e-12)
    # Each seam face's neighbour is its own twin's cell, so the image is that cell moved by the one
    # period: a face paired with another face's twin would need a different offset to look right.
    offset = np.asarray(fc.neighbour_offset)
    seam = np.any(offset != 0.0, axis=1)
    np.testing.assert_allclose(offset[seam], np.tile([LX, 0.0, 0.0], (NY * NZ, 1)), atol=1e-12)
    # The seam is NY * NZ faces more interior, the left patch's faces gone, the rest kept.
    assert interior.sum() == np.asarray(mesh.face_cells.interior).sum() + NY * NZ
    assert joined.n_faces == mesh.n_faces - NY * NZ
    assert "left" not in joined.face_patches.names and "right" not in joined.face_patches.names
    for name in ("bottom", "top", "back", "front"):
        assert (
            np.asarray(joined.face_patches.mask(name)).sum()
            == np.asarray(mesh.face_patches.mask(name)).sum()
        )
    np.testing.assert_allclose(np.asarray(geometry.cell.volume).sum(), LX * LY * LZ)


@pytest.mark.parametrize(
    ("join", "mesh", "match"),
    [
        (PeriodicJoin("right", "nowhere"), _box(), r"no patch 'nowhere'"),
        (PeriodicJoin("right", "top"), _box(), r"different face counts"),
    ],
    ids=["a missing patch", "patches of different sizes"],
)
def test_a_join_that_cannot_be_made_is_refused(join, mesh, match):
    with pytest.raises(ValueError, match=match):
        join.join(mesh)


def test_patches_that_are_not_one_translation_apart_are_refused():
    """One node of the left side moved: its faces no longer land on their twins."""
    mesh = _box()
    left = np.asarray(mesh.face_patches.mask("left"))
    offsets = np.asarray(mesh.face_nodes.offsets)
    first = int(np.asarray(mesh.face_nodes.face_node_indices)[offsets[np.nonzero(left)[0][0]]])
    nodes = np.asarray(mesh.node_coords).copy()
    nodes[first, 1] += 0.2 * LY / NY
    moved = Mesh.from_csr(
        nodes,
        offsets,
        np.asarray(mesh.face_nodes.face_node_indices),
        np.asarray(mesh.face_cells.owner),
        np.asarray(mesh.face_cells.neighbour),
        mesh.n_cells,
        face_patches={
            name: np.nonzero(np.asarray(mesh.face_patches.mask(name)))[0]
            for name in ("left", "right")
        },
    )
    with pytest.raises(ValueError, match=r"could not be matched face-for-face"):
        PeriodicJoin("right", "left").join(moved)


def test_a_periodic_mesh_is_not_joined_again():
    joined = PeriodicJoin("right", "left").join(_box())
    with pytest.raises(ValueError, match=r"periodic already"):
        PeriodicJoin("top", "bottom").join(joined)


def test_a_patch_is_not_joined_to_itself():
    with pytest.raises(ValueError, match=r"two different patches"):
        PeriodicJoin("left", "left")
