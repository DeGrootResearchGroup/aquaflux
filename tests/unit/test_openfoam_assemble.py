"""Unit tests for the OpenFOAM assembler (parsed arrays -> Mesh), with no files.

Drives :func:`aquaflux.io.openfoam.assembler.assemble` from a hand-built ``PolyMeshData`` so the
semantic conventions — neighbour padding, cell count, patch/zone mapping, reserved-name guarding —
are exercised in isolation from text parsing.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.io.openfoam.records import FoamPatch, PolyMeshData

from tests.support.polymesh import two_cube_polymesh_data


def test_assembles_two_cube_mesh():
    mesh = assemble(two_cube_polymesh_data())
    assert mesh.dim == 3
    assert mesh.n_cells == 2
    assert mesh.n_faces == 11
    # One interior face (the shared x = 1 plane), so ten boundary faces.
    assert int(np.sum(np.asarray(mesh.face_cells.interior))) == 1
    volumes = np.asarray(mesh.geometry().cell.volume)
    np.testing.assert_allclose(volumes, 1.0)


def test_neighbour_is_padded_with_boundary_sentinel():
    mesh = assemble(two_cube_polymesh_data())
    neighbour = np.asarray(mesh.face_cells.neighbour)
    assert neighbour[0] == 1  # the single interior face
    assert np.all(neighbour[1:] == -1)  # every boundary face


def test_patches_and_zones_are_named():
    mesh = assemble(two_cube_polymesh_data())
    assert mesh.face_patches.size("inlet") == 1
    assert mesh.face_patches.size("outlet") == 1
    assert mesh.face_patches.size("walls") == 8
    assert mesh.face_patches.is_boundary_patch("walls", mesh.face_cells)
    np.testing.assert_array_equal(np.asarray(mesh.cell_zones.indices("left")), [0])
    np.testing.assert_array_equal(np.asarray(mesh.cell_zones.indices("right")), [1])


def test_reserved_patch_name_rejected():
    data = two_cube_polymesh_data()._replace(patches=(FoamPatch("boundary", "wall", 1, 10),))
    with pytest.raises(ValueError, match="reserved"):
        assemble(data)


def test_neighbour_longer_than_faces_rejected():
    data = two_cube_polymesh_data()._replace(neighbour_internal=np.arange(20, dtype=np.int64))
    with pytest.raises(ValueError, match="longer than the face count"):
        assemble(data)


def test_no_faces_rejected():
    empty = PolyMeshData(
        points=np.zeros((3, 3)),
        face_node_offsets=np.array([0], dtype=np.int64),
        face_node_indices=np.zeros(0, dtype=np.int64),
        owner=np.zeros(0, dtype=np.int64),
        neighbour_internal=np.zeros(0, dtype=np.int64),
        patches=(),
        cell_zones=(),
    )
    with pytest.raises(ValueError, match="no faces"):
        assemble(empty)
