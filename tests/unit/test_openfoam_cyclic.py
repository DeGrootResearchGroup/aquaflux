"""Unit tests for cyclic-patch fusion (parsed arrays -> a periodic interior seam), no files.

Drives :func:`aquaflux.io.openfoam.assembler.assemble` on hand-built ``PolyMeshData`` fixtures
(:mod:`tests.support.polymesh`) so both the fusion arithmetic (owner/neighbour/offset on the
fused seam, patch removal) and its error paths are exercised in isolation from text parsing, and
cross-checks a fused structured slab against ``structured_grid_2d(periodic=("x",))`` — the
independent oracle for the actual periodic connectivity.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.io.openfoam.records import FoamPatch
from aquaflux.mesh import collapse_extruded_direction, structured_grid_2d

from tests.support.meshes import geometry_invariants
from tests.support.polymesh import (
    cyclic_slab_polymesh_data,
    cyclic_two_cube_polymesh_data,
    two_cube_polymesh_data,
)


def test_fuses_cyclic_pair_into_an_interior_seam():
    mesh = assemble(cyclic_two_cube_polymesh_data())

    assert mesh.n_faces == 10  # the donor ("outlet") face is dropped, not just relabelled
    assert int(np.sum(np.asarray(mesh.face_cells.interior))) == 2  # the original seam + this one
    names = set(mesh.face_patches.names)
    assert "inlet" not in names and "outlet" not in names
    assert mesh.face_patches.size("walls") == 8

    # The former "inlet" face (global index 1) survives at the same position: nothing before it
    # in file order was dropped.
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    offset = np.asarray(mesh.face_cells.neighbour_offset)
    assert owner[1] == 0
    assert neighbour[1] == 1
    np.testing.assert_allclose(offset[1], [-2.0, 0.0, 0.0])
    np.testing.assert_allclose(offset[np.arange(10) != 1], 0.0)

    volumes = np.asarray(mesh.geometry().cell.volume)
    np.testing.assert_allclose(volumes, 1.0)


def test_non_cyclic_mesh_carries_no_offset():
    mesh = assemble(two_cube_polymesh_data())
    assert mesh.face_cells.neighbour_offset is None


def test_missing_neighbour_patch_raises():
    data = cyclic_two_cube_polymesh_data()
    data = data._replace(patches=(FoamPatch("inlet", "cyclic", 1, 1), *data.patches[1:]))
    with pytest.raises(ValueError, match="no neighbourPatch entry"):
        assemble(data)


def test_unknown_neighbour_patch_raises():
    data = cyclic_two_cube_polymesh_data()
    data = data._replace(
        patches=(FoamPatch("inlet", "cyclic", 1, 1, "nonexistent"), *data.patches[1:])
    )
    with pytest.raises(ValueError, match="does not exist"):
        assemble(data)


def test_self_referencing_neighbour_patch_raises():
    data = cyclic_two_cube_polymesh_data()
    data = data._replace(patches=(FoamPatch("inlet", "cyclic", 1, 1, "inlet"), *data.patches[1:]))
    with pytest.raises(ValueError, match="own neighbourPatch"):
        assemble(data)


def test_partner_not_cyclic_raises():
    data = cyclic_two_cube_polymesh_data()
    data = data._replace(
        patches=(
            data.patches[0],
            FoamPatch("outlet", "patch", 2, 1, "inlet"),  # no longer type "cyclic"
            data.patches[2],
        )
    )
    with pytest.raises(ValueError, match="not itself a cyclic patch"):
        assemble(data)


def test_asymmetric_neighbour_patch_raises():
    data = cyclic_two_cube_polymesh_data()
    data = data._replace(
        patches=(
            data.patches[0],
            FoamPatch("outlet", "cyclic", 2, 1, "walls"),  # points elsewhere, not back to "inlet"
            data.patches[2],
        )
    )
    with pytest.raises(ValueError, match="instead of 'inlet'"):
        assemble(data)


def test_mismatched_face_count_raises():
    data = cyclic_slab_polymesh_data(3, 2)
    # Shrink "right" by one face without touching "left", so the counts disagree.
    patches = list(data.patches)
    right = patches[1]
    patches[1] = right._replace(n_faces=right.n_faces - 1)
    with pytest.raises(ValueError, match="different face counts"):
        assemble(data._replace(patches=tuple(patches)))


def test_non_translational_pair_raises():
    # "left" varies along y (at x = 0) and "bottom" varies along x (at y = 0): with matching face
    # counts (nx == ny == 2) but no single translation maps one row of centroids onto the other.
    data = cyclic_slab_polymesh_data(2, 2)
    left, right, bottom, top, front_and_back = data.patches
    patches = (
        left._replace(neighbour_patch="bottom"),
        right._replace(type_="patch", neighbour_patch=""),
        bottom._replace(type_="cyclic", neighbour_patch="left"),
        top,
        front_and_back,
    )
    with pytest.raises(ValueError, match="could not be matched"):
        assemble(data._replace(patches=patches))


@pytest.mark.parametrize(("nx", "ny"), [(2, 1), (3, 2), (4, 4)])
def test_cyclic_slab_matches_periodic_structured_grid(nx, ny):
    lx, ly = 2.0, 3.0
    mesh = assemble(cyclic_slab_polymesh_data(nx, ny, lx=lx, ly=ly))
    collapsed = collapse_extruded_direction(mesh, ["frontAndBack"])
    reference = structured_grid_2d(nx, ny, lx=lx, ly=ly, periodic=("x",))

    got = geometry_invariants(collapsed)
    want = geometry_invariants(reference)
    assert got["dim"] == want["dim"] == 2
    assert got["n_cells"] == want["n_cells"]
    assert got["n_faces"] == want["n_faces"]
    assert got["n_interior"] == want["n_interior"]
    np.testing.assert_allclose(got["volumes"], want["volumes"])
    np.testing.assert_allclose(got["areas"], want["areas"])

    # No boundary faces survive on the periodic axis: only "bottom"/"top" are named.
    assert set(collapsed.face_patches.names) & {"left", "right"} == set()
    assert {"bottom", "top"} <= set(collapsed.face_patches.names)
