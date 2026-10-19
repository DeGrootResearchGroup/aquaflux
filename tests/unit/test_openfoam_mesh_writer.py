"""Unit tests for the OpenFOAM polyMesh writer: the face order, the winding, and the round trip.

The order and winding the format requires are checked as properties of the written arrays --
upper-triangular interior faces, contiguous patches, every ring's Newell normal out of its owner --
with an orientation test that shares nothing with the mesh's own face geometry, rather than against
a hand-written expected array. The round trip is checked by reading the written files back with
the existing reader and comparing geometry cell by cell: cells must come back unrenumbered.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from aquaflux.io import read_openfoam, write_openfoam
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.io.openfoam.disassembler import DEFAULT_PATCH_TYPE, disassemble
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d

from tests.support.meshes import (
    hexahedron_beside_a_prism,
    newell_outward,
    perturbed_grid_3d,
    restated,
    tetrahedral_grid_3d,
)
from tests.support.polymesh import cyclic_two_cube_polymesh_data


def _with_walls(mesh: Mesh) -> Mesh:
    """``mesh`` with every boundary face in one patch ``walls`` (the fixtures name none)."""
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    return Mesh.from_csr(
        mesh.node_coords,
        mesh.face_nodes.offsets,
        mesh.face_nodes.face_node_indices,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.n_cells,
        face_patches={"walls": boundary},
    )


def _labelled_box() -> Mesh:
    """A named-boundary box with patch types, a patch group and two cell zones."""
    box = structured_grid_3d(3, 2, 2, named_boundaries=True)
    patches = box.face_patches
    return Mesh.from_csr(
        box.node_coords,
        box.face_nodes.offsets,
        box.face_nodes.face_node_indices,
        box.face_cells.owner,
        box.face_cells.neighbour,
        box.n_cells,
        cell_zones={"inner": [1, 4, 7, 10], "corner": [0]},
        face_patches={name: np.asarray(patches.indices(name)) for name in patches.names[2:]},
        patch_types={"left": "patch", "right": "patch", "top": "wall", "bottom": "wall"},
        patch_groups={"sides": ["front", "back"], "walls": ["top", "bottom"]},
    )


MESHES = {
    "box": lambda: structured_grid_3d(3, 2, 2, named_boundaries=True),
    "skewed box": lambda: _with_walls(perturbed_grid_3d(3, 3, 2, perturb=0.25, seed=4)),
    "tetrahedra": lambda: _with_walls(tetrahedral_grid_3d(2, perturb=0.15, seed=5)),
    "hexahedron beside a prism": lambda: _with_walls(hexahedron_beside_a_prism()),
    "labelled box": _labelled_box,
}


def _scrambled(mesh: Mesh, seed: int) -> Mesh:
    """``mesh`` with its faces shuffled, half its rings reversed and half its interior turned over."""
    rng = np.random.default_rng(seed)
    interior = np.asarray(mesh.face_cells.neighbour) >= 0
    return restated(
        mesh,
        order=rng.permutation(mesh.n_faces),
        reverse=rng.random(mesh.n_faces) < 0.5,
        swap=interior & (rng.random(mesh.n_faces) < 0.5),
    )


def _rings(data):
    offsets, indices = data.face_node_offsets, data.face_node_indices
    return [tuple(indices[a:b].tolist()) for a, b in pairwise(offsets)]


@pytest.mark.parametrize("name", MESHES)
def test_interior_faces_are_upper_triangular(name):
    data = disassemble(_scrambled(MESHES[name](), seed=1))
    n_interior = data.neighbour_internal.size
    owner, neighbour = data.owner[:n_interior], data.neighbour_internal
    assert np.all(owner < neighbour)
    keys = owner * (int(neighbour.max()) + 1) + neighbour
    assert np.all(np.diff(keys) > 0)


@pytest.mark.parametrize("name", MESHES)
def test_each_patch_is_one_block_after_the_interior_faces(name):
    mesh = _scrambled(MESHES[name](), seed=2)
    data = disassemble(mesh)
    starts = [patch.start_face for patch in data.patches]
    ends = [patch.start_face + patch.n_faces for patch in data.patches]
    assert starts[0] == data.neighbour_internal.size
    assert starts[1:] == ends[:-1]
    assert ends[-1] == data.owner.size
    written = _rings(data)
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    for patch in data.patches:
        members = np.asarray(mesh.face_patches.indices(patch.name))
        expected = sorted(tuple(sorted(indices[offsets[f] : offsets[f + 1]])) for f in members)
        block = written[patch.start_face : patch.start_face + patch.n_faces]
        assert sorted(tuple(sorted(ring)) for ring in block) == expected


@pytest.mark.parametrize("name", MESHES)
def test_every_ring_points_out_of_its_owner(name):
    data = disassemble(_scrambled(MESHES[name](), seed=3))
    # Assembled as written, without the mesh's own normal orientation entering the check.
    assert np.all(newell_outward(assemble(data)))


@pytest.mark.parametrize("name", MESHES)
def test_how_the_mesh_was_stored_does_not_change_what_is_written(name):
    # Interior faces have one written order, so they must come out identical; a patch's faces
    # keep the order they were stored in, so each patch is compared as a set of (ring, owner).
    mesh = MESHES[name]()
    plain, scrambled = disassemble(mesh), disassemble(_scrambled(mesh, seed=4))
    n_interior = plain.neighbour_internal.size
    assert _rings(plain)[:n_interior] == _rings(scrambled)[:n_interior]
    np.testing.assert_array_equal(plain.owner[:n_interior], scrambled.owner[:n_interior])
    np.testing.assert_array_equal(plain.neighbour_internal, scrambled.neighbour_internal)
    assert [p[:4] for p in plain.patches] == [p[:4] for p in scrambled.patches]
    for patch in plain.patches:
        block = slice(patch.start_face, patch.start_face + patch.n_faces)

        def faces(data, block=block):
            return sorted(zip(_rings(data)[block], data.owner[block].tolist(), strict=True))

        assert faces(plain) == faces(scrambled)


@pytest.mark.parametrize("name", MESHES)
def test_the_mesh_reads_back_with_its_cells_in_place(name, tmp_path):
    mesh = MESHES[name]()
    write_openfoam(_scrambled(mesh, seed=5), tmp_path)
    back = read_openfoam(tmp_path)
    assert back.n_cells == mesh.n_cells
    original, reread = mesh.geometry().cell, back.geometry().cell
    # Points are written exactly, so the geometry agrees to rounding in the face summation order.
    np.testing.assert_allclose(reread.volume, original.volume, rtol=1e-12)
    np.testing.assert_allclose(reread.centroid, original.centroid, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(back.node_coords), np.asarray(mesh.node_coords))


def test_patches_zones_types_and_groups_survive_the_round_trip(tmp_path):
    mesh = _labelled_box()
    write_openfoam(mesh, tmp_path)
    back = read_openfoam(tmp_path)

    def faces_of(m, name):
        rings = _rings(disassemble(m))
        data = disassemble(m)
        (patch,) = [p for p in data.patches if p.name == name]
        return sorted(
            tuple(sorted(r)) for r in rings[patch.start_face : patch.start_face + patch.n_faces]
        )

    named = mesh.face_patches.names[2:]
    assert back.face_patches.names[2:] == named
    for name in named:
        assert faces_of(back, name) == faces_of(mesh, name)
    types = dict(back.face_patches.patch_types)
    assert types == {
        name: dict(mesh.face_patches.patch_types).get(name, DEFAULT_PATCH_TYPE) for name in named
    }
    assert dict(back.face_patches.patch_groups) == dict(mesh.face_patches.patch_groups)
    for zone in ("inner", "corner"):
        np.testing.assert_array_equal(back.cell_zones.indices(zone), mesh.cell_zones.indices(zone))


def test_an_existing_mesh_is_kept_unless_overwrite_is_asked_and_then_none_of_it_is_left(tmp_path):
    write_openfoam(_labelled_box(), tmp_path)
    plain = structured_grid_3d(2, 2, 2, named_boundaries=True)
    with pytest.raises(FileExistsError, match="already holds a mesh"):
        write_openfoam(plain, tmp_path)
    write_openfoam(plain, tmp_path, overwrite=True)
    # The labelled box's cellZones would otherwise be read against the new mesh's cells.
    assert not (tmp_path / "constant" / "polyMesh" / "cellZones").exists()
    assert read_openfoam(tmp_path).cell_zones.names == ("default",)


def test_a_two_dimensional_mesh_is_refused():
    with pytest.raises(ValueError, match="three-dimensional; this mesh has dim 2"):
        disassemble(structured_grid_2d(2, 2, named_boundaries=True))


def test_a_periodic_mesh_is_refused():
    with pytest.raises(NotImplementedError, match="periodic"):
        disassemble(assemble(cyclic_two_cube_polymesh_data()))


def test_a_cyclic_patch_type_is_refused():
    box = structured_grid_3d(2, 2, 2, named_boundaries=True)
    patches = box.face_patches
    typed = Mesh.from_csr(
        box.node_coords,
        box.face_nodes.offsets,
        box.face_nodes.face_node_indices,
        box.face_cells.owner,
        box.face_cells.neighbour,
        box.n_cells,
        face_patches={name: np.asarray(patches.indices(name)) for name in patches.names[2:]},
        patch_types={"left": "cyclic"},
    )
    with pytest.raises(NotImplementedError, match=r"\['left'\] are declared cyclic"):
        disassemble(typed)


def test_boundary_faces_in_no_named_patch_are_refused():
    with pytest.raises(ValueError, match="24 boundary face"):
        disassemble(structured_grid_3d(2, 2, 2))


def test_a_patch_holding_interior_faces_is_refused():
    box = structured_grid_3d(2, 2, 2, named_boundaries=True)
    patches = box.face_patches
    interior = np.flatnonzero(np.asarray(box.face_cells.neighbour) >= 0)
    baffled = Mesh.from_csr(
        box.node_coords,
        box.face_nodes.offsets,
        box.face_nodes.face_node_indices,
        box.face_cells.owner,
        box.face_cells.neighbour,
        box.n_cells,
        face_patches={
            **{name: np.asarray(patches.indices(name)) for name in patches.names[2:]},
            "baffle": interior[:2],
        },
    )
    with pytest.raises(ValueError, match=r"\['baffle'\] hold interior faces"):
        disassemble(baffled)
