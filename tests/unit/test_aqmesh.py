"""aquaflux's own mesh file (``.aqmesh``): written and read back exactly, or refused.

"Exactly" is checked as the mesh is held: the same pytree structure (every static field -- patch
names, types, groups, zone names, sizes) and every array leaf equal, dtype included. The fixtures
differ in shape, not only in size: a periodic, graded two-dimensional grid; a perturbed hexahedral
grid with patch types, a patch group and a cell zone; and a mixed mesh of tetrahedra (unequal face
sizes are the same arrays as hexahedra, but a mesh of only quadrilaterals would not show a mix-up).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from aquaflux.case import AquafluxMesh, mesh_source_from_mapping
from aquaflux.io import (
    AquafluxMeshWriter,
    read_aqmesh,
    read_aqmesh_header,
    write_aqmesh,
)
from aquaflux.io.aqmesh.mesh_file import AQMESH_VERSION
from aquaflux.mesh import Mesh, graded_nodes, mesh_digest, structured_grid_2d

from tests.support.meshes import hexahedron_beside_a_prism, perturbed_grid_3d, tetrahedral_grid_3d


def _same(first: Mesh, second: Mesh) -> bool:
    """Whether two meshes are the same pytree with equal leaves of equal dtypes."""
    leaves_a, tree_a = jax.tree_util.tree_flatten(first)
    leaves_b, tree_b = jax.tree_util.tree_flatten(second)
    return tree_a == tree_b and all(
        np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves_a, leaves_b, strict=True)
    )


def _periodic_channel() -> Mesh:
    return structured_grid_2d(
        4, 6, 1.0, 2.0, named_boundaries=True, periodic=("x",),
        y_nodes=graded_nodes(6, 2.0, 1.2),
    )  # fmt: skip


def _labelled_hexahedra() -> Mesh:
    """A perturbed hexahedral grid restated with patch types, a patch group and a cell zone."""
    grid = perturbed_grid_3d(3, 2, 2, perturb=0.2, seed=3)
    owner = np.asarray(grid.face_cells.owner)
    neighbour = np.asarray(grid.face_cells.neighbour)
    boundary = np.flatnonzero(neighbour < 0)
    halves = {"inlet": boundary[: boundary.size // 2], "walls": boundary[boundary.size // 2 :]}
    return Mesh.from_csr(
        grid.node_coords, grid.face_nodes.offsets, grid.face_nodes.face_node_indices, owner,
        neighbour, grid.n_cells,
        face_patches=halves,
        patch_types={"inlet": "patch", "walls": "wall"},
        patch_groups={"solid": ["walls"]},
        cell_zones={"porous": np.arange(0, grid.n_cells, 2)},
    )  # fmt: skip


@pytest.mark.parametrize(
    "build",
    [_periodic_channel, _labelled_hexahedra, lambda: tetrahedral_grid_3d(2, perturb=0.2),
     hexahedron_beside_a_prism],
)  # fmt: skip
def test_a_mesh_reads_back_exactly_as_it_was_written(build, tmp_path):
    mesh = build()
    path = write_aqmesh(mesh, tmp_path / "mesh.aqmesh")
    assert _same(read_aqmesh(path), mesh)


def test_the_header_says_what_the_mesh_is_without_reading_it(tmp_path):
    mesh = _labelled_hexahedra()
    path = write_aqmesh(mesh, tmp_path / "mesh.aqmesh", provenance={"made by": "a test", "n": 3})
    header = read_aqmesh_header(path)
    assert (header.dim, header.n_nodes, header.n_faces, header.n_cells) == (
        3, mesh.n_nodes, mesh.n_faces, mesh.n_cells,
    )  # fmt: skip
    assert header.patches == ("inlet", "walls")
    assert dict(header.patch_types) == {"inlet": "patch", "walls": "wall"}
    assert header.patch_groups == (("solid", ("walls",)),)
    assert header.zones == ("default", "porous")
    assert (
        not header.periodic
        and read_aqmesh_header(
            write_aqmesh(_periodic_channel(), tmp_path / "channel.aqmesh")
        ).periodic
    )
    assert header.digest == mesh_digest(mesh)
    assert dict(header.provenance) == {"made by": "a test", "n": 3}
    assert header.version == AQMESH_VERSION


def _rewritten(path: Path, **changes) -> None:
    """The archive at ``path`` written again with some members replaced (or, ``None``, removed)."""
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    for name, value in changes.items():
        if value is None:
            members.pop(name)
        else:
            members[name] = value
    with open(path, "wb") as file:
        np.savez(file, **members)


def _header_with(path: Path, **fields) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(archive["header"].tobytes().decode())
    header.update(fields)
    return np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)


def test_a_mesh_changed_since_it_was_written_is_refused(tmp_path):
    path = write_aqmesh(_labelled_hexahedra(), tmp_path / "mesh.aqmesh")
    with np.load(path, allow_pickle=False) as archive:
        nodes = archive["node_coords"].copy()
    nodes[0, 0] += 0.1
    _rewritten(path, node_coords=nodes)
    with pytest.raises(ValueError, match="digest differs"):
        read_aqmesh(path)


def test_a_newer_version_is_refused_and_named(tmp_path):
    path = write_aqmesh(_labelled_hexahedra(), tmp_path / "mesh.aqmesh")
    _rewritten(path, header=_header_with(path, version=AQMESH_VERSION + 1))
    with pytest.raises(ValueError, match=f"version {AQMESH_VERSION + 1}, newer"):
        read_aqmesh_header(path)


def test_a_file_of_another_kind_or_missing_an_array_is_refused(tmp_path):
    other = tmp_path / "other.aqmesh"
    with open(other, "wb") as file:
        np.savez(file, points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="has no header"):
        read_aqmesh(other)
    path = write_aqmesh(_periodic_channel(), tmp_path / "mesh.aqmesh")
    _rewritten(path, neighbour_offset=None)
    with pytest.raises(ValueError, match="missing its neighbour_offset"):
        read_aqmesh(path)
    with pytest.raises(FileNotFoundError):
        read_aqmesh(tmp_path / "absent.aqmesh")


def test_an_existing_file_is_kept_unless_overwrite_is_given(tmp_path):
    path = write_aqmesh(_periodic_channel(), tmp_path / "mesh.aqmesh")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_aqmesh(_labelled_hexahedra(), path)
    assert path.read_bytes() == before
    write_aqmesh(_labelled_hexahedra(), path, overwrite=True)
    assert _same(read_aqmesh(path), _labelled_hexahedra())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mesh.aqmesh"]  # no partial file left


def test_a_file_without_the_suffix_or_unserializable_provenance_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"ends in \.aqmesh"):
        AquafluxMeshWriter(tmp_path / "mesh.npz")
    with pytest.raises(ValueError, match="JSON-serializable"):
        AquafluxMeshWriter(tmp_path / "mesh.aqmesh", provenance={"at": object()})


def test_a_case_reads_its_aquaflux_mesh_relative_to_its_own_directory(tmp_path):
    mesh = _periodic_channel()
    write_aqmesh(mesh, tmp_path / "meshes" / "channel.aqmesh")
    source = mesh_source_from_mapping({"kind": "AquafluxMesh", "path": "meshes/channel.aqmesh"})
    assert source == AquafluxMesh("meshes/channel.aqmesh")
    assert _same(source.read(tmp_path), mesh)
    with pytest.raises(ValueError, match="an OpenFOAMMesh"):
        AquafluxMesh("constant/polyMesh")
