"""Boundary patches written as one ``.vtp`` per patch, indexed by a ``.vtm``.

Each file is read back by an independent decoder of the XML and its appended block, and checked
against the mesh it was written from: every face's ring, in order and wound out of the domain, and every
field value on the face it belongs to.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aquaflux.io import write_patches
from aquaflux.io.vtk import boundary_patches
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d

from tests.unit.test_mesh_patch_triangles import rebuilt_with_reversed_rings
from tests.unit.test_vtk_xml import _decode

SIDES = ("left", "right", "bottom", "top", "back", "front")


def _box() -> Mesh:
    # Unequal counts and lengths, so a patch read back as another, or a transposed axis, shows.
    return structured_grid_3d(3, 2, 4, 1.0, 2.0, 3.0, named_boundaries=True)


def _read(path):
    return _decode(path.read_bytes())


def _rings(values, kind="Polys"):
    """Each polygon's point indices, from VTK's connectivity and end offsets."""
    del kind
    ends = values["offsets"]
    starts = np.concatenate([[0], ends[:-1]])
    return [values["connectivity"][a:b] for a, b in zip(starts, ends, strict=True)]


def test_every_boundary_patch_is_a_block_named_by_it_and_a_file_beside_the_index(tmp_path) -> None:
    mesh = _box()
    index = write_patches(mesh, None, tmp_path / "out" / "patches.vtm")
    root = ET.fromstring(index.read_text())
    assert root.attrib["type"] == "vtkMultiBlockDataSet"
    blocks = [(d.attrib["name"], d.attrib["file"]) for d in root.iter("DataSet")]
    assert blocks == [(name, f"patches/{name}.vtp") for name in SIDES]
    assert [d.attrib["index"] for d in root.iter("DataSet")] == [str(i) for i in range(6)]
    for _, file in blocks:
        assert (index.parent / file).is_file()


@pytest.mark.parametrize("binary", [True, False])
def test_each_face_is_its_own_ring_wound_out_of_the_domain_whatever_order_it_was_stored_in(
    tmp_path, binary
) -> None:
    mesh = _box()
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    mesh = rebuilt_with_reversed_rings(mesh, boundary[::2])
    geometry = mesh.geometry()
    write_patches(mesh, None, tmp_path / "patches.vtm", binary=binary)
    offsets = np.asarray(mesh.face_nodes.offsets)
    stored = np.asarray(mesh.face_nodes.face_node_indices)
    for name in SIDES:
        root, values = _read(tmp_path / "patches" / f"{name}.vtp")
        piece = next(root.iter("Piece"))
        faces = np.asarray(mesh.face_patches.indices(name))
        assert int(piece.attrib["NumberOfPolys"]) == len(faces)
        points = values["Points"].reshape(-1, 3)
        # The patch's own nodes only, each once.
        assert int(piece.attrib["NumberOfPoints"]) == len(np.unique(stored[np.concatenate(
            [np.arange(offsets[f], offsets[f + 1]) for f in faces]
        )]))  # fmt: skip
        for face, ring in zip(faces, _rings(values), strict=True):
            corners = points[ring]
            mine = np.asarray(mesh.node_coords)[stored[offsets[face] : offsets[face + 1]]]
            # The same nodes, in the stored cyclic order or its reverse.
            assert {tuple(c) for c in corners} == {tuple(c) for c in mine}
            # Newell's normal of the ring points out of the domain, along the owner-outward normal.
            newell = np.sum(np.cross(corners, np.roll(corners, -1, axis=0)), axis=0)
            assert np.dot(newell, np.asarray(geometry.face.normal)[face]) > 0.0


def test_a_field_lands_on_the_face_it_belongs_to_and_a_patch_without_one_has_none(tmp_path) -> None:
    mesh = _box()
    geometry = mesh.geometry()
    top = np.asarray(mesh.face_patches.indices("top"))
    centroid = np.asarray(geometry.face.centroid)
    # A field whose value encodes where its face is, so a shuffled face would carry a wrong value.
    fields = {
        "top": {
            "E": centroid[top, 0] + 10.0 * centroid[top, 2],
            "flux": np.asarray(geometry.face.normal)[top] * 2.0,
        }
    }
    write_patches(mesh, fields, tmp_path / "patches.vtm")
    _, values = _read(tmp_path / "patches" / "top.vtp")
    points = values["Points"].reshape(-1, 3)
    centres = np.array([points[ring].mean(axis=0) for ring in _rings(values)])
    np.testing.assert_allclose(values["E"], centres[:, 0] + 10.0 * centres[:, 2], rtol=1e-14)
    np.testing.assert_allclose(values["flux"], np.tile([0.0, 2.0, 0.0], (len(top), 1)))
    root, bare = _read(tmp_path / "patches" / "left.vtp")
    assert set(bare) == {"Points", "connectivity", "offsets"}
    assert next(root.iter("CellData")).find("DataArray") is None


def test_a_two_dimensional_mesh_writes_its_boundary_as_line_segments_in_the_plane(tmp_path) -> None:
    mesh = structured_grid_2d(3, 2, 1.0, 2.0, named_boundaries=True)
    write_patches(mesh, {"top": {"v": np.ones((3, 2))}}, tmp_path / "patches.vtm")
    root, values = _read(tmp_path / "patches" / "top.vtp")
    piece = next(root.iter("Piece"))
    assert piece.attrib["NumberOfLines"] == "3" and piece.attrib["NumberOfPolys"] == "0"
    assert root.find(".//Lines") is not None
    points = values["Points"].reshape(-1, 3)
    np.testing.assert_array_equal(points[:, 2], 0.0)
    np.testing.assert_allclose(points[:, 1], 2.0)
    # A two-component vector is padded on its trailing axis, as the cell writer pads it.
    np.testing.assert_array_equal(values["v"], np.tile([1.0, 1.0, 0.0], (3, 1)))


def test_only_the_patches_asked_for_are_written(tmp_path) -> None:
    index = write_patches(_box(), None, tmp_path / "boundary.vtm", patches=["front", "left"])
    root = ET.fromstring(index.read_text())
    assert [d.attrib["file"] for d in root.iter("DataSet")] == [
        "boundary/front.vtp",
        "boundary/left.vtp",
    ]
    assert sorted(p.name for p in (tmp_path / "boundary").iterdir()) == ["front.vtp", "left.vtp"]


def test_what_cannot_be_written_as_asked_is_refused(tmp_path) -> None:
    mesh = _box()
    assert boundary_patches(mesh) == SIDES
    with pytest.raises(ValueError, match=r"\['roof'\] is not a boundary patch"):
        write_patches(mesh, None, tmp_path / "p.vtm", patches=["roof"])
    with pytest.raises(ValueError, match=r"fields are given for \['top'\], which are not among"):
        write_patches(mesh, {"top": {"E": np.zeros(6)}}, tmp_path / "p.vtm", patches=["left"])
    with pytest.raises(
        ValueError, match=r"patch 'top': field 'E' has 5 values, but the patch has 12"
    ):
        write_patches(mesh, {"top": {"E": np.zeros(5)}}, tmp_path / "p.vtm")
