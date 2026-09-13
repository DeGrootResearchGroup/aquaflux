"""Unit tests for the VTK XML serialization, by reading each document back.

A written file is only worth as much as a reader gets out of it, so every assertion here goes
through a parse rather than over the text: the ASCII form through the standard library's XML
parser, and the appended-binary form through the byte offsets its own tags declare. The two forms
are then held to carrying the same values, which is the property that lets the readable one be used
to check the fast one.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aquaflux.io.vtk.topology import build_vtk_cells
from aquaflux.io.vtk.xml import cell_data_arrays, pvd_document, vtu_parts
from aquaflux.mesh import structured_grid_2d, structured_grid_3d

_NUMPY_OF_VTK = {
    "Float64": np.float64,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt64": np.uint64,
}


def _decode(document: bytes) -> tuple[ET.Element, dict[str, np.ndarray]]:
    """The document's XML tree, and every DataArray's values, whichever form they are written in."""
    marker = b'<AppendedData encoding="raw">\n_'
    split = document.find(marker)
    if split < 0:
        root = ET.fromstring(document.decode())
        payload = b""
    else:
        head = document[:split] + b"</VTKFile>"
        root = ET.fromstring(head.decode())
        payload = document[split + len(marker) :]

    header = np.dtype(_NUMPY_OF_VTK[root.attrib["header_type"]]) if split >= 0 else None
    values = {}
    for array in root.iter("DataArray"):
        dtype = np.dtype(_NUMPY_OF_VTK[array.attrib["type"]])
        if array.attrib["format"] == "ascii":
            flat = np.fromstring(array.text, sep=" ", dtype=dtype)
        else:
            at = int(array.attrib["offset"])
            size = int(np.frombuffer(payload, header, count=1, offset=at)[0])
            flat = np.frombuffer(
                payload, dtype, count=size // dtype.itemsize, offset=at + header.itemsize
            )
        components = int(array.attrib.get("NumberOfComponents", 1))
        values[array.attrib["Name"]] = flat.reshape(-1, components) if components > 1 else flat
    return root, values


def _document(mesh, fields=(), *, binary=True) -> bytes:
    cells = build_vtk_cells(mesh)
    arrays = cell_data_arrays(dict(fields), mesh.n_cells, mesh.dim)
    return b"".join(vtu_parts(cells, arrays, binary=binary))


@pytest.mark.parametrize("binary", [True, False])
def test_a_3d_document_declares_its_piece_and_carries_every_cell_array(binary):
    mesh = structured_grid_3d(2, 2, 1)
    fields = {
        "p": np.arange(mesh.n_cells, dtype=float),
        "U": np.asarray(mesh.geometry().cell.centroid),
    }
    root, values = _decode(_document(mesh, fields, binary=binary))

    piece = root.find("./UnstructuredGrid/Piece")
    assert int(piece.attrib["NumberOfPoints"]) == mesh.n_nodes
    assert int(piece.attrib["NumberOfCells"]) == mesh.n_cells
    assert set(values) == {
        "Points",
        "connectivity",
        "offsets",
        "types",
        "faces",
        "faceoffsets",
        "p",
        "U",
    }

    cells = build_vtk_cells(mesh)
    np.testing.assert_allclose(values["Points"], cells.points)
    np.testing.assert_array_equal(values["connectivity"], cells.connectivity)
    np.testing.assert_array_equal(values["offsets"], cells.offsets)
    np.testing.assert_array_equal(values["types"], cells.types)
    np.testing.assert_array_equal(values["faces"], cells.faces)
    np.testing.assert_array_equal(values["faceoffsets"], cells.face_offsets)
    np.testing.assert_allclose(values["p"], fields["p"])
    np.testing.assert_allclose(values["U"], fields["U"])


def test_a_2d_document_carries_no_face_stream():
    root, values = _decode(_document(structured_grid_2d(2, 2)))
    assert "faces" not in values and "faceoffsets" not in values
    assert {array.attrib["Name"] for array in root.iter("DataArray")} == {
        "Points",
        "connectivity",
        "offsets",
        "types",
    }


def test_the_ascii_and_binary_forms_carry_the_same_values():
    mesh = structured_grid_3d(2, 1, 2)
    fields = {
        "p": np.linspace(-1.0, 1.0, mesh.n_cells),
        "U": np.asarray(mesh.geometry().cell.centroid),
    }
    _, text = _decode(_document(mesh, fields, binary=False))
    _, raw = _decode(_document(mesh, fields, binary=True))

    assert set(text) == set(raw)
    for name in text:
        # Exact, not close: the readable form is only useful for checking the other one if it is a
        # rendering of the same doubles rather than a rounding of them.
        np.testing.assert_array_equal(text[name], raw[name], err_msg=name)


def test_a_2d_vector_is_padded_to_three_components_on_its_trailing_axis():
    mesh = structured_grid_2d(3, 2)
    velocity = np.stack([np.arange(mesh.n_cells), -np.arange(mesh.n_cells)], axis=1).astype(float)
    (field,) = cell_data_arrays({"U": velocity}, mesh.n_cells, mesh.dim)

    assert field.components == 3
    np.testing.assert_array_equal(field.values[:, :2], velocity)
    np.testing.assert_array_equal(field.values[:, 2], 0.0)


def test_a_3d_vector_keeps_its_own_three_components():
    velocity = np.arange(12.0).reshape(4, 3)
    (field,) = cell_data_arrays({"U": velocity}, 4, 3)
    assert field.components == 3
    np.testing.assert_array_equal(field.values, velocity)


def test_a_scalar_is_written_as_a_single_component():
    (field,) = cell_data_arrays({"p": np.arange(4.0)}, 4, 2)
    assert field.components == 1
    np.testing.assert_array_equal(field.values.ravel(), np.arange(4.0))


def test_a_field_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="has 3 values but the mesh has 4 cells"):
        cell_data_arrays({"p": np.zeros(3)}, 4, 2)


def test_a_field_of_an_unusable_rank_is_refused():
    with pytest.raises(ValueError, match="expected"):
        cell_data_arrays({"tau": np.zeros((4, 3, 3))}, 4, 3)


def test_an_array_narrows_to_32_bit_indices_when_they_fit():
    root, _ = _decode(_document(structured_grid_3d(2, 2, 2)))
    types = {a.attrib["Name"]: a.attrib["type"] for a in root.iter("DataArray")}
    assert types["connectivity"] == "Int32"
    assert types["faces"] == "Int32"
    assert types["Points"] == "Float64"
    assert types["types"] == "UInt8"


def test_a_mesh_with_no_fields_writes_an_empty_cell_data_block():
    root, values = _decode(_document(structured_grid_2d(2, 2)))
    assert root.find("./UnstructuredGrid/Piece/CellData").findall("DataArray") == []
    assert "connectivity" in values


def test_a_pvd_collection_indexes_each_file_by_its_time():
    root = ET.fromstring(pvd_document([(0.0, "a.vtu"), (0.5, "b.vtu")]).decode())
    entries = root.findall("./Collection/DataSet")
    assert [e.attrib["file"] for e in entries] == ["a.vtu", "b.vtu"]
    assert [float(e.attrib["timestep"]) for e in entries] == [0.0, 0.5]


def test_a_field_name_needing_escaping_survives_the_round_trip():
    mesh = structured_grid_2d(2, 1)
    name = 'grad("p") & <k>'
    _, values = _decode(_document(mesh, {name: np.arange(mesh.n_cells, dtype=float)}))
    np.testing.assert_array_equal(values[name], np.arange(mesh.n_cells))


def test_the_appended_block_offsets_are_where_the_tags_say():
    document = _document(structured_grid_3d(2, 1, 1), {"p": np.zeros(2)})
    payload = document[document.find(b'<AppendedData encoding="raw">\n_') + 31 :]
    offsets = [int(m) for m in re.findall(rb'offset="(\d+)"', document)]
    assert offsets[0] == 0
    running = 0
    for offset in offsets:
        assert offset == running
        running += 8 + int(np.frombuffer(payload, np.uint64, count=1, offset=offset)[0])
    assert payload[running:] == b"\n  </AppendedData>\n</VTKFile>\n"
