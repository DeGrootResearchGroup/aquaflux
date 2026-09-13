"""End-to-end tests for the VTK writer: write real files, then read them back.

The reconstruction and the serialization are covered on their own; what is left to establish here is
that the shell around them puts the right bytes on disk, creates what it needs to, and refuses what
it cannot write. Files are read back with the standard library, since a VTK library is not a
dependency of this package -- reading them with VTK itself is the separate, manual check that a
viewer opens them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aquaflux.io import write_pvd, write_vtu
from aquaflux.mesh import structured_grid_2d, structured_grid_3d


def _header(path):
    """The XML above a file's appended data, parsed -- enough for structure but not for values."""
    document = path.read_bytes()
    split = document.find(b'<AppendedData encoding="raw">\n_')
    head = document if split < 0 else document[:split] + b"</VTKFile>"
    return ET.fromstring(head.decode())


def test_writing_a_3d_mesh_with_fields(tmp_path):
    mesh = structured_grid_3d(2, 2, 2)
    fields = {
        "p": np.arange(mesh.n_cells, dtype=float),
        "U": np.asarray(mesh.geometry().cell.centroid),
    }

    written = write_vtu(mesh, fields, tmp_path / "case.vtu")

    assert written == tmp_path / "case.vtu"
    root = _header(written)
    piece = root.find("./UnstructuredGrid/Piece")
    assert int(piece.attrib["NumberOfCells"]) == mesh.n_cells
    assert int(piece.attrib["NumberOfPoints"]) == mesh.n_nodes
    data = {a.attrib["Name"]: a for a in piece.find("CellData")}
    assert data["p"].attrib.get("NumberOfComponents", "1") == "1"
    assert data["U"].attrib["NumberOfComponents"] == "3"
    assert {a.attrib["Name"] for a in piece.find("Cells")} == {
        "connectivity",
        "offsets",
        "types",
        "faces",
        "faceoffsets",
    }


def test_writing_a_mesh_alone(tmp_path):
    # Inspecting a mesh before anything has been solved on it is the reason `fields` is optional.
    written = write_vtu(structured_grid_2d(3, 2), None, tmp_path / "mesh.vtu")
    assert _header(written).find("./UnstructuredGrid/Piece/CellData").findall("DataArray") == []


def test_the_destination_directory_is_created(tmp_path):
    written = write_vtu(structured_grid_2d(2, 2), {}, tmp_path / "deep" / "under" / "case.vtu")
    assert written.is_file()


def test_the_two_forms_differ_only_in_how_the_arrays_are_carried(tmp_path):
    mesh = structured_grid_3d(4, 4, 4)
    fields = {"p": np.arange(mesh.n_cells, dtype=float)}
    raw = write_vtu(mesh, fields, tmp_path / "raw.vtu")
    text = write_vtu(mesh, fields, tmp_path / "text.vtu", binary=False)

    assert b'format="appended"' in raw.read_bytes()
    assert b"AppendedData" not in text.read_bytes()
    # The text form has to stay parseable as plain XML; the raw one deliberately is not, since its
    # arrays are bytes rather than markup. (Which of the two is *smaller* depends on the mesh: a
    # toy mesh's indices are one or two characters against a fixed four bytes, and the order
    # reverses as the indices grow. Size is not why the default is raw -- load time is.)
    assert ET.fromstring(text.read_text()).find("./UnstructuredGrid/Piece") is not None
    for name in ("connectivity", "faces", "p"):
        assert f'Name="{name}"' in text.read_text()


def test_a_field_that_does_not_fit_the_mesh_is_refused(tmp_path):
    mesh = structured_grid_2d(2, 2)
    with pytest.raises(ValueError, match="but the mesh has 4 cells"):
        write_vtu(mesh, {"p": np.zeros(5)}, tmp_path / "case.vtu")


def test_a_non_finite_field_is_written_rather_than_refused(tmp_path):
    # A file for looking at, and where a solution went non-finite is one of the things to look at.
    mesh = structured_grid_2d(2, 2)
    values = np.array([1.0, np.nan, np.inf, -np.inf])
    written = write_vtu(mesh, {"p": values}, tmp_path / "case.vtu", binary=False)
    text = written.read_text()
    assert "nan" in text and "inf" in text


def test_a_pvd_collection_points_at_the_frames_written_beside_it(tmp_path):
    mesh = structured_grid_2d(2, 2)
    frames = []
    for step, time in enumerate([0.0, 0.5]):
        name = f"case_{step}.vtu"
        write_vtu(mesh, {"p": np.full(mesh.n_cells, time)}, tmp_path / name)
        frames.append((time, name))

    written = write_pvd(tmp_path / "case.pvd", frames)

    entries = ET.fromstring(written.read_text()).findall("./Collection/DataSet")
    assert [e.attrib["file"] for e in entries] == ["case_0.vtu", "case_1.vtu"]
    assert all((tmp_path / e.attrib["file"]).is_file() for e in entries)
