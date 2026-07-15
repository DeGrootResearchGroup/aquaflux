"""End-to-end tests for the OpenFOAM reader on small on-disk polyMesh fixtures.

Reads the committed fixtures and cross-checks the resulting mesh against the trusted structured-grid
generators (an independent oracle): the two-cube case against a 2x1x1 grid, and the collapsed 2D
slab against the corresponding 2D grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aquaflux.io import OpenFOAMReader, read_openfoam
from aquaflux.io.openfoam.foamfile import parse_foamfile
from aquaflux.mesh import structured_grid_2d, structured_grid_3d

_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _sorted_volumes(mesh):
    return np.sort(np.asarray(mesh.geometry().cell.volume))


def _sorted_areas(mesh):
    return np.sort(np.asarray(mesh.geometry().face.area))


def test_reads_3d_two_cube_mesh():
    mesh = read_openfoam(_FIXTURES / "polymesh_3d_two_cubes")
    assert mesh.dim == 3
    assert mesh.n_cells == 2
    assert mesh.n_faces == 11
    assert int(np.sum(np.asarray(mesh.face_cells.interior))) == 1
    np.testing.assert_allclose(_sorted_volumes(mesh), 1.0)

    # Named patches and zones survive the read.
    assert mesh.face_patches.size("inlet") == 1
    assert mesh.face_patches.size("walls") == 8
    np.testing.assert_array_equal(np.asarray(mesh.cell_zones.indices("left")), [0])


def test_3d_geometry_matches_structured_grid():
    mesh = read_openfoam(_FIXTURES / "polymesh_3d_two_cubes")
    reference = structured_grid_3d(2, 1, 1, lx=2.0, ly=1.0, lz=1.0)
    np.testing.assert_allclose(_sorted_volumes(mesh), _sorted_volumes(reference))
    np.testing.assert_allclose(_sorted_areas(mesh), _sorted_areas(reference))


def test_reads_and_collapses_2d_slab():
    mesh = read_openfoam(_FIXTURES / "polymesh_2d_slab")
    assert mesh.dim == 2
    assert mesh.n_cells == 2
    # The empty caps are gone; the four side patches survive.
    names = set(mesh.face_patches.names)
    assert "back" not in names and "front" not in names
    assert {"left", "right", "bottom", "top"} <= names

    reference = structured_grid_2d(2, 1, lx=2.0, ly=1.0)
    np.testing.assert_allclose(_sorted_volumes(mesh), _sorted_volumes(reference))
    np.testing.assert_allclose(_sorted_areas(mesh), _sorted_areas(reference))


def test_reads_and_collapses_2d_slab_single_frontandback():
    # The standard OpenFOAM 2D form: a single "frontAndBack" empty patch instead of two caps.
    mesh = read_openfoam(_FIXTURES / "polymesh_2d_slab_frontandback")
    assert mesh.dim == 2
    assert mesh.n_cells == 2
    names = set(mesh.face_patches.names)
    assert "frontAndBack" not in names
    assert {"left", "right", "bottom", "top"} <= names

    reference = structured_grid_2d(2, 1, lx=2.0, ly=1.0)
    np.testing.assert_allclose(_sorted_volumes(mesh), _sorted_volumes(reference))
    np.testing.assert_allclose(_sorted_areas(mesh), _sorted_areas(reference))


def test_reader_resolves_case_directory(tmp_path):
    # A reader pointed at a case directory finds constant/polyMesh under it.
    case = tmp_path / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        (case / "constant" / "polyMesh" / name).write_text(
            (_FIXTURES / "polymesh_3d_two_cubes" / name).read_text()
        )
    mesh = OpenFOAMReader(case).read()
    assert mesh.n_cells == 2


def test_binary_format_rejected(tmp_path):
    src = _FIXTURES / "polymesh_3d_two_cubes"
    for name in ("faces", "owner", "neighbour", "boundary", "cellZones"):
        (tmp_path / name).write_text((src / name).read_text())
    # Flip only the points file to binary format.
    points = parse_foamfile((src / "points").read_text())
    (tmp_path / "points").write_text(
        "FoamFile { format binary; class vectorField; object points; }\n" + points.body
    )
    with pytest.raises(NotImplementedError, match="binary"):
        read_openfoam(tmp_path)


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="no polyMesh"):
        read_openfoam(tmp_path / "does_not_exist")
