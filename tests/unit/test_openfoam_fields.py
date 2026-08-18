"""Unit tests for reading an OpenFOAM scalar field onto an imported mesh.

Split the same way the polyMesh reader is: :func:`parse_scalar_field` is pure and tests on string
snippets, while :func:`read_surface_scalar_field` is exercised end to end on a committed ASCII
fixture with a hand-written field beside it.

The property that actually matters is **placement** -- that value *i* in the file lands on face *i*
of the mesh -- because getting it wrong produces a plausible field rather than an error. So the
end-to-end test writes a field whose value encodes its own face index, and the reader's own
ordering checks are tested by feeding them a mesh whose ordering does not hold.
"""

from __future__ import annotations

from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.io.openfoam import (
    parse_scalar_field,
    read_surface_scalar_field,
    read_volume_scalar_field,
)
from aquaflux.io.openfoam.reader import read_openfoam
from aquaflux.mesh import structured_grid_2d

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "polymesh_3d_two_cubes"

_HEADER = """
FoamFile
{
    format      ascii;
    class       surfaceScalarField;
    object      phi;
}
dimensions      [0 3 -1 0 0 0 0];
"""


def _field_text(internal: str, patches: dict[str, str]) -> str:
    """An OpenFOAM field file body from an internal entry and per-patch entries."""
    blocks = "\n".join(
        f"    {name}\n    {{\n        type calculated;\n        value {entry};\n    }}"
        for name, entry in patches.items()
    )
    return f"{_HEADER}\ninternalField   {internal};\n\nboundaryField\n{{\n{blocks}\n}}\n"


def _nonuniform(values) -> str:
    return f"nonuniform List<scalar> {len(values)} ({' '.join(str(v) for v in values)})"


def test_parse_reads_a_nonuniform_internal_block() -> None:
    """The internal list comes back in file order."""
    body = _field_text(_nonuniform([1.0, 2.0, 3.0]), {})

    assert np.array_equal(parse_scalar_field(body, 3, {}), np.array([1.0, 2.0, 3.0]))


def test_parse_handles_uniform_and_nonuniform_in_one_file() -> None:
    """Both spellings occur together -- a wall's flux is ``uniform 0``, an inlet's is a list.

    A reader that handles only the list form fails on every wall, which is most of the boundary.
    """
    body = _field_text(
        _nonuniform([1.0, 2.0]),
        {"inlet": _nonuniform([-5.0, -6.0]), "wall": "uniform 0"},
    )

    values = parse_scalar_field(body, 2, {"inlet": 2, "wall": 3})

    assert np.array_equal(values, np.array([1.0, 2.0, -5.0, -6.0, 0.0, 0.0, 0.0]))


def test_parse_rejects_a_length_mismatch() -> None:
    """A patch whose list is the wrong length is an error, never silently truncated or padded."""
    body = _field_text(_nonuniform([1.0]), {"inlet": _nonuniform([1.0, 2.0])})

    with pytest.raises(ValueError, match="patch 'inlet'"):
        parse_scalar_field(body, 1, {"inlet": 3})


def test_parse_rejects_a_missing_patch() -> None:
    """A patch present on the mesh but absent from the file is an error, not a zero fill."""
    body = _field_text(_nonuniform([1.0]), {"inlet": "uniform 0"})

    with pytest.raises(ValueError, match="no boundaryField entry for patch 'outlet'"):
        parse_scalar_field(body, 1, {"outlet": 2})


def test_values_land_on_the_faces_they_were_written_for(tmp_path) -> None:
    """The placement property, on the real import path: value ``i`` lands on face ``i``.

    Each value encodes its own face index, so any permutation -- a patch written in the wrong order,
    an off-by-one at the internal/boundary join -- shows up as a mismatch rather than as a
    plausible-looking field.
    """
    mesh = read_openfoam(FIXTURE)
    interior = np.asarray(mesh.face_cells.interior)
    n_internal = int(interior.sum())

    patches = {}
    for name in mesh.face_patches.names:
        indices = np.asarray(mesh.face_patches.indices(name))
        if indices.size:
            patches[name] = _nonuniform([float(i) for i in indices])
    text = _field_text(_nonuniform([float(i) for i in range(n_internal)]), patches)
    path = tmp_path / "phi"
    path.write_text(text)

    values = read_surface_scalar_field(path, mesh)

    assert values.shape == (mesh.n_faces,)
    assert np.array_equal(values, np.arange(mesh.n_faces, dtype=np.float64))


def test_a_mesh_that_is_not_in_openfoam_order_is_refused(tmp_path) -> None:
    """The correspondence is checked, not assumed -- a mesh that breaks it raises.

    A generated grid does not interleave its faces the way an imported polyMesh does, so it is a
    standing example of the ordering this reader must refuse rather than silently mis-place values
    on. The same guard is what stops a collapsed 2D case being read, since that transform rebuilds
    and renumbers the mesh.
    """
    mesh = structured_grid_2d(3, 3, named_boundaries=True)
    interior = np.asarray(mesh.face_cells.interior)
    if interior[: int(interior.sum())].all() and not interior[int(interior.sum()) :].any():
        pytest.skip("this generated grid happens to be in interior-first order")
    path = tmp_path / "phi"
    path.write_text(_field_text("uniform 0", {}))

    with pytest.raises(ValueError, match=r"not OpenFOAM's|not a contiguous block"):
        read_surface_scalar_field(path, mesh)


def test_a_volume_field_reads_its_internal_block_onto_cells(tmp_path) -> None:
    """A ``volScalarField`` places by CELL, so it needs no ordering guard and reads no patches.

    Cell indices come straight from the ``owner``/``neighbour`` labels the assembler reads, so they
    are OpenFOAM's own by construction -- unlike face indices, which rest on the interior-first
    convention the surface reader has to check. The value written here encodes its own cell index,
    so a permutation would show as a mismatch rather than as a plausible field.
    """
    mesh = read_openfoam(FIXTURE)
    path = tmp_path / "nut"
    path.write_text(_field_text(_nonuniform(list(range(mesh.n_cells))), {}))

    values = read_volume_scalar_field(path, mesh)

    assert values.shape == (mesh.n_cells,)
    assert np.array_equal(values, np.arange(mesh.n_cells, dtype=np.float64))


def test_a_volume_field_of_the_wrong_length_is_refused(tmp_path) -> None:
    """A field written on a different mesh is a length mismatch, and must raise rather than pad."""
    mesh = read_openfoam(FIXTURE)
    path = tmp_path / "nut"
    path.write_text(_field_text(_nonuniform([1.0] * (mesh.n_cells + 1)), {}))

    with pytest.raises(ValueError, match="internalField"):
        read_volume_scalar_field(path, mesh)
