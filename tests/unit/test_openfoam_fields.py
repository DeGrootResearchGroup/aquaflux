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
    parse_vector_field,
    read_openfoam_time,
    read_surface_scalar_field,
    read_volume_scalar_field,
    write_openfoam_time,
)
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.io.openfoam.reader import read_openfoam
from aquaflux.mesh import structured_grid_2d, structured_grid_3d

from tests.support.polymesh import two_cube_polymesh_data
from tests.unit.test_openfoam_field_writer import _polymesh_with_extents

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


def test_a_patch_declared_out_of_ascending_start_face_order_still_lands_correctly(
    tmp_path,
) -> None:
    """A ``boundary`` file need not declare its patches in ascending-startFace order.

    The two-cube fixture on disk happens to declare ``inlet`` (startFace 1) before ``outlet``
    (startFace 2), which is also their ascending face order -- so nothing distinguishes "laid out
    by startFace" from "laid out in file/declaration order." Here ``outlet`` is declared before
    ``inlet`` while both keep their real face ranges (a legal, if less common, OpenFOAM layout), so
    ``mesh.face_patches.names`` yields them in that same non-ascending order. The reader must still
    place each patch's values on its own face range, keyed on face index rather than on declaration
    order -- exactly the property the ascending-order sort exists to guarantee.
    """
    original = two_cube_polymesh_data()
    inlet, outlet, walls = original.patches
    assert inlet.start_face < outlet.start_face
    mesh = assemble(original._replace(patches=(outlet, inlet, walls)))
    declared = [name for name in mesh.face_patches.names if name not in ("interior", "boundary")]
    assert declared == ["outlet", "inlet", "walls"]

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


def test_a_periodic_mesh_is_refused(tmp_path) -> None:
    """A periodic seam (a fused ``cyclic`` patch pair, or a generator's ``periodic=``) is refused.

    Its seam face is interior but sat in whichever boundary patch declared it in the original
    file, which can coincidentally still pass the leading-block check above -- so this is a direct
    check on ``neighbour_offset`` instead of relying on that one to catch every case.
    """
    mesh = structured_grid_2d(3, 3, periodic=("x",))
    path = tmp_path / "phi"
    path.write_text(_field_text("uniform 0", {}))

    with pytest.raises(ValueError, match="periodic seam"):
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


# --- reading a time directory's cell fields ---------------------------------------------------------


def _field_file(directory: Path, name: str, kind: str, internal: str) -> None:
    """A field file with the given ``internalField`` text and one patch, as a case would hold."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        f"FoamFile\n{{\n    format ascii;\n    class {kind};\n    object {name};\n}}\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        f"internalField {internal};\n"
        "boundaryField\n{\n    wall\n    {\n        type fixedValue;\n"
        "        value uniform (7 7 7);\n    }\n}\n"
    )


def _vector_list(rows) -> str:
    return (
        f"nonuniform List<vector> {len(rows)}\n(\n"
        + "\n".join(f"({x:.17g} {y:.17g} {z:.17g})" for x, y, z in rows)
        + "\n)"
    )


class TestParseVectorField:
    def test_it_reads_the_list_form_with_each_vector_in_cell_order(self):
        body = "internalField " + _vector_list([(1, 2, 3), (4, 5, 6)]) + ";\nboundaryField { }"
        np.testing.assert_array_equal(parse_vector_field(body, 2), [[1, 2, 3], [4, 5, 6]])

    def test_it_reads_a_uniform_vector_onto_every_cell(self):
        np.testing.assert_array_equal(
            parse_vector_field("internalField uniform (1 0 -2.5);", 3),
            np.tile([1, 0, -2.5], (3, 1)),
        )

    def test_it_reads_the_internal_block_not_a_patchs_value(self):
        """A patch's own ``value`` in the same file is a different quantity and must not be read."""
        body = (
            "internalField uniform (1 2 3);\nboundaryField\n{\n  wall\n  {\n"
            "    value nonuniform List<vector> 1((9 9 9));\n  }\n}\n"
        )
        np.testing.assert_array_equal(parse_vector_field(body, 2), np.tile([1, 2, 3], (2, 1)))

    def test_it_refuses_the_wrong_number_of_vectors(self):
        body = "internalField " + _vector_list([(1, 2, 3), (4, 5, 6)]) + ";"
        with pytest.raises(ValueError, match="2 values but the mesh has 3 cells"):
            parse_vector_field(body, 3)

    def test_it_refuses_a_uniform_entry_that_is_not_a_vector(self):
        with pytest.raises(ValueError, match="neither a 'uniform' nor a 'nonuniform' vector"):
            parse_vector_field("internalField uniform 5;", 3)

    def test_it_refuses_a_field_with_no_internal_entry(self):
        with pytest.raises(ValueError, match="no internalField"):
            parse_vector_field("boundaryField { }", 3)


class TestReadOpenfoamTime:
    @staticmethod
    def _slab(tmp_path):
        """A 3 x 2 collapsed mesh and a time directory ``5`` of fields whose values encode their cell."""
        mesh = structured_grid_2d(3, 2)
        n = mesh.n_cells
        index = np.arange(n, dtype=float)
        time = tmp_path / "5"
        scalars = "\n".join(f"{v:.17g}" for v in index * 1.5 + 0.25)
        _field_file(time, "p", "volScalarField", f"nonuniform List<scalar> {n}\n(\n{scalars}\n)")
        # A planar velocity extruded along y: the dropped component sits in the middle slot.
        _field_file(
            time, "U", "volVectorField", _vector_list([(i, 0.0, 10.0 + i) for i in range(n)])
        )
        return mesh, index

    def test_a_scalar_lands_on_the_cell_it_was_written_for(self, tmp_path):
        mesh, index = self._slab(tmp_path)
        fields = read_openfoam_time(tmp_path, 5, ["p"], mesh)
        np.testing.assert_array_equal(fields["p"], index * 1.5 + 0.25)

    def test_a_2d_vector_drops_the_axis_it_is_told_to(self, tmp_path):
        mesh, index = self._slab(tmp_path)
        fields = read_openfoam_time(tmp_path, "5", ["U"], mesh, extruded_axis=1)
        np.testing.assert_array_equal(fields["U"], np.column_stack([index, 10.0 + index]))

    def test_a_2d_vector_recovers_the_axis_from_the_cases_polymesh(self, tmp_path):
        mesh, index = self._slab(tmp_path)
        planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
        _polymesh_with_extents(tmp_path, np.insert(planar, 1, 0.01))  # extruded along y
        fields = read_openfoam_time(tmp_path, 5, ["U"], mesh)
        np.testing.assert_array_equal(fields["U"], np.column_stack([index, 10.0 + index]))

    def test_the_wrong_axis_is_refused_rather_than_dropping_a_real_component(self, tmp_path):
        """Dropping z here would silently discard the velocity the file holds in that slot."""
        mesh, _ = self._slab(tmp_path)
        with pytest.raises(ValueError, match="nonzero component along axis 2"):
            read_openfoam_time(tmp_path, 5, ["U"], mesh, extruded_axis=2)

    def test_a_scalar_only_read_never_consults_the_polymesh(self, tmp_path):
        mesh, _ = self._slab(tmp_path)  # no polyMesh exists: passes only if none was opened
        assert read_openfoam_time(tmp_path, 5, ["p"], mesh)["p"].shape == (mesh.n_cells,)

    def test_a_3d_vector_keeps_all_three_components(self, tmp_path):
        mesh = structured_grid_3d(2, 1, 1)
        _field_file(tmp_path / "1", "U", "volVectorField", _vector_list([(1, 2, 3), (4, 5, 6)]))
        fields = read_openfoam_time(tmp_path, 1, ["U"], mesh)
        np.testing.assert_array_equal(fields["U"], [[1, 2, 3], [4, 5, 6]])

    def test_a_field_written_by_the_writer_reads_back_exactly(self, tmp_path):
        """The two directions agree, on a vector as well, with a value in every slot a swap would show."""
        mesh = structured_grid_2d(3, 2)
        zero = tmp_path / "0"
        _field_file(zero, "U", "volVectorField", "uniform (0 0 0)")
        _field_file(zero, "p", "volScalarField", "uniform 0")
        planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
        _polymesh_with_extents(tmp_path, np.insert(planar, 2, 0.01))
        rng = np.random.default_rng(0)
        written = {"U": rng.normal(size=(mesh.n_cells, 2)), "p": rng.normal(size=mesh.n_cells)}
        write_openfoam_time(tmp_path, 7, written, mesh)
        read = read_openfoam_time(tmp_path, 7, ["U", "p"], mesh)
        for name, values in written.items():
            np.testing.assert_allclose(read[name], values, rtol=1e-11, err_msg=name)

    def test_a_missing_field_lists_what_the_directory_holds(self, tmp_path):
        mesh, _ = self._slab(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"no field 'k' in .*; it holds \['U', 'p'\]"):
            read_openfoam_time(tmp_path, 5, ["p", "k"], mesh)

    def test_a_missing_time_directory_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no time directory"):
            read_openfoam_time(tmp_path, 9, ["p"], structured_grid_2d(3, 2))

    def test_a_field_of_the_wrong_size_is_refused(self, tmp_path):
        self._slab(tmp_path)
        with pytest.raises(ValueError, match=r"internalField has 6 values but 4 are expected"):
            read_openfoam_time(tmp_path, 5, ["p"], structured_grid_2d(2, 2))

    def test_a_surface_field_is_refused_as_not_a_cell_field(self, tmp_path):
        _field_file(tmp_path / "3", "phi", "surfaceScalarField", "uniform 0")
        with pytest.raises(ValueError, match="'surfaceScalarField', not a cell field"):
            read_openfoam_time(tmp_path, 3, ["phi"], structured_grid_2d(2, 2))
