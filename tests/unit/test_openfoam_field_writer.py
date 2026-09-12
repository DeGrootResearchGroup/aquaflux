"""Unit tests for writing computed cell fields back into an OpenFOAM case.

Split the way the reader is: :func:`parse_field_template` and :func:`format_volume_field` are pure
and test on string snippets, while :func:`write_openfoam_time` is exercised end to end against the
committed polyMesh fixture with a template field beside it.

Two properties carry the feature and are what these tests are built around. **The values must land
on the cells they came from** -- the internal block is written in mesh cell order and read back in
the same order, so a round trip through the reader must be exact rather than close. And **the
boundary conditions must survive verbatim**, because that is what makes a written directory a
restart state rather than merely a picture: a patch type this package never models (``empty``, which
the two-dimensional collapse removes from the mesh entirely) has to come out the other side intact.
"""

from __future__ import annotations

from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.io.openfoam import (
    format_volume_field,
    infer_extruded_axis,
    parse_field_template,
    parse_scalar_field,
    read_field_template,
    write_openfoam_field,
    write_openfoam_time,
)
from aquaflux.mesh import structured_grid_2d, structured_grid_3d

_SCALAR_TEMPLATE = """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 5;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""

_NESTED_TEMPLATE = """
dimensions      [0 1 -1 0 0 0 0];

boundaryField
{
    outlet
    {
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           nonuniform List<vector>
2
(
(1 2 3)
(4 5 6)
)
;
    }
}
"""


def _template(text: str, class_name: str = "volScalarField"):
    return parse_field_template(text, {"class": class_name, "object": "q"})


class TestParseFieldTemplate:
    def test_it_takes_the_class_dimensions_and_patch_blocks(self):
        template = _template(_SCALAR_TEMPLATE)
        assert template.class_name == "volScalarField"
        assert template.dimensions == "[0 2 -2 0 0 0 0]"
        assert list(template.boundary) == ["inlet", "frontAndBack"]
        assert "fixedValue" in template.boundary["inlet"]

    def test_it_refuses_a_class_it_cannot_write(self):
        with pytest.raises(ValueError, match="surfaceScalarField"):
            _template(_SCALAR_TEMPLATE, "surfaceScalarField")

    def test_it_refuses_a_header_with_no_class(self):
        with pytest.raises(ValueError, match="no 'class' entry"):
            parse_field_template(_SCALAR_TEMPLATE, {})

    def test_it_refuses_a_body_with_no_dimensions(self):
        with pytest.raises(ValueError, match="no 'dimensions' entry"):
            _template("boundaryField { }")


_MACRO_TEMPLATE = """
dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 440.15;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           $internalField;
    }
    upperWall
    {
        type            omegaWallFunction;
        value           ${internalField};
    }
}
"""

# A patch dictionary as a solver writes one: the list's count, parentheses and entries sit at
# column zero, below entries that are indented normally.
_COLUMN_ZERO_TEMPLATE = """
dimensions      [0 0 -1 0 0 0 0];

boundaryField
{
    upperWall
    {
        beta1           0.075;
        type            omegaWallFunction;
        value           nonuniform List<scalar>
3
(
18117.5
23573.4
25568.8
)
;
    }
}
"""


class TestFormatVolumeField:
    def test_a_scalar_field_reads_back_through_the_field_parser(self):
        values = np.array([1.0, -2.5, 3.75e-8, 1e12])
        text = format_volume_field(
            values, _template(_SCALAR_TEMPLATE), object_name="p", location="7"
        )
        body = text.split("* //", 1)[1]
        assert np.array_equal(parse_scalar_field(body, len(values), {}), values)

    def test_the_header_and_dimensions_come_from_the_template(self):
        text = format_volume_field(
            np.zeros(3), _template(_SCALAR_TEMPLATE), object_name="p", location="7"
        )
        assert "class       volScalarField;" in text
        assert 'location    "7";' in text
        assert "object      p;" in text
        assert "dimensions      [0 2 -2 0 0 0 0];" in text

    def test_a_patch_type_the_package_does_not_model_survives_verbatim(self):
        text = format_volume_field(
            np.zeros(3), _template(_SCALAR_TEMPLATE), object_name="p", location="7"
        )
        assert "frontAndBack" in text
        assert "type            empty;" in text

    def test_a_nested_patch_entry_keeps_its_structure(self):
        # Re-indenting must not flatten a block: an `inletOutlet` carries a multi-line list, and a
        # per-line strip would merge it into the entry above.
        text = format_volume_field(
            np.zeros((2, 3)),
            _template(_NESTED_TEMPLATE, "volVectorField"),
            object_name="U",
            location="7",
        )
        block = text.split("outlet", 1)[1]
        assert "inletValue      uniform (0 0 0);" in block
        assert "(1 2 3)" in block and "(4 5 6)" in block

    def test_a_two_dimensional_vector_is_padded_on_the_named_axis(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        template = _template(_NESTED_TEMPLATE, "volVectorField")
        last = format_volume_field(values, template, object_name="U", location="7")
        assert "(1 2 0)" in last and "(3 4 0)" in last
        middle = format_volume_field(
            values, template, object_name="U", location="7", extruded_axis=1
        )
        assert "(1 0 2)" in middle and "(3 0 4)" in middle

    def test_it_refuses_an_axis_that_names_no_component(self):
        # Normalizing first would make 5 mean 2 and write a plausible file; the point of the
        # argument is that the caller knows which axis the collapse removed.
        with pytest.raises(ValueError, match="extruded_axis must be one of"):
            format_volume_field(
                np.zeros((2, 2)),
                _template(_NESTED_TEMPLATE, "volVectorField"),
                object_name="U",
                location="7",
                extruded_axis=5,
            )

    def test_a_three_component_vector_is_written_unchanged(self):
        text = format_volume_field(
            np.array([[1.0, 2.0, 3.0]]),
            _template(_NESTED_TEMPLATE, "volVectorField"),
            object_name="U",
            location="7",
        )
        assert "(1 2 3)" in text

    @pytest.mark.parametrize(
        ("values", "class_name"),
        [
            (np.zeros((3, 2)), "volScalarField"),
            (np.zeros(3), "volVectorField"),
            (np.zeros((3, 4)), "volVectorField"),
        ],
    )
    def test_it_refuses_values_whose_shape_the_class_cannot_hold(self, values, class_name):
        with pytest.raises(ValueError, match="expects values of shape"):
            format_volume_field(
                values, _template(_NESTED_TEMPLATE, class_name), object_name="q", location="7"
            )

    def test_it_refuses_a_non_finite_field_by_default(self):
        values = np.array([1.0, np.nan, np.inf])
        with pytest.raises(ValueError, match="2 non-finite"):
            format_volume_field(values, _template(_SCALAR_TEMPLATE), object_name="p", location="7")

    def test_a_non_finite_field_can_be_written_deliberately(self):
        values = np.array([1.0, np.nan])
        text = format_volume_field(
            values,
            _template(_SCALAR_TEMPLATE),
            object_name="p",
            location="7",
            allow_non_finite=True,
        )
        assert "nan" in text


class TestCopiedPatchDictionaries:
    """The two shapes a real template puts in a patch block, both found by running the solver."""

    def test_the_internalField_macro_resolves_against_the_TEMPLATE_not_the_written_values(self):
        # `value $internalField;` is how a `0` directory gives a patch the interior value, and it is
        # only meaningful while that value is uniform. Carried through literally it expands every
        # written cell onto the patch, and the solver rejects the file.
        text = format_volume_field(
            np.arange(5.0), _template(_MACRO_TEMPLATE), object_name="omega", location="7"
        )
        assert "$internalField" not in text
        assert "${internalField}" not in text
        assert text.count("value           uniform 440.15;") == 2

    def test_it_says_so_when_the_macro_cannot_be_resolved(self):
        no_internal = _MACRO_TEMPLATE.replace("internalField   uniform 440.15;", "")
        with pytest.raises(ValueError, match=r"refers to \$internalField"):
            format_volume_field(
                np.arange(5.0), _template(no_internal), object_name="omega", location="7"
            )

    def test_a_list_written_at_column_zero_is_indented_with_its_own_entry(self):
        # Measuring the smallest indent in the block would measure the list and dedent nothing,
        # leaving the entries above it twice as deep as the ones below.
        text = format_volume_field(
            np.arange(5.0), _template(_COLUMN_ZERO_TEMPLATE), object_name="omega", location="7"
        )
        block = text.split("upperWall", 1)[1].split("\n    }", 1)[0]
        body = [line for line in block.splitlines() if line.strip() and line.strip() != "{"]
        indents = {len(line) - len(line.lstrip()) for line in body}
        assert indents == {8}, indents
        assert "18117.5" in text and "25568.8" in text


def _polymesh_with_extents(root: Path, extents) -> Path:
    """A case whose polyMesh ``points`` span ``extents`` -- enough for the axis to be recovered."""
    directory = root / "constant" / "polyMesh"
    directory.mkdir(parents=True, exist_ok=True)
    corners = np.array([[0.0, 0.0, 0.0], list(extents)])
    body = "\n".join(f"({x} {y} {z})" for x, y, z in corners)
    (directory / "points").write_text(
        "FoamFile\n{\n    format ascii;\n    class vectorField;\n    object points;\n}\n"
        f"{len(corners)}\n(\n{body}\n)\n"
    )
    return root


class TestInferExtrudedAxis:
    """The collapsed mesh cannot answer this; the case's own polyMesh can."""

    @pytest.mark.parametrize("dropped", [0, 1, 2])
    def test_it_recovers_each_of_the_three_axes(self, tmp_path, dropped):
        # The 2D mesh is the same in all three arms -- which is the point. Only the polyMesh
        # distinguishes them, and the default (last axis) is right in exactly one.
        mesh = structured_grid_2d(3, 2)
        planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
        extents = np.insert(planar, dropped, 0.01)
        case = _polymesh_with_extents(tmp_path / f"drop{dropped}", extents)
        assert infer_extruded_axis(case, mesh) == dropped

    def test_it_refuses_a_mesh_that_was_never_collapsed(self, tmp_path):
        mesh = structured_grid_3d(2, 2, 1)
        with pytest.raises(ValueError, match="expected a 2D mesh"):
            infer_extruded_axis(tmp_path, mesh)

    def test_it_says_so_rather_than_guessing_when_the_extents_cannot_decide(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
        # A cube of the mesh's own extent: every candidate matches, so no axis is identifiable.
        case = _polymesh_with_extents(tmp_path, [planar[0], planar[0], planar[0]])
        with pytest.raises(ValueError, match="pass extruded_axis explicitly"):
            infer_extruded_axis(case, mesh)


class TestWriteOpenfoamTime:
    @staticmethod
    def _case(tmp_path, names=("p",), class_name="volScalarField", body=_SCALAR_TEMPLATE):
        zero = tmp_path / "0"
        zero.mkdir(parents=True)
        for name in names:
            (zero / name).write_text(
                f"FoamFile\n{{\n    format ascii;\n    class {class_name};\n"
                f"    object {name};\n}}\n{body}"
            )
        return tmp_path

    def test_it_writes_one_file_per_field_into_the_named_time(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path, names=("p", "q"))
        out = write_openfoam_time(
            case, 100, {"p": np.zeros(mesh.n_cells), "q": np.ones(mesh.n_cells)}, mesh
        )
        assert out == case / "100"
        assert sorted(f.name for f in out.iterdir()) == ["p", "q"]

    def test_the_values_round_trip_exactly_through_the_reader(self, tmp_path):
        mesh = structured_grid_2d(3, 2)
        case = self._case(tmp_path)
        # A value that encodes its own cell index, so a permutation is visible rather than plausible.
        values = np.arange(mesh.n_cells, dtype=float) * 1.5 - 0.25
        out = write_openfoam_time(case, 100, {"p": values}, mesh)
        from aquaflux.io.openfoam import read_volume_scalar_field

        assert np.array_equal(read_volume_scalar_field(out / "p", mesh), values)

    def test_a_written_field_is_itself_a_usable_template(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path)
        out = write_openfoam_time(case, 100, {"p": np.zeros(mesh.n_cells)}, mesh)
        again = read_field_template(out / "p")
        original = read_field_template(case / "0" / "p")
        assert again.class_name == original.class_name
        assert again.dimensions == original.dimensions
        assert list(again.boundary) == list(original.boundary)

    def test_it_refuses_a_field_of_the_wrong_length(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path)
        with pytest.raises(ValueError, match=f"but the mesh has {mesh.n_cells} cells"):
            write_openfoam_time(case, 100, {"p": np.zeros(mesh.n_cells + 1)}, mesh)

    def test_it_names_the_missing_template(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"no template .*absent"):
            write_openfoam_time(case, 100, {"absent": np.zeros(mesh.n_cells)}, mesh)

    def test_it_names_the_missing_template_directory(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        with pytest.raises(FileNotFoundError, match="no template time directory"):
            write_openfoam_time(tmp_path, 100, {"p": np.zeros(mesh.n_cells)}, mesh)

    def test_a_2d_vector_is_padded_on_the_axis_the_polyMesh_says(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path, names=("U",), class_name="volVectorField")
        planar = np.ptp(np.asarray(mesh.node_coords, dtype=float), axis=0)
        _polymesh_with_extents(case, np.insert(planar, 1, 0.01))  # extruded along y, not z
        values = np.tile([1.0, 3.0], (mesh.n_cells, 1))
        out = write_openfoam_time(case, 100, {"U": values}, mesh)
        # Default convention would have written (1 3 0); the polyMesh says the zero belongs in y.
        assert "(1 0 3)" in (out / "U").read_text()

    def test_a_scalar_only_write_never_consults_the_polyMesh(self, tmp_path):
        # No polyMesh exists here at all, so this passes only if nothing tried to read one.
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path)
        out = write_openfoam_time(case, 100, {"p": np.zeros(mesh.n_cells)}, mesh)
        assert (out / "p").is_file()

    def test_an_explicit_axis_overrides_the_polyMesh(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path, names=("U",), class_name="volVectorField")
        values = np.tile([1.0, 3.0], (mesh.n_cells, 1))
        out = write_openfoam_time(case, 100, {"U": values}, mesh, extruded_axis=0)
        assert "(0 1 3)" in (out / "U").read_text()

    def test_a_field_can_be_written_with_a_template_of_another_name(self, tmp_path):
        mesh = structured_grid_2d(2, 2)
        case = self._case(tmp_path)
        written = write_openfoam_field(
            case / "100" / "pMean", np.zeros(mesh.n_cells), template=case / "0" / "p"
        )
        assert "object      pMean;" in written.read_text()
        assert 'location    "100";' in written.read_text()
