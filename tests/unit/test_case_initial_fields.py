"""A case's ``Fields`` starting state: the cells of an OpenFOAM time directory, in this case's units.

What a run does with one -- writing a time directory, then starting another run from it -- is
``tests/integration/test_case_run.py``'s. These pin the reading: that each file lands on the right
cells and components, that the pressure is taken from OpenFOAM's per-unit-density form to the case's
own, and each way a time directory is refused.
"""

from __future__ import annotations

from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.case import Fields, case_spec_from_mapping, case_spec_to_mapping
from aquaflux.case.initial import StartingFields

from tests.support.polymesh import copy_slab_polymesh


def _sections(density: float = 1.0, physics: str = "Laminar", **overrides: object) -> dict:
    """The two-cell OpenFOAM slab as a channel, as a file would state it."""
    rans = physics == "RANS"
    inlet = {"kind": "Inlet", "velocity": [1.0, 0.0]}
    if rans:
        inlet["turbulence"] = {"kind": "FixedTurbulence", "k": 1e-3, "omega": 2.0}
    return {
        "mesh": {"kind": "OpenFOAMMesh", "path": "of/constant/polyMesh"},
        "fluid": {"density": density, "kinematic_viscosity": 1.0e-2},
        "physics": {"kind": physics},
        "boundaries": {
            "left": inlet,
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
        },
        "numerics": {
            "momentum_advection": {"kind": "FirstOrderUpwind"},
            **({"turbulence_advection": {"kind": "FirstOrderUpwind"}} if rans else {}),
        },
        **overrides,
    }


def _field(directory: Path, name: str, kind: str, internal: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        f"FoamFile\n{{\n    format ascii;\n    class {kind};\n    object {name};\n}}\n"
        f"dimensions [0 0 0 0 0 0 0];\ninternalField {internal};\n"
        "boundaryField\n{\n    left { type zeroGradient; }\n}\n"
    )


def _time_directory(root: Path, time: str = "5", fields=("U", "p"), thickness: float = 0.5) -> Path:
    """An OpenFOAM case with the slab's mesh and a time directory whose values name their cell."""
    copy_slab_polymesh(root / "of" / "constant" / "polyMesh", thickness)
    directory = root / "of" / time
    values = {
        "U": ("volVectorField", "nonuniform List<vector> 2\n(\n(1.5 0.5 0)\n(2.5 -0.5 0)\n)"),
        "p": ("volScalarField", "nonuniform List<scalar> 2\n(\n10\n30\n)"),
        "k": ("volScalarField", "nonuniform List<scalar> 2\n(\n0.1\n0.2\n)"),
        "omega": ("volScalarField", "nonuniform List<scalar> 2\n(\n5\n7\n)"),
    }
    for name in fields:
        _field(directory, name, *values[name])
    return root


def _read(root: Path, spec_sections: dict, state: Fields) -> StartingFields:
    spec = case_spec_from_mapping(spec_sections)
    mesh = spec.mesh.read(root).validate()
    return state.read(root, spec, mesh)


def test_each_field_lands_on_its_cell_with_the_dropped_component_taken_out(tmp_path) -> None:
    """The slab is extruded along z, so the file's third slot (zero) goes and (x, y) stay in order."""
    root = _time_directory(tmp_path)
    start = _read(root, _sections(), Fields(path="of", time="5"))
    np.testing.assert_array_equal(start.fields["U"], [[1.5, 0.5], [2.5, -0.5]])
    np.testing.assert_array_equal(start.fields["p"], [10.0, 30.0])
    assert set(start.fields) == {"U", "p"}
    assert start.resumption is None
    assert start.source["kind"] == "Fields"
    assert start.source["file"] == str(root / "of" / "5")


def test_the_pressure_is_the_files_per_unit_density_pressure_times_the_density(tmp_path) -> None:
    """At density 2 a file value of 10 is a pressure of 20; reading it unscaled is the wrong answer."""
    root = _time_directory(tmp_path)
    start = _read(root, _sections(density=2.0), Fields(path="of", time="5"))
    np.testing.assert_array_equal(start.fields["p"], [20.0, 60.0])
    np.testing.assert_array_equal(start.fields["U"], [[1.5, 0.5], [2.5, -0.5]])  # only p is scaled
    assert start.source["density"] == 2.0


def test_the_dropped_axis_comes_from_the_cases_mesh_not_from_the_fields_directory(tmp_path) -> None:
    """The polyMesh the case reads need not sit in the case that holds the time directory."""
    root = _time_directory(tmp_path)
    (root / "of" / "constant").rename(root / "mesh_case_constant")
    (root / "mesh_case").mkdir()
    (root / "mesh_case_constant").rename(root / "mesh_case" / "constant")
    sections = _sections() | {
        "mesh": {"kind": "OpenFOAMMesh", "path": "mesh_case/constant/polyMesh"}
    }
    start = _read(root, sections, Fields(path="of", time="5"))
    np.testing.assert_array_equal(start.fields["U"], [[1.5, 0.5], [2.5, -0.5]])


def test_a_mesh_whose_extruded_axis_is_ambiguous_reads_where_the_file_says(tmp_path) -> None:
    """A slab as thick as it is tall cannot say which axis was extruded; the case file can.

    Wrong answers this catches: a stated axis that is ignored (the ambiguity then raises), a letter
    mapped to the wrong component (``y`` would drop the nonzero component and be refused), and an axis
    that reaches the reader but not the dropped component.
    """
    root = _time_directory(tmp_path, thickness=1.0)
    with pytest.raises(ValueError, match="pass extruded_axis explicitly"):
        _read(root, _sections(), Fields(path="of", time="5"))
    start = _read(root, _sections(), Fields(path="of", time="5", extruded_axis="z"))
    np.testing.assert_array_equal(start.fields["U"], [[1.5, 0.5], [2.5, -0.5]])
    with pytest.raises(ValueError, match="nonzero component along axis 1"):
        _read(root, _sections(), Fields(path="of", time="5", extruded_axis="y"))


def test_an_extruded_axis_on_a_starting_state_is_refused_for_a_three_dimensional_mesh() -> None:
    with pytest.raises(ValueError, match=r"initial: extruded_axis .* three-dimensional"):
        Fields(path="of", time="5", extruded_axis="z").refuse_for_dimension(3)
    Fields(path="of", time="5", extruded_axis="z").refuse_for_dimension(2)
    Fields(path="of", time="5").refuse_for_dimension(3)


def test_a_rans_case_reads_the_closures_fields_too(tmp_path) -> None:
    root = _time_directory(tmp_path, fields=("U", "p", "k", "omega"))
    start = _read(root, _sections(physics="RANS"), Fields(path="of", time="5"))
    assert set(start.fields) == {"U", "p", "k", "omega"}
    np.testing.assert_array_equal(start.fields["omega"], [5.0, 7.0])


def test_a_field_the_physics_needs_but_the_directory_lacks_is_named(tmp_path) -> None:
    root = _time_directory(tmp_path, fields=("U", "p"))
    with pytest.raises(FileNotFoundError, match=r"no field 'k' in .*; it holds \['U', 'p'\]"):
        _read(root, _sections(physics="RANS"), Fields(path="of", time="5"))


def test_a_time_that_is_not_there_is_named(tmp_path) -> None:
    root = _time_directory(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"no time directory .*of/9"):
        _read(root, _sections(), Fields(path="of", time="9"))


def test_a_value_that_is_not_finite_is_refused_naming_the_field(tmp_path) -> None:
    root = _time_directory(tmp_path)
    _field(root / "of" / "5", "p", "volScalarField", "nonuniform List<scalar> 2\n(\n10\nnan\n)")
    with pytest.raises(ValueError, match=r"not finite in \['p'\]"):
        _read(root, _sections(), Fields(path="of", time="5"))


def test_a_case_on_another_kind_of_mesh_is_refused(tmp_path) -> None:
    """The cells of an OpenFOAM case are numbered by its mesh, which a generated grid does not share."""
    root = _time_directory(tmp_path)
    grid = {"kind": "StructuredGrid", "cells": [2, 1], "lengths": [2.0, 1.0]}
    spec = case_spec_from_mapping(_sections() | {"mesh": grid})
    mesh = spec.mesh.read(root).validate()
    with pytest.raises(ValueError, match="must be that case's OpenFOAMMesh, not a StructuredGrid"):
        Fields(path="of", time="5").read(root, spec, mesh)


def test_the_section_reads_and_writes_as_a_file_states_it() -> None:
    sections = _sections(initial={"kind": "Fields", "path": "of", "time": "5"})
    spec = case_spec_from_mapping(sections)
    assert spec.initial == Fields(path="of", time="5")
    assert case_spec_to_mapping(spec)["initial"] == {"kind": "Fields", "path": "of", "time": "5"}


def test_the_extruded_axis_reads_and_writes_as_a_file_states_it() -> None:
    state = {"kind": "Fields", "path": "of", "time": "5", "extruded_axis": "z"}
    spec = case_spec_from_mapping(_sections(initial=state))
    assert spec.initial == Fields(path="of", time="5", extruded_axis="z")
    assert case_spec_to_mapping(spec)["initial"] == state
    with pytest.raises(ValueError, match=r"initial.*extruded_axis"):
        case_spec_from_mapping(_sections(initial=state | {"extruded_axis": "w"}))


def test_a_time_written_as_a_number_is_refused_and_says_to_quote_it() -> None:
    with pytest.raises(ValueError, match=r"initial.*time"):
        case_spec_from_mapping(_sections(initial={"kind": "Fields", "path": "of", "time": 5}))


def test_an_empty_path_or_time_is_refused() -> None:
    with pytest.raises(ValueError, match=r"Fields\.path names where"):
        Fields(path="", time="5")
    with pytest.raises(ValueError, match="names the time directory"):
        Fields(path="of", time="")


def test_a_radiation_case_refuses_the_section() -> None:
    radiation = {
        "mesh": {"kind": "StructuredGrid", "cells": [2, 2, 2], "lengths": [1.0, 1.0, 1.0]},
        "physics": {
            "kind": "Radiation",
            "medium": {"kind": "UniformMedium", "absorption": 1.0},
        },
        "boundaries": {
            "left": {
                "kind": "Lamp",
                "profile": {"kind": "LambertianProfile"},
                "power": 1.0,
            },
            **{name: {"kind": "Wall"} for name in ("right", "bottom", "top", "back", "front")},
        },
        "initial": {"kind": "Fields", "path": "of", "time": "5"},
    }
    with pytest.raises(ValueError, match="initial"):
        case_spec_from_mapping(radiation)
