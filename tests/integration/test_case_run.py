"""A case file run end to end, as ``aquaflux run case.yaml`` runs it: solved, and its outputs written.

Each run's results are compared against the library solve called directly on the same problem, so a
field written from anything but the converged root -- a stale state, the seed, the wrong block -- fails.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import yaml
from aquaflux.__main__ import main
from aquaflux.case import FlowMarch, read_case
from aquaflux.flow import solve_flow_march
from aquaflux.io import read_volume_scalar_field

REPO = Path(__file__).resolve().parents[2]
SLAB = REPO / "tests" / "fixtures" / "polymesh_2d_slab_frontandback"

#: A laminar channel at a Reynolds number of about 50, fed a uniform velocity at its left side.
_CHANNEL = {
    "mesh": {"kind": "StructuredGrid", "cells": [8, 4], "lengths": [2.0, 1.0]},
    "fluid": {"density": 1.0, "kinematic_viscosity": 2.0e-2},
    "physics": {"kind": "Laminar"},
    "boundaries": {
        "left": {"kind": "Inlet", "velocity": [1.0, 0.0]},
        "right": {"kind": "Outlet", "pressure": 0.0},
        "bottom": {"kind": "Wall"},
        "top": {"kind": "Wall"},
    },
    "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
}

#: A pressure template for the slab fixture's patches, as an OpenFOAM case's ``0/p`` states it.
_PRESSURE_TEMPLATE = """FoamFile
{
    format ascii;
    class volScalarField;
    object p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    left { type zeroGradient; }
    right { type fixedValue; value uniform 0; }
    bottom { type zeroGradient; }
    top { type zeroGradient; }
    frontAndBack { type empty; }
}
"""


def _write(path: Path, sections: dict) -> Path:
    path.write_text(yaml.safe_dump(sections))
    return path


def test_a_run_writes_the_converged_fields_its_log_its_checkpoints_and_its_records(
    tmp_path,
) -> None:
    sections = _CHANNEL | {"outputs": {"checkpoints": {"kind": "Checkpoints", "keep": 1}}}
    case = _write(tmp_path / "case.yaml", sections)
    assert main(["run", str(case)]) == 0
    results = tmp_path / "results"
    assert sorted(path.name for path in results.iterdir()) == [
        "case.yaml",
        "checkpoints",
        "fields.vtu",
        "march.log",
        "run.yaml",
    ]

    # The root the library reaches by itself on the same problem.
    checked = read_case(case).check()
    problem = checked.build()
    root = np.asarray(solve_flow_march(problem))
    (checkpoint,) = (results / "checkpoints").iterdir()
    np.testing.assert_array_equal(np.load(checkpoint)["state"], root)

    record = yaml.safe_load((results / "run.yaml").read_text())
    assert record["converged"] is True and record["message"] is None
    assert record["solver"] == "FlowMarch"
    assert record["steps"] == int(checkpoint.stem.split("-")[1])
    assert record["residual"] < 1e-8
    assert record["written"] == ["fields.vtu", "march.log", "checkpoints", "case.yaml"]

    vtu = (results / "fields.vtu").read_text(errors="replace")
    assert 'Name="U"' in vtu and 'Name="p"' in vtu
    log = (results / "march.log").read_text()
    assert log.count("\n|") >= record["steps"]  # one table row per step, at least

    # The case as it ran: the default solver written out, the outputs pointing at themselves.
    ran = read_case(results / "case.yaml").spec
    assert ran.solver == FlowMarch()
    assert ran.outputs.directory == "."
    assert ran.mesh == checked.spec.mesh and ran.boundaries == checked.spec.boundaries


def test_a_run_that_stops_short_writes_no_fields_and_says_so(tmp_path) -> None:
    solver = {"kind": "FlowMarch", "max_steps": 1}
    case = _write(tmp_path / "case.yaml", _CHANNEL | {"solver": solver})
    assert main(["run", str(case)]) == 1
    results = tmp_path / "results"
    assert not (results / "fields.vtu").exists()
    record = yaml.safe_load((results / "run.yaml").read_text())
    assert record["converged"] is False
    assert record["message"]
    assert record["steps"] == 1
    assert "did not converge" in (results / "march.log").read_text()


def test_a_run_writes_an_openfoam_time_directory_the_case_can_restart_from(tmp_path) -> None:
    of = tmp_path / "of"
    shutil.copytree(SLAB, of / "constant" / "polyMesh")
    (of / "0").mkdir()
    (of / "0" / "p").write_text(_PRESSURE_TEMPLATE)
    sections = _CHANNEL | {
        "mesh": {"kind": "OpenFOAMMesh", "path": "of/constant/polyMesh"},
        # The two-cell slab leaves the default reconstruction underdetermined; the compact one is not.
        "numerics": {
            "momentum_advection": {"kind": "FirstOrderUpwind"},
            "gradient": {"kind": "CompactGreenGauss"},
        },
        "outputs": {
            "directory": "out",
            "fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "5", "fields": ["p"]}],
        },
    }
    case = _write(tmp_path / "case.yaml", sections)
    assert main(["run", str(case)]) == 0

    checked = read_case(case).check()
    problem = checked.build()
    _, pressure = problem.unpack(solve_flow_march(problem))
    written = read_volume_scalar_field(of / "5" / "p", checked.mesh)
    np.testing.assert_allclose(written, np.asarray(pressure), rtol=1e-12, atol=0.0)

    # The record's relative paths were re-based on its own directory, so it checks where it lies.
    ran = read_case(tmp_path / "out" / "case.yaml")
    assert ran.spec.mesh.path == "../of/constant/polyMesh"
    assert ran.check().mesh.n_cells == checked.mesh.n_cells
    assert ran.spec.outputs.fields[0].case == "../of"
