"""A case file run end to end, as ``aquaflux run case.yaml`` runs it: solved, and its outputs written.

Each run's results are compared against the library solve called directly on the same problem, so a
field written from anything but the converged root -- a stale state, the seed, the wrong block -- fails.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
import yaml
from aquaflux.__main__ import main
from aquaflux.case import FlowMarch, read_case
from aquaflux.case.restart_file import read_restart
from aquaflux.flow import solve_flow_march
from aquaflux.io import read_openfoam_time, read_volume_scalar_field

from tests.support.polymesh import copy_slab_polymesh

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


#: A velocity template for the same patches, as an OpenFOAM case's ``0/U`` states it.
_VELOCITY_TEMPLATE = """FoamFile
{
    format ascii;
    class volVectorField;
    object U;
}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    left { type fixedValue; value uniform (1 0 0); }
    right { type zeroGradient; }
    bottom { type noSlip; }
    top { type noSlip; }
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
        "history.csv",
        "march.log",
        "run.yaml",
    ]

    # The root the library reaches by itself on the same problem.
    checked = read_case(case).check()
    problem = checked.build()
    root = np.asarray(solve_flow_march(problem))
    (checkpoint,) = (results / "checkpoints").iterdir()
    restart = read_restart(checkpoint)
    velocity, pressure = problem.unpack(root)
    np.testing.assert_array_equal(restart.fields["U"], np.asarray(velocity))
    np.testing.assert_array_equal(restart.fields["p"], np.asarray(pressure))

    record = yaml.safe_load((results / "run.yaml").read_text())
    assert record["converged"] is True and record["message"] is None
    assert record["solver"] == "FlowMarch"
    assert record["steps"] == int(checkpoint.stem.split("-")[1])
    assert record["residual"] < 1e-8
    assert record["written"] == [
        "fields.vtu",
        "march.log",
        "history.csv",
        "checkpoints",
        "case.yaml",
    ]

    vtu = (results / "fields.vtu").read_text(errors="replace")
    assert 'Name="U"' in vtu and 'Name="p"' in vtu
    log = (results / "march.log").read_text()
    assert log.count("\n|") >= record["steps"]  # one table row per step, at least
    # One history row per step, the last ending where the run record says the march did.
    history = list(csv.DictReader((results / "history.csv").open()))
    assert [int(row["step"]) for row in history] == list(range(1, record["steps"] + 1))
    assert float(history[-1]["residual_norm"]) == record["residual"]

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


def _restart_case(tmp_path: Path, name: str, outputs: dict, **sections: object) -> Path:
    """A copy of the channel writing into its own directory, so a restart never reads where it writes."""
    return _write(
        tmp_path / name, _CHANNEL | {"outputs": outputs | {"directory": name[:-5]}} | sections
    )


def _reference_residual(directory: Path) -> float:
    """The residual a run's log says it started from."""
    match = re.search(r"reference \|R0\| = (\S+)", (directory / "march.log").read_text())
    return float(match.group(1))


def test_a_run_that_stopped_short_is_resumed_from_its_checkpoint_as_the_same_march(
    tmp_path,
) -> None:
    """The point of a restart: a stopped run's march goes on, step for step, to the same root.

    The stopped run is cut off at three steps, leaving its checkpoint. The resumed run must begin where
    that run ended -- not from scratch, which a restart that ignored its seed would -- and then take the
    steps the uninterrupted run took after its third, because the checkpoint carries the one piece of
    the march's history its damping ramp and stopping bar are measured against: the residual it began
    at. A restart that re-measured at the resumed state would take a different path and stop against a
    different bar.
    """
    checkpoints = {"checkpoints": {"kind": "Checkpoints", "keep": 1}}
    fresh = _restart_case(tmp_path, "fresh.yaml", checkpoints)
    assert main(["run", str(fresh)]) == 0
    fresh_record = yaml.safe_load((tmp_path / "fresh" / "run.yaml").read_text())

    stopped = _restart_case(
        tmp_path, "stopped.yaml", checkpoints, solver={"kind": "FlowMarch", "max_steps": 3}
    )
    assert main(["run", str(stopped)]) == 1
    left_at = yaml.safe_load((tmp_path / "stopped" / "run.yaml").read_text())["residual"]

    resumed = _restart_case(
        tmp_path,
        "resumed.yaml",
        checkpoints,
        initial={"kind": "Checkpoint", "path": "stopped/checkpoints"},
    )
    assert main(["run", str(resumed)]) == 0
    resumed_record = yaml.safe_load((tmp_path / "resumed" / "run.yaml").read_text())

    # It picks the stopped march up where it left off ...
    assert resumed_record["initial"]["residual"] == pytest.approx(left_at, rel=1e-12)
    # ... against the reference the uninterrupted march judged itself by, so the step the log reports
    # as its target is the same one ...
    # (the log prints five significant figures)
    began_at = _reference_residual(tmp_path / "fresh")
    assert _reference_residual(tmp_path / "resumed") == pytest.approx(began_at, rel=1e-4)
    assert resumed_record["initial"]["resumption"]["reference_residual"] == pytest.approx(
        began_at, rel=1e-4
    )
    # ... and takes exactly the steps the uninterrupted run had left.
    assert resumed_record["steps"] == fresh_record["steps"] - 3

    problem = read_case(fresh).check().build()
    velocity, pressure = problem.unpack(solve_flow_march(problem))
    (final,) = (tmp_path / "resumed" / "checkpoints").iterdir()
    reached = read_restart(final)
    np.testing.assert_allclose(reached.fields["U"], np.asarray(velocity), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(reached.fields["p"], np.asarray(pressure), rtol=1e-6, atol=1e-9)


def _shifts(directory: Path) -> list[float]:
    """The shift each step of a run took, from its step history."""
    with open(directory / "history.csv", newline="") as handle:
        return [float(row["shift"]) for row in csv.DictReader(handle)]


def test_a_dual_time_run_resumes_the_shift_its_stopped_run_had_walked_down_to(tmp_path) -> None:
    """The Courant ramp's shift is march history a checkpoint carries, and a resume must not reset it.

    The wrong answer this catches: the resumed run opening at the ramp's starting shift again, which
    is what a restart that carried only the residual did, and then walking the whole ramp a second time.
    """
    march = {
        "kind": "FlowMarch",
        "dual_time": {"kind": "DualTimeLoop", "inner_steps": 3},
    }
    checkpoints = {"checkpoints": {"kind": "Checkpoints", "keep": 1}}
    stopped = _restart_case(tmp_path, "stopped.yaml", checkpoints, solver=march | {"max_steps": 6})
    assert main(["run", str(stopped)]) == 1
    walked = _shifts(tmp_path / "stopped")
    assert walked[-1] < 0.5 * walked[0]  # the ramp has come down by the time it is cut off

    resumed = _restart_case(
        tmp_path,
        "resumed.yaml",
        checkpoints,
        solver=march,
        initial={"kind": "Checkpoint", "path": "stopped/checkpoints"},
    )
    assert main(["run", str(resumed)]) == 0
    assert _shifts(tmp_path / "resumed")[0] == pytest.approx(walked[-1])


def test_a_restart_of_a_changed_case_starts_its_march_afresh_from_the_checkpoint(tmp_path) -> None:
    """A different fluid is a different problem, so the stopped run's reference residual is not carried."""
    checkpoints = {"checkpoints": {"kind": "Checkpoints", "keep": 1}}
    stopped = _restart_case(
        tmp_path, "stopped.yaml", checkpoints, solver={"kind": "FlowMarch", "max_steps": 3}
    )
    assert main(["run", str(stopped)]) == 1

    changed = _restart_case(
        tmp_path,
        "changed.yaml",
        checkpoints,
        fluid={"density": 1.0, "kinematic_viscosity": 4.0e-2},
        initial={"kind": "Checkpoint", "path": "stopped/checkpoints"},
    )
    assert main(["run", str(changed)]) == 0
    record = yaml.safe_load((tmp_path / "changed" / "run.yaml").read_text())
    assert record["initial"]["resumption"] is None


def test_a_run_started_from_a_converged_checkpoint_has_nothing_left_to_do(tmp_path) -> None:
    checkpoints = {"checkpoints": {"kind": "Checkpoints", "keep": 1}}
    first = _restart_case(tmp_path, "first.yaml", checkpoints)
    assert main(["run", str(first)]) == 0
    first_steps = yaml.safe_load((tmp_path / "first" / "run.yaml").read_text())["steps"]

    again = _restart_case(
        tmp_path, "again.yaml", {}, initial={"kind": "Checkpoint", "path": "first/checkpoints"}
    )
    assert main(["run", str(again)]) == 0
    record = yaml.safe_load((tmp_path / "again" / "run.yaml").read_text())
    # Already at the root, so the march's first residual check passes and no step is taken.
    assert (record["steps"] or 0) <= 2 < first_steps

    # The run records what it started from, and its case record points at the same place.
    assert record["initial"]["kind"] == "Checkpoint"
    assert Path(record["initial"]["file"]).name == f"state-{first_steps:05d}.npz"
    assert 0.0 < record["initial"]["residual"] < 1e-8
    ran = read_case(tmp_path / "again" / "case.yaml").spec
    assert (
        ran.initial.location(tmp_path / "again") == (tmp_path / "first" / "checkpoints").resolve()
    )


def test_a_case_starts_from_the_openfoam_time_directory_another_run_wrote(tmp_path) -> None:
    """Out through ``OpenFOAMTime`` and back in through ``Fields``: the pressure, at a density of two.

    The file holds ``p / 2`` because that is what an OpenFOAM incompressible solver reads; the run that
    starts from it multiplies by two again. A writer that stored the pressure itself, or a reader that
    did not scale, would each leave the second run a factor of two off the first's pressure -- far
    enough from a root that it could not stop at once.
    """
    of = tmp_path / "of"
    copy_slab_polymesh(of / "constant" / "polyMesh")
    (of / "0").mkdir()
    (of / "0" / "p").write_text(_PRESSURE_TEMPLATE)
    (of / "0" / "U").write_text(_VELOCITY_TEMPLATE)
    common = _CHANNEL | {
        "mesh": {"kind": "OpenFOAMMesh", "path": "of/constant/polyMesh"},
        "fluid": {"density": 2.0, "kinematic_viscosity": 2.0e-2},
        "numerics": {
            "momentum_advection": {"kind": "FirstOrderUpwind"},
            "gradient": {"kind": "CompactGreenGauss"},
        },
    }
    first = _write(
        tmp_path / "first.yaml",
        common
        | {
            "outputs": {
                "directory": "first",
                "fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "5"}],
            }
        },
    )
    assert main(["run", str(first)]) == 0

    problem = read_case(first).check().build()
    velocity, pressure = problem.unpack(solve_flow_march(problem))
    mesh = read_case(first).check().mesh
    written = read_volume_scalar_field(of / "5" / "p", mesh)
    np.testing.assert_allclose(written, np.asarray(pressure) / 2.0, rtol=1e-6, atol=1e-9)
    assert float(np.max(np.abs(pressure))) > 1e-3  # the comparison is not between zeros

    second = _write(
        tmp_path / "second.yaml",
        common
        | {
            "outputs": {
                "directory": "second",
                "fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "6"}],
            },
            "initial": {"kind": "Fields", "path": "of", "time": "5"},
        },
    )
    assert main(["run", str(second)]) == 0
    record = yaml.safe_load((tmp_path / "second" / "run.yaml").read_text())
    # Already at the root, so the march takes no step; a pressure left unscaled by the density is off
    # by a factor of two and would need some.
    assert (record["steps"] or 0) == 0
    assert record["initial"]["kind"] == "Fields" and record["initial"]["density"] == 2.0
    # What the second run wrote is what the first did: the state went out and came back unchanged.
    again = read_openfoam_time(of, "6", ["U", "p"], mesh)
    first_written = read_openfoam_time(of, "5", ["U", "p"], mesh)
    np.testing.assert_allclose(again["p"], first_written["p"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(again["U"], first_written["U"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(first_written["U"], np.asarray(velocity), rtol=1e-6, atol=1e-9)
