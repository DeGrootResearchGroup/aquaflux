"""A case file's ``outputs`` section, the fields each physics writes, and the checks a run makes before it starts.

What a whole run writes -- a solve, its fields, its records -- is ``tests/integration/test_case_run.py``'s.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import aquaflux.case.solver as solver_module
import numpy as np
import pytest
from aquaflux.__main__ import main
from aquaflux.case import (
    Checkpoints,
    CoupledMarch,
    FlowMarch,
    NotConverged,
    OpenFOAMTime,
    Outputs,
    Segregated,
    ViscosityRamp,
    Vtk,
    case_spec_from_mapping,
    case_spec_to_mapping,
    prepare_run,
)
from aquaflux.case.run import _checkout_state
from aquaflux.solve import (
    DualTimeLoop,
    FieldSplit,
    JacobiSmoothed,
    MaterializedJacobian,
    SimpleSmoothed,
)
from aquaflux.turbulence import BlockDiagonal, solve_segregated

REPO = Path(__file__).resolve().parents[2]
SLAB = REPO / "tests" / "fixtures" / "polymesh_2d_slab_frontandback"


def _sections(physics: str = "Laminar", mesh=None, **overrides: object) -> dict[str, object]:
    """A channel on the two-cell slab fixture, as a file would state it."""
    rans = physics == "RANS"
    inlet = {"kind": "Inlet", "velocity": [1.0, 0.0]}
    if rans:
        inlet["turbulence"] = {"kind": "FixedTurbulence", "k": 1e-3, "omega": 2.0}
    sections = {
        "mesh": mesh or {"kind": "OpenFOAMMesh", "path": str(SLAB)},
        "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-2},
        "physics": {
            "kind": physics,
            **({"advection": {"kind": "FirstOrderUpwind"}} if rans else {}),
        },
        "boundaries": {
            "left": inlet,
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
        },
        "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
    }
    return {**sections, **overrides}


# --- the section ---------------------------------------------------------------------------------


def test_a_file_with_no_outputs_section_writes_vtk_and_the_log_into_results() -> None:
    spec = case_spec_from_mapping(_sections())
    assert spec.outputs == Outputs(directory="results", fields=(Vtk(),), log="march.log")
    assert "outputs" not in case_spec_to_mapping(spec)


def test_an_outputs_section_reads_without_its_kind_and_writes_back_equal() -> None:
    section = {
        "directory": "out",
        "log": None,
        "checkpoints": {"kind": "Checkpoints", "every": 5, "keep": 2},
        "fields": [
            {"kind": "Vtk", "file": "flow.vtu", "fields": ["U"]},
            {"kind": "OpenFOAMTime", "case": "of", "time": "10", "template_time": "0"},
        ],
    }
    spec = case_spec_from_mapping(_sections(outputs=section))
    assert spec.outputs == Outputs(
        directory="out",
        fields=(
            Vtk(file="flow.vtu", fields=("U",)),
            OpenFOAMTime(case="of", time="10", template_time="0"),
        ),
        log=None,
        checkpoints=Checkpoints(every=5, keep=2),
    )
    assert case_spec_to_mapping(spec)["outputs"] == section
    assert case_spec_from_mapping(case_spec_to_mapping(spec)) == spec


@pytest.mark.parametrize(
    ("outputs", "match"),
    [
        (
            {"fields": [{"kind": "Vtk", "file": "flow.vtk"}]},
            r"Vtk.file is a file name ending in .vtu",
        ),
        (
            {"fields": [{"kind": "Vtk", "file": "sub/flow.vtu"}]},
            r"Vtk.file is a file name ending in .vtu",
        ),
        ({"fields": [{"kind": "OpenFOAMTime", "case": "of"}]}, r"OpenFOAMTime .* needs 'time'"),
        (
            {"fields": [{"kind": "OpenFOAMTime", "case": "", "time": "1"}]},
            r"needs the case directory",
        ),
        ({"checkpoints": {"kind": "Checkpoints", "every": 0}}, r"Checkpoints.every must be >= 1"),
        ({"checkpoints": {"kind": "Checkpoints", "keep": 0}}, r"Checkpoints.keep must be >= 1"),
        ({"log": "logs/march.log"}, r"Outputs.log is a file name"),
        ({"directory": ""}, r"Outputs.directory names the directory"),
    ],
    ids=[
        "vtk-wrong-suffix",
        "vtk-not-a-file-name",
        "openfoam-no-time",
        "openfoam-no-case",
        "checkpoints-never",
        "checkpoints-keep-none",
        "log-in-a-subdirectory",
        "no-directory",
    ],
)
def test_an_outputs_section_that_cannot_be_written_is_refused(outputs, match) -> None:
    with pytest.raises(ValueError, match=match):
        case_spec_from_mapping(_sections(outputs=outputs))


def test_an_openfoam_time_directory_is_refused_for_a_mesh_that_is_not_openfoams() -> None:
    grid = {"kind": "StructuredGrid", "cells": [4, 4], "lengths": [1.0, 1.0]}
    with pytest.raises(ValueError, match=r"the mesh is a StructuredGrid. Write the fields as Vtk"):
        case_spec_from_mapping(
            _sections(
                mesh=grid,
                outputs={"fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "1"}]},
            )
        )


def test_a_writer_writes_the_fields_it_names_and_refuses_one_the_case_does_not_produce() -> None:
    fields = {"U": "u", "p": "p", "k": "k"}
    assert Vtk().chosen(fields) == fields
    assert Vtk(fields=("k", "U")).chosen(fields) == {"k": "k", "U": "u"}
    with pytest.raises(
        ValueError, match=r"Vtk.fields names \['nut'\], which this case does not produce"
    ):
        Vtk(fields=("nut",)).chosen(fields)


# --- what each physics writes ------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:MultipleCorrectionGradient.*underdetermined:UserWarning")
def test_a_laminar_case_writes_its_velocity_and_its_solved_pressure() -> None:
    spec = case_spec_from_mapping(_sections())
    problem = spec.physics.build(spec, *_mesh_and_geometry(spec))
    state = np.arange(3 * problem.mesh.n_cells, dtype=float) + 0.5
    fields = spec.physics.output_fields(problem, state)
    velocity, pressure = problem.unpack(state)
    assert list(fields) == ["U", "p"]
    np.testing.assert_array_equal(fields["U"], np.asarray(velocity))
    # The solved pressure, not the gauge-free one the log reports.
    np.testing.assert_array_equal(fields["p"], np.asarray(pressure))
    assert fields["p"].mean() != 0.0
    assert spec.physics.progress_fields(problem) is None


@pytest.mark.filterwarnings("ignore:MultipleCorrectionGradient.*underdetermined:UserWarning")
def test_a_rans_case_writes_k_omega_and_the_eddy_viscosity_they_give() -> None:
    spec = case_spec_from_mapping(_sections("RANS"))
    problem = spec.physics.build(spec, *_mesh_and_geometry(spec))
    n = problem.momentum.mesh.n_cells
    flow = np.linspace(0.1, 0.9, 3 * n)
    k, omega = np.array([0.02, 0.05]), np.array([3.0, 7.0])
    fields = spec.physics.output_fields(problem, (flow, k, omega))
    assert list(fields) == ["U", "p", "k", "omega", "nut"]
    np.testing.assert_array_equal(fields["k"], k)
    np.testing.assert_array_equal(fields["omega"], omega)
    momentum = problem.momentum
    nut = problem.turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    np.testing.assert_array_equal(fields["nut"], np.asarray(nut))
    assert spec.physics.progress_fields(problem) is not None


def _mesh_and_geometry(spec):
    mesh = spec.mesh.read(REPO)
    return mesh, mesh.geometry()


# --- how each solve is observed ------------------------------------------------------------------


class _Logger:
    """Stands in for a MarchLogger: each hook a distinct marker."""

    on_checkpoint, on_retry, on_inner, on_refresh = "step", "retry", "inner", "refresh"

    def __init__(self) -> None:
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)


def test_a_march_logs_each_step_retry_and_inner_iteration_it_has() -> None:
    logger = _Logger()
    assert FlowMarch().observers_for(logger, None) == {"on_checkpoint": "step", "on_retry": "retry"}
    observed = FlowMarch(dual_time=DualTimeLoop(inner_steps=3)).observers_for(logger, None)
    assert observed == {"on_checkpoint": "step", "on_retry": "retry", "inner_observer": "inner"}


def test_a_march_hands_each_step_to_the_log_and_to_the_checkpoints() -> None:
    seen = []
    logger = types.SimpleNamespace(
        on_checkpoint=lambda report, state: seen.append(("log", state)), on_retry=None
    )
    checkpointer = types.SimpleNamespace(
        on_checkpoint=lambda report, state: seen.append(("file", state))
    )
    FlowMarch().observers_for(logger, checkpointer)["on_checkpoint"]("report", "state")
    assert seen == [("log", "state"), ("file", "state")]


def test_a_coupled_march_also_logs_its_refreshes_and_its_ramp() -> None:
    logger = _Logger()
    assert "session_options" not in CoupledMarch(preconditioner=BlockDiagonal()).observers_for(
        logger, None
    )
    observed = CoupledMarch(
        preconditioner=MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed())),
        continuation=ViscosityRamp(anchor=10.0, stations=2, steps_per_station=1),
    ).observers_for(logger, None)
    assert observed["session_options"] == {"observer": "refresh"}
    assert (
        observed["point_setup"]("companion", "state", types.SimpleNamespace(label="point 1/1"))
        is None
    )
    assert logger.notes == ["[point 1/1]"]


def test_the_segregated_loop_takes_no_observers() -> None:
    assert Segregated(sweeps=3).observers_for(_Logger(), object()) == {}


# --- a segregated solve that runs out of sweeps -------------------------------------------------


def test_a_segregated_solve_that_runs_out_of_sweeps_is_not_converged(monkeypatch) -> None:
    def warns(*args, **kwargs):
        warnings.warn(
            "segregated coupling did not reach increment_tol=1e-06 within 3 sweeps (last increment "
            "0.1); the returned fields may be under-converged.",
            stacklevel=2,
        )
        return "fields"

    for name in ("bulk_velocity_flow_solve", "reused_flow_solve", "scalar_pseudo_transient_solve"):
        monkeypatch.setattr(solver_module, name, lambda *a, **k: None)
    monkeypatch.setattr(solver_module, "sst_initial_fields", lambda *a: (1, 2, 3))
    monkeypatch.setattr(solver_module, "solve_segregated", warns)
    problem = types.SimpleNamespace(momentum=types.SimpleNamespace(drive=None), turbulence=None)
    with pytest.raises(NotConverged, match=r"within 3 sweeps"):
        Segregated(sweeps=3).solve(problem)
    monkeypatch.setattr(solver_module, "solve_segregated", lambda *a, **k: "fields")
    assert Segregated(sweeps=3).solve(problem) == "fields"


def test_the_warning_the_case_layer_turns_into_non_convergence_is_the_one_the_loop_gives() -> None:
    """The case layer matches the loop's warning by its opening words, so they must be the loop's own."""
    assert solver_module._SEGREGATED_NOT_CONVERGED in inspect.getsource(solve_segregated)


# --- before a run starts -----------------------------------------------------------------------


def _write(path: Path, sections: dict) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(sections))
    return path


def test_a_run_refuses_to_replace_an_earlier_runs_results(tmp_path: Path) -> None:
    case = _write(tmp_path / "case.yaml", _sections())
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fields.vtu").write_text("an earlier run")
    with pytest.raises(FileExistsError, match=r"results already holds results; .* --overwrite"):
        prepare_run(case)
    assert (tmp_path / "results" / "fields.vtu").read_text() == "an earlier run"


def test_an_empty_output_directory_is_not_an_earlier_run(tmp_path: Path) -> None:
    case = _write(tmp_path / "case.yaml", _sections())
    (tmp_path / "results").mkdir()
    assert prepare_run(case).directory == (tmp_path / "results").resolve()


def test_overwriting_clears_the_earlier_checkpoints_and_nothing_else(tmp_path: Path) -> None:
    case = _write(tmp_path / "case.yaml", _sections())
    checkpoints = tmp_path / "results" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "state-00007.npz").write_text("old")
    (tmp_path / "results" / "notes.txt").write_text("mine")
    prepare_run(case, overwrite=True)
    assert not checkpoints.exists()
    assert (tmp_path / "results" / "notes.txt").read_text() == "mine"


def test_a_run_refuses_to_replace_an_openfoam_time_directory(tmp_path: Path) -> None:
    (tmp_path / "of" / "5").mkdir(parents=True)
    outputs = {"fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "5"}]}
    case = _write(tmp_path / "case.yaml", _sections(outputs=outputs))
    with pytest.raises(FileExistsError, match=r"of/5 already holds results"):
        prepare_run(case)
    assert prepare_run(case, overwrite=True).solver == FlowMarch()


def test_a_case_that_cannot_be_solved_is_refused_before_a_run_starts(tmp_path: Path) -> None:
    """A held bulk velocity with no solver stated has no default solve: refused while reading."""
    grid = {"kind": "StructuredGrid", "cells": [4, 4], "lengths": [1.0, 2.0], "periodic": ["x"]}
    sections = _sections(
        "RANS",
        mesh=grid,
        drive={"kind": "BulkVelocity", "target": 1.0},
        pressure_datum={"kind": "PinnedPoint", "point": [0.0, 0.0]},
    )
    sections["boundaries"] = {"bottom": {"kind": "Wall"}, "top": {"kind": "Wall"}}
    case = _write(tmp_path / "case.yaml", sections)
    with pytest.raises(
        ValueError, match=r"the case states no solver, and its default cannot solve it"
    ):
        prepare_run(case)
    assert not (tmp_path / "results").exists()


def test_a_run_records_the_commit_it_ran_from() -> None:
    commit, modified = _checkout_state()
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert commit == head
    assert isinstance(modified, bool)


# --- the command ---------------------------------------------------------------------------------


def test_check_says_what_the_file_describes(tmp_path: Path, capsys) -> None:
    case = _write(tmp_path / "case.yaml", _sections())
    assert main(["check", str(case)]) == 0
    out = capsys.readouterr().out
    assert "a Laminar case on 2 cells (2D)" in out and "FlowMarch (the default)" in out


def test_check_names_the_patches_a_group_key_reaches(tmp_path: Path, capsys) -> None:
    mesh = tmp_path / "polyMesh"
    shutil.copytree(SLAB, mesh)
    boundary = mesh / "boundary"
    text = boundary.read_text()
    for wall in ("bottom", "top"):
        text = text.replace(
            f"    {wall}\n    {{\n", f"    {wall}\n    {{\n        inGroups 1(walls);\n"
        )
    boundary.write_text(text)
    boundaries = {**_sections()["boundaries"], "walls": {"kind": "Wall"}}
    del boundaries["bottom"], boundaries["top"]
    sections = _sections(mesh={"kind": "OpenFOAMMesh", "path": str(mesh)}, boundaries=boundaries)
    assert main(["check", str(_write(tmp_path / "case.yaml", sections))]) == 0
    assert "patches left, right, bottom, top;" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["check", "run"])
def test_a_refused_file_exits_with_status_2_and_says_why(tmp_path: Path, capsys, command) -> None:
    case = _write(tmp_path / "case.yaml", {**_sections(), "fluid": {"density": 1.0}})
    assert main([command, str(case)]) == 2
    assert "exactly one of kinematic_viscosity and dynamic_viscosity" in capsys.readouterr().err


def test_a_run_over_earlier_results_exits_with_status_2(tmp_path: Path, capsys) -> None:
    case = _write(tmp_path / "case.yaml", _sections())
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run.yaml").write_text("converged: true")
    assert main(["run", str(case)]) == 2
    assert "already holds results" in capsys.readouterr().err


def test_the_package_runs_as_the_aquaflux_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aquaflux", "--help"], capture_output=True, text=True, cwd=REPO
    )
    assert result.returncode == 0
    assert "check" in result.stdout and "run" in result.stdout
