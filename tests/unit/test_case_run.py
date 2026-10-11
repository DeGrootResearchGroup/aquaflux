"""A case file's ``outputs`` section, the fields each physics writes, and the checks a run makes before it starts.

What a whole run writes -- a solve, its fields, its records -- is ``tests/integration/test_case_run.py``'s.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import sys
import tempfile
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
    }
    return {**sections, **overrides}


# --- the section ---------------------------------------------------------------------------------


def test_a_file_with_no_outputs_section_writes_vtk_the_log_and_the_history_into_results() -> None:
    spec = case_spec_from_mapping(_sections())
    assert spec.outputs == Outputs(
        directory="results", fields=(Vtk(),), log="march.log", history="history.csv"
    )
    assert "outputs" not in case_spec_to_mapping(spec)


def test_an_outputs_section_reads_without_its_kind_and_writes_back_equal() -> None:
    section = {
        "directory": "out",
        "log": None,
        "history": "steps.csv",
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
        history="steps.csv",
        checkpoints=Checkpoints(every=5, keep=2),
    )
    assert case_spec_to_mapping(spec)["outputs"] == section
    assert case_spec_from_mapping(case_spec_to_mapping(spec)) == spec


def _slab_case_with_a_velocity_template(root: Path) -> Path:
    """An OpenFOAM case on the slab -- whose extents leave its extruded axis ambiguous -- with a ``U`` template."""
    shutil.copytree(SLAB, root / "of" / "constant" / "polyMesh")
    (root / "of" / "0").mkdir()
    (root / "of" / "0" / "U").write_text(
        "FoamFile\n{\n    format ascii;\n    class volVectorField;\n    object U;\n}\n"
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        "boundaryField\n{\n    left { type fixedValue; value uniform (1 0 0); }\n}\n"
    )
    return root


def test_an_openfoam_time_writes_a_vector_with_its_zero_on_the_axis_the_file_states(
    tmp_path,
) -> None:
    """A slab whose extents cannot say which axis was extruded is written where the file says.

    Wrong answers this catches: a stated axis that is ignored (the ambiguous slab then raises), the
    letters mapped to the wrong component (the zero lands one slot over, so reading the file back
    along ``z`` refuses it), and the axis not reaching the writer at all.
    """
    from aquaflux.case import RunFields
    from aquaflux.io import read_openfoam, read_openfoam_time

    root = _slab_case_with_a_velocity_template(tmp_path)
    mesh = read_openfoam(root / "of")
    velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
    fields = RunFields(cells={"U": velocity})

    with pytest.raises(ValueError, match="pass extruded_axis explicitly"):
        OpenFOAMTime(case="of", time="1").write(tmp_path, root, mesh, fields)

    OpenFOAMTime(case="of", time="2", extruded_axis="y").write(tmp_path, root, mesh, fields)
    read_back = read_openfoam_time(root / "of", "2", ["U"], mesh, extruded_axis=1)
    np.testing.assert_array_equal(read_back["U"], velocity)
    with pytest.raises(ValueError, match="nonzero component along axis 2"):
        read_openfoam_time(root / "of", "2", ["U"], mesh, extruded_axis=2)

    OpenFOAMTime(case="of", time="3", extruded_axis="z").write(tmp_path, root, mesh, fields)
    np.testing.assert_array_equal(
        read_openfoam_time(root / "of", "3", ["U"], mesh, extruded_axis=2)["U"], velocity
    )


def test_an_extruded_axis_is_refused_for_a_three_dimensional_mesh_and_only_then() -> None:
    writer = OpenFOAMTime(case="of", time="1", extruded_axis="z")
    with pytest.raises(ValueError, match=r"outputs.fields: extruded_axis .* three-dimensional"):
        writer.refuse_for_dimension(3)
    writer.refuse_for_dimension(2)
    OpenFOAMTime(case="of", time="1").refuse_for_dimension(3)


def test_the_case_refuses_an_extruded_axis_its_mesh_has_no_use_for(tmp_path) -> None:
    """The check against the mesh reports it with the rest, from the writers and the starting state."""
    sections = _sections(
        outputs={
            "fields": [{"kind": "OpenFOAMTime", "case": "of", "time": "1", "extruded_axis": "z"}]
        },
        initial={"kind": "Fields", "path": "of", "time": "0", "extruded_axis": "z"},
    )
    spec = case_spec_from_mapping(sections)
    mesh = spec.mesh.read(tmp_path)
    spec.check_against(mesh)  # a two-dimensional mesh takes both
    three = shutil.copytree(
        REPO / "tests" / "fixtures" / "polymesh_3d_two_cubes", tmp_path / "three" / "polyMesh"
    )
    spec_3d = case_spec_from_mapping(
        _sections(
            mesh={"kind": "OpenFOAMMesh", "path": str(three)},
            outputs=sections["outputs"],
            initial=sections["initial"],
        )
    )
    with pytest.raises(ValueError) as refused:
        spec_3d.check_against(spec_3d.mesh.read(tmp_path))
    assert "outputs.fields: extruded_axis" in str(refused.value)
    assert "initial: extruded_axis" in str(refused.value)


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
        ({"history": "logs/history.csv"}, r"Outputs.history is a file name"),
        ({"log": "steps.txt", "history": "steps.txt"}, r"log and Outputs.history both name"),
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
        "history-in-a-subdirectory",
        "log-and-history-one-file",
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
    return mesh, mesh.geometry(), REPO


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


def _hooks(seen: list, who: str, **extra: object) -> types.SimpleNamespace:
    """A log or a recorder whose every march hook records who heard what."""
    hooks = {
        "on_checkpoint": lambda report, state: seen.append((who, "step", state)),
        "on_retry": lambda reason, attempt, beta: seen.append((who, "retry", reason)),
        "on_refresh": lambda timing: seen.append((who, "refresh", timing)),
        "on_residuals": None,
        "on_inner": None,
    }
    return types.SimpleNamespace(**(hooks | extra))


def test_a_march_hands_each_step_and_retry_to_the_log_and_to_the_recorder() -> None:
    seen = []
    observers = FlowMarch().observers_for(_hooks(seen, "log"), _hooks(seen, "file"))
    observers["on_checkpoint"]("report", "state")
    observers["on_retry"]("alpha", 1, 0.5)
    assert seen == [
        ("log", "step", "state"),
        ("file", "step", "state"),
        ("log", "retry", "alpha"),
        ("file", "retry", "alpha"),
    ]


def test_the_per_equation_residuals_are_asked_for_only_when_a_recorder_keeps_them() -> None:
    # Each costs a residual evaluation per step, so a march without a taker must not be handed the hook.
    seen = []
    assert "on_residuals" not in FlowMarch().observers_for(_Logger(), None)
    assert "on_residuals" not in FlowMarch().observers_for(
        _Logger(), _hooks(seen, "file", on_residuals=None)
    )
    observers = FlowMarch().observers_for(
        _Logger(), _hooks(seen, "file", on_residuals=lambda terms: seen.append(terms))
    )
    observers["on_residuals"]({"u": 1.0})
    assert seen == [{"u": 1.0}]


def test_a_coupled_march_hands_each_refit_to_the_log_and_to_the_recorder() -> None:
    seen = []
    observers = CoupledMarch(
        preconditioner=MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed()))
    ).observers_for(_hooks(seen, "log"), _hooks(seen, "file"))
    observers["session_options"]["observer"]("timing")
    assert seen == [("log", "refresh", "timing"), ("file", "refresh", "timing")]


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


def test_the_run_hands_each_march_hook_to_every_recorder_that_has_it() -> None:
    from aquaflux.case.run import _StepCount

    seen = []
    history = types.SimpleNamespace(
        on_checkpoint=lambda report, state: seen.append(("history", "step")),
        on_retry=lambda reason, attempt, beta: seen.append(("history", reason)),
        on_refresh=lambda timing: seen.append(("history", timing)),
        on_residuals=lambda terms: seen.append(("history", terms)),
    )
    checkpoints = types.SimpleNamespace(
        on_checkpoint=lambda report, state: seen.append(("checkpoints", "step"))
    )
    steps = _StepCount([history, checkpoints])
    steps.on_retry("alpha", 1, 0.5)
    steps.on_refresh("timing")
    steps.on_residuals({"u": 1.0})
    steps.on_checkpoint(types.SimpleNamespace(residual_norm=0.25), None)
    assert seen == [
        ("history", "alpha"),
        ("history", "timing"),
        ("history", {"u": 1.0}),
        ("history", "step"),
        ("checkpoints", "step"),
    ]
    assert (steps.count, steps.residual) == (1, 0.25)
    # With no recorder keeping them, the per-equation residuals are not asked for at all.
    assert _StepCount([checkpoints]).on_residuals is None


@pytest.mark.parametrize("missing", ["on_checkpoint", "on_retry", "on_refresh", "on_residuals"])
def test_the_run_hands_the_rest_of_the_march_only_to_a_recorder_of_the_whole_march(missing) -> None:
    from aquaflux.case.run import _StepCount
    from aquaflux.solve import MarchRecorder, StateCheckpointer, StepHistory

    assert isinstance(StepHistory(Path(tempfile.mkdtemp()) / "history.csv"), MarchRecorder)
    assert not isinstance(StateCheckpointer(tempfile.mkdtemp()), MarchRecorder)
    seen = []
    hooks = {
        "on_checkpoint": lambda report, state: seen.append("step"),
        "on_retry": lambda reason, attempt, beta: seen.append(reason),
        "on_refresh": lambda timing: seen.append(timing),
        "on_residuals": lambda terms: seen.append(terms),
    }
    whole = types.SimpleNamespace(**hooks)
    del hooks[missing]
    partial = types.SimpleNamespace(**hooks)
    assert isinstance(whole, MarchRecorder)
    assert not isinstance(partial, MarchRecorder)
    # Offering some of the hooks does not make a recorder of the whole march, so it is handed none.
    steps = _StepCount([partial])
    steps.on_retry("alpha", 1, 0.5)
    steps.on_refresh("timing")
    assert seen == []
    assert steps.on_residuals is None
    # A whole-march recorder that wants no per-equation residuals does not cost the march them either.
    assert (
        _StepCount([types.SimpleNamespace(**(vars(whole) | {"on_residuals": None}))]).on_residuals
        is None
    )
