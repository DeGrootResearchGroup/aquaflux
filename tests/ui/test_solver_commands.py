"""Tests for the solver's case-file commands as the page calls them: arguments, replies, failures.

Most replace the process with a stub runner, recording what it was asked; one runs the real command,
which pins that the installed solver is reached through this interpreter at all.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("trame")

from aquaflux_ui.solver_commands import CASE_FILE, MESH_FILE, SolverCommands
from aquaflux_ui.solver_worker import CommandResult


class Recorder:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, arguments, stdin):
        self.calls.append((list(arguments), stdin))
        return self.results.pop(0)


def test_the_schema_is_asked_for_once():
    runner = Recorder(CommandResult(0, json.dumps({"root": "CaseSpec", "kinds": {}})))
    commands = SolverCommands(runner)
    assert commands.schema().root == "CaseSpec" and commands.schema() is commands.schema()
    assert runner.calls == [(["schema"], None)]


def test_a_failing_schema_command_says_why():
    commands = SolverCommands(
        Recorder(CommandResult(1, "", "Traceback ...\nModuleNotFoundError: jax"))
    )
    with pytest.raises(RuntimeError, match="ModuleNotFoundError: jax"):
        commands.schema()


def test_show_returns_the_case_and_the_refusal():
    reply = {"error": "Wall at 'boundaries.x': ...", "case": {"mesh": {}}}
    document = SolverCommands(Recorder(CommandResult(0, json.dumps(reply)))).show("c.yaml")
    assert document.case == {"mesh": {}} and document.error.startswith("Wall")


def test_write_sends_the_case_and_where_its_paths_are_relative_to():
    runner = Recorder(CommandResult(0, json.dumps({"error": None})))
    assert SolverCommands(runner).write("new/c.yaml", {"fluid": {}}, "old") is None
    ((arguments, stdin),) = runner.calls
    assert arguments == ["write", "new/c.yaml", "--relative-to", "old"]
    assert json.loads(stdin) == {"case": {"fluid": {}}}


def test_a_reply_that_is_not_json_is_an_error_naming_the_last_line():
    commands = SolverCommands(Recorder(CommandResult(2, "", "line one\nthe real reason")))
    assert commands.write("c.yaml", {}) == "the real reason"


def test_check_reports_success_and_failure():
    commands = SolverCommands(
        Recorder(
            CommandResult(0, "c.yaml: ... checked.\n"), CommandResult(2, "", "aquaflux: bad\n")
        )
    )
    assert commands.check("c.yaml") == (True, "c.yaml: ... checked.")
    assert commands.check("c.yaml") == (False, "aquaflux: bad")


def test_the_real_command_answers():
    schema = SolverCommands().schema()
    assert schema.root == "CaseSpec" and schema.field("CaseSpec", "boundaries") is not None


def test_mesh_sends_the_section_and_reads_the_report():
    report = {
        "error": None,
        "cells": 12,
        "dim": 2,
        "patches": [{"name": "top", "faces": 4}],
        "groups": {},
        "made": {"quality": {"cells": 12}},
    }
    runner = Recorder(CommandResult(0, json.dumps(report)))
    section = {"kind": "AquafluxMesh", "path": "mesh.aqmesh"}
    export = SolverCommands(runner).mesh(section, "case_dir", "out")
    ((arguments, stdin),) = runner.calls
    assert arguments == ["export-mesh", "out", "--relative-to", "case_dir"]
    assert json.loads(stdin) == {"mesh": section}
    assert (export.error, export.cells, export.dim, export.patches, export.made) == (
        None,
        12,
        2,
        report["patches"],
        report["made"],
    )


def test_a_mesh_that_cannot_be_read_says_why():
    runner = Recorder(CommandResult(2, json.dumps({"error": "no polyMesh there"})))
    export = SolverCommands(runner).mesh({}, ".", "out")
    assert export.error == "no polyMesh there" and export.directory is None


def test_plan_reads_the_run_s_output_paths_and_what_it_would_replace(tmp_path):
    reply = {
        "error": None,
        "directory": str(tmp_path / "results"),
        "log": str(tmp_path / "results" / "march.log"),
        "history": None,
        "occupied": [str(tmp_path / "results")],
    }
    runner = Recorder(CommandResult(0, json.dumps(reply)))
    plan = SolverCommands(runner).plan(tmp_path / "case.yaml")
    assert runner.calls == [(["plan", str(tmp_path / "case.yaml")], None)]
    assert plan.error is None and plan.directory == tmp_path / "results"
    assert plan.log == tmp_path / "results" / "march.log" and plan.history is None
    assert plan.occupied == (tmp_path / "results",)


def test_a_refused_plan_says_why():
    runner = Recorder(CommandResult(2, json.dumps({"error": "case.yaml: no such file"})))
    plan = SolverCommands(runner).plan("case.yaml")
    assert plan.error == "case.yaml: no such file" and plan.directory is None


def test_a_mesh_file_is_asked_for_by_its_own_flag_and_its_schema_kept_apart():
    case = {"root": "CaseSpec", "kinds": {}}
    mesh = {"root": "MeshSpec", "kinds": {}}
    shown = {"error": None, "case": {"drawing": {"file": "a.step"}}}
    runner = Recorder(
        *(CommandResult(0, json.dumps(reply)) for reply in (mesh, case, shown, {"error": None}))
    )
    commands = SolverCommands(runner)
    assert commands.schema(MESH_FILE).root == "MeshSpec"
    assert commands.schema(CASE_FILE).root == "CaseSpec"
    assert commands.schema(MESH_FILE).root == "MeshSpec"  # kept, not asked again
    assert commands.show("m.yaml", MESH_FILE).case == shown["case"]
    assert commands.write("m.yaml", {}, "old", MESH_FILE) is None
    assert [arguments for arguments, _ in runner.calls] == [
        ["schema", "--mesh-file"],
        ["schema"],
        ["show", "m.yaml", "--mesh-file"],
        ["write", "m.yaml", "--mesh-file", "--relative-to", "old"],
    ]


def test_the_drawing_sends_the_mesh_file_and_reads_its_faces():
    reply = {
        "error": None,
        "surface": "out/drawing.vtp",
        "solids": ["tank"],
        "faces": [{"kind": "plane", "area": 1.0, "point": [0, 0, 0]}],
        "patches": ["walls"],
        "landed": {},
        "pipes": [0],
        "ducts": [1, 2],
        "selections_error": "two patches share a face",
    }
    runner = Recorder(
        CommandResult(0, json.dumps(reply)), CommandResult(2, json.dumps({"error": "no drawing"}))
    )
    commands = SolverCommands(runner)
    view = commands.drawing({"drawing": {"file": "tank.step"}}, "mesh_dir", "out")
    ((arguments, stdin),) = runner.calls
    assert arguments == ["drawing", "out", "--relative-to", "mesh_dir"]
    assert json.loads(stdin) == {"mesh_file": {"drawing": {"file": "tank.step"}}}
    assert (str(view.surface), view.solids, view.patches, view.pipes, view.ducts) == (
        "out/drawing.vtp", ["tank"], ["walls"], [0], [1, 2],
    )  # fmt: skip
    assert view.selections_error == "two patches share a face"
    assert commands.drawing({}, "mesh_dir", "out").error == "no drawing"
