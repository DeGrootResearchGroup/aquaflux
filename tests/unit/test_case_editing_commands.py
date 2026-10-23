"""Tests for the ``aquaflux schema``, ``show``, ``plan``, ``write``, ``mesh`` and ``serve`` commands a case-file editor works through.

Run in this process through :func:`aquaflux.__main__.main`, reading what each prints, so what is
pinned is the contract a program relies on: JSON on standard output, the refusal's reason in it, and
exit status 2 for a refusal.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
from pathlib import Path

import pytest
from aquaflux.__main__ import main, serve
from aquaflux.case import (
    Fluid,
    OpenFOAMTime,
    prepare_run,
    read_case,
    read_case_document,
)
from aquaflux.case.paths import relocated

from tests.support.case_meshes import grid_mesh

REPO = Path(__file__).resolve().parents[2]
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"


def _run(capsys, argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    status = main(argv)
    return status, json.loads(capsys.readouterr().out)


def test_schema_describes_the_case_file_from_its_top_level_sections(capsys):
    status, schema = _run(capsys, ["schema"])
    assert status == 0 and schema["root"] == "CaseSpec"
    sections = [field["name"] for field in schema["kinds"]["CaseSpec"]["fields"]]
    assert sections[:5] == ["mesh", "physics", "boundaries", "fluid", "numerics"]
    assert set(schema["one_form_sections"]) == {"fluid", "numerics", "outputs"}
    # A boundary is one of the patch kinds the registry holds -- read from it, not listed here.
    (boundaries,) = [f for f in schema["kinds"]["CaseSpec"]["fields"] if f["name"] == "boundaries"]
    assert boundaries["accepts"][0]["of"][0]["kinds"] == [
        "Inlet",
        "DevelopedInlet",
        "Outlet",
        "Wall",
        "Lamp",
    ]


def test_show_prints_the_document_as_aquaflux_reads_it(capsys):
    status, shown = _run(capsys, ["show", str(PITZDAILY)])
    assert status == 0 and shown["error"] is None
    assert shown["case"] == read_case_document(PITZDAILY)


def test_show_still_shows_a_case_it_refuses_and_says_why(capsys, tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text(PITZDAILY.read_text().replace("kind: Wall", "kind: Wal", 1))
    status, shown = _run(capsys, ["show", str(case)])
    assert status == 0
    assert "unknown kind 'Wal'" in shown["error"] and "boundaries" in shown["error"]
    assert shown["case"]["boundaries"]


def test_show_refuses_a_file_that_is_not_yaml(capsys, tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text("mesh: [unclosed\n")
    status, shown = _run(capsys, ["show", str(case)])
    assert status == 2 and str(case) in shown["error"]


def test_plan_says_where_a_run_writes_and_what_it_would_replace(capsys, tmp_path, monkeypatch):
    case = tmp_path / "case.yaml"
    status, reply = _run(capsys, ["show", str(PITZDAILY)])
    _run(
        capsys,
        ["write", str(case), "--relative-to", str(PITZDAILY.parent)],
        json.dumps(reply),
        monkeypatch,
    )
    status, plan = _run(capsys, ["plan", str(case)])
    results = (tmp_path / "results").resolve()
    assert status == 0 and plan["error"] is None
    assert plan["directory"] == str(results)
    assert (plan["log"], plan["history"]) == (
        str(results / "march.log"),
        str(results / "history.csv"),
    )
    assert plan["occupied"] == []

    # What `aquaflux run` would refuse to replace, by the same rule.
    results.mkdir()
    (results / "history.csv").write_text("")
    assert _run(capsys, ["plan", str(case)])[1]["occupied"] == [str(results)]
    with pytest.raises(FileExistsError, match="already holds results"):
        prepare_run(case)


def test_plan_refuses_a_file_it_cannot_read(capsys, tmp_path):
    status, reply = _run(capsys, ["plan", str(tmp_path / "missing.yaml")])
    assert status == 2 and "missing.yaml" in reply["error"]


def test_write_writes_a_case_that_reads_back_equal(capsys, tmp_path, monkeypatch):
    written = tmp_path / "copy.yaml"
    document = read_case_document(PITZDAILY)
    status, result = _run(
        capsys, ["write", str(written)], json.dumps({"case": document}), monkeypatch
    )
    assert status == 0 and result == {"error": None, "path": str(written)}
    assert read_case(written).spec == read_case(PITZDAILY).spec


def test_write_refuses_a_case_it_cannot_read_and_writes_nothing(capsys, tmp_path, monkeypatch):
    written = tmp_path / "copy.yaml"
    document = read_case_document(PITZDAILY)
    document["fluid"]["density"] = "heavy"
    status, result = _run(
        capsys, ["write", str(written)], json.dumps({"case": document}), monkeypatch
    )
    assert status == 2 and "'heavy'" in result["error"]
    assert not written.exists()


@pytest.mark.parametrize("given", ["not json", "{}"])
def test_write_refuses_input_that_holds_no_case(capsys, tmp_path, monkeypatch, given):
    status, result = _run(capsys, ["write", str(tmp_path / "x.yaml")], given, monkeypatch)
    assert status == 2 and result["error"]


def test_a_copy_written_elsewhere_has_its_paths_re_based_so_it_finds_its_mesh(
    capsys, tmp_path, monkeypatch
):
    written = tmp_path / "elsewhere" / "copy.yaml"
    written.parent.mkdir()
    document = read_case_document(PITZDAILY)
    status, _ = _run(
        capsys,
        ["write", str(written), "--relative-to", str(PITZDAILY.parent)],
        json.dumps({"case": document}),
        monkeypatch,
    )
    assert status == 0
    copy, original = read_case(written), read_case(PITZDAILY)
    mesh = (written.parent / copy.spec.mesh.path).resolve()
    assert mesh == (PITZDAILY.parent / original.spec.mesh.path).resolve()
    # The output directory stays beside the case file that runs it.
    assert copy.spec.outputs.directory == original.spec.outputs.directory


def test_relocating_a_case_re_bases_each_relative_path_it_states_and_no_other(tmp_path):
    spec = read_case(PITZDAILY).spec
    spec = dataclasses.replace(
        spec,
        outputs=dataclasses.replace(
            spec.outputs,
            fields=(*spec.outputs.fields, OpenFOAMTime(case="of_case", time="5")),
        ),
    )
    moved = relocated(spec, Path("/a/b"), Path("/a/c/d"))
    assert moved.mesh.path == "../../b/" + spec.mesh.path
    assert moved.outputs.fields[-1].case == "../../b/of_case"
    assert moved.outputs.fields[0] == spec.outputs.fields[0]  # a Vtk writer has no path of its own
    assert moved.outputs.directory == spec.outputs.directory  # where this file's runs write
    absolute = dataclasses.replace(spec, mesh=dataclasses.replace(spec.mesh, path="/meshes/m"))
    assert relocated(absolute, Path("/a"), Path("/z")).mesh.path == "/meshes/m"
    fluid = Fluid(density=1.0, kinematic_viscosity=1.0e-5)  # names no file
    assert relocated(fluid, Path("/a"), Path("/z")) is fluid


def test_export_mesh_writes_a_mesh_and_its_patches_as_vtk(capsys, tmp_path, monkeypatch):
    section = grid_mesh([4, 3], [2.0, 1.0])
    status, reply = _run(
        capsys, ["export-mesh", str(tmp_path)], json.dumps({"mesh": section}), monkeypatch
    )
    assert status == 0 and reply["error"] is None
    assert (tmp_path / "mesh.vtu").is_file() and reply["mesh"] == str(tmp_path / "mesh.vtu")
    assert (reply["cells"], reply["dim"]) == (12, 2)
    assert {p["name"]: p["faces"] for p in reply["patches"]} == {
        "left": 3, "right": 3, "bottom": 4, "top": 4,
    }  # fmt: skip
    # The patches are written too, one block each named by its patch, for a viewer to highlight.
    index = (tmp_path / "patches.vtm").read_text()
    assert sorted(re.findall(r'name="([^"]+)"', index)) == ["bottom", "left", "right", "top"]


def test_export_mesh_reads_a_mesh_relative_to_the_case_and_reports_its_groups(
    capsys, tmp_path, monkeypatch
):
    section = read_case_document(PITZDAILY)["mesh"]
    status, reply = _run(
        capsys,
        ["export-mesh", str(tmp_path), "--relative-to", str(PITZDAILY.parent)],
        json.dumps({"mesh": section}),
        monkeypatch,
    )
    assert status == 0 and reply["cells"] == 12225
    assert reply["groups"] == {"wall": ["upperWall", "lowerWall"]}


@pytest.mark.parametrize(
    ("section", "match"),
    [({"kind": "Wall"}, "names a mesh source"), ({"kind": "OpenFOAMMesh", "path": "nowhere"}, "")],
)
def test_export_mesh_refuses_a_section_that_is_no_mesh(
    capsys, tmp_path, monkeypatch, section, match
):
    status, reply = _run(
        capsys, ["export-mesh", str(tmp_path)], json.dumps({"mesh": section}), monkeypatch
    )
    assert status == 2 and match in reply["error"]


def _serve(*requests):
    replies = io.StringIO()
    serve(io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), replies)
    return [json.loads(line) for line in replies.getvalue().splitlines()]


def test_serve_answers_each_request_as_the_command_would_have(tmp_path):
    case = read_case_document(PITZDAILY)
    show, write, reread = _serve(
        {"arguments": ["show", str(PITZDAILY)], "stdin": None},
        {
            "arguments": [
                "write",
                str(tmp_path / "copy.yaml"),
                "--relative-to",
                str(PITZDAILY.parent),
            ],
            "stdin": json.dumps({"case": case}),  # a request's own standard input
        },
        {"arguments": ["show", str(tmp_path / "copy.yaml")]},
    )
    assert show["status"] == 0 and json.loads(show["output"])["case"] == case
    assert write["status"] == 0 and json.loads(write["output"])["error"] is None
    assert reread["status"] == 0 and json.loads(reread["output"])["error"] is None


def test_serve_reports_a_refusal_and_keeps_answering(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mesh: [unclosed\n")
    refused, parse_error, unserved, after = _serve(
        {"arguments": ["check", str(bad)]},
        {"arguments": ["show"]},  # no case file: the argument parser refuses it
        # A solve is not answered here; the file does not exist, so even answered it would solve nothing.
        {"arguments": ["run", str(tmp_path / "absent.yaml")]},
        {"arguments": ["schema"]},
    )
    assert refused["status"] == 2 and "aquaflux:" in refused["errors"]
    assert parse_error["status"] == 2 and "required" in parse_error["errors"]
    assert unserved["status"] == 2 and "answers only" in unserved["errors"]
    assert after["status"] == 0 and json.loads(after["output"])["root"] == "CaseSpec"


def test_serve_answers_a_request_that_fails_unexpectedly_with_its_traceback(monkeypatch):
    def broken():
        raise RuntimeError("an unforeseen failure")

    import aquaflux.__main__ as command

    monkeypatch.setattr(
        command, "_CASE_FILE", dataclasses.replace(command._CASE_FILE, schema=broken)
    )
    failed, bad_line = _serve({"arguments": ["schema"]}, {"no arguments": True})
    assert failed["status"] == 1 and "an unforeseen failure" in failed["errors"]
    assert bad_line["status"] == 2 and "bad request" in bad_line["errors"]


def test_serve_replies_on_a_channel_nothing_else_writes_to():
    """Compiled code writing to standard output directly must not land among the replies."""
    import subprocess
    import sys

    script = (
        "import dataclasses, os, aquaflux.__main__ as m\n"
        "def noisy():\n"
        "    os.write(1, b'written straight to the file descriptor\\n')\n"
        "    print('printed')\n"
        "    return {'root': 'CaseSpec'}\n"
        "m._CASE_FILE = dataclasses.replace(m._CASE_FILE, schema=noisy)\n"
        "raise SystemExit(m._serve_process())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps({"arguments": ["schema"]}) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (reply,) = [json.loads(line) for line in done.stdout.splitlines()]
    assert done.returncode == 0 and json.loads(reply["output"].splitlines()[-1]) == {
        "root": "CaseSpec"
    }
    assert "written straight to the file descriptor" in done.stderr


def test_serve_returns_quietly_when_interrupted_or_when_its_reader_has_gone():
    def interrupted():
        yield json.dumps({"arguments": ["schema"]}) + "\n"
        raise KeyboardInterrupt  # Ctrl-C while waiting for the next request

    replies = io.StringIO()
    serve(interrupted(), replies)
    assert json.loads(replies.getvalue())["status"] == 0  # the request before it was answered

    class Closed(io.StringIO):
        def write(self, text):
            raise BrokenPipeError

    serve(io.StringIO(json.dumps({"arguments": ["schema"]}) + "\n"), Closed())


# --- the same commands for a mesh file ---------------------------------------------------------

MESH_FILE = {
    "drawing": {"file": "parts/vessel.step"},
    "region": {"solids": ["vessel"]},
    "sizes": {"max_size": 0.02, "feature_angle": 0.785},
    "patches": {"inlet": {"kind": "FacesNear", "points": [[0.0, 0.0, 1.0]]}},
}


def test_schema_describes_a_mesh_file_when_asked_for_one(capsys):
    status, schema = _run(capsys, ["schema", "--mesh-file"])
    assert status == 0 and schema["root"] == "MeshSpec"
    sections = [field["name"] for field in schema["kinds"]["MeshSpec"]["fields"]]
    assert sections[:4] == ["drawing", "region", "sizes", "patches"]


def test_a_mesh_file_is_shown_and_written_as_a_mesh_file(capsys, tmp_path, monkeypatch):
    original = tmp_path / "meshes" / "mesh.yaml"
    original.parent.mkdir()
    given = json.dumps({"case": MESH_FILE})
    status, result = _run(capsys, ["write", str(original), "--mesh-file"], given, monkeypatch)
    assert status == 0 and result["error"] is None
    status, shown = _run(capsys, ["show", str(original), "--mesh-file"])
    assert status == 0 and shown == {"error": None, "case": MESH_FILE}
    # Read as a case file it is refused, but still shown, for what it is not.
    status, shown = _run(capsys, ["show", str(original)])
    assert shown["case"] == MESH_FILE and "mesh" in shown["error"]
    # A copy saved elsewhere still finds its drawing.
    copy = tmp_path / "copy.yaml"
    arguments = ["write", str(copy), "--mesh-file", "--relative-to", str(original.parent)]
    status, _ = _run(capsys, arguments, given, monkeypatch)
    _, shown = _run(capsys, ["show", str(copy), "--mesh-file"])
    assert shown["case"]["drawing"]["file"] == "meshes/parts/vessel.step"


def test_a_mesh_file_without_its_sizes_is_refused_and_writes_nothing(capsys, tmp_path, monkeypatch):
    unfinished = {key: value for key, value in MESH_FILE.items() if key != "sizes"}
    target = tmp_path / "mesh.yaml"
    status, result = _run(
        capsys, ["write", str(target), "--mesh-file"], json.dumps({"case": unfinished}), monkeypatch
    )
    assert status == 2 and "sizes" in result["error"] and not target.exists()


def test_export_mesh_says_how_a_mesh_was_made_as_its_file_records_it(capsys, tmp_path, monkeypatch):
    section = grid_mesh([4, 3], [2.0, 1.0])  # written without provenance
    _, reply = _run(
        capsys, ["export-mesh", str(tmp_path)], json.dumps({"mesh": section}), monkeypatch
    )
    assert reply["made"] == {}


def test_the_drawing_of_a_mesh_file_without_one_is_refused(capsys, tmp_path, monkeypatch):
    status, reply = _run(
        capsys, ["drawing", str(tmp_path)], json.dumps({"mesh_file": {"region": {}}}), monkeypatch
    )
    assert status == 2 and "drawing is needed" in reply["error"]
