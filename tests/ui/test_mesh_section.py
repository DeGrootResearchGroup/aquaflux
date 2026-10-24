"""The Mesh section on a built page: a mesh file opened, its patches picked, its mesh made and used.

The page is built whole with the solver's commands answered by a stub: the schemas are the solver's
own, the drawing a square's four faces written as real triangles, the mesh a real box written as VTK.
Nothing is rendered, so a pick is answered by the test, not by the view. What is pinned is the
section's own part -- what it shows of the solver's answers, the edits a click makes, the making of
the mesh in a process of its own, and the case given the mesh by a path from its own directory.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trame")

import pyvista as pv
from aquaflux.case import case_schema, mesh_schema
from aquaflux.io import write_patches, write_triangles, write_vtu
from aquaflux.mesh import structured_grid_3d
from aquaflux_ui.case_form import CaseSchema
from aquaflux_ui.mesh_section import quality_tiles
from aquaflux_ui.solver_commands import CaseDocument, DrawingView, MeshExport

pytestmark = pytest.mark.skipif(
    not pv.system_supports_plotting(), reason="this machine cannot render: no display"
)

#: A square's four sides, a triangle each, its faces 0..3; a point on each.
CORNERS = np.array(
    [
        [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
        [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [0, 0, 0], [0, 0, 1]],
    ],
    dtype=float,
)
POINTS = [[0.5, 0.0, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.0, 0.5, 0.5]]
MESH_FILE = {
    "drawing": {"file": "square.step"},
    "region": {"solids": ["square"]},
    "sizes": {"max_size": 0.1, "feature_angle": 0.785},
}
QUALITY = {
    "cells": 12345,
    "non_orthogonality_max": 61.4,
    "non_orthogonality_p99": 31.6,
    "volume_ratio_max": 40.0,
    "volume_ratio_p99": 15.83,
    "smallest_volume": 1e-9,
    "concave_cells": 350,
}


class StubCommands:
    """The solver's commands as the section calls them, the drawing's patches by the points given:
    each point lands on the face whose point it is."""

    def __init__(self, document: dict, pipes=(), ducts=None) -> None:
        self.document = document
        self.pipes, self.ducts = list(pipes), ducts
        self.drawings: list[dict] = []
        self.made: dict | None = {"quality": QUALITY, "seconds": 7.04}

    def schema(self, kind=None):
        name = getattr(kind, "name", "case file")
        return CaseSchema(mesh_schema() if name == "mesh file" else case_schema())

    def show(self, path, kind=None):
        return CaseDocument(Path(path), json.loads(json.dumps(self.document)), None)

    def drawing(self, mesh_file, relative_to, directory):
        self.drawings.append(json.loads(json.dumps(mesh_file)))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        surface = write_triangles(CORNERS, {"face": np.arange(4)}, directory / "drawing.vtp")
        patches, landed = ["walls"] * 4, {}
        for name, selection in (mesh_file.get("patches") or {}).items():
            landed[name] = [POINTS.index(point) for point in selection["points"]]
            for face in landed[name]:
                patches[face] = name
        faces = [{"kind": "plane", "area": 1.0, "point": point} for point in POINTS]
        return DrawingView(
            None, surface, ["square"], faces, patches, landed, pipes=self.pipes, ducts=self.ducts
        )

    def write(self, path, case, relative_to=None, kind=None):
        self.written = (Path(path), case, relative_to, getattr(kind, "name", None))

    def mesh(self, section, relative_to, directory):
        mesh = structured_grid_3d(2, 2, 2, named_boundaries=True)
        write_vtu(mesh, None, Path(directory) / "mesh.vtu")
        write_patches(mesh, None, Path(directory) / "patches.vtm")
        return MeshExport(None, Path(directory), mesh.n_cells, 3, [], {}, self.made)


def _page(tmp_path, document=MESH_FILE, **faces):
    from aquaflux_ui.app import Workspace
    from trame.app import get_server

    server = get_server(name=f"test-mesh-{tmp_path.name}", client_type="vue3")
    commands = StubCommands(document, **faces)
    workspace = Workspace(None, server, commands=commands)
    for name in ("meshing_drawing_update", "meshing_mesh_update", "mesh_view_update"):
        setattr(server.controller, name, lambda: None)
    return workspace, commands, server.state


def _click(section, face):
    """A click the view would have answered with ``face``."""
    section.drawing.pick = lambda x, y: face
    section.clicked({"position": {"x": 1, "y": 1}})


def test_quality_is_shown_as_its_file_records_it_and_nothing_when_it_records_none():
    tiles = quality_tiles({"quality": QUALITY, "seconds": 7.04})
    assert [(tile["label"], tile["value"]) for tile in tiles] == [
        ("Cells", "12,345"),
        ("Non-orthogonality p99 / max", "32° / 61°"),
        ("Volume ratio p99", "15.8"),
        ("Concave cells", "350"),
        ("Meshed in", "7.0 s"),
    ]
    assert quality_tiles({}) == [] and quality_tiles(None) == []


def test_an_opened_mesh_file_lists_its_patches_and_offers_the_drawings_solids(tmp_path):
    workspace, commands, state = _page(tmp_path)

    async def page():
        await workspace.mesh.form.open(tmp_path / "mesh.yaml")

    asyncio.run(page())
    assert commands.drawings == [MESH_FILE]
    assert [(row["name"], row["faces"], row["chosen_by"]) for row in state.meshing_patches] == [
        ("walls", 4, "others")
    ]
    assert workspace.mesh.drawing.face.tolist() == [0, 1, 2, 3]
    rows = {row["path"]: row for section in state.meshfile_sections for row in section["rows"]}
    solids = rows[json.dumps(["region", "solids"])]
    assert solids["widget"] == "choices" and [i["value"] for i in solids["items"]] == ["square"]


def _offered(state, *path):
    """The names the form offers for the list at ``path``."""
    rows = {row["path"]: row for section in state.meshfile_sections for row in section["rows"]}
    row = rows[json.dumps(list(path))]
    assert row["widget"] == "choices"
    return [item["value"] for item in row["items"]]


def test_pipes_and_ducts_are_offered_the_patches_every_face_of_which_is_one(tmp_path):
    # The square's faces 0 and 1 are a duct's sides, face 2 a pipe: the patch holding 0 and 1 is
    # offered as ducts, the one holding 2 as pipes, and the others patch (face 3) as neither, nor is
    # a patch holding a duct's side and a face that is not one.
    patches = {
        "sides": {"kind": "FacesNear", "points": [POINTS[0], POINTS[1]]},
        "tube": {"kind": "FacesNear", "points": [POINTS[2]]},
    }
    document = {**MESH_FILE, "patches": patches}
    workspace, _, state = _page(tmp_path, document, pipes=[2], ducts=[0, 1])
    asyncio.run(workspace.mesh.form.open(tmp_path / "mesh.yaml"))
    assert _offered(state, "pipes") == ["tube"]
    assert _offered(state, "ducts") == ["sides"]
    mixed = {**MESH_FILE, "patches": {"sides": {"kind": "FacesNear", "points": POINTS[:3]}}}
    workspace, _, state = _page(tmp_path / "mixed", mixed, pipes=[2], ducts=[0, 1])
    asyncio.run(workspace.mesh.form.open(tmp_path / "mesh.yaml"))
    assert _offered(state, "ducts") == [] and _offered(state, "pipes") == []


def test_ducts_are_offered_nothing_until_the_solver_can_tell_them(tmp_path):
    # Without a feature angle the solver tells no ducts (None): no patch is offered.
    workspace, _, state = _page(tmp_path, ducts=None)
    asyncio.run(workspace.mesh.form.open(tmp_path / "mesh.yaml"))
    assert _offered(state, "ducts") == []


def test_an_edit_of_the_feature_angle_asks_for_the_drawing_again_and_of_the_size_does_not(tmp_path):
    # The feature angle decides which faces are ducts' sides; the size changes nothing drawn.
    workspace, _, _ = _page(tmp_path)
    section = workspace.mesh
    asked = []

    async def recorded(frame=False):
        asked.append(frame)

    async def page():
        await section.form.open(tmp_path / "mesh.yaml")
        section.load_drawing = recorded
        document = section.form.document
        sizes = document["sizes"]
        section._changed(document, {**document, "sizes": {**sizes, "max_size": 0.2}})
        await asyncio.sleep(0)
        after_size = len(asked)
        section._changed(document, {**document, "sizes": {**sizes, "feature_angle": 1.0}})
        await asyncio.sleep(0)
        return after_size

    after_size = asyncio.run(page())
    assert (after_size, len(asked)) == (0, 1)


def test_a_click_adds_a_face_to_the_chosen_patch_and_again_takes_it_away(tmp_path):
    workspace, commands, state = _page(tmp_path)
    section = workspace.mesh

    async def page():
        await section.form.open(tmp_path / "mesh.yaml")
        section.add_patch("inlet")
        assert state.meshing_chosen == "inlet"
        _click(section, 2)
        await section.load_drawing()
        added = section.form.document["patches"]["inlet"]
        _click(section, 2)
        await section.load_drawing()
        return added

    added = asyncio.run(page())
    assert added == {"kind": "FacesNear", "points": [POINTS[2]]}
    assert "patches" not in section.form.document and state.meshfile_modified
    # The drawing was asked again after each click, with the file as it then stood.
    assert commands.drawings[1]["patches"]["inlet"]["points"] == [POINTS[2]]
    # The patch whose last face left is kept in the list, still chosen, to be given another.
    assert state.meshing_chosen == "inlet"
    assert [row["name"] for row in state.meshing_patches] == ["inlet", "walls"]


def test_a_click_ending_a_drag_picks_nothing_and_hiding_draws_a_face_no_more(tmp_path):
    workspace, _, state = _page(tmp_path)
    section = workspace.mesh

    async def page():
        await section.form.open(tmp_path / "mesh.yaml")
        section.add_patch("inlet")
        section.drawing.pressed = ((9.0, 9.0, 9.0), (0.0, 0.0, 0.0))  # the camera moved since
        _click(section, 2)
        assert "patches" not in section.form.document
        section.drawing.pressed = None
        state.meshing_click_mode = "hide"
        _click(section, 1)

    asyncio.run(page())
    assert "patches" not in section.form.document and state.meshing_hidden == 1
    shown = np.asarray(section.drawing._shown.cell_data["face"]).tolist()
    assert shown == [0, 2, 3]
    section.show_all()
    assert state.meshing_hidden == 0 and section.drawing._shown.n_cells == 4


def test_the_mesh_is_made_in_a_process_of_its_own_and_its_quality_shown(tmp_path, monkeypatch):
    import aquaflux_ui.mesh_section as module
    from aquaflux_ui.run_process import SolverRun

    stand_in = tmp_path / "stand_in.py"
    # Stands in for `aquaflux mesh`: prints its stages, writes the mesh, and exits as it would.
    stand_in.write_text(
        "import sys, pathlib\n"
        "print('   0.0 s  reading the drawing', flush=True)\n"
        "out = pathlib.Path(sys.argv[2]).parent / 'mesh.aqmesh'\n"
        "if out.exists() and '--overwrite' not in sys.argv:\n"
        "    print('aquaflux: ' + str(out) + ' already there', file=sys.stderr); sys.exit(2)\n"
        "out.write_text('a mesh')\n"
    )
    asked = []

    def standing_in(subcommand, file, console, *, overwrite=False):
        asked.append((subcommand, Path(file), overwrite))
        return SolverRun(
            subcommand, file, console, overwrite=overwrite, command=[sys.executable, str(stand_in)]
        )

    monkeypatch.setattr(module, "SolverRun", standing_in)
    workspace, _, state = _page(tmp_path)
    section = workspace.mesh
    path = tmp_path / "mesh.yaml"

    async def page():
        await section.form.open(path)
        for _ in range(2):  # the second meets the first's mesh
            section.make()
            await asyncio.wait_for(asyncio.gather(*section.form._tasks), timeout=60)
            yield {
                key: state[key]
                for key in ("meshing_state", "meshing_state_text", "meshing_log", "meshing_result")
            } | {"meshing_tiles": list(state.meshing_tiles)}

    async def run():
        return [snapshot async for snapshot in page()]

    first, second = asyncio.run(run())
    assert asked == [("mesh", path, False), ("mesh", path, False)]
    assert first["meshing_state"] == "done" and "reading the drawing" in first["meshing_log"]
    assert first["meshing_result"] and first["meshing_tiles"][0]["value"] == "12,345"
    assert second["meshing_state"] == "failed" and "Replace" in second["meshing_state_text"]


def test_the_mesh_made_becomes_the_cases_mesh_by_a_path_from_the_cases_directory(tmp_path):
    workspace, _, state = _page(tmp_path)
    mesh_directory = tmp_path / "meshes"
    mesh_directory.mkdir()
    (mesh_directory / "mesh.aqmesh").write_text("a mesh")
    case = workspace.setup.form

    async def page():
        await workspace.mesh.form.open(mesh_directory / "mesh.yaml")
        workspace.mesh.use_in_case()  # no case open yet
        assert "Open or create a case in Setup first" in state.meshfile_message
        case.schema = CaseSchema(case_schema())
        case.document, case.path = (
            {"mesh": {"kind": "OpenFOAMMesh", "path": "old"}},
            tmp_path / "cases" / "c.yaml",
        )
        workspace.mesh.use_in_case()

    asyncio.run(page())
    assert case.document["mesh"] == {"kind": "AquafluxMesh", "path": "../meshes/mesh.aqmesh"}
    assert state.section == "setup" and state.case_modified


def test_a_new_mesh_file_starts_from_its_drawing_and_is_re_based_where_it_is_saved(tmp_path):
    workspace, commands, state = _page(tmp_path)
    section = workspace.mesh
    drawings = tmp_path / "drawings"

    async def page():
        section._drawing_chosen(str(drawings / "tank.step"), "open")
        await asyncio.wait_for(asyncio.gather(*section.form._tasks), timeout=30)
        assert (section.form.path, section.form.directory) == (None, drawings.resolve())
        assert state.meshfile_name == "Not saved" and state.meshfile_modified
        await section.form.save(tmp_path / "meshes" / "mesh.yaml")

    asyncio.run(page())
    assert commands.drawings[0] == {"drawing": {"file": "tank.step"}}
    path, document, relative_to, kind = commands.written
    assert document == {"drawing": {"file": "tank.step"}}  # as started, nothing added
    assert (path, relative_to, kind) == (
        (tmp_path / "meshes" / "mesh.yaml").resolve(), drawings.resolve(), "mesh file",
    )  # fmt: skip
    assert section.form.directory == (tmp_path / "meshes").resolve()
