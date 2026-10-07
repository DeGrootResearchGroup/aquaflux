"""The whole page is built, and the Results section's entry methods change the page's state.

Building the page constructs every section's widgets and an off-screen render window, so a broken
widget argument or a section that fails to build is caught here rather than in a browser. Nothing is
rendered and no browser connects; the test skips where this machine cannot render at all (a Linux
machine without a display).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("trame")

import pyvista as pv
from aquaflux.io import write_vtu
from aquaflux.mesh import structured_grid_3d
from aquaflux_ui import VtkFiles

pytestmark = pytest.mark.skipif(
    not pv.system_supports_plotting(), reason="this machine cannot render: no display"
)


@pytest.fixture
def workspace(tmp_path):
    from aquaflux_ui.app import Workspace
    from trame.app import get_server

    mesh = structured_grid_3d(3, 2, 2)
    write_vtu(mesh, {"p": np.arange(mesh.n_cells, dtype=float)}, tmp_path / "fields.vtu")
    server = get_server(name=f"test-{tmp_path.name}", client_type="vue3")
    return Workspace(VtkFiles([tmp_path / "fields.vtu"]), server)


def test_the_page_opens_on_results_with_every_section_built(workspace):
    assert [section.key for section in workspace.sections] == ["setup", "run", "results"]
    state = workspace.server.state
    assert state.section == "results" and state.theme == "light"
    assert state.dataset == "fields" and state.field == "p"


def test_slices_and_thresholds_are_added_edited_and_removed_through_the_page(workspace):
    results, state = workspace.results, workspace.server.state
    results.add_slice()
    results.add_slice()
    assert len(state.slices) == 2 and state.slices[0]["id"] != state.slices[1]["id"]
    assert state.surface_opacity == 0.2  # faded for the first one only
    results.edit_slice(1, "axis", "x")
    results.edit_slice(1, "coordinate", "0.25")
    assert (state.slices[1]["axis"], state.slices[1]["coordinate"]) == ("x", 0.25)
    results.remove_slice(0)
    assert [entry["axis"] for entry in state.slices] == ["x"]

    results.add_threshold()
    results.edit_threshold(0, "low", "3")
    assert (state.thresholds[0]["field"], state.thresholds[0]["low"]) == ("p", 3.0)
    results.remove_threshold(0)
    assert state.thresholds == []


def test_a_page_opened_on_a_case_file_opens_on_setup_with_no_results(tmp_path):
    from aquaflux_ui.app import NoResultsSection, Workspace
    from trame.app import get_server

    server = get_server(name=f"test-case-{tmp_path.name}", client_type="vue3")
    workspace = Workspace(None, server, case=tmp_path / "case.yaml")
    assert server.state.section == "setup"
    assert isinstance(workspace.results, NoResultsSection)
    assert server.state.case_loaded is False  # opened once the page is served


def test_choosing_a_boundary_highlights_its_patches_and_choosing_it_again_clears_them(tmp_path):
    from aquaflux_ui.app import Workspace
    from trame.app import get_server

    server = get_server(name=f"test-select-{tmp_path.name}", client_type="vue3")
    setup = Workspace(None, server, case=tmp_path / "case.yaml").setup
    server.controller.mesh_view_update = lambda: None
    setup._groups = {"wall": ["upperWall", "lowerWall"]}
    server.state.mesh_names = ["inlet", "upperWall", "lowerWall", "wall"]
    setup._label_clicked("entry", "wall")
    assert server.state.mesh_selected == ["upperWall", "lowerWall"]
    setup._label_clicked("entry", "wall")
    assert server.state.mesh_selected == []
    setup._label_clicked("number", "inlet")  # a setting's name, not a table entry
    assert server.state.mesh_selected == []
    setup._label_clicked("entry", "inlet")
    assert server.state.mesh_selected == ["inlet"]


class SlowCommands:
    """Solver commands answering at once, except a schema that takes a moment and may fail first."""

    def __init__(self, server, failures=0):
        import json

        from aquaflux_ui.solver_commands import SolverCommands
        from aquaflux_ui.solver_worker import CommandResult

        self.server, self.failures, self.schema_calls, self.opening_seen = server, failures, 0, []
        kinds = {"CaseSpec": {"summary": "A case.", "fields": []}}
        reply = CommandResult(0, json.dumps({"root": "CaseSpec", "kinds": kinds}))
        self._real = SolverCommands(lambda arguments, stdin: reply)

    def schema(self):
        import time

        self.schema_calls += 1
        time.sleep(0.05)
        if self.schema_calls <= self.failures:
            raise RuntimeError("`aquaflux schema` failed: no solver")
        return self._real.schema()

    def show(self, path):
        from aquaflux_ui.solver_commands import CaseDocument

        self.opening_seen.append(self.server.state.case_opening)
        return CaseDocument(path, {}, None)


def _setup_with(commands_for, name):
    from aquaflux_ui.app import Workspace
    from trame.app import get_server

    server = get_server(name=name, client_type="vue3")
    commands = commands_for(server)
    return Workspace(None, server, commands=commands).setup, commands, server


def test_a_case_opened_while_the_schema_is_still_being_fetched_waits_for_that_one_request(tmp_path):
    import asyncio

    setup, commands, server = _setup_with(SlowCommands, f"test-prefetch-{tmp_path.name}")

    async def page():
        prefetch = asyncio.ensure_future(setup._prefetch_schema())
        await asyncio.sleep(0)  # the prefetch is under way when the case is chosen
        assert await setup.open(tmp_path / "pitz.yaml") is None
        await prefetch

    asyncio.run(page())
    assert commands.schema_calls == 1
    assert commands.opening_seen == ["pitz.yaml"]  # said while the solver reads it
    assert server.state.case_opening == "" and server.state.case_loaded is True


def test_a_schema_that_failed_in_the_background_is_asked_for_again_when_a_case_is_opened(tmp_path):
    import asyncio

    setup, commands, server = _setup_with(
        lambda server: SlowCommands(server, failures=1), f"test-retry-{tmp_path.name}"
    )

    async def page():
        await setup._prefetch_schema()  # fails quietly
        assert await setup.open(tmp_path / "pitz.yaml") is None

    asyncio.run(page())
    assert commands.schema_calls == 2 and server.state.case_loaded is True
