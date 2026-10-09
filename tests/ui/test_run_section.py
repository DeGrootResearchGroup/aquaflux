"""The Run section on a built page: when Run may start, what it says the case will do, an earlier run.

The page is built whole, with the committed pitzDaily case open and the solver's plan answered by a
stub, so nothing is solved; what is pinned is the section's own rules -- a run reads the saved file,
replaces earlier results only when told to, and says why it will not start when it will not.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("trame")

import pyvista as pv
from aquaflux.case import case_schema, read_case_document
from aquaflux.solve import StepHistory, StepReport
from aquaflux_ui.case_form import CaseSchema
from aquaflux_ui.solver_commands import RunPlan

pytestmark = pytest.mark.skipif(
    not pv.system_supports_plotting(), reason="this machine cannot render: no display"
)

REPO = Path(__file__).resolve().parents[2]
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"


class _Plans:
    """Answers ``plan`` with :attr:`answer`, which a test changes as it goes."""

    def __init__(self, answer: RunPlan) -> None:
        self.answer = answer

    def plan(self, path) -> RunPlan:
        del path
        return self.answer


def _page(tmp_path, plan: RunPlan):
    from aquaflux_ui.app import Workspace
    from trame.app import get_server

    server = get_server(name=f"test-run-{tmp_path.name}", client_type="vue3")
    commands = _Plans(plan)
    workspace = Workspace(None, server, commands=commands)
    setup = workspace.setup
    setup.schema = CaseSchema(case_schema())
    setup.document = read_case_document(PITZDAILY)
    setup.path = PITZDAILY
    server.state.update({"case_loaded": True, "case_path": str(PITZDAILY), "case_modified": False})
    return workspace, commands, server.state


def _plan(directory: Path, *occupied: Path) -> RunPlan:
    return RunPlan(None, directory, directory / "march.log", directory / "history.csv", occupied)


def test_run_starts_only_from_a_saved_case_and_replaces_results_only_when_told_to(tmp_path):
    results = tmp_path / "results"

    async def scenario():
        workspace, commands, state = _page(tmp_path, _plan(results))
        run = workspace.run
        await run.replan()
        assert (state.run_ready, state.run_block) == (True, "")

        state.case_modified = True
        run._update_ready()
        assert not state.run_ready and state.run_block.startswith("Save the case first")

        state.case_modified = False
        commands.answer = _plan(results, results)
        await run.replan()
        assert not state.run_ready and "holds results" in state.run_block
        # The note beside the switch names the directory as the case file sees it.
        assert state.run_directory_shown == os.path.relpath(results, PITZDAILY.parent) + "/"
        state.run_overwrite = True
        run._update_ready()
        assert state.run_ready

        state.case_loaded = False
        run._update_ready()
        assert not state.run_ready and state.run_block.startswith("Open a case")

    asyncio.run(scenario())


def test_the_summary_says_what_the_case_file_sets_the_run_to_do(tmp_path):
    async def scenario():
        workspace, _, state = _page(tmp_path, _plan(PITZDAILY.parent / "results"))
        await workspace.run.replan()
        return {row["label"]: row["value"] for row in state.run_summary}

    summary = asyncio.run(scenario())
    assert summary == {
        "Physics": "RANS",
        "Solver": "CoupledMarch",
        "Continuation": "ViscosityRamp, 16 stations",
        "Inner loop": "Dual time, ≤ 5 iterations",
        "Max steps": "150 per segment",
        "Stops below": "1.0e-05",
        "Output": "results/",
    }


def test_an_earlier_runs_history_is_shown_until_a_new_run_starts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    with StepHistory(results / "history.csv") as history:
        for step, residual in enumerate([1.0, 0.1, 0.01]):
            history.on_checkpoint(StepReport(step, 3, residual, residual, 1.0))

    async def scenario():
        workspace, _, state = _page(tmp_path, _plan(results, results))
        await workspace.run.replan()
        return state

    state = asyncio.run(scenario())
    assert (state.run_state, state.run_state_text) == ("earlier", "Showing the earlier run")
    assert [tile["value"] for tile in state.run_tiles][:2] == ["1.00e-02", "3"]


def test_the_per_equation_view_redraws_the_residual_from_each_equations_column(tmp_path):
    from aquaflux_ui.run_section import BY_EQUATION, OVERALL

    results = tmp_path / "results"
    results.mkdir()
    with StepHistory(results / "history.csv") as history:
        for step, residual in enumerate([1.0, 0.1]):
            history.on_residuals({"u": residual / 2, "p": residual / 3})
            history.on_checkpoint(StepReport(step, 3, residual, residual, 1.0))

    async def scenario():
        workspace, _, state = _page(tmp_path, _plan(results, results))
        drawn = []
        workspace.run._figures = dict.fromkeys(workspace.run._figures, lambda figure: None)
        workspace.run._figures["residual"] = drawn.append
        await workspace.run.replan()
        assert [trace.name for trace in drawn[-1].data] == ["Residual"]
        # The page's toggle holds one of these two strings; the view follows it.
        assert {OVERALL, BY_EQUATION} == {"overall", "equations"}
        state.run_residual_view = BY_EQUATION
        workspace.run._redraw(force=True)
        assert [trace.name for trace in drawn[-1].data] == ["u", "p"]

    asyncio.run(scenario())


def test_a_history_older_than_the_run_being_followed_is_not_shown_as_its_own(tmp_path):
    # Replacing an earlier run, its history sits in the directory until the new run's first step.
    import time

    results = tmp_path / "results"
    results.mkdir()
    with StepHistory(results / "history.csv") as history:
        history.on_checkpoint(StepReport(0, 3, 1.0, 1.0, 1.0))

    async def scenario():
        workspace, _, _ = _page(tmp_path, _plan(results, results))
        run = workspace.run
        await run.replan()
        assert run._history(since=None).n_steps == 1
        assert run._history(since=time.time() + 60.0) is None

    asyncio.run(scenario())


def test_a_finished_run_keeps_its_ending_and_the_next_run_asks_again_to_replace(tmp_path):
    from aquaflux_ui.run_process import CaseRun

    results = tmp_path / "results"
    results.mkdir()
    stand_in = tmp_path / "stand_in.py"
    # Stands in for `aquaflux run`: writes a one-step history and exits converged.
    stand_in.write_text(
        f"open({str(results / 'history.csv')!r}, 'w').write("
        "'step,seconds,residual_norm,residual_ratio\\n1,0.5,1e-06,1e-06\\n')\n"
    )

    async def scenario():
        workspace, _, state = _page(tmp_path, _plan(results, results))
        run = workspace.run
        await run.replan()
        state.run_overwrite = True
        run._update_ready()
        original = CaseRun.__init__

        def standing_in(self, case, console, *, overwrite=False, command=None):
            original(
                self, case, console, overwrite=overwrite, command=[sys.executable, str(stand_in)]
            )

        CaseRun.__init__ = standing_in
        try:
            run.start()
            assert state.run_overwrite is False  # asked again for the next run
            await asyncio.wait_for(asyncio.gather(*run._tasks), timeout=60)
        finally:
            CaseRun.__init__ = original
        return state

    state = asyncio.run(scenario())
    assert (state.run_state, state.run_state_text) == ("converged", "Converged")
    assert state.run_tiles[1]["value"] == "1"
