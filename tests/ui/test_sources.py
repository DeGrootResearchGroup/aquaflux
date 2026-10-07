"""Tests for the viewer's sources: a run's output directory, and VTK files named directly.

The files are written by the solver's own writers (``write_vtu``, ``write_pvd``, ``StepHistory``), so
these also pin that what the solver writes is what the viewer reads.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

pv = pytest.importorskip("pyvista")

from aquaflux.io import write_pvd, write_vtu  # noqa: E402
from aquaflux.mesh import structured_grid_3d  # noqa: E402
from aquaflux.solve import StepHistory, StepReport  # noqa: E402
from aquaflux_ui import RunDirectory, VtkFiles, open_source  # noqa: E402


@pytest.fixture(scope="module")
def mesh():
    return structured_grid_3d(3, 2, 2)


def _run_directory(path, mesh, *, record=True, history=True):
    """What ``aquaflux run`` leaves: fields, history, the case and run records."""
    path.mkdir()
    write_vtu(mesh, {"p": np.arange(mesh.n_cells, dtype=float)}, path / "fields.vtu")
    written = ["fields.vtu"]
    if history:
        with StepHistory(path / "history.csv") as steps:
            steps.on_checkpoint(StepReport(0, 3, 0.5, 0.25, 1.0))
        written.append("history.csv")
    (path / "case.yaml").write_text(
        yaml.safe_dump(
            {"physics": {"kind": "Laminar"}, "mesh": {"kind": "OpenFOAMMesh", "path": "m"}}
        )
    )
    if record:
        (path / "run.yaml").write_text(
            yaml.safe_dump(
                {
                    "case": "/somewhere/channel.yaml",
                    "aquaflux": {"version": "0.1", "commit": "0123456789abcdef", "modified": True},
                    "solver": "FlowMarch",
                    "converged": True,
                    "steps": 1,
                    "residual": 0.5,
                    "written": [*written, "case.yaml"],
                }
            )
        )
    return path


def test_a_run_directory_shows_the_fields_the_run_record_lists(tmp_path, mesh):
    source = RunDirectory(_run_directory(tmp_path / "results", mesh))

    assert source.title == "channel.yaml"
    assert source.times() == (0.0,)
    frame = source.frame(0)
    assert list(frame) == ["fields"]
    grid = frame["fields"]
    assert grid.n_cells == mesh.n_cells
    np.testing.assert_array_equal(grid.cell_data["p"], np.arange(mesh.n_cells))

    info = source.info()
    assert info["Physics"] == "Laminar" and info["Solver"] == "FlowMarch"
    assert info["Version"] == "0.1 0123456789 (modified)"
    assert source.history().n_steps == 1


def test_only_the_files_the_run_record_lists_are_shown(tmp_path, mesh):
    # A file left in the directory by something else is not this run's result.
    directory = _run_directory(tmp_path / "results", mesh)
    write_vtu(mesh, {"q": np.zeros(mesh.n_cells)}, directory / "stray.vtu")
    assert list(RunDirectory(directory).frame(0)) == ["fields"]


def test_a_directory_without_a_run_record_shows_every_vtk_file_in_it(tmp_path, mesh):
    directory = _run_directory(tmp_path / "results", mesh, record=False)
    write_vtu(mesh, {"q": np.zeros(mesh.n_cells)}, directory / "more.vtu")
    source = RunDirectory(directory)
    assert sorted(source.frame(0)) == ["fields", "more"]
    assert source.history() is not None  # history.csv, by its default name
    assert source.title == "results"


def test_a_run_with_no_history_has_none(tmp_path, mesh):
    source = RunDirectory(_run_directory(tmp_path / "results", mesh, history=False))
    assert source.history() is None


def test_a_run_that_did_not_converge_says_why_it_has_nothing_to_show(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "run.yaml").write_text(
        yaml.safe_dump({"converged": False, "written": ["march.log", "history.csv"]})
    )
    with pytest.raises(FileNotFoundError, match="did not converge, so it wrote no fields"):
        RunDirectory(directory)


def test_a_series_shows_each_step_and_keeps_the_unchanging_file_as_one_object(tmp_path, mesh):
    for step in range(3):
        write_vtu(mesh, {"t": np.full(mesh.n_cells, float(step))}, tmp_path / f"s{step}.vtu")
    write_pvd(tmp_path / "series.pvd", [(0.0, "s0.vtu"), (0.5, "s1.vtu"), (1.0, "s2.vtu")])
    write_vtu(mesh, None, tmp_path / "mesh.vtu")

    source = VtkFiles([tmp_path / "series.pvd", tmp_path / "mesh.vtu"])
    assert source.times() == (0.0, 0.5, 1.0)
    frames = [source.frame(index) for index in range(3)]
    for step, frame in enumerate(frames):
        np.testing.assert_array_equal(frame["series"].cell_data["t"], step)
    # Read once and handed out again: a viewer recognizes an unchanged dataset by identity.
    assert frames[0]["mesh"] is frames[2]["mesh"]
    with pytest.raises(IndexError):
        source.frame(3)


def test_a_multiblock_contributes_each_block_by_its_name(tmp_path):
    blocks = pv.MultiBlock(
        {"floor": pv.Plane(i_resolution=2, j_resolution=2), "bunny": pv.Sphere()}
    )
    blocks.save(tmp_path / "patches.vtm")
    frame = VtkFiles([tmp_path / "patches.vtm"]).frame(0)
    assert sorted(frame) == ["bunny", "floor"]
    assert frame["floor"].n_cells == 4


def test_two_datasets_with_one_name_are_both_kept(tmp_path, mesh):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    write_vtu(mesh, None, tmp_path / "a" / "fields.vtu")
    write_vtu(mesh, None, tmp_path / "b" / "fields.vtu")
    frame = VtkFiles([tmp_path / "a" / "fields.vtu", tmp_path / "b" / "fields.vtu"]).frame(0)
    assert list(frame) == ["fields", "fields (2)"]


def test_open_source_takes_a_directory_or_a_file(tmp_path, mesh):
    directory = _run_directory(tmp_path / "results", mesh)
    assert isinstance(open_source(directory), RunDirectory)
    assert isinstance(open_source(directory / "fields.vtu"), VtkFiles)


@pytest.mark.parametrize(
    ("name", "error", "match"),
    [
        ("fields.txt", ValueError, "is not one of"),
        ("absent.vtu", FileNotFoundError, "does not exist"),
    ],
)
def test_a_file_that_cannot_be_shown_is_refused(tmp_path, name, error, match):
    (tmp_path / "fields.txt").write_text("")
    with pytest.raises(error, match=match):
        VtkFiles([tmp_path / name])
