"""Where a viewer's data comes from: a finished run's output directory, or VTK files named directly.

A :class:`ResultSource` hands the viewer everything it shows -- the datasets of each snapshot as
PyVista objects, the times the snapshots are at, a few lines describing the run, and the march's
convergence history -- and nothing in it knows how they will be drawn. That interface is the seam a
source backed by a run still in progress would implement too; the viewer asks for the snapshot times
and the history each time it refreshes rather than holding them.

Everything is read from files, as NumPy-backed VTK objects. Nothing here imports the solver or JAX,
so the viewer runs on a machine with neither, and loads results copied from somewhere else.
"""

from __future__ import annotations

import abc
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyvista as pv
import yaml

from .history import ConvergenceHistory

__all__ = ["ResultSource", "RunDirectory", "VtkFiles", "multiblock_leaves", "open_source"]

#: The file types a source reads: one unstructured grid, a time series of them, polygonal surfaces,
#: and a multiblock collection of either.
VTK_SUFFIXES = (".vtu", ".pvd", ".vtp", ".vtm")

#: The run record and the case record a run writes into its output directory.
RUN_RECORD, CASE_RECORD = "run.yaml", "case.yaml"


class ResultSource(abc.ABC):
    """What a viewer shows: datasets per snapshot, the snapshots' times, a description, a history."""

    @property
    @abc.abstractmethod
    def title(self) -> str:
        """A short name for the results, for the page's heading."""

    @abc.abstractmethod
    def info(self) -> dict[str, str]:
        """Lines describing the results, display label to value, in display order."""

    @abc.abstractmethod
    def times(self) -> tuple[float, ...]:
        """The times of the snapshots, ascending; one entry for results with no time series."""

    @abc.abstractmethod
    def frame(self, index: int) -> dict[str, pv.DataSet]:
        """The datasets of one snapshot, by name.

        Parameters
        ----------
        index : int
            Into :meth:`times`.

        Returns
        -------
        dict of {str: pyvista.DataSet}
            Each dataset by a name unique within the snapshot. A dataset that does not change from
            one snapshot to the next is the *same* object in each, so a viewer can tell it need not
            recompute anything derived from it.
        """

    @abc.abstractmethod
    def history(self) -> ConvergenceHistory | None:
        """The march's convergence history, or ``None`` if there is none."""


class VtkFiles(ResultSource):
    """Results held in VTK files: unstructured grids, surfaces, multiblocks and ``.pvd`` series.

    A file that is not a series appears in every snapshot, read once. A ``.pvd`` series is read one
    step at a time, as it is asked for; at a time between two of its steps it shows the earlier one.

    Parameters
    ----------
    files : sequence of path-like
        The files, each ending in ``.vtu``, ``.pvd``, ``.vtp`` or ``.vtm``.
    title : str, optional
        The heading; unset, the first file's name.
    info : mapping of {str: str}, optional
        Lines describing the results.
    history : path-like, optional
        A convergence history file to plot.

    Raises
    ------
    ValueError
        If ``files`` is empty or names a file of another type.
    FileNotFoundError
        If a file does not exist.
    """

    def __init__(
        self,
        files: Sequence[str | os.PathLike[str]],
        *,
        title: str | None = None,
        info: Mapping[str, str] | None = None,
        history: str | os.PathLike[str] | None = None,
    ) -> None:
        self._files = tuple(Path(file) for file in files)
        if not self._files:
            raise ValueError("VtkFiles needs at least one file.")
        for file in self._files:
            if file.suffix not in VTK_SUFFIXES:
                raise ValueError(f"{file} is not one of {', '.join(VTK_SUFFIXES)}.")
            if not file.is_file():
                raise FileNotFoundError(f"{file} does not exist.")
        self._title = title if title is not None else self._files[0].name
        self._info = dict(info or {})
        self._history = None if history is None else Path(history)
        self._series = {file: _Series(file) for file in self._files if file.suffix == ".pvd"}
        self._static: dict[str, pv.DataSet] | None = None

    @property
    def title(self) -> str:
        """The heading -- see :meth:`ResultSource.title`."""
        return self._title

    @property
    def files(self) -> tuple[Path, ...]:
        """The files read."""
        return self._files

    def info(self) -> dict[str, str]:
        """The lines given, then the files read -- see :meth:`ResultSource.info`."""
        return self._info | {"Files": ", ".join(file.name for file in self._files)}

    def times(self) -> tuple[float, ...]:
        """Every step time of every series, merged -- see :meth:`ResultSource.times`."""
        merged = sorted({time for series in self._series.values() for time in series.times})
        return tuple(merged) or (0.0,)

    def frame(self, index: int) -> dict[str, pv.DataSet]:
        """The static files' datasets and each series' step at that time -- see :meth:`ResultSource.frame`."""
        times = self.times()
        if not 0 <= index < len(times):
            raise IndexError(f"snapshot {index} of {len(times)}.")
        if self._static is None:
            self._static = {}
            for file in self._files:
                if file not in self._series:
                    _add(self._static, file.stem, pv.read(file))
        frame = dict(self._static)
        for file, series in self._series.items():
            _add(frame, file.stem, series.at(times[index]))
        return frame

    def history(self) -> ConvergenceHistory | None:
        """The history file, read again on each call so a growing one is followed -- see :meth:`ResultSource.history`."""
        if self._history is None or not self._history.is_file():
            return None
        return ConvergenceHistory.read(self._history)


class RunDirectory(ResultSource):
    """The output directory of an ``aquaflux run``: its fields, its records and its history.

    The run record (``run.yaml``) lists what the run wrote, and that list is what is read -- the VTK
    files among it, and the comma-separated-values file as the history -- so a run whose outputs were
    given other names opens the same way. A directory with no run record (a run interrupted before it
    wrote one) falls back to every VTK file in it and a ``history.csv`` if there is one.

    Parameters
    ----------
    directory : path-like
        The output directory.

    Raises
    ------
    FileNotFoundError
        If it is not a directory, or holds no VTK file to show.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"{self.directory} is not a directory.")
        record = _read_yaml(self.directory / RUN_RECORD)
        written = [self.directory / name for name in record.get("written", [])]
        if written:
            history = next((path for path in written if path.suffix == ".csv"), None)
        else:
            written = sorted(self.directory.iterdir())
            history = self.directory / "history.csv"
        files = [path for path in written if path.suffix in VTK_SUFFIXES and path.is_file()]
        if not files:
            raise FileNotFoundError(
                f"{self.directory} holds no {', '.join(VTK_SUFFIXES)} file to show"
                + (
                    "; the run did not converge, so it wrote no fields."
                    if record.get("converged") is False
                    else "."
                )
            )
        self._files = VtkFiles(
            files,
            title=self._title(record),
            info=_describe(record, _read_yaml(self.directory / CASE_RECORD)),
            history=history,
        )

    def _title(self, record: Mapping[str, object]) -> str:
        case = record.get("case")
        return Path(str(case)).name if case else self.directory.name

    @property
    def title(self) -> str:
        """The case file's name -- see :meth:`ResultSource.title`."""
        return self._files.title

    def info(self) -> dict[str, str]:
        """The run and case records, summarized -- see :meth:`ResultSource.info`."""
        return self._files.info()

    def times(self) -> tuple[float, ...]:
        """See :meth:`ResultSource.times`."""
        return self._files.times()

    def frame(self, index: int) -> dict[str, pv.DataSet]:
        """See :meth:`ResultSource.frame`."""
        return self._files.frame(index)

    def history(self) -> ConvergenceHistory | None:
        """See :meth:`ResultSource.history`."""
        return self._files.history()


def open_source(path: str | os.PathLike[str]) -> ResultSource:
    """The source for a path: a run's output directory, or a VTK file.

    Parameters
    ----------
    path : path-like
        A directory written by ``aquaflux run``, or one ``.vtu``, ``.pvd``, ``.vtp`` or ``.vtm`` file.

    Returns
    -------
    ResultSource
        A :class:`RunDirectory` or a :class:`VtkFiles`.
    """
    path = Path(path)
    return RunDirectory(path) if path.is_dir() else VtkFiles([path])


class _Series:
    """One ``.pvd`` time series, read a step at a time."""

    def __init__(self, path: Path) -> None:
        self._reader = pv.get_reader(path)
        self.times = tuple(float(time) for time in self._reader.time_values)
        self._cache: tuple[float, pv.DataSet] | None = None

    def at(self, time: float) -> pv.DataSet:
        """The step at ``time``, or the last one before it; the first if ``time`` precedes them all."""
        step = max((t for t in self.times if t <= time), default=self.times[0])
        if self._cache is None or self._cache[0] != step:
            self._reader.set_active_time_value(step)
            self._cache = (step, self._reader.read())
        return self._cache[1]


def _add(frame: dict[str, pv.DataSet], stem: str, data: pv.DataSet) -> None:
    """Add a file's data to a frame: a multiblock as its leaves, each named by its block."""
    if not isinstance(data, pv.MultiBlock):
        frame[_unique(frame, stem)] = data
        return
    leaves = [
        (name, block)
        for name, block in multiblock_leaves(data)
        if block is not None and block.n_cells
    ]
    if len(leaves) == 1:
        frame[_unique(frame, stem)] = leaves[0][1]
        return
    for name, block in leaves:
        frame[_unique(frame, name or stem)] = block


def multiblock_leaves(
    blocks: pv.MultiBlock, prefix: str = ""
) -> list[tuple[str, pv.DataSet | None]]:
    """Every dataset in a (possibly nested) multiblock, by its block name, nested names joined by ``/``."""
    leaves = []
    for index, block in enumerate(blocks):
        name = blocks.get_block_name(index) or f"block{index}"
        name = f"{prefix}/{name}" if prefix else name
        if isinstance(block, pv.MultiBlock):
            leaves.extend(multiblock_leaves(block, name))
        else:
            leaves.append((name, block))
    return leaves


def _unique(frame: Mapping[str, object], name: str) -> str:
    """``name``, or ``name`` with the first free numeric suffix if a dataset already has it."""
    candidate, n = name, 2
    while candidate in frame:
        candidate, n = f"{name} ({n})", n + 1
    return candidate


def _read_yaml(path: Path) -> dict:
    """A record's mapping; empty if the file is absent."""
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


#: The run record's entries worth showing, with the label each is shown under.
_RUN_LINES = {
    "solver": "Solver",
    "converged": "Converged",
    "steps": "Steps",
    "residual": "Residual",
    "seconds": "Run time (s)",
    "started": "Started",
    "message": "Message",
}


def _describe(run: Mapping[str, object], case: Mapping[str, object]) -> dict[str, str]:
    """The lines worth showing from a run record and a case record, under their display labels."""
    lines: dict[str, str] = {}
    physics = case.get("physics")
    if isinstance(physics, Mapping) and "kind" in physics:
        lines["Physics"] = str(physics["kind"])
    mesh = case.get("mesh")
    if isinstance(mesh, Mapping):
        lines["Mesh"] = str(mesh.get("path", mesh.get("kind", "")))
    for key, label in _RUN_LINES.items():
        if run.get(key) is not None:
            lines[label] = str(run[key])
    version = run.get("aquaflux")
    if isinstance(version, Mapping):
        commit = str(version.get("commit") or "")[:10]
        lines["Version"] = " ".join(
            part
            for part in (
                str(version.get("version", "")),
                commit,
                "(modified)" if version.get("modified") else "",
            )
            if part
        )
    return lines
