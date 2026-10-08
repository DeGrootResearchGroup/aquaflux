"""Running a case file: read, check, build, solve, and write what its ``outputs`` section asks for.

:func:`prepare_run` does the cheap part -- it reads the file, refuses to replace the results of an
earlier run, and checks the case against its mesh -- so a mistake in the file is reported before
anything expensive starts. :meth:`PreparedRun.run` then builds and solves the case, logging each step
of the march as it goes, and writes:

* the converged fields, by each of the section's field writers;
* the log, one row per outer step, also echoed to the terminal;
* the history, the same steps as a comma-separated-values file for a program to read;
* the checkpoints, when asked for;
* ``case.yaml``, the case as it ran -- the solver written out even when the file left it to the
  default -- and ``run.yaml``, a record of the run: the aquaflux version and commit, when it ran,
  how many steps it took, where its residual ended and whether it converged, and the physics' scalar
  results (a radiation case's lamp power and where it goes).

A solve that stops short of its stopping test -- or is interrupted -- writes no fields, since what it
holds is not a solution, but it still writes its log, its checkpoints and ``run.yaml``.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

import equinox as eqx
import yaml

import aquaflux
from aquaflux.solve import (
    MarchLogger,
    RefreshTiming,
    StateCheckpointer,
    StepHistory,
    StepReport,
    combine_observers,
)

from .case_file import CaseFile, CheckedCase, read_case, write_case
from .outputs import RunFields
from .paths import relocated
from .solver import NotConverged, SolverSpec, solver_for

__all__ = ["PreparedRun", "RunPlan", "RunRecord", "plan_run", "prepare_run"]

#: The files a run writes into its output directory besides its fields and its log.
_CASE_RECORD, _RUN_RECORD, _CHECKPOINTS = "case.yaml", "run.yaml", "checkpoints"


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """What a run did.

    Attributes
    ----------
    directory : pathlib.Path
        Where it wrote.
    converged : bool
        Whether the solve reached its stopping test.
    steps : int or None
        The outer steps the march took; ``None`` for a solve with no steps to count (the segregated
        loop).
    residual : float or None
        The march's last residual, in the measure it was steered by; ``None`` as for ``steps``.
    written : tuple of pathlib.Path
        Every file or directory written, fields first.
    message : str or None
        Why the solve stopped short, when it did.
    results : dict
        The physics' scalar results, as recorded in ``run.yaml``; empty when it has none or the solve
        stopped short.
    """

    directory: Path
    converged: bool
    steps: int | None
    residual: float | None
    written: tuple[Path, ...]
    message: str | None = None
    results: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PreparedRun:
    """A case that has been read and checked, with somewhere to write that holds no earlier results.

    Attributes
    ----------
    source : pathlib.Path
        The case file.
    checked : CheckedCase
        The case with its validated mesh.
    solver : SolverSpec
        The solver it runs -- the file's, or its physics' default.
    directory : pathlib.Path
        The output directory.
    """

    source: Path
    checked: CheckedCase
    solver: SolverSpec
    directory: Path

    def run(self, terminal: IO[str] | None = None) -> RunRecord:
        """Build and solve the case, and write its outputs.

        Parameters
        ----------
        terminal : text stream, optional
            Where the log is echoed as it is written; ``sys.stdout`` unset.

        Returns
        -------
        RunRecord
            What the run did. A solve that stops short is a record with ``converged`` false, not an
            error.
        """
        spec = self.checked.spec
        outputs = spec.outputs
        terminal = sys.stdout if terminal is None else terminal
        self.directory.mkdir(parents=True, exist_ok=True)
        started = datetime.datetime.now(datetime.UTC)
        clock = time.perf_counter()
        log = None if outputs.log is None else (self.directory / outputs.log).open("w")
        history = None if outputs.history is None else StepHistory(self.directory / outputs.history)
        try:
            stream = _Tee([terminal, *([] if log is None else [log])])
            stream.write(f"aquaflux {aquaflux.__version__}: {self.source}\n")
            stream.write(f"  solver: {type(self.solver).__name__}; writing to {self.directory}\n")
            problem = self.checked.build()
            logger = MarchLogger(stream, fields=spec.physics.progress_fields(problem))
            checkpointer = (
                None
                if outputs.checkpoints is None
                else StateCheckpointer(
                    self.directory / _CHECKPOINTS,
                    every=outputs.checkpoints.every,
                    keep=outputs.checkpoints.keep,
                )
            )
            steps = _StepCount([recorder for recorder in (history, checkpointer) if recorder])
            observers = self.solver.observers_for(logger, steps)
            converged, message, written, results = True, None, [], {}
            try:
                solution = self.solver.solve(problem, **observers)
            except (NotConverged, eqx.EquinoxRuntimeError) as error:
                converged, message = False, str(error).strip().splitlines()[0]
                logger.note(f"did not converge: {message}")
            except KeyboardInterrupt:
                # Stopped by hand (Ctrl-C, or the browser interface's Stop): a run that stopped
                # short, recorded as one, rather than one that leaves no record of how it ended.
                converged, message = False, "interrupted before it converged"
                logger.note(f"did not converge: {message}")
            if converged:
                fields = RunFields(
                    cells=spec.physics.output_fields(problem, solution),
                    patches=spec.physics.output_patch_fields(problem, solution),
                )
                results = spec.physics.results(problem, solution)
                for writer in outputs.fields:
                    written.append(
                        writer.write(
                            self.directory, self.checked_directory, self.checked.mesh, fields
                        )
                    )
            if log is not None:
                written.append(self.directory / outputs.log)
            if history is not None:
                written.append(history.path)
            if outputs.checkpoints is not None and steps.count:
                written.append(self.directory / _CHECKPOINTS)
            written.append(self._write_case_record())
            record = RunRecord(
                directory=self.directory,
                converged=converged,
                steps=steps.count if steps.count else None,
                residual=steps.residual,
                written=tuple(written),
                message=message,
                results=results,
            )
            written_record = self._write_run_record(record, started, time.perf_counter() - clock)
            record = dataclasses.replace(record, written=(*record.written, written_record))
            logger.note(
                f"{'converged' if converged else 'NOT converged'}; wrote "
                + ", ".join(str(path) for path in record.written)
            )
            return record
        finally:
            if log is not None:
                log.close()
            if history is not None:
                history.close()

    @property
    def checked_directory(self) -> Path:
        """The directory the case file sits in, which its relative paths are taken from."""
        return self.source.parent

    def _write_case_record(self) -> Path:
        """``case.yaml``: the case as it ran, every file it names re-based on the output directory.

        The solver is written out whether the file stated it or left it to the default, so the record
        says what ran rather than what was omitted.
        """
        here = self.directory
        spec = relocated(self.checked.spec, self.checked_directory, here)
        recorded = dataclasses.replace(
            spec,
            solver=self.solver,
            outputs=dataclasses.replace(spec.outputs, directory="."),
        )
        path = here / _CASE_RECORD
        write_case(recorded, path)
        return path

    def _write_run_record(
        self, record: RunRecord, started: datetime.datetime, seconds: float
    ) -> Path:
        """``run.yaml``: what ran, when, and how it ended."""
        commit, modified = _checkout_state()
        document = {
            "case": str(self.source),
            "aquaflux": {"version": aquaflux.__version__, "commit": commit, "modified": modified},
            "started": started.isoformat(timespec="seconds"),
            "seconds": round(seconds, 1),
            "solver": type(self.solver).__name__,
            "converged": record.converged,
            "steps": record.steps,
            "residual": record.residual,
            "message": record.message,
            "written": [os.path.relpath(path, self.directory) for path in record.written],
        }
        if record.results:
            document["results"] = record.results
        path = self.directory / _RUN_RECORD
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, sort_keys=False, default_flow_style=False)
        return path


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """Where a run of a case would write, and what earlier results that would replace.

    Attributes
    ----------
    directory : pathlib.Path
        The output directory.
    log, history : pathlib.Path or None
        The log and the per-step history the run would write there; ``None`` for one the outputs
        section turns off.
    occupied : tuple of pathlib.Path
        What already holds results: the output directory, when it holds anything, and any field
        writer's target outside it that exists. Empty when the run would replace nothing.
    """

    directory: Path
    log: Path | None
    history: Path | None
    occupied: tuple[Path, ...]


def plan_run(path: str | Path) -> RunPlan:
    """Say where a run of the case file at ``path`` would write, without checking its mesh.

    Parameters
    ----------
    path : str or path-like
        The case file.

    Returns
    -------
    RunPlan
        What :func:`prepare_run` would refuse to replace unless told to overwrite.

    Raises
    ------
    ValueError, TypeError, FileNotFoundError
        If the file cannot be read or is refused (see :func:`~aquaflux.case.read_case`).
    """
    case_file = read_case(Path(path).resolve())
    return _plan(case_file)


def prepare_run(path: str | Path, *, overwrite: bool = False) -> PreparedRun:
    """Read a case file and check it, refusing to replace an earlier run's results.

    Parameters
    ----------
    path : str or path-like
        The case file.
    overwrite : bool
        Replace an earlier run's results rather than refuse: the files this run writes are replaced
        and its checkpoints are cleared first. Anything else in the directory is left alone.

    Returns
    -------
    PreparedRun
        Ready to :meth:`~PreparedRun.run`.

    Raises
    ------
    FileExistsError
        If the output directory holds anything, or a field writer's target exists, and ``overwrite``
        is false.
    ValueError, TypeError
        If the file is refused (see :func:`~aquaflux.case.read_case`), its case states no solver and
        its physics' default cannot solve it, or the case does not fit its mesh.
    FileNotFoundError
        If the file or its mesh cannot be found.
    """
    source = Path(path).resolve()
    case_file = read_case(source)
    plan = _plan(case_file)
    if plan.occupied and not overwrite:
        raise FileExistsError(
            f"{', '.join(map(str, plan.occupied))} already "
            f"{'holds' if len(plan.occupied) == 1 else 'hold'} results; move them, or replace them "
            "with --overwrite (overwrite=True)."
        )
    if overwrite and (plan.directory / _CHECKPOINTS).is_dir():
        shutil.rmtree(plan.directory / _CHECKPOINTS)
    solver = solver_for(case_file.spec)
    return PreparedRun(
        source=source, checked=case_file.check(), solver=solver, directory=plan.directory
    )


def _plan(case_file: CaseFile) -> RunPlan:
    """Where ``case_file``'s run writes, and which of its targets already hold results."""
    spec = case_file.spec
    outputs = spec.outputs
    directory = outputs.output_directory(case_file.directory).resolve()
    occupied = [directory] if directory.is_dir() and any(directory.iterdir()) else []
    occupied += [
        target
        for writer in outputs.fields
        for target in writer.targets(directory, case_file.directory)
        if target.exists() and not target.is_relative_to(directory)
    ]
    return RunPlan(
        directory=directory,
        log=None if outputs.log is None else directory / outputs.log,
        history=None if outputs.history is None else directory / outputs.history,
        occupied=tuple(occupied),
    )


class _Tee:
    """A text stream writing to several, flushing each write, so a log file and a terminal agree line for line."""

    def __init__(self, streams: Sequence[IO[str]]) -> None:
        self._streams = list(streams)

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class _StepCount:
    """Counts the march's steps and keeps its last residual, forwarding each march hook to its recorders.

    A recorder is anything with ``on_checkpoint(report, state)`` -- the history, the checkpoints -- and
    optionally ``on_retry``, ``on_refresh`` and ``on_residuals``, which reach the recorders that have
    them.
    """

    def __init__(self, recorders: Sequence[StepHistory | StateCheckpointer]) -> None:
        self._recorders = tuple(recorders)
        self.count = 0
        self.residual: float | None = None

    def on_checkpoint(self, report: StepReport, state: object) -> None:
        self.count += 1
        self.residual = float(report.residual_norm)
        for recorder in self._recorders:
            recorder.on_checkpoint(report, state)

    def on_retry(self, reason: str, attempt: int, beta: float) -> None:
        for hook in self._hooks("on_retry"):
            hook(reason, attempt, beta)

    def on_refresh(self, timing: RefreshTiming) -> None:
        for hook in self._hooks("on_refresh"):
            hook(timing)

    @property
    def on_residuals(self) -> Callable[[Mapping[str, float]], None] | None:
        """The per-equation hook, or ``None`` when no recorder keeps them, so the march skips their cost."""
        hooks = self._hooks("on_residuals")
        if not hooks:
            return None
        return combine_observers(*hooks)

    def _hooks(self, name: str) -> list[Callable]:
        return [getattr(recorder, name) for recorder in self._recorders if hasattr(recorder, name)]


def _checkout_state() -> tuple[str | None, bool | None]:
    """The commit aquaflux was run from and whether its tracked files were modified, if it is a git checkout."""
    root = Path(aquaflux.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())
