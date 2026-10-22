"""The solver's file commands, asked of a separate process: schema, show, write, mesh, drawing, check,
plan.

This package does not import the solver -- that would bring JAX into the page's process -- so
everything it needs to know about a case file it asks the ``aquaflux`` command, which prints JSON
(JavaScript Object Notation) for exactly this. By default the commands go to one long-lived solver
process (:class:`~aquaflux_ui.solver_worker.SolverWorker`), run with the same Python interpreter as
the page, so it is the solver installed beside it, whichever environment that is.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .case_form import CaseSchema
from .solver_worker import CommandResult, SolverWorker

__all__ = [
    "CASE_FILE",
    "MESH_FILE",
    "CaseDocument",
    "DocumentKind",
    "DrawingView",
    "MeshExport",
    "RunPlan",
    "Runner",
    "SolverCommands",
]

#: Runs ``aquaflux <arguments>`` with the given standard input; the seam a test replaces.
Runner = Callable[[Sequence[str], str | None], CommandResult]


@dataclasses.dataclass(frozen=True)
class DocumentKind:
    """A kind of file the solver's editing commands read and write.

    Attributes
    ----------
    name : str
        What the page calls it: ``case file``, ``mesh file``.
    flags : tuple of str
        What the ``schema``, ``show`` and ``write`` commands are told to read it: none for a case
        file.
    """

    name: str
    flags: tuple[str, ...] = ()


#: A case file: the settings of a solve.
CASE_FILE = DocumentKind("case file")

#: A mesh file: how to mesh a region of a drawing, which ``aquaflux mesh`` makes.
MESH_FILE = DocumentKind("mesh file", ("--mesh-file",))


@dataclasses.dataclass(frozen=True)
class CaseDocument:
    """A case file as the solver read it.

    Attributes
    ----------
    path : pathlib.Path
        The file.
    case : dict or None
        Its content, or ``None`` if it could not be read at all.
    error : str or None
        Why the solver refuses it, if it does; a file it refuses can still be shown and corrected.
    """

    path: Path
    case: dict | None
    error: str | None


@dataclasses.dataclass(frozen=True)
class MeshExport:
    """A case's mesh, as the solver read it and wrote it for viewing.

    Attributes
    ----------
    error : str or None
        Why it could not be read, if it could not; the other attributes are then unset.
    directory : pathlib.Path or None
        Where ``mesh.vtu`` (and ``patches.vtm``, when the solver writes the boundary) were written.
    cells, dim : int
        Its cell count and spatial dimension.
    patches : list of dict
        Its boundary patches holding faces: ``{"name", "faces"}`` each, in the mesh's order.
    groups : dict
        Its patch groups, each with the patches in it.
    made : dict or None
        How the mesh was made, as its file records it -- for a mesh ``aquaflux mesh`` made, its
        settings, when, how long, and its ``quality``; ``None`` for a file that records nothing.
    """

    error: str | None
    directory: Path | None = None
    cells: int = 0
    dim: int = 3
    patches: list = dataclasses.field(default_factory=list)
    groups: dict = dataclasses.field(default_factory=dict)
    made: dict | None = None


@dataclasses.dataclass(frozen=True)
class DrawingView:
    """A mesh file's drawing as the solver read it, written for viewing, and the patches of its faces.

    Attributes
    ----------
    error : str or None
        Why the drawing could not be read, if it could not; the other attributes are then unset.
    surface : pathlib.Path or None
        The triangles written, each carrying ``face``, the face of the drawing it lies on.
    solids : list of str
        The drawing's solids.
    faces : list of dict
        Per face: its ``kind`` (``plane``, ``cylinder``, ...), ``area`` and a ``point`` on it.
    patches : list of str or None
        Each face's patch by the file's selections; ``None`` when they are refused.
    landed : dict
        For each patch chosen by points, the face each of its points lands on, in order.
    pipes : list of int
        The faces that are pipes.
    selections_error : str or None
        Why the selections are refused, if they are: the drawing is still shown.
    """

    error: str | None
    surface: Path | None = None
    solids: list = dataclasses.field(default_factory=list)
    faces: list = dataclasses.field(default_factory=list)
    patches: list | None = None
    landed: dict = dataclasses.field(default_factory=dict)
    pipes: list = dataclasses.field(default_factory=list)
    selections_error: str | None = None


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """Where a run of a case file would write, as the solver works it out.

    Attributes
    ----------
    error : str or None
        Why the file could not be read, if it could not; the other attributes are then unset.
    directory : pathlib.Path or None
        The output directory.
    log, history : pathlib.Path or None
        The log and the per-step history the run writes there; ``None`` for one the case turns off.
    occupied : tuple of pathlib.Path
        What already holds results, which the run replaces only when told to overwrite.
    """

    error: str | None
    directory: Path | None = None
    log: Path | None = None
    history: Path | None = None
    occupied: tuple[Path, ...] = ()


class SolverCommands:
    """The case-file commands of the installed solver.

    Parameters
    ----------
    runner : callable, optional
        ``(arguments, stdin) -> CommandResult``; unset, a new
        :class:`~aquaflux_ui.solver_worker.SolverWorker`.
    """

    def __init__(self, runner: Runner | None = None) -> None:
        self._run = runner if runner is not None else SolverWorker()
        self._schemas: dict[DocumentKind, CaseSchema] = {}

    def schema(self, kind: DocumentKind = CASE_FILE) -> CaseSchema:
        """The schema of a kind of file, asked for once and kept.

        Raises
        ------
        RuntimeError
            If the command fails -- the solver is not installed beside the page, say.
        """
        if kind not in self._schemas:
            result = self._run(["schema", *kind.flags], None)
            if result.status != 0:
                raise RuntimeError(f"`aquaflux schema` failed: {_reason(result)}")
            self._schemas[kind] = CaseSchema(json.loads(result.output))
        return self._schemas[kind]

    def show(self, path: str | Path, kind: DocumentKind = CASE_FILE) -> CaseDocument:
        """A file as the solver reads it."""
        result = self._run(["show", str(path), *kind.flags], None)
        reply = _reply(result)
        return CaseDocument(Path(path), reply.get("case"), reply.get("error"))

    def write(
        self,
        path: str | Path,
        case: Mapping,
        relative_to: str | Path | None = None,
        kind: DocumentKind = CASE_FILE,
    ) -> str | None:
        """Write ``case`` to ``path`` through the solver's own writer.

        Parameters
        ----------
        path : path-like
            The file to write.
        case : mapping
            The file's content.
        relative_to : path-like, optional
            The directory the file's relative paths mean now -- where it was opened from. They are
            re-based onto ``path``'s directory, so a copy saved elsewhere still finds its mesh.
        kind : DocumentKind, optional
            The kind of file; unset, a case file.

        Returns
        -------
        str or None
            Why it was refused, or ``None`` once it is written. A refused file writes nothing.
        """
        arguments = ["write", str(path), *kind.flags]
        if relative_to is not None:
            arguments += ["--relative-to", str(relative_to)]
        return _reply(self._run(arguments, json.dumps({"case": case}))).get("error")

    def drawing(
        self, mesh_file: Mapping, relative_to: str | Path, directory: str | Path
    ) -> DrawingView:
        """Read a mesh file's drawing, as the file stands, written into ``directory`` for viewing.

        Parameters
        ----------
        mesh_file : mapping
            The mesh file's content, saved or not: its drawing and region must be set.
        relative_to : path-like
            The directory its relative paths are relative to: the mesh file's.
        directory : path-like
            Where the solver writes the drawing's triangles.

        Returns
        -------
        DrawingView
        """
        arguments = ["drawing", str(directory), "--relative-to", str(relative_to)]
        reply = _reply(self._run(arguments, json.dumps({"mesh_file": mesh_file})))
        if reply.get("error"):
            return DrawingView(reply["error"])
        return DrawingView(
            None,
            Path(reply["surface"]),
            reply.get("solids", []),
            reply.get("faces", []),
            reply.get("patches"),
            reply.get("landed", {}),
            reply.get("pipes", []),
            reply.get("selections_error"),
        )

    def mesh(self, section: Mapping, relative_to: str | Path, directory: str | Path) -> MeshExport:
        """Read a case's mesh from its ``mesh`` section, written into ``directory`` for viewing.

        Parameters
        ----------
        section : mapping
            The case file's ``mesh`` section, as it stands in the page -- saved or not.
        relative_to : path-like
            The directory its relative paths are relative to: the case file's.
        directory : path-like
            Where the solver writes the mesh for viewing.

        Returns
        -------
        MeshExport
        """
        arguments = ["export-mesh", str(directory), "--relative-to", str(relative_to)]
        reply = _reply(self._run(arguments, json.dumps({"mesh": section})))
        if reply.get("error"):
            return MeshExport(reply["error"])
        return MeshExport(
            None,
            Path(directory),
            reply.get("cells", 0),
            reply.get("dim", 3),
            reply.get("patches", []),
            reply.get("groups", {}),
            reply.get("made"),
        )

    def check(self, path: str | Path) -> tuple[bool, str]:
        """Check a case file against its mesh.

        Returns
        -------
        tuple of (bool, str)
            Whether it passed, and what the command said.
        """
        result = self._run(["check", str(path)], None)
        message = (result.output if result.status == 0 else result.errors or result.output).strip()
        return result.status == 0, message

    def plan(self, path: str | Path) -> RunPlan:
        """Where a run of the case file at ``path`` would write, and what it would replace."""
        reply = _reply(self._run(["plan", str(path)], None))
        if reply.get("error"):
            return RunPlan(reply["error"])
        return RunPlan(
            None,
            Path(reply["directory"]),
            _path(reply.get("log")),
            _path(reply.get("history")),
            tuple(Path(target) for target in reply.get("occupied", [])),
        )


def _path(text: str | None) -> Path | None:
    return None if text is None else Path(text)


def _reply(result: CommandResult) -> dict:
    """A JSON command's reply, or an error reply if it printed none."""
    try:
        reply = json.loads(result.output)
    except json.JSONDecodeError:
        return {"error": _reason(result)}
    return reply if isinstance(reply, dict) else {"error": _reason(result)}


def _reason(result: CommandResult) -> str:
    """The last line the command printed about a failure."""
    text = (result.errors or result.output).strip()
    return text.splitlines()[-1] if text else f"it exited with status {result.status}"
