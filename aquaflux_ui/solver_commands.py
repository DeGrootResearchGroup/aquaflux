"""The solver's case-file commands, asked of a separate process: schema, show, write, mesh and check.

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

__all__ = ["CaseDocument", "MeshExport", "Runner", "SolverCommands"]

#: Runs ``aquaflux <arguments>`` with the given standard input; the seam a test replaces.
Runner = Callable[[Sequence[str], str | None], CommandResult]


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
    """A case's mesh, as the solver read or generated it and wrote it for viewing.

    Attributes
    ----------
    error : str or None
        Why it could not be read or generated, if it could not; the other attributes are then unset.
    directory : pathlib.Path or None
        Where ``mesh.vtu`` (and ``patches.vtm``, when the solver writes the boundary) were written.
    cells, dim : int
        Its cell count and spatial dimension.
    patches : list of dict
        Its boundary patches holding faces: ``{"name", "faces"}`` each, in the mesh's order.
    groups : dict
        Its patch groups, each with the patches in it.
    """

    error: str | None
    directory: Path | None = None
    cells: int = 0
    dim: int = 3
    patches: list = dataclasses.field(default_factory=list)
    groups: dict = dataclasses.field(default_factory=dict)


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
        self._schema: CaseSchema | None = None

    def schema(self) -> CaseSchema:
        """The case-file schema, asked for once and kept.

        Raises
        ------
        RuntimeError
            If the command fails -- the solver is not installed beside the page, say.
        """
        if self._schema is None:
            result = self._run(["schema"], None)
            if result.status != 0:
                raise RuntimeError(f"`aquaflux schema` failed: {_reason(result)}")
            self._schema = CaseSchema(json.loads(result.output))
        return self._schema

    def show(self, path: str | Path) -> CaseDocument:
        """A case file as the solver reads it."""
        result = self._run(["show", str(path)], None)
        reply = _reply(result)
        return CaseDocument(Path(path), reply.get("case"), reply.get("error"))

    def write(
        self, path: str | Path, case: Mapping, relative_to: str | Path | None = None
    ) -> str | None:
        """Write ``case`` to ``path`` through the solver's own writer.

        Parameters
        ----------
        path : path-like
            The file to write.
        case : mapping
            The case file's content.
        relative_to : path-like, optional
            The directory the case's relative paths mean now -- where it was opened from. They are
            re-based onto ``path``'s directory, so a copy saved elsewhere still finds its mesh.

        Returns
        -------
        str or None
            Why it was refused, or ``None`` once it is written. A refused case writes nothing.
        """
        arguments = ["write", str(path)]
        if relative_to is not None:
            arguments += ["--relative-to", str(relative_to)]
        return _reply(self._run(arguments, json.dumps({"case": case}))).get("error")

    def mesh(self, section: Mapping, relative_to: str | Path, directory: str | Path) -> MeshExport:
        """Read or generate a case's mesh from its ``mesh`` section, written into ``directory``.

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
        arguments = ["mesh", str(directory), "--relative-to", str(relative_to)]
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
