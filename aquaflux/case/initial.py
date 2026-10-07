"""What a case starts from, when it does not start from scratch: the ``initial`` section.

A case with no ``initial`` section builds its own starting state (a potential flow, or for a
Reynolds-averaged (RANS) case a hybrid initial condition) and marches from it. With one it starts from a
state that already exists -- :class:`Checkpoint`, the checkpoints an earlier run of a case file wrote,
which is how a run that stopped short is resumed.

Reading a starting state is split from using it. :meth:`InitialState.read` finds the file and checks
it against the case's mesh and physics, needing neither geometry nor equations, so a state that does
not fit is refused before anything expensive is built; :func:`starting_seed` then maps the fields it
read onto the problem the case built.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from aquaflux.solve import find_checkpoint

from .restart_file import RestartHeader, read_restart

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh

    from .physics import Physics

__all__ = ["Checkpoint", "InitialState", "StartingFields", "starting_seed"]


@dataclasses.dataclass(frozen=True)
class StartingFields:
    """Physical fields read from a source and checked against the case, not yet mapped onto its problem.

    Attributes
    ----------
    fields : mapping of {str: np.ndarray}
        The fields by name -- ``U`` ``(n_cells, dim)``, ``p`` ``(n_cells,)``, and under RANS ``k`` and
        ``omega``.
    source : mapping of {str: object}
        What they were read from, for a run's record: the kind of source, the file, the residual the
        run that wrote it had reached, and the digest of the case that wrote it.
    """

    fields: Mapping[str, np.ndarray]
    source: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class InitialState(abc.ABC):
    """Where a case's starting state comes from: :class:`Checkpoint`.

    Attributes
    ----------
    path : str
        Where the state is, relative to the case file unless absolute. What it names depends on the
        kind.
    """

    path: str

    #: Which fields name a file or directory, relative to the case file (read by the case's path rules).
    path_fields: ClassVar[tuple[str, ...]] = ("path",)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError(f"{type(self).__name__}.path names where the starting state is.")

    def location(self, case_directory: Path) -> Path:
        """Where this state is read from, resolved from ``case_directory``.

        Parameters
        ----------
        case_directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        pathlib.Path
            An absolute path, with symbolic links resolved.
        """
        return (Path(case_directory) / self.path).resolve()

    @abc.abstractmethod
    def read(self, case_directory: Path, physics: Physics, mesh: Mesh) -> StartingFields:
        """Read the state and check that it belongs to this case.

        Parameters
        ----------
        case_directory : pathlib.Path
            The directory the case file sits in.
        physics : Physics
            The case's physics.
        mesh : Mesh
            The case's mesh, as read: its topology and nodes are compared, so no geometry is needed.

        Returns
        -------
        StartingFields

        Raises
        ------
        FileNotFoundError
            If there is nothing at the path.
        ValueError
            If what is there cannot start this case: it has the wrong physics, cell count or mesh, or
            is not a case checkpoint at all, or holds values that are not finite.
        """


@dataclasses.dataclass(frozen=True)
class Checkpoint(InitialState):
    """Start from a checkpoint an earlier run of a case file wrote, to resume a run that stopped short.

    The file holds the physical fields (``U``, ``p``, and under RANS ``k`` and ``omega``), so a case
    may start from one whatever variables it solves in. It must be on the same mesh and of the same
    physics; the viscosity, the boundary values and the solver settings may all differ.

    A viscosity ramp cannot start from a state -- it opens on a seed fitted to its own anchor
    station -- so a case with one refuses this section. Drop the ramp to resume at the case's own
    viscosity.

    Attributes
    ----------
    path : str
        The ``checkpoints`` directory of the earlier run, relative to the case file unless absolute.
        The run writing this case must write elsewhere: it replaces what it finds in its own output
        directory.
    step : int or "latest"
        Which checkpoint: the highest step present (``latest``), or a step by number, which must still
        be among those the earlier run kept.

    Raises
    ------
    ValueError
        If ``path`` is empty or ``step`` is neither ``latest`` nor at least 1.
    """

    step: int | Literal["latest"] = "latest"

    def __post_init__(self) -> None:
        super().__post_init__()
        # `True` is an int in Python, and a step of "true" names nothing.
        is_step = isinstance(self.step, int) and not isinstance(self.step, bool) and self.step >= 1
        if self.step != "latest" and not is_step:
            raise ValueError(f"Checkpoint.step is 'latest' or a step >= 1, got {self.step!r}.")

    def read(self, case_directory: Path, physics: Physics, mesh: Mesh) -> StartingFields:
        """The checkpoint's fields, checked against the case -- see :meth:`InitialState.read`."""
        file = find_checkpoint(self.location(case_directory), self.step)
        restart = read_restart(file)
        restart.header.refuse_unless_fits(RestartHeader.of(physics, mesh, ""), file)
        restart.refuse_if_not_finite()
        return StartingFields(
            fields=restart.fields,
            source={
                "kind": type(self).__name__,
                "file": str(file),
                "residual": restart.residual,
                "case_digest": restart.header.case_digest,
            },
        )


def starting_seed(
    starting: StartingFields | None, physics: Physics, problem: object
) -> object | None:
    """The starting state a solve takes, from fields read for the case; ``None`` to start from scratch.

    Parameters
    ----------
    starting : StartingFields or None
        What the case's ``initial`` section read, or ``None`` when it has none.
    physics : Physics
        The case's physics.
    problem : object
        What the case built.

    Returns
    -------
    object or None
        The flow state for a laminar case, ``(flow, k, omega)`` for a Reynolds-averaged one.

    Raises
    ------
    ValueError
        If the fields lack one the physics needs.
    """
    return None if starting is None else physics.initial_fields(problem, starting.fields)
