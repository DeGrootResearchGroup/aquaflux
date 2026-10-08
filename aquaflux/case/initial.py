"""What a case starts from, when it does not start from scratch: the ``initial`` section.

A case with no ``initial`` section builds its own starting state (a potential flow, or for a
Reynolds-averaged (RANS) case a hybrid initial condition) and marches from it. With one it starts from a
state that already exists, of one of two kinds. :class:`Checkpoint` is the checkpoints an earlier run of
a case file wrote, which is how a run that stopped short is resumed: the physical fields, and what the
stopped march began at, so the march goes on rather than starts again. :class:`Fields` is one time
directory of an OpenFOAM case -- another program's converged solution, or an earlier run's
:class:`~aquaflux.case.OpenFOAMTime` output -- which carries no march to continue.

Reading a starting state is split from using it. :meth:`InitialState.read` finds the file and checks
it against the case's mesh and physics, needing neither geometry nor equations, so a state that does
not fit is refused before anything expensive is built; :func:`starting_arguments` then maps what it
read onto the problem the case built, as the keywords a solve takes.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from aquaflux.io import infer_extruded_axis, read_openfoam_time
from aquaflux.solve import find_checkpoint

from .kinematic import pressure_from_kinematic
from .mesh_source import OpenFOAMMesh
from .restart_file import RestartHeader, read_restart, refuse_non_finite

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh

    from .physics import Physics
    from .spec import CaseSpec

__all__ = ["Checkpoint", "Fields", "InitialState", "StartingFields", "starting_arguments"]


@dataclasses.dataclass(frozen=True)
class StartingFields:
    """Physical fields read from a source and checked against the case, not yet mapped onto its problem.

    Attributes
    ----------
    fields : mapping of {str: np.ndarray}
        The fields by name -- ``U`` ``(n_cells, dim)``, ``p`` ``(n_cells,)``, and under RANS ``k`` and
        ``omega``.
    source : mapping of {str: object}
        What they were read from, for a run's record: the kind of source and the file, and what that
        kind knows of it -- a checkpoint's residual, the digest of the case that wrote it and the
        reference residual carried to the solve, an OpenFOAM time's density.
    reference_residual : float or None
        The residual norm the run that wrote the fields took at its own first state, when the case
        starting from them states the same problem and judges a residual the same way, so that a march
        resumed from them continues the stopped march's damping and stops against the same bar. ``None``
        when the source has no such history, or the case differs from the one that wrote it.
    """

    fields: Mapping[str, np.ndarray]
    source: Mapping[str, object]
    reference_residual: float | None = None


@dataclasses.dataclass(frozen=True)
class InitialState(abc.ABC):
    """Where a case's starting state comes from: :class:`Checkpoint` or :class:`Fields`.

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
    def read(self, case_directory: Path, spec: CaseSpec, mesh: Mesh) -> StartingFields:
        """Read the state and check that it belongs to this case.

        Parameters
        ----------
        case_directory : pathlib.Path
            The directory the case file sits in.
        spec : CaseSpec
            The case that is to start from it.
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

    A march that is resumed continues the stopped one when the case states the same problem and judges
    a residual in the same measure: the file records the residual the stopped march began at, and the
    resumed march measures its damping and its stopping bar against that one instead of against the
    residual at the state it is handed (which would restart its damping at the opening strength). If
    the problem or the measure differs, the fields still start the case but the march is a new one. A
    march that had refreshed its preconditioner before it stopped re-based its damping at the refresh,
    which the file does not record, so its resumed march continues from the residual it began at.

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

    def read(self, case_directory: Path, spec: CaseSpec, mesh: Mesh) -> StartingFields:
        """The checkpoint's fields, checked against the case -- see :meth:`InitialState.read`.

        The stopped march's reference residual is carried only when the case states the same problem as
        the one that wrote the file, and judges a residual in the same measure: it is a scale for that
        problem and means nothing for another.
        """
        file = find_checkpoint(self.location(case_directory), self.step)
        restart = read_restart(file)
        expected = RestartHeader.of(spec, mesh)
        restart.header.refuse_unless_fits(expected, file)
        restart.refuse_if_not_finite()
        same_problem = restart.header.problem_digest == expected.problem_digest
        carried = restart.reference_residual if same_problem else None
        return StartingFields(
            fields=restart.fields,
            source={
                "kind": type(self).__name__,
                "file": str(file),
                "residual": restart.residual,
                "case_digest": restart.header.case_digest,
                "reference_residual": carried,
            },
            reference_residual=carried,
        )


@dataclasses.dataclass(frozen=True)
class Fields(InitialState):
    """Start from the fields of one time directory of an OpenFOAM case.

    The fields of another program's solution -- a converged OpenFOAM run -- or of an earlier run's
    :class:`~aquaflux.case.OpenFOAMTime` output, read as that case's own files. Each field is a file
    named for it in the time directory: ``U`` and ``p``, and under RANS ``k`` and ``omega``. The
    pressure is read as OpenFOAM's incompressible solvers hold it, per unit density, and multiplied by
    the case's fluid density; a case of density one reads it unchanged.

    The case's mesh must be the OpenFOAM mesh the fields were written on, which is what numbers the
    cells. Only the cell count can be checked against it: fields of another mesh with as many cells
    would be read without complaint. A march resumed from a time directory has no history to carry, so
    it begins as a new one from the state.

    Attributes
    ----------
    path : str
        The OpenFOAM case directory holding the time directory, relative to the case file unless
        absolute.
    time : str
        The time directory's name, used verbatim. Quote it in a file (``time: "1000"``), since a bare
        number reads as a number.

    Raises
    ------
    ValueError
        If ``path`` or ``time`` is empty.
    """

    time: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.time:
            raise ValueError("Fields.time names the time directory to start from.")

    def read(self, case_directory: Path, spec: CaseSpec, mesh: Mesh) -> StartingFields:
        """The time directory's fields, in this case's units -- see :meth:`InitialState.read`.

        Raises
        ------
        FileNotFoundError
            If the time directory, or a field the physics needs, is missing.
        ValueError
            If the case's mesh is not an OpenFOAM one, a field has the wrong number of values, or a
            value is not finite.
        """
        if not isinstance(spec.mesh, OpenFOAMMesh):
            raise ValueError(
                "initial: a Fields state is the cells of an OpenFOAM case in its own numbering, so "
                f"the case's mesh must be that case's OpenFOAMMesh, not a {type(spec.mesh).__name__}."
            )
        directory = self.location(case_directory)
        axis = (
            infer_extruded_axis(Path(case_directory) / spec.mesh.path, mesh)
            if mesh.dim == 2
            else None
        )
        read = read_openfoam_time(
            directory, self.time, spec.physics.state_fields, mesh, extruded_axis=axis
        )
        density = float(spec.fluid.density)
        fields = {**read, "p": pressure_from_kinematic(read["p"], density)}
        where = directory / self.time
        refuse_non_finite(fields, f"{where} holds values that are not finite in {{bad}}.")
        return StartingFields(
            fields=fields,
            source={"kind": type(self).__name__, "file": str(where), "density": density},
        )


def starting_arguments(
    starting: StartingFields | None, physics: Physics, problem: object
) -> dict[str, object]:
    """The keywords a solve takes to start from what the case's ``initial`` section read.

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
    dict
        ``{"initial": <the flow state, or (flow, k, omega) for a Reynolds-averaged case>,
        "reference_residual": <float or None>}``; both ``None`` to start from scratch.

    Raises
    ------
    ValueError
        If the fields lack one the physics needs.
    """
    if starting is None:
        return {"initial": None, "reference_residual": None}
    return {
        "initial": physics.initial_fields(problem, starting.fields),
        "reference_residual": starting.reference_residual,
    }
