"""What a run of a case writes, and where: the fields at the converged root, the log and the checkpoints.

A case file's ``outputs`` section names a directory, relative to the case file, and what goes in it:

* ``fields`` -- a list of writers, each writing the converged fields in one format:
  :class:`Vtk` (a VTK unstructured-grid file, for any mesh) or :class:`OpenFOAMTime` (a time
  directory of an OpenFOAM case, which restarts in the solver that case is set up for);
* ``log`` -- the per-step table of the march, written as the run goes;
* ``checkpoints`` -- the march state every few steps (:class:`Checkpoints`), so a run that stops
  has not lost its work.

Every part is optional. A file with no ``outputs`` section writes the fields as VTK and the log into
``results/`` beside the case file.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping
from pathlib import Path

from aquaflux.io import write_openfoam_time, write_vtu
from aquaflux.mesh import Mesh

__all__ = ["Checkpoints", "FieldWriter", "OpenFOAMTime", "Outputs", "Vtk"]


@dataclasses.dataclass(frozen=True)
class FieldWriter(abc.ABC):
    """Writes the converged fields in one format: :class:`Vtk` or :class:`OpenFOAMTime`.

    Attributes
    ----------
    fields : tuple of str
        The fields to write, by name (``U``, ``p``, and under RANS ``k``, ``omega``, ``nut``); empty,
        every field the case solves for.
    """

    fields: tuple[str, ...] = ()

    def chosen(self, fields: Mapping[str, object]) -> dict[str, object]:
        """The fields this writer writes, of those the case produced.

        Parameters
        ----------
        fields : mapping of {str: array-like}
            Every output field of the case, by name.

        Returns
        -------
        dict
            The named subset, in this writer's order, or all of them.

        Raises
        ------
        ValueError
            If a named field is not one the case produces.
        """
        if not self.fields:
            return dict(fields)
        unknown = [name for name in self.fields if name not in fields]
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.fields names {unknown}, which this case does not produce; "
                f"it produces {list(fields)}."
            )
        return {name: fields[name] for name in self.fields}

    @abc.abstractmethod
    def targets(self, directory: Path, case_directory: Path) -> tuple[Path, ...]:
        """The paths this writer would create, so a run can refuse to replace them before it starts.

        Parameters
        ----------
        directory : pathlib.Path
            The run's output directory.
        case_directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        tuple of pathlib.Path
            Each file or directory it writes.
        """

    @abc.abstractmethod
    def write(
        self, directory: Path, case_directory: Path, mesh: Mesh, fields: Mapping[str, object]
    ) -> Path:
        """Write ``fields`` on ``mesh``.

        Parameters
        ----------
        directory : pathlib.Path
            The run's output directory, which exists.
        case_directory : pathlib.Path
            The directory the case file sits in.
        mesh : Mesh
            The case's mesh.
        fields : mapping of {str: array-like}
            Every output field of the case, by name: ``(n_cells,)`` for a scalar, ``(n_cells, dim)``
            for a vector.

        Returns
        -------
        pathlib.Path
            What was written.
        """


@dataclasses.dataclass(frozen=True)
class Vtk(FieldWriter):
    """The fields as one VTK XML unstructured-grid file (``.vtu``), readable by ParaView and VisIt.

    Attributes
    ----------
    file : str
        The file's name in the output directory.

    Raises
    ------
    ValueError
        If ``file`` does not end in ``.vtu`` or is not a plain file name.
    """

    file: str = "fields.vtu"

    def __post_init__(self) -> None:
        if not self.file.endswith(".vtu") or Path(self.file).name != self.file:
            raise ValueError(f"Vtk.file is a file name ending in .vtu, got {self.file!r}.")

    def targets(self, directory: Path, case_directory: Path) -> tuple[Path, ...]:
        """The one file -- see :meth:`FieldWriter.targets`."""
        del case_directory
        return (directory / self.file,)

    def write(
        self, directory: Path, case_directory: Path, mesh: Mesh, fields: Mapping[str, object]
    ) -> Path:
        """Write the file -- see :meth:`FieldWriter.write`."""
        del case_directory
        return write_vtu(mesh, self.chosen(fields), directory / self.file)


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenFOAMTime(FieldWriter):
    """The fields as one time directory of an OpenFOAM case, a valid restart state for its solver.

    Each field takes its dimensions and boundary conditions from the file of the same name in the
    case's ``template_time`` directory (:func:`~aquaflux.io.write_openfoam_time`), so the case must
    hold a template for every field written -- name the ones it has in ``fields``. It writes into
    ``case``, not into the run's output directory.

    Attributes
    ----------
    case : str
        The OpenFOAM case directory the time directory is written into, relative to the case file
        unless absolute; it holds the template time.
    time : str
        The time directory's name, used verbatim -- quote it in a file (``time: "1000"``), since a
        bare number reads as a number.
    template_time : str or None
        The time directory whose fields are the templates; unset, ``0``.

    Raises
    ------
    ValueError
        If ``case`` or ``time`` is empty.
    """

    case: str
    time: str
    template_time: str | None = None

    def __post_init__(self) -> None:
        if not self.case or not self.time:
            raise ValueError(
                "OpenFOAMTime needs the case directory to write into and the time to write, got "
                f"case={self.case!r}, time={self.time!r}."
            )

    def targets(self, directory: Path, case_directory: Path) -> tuple[Path, ...]:
        """The time directory -- see :meth:`FieldWriter.targets`."""
        del directory
        return (case_directory / self.case / self.time,)

    def write(
        self, directory: Path, case_directory: Path, mesh: Mesh, fields: Mapping[str, object]
    ) -> Path:
        """Write the time directory -- see :meth:`FieldWriter.write`."""
        del directory
        options = {} if self.template_time is None else {"template_time": self.template_time}
        return write_openfoam_time(
            case_directory / self.case, self.time, self.chosen(fields), mesh, **options
        )


@dataclasses.dataclass(frozen=True)
class Checkpoints:
    """Write the march state every few steps, keeping the most recent, in ``checkpoints/``.

    Each file holds the solved-variable state (``omega`` in log form when it is solved so), which is
    what a march resumes from. They are written only by a march; the segregated solve has no steps to
    checkpoint.

    Attributes
    ----------
    every : int
        Write on every ``every``-th step, ``>= 1``.
    keep : int
        How many recent checkpoints to keep, ``>= 1``.

    Raises
    ------
    ValueError
        If a count is not ``>= 1``.
    """

    every: int = 1
    keep: int = 3

    def __post_init__(self) -> None:
        for name in ("every", "keep"):
            if getattr(self, name) < 1:
                raise ValueError(f"Checkpoints.{name} must be >= 1, got {getattr(self, name)!r}.")


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Where a run writes, and what.

    Attributes
    ----------
    directory : str
        The output directory, relative to the case file unless absolute.
    fields : tuple of FieldWriter
        How the converged fields are written; empty writes none.
    log : str or None
        The per-step log's file name in the output directory; ``None`` writes the log to the terminal
        only.
    checkpoints : Checkpoints or None
        The march state every few steps; unset, none.

    Raises
    ------
    ValueError
        If ``directory`` is empty, or ``log`` is not a plain file name.
    TypeError
        If a field writer or the checkpoints are not values of their family.
    """

    directory: str = "results"
    fields: tuple[FieldWriter, ...] = (Vtk(),)
    log: str | None = "march.log"
    checkpoints: Checkpoints | None = None

    def __post_init__(self) -> None:
        if not self.directory:
            raise ValueError("Outputs.directory names the directory a run writes into.")
        if self.log is not None and (not self.log or Path(self.log).name != self.log):
            raise ValueError(f"Outputs.log is a file name, got {self.log!r}.")
        for writer in self.fields:
            if not isinstance(writer, FieldWriter):
                raise TypeError(
                    f"Outputs.fields holds field writers such as Vtk(), got {writer!r}."
                )
        if self.checkpoints is not None and not isinstance(self.checkpoints, Checkpoints):
            raise TypeError(f"Outputs.checkpoints got {self.checkpoints!r}.")

    def output_directory(self, case_directory: Path) -> Path:
        """The output directory, with a relative one taken from ``case_directory``.

        Parameters
        ----------
        case_directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        pathlib.Path
            Where the run writes.
        """
        return Path(case_directory) / self.directory
