"""Where a case's mesh comes from.

A case file names its mesh by source rather than holding it: a path to read, and whatever that
format's reader needs to know. Relative paths are relative to the directory the case file sits in,
so a case and its mesh can be moved together.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from pathlib import Path

from aquaflux.io import MeshReader, OpenFOAMReader

__all__ = ["MeshSource", "OpenFOAMMesh"]


@dataclasses.dataclass(frozen=True)
class MeshSource(abc.ABC):
    """A mesh a case reads from somewhere.

    Implementations name a format and its settings; :meth:`reader` turns that into the
    :class:`~aquaflux.io.MeshReader` for it, so what the mesh *is* stays with the reader and this
    records only where to find it.
    """

    @abc.abstractmethod
    def reader(self, directory: Path) -> MeshReader:
        """The reader for this mesh, with relative paths taken from ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        MeshReader
            The reader; nothing is read until its ``read()`` is called.
        """


@dataclasses.dataclass(frozen=True)
class OpenFOAMMesh(MeshSource):
    """An OpenFOAM polyMesh, read with :class:`~aquaflux.io.OpenFOAMReader`.

    A case that is one cell thick between ``empty`` patches reads as a two-dimensional mesh, and those
    patches are gone from it -- so a case file never names them.

    Attributes
    ----------
    path : str
        The polyMesh directory, or an OpenFOAM case directory holding ``constant/polyMesh``; relative to
        the case file's directory unless absolute.
    cyclic_match_tolerance : float or None
        The tolerance for matching the faces of a ``cyclic`` patch pair; unset, the reader's own
        default.

    Raises
    ------
    ValueError
        If ``path`` is empty, or the tolerance is not a positive, finite number.
    """

    path: str
    cyclic_match_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("an OpenFOAM mesh needs the path to its polyMesh or case directory.")
        tolerance = self.cyclic_match_tolerance
        if tolerance is not None and not (math.isfinite(tolerance) and tolerance > 0):
            raise ValueError(
                f"cyclic_match_tolerance must be a positive, finite number, got {tolerance!r}."
            )

    def reader(self, directory: Path) -> OpenFOAMReader:
        """The OpenFOAM reader for this mesh -- see :meth:`MeshSource.reader`."""
        options = (
            {}
            if self.cyclic_match_tolerance is None
            else {"cyclic_match_tolerance": self.cyclic_match_tolerance}
        )
        return OpenFOAMReader(Path(directory) / self.path, **options)
