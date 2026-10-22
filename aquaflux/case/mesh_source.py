"""Where a case's mesh comes from.

A case file names its mesh by the file it is read from rather than holding it: aquaflux's own mesh
file (:class:`AquafluxMesh`), which ``aquaflux mesh`` writes from a mesh file
(:mod:`aquaflux.case.mesh_file`), or an OpenFOAM polyMesh (:class:`OpenFOAMMesh`). Relative paths are
relative to the directory the case file sits in, so a case and its mesh can be moved together.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from pathlib import Path
from typing import ClassVar

from aquaflux.io import AQMESH_SUFFIX, AquafluxMeshReader, OpenFOAMReader
from aquaflux.mesh import Mesh

__all__ = ["AquafluxMesh", "MeshSource", "OpenFOAMMesh"]


@dataclasses.dataclass(frozen=True)
class MeshSource(abc.ABC):
    """A mesh a case reads."""

    @abc.abstractmethod
    def read(self, directory: Path) -> Mesh:
        """The mesh, with any relative path taken from ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        Mesh
            The mesh.
        """

    def provenance(self, directory: Path) -> dict | None:
        """How the mesh was made, as its file records it; ``None`` for a format that records none.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.
        """
        return None


@dataclasses.dataclass(frozen=True)
class AquafluxMesh(MeshSource):
    """aquaflux's own mesh file (``.aqmesh``), read with :class:`~aquaflux.io.AquafluxMeshReader`.

    The mesh comes back exactly as it was written -- the same cells, faces, patches, groups and
    zones, a two-dimensional or periodic mesh as such -- and a file changed since it was written is
    refused.

    Attributes
    ----------
    path : str
        The ``.aqmesh`` file; relative to the case file's directory unless absolute.

    Raises
    ------
    ValueError
        If ``path`` does not name a ``.aqmesh`` file.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("path",)

    path: str

    def __post_init__(self) -> None:
        if Path(self.path).suffix != AQMESH_SUFFIX:
            raise ValueError(
                f"an AquafluxMesh names a {AQMESH_SUFFIX} file, got {self.path!r}; an OpenFOAM "
                "polyMesh is an OpenFOAMMesh."
            )

    def read(self, directory: Path) -> Mesh:
        """The mesh in the file -- see :meth:`MeshSource.read`."""
        return AquafluxMeshReader(Path(directory) / self.path).read()

    def provenance(self, directory: Path) -> dict | None:
        """The header's record of how the mesh was made -- see :meth:`MeshSource.provenance`.

        A mesh ``aquaflux mesh`` made records the mesh file's settings as they ran, the aquaflux
        version and commit, when it started and how long it took, and a summary of its quality.
        """
        return AquafluxMeshReader(Path(directory) / self.path).header().provenance


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

    path_fields: ClassVar[tuple[str, ...]] = ("path",)

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
        """The reader for this mesh, with a relative :attr:`path` taken from ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        OpenFOAMReader
            The reader; nothing is read until its ``read()`` is called.
        """
        options = (
            {}
            if self.cyclic_match_tolerance is None
            else {"cyclic_match_tolerance": self.cyclic_match_tolerance}
        )
        return OpenFOAMReader(Path(directory) / self.path, **options)

    def read(self, directory: Path) -> Mesh:
        """The polyMesh, read -- see :meth:`MeshSource.read`."""
        return self.reader(directory).read()
