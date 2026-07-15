"""Read an OpenFOAM polyMesh directory into an aquaflux :class:`~aquaflux.mesh.Mesh`.

:class:`OpenFOAMReader` is the only layer that touches the filesystem: it reads the polyMesh files,
delegates their text to the :mod:`.grammar` parsers, assembles the arrays into a 3D mesh
(:func:`.assembler.assemble`), and — for a two-dimensional case, encoded as a one-cell-thick mesh
capped by two ``empty`` patches — collapses the through-thickness direction to return a genuine 2D
mesh. Only ASCII polyMesh files are supported; a binary file is reported rather than misread.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import numpy as np

from aquaflux.io.reader import MeshReader
from aquaflux.mesh import Mesh, collapse_extruded_direction

from .assembler import assemble
from .foamfile import is_binary, parse_foamfile
from .grammar import (
    parse_boundary,
    parse_cell_zones,
    parse_face_list,
    parse_scalar_list,
    parse_vector_list,
)
from .records import PolyMeshData


def _resolve_polymesh_dir(path) -> Path:
    """Resolve ``path`` to a polyMesh directory, accepting either it or an enclosing case directory.

    Raises
    ------
    FileNotFoundError
        If neither ``path`` nor ``path/constant/polyMesh`` contains a ``points`` file.
    """
    path = Path(path)
    for candidate in (path, path / "constant" / "polyMesh"):
        if (candidate / "points").exists():
            return candidate
    raise FileNotFoundError(
        f"no polyMesh found at {path} (looked for a 'points' file there and in constant/polyMesh)"
    )


class OpenFOAMReader(MeshReader):
    """Read an OpenFOAM polyMesh directory into a :class:`~aquaflux.mesh.Mesh`.

    Attributes
    ----------
    directory : str
        The resolved polyMesh directory (the one holding ``points`` / ``faces`` / ``owner`` / …).
    """

    directory: str = eqx.field(static=True)

    def __init__(self, directory):
        """Construct a reader for a polyMesh or case directory.

        Parameters
        ----------
        directory : str or path-like
            Either the polyMesh directory itself or an OpenFOAM case directory containing
            ``constant/polyMesh``.
        """
        self.directory = str(_resolve_polymesh_dir(directory))

    def _read_field(self, filename: str, parser: Callable, *, required: bool = True):
        """Read and parse one polyMesh file's body; ``None`` if optional and absent.

        Raises
        ------
        FileNotFoundError
            If a required file is missing.
        NotImplementedError
            If the file is in binary format (only ASCII is supported).
        """
        path = Path(self.directory) / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"polyMesh file not found: {path}")
            return None
        foam = parse_foamfile(path.read_text())
        if is_binary(foam):
            raise NotImplementedError(
                f"{path} is in binary format; only ASCII polyMesh files are supported"
            )
        return parser(foam.body)

    def read_polymesh(self) -> PolyMeshData:
        """Read and parse the polyMesh files into raw arrays and records (the only file I/O).

        Returns
        -------
        PolyMeshData
            The parsed points, CSR face connectivity, owner, interior neighbour, boundary patches,
            and cell zones (empty when the optional ``neighbour`` / ``cellZones`` files are absent).
        """
        points = self._read_field("points", parse_vector_list)
        offsets, indices = self._read_field("faces", parse_face_list)
        owner = self._read_field("owner", parse_scalar_list)
        neighbour = self._read_field("neighbour", parse_scalar_list, required=False)
        patches = self._read_field("boundary", parse_boundary)
        zones = self._read_field("cellZones", parse_cell_zones, required=False)
        return PolyMeshData(
            points=points,
            face_node_offsets=offsets,
            face_node_indices=indices,
            owner=owner,
            neighbour_internal=np.zeros(0, dtype=np.int64) if neighbour is None else neighbour,
            patches=patches,
            cell_zones=() if zones is None else zones,
        )

    def read(self) -> Mesh:
        """Read the polyMesh into a mesh, collapsing a two-dimensional (``empty``-capped) case.

        Returns
        -------
        Mesh
            The assembled mesh: 3D in general, or the collapsed 2D mesh when the case is one cell
            thick between two ``empty`` patches.
        """
        data = self.read_polymesh()
        mesh = assemble(data)
        empty_patches = [patch.name for patch in data.patches if patch.type_ == "empty"]
        if empty_patches:
            mesh = collapse_extruded_direction(mesh, empty_patches)
        return mesh


def read_openfoam(directory) -> Mesh:
    """Read an OpenFOAM polyMesh (or case) directory into a :class:`~aquaflux.mesh.Mesh`.

    Parameters
    ----------
    directory : str or path-like
        Either a polyMesh directory or an OpenFOAM case directory containing ``constant/polyMesh``.

    Returns
    -------
    Mesh
        The assembled mesh (2D when the case is a one-cell-thick ``empty``-capped extrusion, else 3D).
    """
    return OpenFOAMReader(directory).read()
