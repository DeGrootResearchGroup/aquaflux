"""Write an aquaflux :class:`~aquaflux.mesh.Mesh` as an OpenFOAM polyMesh directory (ASCII).

The counterpart of :mod:`.reader`. :func:`.disassembler.disassemble` lays the mesh out in the order
and winding the format requires; the ``format_*`` functions here turn those arrays into the text of
each file, pure and testable on their own; and :class:`OpenFOAMWriter` is the only part that touches
the filesystem.

Points are written at full double precision -- the shortest decimal that reads back as the same
number -- so a mesh read back from what was written has bit-identical coordinates and geometry.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np

from aquaflux.io.writer import MeshWriter

from .disassembler import disassemble
from .foamfile import SEPARATOR, format_header

if TYPE_CHECKING:  # pragma: no cover
    from aquaflux.mesh import Mesh

    from .records import CellZone, FoamPatch, PolyMeshData

#: Where a case keeps its mesh, relative to the case directory.
POLYMESH_LOCATION = "constant/polyMesh"

#: Every file the format reads as part of a mesh. Overwriting a mesh removes all of them first, so a
#: zone file left over from the previous mesh can never be read against the new one's labels.
MESH_FILES = (
    "points",
    "faces",
    "owner",
    "neighbour",
    "boundary",
    "cellZones",
    "faceZones",
    "pointZones",
)


def _counted_list(entries: list[str]) -> list[str]:
    """The ``N ( ... )`` frame every list in the format uses, one entry per line."""
    return [str(len(entries)), "(", *entries, ")"]


def _file(class_name: str, object_name: str, body: list[str], *, note: str | None = None) -> str:
    """A whole polyMesh file: header, body, closing rule."""
    lines = [*format_header(class_name, POLYMESH_LOCATION, object_name, note=note), "", *body]
    return "\n".join([*lines, "", SEPARATOR, ""])


def format_points(points: np.ndarray) -> str:
    """The ``points`` file: one ``(x y z)`` per node, each coordinate exactly round-tripping.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_nodes, 3)``

    Returns
    -------
    str
    """
    rows = np.asarray(points, dtype=np.float64).tolist()
    return _file(
        "vectorField", "points", _counted_list([f"({x!r} {y!r} {z!r})" for x, y, z in rows])
    )


def format_faces(face_node_offsets: np.ndarray, face_node_indices: np.ndarray) -> str:
    """The ``faces`` file: each face as ``n(i0 i1 ...)``, in the order given.

    Parameters
    ----------
    face_node_offsets : np.ndarray of int, shape ``(n_faces + 1,)``
        Compressed-sparse-row (CSR) row pointers: face ``f``'s nodes are
        ``face_node_indices[face_node_offsets[f]:face_node_offsets[f + 1]]``.
    face_node_indices : np.ndarray of int
        Every face's nodes, concatenated.

    Returns
    -------
    str
    """
    offsets = np.asarray(face_node_offsets).tolist()
    nodes = [str(i) for i in np.asarray(face_node_indices).tolist()]
    entries = [f"{end - start}({' '.join(nodes[start:end])})" for start, end in pairwise(offsets)]
    return _file("faceList", "faces", _counted_list(entries))


def format_labels(labels: np.ndarray, object_name: str, *, note: str | None = None) -> str:
    """A ``labelList`` file (``owner`` or ``neighbour``): one cell index per line.

    Parameters
    ----------
    labels : np.ndarray of int
    object_name : str
        ``"owner"`` or ``"neighbour"``.
    note : str, optional
        Written as the header's ``note`` entry.

    Returns
    -------
    str
    """
    entries = [str(i) for i in np.asarray(labels).tolist()]
    return _file("labelList", object_name, _counted_list(entries), note=note)


def format_boundary(patches: tuple[FoamPatch, ...]) -> str:
    """The ``boundary`` file: one dictionary per patch, in the order given.

    Parameters
    ----------
    patches : tuple of FoamPatch

    Returns
    -------
    str
    """
    entries: list[str] = []
    for patch in patches:
        entries += [f"    {patch.name}", "    {", f"        type            {patch.type_};"]
        if patch.in_groups:
            groups = " ".join(patch.in_groups)
            entries.append(f"        inGroups        List<word> {len(patch.in_groups)}({groups});")
        entries += [
            f"        nFaces          {patch.n_faces};",
            f"        startFace       {patch.start_face};",
            "    }",
        ]
    body = [str(len(patches)), "(", *entries, ")"]
    return _file("polyBoundaryMesh", "boundary", body)


def format_cell_zones(zones: tuple[CellZone, ...]) -> str:
    """The ``cellZones`` file: one dictionary per zone, holding its cells' labels.

    Parameters
    ----------
    zones : tuple of CellZone

    Returns
    -------
    str
    """
    entries: list[str] = []
    for zone in zones:
        labels = [str(i) for i in np.asarray(zone.cell_labels).tolist()]
        entries += [
            zone.name,
            "{",
            "    type            cellZone;",
            f"    cellLabels      List<label> {len(labels)}",
            "(",
            *labels,
            ")",
            ";",
            "}",
        ]
    body = [str(len(zones)), "(", *entries, ")"]
    return _file("regIOobject", "cellZones", body)


def format_polymesh(data: PolyMeshData, n_cells: int) -> dict[str, str]:
    """The text of every file of a polyMesh directory, by file name.

    Parameters
    ----------
    data : PolyMeshData
        The arrays, already in the format's order and winding (:func:`.disassembler.disassemble`).
    n_cells : int
        The number of cells, written into the ``owner`` file's ``note`` as the format does.

    Returns
    -------
    dict of {str: str}
        ``points``, ``faces``, ``owner``, ``neighbour`` and ``boundary``, and ``cellZones`` when there
        are zones.
    """
    note = (
        f"nPoints: {data.points.shape[0]} nCells: {n_cells} "
        f"nFaces: {data.owner.shape[0]} nInternalFaces: {data.neighbour_internal.shape[0]}"
    )
    files = {
        "points": format_points(data.points),
        "faces": format_faces(data.face_node_offsets, data.face_node_indices),
        "owner": format_labels(data.owner, "owner", note=note),
        "neighbour": format_labels(data.neighbour_internal, "neighbour", note=note),
        "boundary": format_boundary(data.patches),
    }
    if data.cell_zones:
        files["cellZones"] = format_cell_zones(data.cell_zones)
    return files


class OpenFOAMWriter(MeshWriter):
    """Write a mesh as the ``constant/polyMesh`` directory of an OpenFOAM case.

    Attributes
    ----------
    directory : str
        The case directory. The mesh goes in its ``constant/polyMesh``, created if absent.
    overwrite : bool
        Replace a mesh already there. Without it, a polyMesh directory that already holds any of
        :data:`MESH_FILES` is refused rather than mixed with the new files.
    """

    directory: str = eqx.field(static=True)
    overwrite: bool = eqx.field(static=True, default=False)

    def __init__(self, directory, *, overwrite: bool = False):
        """Construct a writer for a case directory.

        Parameters
        ----------
        directory : str or path-like
            The OpenFOAM case directory (the one that holds, or will hold, ``constant/polyMesh``).
        overwrite : bool
            Replace a mesh already in the case.
        """
        self.directory = str(directory)
        self.overwrite = overwrite

    def replaces(self) -> tuple[Path, ...]:
        """The files of :data:`MESH_FILES` already in the case's polyMesh directory."""
        target = Path(self.directory) / POLYMESH_LOCATION
        return tuple(target / name for name in MESH_FILES if (target / name).exists())

    def write(self, mesh: Mesh) -> Path:
        """Write ``mesh`` and return the polyMesh directory.

        Parameters
        ----------
        mesh : Mesh
            A mesh :func:`.disassembler.disassemble` accepts.

        Returns
        -------
        Path
            The ``constant/polyMesh`` directory written.

        Raises
        ------
        FileExistsError
            If the case already has a mesh and ``overwrite`` is not set.
        ValueError, NotImplementedError
            If the mesh is one the format cannot hold as written (see
            :func:`.disassembler.disassemble`). Raised before anything is written.
        """
        files = format_polymesh(disassemble(mesh), mesh.n_cells)
        target = Path(self.directory) / POLYMESH_LOCATION
        existing = self.replaces()
        if existing and not self.overwrite:
            raise FileExistsError(
                f"{target} already holds a mesh ({', '.join(path.name for path in existing)}); "
                "pass overwrite=True to replace it"
            )
        target.mkdir(parents=True, exist_ok=True)
        for path in existing:
            path.unlink()
        for name, text in files.items():
            (target / name).write_text(text)
        return target


def write_openfoam(mesh: Mesh, directory, *, overwrite: bool = False) -> Path:
    """Write ``mesh`` as the ``constant/polyMesh`` of the OpenFOAM case at ``directory``.

    Parameters
    ----------
    mesh : Mesh
        A three-dimensional, non-periodic mesh whose boundary faces all belong to named patches (see
        :func:`.disassembler.disassemble`).
    directory : str or path-like
        The case directory.
    overwrite : bool
        Replace a mesh already in the case.

    Returns
    -------
    Path
        The ``constant/polyMesh`` directory written. :func:`~aquaflux.io.read_openfoam` of the case
        reads the mesh back with the same cells, in the same order.
    """
    return OpenFOAMWriter(directory, overwrite=overwrite).write(mesh)
