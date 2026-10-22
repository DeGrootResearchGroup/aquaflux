"""Write a mesh and its cell fields to VTK XML files -- the only file I/O in this package.

The reconstruction (:mod:`.topology`) and the serialization (:mod:`.xml`) are pure and test without
a filesystem; this module is the thin shell that opens a file and streams the parts into it, in the
same shape as the readers' split between parsing and I/O.

**Nothing here refuses a non-finite value.** That is a deliberate difference from writing a field
back into a solver's own case, where a ``NaN`` restart fails somewhere that never names the file
responsible. Here the file is for looking at, and looking at where a solution went non-finite is one
of the things it is for -- a viewer draws those cells as blanks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .topology import VTK_HEXAHEDRON, VtkCells, build_vtk_cells
from .xml import cell_data_arrays, pvd_document, vtu_parts

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from aquaflux.mesh import Mesh


def write_vtu(
    mesh: Mesh, fields: Mapping[str, object] | None, path, *, binary: bool = True
) -> Path:
    """Write a mesh and its cell-centred fields as a VTK unstructured-grid file.

    The mesh's cells are written as arbitrary polygons in two dimensions and arbitrary polyhedra in
    three, which is what its face-based storage already describes -- except that a cell bounded by
    six quadrilaterals is written as a hexahedron, VTK's own type for it, which a viewer slices and
    thresholds far more cheaply than the polyhedron it would otherwise be. On a periodic mesh, the
    cell that sees a periodic seam face through its periodic image is written with that face's nodes
    on its own side of the domain, so every cell is drawn where it is.

    Parameters
    ----------
    mesh : Mesh
        The mesh, two- or three-dimensional. A two-dimensional mesh is written in the plane
        ``z = 0``, and a two-component field on it is padded on the same trailing axis so the
        vectors line up with the geometry they are drawn on.
    fields : mapping of {str: array-like} or None
        Field name to cell values -- ``(n_cells,)`` for a scalar, ``(n_cells, k)`` otherwise.
        ``None`` writes the mesh alone, which is how a mesh is inspected before anything is solved
        on it.
    path : str or Path
        Destination file, conventionally with a ``.vtu`` suffix. Its parent directory is created if
        absent.
    binary : bool, optional
        Write the numeric arrays as raw appended bytes (default) or as decimal text. Text is a
        debugging convenience for a small mesh: it carries the same values to the last bit, but a
        polyhedral cell records its whole face stream as integers, and a viewer has to scan and
        convert every one of them rather than copy them, which dominates the time to open a file of
        any size.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If the mesh is neither two- nor three-dimensional, a field's shape or length does not match
        it, or the mesh is periodic and the two sides of a periodic seam do not match node for node.
    """
    cells = build_vtk_cells(mesh)
    arrays = cell_data_arrays(fields or {}, mesh.n_cells, mesh.dim)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for part in vtu_parts(cells, arrays, binary=binary):
            handle.write(part)
    return destination


#: Where VTK's hexahedron takes each corner from, for corners numbered as a unit cube's are by binary
#: index, corner ``4 i + 2 j + k`` at ``(i, j, k)``: the bottom face as a ring whose right-hand normal
#: points into the cell, then the top face in the same order.
_BINARY_TO_VTK_HEXAHEDRON = np.array([0, 4, 6, 2, 1, 5, 7, 3])


def write_hexahedra(
    points,
    corners,
    path,
    *,
    point_fields: Mapping[str, object] | None = None,
    cell_fields: Mapping[str, object] | None = None,
    binary: bool = True,
) -> Path:
    """Write axis-aligned hexahedra -- the cells of a block grid or an octree -- as a VTK file.

    Parameters
    ----------
    points : array_like, shape ``(n_points, 3)``
        The corners' positions.
    corners : array_like of int, shape ``(n_cells, 8)``
        Each cell's corners, as rows of ``points``, numbered as a unit cube's corners are by binary
        index: corner ``4 i + 2 j + k`` is the one at ``(i, j, k)``, with ``i``, ``j`` and ``k`` along
        increasing ``x``, ``y`` and ``z``. A cell is written with its own eight corners only, so where
        a coarse cell meets finer ones a viewer interpolates the coarse cell from its corners and
        nothing else.
    path : str or Path
        Destination file, conventionally with a ``.vtu`` suffix. Its parent directory is created if
        absent.
    point_fields, cell_fields : mapping of {str: array-like}, optional
        Fields at the points, ``(n_points,)`` or ``(n_points, k)``, and at the cells,
        ``(n_cells,)`` or ``(n_cells, k)``.
    binary : bool, optional
        As :func:`write_vtu`.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``corners`` is not ``(n_cells, 8)`` or names a point that does not exist, or a field's
        length does not match.
    """
    points = np.ascontiguousarray(points, dtype=float)
    corners = np.asarray(corners, dtype=np.int64)
    if corners.ndim != 2 or corners.shape[1] != 8:
        raise ValueError(f"write_hexahedra: corners must be (n_cells, 8), got {corners.shape}.")
    if corners.size and (corners.min() < 0 or corners.max() >= len(points)):
        raise ValueError("write_hexahedra: corners name a point that does not exist.")
    n_cells = len(corners)
    cells = VtkCells(
        points=points,
        connectivity=np.ascontiguousarray(corners[:, _BINARY_TO_VTK_HEXAHEDRON].reshape(-1)),
        offsets=8 * np.arange(1, n_cells + 1, dtype=np.int64),
        types=np.full(n_cells, VTK_HEXAHEDRON, dtype=np.uint8),
        faces=None,
        face_offsets=None,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts = vtu_parts(
        cells,
        cell_data_arrays(cell_fields or {}, n_cells, 3),
        point_fields=cell_data_arrays(point_fields or {}, len(points), 3),
        binary=binary,
    )
    with destination.open("wb") as handle:
        for part in parts:
            handle.write(part)
    return destination


def write_pvd(path, frames: Sequence[tuple[float, str]]) -> Path:
    """Write a ``.pvd`` collection indexing per-step files, so a series opens as one dataset.

    Parameters
    ----------
    path : str or Path
        Destination file, conventionally with a ``.pvd`` suffix. Its parent directory is created if
        absent.
    frames : sequence of (float, str)
        One ``(time, file)`` pair per step. The file is recorded verbatim; a path relative to
        ``path``'s own directory keeps the collection and its steps movable together.

    Returns
    -------
    Path
        The file written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pvd_document(frames))
    return destination
