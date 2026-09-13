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

from .topology import build_vtk_cells
from .xml import cell_data_arrays, pvd_document, vtu_parts

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from aquaflux.mesh import Mesh


def write_vtu(
    mesh: Mesh, fields: Mapping[str, object] | None, path, *, binary: bool = True
) -> Path:
    """Write a mesh and its cell-centred fields as a VTK unstructured-grid file.

    The mesh's cells are written as arbitrary polygons in two dimensions and arbitrary polyhedra in
    three, which is what its face-based storage already describes; no reconstruction into standard
    element types is attempted, and none is needed.

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
        If the mesh is neither two- nor three-dimensional, or a field's shape or length does not
        match it.
    """
    cells = build_vtk_cells(mesh)
    arrays = cell_data_arrays(fields or {}, mesh.n_cells, mesh.dim)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for part in vtu_parts(cells, arrays, binary=binary):
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
