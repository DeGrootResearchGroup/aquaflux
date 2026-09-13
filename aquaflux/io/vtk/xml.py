"""Serialize VTK cell connectivity and cell fields as XML documents. Pure: no filesystem.

Two documents are built here, both as a sequence of byte-like *parts* a caller concatenates or
streams:

- **``.vtu``** -- one unstructured-grid piece: the points, the cell connectivity from
  :mod:`.topology`, and the cell-centred fields. Cell-centred is the honest finite-volume
  representation, written as VTK ``CellData`` with no interpolation to the nodes (a viewer can
  interpolate for display on demand, which is a display choice rather than a property of the data).
- **``.pvd``** -- a collection indexing per-step ``.vtu`` files by time, which is how a transient
  series is opened as one dataset.

Numeric arrays go into a single ``<AppendedData encoding="raw">`` block by default, each preceded by
a 64-bit byte count, with the ``<DataArray>`` tags carrying byte offsets into it. That is the form
that scales, because of how much of a polyhedral file is integers: a cell's topology is its whole
face stream -- a node count and the nodes, for every face, twice over for the interior ones -- which
runs to tens of millions of entries on a mesh of a few million cells. Written as raw bytes those are
copied straight into the reader's arrays; written as decimal every one of them has to be scanned and
converted, which is what dominates the time to open the file. (Size follows the mesh rather than the
format: a small mesh's indices are fewer characters than the fixed width of an integer, and the
order reverses as they grow.) An ASCII rendering stays available for reading a small mesh by eye,
and writes the same values to the last bit.

Returning parts rather than one finished buffer is what lets a caller stream a file larger than it
would want to hold twice; each part is either a small piece of markup or a view onto an array that
already exists.

A parallel ``.pvtu`` index would reuse :func:`cell_data_arrays` unchanged: it declares the same
array names and component counts, without any data, alongside the list of piece files.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, NamedTuple
from xml.sax.saxutils import quoteattr

import numpy as np

from aquaflux.io.cell_fields import as_cell_values

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from .topology import VtkCells

#: Marks where the appended-data block's payload starts; offsets are counted from just after it.
_APPENDED_MARKER = b"_"

#: The byte count that precedes each appended block, named in the file's ``header_type``.
_BLOCK_HEADER_DTYPE = np.uint64

#: Significant digits per ASCII value. Enough to round-trip a double exactly, so reading a mesh by
#: eye and reading it back give the same numbers as the binary form.
_ASCII_PRECISION = 17

_BYTE_ORDER = "LittleEndian" if sys.byteorder == "little" else "BigEndian"

#: VTK's spelling of the NumPy scalar types this writer emits.
_TYPE_NAMES = {
    np.dtype(np.float64): "Float64",
    np.dtype(np.int32): "Int32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.uint8): "UInt8",
    np.dtype(np.uint64): "UInt64",
}


class CellField(NamedTuple):
    """One cell-centred field, shaped for VTK.

    Attributes
    ----------
    name : str
        The array name, as it appears in a viewer.
    values : np.ndarray
        Values, shape ``(n_cells, n_components)``.
    """

    name: str
    values: np.ndarray

    @property
    def components(self) -> int:
        """Number of components per cell."""
        return self.values.shape[1]


def cell_data_arrays(fields: Mapping[str, object], n_cells: int, dim: int) -> tuple[CellField, ...]:
    """Validate and shape a field mapping into VTK cell arrays.

    Parameters
    ----------
    fields : mapping of {str: array-like}
        Field name to cell values: shape ``(n_cells,)`` for a scalar, ``(n_cells, k)`` otherwise.
        Converted with ``np.asarray``, so a JAX array is accepted.
    n_cells : int
        The mesh's cell count, which every field's length must match.
    dim : int
        The mesh's spatial dimension. On a two-dimensional mesh a two-component field is a vector
        and is padded to three components with a **trailing** zero, matching the ``z = 0`` padding
        of the points it is drawn on -- see :func:`~aquaflux.io.vtk.topology.build_vtk_cells` on why
        the axis the case was extruded along is not the one to restore here.

    Returns
    -------
    tuple of CellField
        One entry per field, in the mapping's order.

    Raises
    ------
    ValueError
        If a field's length is not ``n_cells``, or its shape is neither a scalar nor a
        two-dimensional array of components.
    """
    arrays = []
    for name, values in fields.items():
        array = as_cell_values(name, values, n_cells)
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim == 2:
            if dim == 2 and array.shape[1] == 2:
                array = np.pad(array, ((0, 0), (0, 1)))
        else:
            raise ValueError(
                f"field '{name}' has shape {np.shape(values)}; expected (n_cells,) for a scalar or "
                f"(n_cells, n_components) otherwise"
            )
        arrays.append(CellField(name, np.ascontiguousarray(array)))
    return tuple(arrays)


def _narrowed(values: np.ndarray) -> np.ndarray:
    """An index array as the narrowest signed type that holds it, halving the largest blocks."""
    array = np.asarray(values)
    info = np.iinfo(np.int32)
    fits = array.size == 0 or (int(array.min()) >= info.min and int(array.max()) <= info.max)
    return np.ascontiguousarray(array, dtype=np.int32 if fits else np.int64)


def _ascii_text(values: np.ndarray) -> str:
    """An array's values as whitespace-separated decimal text, one cell's tuple per line."""
    flat = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(-1, 1)
    if np.issubdtype(values.dtype, np.floating):
        rows = (" ".join(f"{v:.{_ASCII_PRECISION}g}" for v in row) for row in flat)
    else:
        rows = (" ".join(str(v) for v in row) for row in flat)
    return "\n".join(rows)


class _Arrays:
    """Collects the numeric arrays of a document and renders each one's ``<DataArray>`` tag.

    Holds the appended-block offsets, which is the one piece of state serialization needs: an
    array's tag names the byte position of its data, so the tags cannot be written until the sizes
    of everything before them are known.
    """

    def __init__(self, binary: bool) -> None:
        self._binary = binary
        self._blocks: list[np.ndarray] = []
        self._offset = 0

    def tag(self, array: np.ndarray, name: str | None = None, *, indent: str) -> str:
        """The ``<DataArray>`` element for ``array``, registering its data for the appended block."""
        attributes = [f'type="{_TYPE_NAMES[array.dtype]}"']
        if name is not None:
            attributes.append(f"Name={quoteattr(name)}")
        if array.ndim > 1:
            attributes.append(f'NumberOfComponents="{array.shape[1]}"')
        if not self._binary:
            body = _ascii_text(array)
            return (
                f'{indent}<DataArray {" ".join(attributes)} format="ascii">\n'
                f"{body}\n{indent}</DataArray>"
            )
        attributes.append(f'format="appended" offset="{self._offset}"')
        self._blocks.append(array)
        self._offset += _BLOCK_HEADER_DTYPE().itemsize + array.nbytes
        return f"{indent}<DataArray {' '.join(attributes)}/>"

    def appended(self) -> list[bytes | memoryview]:
        """The appended-data section: the marker, then each block's byte count and payload."""
        if not self._binary:
            return []
        parts: list[bytes | memoryview] = [_APPENDED_MARKER]
        for array in self._blocks:
            parts.append(_BLOCK_HEADER_DTYPE(array.nbytes).tobytes())
            parts.append(memoryview(array).cast("B"))
        return parts


def vtu_parts(
    cells: VtkCells,
    fields: Iterable[CellField] = (),
    *,
    binary: bool = True,
) -> tuple[bytes | memoryview, ...]:
    """Serialize one unstructured-grid piece as the parts of a ``.vtu`` document.

    Parameters
    ----------
    cells : VtkCells
        Points and connectivity, from :func:`~aquaflux.io.vtk.topology.build_vtk_cells`.
    fields : iterable of CellField, optional
        Cell-centred fields, from :func:`cell_data_arrays`. Default: no fields, i.e. the mesh alone.
    binary : bool, optional
        Write the numeric arrays as raw bytes in an appended block (default), or as decimal text
        inline. Text is for reading a small mesh by eye; it carries the same values but is several
        times larger and much slower to load.

    Returns
    -------
    tuple of bytes-like
        The document in order. Join them for the complete file, or write them one after another to
        stream it. Each is either markup or a view onto an array that already exists, so joining is
        the only step that needs the whole document in memory at once.
    """
    arrays = _Arrays(binary)
    fields = tuple(fields)
    lines = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="UnstructuredGrid" version="1.0" byte_order="{_BYTE_ORDER}" '
        f'header_type="{_TYPE_NAMES[np.dtype(_BLOCK_HEADER_DTYPE)]}">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{cells.n_points}" NumberOfCells="{cells.n_cells}">',
        "      <Points>",
        arrays.tag(np.ascontiguousarray(cells.points), "Points", indent=" " * 8),
        "      </Points>",
        "      <Cells>",
        arrays.tag(_narrowed(cells.connectivity), "connectivity", indent=" " * 8),
        arrays.tag(_narrowed(cells.offsets), "offsets", indent=" " * 8),
        arrays.tag(np.ascontiguousarray(cells.types), "types", indent=" " * 8),
    ]
    if cells.faces is not None:
        # Both are required whenever any cell is a polyhedron, and meaningless otherwise.
        lines.append(arrays.tag(_narrowed(cells.faces), "faces", indent=" " * 8))
        lines.append(arrays.tag(_narrowed(cells.face_offsets), "faceoffsets", indent=" " * 8))
    lines.append("      </Cells>")
    lines.append("      <CellData>")
    lines.extend(arrays.tag(field.values, field.name, indent=" " * 8) for field in fields)
    lines.extend(["      </CellData>", "    </Piece>", "  </UnstructuredGrid>"])

    appended = arrays.appended()
    if appended:
        lines.append('  <AppendedData encoding="raw">')
    header = ("\n".join(lines) + "\n").encode()
    if not appended:
        return (header, b"</VTKFile>\n")
    return (header, *appended, b"\n  </AppendedData>\n</VTKFile>\n")


def pvd_document(frames: Sequence[tuple[float, str]]) -> bytes:
    """A ``.pvd`` collection indexing per-step files by time.

    Parameters
    ----------
    frames : sequence of (float, str)
        One ``(time, file)`` pair per step, in the order they should appear. The file is written
        verbatim, so a path relative to the collection's own directory keeps the pair movable.

    Returns
    -------
    bytes
        The complete document.
    """
    lines = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="Collection" version="1.0" byte_order="{_BYTE_ORDER}">',
        "  <Collection>",
        *(
            f'    <DataSet timestep="{time:.{_ASCII_PRECISION}g}" part="0" '
            f"file={quoteattr(str(file))}/>"
            for time, file in frames
        ),
        "  </Collection>",
        "</VTKFile>",
    ]
    return ("\n".join(lines) + "\n").encode()
