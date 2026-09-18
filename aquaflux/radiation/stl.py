"""Reading triangulated surfaces from STL files, in both the ASCII and the binary form.

An STL file is a *triangle soup*: an unordered list of triangles with no shared-vertex table
and no topology. That is all the surface side of a gather needs, which is why the format is
read here rather than through the mesh-import machinery — a triangle soup is not a mesh, and
forcing it through a reader contract whose product is one would mean inventing connectivity
nobody uses.

Each triangle also carries a stored normal, which this module returns as read rather than
trusting: exporters routinely write zeros, and a stored normal disagreeing with the vertex
winding is a real and common defect. Deciding what to do about that is a separate concern and
lives with the other build-time geometry checks.

ASCII files may name several bodies (``solid lampWall`` … ``endsolid``), and those names are
how optical properties get assigned — a sleeve emits, a wall reflects. The binary format has
no such structure: it holds exactly one unnamed body.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

__all__ = ["TriangleSoup", "read_stl"]

_BINARY_HEADER_BYTES = 80
_BINARY_COUNT_BYTES = 4
_BINARY_RECORD_BYTES = 50

_BINARY_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)

_SOLID_LINE = re.compile(rb"^[ \t]*solid[ \t]*(.*?)[ \t\r]*$", re.MULTILINE | re.IGNORECASE)
_VERTEX = re.compile(
    rb"\bvertex\s+(\S+)\s+(\S+)\s+(\S+)",
    re.IGNORECASE,
)
_FACET_NORMAL = re.compile(rb"\bfacet\s+normal\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class TriangleSoup:
    """Triangles as read from a surface file, before any interpretation.

    Plain arrays rather than a traced pytree: reading a file is a build-time step that happens
    once, outside anything compiled.

    Attributes
    ----------
    vertices : np.ndarray, shape ``(n_facets, 3, 3)``
        Triangle vertices; the second axis indexes the three corners in file order, which is
        what defines the winding.
    solid_id : np.ndarray of int, shape ``(n_facets,)``
        Index into :attr:`solid_names` of the body each triangle came from.
    solid_names : tuple of str
        Body names in ``solid_id`` order.
    stored_normal : np.ndarray, shape ``(n_facets, 3)``
        The normal recorded in the file, **as read and not normalized**. Frequently zero, and
        not to be used in place of the winding normal without being checked against it.
    """

    vertices: np.ndarray
    solid_id: np.ndarray
    solid_names: tuple[str, ...]
    stored_normal: np.ndarray

    @property
    def n_facets(self) -> int:
        """Number of triangles read."""
        return int(self.vertices.shape[0])


def _looks_binary(raw: bytes) -> bool:
    """Decide the format by size arithmetic rather than by the leading keyword.

    The usual test — does the file begin with ``solid``? — is unreliable in the one direction
    that matters: a binary file's 80-byte header is arbitrary text, and exporters have been
    known to start it with the word ``solid``, so an ASCII parse is attempted on binary data
    and fails with a message about unparsable numbers. A binary file's length is fixed by its
    own facet count, which no ASCII file matches except by coincidence, so that is the test
    used here.
    """
    if len(raw) < _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES:
        return False
    declared = int(np.frombuffer(raw, dtype="<u4", count=1, offset=_BINARY_HEADER_BYTES)[0])
    expected = _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES + declared * _BINARY_RECORD_BYTES
    return len(raw) == expected


def _read_binary(raw: bytes, name: str) -> TriangleSoup:
    offset = _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES
    records = np.frombuffer(raw, dtype=_BINARY_RECORD, offset=offset)
    return TriangleSoup(
        vertices=np.asarray(records["vertices"], dtype=float),
        solid_id=np.zeros(len(records), dtype=np.int32),
        solid_names=(name,),
        stored_normal=np.asarray(records["normal"], dtype=float),
    )


def _read_ascii(raw: bytes, source: Path) -> TriangleSoup:
    vertex_matches = list(_VERTEX.finditer(raw))
    if len(vertex_matches) % 3 != 0:
        msg = (
            f"{source}: {len(vertex_matches)} vertex records is not a whole number of "
            "triangles -- the file is truncated or malformed"
        )
        raise ValueError(msg)

    try:
        numbers = np.array(
            [[float(group) for group in match.groups()] for match in vertex_matches],
            dtype=float,
        )
    except ValueError as error:
        msg = f"{source}: unparsable vertex coordinate ({error})"
        raise ValueError(msg) from error
    vertices = numbers.reshape(-1, 3, 3) if len(numbers) else np.zeros((0, 3, 3))

    normal_matches = list(_FACET_NORMAL.finditer(raw))
    if len(normal_matches) == len(vertices):
        stored_normal = np.array(
            [[float(group) for group in match.groups()] for match in normal_matches], dtype=float
        )
    else:
        # Not every writer emits one `facet normal` per triangle. The winding is authoritative
        # anyway, so a missing record is recorded as "none stored" rather than refused.
        stored_normal = np.zeros((len(vertices), 3))

    # Bodies are attributed by file position: a triangle belongs to the last `solid` line
    # declared before its first vertex.
    solid_matches = list(_SOLID_LINE.finditer(raw))
    if not solid_matches:
        msg = f"{source}: no `solid` declaration and not a valid binary file"
        raise ValueError(msg)
    names = [
        match.group(1).decode("utf-8", errors="replace").strip() or f"solid{index}"
        for index, match in enumerate(solid_matches)
    ]
    starts = np.array([match.start() for match in solid_matches])
    first_vertex = np.array([match.start() for match in vertex_matches[::3]], dtype=np.int64)
    solid_id = (np.searchsorted(starts, first_vertex, side="right") - 1).astype(np.int32)

    return TriangleSoup(
        vertices=vertices,
        solid_id=solid_id,
        solid_names=tuple(names),
        stored_normal=stored_normal,
    )


def read_stl(path) -> TriangleSoup:
    """Read a triangulated surface from an STL file in either format.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to read. The format is detected from the file's own length, not from whether
        it begins with the word ``solid`` -- see :func:`_looks_binary`.

    Returns
    -------
    TriangleSoup
        The triangles, the body each came from, and the normals as stored.

    Raises
    ------
    ValueError
        If the file is empty, truncated, or holds no recognizable triangles.
    """
    source = Path(path)
    raw = source.read_bytes()
    if not raw:
        msg = f"{source}: empty file"
        raise ValueError(msg)
    soup = _read_binary(raw, source.stem) if _looks_binary(raw) else _read_ascii(raw, source)
    if soup.n_facets == 0:
        msg = f"{source}: no triangles found"
        raise ValueError(msg)
    return soup
