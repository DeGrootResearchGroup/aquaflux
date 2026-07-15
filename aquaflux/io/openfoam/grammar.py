"""Body-grammar parsers for the individual polyMesh payload files (ASCII).

Each polyMesh file's payload is a counted list ``N ( … )`` whose element grammar differs by file:
a ``vectorField`` of points, a ragged ``faceList``, a ``labelList`` of owners/neighbours, a
``polyBoundaryMesh`` dictionary, or a ``cellZones`` list of labelled sets. The four parsers here are
build-time free functions over the comment-stripped body string (:mod:`.foamfile` supplies it), each
sharing :func:`_list_envelope` for the common ``N ( … )`` frame and its count check. The file kind is
known statically at every call site, so these stay plain functions rather than a parser hierarchy.
"""

from __future__ import annotations

import re

import numpy as np

from .records import CellZone, FoamPatch

_COUNT_OPEN_RE = re.compile(r"(\d+)\s*\(")
_PAREN_GROUP_RE = re.compile(r"\(([^()]*)\)")
_FACE_RE = re.compile(r"(\d+)\s*\(([^()]*)\)")
_DICT_BLOCK_RE = re.compile(r"(\w+)\s*\{(.*?)\}", re.DOTALL)
_CELL_LABELS_RE = re.compile(r"cellLabels\s+(?:List<label>\s*)?(\d+)\s*\((.*?)\)", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"(\w+)\s+([^;{}]+);")


def _list_envelope(body: str) -> tuple[int, str]:
    """Return the declared count and the inner text of the leading ``N ( … )`` list.

    Locates the count and its opening parenthesis, then matches that parenthesis by depth so
    balanced inner parentheses (face node lists, ``inGroups`` entries) are handled correctly.

    Raises
    ------
    ValueError
        If no ``N ( … )`` list is present or its parentheses are unbalanced.
    """
    match = _COUNT_OPEN_RE.search(body)
    if match is None:
        raise ValueError("expected a 'N ( ... )' list body")
    count = int(match.group(1))
    open_index = match.end() - 1
    depth = 0
    for i in range(open_index, len(body)):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                return count, body[open_index + 1 : i]
    raise ValueError("unbalanced parentheses in list body")


def _check_count(kind: str, declared: int, found: int) -> None:
    """Raise if a list's declared count disagrees with the number of elements parsed."""
    if declared != found:
        raise ValueError(f"{kind} declares {declared} entries but lists {found}")


def parse_vector_list(body: str) -> np.ndarray:
    """Parse a ``vectorField`` body (the ``points`` file) into an ``(n_nodes, 3)`` float array."""
    count, inner = _list_envelope(body)
    rows = _PAREN_GROUP_RE.findall(inner)
    _check_count("vector list", count, len(rows))
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    points = np.array([[float(x) for x in row.split()] for row in rows], dtype=np.float64)
    if points.shape[1] != 3:
        raise ValueError("each point must have three coordinates")
    return points


def parse_scalar_list(body: str) -> np.ndarray:
    """Parse a ``labelList`` body (the ``owner`` / ``neighbour`` files) into an int array."""
    count, inner = _list_envelope(body)
    tokens = inner.split()
    _check_count("label list", count, len(tokens))
    return np.array(tokens, dtype=np.int64) if tokens else np.zeros(0, dtype=np.int64)


def parse_face_list(body: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a ``faceList`` body (the ``faces`` file) into CSR ``(offsets, indices)``.

    Each face is ``n(i0 i1 … i{n-1})`` with its nodes already in perimeter order; the result is the
    ragged CSR pair :meth:`aquaflux.mesh.Mesh.from_csr` consumes directly.
    """
    count, inner = _list_envelope(body)
    faces = _FACE_RE.findall(inner)
    _check_count("face list", count, len(faces))
    offsets = [0]
    flat: list[int] = []
    for declared_n, nodes_text in faces:
        nodes = nodes_text.split()
        if len(nodes) != int(declared_n):
            raise ValueError(f"a face declares {declared_n} nodes but lists {len(nodes)}")
        flat.extend(int(node) for node in nodes)
        offsets.append(len(flat))
    return np.array(offsets, dtype=np.int64), np.array(flat, dtype=np.int64)


def parse_boundary(body: str) -> tuple[FoamPatch, ...]:
    """Parse a ``polyBoundaryMesh`` body (the ``boundary`` file) into boundary-patch records."""
    count, inner = _list_envelope(body)
    patches = []
    for name, block in _DICT_BLOCK_RE.findall(inner):
        entries = {key: value.strip() for key, value in _KEY_VALUE_RE.findall(block)}
        try:
            patches.append(
                FoamPatch(
                    name=name,
                    type_=entries.get("type", ""),
                    start_face=int(entries["startFace"]),
                    n_faces=int(entries["nFaces"]),
                )
            )
        except KeyError as missing:
            raise ValueError(f"boundary patch '{name}' is missing {missing}") from None
    _check_count("boundary", count, len(patches))
    return tuple(patches)


def parse_cell_zones(body: str) -> tuple[CellZone, ...]:
    """Parse a ``cellZones`` body into zone records (name + cell labels)."""
    count, inner = _list_envelope(body)
    zones = []
    for name, block in _DICT_BLOCK_RE.findall(inner):
        match = _CELL_LABELS_RE.search(block)
        if match is None:
            raise ValueError(f"cellZone '{name}' has no cellLabels list")
        declared_n = int(match.group(1))
        tokens = match.group(2).split()
        _check_count(f"cellZone '{name}'", declared_n, len(tokens))
        labels = np.array(tokens, dtype=np.int64) if tokens else np.zeros(0, dtype=np.int64)
        zones.append(CellZone(name=name, cell_labels=labels))
    _check_count("cellZones", count, len(zones))
    return tuple(zones)
