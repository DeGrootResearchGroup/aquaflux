"""Boundary patches and the fields on their faces, as VTK XML polygonal data bound by one index file.

A quantity that lives on a boundary -- the irradiance on a wall, what a surface absorbs -- belongs to
the boundary's faces, not to the cells beside them, so it is written on the faces themselves:

- each patch is one **``.vtp``** file (VTK XML ``PolyData``): the patch's own nodes, its faces as
  polygons -- or, on a two-dimensional mesh, as line segments -- and the face fields as ``CellData``
  under the fields' own names;
- one **``.vtm``** file (a ``vtkMultiBlockDataSet``) lists them, one block per patch, named by the
  patch, so a viewer opens every patch at once and can still pick them out by name.

The patch files sit in a directory named after the index file without its suffix: ``patches.vtm``
indexes ``patches/<patch>.vtp``, and the index names each by that relative path, so the two move
together.

Every boundary patch is written whether it carries fields or not, so the whole boundary can be drawn;
a field appears on the patches it was computed for. Each face is wound so that its normal points out
of the domain, the usual convention for a boundary, whatever order its nodes were stored in.

The numeric arrays are written as by :mod:`.xml`: raw appended bytes by default, decimal text on
request.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np

from aquaflux import ragged

from .topology import stored_ring_is_outward
from .xml import (
    _BLOCK_HEADER_DTYPE,
    _BYTE_ORDER,
    _TYPE_NAMES,
    CellField,
    _Arrays,
    _narrowed,
    cell_data_arrays,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from aquaflux.mesh import Mesh

__all__ = ["boundary_patches", "vtm_document", "vtp_parts", "write_patches"]


def boundary_patches(mesh: Mesh) -> tuple[str, ...]:
    """The mesh's boundary patches that hold faces, in the mesh's patch order.

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    tuple of str
    """
    patches = mesh.face_patches
    return tuple(
        name
        for name in patches.names
        if patches.size(name) and patches.is_boundary_patch(name, mesh.face_cells)
    )


def vtp_parts(
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    fields: Iterable[CellField] = (),
    *,
    lines: bool = False,
    binary: bool = True,
) -> tuple[bytes | memoryview, ...]:
    """Serialize one polygonal piece as the parts of a ``.vtp`` document.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    connectivity : np.ndarray of int
        Every polygon's points, one after another.
    offsets : np.ndarray of int, shape ``(n_polygons,)``
        Where each polygon ends in ``connectivity`` (VTK's convention: the end, not the start).
    fields : iterable of CellField, optional
        Per-polygon fields, from :func:`~aquaflux.io.vtk.cell_data_arrays`.
    lines : bool, optional
        Write the polygons as line segments (``Lines``) rather than as faces (``Polys``) -- the
        boundary of a two-dimensional mesh.
    binary : bool, optional
        Raw appended bytes (default) or decimal text.

    Returns
    -------
    tuple of bytes-like
        The document in order, to be joined or streamed.
    """
    arrays = _Arrays(binary)
    kind = "Lines" if lines else "Polys"
    counts = {"Verts": 0, "Lines": 0, "Strips": 0, "Polys": 0, kind: len(offsets)}
    header = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="PolyData" version="1.0" byte_order="{_BYTE_ORDER}" '
        f'header_type="{_TYPE_NAMES[np.dtype(_BLOCK_HEADER_DTYPE)]}">',
        "  <PolyData>",
        f'    <Piece NumberOfPoints="{len(points)}" '
        + " ".join(f'NumberOf{name}="{count}"' for name, count in counts.items())
        + ">",
        "      <Points>",
        arrays.tag(np.ascontiguousarray(points, dtype=float), "Points", indent=" " * 8),
        "      </Points>",
        f"      <{kind}>",
        arrays.tag(_narrowed(connectivity), "connectivity", indent=" " * 8),
        arrays.tag(_narrowed(offsets), "offsets", indent=" " * 8),
        f"      </{kind}>",
        "      <CellData>",
        *(arrays.tag(field.values, field.name, indent=" " * 8) for field in fields),
        "      </CellData>",
        "    </Piece>",
        "  </PolyData>",
    ]
    appended = arrays.appended()
    if appended:
        header.append('  <AppendedData encoding="raw">')
    text = ("\n".join(header) + "\n").encode()
    if not appended:
        return (text, b"</VTKFile>\n")
    return (text, *appended, b"\n  </AppendedData>\n</VTKFile>\n")


def vtm_document(blocks: Sequence[tuple[str, str]]) -> bytes:
    """A ``.vtm`` index of named blocks, each a file.

    Parameters
    ----------
    blocks : sequence of (str, str)
        One ``(name, file)`` pair per block, in order; the file is written verbatim, so a path relative
        to the index's own directory keeps the two movable.

    Returns
    -------
    bytes
    """
    lines = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="{_BYTE_ORDER}">',
        "  <vtkMultiBlockDataSet>",
        *(
            f'    <DataSet index="{index}" name={quoteattr(name)} file={quoteattr(file)}/>'
            for index, (name, file) in enumerate(blocks)
        ),
        "  </vtkMultiBlockDataSet>",
        "</VTKFile>",
    ]
    return ("\n".join(lines) + "\n").encode()


def _patch_piece(mesh: Mesh, faces: np.ndarray, outward: np.ndarray):
    """A patch's own points, and its faces' rings in them, each wound out of the domain."""
    face_nodes = mesh.face_nodes
    nodes, face_of, counts = ragged.rows(
        np.asarray(face_nodes.offsets), np.asarray(face_nodes.face_node_indices), faces
    )
    # A ring stored running into the domain is read backwards: position k of a face of n nodes takes
    # its node n-1-k.
    start = np.repeat(np.cumsum(counts) - counts, counts)
    rank = np.arange(len(nodes)) - start
    rank = np.where(~outward[faces][face_of], counts[face_of] - 1 - rank, rank)
    nodes = nodes[start + rank]
    used, local = np.unique(nodes, return_inverse=True)
    points = np.asarray(mesh.node_coords)[used]
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    return points, local, np.cumsum(counts)


def write_patches(
    mesh: Mesh,
    fields: Mapping[str, Mapping[str, object]] | None,
    path,
    *,
    patches: Sequence[str] | None = None,
    binary: bool = True,
) -> Path:
    """Write boundary patches and their face fields: one ``.vtp`` per patch, indexed by a ``.vtm``.

    Parameters
    ----------
    mesh : Mesh
        The mesh, two- or three-dimensional.
    fields : mapping of {str: mapping of {str: array-like}} or None
        Per patch, field name to face values -- ``(n_faces,)`` for a scalar, ``(n_faces, k)``
        otherwise -- in the order the patch's faces appear in the mesh. A patch with no entry is
        written without fields; ``None`` writes the patches alone.
    path : str or Path
        The index file, conventionally ``patches.vtm``. The patch files go in the directory beside it
        named by its stem, which is created if absent.
    patches : sequence of str, optional
        The patches to write; unset, every boundary patch that holds faces.
    binary : bool, optional
        Raw appended bytes (default) or decimal text.

    Returns
    -------
    Path
        The index file written.

    Raises
    ------
    ValueError
        If a patch named, or one given fields, is not a boundary patch of the mesh holding faces, if
        a patch's name cannot be a file name, or if a field's length is not the patch's face count.
    """
    fields = dict(fields or {})
    available = boundary_patches(mesh)
    chosen = tuple(available if patches is None else patches)
    unknown = sorted({*chosen, *fields} - set(available))
    if unknown:
        raise ValueError(
            f"{unknown} {'is not a boundary patch' if len(unknown) == 1 else 'are not boundary patches'} "
            f"of the mesh holding faces; those are {list(available)}."
        )
    strays = sorted(set(fields) - set(chosen))
    if strays:
        raise ValueError(f"fields are given for {strays}, which are not among the patches written.")
    unnameable = [name for name in chosen if Path(name).name != name or name in (".", "..")]
    if unnameable:
        raise ValueError(f"patch names {unnameable} cannot be file names.")

    index = Path(path)
    folder = index.parent / index.stem
    folder.mkdir(parents=True, exist_ok=True)
    outward = stored_ring_is_outward(mesh)
    blocks = []
    for name in chosen:
        faces = np.asarray(mesh.face_patches.indices(name))
        given = fields.get(name, {})
        for field, values in given.items():
            if len(np.asarray(values)) != len(faces):
                raise ValueError(
                    f"patch {name!r}: field {field!r} has {len(np.asarray(values))} values, but the "
                    f"patch has {len(faces)} faces."
                )
        points, connectivity, offsets = _patch_piece(mesh, faces, outward)
        arrays = cell_data_arrays(given, len(faces), mesh.dim)
        piece = folder / f"{name}.vtp"
        with piece.open("wb") as handle:
            for part in vtp_parts(
                points, connectivity, offsets, arrays, lines=mesh.dim == 2, binary=binary
            ):
                handle.write(part)
        blocks.append((name, f"{folder.name}/{piece.name}"))
    index.write_bytes(vtm_document(blocks))
    return index
