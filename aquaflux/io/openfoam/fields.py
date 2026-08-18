"""Read an OpenFOAM scalar field written on a mesh aquaflux has already imported.

A ``volScalarField`` (a cell field) or ``surfaceScalarField`` (a face field) is an internal list
plus a ``boundaryField`` dictionary of per-patch values. The one that matters here is the face flux
``phi``: a scalar transported by an imported flow must ride the flux that flow's continuity closes
on, and ``phi`` is that flux -- rebuilding it as ``(u . n) A`` from the cell velocities satisfies no
discrete continuity.

**Why the face indices line up.** OpenFOAM orders faces interior-first (the upper-triangular
ordering), then boundary faces grouped by patch in ``boundary``-file order, and each patch occupies
a contiguous range. The aquaflux reader carries ``owner`` through unchanged and pads the
interior-only ``neighbour`` list to full length, so it never renumbers a face: aquaflux face ``i``
*is* OpenFOAM face ``i``. A patch's aquaflux indices come back ascending from ``face_patches``,
which for a contiguous range is exactly the order the patch's values are written in.

That correspondence is an inherited convention rather than something this module can enforce, so it
is **checked rather than assumed**: :func:`read_surface_scalar_field` verifies that the mesh's
interior faces really are the leading ``n`` and that each patch's length matches, and raises
naming the mismatch instead of silently placing values on the wrong faces. A two-dimensional case is
the known exception -- the ``empty``-patch collapse rebuilds the mesh and does renumber -- so this
refuses to run on one.
"""

from __future__ import annotations

import re

import numpy as np

from aquaflux.mesh import Mesh

from .foamfile import read_foam_body
from .grammar import list_envelope

# A ``boundaryField { patch { … } patch { … } }`` entry: the patch name and its dictionary body,
# matched by brace depth rather than by regex so a nested ``value nonuniform … ( … )`` list is safe.
_BOUNDARY_FIELD_RE = re.compile(r"\bboundaryField\s*\{", re.DOTALL)
_UNIFORM_RE = re.compile(r"\buniform\s+(-?[\d.eE+-]+)")


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_index``."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced braces")


def _values(body: str, count: int, what: str) -> np.ndarray:
    """Parse a ``uniform X`` or ``nonuniform List<scalar> N ( … )`` entry into ``count`` values.

    Both spellings occur in one file -- a wall's flux is written ``uniform 0`` while an inlet's is a
    full list -- so a reader that handles only the list form silently fails on the walls.
    """
    if "nonuniform" in body:
        declared, inner = list_envelope(body)
        tokens = inner.split()
        if declared != len(tokens):
            raise ValueError(f"{what} declares {declared} values but lists {len(tokens)}")
        if declared != count:
            raise ValueError(f"{what} has {declared} values but the patch has {count} faces")
        return np.array(tokens, dtype=np.float64)
    match = _UNIFORM_RE.search(body)
    if match is None:
        raise ValueError(f"{what} is neither a 'uniform' nor a 'nonuniform' entry")
    return np.full(count, float(match.group(1)), dtype=np.float64)


def parse_scalar_field(body: str, n_internal: int, patch_sizes: dict[str, int]) -> np.ndarray:
    """Parse a scalar-field body into one flat array over internal faces then patch faces.

    Pure: no filesystem, so it tests on a string snippet.

    Parameters
    ----------
    body : str
        The payload of the field file (from :func:`~aquaflux.io.openfoam.read_foam_body`).
    n_internal : int
        Number of internal entries expected (internal faces for a surface field).
    patch_sizes : dict of {str: int}
        Face count per patch, in the order the values are to be laid out after the internal block.

    Returns
    -------
    np.ndarray
        Concatenated ``[internal, patch_0, patch_1, …]``, length
        ``n_internal + sum(patch_sizes.values())``.

    Raises
    ------
    ValueError
        If the internal block or any patch entry is missing, malformed, or the wrong length.
    """
    internal_match = re.search(r"\binternalField\b", body)
    if internal_match is None:
        raise ValueError("field has no internalField entry")
    boundary_match = _BOUNDARY_FIELD_RE.search(body)
    internal_end = boundary_match.start() if boundary_match else len(body)
    parts = [_values(body[internal_match.end() : internal_end], n_internal, "internalField")]

    if patch_sizes and boundary_match is None:
        raise ValueError("field has no boundaryField entry")
    blocks: dict[str, str] = {}
    if boundary_match is not None:
        open_index = boundary_match.end() - 1
        inner = body[open_index + 1 : _matching_brace(body, open_index)]
        cursor = 0
        while (brace := inner.find("{", cursor)) != -1:
            name = inner[cursor:brace].split()[-1] if inner[cursor:brace].split() else ""
            close = _matching_brace(inner, brace)
            blocks[name] = inner[brace + 1 : close]
            cursor = close + 1

    for name, size in patch_sizes.items():
        if name not in blocks:
            raise ValueError(f"field has no boundaryField entry for patch '{name}'")
        parts.append(_values(blocks[name], size, f"patch '{name}'"))
    return np.concatenate(parts)


def read_volume_scalar_field(path, mesh: Mesh) -> np.ndarray:
    """Read a ``volScalarField``'s internal (cell) values onto ``mesh``'s cell ordering.

    The cell-field counterpart of :func:`read_surface_scalar_field`, for reading a reference
    solution -- an eddy viscosity, a transported scalar -- back onto the mesh it was computed on.
    Only the ``internalField`` is returned: a ``volScalarField``'s ``boundaryField`` holds *face*
    values, a different quantity on a different index space, so folding the two into one flat array
    would produce something no consumer wants.

    Cell placement needs no ordering guard the way face placement does. ``assemble`` derives cell
    indices directly from the ``owner``/``neighbour`` labels it reads, so a cell's index is
    OpenFOAM's own by construction rather than by an inherited convention.

    Parameters
    ----------
    path : str or Path
        The field file, e.g. ``<case>/2000/nut``.
    mesh : Mesh
        The mesh the field was written on.

    Returns
    -------
    np.ndarray
        The per-cell values, shape ``(n_cells,)``.

    Raises
    ------
    ValueError
        If the internal block is missing, malformed, or not ``n_cells`` long.
    """
    return parse_scalar_field(read_foam_body(path), mesh.n_cells, {})


def read_surface_scalar_field(path, mesh: Mesh) -> np.ndarray:
    """Read a ``surfaceScalarField`` (e.g. ``phi``) onto ``mesh``'s face ordering.

    Parameters
    ----------
    path : str or Path
        The field file, e.g. ``<case>/2000/phi``.
    mesh : Mesh
        The mesh the field was written on -- imported from that case's ``constant/polyMesh``, so the
        face ordering corresponds (see the module docstring).

    Returns
    -------
    np.ndarray
        The per-face values, shape ``(n_faces,)``, owner-outward (an inflow is negative).

    Raises
    ------
    ValueError
        If the mesh's face ordering is not the imported OpenFOAM one -- interior faces leading,
        each patch a contiguous ascending block -- or a patch's length disagrees with the file.
    """
    face_cells = mesh.face_cells
    interior = np.asarray(face_cells.interior)
    n_internal = int(interior.sum())

    # The correspondence this module depends on, checked rather than assumed. It fails on a
    # collapsed 2D mesh, which is rebuilt by the empty-patch transform and renumbered.
    if not interior[:n_internal].all() or interior[n_internal:].any():
        raise ValueError(
            "mesh face ordering is not OpenFOAM's (interior faces are not the leading block), so "
            "field values cannot be placed by index; a 2D case collapsed from an empty-capped "
            "polyMesh is renumbered and cannot be used here"
        )

    patch_sizes: dict[str, int] = {}
    for name in mesh.face_patches.names:
        # "interior" and "boundary" are assigned automatically from the boundary mask rather than
        # read from the polyMesh, so neither names a patch the field file writes. "interior" holds
        # the interior faces, already covered by the internal block; "boundary" holds boundary faces
        # no named patch claimed -- legal in a mesh, but with nowhere to read their values from.
        if name == "interior":
            continue
        indices = np.asarray(mesh.face_patches.indices(name))
        if indices.size == 0:
            continue
        if name == "boundary":
            raise ValueError(
                f"{indices.size} boundary faces are not in a named patch, so the field has no "
                "values for them; the polyMesh must tile its boundary with named patches"
            )
        if indices[0] < n_internal:
            raise ValueError(
                f"patch '{name}' includes an interior face; ordering is not OpenFOAM's"
            )
        if not np.array_equal(indices, np.arange(indices[0], indices[0] + indices.size)):
            raise ValueError(f"patch '{name}' is not a contiguous block of faces")
        patch_sizes[name] = int(indices.size)

    # Lay the patches out in ascending face order, which is how the file writes them.
    ordered = dict(
        sorted(
            patch_sizes.items(), key=lambda kv: int(np.asarray(mesh.face_patches.indices(kv[0]))[0])
        )
    )
    values = parse_scalar_field(read_foam_body(path), n_internal, ordered)
    if values.shape[0] != mesh.n_faces:
        raise ValueError(
            f"field has {values.shape[0]} values but the mesh has {mesh.n_faces} faces"
        )
    return values
