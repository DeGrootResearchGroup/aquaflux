"""Read an OpenFOAM scalar field written on a mesh aquaflux has already imported.

A ``volScalarField`` (a cell field) or ``surfaceScalarField`` (a face field) is an internal list
plus a ``boundaryField`` dictionary of per-patch values. The one that matters here is the face flux
``phi``: a scalar transported by an imported flow must ride the flux that flow's continuity closes
on, and ``phi`` is that flux -- rebuilding it as ``(u . n) A`` from the cell velocities satisfies no
discrete continuity.

**Why the face indices line up.** OpenFOAM orders faces interior-first (the upper-triangular
ordering), then boundary faces grouped by patch in ``boundary``-file order, and each patch occupies
a contiguous range. The aquaflux reader carries ``owner`` through unchanged and pads the
interior-only ``neighbour`` list to full length, so an ordinary read never renumbers a face:
aquaflux face ``i`` *is* OpenFOAM face ``i``. A patch's aquaflux indices come back ascending from
``face_patches``, which for a contiguous range is exactly the order the patch's values are written
in.

That correspondence is an inherited convention rather than something this module can enforce, so it
is **checked rather than assumed**: :func:`read_surface_scalar_field` verifies that the mesh's
interior faces really are the leading ``n`` and that each patch's length matches, and raises
naming the mismatch instead of silently placing values on the wrong faces. Two cases are known to
break it, and both refuse to run: a two-dimensional case, whose ``empty``-patch collapse rebuilds
the mesh and renumbers; and a mesh with a fused ``cyclic`` patch pair, whose periodic seam face sits
in the file's boundary block, not its internal one -- refused directly (a ``neighbour_offset`` check)
rather than relying on the leading-block check, which a fused seam can coincidentally still pass.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aquaflux.mesh import Mesh

from .field_writer import infer_extruded_axis
from .foamfile import read_foam_body, read_foam_file
from .grammar import (
    BOUNDARY_FIELD_RE,
    _check_count,
    list_envelope,
    parse_vector_list,
    split_boundary_blocks,
)

_UNIFORM_RE = re.compile(r"\buniform\s+(-?[\d.eE+-]+)")
_UNIFORM_VECTOR_RE = re.compile(r"\buniform\s*\(([^()]*)\)")

#: A component along the dropped axis is treated as zero if it is no larger than this fraction of the
#: field's largest kept component (or of one, for a field that is itself zero).
_DROPPED_COMPONENT_TOLERANCE = 1.0e-8


def _values(body: str, count: int, what: str) -> np.ndarray:
    """Parse a ``uniform X`` or ``nonuniform List<scalar> N ( … )`` entry into ``count`` values.

    Both spellings occur in one file -- a wall's flux is written ``uniform 0`` while an inlet's is a
    full list -- so a reader that handles only the list form silently fails on the walls.
    """
    if "nonuniform" in body:
        declared, inner = list_envelope(body)
        tokens = inner.split()
        _check_count(what, declared, len(tokens))
        if declared != count:
            raise ValueError(f"{what} has {declared} values but {count} are expected")
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
    boundary_match = BOUNDARY_FIELD_RE.search(body)
    internal_end = boundary_match.start() if boundary_match else len(body)
    parts = [_values(body[internal_match.end() : internal_end], n_internal, "internalField")]

    if patch_sizes and boundary_match is None:
        raise ValueError("field has no boundaryField entry")
    blocks = split_boundary_blocks(body)

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

    # A periodic seam is a fused cyclic patch pair (or a generator's own periodic=...): the seam
    # face is interior now but sat in a boundary patch's block in the original OpenFOAM file, so
    # index correspondence is gone even where the leading-block check below would not catch it --
    # a seam face can land immediately after the true interior block by coincidence of patch
    # declaration order, passing that check while still reading the wrong file entries.
    if face_cells.neighbour_offset is not None:
        raise ValueError(
            "mesh has a periodic seam (neighbour_offset is set), so its face indices do not "
            "correspond to an OpenFOAM file's ordering; reading a surface field on a periodic or "
            "cyclic-fused mesh is not supported"
        )

    # The correspondence this module depends on otherwise, checked rather than assumed. It fails
    # on a collapsed 2D mesh, which is rebuilt by the empty-patch transform and renumbered.
    if not interior[:n_internal].all() or interior[n_internal:].any():
        raise ValueError(
            "mesh face ordering is not OpenFOAM's (interior faces are not the leading block), so "
            "field values cannot be placed by index; a 2D case collapsed from an empty-capped "
            "polyMesh is renumbered and cannot be used here"
        )

    # name -> (start_face, n_faces); the start is kept alongside the size so laying the patches
    # out in file order below never has to call `.indices()` a second time -- it is an uncached
    # O(n_faces) boolean-compare-plus-compaction on `LabelledGroups`, paid once per patch here.
    patch_ranges: dict[str, tuple[int, int]] = {}
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
        start = int(indices[0])
        if start < n_internal:
            raise ValueError(
                f"patch '{name}' includes an interior face; ordering is not OpenFOAM's"
            )
        if not np.array_equal(indices, np.arange(start, start + indices.size)):
            raise ValueError(f"patch '{name}' is not a contiguous block of faces")
        patch_ranges[name] = (start, int(indices.size))

    # Lay the patches out in ascending face order, which is how the file writes them.
    ordered = {
        name: size for name, (_start, size) in sorted(patch_ranges.items(), key=lambda kv: kv[1][0])
    }
    values = parse_scalar_field(read_foam_body(path), n_internal, ordered)
    if values.shape[0] != mesh.n_faces:
        raise ValueError(
            f"field has {values.shape[0]} values but the mesh has {mesh.n_faces} faces"
        )
    return values


def parse_vector_field(body: str, n_cells: int) -> np.ndarray:
    """Parse a vector field's ``internalField`` into an ``(n_cells, 3)`` array. Pure: tests on a snippet.

    Both spellings occur -- ``uniform (1 0 0)`` and ``nonuniform List<vector> N ( (…) … )`` -- and a
    reader of the list form alone fails on a field a case sets up uniformly.

    Parameters
    ----------
    body : str
        The payload of the field file (from :func:`~aquaflux.io.openfoam.read_foam_body`).
    n_cells : int
        The number of cells expected.

    Returns
    -------
    np.ndarray
        The per-cell vectors, shape ``(n_cells, 3)``.

    Raises
    ------
    ValueError
        If there is no ``internalField``, it is neither spelling, or it holds the wrong number of
        vectors or components.
    """
    internal_match = re.search(r"\binternalField\b", body)
    if internal_match is None:
        raise ValueError("field has no internalField entry")
    boundary_match = BOUNDARY_FIELD_RE.search(body)
    end = boundary_match.start() if boundary_match else len(body)
    entry = body[internal_match.end() : end]
    if "nonuniform" in entry:
        vectors = parse_vector_list(entry)
        if len(vectors) != n_cells:
            raise ValueError(
                f"internalField has {len(vectors)} values but the mesh has {n_cells} cells"
            )
        return vectors
    match = _UNIFORM_VECTOR_RE.search(entry)
    if match is None:
        raise ValueError("internalField is neither a 'uniform' nor a 'nonuniform' vector entry")
    components = [float(token) for token in match.group(1).split()]
    if len(components) != 3:
        raise ValueError(f"a uniform vector has three components, got {len(components)}")
    return np.tile(np.asarray(components, dtype=np.float64), (n_cells, 1))


def read_openfoam_time(
    case,
    time,
    names: Sequence[str],
    mesh: Mesh,
    *,
    extruded_axis: int | None = None,
) -> dict[str, np.ndarray]:
    """Read cell fields from one time directory of an OpenFOAM case onto ``mesh``'s cells.

    The counterpart of :func:`~aquaflux.io.write_openfoam_time`, with the same reading of a
    two-dimensional case: a vector is stored with three components and a collapsed mesh has two, so
    the component along the axis the collapse removed is dropped -- after checking that it is zero,
    since a nonzero one means the wrong axis (or a flow that is not planar) and dropping it would
    start a solve from a different field than the file holds.

    A cell's index here is OpenFOAM's own, by construction (see the module docstring), so a cell
    field needs no permutation. Only the cell **count** can be checked against the mesh: a file on a
    different mesh of the same size would be read without complaint.

    Parameters
    ----------
    case : str or Path
        The OpenFOAM case directory holding the time directory, whose ``constant/polyMesh`` is where
        the extruded axis is recovered from if it must be.
    time : str or float
        The time directory, used verbatim when a string.
    names : sequence of str
        The fields to read -- each a file of that name, a ``volScalarField`` or a ``volVectorField``.
    mesh : Mesh
        The mesh the fields were written on, as read from the case.
    extruded_axis : int, optional
        Which axis the two-dimensional collapse removed. ``None`` recovers it from the case's polyMesh
        (:func:`~aquaflux.io.infer_extruded_axis`), and only when a vector field is read onto a
        two-dimensional mesh.

    Returns
    -------
    dict of {str: np.ndarray}
        Each field by name: ``(n_cells,)`` for a scalar, ``(n_cells, mesh.dim)`` for a vector.

    Raises
    ------
    FileNotFoundError
        If the time directory, or a named field in it, is missing -- listing what the directory holds.
    ValueError
        If a field is not a cell field, holds the wrong number of values, or has a nonzero component
        along the axis a two-dimensional mesh dropped.
    """
    directory = Path(case) / str(time)
    if not directory.is_dir():
        raise FileNotFoundError(f"no time directory {directory}")
    fields: dict[str, np.ndarray] = {}
    axis = extruded_axis
    for name in names:
        path = directory / name
        if not path.is_file():
            held = sorted(entry.name for entry in directory.iterdir() if entry.is_file())
            raise FileNotFoundError(f"no field {name!r} in {directory}; it holds {held}")
        foam = read_foam_file(path)
        kind = foam.header.get("class")
        if kind == "volScalarField":
            fields[name] = parse_scalar_field(foam.body, mesh.n_cells, {})
        elif kind == "volVectorField":
            vectors = parse_vector_field(foam.body, mesh.n_cells)
            if mesh.dim == 2:
                if axis is None:
                    axis = infer_extruded_axis(case, mesh)
                vectors = _without_the_dropped_component(name, vectors, axis)
            fields[name] = vectors
        else:
            raise ValueError(
                f"{path} is a {kind!r}, not a cell field: only volScalarField and volVectorField "
                "can start a case"
            )
    return fields


def _without_the_dropped_component(name: str, vectors: np.ndarray, axis: int) -> np.ndarray:
    """``vectors`` less their component along ``axis``, refusing a field that is not zero there."""
    if not -3 <= axis <= 2:
        raise ValueError(f"extruded_axis must be one of -3..2; got {axis}")
    dropped = vectors[:, axis]
    kept = np.delete(vectors, axis, axis=1)
    scale = max(float(np.max(np.abs(kept), initial=0.0)), 1.0)
    if float(np.max(np.abs(dropped), initial=0.0)) > _DROPPED_COMPONENT_TOLERANCE * scale:
        raise ValueError(
            f"field {name!r} has a nonzero component along axis {axis % 3}, which the "
            "two-dimensional mesh dropped: the wrong axis was taken, or the flow is not planar"
        )
    return kept
