"""Collapse a one-cell-thick extruded 3D mesh to the equivalent genuine 2D mesh.

Some 3D mesh sources represent a two-dimensional problem as a **single layer of cells**
extruded a short distance along one axis, capped on both sides by a pair of planar boundary
patches normal to that axis. The through-thickness direction carries no physics — it exists only
because the source format has no genuine 2D mode. :func:`collapse_extruded_direction` removes it,
returning a ``dim == 2`` :class:`~aquaflux.mesh.Mesh` whose cells map one-to-one onto the original
layer:

- the two capping patches (named by the caller) are dropped along with the through-thickness axis;
- coincident front/back nodes collapse to one node per in-plane position;
- every remaining face is an extruded side quad, which reduces to the 2D edge joining its two
  distinct in-plane endpoints;
- owner/neighbour, the cell count, and the cell zones carry through unchanged (cells are 1:1),
  and the surviving face patches are re-indexed onto the reduced face numbering.

This is a build-time preprocessing transform (eager numpy, then a validated rebuild), not part of
the differentiable solve — it changes the mesh topology, not any field. It is written against the
mesh's public objects so it is independent of how the extruded mesh was produced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .mesh import Mesh

# Faces normal to the extruded axis are constant along it to within this fraction of the mesh
# extent; the same tolerance quantizes in-plane positions when deduplicating front/back nodes.
_RELATIVE_TOLERANCE = 1e-8


def collapse_extruded_direction(mesh: Mesh, removed_patch_names: Sequence[str]) -> Mesh:
    """Collapse a one-cell-thick extruded 3D mesh to the equivalent 2D mesh.

    The two named patches must be the planar caps normal to the extruded axis (the front and back
    of the layer). The extruded axis is inferred from them: it is the single axis along which each
    cap's nodes share a constant coordinate, the two caps lying at different constants.

    Parameters
    ----------
    mesh : Mesh
        A 3D mesh that is one cell thick along one axis, capped by the two named patches.
    removed_patch_names : sequence of str
        The names of exactly the two capping face patches (e.g. an OpenFOAM case's front/back
        ``empty`` patches). Both must exist in ``mesh.face_patches``.

    Returns
    -------
    Mesh
        The collapsed 2D mesh (``dim == 2``), validated. Cells keep their indices and zones; the
        surviving face patches keep their names on the reduced face numbering.

    Raises
    ------
    ValueError
        If ``mesh`` is not 3D; if ``removed_patch_names`` does not name exactly two patches; if a
        named cap is not planar and normal to a single axis, or the two caps share their constant
        coordinate; or if removing the caps leaves a face that is not an extruded quad (so the mesh
        is not a genuine one-cell-thick extrusion of the removed patches).
    """
    removed = list(removed_patch_names)
    if len(removed) != 2:
        raise ValueError(f"expected exactly two capping patches to remove; got {removed}")

    node_coords = np.asarray(mesh.node_coords)
    if node_coords.shape[1] != 3:
        raise ValueError(f"collapse_extruded_direction expects a 3D mesh; got dim {mesh.dim}")

    extent = node_coords.max(axis=0) - node_coords.min(axis=0)
    scale = float(np.max(extent))
    if scale <= 0.0:
        raise ValueError("mesh has zero spatial extent")
    tolerance = _RELATIVE_TOLERANCE * scale

    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)

    cap_masks = [np.asarray(mesh.face_patches.mask(name)) for name in removed]
    extruded_axis = _extruded_axis(removed, cap_masks, offsets, indices, node_coords, tolerance)
    kept_axes = [axis for axis in range(3) if axis != extruded_axis]

    node_map, new_coords = _collapse_nodes(node_coords[:, kept_axes], tolerance)

    removed_faces = cap_masks[0] | cap_masks[1]
    kept_faces = np.nonzero(~removed_faces)[0]
    edge_nodes = _side_faces_to_edges(kept_faces, offsets, indices, node_map)

    edge_offsets = np.arange(kept_faces.size + 1) * 2
    owner = np.asarray(mesh.face_cells.owner)[kept_faces]
    neighbour = np.asarray(mesh.face_cells.neighbour)[kept_faces]

    return Mesh.from_csr(
        new_coords,
        edge_offsets,
        edge_nodes.ravel(),
        owner,
        neighbour,
        n_cells=mesh.n_cells,
        cell_zones=_carry_cell_zones(mesh),
        face_patches=_carry_face_patches(mesh, removed, kept_faces),
    )


def _extruded_axis(
    removed: list[str],
    cap_masks: list[np.ndarray],
    offsets: np.ndarray,
    indices: np.ndarray,
    node_coords: np.ndarray,
    tolerance: float,
) -> int:
    """Infer the extruded axis: the single axis each cap is constant along, the caps differing on it."""
    axes = []
    constants = []
    for name, mask in zip(removed, cap_masks, strict=True):
        faces = np.nonzero(mask)[0]
        if faces.size == 0:
            raise ValueError(f"capping patch '{name}' is empty")
        nodes = np.unique(np.concatenate([indices[offsets[f] : offsets[f + 1]] for f in faces]))
        spread = np.ptp(node_coords[nodes], axis=0)
        flat = np.nonzero(spread <= tolerance)[0]
        if flat.size != 1:
            raise ValueError(
                f"capping patch '{name}' is not planar and normal to a single axis "
                f"(constant along {flat.size} axes)"
            )
        axis = int(flat[0])
        axes.append(axis)
        constants.append(float(node_coords[nodes, axis].mean()))
    if axes[0] != axes[1]:
        raise ValueError(
            f"capping patches '{removed[0]}' and '{removed[1]}' are normal to different axes "
            f"({axes[0]} and {axes[1]}); they must be the two ends of one extrusion"
        )
    if abs(constants[0] - constants[1]) <= tolerance:
        raise ValueError(
            f"capping patches '{removed[0]}' and '{removed[1]}' lie at the same coordinate; "
            "they must be the front and back of the extrusion"
        )
    return axes[0]


def _collapse_nodes(in_plane: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate coincident in-plane nodes; return ``node_map[old] -> new`` and the 2D coords.

    Front and back nodes share an in-plane position, so quantizing to the tolerance and taking the
    unique positions collapses each front/back pair to one 2D node.
    """
    quantized = np.round(in_plane / tolerance).astype(np.int64)
    unique, node_map = np.unique(quantized, axis=0, return_inverse=True)
    node_map = node_map.reshape(-1)
    new_coords = np.zeros((unique.shape[0], 2), dtype=in_plane.dtype)
    new_coords[node_map] = in_plane  # coincident originals write the same position
    return node_map, new_coords


def _side_faces_to_edges(
    kept_faces: np.ndarray, offsets: np.ndarray, indices: np.ndarray, node_map: np.ndarray
) -> np.ndarray:
    """Reduce each extruded side face to the 2D edge of its two distinct in-plane endpoints.

    Every kept face is a quad whose four nodes project onto two in-plane positions; the edge joins
    those two. The endpoints are read in perimeter order (the first node, then the first that maps
    to a different node); their order does not affect the 2D edge geometry.
    """
    edges = np.empty((kept_faces.size, 2), dtype=np.int64)
    for row, face in enumerate(kept_faces):
        mapped = node_map[indices[offsets[face] : offsets[face + 1]]]
        distinct = mapped[np.sort(np.unique(mapped, return_index=True)[1])]
        if distinct.size != 2:
            raise ValueError(
                f"face {int(face)} reduces to {distinct.size} distinct in-plane node(s), not 2; "
                "the mesh is not a one-cell-thick extrusion of the removed patches"
            )
        edges[row] = distinct
    return edges


def _carry_cell_zones(mesh: Mesh) -> dict[str, np.ndarray] | None:
    """Rebuild the named (non-default) cell zones as an index dict; cells are unchanged (1:1)."""
    named = {
        name: np.nonzero(np.asarray(mesh.cell_zones.mask(name)))[0]
        for name in mesh.cell_zones.names
        if name != "default"
    }
    return named or None


def _carry_face_patches(
    mesh: Mesh, removed: list[str], kept_faces: np.ndarray
) -> dict[str, np.ndarray] | None:
    """Re-index the surviving named face patches onto the reduced (kept-face) numbering.

    The two removed caps and the automatic ``"interior"``/``"boundary"`` patches are dropped; the
    latter are reassigned automatically from the collapsed neighbour array by the mesh constructor.
    """
    new_of_old = np.full(mesh.n_faces, -1, dtype=np.int64)
    new_of_old[kept_faces] = np.arange(kept_faces.size)
    skip = {"interior", "boundary", *removed}
    carried = {}
    for name in mesh.face_patches.names:
        if name in skip:
            continue
        old = np.nonzero(np.asarray(mesh.face_patches.mask(name)))[0]
        carried[name] = new_of_old[old]
    return carried or None
