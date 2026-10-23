"""The cell zones and face patches of a mesh, carried onto a rebuilt one.

A transform that rebuilds a mesh with its cells unchanged but some faces dropped (a two-dimensional
collapse removing its caps, a periodic join removing one side of a seam) passes the named zones and
patches on to :meth:`~aquaflux.mesh.Mesh.from_csr` in its index form; these two functions are that
form, shared by every such transform.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .mesh import Mesh


def carried_cell_zones(mesh: Mesh) -> dict[str, np.ndarray] | None:
    """The named (non-default) cell zones as an index dict, for a rebuild that keeps every cell.

    Parameters
    ----------
    mesh : Mesh
        The mesh being rebuilt.

    Returns
    -------
    dict of {str: np.ndarray} or None
        Each named zone's cells; ``None`` when there is none.
    """
    named = {
        name: np.nonzero(np.asarray(mesh.cell_zones.mask(name)))[0]
        for name in mesh.cell_zones.names
        if name != "default"
    }
    return named or None


def carried_face_patches(
    mesh: Mesh, removed: Iterable[str], kept_faces: np.ndarray
) -> dict[str, np.ndarray] | None:
    """The surviving named face patches, re-indexed onto the kept faces.

    The removed patches and the automatic ``"interior"``/``"boundary"`` patches are dropped; the
    latter are reassigned by the mesh constructor from the rebuilt neighbour array.

    Parameters
    ----------
    mesh : Mesh
        The mesh being rebuilt.
    removed : iterable of str
        The patches that do not survive.
    kept_faces : np.ndarray of int, shape ``(n_kept,)``
        The faces kept, in their new order; every face of a surviving patch must be among them.

    Returns
    -------
    dict of {str: np.ndarray} or None
        Each surviving patch's faces in the new numbering; ``None`` when there is none.
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
