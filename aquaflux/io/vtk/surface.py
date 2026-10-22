"""A surface of triangles and the values on them, as one VTK XML polygonal-data (``.vtp``) file.

For a surface that is not a mesh's boundary -- a drawing's faces, triangulated to be looked at and
picked from -- with a value per triangle, such as the face of the drawing each lies on. Corners
shared by triangles are written once.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .patches import vtp_parts
from .xml import cell_data_arrays

__all__ = ["write_triangles"]


def write_triangles(
    corners: np.ndarray, fields: Mapping[str, object] | None, path, *, binary: bool = True
) -> Path:
    """Write triangles, and a value per triangle, as one ``.vtp`` file.

    Parameters
    ----------
    corners : np.ndarray, shape ``(n_triangles, 3, 3)``
        Each triangle's corners, in the order its normal is wound by.
    fields : mapping of {str: array-like} or None
        Field name to per-triangle values, ``(n_triangles,)`` or ``(n_triangles, k)``.
    path : str or Path
        The file to write.
    binary : bool, optional
        Raw appended bytes (default) or decimal text.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``corners`` is not of shape ``(n, 3, 3)``, or a field's length is not ``n``.
    """
    corners = np.asarray(corners, dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (3, 3):
        raise ValueError(f"triangles' corners have shape (n, 3, 3), got {corners.shape}.")
    points, connectivity = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    arrays = cell_data_arrays(dict(fields or {}), corners.shape[0], 3)
    offsets = 3 * np.arange(1, corners.shape[0] + 1)
    path = Path(path)
    with path.open("wb") as handle:
        for part in vtp_parts(points, connectivity.reshape(-1), offsets, arrays, binary=binary):
            handle.write(part)
    return path
