"""What every cell-field writer needs from the values it is handed, in one place.

A writer takes a ``{name: values}`` mapping and has to make the same two checks before it can do
anything format-specific: the values have to become a plain float array (they usually arrive as JAX
arrays off a solve), and their length has to match the mesh they claim to live on. The second is the
error worth catching early -- a field of the wrong length otherwise serializes into a perfectly
well-formed file that no reader can use, and the message that says so should not depend on which
format was being written.
"""

from __future__ import annotations

import numpy as np


def as_cell_values(name: str, values, n_cells: int) -> np.ndarray:
    """One field's values as a float array, checked against the mesh's cell count.

    Parameters
    ----------
    name : str
        The field's name. A writer is handed a whole mapping at once, so *which* entry is wrong is
        the only useful thing an error can say about it.
    values : array-like
        Cell values: ``(n_cells,)`` for a scalar field, ``(n_cells, ...)`` for one carrying
        components. Converted with ``np.asarray``, so a JAX array is accepted. What shapes beyond
        the leading axis mean is the caller's question, not this one's.
    n_cells : int
        The cell count the field must match.

    Returns
    -------
    np.ndarray of float
        The values, converted.

    Raises
    ------
    ValueError
        If the values carry no per-cell axis at all, or their length is not ``n_cells``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"field '{name}' is a single value, not one value per cell")
    if array.shape[0] != n_cells:
        raise ValueError(
            f"field '{name}' has {array.shape[0]} values but the mesh has {n_cells} cells"
        )
    return array
