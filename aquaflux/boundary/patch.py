"""Apply a per-patch closure over named boundary patches — the one iterate-gather-scatter loop.

Both the scalar residual assembler and the coupled-flow assembler build a per-face array by
walking each named boundary patch, gathering that patch's owner cells, evaluating the patch's
boundary closure, and setting the result into the patch's face rows. The iteration is identical;
only the closure differs (a scalar face value vs. a flow velocity / pressure / mass-flux). This
owns that loop in one place, so neither assembler re-open-codes it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity


def apply_per_patch(
    boundary_conditions: Sequence,
    boundary_faces: Sequence[jnp.ndarray],
    face_cells: FaceCellConnectivity,
    init: jnp.ndarray,
    closure: Callable[[object, jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Fold each named patch's closure into ``init``, setting that patch's face rows.

    For each ``(bc, faces)`` pair the patch's owner cells are gathered and
    ``closure(bc, faces, owner)`` is evaluated for the values written at ``init[faces]``. Faces in
    no named patch (interior faces, unlisted boundary faces) keep their ``init`` value.

    Parameters
    ----------
    boundary_conditions : sequence
        One boundary closure per named patch (any type; ``closure`` interprets it).
    boundary_faces : sequence of jnp.ndarray
        The face-index array for each patch, aligned with ``boundary_conditions``.
    face_cells : FaceCellConnectivity
        Owner/neighbour incidence (``mesh.face_cells``) — supplies the per-patch owner gather.
    init : jnp.ndarray
        The array to fold patches into, shape ``(n_faces, ...)`` (typically zeros).
    closure : callable
        ``closure(bc, faces, owner) -> values`` giving the ``init[faces]`` entries for one patch.

    Returns
    -------
    jnp.ndarray
        ``init`` with each patch's rows set, same shape.
    """
    result = init
    for bc, faces in zip(boundary_conditions, boundary_faces, strict=True):
        owner = face_cells.owner[faces]
        result = result.at[faces].set(closure(bc, faces, owner))
    return result
