"""A problem's named boundary closures, bound to a mesh's patches.

Both the scalar residual assembler and the coupled-flow assembler drive their boundary conditions
the same way: for each named patch, gather that patch's owner cells, evaluate the patch's closure,
and set the result into the patch's face rows. The iteration is identical; only the closure differs
(a scalar face value vs. a flow velocity / pressure / mass-flux).

:class:`BoundaryConditions` is the named ``{patch: closure}`` collection — constructed the same way
as a property model (``BoundaryConditions({"left": ..., "right": ...})``) and handed to an
assembler's ``build``. The assembler binds it to the mesh once via :meth:`resolve` (turning patch
*names* into concrete boundary-face indices, off the jit path) and then composes :meth:`apply` — the
one iterate-gather-scatter fold — inside the residual. Neither assembler re-open-codes the loop, and
each holds a single ``boundary`` field rather than parallel name / closure / face tuples. The
closures are opaque here (single-field scalar conditions or multi-field flow bundles), so the object
is generic over the closure type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, FacePatches


class BoundaryConditions(eqx.Module):
    """A named ``{patch: closure}`` boundary-condition collection, bound to a mesh on demand.

    Construct from a mapping of patch name to boundary closure, exactly as a property model is
    constructed from a mapping of property name to property —
    ``BoundaryConditions({"left": Dirichlet(1.0), "right": ZeroGradient()})``. The collection is
    initially *unbound*: it knows patch names, not face indices. An assembler's ``build`` binds it
    to the mesh with :meth:`resolve`, which looks each patch name up in ``mesh.face_patches`` for its
    boundary-face indices (a data-dependent lookup, so it runs once off the jit path). :meth:`apply`
    — the per-patch gather-set fold — then runs inside the differentiable residual.

    An ``equinox.Module`` pytree: the closures are dynamic leaves, so a differentiable boundary
    parameter (e.g. a Biot number held inside a convective closure) is a leaf of this tree and
    gradients flow through it.

    Attributes
    ----------
    conditions : dict of {str: closure}
        The boundary closure per patch name (a single-field ``BoundaryCondition`` or a multi-field
        flow bundle; :meth:`apply` leaves interpreting it to the caller's closure). Iteration order
        is preserved.
    faces : dict of {str: jnp.ndarray} or None
        Boundary-face indices per patch once :meth:`resolve` has bound the collection to a mesh;
        ``None`` before then.
    """

    conditions: dict
    faces: dict | None

    def __init__(self, conditions: dict[str, object], _faces: dict | None = None):
        """Collect the ``{patch name: closure}`` mapping (unbound to any mesh).

        Parameters
        ----------
        conditions : dict of {str: closure}
            Boundary closure per patch name; iteration order is preserved.
        _faces : dict of {str: jnp.ndarray}, optional
            Pre-resolved per-patch face indices, set internally by :meth:`resolve` — not by callers.
        """
        self.conditions = dict(conditions)
        self.faces = _faces

    def resolve(self, face_patches: FacePatches) -> BoundaryConditions:
        """Bind to a mesh: look each patch name up in ``face_patches`` for its boundary-face indices.

        The name→index lookup is data-dependent (dynamic shapes), so it runs here — once, off the jit
        path; the resulting index arrays are constant inputs to the differentiable residual. Raises
        ``ValueError`` (from ``face_patches``) if a patch name is absent from the mesh.

        Parameters
        ----------
        face_patches : FacePatches
            The mesh's named face partition (``mesh.face_patches``).

        Returns
        -------
        BoundaryConditions
            A bound copy carrying the same closures plus their per-patch face indices. Idempotent:
            an already-resolved collection is returned unchanged, so re-binding (e.g. a coupled
            residual that reuses a pre-resolved boundary inside its jit) does not re-run the
            dynamic-shape ``nonzero`` lookup on traced mesh labels.
        """
        if self.faces is not None:
            return self
        faces = {name: jnp.asarray(face_patches.indices(name)) for name in self.conditions}
        return BoundaryConditions(self.conditions, _faces=faces)

    def apply(
        self,
        face_cells: FaceCellConnectivity,
        init: jnp.ndarray,
        closure: Callable[[object, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """Fold each patch's closure into ``init``, setting that patch's face rows.

        For each ``(bc, faces)`` pair the patch's owner cells are gathered and
        ``closure(bc, faces, owner)`` is evaluated for the values written at ``init[faces]``. Faces
        in no named patch (interior faces, unlisted boundary faces) keep their ``init`` value.

        Parameters
        ----------
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

        Raises
        ------
        ValueError
            If the collection has not been bound to a mesh with :meth:`resolve`.
        """
        if self.faces is None:
            raise ValueError(
                "BoundaryConditions must be bound to a mesh via resolve(face_patches) before apply()"
            )
        result = init
        for name, bc in self.conditions.items():
            faces = self.faces[name]
            owner = face_cells.owner[faces]
            result = result.at[faces].set(closure(bc, faces, owner))
        return result
