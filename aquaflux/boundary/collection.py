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

    def resolve(
        self, face_patches: FacePatches, face_cells: FaceCellConnectivity
    ) -> BoundaryConditions:
        """Bind to a mesh: look each patch name up for its face indices, and check every boundary face is covered.

        The name→index lookup is data-dependent (dynamic shapes), so it runs here — once, off the jit
        path; the resulting index arrays are constant inputs to the differentiable residual.

        A boundary face that no closure claims is refused. Left alone it would keep the zero
        placeholder :meth:`apply` starts from — a zero face value, which is a boundary condition
        nobody chose — and a case missing a patch would still converge, to the wrong answer. Every
        uncovered patch is reported at once, including the automatic ``"boundary"`` patch that holds
        boundary faces no named patch claims (naming ``"boundary"`` covers them). Interior faces,
        and the ``"padding"`` faces a distributed partition adds to reach a uniform shape, need no
        closure.

        Parameters
        ----------
        face_patches : FacePatches
            The mesh's named face partition (``mesh.face_patches``).
        face_cells : FaceCellConnectivity
            The mesh's face→cell incidence (``mesh.face_cells``), which says which faces are
            boundary faces.

        Returns
        -------
        BoundaryConditions
            A bound copy carrying the same closures plus their per-patch face indices. Idempotent:
            an already-resolved collection is returned unchanged, so re-binding (e.g. a coupled
            residual that reuses a pre-resolved boundary inside its jit) does not re-run the
            dynamic-shape ``nonzero`` lookup on traced mesh labels.

        Raises
        ------
        ValueError
            If a closure names a patch the mesh does not have, or if any boundary face is in a patch
            with no closure (the message lists every such patch and its face count).
        """
        if self.faces is not None:
            return self
        faces = {name: jnp.asarray(face_patches.indices(name)) for name in self.conditions}
        uncovered = face_patches.uncovered_boundary_faces(self.conditions, face_cells)
        if uncovered:
            listed = ", ".join(
                f"'{name}' ({count} face{'' if count == 1 else 's'})"
                for name, count in uncovered.items()
            )
            unnamed = (
                " ('boundary' holds the boundary faces no named patch claims; give it a closure "
                "under that name)"
                if "boundary" in uncovered
                else ""
            )
            raise ValueError(
                f"boundary faces with no boundary condition: {listed}{unnamed}. Every boundary "
                "face needs a closure -- an omitted patch would silently keep a zero face value."
            )
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
        no closure claims keep their ``init`` value: interior faces, and a distributed partition's
        padding faces. A boundary face cannot be among them -- :meth:`resolve` refuses a collection
        that leaves one uncovered.

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
                "BoundaryConditions must be bound to a mesh via resolve(face_patches, face_cells) "
                "before apply()"
            )
        result = init
        for name, bc in self.conditions.items():
            faces = self.faces[name]
            owner = face_cells.owner[faces]
            result = result.at[faces].set(closure(bc, faces, owner))
        return result


def _named(fields: tuple[str, ...]) -> str:
    """``fields`` as prose -- ``"velocity, pressure and mdot"``, or the single-field phrasing."""
    if not fields:
        return "one field, its host equation's"
    if len(fields) == 1:
        return fields[0]
    return f"{', '.join(fields[:-1])} and {fields[-1]}"


def refuse_a_closure_that_closes_other_fields(
    boundary: BoundaryConditions, closes: tuple[str, ...], caller: str
) -> None:
    """Refuse any closure in ``boundary`` that does not close exactly ``closes``.

    The two closure families differ in arity -- a flow bundle closes the velocity, the pressure and
    the mass flux, while a scalar condition closes one field -- and nothing about handing one where
    the other belongs is caught by the shapes. Left unchecked it surfaces from inside an assembler
    as an ``AttributeError`` for whichever method the wrong family lacks: that names an internal
    method rather than the mistake, and does not say which patch carries it.

    This is deliberately **not** inside :meth:`BoundaryConditions.resolve`. The collection is generic
    over its closure type -- that is what lets it carry a plain per-patch value driven by a caller's
    own callable -- so the arity a given assembler needs is the assembler's knowledge, not the
    collection's. Every patch that disagrees is reported at once, not just the first.

    Parameters
    ----------
    boundary : BoundaryConditions
        The collection to check, bound or unbound.
    closes : tuple of str
        What each closure must declare from its ``closes()``; empty for a single-field equation
        (:data:`~aquaflux.boundary.HOST_EQUATION_FIELD`).
    caller : str
        The builder to name in the message.

    Raises
    ------
    ValueError
        If any closure declares a different set, or declares nothing at all.
    """
    wrong = {
        name: getattr(closure, "closes", None)
        for name, closure in boundary.conditions.items()
        if not callable(getattr(closure, "closes", None)) or closure.closes() != closes
    }
    if not wrong:
        return
    listed = "; ".join(
        f"'{name}' has a {type(boundary.conditions[name]).__name__}, which closes "
        + (
            _named(boundary.conditions[name].closes())
            if callable(declared)
            else "nothing it declares"
        )
        for name, declared in wrong.items()
    )
    raise ValueError(
        f"{caller}: {listed}. This equation needs every patch closed for {_named(closes)}."
    )
