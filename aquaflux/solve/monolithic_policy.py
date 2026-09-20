"""A shift policy preconditioned by one monolithic, frozen inverse of the assembled Jacobian.

The complete LU, the multigrid V-cycle and the field split are all built off the jit path from a
materialized Jacobian and applied through a host callback. What a march needs of any of them is the same
two things: the base policy's shift diagonal paired with a preconditioner that ignores the shift strength
(the inverse is frozen), and a transposed apply for the adjoint. Nothing here names a residual.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from .amg_preconditioner import MonolithicAmgPreconditioner
from .continuation import ShiftPolicy, ShiftTerm
from .lu_preconditioner import MonolithicLuPreconditioner
from .root_adjoint import TransposedPreconditioner

__all__ = ["FrozenTransposeFactory", "MonolithicFactorShiftPolicy"]


class MonolithicFactorShiftPolicy(eqx.Module):
    """A :class:`~aquaflux.solve.ShiftPolicy` that preconditions the whole coupled state with one
    monolithic inverse of the assembled Jacobian, in place of a block-diagonal composition.

    Reuses a base policy's pseudo-transient shift diagonal -- the physics, whatever rows and scales it
    chose -- but replaces its preconditioner with a single monolithic inverse of the assembled Jacobian,
    which forms the true pressure Schur coupling rather than approximating it. That inverse is a complete LU
    (:class:`~aquaflux.solve.MonolithicLuPreconditioner`, exact, one cycle), or a multigrid V-cycle
    (:class:`~aquaflux.solve.MonolithicAmgPreconditioner`, bounded memory on a large three-dimensional
    mesh) -- this policy is agnostic to which, needing only the shared callback-matvec interface. On a
    convection-dominated collocated Rhie--Chow RANS saddle either reaches the forward tolerance
    where the block-triangular preconditioner needs hundreds of cycles.

    The inverse is frozen at a reference state and shift (built off the jit path by a
    :class:`~aquaflux.solve.MaterializedJacobian` session). Unlike the block
    preconditioner's live ``a_P`` rescaling it does not track the developing state; being a far stronger
    preconditioner it tolerates that freezing at a cost of a few extra cycles, and the shift vanishes at
    the root so the frozen inverse never changes the converged solution or its adjoint. Because it
    is a host object (``scipy`` / UMFPACK / PETSc) it rides as a **static** field rather than a traced
    pytree leaf, and is applied inside the jitted Krylov solve through the callback matvec.

    Attributes
    ----------
    base : ShiftPolicy
        The policy supplying the pseudo-transient shift diagonal (and its per-row relaxation).
    preconditioner : MonolithicLuPreconditioner or MonolithicAmgPreconditioner
        The frozen coupled inverse (a static field). Any object exposing the ``matvec`` /
        ``matvec(transpose=True)`` callback interface works.
    """

    base: ShiftPolicy
    preconditioner: MonolithicLuPreconditioner | MonolithicAmgPreconditioner = eqx.field(
        static=True
    )

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The block policy's shift diagonal, glued to the frozen factorization preconditioner.

        The preconditioner is a single frozen apply, and the step solves the shifted system with the
        JAX-side Krylov.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state.
        """
        # ⚠️ Forward BOTH of the base term's β-dependent parts. A wrapper that rebuilds a `ShiftTerm`
        # from only `.diagonal` silently discards whatever else the base put there, and the loss is
        # invisible: the march runs, and the dropped behaviour simply never happens. That is exactly how
        # an earlier per-block damping measured as a no-op on this path.
        base = self.base.shift_term(phi, residual)
        diagonal = base.diagonal
        apply = self.preconditioner.matvec()
        # The factorization is frozen, so the preconditioner does not depend on the shift strength.
        return ShiftTerm(diagonal, lambda relaxation: apply, base.row_relaxation)

    def adjoint_factory(self) -> TransposedPreconditioner:
        """The ``state -> M^T`` factory for the adjoint transpose solve.

        The converged-state adjoint preconditions the (unshifted) transposed coupled Jacobian with the
        frozen factorization's transpose -- the same factors applied with a transposed triangular solve.
        Wrapped in a :class:`~aquaflux.solve.TransposedPreconditioner` because it
        already returns ``M^T``: the generic adjoint machinery derives the transpose with
        :func:`jax.linear_transpose`, which cannot handle the host-callback factorization, so it is
        applied directly instead.
        """
        return TransposedPreconditioner(FrozenTransposeFactory(self.preconditioner))


@dataclasses.dataclass(frozen=True)
class FrozenTransposeFactory:
    """``state -> M^T`` for a frozen monolithic factorization, as a value object rather than a closure.

    The transpose is state-independent -- the factorization is frozen, so the same ``M^T`` serves every
    state -- which is exactly why this can be a value whose equality is the preconditioner's identity.

    That matters because it ends up in a strategy's ``adjoint_preconditioner_factory``, a *static*
    field and hence part of the compiled step's cache key. As a lambda it compared by identity, so a
    Reynolds-continuation rung that rebuilt its engine got a fresh key and recompiled the coupled solve
    even when it was reusing the very same preconditioner. As a value object, two engines sharing one
    preconditioner produce equal factories and the rebuild is a cache hit.

    Attributes
    ----------
    preconditioner : object
        The frozen factorization, supplying ``matvec(transpose=True)``. Compared by identity, which is
        the intended meaning: the same preconditioner object *is* the same operator, and two distinct
        objects generally are not.
    """

    preconditioner: object

    def __call__(self, state: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        del state  # frozen: the transpose does not depend on where the adjoint is taken
        return self.preconditioner.matvec(transpose=True)
