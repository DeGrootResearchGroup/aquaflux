"""A shift policy preconditioned by one monolithic, frozen inverse of the assembled Jacobian.

A field split, or one block inverse over the whole state, is built off the jit path from a materialized
Jacobian. What a march needs of it is the same two things whichever form it takes: the base policy's
shift diagonal paired with a preconditioner that ignores the shift strength (the inverse is frozen), and
a transposed apply for the adjoint. Nothing here names a residual.

The inverse reaches the step in one of two forms. A host inverse is applied through a host callback and
rides in a static field, re-fitted in place (:class:`MonolithicFactorShiftPolicy`). A field split whose
blocks are both traced multigrid cycles is applied on device and rides as data, so a refresh hands the
march a step carrying the re-fitted arrays (:class:`TracedSplitShiftPolicy`, :func:`with_traced_split`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from ..solve import NewtonStrategy, ShiftPolicy, ShiftTerm, TransposedPreconditioner
from .host_preconditioner import HostPreconditioner
from .traced_field_split import TracedFieldSplit

__all__ = [
    "FrozenTransposeFactory",
    "MonolithicFactorShiftPolicy",
    "TracedSplitShiftPolicy",
    "with_traced_split",
]


class MonolithicFactorShiftPolicy(eqx.Module):
    """A :class:`~aquaflux.solve.ShiftPolicy` that preconditions the whole coupled state with one
    monolithic inverse of the assembled Jacobian, in place of a block-diagonal composition.

    Reuses a base policy's pseudo-transient shift diagonal -- the physics, whatever rows and scales it
    chose -- but replaces its preconditioner with a single monolithic inverse of the assembled Jacobian,
    which forms the true pressure Schur coupling rather than approximating it. That inverse is a field
    split of multigrid V-cycles (:class:`~aquaflux.precondition.FieldSplitPreconditioner`, bounded memory
    on a large three-dimensional mesh), or one block inverse over the whole state
    (:class:`~aquaflux.precondition.block_preconditioner.MaterializedBlockPreconditioner`) -- this policy
    is agnostic to which, needing only the shared callback-matvec interface.

    The inverse is frozen at a reference state and shift (built off the jit path by a
    :class:`~aquaflux.precondition.MaterializedJacobian` session). Unlike the block
    preconditioner's live ``a_P`` rescaling it does not track the developing state; being a far stronger
    preconditioner it tolerates that freezing at a cost of a few extra cycles, and the shift vanishes at
    the root so the frozen inverse never changes the converged solution or its adjoint. Because it
    is a host object it rides as a **static** field rather than a traced
    pytree leaf, and is applied inside the jitted Krylov solve through the callback matvec.

    Attributes
    ----------
    base : ShiftPolicy
        The policy supplying the pseudo-transient shift diagonal (and its per-row shift scale).
    preconditioner : HostPreconditioner
        The frozen coupled inverse (a static field), applied through its ``matvec`` /
        ``matvec(transpose=True)`` callbacks.
    """

    base: ShiftPolicy
    preconditioner: HostPreconditioner = eqx.field(static=True)

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
        return ShiftTerm(diagonal, lambda shift_strength: apply, base.row_shift_scale)

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

    That matters because it ends up in a strategy's ``adjoint_preconditioner_factory``, where, holding
    no arrays, it filters to the static side and so is part of the compiled step's cache key. As a lambda it compared by identity, so a
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


class TracedSplitShiftPolicy(eqx.Module):
    """A :class:`~aquaflux.solve.ShiftPolicy` preconditioned by a field split applied entirely on device.

    The counterpart of :class:`MonolithicFactorShiftPolicy` for a split whose two blocks are both traced
    multigrid cycles: the same base shift diagonal, the same frozen inverse, but the inverse is a
    :class:`~aquaflux.precondition.traced_field_split.TracedFieldSplit` held as **data** rather than a host
    object in a static field. Its hierarchies are therefore arguments of the compiled step, not constants
    baked into it, which is what lets a re-fitted split reuse the compiled step: a refresh does not
    mutate this policy, it replaces the split in the step (:func:`with_traced_split`), and at unchanged
    array shapes that is a compilation-cache hit that applies the new values.

    No host callback is emitted when the preconditioner is applied, so on an accelerator a Krylov
    iteration does not round-trip the state vector through the host.

    Attributes
    ----------
    base : ShiftPolicy
        The policy supplying the pseudo-transient shift diagonal (and its per-row shift scale).
    split : TracedFieldSplit
        The frozen field split, as array leaves.
    """

    base: ShiftPolicy
    split: TracedFieldSplit

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The base policy's shift diagonal, glued to the traced split.

        Both of the base term's shift-dependent parts are forwarded, as in
        :meth:`MonolithicFactorShiftPolicy.shift_term`, and for the same reason.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat state.
        residual : jnp.ndarray or None
            ``R(phi)`` when the caller has it, handed on to the base policy.
        """
        base = self.base.shift_term(phi, residual)
        apply = self.split.matvec()
        return ShiftTerm(base.diagonal, lambda shift_strength: apply, base.row_shift_scale)

    def adjoint_factory(self) -> _TracedSplitAdjoint:
        """The ``state -> M`` factory for the adjoint, transposed there by :func:`jax.linear_transpose`.

        The split is a traced linear map, so unlike the host form it needs no hand-written transpose:
        the generic adjoint machinery derives ``M^T`` from the forward code.
        """
        return _TracedSplitAdjoint(self.split)


class _TracedSplitAdjoint(eqx.Module):
    """``state -> M`` for a traced split, as a module so the split rides in the step as data.

    The step's adjoint factory is a data field, so this module's arrays are leaves of the step like the
    policy's own; a refit replaces both (:func:`with_traced_split`).
    """

    split: TracedFieldSplit

    def __call__(self, state: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        del state  # frozen: the split does not depend on where the adjoint is taken
        return self.split.matvec()


def with_traced_split(step: NewtonStrategy, split: TracedFieldSplit) -> NewtonStrategy:
    """``step`` with its traced field split replaced by ``split``: the result of a refit, as a step.

    A traced split rides in two places of a step, the shift policy and the adjoint factory, and a refit
    must reach both or the adjoint would precondition with the split the march started from. Both are
    replaced together here, with ``equinox.tree_at``, so a split of the same shapes gives a step of the
    same structure -- the compiled step is then a cache hit.

    Parameters
    ----------
    step : NewtonStrategy
        A shifted step whose policy is a :class:`TracedSplitShiftPolicy`.
    split : TracedFieldSplit
        The re-fitted split.

    Raises
    ------
    TypeError
        If the step's policy is not a :class:`TracedSplitShiftPolicy`.
    """
    if not isinstance(step.shift_policy, TracedSplitShiftPolicy):
        raise TypeError(
            "with_traced_split needs a step preconditioned by a TracedSplitShiftPolicy, not "
            f"{type(step.shift_policy).__name__}."
        )
    return eqx.tree_at(
        lambda s: (s.shift_policy.split, s.adjoint_preconditioner_factory.split),
        step,
        (split, split),
    )
