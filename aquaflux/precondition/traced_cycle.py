"""A block inverse's traced cycle as a value: the state it runs on, held apart from the code that runs it.

A traced block inverse is a fixed linear map ``r -> M r`` computed by a compiled multigrid cycle over a
frozen hierarchy. Calling it from **inside** an outer trace -- a jitted Krylov solve, or the on-device
field split -- is where the obvious form goes wrong. A bound method such as ``inverse._solve`` reads the
inverse's hierarchy at trace time, so the outer trace closes over the hierarchy's arrays as compile-time
constants. An in-place refresh then changes the arrays but not the method, so the outer compilation cache
still hits and the compiled program applies the hierarchy it was traced with: a stale preconditioner, with
no retrace and no error to say so.

:class:`TracedCycle` removes that shape. Its ``state`` -- the hierarchy and whatever a smoother derives
from it -- is a pytree of ordinary array leaves, so it enters an outer trace as an argument; only
``step``, the code, is static, and it compares by value. A refresh at unchanged shapes is therefore both a
compilation-cache hit and a changed answer, which is the pair of properties a mid-march refresh needs.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["OffersTracedCycle", "TracedCycle", "TracedForm", "TracedSingleBlock"]


def in_precision(tree, precision: str):
    """``tree`` with every floating-point array leaf cast to ``precision``; other leaves unchanged.

    A leaf already in ``precision`` is returned as the same array, not a copy, so narrowing something
    already narrow costs nothing.

    Parameters
    ----------
    tree : pytree
        Any pytree, such as a hierarchy or a traced split.
    precision : {"float64", "float32"}
        The floating-point type to cast to.
    """
    return jax.tree.map(
        lambda leaf: (
            leaf.astype(precision)
            if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        tree,
    )


class TracedCycle(eqx.Module):
    """One block inverse's traced map ``r -> M r``, with its state as leaves and its code as static.

    Attributes
    ----------
    state : pytree
        Everything the cycle reads that a refresh changes -- the hierarchy and any per-level records
        derived from it. Array leaves, so a new value of the same shapes reuses the compiled program.
    step : callable
        ``(state, residual) -> M residual``. Static, so it must be hashable and compare by value (a
        frozen dataclass of the smoother's settings, or a module-level function); two inverses at the
        same settings then share one compiled cycle.
    """

    state: object
    step: Callable = eqx.field(static=True)

    def __call__(self, residual: jnp.ndarray) -> jnp.ndarray:
        """Apply the cycle to ``residual``, shape ``(n_dofs,)``."""
        return self.step(self.state, residual)


@runtime_checkable
class OffersTracedCycle(Protocol):
    """A block inverse whose cycle can run inside an outer trace.

    Implemented by the multigrid inverses, whose cycles are traced JAX. A host factorization has no such
    cycle and does not implement it -- a sequential triangular solve belongs on a CPU.
    """

    def traced_cycle(self) -> TracedCycle:
        """The cycle over what this inverse currently holds, as a :class:`TracedCycle`."""
        ...


class TracedForm(eqx.Module):
    """A fitted frozen inverse in the form a jitted Krylov solve applies on device: a fixed linear ``r -> M r``.

    What a step needs of the inverse it carries as data, whichever form that inverse takes -- a field
    split of two traced cycles, or one traced cycle over the whole state: the degrees of freedom it
    spans, the precision it is applied in, and the map itself. Its arrays are leaves, so a form of the
    same shapes taken after a re-fit reuses the compiled step and applies the re-fitted values.

    There is no ``transpose`` argument: the map is traced and linear, so :func:`jax.linear_transpose`
    derives ``M^T`` from the forward code, once :func:`~aquaflux.solve.with_transposes` has packed the
    transposes of its operators, which a fitted form does not hold (an adjoint's transpose solve packs
    them itself).

    Every subclass provides two more attributes, as fields or properties, which this base reads: ``n_dofs``,
    the degrees of freedom the map is defined over, and ``precision``, ``"float64"`` or ``"float32"``,
    the precision the map is applied in (a static field, so :meth:`in_precision` can replace it).
    """

    @abc.abstractmethod
    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        """``M residual``, shape ``(n_dofs,)``, in the residual's own dtype."""

    def in_precision(self, precision: str):
        """This form with every floating-point array cast to ``precision``, and applied in it.

        Integer index arrays are unchanged. Narrowing rounds the fitted entries once; widening a
        narrowed form back gives the same rounded entries applied in double precision, which is an
        exactly linear map again -- what a transpose solve run to a tight tolerance needs.
        """
        return dataclasses.replace(in_precision(self, precision), precision=precision)

    def matvec(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner matvec, as the ``residual -> M residual`` callable a Krylov solve takes.

        The traced counterpart of :meth:`~aquaflux.precondition.HostPreconditioner.matvec`, and
        deliberately the same shape -- so a caller selects between a host inverse and a traced one by
        construction rather than by branching. Where that one returns a :func:`jax.pure_callback`, this
        returns the map itself, so the Krylov solve calls straight into it.

        Returned as a plain closure rather than as the bound :meth:`apply`. ``jax.jit`` keys its cache
        on the callable, and a bound method of an ``equinox.Module`` carries the module into that key --
        which raises here, since this one holds array leaves and is deliberately not hashable.
        """

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return self.apply(residual)

        return apply


class TracedSingleBlock(TracedForm):
    """One block inverse's traced cycle applied to the whole state, on device.

    The form of an inverse fitted to the whole materialized Jacobian -- a state with a single group of
    fields, such as a laminar flow's ``(u, p)`` saddle under one hierarchy. Applied through a host
    callback instead, every Krylov iteration would copy the state vector to the host and back although
    the cycle itself is traced.

    Attributes
    ----------
    cycle : TracedCycle
        The inverse's cycle, its hierarchy held as leaves.
    n_dofs : int
        Degrees of freedom of the whole state. Static.
    precision : str
        The precision the cycle is applied in. Static.
    """

    cycle: TracedCycle
    n_dofs: int = eqx.field(static=True)
    precision: str = eqx.field(static=True, default="float64")

    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        """Apply the cycle to ``residual``, shape ``(n_dofs,)``, in :attr:`precision`."""
        return self.cycle(residual.astype(self.precision)).astype(residual.dtype)
