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

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["OffersTracedCycle", "TracedCycle"]


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


def host_in_precision(values: np.ndarray, precision: str) -> np.ndarray:
    """``values`` rounded to ``precision`` exactly as :func:`in_precision` rounds an array leaf.

    For an operator held on the host in the precision its traced form is applied in. The conversion
    runs through JAX rather than numpy because the two differ below the single-precision normal range:
    JAX's CPU conversion flushes such a value to a signed zero, numpy's cast keeps it as a subnormal.
    Rounding here and composing the traced form afterwards therefore gives the same entries, bit for
    bit, as composing it from the double values and rounding there.

    Parameters
    ----------
    values : np.ndarray
        Floating-point values, any shape.
    precision : {"float64", "float32"}
        The floating-point type to round to.

    Returns
    -------
    np.ndarray
        ``values`` itself when it is already in ``precision``; otherwise a new writable array.
    """
    if values.dtype == np.dtype(precision):
        return values
    return np.array(jnp.asarray(values).astype(precision))


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
