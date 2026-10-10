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
import jax.numpy as jnp

__all__ = ["OffersTracedCycle", "TracedCycle"]


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
