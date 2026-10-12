"""What a march holds between its steps: a preconditioner kept current across every step it serves.

The staged driver re-freezes the Newton step between segments, and a continuation builds one step per
rung. Both need the frozen inverse and its per-step refresh hook to survive those rebuilds unchanged,
which is a contract on the session, not on any one preconditioner: a materialized-Jacobian session and
a block-diagonal one both satisfy it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import jax.numpy as jnp

from .strategy import NewtonStrategy

__all__ = ["PreconditionerSession"]


class PreconditionerSession(Protocol):
    """One preconditioner, kept current across every step it serves.

    A session is what a march holds on to between its steps: the frozen inverse, the colouring probe it was
    materialized with, and the per-step refresh hook. Those must outlive a single Newton step -- a
    continuation builds a step per rung, a refresh builds one per segment -- and must be the *same
    objects* each time, because the inverse and the hooks ride in static fields of the step and a new
    object recompiles the whole solve.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        ``(step, state) -> None``, called by the march before every step to re-fit the inverse at that
        step's shift; ``None`` for a family with nothing to re-fit.
    """

    refresh_preconditioner: Callable[[NewtonStrategy, jnp.ndarray], None] | None

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        """The Newton step at ``state``, configured by the march keywords."""
        ...

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        """Re-freeze at the developed ``state``; ``previous`` is the step it replaces."""
        ...

    def rebind(self, assembler: object) -> None:
        """Point the session at another companion of the same case, such as the next continuation rung."""
        ...
