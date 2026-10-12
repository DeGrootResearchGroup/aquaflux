"""What a march holds between its steps: a preconditioner kept current across every step it serves.

The staged driver re-freezes the Newton step between segments, and a continuation builds one step per
rung. Both need the frozen inverse and its per-step refresh hook to survive those rebuilds unchanged,
which is a contract on the session, not on any one preconditioner: a materialized-Jacobian session and
a block-diagonal one both satisfy it.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Protocol

import jax.numpy as jnp

from .strategy import NewtonStrategy, StopsForRefresh

__all__ = ["PreconditionerSession", "RebuildingSession"]


class PreconditionerSession(Protocol):
    """One preconditioner, kept current across every step it serves.

    A session is what a march holds on to between its steps: the frozen inverse, the colouring probe it was
    materialized with, and the per-step refresh hook. Those must outlive a single Newton step -- a
    continuation builds a step per rung, a refresh builds one per segment -- and must be the *same
    objects* each time: a host inverse and the hooks ride in static fields of the step, where a new
    object recompiles the whole solve, and an inverse carried as data must keep its array shapes.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        ``(step, state, *, due=False) -> update or None``, called by the march before every step to
        re-fit the inverse at that step's shift -- unconditionally when ``due`` -- or ``None`` for a
        family with nothing to re-fit. It returns ``None`` when the step needs nothing new (nothing was
        re-fitted, or the inverse is re-fitted in place) and a ``step -> step`` update when the step
        carries the inverse as data and must be handed the re-fitted one.
    """

    refresh_preconditioner: Callable[..., Callable[[NewtonStrategy], NewtonStrategy] | None] | None

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


class RebuildingSession(abc.ABC):
    """A session whose steps are built whole from the problem it is pointed at, with nothing to re-fit.

    Its preconditioner is frozen when a step is built and stays frozen; a session of this kind keeps no
    inverse it could re-fit at a later state or shift. What it can follow is a **change of problem**:
    :meth:`rebind` points it at another, and the march's next call to :meth:`refresh_preconditioner`
    makes a step on that problem at the state the march has reached (:meth:`_repointed`) and hands the
    march its shift, preconditioner and operator stand-in
    (:meth:`~aquaflux.solve.ShiftedStep.preconditioned_as`). The step the march holds keeps the shift
    strength its step control set, which rebuilding the whole step would discard. Until it is re-pointed
    the hook hands the march nothing, so a march over one problem is unchanged by having it.

    The step a re-pointing makes is derived from the one the march holds, so that it differs from it in
    array values alone. The march's step is compiled against the one it holds, so a step of the same
    structure is a compilation-cache hit, and one built from scratch -- whose preconditioner objects or
    coarse-grid shapes differ -- recompiles the whole solve at every re-pointing.

    A subclass supplies :meth:`_build` (the step a build makes from :attr:`assembler`),
    :meth:`_repointed` and :meth:`refresh`.

    Attributes
    ----------
    assembler : object
        The problem the next build is made on: the one the session was opened with, or the one it was
        last re-pointed at.
    """

    def __init__(self, assembler: object) -> None:
        self.assembler = assembler
        # The march keywords of the last build, which a rebuild after a re-pointing is made with.
        self._march: dict = {}
        # Whether the problem has changed since the last build, so the steps in use were built on
        # another one.
        self._repointed_since_build = False

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        """The step at ``state`` on :attr:`assembler`, configured by the march keywords.

        Raises
        ------
        ValueError
            If the step would stop for a preconditioner re-fit: nothing here re-fits, so every hard
            step would end early for a re-fit that never comes.
        """
        step = self._build(state, march)
        if isinstance(step, StopsForRefresh) and step.refresh_on_cycles is not None:
            raise ValueError(
                f"the step stops for a preconditioner re-fit (refresh_on_cycles="
                f"{step.refresh_on_cycles}) but this session keeps no inverse it could re-fit, so every "
                "hard step would end early and nothing would be refreshed. Leave refresh_on_cycles "
                "unset, or precondition with a session that re-fits (a materialized Jacobian)."
            )
        self._march, self._repointed_since_build = march, False
        return step

    def rebind(self, assembler: object) -> None:
        """Point the session at ``assembler``; the next step the march takes is rebuilt on it."""
        self.assembler, self._repointed_since_build = assembler, True

    def refresh_preconditioner(
        self, step: NewtonStrategy, state: jnp.ndarray, *, due: bool = False
    ) -> Callable[[NewtonStrategy], NewtonStrategy] | None:
        """After a re-pointing, the march's steps preconditioned for the new problem at ``state``.

        ``due`` is not consulted: a step asking for a re-fit is refused when it is built.
        """
        del due
        if not self._repointed_since_build:
            return None
        rebuilt = self._repointed(state, step, self._march)
        self._repointed_since_build = False
        return lambda held: held.preconditioned_as(rebuilt)

    @abc.abstractmethod
    def _build(self, state: jnp.ndarray, march: dict) -> NewtonStrategy:
        """The step a build makes at ``state`` on :attr:`assembler`."""

    @abc.abstractmethod
    def _repointed(self, state: jnp.ndarray, held: NewtonStrategy, march: dict) -> NewtonStrategy:
        """The step on :attr:`assembler` at ``state``, made from ``held`` so it keeps ``held``'s structure.

        ``held`` is the step the march holds, built on the problem the session pointed at before; the
        result differs from it in array values alone (see the class summary).
        """

    @abc.abstractmethod
    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        """Re-freeze at the developed ``state``; ``previous`` is the step it replaces."""
