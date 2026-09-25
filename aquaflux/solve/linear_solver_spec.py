"""A linear solver named by its settings, so a case description can choose one.

A ``lineax`` solver is an object built in code, which a case file cannot hold. The values here are the
settings a file can state instead, each building the solver it names: :class:`GmresSolve`, a restarted
GMRES stopping at a relative residual, and :class:`DirectSolve`, a factorization for a system small
enough to factor.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import lineax as lx

from .linear import relative_residual_gmres
from .settings_value import SettingsValue

__all__ = ["DirectSolve", "GmresSolve", "LinearSolverSpec"]


@dataclasses.dataclass(frozen=True)
class LinearSolverSpec(SettingsValue, abc.ABC):
    """A linear solver, by its settings: :class:`GmresSolve` or :class:`DirectSolve`."""

    @abc.abstractmethod
    def build(self) -> lx.AbstractLinearSolver:
        """The solver these settings name.

        Returns
        -------
        lineax.AbstractLinearSolver
            A new solver; two builds from equal settings are equal solvers.
        """


@dataclasses.dataclass(frozen=True)
class GmresSolve(LinearSolverSpec):
    """A restarted GMRES stopping at a Euclidean relative residual, by :func:`relative_residual_gmres`.

    Attributes
    ----------
    rtol : float
        The relative residual ``|A x - b| / |b|`` it stops at, ``> 0``.
    restart : int or None
        The Krylov subspace size before a restart; unset, :func:`relative_residual_gmres`'s own.
    stagnation_iters : int or None
        The restart cycles without progress after which it gives up; unset, its own.
    max_restarts : int or None
        A hard cap on the restart cycles; unset, no cap beyond ``lineax``'s.

    Raises
    ------
    ValueError
        If ``rtol`` is not a positive, finite number, or a count is not ``>= 1``.
    """

    rtol: float
    restart: int | None = None
    stagnation_iters: int | None = None
    max_restarts: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rtol) and self.rtol > 0):
            raise ValueError(
                f"GmresSolve.rtol must be a positive, finite number, got {self.rtol!r}."
            )
        for name in ("restart", "stagnation_iters", "max_restarts"):
            count = getattr(self, name)
            if count is not None and count < 1:
                raise ValueError(f"GmresSolve.{name} must be >= 1, got {count!r}.")

    def build(self) -> lx.AbstractLinearSolver:
        """The GMRES -- see :meth:`LinearSolverSpec.build`."""
        return relative_residual_gmres(
            self.rtol, **{name: value for name, value in self.settings().items() if name != "rtol"}
        )


@dataclasses.dataclass(frozen=True)
class DirectSolve(LinearSolverSpec):
    """A direct factorization (``lineax.AutoLinearSolver(well_posed=True)``), exact in one solve.

    Its cost and memory grow far faster than the system, so it suits a small system only -- a channel
    a few cells wide, say.
    """

    def build(self) -> lx.AbstractLinearSolver:
        """The factorization -- see :meth:`LinearSolverSpec.build`."""
        return lx.AutoLinearSolver(well_posed=True)
