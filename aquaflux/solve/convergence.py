"""When a nonlinear solve has converged: the residual measure and the tolerances taken in it, as one value.

A stopping test ``measure(R) <= atol + rtol * measure(R0)`` has no meaning without its measure. On a
coupled turbulent system the plain Euclidean norm of the residual is almost entirely the ``omega``
block, while a row-scaled measure reports a fractional change per equation, so the same ``rtol`` asks
for very different convergence in the two. :class:`Convergence` therefore carries the measure and the
two tolerances together, and a solver takes all three as one setting.

The measure is chosen by value -- :class:`Euclidean`, :class:`RowScaled` or :class:`BlockScaled` -- and
built against the problem being solved, which supplies whatever the measure needs through
:class:`ResidualMeasures`. A problem that cannot supply a measure refuses it by name when the solve
starts, rather than silently measuring in something else.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp

from .settings_value import SettingsValue

if TYPE_CHECKING:
    from .norm import ResidualNorm
    from .strategy import NewtonStrategy

__all__ = [
    "PLAIN_RESIDUAL",
    "BlockScaled",
    "Convergence",
    "Euclidean",
    "MeasureBuilder",
    "ResidualMeasure",
    "ResidualMeasures",
    "RowScaled",
]

#: ``(step, state) -> ResidualNorm``: the measure a march judges one outer iteration by, built at the
#: state that iteration starts from. ``step`` is the Newton step the iteration runs, for a measure that
#: reads its scales from the step's own shift.
MeasureBuilder = Callable[["NewtonStrategy", jnp.ndarray], "ResidualNorm"]


class ResidualMeasures(Protocol):
    """What a problem can measure its residual in, beyond the Euclidean norm.

    A solve hands one of these to the measure it was configured with. A problem that has no row
    diagonals, or no block structure, raises from the method it cannot support, naming the measure.
    """

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> ResidualNorm:
        """The row-equilibrated, field-normalized measure at ``state``.

        Parameters
        ----------
        step : NewtonStrategy
            The step the measured iteration runs, whose shift supplies the row diagonals.
        state : jnp.ndarray
            The state the scales are taken at.

        Returns
        -------
        ResidualNorm
            The measure, its scales fixed at ``state``.
        """
        ...

    def block_scaled(self, state: jnp.ndarray) -> ResidualNorm:
        """The measure scaling each block of the residual by its own magnitude at ``state``.

        Parameters
        ----------
        state : jnp.ndarray
            The state whose per-block residual magnitudes become the scales.

        Returns
        -------
        ResidualNorm
            The measure, its scales fixed at ``state``.
        """
        ...


class _PlainResidual:
    """A residual with no structure a scaled measure could use: only :class:`Euclidean` applies."""

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> ResidualNorm:
        del step, state
        raise TypeError(
            "RowScaled() divides each row by its own diagonal and needs a problem that supplies them, "
            "such as the coupled RANS solve; this solve measures only a plain residual. Use Euclidean()."
        )

    def block_scaled(self, state: jnp.ndarray) -> ResidualNorm:
        del state
        raise TypeError(
            "BlockScaled() scales each field block by its own magnitude and needs a problem that knows "
            "its blocks, such as the coupled RANS solve; this solve measures only a plain residual. Use "
            "Euclidean()."
        )


#: The measures of a residual with no known structure: Euclidean only.
PLAIN_RESIDUAL: ResidualMeasures = _PlainResidual()


class ResidualMeasure(SettingsValue, abc.ABC):
    """Which measure a nonlinear solve judges its residual by, as a value.

    Each concrete value names one measure. The measure a solve steers by -- its line search, its
    pseudo-transient shift, its inner linear solve's stop -- and the one its convergence is judged in are
    the same one.

    This class is abstract: construct :class:`Euclidean`, :class:`RowScaled` or :class:`BlockScaled`.
    """

    @abc.abstractmethod
    def _builder(self, measures: ResidualMeasures, initial_state: jnp.ndarray) -> MeasureBuilder:
        """How this measure is built for each outer iteration of one solve.

        Called once, when the solve starts, so a measure the problem cannot supply is refused before
        any step is taken.

        Parameters
        ----------
        measures : ResidualMeasures
            What the problem can measure its residual in.
        initial_state : jnp.ndarray
            The state the solve starts from.

        Returns
        -------
        MeasureBuilder
            ``(step, state) -> ResidualNorm``.
        """


@dataclasses.dataclass(frozen=True)
class Euclidean(ResidualMeasure):
    """The plain 2-norm of the residual.

    Correct when every unknown sits on a comparable scale. On a coupled system whose blocks differ by
    orders of magnitude it judges only the largest block.
    """

    def _builder(self, measures: ResidualMeasures, initial_state: jnp.ndarray) -> MeasureBuilder:
        del measures, initial_state
        return _euclidean


def _euclidean(step: NewtonStrategy, state: jnp.ndarray) -> ResidualNorm:
    del step, state
    return jnp.linalg.norm


@dataclasses.dataclass(frozen=True)
class RowScaled(ResidualMeasure):
    """Each row divided by its own diagonal, each block by its field's magnitude, rebuilt every iteration.

    Reports a fractional change per equation, so every block contributes comparably whatever units its
    equation is written in (:class:`~aquaflux.solve.RowScaledNorm`). The scales are taken afresh at the
    start of each outer iteration and held for that iteration's line search, so they follow the
    developing flow rather than the initial condition: scales frozen at a cold start over-report a
    developed residual, and a tolerance taken in them can ask for a residual the march never reaches.
    Holding them within an iteration keeps a trial step from being preferred for shrinking its own
    divisor rather than its residual.
    """

    def _builder(self, measures: ResidualMeasures, initial_state: jnp.ndarray) -> MeasureBuilder:
        del initial_state
        return measures.row_scaled


@dataclasses.dataclass(frozen=True)
class BlockScaled(ResidualMeasure):
    """Each block of the residual divided by its own magnitude at the state the solve starts from.

    The coarser field-aware measure (:class:`~aquaflux.solve.BlockScaledNorm`), with one scale per block.
    The scales are taken once and held for the whole solve, including across a preconditioner refresh:
    the measure normalizes itself, so re-taking them at a developed state would re-base every later
    residual toward one and leave the stopping target, measured at the start, out of reach.
    """

    def _builder(self, measures: ResidualMeasures, initial_state: jnp.ndarray) -> MeasureBuilder:
        norm = measures.block_scaled(initial_state)

        def held(step: NewtonStrategy, state: jnp.ndarray) -> ResidualNorm:
            del step, state
            return norm

        return held


@dataclasses.dataclass(frozen=True)
class Convergence(SettingsValue):
    """The stopping test of a nonlinear solve: ``measure(R) <= atol + rtol * measure(R0)``.

    ``R0`` is the residual at the state the solve starts from, taken in the same measure. Every field
    left unset takes the default of the solve it is given to, so ``Convergence(atol=1e-5)`` changes the
    absolute tolerance and nothing else.

    ⚠️ **A relative tolerance depends on the starting state.** ``rtol`` asks for a fraction of the
    initial residual, so a better initial guess asks for a smaller residual. For a solve that must reach
    one level whatever it starts from -- each rung of a continuation, say -- set ``rtol=0`` and give that
    level as ``atol``.

    Attributes
    ----------
    measure : ResidualMeasure or None
        The measure the residual is judged in, and the march steered by.
    rtol : float or None
        The relative tolerance, a fraction of the initial residual in ``measure``.
    atol : float or None
        The absolute tolerance, in ``measure``.

    Raises
    ------
    TypeError
        If ``measure`` is not a :class:`ResidualMeasure` value.
    """

    measure: ResidualMeasure | None = None
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self) -> None:
        if self.measure is not None and not isinstance(self.measure, ResidualMeasure):
            raise TypeError(
                "measure must be a residual-measure value such as Euclidean(), RowScaled() or "
                f"BlockScaled(), got {type(self.measure).__name__}."
            )
