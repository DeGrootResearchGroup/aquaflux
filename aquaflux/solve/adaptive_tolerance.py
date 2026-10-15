"""How tightly each step of a march solves its linear system -- an adaptive inexact-Newton tolerance.

An inexact Newton step solves its linear system only to a relative tolerance ``eta`` (the
"forcing term" of Dembo, Eisenstat & Steihaug 1982): ``norm(J d + R) <= eta norm(R)``. A fixed ``eta`` is
cheap while the march is far from the root, but it costs in two places, and both are visible only when
each solve stops on its residual alone (the residual-only GMRES stop) rather than over-solving it:

* **Near the root.** Once Newton is otherwise exact, a step solved to ``eta`` removes about a factor
  ``eta`` of the residual, so the end of the march converges *linearly* at that rate where a tight solve
  converges quadratically. Measured on a 20 x 14 coupled channel solved to a relative tolerance of
  ``1e-10``: 41 steps at ``eta = 0.3`` against 27 for a solver that over-solves every step.
* **After a setback.** A loose direction that the line search has to cut back, or that raises the
  residual, can lead a march into a region where every later direction is poor; re-solving the bad step
  more tightly from the same state does not recover it, while solving the *next* steps more tightly keeps
  the march out of that region. Measured on a coupled channel's Reynolds-continuation target march: at
  ``eta = 0.3`` throughout it failed after 60 steps, and with this rule it converged in 12.

Tightening every step instead (the Eisenstat & Walker 1996 forcing terms) was measured to cost far more
than it saves on a pseudo-transient march, whose steps are mostly limited by the pseudo-time shift rather
than by the linear solve. So this rule tightens only where the evidence says the solve is what is
holding the march back, and returns to the base tolerance otherwise.
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["AdaptiveLinearTolerance"]


@dataclasses.dataclass(frozen=True)
class AdaptiveLinearTolerance:
    """The relative tolerance each step of a march solves its linear system to, adapted step by step.

    The march starts every solve at the **base** tolerance its linear solver was built with (a
    preconditioner family's ``LinearSolveSettings.rtol``) and, after each accepted step, asks
    :meth:`next_tolerance` for the next one. Two rules, each switched off by setting it to ``None``:

    * **after a setback** -- a step whose line search cut it back (``alpha < 1``) or whose residual rose
      is followed by a step solved to ``after_setback``;
    * **in the tail** -- a full-length step (``alpha = 1``) whose residual fell by no more than
      ``tail_gate`` times the base tolerance was limited by its own linear solve, so the next one is
      solved to ``tail_scale * reduction**2`` (the superlinear choice of Eisenstat & Walker 1996), but
      never tighter than ``target_margin * target / residual`` -- a step need not solve past what
      reaching the stopping target asks of it -- nor than ``minimum``. A step limited by something else,
      such as the pseudo-time shift, reduces the residual by more than that gate allows, and keeps the
      base tolerance.

    Neither rule ever loosens a solve beyond the base tolerance, and a step that fits neither returns to
    it, so a march whose steps all reduce the residual slowly at full length is unchanged. The shift and
    the tolerance both vanish from the converged state, so the root and its adjoint are unaffected; only
    the path to the root changes.

    Attributes
    ----------
    after_setback : float or None
        The tolerance a step after a setback is solved to (default ``0.1``). ``None`` keeps the base.
        Measured on a coupled channel's Reynolds-continuation target march: ``0.1`` converged in 12
        steps, ``0.03`` in 24, so tighter is not monotonically better here.
    tail_gate : float or None
        A full-length step counts as limited by its linear solve when its residual reduction is at most
        this multiple of the base tolerance (default ``1.5``). ``None`` keeps the base in the tail.
    tail_scale : float
        The factor in ``tail_scale * reduction**2`` (default ``0.9``).
    target_margin : float
        The fraction of the stopping target a tail step aims below (default ``0.5``): its tolerance is
        never tighter than ``target_margin * target / residual``.
    minimum : float
        The tightest tolerance either rule asks for (default ``1e-10``).

    Examples
    --------
    Opt in on a coupled solve, with both rules at their defaults::

        from aquaflux.solve import AdaptiveLinearTolerance
        from aquaflux.turbulence import solve_coupled

        flow, k, omega = solve_coupled(
            coupled, flow0, k0, omega0, adaptive_tolerance=AdaptiveLinearTolerance()
        )
    """

    after_setback: float | None = 0.1
    tail_gate: float | None = 1.5
    tail_scale: float = 0.9
    target_margin: float = 0.5
    minimum: float = 1e-10

    def __post_init__(self) -> None:
        name = type(self).__name__
        if self.after_setback is not None and not 0.0 < self.after_setback < 1.0:
            raise ValueError(
                f"{name}.after_setback must lie in (0, 1), got {self.after_setback!r}."
            )
        if self.tail_gate is not None and not self.tail_gate > 0.0:
            raise ValueError(f"{name}.tail_gate must be positive, got {self.tail_gate!r}.")
        if not self.tail_scale > 0.0:
            raise ValueError(f"{name}.tail_scale must be positive, got {self.tail_scale!r}.")
        if not 0.0 < self.target_margin <= 1.0:
            raise ValueError(
                f"{name}.target_margin must lie in (0, 1], got {self.target_margin!r}."
            )
        if not 0.0 < self.minimum < 1.0:
            raise ValueError(f"{name}.minimum must lie in (0, 1), got {self.minimum!r}.")

    def next_tolerance(
        self,
        base: float,
        *,
        alpha: float,
        reduction: float,
        stopping_norm: float,
        target: float,
    ) -> float:
        """The relative tolerance the next step solves to, from how the step just taken went.

        Parameters
        ----------
        base : float
            The tolerance the march's linear solver was built with; never exceeded.
        alpha : float
            The step's line-search factor (for an inner loop, the minimum over its iterations).
        reduction : float
            The step's residual over the residual the march reported before it. A march that rebuilds its
            measure every step reads the two in different scales; a rise between them is still the
            setback this rule responds to.
        stopping_norm : float
            The residual the march's stopping test read at the state the step reached.
        target : float
            The residual the stopping test stops at, in the same measure as ``stopping_norm``.

        Returns
        -------
        float
            The next step's tolerance, in ``[min(minimum, base), base]``.
        """
        if not (math.isfinite(alpha) and math.isfinite(reduction)):
            return base
        if alpha < 1.0 or reduction > 1.0:
            return base if self.after_setback is None else min(base, self.after_setback)
        if self.tail_gate is not None and reduction <= self.tail_gate * base:
            floor = self.minimum
            if stopping_norm > 0.0 and math.isfinite(stopping_norm):
                floor = max(floor, self.target_margin * target / stopping_norm)
            return min(base, max(self.tail_scale * reduction**2, floor))
        return base
