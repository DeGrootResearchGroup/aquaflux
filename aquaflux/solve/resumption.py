"""What a march resumed from an interrupted one needs to know, to continue it rather than start over.

A march handed a state it did not start from has no memory of how it got there. Three of its quantities
are measured *relative to the history it lacks*, and a resume that leaves them to be re-measured at the
state it resumes from does not continue the interrupted march:

* the **stopping scale** -- the stopping target is ``atol + rtol * reference`` with the reference taken
  at the march's first state, so re-taking it at a developed state moves the bar the march stops at;
* the **damping anchor** -- the switched-evolution shift ``max(floor, beta0 (|R| / |R0|)^p)`` is judged
  against the residual at the start of the *segment* the march is in (a refresh re-bases it), so
  re-taking it at the resumed state restarts the ramp at its opening strength;
* the **shift of a step control** -- a Courant ramp walks the shift down step by step from its first
  value, so a control that starts afresh retraces the ramp the interrupted march had already walked.

:class:`Resumption` carries the three, and each consumer takes what it has a place for.
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["Resumption"]


def _positive_finite(name: str, value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"Resumption.{name} must be a positive finite number, got {value!r}.")


@dataclasses.dataclass(frozen=True)
class Resumption:
    """The history of an interrupted march, for the march resuming it.

    Every value is a measurement the interrupted march made, in the residual measure it was steered
    by; resuming under a different problem or measure makes them meaningless, so a caller supplies them
    only when the problem and the measure are unchanged.

    Attributes
    ----------
    reference_residual : float
        The residual norm the march measured at its **first** state. It is the scale of the stopping
        target throughout, and the damping anchor of the first segment unless ``damping_reference``
        says otherwise.
    damping_reference : float or None
        The residual norm the segment the march was interrupted in was anchored at -- its first state's
        under a preconditioner refresh, so it differs from ``reference_residual`` once the march has
        refreshed. ``None`` anchors the first segment at ``reference_residual``.
    shift : float or None
        The shift strength the interrupted march's last step ran at, which a step control resumes from
        instead of its opening value. ``None`` starts the control afresh. A march with no step control
        has no shift to resume, and this is not used.

    Raises
    ------
    ValueError
        If a value given is not a positive finite number.
    """

    reference_residual: float
    damping_reference: float | None = None
    shift: float | None = None

    def __post_init__(self) -> None:
        _positive_finite("reference_residual", self.reference_residual)
        _positive_finite("damping_reference", self.damping_reference)
        _positive_finite("shift", self.shift)

    @property
    def anchor(self) -> float:
        """The residual norm the first segment's damping is anchored at."""
        return self.reference_residual if self.damping_reference is None else self.damping_reference
