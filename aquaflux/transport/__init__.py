"""Scalar transport by a converged flow: species concentration, temperature, passive tracers.

The equation a water or environmental reactor is ultimately asked about -- what a tracer does in a
contactor, where a reagent goes, how long fluid spends where -- solved on the flow the coupled
momentum--continuity block produces. It reuses that flow's own Rhie--Chow face flux, so the scalar
stays discretely conservative with continuity, and it is assembled from the same operators every
other transport equation uses.
"""

from __future__ import annotations

from .scalar import DIFFUSIVITY, ScalarTransport, effective_diffusivity

__all__ = [
    "DIFFUSIVITY",
    "ScalarTransport",
    "effective_diffusivity",
]
