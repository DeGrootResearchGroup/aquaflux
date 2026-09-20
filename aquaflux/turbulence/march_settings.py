"""The coupled march's settings, grouped into values by what each group configures.

A coupled march is configured by many settings that only mean something together. Written as loose
keywords they are copied into every builder's signature and every forwarding wrapper, and a setting
added for one builder has to be threaded by hand through the rest -- which drifts. Grouping each
set into one value removes that: there is nothing left to copy, and a field added to the value
reaches every builder that takes it.

**Every field defaults to** ``None``, **meaning "not set here".** An unset field is resolved by the
builder the value is handed to, so each default is written once, beside the code that applies it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aquaflux.solve import ShiftSettings

if TYPE_CHECKING:
    from .coupled import TurbulenceDamping

__all__ = ["CoupledShiftSettings", "merged_march_options"]


class CoupledShiftSettings(ShiftSettings):
    """How the pseudo-time shift diagonal is formed for the coupled flow-turbulence march.

    The flow's part -- ``basis`` and ``velocity_parts`` -- is :class:`~aquaflux.solve.ShiftSettings`,
    the same value a laminar flow march takes; this adds the one setting that belongs to the closure.
    Unset, the shift is the full operator diagonal on every row with the flow and closure damped alike.

    ⚠️ **The fields may be bound to a state, so this is not plain configuration.** A
    :class:`~aquaflux.turbulence.LiveViscosityVelocityParts` holds the assemblers it reads the viscosity
    from, and a :class:`~aquaflux.turbulence.ResidualTaperedDamping` holds a reference residual taken at
    one state. A value carrying either and shared across the points of a Reynolds continuation carries
    that state to every point -- exactly as the same objects passed as separate keywords would.

    Attributes
    ----------
    turbulence_damping : TurbulenceDamping, float or None
        A multiplier on the shift strength of the ``k`` and ``omega`` rows only. A plain number is a
        constant ratio. Unset, ``1``: the closure damped like the flow. It changes only the path -- the
        shift vanishes at the root.
    """

    turbulence_damping: TurbulenceDamping | float | None = None


def merged_march_options(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Two sets of march options as one, with a settings value merged field by field.

    A continuation merges its shared options with each point's own. For a loose keyword the point's
    value simply wins, as a dictionary merge gives. For a settings value that would be wrong: shared
    options carrying ``CoupledShiftSettings(basis=...)`` and a point returning
    ``CoupledShiftSettings(turbulence_damping=...)`` would lose the basis without a word, where the same two
    settings as separate keywords combine. So when both sides give the same key a value of the same
    settings type -- a :class:`CoupledShiftSettings`, :class:`~aquaflux.solve.LinearSolveSettings`,
    :class:`~aquaflux.solve.Convergence`, :class:`~aquaflux.solve.DualTimeLoop` or
    :class:`~aquaflux.solve.Globalization` -- the point's value
    keeps the fields it sets and takes the rest from the shared one (:func:`~aquaflux.solve.filled_from`).

    ⚠️ **A point cannot reset a shared field to its default through a value.** Leaving the field unset
    takes the shared setting. Write the default out where it is a number; where the default is ``None``
    itself -- ``CoupledShiftSettings.velocity_parts``, say -- keep that setting out of the shared options and set
    it per point instead.

    Parameters
    ----------
    base : dict
        The shared options.
    override : dict
        The options that take precedence, such as one continuation point's.

    Returns
    -------
    dict
        The merged options.
    """
    merged = {**base, **override}
    for name, value in override.items():
        prior = base.get(name)
        if type(prior) is type(value) and hasattr(value, "filled_from"):
            merged[name] = value.filled_from(prior)
    return merged
