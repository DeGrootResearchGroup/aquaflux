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

import dataclasses
from typing import TYPE_CHECKING

import equinox as eqx

from aquaflux.solve import SettingsValue, filled_from

if TYPE_CHECKING:
    from aquaflux.solve import ShiftBasis, VelocityShiftParts

    from .coupled import TurbulenceDamping

__all__ = ["ForwardSolve", "ShiftSettings", "merged_march_options"]


class ShiftSettings(eqx.Module):
    """How the pseudo-time shift diagonal is formed, for every coupled builder.

    The three settings describe the *shift* -- how much each row is damped per unit of the shift
    strength ``beta`` -- and nothing about the preconditioner or the march's schedule, which is why
    they are a value of their own rather than fields of either. Unset, the shift is the full operator
    diagonal on every row with the flow and closure damped alike.

    ⚠️ **Two of the fields may be bound to a state, so this is not plain configuration.** A
    :class:`~aquaflux.turbulence.LiveViscosityVelocityParts` holds the assemblers it reads the viscosity
    from, and a :class:`~aquaflux.turbulence.ResidualTaperedDamping` holds a reference residual taken at
    one state. A value carrying either and shared across the points of a Reynolds continuation carries
    that state to every point -- exactly as the same objects passed as separate keywords would. Only
    ``basis`` is a plain setting.

    Attributes
    ----------
    basis : ShiftBasis or None
        How each block's shift diagonal is combined from its convective and dissipative parts. Unset,
        :class:`~aquaflux.solve.LocalCourantBasis` with its defaults: the full operator diagonal.
    velocity_parts : VelocityShiftParts or None
        Where the velocity shift's two diagonal buckets come from. Unset, the flow assembler's frozen
        momentum diagonal at the reference state.
    turbulence_damping : TurbulenceDamping, float or None
        A multiplier on the shift strength of the ``k`` and ``omega`` rows only. A plain number is a
        constant ratio. Unset, ``1``: the closure damped like the flow. It changes only the path -- the
        shift vanishes at the root.
    """

    basis: ShiftBasis | None = None
    velocity_parts: VelocityShiftParts | None = None
    turbulence_damping: TurbulenceDamping | float | None = None

    def filled_from(self, base: ShiftSettings) -> ShiftSettings:
        """These settings, with each field left unset taken from ``base`` (see :func:`~aquaflux.solve.filled_from`)."""
        return filled_from(self, base)


@dataclasses.dataclass(frozen=True)
class ForwardSolve(SettingsValue):
    """The shifted forward solve's Krylov regime, as one value.

    Each field is resolved against the chosen preconditioner family's own regime when unset: restart
    ``120`` for the block-diagonal family, ``10`` for a complete LU, ``15`` for a multigrid V-cycle or a
    field split, each at a relative tolerance of ``0.3``, and ``1e-2`` with restart ``120`` for the
    mass-flow-constrained march. A builder takes this value or a whole ``lineax`` solver in the same
    parameter, never both, because a solver replaces the regime -- and the stopping measure with it.

    ⚠️ **``rtol`` is measured in the march's own progress measure**, not the Euclidean norm, except on
    the mass-flow-constrained march, whose measure is Euclidean. The coupled residual's 2-norm is almost
    entirely ``omega``, so a Euclidean stop halts while the flow-dominated part of the step is still
    coarse. The same number is therefore not the same tightness under a different measure.

    Attributes
    ----------
    rtol : float or None
        The relative tolerance each shifted solve stops at, in the progress measure.
    restart : int or None
        The Arnoldi restart length.
    max_restarts : int or None
        The restart-cycle cap, in raw ``lineax`` restarts -- the only bound on a single running solve.
    """

    rtol: float | None = None
    restart: int | None = None
    max_restarts: int | None = None


def merged_march_options(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Two sets of march options as one, with a settings value merged field by field.

    A continuation merges its shared options with each point's own. For a loose keyword the point's
    value simply wins, as a dictionary merge gives. For a settings value that would be wrong: shared
    options carrying ``ShiftSettings(basis=...)`` and a point returning
    ``ShiftSettings(turbulence_damping=...)`` would lose the basis without a word, where the same two
    settings as separate keywords combine. So when both sides give the same key a value of the same
    settings type -- a :class:`ShiftSettings`, :class:`ForwardSolve`,
    :class:`~aquaflux.solve.DualTimeLoop` or :class:`~aquaflux.solve.Globalization` -- the point's value
    keeps the fields it sets and takes the rest from the shared one (:func:`~aquaflux.solve.filled_from`).

    ⚠️ **A point cannot reset a shared field to its default through a value.** Leaving the field unset
    takes the shared setting. Write the default out where it is a number; where the default is ``None``
    itself -- ``ShiftSettings.velocity_parts``, say -- keep that setting out of the shared options and set
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
