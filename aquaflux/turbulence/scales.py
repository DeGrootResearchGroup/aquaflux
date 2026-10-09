"""Reference magnitudes of ``k`` and ``omega``, derived from the flow's speed and the domain's size.

A numerical setting scaled by the size of the field it acts on -- the softening of a slope limiter on
the turbulence advection -- needs a magnitude for ``k`` and for ``omega`` before any turbulence has
been computed. Neither is the inlet value. The turbulence a flow develops is set by its shear, not by
what enters it: an inlet carrying 5 % intensity feeds a separated shear layer that reaches ten times its
``k``, and a periodic channel has no inlet at all. So both are taken from the flow's own velocity
scale ``U`` and length scale ``h``, the two numbers every flow problem states:

* ``k_ref = 1.5 (I U)^2``, the turbulent kinetic energy at a reference intensity ``I`` of 10 %. In
  developed wall-bounded and free shear flows the peak ``k`` lies between about ``0.01 U^2`` (a plane
  channel, ``U`` its bulk velocity) and ``0.05 U^2`` (a mixing layer or a separated shear layer,
  ``U`` the velocity difference across it); ``1.5 (0.1 U)^2 = 0.015 U^2`` sits inside that range, at
  its lower end, where a scale that errs makes a limiter stricter rather than looser.
* ``omega_ref = sqrt(k_ref) / (beta_star^(1/4) l)``, the specific dissipation of eddies of size
  ``l = 0.09 h``, the outer mixing length of a channel of hydraulic length ``h`` -- the same length
  the hybrid initial condition sizes a channel's turbulence from, so the two describe the same flow.

``omega_ref`` is deliberately the **outer-flow** level and not the range of the field. ``omega``
grows like ``1 / y^2`` towards a wall, so its range over a domain is set by the first cell's height
and changes with the mesh; a scale built from it would make the limiter's behaviour mesh-dependent,
which is what a scaled softening exists to prevent, and would switch limiting off everywhere but the
near-wall cells. With the outer level, the softening acts where ``omega`` is transported -- the core
and the shear layers -- and near a wall, where the field and its variations are far above
``omega_ref``, the limiter acts as the unsoftened one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .boundary import inlet_k, inlet_omega

if TYPE_CHECKING:
    from .sst import SSTModel

__all__ = [
    "OUTER_MIXING_LENGTH_FACTOR",
    "REFERENCE_INTENSITY",
    "TurbulenceScales",
    "turbulence_scales",
]

#: The turbulence intensity ``k_ref`` is formed at (see the module docstring).
REFERENCE_INTENSITY = 0.1

#: The outer mixing length as a fraction of the hydraulic length, ``l = 0.09 h``: the eddy viscosity
#: a developed channel carries in its core, ``~0.09 u_tau h``.
OUTER_MIXING_LENGTH_FACTOR = 0.09


class TurbulenceScales(NamedTuple):
    """The reference magnitudes of the two turbulence fields.

    Attributes
    ----------
    k : float
        Of the turbulent kinetic energy, ``m^2/s^2``.
    omega : float
        Of the specific dissipation rate, ``1/s``.
    """

    k: float
    omega: float


def turbulence_scales(
    velocity_scale: float,
    hydraulic_length: float,
    model: SSTModel,
    *,
    intensity: float = REFERENCE_INTENSITY,
    length_factor: float = OUTER_MIXING_LENGTH_FACTOR,
) -> TurbulenceScales:
    """The reference ``k`` and ``omega`` of a flow of speed ``velocity_scale`` and size ``hydraulic_length``.

    Parameters
    ----------
    velocity_scale : float
        The flow's speed (:func:`~aquaflux.flow.reference_speed`), ``m/s``.
    hydraulic_length : float
        The domain volume over its wall area (:func:`~aquaflux.flow.wetted_length`), ``m``.
    model : SSTModel
        The closure constants (reads ``beta_star``).
    intensity : float
        The reference turbulence intensity ``k`` is formed at.
    length_factor : float
        The mixing length as a fraction of ``hydraulic_length``.

    Returns
    -------
    TurbulenceScales
        ``k = 1.5 (intensity velocity_scale)^2`` and
        ``omega = sqrt(k) / (beta_star^(1/4) length_factor hydraulic_length)``.

    Raises
    ------
    ValueError
        If either scale is not positive -- a flow with no speed, or a domain with no wall to take a
        length from.
    """
    if not velocity_scale > 0.0:
        raise ValueError(
            "the turbulence reference scales are taken from the flow's speed, and this flow has "
            f"none (velocity scale {velocity_scale!r})."
        )
    if not hydraulic_length > 0.0:
        raise ValueError(
            "the omega reference scale is taken from the domain's hydraulic length (its volume over "
            f"its wall area), and this domain has no wall to take one from ({hydraulic_length!r})."
        )
    k = float(inlet_k(velocity_scale, intensity))
    omega = float(inlet_omega(k, length_factor * hydraulic_length, model))
    return TurbulenceScales(k=k, omega=omega)
