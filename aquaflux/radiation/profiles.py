"""How a source distributes its radiant power over direction.

A profile is a **normalized angular distribution of radiant intensity**: the fraction of a
source's total power that leaves per unit solid angle in a given direction, so that

    integral over the whole sphere of f(omega) d.omega  =  1

and a source of total power ``P`` has intensity ``P f(omega)`` watts per steradian. Keeping the
distribution normalized and the power separate is the convention the illumination-design tools
use, and it is what lets the same object describe a point source and a surface: the point
source's contribution to a fluence rate is ``P f / r^2``, and a facet's is its radiance times
the solid angle it subtends.

**Both quantities come from one distribution, but the gather needs the second one directly.**
A facet of area ``A`` and exitance ``M`` radiates ``P = M A``, so its radiance at an angle
``theta`` from its own normal is ``M f(cos theta) / cos theta``. Written that way the Lambertian
case is ``0/0`` at grazing incidence, and Lambertian is the default path and carries every
reflected ray in the model. Each profile therefore supplies the ratio itself, already reduced:
:meth:`Profile.radiance_per_exitance`. For a Lambertian emitter it is the constant ``1/pi``,
with nothing to cancel at run time.

**A profile is asked about a direction, not an angle.** Its two methods take the unit direction
from the source towards the receiver and the source facet's outward normal. Most distributions
depend on nothing but the angle between the two -- :class:`AxisymmetricProfile`, which reduces
the pair to its cosine and is written in terms of it -- but a measured luminaire does not: its
intensity varies around its axis too (:class:`~aquaflux.radiation.photometry.PhotometricProfile`).

The classes are ``equinox`` modules rather than a kind flag with a parameter bag, so a profile's
parameters stay differentiable leaves and the choice of profile resolves when the program is
traced -- the compiled code contains no branch on which kind is in use.
"""

from __future__ import annotations

import abc
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot

__all__ = ["AxisymmetricProfile", "CosinePower", "Isotropic", "Lambertian", "Profile"]

_FOUR_PI = 4.0 * jnp.pi


class Profile(eqx.Module):
    """A normalized angular distribution of radiant intensity.

    Subclasses supply two views of the same distribution, and the pair must agree:
    ``radiance_per_exitance(d, n) * dot(d, n) == intensity_fraction(d, n)`` wherever
    ``dot(d, n) > 0``. They are both defined rather than one deriving the other because the
    derivation divides by the cosine, and the division is exactly cancellable in the case that
    matters most.

    Attributes
    ----------
    dark_behind : bool
        Whether an areal facet with this distribution sends exactly nothing into the half-space
        behind it: ``radiance_per_exitance`` is exactly zero wherever ``dot(d, n) <= 0``, not
        merely small. A class-level declaration, ``False`` unless a subclass says otherwise,
        because what reads it skips work on the strength of it: a shadow mask for points in the
        volume casts no ray for a pair whose source faces away, since the gather multiplies
        whatever that ray would say by zero. Declared rather than tested, since a test on sample
        directions cannot prove a zero everywhere, and a subclass that is dark behind and does
        not say so only costs rays.
    """

    dark_behind: ClassVar[bool] = False

    @abc.abstractmethod
    def intensity_fraction(self, direction: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
        """Fraction of total power leaving per steradian towards ``direction``.

        Parameters
        ----------
        direction : jnp.ndarray, shape ``(..., 3)``
            Unit vector from the source towards the receiver.
        normal : jnp.ndarray, shape ``(..., 3)``
            The source facet's outward unit normal (zero on a point source). Broadcasts against
            ``direction``.

        Returns
        -------
        jnp.ndarray, shape ``(...)``
            ``f(omega)`` in inverse steradians, integrating to one over the whole sphere.
        """

    @abc.abstractmethod
    def radiance_per_exitance(self, direction: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
        """Radiance leaving a facet towards ``direction``, per unit of its exitance.

        That is ``f / cos theta``, with ``cos theta = dot(direction, normal)``; the arguments are
        those of :meth:`intensity_fraction`.

        Returns
        -------
        jnp.ndarray, shape ``(...)``
            ``L / M`` in inverse steradians, zero where the direction is behind the facet.
        """

    @abc.abstractmethod
    def mirrored(self, normal) -> Profile:
        """This distribution as its mirror image in a plane with unit normal ``normal`` emits it.

        The mirror image of a source sends along the reflection of each direction what the source
        sends along that direction, about the reflection of its own normal. So the profile
        returned, ``q``, satisfies ``q.intensity_fraction(R d, R n) == intensity_fraction(d, n)``
        for the reflection ``R`` in the plane, and the same for the radiance. A distribution of
        the angle from the normal alone is its own mirror image; one that also varies round the
        normal is not, because a reflection reverses the sense in which that angle increases.

        Parameters
        ----------
        normal : array_like, shape ``(3,)``
            Unit normal of the mirror's plane. Only its direction matters, and either sign of it
            gives the same image.

        Returns
        -------
        Profile
        """

    def refuse_normals(self, normals) -> str | None:
        """Why facets with these normals cannot carry this profile, or ``None`` if they can.

        Asked by :func:`~aquaflux.radiation.checks.check_profiles` of the areal facets that carry
        the profile. Nothing is refused unless a subclass measures an angle that some normals
        leave undefined.

        Parameters
        ----------
        normals : array_like, shape ``(n, 3)``
            Outward unit normals.
        """
        del normals
        return None


class AxisymmetricProfile(Profile):
    """A distribution that depends only on the angle from the facet normal.

    Written in terms of ``cos theta``: a subclass supplies :meth:`intensity_fraction_at` and
    :meth:`radiance_per_exitance_at`, and the direction-based methods reduce a direction to its
    cosine and delegate. The cosine forms are public because some consumers only ever hold the
    cosine -- the surface transfer freezes one per pair of facets when it is built.
    """

    @abc.abstractmethod
    def intensity_fraction_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """Fraction of total power leaving per steradian, at ``cos theta`` from the axis.

        Parameters
        ----------
        cos_theta : jnp.ndarray
            Cosine of the angle between the source's normal (or axis) and the direction to the
            receiver. Negative values are behind the source.

        Returns
        -------
        jnp.ndarray
            ``f(omega)`` in inverse steradians, integrating to one over the whole sphere.
        """

    @abc.abstractmethod
    def radiance_per_exitance_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """Radiance leaving a facet, per unit of its exitance — ``f(cos theta) / cos theta``.

        Returns
        -------
        jnp.ndarray
            ``L / M`` in inverse steradians, zero where the direction is behind the facet.
        """

    def intensity_fraction(self, direction: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
        """:meth:`intensity_fraction_at` the cosine of ``direction`` from ``normal``."""
        return self.intensity_fraction_at(dot(direction, normal))

    def radiance_per_exitance(self, direction: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
        """:meth:`radiance_per_exitance_at` the cosine of ``direction`` from ``normal``."""
        return self.radiance_per_exitance_at(dot(direction, normal))

    def mirrored(self, normal) -> AxisymmetricProfile:
        """Itself: a reflection keeps the angle between a direction and the normal."""
        del normal
        return self


class Isotropic(AxisymmetricProfile):
    """Equal intensity in every direction — ``f = 1 / (4 pi)``.

    This is the profile of a **point source**, and the only one a point source may carry: a
    zero-area facet has no surface normal, so no direction-dependent distribution has anything
    to measure its angle against. Asking it for a radiance is therefore a category error rather
    than a missing feature, and it says so instead of returning a number.
    """

    def intensity_fraction_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """Uniform over the sphere, and independent of direction."""
        return jnp.full_like(jnp.asarray(cos_theta, dtype=float), 1.0 / _FOUR_PI)

    def radiance_per_exitance_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """Not defined: an isotropic source is a point source and has no emitting surface."""
        msg = (
            "Isotropic is a point-source profile and has no radiance per unit exitance: a "
            "zero-area facet has no surface for an exitance to be defined on. Give areal "
            "facets Lambertian or CosinePower, and keep Isotropic for facets carrying power."
        )
        raise NotImplementedError(msg)


class Lambertian(AxisymmetricProfile):
    """The diffuse emitter: constant radiance in every visible direction.

    ``f = max(cos theta, 0) / pi``, which is the cosine law — intensity falls off as the
    cosine, exactly compensating the foreshortening of the emitting area, so the surface looks
    equally bright from every angle. It is the default, it is what every reflected ray leaves
    by, and its radiance per unit exitance is the constant ``1 / pi``.
    """

    dark_behind: ClassVar[bool] = True

    def intensity_fraction_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """``max(cos theta, 0) / pi``."""
        return jnp.maximum(jnp.asarray(cos_theta, dtype=float), 0.0) / jnp.pi

    def radiance_per_exitance_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """``1 / pi`` in front of the facet and zero behind it — the cosine cancels exactly."""
        forward = jnp.asarray(cos_theta, dtype=float) > 0.0
        return jnp.where(forward, 1.0 / jnp.pi, 0.0)


class CosinePower(AxisymmetricProfile):
    """A narrowed beam: ``f = (n + 1) max(cos theta, 0)^n / (2 pi)``.

    The generalized-Lambertian form used throughout illumination optics, where the exponent is
    set from the half-intensity angle by ``n = -ln 2 / ln(cos theta_half)``. **At ``n = 1`` it
    reduces exactly to Lambertian**, ``cos theta / pi``, which is the reduction to check against
    when the factor of ``2 pi`` looks wrong: the normalization is over a hemisphere, so the
    constant is ``(n + 1) / (2 pi)`` and not ``(n + 1) / pi``.

    Attributes
    ----------
    exponent : jnp.ndarray
        The exponent ``n``, a differentiable leaf. Must be at least one: below that the
        radiance is unbounded at grazing incidence, which is not a surface emitter.
    """

    dark_behind: ClassVar[bool] = True

    exponent: jnp.ndarray

    def __init__(self, exponent):
        self.exponent = jnp.asarray(exponent, dtype=float)

    def __check_init__(self):
        # Checked here rather than in the gather: a concrete exponent is known when the profile
        # is built, and the alternative is discovering it as an infinity in a fluence field.
        if not isinstance(self.exponent, jnp.ndarray) or self.exponent.shape != ():
            return
        try:
            value = float(self.exponent)
        except TypeError:  # a tracer under jit -- nothing to check against
            return
        if value < 1.0:
            msg = (
                f"CosinePower exponent must be at least 1; got {value}. Below one the radiance "
                "grows without bound at grazing incidence, which no surface emitter does. An "
                "exponent of exactly 1 is Lambertian."
            )
            raise ValueError(msg)

    def intensity_fraction_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """``(n + 1) max(cos theta, 0)^n / (2 pi)``."""
        forward = jnp.maximum(jnp.asarray(cos_theta, dtype=float), 0.0)
        return (self.exponent + 1.0) * forward**self.exponent / (2.0 * jnp.pi)

    def radiance_per_exitance_at(self, cos_theta: jnp.ndarray) -> jnp.ndarray:
        """``(n + 1) max(cos theta, 0)^(n - 1) / (2 pi)`` — one power of the cosine cancelled."""
        cos_theta = jnp.asarray(cos_theta, dtype=float)
        forward = jnp.maximum(cos_theta, 0.0)
        # The cancelled power is taken analytically rather than by dividing, so that n = 1 is
        # the exact constant 1/pi instead of a zero over a zero at grazing incidence.
        magnitude = (self.exponent + 1.0) * forward ** (self.exponent - 1.0) / (2.0 * jnp.pi)
        return jnp.where(cos_theta > 0.0, magnitude, 0.0)
