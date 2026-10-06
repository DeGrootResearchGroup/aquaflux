"""What each boundary patch of a case is, stated once for every field.

A case file describes a boundary physically -- *this patch is an inlet at this velocity*, *that one is
a wall* -- rather than as one closure per solved field. The closures each equation needs (the flow's
velocity, pressure and mass flux, a turbulence closure's ``k`` and ``omega``) and the set of walls a
closure measures its wall distance from are all derived from that one statement, so they cannot be
written in a way that disagrees: a patch cannot be a wall to the flow and an inflow to the turbulence
closure.

Each kind builds the closures it stands for: :meth:`PatchCondition.flow_closure` the velocity, pressure
and mass-flux closure of the flow, and :meth:`PatchCondition.turbulence_closures` the ``k`` and
``omega`` ones. A wall's ``omega`` closure is a placeholder: the closure fixes ``omega`` in the cells
next to a wall instead, so the value its boundary faces would carry is never read.

A setting that only a turbulence closure reads -- an inlet's turbulence, a wall's ``k`` condition --
sits on the patch it belongs to, and the case's physics decides whether it may be there: a laminar
case refuses one, and a Reynolds-averaged case requires the inflow turbulence at every inlet. The same
holds for light: a wall's reflectance and a :class:`Lamp`'s photometry are read by a radiation case
alone, and a flow case refuses them, while a radiation case refuses what only the flow reads -- an
inlet, an outlet, a wall's velocity.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Literal

from aquaflux.boundary import BoundaryCondition, Dirichlet, ZeroGradient
from aquaflux.flow import FlowBoundary, MovingWall, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.turbulence import SSTModel

from .radiation import LampProfile, SurfaceSource

__all__ = [
    "FixedTurbulence",
    "Inlet",
    "InletTurbulence",
    "IntensityLength",
    "Lamp",
    "Outlet",
    "PatchCondition",
    "Wall",
]


def _refuse_non_finite(owner: str, name: str, values: tuple[float, ...]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{owner}.{name} must be finite, got {values!r}.")


def _refuse_a_bad_velocity(owner: str, label: str, velocity: tuple[float, ...]) -> None:
    """Refuse a patch velocity that is not two or three finite components.

    ``owner`` names the class (``Inlet``), ``label`` the same thing in prose (``an inlet``).
    """
    if len(velocity) not in (2, 3):
        raise ValueError(
            f"{label} velocity has two or three components, got {len(velocity)}: {velocity!r}."
        )
    _refuse_non_finite(owner, "velocity", velocity)


def _refuse_a_velocity_of_the_wrong_dimension(
    velocity: tuple[float, ...], dim: int, patch: str, what: str
) -> None:
    """Refuse a patch velocity whose component count is not the mesh's dimension."""
    if len(velocity) != dim:
        raise ValueError(
            f"boundaries.{patch}: the {what} velocity {velocity!r} has {len(velocity)} components, "
            f"but the mesh is {dim}-dimensional."
        )


@dataclasses.dataclass(frozen=True)
class InletTurbulence(abc.ABC):
    """The turbulence an inlet admits, for a Reynolds-averaged case.

    :class:`FixedTurbulence` prescribes ``k`` and ``omega`` outright; :class:`IntensityLength` gives
    them as a turbulence intensity and a length scale, relative to the inflow velocity and the model's
    own constants -- which is why both are handed in.
    """

    @abc.abstractmethod
    def inflow(self, velocity: tuple[float, ...], model: SSTModel) -> tuple[float, float]:
        """The ``(k, omega)`` admitted through an inlet at ``velocity``.

        Parameters
        ----------
        velocity : tuple of float
            The inlet's prescribed velocity.
        model : SSTModel
            The case's turbulence model, whose constants relate a length scale to ``omega``.

        Returns
        -------
        tuple of float
            The turbulent kinetic energy and the specific dissipation rate.
        """


@dataclasses.dataclass(frozen=True)
class FixedTurbulence(InletTurbulence):
    """Prescribed turbulent kinetic energy ``k`` and specific dissipation rate ``omega``.

    Attributes
    ----------
    k : float
        The turbulent kinetic energy admitted, ``>= 0``.
    omega : float
        The specific dissipation rate admitted, ``> 0``.

    Raises
    ------
    ValueError
        If ``k`` is negative or ``omega`` is not positive, or either is not finite.
    """

    k: float
    omega: float

    def __post_init__(self) -> None:
        _refuse_non_finite("FixedTurbulence", "k and omega", (self.k, self.omega))
        if self.k < 0:
            raise ValueError(f"FixedTurbulence.k must be >= 0, got {self.k!r}.")
        if self.omega <= 0:
            raise ValueError(f"FixedTurbulence.omega must be > 0, got {self.omega!r}.")

    def inflow(self, velocity: tuple[float, ...], model: SSTModel) -> tuple[float, float]:
        """``(k, omega)`` as given, whatever the velocity and the model."""
        del velocity, model
        return self.k, self.omega


@dataclasses.dataclass(frozen=True)
class IntensityLength(InletTurbulence):
    """Inflow turbulence given as an intensity and a length scale, the way it is usually known.

    ``k = 1.5 (I |U|)^2`` for an intensity ``I`` of the inflow speed ``|U|`` (the turbulent kinetic
    energy of velocity fluctuations of r.m.s. ``I |U|`` in each of three directions), and
    ``omega = sqrt(k) / (C_mu^(1/4) L)`` for a turbulence length scale ``L`` (the mixing-length
    relation), with ``C_mu`` the model's ``beta_star``. A 5% intensity and a length scale of a tenth of
    the inlet height is a common choice for a developed inflow.

    Attributes
    ----------
    intensity : float
        The r.m.s. velocity fluctuation as a fraction of the inflow speed, ``> 0`` -- ``0.05`` for 5%.
    length : float
        The turbulence length scale, ``> 0``, in the mesh's length units.

    Raises
    ------
    ValueError
        If either is not a positive, finite number.
    """

    intensity: float
    length: float

    def __post_init__(self) -> None:
        for name in ("intensity", "length"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"IntensityLength.{name} must be a positive, finite number, got {value!r}."
                )

    def inflow(self, velocity: tuple[float, ...], model: SSTModel) -> tuple[float, float]:
        """``k`` from the intensity of the inflow speed, ``omega`` from ``k`` and the length scale.

        Raises
        ------
        ValueError
            If the inflow velocity is zero, which carries no turbulence an intensity could be taken of.
        """
        speed = math.hypot(*velocity)
        if speed == 0.0:
            raise ValueError(
                "IntensityLength gives turbulence relative to the inflow speed, and this inlet's "
                "velocity is zero; state it as FixedTurbulence(k, omega) instead."
            )
        k = 1.5 * (self.intensity * speed) ** 2
        return k, math.sqrt(k) / (model.beta_star**0.25 * self.length)


@dataclasses.dataclass(frozen=True)
class PatchCondition(abc.ABC):
    """What one boundary patch is: an :class:`Inlet`, an :class:`Outlet`, a :class:`Wall` or a :class:`Lamp`.

    Each builds the closures it stands for, and answers the questions a case asks of its boundaries
    before anything is built, so that a case's physics and its mesh can refuse what does not fit them.
    """

    @abc.abstractmethod
    def flow_closure(self) -> FlowBoundary:
        """The flow's closure on this patch: its velocity, pressure and mass flux.

        Everything the flow knows about the patch is asked of this closure rather than restated here --
        whether it is a wall, and whether it fixes the pressure level.
        """

    @abc.abstractmethod
    def turbulence_closures(self, model: SSTModel) -> tuple[BoundaryCondition, BoundaryCondition]:
        """The ``(k, omega)`` closures on this patch, for a Reynolds-averaged case.

        Parameters
        ----------
        model : SSTModel
            The case's turbulence model.

        Returns
        -------
        tuple of BoundaryCondition
            The ``k`` closure and the ``omega`` closure.
        """

    def turbulence_settings(self) -> tuple[str, ...]:
        """The settings given here that only a turbulence closure reads; none by default."""
        return ()

    def missing_turbulence_settings(self) -> tuple[str, ...]:
        """The settings a turbulence closure needs here that this patch does not give; none by default."""
        return ()

    def flow_settings(self) -> tuple[str, ...]:
        """The settings given here that only the flow reads; none by default."""
        return ()

    def radiation_settings(self) -> tuple[str, ...]:
        """The settings given here that only a radiation case reads; none by default."""
        return ()

    def refuse_for_dimension(self, dim: int, patch: str) -> None:
        """Refuse this patch on a mesh of ``dim`` spatial dimensions if it cannot apply there.

        Parameters
        ----------
        dim : int
            The mesh's spatial dimension.
        patch : str
            The patch's name, for the message.

        Raises
        ------
        ValueError
            If the patch's settings do not fit a ``dim``-dimensional mesh.
        """
        del dim, patch


@dataclasses.dataclass(frozen=True)
class Inlet(PatchCondition):
    """A velocity inlet: the velocity is prescribed, and the pressure extrapolates from inside.

    Attributes
    ----------
    velocity : tuple of float
        The inflow velocity, one component per spatial dimension of the mesh.
    turbulence : InletTurbulence or None
        The turbulence admitted -- required by a Reynolds-averaged case and refused by a laminar one.

    Raises
    ------
    ValueError
        If the velocity does not have two or three finite components.
    """

    velocity: tuple[float, ...]
    turbulence: InletTurbulence | None = None

    def __post_init__(self) -> None:
        _refuse_a_bad_velocity("Inlet", "an inlet", self.velocity)
        if self.turbulence is not None and not isinstance(self.turbulence, InletTurbulence):
            raise TypeError(
                "Inlet.turbulence must be an inlet-turbulence value such as FixedTurbulence(k, omega), "
                f"got {self.turbulence!r}."
            )

    def flow_closure(self) -> VelocityInlet:
        """A :class:`~aquaflux.flow.VelocityInlet` at :attr:`velocity`."""
        return VelocityInlet(velocity=self.velocity)

    def turbulence_closures(self, model: SSTModel) -> tuple[Dirichlet, Dirichlet]:
        """The inflow ``k`` and ``omega``, each prescribed.

        Raises
        ------
        ValueError
            If the inlet gives no inflow turbulence -- which a Reynolds-averaged case refuses when it is
            constructed, so this is reached only by a caller that skipped that check.
        """
        if self.turbulence is None:
            raise ValueError(
                "this inlet gives no inflow turbulence, so it has no k or omega closure to build."
            )
        k, omega = self.turbulence.inflow(self.velocity, model)
        return Dirichlet(k), Dirichlet(omega)

    def turbulence_settings(self) -> tuple[str, ...]:
        """``("turbulence",)`` if the inflow turbulence is given."""
        return () if self.turbulence is None else ("turbulence",)

    def missing_turbulence_settings(self) -> tuple[str, ...]:
        """``("turbulence",)`` if it is not -- every inflow carries turbulence into a closure."""
        return ("turbulence",) if self.turbulence is None else ()

    def flow_settings(self) -> tuple[str, ...]:
        """``("velocity",)``: an inlet is a statement about the flow alone."""
        return ("velocity",)

    def refuse_for_dimension(self, dim: int, patch: str) -> None:
        """Refuse a velocity whose component count is not the mesh's dimension."""
        _refuse_a_velocity_of_the_wrong_dimension(self.velocity, dim, patch, "inlet")


@dataclasses.dataclass(frozen=True)
class Outlet(PatchCondition):
    """A pressure outlet: the pressure is prescribed, and the velocity extrapolates from inside.

    Attributes
    ----------
    pressure : float
        The pressure imposed on the patch -- the level every pressure in the solution is measured from.

    Raises
    ------
    ValueError
        If the pressure is not finite.
    """

    pressure: float

    def __post_init__(self) -> None:
        _refuse_non_finite("Outlet", "pressure", (self.pressure,))

    def flow_closure(self) -> PressureOutlet:
        """A :class:`~aquaflux.flow.PressureOutlet` at :attr:`pressure`."""
        return PressureOutlet(pressure=self.pressure)

    def turbulence_closures(self, model: SSTModel) -> tuple[ZeroGradient, ZeroGradient]:
        """``k`` and ``omega`` leave with the flow: a zero gradient for each."""
        del model
        return ZeroGradient(), ZeroGradient()

    def flow_settings(self) -> tuple[str, ...]:
        """``("pressure",)``: an outlet is a statement about the flow alone."""
        return ("pressure",)


@dataclasses.dataclass(frozen=True)
class Wall(PatchCondition):
    """A solid wall: no slip, no through-flow -- stationary, or moving in its own plane.

    A moving wall (the driven lid of a cavity) holds the fluid at its own velocity instead of at rest,
    and is a wall in every other respect: it passes no fluid, and in a Reynolds-averaged case it is
    where the closure measures its wall distance from and fixes ``omega`` near, exactly as a
    stationary one is.

    To light, a wall is black unless it is given a reflectance, and then reflects diffusely.

    Attributes
    ----------
    k : {"zero_gradient", "zero"} or None
        The condition on the turbulent kinetic energy at the wall: a zero normal gradient, or a zero
        value. A turbulence setting, so a laminar case refuses it; unset, a Reynolds-averaged case takes
        the zero gradient.
    velocity : tuple of float or None
        The wall's velocity, one component per spatial dimension; unset, the wall is at rest. Any normal
        component is ignored for continuity -- a wall, however it moves, passes no fluid. A flow
        setting, so a radiation case refuses it.
    reflectance : float or None
        The fraction of the light arriving that the wall reflects, diffusely, in ``[0, 1]``; unset, it
        reflects none. A radiation setting, so a flow case refuses it.
    geometry : SurfaceSource or None
        Where a reflecting wall's triangles come from -- the drawing
        (:class:`~aquaflux.case.CadSurface`, :class:`~aquaflux.case.StlSurface`) or, unset, the mesh's
        own patch. Only a reflecting wall has a surface for light to leave, so a black wall refuses
        one.

    Raises
    ------
    ValueError
        If a velocity is given and does not have two or three finite components, if the reflectance is
        outside ``[0, 1]``, or if a geometry is given to a wall that reflects nothing.
    """

    k: Literal["zero_gradient", "zero"] | None = None
    velocity: tuple[float, ...] | None = None
    reflectance: float | None = None
    geometry: SurfaceSource | None = None

    def __post_init__(self) -> None:
        if self.velocity is not None:
            _refuse_a_bad_velocity("Wall", "a wall", self.velocity)
        if self.reflectance is not None and not (
            math.isfinite(self.reflectance) and 0.0 <= self.reflectance <= 1.0
        ):
            raise ValueError(f"Wall.reflectance must lie in [0, 1], got {self.reflectance!r}.")
        if self.geometry is not None:
            if not isinstance(self.geometry, SurfaceSource):
                raise TypeError(
                    f"Wall.geometry must be a surface source such as StlSurface, got {self.geometry!r}."
                )
            if not self.reflects:
                raise ValueError(
                    "Wall.geometry is the surface a reflecting wall sends light back from, but this "
                    "wall reflects nothing, so nothing would read it. Give a reflectance, or remove it."
                )

    @property
    def reflects(self) -> bool:
        """Whether the wall sends any light back."""
        return bool(self.reflectance)

    def flow_closure(self) -> NoSlipWall | MovingWall:
        """A :class:`~aquaflux.flow.NoSlipWall`, or a :class:`~aquaflux.flow.MovingWall` at :attr:`velocity`."""
        return NoSlipWall() if self.velocity is None else MovingWall(velocity=self.velocity)

    def turbulence_closures(self, model: SSTModel) -> tuple[BoundaryCondition, ZeroGradient]:
        """``k`` by :attr:`k` (a zero gradient when unset), and the placeholder ``omega`` closure.

        ``omega`` is fixed in the cells next to the wall rather than at its faces, so its face closure
        is never read; a zero gradient stands in for it.
        """
        del model
        k = Dirichlet(0.0) if self.k == "zero" else ZeroGradient()
        return k, ZeroGradient()

    def turbulence_settings(self) -> tuple[str, ...]:
        """``("k",)`` if the wall's ``k`` condition is given."""
        return () if self.k is None else ("k",)

    def flow_settings(self) -> tuple[str, ...]:
        """``("velocity",)`` if the wall moves."""
        return () if self.velocity is None else ("velocity",)

    def radiation_settings(self) -> tuple[str, ...]:
        """The reflectance and the geometry, whichever is given."""
        return tuple(
            name for name in ("reflectance", "geometry") if getattr(self, name) is not None
        )

    def refuse_for_dimension(self, dim: int, patch: str) -> None:
        """Refuse a wall velocity whose component count is not the mesh's dimension."""
        if self.velocity is not None:
            _refuse_a_velocity_of_the_wrong_dimension(self.velocity, dim, patch, "wall")


@dataclasses.dataclass(frozen=True)
class Lamp(PatchCondition):
    """A wall that emits light: an ultraviolet lamp's window, or the quartz sleeve around one.

    A lamp emits its power from the patch's surface, spread evenly over its area, and distributed over
    direction by its profile. It is black to the light that arrives on it. To a flow it is a stationary
    wall, so a lamp patch is stated once for both; a flow case still refuses one, since nothing in it
    would read the photometry.

    Attributes
    ----------
    profile : LampProfile
        How the lamp distributes its light over direction: :class:`~aquaflux.case.LambertianProfile`,
        :class:`~aquaflux.case.CosinePowerProfile`, or a measured table,
        :class:`~aquaflux.case.IesProfile`.
    power : float or None
        The radiant power emitted, in W. Unset, the profile's own -- which only a photometry file
        stating its intensities in a radiant unit has.
    geometry : SurfaceSource or None
        Where the lamp's triangles come from -- the drawing (:class:`~aquaflux.case.CadSurface`,
        :class:`~aquaflux.case.StlSurface`) or, unset, the mesh's own patch.

    Raises
    ------
    ValueError
        If the power is not positive, or is unset for a profile that cannot state one.
    TypeError
        If the profile or the geometry is not a value of its family.
    """

    profile: LampProfile
    power: float | None = None
    geometry: SurfaceSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, LampProfile):
            raise TypeError(
                f"Lamp.profile must be a lamp profile such as LambertianProfile(), got {self.profile!r}."
            )
        if self.geometry is not None and not isinstance(self.geometry, SurfaceSource):
            raise TypeError(
                f"Lamp.geometry must be a surface source such as StlSurface, got {self.geometry!r}."
            )
        if self.power is not None and not (math.isfinite(self.power) and self.power > 0):
            raise ValueError(f"Lamp.power must be a positive, finite number, got {self.power!r}.")
        if self.power is None and not self.profile.can_state_power:
            raise ValueError(
                f"Lamp.power is unset, and a {type(self.profile).__name__} states no power of its own; "
                "give the lamp's radiant power in W."
            )

    def flow_closure(self) -> NoSlipWall:
        """A :class:`~aquaflux.flow.NoSlipWall`: to the flow a lamp is a stationary wall."""
        return Wall().flow_closure()

    def turbulence_closures(self, model: SSTModel) -> tuple[BoundaryCondition, ZeroGradient]:
        """A stationary wall's -- see :meth:`Wall.turbulence_closures`."""
        return Wall().turbulence_closures(model)

    def radiation_settings(self) -> tuple[str, ...]:
        """The profile, and the power and geometry where given: everything a lamp states is light."""
        return (
            "profile",
            *(name for name in ("power", "geometry") if getattr(self, name) is not None),
        )

    def refuse_for_dimension(self, dim: int, patch: str) -> None:
        """Refuse a lamp on a mesh that is not three-dimensional: light is gathered in three."""
        if dim != 3:
            raise ValueError(
                f"boundaries.{patch}: a lamp lights a three-dimensional domain, but the mesh is "
                f"{dim}-dimensional"
            )
