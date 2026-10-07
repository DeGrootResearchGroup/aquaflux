"""Lamps lighting a scene of reflecting surfaces, with the light gathered where it is wanted.

A lamp's *emitted* light and *reflected* light leave the facet-to-facet transfer in different ways,
and the difference decides the assembly. Reflected light leaves every surface Lambertian, whatever
lit it, so reflection is exactly what :func:`~aquaflux.radiation.model.build_radiation_model` is
for. A lamp's own light need not be: a measured luminaire is a table over two angles
(:class:`~aquaflux.radiation.photometry.PhotometricProfile`), which the transfer cannot carry,
because it freezes one direction per pair of facets. So the lamps' **emission is kept out of the
transfer, and their surfaces are kept in it**:

- the lamps' direct light on every facet that exchanges light -- each reflecting facet and each
  lamp facet -- is gathered with their own profiles, averaged over points spread evenly across the
  facet so a shadow falling partly across it is not decided at one point, and handed to the surface
  solve as an irradiance arriving from outside it;
- the solve then carries only reflected, Lambertian light among the reflectors and the lamps
  together, and gives each facet its radiosity: what it reflects, by its own
  ``diffuse_reflectance``;
- the light anywhere is the lamps' direct light, gathered with their profiles, plus the reflected
  light, gathered from every facet as a Lambertian source of its radiosity.

A lamp is therefore a surface like any other to the light arriving on it: it reflects its
``diffuse_reflectance`` of it diffusely and absorbs the rest, and its triangles stand in the way of
reflected light as a wall's do. With no reflectance it is black, and still casts its shadow. A point
source has no surface, so it neither reflects nor shadows.

:class:`Scene` holds the lamps, the reflecting surfaces, whatever stands in the way and the medium
between them, and the points the light is wanted at -- a set of points in the medium
(:class:`VolumeReceivers`), and any number of named sets of points on surfaces, each facing the way
the surface does (:class:`SurfaceReceivers`). :func:`solve_scene` returns the fluence rate at the
first and the irradiance at the others, split into the direct and the reflected parts
(:class:`SceneSolution`), together with what the medium and the lamps absorb.

A point on a surface that exchanges light lies on one of that surface's facets. That facet's own
light needs no exclusion -- a facet lights nothing in its own plane, a point a rounding in front of
it is behind its receiving half-space and a point a rounding behind it is behind the facet -- but its
triangle does, from any test of what shadows the point: a ray from another facet ends in it, and
would read it as a blocker. Each set of points on such a surface therefore names the surface's body,
and the facet of that body nearest each point is left out of the shadow test at the point's end of
every ray; the points the lamps' light is averaged over on a lamp facet leave that facet out the same
way.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.checks import check_profiles
from aquaflux.radiation.coarsen import _point_triangle_distance
from aquaflux.radiation.gather import direct_irradiance, streamed_fluence_rate, summed_fluence_rate
from aquaflux.radiation.model import RadiationSettings, build_radiation_model, surface_irradiance
from aquaflux.radiation.profiles import Lambertian
from aquaflux.radiation.self_occlusion import NoOcclusion, SilhouetteOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import build_visibility
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = [
    "DEFAULT_LAMP_SAMPLES",
    "Scene",
    "SceneSolution",
    "SurfaceReceivers",
    "VolumeReceivers",
    "solve_scene",
    "subtriangle_centroids",
]

#: Points along each edge of a reflecting facet at which the lamps' light on it is averaged: the
#: facet is cut into this many squared similar triangles and each is sampled at its centroid.
DEFAULT_LAMP_SAMPLES = 4

#: Facets of a reflecting body searched for the ones a surface point lies on, nearest first by
#: centroid; the exact distance then picks among them.
_OWN_FACET_CANDIDATES = 32

#: How much farther than the nearest a facet may be and still hold the point, as a share of the
#: body's extent: a rounding of the point's coordinates.
_TIE = 1e-12

#: How far in front of its own facet a point the lamps' light is averaged over on a lamp facet is
#: moved, as a share of the scene's size. In the facet's plane the point can see its own facet as a
#: whole hemisphere, wherever a rounding leaves the facet's corners undecidably in the point's own
#: plane -- the kernels' convention for a point inside a facet -- and read a lamp as lighting
#: itself. A hair in front, the facet lies decidably behind the point's receiving plane and is
#: clipped away whatever the lamp's profile, and the point stays on the fluid's side of the lamp,
#: where a body holding the fluid does not refuse it. Far above a rounding of the coordinates, far
#: below anything geometric.
_IN_FRONT_OF_OWN_FACET = 1e-12


@dataclasses.dataclass(frozen=True)
class VolumeReceivers:
    """Points in the medium where the fluence rate is wanted.

    Attributes
    ----------
    points : np.ndarray, shape ``(n, 3)``
        Where -- cell centres, usually.
    volumes : np.ndarray, shape ``(n,)``, or None
        The volume each point stands for, so the field can be integrated over the medium; unset,
        no volume integral is formed.

    Raises
    ------
    ValueError
        If the arrays are not of those shapes.
    """

    points: np.ndarray
    volumes: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        _refuse_shape("VolumeReceivers.points", points, (None, 3))
        object.__setattr__(self, "points", points)
        if self.volumes is not None:
            volumes = np.asarray(self.volumes, dtype=float)
            _refuse_shape("VolumeReceivers.volumes", volumes, (len(points),))
            object.__setattr__(self, "volumes", volumes)


@dataclasses.dataclass(frozen=True)
class SurfaceReceivers:
    """Points on a surface where the irradiance is wanted, each facing the way the surface does.

    Attributes
    ----------
    points : np.ndarray, shape ``(n, 3)``
        Where -- the centres of a boundary patch's faces, usually.
    normals : np.ndarray, shape ``(n, 3)``
        Unit normals pointing the way the light arrives from: into the medium.
    areas : np.ndarray, shape ``(n,)``, or None
        The area each point stands for, so the irradiance can be integrated into a power; unset,
        no power is formed.
    reflectance : float
        The surface's diffuse reflectance, so that what it absorbs is ``1 - reflectance`` of what
        arrives. Read for that and nothing else: whether the surface sends light back out is decided
        by the facets' own ``diffuse_reflectance``.
    reflector : str or None
        The body the points lie on, if they lie on one that exchanges light: a body of
        :attr:`Scene.reflectors` or of :attr:`Scene.lamps`. The facet of that body nearest each point
        is then left out of the test of what shadows it, in which a ray from another facet would
        otherwise end in that facet and be read as blocked.

    Raises
    ------
    ValueError
        If the arrays are not of those shapes, or the reflectance is outside ``[0, 1]``.
    """

    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray | None = None
    reflectance: float = 0.0
    reflector: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        _refuse_shape("SurfaceReceivers.points", points, (None, 3))
        _refuse_shape("SurfaceReceivers.normals", normals, points.shape)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "normals", normals)
        if self.areas is not None:
            areas = np.asarray(self.areas, dtype=float)
            _refuse_shape("SurfaceReceivers.areas", areas, (len(points),))
            object.__setattr__(self, "areas", areas)
        if not 0.0 <= self.reflectance <= 1.0:
            raise ValueError(
                f"SurfaceReceivers.reflectance must lie in [0, 1], got {self.reflectance!r}."
            )


@dataclasses.dataclass(frozen=True)
class Scene:
    """Lamps, the surfaces they light, what stands in the way, and where the light is wanted.

    Attributes
    ----------
    lamps : Surfaces
        What emits. Any angular profile; its emission is gathered directly and never carried by the
        transfer. Its areal facets take part in the transfer as surfaces all the same: each reflects
        its ``diffuse_reflectance`` of the light arriving on it, diffusely, and absorbs the rest, and
        each stands in the way of reflected light. With no reflectance a lamp is black.
    reflectors : Surfaces or None
        The surfaces that reflect, diffusely, by each facet's ``diffuse_reflectance``, and emit
        nothing of their own: an emitting surface is a lamp. Unset, only the lamps can reflect.
    occluders : tuple of aquaflux.solids.Body
        Bodies that shadow the light from every source, lamps and reflectors alike.
    absorption : Absorption or None
        The medium between them; unset, it absorbs nothing.
    volume : VolumeReceivers or None
        Points in the medium where the fluence rate is wanted.
    surfaces : mapping of {str: SurfaceReceivers}
        Named sets of points on surfaces where the irradiance is wanted.
    lamp_samples : int
        Points along each edge of a reflecting facet at which the lamps' light on it is averaged;
        the facet is sampled at ``lamp_samples**2`` points.
    settings : RadiationSettings
        Build-time choices: how the surfaces shadow themselves and the points they light, how the
        bodies' shadows are worked out, and how many pairs a pass may form. Read for every gather and
        for the surface solve alike, so the two cannot be built differently.

    Raises
    ------
    ValueError
        If a reflector emits; if a lamp or a reflector reflects specularly, which the scene does not
        carry; if the lamps and the reflectors share a body name, which would leave a set of surface
        points unable to say which it lies on; if a set of surface points names a body neither has;
        or if ``lamp_samples`` is less than one.
    """

    lamps: Surfaces
    reflectors: Surfaces | None = None
    occluders: tuple = ()
    absorption: Absorption | None = None
    volume: VolumeReceivers | None = None
    surfaces: Mapping[str, SurfaceReceivers] = dataclasses.field(
        default_factory=lambda: types.MappingProxyType({})
    )
    lamp_samples: int = DEFAULT_LAMP_SAMPLES
    settings: RadiationSettings = dataclasses.field(default_factory=RadiationSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "occluders", tuple(self.occluders))
        object.__setattr__(self, "surfaces", types.MappingProxyType(dict(self.surfaces)))
        if self.lamp_samples < 1:
            raise ValueError(f"Scene.lamp_samples must be >= 1, got {self.lamp_samples!r}.")
        if self.reflectors is not None:
            if np.any(np.asarray(self.reflectors.emission) != 0.0) or np.any(
                np.asarray(self.reflectors.power) != 0.0
            ):
                raise ValueError(
                    "Scene.reflectors emit light of their own; an emitting surface belongs in "
                    "Scene.lamps, which are gathered with their own profiles."
                )
        for name, surfaces in (("lamps", self.lamps), ("reflectors", self.reflectors)):
            if surfaces is not None and np.any(np.asarray(surfaces.specular_reflectance) != 0.0):
                raise ValueError(
                    f"Scene.{name} reflect specularly; a scene carries diffuse reflection only, so "
                    "the light a mirror would send on would be lost without a word."
                )
        reflecting = () if self.reflectors is None else self.reflectors.solid_names
        shared = sorted(set(reflecting) & set(self.lamps.solid_names))
        if shared:
            raise ValueError(
                f"Scene: the lamps and the reflectors both have bodies named {shared}; give them "
                "distinct solid_names, which is how a set of surface points says what it lies on."
            )
        names = (*reflecting, *self.lamps.solid_names)
        unknown = sorted(
            f"{name!r} names {receivers.reflector!r}"
            for name, receivers in self.surfaces.items()
            if receivers.reflector is not None and receivers.reflector not in names
        )
        if unknown:
            raise ValueError(
                f"Scene.surfaces: {', '.join(unknown)}, which is not a body of the lamps or the "
                f"reflectors; their bodies are {list(names)}."
            )


@dataclasses.dataclass(frozen=True)
class SceneSolution:
    """The light in a :class:`Scene`, direct and reflected, and what each part of it absorbs.

    Every field array is in the order of the points it was gathered at. A reflected part is ``None``
    when nothing reflects.

    Attributes
    ----------
    lamp_power : float
        The power the lamps emit, in W.
    fluence_rate_direct, fluence_rate_reflected : np.ndarray or None, shape ``(n_volume,)``
        Fluence rate at the volume points from the lamps directly, and after at least one
        reflection, in W/m²; ``None`` without volume points.
    irradiance_direct, irradiance_reflected : mapping of {str: np.ndarray}
        Irradiance at each named set of surface points, likewise.
    radiosity : np.ndarray or None, shape ``(n_reflector_facets,)``
        What each reflecting facet sends out, in W/m²; ``None`` without reflectors.
    reflector_irradiance : np.ndarray or None, shape ``(n_reflector_facets,)``
        What lands on each reflecting facet, in W/m²; ``None`` without reflectors.
    lamp_irradiance : np.ndarray or None, shape ``(n_lamp_facets,)``
        What lands on each lamp facet from the other lamps and from everything that reflects, in
        W/m²; zero on a point source, which has no surface to land on. ``None`` when nothing
        exchanges light: no reflectors, and no lamp that reflects.
    lamp_absorbed_power : float or None
        What the lamps absorb of :attr:`lamp_irradiance`, ``Σ (1 - ρ) E A`` over their facets, in W;
        ``None`` with it.
    cycles : int or None
        The surface solve's restart cycles; ``None`` when nothing exchanges light.
    medium_absorbed_power : float or None
        What the medium absorbs, ``∫ a G dV``, in W; ``None`` without volume points or volumes.
    """

    lamp_power: float
    fluence_rate_direct: np.ndarray | None
    fluence_rate_reflected: np.ndarray | None
    irradiance_direct: Mapping[str, np.ndarray]
    irradiance_reflected: Mapping[str, np.ndarray] | None
    radiosity: np.ndarray | None
    reflector_irradiance: np.ndarray | None
    lamp_irradiance: np.ndarray | None
    lamp_absorbed_power: float | None
    cycles: int | None
    medium_absorbed_power: float | None

    @property
    def fluence_rate(self) -> np.ndarray | None:
        """The fluence rate at the volume points, direct and reflected together, in W/m²."""
        if self.fluence_rate_direct is None or self.fluence_rate_reflected is None:
            return self.fluence_rate_direct
        return self.fluence_rate_direct + self.fluence_rate_reflected

    def irradiance(self, name: str) -> np.ndarray:
        """The irradiance at one named set of surface points, direct and reflected together, in W/m²."""
        if self.irradiance_reflected is None:
            return self.irradiance_direct[name]
        return self.irradiance_direct[name] + self.irradiance_reflected[name]


def subtriangle_centroids(vertices, per_side: int) -> np.ndarray:
    """The centroids of the ``per_side**2`` similar triangles each triangle divides into.

    Cutting each edge into ``per_side`` equal parts and joining the cuts parallel to the edges divides
    a triangle into ``per_side**2`` triangles of equal area, so the mean of a quantity over their
    centroids is an equal-weight average over the whole triangle -- one that samples a shadow edge
    crossing it at as many places as it has rows, where one centroid would decide the whole facet.

    Parameters
    ----------
    vertices : array_like, shape ``(n, 3, 3)``
    per_side : int
        Divisions of each edge, ``>= 1``; one gives the centroid.

    Returns
    -------
    np.ndarray, shape ``(n, per_side**2, 3)``
    """
    vertices = np.asarray(vertices, dtype=float)
    if per_side < 1:
        raise ValueError(f"per_side must be >= 1, got {per_side!r}.")
    origin = vertices[:, 0]
    first, second = vertices[:, 1] - origin, vertices[:, 2] - origin
    weights = []
    for i in range(per_side):
        for j in range(per_side - i):
            # The upright triangle with its corner at (i, j), and the inverted one beside it.
            weights.append(((i + 1 / 3) / per_side, (j + 1 / 3) / per_side))
            if i + j < per_side - 1:
                weights.append(((i + 2 / 3) / per_side, (j + 2 / 3) / per_side))
    u, v = np.asarray(weights).T
    return (
        origin[:, None, :]
        + u[None, :, None] * first[:, None, :]
        + v[None, :, None] * second[:, None, :]
    )


def solve_scene(
    scene: Scene, *, solver=None, report: Callable[[str], None] | None = None
) -> SceneSolution:
    """Light a scene: the lamps directly, then whatever the reflectors and the lamps send back.

    Parameters
    ----------
    scene : Scene
    solver : lineax.AbstractLinearSolver, optional
        How the surface solve is solved; unset, :func:`~aquaflux.radiation.model.radiosity`'s default.
    report : callable, optional
        Called with a line of text as each stage starts, for a log.

    Returns
    -------
    SceneSolution

    Raises
    ------
    ValueError
        If a lamp's profile does not suit its facets (see
        :func:`~aquaflux.radiation.checks.check_profiles`), or a point lies inside a body.
    """
    say = report or (lambda line: None)
    check_profiles(scene.lamps)
    lamps = scene.lamps
    lamp_power = float(
        np.sum(np.asarray(lamps.emission) * np.asarray(lamps.area))
        + np.sum(np.asarray(lamps.power))
    )
    say(f"lamps: {lamps.n_facets} facets, {lamp_power:.6g} W")

    exchange = _Exchange.of(scene)
    bounced, exchanged = None, None
    if exchange is not None:
        exchanged = _solve_exchange(scene, exchange, solver, say)
        bounced = exchange.surfaces.with_optics(
            emission=jnp.asarray(exchanged.radiosity), profiles=(Lambertian(),)
        )

    direct_g, reflected_g, medium = None, None, None
    if scene.volume is not None:
        points = scene.volume.points
        say(f"fluence rate at {len(points)} points: direct")
        direct_g = _fluence(scene, lamps, points)
        if bounced is not None:
            say(f"fluence rate at {len(points)} points: reflected")
            reflected_g = _fluence(scene, bounced, points)
        if scene.volume.volumes is not None:
            total = direct_g if reflected_g is None else direct_g + reflected_g
            coefficient = (
                np.zeros(len(points))
                if scene.absorption is None
                else np.asarray(scene.absorption.sample(jnp.asarray(points)))
            )
            medium = float(np.sum(coefficient * total * scene.volume.volumes))

    direct_e, reflected_e = {}, None if bounced is None else {}
    for name, receivers in scene.surfaces.items():
        say(f"irradiance on {name!r}, {len(receivers.points)} points: direct")
        direct_e[name] = _irradiance(
            scene,
            lamps,
            receivers.points,
            receivers.normals,
            own=_own_facets(lamps, receivers.reflector, receivers.points),
        )
        if bounced is not None:
            say(f"irradiance on {name!r}: reflected")
            reflected_e[name] = _irradiance(
                scene,
                bounced,
                receivers.points,
                receivers.normals,
                own=_own_facets(bounced, receivers.reflector, receivers.points),
            )
    return SceneSolution(
        lamp_power=lamp_power,
        fluence_rate_direct=direct_g,
        fluence_rate_reflected=reflected_g,
        irradiance_direct=types.MappingProxyType(direct_e),
        irradiance_reflected=None if reflected_e is None else types.MappingProxyType(reflected_e),
        radiosity=None if exchanged is None else exchanged.reflector_radiosity,
        reflector_irradiance=None if exchanged is None else exchanged.reflector_irradiance,
        lamp_irradiance=None if exchanged is None else exchanged.lamp_irradiance,
        lamp_absorbed_power=None if exchanged is None else exchanged.lamp_absorbed_power,
        cycles=None if exchanged is None else exchanged.cycles,
        medium_absorbed_power=medium,
    )


@dataclasses.dataclass(frozen=True)
class _Exchange:
    """The facets that exchange reflected light: every reflector, then every areal lamp facet.

    The lamps' facets carry no emission here -- the lamps' own light arrives from outside the solve,
    gathered with their profiles -- only their geometry and their diffuse reflectance, so each lamp
    reflects and shadows as a wall does. A point source has no surface and is left out.

    Attributes
    ----------
    surfaces : Surfaces
        The reflectors' facets followed by the lamps' areal facets, with their bodies' names.
    n_reflector_facets : int
        How many of the facets are the reflectors'; the rest are the lamps'.
    lamp_facets : np.ndarray of int, shape ``(n_surfaces - n_reflector_facets,)``
        Which facet of :attr:`Scene.lamps` each of the rest is.
    """

    surfaces: Surfaces
    n_reflector_facets: int
    lamp_facets: np.ndarray

    @classmethod
    def of(cls, scene: Scene) -> _Exchange | None:
        """The exchanging facets of ``scene``; ``None`` when nothing can reflect.

        Without reflectors and with no lamp that reflects, no light is ever sent back, so nothing
        is exchanged and nothing is gathered as reflected -- the lamps' shadows on reflected light
        then shadow nothing.
        """
        lamps, reflectors = scene.lamps, scene.reflectors
        lamp_facets = np.flatnonzero(~lamps.is_point_source)
        lamp_reflectance = np.asarray(lamps.diffuse_reflectance)[lamp_facets]
        if reflectors is None and not np.any(lamp_reflectance > 0.0):
            return None
        pieces = [] if reflectors is None else [reflectors]
        n_reflector_facets = 0 if reflectors is None else reflectors.n_facets
        reflecting_names = () if reflectors is None else reflectors.solid_names
        vertices = [np.asarray(piece.vertices) for piece in pieces]
        solid_id = [np.asarray(piece.solid_id) for piece in pieces]
        reflectance = [np.asarray(piece.diffuse_reflectance) for piece in pieces]
        vertices.append(np.asarray(lamps.vertices)[lamp_facets])
        solid_id.append(np.asarray(lamps.solid_id)[lamp_facets] + len(reflecting_names))
        reflectance.append(lamp_reflectance)
        surfaces = Surfaces.from_triangles(
            np.concatenate(vertices),
            solid_id=np.concatenate(solid_id),
            solid_names=(*reflecting_names, *lamps.solid_names),
            diffuse_reflectance=np.concatenate(reflectance),
            point_sources=(),
        )
        return cls(surfaces, n_reflector_facets, lamp_facets)


@dataclasses.dataclass(frozen=True)
class _Exchanged:
    """What the surface solve gives each exchanging facet, split back into reflectors and lamps."""

    radiosity: np.ndarray
    reflector_radiosity: np.ndarray | None
    reflector_irradiance: np.ndarray | None
    lamp_irradiance: np.ndarray
    lamp_absorbed_power: float
    cycles: int


def _solve_exchange(scene: Scene, exchange: _Exchange, solver, say) -> _Exchanged:
    """Solve the reflected light among the exchanging facets, lit from outside by the lamps.

    The lamps' light on each facet is averaged over its sub-triangle centroids. On a lamp facet
    those points are moved a hair in front of the facet, so that it lights none of them whatever a
    rounding of their heights says, and its triangle is left out of their shadow test.
    """
    surfaces, lamps = exchange.surfaces, scene.lamps
    samples = subtriangle_centroids(surfaces.vertices, scene.lamp_samples)
    per_facet = samples.shape[1]
    normals = np.repeat(np.asarray(surfaces.normal)[:, None, :], per_facet, axis=1)
    say(
        f"exchange: {exchange.n_reflector_facets} reflector and "
        f"{len(exchange.lamp_facets)} lamp facets; the lamps' light on them at "
        f"{samples.shape[0] * per_facet} points"
    )
    split = exchange.n_reflector_facets
    external = np.zeros(samples.shape[:2])
    if split:
        external[:split] = _irradiance(
            scene, lamps, samples[:split].reshape(-1, 3), normals[:split].reshape(-1, 3)
        ).reshape(split, per_facet)
    if len(exchange.lamp_facets):
        vertices = np.asarray(surfaces.vertices)
        size = np.max(np.ptp(vertices.reshape(-1, 3), axis=0)) + np.max(np.abs(vertices))
        in_front = samples[split:] + _IN_FRONT_OF_OWN_FACET * size * normals[split:]
        external[split:] = _irradiance(
            scene,
            lamps,
            in_front.reshape(-1, 3),
            normals[split:].reshape(-1, 3),
            own=np.repeat(exchange.lamp_facets, per_facet)[:, None],
        ).reshape(-1, per_facet)
    say("exchange: building the facet-to-facet transfer")
    model = build_radiation_model(
        np.zeros((0, 3)), surfaces, occluders=scene.occluders, settings=scene.settings
    )
    landing, solved_cycles = surface_irradiance(
        model,
        surfaces,
        absorption=scene.absorption,
        external_irradiance=jnp.asarray(external.mean(axis=1)),
        solver=solver,
    )
    arriving = np.asarray(landing)
    cycles = int(solved_cycles)
    say(f"exchange: radiosity solved in {cycles} restart cycle(s)")
    # What each facet sends out is what it reflects: the lamps' own light is gathered apart.
    reflectance = np.asarray(surfaces.diffuse_reflectance)
    radiosity = reflectance * arriving
    lamp_irradiance = np.zeros(lamps.n_facets)
    lamp_irradiance[exchange.lamp_facets] = arriving[split:]
    absorbed = (1.0 - reflectance[split:]) * arriving[split:] * np.asarray(surfaces.area)[split:]
    has_reflectors = scene.reflectors is not None
    return _Exchanged(
        radiosity=radiosity,
        reflector_radiosity=radiosity[:split] if has_reflectors else None,
        reflector_irradiance=arriving[:split] if has_reflectors else None,
        lamp_irradiance=lamp_irradiance,
        lamp_absorbed_power=float(np.sum(absorbed)),
        cycles=cycles,
    )


def _pair_limit(settings: RadiationSettings) -> int:
    """The pairs a pass may form, as the settings say or by the library's default."""
    return settings.gather_options().get("pair_limit", DEFAULT_PAIR_LIMIT)


def _casts_shadows(scene: Scene) -> bool:
    """Whether anything can shadow a point: a body, or the sources' own triangles.

    The sources shadow themselves unless the settings switch that off, since unset is the ray test.
    """
    occlusion = scene.settings.receiver_visibility_options().get("self_occlusion")
    return bool(scene.occluders) or not isinstance(occlusion, NoOcclusion)


def _fluence(scene: Scene, sources: Surfaces, points: np.ndarray) -> np.ndarray:
    """Fluence rate from ``sources`` at ``points``, through the scene's shadows, mask streamed."""
    pair_limit = _pair_limit(scene.settings)
    if not _casts_shadows(scene):
        field = summed_fluence_rate(
            (sources,), jnp.asarray(points), absorption=scene.absorption, pair_limit=pair_limit
        )
        return np.asarray(field)
    options = dict(scene.settings.receiver_visibility_options())
    self_occlusion = options.pop("self_occlusion", None)
    field = streamed_fluence_rate(
        (sources,),
        jnp.asarray(points),
        # Shadows are cast from positions alone; nothing here is differentiated through them.
        shadow_geometry=jax.lax.stop_gradient(sources),
        occluders=scene.occluders,
        self_occlusion=self_occlusion,
        visibility_options=options,
        absorption=scene.absorption,
        pair_limit=pair_limit,
    )
    return np.asarray(field)


def _irradiance(
    scene: Scene, sources: Surfaces, points: np.ndarray, normals: np.ndarray, own=None
) -> np.ndarray:
    """Irradiance from ``sources`` at oriented points, a pass of points at a time.

    Each pass builds the mask for its own points and drops it, so memory is a pass's whatever the
    number of points. ``own`` gives, per point, the facet of ``sources`` it lies on, which the shadow
    test leaves out at the point's end of every ray.
    """
    pair_limit = _pair_limit(scene.settings)
    if not _casts_shadows(scene):
        return np.asarray(
            direct_irradiance(
                sources,
                jnp.asarray(points),
                jnp.asarray(normals),
                absorption=scene.absorption,
                pair_limit=pair_limit,
            )
        )
    per_pass = receivers_per_pass(pair_limit, sources.n_facets * max(1, len(scene.occluders)))
    options = scene.settings.receiver_visibility_options()
    out = np.empty(len(points))
    for start in range(0, len(points), per_pass):
        stop = min(start + per_pass, len(points))
        chunk = points[start:stop]
        facet = None if own is None else own[start:stop]
        if facet is not None and isinstance(options.get("self_occlusion"), SilhouetteOcclusion):
            # The clip projects its shares about one facet's normal: the nearest.
            facet = facet[:, 0]
        visibility = build_visibility(
            scene.occluders,
            sources,
            chunk,
            receiver_facet=facet,
            pair_limit=pair_limit,
            **options,
        )
        out[start:stop] = np.asarray(
            direct_irradiance(
                sources,
                jnp.asarray(chunk),
                jnp.asarray(normals[start:stop]),
                absorption=scene.absorption,
                visibility=visibility,
                pair_limit=pair_limit,
            )
        )
    return out


def _own_facets(surfaces: Surfaces, body: str | None, points: np.ndarray) -> np.ndarray | None:
    """The facets of ``body`` each point lies on: the nearest, and any as near as it.

    A point on a shared edge or vertex lies on several -- a face centre is the vertex all the
    triangles of its own centre fan share -- and every one of them must be left out of the shadow
    test. Returned as ``(n_points, k)`` rows, ``-1`` filling a row that names fewer; ``None`` when
    the points lie on no body of ``surfaces``, so nothing is left out.
    """
    if body is None or body not in surfaces.solid_names:
        return None
    candidates = np.flatnonzero(np.asarray(surfaces.solid_id) == surfaces.solid_names.index(body))
    corners = np.asarray(surfaces.vertices)[candidates]
    count = min(_OWN_FACET_CANDIDATES, len(candidates))
    _, nearest = cKDTree(corners.mean(axis=1)).query(points, k=count)
    nearest = np.asarray(nearest).reshape(len(points), count)
    distance = _point_triangle_distance(points[:, None, :], corners[nearest])
    # As near as the nearest, to a rounding of the body's size.
    span = float(np.max(np.ptp(corners.reshape(-1, 3), axis=0)))
    tied = distance <= distance.min(axis=1, keepdims=True) + _TIE * span
    width = int(tied.sum(axis=1).max())
    order = np.argsort(~tied, axis=1, kind="stable")[:, :width]
    chosen = np.take_along_axis(candidates[nearest], order, axis=1)
    return np.where(np.take_along_axis(tied, order, axis=1), chosen, -1)


def _refuse_shape(name: str, array: np.ndarray, shape: tuple) -> None:
    """Refuse an array whose shape is not ``shape``, where ``None`` matches any length."""
    if array.ndim != len(shape) or any(
        want is not None and have != want for have, want in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
