"""Lamps lighting a scene of reflecting surfaces, with the light gathered where it is wanted.

A lamp and a reflecting wall leave the facet-to-facet transfer in different ways, and the
difference decides the assembly. Reflected light leaves every surface Lambertian, whatever lit it,
so the reflecting surfaces are exactly what :func:`~aquaflux.radiation.model.build_radiation_model`
is for. A lamp's light need not be: a measured luminaire is a table over two angles
(:class:`~aquaflux.radiation.photometry.PhotometricProfile`), which the transfer cannot carry,
because it freezes one direction per pair of facets. So the lamps are **kept out of the
transfer**:

- their direct light on each reflecting facet is gathered with their own profiles, averaged over
  points spread evenly across the facet so a shadow falling partly across it is not decided at one
  point, and handed to the surface solve as an irradiance arriving from outside it;
- the solve then carries only reflected, Lambertian light, and gives each reflecting facet its
  radiosity;
- the light anywhere is the lamps' direct light, gathered with their profiles, plus the reflected
  light, gathered from the facets as Lambertian sources of their radiosity.

A lamp keeps none of the light that lands on it: it is black to arriving light, as a lamp window
modelled as an emitting boundary usually is.

:class:`Scene` holds the lamps, the reflecting surfaces, whatever stands in the way and the medium
between them, and the points the light is wanted at -- a set of points in the medium
(:class:`VolumeReceivers`), and any number of named sets of points on surfaces, each facing the way
the surface does (:class:`SurfaceReceivers`). :func:`solve_scene` returns the fluence rate at the
first and the irradiance at the others, split into the direct and the reflected parts
(:class:`SceneSolution`), together with what each set of points absorbs.

A point on a reflecting surface lies on one of that surface's facets. That facet's own light needs no
exclusion -- a facet lights nothing in its own plane, a point a rounding in front of it is behind its
receiving half-space and a point a rounding behind it is behind the facet -- but its triangle does,
from any test of what shadows the point: a ray from another facet ends in it, and would read it as a
blocker. Each set of points on a reflecting surface therefore names the surface's body, and the facet
of that body nearest each point is left out of the shadow test at the point's end of every ray.
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
from aquaflux.radiation.self_occlusion import NoOcclusion
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
        by :attr:`Scene.reflectors`.
    reflector : str or None
        The body of :attr:`Scene.reflectors` the points lie on, if they lie on one. The facet of that
        body nearest each point is then left out of the test of what shadows it, in which a ray from
        another facet would otherwise end in that facet and be read as blocked.

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
        What emits. Any angular profile; gathered directly and never part of the transfer, so a lamp
        is black to the light arriving on it.
    reflectors : Surfaces or None
        The surfaces that reflect, diffusely, by each facet's ``diffuse_reflectance``. They emit
        nothing of their own: an emitting surface is a lamp. Unset, nothing reflects.
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
        If a reflector emits, if a set of surface points names a body the reflectors do not have, or
        if ``lamp_samples`` is less than one.
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
        names = () if self.reflectors is None else self.reflectors.solid_names
        unknown = sorted(
            f"{name!r} names {receivers.reflector!r}"
            for name, receivers in self.surfaces.items()
            if receivers.reflector is not None and receivers.reflector not in names
        )
        if unknown:
            raise ValueError(
                f"Scene.surfaces: {', '.join(unknown)}, which is not a reflecting body; the "
                f"reflecting bodies are {list(names)}."
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
        What each reflecting facet sends out, in W/m².
    reflector_irradiance : np.ndarray or None, shape ``(n_reflector_facets,)``
        What lands on each reflecting facet, in W/m².
    cycles : int or None
        The surface solve's restart cycles.
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
    """Light a scene: the lamps directly, then whatever the reflecting surfaces send back.

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

    bounced, radiosity, arriving, cycles = None, None, None, None
    reflectors = scene.reflectors
    if reflectors is not None:
        samples = subtriangle_centroids(reflectors.vertices, scene.lamp_samples)
        normals = np.repeat(np.asarray(reflectors.normal)[:, None, :], samples.shape[1], axis=1)
        say(
            f"reflectors: {reflectors.n_facets} facets; the lamps' light on them at "
            f"{samples.shape[0] * samples.shape[1]} points"
        )
        external = _irradiance(scene, lamps, samples.reshape(-1, 3), normals.reshape(-1, 3))
        external = external.reshape(samples.shape[:2]).mean(axis=1)
        say("reflectors: building the facet-to-facet transfer")
        model = build_radiation_model(
            np.zeros((0, 3)), reflectors, occluders=scene.occluders, settings=scene.settings
        )
        landing, solved_cycles = surface_irradiance(
            model,
            reflectors,
            absorption=scene.absorption,
            external_irradiance=jnp.asarray(external),
            solver=solver,
        )
        arriving = np.asarray(landing)
        cycles = int(solved_cycles)
        # What each facet sends out is what it reflects, since a reflector emits nothing of its own.
        radiosity = np.asarray(reflectors.diffuse_reflectance) * arriving
        bounced = reflectors.with_optics(emission=jnp.asarray(radiosity), profiles=(Lambertian(),))
        say(f"reflectors: radiosity solved in {cycles} restart cycle(s)")

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
        direct_e[name] = _irradiance(scene, lamps, receivers.points, receivers.normals)
        if bounced is not None:
            say(f"irradiance on {name!r}: reflected")
            own = (
                None
                if receivers.reflector is None
                else _own_facets(bounced, receivers.reflector, receivers.points)
            )
            reflected_e[name] = _irradiance(
                scene, bounced, receivers.points, receivers.normals, own=own
            )
    return SceneSolution(
        lamp_power=lamp_power,
        fluence_rate_direct=direct_g,
        fluence_rate_reflected=reflected_g,
        irradiance_direct=types.MappingProxyType(direct_e),
        irradiance_reflected=None if reflected_e is None else types.MappingProxyType(reflected_e),
        radiosity=radiosity,
        reflector_irradiance=arriving,
        cycles=cycles,
        medium_absorbed_power=medium,
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
    number of points. ``own`` gives, per point, the facets of ``sources`` it lies on, which the
    shadow test leaves out at the point's end of every ray. The mask is told which way every point
    faces whether or not it lies on one of them -- the lamps' light on a reflecting wall lies on no
    lamp facet -- so a share of a partly hidden source is a share of the measure an irradiance
    weights by.
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
        visibility = build_visibility(
            scene.occluders,
            sources,
            chunk,
            receiver_facet=None if own is None else own[start:stop],
            receiver_normal=normals[start:stop],
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


def _own_facets(surfaces: Surfaces, body: str, points: np.ndarray) -> np.ndarray:
    """The facets of ``body`` each point lies on: the nearest, and any as near as it.

    A point on a shared edge or vertex lies on several -- a face centre is the vertex all the
    triangles of its own centre fan share -- and every one of them must be left out of the shadow
    test. Returned as ``(n_points, k)`` rows, ``-1`` filling a row that names fewer.
    """
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
