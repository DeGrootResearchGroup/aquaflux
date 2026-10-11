"""A scene with transparent solids: each pair of a source and a point gathered in the right medium.

A point is lit by the sources in its own medium along straight lines, in that medium's absorption and
through whatever regions the line passes straight through, and by the sources in other media along
refracted paths. With every index equal and nothing absorbing the paths are the straight lines and
no surface takes anything, so the scene must give exactly what it gives with no media at all -- any
pair counted twice, left out, or routed to the wrong medium moves that apart. The other checks pin
what equal indices cannot see: the absorption each medium applies, a region crossed straight, and the
facet a point on a lamp lies on.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    NoOcclusion,
    RadiationSettings,
    RayCastOcclusion,
    Scene,
    SurfaceReceivers,
    Surfaces,
    UniformAbsorption,
    VolumeReceivers,
    direct_fluence_rate,
    solve_scene,
)
from aquaflux.radiation.refraction import Media, Transparent
from aquaflux.solids import Box, Cylinder

from tests.unit.radiation_references import inward_box

WATER, QUARTZ, AIR = 1.376, 1.5048, 1.0003
ARC, INNER, OUTER = 0.0075, 0.01025, 0.0115


def _drum(radius: float, height: float, sectors: int, slices: int, centre=(0.0, 0.0)):
    """An open drum about an axis parallel to z, wound outward, ``(n, 3, 3)``."""
    angle = np.linspace(0.0, 2.0 * np.pi, sectors + 1)[:-1]
    z = np.linspace(-height / 2, height / 2, slices + 1)
    triangles = []
    for i in range(sectors):
        a, b = angle[i], angle[(i + 1) % sectors]

        def at(t, h):
            return [centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t), h]

        for k in range(slices):
            triangles.append([at(a, z[k]), at(b, z[k]), at(b, z[k + 1])])
            triangles.append([at(a, z[k]), at(b, z[k + 1]), at(a, z[k + 1])])
    return np.asarray(triangles)


def _lamps(*centres, exitance=50.0) -> Surfaces:
    """An arc about each centre, each its own body."""
    pieces = [_drum(ARC, 0.2, 16, 6, centre) for centre in centres]
    surfaces = Surfaces.from_triangles(
        np.concatenate(pieces),
        solid_id=np.repeat(np.arange(len(pieces)), [len(piece) for piece in pieces]),
        solid_names=tuple(f"lamp{k}" for k in range(len(pieces))),
    )
    return surfaces.with_optics(emission=jnp.full(surfaces.n_facets, exitance))


def _sleeves(*centres, water=WATER, quartz=QUARTZ, air=AIR, absorption=(None, None, None)):
    """A sleeve about each centre; ``absorption`` is the water's, the quartz's and the air's."""
    axis = [0.0, 0.0, 1.0]
    regions = []
    for x, y in centres:
        gap = Transparent(Cylinder([x, y, 0.0], axis, INNER, 0.4), air, absorption[2])
        regions.append(
            Transparent(Cylinder([x, y, 0.0], axis, OUTER, 0.4), quartz, absorption[1], (gap,))
        )
    return Media(water, tuple(regions), absorption[0])


#: Points in the water, and two in the first lamp's air gap.
_WATER_POINTS = np.array([[0.02, 0.0, 0.0], [0.015, 0.012, 0.03], [0.05, -0.02, 0.06]])
_GAP_POINTS = np.array([[0.009, 0.0, 0.01], [0.0, -0.0095, -0.02]])
_POINTS = np.concatenate([_WATER_POINTS[:2], _GAP_POINTS, _WATER_POINTS[2:]])


def test_with_every_index_equal_and_nothing_absorbing_the_scene_is_the_scene_without_media():
    """Water points take every lamp by refracted paths; gap points take their own lamp straight.

    Both lamps light every point: the second lamp's light reaches the first lamp's gap through two
    sleeves, so it is refracted there too.
    """
    lamps = _lamps((0.0, 0.0), (0.0, 0.04))
    normals = np.tile([-1.0, 0.0, 0.0], (len(_POINTS), 1))
    settings = RadiationSettings(self_occlusion=NoOcclusion())
    kwargs = {
        "lamps": lamps,
        "volume": VolumeReceivers(_POINTS),
        "surfaces": {"probe": SurfaceReceivers(_POINTS, normals)},
        "settings": settings,
    }
    plain = solve_scene(Scene(**kwargs))
    media = _sleeves((0.0, 0.0), (0.0, 0.04), water=1.3, quartz=1.3, air=1.3)
    through = solve_scene(Scene(media=media, **kwargs))
    np.testing.assert_allclose(through.fluence_rate_direct, plain.fluence_rate_direct, rtol=1e-10)
    np.testing.assert_allclose(
        through.irradiance_direct["probe"], plain.irradiance_direct["probe"], rtol=1e-10
    )
    assert np.all(plain.fluence_rate_direct > 0.0)


def test_a_point_is_lit_through_its_own_medium_s_absorption():
    """In the gap the light of its own lamp is absorbed by the air, not by the water.

    And what the medium absorbs is each point's own coefficient times its fluence rate.
    """
    lamps = _lamps((0.0, 0.0))
    water, air = 30.0, 2.0
    media = _sleeves(
        (0.0, 0.0),
        absorption=(UniformAbsorption(water), None, UniformAbsorption(air)),
    )
    volumes = np.full(len(_POINTS), 1e-6)
    solved = solve_scene(
        Scene(
            lamps,
            media=media,
            volume=VolumeReceivers(_POINTS, volumes),
            settings=RadiationSettings(self_occlusion=NoOcclusion()),
        )
    )
    gap = np.asarray(direct_fluence_rate(lamps, _GAP_POINTS, absorption=UniformAbsorption(air)))
    np.testing.assert_allclose(solved.fluence_rate_direct[2:4], gap, rtol=1e-12)
    coefficient = np.where(np.isin(np.arange(len(_POINTS)), [2, 3]), air, water)
    assert solved.medium_absorbed_power == pytest.approx(
        float(np.sum(coefficient * solved.fluence_rate_direct * volumes)), rel=1e-12
    )


def test_a_slab_between_a_small_lamp_and_a_point_in_one_medium_brings_the_lamp_nearer():
    """A thick quartz slab across the line from a small lamp to a point, all in water.

    The light is bent through the slab, not taken straight: near the axis a slab of thickness ``t``
    shows the lamp ``t (1 - n_water / n_quartz)`` nearer, so its solid angle grows by the square of
    the distance over the apparent distance. Times ``(1 - R)^2`` at normal incidence and the
    quartz's excess absorption across the slab. The lamp is a millimetre at a tenth of a metre, so
    the near-axis form holds to about one part in ten thousand; straight through the slab would be
    9 % lower.
    """
    lamp = Surfaces.from_triangles(
        np.array([[[0.0, -1.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]]) * 1e-3 / 3
    ).with_optics(emission=jnp.array([20.0]))
    point = np.array([[0.1, 0.0, 0.0]])
    water, quartz, half = 4.0, 9.0, 0.02
    media = Media(
        WATER,
        (
            Transparent(
                Box([0.05, 0.0, 0.0], [half, 0.05, 0.05]), QUARTZ, UniformAbsorption(quartz)
            ),
        ),
        UniformAbsorption(water),
    )
    solved = solve_scene(
        Scene(
            lamp,
            media=media,
            volume=VolumeReceivers(point),
            settings=RadiationSettings(self_occlusion=NoOcclusion()),
        )
    )
    straight = float(direct_fluence_rate(lamp, point, absorption=UniformAbsorption(water))[0])
    kept = 1.0 - ((WATER - QUARTZ) / (WATER + QUARTZ)) ** 2
    nearer = (0.1 / (0.1 - 2 * half * (1.0 - WATER / QUARTZ))) ** 2
    expected = straight * kept**2 * np.exp(-(quartz - water) * 2 * half) * nearer
    assert solved.fluence_rate_direct[0] == pytest.approx(expected, rel=1e-3)


def test_points_on_a_lamp_take_another_lamp_s_light_as_points_just_off_it_do():
    """The last leg of a refracted path ends in the facet the point lies on, and is not cut by it.

    Each point is a facet centroid of the second lamp, named by its body; the reference is the same
    points a hair in front of their facets, named by nothing.
    """
    lamps = _lamps((0.0, 0.0), (0.0, 0.04))
    second = np.flatnonzero(np.asarray(lamps.solid_id) == 1)[::7]
    on = np.asarray(lamps.centroid)[second]
    normals = np.asarray(lamps.normal)[second]
    media = _sleeves((0.0, 0.0), (0.0, 0.04))
    settings = RadiationSettings(self_occlusion=RayCastOcclusion())

    def irradiance(points, reflector):
        scene = Scene(
            lamps,
            media=media,
            surfaces={"lamp": SurfaceReceivers(points, normals, reflector=reflector)},
            settings=settings,
        )
        return solve_scene(scene).irradiance_direct["lamp"]

    named = irradiance(on, "lamp1")
    off = irradiance(on + 1e-7 * normals, None)
    facing = named > 0.0
    assert facing.sum() >= 3
    np.testing.assert_allclose(named, off, rtol=1e-4, atol=1e-6 * named.max())


def test_a_scene_with_media_refuses_a_second_absorption_and_anything_to_exchange():
    lamps = _lamps((0.0, 0.0))
    media = _sleeves((0.0, 0.0))
    with pytest.raises(ValueError, match="both absorption and media"):
        Scene(lamps, absorption=UniformAbsorption(1.0), media=media)
    box = inward_box(2)
    walls = Surfaces.from_triangles(box).with_optics(diffuse_reflectance=jnp.full(len(box), 0.5))
    with pytest.raises(NotImplementedError, match="not yet carried through"):
        Scene(lamps, reflectors=walls, media=media)
    reflecting = lamps.with_optics(diffuse_reflectance=jnp.full(lamps.n_facets, 0.1))
    with pytest.raises(NotImplementedError, match="not yet carried through"):
        Scene(reflecting, media=media)
