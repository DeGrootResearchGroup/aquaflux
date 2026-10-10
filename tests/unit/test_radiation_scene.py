"""Lamps' emission kept out of the transfer: the scene's light against the model with the lamp inside it.

A Lambertian lamp can be carried either way, which is what makes the comparison: inside the model's
surface set, its light on each facet goes through the transfer; kept out, it is gathered directly and
handed to the surface solve. With one receiver point per facet in the transfer and one sample per
facet in the scene, the two evaluate the same projected solid angles at the same points, so the fields
agree to rounding -- and any slip in the assembly (the lamp's light not reaching the reflectors, the
reflected light gathered with the wrong radiosity, a facet left out) moves them apart by a visible
fraction. The lamp's facets stay in the transfer either way, reflecting and shadowing as walls do, so
the same comparison holds for a lamp that reflects and for one that stands in the walls' light.
"""

from __future__ import annotations

import ast
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    DEFAULT_LAMP_SAMPLES,
    Isotropic,
    Lambertian,
    NoOcclusion,
    RadiationSettings,
    RayCastOcclusion,
    Scene,
    SilhouetteOcclusion,
    SurfaceReceivers,
    Surfaces,
    UniformAbsorption,
    VolumeReceivers,
    build_radiation_model,
    build_visibility,
    direct_irradiance,
    fluence_rate,
    radiosity,
    solve_scene,
    subtriangle_centroids,
    surface_irradiance,
)
from aquaflux.radiation.gather import summed_fluence_rate
from aquaflux.solids import Box, Cylinder, Difference, Outside, Sphere

from tests.unit.radiation_references import (
    closed_drum,
    inward_box,
    rectangle_triangles,
    sampled_fraction,
    tilt_rotation,
    tilted,
)

LAMP_EXITANCE = 7.0
REFLECTANCE = 0.6
SETTINGS = RadiationSettings(receiver_quadrature=1, self_occlusion=NoOcclusion())


def _box_split(divisions: int = 3):
    """An inward unit box: the facets of its top face (z = 1), and the rest."""
    triangles = inward_box(divisions)
    top = np.all(np.isclose(triangles[:, :, 2], 1.0), axis=1)
    return triangles, top


LAMP_REFLECTANCE = 0.4


def _drum_in_box(divisions: int = 3, sectors: int = 12):
    """An inward unit box and a closed drum standing in its middle, wound to face the box."""
    drum = closed_drum(sectors, radius=0.12, half_height=0.25) + 0.5
    return inward_box(divisions), drum


def _around_the_drum() -> np.ndarray:
    """Points in the box clear of the drum, several of them with the drum between them and a wall."""
    points = _inside_points()
    return points[np.hypot(points[:, 0] - 0.5, points[:, 1] - 0.5) > 0.2]


def _inside_points() -> np.ndarray:
    grid = np.linspace(0.15, 0.85, 4)
    return np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)


def test_a_lamp_kept_out_of_the_transfer_lights_the_box_as_the_model_does_with_it_inside() -> None:
    triangles, top = _box_split()
    whole = Surfaces.from_triangles(
        triangles,
        emission=np.where(top, LAMP_EXITANCE, 0.0),
        diffuse_reflectance=np.where(top, 0.0, REFLECTANCE),
    )
    points = _inside_points()
    expected, _ = fluence_rate(build_radiation_model(points, whole, settings=SETTINGS), whole)
    outgoing, _ = radiosity(build_radiation_model(points, whole, settings=SETTINGS), whole)

    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            volume=VolumeReceivers(points=points),
            lamp_samples=1,
            settings=SETTINGS,
        )
    )
    np.testing.assert_allclose(solution.fluence_rate, np.asarray(expected), rtol=1e-11)
    np.testing.assert_allclose(solution.radiosity, np.asarray(outgoing)[~top], rtol=1e-11)
    # The reflected part is a large share of the field, so the agreement is not the direct light's alone.
    assert np.min(solution.fluence_rate_reflected / solution.fluence_rate) > 0.3


def test_the_lamps_light_on_a_facet_is_its_average_over_the_facet_not_its_centroids() -> None:
    # Averaged over more points, the light the lamp lands on each reflecting facet converges on the
    # transfer's own twelve-point integral over the facet; taken at the centroid, it does not. Both
    # solves carry the reflectors' exchange by the same twelve-point rule, so the lamp's light on the
    # facets is the only thing that differs.
    triangles, top = _box_split(2)
    whole = Surfaces.from_triangles(
        triangles,
        emission=np.where(top, LAMP_EXITANCE, 0.0),
        diffuse_reflectance=np.where(top, 0.0, REFLECTANCE),
    )
    fine = RadiationSettings(receiver_quadrature=12, self_occlusion=NoOcclusion())
    model = build_radiation_model(np.zeros((0, 3)), whole, settings=fine)
    expected = np.asarray(radiosity(model, whole)[0])[~top]

    def error(samples: int) -> float:
        solution = solve_scene(
            Scene(
                lamps=Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE),
                reflectors=Surfaces.from_triangles(
                    triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
                ),
                lamp_samples=samples,
                settings=fine,
            )
        )
        return float(np.max(np.abs(solution.radiosity - expected) / expected))

    assert error(8) < 0.05 * error(1)


def test_the_irradiance_on_a_wall_is_what_the_solved_radiosity_sends_it() -> None:
    triangles, top = _box_split()
    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    # Points on the floor, in its plane, facing up.
    grid = np.linspace(0.1, 0.9, 5)
    points = np.stack([*np.meshgrid(grid, grid, indexing="ij"), np.zeros((5, 5))], axis=-1).reshape(
        -1, 3
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={
                "floor": SurfaceReceivers(
                    points, normals, reflectance=REFLECTANCE, reflector="walls"
                )
            },
            settings=SETTINGS,
        )
    )
    # What the floor keeps is what it does not reflect of everything arriving, direct and reflected.
    np.testing.assert_allclose(
        solution.irradiance_absorbed["floor"],
        (1.0 - REFLECTANCE) * solution.irradiance("floor"),
        rtol=1e-15,
    )
    emission = np.full(len(triangles), LAMP_EXITANCE)
    emission[~top] = solution.radiosity
    everything = Surfaces.from_triangles(triangles, emission=emission)
    expected = direct_irradiance(everything, jnp.asarray(points), jnp.asarray(normals))
    np.testing.assert_allclose(solution.irradiance("floor"), np.asarray(expected), rtol=1e-12)
    assert np.all(solution.irradiance_reflected["floor"] > 0.1 * solution.irradiance("floor"))


def test_a_point_on_a_reflecting_wall_is_not_shadowed_by_the_facet_it_lies_on() -> None:
    # Points in the floor's own plane, under the ray test among the reflectors' triangles. A ray from
    # any other facet ends in the facet under the point; left in the test, that facet blocks it.
    triangles, top = _box_split()
    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    grid = np.linspace(0.1, 0.9, 4)
    points = np.stack([*np.meshgrid(grid, grid, indexing="ij"), np.zeros((4, 4))], axis=-1).reshape(
        -1, 3
    )
    up = np.tile([0.0, 0.0, 1.0], (len(points), 1))

    def parts(occlusion, on):
        """The lamps' direct light on the points and the reflected light, side by side."""
        settings = RadiationSettings(receiver_quadrature=1, self_occlusion=occlusion)
        scene = Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={"floor": SurfaceReceivers(points, up, reflector=on)},
            settings=settings,
        )
        solution = solve_scene(scene)
        return np.stack(
            [solution.irradiance_direct["floor"], solution.irradiance_reflected["floor"]]
        )

    # A box is convex, so nothing shadows anything: the ray test must change nothing. The walls'
    # triangles stand in the way of the lamps' light as well as the walls', so both parts need the
    # facet under each point left out.
    unshadowed = parts(NoOcclusion(), None)
    np.testing.assert_allclose(parts(RayCastOcclusion(), "walls"), unshadowed, rtol=1e-12)
    assert np.all(parts(RayCastOcclusion(), None) < 0.5 * unshadowed)


def test_a_point_inside_a_tilted_reflecting_facet_is_not_lit_by_that_facet() -> None:
    # Off an axis-aligned plane, a point inside a facet's triangle sees that facet's corners at
    # heights of rounding noise, and the gather can read it the whole hemisphere: the facet's own
    # radiosity, at full strength. The reference moves each point a hair in front of its wall,
    # where nothing is undecidable and its own facet is behind it.
    triangles, top = _box_split()
    triangles = tilted(triangles)
    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    floor = np.isclose(np.asarray(reflectors.normal) @ np.asarray(lamps.normal)[0], -1.0)
    points = subtriangle_centroids(np.asarray(reflectors.vertices)[floor], 3).reshape(-1, 3)
    normals = np.tile(np.asarray(reflectors.normal)[floor][0], (len(points), 1))

    def reflected(occlusion, at, on):
        settings = RadiationSettings(receiver_quadrature=1, self_occlusion=occlusion)
        scene = Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={"floor": SurfaceReceivers(at, normals, reflector=on)},
            settings=settings,
        )
        return solve_scene(scene).irradiance_reflected["floor"]

    reference = reflected(NoOcclusion(), points + 1e-9 * normals, None)
    np.testing.assert_allclose(reflected(NoOcclusion(), points, "walls"), reference, rtol=1e-6)
    # The ray test takes the other path through the scene, which must leave the facets out too.
    np.testing.assert_allclose(reflected(RayCastOcclusion(), points, "walls"), reference, rtol=1e-6)


def test_the_wall_irradiance_is_the_same_however_many_points_a_pass_holds() -> None:
    triangles, top = _box_split()
    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    points = _inside_points()
    # Each point faces its own way, so a pass handed another pass's normals would show.
    angle = np.linspace(0.0, np.pi, len(points))
    normals = np.stack([np.cos(angle), np.zeros_like(angle), np.sin(angle)], axis=1)
    sphere = Sphere(centre=jnp.asarray([0.5, 0.5, 0.75]), radius=0.1)
    small = RadiationSettings(self_occlusion=NoOcclusion(), gather_pair_limit=5 * lamps.n_facets)
    solution = solve_scene(
        Scene(
            lamps=lamps,
            occluders=(sphere,),
            surfaces={"probe": SurfaceReceivers(points, normals)},
            settings=small,
        )
    )
    visibility = build_visibility((sphere,), lamps, points, self_occlusion=NoOcclusion())
    expected = direct_irradiance(lamps, points, normals, visibility=visibility)
    np.testing.assert_allclose(solution.irradiance("probe"), np.asarray(expected), rtol=1e-13)
    # The sphere shadows some of the points, so the shadows were applied in every pass.
    unshadowed = direct_irradiance(lamps, points, normals)
    assert np.sum(np.asarray(expected) < 0.9 * np.asarray(unshadowed)) > 3


def test_the_medium_absorbs_its_coefficient_times_the_volume_integral_of_the_fluence_rate() -> None:
    triangles, top = _box_split(2)
    points = _inside_points()
    volumes = np.linspace(1.0, 2.0, len(points)) / len(points)
    solution = solve_scene(
        Scene(
            lamps=Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE),
            absorption=UniformAbsorption(3.0),
            volume=VolumeReceivers(points=points, volumes=volumes),
            settings=SETTINGS,
        )
    )
    assert solution.medium_absorbed_power == pytest.approx(
        3.0 * float(np.sum(solution.fluence_rate * volumes)), rel=1e-14
    )
    assert solution.lamp_power == pytest.approx(LAMP_EXITANCE, rel=1e-12)
    assert solution.fluence_rate_reflected is None and solution.radiosity is None


@pytest.mark.parametrize("per_side", [1, 2, 3, 5])
def test_the_subtriangle_centroids_are_an_equal_area_average_over_the_triangle(per_side) -> None:
    triangle = np.array([[[0.2, 0.1, 0.0], [1.7, 0.4, 0.3], [0.5, 1.9, -0.2]]])
    samples = subtriangle_centroids(triangle, per_side)[0]
    assert samples.shape == (per_side**2, 3)
    assert len(np.unique(np.round(samples, 12), axis=0)) == per_side**2
    # A linear function's mean over equal-area pieces is its value at the centroid.
    np.testing.assert_allclose(samples.mean(axis=0), triangle[0].mean(axis=0), atol=1e-15)
    # And every sample lies inside: non-negative barycentric coordinates.
    edges = np.stack([triangle[0, 1] - triangle[0, 0], triangle[0, 2] - triangle[0, 0]], axis=1)
    weights, *_ = np.linalg.lstsq(edges, (samples - triangle[0, 0]).T, rcond=None)
    assert np.all(weights > 0) and np.all(weights.sum(axis=0) < 1)


def test_two_by_two_samples_are_the_centroids_of_the_four_midpoint_triangles() -> None:
    a, b, c = np.eye(3)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    expected = {
        tuple(np.round(np.mean(t, axis=0), 12))
        for t in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])
    }
    got = {tuple(row) for row in np.round(subtriangle_centroids(np.array([[a, b, c]]), 2)[0], 12)}
    assert got == expected


def test_a_scene_refuses_what_it_cannot_light_as_described() -> None:
    triangles, top = _box_split(2)
    lamps = Surfaces.from_triangles(triangles[top], emission=LAMP_EXITANCE)
    with pytest.raises(ValueError, match="reflectors emit light of their own"):
        Scene(lamps=lamps, reflectors=Surfaces.from_triangles(triangles[~top], emission=1.0))
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=0.5
    )
    probe = SurfaceReceivers(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), reflector="floor")
    with pytest.raises(
        ValueError, match=r"'probe' names 'floor', which is not a body of the lamps"
    ):
        Scene(lamps=lamps, reflectors=reflectors, surfaces={"probe": probe})
    with pytest.raises(ValueError, match="lamp_samples must be >= 1"):
        Scene(lamps=lamps, lamp_samples=0)
    with pytest.raises(ValueError, match="reflectance must lie in"):
        SurfaceReceivers(np.zeros((1, 3)), np.ones((1, 3)), reflectance=1.5)
    with pytest.raises(ValueError, match="normals must have shape"):
        SurfaceReceivers(np.zeros((2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match=r"both have bodies named \['surface'\]"):
        Scene(lamps=lamps, reflectors=Surfaces.from_triangles(triangles[~top]))
    for mirror in (
        {"lamps": lamps.with_optics(specular_reflectance=0.5)},
        {"lamps": lamps, "reflectors": reflectors.with_optics(specular_reflectance=0.2)},
    ):
        with pytest.raises(ValueError, match="reflect specularly"):
            Scene(**mirror)


def test_a_lamp_that_reflects_lights_the_box_as_the_model_does_with_it_inside() -> None:
    # The lamp's facets reflect as well as emit. Kept out of the transfer, its light still arrives
    # on the walls from outside the solve, and its facets reflect what the walls send back: so the
    # field, the walls' radiosity and what lands on the lamp all equal the model's, which carries
    # the lamp inside its surface set, emitting and reflecting at once.
    triangles, top = _box_split()
    whole = Surfaces.from_triangles(
        triangles,
        emission=np.where(top, LAMP_EXITANCE, 0.0),
        diffuse_reflectance=np.where(top, LAMP_REFLECTANCE, REFLECTANCE),
    )
    points = _inside_points()
    model = build_radiation_model(points, whole, settings=SETTINGS)
    expected, _ = fluence_rate(model, whole)
    landing, _ = surface_irradiance(model, whole)

    lamps = Surfaces.from_triangles(
        triangles[top], emission=LAMP_EXITANCE, diffuse_reflectance=LAMP_REFLECTANCE
    )
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            volume=VolumeReceivers(points=points),
            lamp_samples=1,
            settings=SETTINGS,
        )
    )
    np.testing.assert_allclose(solution.fluence_rate, np.asarray(expected), rtol=1e-11)
    np.testing.assert_allclose(solution.lamp_irradiance, np.asarray(landing)[top], rtol=1e-11)
    np.testing.assert_allclose(solution.reflector_irradiance, np.asarray(landing)[~top], rtol=1e-11)
    # The lamp's reflection is a visible share of the field, so it is not agreeing by being nothing.
    black = solve_scene(
        Scene(
            lamps=lamps.with_optics(diffuse_reflectance=0.0),
            reflectors=reflectors,
            volume=VolumeReceivers(points=points),
            lamp_samples=1,
            settings=SETTINGS,
        )
    )
    assert np.min(solution.fluence_rate / black.fluence_rate) > 1.05


def test_a_lamp_shadows_the_light_the_walls_reflect_past_it() -> None:
    # A black drum in a reflecting box, under the ray test: the light the walls send back is cut by
    # the drum on its way to the points behind it, exactly as the model cuts it with the drum among
    # its facets. Gathered from the walls alone, it would pass straight through the drum.
    walls, drum = _drum_in_box()
    settings = RadiationSettings(receiver_quadrature=1, self_occlusion=RayCastOcclusion())
    whole = Surfaces.from_triangles(
        np.concatenate([walls, drum]),
        emission=np.r_[np.zeros(len(walls)), np.full(len(drum), LAMP_EXITANCE)],
        diffuse_reflectance=np.r_[np.full(len(walls), REFLECTANCE), np.zeros(len(drum))],
    )
    points = _around_the_drum()
    model = build_radiation_model(points, whole, settings=settings)
    expected, _ = fluence_rate(model, whole)

    lamps = Surfaces.from_triangles(drum, solid_names=("lamp",), emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        walls, solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            volume=VolumeReceivers(points=points),
            lamp_samples=1,
            settings=settings,
        )
    )
    np.testing.assert_allclose(solution.fluence_rate, np.asarray(expected), rtol=1e-11)
    # The walls' light gathered as though the drum were not there is brighter at some point, so the
    # agreement above is the drum's shadow and not a scene in which it casts none.
    unshadowed = summed_fluence_rate(
        (reflectors.with_optics(emission=jnp.asarray(solution.radiosity)),), jnp.asarray(points)
    )
    assert np.max(np.asarray(unshadowed) / solution.fluence_rate_reflected) > 1.05


def _lamp_under_a_plate_under_a_ceiling():
    """A lamp facing up, a black plate over it facing down, and a reflecting ceiling over both.

    The plate hides the lamp from the middle of the ceiling and from the points above it, and leaves
    the outer ceiling tiles and the points beside it partly or wholly lit. It is black, so that no
    light it might send back from behind itself reaches the ceiling: what is compared below is the
    shadow it casts and nothing else.
    """
    x, y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    lamp = rectangle_triangles([0.0, 0.0, 0.0], 0.1 * x, 0.1 * y)
    plate = rectangle_triangles([0.0, 0.0, 1.0], 0.5 * y, 0.5 * x)
    ceiling = np.concatenate(
        [
            rectangle_triangles([i, j, 2.0], 0.5 * y, 0.5 * x)
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
        ]
    )
    grid = np.linspace(-1.2, 1.2, 5)
    points = np.array([[a, b, z] for a in grid for b in grid for z in (0.5, 1.5)])
    return lamp, plate, ceiling, points


def test_a_reflector_between_a_lamp_and_a_point_shadows_the_lamp_s_direct_light() -> None:
    # The plate is a reflector, and a reflector's triangles stand in the way of the lamps' own light
    # as they do of the reflected light: the field in the room, the light landing on the ceiling and
    # the ceiling points' irradiance all equal the model's, which holds every facet in one set.
    lamp, plate, ceiling, points = _lamp_under_a_plate_under_a_ceiling()
    settings = RadiationSettings(receiver_quadrature=1, self_occlusion=RayCastOcclusion())
    walls = np.concatenate([plate, ceiling])
    reflectance = np.r_[np.zeros(len(plate)), np.full(len(ceiling), REFLECTANCE)]
    whole = Surfaces.from_triangles(
        np.concatenate([lamp, walls]),
        emission=np.r_[np.full(len(lamp), LAMP_EXITANCE), np.zeros(len(walls))],
        diffuse_reflectance=np.r_[np.zeros(len(lamp)), reflectance],
    )
    model = build_radiation_model(points, whole, settings=settings)
    expected, _ = fluence_rate(model, whole)
    landing, _ = surface_irradiance(model, whole)

    lamps = Surfaces.from_triangles(lamp, solid_names=("lamp",), emission=LAMP_EXITANCE)
    reflectors = Surfaces.from_triangles(
        walls,
        solid_id=np.r_[np.zeros(len(plate), int), np.ones(len(ceiling), int)],
        solid_names=("plate", "ceiling"),
        diffuse_reflectance=reflectance,
    )
    centres = np.asarray(reflectors.centroid)[len(plate) :]
    down = np.tile([0.0, 0.0, -1.0], (len(centres), 1))
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            volume=VolumeReceivers(points),
            surfaces={"ceiling": SurfaceReceivers(centres, down, reflector="ceiling")},
            lamp_samples=1,
            settings=settings,
        )
    )
    np.testing.assert_allclose(solution.fluence_rate, np.asarray(expected), rtol=1e-11)
    np.testing.assert_allclose(
        solution.reflector_irradiance, np.asarray(landing)[len(lamp) :], rtol=1e-11
    )
    # The transfer leaves the shadowed tiles a rounding of zero from their coplanar neighbours.
    on_tiles = np.asarray(landing)[len(lamp) + len(plate) :]
    np.testing.assert_allclose(
        solution.irradiance("ceiling"), on_tiles, rtol=1e-11, atol=1e-14 * on_tiles.max()
    )
    # The plate hides the lamp wholly from some points and some ceiling tiles and leaves others lit,
    # so the agreement is its shadow on the direct light, not a field it does not reach.
    direct = solution.fluence_rate_direct
    unshadowed = np.asarray(summed_fluence_rate((lamps,), jnp.asarray(points)))
    assert np.sum(direct == 0.0) >= 5 and np.sum(np.isclose(direct, unshadowed, rtol=1e-12)) >= 5
    on_ceiling = solution.irradiance_direct["ceiling"]
    assert np.sum(on_ceiling == 0.0) >= 3 and np.sum(on_ceiling > 0.0) >= 3


def test_a_point_source_under_a_reflector_is_shadowed_by_it_and_lights_the_rest_as_a_point() -> (
    None
):
    # A point source gathered among the reflectors' facets stays a point source: beside the plate it
    # delivers P / (4 pi r^2), the isotropic point source's closed form, and over the plate nothing.
    _, plate, _, points = _lamp_under_a_plate_under_a_ceiling()
    power = 2.0
    lamps = Surfaces.from_triangles(
        np.zeros((1, 3, 3)), solid_names=("bulb",), power=power, profiles=(Isotropic(),)
    )
    reflectors = Surfaces.from_triangles(plate, solid_names=("plate",))
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            volume=VolumeReceivers(points),
            settings=RadiationSettings(self_occlusion=RayCastOcclusion()),
        )
    )
    direct = solution.fluence_rate_direct
    hidden = (points[:, 2] > 1.0) & np.all(np.abs(points[:, :2]) < 0.5 * points[:, 2:], axis=1)
    closed_form = power / (4.0 * np.pi * np.sum(points**2, axis=1))
    np.testing.assert_allclose(direct[~hidden], closed_form[~hidden], rtol=1e-12)
    assert np.all(direct[hidden] == 0.0) and hidden.sum() >= 5


def test_what_the_lamps_and_the_walls_absorb_is_what_the_lamps_emit() -> None:
    # A closed box with nothing in it to absorb but its walls and its lamp: every watt the lamp
    # emits ends on one of them. Leave the lamp's share out, or take its reflectance for what it
    # absorbs, and the books are out by a sixth and by a twentieth. They close only to what the
    # one-ray-per-pair shadows of the drum cost on a mesh this coarse, about a percent.
    walls, drum = _drum_in_box()
    settings = RadiationSettings(self_occlusion=RayCastOcclusion())
    lamps = Surfaces.from_triangles(
        drum, solid_names=("lamp",), emission=LAMP_EXITANCE, diffuse_reflectance=LAMP_REFLECTANCE
    )
    reflectors = Surfaces.from_triangles(walls, solid_names=("walls",), diffuse_reflectance=0.8)
    solution = solve_scene(Scene(lamps=lamps, reflectors=reflectors, settings=settings))
    walls_absorb = float(np.sum(0.2 * solution.reflector_irradiance * np.asarray(reflectors.area)))
    lamp_absorbs = float(
        np.sum((1.0 - LAMP_REFLECTANCE) * solution.lamp_irradiance * np.asarray(lamps.area))
    )
    assert solution.lamp_absorbed_power == pytest.approx(lamp_absorbs, rel=1e-12)
    assert solution.reflector_absorbed_power == pytest.approx(walls_absorb, rel=1e-12)
    assert lamp_absorbs > 0.1 * solution.lamp_power
    # The books close from the solution alone.
    assert solution.reflector_absorbed_power + solution.lamp_absorbed_power == pytest.approx(
        solution.lamp_power, rel=0.025
    )


def test_lamps_with_no_reflectors_exchange_light_only_when_one_of_them_reflects() -> None:
    # Two plates facing each other, each a lamp. Black, nothing is sent back and nothing exchanged.
    # Once they reflect, each sends back its share of the other's light, which a model with both
    # plates inside it agrees with.
    triangles, top = _box_split()
    bottom = np.all(np.isclose(triangles[:, :, 2], 0.0), axis=1)
    plates = np.concatenate([triangles[top], triangles[bottom]])
    names = np.r_[np.zeros(top.sum(), int), np.ones(bottom.sum(), int)]
    points = _inside_points()

    def lit(reflectance):
        lamps = Surfaces.from_triangles(
            plates,
            solid_id=names,
            solid_names=("top", "bottom"),
            emission=LAMP_EXITANCE,
            diffuse_reflectance=reflectance,
        )
        return lamps, solve_scene(
            Scene(
                lamps=lamps,
                volume=VolumeReceivers(points=points),
                lamp_samples=1,
                settings=SETTINGS,
            )
        )

    _, black = lit(0.0)
    assert black.fluence_rate_reflected is None and black.lamp_irradiance is None
    lamps, shining = lit(LAMP_REFLECTANCE)
    expected, _ = fluence_rate(build_radiation_model(points, lamps, settings=SETTINGS), lamps)
    np.testing.assert_allclose(shining.fluence_rate, np.asarray(expected), rtol=1e-11)
    assert shining.radiosity is None and shining.reflector_irradiance is None
    assert np.all(shining.lamp_irradiance > 0.0)


def test_a_point_source_lamp_neither_reflects_nor_shadows_and_receives_nothing() -> None:
    # A point source has no surface: the walls reflect around it, and what lands on it reads zero.
    triangles, top = _box_split()
    point = np.repeat(np.array([[[0.5, 0.5, 0.5]]]), 3, axis=1)
    lamps = Surfaces.from_triangles(
        np.concatenate([triangles[top], point]),
        emission=np.r_[np.full(top.sum(), LAMP_EXITANCE), 0.0],
        power=np.r_[np.zeros(top.sum()), 2.0],
        diffuse_reflectance=np.r_[np.full(top.sum(), LAMP_REFLECTANCE), 0.0],
        profiles=(Lambertian(), Isotropic()),
        profile_index=np.r_[np.zeros(top.sum(), int), 1],
    )
    reflectors = Surfaces.from_triangles(
        triangles[~top], solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    solution = solve_scene(
        Scene(lamps=lamps, reflectors=reflectors, lamp_samples=1, settings=SETTINGS)
    )
    assert solution.lamp_irradiance.shape == (lamps.n_facets,)
    assert solution.lamp_irradiance[-1] == 0.0
    assert np.all(solution.lamp_irradiance[:-1] > 0.0)


def test_points_on_a_lamp_are_not_shadowed_by_the_facet_they_lie_on() -> None:
    # Points on the drum's side, facing out, under the ray test among every triangle. A ray from
    # a wall ends in the facet under the point; left in the test, that facet blocks it, and the drum
    # reads dark where the walls light it. Named, the points read what the same points a micrometre
    # off the drum read, where no facet is under them to leave out.
    walls, drum = _drum_in_box(sectors=8)
    lamps = Surfaces.from_triangles(
        drum, solid_names=("lamp",), emission=LAMP_EXITANCE, diffuse_reflectance=LAMP_REFLECTANCE
    )
    reflectors = Surfaces.from_triangles(
        walls, solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    side = np.abs(np.asarray(lamps.normal)[:, 2]) < 0.5
    points = np.asarray(lamps.centroid)[side]
    normals = np.asarray(lamps.normal)[side]
    settings = RadiationSettings(receiver_quadrature=1, self_occlusion=RayCastOcclusion())

    def irradiance(at, on):
        scene = Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={"sleeve": SurfaceReceivers(at, normals, reflector=on)},
            settings=settings,
        )
        return solve_scene(scene).irradiance("sleeve")

    off = irradiance(points + 1e-6 * normals, None)
    np.testing.assert_allclose(irradiance(points, "lamp"), off, rtol=1e-4)
    # Unnamed, most of the points read under half of it: the rays their own facet cuts.
    assert np.mean(irradiance(points, None) < 0.5 * off) > 0.5


def test_one_lamp_s_light_on_another_lands_as_the_model_with_both_inside_says() -> None:
    # Two drums under the ray test. The points one lamp's light is averaged over on the other lie in
    # that lamp's facets, so a ray from the first ends in the facet under the point: left in the
    # test, that facet cuts it, and the second lamp reads dark where the first lights it. The model
    # carries both lamps in its surface set, where each facet is left out of its own rays, so it says
    # what lands with that done. One sample and one quadrature point per facet: the same points.
    walls = inward_box(3)
    drums = np.concatenate(
        [
            closed_drum(8, radius=0.1, half_height=0.25) + np.array([0.3, 0.5, 0.5]),
            closed_drum(8, radius=0.1, half_height=0.25) + np.array([0.7, 0.5, 0.5]),
        ]
    )
    is_lamp = np.arange(len(walls) + len(drums)) >= len(walls)
    settings = RadiationSettings(receiver_quadrature=1, self_occlusion=RayCastOcclusion())
    whole = Surfaces.from_triangles(
        np.concatenate([walls, drums]),
        emission=np.where(is_lamp, LAMP_EXITANCE, 0.0),
        diffuse_reflectance=np.where(is_lamp, LAMP_REFLECTANCE, REFLECTANCE),
    )
    landing, _ = surface_irradiance(
        build_radiation_model(np.zeros((0, 3)), whole, settings=settings), whole
    )
    lamps = Surfaces.from_triangles(
        drums, solid_names=("lamps",), emission=LAMP_EXITANCE, diffuse_reflectance=LAMP_REFLECTANCE
    )
    reflectors = Surfaces.from_triangles(
        walls, solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    solution = solve_scene(
        Scene(lamps=lamps, reflectors=reflectors, lamp_samples=1, settings=settings)
    )
    np.testing.assert_allclose(solution.lamp_irradiance, np.asarray(landing)[is_lamp], rtol=1e-11)
    # Each lamp lights the other directly: the walls alone, black, leave the lamps far darker.
    dark = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors.with_optics(diffuse_reflectance=0.0),
            lamp_samples=1,
            settings=settings,
        )
    )
    assert np.max(dark.lamp_irradiance) > 0.1 * np.max(solution.lamp_irradiance)


def test_the_lamps_light_on_a_lamp_is_taken_on_the_fluid_s_side_of_it() -> None:
    # The water as a body -- the box less a cylinder standing inside the drum, its caps on the
    # drum's caps, as a drawing's fluid leaves out its lamp. The points the lamps' light is averaged
    # over on a lamp facet must not be moved off it into the lamp: a hair behind a cap they are
    # outside the water, and the scene refuses them. Standing wholly behind the lamp's facets, the
    # cylinder shadows nothing the lamp does not, so what lands on the lamp is what it was without
    # the water.
    walls, drum = _drum_in_box()
    lamps = Surfaces.from_triangles(
        drum, solid_names=("lamp",), emission=LAMP_EXITANCE, diffuse_reflectance=LAMP_REFLECTANCE
    )
    reflectors = Surfaces.from_triangles(
        walls, solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    lamp = Cylinder(
        centre=jnp.asarray([0.5, 0.5, 0.5]),
        axis=jnp.asarray([0.0, 0.0, 1.0]),
        radius=0.12 * np.cos(np.pi / 12),
        half_length=0.25,
    )
    water = Outside(Difference(Box(centre=[0.5, 0.5, 0.5], half_sizes=0.5), lamp))
    settings = RadiationSettings(self_occlusion=RayCastOcclusion())

    def landing(occluders):
        scene = Scene(lamps=lamps, reflectors=reflectors, occluders=occluders, settings=settings)
        return solve_scene(scene).lamp_irradiance

    np.testing.assert_allclose(landing((water,)), landing(()), rtol=1e-12)


def test_a_tilted_lamp_flush_with_the_water_s_wall_is_lit_as_without_the_water() -> None:
    # A box lamp whose facets are the faces of the box the water leaves out, turned off every axis.
    # The points the lamps' light is averaged over then lie in those faces, at heights above the
    # water's wall that are rounding noise of either sign -- the same noise as the facet centroids,
    # which the scene checks too. A water body with a tolerance above that rounding admits the
    # facets, and so admits their sample points as they are: none needs moving off its facet. And
    # its own facet, named and left out, does not light it, so what lands is what it was without
    # the water.
    half = np.array([0.08, 0.08, 0.2])
    turn = tilt_rotation()
    # The unit box's inward triangles, wound to face outward, sized to the lamp and turned.
    lamp_triangles = ((inward_box(1)[:, ::-1, :] - 0.5) * 2.0 * half) @ turn.T + 0.5
    lamps = Surfaces.from_triangles(
        lamp_triangles,
        solid_names=("lamp",),
        emission=LAMP_EXITANCE,
        diffuse_reflectance=LAMP_REFLECTANCE,
    )
    reflectors = Surfaces.from_triangles(
        inward_box(3), solid_names=("walls",), diffuse_reflectance=REFLECTANCE
    )
    lamp = Box(centre=[0.5, 0.5, 0.5], half_sizes=half, axes=turn.T)
    water = Outside(Difference(Box(centre=[0.5, 0.5, 0.5], half_sizes=0.5), lamp), tolerance=1e-12)
    samples = subtriangle_centroids(np.asarray(lamps.vertices), DEFAULT_LAMP_SAMPLES)
    off_by_rounding = np.abs(np.asarray(lamp.signed_distance(jnp.asarray(samples.reshape(-1, 3)))))
    # The samples do lie in the lamp's faces, and some on the lamp's side of them by a rounding:
    # without the tolerance those would be refused.
    assert np.max(off_by_rounding) < 1e-14
    assert np.any(np.asarray(Outside(water.fluid).contains(jnp.asarray(samples.reshape(-1, 3)))))
    settings = RadiationSettings(self_occlusion=RayCastOcclusion())

    def landing(occluders):
        scene = Scene(lamps=lamps, reflectors=reflectors, occluders=occluders, settings=settings)
        return solve_scene(scene).lamp_irradiance

    unbounded = landing(())
    assert np.all(unbounded > 0.0)
    np.testing.assert_allclose(landing((water,)), unbounded, rtol=1e-12)


def _scene_with_a_sheet_in_each_set(occlusion) -> Scene:
    """A lamp with a baffle among the lamps, and a ceiling with a shelf among the reflectors.

    Each sheet faces away from the point it is meant to shade, so counted from the side it faces it
    hides nothing, and only its declaration as two-sided darkens that point. The first volume point
    sits above the baffle, on the far side of it from the lamp, and away from the shelf; the second
    sits under the shelf, which hides the whole ceiling from it, and away from the baffle. The baffle
    is clear of every line from the lamp to the ceiling, so the ceiling is lit either way.
    """
    up, down = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    lamp = rectangle_triangles([0.0, 0.0, 0.0], *[0.1 * np.asarray(e) for e in up])
    baffle = rectangle_triangles([0.55, 0.0, 0.3], [0.0, 0.4, 0.0], [0.25, 0.0, 0.0])
    ceiling = rectangle_triangles([0.0, 0.0, 2.0], *down)
    shelf = rectangle_triangles([-3.0, 0.0, 1.5], [1.75, 0.0, 0.0], [0.0, 1.5, 0.0])
    lamps = Surfaces.from_triangles(
        np.concatenate([lamp, baffle]),
        emission=[LAMP_EXITANCE] * 2 + [0.0] * 2,
        solid_id=[0, 0, 1, 1],
        solid_names=("lamp", "baffle"),
    )
    reflectors = Surfaces.from_triangles(
        np.concatenate([ceiling, shelf]),
        diffuse_reflectance=[REFLECTANCE] * 2 + [0.0] * 2,
        solid_id=[0, 0, 1, 1],
        solid_names=("ceiling", "shelf"),
    )
    return Scene(
        lamps=lamps,
        reflectors=reflectors,
        volume=VolumeReceivers(np.array([[1.0, 0.0, 0.6], [-3.0, 0.0, 1.2]])),
        settings=RadiationSettings(receiver_quadrature=1, self_occlusion=occlusion),
    )


@pytest.mark.parametrize(
    ("two_sided", "dark"),
    [((), ()), (("baffle",), (0,)), (("shelf",), (1,)), (("baffle", "shelf"), (0, 1))],
    ids=["neither", "baffle", "shelf", "both"],
)
def test_a_sheet_declared_two_sided_shades_from_behind_whichever_set_it_belongs_to(
    two_sided, dark
) -> None:
    # The lamp and the ceiling are sheets too, open at every edge; declaring them keeps the build
    # from warning about them, and changes nothing, since nothing lies behind either.
    sheets = ("lamp", "ceiling", *two_sided)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solution = solve_scene(
            _scene_with_a_sheet_in_each_set(SilhouetteOcclusion(two_sided=sheets))
        )
    # Every build warns about the open sheets of its own set that were left undeclared, and only
    # about those.
    warned = {
        name
        for w in caught
        for match in re.findall(r"facet\(s\) of (\[[^\]]*\]) belong", str(w.message))
        for name in ast.literal_eval(match)
    }
    assert warned == {"baffle", "shelf"} - set(two_sided)
    clear = solve_scene(_scene_with_a_sheet_in_each_set(NoOcclusion()))
    # The baffle is among the lamps, so it shades the lamps' light; the shelf is among the
    # reflectors, so it shades theirs. Each point is shaded only by the part its sheet can hide.
    shaded = (solution.fluence_rate_direct[0], solution.fluence_rate_reflected[1])
    unshaded = (clear.fluence_rate_direct[0], clear.fluence_rate_reflected[1])
    for point, (field, reference) in enumerate(zip(shaded, unshaded, strict=True)):
        assert reference > 0.0
        if point in dark:
            assert field == pytest.approx(0.0, abs=1e-12 * reference)
        else:
            assert field == pytest.approx(reference, rel=1e-12)
    # A sheet among the lamps blocks no lamp light from the ceiling, so the reflected light is the
    # same whatever the lamps declare.
    np.testing.assert_allclose(solution.radiosity, clear.radiosity, rtol=1e-12)


@pytest.mark.parametrize("field", ["self_occlusion", "receiver_occlusion"])
def test_a_scene_refuses_a_sheet_that_neither_set_has(field) -> None:
    scene = _scene_with_a_sheet_in_each_set(NoOcclusion())
    misspelt = RadiationSettings(**{field: SilhouetteOcclusion(two_sided=("baffle", "shelff"))})
    with pytest.raises(
        ValueError,
        match=r"two_sided names no body of these surfaces: \['shelff'\]; "
        r"have \['ceiling', 'shelf', 'lamp', 'baffle'\]",
    ):
        Scene(lamps=scene.lamps, reflectors=scene.reflectors, settings=misspelt)


def test_settings_cut_down_to_a_set_keep_its_own_sheets_and_nothing_else_changes() -> None:
    settings = RadiationSettings(
        receiver_quadrature=3,
        self_occlusion=SilhouetteOcclusion(two_sided=("baffle", "shelf"), cluster_size=8),
        receiver_occlusion=SilhouetteOcclusion(two_sided=("shelf", "lamp")),
        gather_pair_limit=1234,
    )
    lamps = settings.for_bodies(("lamp", "baffle"))
    assert lamps.self_occlusion == SilhouetteOcclusion(two_sided=("baffle",), cluster_size=8)
    assert lamps.receiver_occlusion == SilhouetteOcclusion(two_sided=("lamp",))
    assert (lamps.receiver_quadrature, lamps.gather_pair_limit) == (3, 1234)
    walls = settings.for_bodies(("shelf",))
    assert walls.self_occlusion.two_sided == walls.receiver_occlusion.two_sided == ("shelf",)
    # A choice that names no bodies is the same choice on every set, and an unset one stays unset.
    plain = RadiationSettings(self_occlusion=RayCastOcclusion())
    assert plain.for_bodies(("anything",)) == plain


#: A lamp seen obliquely from the origin, behind a baffle that hides its overhead end. Wound to
#: face the origin, so the lamp shines on it and the baffle stands in front of it.
OBLIQUE_LAMP = np.array([[-0.5, -1.0, 1.0], [4.0, -1.0, 1.0], [4.0, 1.5, 1.0]])
BAFFLE = np.array([[-9.0, -9.0, 0.5], [0.5, -9.0, 0.5], [0.5, 9.0, 0.5]])
UP = np.array([0.0, 0.0, 1.0])


def test_a_partly_shadowed_lamp_lights_a_wall_by_the_share_of_its_projected_solid_angle() -> None:
    """The lamps' light on a reflecting wall, and on a set of wall points, under the silhouette clip.

    Neither kind of point lies on a lamp facet, and both face a way. The baffle is one of the lamp
    set's own triangles, so the clip hides a share of the lamp, and the share an irradiance wants
    is of the lamp's projected solid angle -- which on this oblique fixture differs from the share
    of its plain solid angle by twenty times the sampler's noise. A point's irradiance is its
    unshadowed irradiance less that share, Lambertian radiance being the same in every direction;
    the share is checked against the brute-force sampler, which never reads a mask.
    """
    lamps = Surfaces.from_triangles(
        np.stack([OBLIQUE_LAMP[::-1], BAFFLE[::-1]]),
        solid_names=("lamp",),
        emission=np.array([LAMP_EXITANCE, 0.0]),
    )
    # One small reflecting facet whose centroid is the origin, sampled once, at that centroid.
    wall = np.array([[[-0.01, -0.01, 0.0], [0.02, -0.01, 0.0], [-0.01, 0.02, 0.0]]])
    reflectors = Surfaces.from_triangles(
        wall, solid_names=("wall",), diffuse_reflectance=REFLECTANCE
    )
    origin = np.zeros((1, 3))
    solution = solve_scene(
        Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={"floor": SurfaceReceivers(origin, UP[None])},
            lamp_samples=1,
            # Every body is one open sheet; each faces the points it lights or shadows, so this only
            # says they are meant as sheets.
            settings=RadiationSettings(
                receiver_quadrature=1,
                self_occlusion=SilhouetteOcclusion(two_sided=("lamp", "wall")),
            ),
        )
    )
    unshadowed = float(direct_irradiance(lamps, jnp.asarray(origin), jnp.asarray(UP[None]))[0])
    projected = sampled_fraction(np.zeros(3), UP, OBLIQUE_LAMP, BAFFLE, samples=400_000)
    plain = sampled_fraction(np.zeros(3), None, OBLIQUE_LAMP, BAFFLE, samples=400_000)
    assert abs(projected - plain) > 0.08
    for name, landed in (
        ("on the reflecting facet", float(solution.reflector_irradiance[0])),
        ("at the floor point", float(solution.irradiance_direct["floor"][0])),
    ):
        assert landed / unshadowed == pytest.approx(1.0 - projected, abs=4e-3), name
