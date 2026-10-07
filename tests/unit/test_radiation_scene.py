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

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    Isotropic,
    Lambertian,
    NoOcclusion,
    RadiationSettings,
    RayCastOcclusion,
    Scene,
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

from tests.unit.radiation_references import closed_drum, inward_box

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
            surfaces={"floor": SurfaceReceivers(points, normals, reflector="walls")},
            settings=SETTINGS,
        )
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

    def reflected(occlusion, on):
        settings = RadiationSettings(receiver_quadrature=1, self_occlusion=occlusion)
        scene = Scene(
            lamps=lamps,
            reflectors=reflectors,
            surfaces={"floor": SurfaceReceivers(points, up, reflector=on)},
            settings=settings,
        )
        return solve_scene(scene).irradiance_reflected["floor"]

    # A box is convex, so nothing shadows anything: the ray test must change nothing.
    unshadowed = reflected(NoOcclusion(), None)
    np.testing.assert_allclose(reflected(RayCastOcclusion(), "walls"), unshadowed, rtol=1e-12)
    assert np.all(reflected(RayCastOcclusion(), None) < 0.5 * unshadowed)


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
    assert lamp_absorbs > 0.1 * solution.lamp_power
    assert walls_absorb + solution.lamp_absorbed_power == pytest.approx(
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


def test_the_lamps_light_on_a_lamp_is_taken_on_the_fluid_s_side_of_it() -> None:
    # The water as a body -- the box less a cylinder standing inside the drum, its caps on the
    # drum's caps, as a drawing's fluid leaves out its lamp. The points the lamps' light is averaged
    # over on a lamp facet must lie in that water: a hair behind a cap they are inside the lamp,
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
