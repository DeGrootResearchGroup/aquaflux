"""Lamps kept out of the transfer: the scene's light against the model with the lamp inside it.

A Lambertian lamp can be carried either way, which is what makes the comparison: inside the model's
surface set, its light on each facet goes through the transfer; kept out, it is gathered directly and
handed to the surface solve. With one receiver point per facet in the transfer and one sample per
facet in the scene, the two evaluate the same projected solid angles at the same points, so the fields
agree to rounding -- and any slip in the assembly (the lamp's light not reaching the reflectors, the
reflected light gathered with the wrong radiosity, a facet left out) moves them apart by a visible
fraction.
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
    build_radiation_model,
    build_visibility,
    direct_irradiance,
    fluence_rate,
    radiosity,
    solve_scene,
    subtriangle_centroids,
)
from aquaflux.solids import Sphere

from tests.unit.radiation_references import inward_box

LAMP_EXITANCE = 7.0
REFLECTANCE = 0.6
SETTINGS = RadiationSettings(receiver_quadrature=1, self_occlusion=NoOcclusion())


def _box_split(divisions: int = 3):
    """An inward unit box: the facets of its top face (z = 1), and the rest."""
    triangles = inward_box(divisions)
    top = np.all(np.isclose(triangles[:, :, 2], 1.0), axis=1)
    return triangles, top


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
    with pytest.raises(ValueError, match=r"'probe' names 'floor', which is not a reflecting body"):
        Scene(lamps=lamps, reflectors=reflectors, surfaces={"probe": probe})
    with pytest.raises(ValueError, match="lamp_samples must be >= 1"):
        Scene(lamps=lamps, lamp_samples=0)
    with pytest.raises(ValueError, match="reflectance must lie in"):
        SurfaceReceivers(np.zeros((1, 3)), np.ones((1, 3)), reflectance=1.5)
    with pytest.raises(ValueError, match="normals must have shape"):
        SurfaceReceivers(np.zeros((2, 3)), np.ones((1, 3)))
