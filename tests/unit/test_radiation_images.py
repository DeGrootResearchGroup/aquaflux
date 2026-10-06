"""One specular bounce into the volume: mirror images of the sources, seen through the mirror."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import images as images_module
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.gather import direct_fluence_rate, direct_irradiance
from aquaflux.radiation.images import (
    _seen_through_aperture,
    mirrored_fluence_rate,
    mirrored_irradiance,
    plane_exchange,
    summed_mirrored_fluence_rate,
)
from aquaflux.radiation.mirror_visibility import build_mirror_visibility
from aquaflux.radiation.mirrors import Mirror, planar_mirrors
from aquaflux.radiation.photometry import PhotometricProfile
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.quadrature import triangle_quadrature
from aquaflux.radiation.self_occlusion import NoOcclusion
from aquaflux.radiation.silhouette import source_view
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer
from aquaflux.solids import Box, Sphere

from tests.unit.radiation_references import (
    axial_rectangle_solid_angle,
    inward_box,
    rectangle_triangles,
)

#: The mirror of most fixtures: the plane z = 0, facing +z, as two triangles of a square.
FLOOR_HALF = 10.0


def _scene(
    sources,
    *,
    mirror_half=FLOOR_HALF,
    specular=(0.8, 0.8),
    emission=0.0,
    power=0.0,
    profiles=None,
    profile_index=0,
):
    """A square floor mirror at z = 0 (two triangles, facing up), then ``sources``.

    The optics given are the sources' alone; the floor emits nothing and reflects only
    specularly, with ``specular`` on its two triangles. The floor is Lambertian, the first entry
    of the catalogue, and ``profile_index`` indexes ``profiles`` after it.
    """
    floor = rectangle_triangles([0.0, 0.0, 0.0], [mirror_half, 0.0, 0.0], [0.0, mirror_half, 0.0])
    n_sources = len(sources)

    def padded(value):
        return np.concatenate([np.zeros(2), np.broadcast_to(value, (n_sources,))])

    surfaces = Surfaces.from_triangles(
        np.concatenate([floor, sources]),
        solid_id=[0, 0] + [1] * n_sources,
        solid_names=("floor", "lamp"),
        emission=padded(emission),
        power=padded(power),
        specular_reflectance=np.concatenate([specular, np.zeros(n_sources)]),
        profiles=(Lambertian(), *(profiles or (Lambertian(),))),
        profile_index=np.concatenate([[0, 0], 1 + np.broadcast_to(profile_index, (n_sources,))]),
    )
    return surfaces, planar_mirrors(surfaces, ["floor"])


def _asymmetric_table() -> PhotometricProfile:
    """A measured-style table whose intensity varies as ``sin h`` round the facet normal."""
    horizontal = jnp.linspace(0.0, 2.0 * jnp.pi, 9)
    vertical = jnp.linspace(0.0, jnp.pi / 2, 7)
    table = (1.0 + 0.6 * jnp.sin(horizontal))[:, None] * jnp.cos(vertical)[None, :] ** 2
    return PhotometricProfile(
        vertical=vertical, horizontal=horizontal, table=table, up=jnp.asarray([0.3, 1.0, 0.2])
    )


def _point_lamp(position, power, **extra):
    return _scene(
        np.full((1, 3, 3), position),
        power=power,
        profiles=(Isotropic(),),
        profile_index=0,
        **extra,
    )


# ---------------------------------------------------------------------------------------
# A point lamp beside a mirror: the image-source closed form
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("coefficient", [0.0, 0.7])
def test_a_point_lamp_s_image_is_its_power_over_the_unfolded_distance_squared(coefficient):
    """``rho_s P exp(-a r') / (4 pi r'^2)``, with ``r'`` the distance to the image. Swept over
    distance and medium, so an ``r``-for-``r^2`` slip or a missing attenuation cannot hide."""
    lamp = np.array([0.3, 0.2, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0)
    points = np.array([[0.5, -0.4, 0.6], [1.0, 1.7, 2.0], [-2.0, 0.5, 4.5]])
    absorption = UniformAbsorption(coefficient) if coefficient else None
    field = mirrored_fluence_rate(surfaces, mirrors, points, absorption=absorption)
    image = lamp * np.array([1.0, 1.0, -1.0])
    unfolded = np.linalg.norm(points - image, axis=1)
    expected = 0.8 * 10.0 * np.exp(-coefficient * unfolded) / (4.0 * np.pi * unfolded**2)
    np.testing.assert_allclose(field, expected, rtol=1e-13)


def test_a_point_image_is_seen_only_where_the_line_to_it_crosses_the_mirror():
    """Past the mirror's edge the line to the image crosses no facet, and nothing arrives."""
    lamp = np.array([0.0, 0.0, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0, mirror_half=1.0)
    # Reflection points at x = 0.5 (inside), 1.5 (outside) and 1.0 + 1e-9 (just outside).
    points = np.array([[1.0, 0.0, 1.0], [3.0, 0.0, 1.0], [2.0 + 2e-9, 0.0, 1.0]])
    field = np.asarray(mirrored_fluence_rate(surfaces, mirrors, points))
    assert field[0] > 0.0
    np.testing.assert_array_equal(field[1:], 0.0)


def test_a_line_through_the_edge_two_mirror_facets_share_is_counted_once():
    """The lamp over the square's centre and the receiver on its diagonal put the reflection point
    exactly on the edge the two triangles share -- the layout a symmetric reactor gives. It is
    credited to one triangle, not both, and not neither."""
    lamp = np.array([0.0, 0.0, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0, mirror_half=1.0)
    points = np.array([[0.4, 0.4, 1.0], [-0.5, -0.5, 2.0], [0.0, 0.0, 3.0]])
    field = mirrored_fluence_rate(surfaces, mirrors, points)
    unfolded = np.linalg.norm(points - np.array([0.0, 0.0, -1.0]), axis=1)
    np.testing.assert_allclose(field, 0.8 * 10.0 / (4.0 * np.pi * unfolded**2), rtol=1e-13)


def test_each_mirror_facet_reflects_by_its_own_specular_reflectance():
    """The floor's two triangles meet along the diagonal x = y; a reflection point on either side
    of it reads that triangle's reflectance."""
    lamp = np.array([0.0, 0.0, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0, mirror_half=1.0, specular=(0.3, 0.9))
    points = np.array([[1.0, -0.6, 1.0], [-0.6, 1.0, 1.0]])
    field = np.asarray(mirrored_fluence_rate(surfaces, mirrors, points))
    unfolded = np.linalg.norm(points - np.array([0.0, 0.0, -1.0]), axis=1)
    reflectance = field * 4.0 * np.pi * unfolded**2 / 10.0
    assert sorted(np.round(reflectance, 12)) == [0.3, 0.9]


def test_no_image_reaches_a_receiver_behind_the_mirror_or_comes_from_a_lamp_behind_it():
    surfaces, mirrors = _point_lamp(np.array([0.3, 0.2, 1.0]), 10.0)
    behind = mirrored_fluence_rate(surfaces, mirrors, np.array([[0.5, 0.5, -0.5]]))
    np.testing.assert_array_equal(behind, 0.0)
    surfaces, mirrors = _point_lamp(np.array([0.3, 0.2, -1.0]), 10.0)
    from_behind = mirrored_fluence_rate(surfaces, mirrors, np.array([[0.5, 0.5, 0.5]]))
    np.testing.assert_array_equal(from_behind, 0.0)


# ---------------------------------------------------------------------------------------
# Areal sources: the image clipped to the aperture
# ---------------------------------------------------------------------------------------


def _plate(half: float, height: float, *, facing_down: bool = True) -> np.ndarray:
    """A square of half-width ``half`` at ``z = height``, centred on the axis."""
    plate = rectangle_triangles([0.0, 0.0, height], [half, 0.0, 0.0], [0.0, half, 0.0])
    return plate[:, ::-1] if facing_down else plate


def test_an_image_wider_than_the_mirror_is_seen_exactly_through_the_mirror_s_outline():
    """A Lambertian plate facing a parallel square mirror, the receiver on the axis between them.
    Every direction through the mirror reaches the plate's image, so the field is the image's
    radiance times the solid angle of the mirror itself -- the aperture limits, not the source."""
    height, depth, mirror_half = 0.4, 1.0, 0.25
    exitance, specular = 3.0, 0.6
    surfaces, mirrors = _scene(
        _plate(5.0, depth),
        mirror_half=mirror_half,
        specular=(specular, specular),
        emission=exitance,
    )
    field = mirrored_fluence_rate(surfaces, mirrors, np.array([[0.0, 0.0, height]]))
    seen = axial_rectangle_solid_angle(mirror_half, mirror_half, height)
    np.testing.assert_allclose(field, specular * exitance / np.pi * seen, rtol=1e-12)


def test_an_image_narrower_than_the_mirror_is_seen_whole():
    """The other limit: through a large mirror the whole image is seen, at the unfolded distance."""
    height, depth, half = 0.4, 1.0, 0.3
    exitance, specular = 3.0, 0.6
    surfaces, mirrors = _scene(
        _plate(half, depth), specular=(specular, specular), emission=exitance
    )
    field = mirrored_fluence_rate(surfaces, mirrors, np.array([[0.0, 0.0, height]]))
    seen = axial_rectangle_solid_angle(half, half, height + depth)
    np.testing.assert_allclose(field, specular * exitance / np.pi * seen, rtol=1e-12)


def test_through_an_unbounded_mirror_an_image_lights_as_its_source_would_from_the_reflection():
    """A mirror larger than everything the receivers can see through it, of reflectance one, is
    invisible: what arrives is the image's direct field, so this compares against the direct
    gather of the reflected set -- over profiles of every kind and receivers off any axis. The
    measured table varies round its axis, so a source gathered with its own distribution rather
    than its mirror image's would light the receivers the wrong way round."""
    rng = np.random.default_rng(2)
    triangles = rng.uniform(-0.4, 0.4, size=(5, 3, 3)) + np.array([0.0, 0.0, 1.2])
    profiles = (Lambertian(), CosinePower(4.0), _asymmetric_table())
    index = [0, 1, 2, 1, 2]
    surfaces, mirrors = _scene(
        triangles,
        mirror_half=1e3,
        specular=(1.0, 1.0),
        emission=[1.0, 2.0, 0.5, 1.5, 3.0],
        profiles=profiles,
        profile_index=index,
    )
    points = rng.uniform(-1.0, 1.0, size=(30, 3)) + np.array([0.0, 0.0, 1.5])
    lamp_only = Surfaces.from_triangles(
        triangles, emission=surfaces.emission[2:], profiles=profiles, profile_index=index
    )
    (mirror,) = mirrors
    expected = direct_fluence_rate(mirror.image(lamp_only), points)
    field = mirrored_fluence_rate(surfaces, mirrors, points)
    assert float(jnp.min(expected)) > 0.0
    np.testing.assert_allclose(field, expected, rtol=1e-11)


def test_only_the_part_of_a_source_in_front_of_the_mirror_has_an_image():
    """A wall standing across the mirror's plane, half above it and half below: its image through
    the mirror is the image of its upper half alone, which is what the clip's depth cut gives."""
    wall = rectangle_triangles([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0])[:, ::-1]
    upper = rectangle_triangles([1.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5])[:, ::-1]
    points = np.array([[0.3, 0.1, 0.4], [-0.5, -0.2, 1.1], [0.6, 0.3, 0.2]])
    straddling, mirrors = _scene(wall, emission=2.0)
    above, mirrors_above = _scene(upper, emission=2.0)
    field = mirrored_fluence_rate(straddling, mirrors, points)
    expected = mirrored_fluence_rate(above, mirrors_above, points)
    assert float(jnp.min(expected)) > 0.0
    np.testing.assert_allclose(field, expected, rtol=1e-12)


def test_the_two_facets_reflectances_weight_the_parts_of_an_image_each_one_shows():
    """The field is linear in each mirror facet's reflectance, and each facet shows a part."""
    sources = _plate(0.6, 1.0)
    points = np.array([[0.2, -0.1, 0.5], [-0.3, 0.4, 0.8]])

    def field(specular):
        surfaces, mirrors = _scene(sources, mirror_half=0.5, specular=specular, emission=2.0)
        return np.asarray(mirrored_fluence_rate(surfaces, mirrors, points))

    first, second = field((1.0, 0.0)), field((0.0, 1.0))
    assert np.all(first > 0.0) and np.all(second > 0.0)
    np.testing.assert_allclose(field((0.3, 0.7)), 0.3 * first + 0.7 * second, rtol=1e-12)


# ---------------------------------------------------------------------------------------
# The medium, the sets, and what is live
# ---------------------------------------------------------------------------------------


def test_a_graded_medium_is_crossed_along_the_two_real_legs():
    """With a coefficient rising linearly in height, the unfolded straight line and the real
    broken path cross different water. The reference integrates the linear field along each real
    leg in closed form -- its mean times its length -- so it shares nothing with the voxel walk."""
    lamp = np.array([0.3, 0.2, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0)
    # Samples at heights -0.75 to 4.75, so every path (heights 0 to 2) lies where the
    # interpolated field is the linear one rather than clamped.
    heights = -1.0 + 0.5 * (np.arange(12) + 0.5)
    medium = VoxelAbsorption(
        np.broadcast_to(0.2 + 0.5 * heights, (3, 3, 12)),
        origin=[-3.0, -3.0, -1.0],
        spacing=[2.0, 2.0, 0.5],
    )
    points = np.array([[0.5, -0.4, 0.6], [1.0, 1.7, 2.0]])
    field = np.asarray(mirrored_fluence_rate(surfaces, mirrors, points, absorption=medium))

    def leg(a, b):
        return np.linalg.norm(b - a) * (0.2 + 0.5 * (a[2] + b[2]) / 2.0)

    image = lamp * np.array([1.0, 1.0, -1.0])
    for point, value in zip(points, field, strict=True):
        share = point[2] / (point[2] - image[2])
        crossing = point + share * (image - point)
        depth = leg(point, crossing) + leg(crossing, lamp)
        unfolded = np.linalg.norm(point - image)
        expected = 0.8 * 10.0 * np.exp(-depth) / (4.0 * np.pi * unfolded**2)
        np.testing.assert_allclose(value, expected, rtol=1e-12)


def test_summing_sets_in_one_pass_is_summing_their_fields():
    triangles = _plate(0.4, 1.0)
    surfaces, mirrors = _scene(triangles, emission=1.5)
    other = surfaces.with_optics(
        emission=np.concatenate([np.zeros(2), [0.5, 2.5]]),
        profiles=(CosinePower(3.0),),
    )
    points = np.array([[0.2, 0.1, 0.5], [-0.4, 0.3, 1.4]])
    together = summed_mirrored_fluence_rate((surfaces, other), mirrors, points)
    apart = mirrored_fluence_rate(surfaces, mirrors, points) + mirrored_fluence_rate(
        other, mirrors, points
    )
    np.testing.assert_allclose(together, apart, rtol=1e-13)


def test_the_specular_reflectance_and_the_emission_are_live():
    """Both multiply the clipped geometry, so the field is linear in each and its derivative is
    the field at one: exact, and non-zero, which a severed derivative would not be."""
    surfaces, mirrors = _scene(_plate(0.4, 1.0), emission=2.0)
    points = np.array([[0.1, 0.2, 0.5]])

    def total(specular, emission):
        optics = surfaces.with_optics(
            specular_reflectance=jnp.concatenate([specular, jnp.zeros(2)]),
            emission=jnp.concatenate([jnp.zeros(2), emission]),
        )
        return jnp.sum(mirrored_fluence_rate(optics, mirrors, points))

    specular, emission = jnp.array([0.4, 0.4]), jnp.array([2.0, 2.0])
    value = total(specular, emission)
    d_specular, d_emission = jax.grad(total, argnums=(0, 1))(specular, emission)
    assert float(value) > 0.0
    np.testing.assert_allclose(jnp.sum(d_specular * specular), value, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(d_emission * emission), value, rtol=1e-12)


def test_how_the_work_is_cut_into_passes_does_not_change_the_answer():
    rng = np.random.default_rng(4)
    surfaces, mirrors = _scene(_plate(0.4, 1.0), mirror_half=0.5, emission=2.0)
    points = rng.uniform(-0.5, 0.5, size=(37, 3)) + np.array([0.0, 0.0, 0.8])
    whole = mirrored_fluence_rate(surfaces, mirrors, points)
    cut = mirrored_fluence_rate(surfaces, mirrors, points, pair_limit=7)
    np.testing.assert_allclose(cut, whole, rtol=1e-13)


def test_a_mirror_from_another_surface_set_is_refused():
    _, (mirror,) = _scene(_plate(0.4, 1.0), emission=2.0)
    smaller = Surfaces.from_triangles(_plate(0.4, 1.0)[:1], emission=2.0)
    with pytest.raises(ValueError, match="it was found in a different set"):
        mirrored_fluence_rate(smaller, [mirror], np.zeros((1, 3)))


def test_each_mirror_adds_its_own_image():
    """A floor and a wall meeting in a corner, of different reflectances: a point lamp in the corner
    is seen in each, and with one bounce only the field is the sum of the two images."""
    floor = rectangle_triangles([10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0])
    wall = rectangle_triangles([0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0])
    lamp = np.array([1.0, 0.2, 1.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([floor, wall, np.full((1, 3, 3), lamp)]),
        solid_id=[0, 0, 1, 1, 2],
        solid_names=("floor", "wall", "lamp"),
        power=[0.0] * 4 + [10.0],
        specular_reflectance=[0.8, 0.8, 0.5, 0.5, 0.0],
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0, 0, 0, 0, 1],
    )
    mirrors = planar_mirrors(surfaces, ["floor", "wall"])
    points = np.array([[2.0, -0.3, 1.5], [0.7, 0.9, 2.4]])
    field = mirrored_fluence_rate(surfaces, mirrors, points)
    in_floor = np.linalg.norm(points - lamp * np.array([1.0, 1.0, -1.0]), axis=1)
    in_wall = np.linalg.norm(points - lamp * np.array([-1.0, 1.0, 1.0]), axis=1)
    expected = 10.0 / (4.0 * np.pi) * (0.8 / in_floor**2 + 0.5 / in_wall**2)
    np.testing.assert_allclose(field, expected, rtol=1e-13)


def test_traced_receivers_get_what_concrete_ones_do_including_behind_the_mirror():
    """Which receivers and sources lie in front of a mirror is read off concrete positions only, to
    save work; traced, every pair is gathered, and the clip and the crossing test must then give
    the zeros the reading would have skipped -- for a receiver behind the mirror, and for a lamp
    behind it shining through from the wrong side."""
    triangles = np.concatenate(
        [
            _plate(0.3, 1.0),
            np.full((1, 3, 3), [0.2, 0.1, 0.8]),
            np.full((1, 3, 3), [0.1, 0.0, -0.7]),
        ]
    )
    surfaces, mirrors = _scene(
        triangles,
        emission=[2.0, 2.0, 0.0, 0.0],
        power=[0.0, 0.0, 5.0, 7.0],
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0, 0, 1, 1],
    )
    points = np.array([[0.1, 0.2, 0.5], [0.3, -0.2, 1.4], [0.2, 0.1, -0.4]])
    concrete = mirrored_fluence_rate(surfaces, mirrors, points)
    traced = jax.jit(lambda at: mirrored_fluence_rate(surfaces, mirrors, at))(points)
    assert float(concrete[2]) == 0.0
    assert float(jnp.min(concrete[:2])) > 0.0
    np.testing.assert_allclose(traced, concrete, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------------------
# Irradiance on oriented receivers
# ---------------------------------------------------------------------------------------


def _oriented(count, seed):
    """Points above the floor with unit normals in every direction, some facing away."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(count, 3)) + np.array([0.0, 0.0, 1.5])
    normals = rng.normal(size=(count, 3))
    return points, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def test_a_point_lamp_s_image_lands_by_the_cosine_at_the_receiver():
    """``rho_s P cos / (4 pi r'^2)``, with the cosine between the receiver's normal and the
    direction back to the image; a receiver facing away gets nothing."""
    lamp = np.array([0.3, 0.2, 1.0])
    surfaces, mirrors = _point_lamp(lamp, 10.0)
    points = np.array([[0.5, -0.4, 0.6], [1.0, 1.7, 2.0], [-2.0, 0.5, 4.5]])
    image = lamp * np.array([1.0, 1.0, -1.0])
    towards = (image - points) / np.linalg.norm(image - points, axis=1, keepdims=True)
    tilted = towards + np.array([[0.3, 0.0, 0.0], [0.0, -0.4, 0.1], [0.2, 0.2, 0.0]])
    normals = np.concatenate([tilted / np.linalg.norm(tilted, axis=1, keepdims=True), -towards])
    at = np.concatenate([points, points])
    field = np.asarray(mirrored_irradiance(surfaces, mirrors, at, normals))
    unfolded = np.linalg.norm(points - image, axis=1)
    cosine = np.sum(normals[:3] * towards, axis=1)
    expected = 0.8 * 10.0 * cosine / (4.0 * np.pi * unfolded**2)
    np.testing.assert_allclose(field[:3], expected, rtol=1e-13)
    np.testing.assert_array_equal(field[3:], 0.0)


def test_through_an_unbounded_mirror_the_irradiance_is_the_image_s_direct_irradiance():
    """The irradiance counterpart of the unbounded-mirror test: through a mirror covering every
    direction, of reflectance one, the image is seen whole, so the irradiance it gives is the
    direct irradiance of the reflected set -- areal facets and a point source together, on
    receivers facing every way."""
    rng = np.random.default_rng(6)
    triangles = rng.uniform(-0.4, 0.4, size=(4, 3, 3)) + np.array([0.0, 0.0, 1.2])
    sources = np.concatenate([triangles, np.full((1, 3, 3), [0.2, -0.1, 0.9])])
    profiles = (Lambertian(), CosinePower(4.0), Isotropic())
    index = [0, 1, 0, 1, 2]
    emission, power = [1.0, 2.0, 0.5, 1.5, 0.0], [0.0] * 4 + [6.0]
    surfaces, mirrors = _scene(
        sources,
        mirror_half=1e3,
        specular=(1.0, 1.0),
        emission=emission,
        power=power,
        profiles=profiles,
        profile_index=index,
    )
    lamp_only = Surfaces.from_triangles(
        sources, emission=emission, power=power, profiles=profiles, profile_index=index
    )
    points, normals = _oriented(25, seed=7)
    (mirror,) = mirrors
    expected = direct_irradiance(mirror.image(lamp_only), points, normals)
    field = mirrored_irradiance(surfaces, mirrors, points, normals)
    assert float(jnp.max(expected)) > 0.0 and float(jnp.min(expected)) == 0.0
    np.testing.assert_allclose(field, expected, rtol=1e-11, atol=1e-14)

    only_points = mirrored_irradiance(surfaces, mirrors, points, normals, point_sources_only=True)
    lamp_point = lamp_only.with_optics(emission=np.zeros(5))
    expected_points = direct_irradiance(mirror.image(lamp_point), points, normals)
    assert float(jnp.max(expected_points)) > 0.0
    np.testing.assert_allclose(only_points, expected_points, rtol=1e-11, atol=1e-14)


# ---------------------------------------------------------------------------------------
# Facet-to-facet exchange through one mirror
# ---------------------------------------------------------------------------------------


def _half_box_and_its_reflection():
    """The half ``x <= 0.5`` of a closed unit box with a mirror across ``x = 0.5``, and the whole box
    made by reflecting that half -- triangle for triangle, so the two halves are mirror images and
    not merely translates."""
    walls = inward_box(2)
    left = walls[np.all(walls[:, :, 0] <= 0.5, axis=1)]
    right = (left * np.array([-1.0, 1.0, 1.0]) + np.array([1.0, 0.0, 0.0]))[:, ::-1]
    mirror = rectangle_triangles([0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0])
    half = Surfaces.from_triangles(
        np.concatenate([left, mirror]),
        solid_id=[0] * len(left) + [1, 1],
        solid_names=("walls", "mirror"),
    )
    whole = Surfaces.from_triangles(np.concatenate([left, right]))
    return half, whole, len(left)


def test_a_mirror_across_a_symmetric_box_exchanges_exactly_what_the_other_half_would():
    """From a facet in the half box, a facet's image through the mirror is the corresponding facet
    of the other half, seen whole: the exchange equals the whole box's transfer to that facet, to
    rounding, at every pair -- and with it every row of the half box sums to one, as the whole
    box's do."""
    half, whole, n = _half_box_and_its_reflection()
    (mirror,) = planar_mirrors(half, ["mirror"])
    np.testing.assert_allclose(np.asarray(mirror.normal), [-1.0, 0.0, 0.0], atol=1e-15)
    rule = triangle_quadrature(6)
    exchange = plane_exchange(mirror, half, rule.points(half.vertices), rule.weight)
    direct = build_transfer(whole, self_occlusion=NoOcclusion()).geometric
    np.testing.assert_allclose(exchange.geometric[:n, :n], direct[:n, n:], atol=1e-15)
    within = build_transfer(half, self_occlusion=NoOcclusion()).geometric[:n, :n]
    np.testing.assert_allclose(
        jnp.sum(within + exchange.geometric[:n, :n], axis=1), 1.0, rtol=1e-13
    )
    # Nothing is exchanged with or through the mirror's own facets, which lie in its plane.
    np.testing.assert_array_equal(exchange.geometric[n:], 0.0)
    np.testing.assert_array_equal(exchange.geometric[:, n:], 0.0)


def test_the_exchange_s_path_lengths_and_cosines_are_the_unfolded_ones():
    """Distance and source cosine are measured to the image, as the light's unfolded path runs."""
    half, whole, n = _half_box_and_its_reflection()
    (mirror,) = planar_mirrors(half, ["mirror"])
    rule = triangle_quadrature(1)
    exchange = plane_exchange(mirror, half, rule.points(half.vertices), rule.weight)
    direct = build_transfer(whole, receiver_quadrature=1, self_occlusion=NoOcclusion())
    np.testing.assert_allclose(exchange.separation[:n, :n], direct.separation[:n, n:], atol=1e-15)
    np.testing.assert_allclose(
        exchange.source_cosine[:n, :n], direct.source_cosine[:n, n:], atol=1e-15
    )


# ---------------------------------------------------------------------------------------
# Shadows on the reflected paths
# ---------------------------------------------------------------------------------------


def test_a_shadowed_image_is_filtered_once_per_leg_a_body_crosses():
    """A slab both legs pass through, and a ball on the first leg of one receiver's path only: the
    image reaches that receiver times the slab's transmittance squared times the ball's, the
    other times the slab's squared -- for the fluence rate and the irradiance alike."""
    surfaces, mirrors = _point_lamp([0.0, 0.0, 1.0], 4.0)
    points = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    bodies = [Box((0.0, 0.0, 0.5), (5.0, 5.0, 0.1)), Sphere((0.25, 0.0, 0.5), 0.1)]
    masks = [
        build_mirror_visibility(mirror, bodies, surfaces, points, self_occlusion=NoOcclusion())
        for mirror in mirrors
    ]
    expected = np.array([0.5**2 * 0.25, 0.5**2])
    shadows = {"shadows": masks, "transmittance": [0.5, 0.25]}
    np.testing.assert_allclose(
        mirrored_fluence_rate(surfaces, mirrors, points, **shadows),
        expected * mirrored_fluence_rate(surfaces, mirrors, points),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        mirrored_irradiance(surfaces, mirrors, points, normals, **shadows),
        expected * mirrored_irradiance(surfaces, mirrors, points, normals),
        rtol=1e-14,
    )
    # Given no transmittance, every body is opaque, as for the direct gather.
    np.testing.assert_array_equal(
        mirrored_fluence_rate(surfaces, mirrors, points, shadows=masks), [0.0, 0.0]
    )


def test_an_areal_image_is_hidden_by_the_surface_s_own_triangles():
    """A plate across the second leg of one receiver's path hides the lamp patch's image from it
    and from no other receiver."""
    lamp = rectangle_triangles([0.0, 0.0, 1.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])[:, ::-1]
    plate = rectangle_triangles([0.75, 0.0, 0.5], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
    surfaces, mirrors = _scene(np.concatenate([lamp, plate]), emission=[1.0, 1.0, 0.0, 0.0])
    points = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    masks = [build_mirror_visibility(mirror, [], surfaces, points) for mirror in mirrors]
    shadowed = mirrored_fluence_rate(surfaces, mirrors, points, shadows=masks)
    clear = mirrored_fluence_rate(surfaces, mirrors, points)
    assert float(clear[0]) > 0.0
    np.testing.assert_allclose(shadowed, [0.0, clear[1]], rtol=1e-14, atol=0.0)


def test_shadows_must_be_one_mask_per_mirror_and_come_with_them_the_transmittance():
    surfaces, mirrors = _point_lamp([0.0, 0.0, 1.0], 4.0)
    points = np.array([[1.0, 0.0, 1.0]])
    mask = build_mirror_visibility(mirrors[0], [], surfaces, points)
    with pytest.raises(ValueError, match="give one each"):
        mirrored_fluence_rate(surfaces, mirrors, points, shadows=[mask, mask])
    with pytest.raises(ValueError, match="without mirror visibility masks"):
        mirrored_fluence_rate(surfaces, mirrors, points, transmittance=[0.5])


# ---------------------------------------------------------------------------------------
# The cone cull: each image clipped only against the mirror facets it overlaps
# ---------------------------------------------------------------------------------------


def _fine_aperture(cells: int = 10, *, hole: bool = False) -> np.ndarray:
    """The square [-1, 1]^2 at z = 0, facing up, as ``2 * cells**2`` triangles -- less the
    middle cell's two with ``hole``, so the outline has an inner edge as well as a rim."""
    edges = np.linspace(-1.0, 1.0, cells + 1)
    step = edges[1] - edges[0]
    middle = cells // 2
    return np.concatenate(
        [
            rectangle_triangles(
                [x + step / 2, y + step / 2, 0.0], [step / 2, 0, 0], [0, step / 2, 0]
            )
            for i, x in enumerate(edges[:-1])
            for j, y in enumerate(edges[:-1])
            if not (hole and i == middle and j == middle)
        ]
    )


def _floor_path(aperture, weights):
    """The gather's view of a mirror in the plane z = 0 whose facets are ``aperture``."""
    mirror = Mirror(
        point=jnp.zeros(3), normal=jnp.asarray([0.0, 0.0, 1.0]), facets=np.arange(len(aperture))
    )
    return images_module._Path.of(mirror, jnp.asarray(aperture), jnp.asarray(weights))


def _images_and_receivers(seed: int, n_images: int = 7, n_receivers: int = 9):
    """Small image triangles behind the plane z = 0, receivers in front, some with normals.

    Some images lie well inside the mirror, some across its rim or the hole, some outside it.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform([-1.2, -1.2, -1.0], [1.2, 1.2, -0.2], (n_images, 1, 3))
    images = centres + rng.normal(scale=0.08, size=(n_images, 3, 3))
    receivers = rng.uniform([-1.2, -1.2, 0.05], [1.2, 1.2, 1.0], (n_receivers, 3))
    # The first receiver sees the first image through the first facet: the triple a padded
    # batch slot points at carries light, so padding counted as a survivor would show.
    images[0] = images[0] - centres[0] + np.array([-0.9, -0.9, -0.5])
    receivers[0] = [-0.9, -0.9, 0.5]
    # The second image sits under the hole's middle, seen through it by the second receiver.
    images[1] = images[1] - centres[1] + np.array([0.1, 0.1, -0.5])
    receivers[1] = [0.1, 0.1, 0.4]
    normals = rng.normal(size=(n_receivers, 3))
    normals[:, 2] = -np.abs(normals[:, 2])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return jnp.asarray(images), jnp.asarray(receivers), jnp.asarray(normals)


@pytest.mark.parametrize("hole", [False, True], ids=["whole", "holed"])
@pytest.mark.parametrize("weighting", ["one weight", "per facet"])
@pytest.mark.parametrize("oriented", [False, True], ids=["volume", "on a surface"])
def test_culling_and_skipping_the_clip_change_no_answer(oriented, weighting, hole, monkeypatch):
    """The culled clip against the clip of every facet, with a batch small enough that the
    survivors take several passes. With one weight on every facet, images wholly inside the
    outline skip the clip; with a weight per facet -- which a facet swapped for another would
    change -- none may, and every image is clipped."""
    monkeypatch.setattr(images_module, "_CLIP_BATCH", 7)
    aperture = _fine_aperture(hole=hole)
    rng = np.random.default_rng(1)
    weights = (
        np.full(len(aperture), 0.6)
        if weighting == "one weight"
        else rng.uniform(0.2, 1.0, len(aperture))
    )
    path = _floor_path(aperture, weights)
    images, receivers, normals = _images_and_receivers(0)
    normals = normals if oriented else None
    every = _seen_through_aperture(path, receivers, normals, images, cull=False)
    culled = _seen_through_aperture(path, receivers, normals, images, cull=True)
    assert float(jnp.min(every)) >= 0.0 and float(jnp.max(every)) > 0.0
    # The every-facet clip leaves last-bit noise on facets that show nothing, so the two agree
    # to a rounding of the largest value rather than bit for bit.
    np.testing.assert_allclose(culled, every, rtol=1e-12, atol=1e-13 * float(jnp.max(every)))


def test_an_image_wholly_inside_the_mirror_needs_no_clip(monkeypatch):
    """With the clip itself stubbed out to clip nothing, the pairs the screen lets skip it still
    get the exact answer -- and they are most of them -- while an image under the hole does not
    skip, and gets nothing."""
    aperture = _fine_aperture(hole=True)
    path = _floor_path(aperture, np.full(len(aperture), 0.6))
    images, receivers, _ = _images_and_receivers(0)
    every = np.asarray(_seen_through_aperture(path, receivers, None, images, cull=False))
    _, (skip, _) = images_module._screen(path, receivers, images)
    skip = np.asarray(skip)
    monkeypatch.setattr(
        images_module,
        "_culled_fractions",
        lambda view, aperture, receivers, normals, kept: jnp.zeros(kept.shape),
    )
    stubbed = np.asarray(_seen_through_aperture(path, receivers, None, images, cull=True))
    assert skip.mean() > 0.4
    assert not skip[1, 1]
    np.testing.assert_allclose(stubbed[skip], every[skip], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(stubbed[~skip], 0.0)


def test_an_image_crossing_the_mirror_s_plane_is_clipped_however_far_inside_it_lies():
    """The part of an image in front of the plane is not seen through the mirror, so an image that
    crosses the plane is clipped even where no edge of the outline comes near it."""
    aperture = _fine_aperture()
    path = _floor_path(aperture, np.full(len(aperture), 0.6))
    images = jnp.asarray([[[-0.1, -0.1, -0.1], [0.1, -0.1, 0.1], [0.0, 0.1, -0.1]]])
    receivers = jnp.asarray([[0.0, 0.0, 0.8], [0.05, -0.02, 0.5]])
    _, (skip, _) = images_module._screen(path, receivers, images)
    every = np.asarray(_seen_through_aperture(path, receivers, None, images, cull=False))
    culled = _seen_through_aperture(path, receivers, None, images, cull=True)
    whole = np.abs(np.asarray(source_view(receivers[:, None], None, images[None]).whole))
    assert not np.any(np.asarray(skip))
    assert np.all(every > 0.0) and np.all(every < 0.9 * 0.6 * whole)
    np.testing.assert_allclose(culled, every, rtol=1e-12, atol=1e-15)


def test_the_screen_keeps_every_facet_that_shows_part_of_an_image_and_few_others():
    """Conservative -- each facet with a non-zero share is kept -- and worth having: on a mirror
    of 200 triangles a small image needs a small share of them."""
    aperture = _fine_aperture()
    images, receivers, _ = _images_and_receivers(2, n_images=5, n_receivers=6)
    kept, _ = images_module._screen(
        _floor_path(aperture, np.ones(len(aperture))), receivers, images
    )
    kept = np.asarray(kept)
    shows = np.stack(
        [
            np.asarray(
                _seen_through_aperture(
                    _floor_path(aperture, np.eye(len(aperture))[k]),
                    receivers,
                    None,
                    images,
                    cull=False,
                )
            )
            for k in range(len(aperture))
        ],
        axis=-1,
    )
    assert np.any(shows > 0.0)
    assert np.all(kept[shows > 0.0])
    assert kept.mean() < 0.25


def test_the_culled_clip_clips_only_what_the_screen_keeps(monkeypatch):
    """With a screen that keeps nothing and lets nothing skip, nothing is seen: the culled clip
    reads its survivors from the screen rather than clipping every facet anyway."""
    monkeypatch.setattr(
        images_module,
        "_screen",
        lambda path, receivers, vertices: (
            jnp.zeros(
                (*receivers.shape[:-1], vertices.shape[-3], path.aperture.shape[0]), dtype=bool
            ),
            None,
        ),
    )
    aperture = _fine_aperture(4)
    path = _floor_path(aperture, np.ones(len(aperture)))
    images, receivers, _ = _images_and_receivers(4)
    every = _seen_through_aperture(path, receivers, None, images, cull=False)
    culled = _seen_through_aperture(path, receivers, None, images, cull=True)
    assert float(jnp.max(every)) > 0.0
    np.testing.assert_array_equal(culled, 0.0)


def test_the_reflectances_stay_differentiable_through_the_culled_clip():
    """Each facet's derivative is the share of the images seen through it -- with one weight on
    every facet as well, where a skipped image would credit its whole share to the one facet the
    line to its centroid crosses. A traced reflectance therefore keeps no outline."""
    aperture = _fine_aperture(4)
    images, receivers, _ = _images_and_receivers(3)
    mirror = _floor_path(aperture, np.ones(len(aperture))).mirror

    def total(weights, cull):
        path = images_module._Path.of(mirror, jnp.asarray(aperture), weights)
        return jnp.sum(_seen_through_aperture(path, receivers, None, images, cull=cull))

    weights = jnp.full(len(aperture), 0.7)
    culled = jax.grad(total)(weights, True)
    assert float(jnp.max(jnp.abs(culled))) > 0.0
    np.testing.assert_allclose(culled, jax.grad(total)(weights, False), rtol=1e-12, atol=1e-15)


def test_only_one_concrete_weight_on_every_facet_lets_an_image_skip_its_clip():
    """The outline the skip tests against is kept for one concrete weight, and for nothing else."""
    aperture = _fine_aperture(2)
    mirror = _floor_path(aperture, np.ones(len(aperture))).mirror
    of = images_module._Path.of
    assert of(mirror, aperture, np.full(len(aperture), 0.4)).outline is not None
    assert of(mirror, aperture, np.linspace(0.2, 0.4, len(aperture))).outline is None
    held = []
    jax.grad(lambda w: held.append(of(mirror, aperture, w).outline) or jnp.sum(w))(
        jnp.ones(len(aperture))
    )
    assert held == [None]
