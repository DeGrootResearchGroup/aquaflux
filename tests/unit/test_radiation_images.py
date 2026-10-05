"""One specular bounce into the volume: mirror images of the sources, seen through the mirror."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.images import mirrored_fluence_rate, summed_mirrored_fluence_rate
from aquaflux.radiation.mirrors import planar_mirrors
from aquaflux.radiation.photometry import PhotometricProfile
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.surfaces import Surfaces

from tests.unit.radiation_references import axial_rectangle_solid_angle, rectangle_triangles

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
