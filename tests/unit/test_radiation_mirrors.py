"""Plane mirrors: grouping a body's facets into the planes they lie in, and the images they form."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.gather import direct_fluence_rate
from aquaflux.radiation.mirrors import Mirror, planar_mirrors
from aquaflux.radiation.photometry import PhotometricProfile
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.surfaces import Surfaces

from tests.unit.radiation_references import cylinder_triangles, inward_box, rectangle_triangles

# ---------------------------------------------------------------------------------------
# Grouping facets into planes
# ---------------------------------------------------------------------------------------


def _heights(mirror: Mirror, surfaces: Surfaces) -> np.ndarray:
    """How far each corner of each of the mirror's facets lies from its plane."""
    corners = np.asarray(surfaces.vertices)[mirror.facets] - np.asarray(mirror.point)
    return corners @ np.asarray(mirror.normal)


def test_a_closed_box_is_one_mirror_per_face_whatever_its_triangulation():
    """Each face is eight triangles; the image a face forms is the same for all of them, so it is
    one mirror -- which is what sets the cost of images by planes rather than by triangles."""
    surfaces = Surfaces.from_triangles(inward_box(2), solid_names=("vessel",))
    mirrors = planar_mirrors(surfaces, ["vessel"])
    assert len(mirrors) == 6
    assert sorted(len(mirror.facets) for mirror in mirrors) == [8] * 6
    np.testing.assert_array_equal(
        np.sort(np.concatenate([mirror.facets for mirror in mirrors])), np.arange(48)
    )
    for mirror in mirrors:
        assert np.abs(_heights(mirror, surfaces)).max() < 1e-14
        # The plane faces the way its facets do, which is into the box.
        np.testing.assert_allclose(
            np.asarray(surfaces.normal)[mirror.facets] @ np.asarray(mirror.normal), 1.0, atol=1e-14
        )
    normals = np.round(np.array([np.asarray(mirror.normal) for mirror in mirrors]), 12)
    assert {tuple(n) for n in normals} == {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    }
    # Ordered by the lowest facet each holds.
    firsts = [int(mirror.facets[0]) for mirror in mirrors]
    assert firsts == sorted(firsts)


@pytest.mark.parametrize("sectors", [12, 180])
def test_a_faceted_cylinder_is_one_mirror_per_flat_strip(sectors):
    """Every strip of a prism is flat, so its slices share one plane; neighbouring strips are not
    coplanar, however fine the faceting. Exactly one mirror per strip means neither merged nor
    split -- at 180 sectors neighbours turn by only two degrees."""
    slices = 5
    tube = cylinder_triangles(0.1, 0.5, sectors=sectors, slices=slices)
    surfaces = Surfaces.from_triangles(tube, solid_names=("sleeve",))
    mirrors = planar_mirrors(surfaces, ["sleeve"])
    assert len(mirrors) == sectors
    assert {len(mirror.facets) for mirror in mirrors} == {2 * slices}


def test_rounding_of_a_single_precision_file_does_not_split_a_plane_but_a_step_does():
    """A binary STL stores single precision, so a flat wall read from one is flat only to a
    rounding; it must still be one plane. A real step out of the plane must not be."""
    vertices = inward_box(2)
    rounded = vertices.astype(np.float32).astype(float) * 1.000_000_1
    surfaces = Surfaces.from_triangles(rounded, solid_names=("vessel",))
    assert len(planar_mirrors(surfaces, ["vessel"])) == 6

    stepped = vertices.copy()
    on_floor = np.flatnonzero(np.all(vertices[:, :, 2] == 0.0, axis=1))
    stepped[on_floor[0], :, 2] = 1e-3
    surfaces = Surfaces.from_triangles(stepped, solid_names=("vessel",))
    assert len(planar_mirrors(surfaces, ["vessel"])) == 7


def test_the_two_sides_of_a_sheet_are_two_mirrors():
    """A mirror reflects only on the side it faces, so the two faces of a thin sheet, though in
    one plane, form different images."""
    sheet = rectangle_triangles([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([sheet, sheet[:, ::-1]]), solid_names=("baffle",)
    )
    mirrors = planar_mirrors(surfaces, ["baffle"])
    assert len(mirrors) == 2
    assert sorted(tuple(mirror.facets) for mirror in mirrors) == [(0, 1), (2, 3)]
    np.testing.assert_allclose(np.asarray(mirrors[0].normal), [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(np.asarray(mirrors[1].normal), [0.0, 0.0, -1.0], atol=1e-15)


def test_only_the_named_bodies_reflect_and_two_bodies_never_share_a_mirror():
    left = rectangle_triangles([-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    right = rectangle_triangles([1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    wall = rectangle_triangles([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5])
    surfaces = Surfaces.from_triangles(
        np.concatenate([left, right, wall]),
        solid_id=[0, 0, 1, 1, 2, 2],
        solid_names=("left", "right", "wall"),
    )
    mirrors = planar_mirrors(surfaces, ["right", "left"])
    assert [tuple(mirror.facets) for mirror in mirrors] == [(0, 1), (2, 3)]
    with pytest.raises(KeyError, match="no such body in this surface set: \\['mirorr'\\]"):
        planar_mirrors(surfaces, ["left", "mirorr"])


def test_a_point_source_has_no_plane_and_joins_no_mirror():
    floor = rectangle_triangles([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([floor, np.zeros((1, 3, 3))]), power=[0.0, 0.0, 5.0]
    )
    (mirror,) = planar_mirrors(surfaces, ["surface"])
    np.testing.assert_array_equal(mirror.facets, [0, 1])


def test_the_largest_facet_sets_a_plane_not_a_sliver_whose_normal_is_rounding():
    """A sliver lying in a wall can compute a normal tilted nearly ninety degrees from it. Were it
    to start the plane, the wall's own facets would not lie in that plane and the wall would split
    into two mirrors. It comes first in the file here, so only the order by area saves it."""
    sliver = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-9, 1e-7]]])
    floor = rectangle_triangles([0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    surfaces = Surfaces.from_triangles(np.concatenate([sliver, floor]))
    assert abs(float(surfaces.normal[0, 2])) < 0.02
    (mirror,) = planar_mirrors(surfaces, ["surface"])
    np.testing.assert_array_equal(mirror.facets, [0, 1, 2])
    np.testing.assert_allclose(np.asarray(mirror.normal), [0.0, 0.0, 1.0], atol=1e-6)


def test_a_tolerance_that_is_not_positive_is_refused():
    surfaces = Surfaces.from_triangles(inward_box(1))
    with pytest.raises(ValueError, match="tolerance must be positive"):
        planar_mirrors(surfaces, ["surface"], tolerance=0.0)


# ---------------------------------------------------------------------------------------
# Reflecting in a mirror
# ---------------------------------------------------------------------------------------


def _mirror(point, normal) -> Mirror:
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    return Mirror(
        point=jnp.asarray(point, dtype=float),
        normal=jnp.asarray(normal),
        facets=np.zeros(0, dtype=int),
    )


def test_a_point_is_reflected_through_the_plane_and_a_point_on_it_stays():
    mirror = _mirror([2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(mirror.reflect_points([[3.0, 1.0, 1.0]]), [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(mirror.reflect_points([2.0, 5.0, -3.0]), [2.0, 5.0, -3.0])
    # Directions do not move with the plane's offset.
    np.testing.assert_allclose(mirror.reflect_directions([1.0, 1.0, 0.0]), [-1.0, 1.0, 0.0])


def _scene() -> Surfaces:
    """Tilted emitters of every kind of distribution, and a point source, away from the origin.

    The photometric table varies as ``sin h`` round its axis, so an image that kept the source's
    sense of rotation would be lit the wrong way round; a table symmetric about its ``0-180``
    plane could not tell.
    """
    rng = np.random.default_rng(11)
    triangles = rng.uniform(-0.3, 0.3, size=(6, 3, 3)) + np.array([0.0, 0.0, 1.0])
    horizontal = jnp.linspace(0.0, 2.0 * jnp.pi, 9)
    vertical = jnp.linspace(0.0, jnp.pi / 2, 7)
    table = (1.0 + 0.6 * jnp.sin(horizontal))[:, None] * jnp.cos(vertical)[None, :] ** 2
    photometric = PhotometricProfile(
        vertical=vertical, horizontal=horizontal, table=table, up=jnp.asarray([0.3, 1.0, 0.2])
    )
    return Surfaces.from_triangles(
        np.concatenate([triangles, np.full((1, 3, 3), [0.1, 0.2, 1.3])]),
        emission=[1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 0.0],
        power=[0.0] * 6 + [4.0],
        diffuse_reflectance=0.25,
        specular_reflectance=0.5,
        profiles=(Lambertian(), CosinePower(4.0), photometric, Isotropic()),
        profile_index=[0, 1, 2, 0, 1, 2, 3],
    )


def test_an_image_lights_each_reflected_point_as_its_source_lights_the_point():
    """The property the image-source method rests on: what a viewer sees in a plane mirror is the
    image, so the direct field of the image at a point's reflection is the source's at the point.
    A winding left unreversed turns every image facet away from the viewer, and an unmirrored
    distribution sends its asymmetric part the wrong way round."""
    surfaces = _scene()
    mirror = _mirror([0.1, -0.2, 0.05], [0.2, 0.3, -1.0])
    image = mirror.image(surfaces)
    rng = np.random.default_rng(5)
    points = rng.uniform(-1.0, 1.0, size=(40, 3)) + np.array([0.0, 0.0, 2.2])
    source_field = direct_fluence_rate(surfaces, points)
    image_field = direct_fluence_rate(image, mirror.reflect_points(points))
    assert float(jnp.min(source_field)) > 0.0
    np.testing.assert_allclose(image_field, source_field, rtol=1e-12)


def test_an_image_carries_every_property_and_label_and_its_own_image_is_the_source():
    surfaces = _scene()
    mirror = _mirror([0.1, -0.2, 0.05], [0.2, 0.3, -1.0])
    image = mirror.image(surfaces)
    np.testing.assert_allclose(image.normal, mirror.reflect_directions(surfaces.normal), atol=1e-14)
    np.testing.assert_allclose(image.area, surfaces.area, rtol=1e-13)
    for name in ("emission", "power", "diffuse_reflectance", "specular_reflectance"):
        np.testing.assert_array_equal(getattr(image, name), getattr(surfaces, name))
    np.testing.assert_array_equal(image.profile_index, surfaces.profile_index)
    assert image.point_source_index == surfaces.point_source_index
    assert image.solid_names == surfaces.solid_names
    again = mirror.image(image)
    np.testing.assert_allclose(again.vertices, surfaces.vertices, atol=1e-14)
    assert again.profiles[2].handedness == surfaces.profiles[2].handedness
