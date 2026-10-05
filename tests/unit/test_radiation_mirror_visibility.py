"""What stands across a path reflected once in a mirror: both legs, every body, the surface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.mirror_visibility import (
    build_mirror_masks,
    build_mirror_visibility,
    reflected_surviving,
)
from aquaflux.radiation.mirrors import planar_mirrors
from aquaflux.radiation.profiles import Isotropic, Lambertian
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.solids import Box, Sphere

from tests.unit.radiation_references import rectangle_triangles

#: A lamp at (0, 0, 1) over the floor mirror reaches (1, 0, 1) through the point (0.5, 0, 0): the
#: first leg runs down from the lamp to there, the second back up to the receiver.
LAMP = (0.0, 0.0, 1.0)
RECEIVER = (1.0, 0.0, 1.0)
FIRST_LEG = (0.25, 0.0, 0.5)
SECOND_LEG = (0.75, 0.0, 0.5)


def _scene(extra=None):
    """A floor mirror at z = 0 facing up, a point lamp at :data:`LAMP`, then ``extra`` triangles.

    The extra triangles are part of the surface: walls, which shadow as its own triangles do.
    """
    floor = rectangle_triangles([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0])
    extra = np.zeros((0, 3, 3)) if extra is None else np.asarray(extra, dtype=float)
    triangles = np.concatenate([floor, np.full((1, 3, 3), LAMP), extra])
    n_extra = len(extra)
    surfaces = Surfaces.from_triangles(
        triangles,
        solid_id=[0, 0, 1] + [2] * n_extra,
        solid_names=("floor", "lamp", "walls"),
        power=np.concatenate([[0.0, 0.0, 4.0], np.zeros(n_extra)]),
        specular_reflectance=np.concatenate([[0.9, 0.9, 0.0], np.zeros(n_extra)]),
        profiles=(Lambertian(), Isotropic()),
        profile_index=np.concatenate([[0, 0, 1], np.zeros(n_extra, dtype=int)]),
    )
    return surfaces, planar_mirrors(surfaces, ["floor"])[0]


def _lamp_column(mask) -> int:
    return int(mask.columns([2])[0])


@pytest.mark.parametrize(
    ("bodies", "expected"),
    [
        ([Sphere(FIRST_LEG, 0.1)], [1]),
        ([Sphere(SECOND_LEG, 0.1)], [1]),
        ([Sphere((0.5, 0.0, 0.5), 0.1)], [0]),
        ([Box((0.0, 0.0, 0.5), (5.0, 5.0, 0.1))], [2]),
        (
            [Sphere(FIRST_LEG, 0.1), Sphere(SECOND_LEG, 0.1), Sphere((0.5, 0.0, 0.9), 0.05)],
            [1, 1, 0],
        ),
    ],
    ids=["first leg", "second leg", "between the legs", "both legs", "each its own"],
)
def test_each_body_counts_the_legs_of_the_path_it_lies_across(bodies, expected):
    """A ball on either leg alone is crossed once; one between the legs, under the turn, not at
    all; a slab both legs pass through, twice; and several bodies each keep their own count."""
    surfaces, mirror = _scene()
    mask = build_mirror_visibility(
        mirror, bodies, surfaces, np.array([RECEIVER]), self_occlusion=NoOcclusion()
    )
    np.testing.assert_array_equal(mask.crossings[:, 0, _lamp_column(mask)], expected)
    assert mask.hidden is None


def test_a_body_crossed_twice_transmits_its_fraction_squared():
    surfaces, mirror = _scene()
    slab = Box((0.0, 0.0, 0.5), (5.0, 5.0, 0.1))
    ball = Sphere(FIRST_LEG, 0.1)
    mask = build_mirror_visibility(
        mirror, [slab, ball], surfaces, np.array([RECEIVER]), self_occlusion=NoOcclusion()
    )
    surviving = mask.surviving(jnp.asarray([0.3, 0.6]))[0, _lamp_column(mask)]
    np.testing.assert_allclose(surviving, 0.3**2 * 0.6, rtol=1e-15)
    slope = jax.grad(lambda t: mask.surviving(jnp.asarray([t, 0.6]))[0, _lamp_column(mask)])(0.3)
    np.testing.assert_allclose(slope, 2 * 0.3 * 0.6, rtol=1e-15)


def test_an_opaque_body_has_a_finite_derivative_whether_or_not_it_is_crossed():
    """At transmittance zero, ``t ** 0`` would differentiate to ``0 * t ** -1``: not a number."""
    crossings = jnp.asarray([[0, 1, 2]], dtype=jnp.uint8)
    slope = jax.jacfwd(lambda t: reflected_surviving(crossings, None, t))(jnp.asarray([0.0]))
    np.testing.assert_array_equal(slope[:, 0], [0.0, 1.0, 0.0])


@pytest.mark.parametrize("leg", ["first", "second"])
def test_the_surface_s_own_triangles_hide_a_path_from_either_leg(leg):
    centre = FIRST_LEG if leg == "first" else SECOND_LEG
    plate = rectangle_triangles(centre, [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])
    surfaces, mirror = _scene(plate)
    mask = build_mirror_visibility(mirror, [], surfaces, np.array([RECEIVER, [-1.0, 0.0, 1.0]]))
    np.testing.assert_array_equal(mask.hidden[:, _lamp_column(mask)], [True, False])


def test_the_mirror_does_not_shadow_the_light_it_reflects():
    """Both legs end where the path meets the mirror, on its triangles; lifted off them, the
    mirror's own facets -- and any other triangle in its plane -- hide nothing."""
    beside = rectangle_triangles([0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.6, 0.0])
    surfaces, mirror = _scene(beside)
    mask = build_mirror_visibility(mirror, [], surfaces, np.array([RECEIVER]))
    assert not bool(jnp.any(mask.hidden))


def test_a_receiver_on_a_facet_ignores_that_facet_on_its_leg():
    """A facet's centroid as the receiver, as between facets: the second leg ends on it."""
    target = rectangle_triangles(RECEIVER, [0.05, 0.0, 0.0], [0.0, 0.05, 0.0])[:, ::-1]
    surfaces, mirror = _scene(target)
    on_target = 3
    receivers = np.asarray(surfaces.centroid)[[on_target]]
    told = build_mirror_visibility(
        mirror, [], surfaces, receivers, receiver_facet=np.array([on_target])
    )
    untold = build_mirror_visibility(mirror, [], surfaces, receivers)
    assert not bool(told.hidden[0, _lamp_column(told)])
    assert bool(untold.hidden[0, _lamp_column(untold)])


def test_only_paths_that_meet_the_mirror_are_tested():
    """A receiver behind the mirror sees nothing in it, and is recorded clear even where a body
    stands across the segment to it; a source behind the mirror has no column at all."""
    surfaces, mirror = _scene(rectangle_triangles([0.0, 0.0, -1.0], [0.1, 0, 0], [0, 0.1, 0]))
    slab = Box((0.0, 0.0, -0.5), (5.0, 5.0, 0.1))
    mask = build_mirror_visibility(mirror, [slab], surfaces, np.array([RECEIVER, [1.0, 0.0, -1.0]]))
    assert list(mask.sources) == [2]
    np.testing.assert_array_equal(mask.crossings[0, 1], [0])
    assert not bool(mask.hidden[1, 0])


def test_the_answer_does_not_depend_on_how_the_pairs_are_cut_into_passes():
    rng = np.random.default_rng(0)
    plates = np.concatenate(
        [
            rectangle_triangles(centre, [0.04, 0.0, 0.0], [0.0, 0.04, 0.0])
            for centre in rng.uniform([-1.0, -1.0, 0.2], [1.0, 1.0, 0.8], (6, 3))
        ]
    )
    surfaces, mirror = _scene(plates)
    points = rng.uniform([-1.5, -1.5, 0.1], [1.5, 1.5, 1.5], (20, 3))
    ball = [Sphere((0.2, 0.1, 0.4), 0.2)]
    whole = build_mirror_visibility(mirror, ball, surfaces, points)
    cut = build_mirror_visibility(mirror, ball, surfaces, points, pair_limit=7)
    gridded = build_mirror_visibility(
        mirror, ball, surfaces, points, self_occlusion=RayCastOcclusion(grid=True)
    )
    assert bool(jnp.any(whole.hidden)) and bool(jnp.any(whole.crossings))
    for other in (cut, gridded):
        np.testing.assert_array_equal(whole.crossings, other.crossings)
        np.testing.assert_array_equal(whole.hidden, other.hidden)


def test_with_nothing_in_the_way_no_mask_is_built():
    surfaces, mirror = _scene()
    points = np.array([RECEIVER])
    assert build_mirror_masks([mirror], [], surfaces, points, self_occlusion=NoOcclusion()) is None
    masks = build_mirror_masks([mirror], [], surfaces, points)
    assert masks is not None and len(masks) == 1


def test_a_mask_refuses_other_receivers_and_other_sources():
    surfaces, mirror = _scene()
    mask = build_mirror_visibility(mirror, [], surfaces, np.array([RECEIVER]))
    with pytest.raises(ValueError, match="mirror visibility mask was built for different"):
        mask.for_receivers(np.array([[1.0, 0.1, 1.0]]))
    with pytest.raises(ValueError, match="no column for facets"):
        mask.columns([0])
    assert mask.for_sources([2]).sources.tolist() == [2]


def test_a_source_straddling_the_mirror_is_left_clear():
    """A facet whose centroid lies behind the plane has a part in front, and so a column, but
    the path through its centroid never meets the mirror: it is recorded clear, not tested."""
    straddling = np.array([[[0.0, -0.1, -0.2], [0.1, 0.1, -0.2], [-0.1, 0.1, 0.1]]])
    surfaces, mirror = _scene(straddling)
    slab = Box((0.0, 0.0, 0.5), (5.0, 5.0, 0.1))
    assert float(np.asarray(surfaces.centroid)[3, 2]) < 0.0
    mask = build_mirror_visibility(
        mirror, [slab], surfaces, np.array([RECEIVER]), self_occlusion=NoOcclusion()
    )
    np.testing.assert_array_equal(mask.crossings[0, 0, mask.columns([2, 3])], [2, 0])
