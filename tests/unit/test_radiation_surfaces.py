"""The surface set: derived geometry, per-body property assignment, and zero-area facets."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.surfaces import Surfaces

RIGHT_TRIANGLE = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])


def test_the_geometry_is_derived_from_the_vertices():
    """Centroid, normal and area cannot be passed in, so they cannot disagree with the shape."""
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE)
    assert float(surfaces.area[0]) == pytest.approx(3.0)
    np.testing.assert_allclose(surfaces.centroid[0], [2 / 3, 1.0, 0.0])
    np.testing.assert_allclose(surfaces.normal[0], [0.0, 0.0, 1.0])


def test_the_normal_follows_the_winding_and_is_a_unit_vector():
    reversed_winding = RIGHT_TRIANGLE[:, ::-1, :]
    np.testing.assert_allclose(Surfaces.from_triangles(reversed_winding).normal[0], [0, 0, -1])
    tilted = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    assert float(jnp.linalg.norm(Surfaces.from_triangles(tilted).normal[0])) == pytest.approx(1.0)


def test_a_zero_area_facet_is_legal_and_yields_no_nan():
    """A point source *is* a zero-area facet, so the guard a reader reaches for is the bug.

    Rejecting a degenerate triangle, or dividing by its area to get a normal, deletes every
    point source in the set or fills the geometry with NaN — and a NaN normal propagates into
    every gradient that touches the surface, not only into the facet that caused it.
    """
    point_source = np.zeros((1, 3, 3))
    surfaces = Surfaces.from_triangles(point_source, power=35.0)
    assert float(surfaces.area[0]) == 0.0
    assert float(surfaces.power[0]) == 35.0
    assert bool(jnp.all(jnp.isfinite(surfaces.normal)))
    np.testing.assert_allclose(surfaces.normal[0], [0.0, 0.0, 0.0])


def test_emission_and_power_are_separate_quantities():
    """One is per unit area and one is not; collapsing them silently rescales a point source."""
    both = np.concatenate([RIGHT_TRIANGLE, np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(both, emission=[10.0, 0.0], power=[0.0, 35.0])
    np.testing.assert_allclose(surfaces.emission, [10.0, 0.0])
    np.testing.assert_allclose(surfaces.power, [0.0, 35.0])


def test_a_scalar_property_is_broadcast_to_every_facet():
    surfaces = Surfaces.from_triangles(np.repeat(RIGHT_TRIANGLE, 4, axis=0), reflectance=0.5)
    np.testing.assert_allclose(surfaces.reflectance, np.full(4, 0.5))


def test_per_body_properties_expand_to_per_facet_values():
    surfaces = Surfaces.from_triangles(
        np.repeat(RIGHT_TRIANGLE, 3, axis=0),
        solid_id=[1, 0, 1],
        solid_names=("wall", "lamp"),
    )
    np.testing.assert_allclose(
        surfaces.per_facet({"wall": 0.0, "lamp": 696.42}), [696.42, 0.0, 696.42]
    )


def test_a_misspelled_body_name_is_refused_rather_than_ignored():
    """Silently dropping an unmatched name would leave the lamp emitting nothing."""
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, solid_names=("lampWall",))
    with pytest.raises(KeyError, match="no such body"):
        surfaces.per_facet({"lampwall": 1.0})


def test_omitting_a_body_is_refused_unless_a_default_is_given():
    surfaces = Surfaces.from_triangles(
        np.repeat(RIGHT_TRIANGLE, 2, axis=0), solid_id=[0, 1], solid_names=("wall", "lamp")
    )
    with pytest.raises(KeyError, match="no value given for body"):
        surfaces.per_facet({"lamp": 1.0})
    np.testing.assert_allclose(surfaces.per_facet({"lamp": 1.0}, default=0.0), [0.0, 1.0])


def test_replacing_the_optics_leaves_the_geometry_identical():
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, emission=1.0, reflectance=0.2)
    updated = surfaces.with_optics(reflectance=0.9)
    np.testing.assert_allclose(updated.reflectance, [0.9])
    np.testing.assert_allclose(updated.emission, surfaces.emission)
    np.testing.assert_array_equal(np.asarray(updated.vertices), np.asarray(surfaces.vertices))
    np.testing.assert_array_equal(np.asarray(updated.normal), np.asarray(surfaces.normal))


def test_one_profile_without_an_index_is_given_to_every_facet():
    """How a set is re-read as purely diffuse, which is what every *reflected* ray leaves by
    whatever the source emitted like."""
    pair = np.repeat(RIGHT_TRIANGLE, 2, axis=0)
    surfaces = Surfaces.from_triangles(
        pair, profiles=(CosinePower(4.0), Isotropic()), profile_index=[0, 1]
    )
    diffuse = surfaces.with_optics(profiles=(Lambertian(),))
    assert diffuse.profiles == (Lambertian(),)
    np.testing.assert_array_equal(np.asarray(diffuse.profile_index), [0, 0])
    np.testing.assert_array_equal(np.asarray(diffuse.vertices), np.asarray(surfaces.vertices))


def test_a_new_catalogue_can_be_given_with_its_own_index():
    pair = np.repeat(RIGHT_TRIANGLE, 2, axis=0)
    surfaces = Surfaces.from_triangles(pair)
    relabelled = surfaces.with_optics(
        profiles=(Lambertian(), CosinePower(4.0)), profile_index=[1, 0]
    )
    assert relabelled.profiles == (Lambertian(), CosinePower(4.0))
    np.testing.assert_array_equal(np.asarray(relabelled.profile_index), [1, 0])


def test_a_longer_catalogue_without_an_index_is_refused():
    """The existing indices would point into a catalogue that has changed under them, which is
    a silent relabelling of which facet emits how rather than an error anywhere later."""
    pair = np.repeat(RIGHT_TRIANGLE, 2, axis=0)
    surfaces = Surfaces.from_triangles(pair)
    with pytest.raises(ValueError, match="no profile_index was given"):
        surfaces.with_optics(profiles=(Lambertian(), Isotropic()))


@pytest.mark.parametrize(
    ("index", "message"),
    [
        ([0], r"profile_index must be \(2,\)"),
        ([0, 0, 0], r"profile_index must be \(2,\)"),
        ([0, 3], "outside the 1 profiles given"),
        ([-1, 0], "outside the 1 profiles given"),
    ],
)
def test_an_index_that_does_not_fit_the_catalogue_is_refused(index, message):
    surfaces = Surfaces.from_triangles(np.repeat(RIGHT_TRIANGLE, 2, axis=0))
    with pytest.raises(ValueError, match=message):
        surfaces.with_optics(profile_index=index)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vertices": np.zeros((2, 3))}, "shape"),
        ({"vertices": np.zeros((2, 4, 3))}, "shape"),
        ({"vertices": np.zeros((2, 3, 3)), "solid_id": [0]}, "solid_id must have shape"),
        ({"vertices": np.zeros((2, 3, 3)), "solid_id": [0, 5]}, "solid_id selects body 5"),
    ],
)
def test_an_inconsistent_construction_is_refused(kwargs, message):
    vertices = kwargs.pop("vertices")
    with pytest.raises(ValueError, match=message):
        Surfaces.from_triangles(vertices, **kwargs)


def test_the_surface_set_is_a_pytree_whose_optics_are_differentiable_leaves():
    """The whole point of the module: a study varies these and differentiates through them."""
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, emission=1.0, reflectance=0.2)

    def total(emission):
        return jnp.sum(surfaces.with_optics(emission=emission).emission * surfaces.area)

    assert float(jax.grad(total)(jnp.asarray(2.0))) == pytest.approx(3.0)
    assert float(jnp.sum(eqx.filter_jit(lambda s: s.area)(surfaces))) == pytest.approx(3.0)


def test_the_body_names_are_static_metadata_and_not_an_array_leaf():
    """Strings cannot be traced; carrying them as a leaf breaks every jit of a surface set."""
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, solid_names=("wall",))
    leaves = jax.tree_util.tree_leaves(surfaces)
    assert all(not isinstance(leaf, str) for leaf in leaves)
    assert surfaces.solid_names == ("wall",)


def test_point_sources_are_labelled_explicitly_and_not_inferred_from_the_area():
    """The label and the area agree when a set is built; they must still be separate things.

    The area is a number a gradient may flow through; the kind is a decision about which code
    path a facet takes, which has to be known before anything is traced. Deriving the second
    from the first ties them together, and the knot only shows up when someone differentiates
    with respect to vertex positions.
    """
    vertices = np.concatenate([RIGHT_TRIANGLE, np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(vertices, power=[0.0, 35.0])
    assert surfaces.point_source_index == (1,)
    np.testing.assert_array_equal(surfaces.is_point_source, [False, True])


def test_the_point_source_labels_can_be_given_explicitly():
    surfaces = Surfaces.from_triangles(np.zeros((2, 3, 3)), point_sources=[0])
    np.testing.assert_array_equal(surfaces.is_point_source, [True, False])


def test_a_label_outside_the_set_is_refused():
    with pytest.raises(ValueError, match="outside the set"):
        Surfaces.from_triangles(RIGHT_TRIANGLE, point_sources=[3])


def test_moving_the_vertices_recomputes_everything_derived_from_them():
    """Substituting the vertices alone would leave centroid, normal and area describing the old
    shape, silently -- nothing downstream can tell a stale normal from a fresh one."""
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, emission=4.0, reflectance=0.3)
    doubled = surfaces.with_geometry(np.asarray(RIGHT_TRIANGLE) * 2.0)
    assert float(doubled.area[0]) == pytest.approx(4.0 * float(surfaces.area[0]))
    np.testing.assert_allclose(doubled.centroid[0], np.asarray(surfaces.centroid[0]) * 2.0)
    np.testing.assert_allclose(doubled.normal, surfaces.normal)


def test_moving_the_vertices_keeps_the_optics_and_the_labels():
    vertices = np.concatenate([RIGHT_TRIANGLE, np.zeros((1, 3, 3))])
    surfaces = Surfaces.from_triangles(
        vertices,
        solid_id=[0, 1],
        solid_names=("wall", "lamp"),
        emission=[7.0, 0.0],
        power=[0.0, 35.0],
        reflectance=0.4,
    )
    moved = surfaces.with_geometry(np.asarray(vertices) + np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(moved.emission, surfaces.emission)
    np.testing.assert_allclose(moved.power, surfaces.power)
    np.testing.assert_array_equal(moved.solid_id, surfaces.solid_id)
    assert moved.solid_names == surfaces.solid_names
    assert moved.point_source_index == surfaces.point_source_index


def test_moving_the_vertices_refuses_a_different_number_of_triangles():
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE)
    with pytest.raises(ValueError, match="expected 1 triangles to move"):
        surfaces.with_geometry(np.repeat(RIGHT_TRIANGLE, 2, axis=0))


def test_the_vertices_may_be_traced_so_a_source_can_move_under_a_gradient():
    """The capability the explicit label unlocks.

    With the kind inferred from the area, a traced vertex made the area a tracer and the kind
    unavailable, so this could not be built at all.
    """
    surfaces = Surfaces.from_triangles(RIGHT_TRIANGLE, emission=1.0)

    def area_of(shift):
        moved = surfaces.with_geometry(jnp.asarray(RIGHT_TRIANGLE) * shift)
        return jnp.sum(moved.area)

    assert float(jax.jit(area_of)(jnp.asarray(2.0))) == pytest.approx(12.0)
    assert float(jax.grad(area_of)(jnp.asarray(1.0))) == pytest.approx(6.0)
