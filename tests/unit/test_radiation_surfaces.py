"""The surface set: derived geometry, per-body property assignment, and zero-area facets."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vertices": np.zeros((2, 3))}, "shape"),
        ({"vertices": np.zeros((2, 4, 3))}, "shape"),
        ({"vertices": np.zeros((2, 3, 3)), "solid_id": [0]}, "solid_id must have shape"),
        ({"vertices": np.zeros((2, 3, 3)), "solid_id": [0, 5]}, "only 1 name"),
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
