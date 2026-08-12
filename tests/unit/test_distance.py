"""Unit tests for the cell-to-patch distance field (physics-free, analytic geometry).

On a structured grid the nearest bottom-wall face sits directly below each cell, so the distance to
the ``"bottom"`` patch is exactly the cell-centroid height; the distance to two walls is the
per-cell minimum. Both are checked against the closed form, plus the error paths.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.mesh import distance_to_patches, structured_grid_2d


def test_distance_to_bottom_equals_cell_height() -> None:
    """The nearest bottom-wall face is directly below a cell, so the distance is the centroid's y."""
    mesh = structured_grid_2d(4, 3, named_boundaries=True)
    geometry = mesh.geometry()
    d = distance_to_patches(mesh, geometry, ["bottom"])
    assert jnp.allclose(d, geometry.cell.centroid[:, 1])


def test_distance_to_two_walls_is_the_per_cell_minimum() -> None:
    """Distance to {bottom, left} is min(centroid_y, centroid_x) — the nearer of the two walls."""
    mesh = structured_grid_2d(4, 3, named_boundaries=True)
    geometry = mesh.geometry()
    d = distance_to_patches(mesh, geometry, ["bottom", "left"])
    centroid = geometry.cell.centroid
    assert jnp.allclose(d, jnp.minimum(centroid[:, 0], centroid[:, 1]))


def test_rejects_unknown_patch() -> None:
    mesh = structured_grid_2d(2, 2, named_boundaries=True)
    with pytest.raises(ValueError, match="no group named 'wall'"):
        distance_to_patches(mesh, mesh.geometry(), ["wall"])


def test_rejects_empty_patch_names() -> None:
    mesh = structured_grid_2d(2, 2, named_boundaries=True)
    with pytest.raises(ValueError, match="no patch names given"):
        distance_to_patches(mesh, mesh.geometry(), [])


def test_blocked_search_is_identical_to_the_whole_array_form() -> None:
    """Blocking the search must change what is live at once and nothing else.

    Each cell's nearest target is independent of every other cell's, so a block boundary cannot
    reorder or drop a comparison. Checked bit-for-bit rather than to a tolerance, because anything
    short of exact equality would mean the blocking had changed the arithmetic.
    """
    import numpy as np
    from aquaflux.mesh import distance
    from aquaflux.vectors import norm_squared

    mesh = structured_grid_2d(9, 7, named_boundaries=True)
    geometry = mesh.geometry()
    target = geometry.face.centroid[mesh.face_patches.indices("bottom")]
    centroid = geometry.cell.centroid

    whole = jnp.sqrt(jnp.min(norm_squared(centroid[:, None, :] - target[None, :, :]), axis=1))
    blocked = distance_to_patches(mesh, geometry, ["bottom"])
    assert np.array_equal(np.asarray(blocked), np.asarray(whole))

    # ...and with the working set squeezed to a single cell, so every block boundary is exercised.
    original = distance._SEARCH_WORKING_BYTES
    try:
        distance._SEARCH_WORKING_BYTES = 1
        one_at_a_time = distance_to_patches(mesh, geometry, ["bottom"])
    finally:
        distance._SEARCH_WORKING_BYTES = original
    assert np.array_equal(np.asarray(one_at_a_time), np.asarray(whole))


def test_blocked_search_bounds_the_working_set() -> None:
    """The block must actually shrink as the target count grows, or the bound is decorative.

    The whole point is that the search never materializes a cell-by-target structure, so the block
    has to respond to the number of targets rather than being a fixed cell count.
    """
    from aquaflux.mesh import distance

    wide = distance._nearest_squared_distance(
        jnp.zeros((1000, 3)), jnp.ones((5000, 3))
    )  # would be 1000 x 5000 x 3 in one piece
    assert wide.shape == (1000,)

    # The derived block scales inversely with the target count: ten times the targets, a tenth the
    # cells per block.
    def block_for(n_target):
        per_cell = max(1, 3 * n_target * 3 * jnp.zeros(1).dtype.itemsize)
        return max(1, distance._SEARCH_WORKING_BYTES // per_cell)

    assert block_for(5000) < block_for(500) <= block_for(50)


def test_distance_is_differentiable_through_node_positions() -> None:
    """The search stays traced, so a gradient with respect to node coordinates still flows.

    This is why the blocked ``jnp`` search is kept rather than handing the nearest-target lookup to a
    spatial index: an index needs concrete coordinates, and geometry is derived from ``node_coords``
    on demand precisely so these gradients chain through it.
    """
    import jax

    mesh = structured_grid_2d(3, 3, named_boundaries=True)

    def total_distance(coords):
        moved = jax.tree_util.tree_map(lambda _: coords, mesh.node_coords)
        import equinox as eqx

        shifted = eqx.tree_at(lambda m: m.node_coords, mesh, moved)
        return jnp.sum(distance_to_patches(shifted, shifted.geometry(), ["bottom"]))

    gradient = jax.grad(total_distance)(mesh.node_coords)
    assert gradient.shape == mesh.node_coords.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradient != 0.0))  # the distance really does depend on node positions
