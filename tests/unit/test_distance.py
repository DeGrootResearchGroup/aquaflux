"""Unit tests for the cell-to-patch distance field (physics-free, analytic geometry).

On a structured grid the nearest bottom-wall face sits directly below each cell, so the distance to
the ``"bottom"`` patch is exactly the cell-centroid height; the distance to two walls is the
per-cell minimum. Both are checked against the closed form, plus the error paths.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import distance, distance_to_patches, structured_grid_2d
from aquaflux.vectors import norm_squared


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


def test_concrete_search_matches_the_whole_array_form() -> None:
    """The k-d tree path (concrete inputs, the ordinary eager call) must match brute force exactly.

    A k-d tree query is a different algorithm from the brute-force minimum, not just a re-blocked
    version of it, so this is checked independently of the traced-path equivalence tests below.
    """
    mesh = structured_grid_2d(9, 7, named_boundaries=True)
    geometry = mesh.geometry()
    target = geometry.face.centroid[mesh.face_patches.indices("bottom")]
    centroid = geometry.cell.centroid

    whole = jnp.sqrt(jnp.min(norm_squared(centroid[:, None, :] - target[None, :, :]), axis=1))
    tree_based = distance_to_patches(mesh, geometry, ["bottom"])
    assert np.array_equal(np.asarray(tree_based), np.asarray(whole))


def test_concrete_inputs_use_the_kdtree_path_and_traced_inputs_do_not() -> None:
    """Pins the dispatch itself, not just its numerical outcome: concrete calls build a k-d tree.

    A regression that accidentally routed concrete calls through the blocked ``jnp`` fallback (or
    traced calls through ``scipy``, which would raise on a :class:`jax.core.Tracer`) would still pass
    the equivalence tests above/below on any input small enough not to time out -- this test instead
    watches which implementation actually ran.
    """
    mesh = structured_grid_2d(4, 3, named_boundaries=True)
    geometry = mesh.geometry()
    target = geometry.face.centroid[mesh.face_patches.indices("bottom")]
    centroid = geometry.cell.centroid

    assert not distance._is_traced(centroid, target)

    calls = []
    real = distance._nearest_squared_distance_concrete

    def spy(c, t):
        calls.append("concrete")
        return real(c, t)

    original = distance._nearest_squared_distance_concrete
    try:
        distance._nearest_squared_distance_concrete = spy
        distance._nearest_squared_distance(centroid, target)
    finally:
        distance._nearest_squared_distance_concrete = original
    assert calls == ["concrete"]

    # A traced call must never reach the concrete (scipy) path -- scipy cannot consume a Tracer.
    traced_calls = []
    real_traced = distance._nearest_squared_distance_traced

    def spy_traced(c, t):
        traced_calls.append("traced")
        return real_traced(c, t)

    original_traced = distance._nearest_squared_distance_traced
    try:
        distance._nearest_squared_distance_traced = spy_traced
        jax.make_jaxpr(distance._nearest_squared_distance)(centroid, target)
    finally:
        distance._nearest_squared_distance_traced = original_traced
    assert traced_calls == ["traced"]


def test_traced_blocked_search_is_identical_to_the_whole_array_form() -> None:
    """The traced fallback's blocking must change what is live at once and nothing else.

    Each cell's nearest target is independent of every other cell's, so a block boundary cannot
    reorder or drop a comparison. Checked bit-for-bit rather than to a tolerance, because anything
    short of exact equality would mean the blocking had changed the arithmetic. Calls the traced
    helper directly, since a concrete call would take the k-d tree path instead.
    """
    mesh = structured_grid_2d(9, 7, named_boundaries=True)
    geometry = mesh.geometry()
    target = geometry.face.centroid[mesh.face_patches.indices("bottom")]
    centroid = geometry.cell.centroid

    whole = jnp.min(norm_squared(centroid[:, None, :] - target[None, :, :]), axis=1)
    blocked = distance._nearest_squared_distance_traced(centroid, target)
    assert np.array_equal(np.asarray(blocked), np.asarray(whole))

    # ...and with the working set squeezed to a single cell, so every block boundary is exercised.
    original = distance._SEARCH_WORKING_BYTES
    try:
        distance._SEARCH_WORKING_BYTES = 1
        one_at_a_time = distance._nearest_squared_distance_traced(centroid, target)
    finally:
        distance._SEARCH_WORKING_BYTES = original
    assert np.array_equal(np.asarray(one_at_a_time), np.asarray(whole))


def test_traced_blocked_search_pads_a_remainder_block_correctly() -> None:
    """A block size that does not evenly divide the cell count must still match the whole-array form.

    The blocked loop runs over a whole number of blocks, padding the tail with a repeated real cell and
    slicing the padding away afterward. A block size that divides ``n_cells`` evenly (as in the test
    above) never exercises that padding at all, so this pins it with a block size chosen not to divide.
    """
    mesh = structured_grid_2d(9, 7, named_boundaries=True)  # 63 cells
    geometry = mesh.geometry()
    target = geometry.face.centroid[mesh.face_patches.indices("bottom")]
    centroid = geometry.cell.centroid
    assert centroid.shape[0] % 8 != 0  # 63 % 8 == 7: the block below leaves a genuine remainder

    whole = jnp.min(norm_squared(centroid[:, None, :] - target[None, :, :]), axis=1)

    n_target, dim = target.shape
    per_cell = max(1, 3 * n_target * dim * centroid.dtype.itemsize)
    original = distance._SEARCH_WORKING_BYTES
    try:
        distance._SEARCH_WORKING_BYTES = per_cell * 8  # forces block == 8
        blocked = distance._nearest_squared_distance_traced(centroid, target)
    finally:
        distance._SEARCH_WORKING_BYTES = original
    assert np.array_equal(np.asarray(blocked), np.asarray(whole))


def test_block_count_does_not_grow_the_compiled_program() -> None:
    """More blocks must cost loop iterations, not a bigger compiled program (issue #240).

    Before this fix each block was a separately traced-and-dispatched XLA program feeding a
    ``jnp.concatenate`` over all of them, so the compiled program's own size grew with the block count
    -- the mechanism that turned this into a multi-minute compile when embedded in a larger jitted
    computation. Run through ``jax.lax.map``, the per-block computation is traced once and looped
    on-device, so the jaxpr's equation count must stay flat as the block count rises by two orders of
    magnitude. A traced call (``jax.make_jaxpr``) is what reaches this path at all.
    """
    centroid = jnp.zeros((2000, 3))
    target = jnp.ones((10, 3))

    def n_equations(working_bytes: int) -> int:
        original = distance._SEARCH_WORKING_BYTES
        try:
            distance._SEARCH_WORKING_BYTES = working_bytes
            jaxpr = jax.make_jaxpr(distance._nearest_squared_distance)(centroid, target)
        finally:
            distance._SEARCH_WORKING_BYTES = original
        return len(jaxpr.jaxpr.eqns)

    few_blocks = n_equations(distance._SEARCH_WORKING_BYTES)  # the shipped default: 1 block here
    many_blocks = n_equations(1)  # forces block == 1, i.e. 2000 blocks
    assert few_blocks == many_blocks


def test_blocked_search_bounds_the_working_set() -> None:
    """The block must actually shrink as the target count grows, or the bound is decorative.

    The whole point is that the search never materializes a cell-by-target structure, so the block
    has to respond to the number of targets rather than being a fixed cell count. Calls the traced
    helper directly, since a concrete call would take the k-d tree path instead, which ignores the
    working-set bound entirely.
    """
    wide = distance._nearest_squared_distance_traced(
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

    This is why the blocked ``jnp`` search is kept as a fallback rather than replacing the search
    outright with a spatial index: an index needs concrete coordinates, and geometry is derived from
    ``node_coords`` on demand precisely so these gradients chain through it -- a differentiated build
    reaches here with traced centroids, which ``jax.grad`` alone (no ``jit``) is enough to trigger.
    """
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
