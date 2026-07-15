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
