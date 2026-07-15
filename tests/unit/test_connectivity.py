"""Unit tests for the face-to-cell connectivity helpers.

These encode the single boundary convention (``neighbour < 0`` marks a boundary face) that
every gather/scatter over faces relies on, so they are tested in isolation on tiny
hand-built index arrays — no mesh, no geometry.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import (
    FaceCellConnectivity,
    FaceNodeConnectivity,
    interior_mask,
)


def test_interior_mask_marks_boundary_faces():
    """A neighbour index ``< 0`` is a boundary face; ``>= 0`` is interior."""
    neighbour = jnp.array([1, -1, 3, -1, 0])
    np.testing.assert_array_equal(
        np.asarray(interior_mask(neighbour)), [True, False, True, False, True]
    )


def test_interior_mask_preserves_numpy_input():
    """The mask is a plain comparison, so a NumPy input stays NumPy (build-time paths)."""
    neighbour = np.array([2, -1, 0])
    mask = interior_mask(neighbour)
    assert isinstance(mask, np.ndarray)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_safe_neighbour_substitutes_owner_on_boundary_faces():
    """Boundary faces get the owner index; interior faces keep the neighbour index."""
    fc = FaceCellConnectivity(jnp.array([0, 5, 2, 7]), jnp.array([1, -1, 3, -1]), n_cells=8)
    np.testing.assert_array_equal(np.asarray(fc.interior), [True, False, True, False])
    # interior faces unchanged; boundary faces read as owner == neighbour.
    np.testing.assert_array_equal(np.asarray(fc.safe_neighbour), [1, 5, 3, 7])


def test_safe_neighbour_index_is_always_in_range():
    """The substituted index is a valid cell, so a per-cell gather never goes out of bounds."""
    n_cells = 4
    # all-boundary (worst case): every neighbour is the sentinel -1
    fc = FaceCellConnectivity(jnp.array([0, 1, 2, 3]), jnp.array([-1, -1, -1, -1]), n_cells=n_cells)
    field = jnp.arange(n_cells, dtype=jnp.float64)
    # gather reads the owner's own value on every boundary face
    np.testing.assert_array_equal(np.asarray(field[fc.safe_neighbour]), np.asarray(field[fc.owner]))


def test_all_interior_leaves_neighbour_untouched():
    """With no boundary faces the safe index is exactly the neighbour and the mask is all True."""
    fc = FaceCellConnectivity(jnp.array([0, 1, 2]), jnp.array([1, 2, 0]), n_cells=3)
    assert bool(jnp.all(fc.interior))
    np.testing.assert_array_equal(np.asarray(fc.safe_neighbour), np.asarray(fc.neighbour))


# --------------------------------------------------------------------------- FaceCellConnectivity
# A 3-cell line with two interior faces (0-1, 1-2) and two boundary faces (on cells 0 and 2).
#   owner = [0, 1, 0, 2],  neighbour = [1, 2, -1, -1]


def _line_face_cells() -> FaceCellConnectivity:
    return FaceCellConnectivity(jnp.array([0, 1, 0, 2]), jnp.array([1, 2, -1, -1]), n_cells=3)


def test_face_cells_scatter_masks_neighbour_on_boundary():
    """`scatter` adds owner_contrib for every face but neighbour_contrib only on interior faces."""
    fc = _line_face_cells()
    owner_contrib = jnp.array([1.0, 2.0, 3.0, 4.0])
    neighbour_contrib = jnp.array([10.0, 20.0, 30.0, 40.0])  # entries 30,40 are on boundary faces
    # owner side: cell0=1+3, cell1=2, cell2=4; neighbour side (boundary masked): cell1+=10, cell2+=20
    np.testing.assert_allclose(
        np.asarray(fc.scatter(owner_contrib, neighbour_contrib)), [4, 12, 24]
    )


def test_face_cells_scatter_conservative_is_antisymmetric():
    """Interior contributions cancel; the cell sum equals the boundary owner-outward flux."""
    fc = _line_face_cells()
    flux = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = fc.scatter_conservative(flux)
    np.testing.assert_allclose(np.asarray(result), [4, 1, 2])
    # only the two boundary fluxes (3 on cell 0, 4 on cell 2) survive the interior cancellation
    assert float(jnp.sum(result)) == 7.0


def test_face_cells_scatter_symmetric_adds_to_both_cells():
    """Both incident cells receive the same contribution (neighbour masked on boundary)."""
    fc = _line_face_cells()
    contrib = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.asarray(fc.scatter_symmetric(contrib)), [4, 3, 6])


def test_face_cells_scatter_broadcasts_vector_contributions():
    """The boundary mask broadcasts against a rank-2 per-face contribution."""
    fc = _line_face_cells()
    flux = jnp.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    result = fc.scatter_conservative(flux)
    # each component is the scalar conservative scatter of that column
    np.testing.assert_allclose(np.asarray(result[:, 0]), [4, 1, 2])
    np.testing.assert_allclose(np.asarray(result[:, 1]), [-4, -1, -2])


def test_combine_face_values_takes_interior_then_boundary():
    """Interior faces get the interior values; boundary faces get the boundary values."""
    fc = _line_face_cells()  # faces 0,1 interior; faces 2,3 boundary
    interior_values = jnp.array([10.0, 11.0, 12.0, 13.0])
    boundary_values = jnp.array([20.0, 21.0, 22.0, 23.0])
    got = fc.combine_face_values(interior_values, boundary_values)
    np.testing.assert_array_equal(np.asarray(got), [10.0, 11.0, 22.0, 23.0])
    # a scalar boundary placeholder (the diffusion denom-guard usage) broadcasts over all faces
    guarded = fc.combine_face_values(interior_values, 1.0)
    np.testing.assert_array_equal(np.asarray(guarded), [10.0, 11.0, 1.0, 1.0])


def test_combine_face_values_broadcasts_over_vector_fields():
    """The per-face interior mask broadcasts over trailing component axes."""
    fc = _line_face_cells()
    interior_values = jnp.arange(8.0).reshape(4, 2)  # (n_faces, dim)
    boundary_values = jnp.full((4, 2), -1.0)
    got = fc.combine_face_values(interior_values, boundary_values)
    expected = np.asarray(interior_values).copy()
    expected[2:] = -1.0  # boundary faces
    np.testing.assert_array_equal(np.asarray(got), expected)


# --------------------------------------------------------------------------- FaceNodeConnectivity
# Two faces sharing an edge: a triangle [0,1,2] and a quad [1,2,3,4]. CSR offsets = [0, 3, 7].


def _tri_quad_face_nodes() -> FaceNodeConnectivity:
    return FaceNodeConnectivity.from_csr(jnp.array([0, 3, 7]), jnp.array([0, 1, 2, 1, 2, 3, 4]))


def test_face_nodes_counts_and_reduce():
    """`counts` is nodes-per-face; reducing ones over incidences recovers it."""
    fn = _tri_quad_face_nodes()
    np.testing.assert_array_equal(np.asarray(fn.counts), [3, 4])
    ones = jnp.ones(7)
    np.testing.assert_allclose(np.asarray(fn.reduce_to_faces(ones)), [3, 4])


def test_face_nodes_perimeter_next_wraps_within_each_face():
    """The perimeter-next map cycles each face's nodes, wrapping the last back to the first."""
    fn = _tri_quad_face_nodes()
    per_incidence = jnp.arange(7)
    # face 0 (positions 0,1,2) → 1,2,0 ; face 1 (positions 3,4,5,6) → 4,5,6,3
    np.testing.assert_array_equal(
        np.asarray(fn.perimeter_next(per_incidence)), [1, 2, 0, 4, 5, 6, 3]
    )


def test_face_nodes_vertex_mean():
    """`vertex_mean` averages each face's node coordinates."""
    fn = _tri_quad_face_nodes()
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 2.0]])
    means = fn.vertex_mean(coords)
    np.testing.assert_allclose(np.asarray(means[0]), [1 / 3, 1 / 3])  # triangle 0,1,2
    np.testing.assert_allclose(np.asarray(means[1]), [5 / 4, 3 / 4])  # quad 1,2,3,4
