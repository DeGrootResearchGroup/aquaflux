"""Unit tests for the face-based-mesh to VTK-connectivity reconstruction. No filesystem.

The property that carries this module is the **winding**: every polyhedron face must be listed
outward from the cell listing it, and a mesh does not store its face rings in any particular
direction. So the checks here are geometric -- each emitted ring's own normal, against the cell it
was emitted for -- rather than comparisons against a hand-written expected array, which would pin
one mesh generator's incidental winding rather than the rule.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from aquaflux.io.vtk.topology import (
    VTK_POLYGON,
    VTK_POLYHEDRON,
    build_vtk_cells,
    stored_ring_is_outward,
)
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d


def _cell_slices(offsets):
    """(start, end) per cell from the VTK end-offset convention."""
    ends = np.asarray(offsets)
    return zip(np.concatenate([[0], ends[:-1]]), ends, strict=True)


def _cell_faces(cells):
    """The face rings of each cell, decoded from the polyhedron face stream."""
    stream = cells.faces
    for start, end in _cell_slices(cells.face_offsets):
        at = int(start)
        n_faces = int(stream[at])
        at += 1
        rings = []
        for _ in range(n_faces):
            count = int(stream[at])
            rings.append(np.asarray(stream[at + 1 : at + 1 + count]))
            at += 1 + count
        assert at == end
        yield rings


def _newell_normal(points):
    """A polygon's area-weighted normal from its ordered vertices, robust to a non-planar ring."""
    return 0.5 * np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)


def _signed_area(points):
    """A planar polygon's signed area in the xy plane; positive when wound counter-clockwise."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _reversed_rings(mesh):
    """The same mesh with every face's node ring stored backwards -- the same faces, wound away."""
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    flipped = np.concatenate([indices[start:end][::-1] for start, end in pairwise(offsets)])
    return Mesh.from_csr(
        mesh.node_coords,
        offsets,
        flipped,
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.n_cells,
    )


def test_3d_cells_are_polyhedra_carrying_every_bounding_face():
    mesh = structured_grid_3d(2, 1, 1)
    cells = build_vtk_cells(mesh)

    assert cells.n_cells == 2
    assert cells.n_points == mesh.n_nodes
    np.testing.assert_array_equal(cells.types, VTK_POLYHEDRON)
    # A hexahedron: six faces of four nodes each, over the cell's eight distinct points.
    assert [len(rings) for rings in _cell_faces(cells)] == [6, 6]
    assert [sorted(map(len, rings)) for rings in _cell_faces(cells)] == [[4] * 6, [4] * 6]
    np.testing.assert_array_equal(cells.offsets, [8, 16])
    for start, end in _cell_slices(cells.offsets):
        assert len(set(cells.connectivity[start:end].tolist())) == 8
    # Per cell: the face count, then each face's own count and its nodes.
    np.testing.assert_array_equal(cells.face_offsets, [1 + 6 * 5, 2 * (1 + 6 * 5)])


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 3, 2)])
def test_every_emitted_polyhedron_face_winds_outward_from_its_own_cell(shape):
    mesh = structured_grid_3d(*shape)
    cells = build_vtk_cells(mesh)
    centroid = np.asarray(mesh.geometry().cell.centroid)

    for index, rings in enumerate(_cell_faces(cells)):
        for ring in rings:
            points = cells.points[ring]
            outward = points.mean(axis=0) - centroid[index]
            assert np.dot(_newell_normal(points), outward) > 0


def test_a_shared_face_is_listed_in_opposite_directions_by_its_two_cells():
    mesh = structured_grid_3d(2, 1, 1)
    left, right = _cell_faces(build_vtk_cells(mesh))

    shared = [
        (a, b)
        for a in left
        for b in right
        if set(a.tolist()) == set(b.tolist()) and len(a) == len(b)
    ]
    assert len(shared) == 1
    a, b = shared[0]
    # Same ring, opposite direction: rotating one reversal onto the other must line up exactly.
    reversed_b = b[::-1]
    roll = int(np.flatnonzero(reversed_b == a[0])[0])
    np.testing.assert_array_equal(np.roll(reversed_b, -roll), a)


def test_a_structured_grid_stores_rings_in_both_directions():
    # The premise the winding rule rests on: "as stored" is not a synonym for "owner-outward", so a
    # writer that kept every owner-listed ring as it found it would emit inward faces on the rest.
    stored = stored_ring_is_outward(structured_grid_3d(2, 2, 2))
    assert 0 < int(np.count_nonzero(stored)) < stored.size


@pytest.mark.parametrize("mesh", [structured_grid_3d(2, 2, 1), structured_grid_2d(3, 2)])
def test_the_output_is_invariant_to_the_direction_the_rings_are_stored_in(mesh):
    # Reversing every stored ring describes the identical mesh, so the reconstruction -- which
    # re-derives each ring's direction rather than trusting it -- must produce identical arrays.
    flipped = build_vtk_cells(_reversed_rings(mesh))
    original = build_vtk_cells(mesh)
    np.testing.assert_array_equal(original.connectivity, flipped.connectivity)
    np.testing.assert_array_equal(original.offsets, flipped.offsets)
    if original.faces is None:
        assert flipped.faces is None
    else:
        np.testing.assert_array_equal(original.faces, flipped.faces)
        np.testing.assert_array_equal(original.face_offsets, flipped.face_offsets)


def test_2d_cells_are_polygons_wound_counter_clockwise_in_a_padded_plane():
    mesh = structured_grid_2d(2, 2)
    cells = build_vtk_cells(mesh)

    assert cells.n_cells == 4
    np.testing.assert_array_equal(cells.types, VTK_POLYGON)
    np.testing.assert_array_equal(cells.offsets, [4, 8, 12, 16])
    assert cells.faces is None and cells.face_offsets is None
    # VTK points always carry three components; a 2D mesh is laid in the plane z = 0.
    assert cells.points.shape == (mesh.n_nodes, 3)
    np.testing.assert_array_equal(cells.points[:, 2], 0.0)
    np.testing.assert_array_equal(cells.points[:, :2], np.asarray(mesh.node_coords))

    for start, end in _cell_slices(cells.offsets):
        ring = cells.connectivity[start:end]
        assert len(set(ring.tolist())) == 4
        assert _signed_area(cells.points[ring]) > 0


def test_a_2d_ring_walks_the_cell_perimeter_edge_by_edge():
    mesh = structured_grid_2d(3, 2)
    cells = build_vtk_cells(mesh)
    edges = {
        frozenset(np.asarray(mesh.face_nodes.face_node_indices)[start:end].tolist())
        for start, end in pairwise(np.asarray(mesh.face_nodes.offsets))
    }
    for start, end in _cell_slices(cells.offsets):
        ring = cells.connectivity[start:end]
        for a, b in zip(ring, np.roll(ring, -1), strict=True):
            assert frozenset((int(a), int(b))) in edges


def test_a_2d_cell_whose_edges_do_not_chain_is_refused():
    # Two triangles claimed as one cell. Each edge is oriented away from the pair's shared centre
    # rather than around a cell, so the chain breaks at the first edge that leads nowhere.
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [4.0, 0.0], [3.0, 1.0]]
    edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]
    mesh = Mesh.from_faces(coords, edges, [0] * 6, [-1] * 6, 1)
    with pytest.raises(ValueError, match="do not form a closed ring"):
        build_vtk_cells(mesh)


def test_a_2d_cell_whose_edges_form_two_rings_is_refused():
    # Concentric squares as one cell: both rings close, and both wind the same way about the shared
    # centre, so every edge does chain -- into two rings rather than one. A walk of a fixed number
    # of steps would go round the outer ring twice and report a ring of the right length.
    outer = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
    inner = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    edges = [[i, (i + 1) % 4] for i in range(4)] + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
    mesh = Mesh.from_faces(outer + inner, edges, [0] * 8, [-1] * 8, 1)
    with pytest.raises(ValueError, match="more than one closed ring"):
        build_vtk_cells(mesh)
