"""Unit tests for the face-based-mesh to VTK-connectivity reconstruction. No filesystem.

The property that carries this module is the **winding**: every polyhedron face must be listed
outward from the cell listing it, and a mesh does not store its face rings in any particular
direction. So the checks here are geometric -- each emitted ring's own normal, against the cell it
was emitted for -- rather than comparisons against a hand-written expected array, which would pin
one mesh generator's incidental winding rather than the rule.
"""

from __future__ import annotations

from itertools import pairwise

import equinox as eqx
import numpy as np
import pytest
from aquaflux.io.openfoam.assembler import assemble
from aquaflux.io.vtk.topology import (
    VTK_HEXAHEDRON,
    VTK_POLYGON,
    VTK_POLYHEDRON,
    build_vtk_cells,
    stored_ring_is_outward,
)
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d

from tests.support.meshes import (
    hexahedron_beside_a_prism,
    perturbed_grid_3d,
    tetrahedral_grid_3d,
)
from tests.support.polymesh import cyclic_slab_polymesh_data, cyclic_two_cube_polymesh_data


def _cell_slices(offsets):
    """(start, end) per cell from the VTK end-offset convention."""
    ends = np.asarray(offsets)
    return zip(np.concatenate([[0], ends[:-1]]), ends, strict=True)


def _cell_faces(cells):
    """The face rings of each polyhedron, decoded from the face stream; other cells are skipped."""
    stream = cells.faces
    ends = np.asarray(cells.face_offsets)
    polyhedra = ends >= 0
    starts = np.concatenate([[0], ends[polyhedra][:-1]])
    for start, end in zip(starts, ends[polyhedra], strict=True):
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


def _hexahedra(cells):
    """The eight ordered vertices of every hexahedral cell, shape ``(n_hexahedra, 8)``."""
    ends = np.asarray(cells.offsets)[cells.types == VTK_HEXAHEDRON]
    return cells.connectivity[ends[:, None] - 8 + np.arange(8)]


#: A VTK hexahedron's six faces, as positions in its eight vertices, each wound to point outward:
#: the base (0-3) faces into the cell and is reversed; the top (4-7) shares its sense, so points out.
_HEXAHEDRON_FACES = (
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)  # fmt: skip


def _outward_rings(cells):
    """Every cell's outward face rings: a polyhedron's from its face stream, a hexahedron's from VTK's
    vertex order."""
    polyhedra = _cell_faces(cells) if cells.faces is not None else iter(())
    for cell_type, (start, end) in zip(cells.types, _cell_slices(cells.offsets), strict=True):
        if cell_type == VTK_HEXAHEDRON:
            vertices = cells.connectivity[start:end]
            yield [vertices[list(face)] for face in _HEXAHEDRON_FACES]
        else:
            yield next(polyhedra)


def _mesh_face_sets(mesh):
    """Every face of the mesh, as a frozenset of its nodes."""
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    return {
        frozenset(indices[start:end].tolist())
        for start, end in pairwise(np.asarray(mesh.face_nodes.offsets))
    }


def test_a_structured_grid_is_written_as_hexahedra_with_no_face_stream():
    mesh = structured_grid_3d(2, 1, 1)
    cells = build_vtk_cells(mesh)

    assert cells.n_cells == 2
    assert cells.n_points == mesh.n_nodes
    np.testing.assert_array_equal(cells.types, VTK_HEXAHEDRON)
    np.testing.assert_array_equal(cells.offsets, [8, 16])
    assert cells.faces is None and cells.face_offsets is None


@pytest.mark.parametrize(
    "mesh",
    [structured_grid_3d(2, 3, 2), perturbed_grid_3d(3, 2, 2, perturb=0.25, seed=3)],
    ids=["structured", "perturbed"],
)
def test_hexahedron_vertices_are_in_vtk_order(mesh):
    # VTK's hexahedron: vertices 0-3 are a face whose right-hand normal points INTO the cell, 4-7
    # the opposite face in the same rotational sense, and each i joined to i + 4 by an edge. Those
    # three properties fix the order up to the choice of base face and starting vertex, which VTK
    # does not care about -- so they are what is checked, geometrically, rather than one array.
    cells = build_vtk_cells(mesh)
    centroid = np.asarray(mesh.geometry().cell.centroid)
    faces = _mesh_face_sets(mesh)
    hexahedra = _hexahedra(cells)
    assert hexahedra.shape == (mesh.n_cells, 8)

    for cell, vertices in enumerate(hexahedra):
        base, top = vertices[:4], vertices[4:]
        assert frozenset(base.tolist()) in faces and frozenset(top.tolist()) in faces
        inward = centroid[cell] - cells.points[base].mean(axis=0)
        assert np.dot(_newell_normal(cells.points[base]), inward) > 0
        # The same rotational sense seen from the base: the top's normal also points base-to-top.
        assert np.dot(_newell_normal(cells.points[top]), inward) > 0
        for i in range(4):
            side = frozenset(vertices[[i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4]].tolist())
            assert side in faces


def test_a_hexahedral_cells_volume_is_the_meshs_own():
    # Independent of the ordering checks above: the volume of the hexahedron VTK would build from
    # these eight vertices -- five tetrahedra of the standard split -- must be the cell's volume.
    mesh = perturbed_grid_3d(3, 3, 2, perturb=0.2, seed=1)
    cells = build_vtk_cells(mesh)
    p = cells.points[_hexahedra(cells)]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]
    volume = sum(
        np.einsum("ij,ij->i", p[:, b] - p[:, a], np.cross(p[:, c] - p[:, a], p[:, d] - p[:, a]))
        / 6.0
        for a, b, c, d in tets
    )
    np.testing.assert_allclose(volume, np.asarray(mesh.geometry().cell.volume), rtol=0.05)
    assert np.all(volume > 0)


def test_tetrahedra_stay_polyhedra_carrying_every_bounding_face():
    mesh = tetrahedral_grid_3d(1)
    cells = build_vtk_cells(mesh)

    np.testing.assert_array_equal(cells.types, VTK_POLYHEDRON)
    rings = list(_cell_faces(cells))
    assert len(rings) == mesh.n_cells
    assert all(sorted(map(len, cell)) == [3, 3, 3, 3] for cell in rings)
    for start, end in _cell_slices(cells.offsets):
        assert len(set(cells.connectivity[start:end].tolist())) == 4
    # Per cell: the face count, then each face's own count and its nodes.
    np.testing.assert_array_equal(np.diff(cells.face_offsets, prepend=0), 1 + 4 * 4)


def test_a_mixed_mesh_carries_a_face_stream_for_its_polyhedra_only():
    mesh = hexahedron_beside_a_prism()
    cells = build_vtk_cells(mesh)

    np.testing.assert_array_equal(cells.types, [VTK_HEXAHEDRON, VTK_POLYHEDRON])
    np.testing.assert_array_equal(cells.offsets, [8, 14])
    # The prism alone: its face count, then two triangles and three quadrilaterals.
    np.testing.assert_array_equal(cells.face_offsets, [-1, 1 + 2 * 4 + 3 * 5])
    assert cells.faces.shape == (1 + 2 * 4 + 3 * 5,)
    (prism,) = _cell_faces(cells)
    assert sorted(map(len, prism)) == [3, 3, 4, 4, 4]
    assert {frozenset(r.tolist()) for r in prism} >= {frozenset({1, 2, 6, 5}), frozenset({1, 2, 8})}


@pytest.mark.parametrize("mesh", [tetrahedral_grid_3d(1), hexahedron_beside_a_prism()])
def test_every_emitted_polyhedron_face_winds_outward_from_its_own_cell(mesh):
    cells = build_vtk_cells(mesh)
    centroid = np.asarray(mesh.geometry().cell.centroid)
    polyhedra = np.flatnonzero(cells.types == VTK_POLYHEDRON)

    for index, rings in zip(polyhedra, _cell_faces(cells), strict=True):
        for ring in rings:
            points = cells.points[ring]
            outward = points.mean(axis=0) - centroid[index]
            assert np.dot(_newell_normal(points), outward) > 0


def test_a_shared_face_is_listed_in_opposite_directions_by_its_two_cells():
    mesh = tetrahedral_grid_3d(1)
    rings = list(_cell_faces(build_vtk_cells(mesh)))
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    face = int(np.flatnonzero(neighbour >= 0)[0])
    nodes = set(np.asarray(mesh.face_nodes.face_node_indices)[
        int(mesh.face_nodes.offsets[face]) : int(mesh.face_nodes.offsets[face + 1])
    ].tolist())  # fmt: skip

    (a,) = [r for r in rings[int(owner[face])] if set(r.tolist()) == nodes]
    (b,) = [r for r in rings[int(neighbour[face])] if set(r.tolist()) == nodes]
    # Same ring, opposite direction: rotating one reversal onto the other must line up exactly.
    reversed_b = b[::-1]
    roll = int(np.flatnonzero(reversed_b == a[0])[0])
    np.testing.assert_array_equal(np.roll(reversed_b, -roll), a)


def test_a_structured_grid_stores_rings_in_both_directions():
    # The premise the winding rule rests on: "as stored" is not a synonym for "owner-outward", so a
    # writer that kept every owner-listed ring as it found it would emit inward faces on the rest.
    stored = stored_ring_is_outward(structured_grid_3d(2, 2, 2))
    assert 0 < int(np.count_nonzero(stored)) < stored.size


@pytest.mark.parametrize(
    "mesh",
    [
        structured_grid_3d(2, 2, 1),
        tetrahedral_grid_3d(1),
        hexahedron_beside_a_prism(),
        structured_grid_2d(3, 2),
    ],
)
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


# --- periodic seams ---------------------------------------------------------------------------------
#
# A periodic seam face is stored once, with its nodes on the owner's side of the domain. Each check
# below recomputes a cell's size *from the emitted connectivity alone* and compares it with the
# mesh's own cell volume: a seam ring handed to the neighbour untranslated reaches across the whole
# period, which in two dimensions breaks the edge chain and in three silently inflates the cell.


def _periodic_grid_2d():
    # Unequal rows, so a ring taken from the wrong row would carry the wrong area.
    return structured_grid_2d(4, 3, periodic=("x",), y_nodes=[0.0, 0.2, 0.7, 1.5])


def _polyhedron_volume(points, rings):
    """Divergence-theorem volume of a closed polyhedron from its outward-wound planar face rings."""
    return (
        sum(
            float(np.dot(points[ring].mean(axis=0), _newell_normal(points[ring]))) for ring in rings
        )
        / 3.0
    )


def test_a_periodic_2d_cell_is_a_ring_of_its_own_nodes_with_its_own_area():
    mesh = _periodic_grid_2d()
    cells = build_vtk_cells(mesh)
    volume = np.asarray(mesh.geometry().cell.volume)

    for index, (start, end) in enumerate(_cell_slices(cells.offsets)):
        ring = cells.connectivity[start:end]
        assert len(set(ring.tolist())) == 4
        np.testing.assert_allclose(_signed_area(cells.points[ring]), volume[index], rtol=1e-12)


@pytest.mark.parametrize(
    "mesh",
    [assemble(cyclic_slab_polymesh_data(4, 3)), assemble(cyclic_two_cube_polymesh_data())],
    ids=["cyclic-slab", "cyclic-two-cube"],
)
def test_a_periodic_3d_cell_is_closed_by_its_own_nodes_with_its_own_volume(mesh):
    assert mesh.face_cells.neighbour_offset is not None  # the fixture really is periodic
    cells = build_vtk_cells(mesh)
    volume = np.asarray(mesh.geometry().cell.volume)
    centroid = np.asarray(mesh.geometry().cell.centroid)

    for index, ((start, end), rings) in enumerate(
        zip(_cell_slices(cells.offsets), _outward_rings(cells), strict=True)
    ):
        assert end - start == 8  # a hexahedron's own eight points, none from across the period
        np.testing.assert_allclose(
            _polyhedron_volume(cells.points, rings), volume[index], rtol=1e-12
        )
        for ring in rings:
            outward = cells.points[ring].mean(axis=0) - centroid[index]
            assert np.dot(_newell_normal(cells.points[ring]), outward) > 0


def test_a_periodic_seam_node_with_no_counterpart_is_refused():
    # Move one node of the x = 0 column (the seam's image side) off the translated position of its
    # x = lx partner by a third of the row height: the two sides no longer match.
    mesh = _periodic_grid_2d()
    coords = np.asarray(mesh.node_coords).copy()
    moved = int(np.flatnonzero((coords[:, 0] == 0.0) & (coords[:, 1] == 0.7))[0])
    coords[moved, 1] += 0.25
    mismatched = eqx.tree_at(lambda m: m.node_coords, mesh, coords)
    with pytest.raises(ValueError, match="no counterpart on the other side"):
        build_vtk_cells(mismatched)
