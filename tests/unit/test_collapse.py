"""Unit tests for the extruded-direction collapse transform.

A one-cell-thick :func:`structured_grid_3d` slab is exactly an extruded 2D grid: collapsing away
its through-thickness (``"back"``/``"front"``) direction must reproduce the geometry of the
corresponding :func:`structured_grid_2d`. Because the collapse renumbers nodes and faces, the
comparison is on order-independent geometric invariants, not element-wise arrays.

The one thing that renumbering can silently lose is a streamwise-periodic seam's per-face
neighbour-image translation, so a hand-built periodic slab (:func:`_periodic_extruded_slab`, since
the structured generators are periodic in 2D only) pins that it survives.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import (
    Mesh,
    closed_cell_residual,
    collapse_extruded_direction,
    structured_grid_2d,
    structured_grid_3d,
)

from tests.support.meshes import geometry_invariants


def _periodic_extruded_slab(nx: int, ny: int, lx: float = 2.0, ly: float = 1.0, lz: float = 0.4):
    """A one-cell-thick 3D slab, periodic along x, capped by a ``"frontAndBack"`` patch in z.

    The structured generators build a periodic mesh in 2D only, so the connectivity is assembled
    here: the x = 0 and x = lx planes are **fused** into one interior seam face per cell row —
    wrapping the last cell of the row back to the first with a ``+lx`` neighbour-image translation
    — while the y planes stay ordinary boundaries and the two z planes are the extrusion caps.
    Collapsing away z must therefore reproduce ``structured_grid_2d(nx, ny, periodic=("x",))``.
    """
    x, y, z = np.linspace(0.0, lx, nx + 1), np.linspace(0.0, ly, ny + 1), np.array([0.0, lz])

    def nid(i, j, k):  # node index (k slowest, i fastest), two node planes in z
        return (k * (ny + 1) + j) * (nx + 1) + i

    def cid(i, j):  # cell index; the layer is one cell thick, so there is no k
        return j * nx + i

    kk, jj, ii = np.meshgrid(np.arange(2), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    coords = np.stack([x[ii].ravel(), y[jj].ravel(), z[kk].ravel()], axis=1)

    nodes: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    neighbour: list[np.ndarray] = []
    offset: list[np.ndarray] = []

    # X-normal faces, one quad per cell row: i in [1, nx] sits at x[i] between cells (i-1, j) and
    # (i mod nx, j). The i == nx face is the seam, wrapping the last cell back to the first with a
    # periodic image +lx along x; the x = 0 plane is that seam's image and is not emitted, so every
    # x-face is interior.
    fi, fj = (a.ravel() for a in np.meshgrid(np.arange(1, nx + 1), np.arange(ny), indexing="ij"))
    nodes.append(
        np.stack([nid(fi, fj, 0), nid(fi, fj + 1, 0), nid(fi, fj + 1, 1), nid(fi, fj, 1)], axis=1)
    )
    owner.append(cid(fi - 1, fj))
    neighbour.append(cid(fi % nx, fj))
    seam = np.zeros((fi.size, 3))
    seam[fi == nx, 0] = lx
    offset.append(seam)

    # Y-normal faces: j in [0, ny], boundary on the j == 0 and j == ny planes.
    fi, fj = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij"))
    nodes.append(
        np.stack([nid(fi, fj, 0), nid(fi + 1, fj, 0), nid(fi + 1, fj, 1), nid(fi, fj, 1)], axis=1)
    )
    low, high = fj > 0, fj < ny
    owner.append(
        np.where(low, cid(fi, np.clip(fj - 1, 0, ny - 1)), cid(fi, np.clip(fj, 0, ny - 1)))
    )
    neighbour.append(np.where(low & high, cid(fi, np.clip(fj, 0, ny - 1)), -1))
    offset.append(np.zeros((fi.size, 3)))

    # Z-normal faces: the two caps of the extrusion, every one a boundary face.
    fi, fj, fk = (
        a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(2), indexing="ij")
    )
    nodes.append(
        np.stack(
            [nid(fi, fj, fk), nid(fi + 1, fj, fk), nid(fi + 1, fj + 1, fk), nid(fi, fj + 1, fk)],
            axis=1,
        )
    )
    owner.append(cid(fi, fj))
    neighbour.append(np.full(fi.size, -1))
    offset.append(np.zeros((fi.size, 3)))

    all_nodes = np.concatenate(nodes, axis=0)
    n_faces = all_nodes.shape[0]
    caps = np.arange(n_faces - fi.size, n_faces)  # the z-normal family, emitted last
    return Mesh.from_csr(
        coords,
        np.arange(n_faces + 1) * 4,
        all_nodes.ravel(),
        np.concatenate(owner),
        np.concatenate(neighbour),
        n_cells=nx * ny,
        face_patches={"frontAndBack": caps},
        neighbour_offset=np.concatenate(offset),
    )


@pytest.mark.parametrize(("nx", "ny"), [(2, 1), (3, 2), (4, 4)])
def test_collapsed_slab_matches_structured_grid_2d(nx, ny):
    slab = structured_grid_3d(nx, ny, 1, lx=2.0, ly=3.0, lz=0.5, named_boundaries=True)
    collapsed = collapse_extruded_direction(slab, ["back", "front"])
    reference = structured_grid_2d(nx, ny, lx=2.0, ly=3.0)

    got = geometry_invariants(collapsed)
    want = geometry_invariants(reference)
    assert got["dim"] == want["dim"] == 2
    assert got["n_cells"] == want["n_cells"]
    assert got["n_faces"] == want["n_faces"]
    assert got["n_interior"] == want["n_interior"]
    np.testing.assert_allclose(got["volumes"], want["volumes"])
    np.testing.assert_allclose(got["areas"], want["areas"])


@pytest.mark.parametrize(("nx", "ny"), [(2, 1), (3, 2), (4, 4)])
def test_collapse_single_frontandback_patch(nx, ny):
    # The standard OpenFOAM 2D convention: one "empty" patch holding both the front and back
    # planes, rather than two separate caps. Collapsing it must match the two-cap result.
    slab = structured_grid_3d(nx, ny, 1, lx=2.0, ly=3.0, lz=0.5, named_boundaries=True)
    front_and_back = np.concatenate(
        [
            np.asarray(slab.face_patches.indices("back")),
            np.asarray(slab.face_patches.indices("front")),
        ]
    )
    merged = Mesh.from_csr(
        slab.node_coords,
        slab.face_nodes.offsets,
        slab.face_nodes.face_node_indices,
        slab.face_cells.owner,
        slab.face_cells.neighbour,
        n_cells=slab.n_cells,
        face_patches={
            **{
                name: np.asarray(slab.face_patches.indices(name))
                for name in ("left", "right", "bottom", "top")
            },
            "frontAndBack": front_and_back,
        },
    )
    collapsed = collapse_extruded_direction(merged, ["frontAndBack"])
    reference = structured_grid_2d(nx, ny, lx=2.0, ly=3.0)

    got = geometry_invariants(collapsed)
    want = geometry_invariants(reference)
    assert got["dim"] == want["dim"] == 2
    assert got["n_cells"] == want["n_cells"]
    assert got["n_faces"] == want["n_faces"]
    assert got["n_interior"] == want["n_interior"]
    np.testing.assert_allclose(got["volumes"], want["volumes"])
    np.testing.assert_allclose(got["areas"], want["areas"])
    names = set(collapsed.face_patches.names)
    assert "frontAndBack" not in names
    assert {"left", "right", "bottom", "top"} <= names


def test_collapse_drops_caps_and_keeps_side_patches():
    slab = structured_grid_3d(3, 2, 1, named_boundaries=True)
    collapsed = collapse_extruded_direction(slab, ["back", "front"])

    names = set(collapsed.face_patches.names)
    assert "back" not in names and "front" not in names
    assert {"left", "right", "bottom", "top"} <= names

    # Each surviving side patch still selects the boundary edges on its own plane.
    centroid = np.asarray(collapsed.geometry().face.centroid)
    left = np.asarray(collapsed.face_patches.mask("left"))
    top = np.asarray(collapsed.face_patches.mask("top"))
    assert np.allclose(centroid[left, 0], 0.0)  # x = 0
    assert np.allclose(centroid[top, 1], 1.0)  # y = ly
    # "left" is the x = 0 boundary: one edge per row of cells (ny = 2).
    reference = structured_grid_2d(3, 2, named_boundaries=True)
    assert collapsed.face_patches.size("left") == reference.face_patches.size("left")


def test_cell_zones_survive_collapse():
    slab = structured_grid_3d(4, 1, 1, named_boundaries=True)
    # Tag two cells as a zone; collapse must carry it through unchanged (cells map 1:1).
    zone_cells = np.array([0, 1])
    zoned = Mesh.from_csr(
        slab.node_coords,
        slab.face_nodes.offsets,
        slab.face_nodes.face_node_indices,
        slab.face_cells.owner,
        slab.face_cells.neighbour,
        n_cells=slab.n_cells,
        cell_zones={"left_half": zone_cells},
        face_patches={
            name: np.asarray(slab.face_patches.indices(name))
            for name in slab.face_patches.names
            if name not in ("interior", "boundary")
        },
    )
    collapsed = collapse_extruded_direction(zoned, ["back", "front"])
    assert "left_half" in collapsed.cell_zones.names
    np.testing.assert_array_equal(np.asarray(collapsed.cell_zones.indices("left_half")), zone_cells)


def test_collapse_carries_the_periodic_offset():
    """A collapsed periodic slab is still periodic — the seam's neighbour-image translation
    survives the face renumbering, projected onto the two surviving axes.

    Guards the single argument that carries it. Without it a seam face's neighbour sits a full
    period away, and the divergence-theorem volume of the boundary-column cells collapses --
    silently, since an absent offset is exactly what an ordinary mesh has.
    """
    nx, ny, lx, ly = 4, 3, 2.0, 1.0
    slab = _periodic_extruded_slab(nx, ny, lx=lx, ly=ly)
    kept = ~np.asarray(slab.face_patches.mask("frontAndBack"))
    # The surviving offsets are the originals' in-plane (x, y) components, the extruded z one
    # dropped: one +lx wrap face per cell row, every other kept face zero.
    expected = np.asarray(slab.face_cells.neighbour_offset)[kept][:, :2]
    assert np.count_nonzero(expected[:, 0] == lx) == ny

    collapsed = collapse_extruded_direction(slab, ["frontAndBack"])

    assert collapsed.face_cells.neighbour_offset is not None
    np.testing.assert_allclose(np.asarray(collapsed.face_cells.neighbour_offset), expected)
    # Every cell keeps its full area; a dropped offset would collapse the x = 0 column's.
    np.testing.assert_allclose(np.asarray(collapsed.geometry().cell.volume), (lx / nx) * (ly / ny))
    assert float(np.max(np.abs(np.asarray(closed_cell_residual(collapsed))))) < 1e-10
    # The collapse reproduces the 2D periodic generator: one seam face per row, no side patches.
    reference = structured_grid_2d(nx, ny, lx=lx, ly=ly, periodic=("x",))
    assert collapsed.n_faces == reference.n_faces
    assert int(np.sum(np.asarray(collapsed.face_cells.interior))) == int(
        np.sum(np.asarray(reference.face_cells.interior))
    )


def test_collapse_leaves_a_non_periodic_slab_offset_free():
    """An ordinary slab gains no offset array from the collapse (``None`` in, ``None`` out)."""
    slab = structured_grid_3d(3, 2, 1, named_boundaries=True)
    assert slab.face_cells.neighbour_offset is None
    assert collapse_extruded_direction(slab, ["back", "front"]).face_cells.neighbour_offset is None


def _extruded_triangle_and_quad(lz: float = 0.5) -> Mesh:
    """Two right triangles (a unit square split by its diagonal) beside a unit-square quad cell,
    extruded in z by ``lz``.

    Every structured-grid fixture in this file caps an extrusion with quads (4-node faces) and
    keeps every cap face at the same node count, so the collapse's per-face reductions never have
    to treat a subset of faces as genuinely ragged. This mesh's caps mix **triangles** (3 nodes,
    cells A and B) with a **quad** (4 nodes, cell C) *within the same removed patch*, which a
    vectorization that silently assumed a uniform node count across a whole subset — reshaping to
    ``(n, k)`` for a fixed ``k``, or reusing one face's row length for another's — would get wrong
    here while still passing on any single-shape mesh.

    2D layout (before extrusion, y up, x right)::

        3 --- 2 --- 5
        | B  /|  C  |
        |  /  |     |
        |/  A |     |
        0 --- 1 --- 4

    Triangle A = (0,1,2), triangle B = (0,2,3), quad C = (1,4,5,2); A and B share the diagonal
    (0,2), A and C share the edge (1,2) — the two surviving interior faces after collapse.
    """
    nodes_2d = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    front = np.hstack([nodes_2d, np.zeros((6, 1))])
    back = np.hstack([nodes_2d, np.full((6, 1), lz)])
    nodes = np.vstack([front, back])  # indices 0-5 front, 6-11 back (+6 offset)

    faces = [
        [0, 1, 2],  # cap front A (triangle)
        [0, 2, 3],  # cap front B (triangle)
        [1, 4, 5, 2],  # cap front C (quad)
        [6, 7, 8],  # cap back A
        [6, 8, 9],  # cap back B
        [7, 10, 11, 8],  # cap back C
        [0, 1, 7, 6],  # side: bottom of A, boundary
        [1, 2, 8, 7],  # side: A/C shared edge, interior
        [0, 2, 8, 6],  # side: A/B shared diagonal, interior
        [2, 3, 9, 8],  # side: top of B, boundary
        [3, 0, 6, 9],  # side: left of B, boundary
        [1, 4, 10, 7],  # side: bottom of C, boundary
        [4, 5, 11, 10],  # side: right of C, boundary
        [5, 2, 8, 11],  # side: top of C, boundary
    ]
    owner = [0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 2, 2, 2]
    neighbour = [-1, -1, -1, -1, -1, -1, -1, 2, 1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(
        nodes,
        faces,
        owner,
        neighbour,
        n_cells=3,
        face_patches={"front": np.array([0, 1, 2]), "back": np.array([3, 4, 5])},
    )


def test_collapse_reduces_mixed_polygon_cap_faces_correctly():
    """The cap-axis inference and side-quad reduction both handle a genuinely ragged subset.

    Collapsing must find z as the extruded axis from cap faces that mix a 3-node and a 4-node
    polygon *in one removed patch*, and correctly reduce all eight side quads to 2D edges,
    reproducing the two right triangles' and the unit square's combined areas exactly. A
    vectorization that assumes one node count per subset (reshape-to-fixed-width, or that reuses
    one face's row length for another's) would corrupt this even though it can pass on a
    same-shape-only fixture.
    """
    mesh = _extruded_triangle_and_quad()
    collapsed = collapse_extruded_direction(mesh, ["front", "back"])

    assert collapsed.dim == 2
    assert collapsed.n_cells == 3
    assert collapsed.n_faces == 8
    assert int(np.sum(np.asarray(collapsed.face_cells.interior))) == 2  # the two shared edges
    volumes = np.sort(np.asarray(collapsed.geometry().cell.volume))
    np.testing.assert_allclose(volumes, [0.5, 0.5, 1.0])
    assert float(np.max(np.abs(np.asarray(closed_cell_residual(collapsed))))) < 1e-10


def test_collapse_rejects_a_non_planar_capping_face():
    """A cap face that is warped out of every coordinate plane is refused, not silently accepted.

    One node of the z = 0 cap is pulled out of plane, so that face's nodes are no longer constant
    along *any* single axis (its spread exceeds tolerance on x, y, *and* z) -- the "capping face is
    not planar" branch, distinct from the "two caps normal to different axes" case covered by
    ``test_caps_on_different_axes_rejected``.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.3],  # pulled out of the z = 0 plane
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = [
        [0, 1, 2, 3],  # cap front -- warped
        [4, 5, 6, 7],  # cap back -- planar
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    owner = [0, 0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1, -1]
    mesh = Mesh.from_faces(
        nodes,
        faces,
        owner,
        neighbour,
        n_cells=1,
        face_patches={"front": np.array([0]), "back": np.array([1])},
    )

    with pytest.raises(ValueError, match="not planar and normal to a single axis"):
        collapse_extruded_direction(mesh, ["front", "back"])


def test_collapse_rejects_a_degenerate_side_edge():
    """A side face that reduces to a single in-plane node (not two) is refused.

    Node 1 sits on top of node 0's ``(x, y)`` position, so the bottom side face's edge degenerates
    to a point once z is dropped -- one *fewer* distinct node than the ``4``-distinct case
    ``test_non_extrusion_rejected`` covers, exercising the other side of the "not 2" check.
    """
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # collapsed onto node 0's (x, y)
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = [
        [0, 1, 2, 3],  # cap front
        [4, 5, 6, 7],  # cap back
        [0, 1, 5, 4],  # bottom side -- degenerate: 0 and 1 coincide in-plane
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    owner = [0, 0, 0, 0, 0, 0]
    neighbour = [-1, -1, -1, -1, -1, -1]
    mesh = Mesh.from_faces(
        nodes,
        faces,
        owner,
        neighbour,
        n_cells=1,
        face_patches={"front": np.array([0]), "back": np.array([1])},
    )

    with pytest.raises(ValueError, match="reduces to 1 distinct"):
        collapse_extruded_direction(mesh, ["front", "back"])


def test_requires_at_least_one_patch():
    slab = structured_grid_3d(2, 1, 1, named_boundaries=True)
    with pytest.raises(ValueError, match="at least one"):
        collapse_extruded_direction(slab, [])


def test_single_cap_plane_rejected():
    # Only one of the two caps: the removed faces span a single plane, not the two an extrusion needs.
    slab = structured_grid_3d(2, 1, 1, named_boundaries=True)
    with pytest.raises(ValueError, match="two parallel planes"):
        collapse_extruded_direction(slab, ["back"])


def test_caps_on_different_axes_rejected():
    slab = structured_grid_3d(2, 1, 1, named_boundaries=True)
    with pytest.raises(ValueError, match="different axes"):
        collapse_extruded_direction(slab, ["left", "bottom"])


def test_non_extrusion_rejected():
    # Two cells thick along x: removing the x-caps leaves a genuine interior quad, not an edge.
    slab = structured_grid_3d(2, 1, 1, named_boundaries=True)
    with pytest.raises(ValueError, match="not a one-cell-thick extrusion"):
        collapse_extruded_direction(slab, ["left", "right"])
