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


def _geometry_invariants(mesh):
    """Order-independent geometry summary: dims, counts, and sorted volume/area multisets."""
    geometry = mesh.geometry()
    return {
        "dim": mesh.dim,
        "n_cells": mesh.n_cells,
        "n_faces": mesh.n_faces,
        "n_interior": int(np.sum(np.asarray(mesh.face_cells.interior))),
        "volumes": np.sort(np.asarray(geometry.cell.volume)),
        "areas": np.sort(np.asarray(geometry.face.area)),
    }


@pytest.mark.parametrize(("nx", "ny"), [(2, 1), (3, 2), (4, 4)])
def test_collapsed_slab_matches_structured_grid_2d(nx, ny):
    slab = structured_grid_3d(nx, ny, 1, lx=2.0, ly=3.0, lz=0.5, named_boundaries=True)
    collapsed = collapse_extruded_direction(slab, ["back", "front"])
    reference = structured_grid_2d(nx, ny, lx=2.0, ly=3.0)

    got = _geometry_invariants(collapsed)
    want = _geometry_invariants(reference)
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

    got = _geometry_invariants(collapsed)
    want = _geometry_invariants(reference)
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
