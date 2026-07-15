"""Unit tests for the extruded-direction collapse transform.

A one-cell-thick :func:`structured_grid_3d` slab is exactly an extruded 2D grid: collapsing away
its through-thickness (``"back"``/``"front"``) direction must reproduce the geometry of the
corresponding :func:`structured_grid_2d`. Because the collapse renumbers nodes and faces, the
comparison is on order-independent geometric invariants, not element-wise arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import (
    collapse_extruded_direction,
    structured_grid_2d,
    structured_grid_3d,
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
    from aquaflux.mesh import Mesh

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
            **{name: np.asarray(slab.face_patches.indices(name)) for name in ("left", "right", "bottom", "top")},
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
    from aquaflux.mesh import Mesh  # local import keeps the zone rebuild explicit

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
