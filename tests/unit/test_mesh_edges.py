"""The edges of a face-based mesh: each once, and the one each face incidence runs along.

Counts are checked against closed forms -- a structured grid's edges by axis, and Euler's formula
``V - E + F - C = 1`` for any mesh of a ball -- rather than a recorded number.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import structured_grid_3d

from tests.support.meshes import hexahedron_beside_a_prism, tetrahedral_grid_3d


def test_a_structured_grid_has_its_edges_by_axis():
    nx, ny, nz = 3, 2, 4
    edge_nodes, _ = structured_grid_3d(nx, ny, nz).face_nodes.edges()
    expected = nx * (ny + 1) * (nz + 1) + (nx + 1) * ny * (nz + 1) + (nx + 1) * (ny + 1) * nz
    assert edge_nodes.shape == (expected, 2)


@pytest.mark.parametrize(
    "mesh",
    [
        tetrahedral_grid_3d(3, perturb=0.2, seed=1),
        hexahedron_beside_a_prism(),
        structured_grid_3d(2, 3, 2),
    ],
)
def test_eulers_formula_holds_for_a_ball(mesh):
    edge_nodes, _ = mesh.face_nodes.edges()
    assert mesh.n_nodes - edge_nodes.shape[0] + mesh.n_faces - mesh.n_cells == 1


def test_each_incidence_runs_along_its_edge_and_edges_are_unique_and_sorted():
    mesh = tetrahedral_grid_3d(2)
    edge_nodes, incidence_edge = mesh.face_nodes.edges()
    start = np.asarray(mesh.face_nodes.face_node_indices)
    end = start[np.asarray(mesh.face_nodes.next_incidence)]
    np.testing.assert_array_equal(
        edge_nodes[incidence_edge], np.sort(np.stack([start, end], axis=1), axis=1)
    )
    assert np.all(edge_nodes[:, 0] < edge_nodes[:, 1])
    keys = edge_nodes[:, 0] * mesh.n_nodes + edge_nodes[:, 1]
    assert np.all(np.diff(keys) > 0)
