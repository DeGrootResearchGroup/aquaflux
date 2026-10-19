"""Unit tests for :func:`aquaflux.mesh.stored_ring_is_outward`.

The answer is checked against an independent orientation test -- each ring's Newell normal against
the direction from its owner cell's vertex mean to the face -- rather than against the mesh's own
face-geometry strategy, which is what the function itself runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.mesh import stored_ring_is_outward, structured_grid_3d

from tests.support.meshes import (
    hexahedron_beside_a_prism,
    newell_outward,
    perturbed_grid_3d,
    restated,
    tetrahedral_grid_3d,
)


def _scrambled(mesh, seed: int):
    """The mesh with a random half of its rings reversed and of its interior faces turned over."""
    rng = np.random.default_rng(seed)
    interior = np.asarray(mesh.face_cells.neighbour) >= 0
    return restated(
        mesh,
        reverse=rng.random(mesh.n_faces) < 0.5,
        swap=interior & (rng.random(mesh.n_faces) < 0.5),
    )


@pytest.mark.parametrize(
    "mesh",
    [
        structured_grid_3d(3, 2, 2),
        perturbed_grid_3d(3, 3, 2, perturb=0.25, seed=1),
        tetrahedral_grid_3d(2, perturb=0.15, seed=2),
        hexahedron_beside_a_prism(),
    ],
)
def test_the_answer_agrees_with_an_independent_orientation_test(mesh):
    scrambled = _scrambled(mesh, seed=7)
    np.testing.assert_array_equal(stored_ring_is_outward(scrambled), newell_outward(scrambled))


def test_a_structured_grid_stores_rings_in_both_directions():
    # The premise every writer that uses this rests on: "as stored" is not a synonym for
    # "owner-outward", so a writer that kept every owner-listed ring as it found it would emit
    # inward faces on the rest.
    stored = stored_ring_is_outward(structured_grid_3d(2, 2, 2))
    assert 0 < int(np.count_nonzero(stored)) < stored.size
