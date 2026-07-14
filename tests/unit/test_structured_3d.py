"""Unit tests for the 3D structured hexahedral grid generator and the fast CSR constructor.

The generator is vectorized (no per-face Python loop) and goes through :meth:`Mesh.from_csr`, so
the invariants to check are: the right cell/face counts, an exactly-recovered domain volume,
closed cells (consistent owner-outward normals), and correctly-sized named boundary patches. A
non-cubic, non-unit box catches axis-transposition and scale-factor errors a unit cube would hide.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual, structured_grid_3d

from tests.support.meshes import perturbed_grid_3d


def _expected_faces(nx, ny, nz):
    return (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)


@pytest.mark.parametrize(
    ("nx", "ny", "nz", "lx", "ly", "lz"),
    [(2, 2, 2, 1.0, 1.0, 1.0), (3, 4, 5, 1.0, 1.0, 1.0), (2, 3, 4, 2.0, 3.0, 4.0)],
)
def test_counts_volume_and_closed_cells(nx, ny, nz, lx, ly, lz) -> None:
    mesh = structured_grid_3d(nx, ny, nz, lx, ly, lz)
    cg = mesh.geometry().cell

    assert mesh.n_cells == nx * ny * nz
    assert mesh.n_faces == _expected_faces(nx, ny, nz)
    assert mesh.dim == 3
    assert np.isclose(float(jnp.sum(cg.volume)), lx * ly * lz)  # exact domain volume
    # every cell volume is the same positive hex volume
    assert np.allclose(np.asarray(cg.volume), (lx / nx) * (ly / ny) * (lz / nz))
    # closed cells => owner-outward normals sum to zero per cell
    assert float(jnp.max(jnp.abs(closed_cell_residual(mesh)))) < 1e-10


def test_named_boundaries_have_correct_face_counts() -> None:
    nx, ny, nz = 3, 4, 5
    mesh = structured_grid_3d(nx, ny, nz, named_boundaries=True)
    patches = mesh.face_patches
    # each side is a plane of cells normal to its axis
    assert patches.size("left") == ny * nz
    assert patches.size("right") == ny * nz
    assert patches.size("bottom") == nx * nz
    assert patches.size("top") == nx * nz
    assert patches.size("back") == nx * ny
    assert patches.size("front") == nx * ny
    # the six sides cover exactly the boundary faces
    total_boundary = 2 * (ny * nz + nx * nz + nx * ny)
    assert int(np.sum(np.asarray(mesh.face_cells.neighbour) < 0)) == total_boundary


def test_perturbation_preserves_domain_and_closure() -> None:
    """Interior-node perturbation warps cells but keeps the box and closed cells."""
    mesh = perturbed_grid_3d(5, 5, 5, perturb=0.2, seed=3)
    cg = mesh.geometry().cell
    assert np.isclose(float(jnp.sum(cg.volume)), 1.0)  # boundary nodes fixed => volume preserved
    assert float(jnp.max(jnp.abs(closed_cell_residual(mesh)))) < 1e-10
    # cell volumes now vary (non-uniform), unlike the orthogonal grid
    assert float(jnp.std(cg.volume)) > 0.0
