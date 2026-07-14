"""Unit tests for the 2D structured quad grid generator (counts, closure, named patches).

Like :mod:`test_structured_3d`, but for :func:`structured_grid_2d`: the generator is vectorized
and goes through :meth:`Mesh.from_csr`, so the invariants to check are the right cell/face counts,
an exactly-recovered domain area, closed cells (consistent owner-outward normals), and
correctly-sized named boundary patches — including under interior-node perturbation.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import closed_cell_residual, structured_grid_2d

from tests.support.meshes import perturbed_grid_2d


def _expected_faces_2d(nx, ny):
    return (nx + 1) * ny + nx * (ny + 1)


@pytest.mark.parametrize(
    ("nx", "ny", "lx", "ly"),
    [(2, 2, 1.0, 1.0), (3, 4, 1.0, 1.0), (4, 3, 2.0, 3.0)],
)
def test_counts_area_and_closed_cells(nx, ny, lx, ly) -> None:
    mesh = structured_grid_2d(nx, ny, lx, ly)
    cg = mesh.geometry().cell

    assert mesh.n_cells == nx * ny
    assert mesh.n_faces == _expected_faces_2d(nx, ny)
    assert mesh.dim == 2
    assert np.isclose(float(jnp.sum(cg.volume)), lx * ly)  # exact domain area
    assert np.allclose(np.asarray(cg.volume), (lx / nx) * (ly / ny))  # uniform cell area
    assert float(jnp.max(jnp.abs(closed_cell_residual(mesh)))) < 1e-10


def test_named_boundaries_have_correct_face_counts() -> None:
    nx, ny = 3, 4
    mesh = structured_grid_2d(nx, ny, named_boundaries=True)
    patches = mesh.face_patches
    assert patches.size("left") == ny
    assert patches.size("right") == ny
    assert patches.size("bottom") == nx
    assert patches.size("top") == nx
    # the four sides cover exactly the boundary faces
    assert int(np.sum(np.asarray(mesh.face_cells.neighbour) < 0)) == 2 * (nx + ny)


def test_perturbation_preserves_domain_and_closure() -> None:
    """Interior-node perturbation warps cells but keeps the box and closed cells."""
    mesh = perturbed_grid_2d(6, 6, perturb=0.2, seed=3)
    cg = mesh.geometry().cell
    assert np.isclose(float(jnp.sum(cg.volume)), 1.0)  # boundary nodes fixed => area preserved
    assert float(jnp.max(jnp.abs(closed_cell_residual(mesh)))) < 1e-10
    assert float(jnp.std(cg.volume)) > 0.0  # cell areas now vary
