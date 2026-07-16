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
from aquaflux.mesh import closed_cell_residual, graded_nodes, structured_grid_2d

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


def test_graded_nodes_span_and_symmetry() -> None:
    """Double-sided grading spans the axis exactly, clusters at both walls, and is symmetric."""
    nodes = graded_nodes(10, 2.0, growth=1.3)
    assert nodes.shape == (11,)
    assert np.isclose(nodes[0], 0.0) and np.isclose(nodes[-1], 2.0)
    sizes = np.diff(nodes)
    assert np.all(sizes > 0.0)  # strictly increasing nodes
    assert np.allclose(sizes, sizes[::-1])  # symmetric about the centre
    assert sizes[0] < sizes[len(sizes) // 2]  # finest at the wall, coarsest at the centre
    assert np.isclose(sizes[1] / sizes[0], 1.3)  # geometric cell-to-cell ratio


def test_graded_nodes_one_sided_is_monotone() -> None:
    """One-sided grading is finest at 0 and coarsens monotonically to the far end."""
    nodes = graded_nodes(8, 1.0, growth=1.2, both_sides=False)
    sizes = np.diff(nodes)
    assert np.all(np.diff(sizes) > 0.0)  # every cell larger than the last
    assert np.allclose(sizes[1:] / sizes[:-1], 1.2)


def test_graded_nodes_uniform_when_growth_one() -> None:
    assert np.allclose(graded_nodes(5, 1.0, growth=1.0), np.linspace(0.0, 1.0, 6))


def test_graded_grid_area_and_closure() -> None:
    """A wall-graded structured grid still recovers the box area with closed cells."""
    y = graded_nodes(16, 1.0, growth=1.2)
    mesh = structured_grid_2d(8, 16, lx=2.0, ly=1.0, y_nodes=y)
    cg = mesh.geometry().cell
    assert mesh.n_cells == 8 * 16
    assert np.isclose(float(jnp.sum(cg.volume)), 2.0)  # exact area despite non-uniform spacing
    assert float(jnp.std(cg.volume)) > 0.0  # cells vary in the graded direction
    assert float(jnp.max(jnp.abs(closed_cell_residual(mesh)))) < 1e-10


def test_structured_grid_rejects_bad_node_coordinates() -> None:
    with pytest.raises(ValueError, match="shape"):
        structured_grid_2d(4, 4, y_nodes=np.linspace(0.0, 1.0, 4))  # wrong length (need ny+1)
    with pytest.raises(ValueError, match="increasing"):
        structured_grid_2d(4, 4, x_nodes=np.array([0.0, 0.3, 0.2, 0.7, 1.0]))  # not monotone
