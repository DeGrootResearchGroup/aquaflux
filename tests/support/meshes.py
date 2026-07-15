"""Skewed structured grids for order-of-accuracy studies (test-only, not a shipped API).

:func:`perturbed_grid_2d` / :func:`perturbed_grid_3d` build a clean structured grid and then
displace its interior lattice nodes by an independent uniform random offset per axis (boundary
nodes stay fixed, so the domain is preserved). The displacement is deliberately *irregular*
(per-node white noise, not a smooth field): a smooth displacement would preserve much of a
structured grid's error cancellation, whereas the irregular offset breaks it and exposes a
scheme's true order of accuracy on skewed, non-orthogonal cells.

This skewing exists purely to stress numerical schemes in the test suite, so it lives here rather
than in the library generators. Displacing node coordinates after construction is exactly
equivalent to perturbing before it (the topology is unchanged and geometry is derived from
``node_coords`` on demand), which is why these are thin wrappers over the clean generators.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d


def _displace_interior(mesh: Mesh, n_per_axis, extent_per_axis, perturb: float, seed: int) -> Mesh:
    """Return ``mesh`` with its interior structured-lattice nodes displaced; boundary nodes fixed.

    Each strictly-interior node of the ``(nx+1) x (ny+1)[ x (nz+1)]`` lattice is moved by a uniform
    random offset in ``[-perturb*h, perturb*h]`` per axis (``h`` the cell size on that axis).
    ``perturb == 0`` returns the mesh unchanged.

    Parameters
    ----------
    mesh : Mesh
        The clean structured grid to displace (its ``node_coords`` are the lattice ``index * h``).
    n_per_axis : tuple of int
        Cell counts ``(nx, ny[, nz])`` in x, y[, z] order.
    extent_per_axis : tuple of float
        Domain extents ``(lx, ly[, lz])`` in the same order.
    perturb : float
        Displacement amplitude as a fraction of the per-axis cell size (0 = no displacement).
    seed : int
        Seed for the reproducible offsets.

    Returns
    -------
    Mesh
        The mesh with displaced interior nodes (same topology).
    """
    if not perturb:
        return mesh
    steps = [length / n for length, n in zip(extent_per_axis, n_per_axis, strict=True)]
    # The lattice is raveled in C-order with x fastest, so the index grids run slowest-to-fastest
    # over the reversed axes ((z, y, x) in 3D, (y, x) in 2D) -- the same order node_coords uses.
    reversed_counts = [n + 1 for n in reversed(n_per_axis)]
    index_grids = np.meshgrid(*[np.arange(c) for c in reversed_counts], indexing="ij")
    interior = np.ones(index_grids[0].shape, dtype=bool)
    for index_grid, n in zip(index_grids, reversed(n_per_axis), strict=True):
        interior &= (index_grid > 0) & (index_grid < n)
    rng = np.random.default_rng(seed)
    # Draw per axis in x, y[, z] order so node_coords columns line up with the offsets.
    offsets = np.stack(
        [
            (interior * rng.uniform(-perturb * h, perturb * h, interior.shape)).ravel()
            for h in steps
        ],
        axis=1,
    )
    return eqx.tree_at(lambda m: m.node_coords, mesh, mesh.node_coords + jnp.asarray(offsets))


def perturbed_grid_2d(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    perturb: float = 0.2,
    seed: int = 0,
    named_boundaries: bool = False,
) -> Mesh:
    """A structured quad grid with interior nodes randomly displaced (see module docstring).

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y.
    lx, ly : float
        Domain size.
    perturb : float
        Interior-node displacement amplitude as a fraction of the cell size (0 = orthogonal).
    seed : int
        Seed for the reproducible displacement.
    named_boundaries : bool
        Passed through to :func:`~aquaflux.mesh.structured_grid_2d`.

    Returns
    -------
    Mesh
    """
    mesh = structured_grid_2d(nx, ny, lx, ly, named_boundaries=named_boundaries)
    return _displace_interior(mesh, (nx, ny), (lx, ly), perturb, seed)


def perturbed_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    perturb: float = 0.2,
    seed: int = 0,
    named_boundaries: bool = False,
) -> Mesh:
    """A structured hexahedral grid with interior nodes randomly displaced (see module docstring).

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along x, y, z.
    lx, ly, lz : float
        Domain extent along each axis.
    perturb : float
        Interior-node displacement amplitude as a fraction of the cell size (0 = orthogonal).
    seed : int
        Seed for the reproducible displacement.
    named_boundaries : bool
        Passed through to :func:`~aquaflux.mesh.structured_grid_3d`.

    Returns
    -------
    Mesh
    """
    mesh = structured_grid_3d(nx, ny, nz, lx, ly, lz, named_boundaries=named_boundaries)
    return _displace_interior(mesh, (nx, ny, nz), (lx, ly, lz), perturb, seed)


def columnwise_perturbed_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    perturb: float = 0.2,
    seed: int = 0,
    named_boundaries: bool = False,
) -> Mesh:
    """A hexahedral grid skewed **in-plane** so every cell face stays exactly planar.

    Each interior column of nodes (a fixed ``(i, j)`` for all ``k``) is displaced by the *same*
    random ``(x, y)`` offset, with ``z`` left on its layer plane. Because the displacement is
    constant along ``z``, every face remains planar — the top/bottom faces are the displaced quads
    lying in their ``z`` plane, and the side faces are straight extrusions of an edge — while the
    cells become genuinely non-orthogonal (skewed) in ``x``–``y``.

    This is the 3D counterpart of :func:`perturbed_grid_2d` for schemes whose accuracy depends on
    planar faces (e.g. a Green–Gauss face integral being exact for a quadratic field). A fully
    perturbed hex grid (:func:`perturbed_grid_3d`) *warps* its quad faces, which breaks that
    property for every Green–Gauss scheme and masks the true skewed-mesh order of accuracy; keeping
    faces planar isolates the skew.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along x, y, z.
    lx, ly, lz : float
        Domain extent along each axis.
    perturb : float
        In-plane displacement amplitude as a fraction of the cell size (0 = orthogonal).
    seed : int
        Seed for the reproducible displacement.
    named_boundaries : bool
        Passed through to :func:`~aquaflux.mesh.structured_grid_3d`.

    Returns
    -------
    Mesh
    """
    mesh = structured_grid_3d(nx, ny, nz, lx, ly, lz, named_boundaries=named_boundaries)
    if not perturb:
        return mesh
    hx, hy = lx / nx, ly / ny
    # Node lattice raveled in C-order ((z, y, x), x fastest) — matches node_coords.
    _iz, iy, ix = np.meshgrid(
        np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij"
    )
    xy_interior = (ix > 0) & (ix < nx) & (iy > 0) & (iy < ny)  # all z-layers of an interior column
    rng = np.random.default_rng(seed)
    # One (x, y) offset per (j, i) column, broadcast identically across every z-layer.
    dx = np.broadcast_to(rng.uniform(-perturb * hx, perturb * hx, (ny + 1, nx + 1)), ix.shape)
    dy = np.broadcast_to(rng.uniform(-perturb * hy, perturb * hy, (ny + 1, nx + 1)), ix.shape)
    offsets = np.stack(
        [(xy_interior * dx).ravel(), (xy_interior * dy).ravel(), np.zeros(ix.size)], axis=1
    )
    return eqx.tree_at(lambda m: m.node_coords, mesh, mesh.node_coords + jnp.asarray(offsets))
