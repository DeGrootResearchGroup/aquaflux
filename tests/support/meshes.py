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

from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import Mesh, structured_grid_2d, structured_grid_3d


def geometry_invariants(mesh: Mesh) -> dict:
    """Order-independent geometry summary: dims, counts, and sorted volume/area multisets.

    Two meshes with the same connectivity up to face/cell renumbering (e.g. a structured grid and
    the same domain read back through a different path) produce the same summary, so comparing it
    is a renumbering-robust equality check.
    """
    geometry = mesh.geometry()
    return {
        "dim": mesh.dim,
        "n_cells": mesh.n_cells,
        "n_faces": mesh.n_faces,
        "n_interior": int(np.sum(np.asarray(mesh.face_cells.interior))),
        "volumes": np.sort(np.asarray(geometry.cell.volume)),
        "areas": np.sort(np.asarray(geometry.face.area)),
    }


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


def tetrahedral_grid_3d(n: int, *, perturb: float = 0.0, seed: int = 0) -> Mesh:
    """A conforming tetrahedral mesh of the unit cube — the fixture with FOUR-faced cells.

    Every other 3D fixture here is hexahedral, so a cell always has six faces. That is not a
    harmless simplification: a preconditioner defect on this scheme was found on an automatically
    generated mesh where **every** affected cell had four faces, and it was invisible to the whole
    unit tier for exactly this reason — twice, once in each direction. This closes that gap.

    Each cube of an ``n x n x n`` grid is split into six tetrahedra by the Kuhn subdivision: from
    the cube's low corner, walk to its high corner one axis at a time, once per ordering of the
    three axes. The six paths give six tetrahedra that tile the cube, and because the subdivision
    depends only on the axis ordering it agrees on every shared cube face — so the mesh is
    conforming, and its interior faces match up without a search.

    Parameters
    ----------
    n : int
        Cubes per axis. The mesh has ``6 n**3`` cells.
    perturb : float, optional
        Random displacement of interior nodes, as a fraction of the cube edge. Boundary nodes are
        held so the domain stays the unit cube. ``0`` (default) leaves the nodes on the lattice.
    seed : int, optional
        Seed for that displacement.

    Returns
    -------
    Mesh
        A mesh of ``6 n**3`` tetrahedra, every cell with exactly four triangular faces.
    """
    step, side = 1.0 / n, n + 1
    coords = np.array(
        [
            (i * step, j * step, k * step)
            for i in range(side)
            for j in range(side)
            for k in range(side)
        ],
        dtype=float,
    )
    if perturb:
        rng = np.random.default_rng(seed)
        on_face = np.zeros(len(coords), dtype=bool)
        for axis in range(3):
            on_face |= np.isclose(coords[:, axis], 0.0) | np.isclose(coords[:, axis], 1.0)
        shift = rng.uniform(-perturb, perturb, size=coords.shape) * step
        coords[~on_face] += shift[~on_face]

    def node(i, j, k):
        return (i * side + j) * side + k

    # The six axis orderings; each is a monotone path from the cube's low corner to its high one.
    orderings = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ordering in orderings:
                    corner, vertices = [i, j, k], [node(i, j, k)]
                    for axis in ordering:
                        corner[axis] += 1
                        vertices.append(node(*corner))
                    cells.append(vertices)

    return tetrahedral_mesh(coords, cells)


def tetrahedral_mesh(points, cells) -> Mesh:
    """A face-based mesh of tetrahedra given by their four point indices, in any order.

    Parameters
    ----------
    points : array-like, shape ``(n_points, 3)``
    cells : array-like of int, shape ``(n_cells, 4)``
    """
    cells = np.asarray(cells).tolist()
    # A triangle is shared by at most two tetrahedra, so the first sighting owns it and the second
    # neighbours it. Keyed on the sorted node triple, which is the face's identity whichever cell
    # names it; the stored order is the owner's, and any order of three nodes is a valid perimeter.
    faces, owner, neighbour, seen = [], [], [], {}
    for index, vertices in enumerate(cells):
        for corner in range(4):
            triangle = vertices[:corner] + vertices[corner + 1 :]
            key = tuple(sorted(triangle))
            if key in seen:
                neighbour[seen[key]] = index
            else:
                seen[key] = len(faces)
                faces.append(triangle)
                owner.append(index)
                neighbour.append(-1)

    return Mesh.from_faces(
        jnp.asarray(np.asarray(points, dtype=float)),
        faces,
        jnp.asarray(owner),
        jnp.asarray(neighbour),
        len(cells),
    )


def hexahedron_beside_a_prism():
    """Two cells of different kinds -- the fixture with a hexahedron AND a polyhedron.

    A unit cube at ``x in [0, 1]`` sharing its ``x = 1`` face with a triangular prism beyond it.

    The smallest mesh on which a writer must give one cell a standard type and the other its face
    stream, so an index that confuses the two kinds of cell has somewhere to go wrong.
    """
    coords = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [2, 0, 0], [2, 0, 1],
    ]  # fmt: skip
    faces = [
        [1, 2, 6, 5],  # shared, x = 1
        [0, 3, 7, 4], [0, 1, 5, 4], [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7],  # the cube's others
        [1, 8, 9, 5], [2, 8, 9, 6], [1, 2, 8], [5, 6, 9],  # the prism's others
    ]  # fmt: skip
    owner = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    neighbour = [1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    return Mesh.from_faces(np.asarray(coords, dtype=float), faces, owner, neighbour, 2)


def restated(mesh: Mesh, *, order=None, reverse=None, swap=None) -> Mesh:
    """The same mesh stated differently: faces reordered, rings reversed, interior faces turned over.

    None of the three changes alters the mesh: faces may be stored in any order, a ring describes
    the same face in either direction, and an interior face is the same face whichever of its two
    cells is called the owner. So a consumer that re-derives order and orientation instead of
    trusting how they are stored must give the same answer for the restated mesh as for the
    original. Zones, patches (moved with their faces), patch types and groups are kept.

    Parameters
    ----------
    mesh : Mesh
        A non-periodic mesh (a seam face's ring belongs to its owner's side, so turning it over would
        move it).
    order : array-like of int, shape ``(n_faces,)``, optional
        The old index of each new face, a permutation. Default: the existing order.
    reverse : array-like of bool, shape ``(n_faces,)``, optional
        Faces (by old index) whose node ring is reversed. Default: none.
    swap : array-like of bool, shape ``(n_faces,)``, optional
        Faces (by old index) whose owner and neighbour are exchanged; only interior faces may be
        chosen. Default: none.
    """
    if mesh.face_cells.neighbour_offset is not None:
        raise ValueError("restated needs a non-periodic mesh")
    n_faces = mesh.n_faces
    order = np.arange(n_faces) if order is None else np.asarray(order, dtype=np.int64)
    reverse = np.zeros(n_faces, dtype=bool) if reverse is None else np.asarray(reverse, dtype=bool)
    swap = np.zeros(n_faces, dtype=bool) if swap is None else np.asarray(swap, dtype=bool)
    owner = np.asarray(mesh.face_cells.owner).copy()
    neighbour = np.asarray(mesh.face_cells.neighbour).copy()
    if np.any(swap & (neighbour < 0)):
        raise ValueError("only an interior face can be turned over")
    owner[swap], neighbour[swap] = neighbour[swap], owner[swap]
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    rings = [indices[start:end] for start, end in pairwise(offsets)]
    rings = [rings[f][::-1] if reverse[f] else rings[f] for f in order]
    new_of_old = np.empty(n_faces, dtype=np.int64)
    new_of_old[order] = np.arange(n_faces)
    patches = mesh.face_patches
    named = [name for name in patches.names if name not in ("interior", "boundary")]
    zones = mesh.cell_zones
    return Mesh.from_faces(
        mesh.node_coords,
        rings,
        owner[order],
        neighbour[order],
        mesh.n_cells,
        cell_zones={name: np.asarray(zones.indices(name)) for name in zones.names[1:]},
        face_patches={
            name: np.sort(new_of_old[np.asarray(patches.indices(name))]) for name in named
        },
        patch_types=dict(patches.patch_types),
        patch_groups=dict(patches.patch_groups),
    )


def newell_outward(mesh: Mesh) -> np.ndarray:
    """Whether each stored ring's Newell normal points from its owner's vertex mean toward the face.

    An orientation test that shares nothing with the mesh's own face-geometry strategy (no centre
    fan, no approximate centroid), for checking code that relies on that strategy. Sound on the
    convex cells of the test fixtures, where the vertex mean lies inside the cell.
    """
    coords = np.asarray(mesh.node_coords)
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    rings = [indices[a:b] for a, b in pairwise(offsets)]
    cell_nodes = [set() for _ in range(mesh.n_cells)]
    for face, ring in enumerate(rings):
        cell_nodes[owner[face]].update(ring.tolist())
        if neighbour[face] >= 0:
            cell_nodes[neighbour[face]].update(ring.tolist())
    centre = np.array([coords[sorted(nodes)].mean(axis=0) for nodes in cell_nodes])
    result = np.empty(len(rings), dtype=bool)
    for face, ring in enumerate(rings):
        p = coords[ring]
        newell = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        result[face] = newell @ (p.mean(axis=0) - centre[owner[face]]) > 0.0
    return result


def without_cells(mesh: Mesh, removed, *, patch: str) -> Mesh:
    """``mesh`` with some cells taken out, the faces they exposed put in a new boundary patch.

    For a non-convex domain from a convex one: an L from a block of hexahedra, say. Remaining cells
    and the nodes still used are renumbered in their original order; existing named patches keep
    their faces. A non-periodic mesh with one cell zone.

    Parameters
    ----------
    mesh : Mesh
    removed : array-like of int
        The cells to take out.
    patch : str
        The name of the patch for the faces between a kept cell and a removed one.
    """
    removed = np.asarray(removed, dtype=np.int64)
    keep = np.ones(mesh.n_cells, dtype=bool)
    keep[removed] = False
    new_cell = np.full(mesh.n_cells, -1, dtype=np.int64)
    new_cell[keep] = np.arange(int(keep.sum()))
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    kept_owner = keep[owner]
    kept_neighbour = (neighbour >= 0) & keep[np.maximum(neighbour, 0)]
    faces = np.flatnonzero(kept_owner | kept_neighbour)
    # A face whose owner went keeps its neighbour as owner; its ring direction does not matter.
    new_owner = np.where(kept_owner[faces], new_cell[owner[faces]], new_cell[neighbour[faces]])
    new_neighbour = np.where(
        kept_owner[faces] & kept_neighbour[faces], new_cell[np.maximum(neighbour[faces], 0)], -1
    )
    exposed = (neighbour[faces] >= 0) & (new_neighbour < 0)
    offsets = np.asarray(mesh.face_nodes.offsets)
    indices = np.asarray(mesh.face_nodes.face_node_indices)
    rings = [indices[offsets[f] : offsets[f + 1]] for f in faces]
    used = np.unique(np.concatenate(rings))
    new_node = np.full(mesh.n_nodes, -1, dtype=np.int64)
    new_node[used] = np.arange(used.size)
    new_face = np.full(mesh.n_faces, -1, dtype=np.int64)
    new_face[faces] = np.arange(faces.size)
    patches = mesh.face_patches
    named = {
        name: new_face[np.asarray(patches.indices(name))]
        for name in patches.names
        if name not in ("interior", "boundary")
    }
    named = {name: idx[idx >= 0] for name, idx in named.items()}
    named[patch] = np.flatnonzero(exposed)
    return Mesh.from_faces(
        np.asarray(mesh.node_coords)[used],
        [new_node[ring] for ring in rings],
        new_owner,
        new_neighbour,
        int(keep.sum()),
        face_patches=named,
        patch_types=dict(patches.patch_types),
        patch_groups=dict(patches.patch_groups),
    )
