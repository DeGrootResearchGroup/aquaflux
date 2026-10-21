"""Hexahedral blocks joined into one conformal mesh of hexahedra.

Blocks that share a face are joined by identifying the points they share: two blocks built on one
face (the same edges, the same surface, the same node fractions) give its points identically up to
rounding, so points closer than a small tolerance are taken to be one. A face of a block then either
meets another block's face whole -- every one of its quadrilaterals is also one of the other's -- or
lies on the boundary. A face met only in part is a non-conformal join, which a mesh cannot hold, and
is refused, as is one whose points inside it are also another block's while its quadrilaterals
are not (two faces cut differently). Two faces that overlap without sharing a single point are not
seen as a join: both stay on the boundary.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from aquaflux.mesh import Mesh

from .blocks import FACES, HexBlock
from .elements import element_mesh, face_keys

__all__ = ["BlockMesh"]

#: A block cell's corners, as offsets ``(di, dj, dk)`` in the hexahedron's corner order.
_CELL_CORNERS = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]


@dataclasses.dataclass(frozen=True)
class BlockMesh:
    """Hexahedra of several blocks, their shared points identified once, and their boundary.

    Build one with :meth:`assemble`.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Each point once, in the order the blocks' nodes first give it (block by block, each in
        ``(i, j, k)`` order with ``k`` fastest).
    hexahedra : np.ndarray of int, shape ``(n_cells, 8)``
        Each hexahedron's corners in the hexahedron's order, right-handed; block by block, each in
        ``(i, j, k)`` order with ``k`` fastest.
    boundary : np.ndarray of int, shape ``(n_faces, 4)``
        The boundary quadrilaterals, each ring in order, block face by block face.
    boundary_block : np.ndarray of int, shape ``(n_faces,)``
        The block each boundary quadrilateral belongs to.
    boundary_face : np.ndarray of int, shape ``(n_faces,)``
        The block face (an index into ``FACES``: ``u-, u+, v-, v+, w-, w+``) it lies on.
    blocks : tuple of HexBlock
    """

    points: np.ndarray
    hexahedra: np.ndarray
    boundary: np.ndarray
    boundary_block: np.ndarray
    boundary_face: np.ndarray
    blocks: tuple

    @classmethod
    def assemble(
        cls, blocks: Sequence[HexBlock], tolerance: float | None = None, count_factor: float = 1.0
    ) -> BlockMesh:
        """Join blocks into one mesh, identifying the points they share.

        Parameters
        ----------
        blocks : sequence of HexBlock
        tolerance : float, optional
            Points closer than this are one point. Unset, a millionth of the shortest edge of any
            cell.
        count_factor : float
            Every block's cell counts scaled by this (:meth:`HexBlock.scaled_counts`): a whole
            mesh made finer or coarser in one setting. Unset, 1.

        Returns
        -------
        BlockMesh

        Raises
        ------
        ValueError
            If two points of one block would be identified (the tolerance is too large, or the
            block is degenerate), or a block face meets other blocks on only part of it, or a
            quadrilateral is on more than two blocks.
        """
        blocks = tuple(blocks)
        if count_factor != 1.0:
            blocks = tuple(block.scaled_counts(count_factor) for block in blocks)
        grids = [block.points() for block in blocks]
        if tolerance is None:
            tolerance = 1e-6 * min(_shortest_edge(grid) for grid in grids)
        flat = np.concatenate([grid.reshape(-1, 3) for grid in grids])
        owner_block = np.concatenate(
            [np.full(grid[..., 0].size, b) for b, grid in enumerate(grids)]
        )
        number = _identified(flat, owner_block, tolerance)
        hexahedra, quads, quad_block, quad_face, inside = [], [], [], [], []
        base = 0
        for b, grid in enumerate(grids):
            index = number[base : base + grid[..., 0].size].reshape(grid.shape[:3])
            base += grid[..., 0].size
            hexahedra.append(_cells(index))
            for face in range(len(FACES)):
                plane = np.take(index, -(face % 2), axis=face // 2)
                quads.append(_quads(plane))
                quad_block.append(np.full(quads[-1].shape[0], b))
                quad_face.append(np.full(quads[-1].shape[0], face))
                inside.append(plane[1:-1, 1:-1].ravel())
        # How many blocks give each point: a point inside a face that another block also gives
        # is a sign the two blocks meet across that face.
        given = np.unique(np.stack([number, owner_block], axis=1), axis=0)
        givers = np.bincount(given[:, 0], minlength=number.max() + 1)
        quads = np.concatenate(quads)
        quad_block, quad_face = np.concatenate(quad_block), np.concatenate(quad_face)
        shared_inside = np.array([np.any(givers[points] > 1) for points in inside])
        on_boundary = _boundary_quads(quads, quad_block, quad_face, shared_inside, len(givers))
        first = np.empty(number.max() + 1, dtype=np.int64)
        first[number[::-1]] = np.arange(number.size)[::-1]  # the earliest node giving each point
        return cls(
            points=flat[first],
            hexahedra=np.concatenate(hexahedra),
            boundary=quads[on_boundary],
            boundary_block=quad_block[on_boundary],
            boundary_face=quad_face[on_boundary],
            blocks=blocks,
        )

    def patches(self, default_patch: str | None = None) -> dict[str, np.ndarray]:
        """The boundary quadrilaterals in each patch the blocks name, ``(n, 4)`` each.

        Parameters
        ----------
        default_patch : str, optional
            The patch of every boundary block face the blocks leave unnamed. Unset, there must be
            none.

        Raises
        ------
        ValueError
            If a boundary block face is in no patch and no default is given.
        """
        names = np.array(
            [
                self.blocks[b].patches.get(FACES[f])
                for b, f in zip(self.boundary_block, self.boundary_face, strict=True)
            ],
            dtype=object,
        )
        if default_patch is not None:
            names[np.equal(names, None)] = default_patch
        unnamed = sorted(
            {
                (int(b), FACES[f])
                for b, f, n in zip(self.boundary_block, self.boundary_face, names, strict=True)
                if n is None
            }
        )
        if unnamed:
            raise ValueError(f"boundary block face(s) {unnamed} (block, face) are in no patch")
        return {str(name): self.boundary[names == name] for name in dict.fromkeys(names)}

    def mesh(
        self,
        named: bool = True,
        default_patch: str | None = None,
        patch_types: Mapping[str, str] | None = None,
    ) -> Mesh:
        """The face-based mesh of the hexahedra.

        Parameters
        ----------
        named : bool
            Put the boundary faces in the blocks' patches (:meth:`patches`); otherwise leave the
            boundary unnamed.
        default_patch : str, optional
            The patch of the boundary faces the blocks leave unnamed (see :meth:`patches`).
        patch_types : mapping of {str: str}, optional
            A type for named patches (``wall``, ``patch``, ``symmetry``, ...), carried on the mesh.

        Returns
        -------
        Mesh
        """
        return element_mesh(
            self.points,
            hexahedra=self.hexahedra,
            patches=self.patches(default_patch) if named else None,
            patch_types=patch_types,
        )


def _shortest_edge(grid) -> float:
    return min(
        float(np.min(np.linalg.norm(np.diff(grid, axis=axis), axis=-1))) for axis in range(3)
    )


def _identified(points, owner_block, tolerance) -> np.ndarray:
    """Each point's number once points within ``tolerance`` are one, numbered as first seen."""
    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    same_block = owner_block[pairs[:, 0]] == owner_block[pairs[:, 1]]
    if np.any(same_block):
        blocks = sorted(set(owner_block[pairs[same_block, 0]].tolist()))
        raise ValueError(
            f"block(s) {blocks} have points closer than the tolerance {tolerance:g}: a degenerate "
            "block, or a tolerance too large for its cells"
        )
    n = points.shape[0]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    # Number each group by its first point, in point order.
    first = np.full(label.max() + 1, n)
    np.minimum.at(first, label, np.arange(n))
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[label]


def _cells(index) -> np.ndarray:
    """A block's hexahedra from its point numbers, ``(i, j, k)`` order with ``k`` fastest."""
    ni, nj, nk = (s - 1 for s in index.shape)
    return np.stack(
        [index[di : di + ni, dj : dj + nj, dk : dk + nk].ravel() for di, dj, dk in _CELL_CORNERS],
        axis=1,
    )


def _quads(plane) -> np.ndarray:
    """The quadrilaterals of a block face's point numbers, each ring in order."""
    na, nb = (s - 1 for s in plane.shape)
    return np.stack(
        [
            plane[:na, :nb].ravel(),
            plane[1:, :nb].ravel(),
            plane[1:, 1:].ravel(),
            plane[:na, 1:].ravel(),
        ],
        axis=1,
    )


def _boundary_quads(quads, quad_block, quad_face, shared_inside, n_points) -> np.ndarray:
    """Which block-face quadrilaterals are on the boundary; refuses a face met only in part.

    A face is met only in part when some but not all of its quadrilaterals are another block's, or
    none are but a point inside it is another block's too (two faces cut differently).
    """
    _, inverse, counts = np.unique(
        face_keys(quads, n_points), axis=0, return_inverse=True, return_counts=True
    )
    count = counts[inverse.ravel()]
    if np.any(count > 2):
        raise ValueError(
            f"{int(np.count_nonzero(count > 2))} block-face quadrilateral(s) are on more than two "
            "blocks"
        )
    lone = count == 1
    face_id = quad_block * len(FACES) + quad_face
    partial = []
    for each in np.unique(face_id):
        on = face_id == each
        n_lone = np.count_nonzero(lone[on])
        if 0 < n_lone < np.count_nonzero(on) or (n_lone > 0 and shared_inside[each]):
            partial.append((int(each // len(FACES)), FACES[each % len(FACES)]))
    if partial:
        raise ValueError(
            f"block face(s) {partial} (block, face) meet another block on only part of them: the "
            "blocks do not join conformally"
        )
    return lone
