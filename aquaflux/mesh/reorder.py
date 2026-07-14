"""Cell renumbering (reordering) strategies and the canonical relabelling transform.

Reordering the cells of an unstructured mesh changes *which integer index* each cell carries,
not the geometry or the physics. It matters for two reasons on large meshes:

- **Sparse-matvec locality.** The residual assembly gathers owner/neighbour cell values per
  face; a spatially-local numbering (adjacent cells close in index) improves cache reuse and,
  on accelerators, memory coalescing of those gathers.
- **Aggregation-multigrid convergence.** The smoothed-aggregation coarse space is built by a
  *greedy* aggregation that visits cells in index order. A spatially-local numbering yields
  compact aggregates and a near-optimal V-cycle contraction factor; an arbitrary (scrambled)
  numbering yields irregular aggregates and a measurably worse factor. The V-cycle *smoother*
  is permutation-invariant, so this is purely a coarse-space-construction effect — but it is
  real, so reordering a poorly-numbered mesh before building the preconditioner restores the
  rate. (Bandwidth-reducing reordering is the classical fix; here it protects the coarse space
  rather than a direct/ILU factorization.)

:func:`permute_cells` applies a cell relabelling to a mesh; a small :class:`CellReordering`
strategy hierarchy *chooses* the permutation (identity, reverse Cuthill--McKee, random). Reordering is a
build-time preprocessing step run once per mesh, before geometry/fields are attached; it is not
part of the differentiable solve.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .connectivity import FaceCellConnectivity, interior_mask
from .graph import cell_adjacency_coo
from .mesh import Mesh


def permute_cells(mesh: Mesh, perm) -> Mesh:
    """Relabel every cell by ``perm`` and return the equivalent renumbered mesh.

    The permutation renames cell ``old`` to ``perm[old]``; node coordinates and face
    connectivity are untouched (faces are not reordered), so the geometry and physics are
    identical up to the relabelling. Concretely the owner/neighbour *values*, the cell-zone
    labels, and (downstream) any per-cell field or the linear system are all mapped by the same
    permutation, giving the symmetric renumbering ``P·A·Pᵀ`` of the assembled operator.

    A per-cell field ``f`` in the old numbering becomes ``f[argsort(perm)]`` in the new one
    (``new_field[perm[old]] = f[old]``); callers that carry per-cell inputs (e.g. a
    spatially-varying coefficient, or a pinned-cell index) remap them with the returned
    permutation the same way.

    Parameters
    ----------
    mesh : Mesh
        The mesh to renumber.
    perm : array-like of int, shape ``(n_cells,)``
        A bijection on ``[0, n_cells)`` with ``perm[old_cell] = new_cell``.

    Returns
    -------
    Mesh
        The renumbered mesh (validated).

    Raises
    ------
    ValueError
        If ``perm`` is not a bijection on ``[0, n_cells)``. A non-bijective ``perm`` would
        silently corrupt the mesh (``argsort`` is no longer a true inverse, cells collapse in the
        owner/neighbour remap), so it is rejected up front rather than producing a broken mesh
        that may still pass :meth:`~aquaflux.mesh.Mesh.validate`.
    """
    perm_np = np.asarray(perm)
    if perm_np.shape != (mesh.n_cells,) or not np.array_equal(
        np.sort(perm_np), np.arange(mesh.n_cells)
    ):
        raise ValueError(
            f"perm must be a permutation of [0, {mesh.n_cells}); got shape {perm_np.shape}"
        )
    perm = jnp.asarray(perm)
    inverse = jnp.argsort(perm)  # inverse[new] = old
    neighbour = mesh.face_cells.neighbour
    owner_new = perm[mesh.face_cells.owner]
    # remap interior neighbours through the permutation; keep the boundary sentinel (-1)
    neighbour_new = jnp.where(interior_mask(neighbour), perm[jnp.clip(neighbour, 0)], -1)
    face_cells = FaceCellConnectivity(owner_new, neighbour_new, mesh.n_cells)
    zones = eqx.tree_at(lambda z: z.label, mesh.cell_zones, mesh.cell_zones.label[inverse])
    renumbered = eqx.tree_at(
        lambda m: (m.face_cells, m.cell_zones),
        mesh,
        (face_cells, zones),
    )
    return renumbered.validate()


class CellReordering(eqx.Module):
    """Strategy interface for choosing a cell renumbering of a mesh.

    A concrete strategy computes a permutation ``perm[old] = new`` from the mesh connectivity;
    :meth:`apply` turns it into the renumbered mesh via the shared :func:`permute_cells`.
    """

    @abc.abstractmethod
    def permutation(self, mesh: Mesh) -> np.ndarray:
        """Return the permutation ``perm`` with ``perm[old_cell] = new_cell``.

        Parameters
        ----------
        mesh : Mesh
            The mesh to renumber.

        Returns
        -------
        numpy.ndarray of int, shape ``(n_cells,)``
            A bijection on ``[0, n_cells)``.
        """

    def apply(self, mesh: Mesh) -> tuple[Mesh, np.ndarray]:
        """Renumber ``mesh`` with this strategy.

        Returns
        -------
        tuple of (Mesh, numpy.ndarray)
            The renumbered mesh and the permutation ``perm`` (``perm[old] = new``) used, so the
            caller can remap per-cell fields consistently.
        """
        perm = self.permutation(mesh)
        return permute_cells(mesh, perm), perm


class IdentityReordering(CellReordering):
    """No-op renumbering — keeps the mesh's existing cell numbering.

    The default: use it to opt out of reordering, or as a baseline against which to measure the
    effect of a real reordering.
    """

    def permutation(self, mesh: Mesh) -> np.ndarray:
        return np.arange(mesh.n_cells)


class ReverseCuthillMcKee(CellReordering):
    """Reverse Cuthill--McKee renumbering — minimizes the cell-graph bandwidth.

    Numbers cells by a breadth-first sweep from a low-degree seed and reverses the result, so
    adjacent cells receive nearby indices. This gives the residual gathers good locality and
    hands the aggregation multigrid a compact-aggregate-friendly numbering (restoring the
    near-optimal V-cycle contraction factor a scrambled mesh would degrade).
    """

    def permutation(self, mesh: Mesh) -> np.ndarray:
        graph = cell_adjacency_coo(mesh).tocsr()
        old_order = reverse_cuthill_mckee(graph, symmetric_mode=True)  # old_order[new] = old
        return np.argsort(old_order)  # perm[old] = new


class RandomReordering(CellReordering):
    """A uniformly random renumbering — the worst-case (locality-destroying) ordering.

    Not for production use: it exists to stress-test ordering-robustness and to measure the
    penalty of a poorly-numbered mesh (the counterpart of :class:`ReverseCuthillMcKee`).
    Deterministic given ``seed``.
    """

    seed: int = eqx.field(static=True)

    def __init__(self, seed: int = 0):
        self.seed = seed

    def permutation(self, mesh: Mesh) -> np.ndarray:
        return np.random.default_rng(self.seed).permutation(mesh.n_cells)
