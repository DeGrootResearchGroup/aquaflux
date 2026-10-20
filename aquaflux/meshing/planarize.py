"""Flatten a mesh's faces by moving its free vertices, never making its cells or faces worse.

A face of a polyhedral dual is a ring of cell centroids, which in general do not lie in one plane.
A non-planar face bounds no definite volume (its cell's volume depends on how the face is
triangulated, :func:`~aquaflux.mesh.triangulation_volume_change`), and its warp makes the cell
stand off its own faces' planes. Nothing ties an interior dual vertex to where it starts: any point
inside its primal cell gives the same topology. So the vertices that carry no boundary or feature
meaning can be moved to flatten the faces around them.

The pass is a damped Gauss--Newton iteration on the total warp ``E = sum_f sum_k (n_f . (v_k -
c_f))^2 / area_f`` -- each vertex's height above its face's vertex-mean plane, squared, over the
face's area so that faces of every size count alike. Per sweep, every free vertex takes the step
that minimizes the warp of its own faces with their planes held, all at once (Jacobi).

Flattening alone is not safe: a vertex dragged to flatten a small face can carry a cell centroid
past one of the cell's faces. So a step is kept only where it leaves the mesh no worse than the
pass found it, by these measures:

* no cell less convex than it started (:func:`~aquaflux.mesh.cell_non_convexity`), none newly
  concave by ``checkMesh``'s face-plane test and none concave more so
  (:func:`~aquaflux.mesh.cell_concavity`), and every volume positive;
* no face more non-orthogonal or more skewed than the worst face of the starting mesh
  (:func:`~aquaflux.mesh.face_non_orthogonality`, :func:`~aquaflux.mesh.face_skewness`). A
  face-by-face bound would hold nearly every vertex still: almost any move tilts some face a little.

Where a step breaks one, the free vertices of the cells it touches are held and the rest retried,
until nothing is worse. A sweep that would raise ``E`` is halved until it does not. The pass stops
when a sweep lowers ``E`` by less than a set fraction, or ``E`` has fallen to a set fraction of
where it started.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux import ragged
from aquaflux.mesh import (
    CONCAVE_COSINE,
    Mesh,
    cell_concavity,
    cell_non_convexity,
    face_non_orthogonality,
    face_skewness,
)

__all__ = ["FacePlanarizer", "Planarization"]


@dataclasses.dataclass(frozen=True)
class Planarization:
    """What a planarization pass produced.

    Attributes
    ----------
    mesh : Mesh
        The mesh with its free vertices moved; topology and every other vertex unchanged.
    warp : tuple of float
        The total warp ``E`` before the first sweep and after each.
    converged : bool
        Whether the pass stopped on its convergence test -- a sweep lowered ``E`` by less than the
        planarizer's ``min_decrease``, or ``E`` fell to its ``flat`` fraction of where it started --
        rather than at ``max_sweeps`` or because no step could be kept.
    """

    mesh: Mesh
    warp: tuple[float, ...]
    converged: bool

    @property
    def sweeps(self) -> int:
        """The number of sweeps run."""
        return len(self.warp) - 1


@dataclasses.dataclass(frozen=True)
class FacePlanarizer:
    """Move a mesh's free vertices to flatten its faces, never making the mesh worse.

    Attributes
    ----------
    min_decrease : float
        The pass stops once a sweep lowers the total warp by less than this fraction of it.
    flat : float
        The pass also stops once the total warp is this fraction of where it started: where the
        faces can be made flat, the warp falls geometrically, and the decrease test never trips.
    damping : float
        Each vertex's step solves ``(M + damping * trace(M) / 3 * I) dx = -r``, with ``M`` and ``r``
        its faces' normal-weighted sums; damping keeps the step bounded where a vertex's faces are
        nearly parallel and so fix it in fewer than three directions.
    max_sweeps : int
        A bound on the sweeps run, reported through :attr:`Planarization.converged`; the pass
        normally stops on ``min_decrease`` before it.
    max_halvings : int
        How many times a sweep that would raise the warp is halved before the pass gives up.
    """

    min_decrease: float = 1e-3
    flat: float = 1e-12
    damping: float = 0.1
    max_sweeps: int = 500
    max_halvings: int = 8

    def planarize(self, mesh: Mesh, movable: np.ndarray) -> Planarization:
        """Flatten ``mesh``'s faces by moving the vertices ``movable`` marks.

        Parameters
        ----------
        mesh : Mesh
            A three-dimensional mesh.
        movable : np.ndarray of bool, shape ``(n_nodes,)``
            The vertices free to move. The caller decides: a vertex on the boundary, on a feature,
            or placed on the surface between two pieces of a cut cell carries a meaning a move
            would destroy, so must not be marked.

        Returns
        -------
        Planarization

        Raises
        ------
        ValueError
            If the mesh is not three-dimensional or ``movable`` has the wrong shape.
        """
        if mesh.dim != 3:
            raise ValueError("planarization needs a three-dimensional mesh")
        movable = np.asarray(movable, dtype=bool)
        if movable.shape != (mesh.n_nodes,):
            raise ValueError(f"movable must have shape ({mesh.n_nodes},), got {movable.shape}")
        state = _State.at(mesh, np.asarray(mesh.node_coords, dtype=np.float64))
        guard = _NoWorse(mesh, state)
        history = [state.warp]
        flat = self.flat * state.warp
        converged = state.warp == 0.0
        while not converged and len(history) <= self.max_sweeps:
            trial = self._kept(mesh, state, self._step(mesh, state, movable), guard)
            if trial is None:
                break
            converged = state.warp - trial.warp < self.min_decrease * state.warp
            state = trial
            history.append(state.warp)
            converged = converged or state.warp <= flat
        moved = eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(state.coords))
        return Planarization(mesh=moved, warp=tuple(history), converged=bool(converged))

    def _step(self, mesh: Mesh, state: _State, movable: np.ndarray) -> np.ndarray:
        """Each free vertex's damped least-squares step onto its faces' planes, planes held."""
        weight = 1.0 / state.area[state.face]
        normal = state.normal[state.face]
        outer = weight[:, None, None] * normal[:, :, None] * normal[:, None, :]
        matrix = np.zeros((mesh.n_nodes, 3, 3))
        np.add.at(matrix, state.node, outer)
        residual = np.zeros((mesh.n_nodes, 3))
        np.add.at(residual, state.node, (weight * state.height)[:, None] * normal)
        trace = np.trace(matrix, axis1=1, axis2=2)
        free = movable & (trace > 0.0)
        damped = matrix[free] + (self.damping * trace[free] / 3.0)[:, None, None] * np.eye(3)
        step = np.zeros((mesh.n_nodes, 3))
        step[free] = -np.linalg.solve(damped, residual[free][:, :, None])[:, :, 0]
        return step

    def _kept(self, mesh: Mesh, state: _State, step: np.ndarray, guard: _NoWorse) -> _State | None:
        """The trial after ``step``, halved while it would raise the warp; ``None`` if none helps."""
        for _ in range(self.max_halvings + 1):
            trial = guard.hold_what_worsens(mesh, state, step)
            if trial is not None and trial.warp <= state.warp:
                return trial
            step = 0.5 * step
        return None


@dataclasses.dataclass(frozen=True)
class _State:
    """Vertex positions with the face planes, warp and quality they give."""

    coords: np.ndarray
    node: np.ndarray  # (n_incidences,): each face incidence's vertex
    face: np.ndarray  # (n_incidences,): its face
    normal: np.ndarray  # (n_faces, 3): unit area vectors
    area: np.ndarray  # (n_faces,)
    height: np.ndarray  # (n_incidences,): the vertex's height above its face's vertex-mean plane
    cell_quality: np.ndarray  # (n_cells, 3): non-convexity, concavity, -volume (larger is worse)
    face_quality: np.ndarray  # (n_faces, 2): non-orthogonality, skewness

    @property
    def warp(self) -> float:
        """The total warp ``sum height^2 / area``."""
        return float(np.sum(self.height**2 / self.area[self.face]))

    @classmethod
    def at(cls, mesh: Mesh, coords: np.ndarray) -> _State:
        moved = eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(coords))
        geometry = moved.geometry()
        face_nodes = mesh.face_nodes
        node = np.asarray(face_nodes.face_node_indices)
        face = np.asarray(face_nodes.face_of_incidence)
        normal = np.asarray(geometry.face.normal)
        mean = np.asarray(face_nodes.vertex_mean(moved.node_coords))
        height = np.einsum("ij,ij->i", coords[node] - mean[face], normal[face])
        cells = np.stack(
            [cell_non_convexity(moved), cell_concavity(moved), -np.asarray(geometry.cell.volume)],
            axis=1,
        )
        faces = np.stack(
            [np.asarray(face_non_orthogonality(moved)), np.asarray(face_skewness(moved))], axis=1
        )
        return cls(coords, node, face, normal, np.asarray(geometry.face.area), height, cells, faces)


class _NoWorse:
    """The bounds a step must keep, from the starting mesh, and the holding of what breaks them."""

    #: Slack on every bound, for rounding only.
    _SLACK = 1e-12

    def __init__(self, mesh: Mesh, start: _State) -> None:
        self._offsets, self._nodes = mesh.cell_nodes()
        self._owner = np.asarray(mesh.face_cells.owner)
        self._neighbour = np.asarray(mesh.face_cells.neighbour)
        cell = start.cell_quality.copy()
        cell[:, 1] = np.maximum(cell[:, 1], CONCAVE_COSINE)  # a convex cell may change, not fold
        cell[:, 2] = 0.0  # a positive volume
        self._cell_bound = cell + self._SLACK
        self._face_bound = start.face_quality.max(axis=0) + self._SLACK

    def hold_what_worsens(self, mesh: Mesh, state: _State, step: np.ndarray) -> _State | None:
        """The state after ``step``, every worsened cell's vertices held; ``None`` if all are."""
        step = step.copy()
        while np.any(step):
            trial = _State.at(mesh, state.coords + step)
            worse = np.any(trial.cell_quality > self._cell_bound, axis=1)
            bad = np.any(trial.face_quality > self._face_bound, axis=1)
            neighbour = self._neighbour[bad]
            worse[self._owner[bad]] = True
            worse[neighbour[neighbour >= 0]] = True
            if not np.any(worse):
                return trial
            held, _, _ = ragged.rows(self._offsets, self._nodes, np.flatnonzero(worse))
            if not np.any(step[held]):
                return None  # holding changes nothing more: the step cannot be made safe
            step[held] = 0.0
        return None
