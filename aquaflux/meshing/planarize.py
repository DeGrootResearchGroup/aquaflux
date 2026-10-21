"""Flatten a mesh's faces by moving its free vertices, never making its cells or faces worse.

A face of a polyhedral dual is a ring of cell centroids, which in general do not lie in one plane.
A non-planar face bounds no definite volume (its cell's volume depends on how the face is
triangulated, :func:`~aquaflux.mesh.triangulation_volume_change`), and its warp makes the cell
stand off its own faces' planes. Nothing ties an interior dual vertex to where it starts: any point
inside its primal cell gives the same topology. So the vertices that carry no boundary or feature
meaning can be moved to flatten the faces around them.

The quantity minimized is the total warp ``E = |r|^2``: ``r`` holds, for every vertex of every
face, its height ``n_f . (v_k - c_f)`` above the face's vertex-mean plane over the root of the face's
area, so that faces of every size count alike. ``r`` is written in JAX as a function of the free
vertices, the planes included -- a face's plane moves with its vertices -- and each iteration takes
a Levenberg--Marquardt step, solving ``(J^T J + lambda I) dx = -J^T r`` by conjugate gradients
with ``J`` applied by automatic differentiation and never formed. The conjugate gradients are
preconditioned by each vertex's 3x3 block ``sum n n^T / area + lambda I`` over its faces, which
carries the ``1 / area`` scaling that otherwise spreads the system's spectrum.

Flattening alone is not safe: a vertex dragged to flatten a small face can carry a cell centroid
past one of the cell's faces. So a step is kept only where it leaves the mesh no worse than the
pass found it, by these measures:

* no cell less convex than the starting mesh's least convex cell
  (:func:`~aquaflux.mesh.cell_non_convexity`), nor with a volume more dependent on how its faces are
  triangulated than the starting mesh's most dependent cell
  (:func:`~aquaflux.mesh.triangulation_volume_change`), none newly concave by ``checkMesh``'s
  face-plane test and none concave more so (:func:`~aquaflux.mesh.cell_concavity`), and every
  volume positive;
* no face more non-orthogonal or more skewed than the worst face of the starting mesh
  (:func:`~aquaflux.mesh.face_non_orthogonality`, :func:`~aquaflux.mesh.face_skewness`).

The convexity, triangulation, non-orthogonality and skewness bounds are each the starting mesh's
worst, not each cell's or face's own starting value: flattening one face moves vertices that several
cells share, improving some and worsening others a little, and a bound per element holds back nearly
every step that would do so, though the mesh's worst cell or face is not made worse by it. Concavity is the
exception, held cell by cell: a cell turning concave is a failure in itself, however mild.

A step that breaks a bound is first shortened, by raising ``lambda``, up to a set multiple of its
starting value; past that, the free vertices of the cells it worsens are pinned and the step solved
again without them. Re-solving matters: a global step with some of its vertices deleted afterwards
is no longer a step that lowers ``E``. A safe step that does not lower ``E`` is shortened too. The
pass stops when an iteration lowers ``E`` by less than a set fraction, or ``E`` has fallen to a set
fraction of where it started.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux import ragged
from aquaflux.mesh import (
    CONCAVE_COSINE,
    FaceNodeConnectivity,
    Mesh,
    MeshQuality,
    PolygonFaceGeometry,
)
from aquaflux.vectors import dot

__all__ = ["FacePlanarizer", "Planarization"]


@dataclasses.dataclass(frozen=True)
class Planarization:
    """What a planarization pass produced.

    Attributes
    ----------
    mesh : Mesh
        The mesh with its free vertices moved; topology and every other vertex unchanged.
    warp : tuple of float
        The total warp ``E`` before the first iteration and after each.
    converged : bool
        Whether the pass stopped on its convergence test -- an iteration lowered ``E`` by less than
        the planarizer's ``min_decrease``, or ``E`` fell to its ``flat`` fraction of where it
        started -- rather than at ``max_iterations`` or because no step could be kept.
    linear_products : int
        The products with ``J^T J + lambda I`` the conjugate gradients used, over the whole pass.
    """

    mesh: Mesh
    warp: tuple[float, ...]
    converged: bool
    linear_products: int

    @property
    def iterations(self) -> int:
        """The number of iterations run (steps kept)."""
        return len(self.warp) - 1


@dataclasses.dataclass(frozen=True)
class FacePlanarizer:
    """Move a mesh's free vertices to flatten its faces, never making the mesh worse.

    Attributes
    ----------
    min_decrease : float
        The pass stops once an iteration lowers the total warp by less than this fraction of it.
    flat : float
        The pass also stops once the total warp is this fraction of where it started: where the
        faces can be made flat, the warp falls geometrically, and the decrease test never trips.
    damping : float
        The starting ``lambda``, as a fraction of the mean ``1 / area`` over the faces -- the scale
        of ``J^T J``'s diagonal.
    shortening : float
        How far ``lambda`` may rise, as a multiple of its start, to keep a step within every bound
        before the cells the step worsens are pinned instead.
    increase, decrease : float
        The factors ``lambda`` is multiplied by when a step is shortened, and divided by after a
        step is kept.
    linear_tolerance : float
        The conjugate gradients stop when the residual is this fraction of the right-hand side's.
    max_linear_products : int
        A bound on one solve's products; the step reached by then is used.
    max_iterations : int
        A bound on the iterations, reported through :attr:`Planarization.converged`; the pass
        normally stops on ``min_decrease`` before it.
    max_attempts : int
        How many solves one iteration may make (shortening and pinning) before the pass gives up.
    """

    min_decrease: float = 1e-3
    flat: float = 1e-12
    damping: float = 1e-3
    shortening: float = 100.0
    increase: float = 4.0
    decrease: float = 3.0
    linear_tolerance: float = 0.1
    max_linear_products: int = 200
    max_iterations: int = 200
    max_attempts: int = 60

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
            If the mesh is not three-dimensional, ``movable`` has the wrong shape, or a cell's
            volume is not positive or its quality not defined -- the bounds, taken from the
            starting mesh, would then hold nothing.
        """
        if mesh.dim != 3:
            raise ValueError("planarization needs a three-dimensional mesh")
        movable = np.asarray(movable, dtype=bool)
        if movable.shape != (mesh.n_nodes,):
            raise ValueError(f"movable must have shape ({mesh.n_nodes},), got {movable.shape}")
        free = np.flatnonzero(movable)
        quality = MeshQuality(mesh)
        state = _State.at(mesh, quality, np.asarray(mesh.node_coords, dtype=np.float64))
        undefined = (state.cell_quality[:, 2] >= 0.0) | ~np.all(
            np.isfinite(state.cell_quality), axis=1
        )
        if np.any(undefined):
            raise ValueError(
                f"{int(undefined.sum())} cell(s) have a volume that is not positive or a quality "
                f"that is not defined (first: cell {int(np.flatnonzero(undefined)[0])}); "
                "planarization keeps a mesh no worse than it was, which needs a valid mesh to start"
            )
        guard = _NoWorse(mesh, state)
        warp = _Warp.of(mesh, free)
        start = self.damping * float(np.mean(1.0 / state.area))
        step = _Step(self, mesh, quality, guard, warp, free, start)
        history, products, lam = [state.warp], 0, start
        flat = self.flat * state.warp
        converged = state.warp == 0.0
        while not converged and free.size and len(history) <= self.max_iterations:
            trial, lam, used = step.take(state, lam)
            products += used
            if trial is None:
                break
            converged = state.warp - trial.warp < self.min_decrease * state.warp
            state, lam = trial, lam / self.decrease
            history.append(state.warp)
            converged = converged or state.warp <= flat
        moved = eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(state.coords))
        return Planarization(moved, tuple(history), bool(converged), products)


class _Step:
    """One iteration's guarded Levenberg--Marquardt step, shortened and pinned until it is safe."""

    def __init__(self, planarizer, mesh, quality, guard, warp, free, start) -> None:
        self._p, self._mesh, self._quality = planarizer, mesh, quality
        self._guard, self._warp, self._free = guard, warp, free
        self._cap = planarizer.shortening * start
        self._slot = np.full(mesh.n_nodes, -1)
        self._slot[free] = np.arange(free.size)

    def take(self, state: _State, lam: float) -> tuple[_State | None, float, int]:
        """The kept state, the ``lambda`` it was found at, and the products used."""
        p, free = self._p, self._free
        x = jnp.asarray(state.coords[free].reshape(-1))
        gradient = np.asarray(_gradient(self._warp, x))
        pinned = np.zeros(free.size, dtype=bool)
        products = 0
        for _ in range(p.max_attempts):
            mask = np.repeat(~pinned, 3).astype(np.float64)

            def apply(v, lam=lam, mask=mask):
                product = _product(self._warp, x, jnp.asarray(mask * v), lam)
                return mask * np.asarray(product)

            inverse = _block_inverse(state, self._slot, free.size, lam, pinned)
            dx, used = _conjugate_gradients(
                apply, -mask * gradient, inverse, p.linear_tolerance, p.max_linear_products
            )
            products += used
            move = np.zeros_like(state.coords)
            move[free] = dx.reshape(-1, 3)
            candidate = _State.at(self._mesh, self._quality, state.coords + move)
            worse = self._guard.worsened(candidate)
            if np.any(worse) and lam < self._cap:
                lam *= p.increase  # a shorter step may keep every bound
                continue
            if np.any(worse):
                held = self._slot[self._guard.vertices_of(worse)]
                held = held[held >= 0]
                if not np.all(pinned[held]):
                    pinned[held] = True  # solve again without the worsened cells' vertices
                    continue
            elif candidate.warp < state.warp:
                return candidate, lam, products
            lam *= p.increase  # too long a step: shorten it, keeping the pins
        return None, lam, products


class _Warp(eqx.Module):
    """The per-incidence warp residual ``r(x)`` of the free vertices ``x`` (flattened)."""

    face_nodes: FaceNodeConnectivity
    base: jnp.ndarray  # (n_nodes, 3): every vertex's position; the free ones are overwritten
    free: jnp.ndarray  # (n_free,): the free vertices

    @classmethod
    def of(cls, mesh: Mesh, free: np.ndarray) -> _Warp:
        return cls(mesh.face_nodes, jnp.asarray(mesh.node_coords), jnp.asarray(free))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        coords = self.base.at[self.free].set(x.reshape(-1, 3))
        area, _, normal, _ = PolygonFaceGeometry().centre_fan(coords, self.face_nodes)
        of_face = self.face_nodes.face_of_incidence
        mean = self.face_nodes.vertex_mean(coords)
        lift = dot(self.face_nodes.gather_node_coords(coords) - mean[of_face], normal[of_face])
        return lift / jnp.sqrt(area[of_face])


@eqx.filter_jit
def _gradient(warp: _Warp, x: jnp.ndarray) -> jnp.ndarray:
    """``J^T r`` at ``x``."""
    residual, backward = jax.vjp(warp, x)
    return backward(residual)[0]


@eqx.filter_jit
def _product(warp: _Warp, x: jnp.ndarray, v: jnp.ndarray, lam: float) -> jnp.ndarray:
    """``(J^T J + lam I) v`` at ``x``."""
    _, forward = jax.linearize(warp, x)
    backward = jax.linear_transpose(forward, x)
    return backward(forward(v))[0] + lam * v


def _block_inverse(state, slot, n_free, lam, pinned) -> np.ndarray:
    """Per free vertex, the inverse of ``sum n n^T / area + lam I`` over its faces; ``(n_free, 3, 3)``.

    These are ``J^T J``'s diagonal blocks with the planes held still; a pinned vertex gets the
    identity, its components being zero throughout the solve.
    """
    at = slot[state.node]
    kept = at >= 0
    normal = state.normal[state.face[kept]]
    weight = 1.0 / state.area[state.face[kept]]
    blocks = np.zeros((n_free, 3, 3))
    np.add.at(blocks, at[kept], weight[:, None, None] * normal[:, :, None] * normal[:, None, :])
    blocks += lam * np.eye(3)
    blocks[pinned] = np.eye(3)
    return np.linalg.inv(blocks)


def _conjugate_gradients(apply, rhs, inverse, tolerance, max_products) -> tuple[np.ndarray, int]:
    """Preconditioned conjugate gradients from zero; the solution and the products used.

    Stops when the plain residual is ``tolerance`` of ``|rhs|``, or after ``max_products``.
    ``inverse`` is applied blockwise, three components per vertex.
    """
    x = np.zeros_like(rhs)
    target = tolerance**2 * float(rhs @ rhs)
    if target == 0.0:
        return x, 0
    r = rhs.copy()
    z = np.einsum("vij,vj->vi", inverse, r.reshape(-1, 3)).reshape(-1)
    p = z.copy()
    rz = float(r @ z)
    for used in range(1, max_products + 1):
        ap = apply(p)
        alpha = rz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        if float(r @ r) <= target:
            return x, used
        z = np.einsum("vij,vj->vi", inverse, r.reshape(-1, 3)).reshape(-1)
        rz_next = float(r @ z)
        p = z + (rz_next / rz) * p
        rz = rz_next
    return x, max_products


@dataclasses.dataclass(frozen=True)
class _State:
    """Vertex positions with the face planes, warp and quality they give."""

    coords: np.ndarray
    node: np.ndarray  # (n_incidences,): each face incidence's vertex
    face: np.ndarray  # (n_incidences,): its face
    normal: np.ndarray  # (n_faces, 3): unit area vectors
    area: np.ndarray  # (n_faces,)
    height: np.ndarray  # (n_incidences,): the vertex's height above its face's vertex-mean plane
    # (n_cells, 4): non-convexity, concavity, -volume, triangulation-volume change (larger is worse)
    cell_quality: np.ndarray
    face_quality: np.ndarray  # (n_faces, 2): non-orthogonality, skewness

    @property
    def warp(self) -> float:
        """The total warp ``sum height^2 / area``."""
        return float(np.sum(self.height**2 / self.area[self.face]))

    @classmethod
    def at(cls, mesh: Mesh, quality: MeshQuality, coords: np.ndarray) -> _State:
        """The state of ``mesh``'s topology with its vertices at ``coords``; ``quality`` checks it."""
        geometry = quality.geometry(coords)
        face_nodes = mesh.face_nodes
        node = np.asarray(face_nodes.face_node_indices)
        face = np.asarray(face_nodes.face_of_incidence)
        normal = np.asarray(geometry.face.normal)
        height = np.asarray(quality.vertex_heights(coords, geometry))
        cells = np.stack(
            [
                quality.cell_non_convexity(coords, geometry),
                quality.cell_concavity(geometry),
                -np.asarray(geometry.cell.volume),
                np.asarray(quality.triangulation_volume_change(coords, geometry)),
            ],
            axis=1,
        )
        faces = np.stack(
            [
                np.asarray(quality.face_non_orthogonality(geometry)),
                np.asarray(quality.face_skewness(coords, geometry)),
            ],
            axis=1,
        )
        return cls(coords, node, face, normal, np.asarray(geometry.face.area), height, cells, faces)


class _NoWorse:
    """The bounds a step must keep, from the starting mesh: which cells a trial worsens."""

    #: Slack on every bound, for rounding only.
    _SLACK = 1e-12

    def __init__(self, mesh: Mesh, start: _State) -> None:
        self._offsets, self._nodes = mesh.cell_nodes()
        self._owner = np.asarray(mesh.face_cells.owner)
        self._neighbour = np.asarray(mesh.face_cells.neighbour)
        cell = start.cell_quality.copy()
        cell[:, 0] = cell[:, 0].max()  # no cell less convex than the start's least convex
        cell[:, 3] = cell[:, 3].max()  # nor more volume-ambiguous than its most
        cell[:, 1] = np.maximum(cell[:, 1], CONCAVE_COSINE)  # a convex cell may change, not fold
        cell[:, 2] = 0.0  # a positive volume
        self._cell_bound = cell + self._SLACK
        self._face_bound = start.face_quality.max(axis=0) + self._SLACK

    def worsened(self, trial: _State) -> np.ndarray:
        """The cells ``trial`` breaks a bound in, or that bound a face breaking one; ``(n_cells,)``."""
        worse = np.any(trial.cell_quality > self._cell_bound, axis=1)
        bad = np.any(trial.face_quality > self._face_bound, axis=1)
        neighbour = self._neighbour[bad]
        worse[self._owner[bad]] = True
        worse[neighbour[neighbour >= 0]] = True
        return worse

    def vertices_of(self, cells: np.ndarray) -> np.ndarray:
        """The vertices of the cells ``cells`` marks."""
        held, _, _ = ragged.rows(self._offsets, self._nodes, np.flatnonzero(cells))
        return held
