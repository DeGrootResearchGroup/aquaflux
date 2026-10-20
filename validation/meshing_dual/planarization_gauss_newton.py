"""Does a Gauss--Newton step on the true warp beat the planarizer's Jacobi sweeps?

``FacePlanarizer`` steps each free vertex onto its faces' planes with the planes held, so it never
sees that a plane moves with its vertices, and converges linearly. Here the same total warp
``E = |r|^2``, ``r`` each face incidence's height above its face's vertex-mean plane over the root
of the face's area, is a JAX function of the free vertices, the planes included; each iteration
solves the Levenberg--Marquardt system ``(J^T J + lambda I) dx = -J^T r`` by conjugate gradients
with ``J`` applied by ``jax.jvp`` / ``jax.vjp`` and never formed. The step is judged by the
planarizer's own bounds (no cell less convex or newly concave, volumes positive, no face beyond
the start's worst non-orthogonality or skewness). Where it breaks one, the free vertices of the
cells it worsens are pinned and the system solved again without them -- deleting them from a
finished step instead left the rest incoherent (the step's decrease collapsed as more was held).
``lambda`` is raised when a safe step does not lower ``E`` and lowered when it does, and the pass stops on the planarizer's test (a decrease
below ``1e-3`` of ``E``, or ``E`` at ``1e-12`` of its start).

For each case both are run from the same dual and the harness prints, per method, the outer
iterations, the guard's quality evaluations, conjugate-gradient products, wall time and
``planarization.describe``'s quality line.

    python validation/meshing_dual/planarization_gauss_newton.py kuhn
    python validation/meshing_dual/planarization_gauss_newton.py L 0.01

``kuhn`` (no Netgen needed) is the dual of perturbed Kuhn tetrahedra of the unit cube (n 8, perturb
0.25); other names and sizes as in ``dual_scale_convexity.py``, in an environment with
``cadquery-ocp-novtk`` and ``netgen-mesher``.
"""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import Mesh, PolygonFaceGeometry
from aquaflux.meshing import FacePlanarizer, PolyhedralDual
from aquaflux.meshing.planarize import _NoWorse, _State
from aquaflux.vectors import dot
from planarization import describe

MIN_DECREASE, FLAT = 1e-3, 1e-12
CG_RTOL, CG_MAX = 0.1, 200
MAX_ITERATIONS, MAX_REJECTIONS, PIN_ROUNDS = 200, 12, 30
#: How far lambda is raised to keep a step safe before the worsened cells are pinned instead.
SHORTEN_BEFORE_PINNING = float(
    next((a.split("=")[1] for a in sys.argv if a.startswith("--shorten=")), 1.0)
)
TRACE = "--trace" in sys.argv


class _Counted:
    """Counts the guard's quality evaluations."""

    calls = 0
    original = _State.at.__func__

    @classmethod
    def at(cls, state_cls, mesh, coords):
        cls.calls += 1
        return cls.original(state_cls, mesh, coords)


_State.at = classmethod(lambda c, mesh, coords: _Counted.at(c, mesh, coords))


def _residual_function(mesh: Mesh, free: np.ndarray):
    face_nodes = mesh.face_nodes
    of_face = face_nodes.face_of_incidence
    base = jnp.asarray(mesh.node_coords)
    scheme = PolygonFaceGeometry()

    def residual(x):
        coords = base.at[free].set(x.reshape(-1, 3))
        area, _, normal, _ = scheme.centre_fan(coords, face_nodes)
        mean = face_nodes.vertex_mean(coords)
        lift = dot(face_nodes.gather_node_coords(coords) - mean[of_face], normal[of_face])
        return lift / jnp.sqrt(area[of_face])

    return residual


def gauss_newton(mesh: Mesh, movable: np.ndarray) -> dict:
    free = np.flatnonzero(movable)
    residual = _residual_function(mesh, free)

    @jax.jit
    def linearized(x):
        r, forward = jax.linearize(residual, x)
        backward = jax.linear_transpose(forward, x)
        return r, (backward(r)[0]), forward, backward

    @jax.jit
    def normal_product(x, v, lam):
        _, forward = jax.linearize(residual, x)
        backward = jax.linear_transpose(forward, x)
        return backward(forward(v))[0] + lam * v

    coords = np.asarray(mesh.node_coords, dtype=np.float64)
    state = _State.at(mesh, coords)
    guard = _NoWorse(mesh, state)
    area = np.asarray(mesh.geometry().face.area)
    lam = lam_start = 1e-3 * float(np.mean(1.0 / area))
    start, history, products, converged = state.warp, [state.warp], 0, False
    x = jnp.asarray(coords[free].reshape(-1))
    slot = np.full(mesh.n_nodes, -1)
    slot[free] = np.arange(free.size)
    for _ in range(MAX_ITERATIONS):
        _, gradient, _, _ = linearized(x)
        pinned = np.zeros(free.size, dtype=bool)
        trial, rounds = None, 0
        for _ in range(MAX_REJECTIONS * PIN_ROUNDS):
            mask = np.repeat(~pinned, 3).astype(np.float64)

            def apply(v, x=x, lam=lam, mask=mask):
                return mask * np.asarray(normal_product(x, jnp.asarray(mask * v), lam))

            dx, used = _cg(apply, -mask * np.asarray(gradient))
            products += used
            step = np.zeros_like(coords)
            step[free] = dx.reshape(-1, 3)
            candidate = _State.at(mesh, state.coords + step)
            worse = guard.worsened(candidate)
            if np.any(worse) and lam < SHORTEN_BEFORE_PINNING * lam_start:
                lam = 4.0 * lam  # a shorter step may keep every bound
                continue
            if np.any(worse) and rounds < PIN_ROUNDS:
                # Pin the free vertices of every worsened cell and solve again without them.
                held = slot[guard.vertices_of(worse)]
                pinned[held[held >= 0]] = True
                rounds += 1
                continue
            if not np.any(worse) and candidate.warp < state.warp:
                trial = candidate
                break
            lam, rounds = 4.0 * lam, 0  # too long a step: shorten it, keeping the pins
        if TRACE and trial is not None:
            print(
                f"    E {state.warp:.4g} -> {trial.warp:.4g} lambda {lam:.3g} "
                f"pinned {pinned.mean():.3f} cg {used}",
                flush=True,
            )
        if trial is None:
            break
        decrease = state.warp - trial.warp
        state, lam = trial, lam / 3.0
        x = jnp.asarray(state.coords[free].reshape(-1))
        history.append(state.warp)
        if decrease < MIN_DECREASE * history[-2] or state.warp <= FLAT * start:
            converged = True
            break
    moved = Mesh.from_csr(
        state.coords, mesh.face_nodes.offsets, mesh.face_nodes.face_node_indices,
        mesh.face_cells.owner, mesh.face_cells.neighbour, mesh.n_cells,
    )  # fmt: skip
    return {"mesh": moved, "warp": history, "converged": converged, "products": products}


def _cg(apply, rhs: np.ndarray) -> tuple[np.ndarray, int]:
    """Conjugate gradients from zero to ``CG_RTOL`` of ``|rhs|``; the solution and products used."""
    x = np.zeros_like(rhs)
    r = rhs.copy()
    p = r.copy()
    rr = float(r @ r)
    target = (CG_RTOL**2) * rr
    for used in range(1, CG_MAX + 1):
        ap = apply(p)
        alpha = rr / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        new = float(r @ r)
        if new <= target:
            return x, used
        p = r + (new / rr) * p
        rr = new
    return x, CG_MAX


def _duals(names: list[str], sizes: list[float]):
    if "kuhn" in names:
        from tests.support.meshes import tetrahedral_grid_3d

        primal = tetrahedral_grid_3d(8, perturb=0.25)
        yield "kuhn n8", primal, PolyhedralDual(feature_angle=math.radians(45)).build(primal)
    rest = [n for n in names if n != "kuhn"]
    if not rest:
        return
    from aquaflux.meshing import NetgenProcess, element_mesh
    from dual_scale_convexity import DUAL, _solids

    netgen = NetgenProcess()
    with tempfile.TemporaryDirectory() as directory:
        for name, path in _solids(pathlib.Path(directory)).items():
            if name not in rest:
                continue
            for maxh in sizes:
                surface = netgen.triangulate(path, max_size=maxh)
                filled = netgen.fill(surface.points, surface.triangles, max_size=maxh)
                primal = element_mesh(
                    filled.points, filled.tetrahedra, patches={"walls": surface.triangles}
                )
                yield f"{name} maxh={maxh:g}", primal, DUAL.build(primal)


def main(names: list[str], sizes: list[float]) -> None:
    for label, primal, dual in _duals(names, sizes):
        movable = np.arange(dual.n_nodes) < primal.n_cells  # the primal cell centroids
        print(f"{label} cells={dual.n_cells} moved={int(movable.sum())}", flush=True)
        print(f"  start:  {describe(dual)}", flush=True)
        for method in ("jacobi", "gauss-newton", "gauss-newton+jacobi"):
            _Counted.calls = 0
            begin = time.perf_counter()
            if method == "jacobi":
                result = FacePlanarizer(max_sweeps=300).planarize(dual, movable)
                mesh, warp, converged, products = result.mesh, result.warp, result.converged, 0
            else:
                out = gauss_newton(dual, movable)
                mesh, warp, converged, products = (
                    out["mesh"], out["warp"], out["converged"], out["products"]
                )  # fmt: skip
                if method == "gauss-newton+jacobi":
                    # The sweeps' bounds come from their start, itself no worse than the dual.
                    polish = FacePlanarizer(max_sweeps=300).planarize(mesh, movable)
                    mesh, converged = polish.mesh, polish.converged
                    warp = list(warp) + list(polish.warp[1:])
            spent = time.perf_counter() - begin
            print(
                f"  {method:12s} iterations {len(warp) - 1} converged {converged} "
                f"evaluations {_Counted.calls} cg products {products} "
                f"total warp {warp[0]:.3g} -> {warp[-1]:.3g} in {spent:.1f} s",
                flush=True,
            )
            print(f"  {'':12s} {describe(mesh)}", flush=True)


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--trace" and not a.startswith("--")]
    main(
        [a for a in arguments if not a[0].isdigit()],
        [float(a) for a in arguments if a[0].isdigit()],
    )
