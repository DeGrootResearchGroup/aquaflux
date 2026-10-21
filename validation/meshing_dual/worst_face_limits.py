"""What keeps a dual's worst faces warped after planarization, when every vertex of them is free?

``planarization.py`` found that most of the worst 1 % of interior faces by warp, after
``FacePlanarizer``, have no fixed vertex. Three things could hold such a face: one of the
planarizer's bounds (a step flattening it would make some cell less convex or concave, or some face
worse than the starting mesh's worst); the stopping test (the pass ends before it gets there); or
its neighbours (the unconstrained least-squares optimum itself leaves it warped, because flattening
it warps the faces it shares vertices with by more).

The probe runs the shipped pass, takes as **tracked faces** the interior faces in the worst 1 % by
warp whose every vertex is free, and reruns the pass from the same dual with: each bound loosened
in turn; every quality bound loosened (positive volume kept); and both the shipped bounds and none
with a much tighter stop. Per arm it prints the tracked faces' warp (median, 90th percentile, max,
and the share halved against the shipped pass), and what the arm costs elsewhere: total warp, worst
interior warp, non-orthogonality and skewness maxima, worst non-convexity, concave cells, time.

Reading: a bound whose loosening flattens the tracked faces is what holds them; if none does even
with the tight stop, the neighbours do.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/worst_face_limits.py L 0.01
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from aquaflux.mesh import (
    CONCAVE_COSINE,
    cell_concavity,
    cell_non_convexity,
    face_non_orthogonality,
    face_skewness,
    face_warp,
)
from aquaflux.meshing import FacePlanarizer, NetgenProcess, element_mesh
from aquaflux.meshing import planarize as planarize_module
from dual_scale_convexity import DUAL, _solids

#: Columns of the guard's bounds: cells (non-convexity, concavity, -volume), faces (non-orth, skew).
CELL, FACE = "cell", "face"
LOOSENED = {
    "shipped": [],
    "no convexity bound": [(CELL, 0)],
    "no concavity bound": [(CELL, 1)],
    "no non-orthogonality bound": [(FACE, 0)],
    "no skewness bound": [(FACE, 1)],
    "no quality bound": [(CELL, 0), (CELL, 1), (FACE, 0), (FACE, 1)],
}
SHIPPED = FacePlanarizer()
TIGHT = FacePlanarizer(min_decrease=1e-6, max_iterations=1000)
ARMS = [(name, SHIPPED) for name in LOOSENED] + [
    ("shipped, tight stop", TIGHT, "shipped"),
    ("no quality bound, tight stop", TIGHT, "no quality bound"),
]


def _with_loosened(columns):
    """Run the planarizer with the given bound columns set to infinity."""
    original = planarize_module._NoWorse.__init__

    def init(self, mesh, start):
        original(self, mesh, start)
        for kind, column in columns:
            bound = self._cell_bound if kind == CELL else self._face_bound
            bound[..., column] = np.inf

    return original, init


def _run(dual, movable, planarizer, columns):
    original, init = _with_loosened(columns)
    planarize_module._NoWorse.__init__ = init
    try:
        begin = time.perf_counter()
        result = planarizer.planarize(dual, movable)
        return result, time.perf_counter() - begin
    finally:
        planarize_module._NoWorse.__init__ = original


def _tracked(mesh, movable) -> np.ndarray:
    """Interior faces in the worst 1 % by warp whose every vertex is free."""
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    of_face = np.repeat(np.arange(mesh.n_faces), np.diff(offsets))
    fixed = np.bincount(of_face, weights=~movable[nodes], minlength=mesh.n_faces) > 0
    interior = np.asarray(mesh.face_cells.interior)
    warp = np.asarray(face_warp(mesh))
    worst = warp >= np.quantile(warp[interior], 0.99)
    return np.flatnonzero(worst & interior & ~fixed)


def _line(name, result, seconds, tracked, reference) -> str:
    mesh = result.mesh
    warp = np.asarray(face_warp(mesh))
    interior = np.asarray(mesh.face_cells.interior)
    mine = warp[tracked]
    halved = float(np.mean(mine <= 0.5 * reference))
    return (
        f"  {name:30s} tracked median {np.median(mine):.3f} p90 {np.quantile(mine, 0.9):.3f} "
        f"max {mine.max():.3f} halved {halved:4.0%} | total warp {result.warp[-1]:8.3g} "
        f"int warp max {warp[interior].max():.3f} | non-orth {np.asarray(face_non_orthogonality(mesh))[interior].max():5.1f} "
        f"skew {np.asarray(face_skewness(mesh)).max():.2f} | non-convexity {cell_non_convexity(mesh).max():.3f} "
        f"concave {int((cell_concavity(mesh) > CONCAVE_COSINE).sum()):4d} | "
        f"{result.iterations:4d} it {seconds:6.1f} s converged {result.converged}"
    )


def main(names: list[str], sizes: list[float]) -> None:
    netgen = NetgenProcess()
    with tempfile.TemporaryDirectory() as directory:
        for name, path in _solids(pathlib.Path(directory)).items():
            if names and name not in names:
                continue
            for maxh in sizes:
                surface = netgen.triangulate(path, max_size=maxh)
                filled = netgen.fill(surface.points, surface.triangles, max_size=maxh)
                primal = element_mesh(
                    filled.points, filled.tetrahedra, patches={"walls": surface.triangles}
                )
                traced = DUAL.build_traced(primal)
                dual, movable = traced.mesh, traced.vertex_cell >= 0
                print(f"{name} maxh={maxh:g} cells={dual.n_cells}", flush=True)
                tracked = reference = None
                for arm in ARMS:
                    label, planarizer = arm[0], arm[1]
                    columns = LOOSENED[arm[2] if len(arm) > 2 else label]
                    result, seconds = _run(dual, movable, planarizer, columns)
                    if tracked is None:
                        tracked = _tracked(result.mesh, movable)
                        reference = np.asarray(face_warp(result.mesh))[tracked]
                        start = np.asarray(face_warp(dual))[tracked]
                        print(
                            f"  {tracked.size} tracked faces; their warp at the start: median "
                            f"{np.median(start):.3f} max {start.max():.3f}",
                            flush=True,
                        )
                    print(_line(label, result, seconds, tracked, reference), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(
        [a for a in arguments if not a[0].isdigit()],
        [float(a) for a in arguments if a[0].isdigit()],
    )
