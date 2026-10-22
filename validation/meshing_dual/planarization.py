"""How much moving the dual's interior vertices flattens its faces, on duals of Netgen tetrahedra.

The solids of ``dual_scale_convexity.py`` (cylinder, sleeve box, L, Sozzi water) are meshed by
Netgen in its child process and dualized with the same feature angle. The vertices moved are the
primal cell centroids -- every interior vertex except those the concave cut placed on the surface
between pieces; the boundary, its features and the cut stay. For each case and ``maxh`` the
harness prints, before and after ``FacePlanarizer``: the total warp it minimizes; face warp (the
largest out-of-plane vertex height over the square root of the face's area) on interior and on
boundary faces; cell non-convexity (largest vertex excursion beyond a face plane over the cell
size); the cells ``checkMesh`` would count concave by face planes; the largest change in a cell's
volume that re-triangulating its faces could make; non-orthogonality (maximum, and the average as
``checkMesh`` takes it, the angle of the mean cosine) and skewness; and the worst closure.

After the pass it prints where the worst are. Each face is a boundary face, an interior face with a
vertex on the boundary, one with a vertex the concave cut placed between pieces, or one whose every
vertex was free to move; each cell is a cut piece (it has a vertex between pieces), a wall cell (it
has a boundary face), or an interior cell. For the worst 1 % of interior faces by warp, and of cells
by non-convexity and by triangulation-volume change, it gives the share in each class, and each
class's own maximum over the whole mesh.

Run in an environment with ``cadquery-ocp-novtk`` and ``netgen-mesher``::

    python validation/meshing_dual/planarization.py L cylinder 0.01 0.005

Names and sizes as in ``dual_scale_convexity.py``; ``--iterations N`` caps the iterations (default 200).
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
    Mesh,
    cell_concavity,
    cell_non_convexity,
    closed_cell_residual,
    face_non_orthogonality,
    face_skewness,
    face_warp,
    triangulation_volume_change,
)
from aquaflux.meshing import FacePlanarizer, NetgenProcess, element_mesh
from dual_scale_convexity import DUAL, _solids


def describe(mesh: Mesh) -> str:
    interior = np.asarray(mesh.face_cells.interior)
    warp = np.asarray(face_warp(mesh))
    convexity = cell_non_convexity(mesh)
    volume = np.asarray(triangulation_volume_change(mesh))
    angle = np.asarray(face_non_orthogonality(mesh))[interior]
    average = np.degrees(np.arccos(np.cos(np.radians(angle)).mean()))
    return (
        f"warp int p99 {np.quantile(warp[interior], 0.99):.3f} max {warp[interior].max():.3f} "
        f"bnd p99 {np.quantile(warp[~interior], 0.99):.3f} max {warp[~interior].max():.3f} | "
        f"non-convexity p99 {np.quantile(convexity, 0.99):.3f} max {convexity.max():.3f} | "
        f"concave {int((cell_concavity(mesh) > CONCAVE_COSINE).sum())} | "
        f"tri-volume p99 {np.quantile(volume, 0.99):.2e} max {volume.max():.2e} | "
        f"non-orth max {angle.max():.1f} avg {average:.1f} skew {np.asarray(face_skewness(mesh)).max():.2f} | "
        f"closure {float(np.max(np.asarray(closed_cell_residual(mesh)))):.1e}"
    )


def where_the_worst_are(mesh: Mesh, movable: np.ndarray) -> str:
    """The classes of the worst faces and cells, as the module docstring describes."""
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    boundary_face = neighbour < 0
    of_face = np.repeat(np.arange(mesh.n_faces), np.diff(offsets))
    on_boundary = np.zeros(mesh.n_nodes, dtype=bool)
    on_boundary[nodes[boundary_face[of_face]]] = True
    between = ~movable & ~on_boundary  # fixed, not on the boundary: placed by the cut
    touches_boundary = np.bincount(of_face, weights=on_boundary[nodes], minlength=mesh.n_faces) > 0
    touches_cut = np.bincount(of_face, weights=between[nodes], minlength=mesh.n_faces) > 0
    face_class = np.select(
        [boundary_face, touches_boundary, touches_cut],
        ["boundary", "by boundary", "by cut"],
        "free",
    )
    cell_offsets, cell_nodes = mesh.cell_nodes()
    of_cell = np.repeat(np.arange(mesh.n_cells), np.diff(cell_offsets))
    piece = np.bincount(of_cell, weights=between[cell_nodes], minlength=mesh.n_cells) > 0
    wall = np.zeros(mesh.n_cells, dtype=bool)
    wall[owner[boundary_face]] = True
    cell_class = np.select([piece, wall], ["cut piece", "wall"], "interior")

    def breakdown(values, classes, names):
        worst = values >= np.quantile(values, 0.99)
        parts = []
        for name in names:
            mine = classes == name
            peak = f"{values[mine].max():.3f}" if mine.any() else "-"
            parts.append(f"{name} {worst[mine].sum() / worst.sum():.0%} (max {peak})")
        return ", ".join(parts)

    warp = np.asarray(face_warp(mesh))
    interior = ~boundary_face
    faces = ("by boundary", "by cut", "free")
    cells = ("cut piece", "wall", "interior")
    boundary_max = warp[boundary_face].max() if boundary_face.any() else 0.0
    return (
        f"worst 1% interior warp: {breakdown(warp[interior], face_class[interior], faces)}; "
        f"boundary faces' max {boundary_max:.3f}\n"
        f"    worst 1% non-convexity: {breakdown(cell_non_convexity(mesh), cell_class, cells)}\n"
        f"    worst 1% tri-volume: "
        f"{breakdown(np.asarray(triangulation_volume_change(mesh)), cell_class, cells)}"
    )


def main(names: list[str], sizes: list[float], iterations: int) -> None:
    netgen = NetgenProcess()
    planarizer = FacePlanarizer(max_iterations=iterations)
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
                print(f"{name} maxh={maxh:g} cells={dual.n_cells} moved={int(movable.sum())}")
                print(f"  before: {describe(dual)}", flush=True)
                start = time.perf_counter()
                result = planarizer.planarize(dual, movable)
                spent = time.perf_counter() - start
                print(
                    f"  iterations {result.iterations} converged {result.converged} "
                    f"linear products {result.linear_products} "
                    f"total warp {result.warp[0]:.3g} -> {result.warp[-1]:.3g} in {spent:.1f} s"
                )
                print(f"  after:  {describe(result.mesh)}", flush=True)
                print(f"  where:  {where_the_worst_are(result.mesh, movable)}", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    cap = 200
    if "--iterations" in arguments:
        at = arguments.index("--iterations")
        cap = int(arguments[at + 1])
        del arguments[at : at + 2]
    main(
        [a for a in arguments if not a[0].isdigit()],
        [float(a) for a in arguments if a[0].isdigit()],
        cap,
    )
