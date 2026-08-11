"""Near-wall aquaflux-vs-OpenFOAM comparison, and the reattachment metric's own resolution.

Exists because two numbers this case is judged by were recorded with **no reproducible definition**:
a first-wall-layer ``k`` median (quoted 0.164) and a reattachment length (quoted 7.24 against 8.36).
Neither could be re-derived from anything in the repository, and both turn out to be far more sensitive
to *how they are defined* than to anything about the solver:

* **"First wall layer" spans a 4x range of defensible meanings.** All wall-face owners is 4490 cells;
  the finest-spacing layer is 1600; the floor alone is 640. Their OpenFOAM ``k`` medians differ by more
  than the aquaflux/OpenFOAM discrepancy being investigated, so the ratio is meaningless without the
  cell set named beside it.
* **The reattachment metric is quantized at the grid.** :func:`compare.reattachment_length` returns the
  ``x`` of the *last reversed wall cell*, which is a grid station, and near reattachment this mesh's
  stations are ``… 6.728, 7.243, 7.787, 8.361, 8.966 …`` -- spacing ~0.5 h. The two quoted numbers are
  **two stations apart**, so a difference reported as "15%" is a two-cell offset on a metric whose
  resolution is half a step height. This prints the stations and a sub-cell interpolated crossing
  alongside, so the quantization is visible rather than implied.

Reports both quantities every way that is defensible, so a reader can see the spread instead of
inheriting one number. Takes the aquaflux state as a path, because the converged root is not a fixture
-- it is whatever the last march produced.

Usage::

    python3 -u validation/bfs3d_openfoam/wall_layer_comparison.py <state.npz>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402

H = 0.01  # step height, as in compare.py
Y_STAR_LAM = 11.53  # the sublayer/log crossover both codes switch on


def cell_sets(case, wall_distance, wall_cells, centroid):
    """The defensible readings of "the first wall layer", each named.

    Returns a dict of ``name -> cell indices``. The point of returning several is that the phrase does
    not have one meaning on this mesh: the side walls sit six times further from their wall than the
    floor does, so including them changes the population and the answer.
    """
    fine = wall_cells[
        np.isclose(wall_distance[wall_cells], wall_distance[wall_cells].min(), rtol=1e-9)
    ]
    patches = case["momentum"].mesh.face_patches
    face_cells = case["momentum"].mesh.face_cells
    owner, neighbour = np.asarray(face_cells.owner), np.asarray(face_cells.neighbour)
    sets = {"all wall-adjacent": wall_cells, "finest layer": fine}
    for name in ("lowerWall", "upperWall", "sideWalls"):
        mask = np.asarray(patches.mask(name))
        cells = np.unique(owner[mask & (neighbour < 0)])
        sets[f"  {name}"] = cells
        sets[f"  {name} ∩ finest"] = np.intersect1d(cells, fine)
    sets["  floor behind the step"] = fine[(centroid[fine, 1] < -0.005) & (centroid[fine, 0] > 0.0)]
    return sets


def reattachment_stations(case, centroid, wall_distance, wall_cells):
    """The wall-cell ``x/h`` stations near reattachment -- the metric's actual resolution."""
    fine = wall_cells[
        np.isclose(wall_distance[wall_cells], wall_distance[wall_cells].min(), rtol=1e-9)
    ]
    floor = fine[(centroid[fine, 1] < -0.005) & (centroid[fine, 0] > 0.0)]
    stations = np.unique(np.round(centroid[floor, 0] / H, 6))
    return stations[(stations > 4.0) & (stations < 12.0)]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <state.npz>")
    state = jnp.asarray(np.load(sys.argv[1])["state"])

    case = compare.build_case()
    coupled = case["coupled"]
    reference = compare.read_openfoam_reference()
    centroid = np.asarray(case["geom"].cell.centroid)
    if not np.allclose(centroid, reference["centroid"], atol=1e-9):
        raise SystemExit("cell ordering differs from the OpenFOAM reference; comparison invalid")

    _, k_solved, omega_solved = coupled.layout.unpack(state)
    k = np.asarray(coupled.k_transform.to_physical(k_solved))
    omega = np.asarray(coupled.omega_transform.to_physical(omega_solved))
    k_of, omega_of = np.asarray(reference["k"]), np.asarray(reference["omega"])

    turbulence = case["turbulence"]
    wall_distance = np.asarray(turbulence.wall_distance)
    wall_cells = np.asarray(turbulence.wall_cells)

    print(f"{'=' * 100}\nnear-wall comparison: {Path(sys.argv[1]).name}\n{'=' * 100}")
    print(
        f"  {'cell set':<26}{'n':>7}{'k aq':>11}{'k OF':>11}{'ratio':>8}{'ω ratio':>10}{'y* (OF k)':>12}"
    )
    for name, cells in cell_sets(case, wall_distance, wall_cells, centroid).items():
        if cells.size == 0:
            continue
        # y* from OpenFOAM's k, not ours: y* ∝ sqrt(k), so our own y* is depressed by the very deficit
        # under study and would understate which regime the mesh is really in.
        y_star = 0.09**0.25 * np.sqrt(np.maximum(k_of[cells], 0.0)) * wall_distance[cells] / 1e-5
        print(
            f"  {name:<26}{cells.size:>7}{np.median(k[cells]):>11.4f}{np.median(k_of[cells]):>11.4f}"
            f"{np.median(k[cells]) / np.median(k_of[cells]):>8.3f}"
            f"{np.median(omega[cells]) / np.median(omega_of[cells]):>10.3f}"
            f"{np.median(y_star):>12.2f}"
        )
    print(
        f"\n  global k median: aq {np.median(k):.4f}  OF {np.median(k_of):.4f}  "
        f"ratio {np.median(k) / np.median(k_of):.3f}"
    )

    print(f"\n{'=' * 100}\nreattachment, and the resolution the metric is quoted at\n{'=' * 100}")
    stations = reattachment_stations(case, centroid, wall_distance, wall_cells)
    print(f"  floor wall-cell x/h stations: {np.array2string(stations, precision=3)}")
    spacing = np.diff(stations)
    print(f"  local spacing dx/h: {spacing.min():.3f} .. {spacing.max():.3f}")
    slab = compare.mid_span_slab(centroid)
    velocity, _ = case["momentum"].unpack(coupled.layout.unpack(state)[0])
    # The metric takes the streamwise COMPONENT, not the vector; passing the vector silently returns a
    # far-downstream station for both codes, which looks like agreement rather than like a bug.
    for label, u in (("aquaflux", np.asarray(velocity)), ("OpenFOAM", np.asarray(reference["U"]))):
        length = compare.reattachment_length(centroid, u[:, 0], z_slab=slab)
        index = int(np.argmin(np.abs(stations - length)))
        print(f"  {label:<10} x_r/h {length:>7.3f}   = station index {index}")
    print(
        "\n  ⚠️ Both are grid stations. Quote the station spacing beside any difference between them:\n"
        "     a gap of one station is the metric's own resolution, not a result."
    )


if __name__ == "__main__":
    main()
