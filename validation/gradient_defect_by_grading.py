"""Where the multiple-correction gradient's second pass is active, read from geometry alone.

The scheme returns ``first - gradient_defect . hessian``: the first-order-exact gradient, less its
first-order error per unit Hessian times a reconstructed Hessian. A cell whose ``gradient_defect`` is
zero takes nothing from the second pass, and so -- in a residual built on this gradient -- couples no
further than the first pass does; a cell whose defect is not zero carries the second pass's extra ring.
Whether a column of a coupled Jacobian reaches distance three on a given mesh therefore starts here.

The defect is a property of the mesh, so this needs no case and no solve. It is reported per unit cell
size (``|defect| / V^(1/3)``, dimensionless), interior and boundary cells apart (a boundary cell's
closure carries a defect of its own on any mesh), on three controls and on both step-case meshes:

  uniform cubes             a centred Green-Gauss gradient is exact for quadratics: zero defect
  uniform, stretched 10:1   aspect ratio alone
  graded 1.25               each cell 1.25x the previous along one axis: spacing that varies
  bfs3d, pitzDaily          the meshes themselves, with interior cells binned by the largest volume
                            ratio to a face neighbour

Usage::

    python3 validation/gradient_defect_by_grading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh.structured import structured_grid_3d  # noqa: E402
from aquaflux.schemes.multiple_correction import OwnerGradient, _build_corrections  # noqa: E402

MESHES = {
    "bfs3d": ROOT / "validation/bfs3d_openfoam/of_case/constant/polyMesh",
    "pitzDaily": ROOT / "validation/pitzdaily_openfoam/runs/kwsst/polyMesh",
}
#: Interior cells are binned by the largest volume ratio to a face neighbour.
VOLUME_RATIO_BINS = ((1.0, 1.0001), (1.0001, 1.05), (1.05, 1.2), (1.2, np.inf))
#: Cells per side of the controls, and the controls' grading.
CONTROL_CELLS = 8
GROWTH = 1.25


def defect_per_cell_size(mesh):
    """``|gradient_defect| / V^(1/3)`` per cell, and which cells own a boundary face."""
    geometry = mesh.geometry()
    corrections = _build_corrections(mesh, geometry, OwnerGradient(), None)
    defect = np.asarray(corrections.gradient_defect)
    size = np.asarray(geometry.cell.volume) ** (1.0 / mesh.dim)
    face_cells = mesh.face_cells
    boundary = np.zeros(mesh.n_cells, bool)
    boundary[np.asarray(face_cells.owner)[~np.asarray(face_cells.interior)]] = True
    return np.linalg.norm(defect.reshape(mesh.n_cells, -1), axis=1) / size, boundary


def largest_neighbour_volume_ratio(mesh):
    volume = np.asarray(mesh.geometry().cell.volume)
    owner, neighbour, _ = mesh.face_cells.interior_edges()
    owner, neighbour = np.asarray(owner), np.asarray(neighbour)
    ratio = np.maximum(volume[owner] / volume[neighbour], volume[neighbour] / volume[owner])
    out = np.ones(mesh.n_cells)
    np.maximum.at(out, owner, ratio)
    np.maximum.at(out, neighbour, ratio)
    return out


def report(label, mesh, by_grading=False):
    defect, boundary = defect_per_cell_size(mesh)
    interior = defect[~boundary]
    print(
        f"{label:28s} {mesh.n_cells:6d} cells  interior median {np.median(interior):.2e} max "
        f"{interior.max():.2e}  boundary median {np.median(defect[boundary]):.2e}",
        flush=True,
    )
    if by_grading:
        ratio = largest_neighbour_volume_ratio(mesh)
        for low, high in VOLUME_RATIO_BINS:
            cells = ~boundary & (ratio >= low) & (ratio < high)
            if cells.any():
                print(
                    f"    interior, volume ratio {low:g}-{high:g}: {int(cells.sum()):6d} cells, median "
                    f"{np.median(defect[cells]):.2e}",
                    flush=True,
                )


def graded(mesh, growth, cells):
    """``mesh`` with its y spacing growing geometrically by ``growth`` per cell over ``[0, 1]``."""
    coords = np.asarray(mesh.node_coords)
    y = (growth ** (coords[:, 1] * cells) - 1.0) / (growth**cells - 1.0)
    return eqx.tree_at(
        lambda m: m.node_coords, mesh, jnp.asarray(np.stack([coords[:, 0], y, coords[:, 2]], 1))
    )


def main():
    n = CONTROL_CELLS
    report("uniform cubes", structured_grid_3d(n, n, n))
    report("uniform, stretched 10:1", structured_grid_3d(n, n, n, 1.0, 1.0, 0.1))
    report(f"graded {GROWTH:g} along y", graded(structured_grid_3d(n, n, n), GROWTH, n))
    for name, path in MESHES.items():
        if not path.exists():
            print(f"{name}: {path} missing", flush=True)
            continue
        report(name, read_openfoam(path), by_grading=True)


if __name__ == "__main__":
    main()
