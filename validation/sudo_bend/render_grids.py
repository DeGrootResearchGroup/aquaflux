"""The grid study's grids in cross-section, side by side.

For each grid (``grids/round36``, ``round48``, ``round64``, meshed by ``aquaflux mesh mesh.yaml``)
the cross-section halfway round the bend, whole and close to its outer wall, drawn into
``figures/grids_sections.png``. Needs the ``ui`` extra (PyVista) and an off-screen OpenGL.

Usage::

    python validation/sudo_bend/render_grids.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from aquaflux.io import read_aqmesh, write_vtu
from geometry import Bend
from render_mesh import cell_kinds, kinds_mesh, plane_camera, section

HERE = Path(__file__).parent
GRIDS = ("round36", "round48", "round64")


def main() -> None:
    bend = Bend()
    r, rc = bend.radius, bend.bend_radius
    # Halfway round the bend, a hair off the cross-section of faces there.
    a = math.radians(45.0) + 1e-4
    centre = np.array([rc * math.sin(a), rc * (1.0 - math.cos(a)), 0.0])
    along = np.array([math.cos(a), math.sin(a), 0.0])
    outer = centre - r * np.array([-math.sin(a), math.cos(a), 0.0])  # away from the bend's centre
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), layout="constrained")
    with tempfile.TemporaryDirectory() as work:
        for column, grid in enumerate(GRIDS):
            mesh = read_aqmesh(HERE / "grids" / grid / "work" / "bend.aqmesh")
            path = write_vtu(mesh, {"hexahedron": cell_kinds(mesh)}, Path(work) / f"{grid}.vtu")
            cut = section(pv.read(path), centre, along)
            for row, (focus, zoom) in enumerate(
                ((centre, 1.06 * r), (0.8 * outer + 0.2 * centre, 0.3 * r))
            ):
                p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
                p.set_background("white")
                kinds_mesh(p, cut)
                plane_camera(p, focus, -along, (0.0, 0.0, 1.0), 1.0, zoom)
                axes[row, column].imshow(p.screenshot(None, return_img=True))
                p.close()
                axes[row, column].axis("off")
            axes[0, column].set_title(f"{grid}: {mesh.n_cells} cells", fontsize=14)
    axes[1, 0].set_ylabel("near the outer wall")
    fig.suptitle(
        "The bend at 45 degrees, seen along the flow (+z up, the bend's inside to the left, the outer wall to the right in the close-ups); "
        "blue hexahedra, orange the core corners' prisms",
        fontsize=13,
    )
    fig.savefig(HERE / "figures" / "grids_sections.png", dpi=100)
    print(f"wrote {HERE / 'figures' / 'grids_sections.png'}", flush=True)


if __name__ == "__main__":
    main()
