"""Pictures of the solved bend: the axial velocity and the secondary flow at the stations, and the
velocity on the plane of symmetry.

Reads the fields ``aquaflux run validation/sudo_bend/case.yaml`` writes and writes
``figures/flow_sections.png`` and ``figures/flow_symmetry_plane.png``. Each cross-section is seen
along the flow, the bend's inside on the left, +z up; the colour is the velocity along the pipe over
the bulk velocity and the arrows the velocity across it (the Dean vortices). Needs the ``ui`` extra
(PyVista) and an off-screen OpenGL.

Usage::

    python validation/sudo_bend/render_flow.py [<fields.vtu>]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from geometry import Bend

HERE = Path(__file__).parent
FIELDS = HERE / "work" / "results" / "fields.vtu"
FIGURES = HERE / "figures"
BULK = 6.0e4 * 1.5e-5 / Bend.diameter


def stations(bend: Bend):
    """Each station's name, centre, unit flow direction and the unit direction toward the bend's
    inside (the in-plane 'left')."""
    rc, d = bend.bend_radius, bend.diameter
    out = [("1 D upstream", np.array([-d, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    for degrees in (30, 60, 90):
        a = math.radians(degrees) - 1e-4  # a hair inside the bend: a station can lie on faces
        out.append(
            (
                f"bend {degrees} deg",
                np.array([rc * math.sin(a), rc * (1.0 - math.cos(a)), 0.0]),
                np.array([math.cos(a), math.sin(a), 0.0]),
            )
        )
    for diameters in (1, 5):
        out.append(
            (
                f"{diameters} D downstream",
                np.array([rc, rc + diameters * d, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        )
    return out


def main(path: Path) -> None:
    bend = Bend()
    r = bend.radius
    grid = pv.read(path)
    print(f"{grid.n_cells} cells read from {path}", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), layout="constrained")
    for axis, (label, centre, along) in zip(axes.flat, stations(bend), strict=True):
        cut = grid.slice(
            normal=along, origin=centre, generate_triangles=True
        ).cell_data_to_point_data()
        # Seen along the flow with +z up, the right-hand side is along x up, and the bend's inside
        # (its centre of curvature) is on the left.
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(along, up)
        points = np.asarray(cut.points) - centre
        x, y = points @ right, points @ up
        velocity = np.asarray(cut.point_data["U"])
        axial = velocity @ along / BULK
        triangles = np.asarray(cut.faces).reshape(-1, 4)[:, 1:]
        shade = axis.tripcolor(x / r, y / r, triangles, axial, shading="gouraud",
                               cmap="viridis", vmin=0.0, vmax=1.4)  # fmt: skip
        pick = np.random.default_rng(0).choice(len(x), size=min(500, len(x)), replace=False)
        across = velocity[pick] - np.outer(velocity[pick] @ along, along)
        axis.quiver(x[pick] / r, y[pick] / r, across @ right, across @ up, color="white",
                    scale=4.0 * BULK, width=0.003)  # fmt: skip
        speed = float(np.linalg.norm(velocity - np.outer(velocity @ along, along), axis=1).max())
        axis.set_title(f"{label}\nlargest cross-flow {speed / BULK:.2f} U_b", fontsize=12)
        axis.set_aspect("equal")
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_xticks([-1, 0, 1], ["inside", "0", "outside"])
    fig.colorbar(shade, ax=axes, shrink=0.6, label="velocity along the pipe / U_b")
    fig.suptitle(
        "Cross-sections seen along the flow (+z up, the bend's inside to the left): colour the "
        "axial velocity, arrows the cross-flow"
    )
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "flow_sections.png", dpi=110)
    plt.close(fig)

    # The plane of symmetry, z = 0, seen from above.
    cut = grid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 1e-4))
    cut.cell_data["speed"] = np.linalg.norm(np.asarray(cut.cell_data["U"]), axis=1) / BULK
    p = pv.Plotter(off_screen=True, window_size=(1600, 1300))
    p.set_background("white")
    p.add_mesh(cut, scalars="speed", cmap="viridis", clim=[0.0, 1.4], lighting=False,
               scalar_bar_args={"title": "|U| / U_b", "color": "black"})  # fmt: skip
    rc = bend.bend_radius
    p.camera_position = [(0.5 * rc, 0.7 * rc, 3.0), (0.5 * rc, 0.7 * rc, 0.0), (0.0, 1.0, 0.0)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = 0.42
    p.screenshot(FIGURES / "flow_symmetry_plane.png")
    p.close()
    print(f"wrote the figures into {FIGURES}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FIELDS)
