"""Pictures of the two tessellations ``run.py`` uses, and the perforated plate's open fraction.

The lamps' triangles emit and the walls' triangles reflect; no shadow is cast by either (the shadows
are exact bodies, see ``geometry.py``). Writes ``lamp_tessellation.png`` and
``wall_tessellation.png`` into a run's directory (``run.py`` calls these for every run) and gives the plate's open fraction three ways:
as designed, as the shadowing body has it (exact circles), and as the reflecting triangles have it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from geometry import Reactor, wall_triangles  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

mpl.use("Agg")
EDGE = {"edgecolor": "k", "linewidth": 0.25}


def flat(ax, triangles, axes, colour="lightsteelblue", **extra):
    """Triangles projected onto two coordinates."""
    ax.add_collection(
        PolyCollection(triangles[:, :, axes] * 1e3, facecolor=colour, **EDGE, **extra)
    )
    ax.set_aspect("equal")


def areas(triangles):
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )


def lamp_figure(
    reactor: Reactor,
    out: Path,
    axial: float,
    sectors: int,
    wall_lighting_axial: float = 0.04,
    shrink: float = 0.000251,
) -> None:
    """The lamps' triangles: a tip in 3D, a cross-section, both axial spacings, all ten end-on."""
    fine, which = reactor.lamp_triangles(sectors, axial)
    coarse, coarse_which = reactor.lamp_triangles(sectors, wall_lighting_axial)
    one, one_coarse = fine[which == 0], coarse[coarse_which == 0]
    y0, z0 = reactor.lamp_centres[0]
    shift = np.array([0.0, y0, z0])
    one, one_coarse = one - shift, one_coarse - shift
    tip = reactor.tip_x

    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(grid[0, :2], projection="3d")
    near = one[one[:, :, 0].max(axis=1) < tip + 0.06]
    ax.add_collection3d(Poly3DCollection(near * 1e3, facecolor="gold", **EDGE))
    ax.set_xlim(1e3 * (tip - 0.015), 1e3 * (tip + 0.06))
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20, 20)
    ax.set_box_aspect((75, 40, 40))
    ax.view_init(elev=20, azim=-60)
    ax.set_title(
        f"Lamp 0, tip and first 60 mm of sleeve (volume gather: {sectors} sectors, "
        f"{axial * 1e3:g} mm rings)"
    )
    ax.set_xlabel("x (mm)")

    ax = fig.add_subplot(grid[0, 2])
    ring = one[np.abs(one[:, :, 0] - (tip + 0.4)).max(axis=1) < 0.0081]
    flat(ax, ring, [1, 2], colour="gold")
    circle = np.linspace(0, 2 * np.pi, 400)
    r = reactor.lamp_radius * 1e3
    ax.plot(r * np.cos(circle), r * np.sin(circle), "r-", lw=0.8, label="true sleeve (10 mm)")
    body = (reactor.lamp_radius - shrink) * 1e3
    ax.plot(body * np.cos(circle), body * np.sin(circle), "b--", lw=0.8,
            label=f"shadowing body ({body:.2f} mm)")  # fmt: skip
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.legend(loc="center", fontsize=8)
    ax.set_title(f"Cross-section: {sectors}-gon on the true circle")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")

    for col, (tris, label) in enumerate(
        ((one, f"{axial * 1e3:g} mm rings: G in the water"),
         (one_coarse, f"{wall_lighting_axial * 1e3:g} mm rings: light on the walls"))
    ):  # fmt: skip
        ax = fig.add_subplot(grid[1, col])
        front = tris[(tris[:, :, 2].mean(axis=1) > 0) & (tris[:, :, 0].min(axis=1) < tip + 0.2)]
        flat(ax, front, [0, 1], colour="gold")
        ax.set_xlim(1e3 * (tip - 0.012), 1e3 * (tip + 0.2))
        ax.set_ylim(-12, 12)
        ax.set_title(f"Side view, first 200 mm ({label})", fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    ax = fig.add_subplot(grid[1, 2])
    flat(ax, fine[fine[:, :, 0].mean(axis=1) > reactor.length - 0.01], [1, 2], colour="gold")
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(150 * np.cos(t), 150 * np.sin(t), "k-", lw=0.8)
    for k, (y, z) in enumerate(reactor.lamp_centres):
        ax.annotate(str(k), (1e3 * y + 12, 1e3 * z + 12), fontsize=8)
    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_title("All ten lamps, end-on (vessel wall in black)")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    fig.suptitle(
        f"Lamp tessellation: {len(fine) // reactor.n_lamps} triangles per lamp for the volume, "
        f"{len(coarse) // reactor.n_lamps} for lighting the walls; every vertex on the sleeve"
    )
    fig.savefig(out / "lamp_tessellation.png", dpi=130)


def wall_figure(
    reactor: Reactor, triangles, part, out: Path, wall_size: float, per_turn: int
) -> None:
    """The reflecting triangles: the vessel, the plate and its holes, the end wall."""
    centroid = triangles.mean(axis=1)
    on_face = (part == 1) & (np.abs(centroid[:, 0] - reactor.plate_x) < 1e-6)
    holes = (part == 1) & ~on_face & (np.abs(centroid[:, 0] - reactor.plate_far_x) > 1e-6)
    end_wall = (part == 0) & (np.abs(centroid[:, 0] - reactor.length) < 1e-6)
    circle = np.linspace(0, 2 * np.pi, 200)

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(grid[0, :], projection="3d")
    vessel = triangles[(part == 0) & (centroid[:, 2] < 0.0)]
    ax.add_collection3d(Poly3DCollection(vessel * 1e3, facecolor="lightsteelblue", **EDGE))
    plate = triangles[on_face]
    ax.add_collection3d(Poly3DCollection(plate * 1e3, facecolor="salmon", **EDGE))
    low, high = reactor.bounds
    ax.set_xlim(1e3 * low[0], 1e3 * high[0])
    ax.set_ylim(-150, 400)
    ax.set_zlim(-150, 150)
    ax.set_box_aspect((high[0] - low[0], 0.55, 0.3))
    ax.view_init(elev=25, azim=-70)
    ax.set_title(
        f"Walls (far half shown, {np.sum(part == 0)} triangles) and the plate's upstream face "
        f"(of {np.sum(part == 1)} plate triangles)"
    )
    ax.set_xlabel("x (mm)")

    ax = fig.add_subplot(grid[1, 0])
    flat(ax, triangles[on_face], [1, 2], colour="salmon")
    for y, z in reactor.hole_centres:
        ax.plot(1e3 * (y + reactor.hole_radius * np.cos(circle)),
                1e3 * (z + reactor.hole_radius * np.sin(circle)), "b-", lw=0.4)  # fmt: skip
    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_title("Plate, upstream face (exact holes in blue)")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")

    ax = fig.add_subplot(grid[1, 1])
    flat(ax, triangles[on_face], [1, 2], colour="salmon")
    for y, z in reactor.hole_centres:
        ax.plot(1e3 * (y + reactor.hole_radius * np.cos(circle)),
                1e3 * (z + reactor.hole_radius * np.sin(circle)), "b-", lw=0.8)  # fmt: skip
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_title(f"Plate close-up: {per_turn}-sided holes ({np.sum(holes)} hole-wall triangles)")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")

    ax = fig.add_subplot(grid[1, 2])
    flat(ax, triangles[end_wall], [1, 2])
    for y, z in reactor.lamp_centres:
        ax.plot(1e3 * (y + reactor.lamp_radius * np.cos(circle)),
                1e3 * (z + reactor.lamp_radius * np.sin(circle)), "r-", lw=0.6)  # fmt: skip
    ax.set_xlim(-155, 155)
    ax.set_ylim(-155, 155)
    ax.set_title("End wall at x = 1100 mm, lamp sleeves in red")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    fig.suptitle(
        f"Reflecting-wall tessellation (gmsh, edges <= {wall_size * 1e3:g} mm, {per_turn} per turn of a "
        f"curve): {len(triangles)} triangles, diffuse reflectance 0.1"
    )
    fig.savefig(out / "wall_tessellation.png", dpi=130)


def open_fraction(reactor: Reactor, triangles, part, out: Path | None = None) -> dict:
    """The plate's open fraction as designed and as the reflecting triangles have it."""
    disc = np.pi * reactor.radius**2
    designed = reactor.n_holes * np.pi * reactor.hole_radius**2 / disc
    centroid = triangles.mean(axis=1)
    face = (part == 1) & (np.abs(centroid[:, 0] - reactor.plate_x) < 1e-6)
    metal = areas(triangles[face]).sum()
    result = {
        "designed": designed,
        "shadowing_body": designed,
        "reflecting_triangles": 1.0 - metal / disc,
    }
    print(f"plate open fraction: designed and shadowed {designed:.4f}, "
          f"reflecting triangles {result['reflecting_triangles']:.4f}")  # fmt: skip
    if out is not None:
        (out / "plate_open_fraction.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    """The baseline's figures, into ``work/tessellation``."""
    out = HERE / "work" / "tessellation"
    out.mkdir(parents=True, exist_ok=True)
    reactor = Reactor()
    triangles, part = wall_triangles(reactor, 0.03)
    open_fraction(reactor, triangles, part, out)
    lamp_figure(reactor, out, 0.008, 24)
    wall_figure(reactor, triangles, part, out, 0.03, 8)
    print(out)


if __name__ == "__main__":
    main()
