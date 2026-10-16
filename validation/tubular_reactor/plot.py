"""Slices of the fluence rate written by ``run.py``: along the axis, across it, and behind the plate.

Every plane is drawn up to the reactor's true boundary (``render.py``): fine pixels kept by the exact
water test, each interpolated from the grid points around it.

Run as ``python plot.py <run directory>``; writes ``slices.png`` and ``slices_upstream.png`` there.
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
import render  # noqa: E402
from geometry import Reactor  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

mpl.use("Agg")

#: The vessel's cross-section, in metres, as drawn.
ACROSS = (-0.155, 0.155, -0.155, 0.155)
#: Distances upstream of the plate's near face at which the grid is cut.
UPSTREAM_DISTANCES = (0.002, 0.006, 0.012, 0.025, 0.050, 0.090)


def load(run: Path):
    data = np.load(run / "fluence_rate.npz")
    shape = tuple(data["shape"])
    field = np.full(int(np.prod(shape)), np.nan)
    field[data["index"]] = data["G"]
    return data["origin"], float(data["h"]), field.reshape(shape)


def shared_norm(arrays, decades: float = 5.0) -> LogNorm:
    """One log scale for several panels: from the largest value in any of them down ``decades``."""
    peak = max(float(np.nanmax(a)) for a in arrays)
    return LogNorm(vmin=peak * 10.0**-decades, vmax=peak)


def draw_log(ax, u, v, values, norm):
    """``values`` (len(u), len(v)) on ``norm``, in millimetres: exactly zero in grey (the plate hides
    every lamp), positive but below the scale in black, outside the water left blank."""
    extent = (1e3 * (u[0] - (u[1] - u[0]) / 2), 1e3 * (u[-1] + (u[1] - u[0]) / 2),
              1e3 * (v[0] - (v[1] - v[0]) / 2), 1e3 * (v[-1] + (v[1] - v[0]) / 2))  # fmt: skip
    zero = np.where(values == 0.0, 1.0, np.nan)
    ax.imshow(zero.T, origin="lower", extent=extent, cmap=ListedColormap(["0.6"]), vmin=0, vmax=2,
              interpolation="nearest")  # fmt: skip
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    cmap.set_bad(alpha=0.0)
    shown = np.where(values > 0.0, values, np.nan)
    image = ax.imshow(shown.T, origin="lower", extent=extent, norm=norm, cmap=cmap,
                      interpolation="nearest")  # fmt: skip
    ax.set_aspect("equal")
    return image


def main(run: Path) -> None:
    reactor = Reactor()
    origin, h, g = load(run)
    summary = json.loads((run / "summary.json").read_text())
    low, high = reactor.bounds
    planes = [render.plane(reactor, g, origin, h, 2, 0.0, (low[0], high[0], low[1], high[1]),
                           pixel=1e-3)]  # fmt: skip
    cuts = (0.097, 0.109, 0.70)
    planes += [render.plane(reactor, g, origin, h, 0, x, ACROSS) for x in cuts]
    norm = shared_norm([p[2] for p in planes], decades=6)

    fig = plt.figure(figsize=(13, 9), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    top = fig.add_subplot(grid[0, :])
    image = draw_log(top, *planes[0], norm)
    top.set_title("G on the plane z = 0 (through the axis and the outlet)")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")
    for col, (x, panel) in enumerate(zip(cuts, planes[1:], strict=True)):
        ax = fig.add_subplot(grid[1, col])
        draw_log(ax, *panel, norm)
        where = (
            "behind the plate"
            if x < 0.1
            else "lamp side of the plate"
            if x < 0.2
            else "among the lamps"
        )
        ax.set_title(f"x = {x * 1e3:.0f} mm ({where})")
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")
    fig.colorbar(image, ax=fig.axes, label="fluence rate G (W/m²)", shrink=0.8, extend="min")
    results = summary["results"]
    fig.suptitle(
        f"Tubular reactor, 10 x {summary['reactor']['lamp_power_W']} W lamps, "
        f"UVT {summary['settings']['uvt_percent']:g} %, wall reflectance "
        f"{summary['settings']['reflectance']:g}; volume-mean G "
        f"{results['mean_G_W_per_m2']:.1f} W/m² (interpolated from the {h * 1e3:g} mm grid)"
    )
    fig.savefig(run / "slices.png", dpi=130)
    print(run / "slices.png")
    upstream(run, reactor, origin, h, g)


def upstream(run: Path, reactor: Reactor, origin, h, g) -> None:
    """Cross-sections behind the plate at each of ``UPSTREAM_DISTANCES``, on one log scale."""
    planes = [render.plane(reactor, g, origin, h, 0, reactor.plate_x - d, ACROSS)
              for d in UPSTREAM_DISTANCES]  # fmt: skip
    norm = shared_norm([p[2] for p in planes])
    fig, axes = plt.subplots(1, len(planes), figsize=(4.2 * len(planes), 4.4), layout="constrained")
    for ax, d, panel in zip(axes, UPSTREAM_DISTANCES, planes, strict=True):
        image = draw_log(ax, *panel, norm)
        ax.set_title(f"x = {(reactor.plate_x - d) * 1e3:g} mm ({d * 1e3:g} mm behind)", fontsize=9)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")
    fig.colorbar(image, ax=axes, shrink=0.8, extend="min", label="G (W/m²)")
    fig.suptitle(
        f"G behind the plate, interpolated from the {h * 1e3:g} mm grid, one scale "
        "(grey: G = 0; black: below the scale)"
    )
    fig.savefig(run / "slices_upstream.png", dpi=110)
    print(run / "slices_upstream.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
