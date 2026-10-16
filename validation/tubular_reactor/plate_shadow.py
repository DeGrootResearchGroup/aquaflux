"""The lamps' light on fine planes either side of the perforated plate, with and without the plate.

The lamps are all downstream of the plate, so the water upstream of it is lit only through the holes.
This gathers the direct fluence rate on a plane ``TUBULAR_GAP`` upstream of the plate's near face
(default 3 mm), once with the plate in the way and once without it, and on a plane the same distance
downstream of its far face. If the plate's shadow is right, the ratio upstream should be close to 1
inside the holes' projections and 0 elsewhere, averaging near the open fraction.

``run.py`` calls :func:`run` for every run, writing ``plate_shadow.png`` and ``plate_shadow.npz``
into its directory. Run alone (``validation/run_case.sh validation/tubular_reactor/plate_shadow.py``)
it does the same for the baseline lamp, into ``work/plate_shadow_<spacing>_<side>``, and
``TUBULAR_PLANE_SPACING`` / ``TUBULAR_PLANE_HALF`` set the sampling and the square's half side.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import jax.numpy as jnp  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import render  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    UniformAbsorption,
    absorption_from_uvt,
    direct_fluence_rate,
)
from geometry import Reactor  # noqa: E402
from plot import draw_log, shared_norm  # noqa: E402

mpl.use("Agg")

SPACING = float(os.environ.get("TUBULAR_PLANE_SPACING", "0.0015"))
GAP = float(os.environ.get("TUBULAR_GAP", "0.003"))
#: Said on every figure drawn from these planes.
DIRECT_ONLY = (
    "Lamps' direct light only: light reflected by the walls is NOT included on these planes"
)
#: Distances upstream of the plate's near face at which the beams through the holes are drawn.
BEAM_DISTANCES = (0.001, 0.003, 0.006, 0.012, 0.025, 0.050)
#: Half the side of the square sampled, centred on the axis; unset, the whole vessel.
HALF = float(os.environ.get("TUBULAR_PLANE_HALF", "0.150"))
#: Grid points closer than this to a wall or a lamp are not gathered at (``run.NODE_MARGIN``).
NODE_MARGIN = 3e-4


def say(line: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def plane(reactor: Reactor, x: float):
    """Points on a square grid across the vessel at ``x`` that are in the water."""
    side = np.arange(-HALF, HALF + SPACING / 2, SPACING)
    y, z = np.meshgrid(side, side, indexing="ij")
    points = np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])
    return side, reactor.in_water(points, NODE_MARGIN), points


def run(reactor: Reactor, fine, occluders, absorption, out: Path) -> None:
    """The planes for these lamps and bodies (``occluders`` = water, plate, lamps), into ``out``:
    ``GAP`` downstream, ``GAP`` upstream without the plate, and every ``BEAM_DISTANCES`` upstream
    with it. Reads them back instead when ``TUBULAR_REPLOT`` is set and they exist."""
    saved = out / "plate_shadow.npz"
    planes = [
        ("downstream", reactor.plate_far_x + GAP, occluders),
        ("upstream_no_plate", reactor.plate_x - GAP, (occluders[0], *occluders[2:])),
    ]
    distances = sorted({GAP, *BEAM_DISTANCES})
    planes += [(_upstream(d), reactor.plate_x - d, occluders) for d in distances]
    if os.environ.get("TUBULAR_REPLOT") and saved.exists():
        data = np.load(saved)
        side = data["side"]
        results = {label: data[label] for label, _, _ in planes}
    else:
        results = {}
        for label, x, bodies in planes:
            side, wet, points = plane(reactor, x)
            tic = time.perf_counter()
            field = np.full(len(points), np.nan)
            field[wet] = np.asarray(
                direct_fluence_rate(
                    fine, jnp.asarray(points[wet]), absorption=absorption, occluders=bodies,
                    self_occlusion=NoOcclusion(),
                )
            )  # fmt: skip
            say(f"plate plane {label}, x = {x * 1e3:.1f} mm: {wet.sum()} points, "
                f"{time.perf_counter() - tic:.0f} s")  # fmt: skip
            results[label] = field.reshape(len(side), len(side))
        positions = {f"x_{label}": x for label, x, _ in planes}
        np.savez_compressed(saved, side=side, **positions, **results)
    at = {label: x for label, x, _ in planes}
    rendered = {label: _rendered(reactor, side, results[label], at[label]) for label in results}
    rendered["upstream"] = rendered[_upstream(GAP)]
    draw(reactor, rendered, out)
    draw_beams(
        reactor, {d: (results[_upstream(d)], rendered[_upstream(d)]) for d in distances}, out
    )


def _rendered(reactor: Reactor, side: np.ndarray, values: np.ndarray, x: float):
    """A plane's samples drawn up to the true boundary (``render.plane``): pixel centres and image."""
    spacing = float(side[1] - side[0])
    return render.plane(reactor, values, (side[0], side[0]), spacing, 0, x,
                        (side[0], side[-1], side[0], side[-1]),
                        pixel=min(render.PIXEL, spacing / 2))  # fmt: skip


def _upstream(distance: float) -> str:
    return f"upstream_{distance * 1e3:g}mm"


def draw_beams(reactor: Reactor, by_distance: dict, out: Path) -> None:
    """The light through the holes at each distance upstream of the plate: whole plate above, a
    close-up below, all on one log scale (grey where the plate hides every lamp). ``by_distance``
    maps each distance to its samples and their rendering."""
    count = len(by_distance)
    fig, axes = plt.subplots(2, count, figsize=(4.0 * count, 8.4), layout="constrained")
    holes = reactor.hole_centres * 1e3
    norm = shared_norm(image for _, (_, _, image) in by_distance.values())
    for col, (distance, (values, (u, v, picture))) in enumerate(sorted(by_distance.items())):
        for row, half in enumerate((min(150, 1e3 * HALF), min(30, 1e3 * HALF))):
            ax = axes[row, col]
            image = draw_log(ax, u, v, picture, norm)
            for cy, cz in holes:
                ax.add_patch(plt.Circle((cy, cz), reactor.hole_radius * 1e3, fill=False,
                                        lw=0.4 if row == 0 else 0.8, color="cyan"))  # fmt: skip
            ax.set_aspect("equal")
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_xlabel("y (mm)")
            ax.set_ylabel("z (mm)")
            if row == 0:
                lit = np.isfinite(values)
                share = np.mean(values[lit] > 0)
                ax.set_title(f"{distance * 1e3:g} mm upstream\nmean G {np.nanmean(values):.2e} W/m², "
                             f"{100 * share:.0f} % lit", fontsize=9)  # fmt: skip
    fig.colorbar(image, ax=axes, shrink=0.6, extend="min", label="G (W/m²)")
    fig.suptitle("Light through the plate's holes at increasing distance upstream, one scale "
                 "(grey: G = 0; black: below the scale; hole outlines in cyan)\n"
                 f"{DIRECT_ONLY}")  # fmt: skip
    fig.savefig(out / "plate_beams.png", dpi=110)
    say(f"wrote {out / 'plate_beams.png'}")


def main() -> None:
    from run import AXIAL, UVT, WATER_TOLERANCE, lamps, shadow_shrink

    out = HERE / "work" / f"plate_shadow_{SPACING * 1e3:g}mm_{2 * HALF * 1e3:g}mm"
    out.mkdir(parents=True, exist_ok=True)
    reactor = Reactor()
    fine = lamps(reactor, AXIAL)
    occluders = (
        reactor.water(WATER_TOLERANCE),
        reactor.plate(),
        *reactor.lamp_bodies(shadow_shrink(reactor, fine)),
    )
    run(reactor, fine, occluders, UniformAbsorption(absorption_from_uvt(UVT)), out)


def draw(reactor: Reactor, rendered: dict, out: Path) -> None:
    """Four planes across the vessel and a close-up of the shadow behind the plate, each drawn up to
    the true boundary; ``rendered`` maps each plane's label to its pixel centres and image.

    On the log scales a fluence rate of exactly zero (the plate's full shadow) is drawn in its own
    colour, grey, so it is not confused with the white outside the water.
    """
    up, free, down = rendered["upstream"], rendered["upstream_no_plate"], rendered["downstream"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (up[0], up[1], up[2] / free[2])
    say(
        f"upstream / no plate (rendered): mean {np.nanmean(ratio[2]):.3f}; mean G upstream "
        f"{np.nanmean(up[2]):.4g}, without the plate {np.nanmean(free[2]):.4g}, downstream "
        f"{np.nanmean(down[2]):.4g} W/m2"
    )
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.4), layout="constrained")
    holes = reactor.hole_centres * 1e3
    gap = f"{GAP * 1e3:g} mm"
    norm = shared_norm((down[2], free[2], up[2]))
    panels = (
        (down, f"{gap} downstream of the plate (lamp side)", 150),
        (free, f"{gap} upstream, plate removed", 150),
        (up, f"{gap} upstream, behind the plate", 150),
        (up, "close-up of the same plane", min(40, 1e3 * HALF)),
        (ratio, "upstream: with plate / without", 150),
    )
    for k, (ax, ((u, v, picture), title, half)) in enumerate(zip(axes, panels, strict=True)):
        if k == 4:
            extent = 1e3 * np.array([u[0], u[-1], v[0], v[-1]])
            ratio_image = ax.imshow(picture.T, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
                                    cmap="viridis", interpolation="nearest")  # fmt: skip
            fig.colorbar(ratio_image, ax=ax, shrink=0.8)
        else:
            image = draw_log(ax, u, v, picture, norm)
        for cy, cz in holes:
            ax.add_patch(
                plt.Circle((cy, cz), reactor.hole_radius * 1e3, fill=False, lw=0.5, color="cyan")
            )
        ax.set_aspect("equal")
        ax.set_xlim(-min(half, 1e3 * HALF), min(half, 1e3 * HALF))
        ax.set_ylim(-min(half, 1e3 * HALF), min(half, 1e3 * HALF))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")
    fig.colorbar(image, ax=axes[:4], shrink=0.8, extend="min", label="G (W/m², one scale)")
    fig.suptitle(
        "Fluence rate either side of the perforated plate (plate at x = 100-106 mm; "
        f"hole outlines in cyan)\n{DIRECT_ONLY}"
    )
    fig.savefig(out / "plate_shadow.png", dpi=120)
    say(f"wrote {out / 'plate_shadow.png'}")


if __name__ == "__main__":
    main()
