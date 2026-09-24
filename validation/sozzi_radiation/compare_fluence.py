"""Compare aquaflux's fluence rate with the discrete-ordinates reference on the Sozzi reactor.

Reads the mesh ``generate_dom_reference.py`` built and the DOM fields it kept, computes aquaflux's
``G`` at every cell centre, and writes what is needed to *see* the differences:

- ``work/compare/sozzi_fluence.vtu`` -- the mesh with ``G_aquaflux``, ``G_dom64``, ``G_dom256``,
  the ratio of each DOM field to aquaflux, and the region each cell is in, for ParaView;
- ``work/compare/*.png`` -- a slice through the lamp axis and profiles along probe lines;
- the volume-weighted mean of each field by region, printed and kept in ``summary.json``.

**The scene.** 35 W lamp as a diffuse emitter, exitance 696.42 W/m^2 on the ``lampWall`` patch
(its cylinder and hemispherical tip), water absorbing at 35.67 /m, every wall black. With black
walls nothing reflects, so the fluence rate is the direct gather from the lamp alone -- no
interreflection solve, and no transfer matrix over the 61,000 wall facets.

**Visibility, exactly and cheaply.** The fluid is three cylinders: a chamber holding the lamp
on its axis, an inlet pipe continuing that axis, and an outlet riser standing on the chamber's
top. The lamp is convex, so it shadows itself through the source-side cosine and needs no ray
test. Every lamp point is inside the chamber, and each cylinder is convex, so a cell in the
chamber sees every lamp facet facing it, and a cell in a pipe sees a lamp point exactly when the
segment between them leaves the chamber through that pipe's opening --
:class:`BranchOpenings`. That is O(receivers x facets). Testing the same segments against the
53,500 wall triangles would be ~6e14 intersections and would give the same bits.

**Discretization.** The lamp's facets are ~4 mm; the gather is exact in solid angle and
evaluates the absorption at each facet's centroid. The harness measures what that costs on the
cells nearest the lamp, where it is largest, by re-gathering a sample against a lamp refined
until each facet is small against its distance -- and reports it, rather than assuming it away.

Run with ``validation/run_case.sh validation/sozzi_radiation/compare_fluence.py`` after
``generate_dom_reference.py`` has produced ``work/case`` and ``work/runs``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam, read_volume_scalar_field, write_vtu  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    Surfaces,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
    read_stl,
    refine_for_receivers,
)
from aquaflux.radiation.occluders import Occluder  # noqa: E402

WORK = HERE / "work"
CASE = WORK / "case"
OUT = WORK / "compare"

# The reactor, from the tutorial's geometry script (millimetres there, metres here).
R_BODY, X_BODY_END = 0.0445, 0.889  # chamber: radius, and the plane where the inlet pipe starts
R_PIPE = 0.00955  # inlet pipe and outlet riser share a radius
X_RISER = 0.04765  # riser axis, a vertical line at y = 0
R_LAMP = 0.010

EXITANCE = 696.42  # W/m^2 on the lamp patch, as the DOM case sets it
ABSORPTION = 35.67  # 1/m, napierian
CHUNK = 20_000
DOM_RUNS = {"G_dom64": "nphi8_ntheta4", "G_dom256": "nphi16_ntheta8"}


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class BranchOpenings(Occluder):
    """The chamber's wall, seen from a cell in one of the two pipes.

    A segment from a lamp point (inside the chamber) to a cell in the inlet pipe is clear
    exactly when it crosses the chamber's end plane inside the pipe's circle; to a cell in the
    riser, exactly when it leaves the chamber's cylinder through the riser's footprint. Either
    way the rest of the segment lies inside one convex cylinder or the other, so nothing else
    can block it. To a cell in the chamber it is always clear.
    """

    def contains(self, position) -> jnp.ndarray:
        """Outside the fluid: in none of the three cylinders."""
        p = jnp.asarray(position, dtype=float)
        return ~(_in_chamber(p) | _in_inlet(p) | _in_riser(p))

    def crossing_ratio(self, origin, target) -> jnp.ndarray:
        """Where each segment crosses its opening, as a fraction of the opening's radius.

        One at the rim, less inside it, more outside. :meth:`blocks` is exactly ``ratio > 1`` for
        a target in a pipe, and this is separate from it so that a study can ask *how far from
        the rim* a sight line passes -- which is what distinguishes a mask that disagrees with
        another because the two describe the opening differently from one that is simply wrong.
        A segment aimed at a cell in the chamber crosses no opening and gets zero.
        """
        o, t = jnp.asarray(origin, dtype=float), jnp.asarray(target, dtype=float)
        d = t - o
        # Inlet: where the segment crosses the end plane.
        at_plane = (
            o
            + ((X_BODY_END - o[..., 0]) / jnp.where(d[..., 0] == 0.0, 1.0, d[..., 0]))[..., None]
            * d
        )
        inlet = jnp.sqrt(at_plane[..., 1] ** 2 + at_plane[..., 2] ** 2) / R_PIPE
        # Riser: where the segment leaves the chamber's cylinder (the origin is inside it).
        a = d[..., 1] ** 2 + d[..., 2] ** 2
        b = 2.0 * (o[..., 1] * d[..., 1] + o[..., 2] * d[..., 2])
        c = o[..., 1] ** 2 + o[..., 2] ** 2 - R_BODY**2
        safe_a = jnp.where(a == 0.0, 1.0, a)
        exit_t = (-b + jnp.sqrt(jnp.maximum(b * b - 4.0 * safe_a * c, 0.0))) / (2.0 * safe_a)
        exit = o + exit_t[..., None] * d
        riser = jnp.where(
            exit[..., 2] > 0.0,
            jnp.sqrt((exit[..., 0] - X_RISER) ** 2 + exit[..., 1] ** 2) / R_PIPE,
            jnp.inf,
        )
        return jnp.where(_in_chamber(t), 0.0, jnp.where(t[..., 0] > X_BODY_END, inlet, riser))

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        del min_distance  # the lamp is not an occluder here, so nothing can shadow itself
        return self.crossing_ratio(origin, target) > 1.0


def _in_chamber(p):
    return (
        (p[..., 0] >= 0.0)
        & (p[..., 0] <= X_BODY_END)
        & (p[..., 1] ** 2 + p[..., 2] ** 2 <= R_BODY**2)
    )


def _in_inlet(p):
    return (p[..., 0] > X_BODY_END) & (p[..., 1] ** 2 + p[..., 2] ** 2 <= R_PIPE**2)


def _in_riser(p):
    return (p[..., 2] > 0.0) & ((p[..., 0] - X_RISER) ** 2 + p[..., 1] ** 2 <= R_PIPE**2)


def lamp_surfaces() -> Surfaces:
    """The emitting patch, wound so its normals face the water."""
    vertices = np.asarray(read_stl(CASE / "constant" / "triSurface" / "lampWall.stl").vertices)
    lamp = Surfaces.from_triangles(vertices, emission=EXITANCE)
    centroid, normal = np.asarray(lamp.centroid), np.asarray(lamp.normal)
    # Outward is away from the lamp's axis, or along +x on the tip.
    axis_point = np.column_stack([np.minimum(centroid[:, 0], 0.80), np.zeros((len(centroid), 2))])
    outward = np.einsum("ij,ij->i", normal, centroid - axis_point) > 0.0
    if outward.mean() < 0.5:
        lamp = Surfaces.from_triangles(vertices[:, ::-1, :], emission=EXITANCE)
        outward = ~outward
    if not outward.all():
        msg = f"{int((~outward).sum())} lamp facets face inward after orientation; fix the STL"
        raise ValueError(msg)
    return lamp


def gather(lamp: Surfaces, points: np.ndarray, region: np.ndarray, label: str) -> np.ndarray:
    """``G`` at ``points``: no mask in the chamber, the opening test in the pipes."""
    water = UniformAbsorption(ABSORPTION)
    field = np.empty(len(points))
    started = time.perf_counter()
    for start in range(0, len(points), CHUNK):
        stop = min(start + CHUNK, len(points))
        chunk, where = points[start:stop], region[start:stop]
        piped = where != 0
        out = np.empty(stop - start)
        if (~piped).any():
            out[~piped] = np.asarray(
                direct_fluence_rate(lamp, jnp.asarray(chunk[~piped]), absorption=water)
            )
        if piped.any():
            mask = build_visibility(
                [BranchOpenings()], lamp, chunk[piped], self_occlusion=NoOcclusion()
            )
            out[piped] = np.asarray(
                direct_fluence_rate(
                    lamp, jnp.asarray(chunk[piped]), absorption=water, visibility=mask
                )
            )
        field[start:stop] = out
        if (start // CHUNK) % 10 == 0 or stop == len(points):
            elapsed = time.perf_counter() - started
            _say(f"  {label}: {stop}/{len(points)} cells, {elapsed:.0f} s")
    return field


def refinement_error(lamp, points, region, coarse, rng) -> dict:
    """How far the coarse lamp is from a refined one, on the cells where it matters most."""
    # One short window along the lamp, so only the facets near it are refined: refining against
    # cells spread along the whole lamp split all 7516 facets up to a thousandfold and exhausted
    # memory. The error is set locally, by the facets nearest a cell, so a window is a fair sample.
    distance = np.hypot(points[:, 1], points[:, 2]) - R_LAMP
    window = (region == 0) & (np.abs(points[:, 0] - 0.40) < 0.01) & (distance < 0.005)
    near = np.flatnonzero(window)
    sample = rng.choice(near, size=min(1500, len(near)), replace=False)
    fine_lamp, _ = refine_for_receivers(lamp, points[sample], max_ratio=0.25, max_levels=4)
    fine = gather(fine_lamp, points[sample], region[sample], "refined lamp, near-lamp sample")
    relative = np.abs(coarse[sample] - fine) / fine
    return {
        "sample_cells": len(sample),
        "within_mm_of_lamp": 5.0,
        "axial_window_m": [0.39, 0.41],
        "coarse_facets": int(lamp.n_facets),
        "refined_facets": int(fine_lamp.n_facets),
        "relative_error_median": float(np.median(relative)),
        "relative_error_p99": float(np.percentile(relative, 99)),
        "relative_error_max": float(relative.max()),
    }


def plots(points, fields, region) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ours = fields["G_aquaflux"]
    # Thick enough to hold a centroid of every cell the plane cuts: the bulk cells are ~7.5 mm, so
    # a thinner slab leaves whole bands of the slice with no cell in it.
    mid = np.abs(points[:, 1]) < 0.004
    log_g = {
        name: np.log10(np.maximum(fields[name], 1e-3)) for name in DOM_RUNS | {"G_aquaflux": 0}
    }
    ratio = {
        name: np.log2(np.maximum(fields[name], 1e-6) / np.maximum(ours, 1e-6)) for name in DOM_RUNS
    }

    def slice_figure(window, path, size, marker):
        rows = [
            ("aquaflux G, log10 W/m^2", log_g["G_aquaflux"], "viridis", (0, 3.2)),
            ("DOM 256 directions G, log10 W/m^2", log_g["G_dom256"], "viridis", (0, 3.2)),
            ("log2(DOM 64 / aquaflux)", ratio["G_dom64"], "RdBu_r", (-1, 1)),
            ("log2(DOM 256 / aquaflux)", ratio["G_dom256"], "RdBu_r", (-1, 1)),
        ]
        fig, axes = plt.subplots(len(rows), 1, figsize=size, sharex=True, layout="constrained")
        for ax, (title, values, cmap, limits) in zip(axes, rows, strict=True):
            shown = ax.scatter(
                points[window, 0], points[window, 2], c=values[window], s=marker, cmap=cmap,
                vmin=limits[0], vmax=limits[1], marker="s", linewidths=0,
            )  # fmt: skip
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.set_ylabel("z (m)")
            fig.colorbar(shown, ax=ax, shrink=0.9, pad=0.01)
        axes[-1].set_xlabel("x (m)")
        fig.savefig(OUT / path, dpi=160)
        plt.close(fig)

    # The whole chamber and the start of the inlet, lamp tip included: where along the lamp they
    # differ. Then the elbow and riser, where visibility through the opening decides everything.
    slice_figure(
        mid & (points[:, 2] < 0.05) & (points[:, 0] < 1.0), "slice_chamber.png", (14, 8), 3
    )
    slice_figure(mid & (points[:, 0] < 0.20) & (points[:, 2] < 0.25), "slice_elbow.png", (7, 16), 6)

    def profile(ax, mask, coordinate, xlabel):
        order = np.flatnonzero(mask)
        order = order[np.argsort(coordinate[order])]
        for name, style in (("G_dom64", "C1."), ("G_dom256", "C0."), ("G_aquaflux", "k.")):
            ax.plot(coordinate[order], fields[name][order], style, ms=2.5, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("G (W/m^2)")
        ax.set_yscale("log")
        ax.legend()

    radius = np.hypot(points[:, 1], points[:, 2])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    profile(axes[0, 0], (np.abs(points[:, 0] - 0.40) < 0.004) & (region == 0), radius,
            "radius at x = 0.40 m, all angles (m)")  # fmt: skip
    profile(axes[0, 1], (np.abs(radius - 0.030) < 0.003) & (region == 0), points[:, 0],
            "x along r = 30 mm, all angles (m)")  # fmt: skip
    profile(axes[1, 0], (region == 2) & (np.hypot(points[:, 0] - X_RISER, points[:, 1]) < 0.003),
            points[:, 2], "height up the riser axis (m)")  # fmt: skip
    profile(
        axes[1, 1], (region == 1) & (radius < 0.003), points[:, 0], "x along the inlet axis (m)"
    )
    fig.savefig(OUT / "profiles.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    _say("reading the mesh")
    mesh = read_openfoam(CASE)
    geometry = mesh.geometry()
    points = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)
    region = np.where(_in_chamber(points), 0, np.where(points[:, 0] > X_BODY_END, 1, 2))
    outside = np.asarray(BranchOpenings().contains(jnp.asarray(points)))
    _say(
        f"{len(points)} cells: chamber {int((region == 0).sum())}, inlet {int((region == 1).sum())}, "
        f"riser {int((region == 2).sum())}; outside all three cylinders {int(outside.sum())}"
    )

    lamp = lamp_surfaces()
    _say(f"lamp: {lamp.n_facets} facets, {float(np.sum(np.asarray(lamp.area))) * EXITANCE:.3f} W")

    saved = OUT / "G_aquaflux.npy"
    if saved.exists() and np.load(saved).shape == (len(points),):
        _say(f"reusing {saved.name} (delete it to recompute)")
        fields = {"G_aquaflux": np.load(saved)}
    else:
        _say("gathering aquaflux G at every cell centre")
        fields = {"G_aquaflux": gather(lamp, points, region, "aquaflux")}
        np.save(saved, fields["G_aquaflux"])

    _say("discretization check near the lamp")
    refinement = refinement_error(lamp, points, region, fields["G_aquaflux"], rng)
    _say(f"  {refinement}")

    for name, run in DOM_RUNS.items():
        fields[name] = np.asarray(read_volume_scalar_field(WORK / "runs" / run / "G", mesh))
    for name in DOM_RUNS:
        fields[f"{name}_over_aquaflux"] = fields[name] / np.maximum(fields["G_aquaflux"], 1e-12)
    fields["region"] = region.astype(float)

    summary = {"cells": len(points), "refinement": refinement, "volume_mean": {}, "ratio": {}}
    for label, mask in (("all", np.ones(len(points), bool)), ("chamber", region == 0),
                        ("inlet", region == 1), ("riser", region == 2)):  # fmt: skip
        weights = volume[mask]
        summary["volume_mean"][label] = {
            name: float(np.sum(fields[name][mask] * weights) / np.sum(weights))
            for name in ("G_aquaflux", *DOM_RUNS)
        }
        _say(f"volume-weighted mean G, {label}: {summary['volume_mean'][label]}")
    lit = fields["G_aquaflux"] > 1e-3 * fields["G_aquaflux"].max()
    for name in DOM_RUNS:
        ratio = fields[f"{name}_over_aquaflux"][lit]
        summary["ratio"][name] = {
            f"p{q}": float(np.percentile(ratio, q)) for q in (1, 10, 50, 90, 99)
        }
        _say(f"{name} / aquaflux over lit cells: {summary['ratio'][name]}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _say("writing the VTU and the plots")
    write_vtu(mesh, fields, OUT / "sozzi_fluence.vtu")
    plots(points, fields, region)
    _say(f"done: {OUT}")


if __name__ == "__main__":
    main()
    sys.exit(0)
