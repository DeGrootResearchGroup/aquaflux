"""Compare the resolution study's runs: the background grid, and each tessellation setting.

Reads every run under ``work/runs`` (each a directory named by its settings, written by ``run.py``)
and writes ``work/study/``: ``runs.csv`` (one row of headline numbers per run), ``pointwise.csv``
(field differences between runs at the same points), ``report.md``, and figures.

How runs are compared point by point:

- **Grid**: ``G`` at a node does not depend on the other nodes, so what the grid spacing changes is
  how well the nodes represent the field between them. Each coarse run is interpolated to every
  node of each finer run (trilinearly in log G from its cell's corners in the water, dry corners
  dropped and the weights rescaled) and compared with the value gathered there. Where the coarse
  nodes are a subset of the fine ones (8 mm in 4 mm, 6 mm in 3 mm) those coincident nodes are
  checked to agree exactly, which is what says the two runs are the same field. Volume-mean
  values are in ``runs.csv``.
- **Tessellation**: runs on one grid have identical nodes, so each is compared with the finest run
  of its family (lamp rings, lamp sectors, wall edges, hole sides) node by node.

Differences are relative to the reference, over nodes where the reference is above ``1e-3`` of its
peak (the lit water), and also reported in bands of distance from the nearest sleeve, where a coarse
lamp is wrong first.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from geometry import Reactor  # noqa: E402

mpl.use("Agg")

RUNS = Path(os.environ.get("TUBULAR_RUNS", HERE / "work" / "runs"))
OUT = Path(os.environ.get("TUBULAR_STUDY", HERE / "work" / "study"))
TAG = re.compile(r"h(?P<h>[\d.]+)_ax(?P<ax>[\d.]+)_s(?P<s>\d+)_w(?P<w>[\d.]+)_t(?P<t>\d+)")
BASE = {"ax": 8.0, "s": 24, "w": 30.0, "t": 8}
#: Bands of distance from the nearest sleeve, in metres.
BANDS = ((0.0, 0.002), (0.002, 0.005), (0.005, 0.010), (0.010, 0.030), (0.030, 1.0))
LIT = 1e-3


class Run:
    def __init__(self, path: Path) -> None:
        match = TAG.fullmatch(path.name)
        self.path = path
        self.tag = path.name
        self.h = float(match["h"]) * 1e-3
        self.settings = {
            "ax": float(match["ax"]), "s": int(match["s"]),
            "w": float(match["w"]), "t": int(match["t"]),
        }  # fmt: skip
        self.summary = json.loads((path / "summary.json").read_text())
        data = np.load(path / "fluence_rate.npz")
        self.origin = data["origin"]
        self.shape = np.asarray(data["shape"])
        self.index = data["index"]
        self.fields = {k: data[k] for k in ("G", "G_direct", "G_reflected")}

    @property
    def ijk(self) -> np.ndarray:
        return np.column_stack(np.unravel_index(self.index, tuple(self.shape)))

    @property
    def points(self) -> np.ndarray:
        return self.origin + self.h * self.ijk

    def is_baseline_tessellation(self) -> bool:
        return self.settings == BASE

    def differs_in(self) -> list[str]:
        return [k for k, v in self.settings.items() if v != BASE[k]]


def load() -> list[Run]:
    # A run still in progress has no summary yet.
    runs = [Run(p) for p in sorted(RUNS.iterdir())
            if TAG.fullmatch(p.name) and (p / "fluence_rate.npz").exists()]  # fmt: skip
    if not runs:
        raise SystemExit(f"no runs under {RUNS}")
    return runs


def headline(run: Run) -> dict:
    r = run.summary["results"]
    plate = run.path / "plate_shadow.npz"
    row = {
        "tag": run.tag, "h_mm": run.h * 1e3, **run.settings,
        "nodes": len(run.index), "lamp_facets": run.summary["counts"]["lamp_facets_volume"],
        "wall_triangles": run.summary["counts"]["wall_triangles"],
        "mean_G": r["mean_G_W_per_m2"],
        "mean_G_downstream": r["mean_G_downstream_of_plate_W_per_m2"],
        "mean_G_upstream": r["mean_G_upstream_of_plate_W_per_m2"],
        "reflected_share": r["reflected_share_of_mean_G"],
        "water_absorbed_W": r["water_absorbed_W_grid_estimate"],
        "walls_absorbed_W": sum(r["walls_absorbed_W"].values()),
        "G_p50": r["G_percentiles_1_10_50_90_99"][2], "G_p99": r["G_percentiles_1_10_50_90_99"][4],
        "seconds": run.summary["seconds"],
    }  # fmt: skip
    if plate.exists():
        data = np.load(plate)
        up, free = data["upstream_3mm"], data["upstream_no_plate"]
        row["plate_transmission_3mm"] = float(np.nanmean(up) / np.nanmean(free))
        row["plate_lit_share_3mm"] = float(np.mean(up[np.isfinite(up)] > 0))
    return row


def relative(test: np.ndarray, reference: np.ndarray, distance: np.ndarray) -> dict:
    """Relative differences over lit nodes, overall and by distance from a sleeve."""
    lit = reference > LIT * reference.max()
    rel = np.abs(test[lit] - reference[lit]) / reference[lit]
    out = {
        "lit_nodes": int(lit.sum()),
        "median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
        "p99": float(np.percentile(rel, 99)), "max": float(rel.max()),
        "mean_signed": float(np.mean((test[lit] - reference[lit]) / reference[lit])),
    }  # fmt: skip
    for low, high in BANDS:
        band = (distance[lit] >= low) & (distance[lit] < high)
        key = f"median_{low * 1e3:g}-{high * 1e3:g}mm"
        out[key] = float(np.median(rel[band])) if band.any() else float("nan")
    return out


def interpolate(coarse: Run, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``coarse``'s G at ``points`` (n, 3): trilinear in log G over the corners of each point's grid
    cell that are in the water, their weights rescaled; also whether the point was reached (some
    corner in the water)."""
    full = np.full(int(np.prod(coarse.shape)), np.nan)
    floor = 1e-12 * coarse.fields["G"].max()
    full[coarse.index] = np.log(np.maximum(coarse.fields["G"], floor))
    scaled = (points - coarse.origin) / coarse.h
    low = np.clip(np.floor(scaled).astype(int), 0, coarse.shape - 2)
    # A fine node past the coarse grid's last plane takes that plane's values, not an extrapolation.
    frac = np.clip(scaled - low, 0.0, 1.0)
    total = np.zeros(len(points))
    weight = np.zeros(len(points))
    for corner in np.ndindex(2, 2, 2):
        c = np.asarray(corner)
        w = np.prod(np.where(c == 1, frac, 1.0 - frac), axis=1)
        value = full[np.ravel_multi_index((low + c).T, tuple(coarse.shape))]
        wet = np.isfinite(value)
        total += np.where(wet, w * np.where(wet, value, 0.0), 0.0)
        weight += np.where(wet, w, 0.0)
    reached = weight > 0
    return np.exp(total / np.where(reached, weight, 1.0)), reached


def coincident(coarse: Run, fine: Run) -> tuple[np.ndarray, np.ndarray] | None:
    """Pairs (coarse position, fine position) of nodes the two grids share, or None if none."""
    ratio = coarse.h / fine.h
    if abs(ratio - round(ratio)) > 1e-9 or not np.allclose(coarse.origin, fine.origin):
        return None
    keys = np.ravel_multi_index((coarse.ijk * round(ratio)).T, tuple(fine.shape))
    where = np.minimum(np.searchsorted(fine.index, keys), len(fine.index) - 1)
    found = fine.index[where] == keys
    return np.flatnonzero(found), where[found]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reactor = Reactor()
    runs = load()
    rows = [headline(r) for r in runs]
    with (OUT / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

    comparisons = []
    # Grid: baseline tessellation, each run interpolated to each finer one's nodes.
    grid_runs = sorted((r for r in runs if r.is_baseline_tessellation()), key=lambda r: -r.h)
    for coarse in grid_runs:
        for fine in grid_runs:
            if fine.h >= coarse.h:
                continue
            shared = coincident(coarse, fine)
            exact = None
            if shared is not None:
                a, b = shared
                exact = float(np.max(np.abs(coarse.fields["G"][a] - fine.fields["G"][b])))
            values, reached = interpolate(coarse, fine.points)
            distance = reactor.lamp_distance(fine.points[reached])
            comparisons.append({
                "study": "grid (interpolated)", "field": "G", "test": coarse.tag,
                "reference": fine.tag, "coincident_max_abs_diff": exact,
                "unreached_nodes": int((~reached).sum()),
                **relative(values[reached], fine.fields["G"][reached], distance),
            })  # fmt: skip
    # Tessellation: on each grid, each family against its finest member and the baseline.
    families = {"ax": min, "s": max, "w": min, "t": max}
    for h in sorted({r.h for r in runs}):
        same = [r for r in runs if r.h == h]
        base = next((r for r in same if r.is_baseline_tessellation()), None)
        if base is None:
            continue
        distance = reactor.lamp_distance(base.points)
        for key, finest in families.items():
            family = [r for r in same if r.differs_in() in ([], [key])]
            if len(family) < 2:
                continue
            reference = next(
                r for r in family if r.settings[key] == finest(m.settings[key] for m in family)
            )
            for test in family:
                if test is reference or not np.array_equal(test.index, reference.index):
                    continue
                for field in ("G", "G_reflected"):
                    comparisons.append({
                        "study": f"tessellation:{key}", "field": field, "test": test.tag,
                        "reference": reference.tag,
                        **relative(test.fields[field], reference.fields[field], distance),
                    })  # fmt: skip
    if comparisons:
        keys = list(dict.fromkeys(k for c in comparisons for k in c))
        with (OUT / "pointwise.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comparisons)
    figures(runs, rows, comparisons)
    report(rows, comparisons)
    print(OUT)


def figures(runs, rows, comparisons) -> None:
    grid = sorted(
        (r for r in rows if all(r[k] == v for k, v in BASE.items())), key=lambda r: r["h_mm"]
    )
    if len(grid) > 1:
        fig, axes = plt.subplots(1, 4, figsize=(19, 4), layout="constrained")
        h = [r["h_mm"] for r in grid]
        finest = min(grid, key=lambda r: r["h_mm"])["tag"]
        errors = [
            c for c in comparisons if c["study"].startswith("grid") and c["reference"] == finest
        ]
        if errors:
            ax = axes[3]
            spacing = [float(TAG.fullmatch(c["test"])["h"]) for c in errors]
            for key in [k for k in errors[0] if k.startswith("median_")] + ["median"]:
                ax.loglog(spacing, [c[key] for c in errors], "o-",
                          label=key.replace("median_", "") + (" from sleeve" if key != "median" else " (all lit)"))  # fmt: skip
            ax.loglog(spacing, [errors[0]["median"] * (x / spacing[0]) ** 2 for x in spacing], "k--",
                      lw=0.8, label="h² slope")  # fmt: skip
            ax.set_xlabel("grid spacing h (mm)")
            ax.set_ylabel(f"median |ΔG|/G, interpolated to the {finest.split('_')[0][1:]} mm grid")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
        for ax, key, label in zip(axes[:3], ("mean_G", "water_absorbed_W", "plate_transmission_3mm"),
                                  ("volume-mean G (W/m²)", "water absorbed, grid estimate (W)",
                                   "plate transmission 3 mm upstream"), strict=True):  # fmt: skip
            ax.plot(h, [r.get(key, np.nan) for r in grid], "o-")
            ax.set_xlabel("grid spacing h (mm)")
            ax.set_ylabel(label)
            ax.invert_xaxis()
            ax.grid(alpha=0.3)
        fig.suptitle("Background grid study (baseline tessellation)")
        fig.savefig(OUT / "grid_study.png", dpi=120)
    tess = [c for c in comparisons if c["study"].startswith("tessellation") and c["field"] == "G"]
    if tess:
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        bands = [k for k in tess[0] if k.startswith("median_")]
        width = 0.8 / len(tess)
        for i, c in enumerate(tess):
            ax.bar(np.arange(len(bands)) + i * width, [c[b] for b in bands], width,
                   label=f"{c['test']} vs {c['reference']}")  # fmt: skip
        ax.set_xticks(np.arange(len(bands)) + 0.4 - width / 2)
        ax.set_xticklabels([b.replace("median_", "") + " from sleeve" for b in bands])
        ax.set_yscale("log")
        ax.set_ylabel("median |ΔG| / G")
        ax.legend(fontsize=7)
        ax.set_title(
            "Tessellation study: G against the finest of each family, by distance from a sleeve"
        )
        fig.savefig(OUT / "tessellation_study.png", dpi=120)


def report(rows, comparisons) -> None:
    lines = ["# Resolution study", "", "## Runs", ""]
    cols = ["tag", "nodes", "lamp_facets", "wall_triangles", "mean_G", "mean_G_upstream",
            "reflected_share", "water_absorbed_W", "plate_transmission_3mm", "seconds"]  # fmt: skip
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    if comparisons:
        lines += ["", "## Point by point", ""]
        cols = ["study", "field", "test", "reference", "lit_nodes", "median", "p90", "p99", "max",
                "mean_signed", "median_0-2mm", "median_2-5mm", "median_10-30mm",
                "coincident_max_abs_diff"]  # fmt: skip
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for c in comparisons:
            lines.append("| " + " | ".join(_fmt(c.get(k, "")) for k in cols) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
