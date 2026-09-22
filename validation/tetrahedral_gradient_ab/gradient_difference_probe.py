"""Issue #435: where do the gradients of the scheme that marches differ from those that do not?

On this mesh ``CorrectedGreenGauss`` converges the coupled march and ``CompactGreenGauss``,
``MultipleCorrectionGradient`` (and option 1) do not, and neither reconstruction accuracy, the
correction's iteration count, nor the k/omega closure devices explains the split. This evaluates every
scheme's gradients at ONE state -- the Re/10 anchor converged with ``CorrectedGreenGauss`` -- so any
difference is the scheme and nothing else, and reports where they differ:

* the velocity-gradient tensor, the strain rate the production terms are built from, ``grad k`` and
  ``grad omega`` -- each compared with ``CorrectedGreenGauss`` per cell, grouped by wall-cell ring, by the
  number of boundary faces a cell owns, by the multiple-correction scheme's own ``max|M2^-1|``, by the
  cell's non-orthogonality and by streamwise position;
* the worst-differing cells, with those attributes, and rank correlations of the difference with them;
* the same schemes' error against the exact gradient of a smooth analytic field, so "which one is more
  accurate on this mesh" is measured rather than assumed.

Compared: ``compact GG``, ``multcorr repaired`` (the scheme as it marches) and ``multcorr first pass``
(its linear-exact first pass with no Hessian correction), which separates what the second pass adds.
Option 1 is not included: its velocity gradient needs the live ``k`` the coupled residual supplies.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/gradient_difference_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnly, cell_non_orthogonality
from seed_state_probe import rings_from_wall
from warm_start_probe import RATIO, SEED_STEPS, march


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-cell ``|a - b| / |b|`` (Frobenius), with ``|b|`` floored at 5 % of its 95th percentile."""
    n = a.shape[0]
    numerator = np.linalg.norm((a - b).reshape(n, -1), axis=1)
    reference = np.linalg.norm(b.reshape(n, -1), axis=1)
    return numerator / np.maximum(reference, 0.05 * np.percentile(reference, 95))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def summary(values: np.ndarray, mask: np.ndarray) -> str:
    if not mask.any():
        return f"{'--':>26}"
    v = values[mask]
    return f"{np.median(v):8.3f} {np.percentile(v, 90):8.3f} {v.max():8.3f}"


def grouped_report(
    title: str, values: dict[str, np.ndarray], groups: dict[str, np.ndarray]
) -> None:
    """One table: rows are cell groups, column blocks are schemes, each ``median p90 max``."""
    print(
        f"  --- {title} (relative difference from corrected GG: median  p90  max) ---", flush=True
    )
    header = "".join(f"{name:>27}" for name in values)
    print(f"  {'group':<22}{'cells':>6}{header}", flush=True)
    for group, mask in groups.items():
        row = "".join(f"  {summary(v, mask)}" for v in values.values())
        print(f"  {group:<22}{int(mask.sum()):>6}{row}", flush=True)


def fields_at(case, flow, k, omega) -> dict[str, np.ndarray]:
    velocity = case.momentum.velocity_fields(flow)
    closure = case.turbulence.closure_fields(velocity, k, omega)
    return {
        "velocity gradient": np.asarray(velocity.gradient),
        "strain rate S": np.asarray(closure.strain_rate)[:, None],
        "grad k": np.asarray(closure.grad_k),
        "grad omega": np.asarray(closure.grad_omega),
    }


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    flow0 = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k0 = jnp.full(n, float(inlet_k(jnp.array(U_IN), INTENSITY)))
    omega0 = jnp.full(n, float(inlet_omega(jnp.array(k0[0]), LENGTH_SCALE, SSTModel())))
    seed = march("seed", seed_case, flow0, k0, omega0, 1e-4, SEED_STEPS)
    if seed is None:
        return
    flow, k, omega = seed

    multcorr = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    schemes = {
        "compact GG": CompactGreenGauss(),
        "multcorr repaired": multcorr,
        "multcorr first pass": FirstPassOnly(multcorr),
    }
    reference = fields_at(seed_case, flow, k, omega)
    compared = {
        name: fields_at(build_case(s).with_scaled_molecular_viscosity(RATIO), flow, k, omega)
        for name, s in schemes.items()
    }

    face_cells = mesh.face_cells
    boundary = ~np.asarray(face_cells.interior)
    owned = np.bincount(np.asarray(face_cells.owner)[boundary], minlength=n)
    rings = rings_from_wall(mesh, seed_case.turbulence.wall_cells)
    m2 = np.max(np.abs(np.asarray(multcorr.prepared.m2_inverse)), axis=(1, 2))
    angle = cell_non_orthogonality(mesh, geometry)
    x = np.asarray(geometry.cell.centroid)[:, 0]
    x_edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    x_bin = np.digitize(x, x_edges)
    angle_edges = np.quantile(angle, [0.25, 0.5, 0.75])
    groups = {
        "all cells": np.ones(n, dtype=bool),
        **{f"ring {r}": rings == r for r in range(int(rings.max()) + 1)},
        "owns 0 bdry faces": owned == 0,
        "owns 1 bdry face": owned == 1,
        "owns >=2 bdry faces": owned >= 2,
        "max|M2^-1| <= 10": m2 <= 10,
        "max|M2^-1| 10-100": (m2 > 10) & (m2 <= 100),
        "max|M2^-1| 100-1e3": (m2 > 100) & (m2 <= 1e3),
        "max|M2^-1| > 1e3": m2 > 1e3,
        **{f"non-orth quartile {q + 1}": np.digitize(angle, angle_edges) == q for q in range(4)},
        **{f"streamwise fifth {b + 1}": x_bin == b for b in range(5)},
    }
    print(
        f"=== at the Re/{RATIO:g} anchor converged with corrected GG: {n} cells, ring sizes "
        f"{np.bincount(rings).tolist()}, {int((owned >= 2).sum())} cells own >=2 boundary faces ===",
        flush=True,
    )

    worst_source = None
    for field in reference:
        diffs = {
            name: relative_difference(compared[name][field], reference[field]) for name in schemes
        }
        if field == "velocity gradient":
            worst_source = diffs
        mask_groups = groups
        if field == "grad omega":
            # omega's wall-fixation cells carry an imposed gradient in every scheme, so ring 0 is not a
            # reconstruction comparison there.
            mask_groups = {g: m & (rings > 0) for g, m in groups.items()}
        grouped_report(field, diffs, mask_groups)

    print(
        "  --- SIGNED: median |scheme| / |corrected GG| per group (>1: the scheme's gradient is larger) ---",
        flush=True,
    )
    for field in ("velocity gradient", "strain rate S", "grad k"):
        magnitude = {
            name: np.linalg.norm(fields[field].reshape(n, -1), axis=1)
            for name, fields in {"corrected GG": reference, **compared}.items()
        }
        print(f"  {field}", flush=True)
        for group in ("ring 0", "ring 1", "ring 2", "ring 3", "owns >=2 bdry faces"):
            mask = groups[group]
            ratios = "  ".join(
                f"{name} {np.median(magnitude[name][mask] / np.maximum(magnitude['corrected GG'][mask], 1e-30)):6.2f}"
                for name in schemes
            )
            print(f"    {group:<22}{ratios}", flush=True)
    print(
        "  --- do the schemes that fail agree WITH EACH OTHER? velocity gradient, median relative difference ---",
        flush=True,
    )
    vg = {name: compared[name]["velocity gradient"] for name in schemes}
    pairs = (
        ("compact GG", "multcorr first pass"),
        ("compact GG", "multcorr repaired"),
        ("multcorr first pass", "multcorr repaired"),
    )
    for group in ("ring 0", "ring 1", "ring 2", "ring 3"):
        mask = groups[group]
        row = "  ".join(
            f"{a[:9]}~{b[:9]} {np.median(relative_difference(vg[a], vg[b])[mask]):6.2f}"
            for a, b in pairs
        )
        print(
            f"    {group:<10}{row}   (vs corrected GG: compact {np.median(relative_difference(vg['compact GG'], reference['velocity gradient'])[mask]):.2f})",
            flush=True,
        )
    diff = worst_source["multcorr repaired"]
    print("  --- worst cells, velocity gradient, multcorr repaired vs corrected GG ---", flush=True)
    print(
        f"  {'cell':>6} {'rel diff':>9} {'compact':>8} {'1st pass':>9} {'ring':>5} {'bdry':>5} "
        f"{'max|M2^-1|':>11} {'non-orth':>9} {'x/L':>6}",
        flush=True,
    )
    for c in np.argsort(diff)[::-1][:12]:
        print(
            f"  {c:>6} {diff[c]:>9.2f} {worst_source['compact GG'][c]:>8.2f} "
            f"{worst_source['multcorr first pass'][c]:>9.2f} {rings[c]:>5} {owned[c]:>5} "
            f"{m2[c]:>11.2e} {angle[c]:>8.1f}° {(x[c] - x.min()) / (x.max() - x.min()):>6.2f}",
            flush=True,
        )
    print(
        "  --- rank correlation of the velocity-gradient difference with a cell property ---",
        flush=True,
    )
    for name, d in worst_source.items():
        print(
            f"  {name:<20} ring {spearman(d, rings):+.2f}  bdry faces {spearman(d, owned):+.2f}  "
            f"max|M2^-1| {spearman(d, m2):+.2f}  non-orth {spearman(d, angle):+.2f}  "
            f"streamwise {spearman(d, x):+.2f}",
            flush=True,
        )

    # Accuracy against an exact gradient, independent of the flow state.
    lo, hi = (
        np.asarray(geometry.cell.centroid).min(axis=0),
        np.asarray(geometry.cell.centroid).max(axis=0),
    )

    def shape(points):
        return (np.asarray(points) - lo) / (hi - lo)

    def analytic(points):
        u = shape(points)
        return np.sin(3.0 * u[:, 0]) * np.cos(2.0 * u[:, 1]) + u[:, 2] ** 2 * u[:, 0] + u[:, 1] ** 3

    def analytic_gradient(points):
        u = shape(points)
        scale = 1.0 / (hi - lo)
        return (
            np.stack(
                [
                    3.0 * np.cos(3.0 * u[:, 0]) * np.cos(2.0 * u[:, 1]) + u[:, 2] ** 2,
                    -2.0 * np.sin(3.0 * u[:, 0]) * np.sin(2.0 * u[:, 1]) + 3.0 * u[:, 1] ** 2,
                    2.0 * u[:, 2] * u[:, 0],
                ],
                axis=1,
            )
            * scale
        )

    cell_x = np.asarray(geometry.cell.centroid)
    face_x = np.asarray(geometry.face.centroid)
    phi = jnp.asarray(analytic(cell_x))
    phi_faces = jnp.asarray(analytic(face_x))
    exact = analytic_gradient(cell_x)
    floor_reference = np.linalg.norm(exact, axis=1)
    errors = {}
    for name, scheme in {"corrected GG": CorrectedGreenGauss(), **schemes}.items():
        bound = scheme.bind(mesh, geometry)
        g = np.asarray(bound.gradients(phi, mesh, geometry, phi_faces))
        errors[name] = np.linalg.norm(g - exact, axis=1) / np.maximum(
            floor_reference, 0.05 * np.percentile(floor_reference, 95)
        )
    print(
        "  --- error against the EXACT gradient of a smooth analytic field (median p90 max) ---",
        flush=True,
    )
    print(f"  {'group':<22}{'cells':>6}" + "".join(f"{name:>27}" for name in errors), flush=True)
    for group in (
        "all cells",
        "ring 0",
        "ring 1",
        "ring 2",
        "owns >=2 bdry faces",
        "max|M2^-1| > 1e3",
    ):
        mask = groups[group]
        print(
            f"  {group:<22}{int(mask.sum()):>6}"
            + "".join(f"  {summary(e, mask)}" for e in errors.values()),
            flush=True,
        )


if __name__ == "__main__":
    main()
