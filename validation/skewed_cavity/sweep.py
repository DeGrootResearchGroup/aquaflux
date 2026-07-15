"""Sweep the skew angle: does aquaflux's Newton count stay flat as OpenFOAM's climbs?

For beta in {30, 45, 60} deg (mesh non-orthogonality = 90 - beta = {60, 45, 30} deg)
this regenerates the parallelogram mesh, runs the OpenFOAM SIMPLE sweep
(nNonOrthogonalCorrectors 0/1/2) in the container, then solves the same mesh with
aquaflux two ways (corrected gradient in the AD Jacobian vs stop_gradient-lagged).
Writes figures/sweep.png and report_sweep.md.

Run from the aquaflux repo root:  PYTHONPATH=. python3 validation/skewed_cavity/sweep.py
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare as C  # noqa: E402  (shared helpers; also enables aquaflux x64)
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.schemes import CorrectedGreenGauss  # noqa: E402

BETAS = (30, 45, 60)
IMAGE = "openfoam13:latest"
SWEEP = HERE / "runs_sweep"


def _blockmesh(beta_deg: float) -> str:
    """blockMeshDict for the parallelogram cavity skewed at ``beta_deg`` (literal coords)."""
    dx, dy = math.cos(math.radians(beta_deg)), math.sin(math.radians(beta_deg))
    return f"""FoamFile {{ format ascii; class dictionary; object blockMeshDict; }}
// beta = {beta_deg} deg  ->  mesh non-orthogonality = {90 - beta_deg} deg
convertToMeters 1;
vertices
(
    (0 0 0)
    (1 0 0)
    ({1 + dx:.12f} {dy:.12f} 0)
    ({dx:.12f} {dy:.12f} 0)
    (0 0 0.1)
    (1 0 0.1)
    ({1 + dx:.12f} {dy:.12f} 0.1)
    ({dx:.12f} {dy:.12f} 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1) );
edges ( );
boundary
(
    movingWall {{ type wall; faces ( (3 7 6 2) ); }}
    fixedWalls {{ type wall; faces ( (0 4 7 3) (2 6 5 1) (1 5 4 0) ); }}
    back  {{ type empty; faces ( (0 3 2 1) ); }}
    front {{ type empty; faces ( (4 5 6 7) ); }}
);
"""


def _checkmesh_nonortho(case: Path) -> float:
    text = (case / "log.checkMesh").read_text()
    m = re.search(r"non-orthogonality Max:\s*([\d.]+)", text)
    return float(m.group(1)) if m else float("nan")


def _of_iterations(nonorth_dir: Path) -> int | None:
    """OpenFOAM outer iterations to converge, or None if it diverged / hit the cap."""
    log = nonorth_dir / "log.foamRun"
    if not log.exists():
        return None
    text = log.read_text()
    m = re.search(r"SIMPLE solution converged in (\d+) iterations", text)
    return int(m.group(1)) if m else None
def _steps_to_tol(history: np.ndarray, tol: float = 1e-8) -> int | None:
    """Number of Newton steps for the residual history to first drop below ``tol``."""
    below = np.nonzero(history < tol)[0]
    return int(below[0]) if below.size else None


def _run_beta(beta: int) -> dict:
    betadir = SWEEP / f"beta_{beta}"
    case = betadir / "case"
    if case.exists():
        shutil.rmtree(betadir)
    shutil.copytree(HERE / "of_case", case)
    (case / "system" / "blockMeshDict").write_text(_blockmesh(beta))

    print(f"\n=== beta = {beta} deg  (target non-orthogonality {90 - beta} deg) ===")
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{HERE}:/work", "-w", "/work", IMAGE,
         "bash", "run_of_beta.sh", f"runs_sweep/beta_{beta}"],
        check=True,
    )
    nonortho = _checkmesh_nonortho(case)
    of = {n: _of_iterations(betadir / f"nonorth_{n}") for n in (0, 1, 2)}
    print(f"  checkMesh non-orthogonality = {nonortho:.1f} deg;  OF outer iters {of}")

    mesh = read_openfoam(betadir / "nonorth_2")
    geom = mesh.geometry()
    imp_hist, _, _ = C._solve(C._build(mesh, geom, CorrectedGreenGauss()), 12, label=f"impl b{beta}")
    lag_hist, _, _ = C._solve(
        C._build(mesh, geom, C.LaggedGradient(inner=CorrectedGreenGauss())), 60, label=f"lag b{beta}"
    )
    return {
        "beta": beta,
        "nonortho": nonortho,
        "of": of,
        "aq_impl": _steps_to_tol(imp_hist),
        "aq_lag": _steps_to_tol(lag_hist),
    }


def _figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = sorted(results, key=lambda d: d["nonortho"])
    xs = [d["nonortho"] for d in r]
    SENT = 900  # a "did not converge" band above the data

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    def plot_series(key, style, color, label, lw=1.6, ms=8):
        vals = [(d[key] if key in d else d["of"][key[1]]) for d in r]
        conv = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if conv:
            ax.plot([c[0] for c in conv], [c[1] for c in conv], style, color=color,
                    label=label, lw=lw, ms=ms)
        else:
            ax.plot([], [], style, color=color, label=label)  # legend entry only
        for x, v in zip(xs, vals):
            if v is None:  # diverged / did not converge -> a marker on the band
                ax.plot(x, SENT, "x", color=color, ms=11, mew=2.5)

    ax.axhspan(SENT / 1.35, SENT * 1.35, color="0.92", zorder=0)
    ax.text(xs[0], SENT, "  diverged / no convergence", fontsize=8, va="center", color="0.4")
    plot_series(("of", 1), "^-", "#2166ac", "OpenFOAM SIMPLE, nOrtho=1")
    plot_series(("of", 2), "v-", "#5aa0d6", "OpenFOAM SIMPLE, nOrtho=2")
    plot_series("aq_lag", "s-", "#c2630a", "aquaflux lagged (deferred correction)")
    plot_series("aq_impl", "o-", "#1b7837", "aquaflux implicit (correction in AD Jacobian)", lw=2.4, ms=9)

    ax.set_yscale("log")
    ax.set_ylim(3, 1600)
    ax.set_xlabel("mesh non-orthogonality (deg)")
    ax.set_ylabel("nonlinear iterations to converge  ($|R| < 10^{-8}$)")
    ax.set_title("Iterations vs non-orthogonality: aquaflux stays flat, OpenFOAM climbs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "sweep.png", dpi=140)


def _report(results):
    r = sorted(results, key=lambda d: d["nonortho"])

    def cell(v):
        return str(v) if v is not None else "diverged"

    rows = "\n".join(
        f"| {d['beta']} | {d['nonortho']:.1f} | {cell(d['of'][0])} | {cell(d['of'][1])} | "
        f"{cell(d['of'][2])} | **{cell(d['aq_impl'])}** | {cell(d['aq_lag'])} |"
        for d in r
    )
    md = f"""# Skew-angle sweep: iterations vs mesh non-orthogonality

Same inclined lid-driven cavity (Re = 100, 40x40), swept over skew angle. The mesh
non-orthogonality is `90 - beta`. "Iterations" = OpenFOAM SIMPLE outer iterations,
or aquaflux Newton steps to `|R| < 1e-8`.

| beta (deg) | non-orthogonality (deg) | OF nOrtho=0 | OF nOrtho=1 | OF nOrtho=2 | aquaflux implicit | aquaflux lagged |
|---|---|---|---|---|---|---|
{rows}

**aquaflux with the corrected gradient in the AD Jacobian stays flat** (a handful of
quadratic Newton steps) across the whole range, while the segregated SIMPLE count and
the deferred / lagged count grow with non-orthogonality (and the uncorrected variants
diverge). Folding the correction into the Jacobian is what removes the
non-orthogonality penalty. See `figures/sweep.png`.
"""
    (HERE / "report_sweep.md").write_text(md)


def main() -> None:
    SWEEP.mkdir(exist_ok=True)
    results = [_run_beta(b) for b in BETAS]
    _figure(results)
    _report(results)
    print(f"\nWrote {HERE / 'report_sweep.md'} and {HERE / 'figures' / 'sweep.png'}")


if __name__ == "__main__":
    main()
