"""The UV dose on the Sozzi & Taghipour reactor, from aquaflux's fluence rate and from DOM's.

Dose is what a UV reactor is designed against, and it is not the fluence rate ``G``: it is what a
particle accumulates along its own path, ``D = integral of G dt``, so the same ``G`` error can matter
a great deal or not at all depending on where the flow carries the particles. This runs
of-optical-radiation's own dose tracker (``radiationDose``) on one converged flow with each fluence
rate in turn -- aquaflux's direct gather and the discrete-ordinates (DOM) solve at 64 and 256
directions -- so the fluence rate is the only thing that changes between them.

**The particles are the same particles.** The tracker draws its injection points and its turbulent
dispersion from a seeded generator, one stream per thread over a static schedule, and dose does not
feed back into the motion (``maxDose 0``). So with one flow, one seed and one thread count, every
run follows identical trajectories and only the dose integrated along them differs -- a *paired*
comparison, particle by particle. That is checked, not assumed: the harness refuses to compare
runs whose end times and end points disagree.

**The fluence rate's patch values are the one choice made here.** The tracker interpolates ``G`` to
the particle from cell *and* boundary values, and aquaflux computes cell values only. So every
field compared is written with ``zeroGradient`` on every patch -- same interior values, same patch
treatment -- and a fourth run keeps DOM 256's own patch values, the file as the solver wrote it, to
measure what that choice moves.

Stages, each skipped when its output exists:

1. compile of-optical-radiation into ``work/foamuser`` (shared with ``generate_dom_reference.py``);
2. copy the meshed case to ``work/dose/case`` and solve the tutorial's steady RANS flow (realizable
   k-epsilon, 25 US gal/min) in parallel, then reconstruct its last time;
3. for each fluence rate, write it as ``G`` in that time and run the tracker (10,000 particles from
   the inlet, the tutorial's settings, trajectories decimated for the VTK), keeping its per-particle
   CSV, summary and trajectories under ``work/dose/runs/<name>/``;
4. compare: the dose distributions, per-particle ratios, log reduction over a range of inactivation
   rates, against Sozzi & Taghipour (2006) where they report a number; plots and ``summary.json``
   in ``work/dose/compare/``, and a VTU of the fluence rates and ``log10 |G_DOM - G_aquaflux|``.

Inputs: ``work/case`` and ``work/runs`` from ``generate_dom_reference.py``, and
``work/compare/G_aquaflux.npy`` from ``compare_fluence.py`` (the case's ``lampWall.stl``, 7,516
facets, exact visibility through the pipe openings).

Run with ``SOZZI_OOR_SOURCE=<checkout> validation/run_case.sh validation/sozzi_radiation/dose_comparison.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
from aquaflux.io import (  # noqa: E402
    FieldTemplate,
    read_field_template,
    read_openfoam,
    read_volume_scalar_field,
    write_openfoam_field,
    write_vtu,
)
from generate_dom_reference import WORK, _in_container, _say, compile_library  # noqa: E402

DOSE = WORK / "dose"
CASE = DOSE / "case"
RUNS = DOSE / "runs"
OUT = DOSE / "compare"

#: MPI ranks for the flow solve, and OpenMP threads for the tracker. The thread count is part of
#: what makes two tracker runs follow the same particles, so it is fixed here rather than left to
#: the container's default.
FLOW_RANKS = 8
TRACKER_THREADS = 8
#: Keep every Nth trajectory vertex in the VTK: the tracker's 5 ms steps over a residence of
#: seconds would otherwise write ~1e7 points per run, far more than a picture needs.
TRAJECTORY_STRIDE = 20

#: Inactivation rate constants (cm^2/mJ) for the log reduction ``-log10 mean(exp(-k D))``. The
#: tutorial reports 0.1; the spread shows how much weight the low-dose tail carries as k grows.
K_INACT = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

#: Sozzi & Taghipour (2006), the L-shaped reactor at 25 US gal/min, Lagrangian dose.
PAPER = {"mean_dose_mJ_cm2": 68.0, "log_reduction_k0.1": 1.87, "dose_range_mJ_cm2": (21.0, 270.0)}

#: Each compared fluence rate: where its interior values come from, and whether its patch values
#: are the zeroGradient ones shared by all (``None``) or the solver's own file.
SOURCES = {
    "aquaflux": {"interior": WORK / "compare" / "G_aquaflux.npy", "native": None},
    "dom64": {"interior": WORK / "runs" / "nphi8_ntheta4" / "G", "native": None},
    "dom256": {"interior": WORK / "runs" / "nphi16_ntheta8" / "G", "native": None},
    "dom256_native_patches": {"interior": None, "native": WORK / "runs" / "nphi16_ntheta8" / "G"},
}
PAIRED = ("aquaflux", "dom64", "dom256")


def _numeric_times(case: Path) -> list[Path]:
    times = [p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.eE+-]+", p.name)]
    return sorted(times, key=lambda p: float(p.name))


# -- stage 2: the flow --------------------------------------------------------------------------


def prepare_case() -> None:
    """Copy the meshed tutorial into this harness's own directory, flow fields only."""
    if (CASE / "constant" / "polyMesh" / "owner").exists():
        _say("dose case already prepared")
        return
    source = WORK / "case"
    if CASE.exists():
        shutil.rmtree(CASE)
    CASE.mkdir(parents=True)
    shutil.copytree(source / "system", CASE / "system")
    shutil.copytree(source / "constant", CASE / "constant", ignore=shutil.ignore_patterns("*.gz"))
    (CASE / "0").mkdir()
    for field in ("U", "p", "k", "epsilon", "nut"):
        shutil.copy(source / "0" / field, CASE / "0" / field)
    (CASE / "out.foam").touch()
    (CASE / "system" / "decomposeParDict").write_text(
        "FoamFile { format ascii; class dictionary; object decomposeParDict; }\n"
        f"numberOfSubdomains {FLOW_RANKS};\nmethod scotch;\n"
    )
    _say(f"dose case prepared at {CASE}")


def solve_flow(oor: Path) -> Path:
    """The tutorial's steady RANS solve, in parallel; returns the reconstructed last time."""
    done = CASE / "flow.json"
    if done.exists():
        latest = CASE / json.loads(done.read_text())["time"]
        _say(f"flow already solved (t = {latest.name})")
        return latest
    mpi = "export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;"
    started = time.perf_counter()
    _say(f"decomposing for {FLOW_RANKS} ranks")
    _in_container(oor, "/work/dose/case", "decomposePar -force", CASE / "log.decomposePar")
    _say("solving the flow (foamRun, PIMPLE with local time stepping)")
    _in_container(
        oor,
        "/work/dose/case",
        f"{mpi} mpirun -np {FLOW_RANKS} foamRun -parallel",
        CASE / "log.foamRun",
    )
    _in_container(oor, "/work/dose/case", "reconstructPar -latestTime", CASE / "log.reconstructPar")
    for processor in CASE.glob("processor*"):
        shutil.rmtree(processor)
    elapsed = time.perf_counter() - started
    latest = _numeric_times(CASE)[-1]
    record = {"time": latest.name, "wall_seconds": round(elapsed, 1), **flow_convergence()}
    done.write_text(json.dumps(record, indent=2) + "\n")
    _say(f"flow solved: {record}")
    return latest


def _area_average(name: str) -> np.ndarray:
    path = next((CASE / "postProcessing" / name).glob("*/surfaceFieldValue.dat"))
    return np.loadtxt(path, comments="#")


def flow_convergence() -> dict:
    """What the flow stopped on, and the integral check: the pressure drop's late drift."""
    log = (CASE / "log.foamRun").read_text().splitlines()
    stops = [line.strip() for line in log if "converged" in line.lower()]
    inlet, outlet = _area_average("inletPressure"), _area_average("outletPressure")
    drop = inlet[:, 1] - outlet[:, 1]
    tail = drop[-100:]
    return {
        "iterations": int(inlet[-1, 0]),
        # The solver's own words, rather than a guess at them: absent means it ran to endTime.
        "stop_message": stops[-1] if stops else None,
        "kinematic_pressure_drop_m2_s2": float(drop[-1]),
        "pressure_drop_relative_spread_last_100": float(np.ptp(tail) / abs(tail.mean())),
    }


# -- stage 3: the dose, one fluence rate at a time ----------------------------------------------


def dose_dictionary() -> None:
    """The tutorial's tracker settings, with the trajectory output decimated."""
    path = CASE / "system" / "postProcess.dict"
    text = path.read_text()
    if "trajectoryStride" in text:
        return
    new, count = re.subn(
        r"(output\s*\{)", rf"\1\n        trajectoryStride  {TRAJECTORY_STRIDE};", text
    )
    if count != 1:
        raise ValueError(f"expected one 'output' block in {path}, found {count}")
    path.write_text(new)


def zero_gradient_template(native: FieldTemplate) -> FieldTemplate:
    """The DOM file's class, units and patches, with every patch value taken from its cell."""
    return native._replace(
        internal_field="uniform 0",
        boundary=dict.fromkeys(native.boundary, "type            zeroGradient;"),
    )


def write_fluence_rate(name: str, destination: Path, mesh, template: FieldTemplate) -> None:
    spec = SOURCES[name]
    if spec["native"] is not None:
        shutil.copy(spec["native"], destination)
        return
    interior = spec["interior"]
    values = (
        np.load(interior)
        if interior.suffix == ".npy"
        else np.asarray(read_volume_scalar_field(interior, mesh))
    )
    write_openfoam_field(destination, values, template=template, object_name="G")


def run_dose(oor: Path, name: str, flow_time: Path, mesh, template: FieldTemplate) -> None:
    run = RUNS / name
    if (run / "doseDistribution.csv").exists():
        _say(f"{name}: dose already run")
        return
    write_fluence_rate(name, flow_time / "G", mesh, template)
    output = CASE / "postProcessing" / "radiationDose"
    if output.exists():
        shutil.rmtree(output)
    run.mkdir(parents=True, exist_ok=True)
    _say(f"{name}: tracking")
    started = time.perf_counter()
    _in_container(
        oor,
        "/work/dose/case",
        f"export OMP_NUM_THREADS={TRACKER_THREADS}; "
        f"foamPostProcess -dict system/postProcess.dict -time {flow_time.name}",
        run / "log.radiationDose",
    )
    elapsed = time.perf_counter() - started
    for item in (output / flow_time.name).iterdir():
        shutil.move(str(item), run / item.name)
    for item in CASE.glob("postProcessing/radiationDose/*"):
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    commit = subprocess.run(
        ["git", "-C", str(oor), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    record = {
        "of_optical_radiation_commit": commit,
        "fluence_rate": {k: (str(v) if v is not None else None) for k, v in SOURCES[name].items()},
        "patch_values": "solver's own" if SOURCES[name]["native"] else "zeroGradient",
        "flow_time": flow_time.name,
        "omp_threads": TRACKER_THREADS,
        "trajectory_stride": TRAJECTORY_STRIDE,
        "wall_seconds": round(elapsed, 1),
    }
    (run / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    _say(f"{name}: done in {elapsed:.0f} s")


# -- stage 4: the comparison --------------------------------------------------------------------


def read_tracks(name: str) -> dict[str, np.ndarray]:
    path = RUNS / name / "doseDistribution.csv"
    rows = [line.split(",") for line in path.read_text().splitlines() if not line.startswith("#")]
    columns = list(zip(*rows, strict=True))
    tracks = {
        "id": np.array(columns[0], dtype=int),
        "reason": np.array(columns[1]),
        "time": np.array(columns[2], dtype=float),
        "dose": np.array(columns[3], dtype=float),
        "end": np.array(columns[4:7], dtype=float).T,
    }
    order = np.argsort(tracks["id"])
    return {key: value[order] for key, value in tracks.items()}


def read_summary(name: str) -> dict[str, float]:
    values = {}
    for line in (RUNS / name / "summary.dat").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            values[parts[0]] = float(parts[-1])
    return values


def log_reduction(dose: np.ndarray, k: float) -> float:
    return float(-np.log10(np.mean(np.exp(-k * dose))))


def check_paired(tracks: dict[str, dict]) -> None:
    """Refuse to compare runs that did not follow the same particles."""
    first, *rest = tracks
    for name in rest:
        for key in ("id", "reason", "time", "end"):
            if not np.array_equal(tracks[first][key], tracks[name][key]):
                differing = int(np.count_nonzero(np.any(
                    np.atleast_2d(tracks[first][key].T != tracks[name][key].T), axis=0
                )))  # fmt: skip
                raise RuntimeError(
                    f"{name} and {first} disagree on '{key}' for {differing} particles: the runs "
                    f"did not follow the same trajectories, so a paired comparison is meaningless"
                )
    _say(f"paired: {len(tracks[first]['id'])} particles, identical ends in all {len(tracks)} runs")


def statistics(dose: np.ndarray) -> dict:
    quantiles = {f"p{q}": float(np.percentile(dose, q)) for q in (1, 5, 10, 50, 90, 95, 99)}
    return {
        "n": int(dose.size),
        "mean": float(dose.mean()),
        "stdev": float(dose.std(ddof=1)),
        "min": float(dose.min()),
        "max": float(dose.max()),
        **quantiles,
        "log_reduction": {str(k): log_reduction(dose, k) for k in K_INACT},
    }


def compare() -> dict:
    tracks = {name: read_tracks(name) for name in SOURCES}
    check_paired(tracks)
    escaped = tracks["aquaflux"]["reason"] == "escaped"
    summary = {
        "particles": int(escaped.size),
        "escaped": int(escaped.sum()),
        "end_reasons": {
            str(reason): int(count)
            for reason, count in zip(
                *np.unique(tracks["aquaflux"]["reason"], return_counts=True), strict=True
            )
        },
        "paper": PAPER,
        "runs": {},
        "paired_ratio_to_aquaflux": {},
    }
    for name in SOURCES:
        dose = tracks[name]["dose"][escaped]
        stats = statistics(dose)
        # The tracker's own summary is a second, independent computation of the same numbers.
        solver = read_summary(name)
        stats["tracker_summary"] = {
            "mean": solver["meanDose_mJcm2"],
            "log_reduction_k0.1": solver["logReduction_k=0.1"],
        }
        summary["runs"][name] = stats
        _say(
            f"{name:>22}: mean {stats['mean']:6.2f} (tracker {solver['meanDose_mJcm2']:6.2f}), "
            f"p5/p50/p95 {stats['p5']:6.1f} {stats['p50']:6.1f} {stats['p95']:6.1f}, "
            f"range {stats['min']:.1f}-{stats['max']:.1f}, "
            f"LR(k=0.1) {stats['log_reduction']['0.1']:.3f} "
            f"(tracker {solver['logReduction_k=0.1']:.3f})"
        )
    ours = tracks["aquaflux"]["dose"][escaped]
    for name in SOURCES:
        if name == "aquaflux":
            continue
        ratio = tracks[name]["dose"][escaped] / ours
        summary["paired_ratio_to_aquaflux"][name] = {
            f"p{q}": float(np.percentile(ratio, q)) for q in (1, 10, 50, 90, 99)
        }
        _say(f"{name} / aquaflux per particle: {summary['paired_ratio_to_aquaflux'][name]}")
    plots(tracks, escaped)
    return summary


def plots(tracks: dict, escaped: np.ndarray) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"aquaflux": "k", "dom64": "C1", "dom256": "C0", "dom256_native_patches": "C0"}
    labels = {
        "aquaflux": "aquaflux",
        "dom64": "DOM, 64 directions",
        "dom256": "DOM, 256 directions",
        "dom256_native_patches": "DOM 256, its own patch values",
    }
    doses = {name: tracks[name]["dose"][escaped] for name in tracks}
    top = max(d.max() for d in doses.values())
    bins = np.linspace(0.0, 1.02 * top, 80)

    fig, (hist, cdf) = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name in PAIRED:
        hist.hist(doses[name], bins=bins, histtype="step", color=styles[name], label=labels[name],
                  density=True, lw=1.4)  # fmt: skip
        values = np.sort(doses[name])
        cdf.plot(values, np.arange(1, values.size + 1) / values.size, color=styles[name],
                 label=labels[name])  # fmt: skip
    for ax in (hist, cdf):
        ax.axvspan(*PAPER["dose_range_mJ_cm2"], color="0.9", zorder=0, label="paper's range")
        ax.axvline(PAPER["mean_dose_mJ_cm2"], color="0.4", ls="--", lw=1, label="paper's mean")
        ax.set_xlabel("dose (mJ/cm²)")
    hist.set_ylabel("probability density")
    cdf.set_ylabel("fraction of particles at or below")
    hist.legend(fontsize=8)
    fig.suptitle(f"UV dose, {int(escaped.sum()):,} particles through the Sozzi reactor")
    fig.savefig(OUT / "dose_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ours = doses["aquaflux"]
    for name in ("dom64", "dom256"):
        axes[0].scatter(ours, doses[name] / ours, s=2, color=styles[name], alpha=0.4,
                        label=labels[name])  # fmt: skip
        axes[1].hist(doses[name] / ours, bins=120, histtype="step", color=styles[name],
                     label=labels[name])  # fmt: skip
    axes[1].hist(doses["dom256_native_patches"] / ours, bins=120, histtype="step",
                 color=styles["dom256"], ls=":", label=labels["dom256_native_patches"])  # fmt: skip
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set_xlabel("aquaflux dose (mJ/cm²)")
    axes[0].set_ylabel("DOM dose / aquaflux dose, same particle")
    axes[1].set_xlabel("DOM dose / aquaflux dose, same particle")
    axes[1].set_ylabel("particles")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(OUT / "dose_paired.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    for name in PAIRED:
        ax.plot(K_INACT, [log_reduction(doses[name], k) for k in K_INACT], "o-",
                color=styles[name], label=labels[name])  # fmt: skip
    ax.plot([0.1], [PAPER["log_reduction_k0.1"]], "s", color="0.4", label="paper, k = 0.1")
    ax.set_xscale("log")
    ax.set_xlabel("inactivation rate constant k (cm²/mJ)")
    ax.set_ylabel("log reduction")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "log_reduction.png", dpi=160)
    plt.close(fig)


def fluence_rate_fields(mesh) -> None:
    """The three fluence rates and how far each DOM field is from aquaflux's, for ParaView."""
    path = OUT / "sozzi_fluence_rates.vtu"
    if path.exists():
        return
    fields = {"G_aquaflux": np.load(SOURCES["aquaflux"]["interior"])}
    for name in ("dom64", "dom256"):
        fields[f"G_{name}"] = np.asarray(read_volume_scalar_field(SOURCES[name]["interior"], mesh))
        # Floored so a cell where the two agree to the last digit shows as far below the rest
        # rather than as -inf.
        fields[f"log10_abs_G_{name}_minus_aquaflux"] = np.log10(
            np.maximum(np.abs(fields[f"G_{name}"] - fields["G_aquaflux"]), 1e-12)
        )
    write_vtu(mesh, fields, path)
    _say(f"wrote {path.name}")


def main() -> None:
    oor = Path(os.environ["SOZZI_OOR_SOURCE"]).resolve()
    for spec in SOURCES.values():
        for path in spec.values():
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{path} is missing; see this script's docstring")
    compile_library(oor)
    prepare_case()
    flow_time = solve_flow(oor)
    dose_dictionary()

    _say("reading the mesh")
    mesh = read_openfoam(CASE)
    template = zero_gradient_template(read_field_template(SOURCES["dom256"]["interior"]))
    for name in SOURCES:
        run_dose(oor, name, flow_time, mesh, template)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"flow": json.loads((CASE / "flow.json").read_text()), **compare()}
    summary["configuration"] = {
        "omp_threads": TRACKER_THREADS,
        "trajectory_stride": TRAJECTORY_STRIDE,
        "flow_ranks": FLOW_RANKS,
        "lamp_exitance_W_per_m2": 696.42,
        "absorption_per_m": 35.67,
        "walls": "black",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fluence_rate_fields(mesh)
    _say(f"done: {OUT}")


if __name__ == "__main__":
    main()
    sys.exit(0)
