"""The radiation swap on the Sozzi reactor's wall-resolved uvmesh mesh: one flow, three fluence rates.

The counterpart of ``compare_fluence.py`` and ``dose_comparison.py`` for the uvmesh mesh (1,232,629
cells: structured lamp and wall layers joined non-conformally to a polyhedral bulk) and its
k-omega SST flow. The flow, the DOM solves and the dose tracking run in of-optical-radiation's
Docker image outside this script; this script computes aquaflux's fluence rate on the mesh, writes
each fluence rate as an OpenFOAM field for the tracker, and compares the doses.

Stages, run as ``python uvmesh_swap.py <stage>`` or
``SOZZI_SWAP_STAGE=<stage> validation/run_case.sh validation/sozzi_radiation/uvmesh_swap.py``:

``gather``
    aquaflux's ``G`` at every cell centre, kept as ``swap/G_aquaflux.npy``. The lamp is the mesh's
    own emitting patches, ``lamp0_wall`` (the 0.80 m cylinder) and ``lamp0_tip_B`` (the
    hemispherical tip) -- the surface the DOM case emits from -- cut into their centre-fan triangles
    and coarsened with the emitted power held (``LAMP_EDGE``, ``LAMP_CHORD``). Exitance
    696.42 W/m^2, absorption 35.67 /m, black walls. The fluid is the same three cylinders as the
    snapped mesh's, so a cell in a pipe sees a lamp point exactly when the segment leaves the
    chamber through that pipe's opening (``compare_fluence.BranchOpenings``); every cell of this
    mesh is inside one of the three, which the stage checks.
``write``
    One tracking case per fluence rate under ``swap/<name>``: the flow's time directory (linked)
    with ``G`` written in it. ``aquaflux``, ``dom72`` and ``dom288`` share the same patch treatment:
    the interior values, ``zeroGradient`` on every ordinary patch and the constraint type on the
    non-conformal coupling patches (the tracker interpolates ``G`` from cell and boundary values,
    and aquaflux computes cells only). ``dom288_native`` keeps the DOM file as the solver wrote it.
``compare``
    After the tracker has run in each case: refuses to compare runs whose particles did not follow
    the same paths (end reasons, times and points must agree), then writes the dose statistics and
    the log reduction over ``K_INACT``, with each field's volume-weighted mean and its ratio to
    aquaflux's cell by cell, to ``swap/summary.json``.

Paths: ``SOZZI_UVMESH_RUN`` (default ``~/aquaflux-runs/sozzi_sst_dose_2026-10-01``) holds
``mesh_uvmesh/polyMesh``, the flow at ``case/uv_arc/<FLOW_TIME>``, and the DOM fields
``case/uv_dom6_lu/<t>/G`` and ``case/uv_dom12_lu/<t>/G``, both linearUpwind rays and 1 x 1 pixels.
Their angular grids are uniform: of-optical-radiation's DOM divides the polar angle over the whole
sphere into ``nTheta`` bins of pi/nTheta and the azimuth into ``2 nPhi`` bins of pi/nPhi, so the bins
are square only when ``nPhi == nTheta``; n = 6 gives 72 directions and n = 12 gives 288 (Fluent's
3 x 3 and 6 x 6 theta x phi divisions per octant).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare_fluence  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh import patch_triangles  # noqa: E402
from aquaflux.radiation import Surfaces, coarsen_surfaces  # noqa: E402
from compare_fluence import (  # noqa: E402
    EXITANCE,
    X_BODY_END,
    _in_chamber,
    _in_inlet,
    _in_riser,
    gather,
)

RUN = Path(
    os.environ.get("SOZZI_UVMESH_RUN", Path.home() / "aquaflux-runs/sozzi_sst_dose_2026-10-01")
)
MESH = RUN / "mesh_uvmesh" / "polyMesh"
FLOW_CASE = RUN / "case" / "uv_arc"
FLOW_TIME = "2059"
DOM = {"dom72": RUN / "case" / "uv_dom6_lu", "dom288": RUN / "case" / "uv_dom12_lu"}
#: The tracker's dictionaries for every run: the tutorial's, with the Langevin dispersion model.
TRACKER_SYSTEM = RUN / "case" / "uv_dom_lu_lgv89" / "system"
OUT = RUN / "swap"

LAMP_PATCHES = ("lamp0_wall", "lamp0_tip_B")
#: Coarsening bounds for the lamp, in metres: longest edge along it, chord around it. On the snapped
#: mesh's lamp patch these moved G by 0.52% median near the lamp (< 5 mm) and 0.14% elsewhere from
#: the exact patch (this directory's README, "The lamp from the mesh's own patch").
LAMP_EDGE, LAMP_CHORD = 4e-3, 1e-4
#: Inactivation rate constants (cm^2/mJ) for the log reduction ``-log10 mean(exp(-k D))``.
K_INACT = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
SOURCES = ("aquaflux", "dom72", "dom288", "dom288_native")
#: Volumetric flow rate through the reactor, m^3/s: 25 US gallons a minute, the flow the case's
#: inlet velocity was set from.
FLOW_RATE = 25 * 3.785411784e-3 / 60
#: J/m^2 to mJ/cm^2.
TO_MJ_PER_CM2 = 0.1
#: Cells per gather chunk. compare_fluence's 20,000 holds a chunk x facets array per step; this
#: lamp has ~2.5x its facets, so the chunk is cut to keep the peak memory alike.
GATHER_CHUNK = 4_000
CONSTRAINT_TYPES = (
    "nonConformalCyclic",
    "nonConformalError",
    "cyclic",
    "empty",
    "symmetryPlane",
    "wedge",
)


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _latest_time(case: Path) -> Path:
    times = [p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.eE+-]+", p.name)]
    return max(times, key=lambda p: float(p.name))


# -- gather ---------------------------------------------------------------------------------------


def fluid_region(points: np.ndarray) -> np.ndarray:
    """0 in the chamber, 1 in the inlet pipe, 2 in the riser, for points known to be in the fluid."""
    return np.where(_in_chamber(points), 0, np.where(points[:, 0] > X_BODY_END, 1, 2))


def mesh_lamp(mesh, geometry) -> tuple[Surfaces, dict]:
    """The mesh's own lamp patches as aquaflux facets, coarsened with the emitted power held.

    Returns the facets and a record of how they were made (the lamp half of ``gather.json``).
    """
    triangles = patch_triangles(mesh, geometry, list(LAMP_PATCHES)).vertices
    exact = Surfaces.from_triangles(np.asarray(triangles), emission=EXITANCE)
    started = time.perf_counter()
    lamp, coarsening = coarsen_surfaces(exact, max_edge=LAMP_EDGE, chord=LAMP_CHORD)
    exact_power = float(np.sum(np.asarray(exact.area))) * EXITANCE
    power = float(np.sum(np.asarray(lamp.area) * np.asarray(lamp.emission)))
    seconds = time.perf_counter() - started
    _say(f"lamp: {len(triangles)} patch triangles ({exact_power:.4f} W) coarsened to "
         f"{lamp.n_facets} facets ({power:.4f} W) in {seconds:.0f} s")  # fmt: skip
    record = {
        "lamp_patches": list(LAMP_PATCHES),
        "patch_triangles": len(triangles),
        "lamp_facets": int(lamp.n_facets),
        "lamp_edge_m": LAMP_EDGE,
        "lamp_chord_m": LAMP_CHORD,
        "exact_patch_power_W": exact_power,
        "coarsened_power_W": power,
        "longest_edge_max_m": float(coarsening.longest_edge.max()),
        "coarsening_seconds": round(seconds, 1),
        "exitance_W_per_m2": EXITANCE,
        "absorption_per_m": 35.67,
        "walls": "black",
        "visibility": "compare_fluence.BranchOpenings in the pipes, none needed in the chamber",
    }
    return lamp, record


def stage_gather() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _say("reading the mesh")
    mesh = read_openfoam(MESH)
    geometry = mesh.geometry()
    points = np.asarray(geometry.cell.centroid)
    region = fluid_region(points)
    inside = np.asarray(_in_chamber(jnp.asarray(points)) | _in_inlet(jnp.asarray(points))
                        | _in_riser(jnp.asarray(points)))  # fmt: skip
    if not inside.all():
        raise RuntimeError(f"{int((~inside).sum())} cells lie outside the three cylinders")
    _say(f"{len(points)} cells: chamber {int((region == 0).sum())}, inlet "
         f"{int((region == 1).sum())}, riser {int((region == 2).sum())}")  # fmt: skip

    lamp, lamp_record = mesh_lamp(mesh, geometry)
    started = time.perf_counter()
    compare_fluence.CHUNK = GATHER_CHUNK
    field = gather(lamp, points, region, "aquaflux")
    seconds = time.perf_counter() - started
    np.save(OUT / "G_aquaflux.npy", field)
    (OUT / "gather.json").write_text(json.dumps({
        "cells": len(points),
        **lamp_record,
        "gather_chunk_cells": GATHER_CHUNK,
        "gather_seconds": round(seconds, 1),
        "jax": __import__("jax").__version__,
    }, indent=2) + "\n")  # fmt: skip
    _say(
        f"aquaflux G gathered in {seconds:.0f} s: min {field.min():.3g}, max {field.max():.4g} W/m^2"
    )


# -- write ----------------------------------------------------------------------------------------


def _binary_or_ascii_internal(path: Path, n: int) -> np.ndarray:
    """The internal field of a scalar OpenFOAM file, written in ASCII or binary."""
    raw = path.read_bytes()
    start = raw.index(b"internalField")
    match = re.search(rb"nonuniform\s+List<scalar>\s*(\d+)\s*\(", raw[start:])
    count = int(match.group(1))
    if count != n:
        raise ValueError(f"{path} holds {count} values, the mesh {n} cells")
    begin = start + match.end()
    if b"format      binary" in raw[:2000] or b"format binary" in raw[:2000]:
        return np.frombuffer(raw[begin : begin + 8 * count], dtype="<f8").copy()
    return np.array(raw[begin : begin + 30 * count].split(maxsplit=count)[:count], dtype=float)


def _patches() -> list[tuple[str, str]]:
    """(name, type) of every boundary patch, in the mesh's order."""
    text = (MESH / "boundary").read_text()
    body = text[text.index("(", text.index("FoamFile")) + 1 :]
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"\n\s*([A-Za-z0-9_]+)\s*\{[^}]*?\btype\s+(\w+);", body)]  # fmt: skip


def write_field(path: Path, values: np.ndarray) -> None:
    """``G`` in ASCII: interior values, zeroGradient patches, constraint types on the couplings."""
    lines = [
        "FoamFile\n{\n    format      ascii;\n    class       volScalarField;\n"
        '    location    "' + FLOW_TIME + '";\n    object      G;\n}\n\n',
        "dimensions      [1 0 -3 0 0 0 0];\n\n",
        f"internalField   nonuniform List<scalar>\n{len(values)}\n(\n",
        "\n".join(f"{v:.10g}" for v in values),
        "\n)\n;\n\nboundaryField\n{\n",
    ]
    for name, kind in _patches():
        entry = kind if kind in CONSTRAINT_TYPES else "zeroGradient"
        lines.append(f"    {name}\n    {{\n        type            {entry};\n    }}\n")
    lines.append("}\n")
    path.write_text("".join(lines))


def tracking_case(
    case: Path, values: np.ndarray | None = None, copy_from: Path | None = None
) -> None:
    """A case the tracker can run in: the flow's time directory (linked) with ``G`` in it.

    ``G`` is either ``values`` at the cells, written by :func:`write_field`, or a copy of
    ``copy_from`` as it stands. The tracker's dictionaries are ``TRACKER_SYSTEM``'s.
    """
    if (values is None) == (copy_from is None):
        raise ValueError("give exactly one of values and copy_from")
    if case.exists():
        shutil.rmtree(case)
    flow = FLOW_CASE / FLOW_TIME
    (case / FLOW_TIME).mkdir(parents=True)
    shutil.copytree(TRACKER_SYSTEM, case / "system")
    # Relative links, so the case also resolves inside a container that mounts RUN elsewhere.
    (case / "constant").symlink_to(os.path.relpath(FLOW_CASE / "constant", case))
    for item in flow.iterdir():
        if item.name != "G":
            (case / FLOW_TIME / item.name).symlink_to(os.path.relpath(item, case / FLOW_TIME))
    (case / "out.foam").touch()
    if values is None:
        shutil.copy(copy_from, case / FLOW_TIME / "G")
    else:
        write_field(case / FLOW_TIME / "G", values)


def stage_write() -> None:
    dictionary = (TRACKER_SYSTEM / "postProcess.dict").read_text()
    if not re.search(r"type\s+langevin;", dictionary):
        raise RuntimeError(f"{TRACKER_SYSTEM} does not select the langevin dispersion model")
    n = len(np.load(OUT / "G_aquaflux.npy"))
    fields = {"aquaflux": np.load(OUT / "G_aquaflux.npy")}
    for name, case in DOM.items():
        fields[name] = _binary_or_ascii_internal(_latest_time(case) / "G", n)
    record = {}
    for name in SOURCES:
        case = OUT / name
        if name == "dom288_native":
            source = _latest_time(DOM["dom288"]) / "G"
            tracking_case(case, copy_from=source)
            record[name] = {"G": str(source), "patch_values": "the solver's own"}
        else:
            tracking_case(case, values=fields[name])
            source = OUT / "G_aquaflux.npy" if name == "aquaflux" else _latest_time(DOM[name]) / "G"
            record[name] = {
                "G": str(source),
                "patch_values": "zeroGradient (coupling patches: their constraint types)",
            }
        _say(f"{name}: case written at {case}")
    (OUT / "write.json").write_text(json.dumps(record, indent=2) + "\n")


# -- compare --------------------------------------------------------------------------------------


def tracks(case: Path) -> dict[str, np.ndarray]:
    """The tracker's particles in ``case``, sorted by id: end reason, time, dose and end point."""
    path = next((case / "postProcessing" / "radiationDose").glob("*/doseDistribution.csv"))
    rows = [line.split(",") for line in path.read_text().splitlines() if not line.startswith("#")]
    columns = list(zip(*rows, strict=True))
    run = {
        "id": np.array(columns[0], dtype=int),
        "reason": np.array(columns[1]),
        "time": np.array(columns[2], dtype=float),
        "dose": np.array(columns[3], dtype=float),
        "end": np.array(columns[4:7], dtype=float).T,
    }
    order = np.argsort(run["id"])
    return {key: value[order] for key, value in run.items()}


def log_reduction(dose: np.ndarray, k: float) -> float:
    return float(-np.log10(np.mean(np.exp(-k * dose))))


def field_statistics() -> dict:
    """Each field's volume-weighted mean by region, its mean dose, and each DO field over aquaflux's.

    The same two measures as ``compare_fluence``'s summary: the mean over all cells, the chamber,
    the inlet pipe and the riser; and percentiles of the cell-by-cell ratio over the cells where
    aquaflux's G exceeds 1e-3 of its peak. And the mean dose the field gives fluid that passes through
    the reactor, ``integral(G dV) / Q`` with G below zero counted as zero, as the tracker counts it: in
    a steady flow the dose averaged over the outflow is the volume integral of G over the flow rate,
    whatever the flow's pattern, so a tracker whose particles sample the fluid correctly must return
    this mean.
    """
    _say("reading the mesh for the field statistics")
    geometry = read_openfoam(MESH).geometry()
    points = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)
    region = np.where(_in_chamber(points), 0, np.where(points[:, 0] > X_BODY_END, 1, 2))
    fields = {"aquaflux": np.load(OUT / "G_aquaflux.npy")}
    for name, case in DOM.items():
        fields[name] = _binary_or_ascii_internal(_latest_time(case) / "G", len(points))
    means = {}
    for label, mask in (("all", np.ones(len(points), bool)), ("chamber", region == 0),
                        ("inlet", region == 1), ("riser", region == 2)):  # fmt: skip
        weights = volume[mask]
        means[label] = {name: float(np.sum(f[mask] * weights) / np.sum(weights))
                        for name, f in fields.items()}  # fmt: skip
        _say(f"volume-weighted mean G, {label}: {means[label]}")
    lit = fields["aquaflux"] > 1e-3 * fields["aquaflux"].max()
    ratio = {}
    for name in DOM:
        values = fields[name][lit] / fields["aquaflux"][lit]
        ratio[name] = {f"p{q}": float(np.percentile(values, q)) for q in (1, 10, 50, 90, 99)}
        _say(f"{name} / aquaflux over {int(lit.sum())} lit cells: {ratio[name]}")
    negative = {name: float(np.mean(fields[name] < 0)) for name in DOM}
    mean_dose = {name: float(np.sum(np.maximum(f, 0.0) * volume)) / FLOW_RATE * TO_MJ_PER_CM2
                 for name, f in fields.items()}  # fmt: skip
    _say(f"integral(G dV) / Q [mJ/cm^2], volume {volume.sum():.6g} m^3: {mean_dose}")
    return {"volume_mean": means, "ratio_over_lit_cells": ratio, "lit_cells": int(lit.sum()),
            "negative_cell_fraction": negative, "volume_m3": float(volume.sum()),
            "flow_rate_m3_per_s": FLOW_RATE, "mean_dose_from_integral": mean_dose}  # fmt: skip


def stage_compare() -> None:
    by_source = {name: tracks(OUT / name) for name in SOURCES}
    first = by_source[SOURCES[0]]
    for name in SOURCES[1:]:
        for key in ("id", "reason", "time", "end"):
            if not np.array_equal(first[key], by_source[name][key]):
                raise RuntimeError(f"{name} and {SOURCES[0]} disagree on '{key}': not paired")
    escaped = first["reason"] == "escaped"
    _say(
        f"paired: {escaped.size} particles, {int(escaped.sum())} escaped, identical ends in all runs"
    )
    runs = {}
    for name in SOURCES:
        dose = by_source[name]["dose"][escaped]
        runs[name] = {
            "n": int(dose.size),
            "mean": float(dose.mean()),
            "min": float(dose.min()),
            "max": float(dose.max()),
            **{f"p{q}": float(np.percentile(dose, q)) for q in (1, 5, 50, 95, 99)},
            "log_reduction": {f"{k:g}": log_reduction(dose, k) for k in K_INACT},
        }
        _say(f"{name:>14}: mean {runs[name]['mean']:.2f}, LR " + ", ".join(
            f"k={k}: {v:.3f}" for k, v in runs[name]["log_reduction"].items()))  # fmt: skip
    ours = by_source["aquaflux"]["dose"][escaped]
    paired = {name: {f"p{q}": float(np.percentile(by_source[name]["dose"][escaped] / ours, q))
                     for q in (1, 10, 50, 90, 99)} for name in SOURCES if name != "aquaflux"}  # fmt: skip
    summary = {
        "particles": int(escaped.size),
        "escaped": int(escaped.sum()),
        "runs": runs,
        "paired_ratio_to_aquaflux": paired,
        "fields_compared": field_statistics(),
        "gather": json.loads((OUT / "gather.json").read_text()),
        "fields": json.loads((OUT / "write.json").read_text()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _say(f"wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    stages = {"gather": stage_gather, "write": stage_write, "compare": stage_compare}
    # The stage comes from the command line, or from SOZZI_SWAP_STAGE when launched through
    # validation/run_case.sh, which passes no arguments to the script.
    stage = sys.argv[1] if len(sys.argv) == 2 else os.environ.get("SOZZI_SWAP_STAGE")
    if stage not in stages:
        sys.exit(f"usage: {Path(__file__).name} {{{'|'.join(stages)}}} (or SOZZI_SWAP_STAGE)")
    stages[stage]()
    sys.exit(0)
