"""Mesh the ray-effects room and run the discrete-ordinates solver on it.

**The case.** A 4 x 4 x 3 m room of air at 222 nm -- transparent and non-scattering, so nothing
masks the angular discretization -- lit by an Ushio Care222 B1 far-UVC module on the ceiling
centre, facing down, with a perforated "Voronoi bunny" hanging with its base at 1.2 m. Every
surface is black. The discrete-ordinates method (DOM) of `of-optical-radiation
<https://github.com/DeGrootResearchGroup/of-optical-radiation>`_ carries radiance along a finite
set of directions, so a small bright source in a large clear room is its worst case: energy
leaves the lamp in discrete beams (the ray effect) and each beam is smeared sideways by the
upwind transport (false scattering). This script produces the DOM fields that
``aquaflux_floor.py``, ``aquaflux_volume.py`` and ``reference.py`` are compared against.

**The lamp** is the ``iesEmitter`` boundary condition on a patch of ceiling faces, fed the
OSLUV-measured photometry of the module (``ushio_b1.ies``, see ``ushio_b1.LICENSE``) and
normalized to the file's own integrated radiant flux. The patch is cut from the 3.125 mm faces
of the refined ceiling: 20 x 14 faces, 6.25 x 4.375 cm, the 6.25 cm side along x, which is the
fixture's h = 0 direction (``fixtureUp``). The module's measured opening is 6 x 4.5 cm.

**Two meshes**, built from one template (``case/``): the room with the bunny, and the empty room
-- the same dictionaries with the bunny taken out of ``snappyHexMeshDict``, so every refinement
box is kept and the two differ only near the bunny.

Stages, each skipped when its output already exists:

1. compile of-optical-radiation into a persistent directory, so a container can be discarded;
2. for each mesh: place the bunny (``surfaceCheck``, recentre, ``surfaceTransformPoints``), then
   ``blockMesh``, ``snappyHexMesh``, ``createPatch`` (the lamp) and ``checkMesh``, in ASCII,
   because aquaflux's OpenFOAM reader refuses binary;
3. for each requested DOM resolution: decompose, run ``opticalRadiationFoam`` in parallel,
   reconstruct, and keep ``G``, ``qin`` (the incident flux on every patch), the log and a
   ``record.json`` under ``work/runs/<mesh>_nphi<P>_ntheta<T>[_px<p>x<t>]/``.

Environment:

``RAY_OOR_SOURCE``
    A checkout of of-optical-radiation with the ``incidentFluxPatches`` output (required). Its
    commit is recorded beside every run.
``RAY_OOR_IMAGE``
    The Docker image; default ``oor:local``, built from that checkout's ``Dockerfile``.
``RAY_BUNNY_STL``
    The bunny surface (required for the bunny mesh): ``voronoi_bunny_open.stl``, made by
    ``voronoi_bunny.py --cell 0.065 --strut 0.008 --shell 0.008 --pitch 0.0025``.
``RAY_NPROCS``
    MPI ranks for the DOM runs; default 8.

``RAY_STAGES``
    What to do, space-separated: ``mesh`` builds both meshes; a run is
    ``<mesh>:<nPhi>x<nTheta>[/<nPixelPhi>x<nPixelTheta>]`` with ``<mesh>`` one of ``bunny`` and
    ``empty`` (e.g. ``bunny:6x6 empty:6x6/3x3``; ``nPhi == nTheta`` gives square bins), and builds its mesh first if needed;
    ``<mesh>+reflecting:<P>x<T>`` runs the room with its floor, ceiling and walls diffusely
    reflecting (``room.WALL_REFLECTANCE``) and the bunny black, into
    ``runs/<mesh>_reflecting_nphi<P>_ntheta<T>/``. Default ``mesh``.

    **Pixels** are the Murthy-Mathur subdivision of each direction's bin, used to split the flux
    of a bin that a face's plane cuts through (an "overhanging" bin). Without them (1 x 1, the
    default) the whole bin goes one way, which on the cut cells of a snapped mesh sends it through
    the wrong face. A run with other than 1 x 1 pixels carries ``_px<p>x<t>`` in its name, so a
    1 x 1 run keeps its name and its result.

Run with ``RAY_OOR_SOURCE=<checkout> RAY_BUNNY_STL=<stl> RAY_STAGES=<stages>
validation/run_case.sh validation/ray_effects_room/generate_dom.py``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import room  # noqa: E402

WORK = HERE / "work"
MESHES = ("bunny", "empty")


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# The container running now, so a signal to this process can stop it: killing the ``docker run``
# client does not stop the container, which would otherwise run on unseen by ``run_case.sh``.
_RUNNING: list[str] = []


def _stop_container(signum, frame) -> None:
    del frame
    for name in _RUNNING:
        subprocess.run(["docker", "stop", "-t", "5", name], capture_output=True, check=False)
    raise SystemExit(128 + signum)


def _in_container(source: Path, workdir: str, command: str, log: Path) -> None:
    """Run ``command`` under OpenFOAM's environment with the checkout, the compiled library and
    the work tree mounted, writing its output to ``log``. Raises if it fails."""
    (WORK / "foamuser").mkdir(parents=True, exist_ok=True)
    name = f"ray-effects-{os.getpid()}-{time.monotonic_ns()}"
    docker = [
        "docker", "run", "--rm", "--name", name, "-e", "USER=root",
        "-v", f"{source}:/code",
        "-v", f"{WORK / 'foamuser'}:/root/OpenFOAM",
        "-v", f"{WORK}:/work",
        "-w", workdir,
        os.environ.get("RAY_OOR_IMAGE", "oor:local"),
        "bash", "-c", f"source /opt/openfoam13/etc/bashrc; set -e; {command}",
    ]  # fmt: skip
    started = time.perf_counter()
    _RUNNING.append(name)
    try:
        with log.open("w") as out:
            result = subprocess.run(docker, stdout=out, stderr=subprocess.STDOUT, check=False)
    finally:
        _RUNNING.remove(name)
    if result.returncode:
        msg = f"container command failed ({result.returncode}); see {log}"
        raise RuntimeError(msg)
    _say(f"  done in {time.perf_counter() - started:.0f} s ({log.name})")


def _set_entry(path: Path, key: str, value: str) -> None:
    """Replace a ``key value;`` entry in an OpenFOAM dictionary (exactly one must exist)."""
    text = path.read_text()
    new, count = re.subn(rf"(?m)^(\s*{key}\s+)[^;]+;", rf"\g<1>{value};", text)
    if count != 1:
        msg = f"expected one '{key}' entry in {path}, found {count}"
        raise ValueError(msg)
    path.write_text(new)


def _commit(source: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def compile_library(source: Path) -> None:
    stamp = WORK / "foamuser" / "commit"
    built = list((WORK / "foamuser").glob("*/platforms/*/bin/opticalRadiationFoam"))
    if built and stamp.exists() and stamp.read_text().strip() == _commit(source):
        _say("library already compiled at this commit")
        return
    _say(f"compiling of-optical-radiation {_commit(source)[:7]}")
    _in_container(source, "/code", "./Allwmake -j", WORK / "log.Allwmake")
    stamp.write_text(_commit(source) + "\n")


def _without_bunny(snappy: Path) -> None:
    """Remove the bunny from a snappyHexMeshDict: its geometry entry and its surface refinement.

    Each is a sub-dictionary named ``bunny``; it is cut from its name to its matching closing
    brace. Exactly two must exist, so a template edit that renames or adds one fails here.
    """
    text = snappy.read_text()
    starts = [m.start() for m in re.finditer(r"(?m)^[ \t]*bunny[ \t]*\n[ \t]*\{", text)]
    if len(starts) != 2:
        msg = f"expected two 'bunny' entries in {snappy}, found {len(starts)}"
        raise ValueError(msg)
    for start in reversed(starts):
        depth, i = 0, text.index("{", start)
        while True:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                break
            i += 1
        text = text[:start] + text[i + 1 :].lstrip("\n")
    snappy.write_text(text)


def place_bunny(source: Path, case: Path) -> dict:
    """Recentre the bunny's bounding box on x = y = 0 and stand its base on z = BUNNY_BASE."""
    stl = Path(os.environ["RAY_BUNNY_STL"]).resolve()
    tri = case / "constant" / "triSurface"
    shutil.copy(stl, tri / "bunny_raw.stl")
    _in_container(source, f"/work/{case.relative_to(WORK)}",
                  "surfaceCheck constant/triSurface/bunny_raw.stl", case / "log.surfaceCheck")  # fmt: skip
    low, high = room.stl_bounds(tri / "bunny_raw.stl")
    shift = (-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, room.BUNNY_BASE - low[2])
    move = f"translate=({shift[0]:.9f} {shift[1]:.9f} {shift[2]:.9f})"
    _in_container(
        source, f"/work/{case.relative_to(WORK)}",
        f'surfaceTransformPoints "{move}" constant/triSurface/bunny_raw.stl '
        "constant/triSurface/bunny.stl",
        case / "log.surfaceTransformPoints",
    )  # fmt: skip
    (tri / "bunny_raw.stl").unlink()
    low, high = room.stl_bounds(tri / "bunny.stl")
    _say(f"  bunny placed: {low.round(4)} to {high.round(4)}")
    return {"source_stl": str(stl), "translate": shift, "bounds": [low.tolist(), high.tolist()]}


MESH_STEPS = ("blockMesh", "snappyHexMesh", "createPatch", "checkMesh -allTopology -allGeometry")


def mesh(source: Path, name: str) -> Path:
    """Build one mesh, resuming after the last step that completed.

    Each step leaves a marker when it succeeds, because a mesh directory is not evidence of a
    finished mesh: ``snappyHexMesh`` writes ``polyMesh`` before the lamp patch is cut from it.
    """
    case = WORK / name / "case"
    record_path = case / "mesh_record.json"
    if record_path.exists():
        _say(f"{name}: already meshed")
        return case
    done = lambda step: case / f".done.{step.split()[0]}"  # noqa: E731
    if not done(MESH_STEPS[0]).exists():
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(HERE / "case", case)
        record = {"bunny": place_bunny(source, case)} if name == "bunny" else {}
        if name != "bunny":
            _without_bunny(case / "system" / "snappyHexMeshDict")
        (case / "placement.json").write_text(json.dumps(record, indent=2) + "\n")
    record = json.loads((case / "placement.json").read_text())
    _say(f"{name}: meshing")
    for step in MESH_STEPS:
        if done(step).exists():
            _say(f"  {step.split()[0]}: done earlier")
            continue
        _in_container(source, f"/work/{name}/case", step, case / f"log.{step.split()[0]}")
        done(step).touch()
    record["check_mesh"] = room.check_mesh_summary(case / "log.checkMesh")
    record["of_optical_radiation_commit"] = _commit(source)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    _say(f"{name}: {record['check_mesh']}")
    return case


def write_cells(source: Path, name: str) -> None:
    """Cell centres and volumes of a mesh, from OpenFOAM, as ``work/<mesh>/cells.npz``.

    OpenFOAM computes them itself (``writeCellCentres``, ``writeCellVolumes``), so aquaflux's
    fluence rate is gathered at exactly the points the discrete-ordinates ``G`` belongs to. The
    fields it writes into ``0/`` are removed afterwards, since every run copies ``0/``.
    """
    target = WORK / name / "cells.npz"
    if target.exists():
        return
    case = WORK / name / "case"
    _in_container(
        source,
        f"/work/{name}/case",
        "foamPostProcess -func writeCellCentres -time 0 && foamPostProcess -func writeCellVolumes -time 0",
        case / "log.writeCells",
    )
    from aquaflux.io.openfoam.foamfile import read_foam_body
    from aquaflux.io.openfoam.grammar import list_envelope, parse_vector_list

    def internal(field: str) -> str:
        body = read_foam_body(case / "0" / field)
        start = body.index("internalField")
        return body[start : body.index("boundaryField", start)]

    centres = parse_vector_list(internal("C").split(">", 1)[1])  # after "List<vector>"
    _, values = list_envelope(internal("Vc").split(">", 1)[1])  # after "List<scalar>"
    volumes = np.array(values.split(), dtype=float)
    if len(centres) != len(volumes):
        msg = f"{name}: {len(centres)} centres but {len(volumes)} volumes"
        raise ValueError(msg)
    np.savez(target, centre=centres, volume=volumes)
    for field in ("C", "Ccx", "Ccy", "Ccz", "Vc"):
        (case / "0" / field).unlink(missing_ok=True)
    _say(f"{name}: {len(volumes)} cell centres and volumes written")


def _estimate_memory_gb(cells: int, faces: int, rays: int) -> float:
    """What the DOM's per-ray fields hold, in GB.

    Each ray keeps one radiance field, a value per cell and per boundary face; with nothing
    scattering, of-optical-radiation allocates no per-ray snapshot beside it. Everything else the
    solver holds is small beside this at the resolutions used here.
    """
    return rays * (cells + faces) * 8 / 1e9


BLACK = """        type            reflective;
        diffuseFraction 0;
        reflectionCoef  0;"""


def _reflecting(field: Path) -> None:
    """Make the room's own surfaces diffuse reflectors in a copy of ``0/I``; the bunny stays black.

    The template gives every black surface one entry, matched by a regex over the patch names;
    it is split into a black ``bunny`` and a reflecting ``(floor|ceiling|walls)``.
    """
    text = field.read_text()
    black_group = '"(floor|ceiling|walls|bunny)"'
    if text.count(black_group) != 1 or text.count(BLACK) != 1:
        msg = f"{field} is not the template this edit expects"
        raise ValueError(msg)
    start = text.index(black_group)
    end = text.index("}", start) + 1
    block = text[start:end]
    reflecting = block.replace(
        black_group, '"(' + "|".join(room.REFLECTING_PATCHES) + ')"'
    ).replace(
        BLACK,
        f"""        type            reflective;
        diffuseFraction 1;
        reflectionCoef  {room.WALL_REFLECTANCE};""",
    )
    field.write_text(
        text[:start] + block.replace(black_group, "bunny") + "\n    " + reflecting + text[end:]
    )


def run_dom(
    source: Path,
    name: str,
    n_phi: int,
    n_theta: int,
    *,
    pixels: tuple[int, int] = (1, 1),
    reflecting: bool = False,
) -> None:
    run_name = f"{name}{'_reflecting' if reflecting else ''}_nphi{n_phi}_ntheta{n_theta}"
    if pixels != (1, 1):
        run_name += f"_px{pixels[0]}x{pixels[1]}"
    run = WORK / "runs" / run_name
    if (run / "record.json").exists():
        _say(f"{run_name}: already run")
        return
    case = mesh(source, name)
    mesh_record = json.loads((case / "mesh_record.json").read_text())
    rays = 2 * n_phi * n_theta
    cells = mesh_record["check_mesh"]["cells"]
    boundary = mesh_record["check_mesh"]["faces"] - _internal_faces(case)
    estimate = _estimate_memory_gb(cells, boundary, rays)
    _say(f"{run_name}: {rays} directions on {cells} cells, ~{estimate:.1f} GB of ray fields")

    staging = WORK / "staging" / run_name
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "constant").mkdir(parents=True)
    shutil.copytree(case / "0", staging / "0")
    shutil.copytree(case / "system", staging / "system")
    shutil.copy(case / "constant" / "opticalRadiationProperties", staging / "constant")
    # The mesh is ~0.5 GB of ASCII; a relative link resolves inside the container too.
    (staging / "constant" / "polyMesh").symlink_to(
        os.path.relpath(case / "constant" / "polyMesh", staging / "constant")
    )
    shutil.copy(room.IES_FILE, staging / "constant" / "lamp.ies")
    power = room.lamp_power()
    text = (staging / "0" / "I").read_text()
    if text.count("LAMP_POWER_W") != 1:
        msg = "0/I must hold exactly one LAMP_POWER_W placeholder"
        raise ValueError(msg)
    (staging / "0" / "I").write_text(text.replace("LAMP_POWER_W", f"{power:.12g}"))
    if reflecting:
        _reflecting(staging / "0" / "I")
    properties = staging / "constant" / "opticalRadiationProperties"
    _set_entry(properties, "nPhi", str(n_phi))
    _set_entry(properties, "nTheta", str(n_theta))
    _set_entry(properties, "nPixelPhi", str(pixels[0]))
    _set_entry(properties, "nPixelTheta", str(pixels[1]))
    ranks = int(os.environ.get("RAY_NPROCS", "8"))
    _set_entry(staging / "system" / "decomposeParDict", "numberOfSubdomains", str(ranks))

    where = f"/work/staging/{run_name}"
    mpi = "export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;"
    _in_container(source, where, "decomposePar -force", staging / "log.decomposePar")
    started = time.perf_counter()
    _in_container(
        source,
        where,
        f"{mpi} mpirun -np {ranks} opticalRadiationFoam -parallel",
        staging / "log.opticalRadiationFoam",
    )
    elapsed = time.perf_counter() - started
    _in_container(source, where, "reconstructPar -latestTime", staging / "log.reconstructPar")

    written = max(
        (p for p in staging.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.eE+-]+", p.name)),
        key=lambda p: float(p.name),
    )
    if written.name == "0":
        msg = f"{run_name}: no time directory was written"
        raise RuntimeError(msg)
    run.mkdir(parents=True, exist_ok=True)
    for field in ("G", "qin"):
        shutil.copy(written / field, run / field)
    for log in ("log.opticalRadiationFoam", "log.decomposePar", "log.reconstructPar"):
        shutil.copy(staging / log, run / log)
    solver_log = (staging / "log.opticalRadiationFoam").read_text()
    iterations = len(re.findall(r"Solving for I_0_0,", solver_log))
    residuals = [float(r) for r in re.findall(r"Initial residual = ([0-9.eE+-]+)", solver_log)]
    per_sweep = len(residuals) // max(iterations, 1)
    last_sweep = residuals[-per_sweep:] if per_sweep else []
    record = {
        "mesh": name,
        "wall_reflectance": room.WALL_REFLECTANCE if reflecting else 0.0,
        "cells": cells,
        "n_phi": n_phi,
        "n_theta": n_theta,
        "directions": rays,
        "n_pixel_phi": int(_entry(properties, "nPixelPhi")),
        "n_pixel_theta": int(_entry(properties, "nPixelTheta")),
        "lamp_power_W": power,
        "lamp_ies": room.IES_FILE.name,
        "of_optical_radiation_commit": _commit(source),
        "mpi_ranks": ranks,
        "docker_cpus": _docker_info("{{.NCPU}}"),
        "docker_memory_bytes": _docker_info("{{.MemTotal}}"),
        "host": platform.platform(),
        "outer_iterations": iterations,
        "max_initial_residual_last_sweep": max(last_sweep) if last_sweep else None,
        "convergence": _entry(staging / "constant" / "opticalRadiationProperties", "convergence"),
        "max_iter": _entry(staging / "constant" / "opticalRadiationProperties", "maxIter"),
        "wall_seconds": round(elapsed, 1),
        "ray_field_memory_estimate_GB": round(estimate, 2),
        "written_at_time": written.name,
    }
    (run / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(staging)
    _say(f"{run_name}: done, {elapsed:.0f} s, {iterations} outer iterations")


def _entry(path: Path, key: str) -> str:
    return re.search(rf"(?m)^\s*{key}\s+([^;]+);", path.read_text()).group(1).strip()


def _internal_faces(case: Path) -> int:
    match = re.search(r"internal faces:\s+(\d+)", (case / "log.checkMesh").read_text())
    return int(match.group(1))


def _docker_info(template: str) -> str:
    return subprocess.run(
        ["docker", "info", "--format", template], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    signal.signal(signal.SIGTERM, _stop_container)
    signal.signal(signal.SIGHUP, _stop_container)
    arguments = os.environ.get("RAY_STAGES", "mesh").split()
    source = Path(os.environ["RAY_OOR_SOURCE"]).resolve()
    WORK.mkdir(exist_ok=True)
    compile_library(source)
    runs = [a for a in arguments if a != "mesh"]
    wanted = (
        MESHES if "mesh" in arguments else sorted({r.split(":")[0].split("+")[0] for r in runs})
    )
    for name in wanted:
        if name not in MESHES:
            msg = f"unknown mesh {name!r}; expected one of {MESHES}"
            raise ValueError(msg)
        mesh(source, name)
        write_cells(source, name)
    for run in runs:
        name, _, resolution = run.partition(":")
        name, _, variant = name.partition("+")
        if variant not in ("", "reflecting"):
            msg = f"unknown variant {variant!r} in {run!r}; the one variant is 'reflecting'"
            raise ValueError(msg)
        resolution, _, pixel_text = resolution.partition("/")
        n_phi, n_theta = (int(v) for v in resolution.split("x"))
        pixels = tuple(int(v) for v in pixel_text.split("x")) if pixel_text else (1, 1)
        if len(pixels) != 2 or min(pixels) < 1:
            msg = f"pixels in {run!r} must be <nPixelPhi>x<nPixelTheta>, each at least 1"
            raise ValueError(msg)
        run_dom(source, name, n_phi, n_theta, pixels=pixels, reflecting=variant == "reflecting")
    _say("done")


if __name__ == "__main__":
    main()
