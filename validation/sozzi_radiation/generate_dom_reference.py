"""Generate the discrete-ordinates fluence-rate reference for the Sozzi & Taghipour reactor.

The comparison this feeds checks aquaflux's fluence rate ``G`` against the finite-volume
discrete-ordinates method (DOM) of `of-optical-radiation
<https://github.com/DeGrootResearchGroup/of-optical-radiation>`_, on that project's
``uvReactorSozzi2006-DOM`` tutorial: a 35 W lamp as a diffuse emitter (exitance 696.42 W/m^2 over
its cylinder, hemispherical tip and base disc), water absorbing at 35.67 /m (70% transmittance
per cm), every other wall black, no scattering. Every one of those is physics aquaflux models, so
the fields are comparable cell for cell on the same mesh.

**DOM is a reference, not the truth.** Its two irreducible errors -- the ray effect and false
scattering -- are worst for a small bright source in a large, weakly absorbing domain, which is
this case. So this runs the solve at several angular resolutions (``nPhi x nTheta``, giving
``2 nPhi nTheta`` directions over the sphere): the spread between them is the DOM's own error,
and it has to be known before any tolerance on the comparison is written.

**No flow solve.** ``opticalRadiationFoam`` reads only the radiance field, so with a constant
extinction model the DOM ``G`` does not depend on the flow; the tutorial's ~45 min RANS solve is
needed only for the dose post-process, not here.

Three stages, each skipped when its output already exists:

1. compile the library into a persistent directory, so the container can be thrown away;
2. mesh the tutorial (``blockMesh``, ``surfaceFeatures``, ``snappyHexMesh``), writing **ASCII**,
   because aquaflux's OpenFOAM reader refuses binary;
3. for each resolution, run ``opticalRadiationFoam`` on a copy of the meshed case and keep its
   ``G`` and log under ``work/runs/nphi<P>_ntheta<T>/``.

Environment:

``SOZZI_OOR_SOURCE``
    A checkout of of-optical-radiation (required). Its commit is recorded beside every run.
``SOZZI_OOR_IMAGE``
    The Docker image to run in; default ``oor:local``, built from that checkout's ``Dockerfile``
    (``docker build -t oor:local <checkout>``). The published image is amd64 only.

Arguments: resolutions as ``nPhi x nTheta`` pairs, default ``8x4 16x8`` (64 and 256 directions;
the tutorial runs 8x4).

Run with ``SOZZI_OOR_SOURCE=<checkout> validation/run_case.sh validation/sozzi_radiation/generate_dom_reference.py``.
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

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
TUTORIAL = "tutorials/uvReactorSozzi2006-DOM"
DEFAULT_RESOLUTIONS = ("8x4", "16x8")


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _in_container(source: Path, workdir: str, command: str, log: Path) -> None:
    """Run ``command`` under OpenFOAM's environment, with the checkout, the compiled library and
    the work tree mounted, writing its output to ``log``. Raises if it fails."""
    (WORK / "foamuser").mkdir(parents=True, exist_ok=True)
    docker = [
        "docker", "run", "--rm", "-e", "USER=root",
        "-v", f"{source}:/code",
        "-v", f"{WORK / 'foamuser'}:/root/OpenFOAM",
        "-v", f"{WORK}:/work",
        "-w", workdir,
        os.environ.get("SOZZI_OOR_IMAGE", "oor:local"),
        "bash", "-c", f"source /opt/openfoam13/etc/bashrc; set -e; {command}",
    ]  # fmt: skip
    started = time.perf_counter()
    with log.open("w") as out:
        result = subprocess.run(docker, stdout=out, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        msg = f"container command failed ({result.returncode}); see {log}"
        raise RuntimeError(msg)
    _say(f"  done in {time.perf_counter() - started:.0f} s ({log.name})")


def _set_entry(path: Path, key: str, value: str) -> None:
    """Replace a top-level ``key value;`` entry in an OpenFOAM dictionary."""
    text = path.read_text()
    new, count = re.subn(rf"(?m)^(\s*{key}\s+)[^;]+;", rf"\g<1>{value};", text)
    if count == 0:
        msg = f"no '{key}' entry in {path}"
        raise ValueError(msg)
    path.write_text(new)


def compile_library(source: Path) -> None:
    if list((WORK / "foamuser").glob("*/platforms/*/bin/opticalRadiationFoam")):
        _say("library already compiled")
        return
    _say("compiling of-optical-radiation (once)")
    _in_container(source, "/code", "./Allwmake -j", WORK / "log.Allwmake")


def mesh_case(source: Path) -> Path:
    case = WORK / "case"
    if (case / "constant" / "polyMesh" / "owner").exists():
        _say("case already meshed")
        return case
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(source / TUTORIAL, case)
    for dictionary in ("controlDict", "controlDict.DOM"):
        _set_entry(case / "system" / dictionary, "writeFormat", "ascii")
    _say("meshing the tutorial")
    _in_container(source, "/work/case", "./Allmesh", WORK / "log.Allmesh")
    return case


def run_dom(source: Path, case: Path, n_phi: int, n_theta: int) -> None:
    name = f"nphi{n_phi}_ntheta{n_theta}"
    run = WORK / "runs" / name
    if (run / "G").exists():
        _say(f"{name}: already run")
        return
    staging = WORK / "staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for part in ("0", "system", "constant"):
        shutil.copytree(case / part, staging / part, symlinks=True)
    shutil.copy(staging / "system" / "controlDict.DOM", staging / "system" / "controlDict")
    properties = staging / "constant" / "opticalRadiationProperties"
    _set_entry(properties, "nPhi", str(n_phi))
    _set_entry(properties, "nTheta", str(n_theta))
    _say(f"{name}: {2 * n_phi * n_theta} directions")
    started = time.perf_counter()
    _in_container(
        source,
        f"/work/staging/{name}",
        "opticalRadiationFoam",
        staging / "log.opticalRadiationFoam",
    )
    elapsed = time.perf_counter() - started
    times = sorted(
        (p for p in staging.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.eE+-]+", p.name)),
        key=lambda p: float(p.name),
    )
    written = times[-1]
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(written / "G", run / "G")
    shutil.copy(staging / "log.opticalRadiationFoam", run / "log.opticalRadiationFoam")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    record = {
        "of_optical_radiation_commit": commit,
        "n_phi": n_phi,
        "n_theta": n_theta,
        "directions": 2 * n_phi * n_theta,
        "absorption_per_m": 35.67,
        "lamp_exitance_W_per_m2": 696.42,
        "wall_reflection": 0.0,
        "written_at_time": written.name,
        "wall_seconds": round(elapsed, 1),
    }
    (run / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(staging)
    _say(f"{name}: G kept, {elapsed:.0f} s")


def main(arguments: list[str]) -> None:
    source = Path(os.environ["SOZZI_OOR_SOURCE"]).resolve()
    resolutions = [
        tuple(int(v) for v in pair.split("x")) for pair in arguments or DEFAULT_RESOLUTIONS
    ]
    WORK.mkdir(exist_ok=True)
    compile_library(source)
    case = mesh_case(source)
    for n_phi, n_theta in resolutions:
        run_dom(source, case, n_phi, n_theta)
    _say("all resolutions done")


if __name__ == "__main__":
    main(sys.argv[1:])
