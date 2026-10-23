"""Every file a shipped case or mesh file names is in the repository, or is declared to be generated.

A case file names its mesh (and, for radiation, a lamp's photometry) by a path relative to itself. A
path into a directory the repository ignores reads cleanly and builds on the machine that ran the
generator, and nowhere else: the 3D backward-facing step once named ``runs/kwsst/polyMesh``, which an
OpenFOAM run writes and ``.gitignore`` excludes, while the same mesh sat tracked in another case's
directory, so from a fresh checkout the case could not be built and nothing said so.

A mesh file names the drawing it meshes and the mesh it writes, so it is read the same way: the
drawing must be tracked, and the mesh it makes is declared as generated.

A path that a fresh checkout cannot have is therefore a decision to write down, with what generates
it, in :data:`GENERATED` -- declared, not discovered. A declared path that is now tracked, or that no
case file names any more, fails too, so the table cannot outlive what it describes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from aquaflux.case import read_case
from aquaflux.case.case_file import read_case_document, read_mesh_file
from aquaflux.case.paths import named_paths

REPO = Path(__file__).resolve().parents[2]

#: Repository-relative paths a shipped case file may name although no file under them is tracked:
#: path -> what writes it.
GENERATED: dict[str, str] = {
    "validation/ray_effects_room/work/bunny/case/constant/polyMesh": (
        "generate_dom.py (blockMesh + snappyHexMesh in OpenFOAM's Docker image)"
    ),
    "validation/ray_effects_room/work/empty/case/constant/polyMesh": (
        "generate_dom.py (blockMesh + snappyHexMesh in OpenFOAM's Docker image)"
    ),
    "validation/sudo_bend/work/bend.aqmesh": "aquaflux mesh validation/sudo_bend/mesh.yaml",
}


def _tracked() -> list[str]:
    """Every tracked file, repository-relative."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, text=True
    )
    return [name for name in listed.stdout.split("\0") if name]


def _yaml_files(tracked: list[str]) -> list[str]:
    return [name for name in tracked if name.startswith("validation/") and name.endswith(".yaml")]


def _is_mesh_file(name: str) -> bool:
    """Whether a shipped YAML file is a mesh file: one naming the drawing it meshes, which no case
    file has."""
    document = read_case_document(REPO / name)
    return isinstance(document, dict) and "drawing" in document


def _case_files(tracked: list[str]) -> list[str]:
    return [name for name in _yaml_files(tracked) if not _is_mesh_file(name)]


def _mesh_files(tracked: list[str]) -> list[str]:
    return [name for name in _yaml_files(tracked) if _is_mesh_file(name)]


def _named(tracked: list[str]) -> dict[str, list[str]]:
    """Each path the shipped case and mesh files name, normalized to the repository -> the files
    naming it."""
    named: dict[str, list[str]] = {}
    read = [(name, read_case(REPO / name)) for name in _case_files(tracked)]
    read += [(name, read_mesh_file(REPO / name)) for name in _mesh_files(tracked)]
    for case_file, case in read:
        for location, path in named_paths(case.spec):
            assert not os.path.isabs(path), f"{case_file} {location} is absolute: {path}"
            target = os.path.relpath(os.path.normpath(case.directory / path), REPO)
            named.setdefault(target, []).append(f"{case_file} ({location})")
    return named


def _is_tracked(path: str, tracked: list[str]) -> bool:
    """Whether ``path`` is a tracked file or a directory holding one."""
    return any(name == path or name.startswith(path + "/") for name in tracked)


def test_the_census_reads_every_shipped_case_file() -> None:
    # The guard below is only as wide as the files it reads: an empty or partial listing would pass it.
    tracked = _tracked()
    case_files = _case_files(tracked)
    assert "validation/bfs3d_openfoam/case.yaml" in case_files
    assert "validation/pitzdaily_openfoam/case.yaml" in case_files
    assert "validation/sudo_bend/mesh.yaml" in _mesh_files(tracked)
    named = _named(tracked)
    assert any(path.endswith("polyMesh") for path in named)
    assert "validation/sudo_bend/bend.step" in named


def test_every_path_a_shipped_case_file_names_is_tracked_or_declared_generated() -> None:
    tracked = _tracked()
    missing = {
        path: where
        for path, where in _named(tracked).items()
        if path not in GENERATED and not _is_tracked(path, tracked)
    }
    assert not missing, (
        "a shipped case file names a path that is not in the repository, so a fresh checkout cannot "
        f"build it (track it, or declare what generates it in GENERATED): {missing}"
    )


def test_every_declared_generated_path_is_named_and_untracked() -> None:
    tracked = _tracked()
    named = _named(tracked)
    stale = {
        path: "no shipped case file names it" if path not in named else "it is tracked"
        for path in GENERATED
        if path not in named or _is_tracked(path, tracked)
    }
    assert not stale, f"a GENERATED entry no longer describes the repository: {stale}"
