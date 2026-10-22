"""Structured grids as the mesh section of a test's case file (test-only, not a shipped API).

A case file reads its mesh from a file. A test that wants a small box or channel writes one with
:func:`grid_mesh`, which builds it with :func:`~aquaflux.mesh.structured_grid_2d` /
:func:`~aquaflux.mesh.structured_grid_3d` (patches named by side, a periodic axis's two sides fused
into a seam), writes it as ``.aqmesh`` once per process into a temporary directory, and returns the
``{kind: AquafluxMesh, path: ...}`` section naming it by an absolute path -- so a case reads it from
wherever the case is read.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import cache
from pathlib import Path

from aquaflux.io import write_aqmesh
from aquaflux.mesh import graded_nodes, structured_grid_2d, structured_grid_3d

__all__ = ["grid_mesh", "grid_mesh_path"]


@cache
def _directory() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="aquaflux-test-grids-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return directory


@cache
def grid_mesh_path(
    cells: tuple[int, ...],
    lengths: tuple[float, ...],
    periodic: tuple[str, ...] = (),
    growth: float | None = None,
    both_sides: bool = True,
) -> str:
    """The ``.aqmesh`` file of the box ``[0, lx] x [0, ly]`` (or a 3D box), written once.

    Parameters
    ----------
    cells, lengths : tuple
        Two counts and two lengths, or three of each (a 3D box is uniform).
    periodic : tuple of {"x"}
        Axes that wrap round (2D only).
    growth, both_sides : float or None, bool
        Wall-normal cells along ``y`` growing geometrically by ``growth`` from both walls, or from
        ``y = 0`` only (2D only); unset, uniform.
    """
    if len(cells) == 3:
        mesh = structured_grid_3d(*cells, *lengths, named_boundaries=True)
    else:
        nodes = {}
        if growth is not None:
            nodes["y_nodes"] = graded_nodes(cells[1], lengths[1], growth, both_sides=both_sides)
        mesh = structured_grid_2d(
            *cells, *lengths, named_boundaries=True, periodic=tuple(periodic), **nodes
        )
    name = "_".join(
        ["grid", *map(str, cells), *map(str, lengths), *periodic, str(growth), str(both_sides)]
    )
    path = _directory() / f"{name}.aqmesh"
    if not path.exists():
        write_aqmesh(mesh, path)
    return str(path)


def grid_mesh(cells, lengths, *, periodic=(), growth=None, both_sides=True) -> dict[str, str]:
    """A case file's mesh section naming :func:`grid_mesh_path`'s file."""
    return {
        "kind": "AquafluxMesh",
        "path": grid_mesh_path(
            tuple(cells), tuple(float(n) for n in lengths), tuple(periodic), growth, both_sides
        ),
    }
