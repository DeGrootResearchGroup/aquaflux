"""The ray-effects room: its dimensions and the small helpers every script here shares.

Kept in one module so the mesh, the DOM runs, the aquaflux gather, the reference and the figures
cannot disagree about where the bunny stands or which patch is the floor.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IES_FILE = HERE / "ushio_b1.ies"

ROOM_LOW = np.array([-2.0, -2.0, 0.0])  # m
ROOM_HIGH = np.array([2.0, 2.0, 3.0])  # m
BUNNY_BASE = 1.2  # m, height of the bunny's lowest point
LAMP_PATCH = "lamp"
FLOOR_PATCH = "floor"

# The line profile across the shadow and the floor window the maps show.
PROFILE_Y = 0.1  # m
PROFILE_X = (-1.0, 1.0)  # m
MAP_HALF_WIDTH = 1.0  # m


# Planes through the volume where the fluence rate is shown and the reference is computed: one
# vertical, through the lamp and the bunny's middle, and one horizontal, between the bunny and the
# floor. Each sits just off a face plane (the mesh's faces lie on multiples of 3.125 mm from the
# walls), so it cuts exactly one layer of cells at every refinement level.
SLICES = {"vertical": (1, 0.003), "horizontal": (2, 0.603)}  # name: (axis, coordinate in m)


def slice_cells(centre: np.ndarray, volume: np.ndarray, name: str) -> np.ndarray:
    """Indices of the cells a slice plane passes through.

    A cell is taken as a cube of side ``volume ** (1/3)`` about its centre, which is exact for the
    hexahedra that fill all but the cells snapped to the bunny.
    """
    axis, value = SLICES[name]
    half = 0.5 * np.cbrt(volume)
    return np.flatnonzero(np.abs(centre[:, axis] - value) < half)


# The reflecting variant: the room's own surfaces reflect diffusely; the bunny and the lamp's window
# stay black. Idealized for 222 nm (ordinary paints are 5-10 %), chosen so the bounce light is a
# large part of the field.
REFLECTING_PATCHES = ("floor", "ceiling", "walls")
WALL_REFLECTANCE = 0.5
# aquaflux's reflecting surfaces: each face of the room cut into squares of this side (and a check
# at half of it). Reflected light is smooth, so these are far coarser than the mesh's faces.
REFLECTING_FACET = 0.2  # m


def room_facets(side: float) -> np.ndarray:
    """The room's six faces as squares of ``side``, two triangles each, wound to face the room.

    Returns
    -------
    np.ndarray, shape ``(n_triangles, 3, 3)``
        Triangles whose normals point into the room: up from the floor, down from the ceiling,
        inwards from each wall.
    """
    triangles = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        nu = round((ROOM_HIGH[u] - ROOM_LOW[u]) / side)
        nv = round((ROOM_HIGH[v] - ROOM_LOW[v]) / side)
        gu = np.linspace(ROOM_LOW[u], ROOM_HIGH[u], nu + 1)
        gv = np.linspace(ROOM_LOW[v], ROOM_HIGH[v], nv + 1)
        for level, inward in ((ROOM_LOW[axis], 1.0), (ROOM_HIGH[axis], -1.0)):
            for i in range(nu):
                for j in range(nv):
                    corners = np.zeros((4, 3))
                    corners[:, axis] = level
                    corners[:, u] = [gu[i], gu[i + 1], gu[i + 1], gu[i]]
                    corners[:, v] = [gv[j], gv[j], gv[j + 1], gv[j + 1]]
                    pair = [corners[[0, 1, 2]], corners[[0, 2, 3]]]
                    normal = np.cross(pair[0][1] - pair[0][0], pair[0][2] - pair[0][0])
                    if normal[axis] * inward < 0:
                        pair = [t[::-1] for t in pair]
                    triangles.extend(pair)
    return np.array(triangles)


# The fixture's h = 0 direction (IES "fixtureUp"), and its axis: the lamp faces straight down.
LAMP_UP = (1.0, 0.0, 0.0)
LAMP_AXIS = (0.0, 0.0, -1.0)


def lamp_photometry():
    """The lamp's measured photometry, read from ``ushio_b1.ies``."""
    from aquaflux.radiation import read_ies

    return read_ies(IES_FILE)


def lamp_power() -> float:
    """The lamp's radiant power in W: the IES table's own integrated flux.

    The file states its intensities in mW/sr (an ``[_INTENSITYUNITS]`` keyword), so its flux is in
    mW; the unit is checked rather than assumed, since LM-63's default unit is the candela.
    """
    photometry = lamp_photometry()
    unit = photometry.keywords.get("_INTENSITYUNITS")
    if unit != "mW/sr":
        msg = f"{IES_FILE.name} states intensities in {unit!r}; expected 'mW/sr'"
        raise ValueError(msg)
    return photometry.flux * 1e-3


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of an STL file's vertices, as ``(low, high)``, each shape ``(3,)``."""
    from aquaflux.radiation import read_stl

    vertices = np.asarray(read_stl(path).vertices).reshape(-1, 3)
    return vertices.min(axis=0), vertices.max(axis=0)


# checkMesh failures a snapped mesh of a lattice carries and the comparison can live with: a few
# faces past its skewness threshold of 4, and the concave cells that -allGeometry adds a test for.
# Both solvers run on the one mesh, so neither is favoured; any other failure stops the run.
TOLERATED_FAILURES = {
    "skew_faces": r"\*\*\*Max skewness = [0-9.eE+-]+, (\d+) highly skew faces",
    "concave_cells": r"\*\*\*Concave cells \(using face planes\) found, number of cells: (\d+)",
}


def check_mesh_summary(log: Path) -> dict:
    """The cell count and the quality figures ``checkMesh`` reports, from its log.

    Raises unless the log ends in ``Mesh OK.`` or every failed check is one of
    ``TOLERATED_FAILURES``, whose counts are then reported.
    """
    text = log.read_text()
    failed = re.search(r"Failed (\d+) mesh checks", text)
    tolerated = {}
    if failed is not None:
        tolerated = {
            name: int(match.group(1))
            for name, pattern in TOLERATED_FAILURES.items()
            if (match := re.search(pattern, text))
        }
        if len(tolerated) != int(failed.group(1)) or len(re.findall(r"\*\*\*", text)) != len(
            tolerated
        ):
            msg = f"checkMesh failed a check that is not tolerated here; see {log}"
            raise RuntimeError(msg)
    elif "Mesh OK." not in text:
        msg = f"checkMesh reported neither 'Mesh OK.' nor its failures; see {log}"
        raise RuntimeError(msg)

    def number(pattern: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            msg = f"no match for {pattern!r} in {log}"
            raise ValueError(msg)
        return float(match.group(1))

    return {
        "cells": int(number(r"cells:\s+(\d+)")),
        "faces": int(number(r"\bfaces:\s+(\d+)")),
        "max_non_orthogonality_deg": number(r"Mesh non-orthogonality Max:\s+([0-9.eE+-]+)"),
        "max_skewness": number(r"Max skewness = ([0-9.eE+-]+)"),
        "max_aspect_ratio": number(r"Max aspect ratio = ([0-9.eE+-]+)"),
        "tolerated_failures": tolerated,
    }
