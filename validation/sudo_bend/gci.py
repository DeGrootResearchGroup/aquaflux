"""The bend's grid convergence index (GCI), by the procedure of Celik et al. (2008).

Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H. & Raad, P. E. (2008). Procedure
for estimation and reporting of uncertainty due to discretization in CFD applications. *J. Fluids
Eng.* 130, 078001. For each quantity ``phi`` solved on three grids of representative size
``h = (V / N)^(1/3)``, finest first, with ``r21 = h2 / h1`` and ``r32 = h3 / h2`` (each above 1.3):
the apparent order ``p`` (solved by fixed-point iteration, since the ratios differ), the
extrapolated value, the approximate and extrapolated relative errors, and the fine-grid GCI with a
safety factor of 1.25. A quantity whose two differences change sign converges oscillatorily, and one
whose differences do not shrink does not converge; both are reported as such rather than given an
order.

The quantities, all read from each grid's ``work/results/fields.vtu``:

* ``K_loss`` -- the pressure drop from 1 D before the bend to 10 D after it, area-averaged over
  each cross-section, over ``rho U_b^2 / 2``;
* ``secondary`` -- the cross-section mean of the cross-flow's squared speed at 90 degrees, over
  ``U_b^2``: the strength of the Dean vortices;
* ``U(r)`` at each station -- the axial velocity over ``U_b`` along the diameter in the plane of
  symmetry, from the bend's inside to its outside, at 21 points (the local GCI, as Celik's Fig. 1
  reports a profile).

First, the procedure is checked against the paper's own worked example (its Table 1, the first
column), and the script stops if it does not reproduce it.

Usage::

    python validation/sudo_bend/gci.py [<grid> <grid> <grid>]   # finest first; default round64 round48 round36
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pyvista as pv
from aquaflux.io import read_aqmesh
from geometry import Bend
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.optimize import brentq

HERE = Path(__file__).parent
GRIDS = HERE / "grids"
DEFAULT = ("round64", "round48", "round36")
RHO, NU = 1.2, 1.5e-5
BULK = 6.0e4 * NU / Bend.diameter
#: How many nearest cells a point's stencil is grown from.
TIED = 4
#: Safety factor for three grids (Celik et al. 2008, step 5).
SAFETY = 1.25


@dataclasses.dataclass(frozen=True)
class Convergence:
    """Celik's step 3 to 5 for one quantity on three grids, finest first."""

    phi: tuple[float, float, float]
    order: float | None
    extrapolated: float | None
    approximate_error: float
    extrapolated_error: float | None
    gci: float | None
    kind: str  # "monotone", "oscillatory" or "divergent"


def celik(h: tuple[float, float, float], phi: tuple[float, float, float]) -> Convergence:
    """The apparent order, extrapolated value, errors and fine-grid GCI of ``phi`` on grids ``h``.

    Parameters
    ----------
    h : tuple of float
        The grids' representative sizes, finest first.
    phi : tuple of float
        The quantity on each, in the same order.
    """
    h1, h2, h3 = h
    phi1, phi2, phi3 = phi
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    approximate = abs(e21 / phi1) if phi1 != 0.0 else abs(e21)
    if e21 == 0.0 or e32 == 0.0:
        return Convergence(phi, None, None, approximate, None, None, "exact")
    s = math.copysign(1.0, e32 / e21)
    kind = "monotone" if s > 0 else "oscillatory"
    ratio = abs(e32 / e21)

    def residual(p: float) -> float:
        q = math.log((r21**p - s) / (r32**p - s))
        return p - abs(math.log(ratio) + q) / math.log(r21)

    try:
        p = brentq(residual, 1e-3, 20.0)
    except ValueError:
        return Convergence(phi, None, None, approximate, None, None, "divergent")
    extrapolated = (r21**p * phi1 - phi2) / (r21**p - 1.0)
    extrapolated_error = abs((extrapolated - phi1) / extrapolated)
    gci = SAFETY * approximate / (r21**p - 1.0)
    return Convergence(phi, p, extrapolated, approximate, extrapolated_error, gci, kind)


def check_against_the_paper() -> None:
    """Celik et al. (2008), Table 1, first column: a 2D case on 18000, 8000 and 4500 cells.

    The paper gives p = 1.53, phi_ext = 6.1685, e_a = 1.5 %, e_ext = 1.7 % and GCI = 2.2 %, to
    the digits it prints.
    """
    h = tuple((1.0 / n) ** 0.5 for n in (18000, 8000, 4500))
    result = celik(h, (6.063, 5.972, 5.863))
    got = (result.order, result.extrapolated, result.approximate_error, result.extrapolated_error,
           result.gci)  # fmt: skip
    expected = (1.53, 6.1685, 0.015, 0.017, 0.022)
    printed = (0.005, 0.00005, 0.0005, 0.0005, 0.0005)
    if not all(abs(g - e) <= t for g, e, t in zip(got, expected, printed, strict=True)):
        raise SystemExit(f"the GCI procedure does not reproduce Celik et al.'s Table 1: {got}")
    print(
        f"check: Celik et al. (2008) Table 1 reproduced: p {got[0]:.2f}, phi_ext {got[1]:.4f}, "
        f"e_a {100 * got[2]:.1f} %, e_ext {100 * got[3]:.1f} %, GCI {100 * got[4]:.1f} %",
        flush=True,
    )


@dataclasses.dataclass(frozen=True)
class Station:
    name: str
    centre: np.ndarray
    along: np.ndarray

    @property
    def right(self) -> np.ndarray:
        """Across the pipe in the plane of symmetry, from the bend's inside to its outside."""
        return np.cross(self.along, [0.0, 0.0, 1.0])


def stations(bend: Bend) -> list[Station]:
    rc, d = bend.bend_radius, bend.diameter
    out = [Station("1 D upstream", np.array([-d, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    for degrees in (30, 60, 90):
        a = math.radians(degrees)
        out.append(
            Station(
                f"bend {degrees} deg",
                np.array([rc * math.sin(a), rc * (1.0 - math.cos(a)), 0.0]),
                np.array([math.cos(a), math.sin(a), 0.0]),
            )
        )
    for diameters in (1, 5, 10):
        out.append(
            Station(
                f"{diameters} D downstream",
                np.array([rc, rc + diameters * d, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )
        )
    return out


class Solution:
    """One grid's solved fields, read at any point by a local linear fit to the cells round it.

    The value at a point is the constant term of a linear least-squares fit, weighted by inverse
    squared distance (bounded beside a centroid), to the cell values over a stencil from the mesh's
    own connectivity: the :data:`TIED` cells whose centroids are nearest the point and two rings of
    face neighbours round each (several, since on the pipe's axis four core cells are equally near,
    and a stencil chosen among them by rounding moved a value by 0.007 U_b). It is exact for a linear field and unique whatever the centroids' arrangement. Two
    plainer readings were measured and dropped: a linear interpolation over a Delaunay
    triangulation of the centroids is not unique where they lie on near-regular planes (the graded
    rows after the bend), and a 1e-7 m change of the mesh moved a sampled velocity by 0.02 U_b; the
    nearest centroids by distance alone all lie in one long row's cross-section there, so the fit
    misses the variation along the pipe.
    """

    def __init__(self, grid: str):
        directory = GRIDS / grid / "work"
        mesh = read_aqmesh(directory / "bend.aqmesh")
        cells = mesh.geometry().cell
        self.grid = grid
        self.n_cells = mesh.n_cells
        self.volume = float(np.asarray(cells.volume).sum())
        self.h = (self.volume / self.n_cells) ** (1.0 / 3.0)
        fields = pv.read(directory / "results" / "fields.vtu")
        self.centroids = np.asarray(cells.centroid)
        self.tree = cKDTree(self.centroids)
        fc = mesh.face_cells
        interior = np.asarray(fc.interior)
        owner, neighbour = np.asarray(fc.owner)[interior], np.asarray(fc.neighbour)[interior]
        n = mesh.n_cells
        ring = sparse.coo_matrix(
            (np.ones(2 * owner.size), (np.r_[owner, neighbour], np.r_[neighbour, owner])), (n, n)
        ).tocsr() + sparse.identity(n, format="csr")
        self.stencils = (ring @ ring).tocsr()
        self.velocity = np.asarray(fields.cell_data["U"])
        self.pressure = np.asarray(fields.cell_data["p"])

    def sample(self, station: Station, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` (per cell) at ``points``, shape ``(m,)``, each from the cells round it."""
        del station
        _, nearest = self.tree.query(points, k=TIED)
        out = np.empty(len(points))
        indptr, indices = self.stencils.indptr, self.stencils.indices
        for i, (point, cells) in enumerate(zip(points, nearest, strict=True)):
            stencil = np.unique(
                np.concatenate([indices[indptr[cell] : indptr[cell + 1]] for cell in cells])
            )
            offsets = self.centroids[stencil] - point
            distance = np.linalg.norm(offsets, axis=1)
            scale = distance.max()
            root = scale / np.maximum(distance, 0.05 * scale)
            design = np.column_stack([np.ones(stencil.size), offsets / scale]) * root[:, None]
            out[i] = np.linalg.lstsq(design, values[stencil] * root, rcond=None)[0][0]
        return out

    def section(self, station: Station, radius: float, count: int = 41):
        """Points over the cross-section on a polar grid of equal areas, inside the first cells."""
        rings = np.sqrt((np.arange(count) + 0.5) / count) * 0.93 * radius
        angles = np.linspace(0.0, 2.0 * math.pi, 4 * count, endpoint=False)
        up = np.array([0.0, 0.0, 1.0])
        r, a = np.meshgrid(rings, angles, indexing="ij")
        return station.centre + (
            (r * np.cos(a)).reshape(-1, 1) * station.right + (r * np.sin(a)).reshape(-1, 1) * up
        )


def quantities(solution: Solution, bend: Bend) -> dict[str, float]:
    by_name = {s.name: s for s in stations(bend)}
    r = bend.radius
    out = {}
    before, after = by_name["1 D upstream"], by_name["10 D downstream"]
    mean_p = [
        float(np.nanmean(solution.sample(s, solution.section(s, r), solution.pressure)))
        for s in (before, after)
    ]
    out["K_loss"] = (mean_p[0] - mean_p[1]) / (0.5 * RHO * BULK**2)
    turn = by_name["bend 90 deg"]
    points = solution.section(turn, r)
    u = np.stack([solution.sample(turn, points, solution.velocity[:, i]) for i in range(3)], 1)
    across = u - np.outer(u @ turn.along, turn.along)
    out["secondary"] = float(np.nanmean(np.sum(across**2, axis=1))) / BULK**2
    line = np.linspace(-0.9, 0.9, 21)
    for station in stations(bend):
        points = station.centre + np.outer(line * r, station.right)
        axial = (
            np.stack(
                [solution.sample(station, points, solution.velocity[:, i]) for i in range(3)], 1
            )
            @ station.along
        )
        for x, value in zip(line, axial, strict=True):
            out[f"U {station.name} {x:+.2f}"] = float(value) / BULK
    return out


def main(grids: tuple[str, str, str]) -> None:
    check_against_the_paper()
    bend = Bend()
    solutions = [Solution(grid) for grid in grids]
    h = tuple(s.h for s in solutions)
    for s in solutions:
        print(f"{s.grid}: {s.n_cells} cells, h {1e3 * s.h:.3f} mm", flush=True)
    print(f"r21 {h[1] / h[0]:.4f}, r32 {h[2] / h[1]:.4f}", flush=True)
    values = [quantities(s, bend) for s in solutions]
    print(f"\n{'quantity':<28} {'fine':>9} {'medium':>9} {'coarse':>9} {'p':>6} "
          f"{'extrap.':>9} {'e_a %':>7} {'GCI %':>7}  kind")  # fmt: skip
    local = {}
    for name in values[0]:
        phi = tuple(v[name] for v in values)
        result = celik(h, phi)
        if name.startswith("U "):
            local.setdefault(name.rsplit(" ", 1)[0], []).append(result)
            continue
        print(_row(name, result), flush=True)
    for station, results in local.items():
        monotone = [x for x in results if x.kind == "monotone"]
        orders = [x.order for x in monotone]
        gcis = [x.gci for x in monotone]
        print(
            f"{station:<28} {len(monotone)}/{len(results)} points monotone"
            + (
                f", p mean {np.mean(orders):.2f}, GCI max {100 * max(gcis):.2f} % mean "
                f"{100 * np.mean(gcis):.2f} %"
                if monotone
                else ""
            ),
            flush=True,
        )


def _row(name: str, x: Convergence) -> str:
    fine, medium, coarse = x.phi
    if x.order is None:
        return f"{name:<28} {fine:9.5f} {medium:9.5f} {coarse:9.5f} {'':>6} {'':>9} " \
               f"{100 * x.approximate_error:7.3f} {'':>7}  {x.kind}"  # fmt: skip
    return (
        f"{name:<28} {fine:9.5f} {medium:9.5f} {coarse:9.5f} {x.order:6.2f} {x.extrapolated:9.5f} "
        f"{100 * x.approximate_error:7.3f} {100 * x.gci:7.3f}  {x.kind}"
    )


if __name__ == "__main__":
    main(tuple(sys.argv[1:4]) if len(sys.argv) > 3 else DEFAULT)
