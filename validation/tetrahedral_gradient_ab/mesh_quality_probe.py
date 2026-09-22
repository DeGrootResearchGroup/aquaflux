"""Issue #435: how good (or bad) is the tetrahedral duct mesh, next to a mesh that is known to march?

Face and cell quality of the duct mesh and of pitzDaily's mesh (hexahedral-equivalent quadrilaterals, the
case the multiple-correction scheme marches), computed the same way for both:

* non-orthogonality -- the angle between the cell-centre line and the face normal, per interior face;
* skewness -- how far the face centroid lies from where the cell-centre line crosses the face plane,
  divided by the cell-centre distance, per interior face;
* size -- the cube root of the volume (``sqrt`` for a 2D mesh), and how many cells span the domain;
* neighbour size ratio -- ``max(V_P, V_N) / min(V_P, V_N)`` per interior face;
* wall-cell spacing -- the wall distance of the cells owning wall faces, relative to the local cell size,
  and the ``y+`` that spacing gives at the duct's anchor and target Reynolds numbers (Blasius friction).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/mesh_quality_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import CorrectedGreenGauss
from compare import NU, POLYMESH, RATIO, U_IN, build_case

REPO = Path(__file__).resolve().parents[2]
PITZ = REPO / "validation" / "pitzdaily_openfoam" / "of_case" / "constant" / "polyMesh"


def describe(name: str, path: Path) -> dict:
    mesh = read_openfoam(path)
    geometry = mesh.geometry()
    face_cells = mesh.face_cells
    interior = np.asarray(face_cells.interior)
    owner = np.asarray(face_cells.owner)[interior]
    neighbour = np.asarray(face_cells.neighbour)[interior]
    x = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)
    face_x = np.asarray(geometry.face.centroid)[interior]
    normal = np.asarray(geometry.face.normal)[interior]

    d = x[neighbour] - x[owner]
    dist = np.linalg.norm(d, axis=1)
    cosine = np.sum(d * normal, axis=1) / dist
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    # Where the cell-centre line crosses the face plane, and how far that is from the face centroid.
    t = np.sum((face_x - x[owner]) * normal, axis=1) / np.sum(d * normal, axis=1)
    crossing = x[owner] + t[:, None] * d
    skew = np.linalg.norm(face_x - crossing, axis=1) / dist
    ratio = np.maximum(volume[owner], volume[neighbour]) / np.minimum(
        volume[owner], volume[neighbour]
    )
    h = volume ** (1.0 / mesh.dim)
    return {
        "name": name,
        "mesh": mesh,
        "n": mesh.n_cells,
        "dim": mesh.dim,
        "angle": angle,
        "skew": skew,
        "ratio": ratio,
        "h": h,
        "extent": np.asarray(geometry.cell.centroid).max(axis=0)
        - np.asarray(geometry.cell.centroid).min(axis=0),
    }


def line(label: str, values: np.ndarray, fmt: str = "{:8.2f}") -> str:
    q = np.percentile(values, [50, 90, 99, 100])
    return f"  {label:<34}" + " ".join(fmt.format(v) for v in [values.mean(), *q])


def main() -> None:
    meshes = [describe("tetrahedral duct", POLYMESH)]
    if PITZ.exists():
        meshes.append(describe("pitzDaily (2D)", PITZ))
    for m in meshes:
        print(
            f"=== {m['name']}: {m['n']} cells, {m['dim']}D; cell size h = V^(1/dim): median "
            f"{np.median(m['h']):.2e} m, extents {np.array2string(m['extent'], precision=3)} m ===",
            flush=True,
        )
        print(
            f"  {'(per interior face)':<34}"
            + " ".join(f"{s:>8}" for s in ("mean", "median", "p90", "p99", "max")),
            flush=True,
        )
        print(line("non-orthogonality (deg)", m["angle"]), flush=True)
        print(line("skewness |x_f - x_cross| / |d|", m["skew"], "{:8.3f}"), flush=True)
        print(line("neighbour volume ratio", m["ratio"]), flush=True)
        print(
            f"  faces above 40 deg: {(m['angle'] > 40).mean():.1%}; above 50 deg: {(m['angle'] > 50).mean():.1%}; "
            f"skewness above 0.3: {(m['skew'] > 0.3).mean():.1%}",
            flush=True,
        )

    duct = meshes[0]
    print("=== the duct against its own geometry ===", flush=True)
    h = np.median(duct["h"])
    print(
        f"  cross-section 0.025 m over median h = {h * 1e3:.1f} mm -> about {0.025 / h:.1f} cells across the duct; "
        f"length 0.25 m -> about {0.25 / h:.0f} along it",
        flush=True,
    )
    case = build_case(CorrectedGreenGauss())
    wall = np.asarray(case.turbulence.wall_cells)
    d = np.asarray(case.turbulence.wall_distance)[wall]
    print(
        f"  wall cells: {wall.size}; wall distance d median {np.median(d) * 1e3:.2f} mm "
        f"(min {d.min() * 1e3:.2f}, max {d.max() * 1e3:.2f}); d / h median {np.median(d / duct['h'][wall]):.2f}",
        flush=True,
    )
    for label, nu in (("target", NU), (f"anchor (nu x {RATIO:g})", NU * RATIO)):
        re = U_IN * 0.025 / nu
        u_tau = U_IN * np.sqrt(0.5 * 0.0791 * re**-0.25)
        yplus = d * u_tau / nu
        print(
            f"  {label}: Re_Dh {re:.0f}, Blasius u_tau {u_tau:.3f} m/s -> wall-cell y+ median {np.median(yplus):.0f} "
            f"(p10 {np.percentile(yplus, 10):.0f}, p90 {np.percentile(yplus, 90):.0f})",
            flush=True,
        )
    print(
        "  (commonly quoted practice, not measured here: OpenFOAM's checkMesh reports non-orthogonality above "
        "70 deg as severe, and unstructured tetrahedral meshes of complex parts often reach 55-65 deg.)",
        flush=True,
    )


if __name__ == "__main__":
    main()
