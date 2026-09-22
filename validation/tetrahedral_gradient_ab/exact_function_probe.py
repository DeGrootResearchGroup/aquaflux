"""Issue #435: does the multiple-correction scheme reproduce EXACT functions on this real mesh?

``MultipleCorrectionGradient`` is designed to be quadratic-exact: given exact boundary values it must
return the exact gradient of any linear or quadratic field, to round-off, in every cell. A cell where it
does not is a defect in the reconstruction itself, independent of any flow, turbulence model or march. The
unit tests check this on a synthetic perturbed-tetrahedron cube; this checks it on the duct mesh the march
fails on, cell by cell, and grades the error by the properties that could explain it (boundary faces
owned, wall ring, ``max|M2^-1|``, non-orthogonality).

Fields, all in coordinates normalized to the mesh's bounding box and given exact values at cell centroids
and at boundary-face centroids: constant, linear, a general quadratic, a cubic (not reproducible -- its
error is the truncation the scheme amplifies) and a smooth non-polynomial field.

Schemes: ``multcorr repaired`` (as it marches), ``multcorr owner`` (no corner repair: the underdetermined
cells stay singular), ``multcorr first pass`` (linear-exact only), ``corrected GG`` and ``compact GG``.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/exact_function_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GmresGradientSolve,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from compare import POLYMESH
from diffusion_operator_probe import FirstPassOnly, cell_non_orthogonality


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    centroid = np.asarray(geometry.cell.centroid)
    face_centroid = np.asarray(geometry.face.centroid)
    lo, hi = centroid.min(axis=0), centroid.max(axis=0)
    rng = np.random.default_rng(0)
    a = rng.normal(size=3)
    q = rng.normal(size=(3, 3))
    q = 0.5 * (q + q.T)
    c3 = rng.normal(size=3)

    def unit(points):
        return (np.asarray(points) - lo) / (hi - lo)

    # value and gradient wrt PHYSICAL coordinates, for each field
    def constant(points):
        return np.full(len(points), 3.0), np.zeros((len(points), 3))

    def linear(points):
        u = unit(points)
        return u @ a + 1.0, np.tile(a / (hi - lo), (len(points), 1))

    def quadratic(points):
        u = unit(points)
        value = np.einsum("ni,ij,nj->n", u, q, u) + u @ a + 1.0
        return value, ((2.0 * u @ q + a) / (hi - lo))

    def cubic(points):
        u = unit(points)
        value = np.sum(c3 * u**3, axis=1)
        return value, (3.0 * c3 * u**2) / (hi - lo)

    def smooth(points):
        u = unit(points)
        value = np.sin(3.0 * u[:, 0]) * np.cos(2.0 * u[:, 1]) + u[:, 2] ** 2 * u[:, 0]
        grad = np.stack(
            [
                3.0 * np.cos(3.0 * u[:, 0]) * np.cos(2.0 * u[:, 1]) + u[:, 2] ** 2,
                -2.0 * np.sin(3.0 * u[:, 0]) * np.sin(2.0 * u[:, 1]),
                2.0 * u[:, 2] * u[:, 0],
            ],
            axis=1,
        )
        return value, grad / (hi - lo)

    fields = {
        "constant": constant,
        "linear": linear,
        "quadratic": quadratic,
        "cubic": cubic,
        "smooth (sin/cos)": smooth,
    }

    owner_scheme = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None)
    repaired = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    schemes = {
        "multcorr repaired": repaired,
        "multcorr owner": owner_scheme.bind(mesh, geometry),
        "multcorr first pass": FirstPassOnly(repaired),
        "corrected GG": CorrectedGreenGauss().bind(mesh, geometry),
        "corrected GG exact": CorrectedGreenGauss(solver=GmresGradientSolve()).bind(mesh, geometry),
        "compact GG": CompactGreenGauss().bind(mesh, geometry),
    }

    face_cells = mesh.face_cells
    boundary = ~np.asarray(face_cells.interior)
    owned = np.bincount(np.asarray(face_cells.owner)[boundary], minlength=n)
    m2 = np.max(np.abs(np.asarray(repaired.prepared.m2_inverse)), axis=(1, 2))
    angle = cell_non_orthogonality(mesh, geometry)
    groups = {
        "all cells": np.ones(n, dtype=bool),
        "owns 0 bdry faces": owned == 0,
        "owns 1 bdry face": owned == 1,
        "owns >=2 bdry faces": owned >= 2,
        "max|M2^-1| <= 10": m2 <= 10,
        "max|M2^-1| 10-100": (m2 > 10) & (m2 <= 100),
        "max|M2^-1| 100-1e3": (m2 > 100) & (m2 <= 1e3),
        "max|M2^-1| > 1e3": m2 > 1e3,
    }
    print(
        f"=== {n} cells; {int((owned >= 2).sum())} own >=2 boundary faces; max|M2^-1| over cells: "
        f"median {np.median(m2):.2f}, max {m2.max():.2e}; non-orthogonality median {np.median(angle):.1f} deg "
        f"max {angle.max():.1f} deg ===",
        flush=True,
    )

    for field_name, field in fields.items():
        value_cells, exact = field(centroid)
        value_faces, _ = field(face_centroid)
        norm = np.linalg.norm(exact, axis=1)
        # A constant field has no gradient to scale by: fall back to one unit-box gradient.
        scale = max(np.percentile(norm, 95), 1.0 / float((hi - lo).max()))
        print(
            f"--- {field_name}: max |grad error| / (95th-percentile |exact grad|), per cell group ---",
            flush=True,
        )
        print(f"  {'group':<22}{'cells':>6}" + "".join(f"{s:>21}" for s in schemes), flush=True)
        errors = {}
        for name, scheme in schemes.items():
            g = np.asarray(
                scheme.gradients(jnp.asarray(value_cells), mesh, geometry, jnp.asarray(value_faces))
            )
            err = np.linalg.norm(g - exact, axis=1) / scale
            errors[name] = np.where(np.isfinite(err), err, np.inf)
        for group, mask in groups.items():
            row = "".join(f"{errors[s][mask].max():>21.2e}" for s in schemes)
            print(f"  {group:<22}{int(mask.sum()):>6}{row}", flush=True)
        if field_name in ("linear", "quadratic"):
            e = errors["multcorr repaired"]
            worst = np.argsort(e)[::-1][:5]
            print(
                "  worst multcorr-repaired cells (cell, err, bdry faces, max|M2^-1|, non-orth): "
                + "; ".join(
                    f"{c}, {e[c]:.1e}, {owned[c]}, {m2[c]:.1e}, {angle[c]:.0f}" for c in worst
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
