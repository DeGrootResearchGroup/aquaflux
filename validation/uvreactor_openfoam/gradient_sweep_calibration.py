"""How many inner sweeps does the Hessian-corrected gradient need *on this mesh*?

``HessianCorrectedGradient`` eliminates the Hessian and solves the remaining system for the gradient,
applying the eliminated Hessian block's inverse by a fixed number of block-Jacobi Richardson sweeps.
A fixed count carries no convergence test, so on a mesh skewed beyond what the default was calibrated
against the reconstruction quietly stops being exact for quadratic fields -- which is the entire
reason to choose this scheme over a corrected Green-Gauss one. The count needed is set by the mesh's
skewness and not by its size, so it cannot be inferred from a small test mesh and has to be measured
on the real one. That is what this script does.

It uses only the public reconstruction API: the same field is reconstructed at a ladder of inner
sweep counts and compared against a reference solve of the same system, so the difference reported is
the inner iteration's own truncation and nothing else. Two references are available -- a long fixed
sweep (cheap, the default) and an exact Krylov solve (``--exact``, slower but independent of the
iteration being calibrated). The long-sweep reference is self-checked against a longer one, so a
reference that is itself under-resolved shows up rather than flattering every arm below it.

The outer system is calibrated the same way. It is well enough conditioned to be swept rather than
Krylov-solved, and a swept outer solve is what removes the last inner product from the scheme -- the
configuration worth having where reductions are expensive. Whether it converges in a practical number
of sweeps on a given mesh is again a mesh property, so it is measured rather than assumed.

Usage
-----
    python3 validation/uvreactor_openfoam/gradient_sweep_calibration.py <path-to-polyMesh-dir>
        [--exact] [--cells N] [--outer-sweeps N]

Scale
-----
Written to be run on the full mesh, which is where the question it answers has a real answer -- and
that is not a laptop-sized job. The reconstruction holds several ``(n_faces, dim, dim)``
intermediates, so at a few million faces each is gigabytes on its own; the banner prints that unit,
and every arm prints peak resident memory beside its wall clock, so a run heading for trouble says so
before it gets there rather than being killed without explanation.

``UV_INNER_LADDER`` (default ``2,4,6,8,10,12,16``) and ``UV_REFERENCE_SWEEPS`` (default 24) shorten
the ladder when a full sweep is more than a node can hold -- each rung is one more full
reconstruction. ``--exact`` swaps the long-sweep reference for a Krylov solve, considerably dearer at
scale; prefer the default reference unless its own self-check reports it unconverged.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh.quality import face_planarity  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    CorrectedGreenGauss,
    GmresGradientSolve,
    HessianCorrectedGradient,
    SweptGradientSolve,
)
from aquaflux.schemes.interpolation import interpolation_factor  # noqa: E402
from aquaflux.vectors import norm_squared  # noqa: E402

INNER_LADDER = tuple(
    int(n) for n in os.environ.get("UV_INNER_LADDER", "2,4,6,8,10,12,16").split(",")
)
REFERENCE_SWEEPS = int(os.environ.get("UV_REFERENCE_SWEEPS", "24"))
REFERENCE_CHECK_SWEEPS = REFERENCE_SWEEPS + 8
OUTER_LADDER = (10, 20, 40, 80)


def peak_rss_gb() -> float:
    """Peak resident set size so far, in GB -- the number that decides whether a run fits.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS, so the unit is chosen by platform rather
    than assumed; getting that wrong reports a thousand-fold wrong figure, which on this question
    would be worse than reporting nothing.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2) if sys.platform.startswith("linux") else peak / 1e9


def footprint_note(mesh) -> str:
    """A per-array footprint estimate, stated before the run rather than discovered during it.

    The Hessian-corrected reconstruction carries several ``(n_faces, dim, dim)`` intermediates -- the
    face-curvature tensor and the owner/neighbour/interpolated Hessian gathers among them -- and at a
    few million faces each is a substantial array on its own. Printing the unit up front is what lets
    a node be sized for the run, instead of the run discovering the node.
    """
    per_face = mesh.n_faces * mesh.dim * mesh.dim * 8 / 1e9
    per_cell = mesh.n_cells * mesh.dim * mesh.dim * 8 / 1e9
    return (
        f"one (n_faces, dim, dim) array = {per_face:.2f} GB; "
        f"one per-cell block = {per_cell:.2f} GB (two are held)"
    )


def skewness_census(mesh, geometry) -> None:
    """Report the two geometric quantities that set the reconstruction's difficulty.

    ``|D_g,ip| / |d|`` is the face centroid's offset from the owner-neighbour line as a fraction of
    the cell spacing -- the skewness the corrections exist for, and what drives both iterations'
    contraction rate. Face planarity is reported beside it because a warped face breaks the
    Green-Gauss face integral's exactness for a quadratic for *every* scheme in this family, so a
    mesh that is badly warped caps the accuracy no sweep count can recover.
    """
    face_cells = mesh.face_cells
    centroid = geometry.cell.centroid
    x_p = centroid[face_cells.owner]
    d = face_cells.neighbour_centroid(centroid) - x_p
    g = interpolation_factor(face_cells, geometry)
    skew = geometry.face.centroid - (x_p + d * g[:, None])
    interior = np.asarray(face_cells.interior)
    spacing = np.sqrt(np.asarray(norm_squared(d)))
    offset = np.sqrt(np.asarray(norm_squared(skew)))
    ratio = offset[interior] / np.maximum(spacing[interior], 1e-300)
    planarity = np.asarray(face_planarity(mesh))

    print("  skewness |D_g,ip| / |d| over interior faces:", flush=True)
    for label, value in (
        ("median", np.median(ratio)),
        ("p99", np.percentile(ratio, 99)),
        ("max", ratio.max()),
    ):
        print(f"    {label:>7s} = {value:.4f}", flush=True)
    print(
        f"  face planarity |S|/sum|tri|:  min = {planarity.min():.6f}, "
        f"p1 = {np.percentile(planarity, 1):.6f}  (1.0 = planar)",
        flush=True,
    )


def probe_field(geometry):
    """A general quadratic in the cell centroids, and its values on the face centroids.

    A quadratic is the field this scheme is *defined* to reconstruct exactly, so it is the field whose
    reconstruction error is entirely attributable to the solve rather than to the discretization. The
    coordinates are shifted to the mesh's own centre and scaled by its extent first: a CAD-derived
    mesh can sit far from the origin, where an unshifted quadratic is dominated by a large constant
    and its gradient loses significance to cancellation.
    """
    x = np.asarray(geometry.cell.centroid)
    xf = np.asarray(geometry.face.centroid)
    centre = x.mean(axis=0)
    extent = max(float(np.abs(x - centre).max()), 1e-300)

    def evaluate(points):
        u = (points - centre) / extent
        value = 0.6 + 1.7 * u[:, 0] - 1.1 * u[:, 1] + 0.4 * u[:, 0] ** 2 + 0.9 * u[:, 0] * u[:, 1]
        if u.shape[1] == 3:
            value = value + 0.8 * u[:, 2] + 0.5 * u[:, 2] ** 2 - 0.3 * u[:, 1] * u[:, 2]
        return value

    return jnp.asarray(evaluate(x)), jnp.asarray(evaluate(xf))


def reconstruct(scheme, field, mesh, geometry, bvals, label):
    """Reconstruct and report wall time, so the calibration also prices each arm."""
    started = time.perf_counter()
    # JAX dispatches asynchronously, so the result has to be waited on before the clock is read --
    # without this the ladder's timings measure dispatch and rank every arm as equally fast.
    gradient = scheme.gradients(field, mesh, geometry, bvals).block_until_ready()
    elapsed = time.perf_counter() - started
    gradient = np.asarray(gradient)
    print(f"    {label:<34s} {elapsed:7.2f} s   peak RSS {peak_rss_gb():5.2f} GB", flush=True)
    return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="path to a polyMesh directory")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="reference the inner ladder against an exact Krylov solve instead of a long sweep",
    )
    parser.add_argument(
        "--outer-sweeps",
        type=int,
        default=None,
        help="calibrate the swept OUTER solve up to this many sweeps (default: skip)",
    )
    args = parser.parse_args()

    print(f"mesh: {args.mesh}", flush=True)
    started = time.perf_counter()
    mesh = read_openfoam(args.mesh.resolve())
    geometry = mesh.geometry()
    print(
        f"  n_cells={mesh.n_cells}  n_faces={mesh.n_faces}  dim={mesh.dim}  "
        f"(import + geometry {time.perf_counter() - started:.1f} s)",
        flush=True,
    )
    block_bytes = mesh.n_cells * mesh.dim * mesh.dim * 8
    print(
        f"  per-cell preconditioner blocks: 2 x (n_cells, {mesh.dim}, {mesh.dim}) "
        f"= 2 x {block_bytes / 1e6:.1f} MB",
        flush=True,
    )
    print(f"  memory unit: {footprint_note(mesh)}", flush=True)
    print(f"  peak RSS after import + geometry: {peak_rss_gb():.2f} GB", flush=True)
    skewness_census(mesh, geometry)

    field, bvals = probe_field(geometry)

    print("\ninner (Hessian) solve -- reconstruction vs a converged reference", flush=True)
    if args.exact:
        reference_scheme = HessianCorrectedGradient(hessian_solver=GmresGradientSolve())
        reference_label = "reference: exact Krylov inner"
    else:
        reference_scheme = HessianCorrectedGradient(
            hessian_solver=SweptGradientSolve(sweeps=REFERENCE_SWEEPS, warn_tol=None)
        )
        reference_label = f"reference: {REFERENCE_SWEEPS} inner sweeps"
    reference = reconstruct(reference_scheme, field, mesh, geometry, bvals, reference_label)
    scale = max(float(np.abs(reference).max()), 1e-300)

    if not args.exact:
        # A fixed-sweep reference that is itself under-resolved would flatter every arm below it, so
        # it is checked against a longer one before anything is measured against it.
        longer = reconstruct(
            HessianCorrectedGradient(
                hessian_solver=SweptGradientSolve(sweeps=REFERENCE_CHECK_SWEEPS, warn_tol=None)
            ),
            field,
            mesh,
            geometry,
            bvals,
            f"reference self-check: {REFERENCE_CHECK_SWEEPS} sweeps",
        )
        drift = float(np.abs(longer - reference).max()) / scale
        verdict = "converged" if drift < 1e-13 else "NOT CONVERGED -- re-run with --exact"
        print(f"    reference self-check drift = {drift:.3e}  ({verdict})", flush=True)

    print("\n  inner sweeps -> relative departure from the reference gradient", flush=True)
    default_sweeps = HessianCorrectedGradient().hessian_solver.sweeps
    for sweeps in INNER_LADDER:
        gradient = reconstruct(
            HessianCorrectedGradient(
                hessian_solver=SweptGradientSolve(sweeps=sweeps, warn_tol=None)
            ),
            field,
            mesh,
            geometry,
            bvals,
            f"inner sweeps = {sweeps}"
            + ("   <- shipped default" if sweeps == default_sweeps else ""),
        )
        error = float(np.abs(gradient - reference).max()) / scale
        print(f"      relative departure = {error:.3e}", flush=True)

    if args.outer_sweeps:
        print(
            "\nouter (Schur) solve -- swept vs the Krylov default, at the shipped inner count",
            flush=True,
        )
        krylov = reconstruct(
            HessianCorrectedGradient(), field, mesh, geometry, bvals, "outer: Krylov (default)"
        )
        for sweeps in [s for s in OUTER_LADDER if s <= args.outer_sweeps]:
            gradient = reconstruct(
                HessianCorrectedGradient(solver=SweptGradientSolve(sweeps=sweeps, warn_tol=None)),
                field,
                mesh,
                geometry,
                bvals,
                f"outer sweeps = {sweeps}",
            )
            error = float(np.abs(gradient - krylov).max()) / max(
                float(np.abs(krylov).max()), 1e-300
            )
            print(f"      relative departure from Krylov = {error:.3e}", flush=True)

    print("\nfor scale -- the scheme this one replaces, on the same field", flush=True)
    corrected = reconstruct(
        CorrectedGreenGauss(), field, mesh, geometry, bvals, "CorrectedGreenGauss (default)"
    )
    print(
        f"    relative departure from the Hessian-corrected reference = "
        f"{float(np.abs(corrected - reference).max()) / scale:.3e}",
        flush=True,
    )
    print(
        "\nRead the ladder for the first sweep count whose departure is below the accuracy you need;\n"
        "the shipped default is calibrated on much smaller meshes and is not a guarantee here.",
        flush=True,
    )


if __name__ == "__main__":
    main()
