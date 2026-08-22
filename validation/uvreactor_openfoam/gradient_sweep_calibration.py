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
import equinox as eqx  # noqa: E402
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
# The self-check needs a LONGER sweep than the reference, and on a large mesh that is the arm most
# likely to be unaffordable -- the unrolled sweep has a memory cliff in the sweep count (measured on
# a 1.6M-cell mesh: every rung to 12 fits in 12 GB, 16 is killed outright). Overridable so a run can
# keep the reference it can afford instead of losing the whole calibration to the check above it.
REFERENCE_CHECK_SWEEPS = int(os.environ.get("UV_REFERENCE_CHECK", REFERENCE_SWEEPS + 8))
OUTER_LADDER = tuple(int(n) for n in os.environ.get("UV_OUTER_LADDER", "3,5,10,20").split(","))
# The outer arms hold the INNER count fixed, because cost here is the product of the two -- the inner
# solve runs once per outer operator apply -- so sweeping one with the other at its default measures
# a program several times larger than anything worth shipping, and walks into the memory cliff on a
# large mesh. Calibrate the inner ladder first and pin its answer here.
OUTER_INNER_SWEEPS = int(os.environ.get("UV_OUTER_INNER", "6"))
# Betchen and Straatman solve this reconstruction by UNDER-RELAXED block-Jacobi, and state that on an
# arbitrary grid a relaxation strictly inside (0, 1] is needed for convergence at all -- they run 0.8.
# So the relaxation is part of what has to be calibrated per mesh, not a constant: swept alongside the
# sweep count, because the two trade against each other (heavier damping needs more sweeps).
OUTER_RELAXATIONS = tuple(float(x) for x in os.environ.get("UV_OUTER_RELAX", "1.0").split(","))


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


def gradient_error(gradient, analytic):
    """Per-cell relative gradient error as ``median / p99 / max``.

    A MAX-NORM ALONE IS NEARLY USELESS ON A MILLION-CELL MESH. One sliver cell -- which an automatic
    mesher does produce -- sets it for the whole grid, so a mesh that reconstructs perfectly
    everywhere except a handful of cells is indistinguishable from one that is wrong everywhere.
    Measured on the reactor mesh: the max is ~7 (a 700% error) while the median is orders below it.
    The three together say both how good the reconstruction usually is and how bad it ever gets, and
    the gap between them is itself the diagnosis -- a large gap is a few bad cells, a small one is a
    scheme that does not work here.

    Normalized per cell by the analytic gradient's own magnitude at that cell, with a floor at a
    small fraction of the field's typical magnitude so that cells where the true gradient nearly
    vanishes cannot manufacture an enormous relative error from a small absolute one.
    """
    magnitude = np.linalg.norm(analytic, axis=-1)
    floor = 1e-6 * float(np.median(magnitude))
    relative = np.linalg.norm(gradient - analytic, axis=-1) / np.maximum(magnitude, floor)
    return (
        float(np.median(relative)),
        float(np.percentile(relative, 99)),
        float(relative.max()),
    )


def format_error(stats):
    """Render a ``gradient_error`` triple as ``median / p99 / max``."""
    return " / ".join(f"{value:.3e}" for value in stats)


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

    def analytic_gradient(points):
        """The exact gradient of ``evaluate``, by hand, in PHYSICAL coordinates.

        The chain rule through the shift-and-scale contributes the ``1 / extent``: the polynomial is
        written in the normalized ``u``, so its derivative there is with respect to ``u``.
        """
        u = (points - centre) / extent
        d0 = 1.7 + 0.8 * u[:, 0] + 0.9 * u[:, 1]
        d1 = -1.1 + 0.9 * u[:, 0]
        if u.shape[1] == 2:
            return np.stack([d0, d1], axis=-1) / extent
        d1 = d1 - 0.3 * u[:, 2]
        d2 = 0.8 + 1.0 * u[:, 2] - 0.3 * u[:, 1]
        return np.stack([d0, d1, d2], axis=-1) / extent

    return jnp.asarray(evaluate(x)), jnp.asarray(evaluate(xf)), analytic_gradient(x)


@eqx.filter_jit
def _reconstruct(scheme, field, mesh, geometry, bvals):
    """The reconstruction, compiled -- which is the only form whose cost means anything.

    A solver calls this from inside a compiled step, where the sweep is unrolled into one program
    and the compiler assigns buffers across it. Run eagerly instead, every intermediate of every
    sweep is a separate live array and each ``(n_faces, dim, dim)`` one is gigabytes, so an eager
    ladder measures dispatch and allocator behaviour rather than the scheme.

    ⚠️ It does NOT bias the arms in the direction one would guess. Measured on a 1.6M-cell mesh, the
    compiled corrected-Green-Gauss arm sped up 2.8x against the compiled Hessian-corrected arm's
    1.7x, so the EAGER comparison flattered the expensive scheme -- 100x against a compiled 163x at
    equal settings. Both schemes sweep, so "the swept arm keeps more intermediates" does not
    separate them; what does is that one nests a solve inside the other's operator.
    """
    return scheme.gradients(field, mesh, geometry, bvals)


def reconstruct(scheme, field, mesh, geometry, bvals, label):
    """Reconstruct and report wall time, so the calibration also prices each arm.

    Compilation is timed and reported separately from execution because they are charged
    differently: a solver pays the compile once per shape and the run once per step, so folding
    them together would price a scheme by a cost that amortizes to nothing.
    """
    started = time.perf_counter()
    # JAX dispatches asynchronously, so the result has to be waited on before the clock is read --
    # without this the ladder's timings measure dispatch and rank every arm as equally fast.
    gradient = _reconstruct(scheme, field, mesh, geometry, bvals).block_until_ready()
    compiled = time.perf_counter() - started

    started = time.perf_counter()
    gradient = _reconstruct(scheme, field, mesh, geometry, bvals).block_until_ready()
    elapsed = time.perf_counter() - started

    gradient = np.asarray(gradient)
    print(
        f"    {label:<34s} {elapsed:7.2f} s  (+{compiled - elapsed:6.1f} s compile)"
        f"   peak RSS {peak_rss_gb():5.2f} GB",
        flush=True,
    )
    return gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Optional, with a ``UV_MESH`` fallback: this is written for a mesh large enough to need
    # ``validation/run_case.sh``, and that runner invokes a script with NO arguments -- so a
    # positional-only mesh path would make the one script that most needs the runner the one script
    # that cannot use it.
    parser.add_argument(
        "mesh",
        type=Path,
        nargs="?",
        default=os.environ.get("UV_MESH"),
        help="path to a polyMesh directory (or set UV_MESH)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="reference the inner ladder against an exact Krylov solve instead of a long sweep",
    )
    parser.add_argument(
        "--outer-sweeps",
        type=int,
        default=int(os.environ["UV_OUTER_SWEEPS"]) if os.environ.get("UV_OUTER_SWEEPS") else None,
        help="calibrate the swept OUTER solve up to this many sweeps (default: skip)",
    )
    args = parser.parse_args()
    if args.mesh is None:
        raise SystemExit("no mesh: pass a polyMesh directory, or set UV_MESH")
    args.mesh = Path(args.mesh)

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

    field, bvals, analytic = probe_field(geometry)

    # ORDER: the ladder runs CHEAPEST-FIRST and the reference runs LAST, which is the opposite of
    # how this reads. The reference is the single most expensive arm -- it is the longest sweep by
    # construction -- so running it first means a node that cannot afford it reports NOTHING, having
    # spent the whole run proving only that the reference is too big. Measured here: a 24-sweep
    # reference on a 1.6M-cell mesh was killed during compilation, discarding a ladder that would
    # every one have fitted. Ladder gradients are (n_cells, dim) and cost ~39 MB each, so holding
    # them all to compare at the end is free next to one more reconstruction.
    print("\ninner (Hessian) solve -- reconstruction vs a converged reference", flush=True)
    print("  inner sweeps (cheapest first; the reference runs last)", flush=True)
    default_sweeps = HessianCorrectedGradient().hessian_solver.sweeps
    measured: list[tuple[int, np.ndarray]] = []
    for sweeps in sorted(INNER_LADDER):
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
        # Reported against the previous rung as we go, so a run that dies later still shows where
        # the iteration was settling. This is a self-consistency measure, not an accuracy one: it
        # says the sweeps stopped moving, which a stalled iteration also satisfies. The reference
        # below is what turns it into an accuracy statement.
        if measured:
            prev_sweeps, prev = measured[-1]
            step = float(np.abs(gradient - prev).max()) / max(float(np.abs(gradient).max()), 1e-300)
            print(f"      moved {step:.3e} since {prev_sweeps} sweeps", flush=True)
        measured.append((sweeps, gradient))

    # AGAINST THE ANALYTIC GRADIENT, which needs no reference and so cannot be quietly wrong.
    # Everything else here is a solve compared with a solve, and when two solves disagree -- on this
    # mesh the outer sweep and an exact Krylov solve differ by 3.6e-01 -- no such comparison can say
    # which one is at fault. The scheme is defined to reconstruct a quadratic exactly, so a departure
    # here is real error whatever produced it. ⚠️ On WARPED faces it is NOT solver error: on a warped
    # 8^3 grid at planarity 0.89 a fully converged solve reconstructs this field to only 1.0e-01,
    # because the derivation assumes a constant normal per face. Read this beside the planarity census.
    print("\n  every arm against the ANALYTIC gradient (relative: median / p99 / max)", flush=True)
    for sweeps, gradient in measured:
        med, p99, worst = gradient_error(gradient, analytic)
        print(
            f"    inner sweeps = {sweeps:<3d} vs analytic = {med:.3e} / {p99:.3e} / {worst:.3e}",
            flush=True,
        )

    print("\n  the reference, and the ladder measured against it", flush=True)
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
        # Graded against what the ladder is actually READ at, not against machine precision. A
        # fixed-sweep reference is never exact, so a machine-precision test reports NOT CONVERGED on
        # every real mesh and sends the reader to the far dearer Krylov reference for nothing --
        # measured here at a drift of 4e-09, which resolves every rung this ladder reports. What
        # matters is whether the reference's own error is small beside the departures being read
        # off it, so the verdict names the resolution it buys and lets the reader judge.
        if drift < 1e-13:
            verdict = "exact to machine precision"
        elif drift < 1e-6:
            verdict = f"fine -- resolves departures down to ~{max(drift * 10, 1e-15):.0e}"
        else:
            verdict = "TOO COARSE for this ladder -- raise UV_REFERENCE_SWEEPS, or use --exact"
        print(f"    reference self-check drift = {drift:.3e}  ({verdict})", flush=True)

    print("\n  inner sweeps -> relative departure from the reference gradient", flush=True)
    for sweeps, gradient in measured:
        error = float(np.abs(gradient - reference).max()) / scale
        print(f"    inner sweeps = {sweeps:<3d} relative departure = {error:.3e}", flush=True)

    if args.outer_sweeps:
        print(
            "\nouter (Schur) solve -- swept against an exact Krylov solve, at a pinned inner count",
            flush=True,
        )
        # The reference is an EXPLICIT Krylov outer solve. It cannot be the constructor default:
        # the default outer solver is a 20-sweep Richardson, so referencing the ladder against
        # `HessianCorrectedGradient()` would compare the swept solve against itself and report a
        # departure of zero at 20 sweeps -- an arm that grades its own homework.
        inner = SweptGradientSolve(sweeps=OUTER_INNER_SWEEPS, warn_tol=None)
        krylov = reconstruct(
            HessianCorrectedGradient(solver=GmresGradientSolve(), hessian_solver=inner),
            field,
            mesh,
            geometry,
            bvals,
            f"outer: Krylov (exact), inner {OUTER_INNER_SWEEPS}",
        )
        arms = [
            (sweeps, relax)
            for relax in OUTER_RELAXATIONS
            for sweeps in OUTER_LADDER
            if sweeps <= args.outer_sweeps
        ]
        for sweeps, relax in arms:
            gradient = reconstruct(
                HessianCorrectedGradient(
                    solver=SweptGradientSolve(sweeps=sweeps, warn_tol=None, relaxation=relax),
                    hessian_solver=inner,
                ),
                field,
                mesh,
                geometry,
                bvals,
                f"outer {sweeps:<3d} relax {relax:<4.2f} (inner {OUTER_INNER_SWEEPS})",
            )
            error = float(np.abs(gradient - krylov).max()) / max(
                float(np.abs(krylov).max()), 1e-300
            )
            print(f"      relative departure from Krylov = {error:.3e}", flush=True)

    # ⚠️ AGAINST AN EXACT GRADIENT, NOT AGAINST THE LADDER'S REFERENCE. The reference above is built
    # with the OUTER solver left at its default, and the outer solve is the half of this scheme that
    # can fail to converge -- measured on a 1.6M-cell mesh at 3.6e-01 from exact after 20 sweeps,
    # oscillating rather than settling. Referencing the replaced scheme against that would report the
    # outer solve's own error as though it were the accuracy difference between the two schemes, and
    # the difference this comparison exists to measure is the one an exact solve would give.
    print("\nis the scheme worth it -- both measured against an EXACT gradient", flush=True)
    exact = reconstruct(
        HessianCorrectedGradient(
            solver=GmresGradientSolve(),
            hessian_solver=SweptGradientSolve(sweeps=REFERENCE_SWEEPS, warn_tol=None),
        ),
        field,
        mesh,
        geometry,
        bvals,
        f"exact: Krylov outer, inner {REFERENCE_SWEEPS}",
    )
    exact_scale = max(float(np.abs(exact).max()), 1e-300)

    shipped = reconstruct(
        HessianCorrectedGradient(),
        field,
        mesh,
        geometry,
        bvals,
        "Hessian-corrected, shipped default",
    )
    print(
        f"      shipped default departs from exact by "
        f"{float(np.abs(shipped - exact).max()) / exact_scale:.3e}"
        f"   |  vs ANALYTIC {format_error(gradient_error(shipped, analytic))}",
        flush=True,
    )
    print(
        f"      exact solve            vs ANALYTIC {format_error(gradient_error(exact, analytic))}",
        flush=True,
    )

    print("\nfor scale -- the scheme this one replaces, on the same field", flush=True)
    corrected = reconstruct(
        CorrectedGreenGauss(), field, mesh, geometry, bvals, "CorrectedGreenGauss (default)"
    )
    print(
        f"      departs from exact by {float(np.abs(corrected - exact).max()) / exact_scale:.3e}"
        f"   |  vs ANALYTIC {format_error(gradient_error(corrected, analytic))}",
        flush=True,
    )
    print(
        "\nRead the ladder for the first sweep count whose departure is below the accuracy you need;\n"
        "the shipped default is calibrated on much smaller meshes and is not a guarantee here.",
        flush=True,
    )


if __name__ == "__main__":
    main()
