"""Which boundary closure does the multiple-correction reconstruction need *on this mesh*?

:class:`~aquaflux.schemes.MultipleCorrectionGradient` closes the gradient on a boundary face with an
injected :class:`~aquaflux.schemes.GradientBoundaryClosure`, and the two shipped choices fail in
opposite regimes. :class:`~aquaflux.schemes.OwnerGradient` reads no boundary value at all, which
makes it immune to a boundary value that is itself a closure -- but on a boundary **tetrahedron** it
supplies no direction the cell did not already have, and the six Hessian components are left
underdetermined. :class:`~aquaflux.schemes.SkewCorrectedGradient` supplies that direction from the
boundary value, and is the closure a coupled RANS march on `pitzdaily_gradient_ab` stalls under.

Which of those matters here is a property of *this* mesh, so it is measured rather than argued. The
reactor is snappyHexMesh rather than tetrahedral, but its cut cells do produce four-faced cells, and
a census of the connectivity is printed alongside the reconstruction so the two can be read together.

The field is a **quadratic**, reused from :mod:`gradient_sweep_calibration` -- the field this scheme
is *defined* to reproduce exactly, so any error is the scheme's and not the discretization's. The
error is reported as median / p99 / max for the reason that module gives: on a million-cell mesh a
max-norm alone is set by one sliver cell, and the gap between the three is itself the diagnosis.

Usage
-----
    python3 validation/uvreactor_openfoam/multiple_correction_closures.py <path-to-polyMesh-dir>

or set ``UV_MESH`` and run it through ``validation/run_case.sh``, which is the way to run it on the
full mesh. Each closure is built and released before the next, so peak memory is one closure's.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from gradient_sweep_calibration import (  # noqa: E402  (the quadratic probe and its error measure)
    format_error,
    gradient_error,
    probe_field,
)

#: The closures to compare, in the order reported. Both are shipped; the first is the default.
CLOSURES = (("OwnerGradient", OwnerGradient), ("SkewCorrectedGradient", SkewCorrectedGradient))


def peak_rss_gb() -> float:
    """Peak resident set size in GB (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def cell_census(mesh) -> None:
    """How many cells have four faces, and how many of those touch a boundary.

    The boundary tetrahedron is the shape the owner closure cannot determine, so this is the
    structural half of the answer -- printed beside the reconstruction rather than instead of it,
    because a face count does **not** settle which cells are underdetermined: each face carries a
    gradient *vector* rather than one number, so a boundary hexahedron has fewer interior faces than
    there are Hessian components and is perfectly determined. Only the reconstruction settles it.
    """
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    interior = np.asarray(face_cells.interior)
    neighbour = np.asarray(face_cells.neighbour)

    total = np.bincount(owner, minlength=mesh.n_cells)
    total += np.bincount(neighbour[interior], minlength=mesh.n_cells)
    at_boundary = np.zeros(mesh.n_cells, dtype=bool)
    at_boundary[owner[~interior]] = True

    print(f"  faces/cell: min {total.min()} median {int(np.median(total))} max {total.max()}")
    print(f"  cells owning a boundary face: {int(at_boundary.sum())} of {mesh.n_cells}")
    tets = total == 4
    print(
        f"  four-faced cells: {int(tets.sum())}, "
        f"of which at a boundary: {int((tets & at_boundary).sum())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mesh",
        type=Path,
        nargs="?",
        default=os.environ.get("UV_MESH"),
        help="path to a polyMesh directory (or set UV_MESH)",
    )
    args = parser.parse_args()
    if args.mesh is None:
        raise SystemExit("no mesh: pass a polyMesh directory, or set UV_MESH")

    print(f"mesh: {args.mesh}", flush=True)
    started = time.perf_counter()
    mesh = read_openfoam(Path(args.mesh).resolve())
    geometry = mesh.geometry()
    print(
        f"  n_cells={mesh.n_cells} n_faces={mesh.n_faces} dim={mesh.dim} "
        f"({time.perf_counter() - started:.1f} s, peak {peak_rss_gb():.2f} GB)",
        flush=True,
    )
    cell_census(mesh)

    cell_values, face_values, analytic = probe_field(geometry)
    print(
        f"\n{'closure':<24} {'gradient error  median / p99 / max':<40} "
        f"{'max |M2^-1|':>12} {'build':>8} {'peak GB':>8}",
        flush=True,
    )
    for name, closure in CLOSURES:
        build_started = time.perf_counter()
        scheme = MultipleCorrectionGradient(boundary_closure=closure()).bind(mesh, geometry)
        build = time.perf_counter() - build_started
        worst = float(jnp.max(jnp.abs(scheme.prepared.m2_inverse)))
        gradient = scheme.gradients(cell_values, mesh, geometry, face_values)
        stats = gradient_error(np.asarray(gradient), analytic)
        print(
            f"{name:<24} {format_error(stats):<40} {worst:>12.2e} "
            f"{build:>7.1f}s {peak_rss_gb():>7.2f}",
            flush=True,
        )
        del scheme, gradient
        gc.collect()

    print(
        "\nA quadratic is what this scheme reproduces exactly, so a median error far above roundoff "
        "means\nthe closure does not work on this mesh. `max |M2^-1|` is the scheme's own detector: "
        "order one\nwhen the Hessian is determined, ~1e16 when a boundary cell leaves it singular.",
        flush=True,
    )


if __name__ == "__main__":
    main()
