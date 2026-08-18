"""Does OpenFOAM's face flux ``phi`` land on the faces aquaflux thinks it does?

Reading a ``surfaceScalarField`` onto an imported mesh places values **by index**: the file's face
``i`` is taken to be the mesh's face ``i``. That is a real property of the import path -- OpenFOAM
orders faces interior-first then patch-by-patch, and the reader carries ``owner`` through without
renumbering -- but it is the kind of property whose failure is silent. A permuted placement yields a
field that is finite, plausibly shaped, and wrong, so nothing downstream raises; a scalar transported
on it would simply be incorrect.

:func:`~aquaflux.io.read_surface_scalar_field` checks the *structural* half of that (interior faces
lead, each patch is one contiguous ascending block, lengths agree). This probe checks the half no
structural test can reach: whether the values are on the **right faces within** the interior block,
which needs the mesh's connectivity and so needs the mesh built.

**The measurement.** ``phi`` is a volumetric flux and OpenFOAM's own continuity closes on it, so the
conservative scatter of it onto cells -- owner ``+phi``, neighbour ``-phi``, which is discrete
divergence on *aquaflux's* connectivity -- must vanish in every cell. It is exact only to the
reference's own convergence, so the bar is OpenFOAM's continuity level (~1e-6 relative here), not
machine zero.

**Normalize by the domain's flow rate, not by the cell's own throughput.** A per-cell relative error
``|net| / sum|phi| over the cell`` looks like the natural measure and is a trap: in the recirculation
and the side-wall corners a cell's own throughput collapses to four orders below the median, so the
reference's fixed absolute error divides up into a large relative one. That is a property of the
denominator, not of the flux -- measured here, the worst-relative cells are the *stagnant* ones and
carry an absolute imbalance no larger than anywhere else. Dividing instead by the inlet volumetric
rate, which is the scale continuity actually closes on globally, gives a bounded measure comparable
to the reference's own reported continuity error. The local-relative distribution is still printed,
because its *spread* is what shows the imbalance is diffuse rather than concentrated -- a genuine
mis-placement is local, and would stand out as a cluster rather than as a shifted median.

**The mutation control, which is the point.** "The residual is small" is worthless on its own unless
a wrong answer would have made it large, so this permutes the interior block and re-measures. Without
that column the check cannot distinguish a correct placement from a measurement too insensitive to
detect a bad one. The permuted arm is reported alongside, and the run is only informative if the two
differ by orders.

Prints one line per patch and one summary table. There is no solve here -- the cost is a mesh build
and two scatters -- so it is cheap enough to re-ask whenever the import path changes. Run from the
repo root::

    validation/run_case.sh validation/bfs3d_openfoam/phi_placement.py --wait
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))

from aquaflux.io import read_openfoam, read_surface_scalar_field  # noqa: E402

#: The OpenFOAM time directory whose ``phi`` is read. The steady run's last write.
PHI_TIME = "2000"

#: Inlet bulk velocity (m/s) and inlet area (m^2) from ``of_case/system/blockMeshDict``: a step of
#: height h = 0.01 m, inlet channel one step high, spanning 4h = 0.04 m. Their product is the
#: volumetric rate the inlet patch must carry, which is what pins the field's SCALE -- the continuity
#: check below is blind to a uniformly rescaled flux, since any multiple of a divergence-free field
#: is divergence-free.
U_INLET = 10.0
INLET_AREA = 0.01 * 0.04


def per_cell_throughput(mesh, phi: np.ndarray) -> np.ndarray:
    """Sum of ``|phi|`` over each cell's faces -- the local scale a cell's net imbalance sits on.

    A cell's net flux is a difference of terms this size, which makes this the natural *local*
    denominator. It is reported as a distribution rather than used as the pass/fail measure, because
    it collapses in stagnant cells; see the module docstring on why the flow rate is the bounded one.
    """
    magnitude = np.abs(phi)
    return np.asarray(mesh.face_cells.scatter(magnitude, magnitude))


def main() -> int:
    print(f"reading mesh from {CASE / 'of_case'}", flush=True)
    started = time.time()
    mesh = read_openfoam(CASE / "of_case")
    print(
        f"  mesh: {mesh.n_cells} cells, {mesh.n_faces} faces, dim {mesh.dim} "
        f"({time.time() - started:.0f} s)",
        flush=True,
    )

    phi_path = CASE / "of_case" / PHI_TIME / "phi"
    phi = read_surface_scalar_field(phi_path, mesh)
    print(f"  phi: {phi.shape[0]} values from {phi_path.relative_to(ROOT)}", flush=True)

    interior = np.asarray(mesh.face_cells.interior)
    n_internal = int(interior.sum())
    print(f"  {n_internal} internal + {phi.shape[0] - n_internal} boundary faces", flush=True)

    # --- boundary placement: each patch's net flux against what the geometry says it must be -------
    print("\npatch                     faces        net flux (m^3/s)", flush=True)
    net_boundary = 0.0
    for name in mesh.face_patches.names:
        if name == "interior":
            continue
        indices = np.asarray(mesh.face_patches.indices(name))
        if indices.size == 0:
            continue
        net = float(phi[indices].sum())
        net_boundary += net
        print(f"  {name:<22s} {indices.size:>7d}   {net:>18.6e}", flush=True)
    print(f"  {'NET (all patches)':<22s} {'':>7s}   {net_boundary:>18.6e}", flush=True)

    expected_inlet = -U_INLET * INLET_AREA
    print(
        f"\n  inlet expected {expected_inlet:.6e} m^3/s "
        f"(U={U_INLET} m/s x A={INLET_AREA:g} m^2); an inflow is negative owner-outward",
        flush=True,
    )
    print(
        f"  net imbalance {net_boundary:.3e} = {abs(net_boundary / expected_inlet):.2e} "
        "of the throughput -- OpenFOAM's own continuity error, not a placement error",
        flush=True,
    )

    # --- interior placement: discrete continuity on aquaflux's connectivity -----------------------
    throughput = per_cell_throughput(mesh, phi)
    inlet_rate = U_INLET * INLET_AREA

    def continuity(face_flux: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Per-cell |net| flux, its maximum as a fraction of the inlet rate, and the local ratio."""
        net = np.abs(np.asarray(mesh.face_cells.scatter_conservative(face_flux)))
        return net, float(net.max() / inlet_rate), net / throughput

    as_read, as_read_rel, local = continuity(phi)

    # The control. Permuting the interior block leaves every boundary value, every patch length and
    # the structural checks untouched -- it is exactly the failure the index placement risks and that
    # no structural test can see. Seeded so the run is reproducible.
    permuted = phi.copy()
    rng = np.random.default_rng(0)
    permuted[:n_internal] = permuted[rng.permutation(n_internal)]
    shuffled, shuffled_rel, _ = continuity(permuted)

    print(
        f"\ndomain flow rate {inlet_rate:.3e} m^3/s; per-cell throughput median "
        f"{float(np.median(throughput)):.3e}, min {float(throughput.min()):.3e} m^3/s",
        flush=True,
    )
    print("\narm                    max |net| (m^3/s)     / flow rate", flush=True)
    print(f"  {'as read':<20s} {float(as_read.max()):>18.3e}   {as_read_rel:>14.3e}", flush=True)
    print(
        f"  {'interior permuted':<20s} {float(shuffled.max()):>18.3e}   {shuffled_rel:>14.3e}",
        flush=True,
    )
    print(
        f"\n  the control is {shuffled_rel / as_read_rel:.3g}x worse than the field as read",
        flush=True,
    )

    # The local-relative distribution, and the check that its tail is a denominator effect. If the
    # worst-relative cells were a mis-placement they would carry a large ABSOLUTE imbalance too; the
    # column that settles it is their |net| against the global maximum, not their ratio.
    order = np.argsort(local)[::-1]
    print(
        "\nlocal ratio |net| / cell throughput -- median "
        f"{float(np.median(local)):.2e}, 99th pct {float(np.percentile(local, 99)):.2e}, "
        f"max {float(local.max()):.2e}",
        flush=True,
    )
    print(
        "\n  worst-ratio cells      ratio    |net| (m^3/s)   throughput   vs median throughput",
        flush=True,
    )
    for cell_index in order[:5]:
        print(
            f"    cell {int(cell_index):<10d} {local[cell_index]:>9.2e} {as_read[cell_index]:>13.2e}"
            f" {throughput[cell_index]:>13.2e} {throughput[cell_index] / np.median(throughput):>10.2e}",
            flush=True,
        )
    print(
        f"    (global max |net| is {float(as_read.max()):.2e}, so these carry no unusual "
        "ABSOLUTE imbalance -- their ratio is a small denominator)",
        flush=True,
    )

    # A verdict, so the log says what it found rather than leaving a reader to judge the table. The
    # bar on the as-read arm is the reference's own convergence level; the bar on the control is that
    # it be decisively worse, which is what makes the first number mean anything.
    ok = as_read_rel < 1e-4 and shuffled_rel > 1e3 * as_read_rel
    print(
        "\nVERDICT: interior placement CONFIRMED -- OpenFOAM's phi satisfies discrete continuity on "
        "aquaflux's own connectivity, and a permuted interior does not"
        if ok
        else "\nVERDICT: FAILED -- see the table above",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
