"""How wrong is the radiation transfer's BINARY per-pair occlusion mask?

``build_visibility`` casts one ray per facet pair, source centroid to receiver centroid, and
records the pair as wholly blocked or wholly clear. With the source side of the transfer in
closed form and the receiver side integrated over six points, that bit is the only
all-or-nothing term left in the surface system — and it is all-or-nothing in exactly the
geometry the package exists for: a lamp sleeve, a baffle, a duct shadowing itself.

This measures the error before anything is designed to fix it, which is what issue #447 asks
for. Nothing here proposes a treatment.

**The instrument.** Two 2 m square plates facing each other across a 2 m gap, with an opaque
cylinder on the axis between them. Each plate is meshed into ``n x n`` quads (two triangles
each). The reference meshes the same plates 36 per side, where a sub-pair's own bit is nearly
exact, and area-averages its transfer back onto the coarse patches — which is the definition of
the coarse form factor, so the comparison is against the right quantity rather than against a
finer answer to a different question.

**The control that makes it a measurement of the mask and not of the mesh.** Run with the
cylinder removed, the same coarse-against-aggregated-reference comparison reads **5.7e-07**.
That is the instrument's own floor — the difference between six quadrature points on one large
receiving triangle and six on each of many small ones — so everything above it is the mask.

**Read the mean, not the maximum.** The reference's shadow edge is itself a staircase of
sub-facet bits, so the worst single pair is a lottery in how the two meshes happen to land
against that edge; it does not settle as the reference is refined (0.123 / 0.151 / 0.135 /
0.145 / 0.144 at 2x / 3x / 4x / 6x / 8x). The mean settles from 4x upward (0.0128 / 0.0223 /
0.0193 / 0.0206 / 0.0209), so the reference is converged for the mean to about 5% of itself and
is not converged for the maximum at all.

Run it with ``validation/run_case.sh validation/radiation_partial_occlusion.py``; it takes a
few minutes and needs no OpenFOAM case.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
from aquaflux.radiation import (
    Cylinder,
    RadiationSettings,
    Surfaces,
    build_radiation_model,
    build_transfer,
    fluence_rate,
)
from tests.unit.radiation_references import area_average_onto, facing_plates

#: Plate half-width, and the half-gap between them, in metres. A square aperture of the same
#: size as its separation, so neither the near nor the far field dominates.
HALF = 1.0
GAP = 1.0

#: Quads per side of the reference plates. 5184 facets, which is where the mean of the error
#: stops moving; see the module docstring.
REFERENCE = 36


def plates(n: int) -> Surfaces:
    """The two facing plates as one emitting set, both fully emitting and non-reflecting."""
    return Surfaces.from_triangles(
        facing_plates(n, half=HALF, gap=GAP), emission=1.0, reflectance=0.0
    )


def cylinder(radius: float) -> Cylinder:
    """An opaque rod across the gap, on the axis, long enough to clear both plates."""
    return Cylinder(centre=[0, 0, 0], axis=[0, 1, 0], radius=radius, half_length=4.0)


def effective_transfer(n: int, occluders) -> tuple[np.ndarray, np.ndarray]:
    """The transfer a solve actually uses: frozen geometry times the live mask, fully opaque."""
    surfaces = plates(n)
    transfer = build_transfer(surfaces, occluders=occluders, self_occlusion=False)
    reflected, _ = transfer.assemble(
        surfaces, transmittance=np.zeros(len(occluders)) if occluders else None
    )
    return np.asarray(reflected), np.asarray(surfaces.area)


def mask_error(n_coarse: int, reference, reference_area, radius: float) -> dict[str, float]:
    """Coarse against the aggregated reference, normalized by the largest reference entry."""
    coarse, coarse_area = effective_transfer(n_coarse, (cylinder(radius),))
    got = area_average_onto(coarse, coarse_area, n_coarse, n_coarse)
    want = area_average_onto(reference, reference_area, REFERENCE, n_coarse)
    scale = float(np.max(want))
    return {
        "max": float(np.max(np.abs(got - want))) / scale,
        "mean": float(np.mean(np.abs(got - want))) / scale,
        "row_sum": float(np.max(np.abs(got.sum(1) - want.sum(1)))),
    }


def field_on_a_line(n: int, occluders, probes: np.ndarray) -> np.ndarray:
    """Fluence rate at ``probes`` from the plates, with whatever mask the mesh produced."""
    surfaces = plates(n)
    model = build_radiation_model(
        probes, surfaces, occluders=occluders, settings=RadiationSettings(self_occlusion=False)
    )
    value, _ = fluence_rate(
        model, surfaces, transmittance=np.zeros(len(occluders)) if occluders else None
    )
    return np.asarray(value)


def main() -> None:
    started = time.time()
    print(f"Plates {2 * HALF} m square at z = +-{GAP} m; reference {REFERENCE} quads per side.")
    print("Error normalized by the largest reference transfer entry.\n", flush=True)

    control, control_area = effective_transfer(REFERENCE, ())
    coarse, coarse_area = effective_transfer(4, ())
    got = area_average_onto(coarse, coarse_area, 4, 4)
    want = area_average_onto(control, control_area, REFERENCE, 4)
    floor = float(np.max(np.abs(got - want)) / np.max(want))
    print(f"CONTROL, nothing in the way, 4x4 against the reference: max {floor:.3e}")
    print("  -- the instrument's own floor; everything below is the mask.\n", flush=True)

    for radius in (0.15, 0.30, 0.60):
        reference, reference_area = effective_transfer(REFERENCE, (cylinder(radius),))
        print(f"cylinder radius {radius} m across a {2 * HALF} m gap")
        print("   coarse    max      mean     row-sum drift", flush=True)
        for n_coarse in (2, 3, 4, 6, 9, 12):
            e = mask_error(n_coarse, reference, reference_area, radius)
            print(
                f"   {n_coarse:2d}x{n_coarse:<2d}   {e['max']:.4f}   {e['mean']:.4f}   "
                f"{e['row_sum']:.4f}",
                flush=True,
            )
        print(flush=True)

    print("Fluence rate on a line at z = 0.5 m, clear of the body, radius 0.30 m")
    probes = np.stack([np.linspace(-0.95, 0.95, 15), np.zeros(15), np.full(15, 0.5)], axis=1)
    reference_field = field_on_a_line(REFERENCE, (cylinder(0.30),), probes)
    unblocked = field_on_a_line(REFERENCE, (), probes)
    shadow = unblocked - reference_field
    print("      x     clear   reference   shadow    n=2      n=4      n=8", flush=True)
    coarse_fields = {n: field_on_a_line(n, (cylinder(0.30),), probes) for n in (2, 4, 8)}
    for i, x in enumerate(probes[:, 0]):
        cells = "".join(
            f"  {100 * (coarse_fields[n][i] - reference_field[i]) / reference_field[i]:+6.2f}%"
            for n in (2, 4, 8)
        )
        print(
            f"   {x:+.2f}  {unblocked[i]:7.4f}  {reference_field[i]:8.4f}  "
            f"{shadow[i]:7.4f} {cells}",
            flush=True,
        )
    for n in (2, 4, 8):
        gap = np.abs(coarse_fields[n] - reference_field)
        print(
            f"   n={n}: worst {100 * np.max(gap / reference_field):5.2f}% of the field, "
            f"{100 * np.max(gap / shadow):5.1f}% of the shadow it is modelling; "
            f"mean {100 * np.mean(gap / reference_field):.2f}%",
            flush=True,
        )
    print(f"\n({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
