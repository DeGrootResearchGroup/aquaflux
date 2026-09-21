"""The live factors evaluated at ONE point per pair, and the parameter that bounds each.

``build_transfer`` integrates the geometric term over the receiving facet, but three factors
multiply it elementwise at a single point per pair, because they are live and differentiable and
folding them inside the quadrature would freeze them. One of the three -- the occlusion mask --
is measured and priced elsewhere. This harness is about the other two, which share a structure:

**A quantity that varies across a facet, evaluated at the facet's centroid, multiplying a
geometric term that was integrated exactly.**

1. **Absorption**, ``exp(-a r)`` off one centroid-to-centroid separation.
2. **A non-Lambertian source profile**, ``radiance_per_exitance`` at one direction, source
   centroid to receiver centroid.

Both are measured here against a dense integral of the same quantity, so each bias is isolated
from every other approximation in the build -- no mesh refinement and no aggregation map, and so
none of the confounds those bring. Each section carries a case whose bias is **exactly zero by
construction** and must measure as such, which is what makes the rest of its numbers evidence:
``a = 0`` for absorption, and the Lambertian ``n = 1`` for the profile, where the cancellation
against the projected solid angle is exact at every geometry.

⚠️ **NEITHER BIAS HAS A FIXED SIGN, AND THE FIRST VERSION OF THIS FILE ASSERTED THAT BOTH WERE
NEGATIVE.** The reasoning was Jensen's inequality on a convex factor, which is sound and is not
the leading term: the centroid separation is not the mean separation, and the centroid direction
is not the mean direction. Measured, both biases run one way for near-axial pairs and the other
way for grazing ones, and the sign changes inside a single scene. **A bound here is on the
magnitude; anyone treating either as a one-way correction will over-shoot it.**

Run with ``validation/run_case.sh validation/radiation_one_point_factors.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
from tests.unit.radiation_references import (
    absorption_bias,
    mean_separation_excess,
    profile_bias,
    profile_cumulant_bias,
    unit_facet,
)

#: Facet separations to sweep, as multiples of the facet's own length scale. A pair of
#: neighbours in a meshed enclosure sits near the top of this range, which is where both biases
#: are worst -- the distant pairs that look benign are not the ones that set the error.
SEPARATIONS = (1.0, 2.0, 4.0, 8.0)


def convergence() -> None:
    """Is the dense reference converged? One that is still moving judges nothing."""
    print("\n## 0. Is the dense reference converged?\n", flush=True)
    source = unit_facet([0.0, 0.0, 0.0], 1.0)
    receiver = unit_facet([0.0, 0.0, 1.0], 1.0, facing_down=True)
    point = np.array([0.0, 0.0, 1.0])
    print(f"{'k':>4} {'absorption, a*w=0.3':>21} {'profile, n=8':>15}", flush=True)
    for k in (6, 12, 24, 36):
        print(
            f"{k:4d} {absorption_bias(source, receiver, 0.3, k):21.8f} "
            f"{profile_bias(source, point, 8.0, k):15.8f}",
            flush=True,
        )
    print("\n-- 24 is the default and is converged to the figures these sweeps report.", flush=True)


def absorption() -> None:
    """Sweep ``a * w`` and separation, and test the mechanism against the bias."""
    print("\n## 1. Absorption\n", flush=True)
    print(
        "`w` is sqrt(area), the module's own length scale. `a * w = 0` is the control.\n"
        "Negative means the shipped form transmits too much -- too BRIGHT.\n",
        flush=True,
    )
    source = unit_facet([0.0, 0.0, 0.0], 1.0)

    print(f"{'a * w':>7}" + "".join(f"{f'gap={g:g}w':>12}" for g in SEPARATIONS), flush=True)
    for coefficient in (0.0, 0.03, 0.1, 0.3, 1.0):
        row = []
        for gap in SEPARATIONS:
            receiver = unit_facet([0.0, 0.0, gap], 1.0, facing_down=True)
            row.append(100.0 * (absorption_bias(source, receiver, coefficient) - 1.0))
        print(f"{coefficient:7.2f}" + "".join(f"{v:11.3f}%" for v in row), flush=True)

    print(
        "\n⚠️ The bias is FIRST order in `a * w`, not second. The leading term is not Jensen's\n"
        "inequality on `exp` -- it is that the centroid separation is not the separation the\n"
        "factor is averaged over, which differs by an amount set by the facet's own extent.\n"
        "Predicted bias `-a * (<r> - r_centroid)`, and the two agree to four figures:\n",
        flush=True,
    )
    print(f"{'gap / w':>8} {'(<r> - r_c) / w':>17} {'measured slope':>16}", flush=True)
    for gap in SEPARATIONS:
        receiver = unit_facet([0.0, 0.0, gap], 1.0, facing_down=True)
        excess = mean_separation_excess(source, receiver)
        slope = (absorption_bias(source, receiver, 0.01) - 1.0) / 0.01
        print(f"{gap:8.1f} {excess:17.4f} {slope:16.4f}", flush=True)

    print(
        "\n-- `<r> - r_centroid` falls like `w**2 / (4 d)`, so the bias is `(a w) * (w / 4d)`:\n"
        "   first order in the absorbance across a facet, damped by the pair's aspect ratio.\n"
        "   Worst between NEIGHBOURS, where d ~ w and it reaches about `0.15 * a * w`.",
        flush=True,
    )


def absorption_off_axis() -> None:
    """The sign flips with lateral offset, which is why the bound is on the magnitude."""
    print("\n## 2. Absorption: the sign is not fixed\n", flush=True)
    source = unit_facet([0.0, 0.0, 0.0], 1.0)
    print(f"{'offset':>8}" + "".join(f"{f'gap={g:g}w':>12}" for g in SEPARATIONS), flush=True)
    for offset in (0.0, 0.5, 1.0, 2.0, 4.0):
        row = []
        for gap in SEPARATIONS:
            receiver = unit_facet([offset, 0.0, gap], 1.0, facing_down=True)
            row.append(100.0 * (absorption_bias(source, receiver, 0.3) - 1.0))
        print(f"{offset:7.1f}w" + "".join(f"{v:11.3f}%" for v in row), flush=True)
    print(
        "\n-- at `a * w = 0.3`. Head-on the centroid separation is the shortest characteristic\n"
        "   distance and the form is too bright; slid sideways, the kernel's `1 / r**2` weight\n"
        "   concentrates on the facing near corners until the separation it averages falls\n"
        "   BELOW the centroid one, and `<r> - r_centroid` -- hence the bias -- changes sign.\n"
        "   Both signs occur inside one enclosure.",
        flush=True,
    )


def profile() -> None:
    """Sweep the exponent and the facet's angular width, with and without the cheap fix."""
    print("\n## 3. A non-Lambertian source profile\n", flush=True)
    print(
        "`n = 1` is the control: Lambertian cancels exactly against the projected solid angle,\n"
        "at every geometry and every distance, so it must read exactly zero.\n"
        "The last column is the residual after the two-moment frozen/live split described in\n"
        "`profile_cumulant_bias` -- two frozen numbers per pair instead of one per sample.\n",
        flush=True,
    )
    source = unit_facet([0.0, 0.0, 0.0], 1.0)
    print(
        f"{'r / w':>7} {'n':>4} {'(n-1)(w/r)^2':>14} {'bias':>10} {'after two moments':>19}",
        flush=True,
    )
    for distance in SEPARATIONS:
        for exponent in (1.0, 2.0, 4.0, 8.0, 16.0):
            point = np.array([0.0, 0.0, distance])
            shipped = profile_bias(source, point, exponent)
            corrected = profile_cumulant_bias(source, point, exponent)
            print(
                f"{distance:7.1f} {exponent:4.0f} {(exponent - 1) / distance**2:14.3f} "
                f"{100 * (shipped - 1):9.2f}% {100 * (corrected - 1):18.3f}%",
                flush=True,
            )
        print(flush=True)
    print(
        "-- the bias collapses onto `(n - 1) * (w / r)**2` with a slope near -0.13 while that\n"
        "   parameter is small, and saturates once it is not. Second order in the angular\n"
        "   width, where absorption's is first order in `a * w`.",
        flush=True,
    )


def profile_off_axis() -> None:
    """Off the axis the profile bias changes sign, as absorption's does."""
    print("\n## 4. The profile bias is not one-way either\n", flush=True)
    source = unit_facet([0.0, 0.0, 0.0], 1.0)
    print(f"{'angle':>7}" + "".join(f"{f'r={r:g}w':>12}" for r in SEPARATIONS), flush=True)
    for degrees in (0, 15, 30, 45, 60, 75):
        radians = np.radians(degrees)
        row = []
        for distance in SEPARATIONS:
            point = distance * np.array([np.sin(radians), 0.0, np.cos(radians)])
            row.append(100.0 * (profile_bias(source, point, 8.0) - 1.0))
        print(f"{degrees:6d}°" + "".join(f"{v:11.2f}%" for v in row), flush=True)
    print(
        "\n-- at `n = 8`, angle measured from the source's own normal. On the axis the centroid\n"
        "   direction sits at the profile's PEAK, so the one-point value is an extreme rather\n"
        "   than an average and the transfer comes out too bright; past about 35° the profile\n"
        "   is convex across the facet's angular span and it goes the other way.",
        flush=True,
    )


if __name__ == "__main__":
    print(__doc__.split("Run with")[0].strip(), flush=True)
    convergence()
    absorption()
    absorption_off_axis()
    profile()
    profile_off_axis()
