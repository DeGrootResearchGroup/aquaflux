"""The developed inflow: the precursor's profiles of axial velocity, k and omega against radius.

Reads a grid's precursor table (``aquaflux run precursor.yaml`` in
``validation/sudo_bend/grids/<grid>``; by default the coarsest grid's) and draws
``figures/inflow_profiles.png``: the velocity against the one-seventh power law of a smooth pipe
at this Reynolds number, and in wall units against the log law. The wall units are Blasius's
friction velocity at this Reynolds number, so the comparison is not of the profile with itself; the
friction velocity that would put the first cell on the log law is printed beside it, a derived
number.

Usage::

    python validation/sudo_bend/plot_inflow.py [<profiles.csv>]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aquaflux.io import read_profile_table
from geometry import Bend
from scipy.optimize import brentq

HERE = Path(__file__).parent
TABLE = HERE / "grids" / "round36" / "work" / "precursor" / "profiles.csv"
NU = 1.5e-5
REYNOLDS = 6.0e4
KAPPA, B = 0.41, 5.2


def main(path: Path) -> None:
    table = read_profile_table(path)
    radius = Bend().radius
    r, u = table.radii, table.columns["U"]
    rr = np.linspace(0.0, radius, 4001)
    # The bulk velocity the precursor holds (Re = 6.0e4); the profile's own integral is printed beside
    # it, held to the wall beyond the last radius, so a little short.
    bulk = REYNOLDS * NU / Bend().diameter
    integral = np.trapezoid(np.interp(rr, r, u) * 2.0 * rr, rr) / radius**2
    wall = radius - r[-1]
    u_tau = bulk * np.sqrt(0.0791 * REYNOLDS**-0.25 / 2.0)  # Blasius: c_f = 0.0791 Re^-1/4
    fitted = brentq(lambda v: u[-1] / v - (np.log(wall * v / NU) / KAPPA + B), 1e-3, 10.0)
    print(
        f"{r.size} radii; bulk {bulk:.4f} m/s (the profile's integral {integral:.4f}), centreline "
        f"{u[0] / bulk:.3f} U_b; Blasius friction velocity {u_tau:.4f} m/s; the one putting the "
        f"first cell on the log law (derived) {fitted:.4f} m/s",
        flush=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    power = (1.0 - rr / radius) ** (1.0 / 7.0)
    power *= bulk / (np.trapezoid(power * 2.0 * rr, rr) / radius**2)
    axes[0].plot(r / radius, u / bulk, "o-", ms=4, label="precursor (k-omega SST)")
    axes[0].plot(rr / radius, power / bulk, "--", label="1/7 power law, same bulk")
    axes[0].set(xlabel="r / R", ylabel="U / U_b", title="axial velocity")
    axes[0].legend()
    y = radius - r
    yplus = y * u_tau / NU
    axes[1].semilogx(yplus, u / u_tau, "o-", ms=4, label="precursor")
    grid = np.logspace(1, np.log10(yplus.max()), 50)
    axes[1].semilogx(grid, np.log(grid) / KAPPA + B, "--", label="log law")
    axes[1].set(xlabel="y+", ylabel="U+", title=f"wall units, Blasius u_tau = {u_tau:.3f} m/s")
    axes[1].legend()
    axes[2].plot(r / radius, table.columns["k"] / u_tau**2, "o-", ms=4)
    twin = axes[2].twinx()
    twin.semilogy(r / radius, table.columns["omega"], "s-", ms=3, color="C3", label="omega")
    axes[2].set(xlabel="r / R", title="turbulence: k (blue, left), omega (red, right)")
    axes[2].set_ylabel("k / u_tau^2", color="C0")
    twin.set_ylabel("omega (1/s)", color="C3")
    fig.suptitle("The developed inflow at Re = 6.0e4, averaged by radius over the periodic pipe")
    fig.tight_layout()
    out = HERE / "figures" / "inflow_profiles.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else TABLE)
