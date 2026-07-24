"""pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST.

A **same-mesh, cell-for-cell** cross-code validation. The OpenFOAM pitzDailySteady tutorial (its RAS
model switched from the shipped kEpsilon to kOmegaSST) is run in the openfoam13 container by
``of_case/run_of.sh``, which writes the converged fields, the mesh, and the SIMPLE residual history to
``runs/kwsst/``. This script then reads that **same mesh** into aquaflux via ``read_openfoam`` and
solves the coupled RANS system on it, so the two solutions live on identical cells and are compared
directly (no interpolation between independent meshes, unlike ``validation/turbulent_channel_openfoam``,
whose cyclic mesh the reader cannot yet import).

aquaflux setup, as requested for this study:

* the **coupled** turbulent solver (:func:`aquaflux.turbulence.solve_coupled` -- one monolithic Newton
  on ``R(u, p, k, omega)``, globalized by pseudo-transient continuation);
* **hybrid initialization** (potential-flow velocity + Laplace-smoothed turbulence), which
  ``solve_coupled`` invokes automatically to self-start;
* **second-order upwind** momentum advection (:class:`aquaflux.discretization.LimitedUpwind` with the
  :class:`aquaflux.schemes.VenkatakrishnanLimiter` -- the upwind cell reconstructed to the face with
  its gradient, slope-limited so the reconstruction stays bounded). The stiff k/omega scalars use
  bounded first-order upwind: a second-order stencil there lets the coupled Newton step drive omega
  negative (a Newton-update, M-matrix effect the limiter does not prevent -- see ``solve_aquaflux``);
* **corrected Green-Gauss** gradients (:class:`aquaflux.schemes.CorrectedGreenGauss`, the
  skewness/non-orthogonality-corrected reconstruction -- the analogue of OpenFOAM's ``corrected``
  surface-normal / non-orthogonal treatment);
* **log-variable omega** (:class:`aquaflux.turbulence.LogScalars` on ``omega_transform``): ``omega =
  e^w`` stays strictly positive under any Newton step. Without it a direct-omega step drives omega
  negative once the recirculation forms, poisoning ``nu_t = k/omega`` while the residual stays finite
  (so the divergence guard never trips) -- the failure this case exposes and log-omega structurally
  removes.

The physics caveat this study documents: the pitzDaily mesh is a **wall-function** mesh (first-cell
``y+`` well above the viscous sublayer), whereas aquaflux's SST is **wall-resolving** (it fixes the
analytical sublayer ``omega`` at the wall-adjacent cell). The comparison therefore isolates the *outer*
flow -- the shear-layer development, the recirculation bubble, and the reattachment length -- where the
near-wall treatment matters least, and reports the near-wall fields as the expected point of departure.

**Cost note (binding for whoever runs this):** the coupled log-omega solve on the full ~12k-cell mesh
is compute-heavy -- each Newton step is several minutes and the march is long, so a full run is a
matter of hours. omega-log is validated on a smaller channel (``tests/integration/test_coupled_rans``);
efficient large-mesh convergence (the reparametrized-block preconditioner scaling and the
globalization) is a known tuning follow-up. Track the **per-field relative** residuals when running --
the absolute ``||R||`` is dominated by omega's ~1e5 scale and is a misleading convergence metric.

Run (after ``run_of.sh``) from the repo root:
    python3 validation/pitzdaily_openfoam/compare.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, SweptGradientSolve, VenkatakrishnanLimiter
from aquaflux.turbulence import CoupledRANS, LogScalars, SSTModel, SSTTurbulence, solve_coupled

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "kwsst"
FIGS = HERE / "figures"

# The pitzDaily operating point (0/ and constant/): U_in = 10 m/s, nu = 1e-5, k_in = 0.375,
# omega_in = 440.15. rho = 1 (incompressible kinematic).
RHO, NU = 1.0, 1e-5
U_IN, K_IN, OMEGA_IN = 10.0, 0.375, 440.15
WALLS = ["upperWall", "lowerWall"]
STEP_X, STEP_Y = 0.0, 0.0  # the step lip; the lower wall drops to y = -0.0254 for x > 0
# The coupled Newton march budget. This is a stiff, separating, high-Re case on a wall-function mesh
# (aquaflux's SST is wall-resolving), so it converges to an engineering tolerance rather than machine
# zero; the cap is generous so the march exits on the tolerance, not the count.
MAX_STEPS, RTOL = 200, 1e-6


# --- OpenFOAM ascii internalField parsing (nonuniform scalar / vector list) ---
def _of_scalar(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    return np.array(re.findall(r"[-+]?\d[\d.eE+-]*", body)[: int(m.group(1))], dtype=float)


def _of_vector(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    tri = re.findall(r"\(([^)]*)\)", body)[: int(m.group(1))]
    return np.array([[float(v) for v in s.split()] for s in tri], dtype=float)


def read_openfoam_reference():
    """The OpenFOAM converged fields + cell centres, keyed to their cell centroids."""
    ccx, ccy = _of_scalar(RUNS / "Ccx"), _of_scalar(RUNS / "Ccy")
    return dict(
        centroid=np.column_stack([ccx, ccy]),
        U=_of_vector(RUNS / "U")[:, :2],
        p=_of_scalar(RUNS / "p"),
        k=_of_scalar(RUNS / "k"),
        omega=_of_scalar(RUNS / "omega"),
        nut=_of_scalar(RUNS / "nut"),
    )


def build_case():
    """Assemble the benchmark: mesh, momentum, turbulence and the coupled residual -- no solve.

    Split out from :func:`solve_aquaflux` so a solver study can re-solve at a saved state (a
    mid-march checkpoint, say) without re-marching to it, and without restating the case. The
    mesh import, boundary conditions, model constants and scheme choices *are* the definition of
    this benchmark; a second copy of them would drift from the one the validation figures use.

    Returns
    -------
    dict
        ``coupled``, ``momentum``, ``turbulence`` and ``geom`` for the assembled case.
    """
    mesh = read_openfoam(RUNS / "polyMesh")
    geom = mesh.geometry()
    # Corrected (non-orthogonal / skewness) Green-Gauss gradients. Its A_g^-1 apply is the default O(n)
    # matrix-free swept solve, not a nested GMRES: identical discretization, but it avoids a nested
    # Krylov solve (carrying its own implicit-diff tangent) inside every coupled-residual evaluation,
    # which otherwise dominates the monolithic Newton cost on this ~12k-cell mesh (measured ~180x per
    # residual eval here).
    grad = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=8))
    # Momentum advection: second-order upwind = Venkatakrishnan-limited linear upwind (the upwind cell
    # reconstructed to the face with its corrected-Green-Gauss gradient, slope-limited so the
    # reconstruction is monotonicity-bounded) -- the analogue of OpenFOAM's `Gauss linearUpwind`.
    momentum_upwind = LimitedUpwind(limiter=VenkatakrishnanLimiter())
    # Turbulence advection: first-order upwind on k and omega. The slope limiter bounds the advective
    # *face value*, but the negative-omega failure of second-order on the stiff omega equation is a
    # Newton-*update* overshoot at the cell centre, not a face-value one: first-order upwind makes the
    # omega transport operator diagonally dominant (an M-matrix) so the pseudo-transient-shifted Newton
    # step preserves positivity, whereas a second-order stencil -- even limited -- weakens that
    # dominance and lets the update drive omega < 0 (then nu_t = k/omega flips sign and poisons the
    # closure while the residual stays finite, so the divergence guard never trips). The structural fix
    # for second-order scalars is log-variable transport (omega = e^w), which is not built here.
    scalar_upwind = FirstOrderUpwind()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
            }
        ),
        advection_scheme=momentum_upwind,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                "upperWall": Dirichlet(0.0),
                "lowerWall": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
            }
        ),
    )
    # Log-transform omega: omega = e^w stays strictly positive under any Newton step. On this stiff
    # separating case a direct-omega step drives omega negative once the recirculation forms (nu_t =
    # k/omega then flips sign and poisons the closure while the residual stays finite, so the divergence
    # guard never trips). k stays direct -- log(k) is ill-conditioned where k -> 0 at the walls.
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    # The monolithic Newton is globalized by the default pseudo-transient continuation: an a_P /
    # transport-diagonal shift that damps each step heavily far from the fixed point and ramps to
    # zero on the residual, recovering the exact steady Newton state at convergence.
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def solve_aquaflux(**solve_kwargs):
    """Solve the coupled RANS system on the imported OpenFOAM mesh; return fields + geometry.

    Parameters
    ----------
    **solve_kwargs
        Forwarded to :func:`~aquaflux.turbulence.solve_coupled`, overriding the defaults set here.
        This is the seam a solver study uses to instrument or reconfigure the march -- an ``on_step``
        observer, a ``refresh_trigger``, a different ``method``.
    """
    case = build_case()
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    solve_options = dict(max_steps=MAX_STEPS, rtol=RTOL) | solve_kwargs
    flow, k, omega = solve_coupled(coupled, **solve_options)
    velocity, pressure = momentum.unpack(flow)
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    return dict(
        centroid=np.asarray(geom.cell.centroid),
        U=np.asarray(velocity),
        p=np.asarray(pressure),
        k=np.asarray(k),
        omega=np.asarray(omega),
        nut=np.asarray(nu_t),
    )


def reattachment_length(centroid, u_x):
    """Lower-wall reattachment length x_r/h behind the step (h = step height = 0.0254 m).

    Reads the sign of the wall-adjacent streamwise velocity along the lower wall downstream of the
    step: the recirculation bubble is where it is negative, and reattachment is the last such x.
    """
    h = 0.0254
    x, y = centroid[:, 0], centroid[:, 1]
    # The wall-adjacent row along the lower wall (y just above the -h floor), downstream of the step.
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    xs = x[band]
    us = u_x[band]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = np.where(us < 0)[0]
    if neg.size == 0:
        return 0.0
    return float(xs[neg[-1]] / h)


def main():
    if not (RUNS / "U").exists():
        raise SystemExit(f"OpenFOAM results not found in {RUNS}; run of_case/run_of.sh first.")
    of = read_openfoam_reference()
    print(
        f"OpenFOAM: {of['centroid'].shape[0]} cells, Ux in "
        f"[{of['U'][:, 0].min():.3f}, {of['U'][:, 0].max():.3f}]",
        flush=True,
    )

    t0 = time.time()
    aq = solve_aquaflux()
    print(
        f"aquaflux coupled solve: {time.time() - t0:.0f}s, "
        f"Ux in [{aq['U'][:, 0].min():.3f}, {aq['U'][:, 0].max():.3f}]",
        flush=True,
    )

    from scipy.spatial import cKDTree

    tree = cKDTree(of["centroid"])
    dist, idx = tree.query(aq["centroid"])
    assert float(dist.max()) < 1e-6, f"mesh mismatch: max centroid distance {dist.max()}"

    def rel_l2(a, b, scale):
        return float(np.sqrt(np.mean((a - b) ** 2)) / scale)

    metrics = dict(
        ux=rel_l2(aq["U"][:, 0], of["U"][idx, 0], U_IN),
        uy=rel_l2(aq["U"][:, 1], of["U"][idx, 1], U_IN),
        xr_aqua=reattachment_length(aq["centroid"], aq["U"][:, 0]),
        xr_of=reattachment_length(of["centroid"], of["U"][:, 0]),
        nut_peak_aqua=float(aq["nut"].max() / NU),
        nut_peak_of=float(of["nut"].max() / NU),
    )
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}", flush=True)

    _figure(of, aq, idx)
    _report(metrics)


def _figure(of, aq, idx):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x, y = aq["centroid"][:, 0], aq["centroid"][:, 1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    vmax = max(abs(of["U"][:, 0]).max(), abs(aq["U"][:, 0]).max())
    for a, data, title in (
        (ax[0], of["U"][idx, 0], "OpenFOAM $U_x$"),
        (ax[1], aq["U"][:, 0], "aquaflux $U_x$"),
    ):
        sc = a.scatter(x, y, c=data, s=3, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    # Reattachment: lower-wall streamwise velocity sign.
    h = 0.0254
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    order = np.argsort(x[band])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(
        x[band][order] / h,
        of["U"][idx][band][order][:, 0],
        "s-",
        ms=3,
        color="C1",
        label="OpenFOAM",
    )
    ax[2].plot(
        x[band][order] / h, aq["U"][band][order][:, 0], ".-", ms=4, color="C0", label="aquaflux"
    )
    ax[2].set_xlabel("$x/h$ behind the step")
    ax[2].set_ylabel("near-wall $U_x$")
    ax[2].set_title("Lower-wall recirculation (sign change = reattachment)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "comparison.png", dpi=130)
    print(f"wrote {FIGS / 'comparison.png'}", flush=True)


def _report(m):
    lines = [
        "# pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST",
        "",
        "The OpenFOAM `pitzDailySteady` tutorial -- its RAS model switched from the shipped `kEpsilon`",
        "to `kOmegaSST` -- run in OpenFOAM, then solved on the **same imported mesh** by aquaflux's",
        "coupled RANS solver (hybrid initialization, second-order upwind momentum advection, corrected",
        "Green-Gauss gradients). U_in = 10 m/s, nu = 1e-5 (Re ~ 25000 on the 25.4 mm inlet).",
        "",
        "## Results",
        "",
        "| quantity | aquaflux | OpenFOAM |",
        "|---|---|---|",
        f"| reattachment length x_r/h (lower wall) | {m['xr_aqua']:.2f} | {m['xr_of']:.2f} |",
        f"| peak nu_t/nu | {m['nut_peak_aqua']:.0f} | {m['nut_peak_of']:.0f} |",
        f"| rel. L2 U_x error (cell-for-cell) | {m['ux']:.3f} | -- |",
        f"| rel. L2 U_y error (cell-for-cell) | {m['uy']:.3f} | -- |",
        "",
        "See `figures/comparison.png`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image) -> runs/kwsst/",
        "cd validation/pitzdaily_openfoam",
        'docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh',
        "# 2. aquaflux coupled solve + comparison (from the repo root)",
        "cd ../..",
        "python3 validation/pitzdaily_openfoam/compare.py",
        "```",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
