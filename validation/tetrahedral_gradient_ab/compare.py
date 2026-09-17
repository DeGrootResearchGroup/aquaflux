"""A tetrahedral mesh with corner cells: does the local gradient repair fix them, and does it march?

Issue #432: ``MultipleCorrectionGradient``'s default (``boundary_closure=OwnerGradient()``,
``fallback=None``) leaves the reconstruction underdetermined at a tetrahedron owning two or more
boundary faces (``bind`` warns, naming the cells), and ``fallback=SkewCorrectedGradient()`` repairs
it -- but every case that has ever marched with that repair is quadrilateral or hexahedral (pitzDaily,
``bfs3d``, the UV reactor), so the repair had never been checked against a real mesh with the cell
shape it exists for. This case builds one.

**Geometry.** A short rectangular duct meshed as tetrahedra (``of_case/make_mesh.py``, gmsh's
OpenCASCADE + Delaunay backends) rather than the hexahedra every other 3D case here uses. Any
tetrahedral mesh of a box has cells owning two or more boundary faces wherever an element touches an
edge of the box -- no special construction is needed, only a genuinely unstructured tet mesh of a
domain with edges. The element size is chosen to avoid a separate, unrelated hazard: an unstructured
Delaunay mesh occasionally leaves a cell (interior or boundary) whose immediate neighbourhood is
nearly coplanar, which makes the Hessian-correction normal equations singular for a reason unrelated
to the boundary-face question this case is about (see ``make_mesh.py``'s ``MESH_SIZE`` comment). At
the shipped size: 2462 cells, 176 of them owning two or more boundary faces.

**What is answered.** ``report_m2_conditioning`` measures the actual quantity the reconstruction is
underdetermined in (``max|M2^-1|`` per cell) on this real mesh, not the synthetic unit fixture every
prior measurement of this defect used. The repair fixes it cleanly: the worst corner cell goes from
``max|M2^-1|`` ~1e16 under ``fallback=None`` to ~10 under ``fallback=SkewCorrectedGradient()``, and the
176-cell underdetermined population drops to zero. This is now on record in issue #432.

**What is NOT yet answered.** Whether the repair is *safe on a real march* -- the question `fallback`
exists for -- is still open. ``run_march_ab`` attempts it and is currently blocked before either arm
reaches a converged state, for reasons independent of the corner-cell defect above (both arms are
affected almost identically): tracked separately as issue #435. Read ``run_march_ab``'s own docstring
before trusting anything it prints; as shipped, expect it to report a failure.

This is **not** a physics-validated case (no OpenFOAM reference is run) -- the mesh is coarse and the
duct short, deliberately, to keep the march cheap once it can run at all. It exists to answer one
question about the reconstruction, not to characterize duct flow.

Run:
    validation/run_case.sh validation/tetrahedral_gradient_ab/compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so
# `import aquaflux` needs the repo root added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.io import read_openfoam
from aquaflux.mesh import distance_to_patches
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.solve import DualTimeLoop, RetryPolicy
from aquaflux.turbulence import (
    CompleteLu,
    CoupledRANS,
    MaterializedJacobian,
    SSTModel,
    SSTTurbulence,
    inlet_k,
    inlet_omega,
    solve_coupled,
)

HERE = Path(__file__).resolve().parent
POLYMESH = HERE / "of_case" / "constant" / "polyMesh"

#: The duct's operating point. Dh = 0.025 m (the cross-section), U_in = 10 m/s, nu = 1e-5 ->
#: Re_Dh = 25000 -- the same order as pitzDaily's inlet Reynolds number, so the k-omega SST wall
#: treatment is in its usual regime rather than a corner case of its own.
RHO, NU = 1.0, 1e-5
U_IN = 10.0
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * 0.025

#: Per rung, for the march attempt (part 2 -- see the module docstring). Deliberately small: as
#: shipped this march does not converge (issue #435), so a large cap only spends more wall time
#: reaching the same "alpha = 0 every step" failure this case already reports at ~15 steps.
#: Raise it if #435 is fixed and this case should actually try to converge.
MAX_STEPS = 15
RTOL, ATOL = 0.0, 1e-5  # the target rung's stop
ANCHOR_RTOL = 0.01  # the anchor rung only needs to be a good enough seed for the target
RATIO = 10.0  # anchor viscosity = target x RATIO (Reynolds number / RATIO)
INNER_STEPS, INNER_TOL = 5, 1e-2

BACKEND = "scipy"  # always available (no petsc4py needed); exact regardless of backend

#: Escalate the pseudo-time shift when a step's line search collapses (alpha < 0.01) -- the default
#: RetryPolicy has `on_alpha=None`, i.e. NO escalation trigger at all, and a step stalled at alpha=0
#: then repeats identically forever (measured: 60 steps, bit-identical |R|, no retry) rather than
#: recovering. Matches PITZ_RETRY_ON_CYCLES-style values elsewhere in validation/.
RETRY = RetryPolicy(on_alpha=0.01, beta_factor=2.0)


def build_case(gradient_scheme) -> CoupledRANS:
    """Assemble the duct's coupled RANS problem with the given gradient reconstruction."""
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    properties = PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)})

    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "walls": NoSlipWall(),
            }
        ),
        gradient_scheme=gradient_scheme,
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        wall_patches=["walls"],
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(k_in),
                "outlet": ZeroGradient(),
                "walls": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(omega_in),
                "outlet": ZeroGradient(),
                "walls": ZeroGradient(),
            }
        ),
        gradient_scheme=gradient_scheme,
    )
    return CoupledRANS.build(momentum, turbulence)


def corner_cell_count(coupled: CoupledRANS) -> tuple[int, int]:
    """(cells with >=2 boundary faces, total cells) -- the population the repair can reach."""
    mesh = coupled.momentum.mesh
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    interior = np.asarray(face_cells.interior)
    counts = np.bincount(owner[~interior], minlength=mesh.n_cells)
    return int((counts >= 2).sum()), mesh.n_cells


#: `MultipleCorrectionGradient._undetermined_cells`'s own threshold -- see its docstring for why this
#: value (a healthy inverse is order unity; an underdetermined one runs to ~1e16).
UNDETERMINED_THRESHOLD = 1e4


def report_m2_conditioning(gradient_scheme, label: str) -> float:
    """Bind ``gradient_scheme`` on this mesh and report ``max|M2^-1|`` per cell -- the actual quantity
    the corner-cell defect is about, measured here on a real mesh rather than a synthetic fixture.

    Returns the worst cell's value, for the before/after comparison in ``main``.
    """
    coupled = build_case(gradient_scheme)
    mesh, geometry = coupled.momentum.mesh, coupled.momentum.geometry
    corners, n_cells = corner_cell_count(coupled)
    bound = coupled.momentum.gradient_scheme.bind(mesh, geometry)
    worst = np.max(np.abs(np.asarray(bound.prepared.m2_inverse)), axis=(1, 2))
    over = int((worst > UNDETERMINED_THRESHOLD).sum())
    print(
        f"[{label}] {n_cells} cells, {corners} owning >=2 boundary faces  |  "
        f"max|M2^-1|: median {np.median(worst):.3g}, worst cell {int(np.argmax(worst))} "
        f"at {worst.max():.3e}, {over} cells above {UNDETERMINED_THRESHOLD:g}",
        flush=True,
    )
    return float(worst.max())


def _hybrid_start(coupled: CoupledRANS) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A wall-tapered initial condition, needing no linear solve at all.

    ⚠️ **Tracked as issue #435, unresolved: this seed does not yet get the march started either.**
    Kept in the tree because it is the current best attempt and rules out several mechanisms (see its
    own docstring in #435) rather than because it works. ``run_march_ab`` is expected to report a
    failure with the current code; do not read a green run as validated until #435 closes.

    ``hybrid_initialize``'s own potential-flow seed is not usable on this mesh, and not for a reason
    worth working around: its AMG-preconditioned Laplace solve stagnates (regardless of closure), and
    the underlying reason turned out to be structural rather than a preconditioner weakness --
    materializing the EXACT reconstruction Jacobian for the same scalar Laplace problem and solving it
    directly gives a condition number of **3.3e19 under `OwnerGradient`/`fallback=None` and 1.7e19
    under the repaired closure** -- both far past float64's ~1e16 solvable range, and close enough to
    each other that this is not primarily the corner-cell defect this case exists to test (measured
    once, kept as a lead rather than chased: a scalar Laplace/Neumann-datum problem on this duct's
    aspect ratio may simply be badly scaled on its own). Either way, no linear solve of this
    reconstruction's Jacobian is currently reliable on this mesh, potential flow included.

    So the seed here does not solve anything: a wall-normal profile from ``dist_to_wall`` that is
    exactly zero at the walls and ``U_IN`` at the wall-farthest cell, zero pressure, and ``k``/``omega``
    tapered the same way (floored well above zero so ``nu_t = k/omega`` stays finite). It satisfies no
    conservation law and is not divergence-free, but it starts the coupled Newton march already close
    to the no-slip condition instead of jumping the whole near-wall shear in one step -- which is what
    a uniform plug does, and does badly (measured: from a uniform plug the very first Newton step finds
    no admissible direction at any pseudo-time shift, `alpha` pinned at 0 through the full escalation
    ladder, reproduced across a Reynolds sweep from the target down to 1/10000th of it).

    ⚠️ **The taper's CURVATURE, not just its value, matters here, and a genuine turbulent power-law
    profile is the wrong shape to feed a second-order reconstruction.** A first attempt used
    ``dist**(1/7)`` (the standard turbulent-pipe exponent): its curvature diverges as `dist -> 0`, and
    at the thinnest near-wall cell that produced a velocity **gradient of 1.3e16** and a strain rate to
    match -- identically under both arms, at a cell with a perfectly healthy `max|M2^-1|` of 2.94 --
    which is the tell that this was the taper's own singular curvature meeting the Hessian-correction
    reconstruction, not the corner-cell defect this case exists to measure. The bounded-curvature taper
    below (``1 - (1-x)**2``) does not trigger *that* mechanism, but the coupled residual it produces is
    still astronomical (2.4e23, again identical between arms) -- a third, still-untraced cause. Do not
    read the absence of the 1/7-power symptom as this seed being sound.
    """
    momentum = coupled.momentum
    mesh, geometry = momentum.mesh, momentum.geometry
    n, dim = mesh.n_cells, mesh.dim

    wall_distance = distance_to_patches(mesh, geometry, ["walls"])
    reference = jnp.max(wall_distance)
    fraction = jnp.clip(wall_distance / reference, 0.0, 1.0)
    shape = 1.0 - (1.0 - fraction) ** 2  # bounded curvature everywhere, unlike a 1/7 power law

    velocity = jnp.zeros((n, dim)).at[:, 0].set(U_IN * shape)
    flow = momentum.pack(velocity, jnp.zeros(n))

    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, SSTModel()))
    floor = 0.1  # keep k/omega well clear of zero everywhere, including at the wall cells
    k = k_in * (floor + (1.0 - floor) * shape)
    omega = omega_in * (1.0 / (floor + (1.0 - floor) * shape))  # higher omega where k is lower
    return flow, k, omega


def run_march_ab(name: str, gradient_scheme) -> dict:
    """Attempt the coupled RANS march -- see ``_hybrid_start``'s docstring; expect a FAILED report.

    Kept as a best-effort attempt (issue #435) rather than removed: it is the harness the fix for #435
    should be checked against, and its failure mode (which rung, which exception) is itself informative
    to whoever picks that issue up.
    """
    coupled = build_case(gradient_scheme)
    corners, n_cells = corner_cell_count(coupled)
    print(f"[{name}] {n_cells} cells, {corners} owning >=2 boundary faces", flush=True)

    def on_step(report, rung="?"):
        print(
            f"  [{name}] {rung:<6} step {report.step:>3d}  |R| {report.residual_norm:.4e}  "
            f"ratio {report.residual_ratio:.3e}  alpha {report.alpha:>8.4g}  "
            f"cyc {report.cycles:>3d}  esc {report.escalations}",
            flush=True,
        )

    started = time.perf_counter()
    preconditioner = MaterializedJacobian(CompleteLu(backend=BACKEND))
    dual_time = DualTimeLoop(inner_steps=INNER_STEPS, inner_tol=INNER_TOL)
    try:
        # A manual two-point ramp (anchor at Re/RATIO, then the target) rather than
        # solve_reynolds_continuation, which always self-starts the anchor through
        # hybrid_initialize -- see _hybrid_start's docstring for why that is avoided here.
        anchor = coupled.with_scaled_molecular_viscosity(RATIO)
        flow0, k0, omega0 = _hybrid_start(coupled)
        flow1, k1, omega1 = solve_coupled(
            anchor,
            flow0,
            k0,
            omega0,
            preconditioner=preconditioner,
            dual_time=dual_time,
            max_steps=MAX_STEPS,
            rtol=ANCHOR_RTOL,
            atol=0.0,
            positivity_projection=True,
            retry=RETRY,
            on_step=lambda r: on_step(r, rung="anchor"),
        )
        flow, k, omega = solve_coupled(
            coupled,
            flow1,
            k1,
            omega1,
            preconditioner=preconditioner,
            dual_time=dual_time,
            max_steps=MAX_STEPS,
            rtol=RTOL,
            atol=ATOL,
            positivity_projection=True,
            retry=RETRY,
            on_step=lambda r: on_step(r, rung="target"),
        )
    except (
        Exception
    ) as exc:  # the march's own guard raises on non-convergence/non-finite -- report it
        elapsed = time.perf_counter() - started
        print(f"[{name}] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}", flush=True)
        return {
            "name": name,
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": elapsed,
        }

    elapsed = time.perf_counter() - started
    state = coupled.pack_state(flow, k, omega)
    residual_norm = float(jnp.linalg.norm(coupled.residual(state)))
    print(
        f"[{name}] converged in {elapsed:.1f}s, |R| = {residual_norm:.3e}, "
        f"k in [{float(jnp.min(k)):.3e}, {float(jnp.max(k)):.3e}], "
        f"omega in [{float(jnp.min(omega)):.3e}, {float(jnp.max(omega)):.3e}]",
        flush=True,
    )
    return {
        "name": name,
        "failed": False,
        "coupled": coupled,
        "flow": flow,
        "k": k,
        "omega": omega,
        "residual_norm": residual_norm,
        "elapsed": elapsed,
    }


def _relative_l2(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main() -> None:
    if not POLYMESH.exists():
        raise SystemExit(
            f"no mesh at {POLYMESH}; build it first (see README.md: make_mesh.py + gmshToFoam)"
        )

    # Part 1: the question this case answers, and does so unconditionally -- no march required.
    print("=== M2 conditioning: max|M2^-1| per cell, before and after the local repair ===\n")
    owner_worst = report_m2_conditioning(
        MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None), "owner"
    )
    repaired_worst = report_m2_conditioning(
        MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ),
        "repaired",
    )
    print(
        f"\n  worst-cell max|M2^-1|: {owner_worst:.3e} (owner) -> {repaired_worst:.3e} (repaired), "
        f"{owner_worst / max(repaired_worst, 1e-300):.1e}x improvement\n"
    )

    # Part 2: whether that fix is safe on a real march -- currently blocked, see issue #435.
    print("=== march attempt (issue #435 -- expect FAILED; see run_march_ab's docstring) ===\n")
    owner_arm = run_march_ab(
        "owner",
        MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
    )

    print("\n=== arm 2: fallback=SkewCorrectedGradient() (repairs the corner cells locally) ===\n")
    repaired_arm = run_march_ab(
        "repaired",
        MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ),
    )

    print("\n=== summary ===")
    for arm in (owner_arm, repaired_arm):
        if arm["failed"]:
            print(f"  {arm['name']:<10} FAILED: {arm['error']}")
        else:
            print(
                f"  {arm['name']:<10} converged, {arm['elapsed']:.1f}s, "
                f"|R| {arm['residual_norm']:.3e}"
            )

    if not owner_arm["failed"] and not repaired_arm["failed"]:
        du = _relative_l2(repaired_arm["flow"], owner_arm["flow"])
        dk = _relative_l2(repaired_arm["k"], owner_arm["k"])
        domega = _relative_l2(repaired_arm["omega"], owner_arm["omega"])
        print(
            f"\n  relative L2 difference (repaired vs owner): flow {du:.3e}, k {dk:.3e}, "
            f"omega {domega:.3e}"
        )
        print(
            "\n  BOTH ARMS MARCH: the local corner-cell repair is safe on the geometry it exists "
            "for, at least on this mesh and operating point."
        )
    elif owner_arm["failed"] != repaired_arm["failed"]:
        broken = "owner" if owner_arm["failed"] else "repaired"
        print(f"\n  ONE ARM FAILED ({broken}): the two closures are NOT equivalent on this mesh.")
    else:
        print(
            "\n  BOTH ARMS FAILED, identically: this is the EXPECTED, currently-unresolved outcome "
            "(issue #435), not a result about the gradient closure -- the M2 conditioning numbers "
            "above are what this case currently has to say about #432."
        )


if __name__ == "__main__":
    main()
