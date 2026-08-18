"""Passive tracer on the finite-width 3D backward-facing step: aquaflux vs OpenFOAM.

A species injected over **part of** the inlet of ``validation/bfs3d_openfoam``'s flow, transported to
steady state, and compared cell for cell against an OpenFOAM ``scalarTransport`` run on the same
mesh. It is the first exercise of :mod:`aquaflux.transport` against another code, and the shape the
reactor cases this project exists for will take: a converged flow, then what a tracer does in it.

**Two arms, reported under different names, because one number cannot answer both questions.**
The bfs3d *flow* itself does not agree between the codes -- reattachment ``x_r/h`` 8.36 against
OpenFOAM's 7.24 -- so a species comparison run on each code's own flow is dominated by that
disagreement and can attribute nothing. Splitting it:

* **Same-flux arm** -- both codes transport on the **identical face flux**: OpenFOAM's own ``phi``,
  imported by :func:`~aquaflux.io.read_surface_scalar_field`, together with OpenFOAM's own ``nut``
  so the diffusivity matches too. Mesh, flux, diffusivity and boundary values are then common, and
  what is left between the codes is the **scalar discretization** and nothing else. This is
  legitimate precisely because ``phi`` satisfies OpenFOAM's *discrete* continuity -- rebuilding
  ``(u . n) A`` from cell velocities would not, and a tracer on such a flux is not conservative.
  That the imported ``phi`` lands on the right faces is measured, not assumed, by
  ``../bfs3d_openfoam/phi_placement.py``.
* **Own-flow arm** -- each code on its own flow and its own eddy viscosity. This is the honest
  "what a user gets" number, and it necessarily carries the flow disagreement as well as the
  transport one. Read it as an end-to-end figure, never as a statement about transport.

The two arms differ **only** in which flux and eddy viscosity are supplied. Every other choice --
mesh, scheme, gradient, boundary closures, diffusivity relation, the injector -- is shared, so
subtracting the arms isolates the flow's contribution.

**Diffusivity: a turbulent Schmidt number of exactly 1, on both sides.** OpenFOAM's
``scalarTransport`` with no ``D`` entry uses the momentum transport model's effective viscosity
``nu + nut``; aquaflux is given ``effective_diffusivity(nu, nut, turbulent_number=1.0)``, which is
the same quantity. That is a **modelling choice made to match the reference**, not this project's
default (which is 0.7), and any result here is a result at ``Sc_t = 1``.

**The injection is a boundary VALUE, not a mesh change.** It is a
:class:`~aquaflux.boundary.DirichletField` on the existing ``inlet`` patch whose value is a function
of the face centroid, so the ``polyMesh`` is untouched and every measurement previously taken on it
stays valid. Its profile has one definition, ``injector.injected_value``, from which the OpenFOAM
case's inlet values are generated -- so the two codes impose identical values face for face rather
than two implementations of one intent.

Run (after the flow case, ``write_inlet_field.py`` and ``of_case/run_of.sh``) from the repo root::

    validation/run_case.sh validation/bfs3d_species/compare.py --wait
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
FLOW_CASE_DIR = ROOT / "validation" / "bfs3d_openfoam"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(FLOW_CASE_DIR))

import jax.numpy as jnp  # noqa: E402
from aquaflux.boundary import BoundaryConditions, DirichletField, ZeroGradient  # noqa: E402
from aquaflux.discretization import LimitedUpwind  # noqa: E402
from aquaflux.flow import volume_flux  # noqa: E402
from aquaflux.io import (  # noqa: E402
    read_openfoam,
    read_surface_scalar_field,
    read_volume_scalar_field,
)
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter  # noqa: E402
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver  # noqa: E402
from aquaflux.transport import ScalarTransport, effective_diffusivity  # noqa: E402
from aquaflux.turbulence import scalar_transport_preconditioner  # noqa: E402
from injector import injected_value  # noqa: E402

OF_FLOW = FLOW_CASE_DIR / "of_case"
OF_SPECIES = CASE / "runs" / "species"
CHECKPOINTS = FLOW_CASE_DIR / "checkpoints"
FIGS = CASE / "figures"

#: The flow case's fluid: kinematic, unit density (so its ``phi`` is already volumetric -- which the
#: inlet flux check confirms, reading exactly ``U A`` rather than ``rho U A``).
RHO, NU = 1.0, 1e-5

#: Step height, and the streamwise stations the plume is profiled at, in step heights.
H = 0.01
STATIONS = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)

#: Turbulent Schmidt number. **1.0 to match OpenFOAM's un-parameterized ``nuEff``**, not the
#: project default of 0.7 -- see the module docstring.
SCHMIDT_T = 1.0

#: The time directory the frozen flow (and its flux) is taken from.
FLOW_TIME = "2000"


def build_transport(mesh, geometry, nut):
    """The tracer equation -- everything except which flow carries it.

    Shared verbatim by both arms, so the only difference between them is the flux and the eddy
    viscosity handed in. Second-order limited upwind advection and corrected Green--Gauss gradients
    mirror what the flow case gives its own transported scalars.

    Parameters
    ----------
    mesh : Mesh
        The imported mesh.
    geometry : MeshGeometry
        Its face and cell metrics.
    nut : np.ndarray
        Eddy viscosity per cell, shape ``(n_cells,)`` -- whichever code's the arm is measuring.

    Returns
    -------
    ScalarTransport
        The configured equation; call ``.residual(flux)`` for a particular flow.
    jnp.ndarray
        The per-cell effective diffusivity, which the solve's preconditioner needs as well.
    """
    boundary = BoundaryConditions(
        {
            # The sub-patch injection: a value that varies over the EXISTING inlet patch.
            "inlet": DirichletField(field_fn=injected_value),
            "outlet": ZeroGradient(),
            "upperWall": ZeroGradient(),
            "lowerWall": ZeroGradient(),
            "sideWalls": ZeroGradient(),
        }
    )
    diffusivity = effective_diffusivity(
        jnp.full(mesh.n_cells, NU), jnp.asarray(nut), turbulent_number=SCHMIDT_T
    )
    transport = ScalarTransport.build(
        mesh,
        geometry,
        diffusivity,
        boundary,
        LimitedUpwind(limiter=VenkatakrishnanLimiter()),
        gradient_scheme=CorrectedGreenGauss(),
    )
    return transport, diffusivity.values


def solve_tracer(transport, mesh, geometry, flux, diffusivity):
    """Solve the steady tracer equation on one frozen flux.

    The equation is **linear** in the concentration -- the flux and the diffusivity are frozen and
    nothing else depends on it -- so Newton reaches the root in one step and the rest only confirm
    it. That is also why neither arm needs the continuation or line search the flow solve did.

    Linear does not mean easy to invert, though, and an unpreconditioned solve does not converge
    here: away from the shear layer the eddy viscosity falls to ``~4e-11``, so the effective
    diffusivity is essentially molecular and the cell Peclet number reaches order ``10^3``. The
    frozen convection-diffusion V-cycle this project already builds for the ``k``/``omega`` scalars
    is the right preconditioner for exactly the same reason it is right there, and it is reused
    rather than rebuilt: it upwinds first order at the frozen flux, which is what makes it an
    M-matrix an aggregation hierarchy can coarsen.
    """
    residual = transport.residual(flux)
    preconditioner = scalar_transport_preconditioner(
        mesh,
        geometry,
        diffusivity,
        flux,
        residual,
        jnp.zeros(mesh.n_cells),
    )
    solver = ImplicitNewtonSolver(
        max_steps=20, forward_step=DampedNewtonStep(preconditioner=preconditioner)
    )
    return solver.solve(lambda c, _: residual(c), jnp.zeros(mesh.n_cells), None)


def continuity_error(mesh, flux, flow_rate):
    """Max per-cell net flux as a fraction of the domain flow rate -- is this flux conservative?

    The property a transported scalar actually depends on: if the flux does not close discretely,
    a uniform tracer would not stay uniform and nothing downstream of it means anything. Measured
    the same way, and by the same scatter, as ``../bfs3d_openfoam/phi_placement.py``.
    """
    net = np.abs(np.asarray(mesh.face_cells.scatter_conservative(jnp.asarray(flux))))
    return float(net.max() / flow_rate)


def injected_throughput(mesh, geometry, flux):
    """Tracer entering through the inlet, ``sum over the inlet of -phi * s_face`` (m^3/s).

    Exact rather than estimated: the inlet is a Dirichlet patch, so its face values *are* the
    injection profile evaluated at the face centroids -- the same values the residual assembles
    from, and the same ones written into the OpenFOAM case. This is the rate the outlet must carry
    at steady state, and it is identical for both arms and both codes by construction.
    """
    indices = np.asarray(mesh.face_patches.indices("inlet"))
    centroids = np.asarray(geometry.face.centroid)[indices]
    return float(-np.sum(np.asarray(flux)[indices] * np.asarray(injected_value(centroids))))


def conservation_error(transport, flux, values):
    """``|sum over all cells of R|`` -- the net tracer flux through the boundary, which must vanish.

    Summing the converged residual telescopes every interior face away, since each contributes
    equally and oppositely to the two cells it separates. What survives is exactly the net boundary
    flux, so this is a *discrete identity* rather than an approximation, and it needs no boundary
    face values of its own. It holds only because the scalar rides a flux on which continuity
    closes -- which is the whole reason the reference's ``phi`` is imported rather than rebuilt.
    """
    return float(abs(jnp.sum(transport.residual(flux)(values))))


def outlet_throughput(mesh, flux, values):
    """Tracer leaving through the outlet, estimated with the owner-cell value on each outlet face.

    An estimate, not the identity above: the outlet is ``ZeroGradient``, so its true face value
    carries a tangential correction this ignores. It is reported because it is computed **the same
    way for both codes**, which makes it a fair comparison even where it is not an exact flux.
    """
    indices = np.asarray(mesh.face_patches.indices("outlet"))
    owner = np.asarray(mesh.face_cells.owner)[indices]
    return float(np.sum(np.asarray(flux)[indices] * np.asarray(values)[owner]))


def slab_profile(centroid, volume, values, station, half_width=0.25 * H):
    """Volume-weighted mean and unmixedness of the tracer in a thin slab at ``x = station * H``.

    Returns the slab's mean concentration and its **unmixedness** -- the variance normalized by
    ``mean * (1 - mean)``, which is 1 for a completely segregated stream and 0 once the slab is
    uniform. It is the mixing measure the two codes are compared on, and it is defined here once so
    the same callable produces both the reported table and any figure.

    A volume-weighted slab average, not a flux-weighted mixing cup: the two codes sit on identical
    cells, so a volume weighting compares like with like and needs no cross-sectional face set.

    ⚠️ **The slab widens to the nearest cell plane when a fixed width would miss.** This mesh grades
    8x in the streamwise direction, so far downstream one cell is wider than ``half_width`` and a
    fixed slab falls in the gap between cell centres and selects nothing -- which showed up as a
    silent ``nan`` for one station while its neighbours reported normally. Falling back to the
    nearest plane of centres makes the station always mean something; the fixed width still governs
    wherever the mesh is fine enough to resolve it.
    """
    x = centroid[:, 0]
    offset = np.abs(x - station * H)
    inside = offset <= max(half_width, 1.01 * float(offset.min()))
    if not inside.any():
        return float("nan"), float("nan")
    w = volume[inside]
    s = values[inside]
    mean = float(np.sum(w * s) / np.sum(w))
    variance = float(np.sum(w * (s - mean) ** 2) / np.sum(w))
    spread = mean * (1.0 - mean)
    return mean, (variance / spread if spread > 1e-12 else float("nan"))


def openfoam_flux_and_nut(mesh):
    """The reference's own face flux and eddy viscosity, on this mesh's ordering.

    Both come from the frozen flow the OpenFOAM tracer run used, so the same-flux arm transports on
    exactly what the reference transported on. The flux placement is checked by the reader; that it
    is *right* rather than merely well-shaped is measured by ``../bfs3d_openfoam/phi_placement.py``.
    """
    flux = read_surface_scalar_field(OF_FLOW / FLOW_TIME / "phi", mesh)
    nut = read_volume_scalar_field(OF_FLOW / FLOW_TIME / "nut", mesh)
    return jnp.asarray(flux), np.asarray(nut)


def _flow_case_module():
    """The flow case's own ``compare.py``, loaded by PATH rather than by name.

    Both cases have a module called ``compare``, so a plain ``import compare`` resolves by whatever
    order the paths happen to sit in -- and would silently pick this file when it is imported rather
    than run. Loading the flow case's file explicitly makes the assembly it exports (``build_case``)
    unambiguous, which matters because the whole point is to transport on the *same* case's flow
    rather than on a second description of it.
    """
    import importlib.util

    path = FLOW_CASE_DIR / "compare.py"
    spec = importlib.util.spec_from_file_location("bfs3d_flow_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aquaflux_flux_and_nut(mesh):
    """aquaflux's own converged flow: its Rhie--Chow flux (volumetric) and its eddy viscosity.

    Read from the flow case's rolling checkpoint rather than re-marched -- a converged state
    otherwise exists only inside the process that computed it, and re-solving the coupled RANS
    system to obtain a flux the tracer will then freeze is an hour spent to reproduce a state
    already on disk.

    ⚠️ **The checkpoint predates the production-limiter default moving to OFF**, so it is not a root
    of today's coupled residual. That matters for an adjoint and does not matter here: what a
    transported scalar requires of a flux is that it close discretely, which is checked directly
    (:func:`continuity_error`) rather than inferred from the state's residual.
    """
    case = _flow_case_module().build_case()
    states = sorted(CHECKPOINTS.glob("state-*.npz"))
    if not states:
        raise SystemExit(
            f"no checkpoint in {CHECKPOINTS.relative_to(ROOT)} -- run the flow case first "
            "(validation/run_case.sh validation/bfs3d_openfoam/compare.py)"
        )
    path = states[-1]
    state = jnp.asarray(np.load(path)["state"])

    coupled, momentum, turbulence = case["coupled"], case["momentum"], case["turbulence"]
    flow, k, omega = coupled.physical_fields(state)
    nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    # A concentration rides the VOLUMETRIC flux, never the mass flux: its balance is a mass balance
    # on the species and has no fluid density in it. (This case runs at rho = 1, where the two are
    # numerically identical -- so this line is a statement of intent that this case cannot verify.
    # tests/integration/test_scalar_transport.py verifies it, at rho = 998.)
    flux = volume_flux(momentum.mass_flux(flow), RHO)
    print(f"  aquaflux flow from {path.name}", flush=True)
    return jnp.asarray(flux), np.asarray(nu_t)


def run_arm(name, mesh, geometry, flux, nut, flow_rate):
    """Solve the tracer on one flux and report what the solve itself guarantees."""
    started = time.time()
    transport, diffusivity = build_transport(mesh, geometry, nut)
    values = np.asarray(solve_tracer(transport, mesh, geometry, flux, diffusivity))

    injected = injected_throughput(mesh, geometry, flux)
    print(
        f"  {name}: solved in {time.time() - started:.0f} s; "
        f"range [{values.min():+.4f}, {values.max():+.4f}]; "
        f"flux continuity {continuity_error(mesh, flux, flow_rate):.2e}; "
        f"conservation |sum R| {conservation_error(transport, flux, jnp.asarray(values)):.2e} "
        f"against {injected:.4e} m^3/s injected",
        flush=True,
    )
    return values


def compare(label, aqua, of, centroid, volume):
    """One arm against the reference: cellwise agreement, then the mixing profile."""
    delta = aqua - of
    print(f"\n--- {label} ---", flush=True)
    print(
        f"  cellwise |delta|: max {np.abs(delta).max():.4f}, rms "
        f"{np.sqrt(np.mean(delta**2)):.4f}, mean {delta.mean():+.4f}",
        flush=True,
    )
    print("\n  x/h     aquaflux mean   OF mean    aquaflux unmixed   OF unmixed", flush=True)
    rows = []
    for station in STATIONS:
        a_mean, a_unmixed = slab_profile(centroid, volume, aqua, station)
        o_mean, o_unmixed = slab_profile(centroid, volume, of, station)
        rows.append((station, a_mean, o_mean, a_unmixed, o_unmixed))
        print(
            f"  {station:5.1f}   {a_mean:12.5f}   {o_mean:8.5f}   {a_unmixed:14.4f}   "
            f"{o_unmixed:10.4f}",
            flush=True,
        )
    return dict(
        max_delta=float(np.abs(delta).max()),
        rms_delta=float(np.sqrt(np.mean(delta**2))),
        rows=rows,
    )


def main() -> int:
    if not (OF_SPECIES / "s").exists():
        raise SystemExit(
            f"no OpenFOAM tracer field at {OF_SPECIES.relative_to(ROOT)}/s -- run\n"
            "  python3 validation/bfs3d_species/write_inlet_field.py\n"
            "  docker run --rm -v $PWD:/work -w /work/validation/bfs3d_species/of_case "
            "openfoam13:latest bash run_of.sh"
        )

    print("reading mesh + reference fields", flush=True)
    mesh = read_openfoam(OF_FLOW)
    geometry = mesh.geometry()
    centroid = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)
    of_values = read_volume_scalar_field(OF_SPECIES / "s", mesh)
    print(
        f"  {mesh.n_cells} cells; OpenFOAM tracer range "
        f"[{of_values.min():+.4f}, {of_values.max():+.4f}]",
        flush=True,
    )

    of_flux, of_nut = openfoam_flux_and_nut(mesh)
    flow_rate = abs(
        float(np.sum(np.asarray(of_flux)[np.asarray(mesh.face_patches.indices("inlet"))]))
    )
    print(f"  domain flow rate {flow_rate:.4e} m^3/s, Sc_t = {SCHMIDT_T}", flush=True)

    print("\nsolving", flush=True)
    same_flux = run_arm("same-flux ", mesh, geometry, of_flux, of_nut, flow_rate)
    aq_flux, aq_nut = aquaflux_flux_and_nut(mesh)
    own_flow = run_arm("own-flow  ", mesh, geometry, aq_flux, aq_nut, flow_rate)

    same = compare(
        "same-flux arm (isolates the scalar discretization)", same_flux, of_values, centroid, volume
    )
    own = compare(
        "own-flow arm (end to end: transport AND the flow difference)",
        own_flow,
        of_values,
        centroid,
        volume,
    )

    # The arms differ only in the flux and nu_t, so their difference is the flow's contribution --
    # the number that says whether an end-to-end discrepancy is transport or is the flow underneath.
    between = own_flow - same_flux
    print(
        f"\nflow's own contribution (own-flow minus same-flux): max "
        f"{np.abs(between).max():.4f}, rms {np.sqrt(np.mean(between**2)):.4f}",
        flush=True,
    )
    print(
        f"\nSUMMARY  same-flux rms {same['rms_delta']:.4f} | own-flow rms {own['rms_delta']:.4f} "
        f"| flow contribution rms {np.sqrt(np.mean(between**2)):.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
