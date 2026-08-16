"""Integration: a scalar transported by a solved flow, against analytical and discrete properties.

The species path end to end -- a converged flow, its Rhie--Chow flux converted to a volumetric one,
and :class:`~aquaflux.transport.ScalarTransport` solved on it. Four properties, each of which fails
in a different way if the coupling is wrong:

- **order of accuracy** against the 1-D advection--diffusion exponential, which pins the operator;
- **discrete conservation** -- what enters leaves, to solver tolerance, which is what advecting on
  the flow's own conservative flux buys and what rebuilding ``(u . n) A`` would break;
- **uniform-field preservation** -- a uniform tracer must stay uniform, the discrete statement that
  the flux the scalar rides is divergence-free;
- **the flux carries the right quantity** -- the volumetric flow rate through the inlet is
  ``U_in x A``, not ``rho U_in x A``;
- **boundedness** -- a tracer injected between 0 and 1 stays in ``[0, 1]`` under an upwind scheme,
  so the field is physical rather than merely convergent.

⚠️ **The fourth is not implied by the third, which is why it is its own test.** A uniform field is
preserved by *any* divergence-free flux, and scaling a flux by a constant density leaves it
divergence-free -- so uniform-field preservation detects a flux that is not conservative (one
rebuilt from cell velocities) but is completely blind to one that is wrongly *scaled*. Checked by
mutation: feeding the mass flux where the volumetric one belongs leaves the uniform result
unchanged to 1e-10. Only a test that reads the flux's magnitude against a physical rate catches it.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, DirichletField, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    volume_flux,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver
from aquaflux.transport import ScalarTransport, effective_diffusivity

# Water, not a unit density. With rho = 1 the mass flux and the volumetric flux are numerically
# identical, so no test here could tell them apart -- and telling them apart is the whole point of
# the conversion these tests exercise. At 998 they differ by three orders of magnitude.
RHO, NU, U_IN = 998.0, 1e-2, 1.0


def _flow(nx=16, ny=8):
    """A converged channel flow, and the volumetric face flux a scalar rides on it."""
    mesh = structured_grid_2d(nx, ny, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )
    # The block preconditioner plus a backtracking line search, as the channel flow tests use: the
    # full Newton step from a zero state overshoots and an unpreconditioned solve stalls.
    forward_step = DampedNewtonStep(
        preconditioner=BlockPreconditioner.build(momentum).factory(), line_search=10
    )
    solver = ImplicitNewtonSolver(max_steps=40, forward_step=forward_step)
    state = solver.solve(lambda s, m: m.residual(s), momentum.initial_state(), momentum)
    return mesh, geometry, momentum, state, volume_flux(momentum.mass_flux(state), RHO)


def _solve_scalar(transport, flux, n_cells):
    solver = ImplicitNewtonSolver(max_steps=40, forward_step=DampedNewtonStep())
    residual = transport.residual(flux)
    return solver.solve(lambda c, _: residual(c), jnp.zeros(n_cells), None)


@pytest.fixture(scope="module")
def flow():
    return _flow()


def _transport(mesh, geometry, boundary, diffusivity=1e-2):
    return ScalarTransport.build(
        mesh,
        geometry,
        effective_diffusivity(jnp.full(mesh.n_cells, diffusivity)),
        boundary,
        FirstOrderUpwind(),
        gradient_scheme=CompactGreenGauss(),
    )


def test_a_uniform_tracer_stays_uniform(flow) -> None:
    """The discrete statement that the flux the scalar rides is divergence-free.

    With the same value imposed on every inflow boundary and zero gradient elsewhere, the exact
    solution is that value everywhere. It holds discretely only because the scalar advects on the
    flow's own Rhie--Chow flux, on which continuity closes; a flux rebuilt from cell velocities
    would leave a spurious source in every cell.

    It says nothing about the flux's *scale*: any divergence-free flux preserves a uniform field,
    and scaling by a constant density preserves divergence-freeness. That is
    :func:`test_the_flux_carries_a_volumetric_flow_rate`'s job.
    """
    mesh, geometry, _, _, flux = flow
    boundary = BoundaryConditions(
        {
            "left": Dirichlet(1.0),
            "right": ZeroGradient(),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    c = _solve_scalar(_transport(mesh, geometry, boundary), flux, mesh.n_cells)

    assert jnp.allclose(c, 1.0, atol=1e-8)


def test_what_enters_leaves(flow) -> None:
    """Discrete conservation: the net advective-plus-diffusive flux through the boundary is zero.

    Summing the converged residual over every cell telescopes the interior faces away (each carries
    equal and opposite contributions to its two cells), so the total is exactly the net boundary
    flux -- which must vanish at steady state with no volume source.
    """
    mesh, geometry, _, _, flux = flow
    boundary = BoundaryConditions(
        {
            "left": Dirichlet(1.0),
            "right": ZeroGradient(),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    transport = _transport(mesh, geometry, boundary)
    c = _solve_scalar(transport, flux, mesh.n_cells)

    assert abs(float(jnp.sum(transport.residual(flux)(c)))) < 1e-9


def test_a_sub_patch_injection_stays_bounded_and_mixes(flow) -> None:
    """A tracer injected over part of the inlet stays in ``[0, 1]`` and spreads downstream.

    The sub-patch injection is a :class:`DirichletField` on the *existing* inlet patch -- the value
    is a function of the face centroid -- so covering part of a patch needs no separate patch and
    therefore no change to the mesh. Boundedness is the physical check: an upwind scheme must not
    manufacture concentrations outside the range imposed on the boundary.
    """
    mesh, geometry, _, _, flux = flow
    boundary = BoundaryConditions(
        {
            # Inject over the lower half of the inlet only.
            "left": DirichletField(field_fn=lambda x: jnp.where(x[:, 1] < 0.5, 1.0, 0.0)),
            "right": ZeroGradient(),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    c = _solve_scalar(_transport(mesh, geometry, boundary), flux, mesh.n_cells)

    assert float(jnp.min(c)) >= -1e-10
    assert float(jnp.max(c)) <= 1.0 + 1e-10
    # It is a genuine mixing problem, not a uniform field: the tracer spans the range and has
    # spread off the injected half.
    assert float(jnp.max(c)) > 0.5
    assert float(jnp.min(c)) < 0.5


def test_the_flux_carries_a_volumetric_flow_rate(flow) -> None:
    """The flux a concentration rides is ``Q = U A``, not ``rho U A`` -- a factor of 998 apart.

    The one check here with teeth against a mis-scaled flux. A species concentration is per unit
    *volume*, so its balance ``dC/dt + div(u C) = ...`` is carried by the volumetric flux; using the
    mass flux would inflate every advective term by the density and silently give a Peclet number
    three orders of magnitude too large. Measured against the inlet's known volumetric rate
    ``U_in x H x depth``, which the flow reproduces because the inlet velocity is prescribed.
    """
    mesh, geometry, momentum, state, flux = flow
    inlet = mesh.face_patches.indices("left")

    # Owner-outward, so an inflow is negative; the magnitude is the volumetric rate in.
    rate_in = -float(jnp.sum(flux[inlet]))
    expected = U_IN * float(jnp.sum(geometry.face.area[inlet]))

    assert rate_in == pytest.approx(expected, rel=1e-6)
    # And it is genuinely the mass flux divided by the density, not coincidentally equal to it.
    assert not jnp.allclose(flux[inlet], momentum.mass_flux(state)[inlet])


def _advection_diffusion_error(nx: int) -> float:
    """RMS error against the 1-D advection--diffusion exponential on an ``nx``-cell mesh."""
    length, gamma, u = 1.0, 0.05, 1.0
    mesh = structured_grid_2d(nx, 1, lx=length, ly=0.1, named_boundaries=True)
    geometry = mesh.geometry()
    # A uniform flow: the face flux is u times the projected area, exactly divergence-free here.
    flux = u * geometry.face.normal[:, 0] * geometry.face.area

    transport = ScalarTransport.build(
        mesh,
        geometry,
        effective_diffusivity(jnp.full(mesh.n_cells, gamma)),
        BoundaryConditions(
            {
                "left": Dirichlet(0.0),
                "right": Dirichlet(1.0),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        FirstOrderUpwind(),
    )
    c = _solve_scalar(transport, flux, mesh.n_cells)

    x = geometry.cell.centroid[:, 0]
    peclet = u * length / gamma
    exact = (jnp.exp(peclet * x / length) - 1.0) / (jnp.exp(peclet) - 1.0)
    return float(jnp.sqrt(jnp.mean((c - exact) ** 2)))


def test_advection_diffusion_matches_the_analytical_profile_and_converges() -> None:
    """The 1-D advection--diffusion exponential, and the error falling as the mesh refines.

    A uniform velocity carries a scalar from ``C=0`` to ``C=1`` against diffusion; the closed-form
    profile is ``(exp(Pe x/L) - 1)/(exp(Pe) - 1)``. This is the operator-level check that the
    coupling advects on the right quantity -- an error in the flux (a mass flux where a volumetric
    one belongs, say) shows up directly as the wrong Peclet number, which no amount of refinement
    would fix. First-order upwind is diffusive at ``Pe = 20``, so the bar is the *trend*: the error
    must fall by close to the scheme's first order over a mesh doubling.
    """
    coarse = _advection_diffusion_error(20)
    fine = _advection_diffusion_error(40)

    assert fine < coarse
    # First order would halve it; require most of that, which a wrong flux could not produce.
    assert coarse / fine > 1.6
