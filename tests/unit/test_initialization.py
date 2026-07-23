"""Unit: the cheap field initializers -- scalar Laplace, potential flow, and the hybrid RANS IC.

Fast checks (linear solves on small meshes, no nonlinear/coupled solve): the Laplace solve reproduces
an analytic harmonic field; the potential-flow velocity matches the through-flow with no wall
penetration and works as a standalone flow-only initializer (and degrades to zero on a closed domain);
and the hybrid IC yields positive fields with the analytical near-wall omega and a small momentum-block
residual. The coupled solve self-starting from this IC is the slow integration test.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    bernoulli_pressure,
    laplace_field,
    potential_flow,
)
from aquaflux.flow.initialization import _pressure_outlet_cells
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
    omega_wall_value,
)

RHO, U_IN, NU = 1.0, 1.0, 1e-2


def _channel(nx=16, ny=12, lx=3.0, ly=1.0, y_nodes=None):
    mesh = structured_grid_2d(nx, ny, lx=lx, ly=ly, named_boundaries=True, y_nodes=y_nodes)
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
    return mesh, geometry, momentum


def test_laplace_field_reproduces_a_linear_harmonic() -> None:
    # phi linear from 0 (left) to 1 (right), zero-gradient top/bottom -> phi = x / lx exactly.
    lx = 3.0
    mesh = structured_grid_2d(12, 8, lx=lx, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    boundary = BoundaryConditions(
        {
            "left": Dirichlet(0.0),
            "right": Dirichlet(1.0),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    phi, _ = laplace_field(mesh, geometry, boundary, gradient_scheme=CompactGreenGauss())
    expected = geometry.cell.centroid[:, 0] / lx
    assert float(jnp.max(jnp.abs(phi - expected))) < 1e-8


def test_potential_flow_matches_the_through_flow_without_wall_penetration() -> None:
    _, _, momentum = _channel()
    flow = potential_flow(momentum)  # flow-only: no turbulence involved
    velocity, _ = momentum.unpack(flow)
    assert bool(jnp.all(jnp.isfinite(flow)))
    # straight channel: streamwise velocity == inlet, no transverse penetration
    assert float(jnp.max(jnp.abs(velocity[:, 0] - U_IN))) < 1e-6
    assert float(jnp.max(jnp.abs(velocity[:, 1]))) < 1e-6


def test_bernoulli_pressure_is_anchored_at_the_outlet_and_tracks_the_dynamic_head() -> None:
    # A velocity that accelerates downstream, so the dynamic head varies across the channel.
    _, geometry, momentum = _channel()
    speed = 1.0 + geometry.cell.centroid[:, 0]
    velocity = jnp.stack([speed, jnp.zeros_like(speed)], axis=1)
    pressure = bernoulli_pressure(momentum, velocity)

    outlet_cells = _pressure_outlet_cells(momentum)
    # Consistent with the p = 0 outlet BC: the mean pressure over the outlet cells is zero.
    assert abs(float(jnp.mean(pressure[outlet_cells]))) < 1e-10
    # Bernoulli's ``p + ½ρ|u|² = const``: the faster cell carries the lower pressure.
    assert float(pressure[jnp.argmax(speed)]) < float(pressure[jnp.argmin(speed)])
    # Exact closed form p = ½ρ(mean_outlet(|u|²) - |u|²).
    reference = 0.5 * RHO * float(jnp.mean(speed[outlet_cells] ** 2))
    expected = reference - 0.5 * RHO * speed**2
    assert float(jnp.max(jnp.abs(pressure - expected))) < 1e-10


def test_bernoulli_pressure_is_uniform_for_a_uniform_velocity() -> None:
    # A uniform through-flow has no dynamic-head variation, so the seed is the (zero) outlet datum.
    _, geometry, momentum = _channel()
    velocity = jnp.broadcast_to(jnp.array([U_IN, 0.0]), (geometry.cell.centroid.shape[0], 2))
    pressure = bernoulli_pressure(momentum, velocity)
    assert float(jnp.max(jnp.abs(pressure))) < 1e-12


def test_potential_flow_is_zero_on_a_closed_domain() -> None:
    # Lid-driven cavity: all walls, no outlet -> no potential through-flow -> zero velocity.
    mesh = structured_grid_2d(8, 8, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_IN, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    velocity, _ = momentum.unpack(potential_flow(momentum))
    assert float(jnp.max(jnp.abs(velocity))) < 1e-8


@pytest.mark.parametrize("growth", [1.8, 2.2])
def test_potential_flow_survives_a_wall_resolved_aspect_ratio(growth: float) -> None:
    # A wall-resolved mesh grades to a near-wall cell of aspect ratio 1e3-1e5, where the Laplacian's
    # condition number (~aspect ratio squared) stagnates an unpreconditioned Krylov solve into a
    # non-finite iterate. The multigrid preconditioner is what keeps the initializer usable there.
    ny, lx, ly = 32, 6.0, 1.0
    y_nodes = graded_nodes(ny, ly, growth)
    mesh = structured_grid_2d(16, ny, lx=lx, ly=ly, named_boundaries=True, y_nodes=y_nodes)
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
    assert (lx / 16) / float(y_nodes[1]) > 1e3  # the regime that used to fail

    velocity, _ = momentum.unpack(potential_flow(momentum))

    assert bool(jnp.all(jnp.isfinite(velocity)))
    # still the straight-channel potential solution, graded mesh notwithstanding
    assert float(jnp.max(jnp.abs(velocity[:, 0] - U_IN))) < 1e-8
    # The transverse component is not machine-zero here: the cell-gradient reconstruction's roundoff
    # is amplified by the near-wall anisotropy (it grows with the aspect ratio, while the streamwise
    # component stays exact). Small is all that is required -- and a non-zero transverse velocity is
    # what lifts the coupled solve's exactly-symmetric degeneracy.
    assert float(jnp.max(jnp.abs(velocity[:, 1]))) < 1e-3


def _turbulence(mesh, geometry, k_in, omega_in):
    return SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )


def test_hybrid_initialize_is_positive_with_analytical_wall_omega() -> None:
    mesh, geometry, momentum = _channel()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), 0.05))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07, model))
    turbulence = _turbulence(mesh, geometry, k_in, omega_in)

    flow, k, omega = hybrid_initialize(momentum, turbulence)

    assert flow.shape == (momentum.mesh.n_cells * (mesh.dim + 1),)
    assert bool(jnp.all(jnp.isfinite(flow)))
    assert float(jnp.min(k)) > 0.0
    assert float(jnp.min(omega)) > 0.0
    assert float(jnp.max(k)) <= k_in + 1e-12  # k floored at the inlet level, still bounded by k_in

    # every cell sits on the analytical viscous-sublayer profile omega(y) = 6 nu / (beta_1 y^2) at its
    # own wall distance, or above it where the interpolant is larger -- so the wall-adjacent cells carry
    # the fixation value exactly and the near-wall band follows the same 1/y^2 decay (not a flat cliff).
    profile = omega_wall_value(turbulence.molecular_viscosity, turbulence.wall_distance, model)
    assert bool(jnp.all(omega >= profile - 1e-12))
    assert jnp.allclose(omega[turbulence.wall_cells], profile[turbulence.wall_cells])
    # a wall-adjacent interior cell (not itself fixed) is lifted onto the profile, above the interpolant
    interior = jnp.ones(mesh.n_cells, bool).at[turbulence.wall_cells].set(False)
    near_wall_interior = interior & (profile > omega_in)
    assert bool(jnp.any(near_wall_interior))
    assert jnp.allclose(omega[near_wall_interior], profile[near_wall_interior])
    # omega is the interpolant *raised* onto the profile, so it is never below the interpolant floor
    # (omega_in here, the constant harmonic solution of the inlet-Dirichlet / zero-gradient-wall data).
    assert float(jnp.min(omega)) >= omega_in * (1.0 - 1e-6)


def test_hybrid_initialize_omega_is_a_smooth_ramp_in_log_space() -> None:
    """On a wall-resolved (graded) mesh the seeded omega is a smooth ramp in the log variable log(omega).

    The analytical profile ``6 nu/(beta_1 y^2)`` gives ``log omega(y) = log(6 nu/beta_1) - 2 log y``,
    whose largest cross-face step is set by the mesh growth ratio, not the Reynolds number. Seeding only
    the wall cells would instead put ``omega_wall`` next to the flat interpolant, a jump of
    ``~log(omega_wall / omega_core)`` in ``log omega`` across the first face that *grows* as the
    near-wall spacing shrinks. That matters wherever the near-wall omega enters logarithmically: pin that
    the seeded field is the ramp -- its max cross-face jump in ``log omega`` is far below that cliff.
    """
    ny, growth, ly = 32, 1.3, 1.0
    y_nodes = graded_nodes(ny, ly, growth)
    mesh, geometry, momentum = _channel(ny=ny, ly=ly, y_nodes=y_nodes)
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), 0.05))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07, model))
    turbulence = _turbulence(mesh, geometry, k_in, omega_in)

    _, _, omega = hybrid_initialize(momentum, turbulence)

    w = jnp.log(omega)
    owner, neighbour = mesh.face_cells.owner, mesh.face_cells.neighbour
    interior = neighbour >= 0
    max_face_jump = float(jnp.max(jnp.abs(w[owner[interior]] - w[neighbour[interior]])))
    # The wall-cell fixation value against the interior interpolant -- the jump the wall-cells-only seed
    # would leave across the first face (the cliff this profile replaces).
    wall_value = float(
        jnp.max(
            omega_wall_value(
                turbulence.molecular_viscosity[turbulence.wall_cells],
                turbulence.wall_distance[turbulence.wall_cells],
                model,
            )
        )
    )
    cliff_jump = float(jnp.log(wall_value) - jnp.log(omega_in))
    assert max_face_jump < 0.5 * cliff_jump  # a ramp set by the grading, not the wall-to-core cliff
    assert max_face_jump < 3.0  # ~2 log(growth) per face, an absolute bound independent of Reynolds


def test_hybrid_initialize_floors_inlet_driven_k_at_the_turbulent_level() -> None:
    """An inlet-driven wall-bounded channel starts with a turbulent-level interior k, not laminar.

    The k Laplace interpolant is pulled toward its wall Dirichlet(0) values, and on a domain several
    heights long the walls dominate the small inlet patch by area, so the raw interior k collapses
    orders of magnitude below ``k_in`` -- the *laminar* field (``nu_t = k/omega ~ 0``). A turbulent case
    started there must then grow k across the whole interior, a swing the coupled Newton must absorb. The
    IC floors k at the inlet turbulence level (the interpolant's own maximum) so it starts turbulent.
    """
    lx, ly = (
        8.0,
        1.0,
    )  # long enough that the raw interpolant collapses (this is a real test of the floor)
    mesh, geometry, momentum = _channel(nx=96, ny=24, lx=lx, ly=ly)
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), 0.05))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07, model))
    turbulence = _turbulence(mesh, geometry, k_in, omega_in)

    # Establish that the raw interpolant really does collapse here -- median orders of magnitude below
    # k_in -- so the assertion below is testing the floor, not a short channel where it barely matters.
    raw, _ = laplace_field(
        mesh, geometry, turbulence.k_boundary, gradient_scheme=CompactGreenGauss()
    )
    assert float(jnp.median(raw)) < 0.01 * k_in

    _, k, _ = hybrid_initialize(momentum, turbulence)
    # The floor lifts the whole interior to the inlet level: even the least cell is turbulent, not ~0.
    assert float(jnp.min(k)) >= 0.5 * k_in
    assert float(jnp.median(k)) >= 0.5 * k_in


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _periodic_channel(beta=0.0035, mu_factor=1.0):
    """A streamwise-periodic channel driven by a body force: no inlet, every patch a wall."""
    mesh = structured_grid_2d(4, 24, lx=1.0, ly=2.0, periodic=("x",), named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU * mu_factor), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
        body_force=(beta, 0.0),
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions({"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}),
        omega_boundary=BoundaryConditions({"bottom": ZeroGradient(), "top": ZeroGradient()}),
    )
    return mesh, momentum, turbulence


def test_hybrid_initialize_starts_a_body_force_channel_in_the_turbulent_regime() -> None:
    """With no inlet, the k interpolant is harmonic between zero wall values -- identically zero.

    Started there the closure has no turbulence at all (``nu_t = 0``), which is not merely a poor
    guess but the *laminar* problem. The equilibrium level from the friction velocity replaces it.
    """
    _, momentum, turbulence = _periodic_channel(beta=0.0035)
    _, k, omega = hybrid_initialize(momentum, turbulence)

    u_tau = (0.0035 * 1.0 / RHO) ** 0.5  # h = V/A_wall = 1 for this ly = 2 channel
    assert float(jnp.min(k)) == pytest.approx(u_tau**2 / SSTModel().beta_star ** 0.5)
    assert float(jnp.min(omega)) > 0.0  # a pure-Neumann omega solve would leave the interior empty


def test_hybrid_initialize_gives_a_developed_channel_eddy_viscosity() -> None:
    """The equilibrium levels must land at the ``~0.09 u_tau h`` eddy viscosity a channel carries.

    Getting k right but omega wrong would be worse than either: ``nu_t = k / omega`` with omega at
    its floor is enormous, so both come from the same friction velocity.
    """
    _, momentum, turbulence = _periodic_channel(beta=0.0035)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    nu_t = turbulence.eddy_viscosity(momentum.velocity_fields(flow).gradient, k, omega)

    u_tau = (0.0035 * 1.0 / RHO) ** 0.5
    assert float(jnp.max(nu_t)) == pytest.approx(0.09 * u_tau * 1.0, rel=0.05)
    # The degenerate interpolant would leave k at its 1e-8 floor, i.e. nu_t ~ 1e-9 -- no
    # turbulence at all. Pin the gap so a regression to that start cannot pass.
    assert float(jnp.max(nu_t)) > 1e-4
