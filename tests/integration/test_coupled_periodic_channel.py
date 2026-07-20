"""Integration: the monolithic coupled RANS Newton solve on a *periodic* body-force channel.

The companion :mod:`test_coupled_rans` drives the coupled solve on an inlet/outlet channel. This
module is the streamwise-**periodic** counterpart -- two no-slip walls, periodic in ``x``, driven by
a uniform ``body_force`` with the pressure pinned at one cell -- the setup a fully-developed turbulent
channel actually uses. It is a distinct convergence path for two reasons the inlet case never
exercises:

* The hybrid initial condition for a body-force domain is the **exactly uniform plug**
  ``scales.body_force_velocity`` (``u_y = 0`` identically), whose velocity gradient -- and hence the
  strain magnitude ``S = sqrt(2 S_ij S_ij)`` -- is exactly zero in every interior cell. The
  ``sqrt`` there has an infinite derivative, so before the guarded ``sqrt`` in
  :func:`~aquaflux.turbulence.strain.strain_rate_magnitude` the coupled Jacobian was NaN in every
  interior cell and the Newton march stalled immediately. With the guard the solve self-starts from
  that uniform plug with **no symmetry-breaking perturbation**, and -- since the true developed
  channel is symmetric about its centreline -- converges to a reflection-symmetric field.
* The near-wall ``omega`` stiffness dominates the coupled residual at the start (the same
  high-aspect-ratio stiffness the segregated loop needed the algebraic-multigrid scalar
  preconditioner for), so the march needs the convection-diffusion AMG the continuation policy
  carries; both the ``air`` and ``twolevel`` coarsenings clear it.

The segregated Picard loop is **not** used to cross-check the fixed point here: on this body-force
channel its outer globalization does not converge (the failure the monolithic engine exists to
overcome), so the fixed point is cross-validated by solving with two independent AMG coarsenings and
checking they land on the same root -- the periodic analogue of the inlet case's coupled-vs-segregated
agreement.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import SSTModel, SSTTurbulence, hybrid_initialize
from aquaflux.turbulence.coupled import CoupledRANS, solve_coupled

RHO, U_B, H = 1.0, 1.0, 2.0
RE_B, NY, GROWTH, BETA0 = 20000, 48, 1.13, 0.004
NU = U_B * H / RE_B  # 1e-4
K_FLOOR = 1e-8  # the hybrid IC's floor; asserted strictly inactive at the converged state
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}
MAX_STEPS = 300


def _periodic_channel():
    """Build the periodic body-force turbulent channel (mesh, momentum, turbulence)."""
    y_nodes = graded_nodes(NY, H, GROWTH)
    mesh = structured_grid_2d(
        4, NY, lx=1.0, ly=H, periodic=("x",), named_boundaries=True, y_nodes=y_nodes
    )
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
        body_force=(BETA0, 0.0),
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


@pytest.fixture(scope="module")
def case():
    """The channel and the ``air``-preconditioned coupled solution, self-started from the hybrid IC."""
    mesh, momentum, turbulence = _periodic_channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    # The hybrid IC is the exactly-symmetric uniform plug (u_y == 0): no perturbation is applied,
    # so this exercises the guarded-sqrt strain fix directly.
    flow0, k0, omega0 = hybrid_initialize(momentum, turbulence)
    flow, k, omega = solve_coupled(
        coupled, flow0, k0, omega0, method="air", max_steps=MAX_STEPS, **PRECONDITIONER
    )
    return {
        "mesh": mesh,
        "momentum": momentum,
        "turbulence": turbulence,
        "coupled": coupled,
        "start": (flow0, k0, omega0),
        "solution": (flow, k, omega),
    }


@pytest.mark.slow
def test_coupled_newton_converges_on_a_periodic_body_force_channel(case) -> None:
    """The coupled solve self-starts from the uniform-plug hybrid IC and converges to a healthy field.

    The uniform plug has ``S = 0`` in every interior cell; this is the case whose NaN Jacobian used to
    stall the march. It now converges to machine precision with a genuinely turbulent, floor-free
    field that is symmetric about the channel centreline -- reached with no symmetry-breaking start.
    """
    coupled, momentum, turbulence = case["coupled"], case["momentum"], case["turbulence"]
    flow, k, omega = case["solution"]

    # Converged to machine precision: a genuine root of the unfrozen coupled residual.
    residual_norm = float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow, k, omega))))
    assert residual_norm < 1e-8

    # Genuinely turbulent, not the laminar (nu_t = 0) branch the degenerate IC would have started.
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 10.0

    # The realizability floors are strictly inactive at the converged state (adjoint honesty): k sits
    # orders of magnitude above its floor, and omega is strictly positive. Not tautological -- the
    # coupled residual carries no in-residual floor, so a diverged field would violate these.
    assert float(jnp.min(k)) > 100.0 * K_FLOOR
    assert float(jnp.min(omega)) > 0.0

    # The developed channel is symmetric about y = H/2, and the coupled solve reaches that symmetry
    # from the exactly-symmetric plug with no perturbation: the wall-normal velocity is ~0 and the
    # streamwise profile equals its mirror (cells sorted by y match the reverse order).
    velocity, _ = momentum.unpack(flow)
    u_x = np.asarray(velocity[:, 0])
    u_y = np.asarray(velocity[:, 1])
    order = np.argsort(np.asarray(case["mesh"].geometry().cell.centroid)[:, 1])
    u_sorted = u_x[order]
    assert np.max(np.abs(u_y)) / np.max(np.abs(u_x)) < 1e-8
    assert np.max(np.abs(u_sorted - u_sorted[::-1])) / np.max(np.abs(u_sorted)) < 1e-6


@pytest.mark.slow
def test_coupled_fixed_point_is_amg_method_independent(case) -> None:
    """Two independent AMG coarsenings converge to the same root -- it is the true coupled fixed point.

    The segregated loop does not converge on this body-force channel, so the fixed point is
    cross-validated the way the inlet case uses the segregated solution: an independent solve (the
    ``twolevel`` coarsening in place of ``air``) must land on the same state to solver tolerance.
    """
    coupled = case["coupled"]
    flow0, k0, omega0 = case["start"]
    flow_air, k_air, omega_air = case["solution"]

    flow_tl, k_tl, omega_tl = solve_coupled(
        coupled, flow0, k0, omega0, method="twolevel", max_steps=400, **PRECONDITIONER
    )

    assert float(jnp.linalg.norm(flow_tl - flow_air) / jnp.linalg.norm(flow_air)) < 1e-6
    assert float(jnp.linalg.norm(k_tl - k_air) / jnp.linalg.norm(k_air)) < 1e-6
    assert float(jnp.linalg.norm(omega_tl - omega_air) / jnp.linalg.norm(omega_air)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
