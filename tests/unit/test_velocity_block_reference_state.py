"""Which flow state the block-diagonal coupled policy freezes its convection velocity block at.

The policy freezes every operator it holds at its own reference state, and the convection velocity
block's mass flux is frozen there too. The block builder's default, the uniform flow the boundary
conditions imply, is far from divergence-free in a duct that changes section; in a contracting one the
frozen upwind operator loses positivity on its first coarse level and the hierarchy refuses to build,
so a coupled solve with the default preconditioner could not start. A reference flow at rest (a closed
domain's potential flow) carries no convection to freeze at, so there the policy keeps the default.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    ConvectionTwoLevel,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.flow.block_preconditioner import _characteristic_reference_state
from aquaflux.initialization import hybrid_initialize
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import ScalarTwoLevel, SSTModel, SSTTurbulence, inlet_k, inlet_omega
from aquaflux.turbulence.coupled import CoupledRANS, _coupled_shift_policy

# A Gaussian bump of 0.3 channel heights on the lower wall: the smallest mesh measured on which the
# boundary-derived uniform flow gives a velocity hierarchy with a negative coarse diagonal.
NX, NY, LENGTH, CREST, HEIGHT, WIDTH = 36, 12, 4.0, 1.0, 0.3, 0.35
NU = 1e-4


def _contracting_duct() -> CoupledRANS:
    mesh = structured_grid_2d(
        NX, NY, lx=LENGTH, ly=1.0, named_boundaries=True, y_nodes=graded_nodes(NY, 1.0, 1.15)
    )
    coords = np.asarray(mesh.node_coords)
    x, y = coords[:, 0], coords[:, 1]
    wall = HEIGHT * np.exp(-(((x - CREST) / WIDTH) ** 2))
    mesh = eqx.tree_at(
        lambda m: m.node_coords, mesh, jnp.asarray(np.stack([x, wall + y * (1.0 - wall)], axis=1))
    )
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(1.0), 0.05))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07, model))
    properties = PropertyModel({"viscosity": Constant(NU), "density": Constant(1.0)})
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(1.0, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=CompactGreenGauss(),
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
    return CoupledRANS.build(momentum, turbulence)


@pytest.fixture(scope="module")
def duct():
    coupled = _contracting_duct()
    state = coupled.state_from_physical(*hybrid_initialize(coupled))
    return coupled, state


def _applied(preconditioner: BlockPreconditioner, state: jnp.ndarray) -> np.ndarray:
    """The preconditioner applied, at ``state``, to one fixed vector -- a fingerprint of the operator."""
    vector = jax.random.normal(jax.random.PRNGKey(0), state.shape)
    return np.asarray(preconditioner.factory()(state)(vector))


def test_the_uniform_boundary_flow_cannot_build_the_block_on_a_contracting_duct(duct) -> None:
    """The failure the coupled policy used to meet: the plug's hierarchy has a negative coarse diagonal."""
    coupled, _ = duct
    with pytest.raises(ValueError, match="operator diagonal must be finite and strictly positive"):
        BlockPreconditioner.build(
            coupled.momentum,
            velocity=ConvectionTwoLevel(),
            reference_state=_characteristic_reference_state(coupled.momentum),
            strength_threshold=0.25,
        )


def test_the_coupled_policy_freezes_its_velocity_block_at_its_own_reference_flow(duct) -> None:
    """The policy builds on the duct, and its velocity block is the one frozen at its reference flow.

    The comparison block is assembled independently, from the same eddy viscosity and the reference
    state's flow, so a policy that froze at any other state -- the boundary-derived plug, or a
    rescaled flow -- would give a different operator (or, for the plug, none at all).
    """
    coupled, state = duct
    policy = _coupled_shift_policy(coupled, state, ScalarTwoLevel())

    flow, _, _ = coupled.physical_fields(state)
    expected = _velocity_block_at(coupled, state, flow)
    np.testing.assert_allclose(
        _applied(policy.flow_preconditioner, flow), _applied(expected, flow), rtol=1e-12, atol=0.0
    )


def _closed_cavity() -> CoupledRANS:
    mesh = structured_grid_2d(6, 6, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    properties = PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)})
    walls = ("bottom", "left", "right", "top")
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=CompactGreenGauss(),
        wall_patches=list(walls),
        k_boundary=BoundaryConditions({w: Dirichlet(0.0) for w in walls}),
        omega_boundary=BoundaryConditions({w: ZeroGradient() for w in walls}),
    )
    return CoupledRANS.build(momentum, turbulence)


def _velocity_block_at(coupled: CoupledRANS, state: jnp.ndarray, reference) -> BlockPreconditioner:
    """The block the policy should build, assembled independently at ``reference``'s flow."""
    flow, k, omega = coupled.physical_fields(state)
    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
    return BlockPreconditioner.build(
        coupled.momentum.with_eddy_viscosity(closure.nu_t),
        velocity=ConvectionTwoLevel(),
        reference_state=reference,
        strength_threshold=0.25,
    )


def test_a_reference_flow_at_rest_keeps_the_boundary_derived_flow() -> None:
    """On a closed lid cavity the hybrid start is at rest, and the block stays at the lid-driven flow.

    Freezing at the state at rest would build a viscous-only block (and say so in a warning); the
    policy instead leaves the builder its own default, as it did before it froze at its reference.

    The cavity is at a low Peclet number: at Reynolds number 100 the boundary-derived flow's own
    hierarchy has a negative coarse diagonal on this mesh too, which is a property of that default and
    not of the choice pinned here.
    """
    coupled = _closed_cavity()
    state = coupled.state_from_physical(*hybrid_initialize(coupled))
    flow, _, _ = coupled.physical_fields(state)
    assert float(jnp.max(jnp.abs(coupled.momentum.mass_flux(flow)))) == 0.0  # the premise
    policy = _coupled_shift_policy(coupled, state, ScalarTwoLevel())
    np.testing.assert_allclose(
        _applied(policy.flow_preconditioner, flow),
        _applied(_velocity_block_at(coupled, state, None), flow),
        rtol=0.0,
        atol=0.0,
    )
