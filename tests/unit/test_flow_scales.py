"""The characteristic velocity scale a flow problem is driven at.

Two drives, two rules: a prescribed boundary velocity gives the fastest patch speed directly, while a
body-force-driven domain (a streamwise-periodic channel prescribes velocity nowhere) has to be sized
from the global force balance against the wall drag. The laminar branch of that balance is exact, so
it is checked against the closed-form plane-channel bulk velocity.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.flow.scales import (
    BULK_TO_FRICTION_RATIO,
    body_force_speed,
    body_force_velocity,
    characteristic_velocity,
    friction_velocity,
    hydraulic_length,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss

H, LX, RHO = 1.0, 2.0, 1.0


def _build(boundary, *, mu=0.1, body_force=None, periodic=False, pin=None):
    kw = {"periodic": ("x",)} if periodic else {}
    mesh = structured_grid_2d(6, 16, lx=LX, ly=H, named_boundaries=True, **kw)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(boundary),
        body_force=body_force,
        pressure_pin=pin,
    )


def _periodic_channel(mu, beta):
    return _build(
        {"bottom": NoSlipWall(), "top": NoSlipWall()},
        mu=mu,
        body_force=(beta, 0.0),
        periodic=True,
        pin=0,
    )


def test_hydraulic_length_is_the_channel_half_height() -> None:
    """``V / A_wall`` for a plane channel is its half-height, independent of the mesh."""
    assert hydraulic_length(_periodic_channel(0.1, 0.1)) == pytest.approx(H / 2.0)


def test_hydraulic_length_is_zero_without_a_wetted_wall() -> None:
    """An all-through-flow domain has nothing for a body force to balance against."""
    channel = _build(
        {"left": VelocityInlet(velocity=(1.0, 0.0)), "right": PressureOutlet(pressure=0.0)}
    )
    assert hydraulic_length(channel) == 0.0


@pytest.mark.parametrize("mu", [0.1, 0.05, 0.01])
def test_body_force_speed_matches_the_analytic_laminar_bulk_velocity(mu) -> None:
    """In the laminar regime the balance is exact: ``U_b = beta H^2 / (12 mu)``."""
    beta = 0.1
    speed = float(body_force_speed(_periodic_channel(mu, beta)))
    assert speed == pytest.approx(beta * H**2 / (12.0 * mu), rel=1e-10)


def test_body_force_speed_caps_at_the_friction_velocity_scaling() -> None:
    """As ``mu`` falls the laminar branch diverges, so the friction-velocity branch must take over."""
    beta, mu = 0.1, 1e-6
    channel = _periodic_channel(mu, beta)
    u_tau = (beta * (H / 2.0) / RHO) ** 0.5
    assert float(body_force_speed(channel)) == pytest.approx(BULK_TO_FRICTION_RATIO * u_tau)
    # The cap is the smaller branch, so it is far below the (unphysical) laminar extrapolation.
    assert float(body_force_speed(channel)) < beta * H**2 / (12.0 * mu)


def test_body_force_speed_is_zero_without_a_force() -> None:
    channel = _build({"bottom": NoSlipWall(), "top": NoSlipWall()}, periodic=True, pin=0)
    assert float(body_force_speed(channel)) == 0.0


def test_characteristic_velocity_follows_the_body_force_direction() -> None:
    """A body-force-driven domain is sized along the force, not across it."""
    velocity = characteristic_velocity(_periodic_channel(0.1, 0.1))
    assert float(velocity[0]) > 0.0
    assert float(velocity[1]) == pytest.approx(0.0)


def test_characteristic_velocity_takes_the_fastest_prescribed_patch() -> None:
    """A prescribed velocity wins outright — the estimate is only for domains that lack one."""
    channel = _build(
        {
            "bottom": NoSlipWall(),
            "top": MovingWall(velocity=(0.25, 0.0)),
            "left": VelocityInlet(velocity=(0.75, 0.0)),
            "right": PressureOutlet(pressure=0.0),
        }
    )
    assert float(jnp.linalg.norm(characteristic_velocity(channel))) == pytest.approx(0.75)


def test_characteristic_velocity_prefers_a_prescribed_velocity_over_a_body_force() -> None:
    """With both drives present the prescribed speed is the real one; the estimate stays unused."""
    channel = _build(
        {
            "bottom": NoSlipWall(),
            "top": NoSlipWall(),
            "left": VelocityInlet(velocity=(0.5, 0.0)),
            "right": PressureOutlet(pressure=0.0),
        },
        body_force=(0.1, 0.0),
    )
    assert float(jnp.linalg.norm(characteristic_velocity(channel))) == pytest.approx(0.5)


def test_characteristic_velocity_is_zero_when_nothing_drives_the_flow() -> None:
    """Every patch a stationary wall and no body force: nothing is making the fluid move."""
    walls = {name: NoSlipWall() for name in ("bottom", "top", "left", "right")}
    assert float(jnp.linalg.norm(characteristic_velocity(_build(walls, pin=0)))) == 0.0


def test_only_solid_patches_shear_the_flow() -> None:
    """``shears_flow`` is what selects the wetted area, so the classification must be exact."""
    assert NoSlipWall().shears_flow()
    assert MovingWall(velocity=(1.0, 0.0)).shears_flow()
    assert not VelocityInlet(velocity=(1.0, 0.0)).shears_flow()
    assert not PressureOutlet(pressure=0.0).shears_flow()


def test_body_force_velocity_ignores_a_moving_wall() -> None:
    """A driven cavity has a prescribed speed but no body force, so it sustains no plug.

    The distinction matters to the initializer: a moving lid drives recirculation with no net
    through-flow, so a uniform plug at the lid speed would violate the three stationary walls.
    """
    cavity = _build(
        {
            "top": MovingWall(velocity=(1.0, 0.0)),
            "bottom": NoSlipWall(),
            "left": NoSlipWall(),
            "right": NoSlipWall(),
        },
        pin=0,
    )
    assert float(jnp.linalg.norm(body_force_velocity(cavity))) == 0.0
    # The characteristic velocity still reports the lid speed — the two answer different questions.
    assert float(jnp.linalg.norm(characteristic_velocity(cavity))) == pytest.approx(1.0)


def test_friction_velocity_closes_the_force_balance() -> None:
    """``u_tau = sqrt(beta h / rho)`` -- the one velocity scale a body-force domain always has."""
    beta = 0.1
    channel = _periodic_channel(0.1, beta)
    assert float(friction_velocity(channel)) == pytest.approx((beta * (H / 2.0) / RHO) ** 0.5)


def test_friction_velocity_is_zero_without_a_body_force() -> None:
    """No force, nothing for the wall drag to balance -- so no scale, and the estimate stays unused."""
    channel = _build({"bottom": NoSlipWall(), "top": NoSlipWall()}, periodic=True, pin=0)
    assert float(friction_velocity(channel)) == 0.0
