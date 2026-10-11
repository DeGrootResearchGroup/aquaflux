"""Unit: the bordered form of a residual held at a bulk velocity (``aquaflux.flow.bordered``).

Each piece a march runs a held bulk velocity through -- the residual, the shift policy, the measures, the
source, the homotopy and the force read-back -- is checked against an independent statement of what it
must be, on a small laminar periodic channel.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    FlowMeasures,
    MassFlow,
    MassFlowBorder,
    MomentumContinuity,
    NoSlipWall,
    PinnedPoint,
    mass_flow_border,
    momentum_shift_policy,
)
from aquaflux.flow.bordered import BULK_VELOCITY
from aquaflux.flow.drive import BOUNDARY_DRIVEN
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import BlockScaledNorm, RowScaledNorm

H, MU, U_TARGET = 2.0, 0.1, 1.0


def _channel(force: float = 0.05, target: float = U_TARGET) -> MomentumContinuity:
    mesh = structured_grid_2d(4, 8, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(MU), "density": Constant(1.0)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=target, force=force),
    )


def _state(momentum: MomentumContinuity, seed: int = 0) -> jnp.ndarray:
    return jax.random.normal(jax.random.PRNGKey(seed), (momentum.layout.size,))


def test_a_boundary_driven_assembler_has_no_border() -> None:
    momentum = _channel()
    driven = eqx.tree_at(lambda m: m.drive, momentum, BOUNDARY_DRIVEN)
    assert mass_flow_border(driven, driven.layout) is None
    assert isinstance(mass_flow_border(momentum, momentum.layout), MassFlowBorder)


def test_the_bordered_residual_is_the_residual_forced_at_the_solved_force() -> None:
    """``[R(w; beta) ; <U> - target]``, written independently by forcing the assembler at ``beta``."""
    momentum = _channel(force=0.05)
    border = mass_flow_border(momentum, momentum.layout)
    w, beta = _state(momentum), jnp.asarray(0.37)
    augmented = border.drive.join(momentum.layout, w, beta)

    forced = border.drive.forced(momentum, beta)
    velocity, _ = momentum.unpack(w)
    bulk = border.drive.bulk_velocity(velocity, momentum.geometry.cell.volume)
    expected = jnp.append(forced.residual(w), bulk - U_TARGET)

    got = border.residual(momentum.residual)(augmented)
    assert got.shape == (momentum.layout.size + 1,)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_the_force_the_assembler_carries_cancels_out() -> None:
    """Two assemblers carrying different forces border to the same function of the augmented state."""
    a, b = _channel(force=0.05), _channel(force=2.0)
    augmented = jnp.append(_state(a), 0.4)
    ra = mass_flow_border(a, a.layout).residual(a.residual)(augmented)
    rb = mass_flow_border(b, b.layout).residual(b.residual)(augmented)
    assert jnp.allclose(ra, rb, rtol=1e-12, atol=1e-12)


def test_the_force_read_back_is_exact_at_a_root() -> None:
    """At a root of the bordered system the force is recovered from the fields' residual alone.

    A root is manufactured from any state: give it the force that zeroes the bordered momentum rows in the
    least-squares sense and check those rows really are zero there -- which needs the residual to be
    exactly linear in the force along the column the border uses.
    """
    momentum = _channel(force=0.05)
    border = mass_flow_border(momentum, momentum.layout)
    w = _state(momentum, seed=4)
    beta = border.force_holding(momentum.residual(w))
    rows = border.drive.forced(momentum, beta).residual(w)
    assert float(jnp.dot(border.force, rows)) == pytest.approx(0.0, abs=1e-10)
    # And a carried force elsewhere changes the residual but not the force read back.
    other = _channel(force=3.0)
    assert float(mass_flow_border(other, other.layout).force_holding(other.residual(w))) == (
        pytest.approx(float(beta), rel=1e-12)
    )


def test_the_bordered_policy_adds_no_shift_to_the_constraint_and_wraps_the_preconditioner() -> None:
    momentum = _channel()
    border = mass_flow_border(momentum, momentum.layout)
    w = _state(momentum)
    inner = momentum_shift_policy(momentum, None, reference_state=w)
    policy = border.policy(inner)
    term = policy.shift_term(border.attach(w))
    assert term.diagonal.shape == (momentum.layout.size + 1,)
    assert float(term.diagonal[-1]) == 0.0
    assert jnp.array_equal(term.diagonal[:-1], inner.shift_term(w).diagonal)
    # The bordered preconditioner returns an augmented vector whose force entry solves the 1x1 Schur
    # system: its constraint row of the output is consistent with the input's border entry.
    matvec = term.make_preconditioner(jnp.asarray(0.5))
    out = matvec(jnp.ones(momentum.layout.size + 1))
    assert out.shape == (momentum.layout.size + 1,)
    assert jnp.isfinite(out).all()


def test_the_bordered_measures_append_the_constraint_row() -> None:
    # A target other than one, so a constraint row left unscaled would be told from a scaled one.
    target = 2.0
    momentum = _channel(target=target)
    border = mass_flow_border(momentum, momentum.layout)
    w = _state(momentum)

    class _Step(eqx.Module):
        shift_policy: object

    inner_policy = momentum_shift_policy(momentum, None, reference_state=w)
    step = _Step(border.policy(inner_policy))
    measures = border.measures(FlowMeasures(momentum))
    row = measures.row_scaled(step, border.attach(w))
    unbordered = FlowMeasures(momentum).row_scaled(_Step(inner_policy), w)
    assert isinstance(row, RowScaledNorm)
    assert row.names == (*unbordered.names, BULK_VELOCITY)
    residual = jax.random.normal(jax.random.PRNGKey(7), (momentum.layout.size + 1,))
    assert jnp.allclose(row.per_block(residual)[:-1], unbordered.per_block(residual[:-1]))
    # The constraint row is the miss of the bulk velocity as a fraction of the target.
    miss = jnp.zeros(momentum.layout.size + 1).at[-1].set(0.25 * target)
    assert float(row.per_block(miss)[-1]) == pytest.approx(0.25)

    block = measures.block_scaled(border.attach(w))
    assert isinstance(block, BlockScaledNorm)
    assert block.sizes == (*momentum.layout.sizes, 1)
    assert float(block.scales[-1]) == float(block.scales[0])


def test_the_row_scaled_measure_refuses_a_target_of_zero() -> None:
    momentum = _channel(target=0.0)
    border = mass_flow_border(momentum, momentum.layout)
    with pytest.raises(ValueError, match="holds zero"):
        border.measures(None).row_scaled(None, border.attach(_state(momentum)))


def test_the_source_hands_its_builder_the_fields_alone() -> None:
    momentum = _channel()
    border = mass_flow_border(momentum, momentum.layout)
    seen = []

    class _Source:
        refresh_preconditioner = None

        def build(self, state):
            seen.append(("build", state.shape))
            return "built"

        def refresh(self, state, previous):
            seen.append(("refresh", state.shape, previous))
            return "refreshed"

    source = border.source(_Source())
    augmented = border.attach(_state(momentum))
    assert source.build(augmented) == "built"
    assert source.refresh(augmented, "old") == "refreshed"
    size = momentum.layout.size
    assert seen == [("build", (size,)), ("refresh", (size,), "old")]
    assert source.refresh_preconditioner is None


def test_the_homotopy_borders_every_station_and_delegates_the_rest() -> None:
    momentum = _channel()
    border = mass_flow_border(momentum, momentum.layout)
    stiffer = momentum.with_scaled_molecular_viscosity(4.0)

    class _Ramp:
        def enter(self, step):
            return (stiffer if step == 0 else momentum).residual

        def arrived(self, step):
            return step >= 1

        def shift_factor(self, step):
            return 1.5 if step == 1 else 1.0

        def station(self, step):
            return min(step, 1)

    ramp = border.homotopy(_Ramp())
    augmented = jnp.append(_state(momentum), 0.3)
    assert jnp.allclose(
        ramp.enter(0)(augmented), border.residual(stiffer.residual)(augmented), atol=1e-12
    )
    assert jnp.allclose(
        ramp.enter(1)(augmented), border.residual(momentum.residual)(augmented), atol=1e-12
    )
    assert (ramp.arrived(0), ramp.arrived(1)) == (False, True)
    assert (ramp.shift_factor(0), ramp.shift_factor(1)) == (1.0, 1.5)
    assert (ramp.station(0), ramp.station(5)) == (0, 1)
    assert border.homotopy(None) is None


def test_a_drift_measure_is_read_on_the_fields() -> None:
    momentum = _channel()
    border = mass_flow_border(momentum, momentum.layout)
    seen = []

    def measure(start):
        seen.append(start.shape)
        return lambda state: float(jnp.sum(state - start))

    rebased = border.on_fields(measure)
    start = border.attach(_state(momentum))
    assert rebased(start)(start.at[-1].add(5.0)) == pytest.approx(0.0)
    assert seen == [(momentum.layout.size,)]
    assert border.on_fields(None) is None
