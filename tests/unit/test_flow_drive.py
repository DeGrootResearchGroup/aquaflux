"""Unit: what drives a flow is one object, and a solved driving force is not a prescribed one.

``MomentumContinuity`` used to carry a ``body_force`` leaf that meant two different things -- a fixed
input on a periodic channel, and the live iterate of a bulk-velocity solve. The two are now two
classes: a prescribed force is a :class:`~aquaflux.flow.UniformBodyForce` source like any other force
per unit volume, and a *solved* one is a :class:`~aquaflux.flow.MassFlow` drive, which is also where
everything the bordered residual needs lives -- the layout the multiplier is attached through, the seed
and the answer, the border column and row, and the bulk-velocity average.

Two things here are worth stating as the reason the tests exist rather than as coverage. A drive
reports **no force at all** as ``None``, never as a zero vector, because a mass-flow solve is
force-driven at the instant its multiplier happens to be zero -- which is every solve's first step, and
the distinction an initializer reading the value alone got wrong. And the border is attached through
the **declared layout**, so the position of the multiplier has one definition rather than six.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    BoundaryDriven,
    Drive,
    MassFlow,
    MomentumContinuity,
    MomentumSource,
    NoSlipWall,
    PinnedPoint,
    UniformBodyForce,
    mass_flow_drive,
    refuse_a_constraint_this_solve_cannot_hold,
)
from aquaflux.flow.drive import BODY_FORCE
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import CellFields, FieldLayout, GlobalDofs, SubLayout
from aquaflux.vectors import scale

H, MU, RHO, TARGET = 2.0, 0.1, 1.0, 1.0


def _channel(drive=None, sources=()):
    """A small streamwise-periodic channel, driven however the caller says."""
    mesh = structured_grid_2d(4, 6, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(RHO * MU), "density": Constant(RHO)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        sources=sources,
        **({} if drive is None else {"drive": drive}),
    )


# --- no force at all is not a force of zero ---------------------------------------------------


def test_a_boundary_driven_flow_reports_no_force_rather_than_a_zero_one() -> None:
    """``None``, not ``zeros(dim)`` -- the two answer different questions.

    A consumer deciding "is this domain force-driven" reads this, and a zero vector would make the
    answer "no" indistinguishable from "yes, and the force is currently zero".
    """
    assert BoundaryDriven().volumetric_force(2) is None
    assert BoundaryDriven().volumetric_force(3) is None


def test_a_mass_flow_drive_at_zero_still_reports_a_force() -> None:
    """The seed of every constrained solve is a multiplier that has not moved yet.

    Reported as a force of zero rather than as no force, so "force-driven" does not flicker off on the
    first step of the march that is about to find the force.
    """
    force = MassFlow(target=TARGET, force=0.0).volumetric_force(2)
    assert force is not None
    assert jnp.array_equal(force, jnp.zeros(2))


def test_the_solved_force_acts_only_along_the_flow_direction() -> None:
    """A scalar multiplier on one axis, zero on the others -- what makes ``dR/dbeta = -V`` exact."""
    assert jnp.array_equal(
        MassFlow(target=TARGET, flow_direction=1, force=0.7).volumetric_force(3),
        jnp.array([0.0, 0.7, 0.0]),
    )


def test_a_prescribed_force_and_a_solved_one_both_reach_the_summed_force() -> None:
    """``uniform_body_force`` is the one place that answers "what is pushing this flow".

    Reading either route alone answers zero for a channel driven by the other, which is what made the
    initializer start a source-driven channel at rest.
    """
    prescribed = _channel(sources=(UniformBodyForce(jnp.array([0.3, 0.0])),))
    solved = _channel(drive=MassFlow(target=TARGET, force=0.3))
    both = _channel(
        drive=MassFlow(target=TARGET, force=0.3),
        sources=(UniformBodyForce(jnp.array([0.3, 0.0])),),
    )
    assert jnp.allclose(prescribed.uniform_body_force(), jnp.array([0.3, 0.0]))
    assert jnp.allclose(solved.uniform_body_force(), jnp.array([0.3, 0.0]))
    assert jnp.allclose(both.uniform_body_force(), jnp.array([0.6, 0.0]))
    assert jnp.allclose(_channel().uniform_body_force(), jnp.zeros(2))


class _Drag(MomentumSource):
    """A velocity-dependent source: a force per unit volume that is not uniform."""

    coefficient: float

    def source(self, fields, geometry, properties):
        return -self.coefficient * scale(fields.velocity, geometry.cell.volume)

    def face_force(self, geometry, properties):
        return None

    def diagonal(self, velocity, geometry, properties):
        return jnp.full_like(velocity, self.coefficient)


def test_a_state_dependent_source_is_not_part_of_the_uniform_force() -> None:
    """Only the uniform part, because only that closes a global force balance.

    A drag term is a force per unit volume too, but it varies with the state, so summing it here would
    hand the initializer a plug velocity derived from a balance that does not hold.
    """
    channel = _channel(sources=(_Drag(0.5), UniformBodyForce(jnp.array([0.3, 0.0]))))
    assert jnp.allclose(channel.uniform_body_force(), jnp.array([0.3, 0.0]))


# --- the border goes through the declared layout ------------------------------------------------


def _fields() -> FieldLayout:
    return FieldLayout(5, (CellFields("velocity", 2), CellFields("pressure", 1)))


def test_the_layout_gains_exactly_one_global_dof_after_the_fields() -> None:
    fields = _fields()
    bordered = MassFlow(target=TARGET).layout(fields)
    assert bordered.blocks[:-1] == fields.blocks
    assert bordered.blocks[-1] == GlobalDofs(BODY_FORCE, 1)
    assert bordered.size == fields.size + 1


def test_a_boundary_driven_state_gains_nothing() -> None:
    fields = _fields()
    assert BoundaryDriven().layout(fields) is fields
    state = jnp.arange(fields.size, dtype=float)
    assert jnp.array_equal(BoundaryDriven().driven_state(fields, state), state)
    assert jnp.array_equal(BoundaryDriven().fields_state(fields, state), state)
    assert BoundaryDriven().settled(fields, state) == BoundaryDriven()


def test_the_border_entry_lands_where_the_layout_says_it_does() -> None:
    """``join`` and the layout must agree, or the measure weighs a partition the state is not in.

    Checked against ``slice_of`` rather than against ``[-1]``: the point of the block is that the
    position is *derived*, and a hand-written index agreeing with it today is the defect, not the test.
    """
    drive = MassFlow(target=TARGET)
    fields = _fields()
    state = jnp.arange(fields.size, dtype=float)
    joined = drive.join(fields, state, jnp.asarray(7.5))
    where = drive.layout(fields).slice_of(BODY_FORCE)
    assert float(joined[where][0]) == 7.5
    assert jnp.array_equal(joined[: where.start], state)


def test_join_and_split_are_inverses() -> None:
    drive = MassFlow(target=TARGET)
    fields = _fields()
    state = jnp.arange(fields.size, dtype=float)
    block, border = drive.split(fields, drive.join(fields, state, jnp.asarray(-2.0)))
    assert jnp.array_equal(block, state)
    assert float(border) == -2.0


def test_the_seed_is_the_drives_own_force_and_the_answer_comes_back_on_it() -> None:
    """The solve starts at the multiplier the drive carries and leaves the converged one there.

    That round trip is the whole interface a segregated outer loop threads ``beta`` through.
    """
    drive = MassFlow(target=TARGET, force=0.04)
    fields = _fields()
    state = jnp.arange(fields.size, dtype=float)
    seeded = drive.driven_state(fields, state)
    assert float(drive.split(fields, seeded)[1]) == pytest.approx(0.04)

    converged = drive.join(fields, state, jnp.asarray(0.31))
    settled = drive.settled(fields, converged)
    assert float(settled.force) == pytest.approx(0.31)
    assert settled.target == drive.target and settled.flow_direction == drive.flow_direction


# --- the assembler side --------------------------------------------------------------------------


def test_forcing_the_assembler_moves_the_multiplier_and_nothing_else() -> None:
    """One iterate written into one leaf: the assembler is otherwise the same object."""
    drive = MassFlow(target=TARGET, force=0.05)
    momentum = _channel(drive=drive)
    forced = drive.forced(momentum, jnp.asarray(0.42))
    assert float(forced.drive.force) == pytest.approx(0.42)
    assert float(momentum.drive.force) == pytest.approx(0.05)  # unchanged, it is frozen
    assert eqx.tree_at(lambda m: m.drive.force, forced, momentum.drive.force) == momentum


def test_the_solved_force_enters_the_residual_as_minus_beta_times_volume() -> None:
    """The drive's force is subtracted per component, exactly as a uniform source would be.

    This is what makes the analytic border column ``a = dR/dbeta = -V`` correct; a drive whose force
    reached the residual scaled, signed or directed differently would leave the bordered solve solving
    a system its preconditioner does not describe.
    """
    beta = 0.35
    at_rest = _channel()
    driven = _channel(drive=MassFlow(target=TARGET, force=beta))
    state = at_rest.initial_state()
    difference, _ = at_rest.unpack(driven.residual(state) - at_rest.residual(state))
    volume = at_rest.geometry.cell.volume
    assert jnp.allclose(difference[:, 0], -beta * volume)
    assert jnp.allclose(difference[:, 1], 0.0)


def test_the_padded_border_is_the_flow_border_and_zeros_elsewhere() -> None:
    """``beta`` enters only momentum and ``<U>`` reads only velocity, so every other block is zero.

    Derived from the outer layout rather than from a count of its blocks, so a coupled state that grows
    a field cannot be padded to the wrong width.
    """
    drive = MassFlow(target=TARGET)
    momentum = _channel(drive=drive)
    flow = momentum.layout
    n = momentum.mesh.n_cells
    outer = FieldLayout(n, (SubLayout("flow", flow), CellFields("k", 1), CellFields("omega", 1)))

    force_flow, average_flow = drive.constraint_vectors(momentum)
    force, average = drive.constraint_vectors_in(momentum, outer)

    assert force.shape == (outer.size,)
    assert jnp.array_equal(force[: flow.size], force_flow)
    assert jnp.array_equal(average[: flow.size], average_flow)
    assert jnp.array_equal(force[flow.size :], jnp.zeros(outer.size - flow.size))
    assert jnp.array_equal(average[flow.size :], jnp.zeros(outer.size - flow.size))


def test_the_bulk_velocity_is_the_volume_weighted_average_of_its_own_component() -> None:
    drive = MassFlow(target=TARGET, flow_direction=1)
    velocity = jnp.array([[1.0, 4.0], [2.0, 6.0]])
    volume = jnp.array([1.0, 3.0])
    assert float(drive.bulk_velocity(velocity, volume)) == pytest.approx((4.0 + 18.0) / 4.0)


def test_an_assembler_with_no_mass_flow_drive_is_refused_by_name() -> None:
    """One refusal for every entry point that borders a residual, raised before anything is built."""
    with pytest.raises(TypeError, match=r"here.*MassFlow|MassFlow"):
        mass_flow_drive(_channel(), "here")
    drive = MassFlow(target=TARGET)
    assert mass_flow_drive(_channel(drive=drive), "here") == drive


def test_a_solve_that_marches_the_fields_alone_refuses_a_constraint_it_cannot_hold() -> None:
    """The dangerous case is the one that would WORK: a solve at the seed force, reported as the answer.

    A ``MassFlow`` drive's force is a perfectly good force, so an unbordered march holds it fixed and
    converges to whatever bulk velocity it happens to produce -- the constraint silently dropped, with
    nothing in the result to say so. Refused at both entry points that march the fields.
    """
    driven = _channel(drive=MassFlow(target=TARGET, force=0.05))
    with pytest.raises(TypeError, match="bulk velocity"):
        refuse_a_constraint_this_solve_cannot_hold(driven, "here")
    # A prescribed force is exactly what that solve IS for, so it passes.
    assert (
        refuse_a_constraint_this_solve_cannot_hold(
            _channel(sources=(UniformBodyForce(jnp.array([0.05, 0.0])),)), "here"
        )
        is None
    )


def test_a_force_written_as_a_number_is_the_same_value_as_one_written_as_an_array() -> None:
    """Converted on the way in, so the two are one pytree and a reused solver is not recompiled."""
    assert MassFlow(target=TARGET, force=0.004) == MassFlow(target=TARGET, force=jnp.asarray(0.004))


def test_every_member_of_the_interface_is_abstract() -> None:
    """A default here would let a bordered drive inherit an answer that drops its multiplier.

    Named as a set rather than checked one at a time, so a member added to the interface without an
    abstract declaration fails here instead of silently acquiring whichever behaviour the base had.
    """
    assert Drive.__abstractmethods__ == frozenset(
        {"volumetric_force", "layout", "driven_state", "fields_state", "settled"}
    )
