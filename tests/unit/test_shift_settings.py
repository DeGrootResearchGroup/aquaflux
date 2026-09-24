"""The pseudo-time shift settings as one value, on every coupled builder.

The three shift settings -- the basis, the velocity parts and the closure's damping ratio -- were loose
keywords copied into every coupled builder's signature. They are one value now: its fields are pinned,
what it leaves unset resolves to the builders' defaults, a set field reaches the shift policy, and a
continuation merges a point's value over the shared one field by field -- which is how the same
settings as separate keywords always combined.
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.solve import CompleteLu, LocalCourantBasis, MaterializedJacobian
from aquaflux.turbulence import (
    BlockDiagonal,
    ConstantDamping,
    CoupledShiftSettings,
    UnpreconditionedScalars,
    coupled_step,
    open_session,
    solve_reynolds_continuation,
)
from aquaflux.turbulence.coupled import (
    _DEFAULT_SHIFT_BASIS,
    _resolved_shift,
    mass_flow_coupled_continuation,
)
from aquaflux.turbulence.march_settings import merged_march_options

from tests.unit.test_coupled_rans import _cavity, _healthy_state, _mass_flow_cavity
from tests.unit.test_reynolds_continuation import _record_solves, _tiny_coupled

_BASIS = LocalCourantBasis(dissipative_weight=0.0)
#: The loose keywords the value replaced -- none may reappear on a builder beside it.
_RETIRED = ("shift_basis", "velocity_shift_parts", "turbulence_damping")


def test_every_coupled_builder_takes_the_value_and_none_of_the_keywords_it_replaced() -> None:
    assert {field.name for field in dataclasses.fields(CoupledShiftSettings)} == {
        "basis",
        "velocity_parts",
        "turbulence_damping",
    }
    for builder in (coupled_step, mass_flow_coupled_continuation):
        parameters = set(inspect.signature(builder).parameters)
        assert "shift" in parameters, builder.__name__
        assert not parameters & set(_RETIRED), builder.__name__


def test_an_unset_shift_resolves_to_the_builders_defaults() -> None:
    basis, parts, damping = _resolved_shift(None)
    assert basis is _DEFAULT_SHIFT_BASIS
    assert parts is None
    assert damping == 1.0
    assert _resolved_shift(CoupledShiftSettings()) == _resolved_shift(None)


def test_a_set_field_passes_through_as_the_very_object_given() -> None:
    damping = ConstantDamping(3.0)
    basis, parts, resolved = _resolved_shift(
        CoupledShiftSettings(basis=_BASIS, turbulence_damping=damping)
    )
    assert basis is _BASIS
    assert parts is None
    assert resolved is damping


def test_the_damping_reaches_the_policy_of_every_family_and_the_mass_flow_builder() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    shift = CoupledShiftSettings(basis=_BASIS, turbulence_damping=2.0)
    relaxation = jnp.asarray(0.5)

    block = coupled_step(
        coupled, state, preconditioner=BlockDiagonal(scalar=UnpreconditionedScalars()), shift=shift
    )
    assert float(block.shift_policy.turbulence_damping.factor(relaxation, None)) == 2.0
    assert block.shift_policy.shift_basis is _BASIS

    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    lu = session.build(state, shift=shift)
    assert float(lu.shift_policy.base.turbulence_damping.factor(relaxation, None)) == 2.0

    mass_flow_mesh, mass_flow_coupled = _mass_flow_cavity(4)
    mass_flow = mass_flow_coupled_continuation(
        mass_flow_coupled,
        _healthy_state(mass_flow_mesh, mass_flow_coupled),
        preconditioner=BlockDiagonal(scalar=UnpreconditionedScalars()),
        shift=shift,
    )
    assert float(mass_flow.shift_policy.inner.turbulence_damping.factor(relaxation, None)) == 2.0


def test_a_point_s_shift_value_is_merged_field_by_field_over_the_shared_one() -> None:
    base = {"shift": CoupledShiftSettings(basis=_BASIS, turbulence_damping=3.0), "rtol": 1e-6}
    override = {"shift": CoupledShiftSettings(turbulence_damping=2.0), "rtol": 1e-8}
    merged = merged_march_options(base, override)
    assert merged["shift"].basis is _BASIS  # kept from the shared value
    assert merged["shift"].turbulence_damping == 2.0  # the point's own wins where it sets one
    assert merged["rtol"] == 1e-8  # a loose keyword is a plain override
    assert merged_march_options(base, {})["shift"] is base["shift"]


def test_the_reynolds_continuation_merges_a_point_s_shift_over_the_shared_one(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(
        _tiny_coupled(),
        n_points=0,
        shift=CoupledShiftSettings(basis=_BASIS),
        point_setup=lambda companion, state, point: {
            "shift": CoupledShiftSettings(turbulence_damping=2.0)
        },
    )
    (call,) = calls
    assert call["kwargs"]["shift"].basis is _BASIS
    assert call["kwargs"]["shift"].turbulence_damping == 2.0
