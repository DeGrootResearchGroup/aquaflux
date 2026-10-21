"""The coloured probe's one-program-per-column-field Jacobian products.

What is pinned, mesh-free on a three-field residual over a chain of cells: that the per-field product
returns the whole-state product's responses for the same seeds; that it prunes -- a field that does not
reach an operator puts no tangent through it, read off the jaxpr -- and that a residual which repacks its
fields into a flat state loses the pruning; that the probe's chunks never mix column fields and recover
the same matrix; and that a problem which does not offer the per-field residual is probed with
whole-state seeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.precondition import JacobianProbe, MaterializedJacobian, SimpleSmoothed
from aquaflux.precondition.materialized_session import (
    BetaTrackingRefresh,
    MaterializedProblem,
    MaterializedSession,
    _probe_products,
    batched_field_jacobian_matvec,
    batched_jacobian_matvec,
)
from aquaflux.precondition.sparse_jacobian import (
    FieldwiseMatvec,
    block_stencil_gather_map,
    column_probe_plan,
    materialize_block_jacobian,
)
from aquaflux.solve import ConstantRelaxation, FieldLayout, ShiftTerm

_N = 9
_LAYOUT = FieldLayout.cell_fields(_N, a=1, b=1, c=1)
_OWNER = np.arange(_N - 1)
_NEIGHBOUR = np.arange(1, _N)


def _shifted(x: jnp.ndarray) -> jnp.ndarray:
    """Each cell's left neighbour's value, zero at the first cell (reach 1, no wrap-around)."""
    return jnp.pad(x, (1, 0))[:-1]


def _difference(x: jnp.ndarray) -> jnp.ndarray:
    """A nearest-neighbour coupling on the chain (reach 1)."""
    padded = jnp.pad(x, 1)
    return 2.0 * x - padded[:-2] - padded[2:]


def _smoothed(x: jnp.ndarray) -> jnp.ndarray:
    """The operator only ``a`` reaches, a three-point convolution (reach 1): the residual's one
    ``conv_general_dilated``, so it is findable in a jaxpr."""
    return jnp.convolve(x, jnp.array([0.25, 0.5, 0.25]), mode="same")


class _ThreeFields(eqx.Module):
    """``R(a, b, c)``: ``a`` alone reaches the smoothing; ``b`` and ``c`` reach products and differences.

    ``route_c_through_the_smoothing`` adds ``c`` to the smoothing's argument, which is the mutation that must
    make the pruning assertion fail; ``repack`` evaluates the per-field residual by packing the fields
    into a flat state and calling :meth:`residual`, the trap the per-field entry point exists to avoid.
    """

    weight: jnp.ndarray
    route_c_through_the_smoothing: bool = eqx.field(static=True, default=False)
    repack: bool = eqx.field(static=True, default=False)

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.residual_of(*_LAYOUT.unpack(state))

    def residual_with_field(self, state, field, values):
        fields = list(_LAYOUT.unpack(state))
        fields[field] = values
        if self.repack:
            return self.residual(_LAYOUT.pack(*fields))
        return self.residual_of(*fields)

    def residual_of(self, a, b, c):
        smoothed = _smoothed(a + c if self.route_c_through_the_smoothing else a)
        return _LAYOUT.pack(
            _difference(a) + smoothed * b + self.weight * c**2,
            _difference(b) + jnp.sin(a) * c,
            _difference(c) + b * _shifted(c) + smoothed,
        )


def _state() -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, _LAYOUT.size))


def _assembler(**options) -> _ThreeFields:
    return _ThreeFields(weight=jnp.asarray(0.7), **options)


def _seeds(field: int, rows: int = 3) -> np.ndarray:
    return np.random.default_rng(field + 1).standard_normal((rows, _N))


def _embedded(field: int, seeds: np.ndarray) -> np.ndarray:
    whole = np.zeros((seeds.shape[0], _LAYOUT.size))
    whole[:, field * _N : (field + 1) * _N] = seeds
    return whole


@pytest.mark.parametrize("field", [0, 1, 2])
def test_the_per_field_product_returns_the_whole_state_products_columns(field: int) -> None:
    """The responses to seeds in one field, by the per-field program and by whole-state seeds, and the
    Jacobian's own column slice, agree; a product that replaced the wrong field, or dropped the others'
    constant values, would not."""
    assembler, state = _assembler(), _state()
    seeds = _seeds(field)
    per_field = np.asarray(
        batched_field_jacobian_matvec(assembler, _LAYOUT, state, field, jnp.asarray(seeds))
    )
    whole = np.asarray(
        batched_jacobian_matvec(assembler, state, jnp.asarray(_embedded(field, seeds)))
    )
    jacobian = np.asarray(jax.jacfwd(assembler.residual)(state))
    np.testing.assert_allclose(per_field, whole, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        per_field, seeds @ jacobian[:, field * _N : (field + 1) * _N].T, rtol=1e-12, atol=1e-12
    )


def _count(jaxpr, primitive: str) -> int:
    """Equations of ``primitive`` in ``jaxpr`` and every jaxpr nested in it (a jit's, a vmap's)."""
    total = 0
    for equation in jaxpr.eqns:
        total += equation.primitive.name == primitive
        for value in equation.params.values():
            for inner in value if isinstance(value, tuple | list) else (value,):
                if isinstance(inner, jax.extend.core.ClosedJaxpr):
                    total += _count(inner.jaxpr, primitive)
                elif isinstance(inner, jax.extend.core.Jaxpr):
                    total += _count(inner, primitive)
    return total


def _smoothings_in_the_product(assembler, field: int) -> int:
    """Convolutions in the per-field product's program: one primal, plus one if a tangent runs."""
    jaxpr = jax.make_jaxpr(
        lambda state, seeds: batched_field_jacobian_matvec(assembler, _LAYOUT, state, field, seeds)
    )(_state(), jnp.asarray(_seeds(field)))
    return _count(jaxpr.jaxpr, "conv_general_dilated")


def test_a_field_that_does_not_reach_an_operator_puts_no_tangent_through_it() -> None:
    """``b`` and ``c`` do not reach the smoothing, so their programs evaluate it once (the primal);
    ``a`` reaches it, so its program also differentiates it. Routing ``c`` through it -- the mutation
    -- makes ``c``'s program differentiate it too, which is what shows the count can fail."""
    pruned = _assembler()
    assert [_smoothings_in_the_product(pruned, field) for field in range(3)] == [2, 1, 1]
    routed = _assembler(route_c_through_the_smoothing=True)
    assert _smoothings_in_the_product(routed, 2) == 2


def test_a_residual_that_repacks_its_fields_loses_the_pruning() -> None:
    """Packing the fields back into a flat state hands every field an explicit zero tangent, so even
    ``b``'s program differentiates the smoothing: the reason the per-field entry point must reach the
    operators through the separate fields."""
    assert _smoothings_in_the_product(_assembler(repack=True), 1) == 2


def _plan():
    """Columns probed at different reaches, so the fields' colour counts differ and a batch of 3 leaves
    every field a short last chunk -- the shape a chunk spanning two fields would come from."""
    return column_probe_plan(_OWNER, _NEIGHBOUR, _N, column_reach=(1, 2, 2), pattern_reach=2)


def test_no_chunk_mixes_column_fields_and_the_matrix_is_the_whole_state_ones() -> None:
    assembler, state = _assembler(), _state()
    plan = _plan()
    assert len(set(plan.n_colours)) > 1  # the fixture's premise
    calls: list[tuple[int, tuple[int, ...]]] = []

    def apply(field, seeds):
        calls.append((field, seeds.shape))
        return batched_field_jacobian_matvec(assembler, _LAYOUT, state, field, seeds)

    structure = block_stencil_gather_map(plan)
    matvec, whole = _probe_products(assembler, state, None)
    per_field = materialize_block_jacobian(
        matvec, plan, batched_matvec=FieldwiseMatvec(apply), probe_batch_size=3, structure=structure
    )
    reference = materialize_block_jacobian(
        matvec, plan, batched_matvec=whole, probe_batch_size=3, structure=structure
    )
    expected = [
        (field, (3, _N))
        for field, count in enumerate(plan.n_colours)
        for _ in range(-(-count // 3))
    ]
    assert calls == expected
    np.testing.assert_array_equal(per_field.indices, reference.indices)
    np.testing.assert_allclose(per_field.toarray(), reference.toarray(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        per_field.toarray(), np.asarray(jax.jacfwd(assembler.residual)(state)), atol=1e-12
    )


def test_one_fields_seeds_refuse_a_range_spanning_two_fields() -> None:
    plan = _plan()
    boundary = plan.probe_base[1]
    with pytest.raises(ValueError, match="read column fields"):
        plan.field_seeds(boundary - 1, boundary + 1)
    np.testing.assert_array_equal(
        plan.field_seeds(boundary, boundary + 1)[0],
        plan.seed_block(boundary, boundary + 1)[0, _N : 2 * _N],
    )


# --- the session and its refresh hook --------------------------------------------------------------------


class _Probed(MaterializedProblem):
    """Just enough of a problem to say whether it is probed one field at a time."""

    def __init__(self, probes_by_field: bool) -> None:
        self.probes_by_field = probes_by_field

    assembler = property(lambda self: _assembler())
    layout = property(lambda self: _LAYOUT)

    def with_assembler(self, assembler):
        return self

    def probe(self, settings, groups):
        raise NotImplementedError

    def groups(self):
        return None

    def bind_march(self, march):
        return march

    def shift_source(self, state, march):
        raise NotImplementedError

    def build_step(self, state, policy, march, base_regime):
        raise NotImplementedError


class _Materializing:
    """A host preconditioner whose refresh materializes the Jacobian through the products it is given."""

    def __init__(self) -> None:
        self.product = None
        self.jacobian = None

    def refresh_in_place(self, matvec, plan, shift, *, batched_matvec, structure, **kwargs):
        self.product = batched_matvec
        self.jacobian = materialize_block_jacobian(
            matvec, plan, batched_matvec=batched_matvec, probe_batch_size=2, structure=structure
        )
        return ()


def _refit(by_field) -> _Materializing:
    plan = _plan()
    preconditioner = _Materializing()
    hook = BetaTrackingRefresh(
        _assembler(),
        JacobianProbe(plan=plan, structure=block_stencil_gather_map(plan)),
        preconditioner,
        by_field=by_field,
    )
    term = ShiftTerm(jnp.ones(_LAYOUT.size), lambda _beta: None)
    step = SimpleNamespace(
        relaxation_schedule=ConstantRelaxation(0.5),
        shift_policy=SimpleNamespace(base=SimpleNamespace(shift_term=lambda _phi, _r=None: term)),
    )
    hook(step, _state())
    return preconditioner


@pytest.mark.parametrize("probes_by_field", [False, True])
def test_a_session_probes_by_field_only_when_its_problem_offers_it(probes_by_field: bool) -> None:
    """The flag reaches the refresh hook's products: whole-state seeds without it, one program per
    field with it, and the same matrix either way."""
    session = MaterializedSession(MaterializedJacobian(SimpleSmoothed()), _Probed(probes_by_field))
    assert session._by_field == (_LAYOUT if probes_by_field else None)
    assert not MaterializedProblem.probes_by_field
    refit = _refit(session._by_field)
    assert isinstance(refit.product, FieldwiseMatvec) == probes_by_field
    np.testing.assert_allclose(
        refit.jacobian.toarray(),
        np.asarray(jax.jacfwd(_assembler().residual)(_state())),
        atol=1e-12,
    )


def test_the_coupled_and_flow_problems_are_probed_by_field() -> None:
    from aquaflux.flow.march import _FlowProblem
    from aquaflux.turbulence.coupled import _CoupledProblem

    assert _CoupledProblem.probes_by_field and _FlowProblem.probes_by_field


# --- the coupled and flow residuals' per-field entry points ---------------------------------------------


def _equations(jaxpr) -> int:
    """Every equation in ``jaxpr`` and in the jaxprs nested in it."""
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for value in equation.params.values():
            for inner in value if isinstance(value, tuple | list) else (value,):
                if isinstance(inner, jax.extend.core.ClosedJaxpr):
                    total += _equations(inner.jaxpr)
                elif isinstance(inner, jax.extend.core.Jaxpr):
                    total += _equations(inner)
    return total


def _log_omega_cavity():
    """The coupled cavity with ``omega`` solved as its logarithm, so the transforms are on the path."""
    import dataclasses

    from aquaflux.turbulence import LogScalars

    from tests.unit.test_coupled_rans import _cavity, _healthy_state

    mesh, coupled = _cavity(4)
    coupled = dataclasses.replace(coupled, omega_transform=LogScalars())
    physical = _healthy_state(mesh, coupled)
    return coupled, coupled.state_from_physical(*coupled.physical_fields(physical))


@pytest.mark.parametrize("which", ["coupled", "flow"])
def test_the_residuals_with_one_field_replaced_are_their_whole_state_residuals(which: str) -> None:
    """Replacing a field by its own values leaves each residual unchanged bit for bit, and each column
    field's product agrees with the whole-state product's columns: a field mapped to the wrong block,
    or a solved ``omega`` not taken back to the physical one, would not."""
    coupled, state = _log_omega_cavity()
    if which == "flow":
        assembler, layout = coupled.momentum, coupled.momentum.layout
        state = state[: layout.size]
    else:
        assembler, layout = coupled, coupled.layout
    rng = np.random.default_rng(5)
    fields = range(layout.n_fields)
    if which == "coupled":
        # Each field costs a compile. The velocity and pressure branches hand over to the momentum
        # block's own `unpack_with_field`, which the flow arm checks field by field; `k` and `omega`
        # (the latter through its solved-to-physical transform) are the coupled residual's own.
        fields = [layout.field_offset("k"), layout.field_offset("omega")]
    for field in fields:
        own = state[layout.field_dofs(field) : layout.field_dofs(field + 1)]
        np.testing.assert_array_equal(
            assembler.residual_with_field(state, field, own), assembler.residual(state)
        )
        seeds = rng.standard_normal((2, layout.n_cells))
        whole = np.zeros((2, layout.size))
        whole[:, layout.field_dofs(field) : layout.field_dofs(field + 1)] = seeds
        expected = np.asarray(batched_jacobian_matvec(assembler, state, jnp.asarray(whole)))
        np.testing.assert_allclose(
            batched_field_jacobian_matvec(assembler, layout, state, field, jnp.asarray(seeds)),
            expected,
            rtol=1e-12,
            atol=1e-12 * np.abs(expected).max(),
        )


def test_the_coupled_pressure_and_turbulence_columns_trace_less_than_the_whole_state() -> None:
    """A pressure, ``k`` or ``omega`` column leaves the velocity gradient and what is built on it out of
    the derivative, so each traces fewer equations than a whole-state product, and the pressure column
    fewer than a velocity column, which cannot shed it. Repacking the fields before assembling would
    leave every column at the whole state's count."""
    coupled, state = _log_omega_cavity()
    layout = coupled.layout

    def traced(product, *arguments) -> int:
        return _equations(jax.make_jaxpr(product)(*arguments).jaxpr)

    whole = traced(
        lambda s, t: batched_jacobian_matvec(coupled, s, t), state, jnp.ones((2, layout.size))
    )
    per_field = [
        traced(
            lambda s, t, f=field: batched_field_jacobian_matvec(coupled, layout, s, f, t),
            state,
            jnp.ones((2, layout.n_cells)),
        )
        for field in range(layout.n_fields)
    ]
    dim = coupled.momentum.mesh.dim
    assert all(count < whole for count in per_field[dim:]), (whole, per_field)
    assert per_field[dim] < min(per_field[:dim]), per_field
