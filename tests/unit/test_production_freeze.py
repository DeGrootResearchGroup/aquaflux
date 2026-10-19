"""Freezing the k-production's eddy viscosity per cell, chosen at each re-fit of a materialized Jacobian.

What is pinned: the stand-in changes the Jacobian on the frozen k rows' diagonal and nowhere else, by
exactly ``production_diagonal_gain``, and leaves the residual's value untouched; the criterion freezes the
rows production has all but cancelled and only those; a session asks for the stand-in at each re-fit,
fits its inverse to the operator the stand-in implies, and installs the stand-in in the step; and the
setting reaches the session from every place a caller may give it, and is refused where it cannot act.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.case import CoupledMarch
from aquaflux.precondition import FieldSplit, JacobiSmoothed, MaterializedJacobian, SimpleSmoothed
from aquaflux.precondition.materialized_preconditioner import MaterializedJacobianPreconditioner
from aquaflux.precondition.materialized_session import frozen_shift_diagonal
from aquaflux.solve import ConstantRelaxation, DualTimeLoop
from aquaflux.turbulence import BlockDiagonal, open_session
from aquaflux.turbulence import coupled as coupled_module
from aquaflux.turbulence import reynolds as reynolds_module
from aquaflux.turbulence.production_freeze import (
    frozen_production_viscosity,
    production_diagonal_gain,
    production_freeze_cells,
    with_frozen_production_cells,
)

from tests.unit.test_coupled_rans import _cavity, _healthy_state

_SPLIT = FieldSplit(SimpleSmoothed(), JacobiSmoothed())


@pytest.fixture(scope="module")
def case():
    mesh, coupled = _cavity()
    return mesh, coupled, _healthy_state(mesh, coupled)


def _k_block(coupled):
    k = coupled.layout.slice_of("k")
    return k.start, k.stop


def test_the_stand_in_changes_only_the_frozen_k_rows_diagonal_by_the_gain(case) -> None:
    """``J_standin v - J v`` is ``cells * gain * v`` on the k rows and zero on every other row.

    Two random directions and a mask that freezes every third cell, so a gain applied to the wrong cells,
    a change leaking off the diagonal (it would mix other entries of ``v`` in), or a gain of the wrong
    sign or size each shows.
    """
    mesh, coupled, state = case
    start, stop = _k_block(coupled)
    cells = np.zeros(mesh.n_cells)
    cells[::3] = 1.0
    gain = np.asarray(production_diagonal_gain(coupled, state))
    stand_in = with_frozen_production_cells(coupled, cells)
    for seed in (1, 2):
        v = jax.random.normal(jax.random.PRNGKey(seed), state.shape)
        difference = np.asarray(
            jax.jvp(stand_in.residual, (state,), (v,))[1]
            - jax.jvp(coupled.residual, (state,), (v,))[1]
        )
        expected = np.zeros(state.shape)
        expected[start:stop] = cells * gain * np.asarray(v[start:stop])
        assert np.abs(expected).max() > 1e-3
        np.testing.assert_allclose(difference, expected, rtol=0, atol=1e-12)


def test_the_stand_in_leaves_the_residual_bit_identical(case) -> None:
    mesh, coupled, state = case
    stand_in = with_frozen_production_cells(coupled, np.ones(mesh.n_cells))
    assert jnp.array_equal(stand_in.residual(state), coupled.residual(state))


def test_the_gain_is_the_frozen_jacobians_k_diagonal_less_the_exact_one(case) -> None:
    """Against the diagonal read one unit tangent at a time, so the all-at-once JVP's locality is tested."""
    mesh, coupled, state = case
    start, _ = _k_block(coupled)
    frozen = frozen_production_viscosity(coupled)
    gain = np.asarray(production_diagonal_gain(coupled, state))
    for cell in (0, mesh.n_cells // 2, mesh.n_cells - 1):
        e = jnp.zeros_like(state).at[start + cell].set(1.0)
        exact = jax.jvp(coupled.residual, (state,), (e,))[1][start + cell]
        freeze = jax.jvp(frozen.residual, (state,), (e,))[1][start + cell]
        assert gain[cell] == pytest.approx(float(freeze - exact), rel=1e-10, abs=1e-15)
    assert (gain > 0).all()


def test_the_criterion_freezes_only_rows_production_has_all_but_cancelled() -> None:
    k_diagonal = np.array([-2.0, 0.05, 1.0, 1.0, -1.0, 0.23])
    gain = np.array([3.0, 1.0, 1.0, 0.0, -1.0, 2.0])
    shift = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Row 0: negative, freezing makes it 1.0 -- frozen. Row 1: 0.05 < 0.1 * 1.05 -- frozen.
    # Row 2: 1.0 >= 0.1 * 2.0 -- kept. Rows 3, 4: freezing would not raise the diagonal -- kept.
    # Row 5: 0.23 < 0.1 * 2.23 is false (0.223) -- kept, just above the threshold's boundary.
    frozen = production_freeze_cells(k_diagonal, gain, shift, 0.1)
    assert frozen.tolist() == [True, True, False, False, False, False]
    # The shift counts on both sides: a large one lifts row 1 clear.
    assert not production_freeze_cells(k_diagonal, gain, shift + 10.0, 0.1)[1]
    # Threshold zero freezes only a row already negative.
    assert production_freeze_cells(k_diagonal, gain, shift, 0.0).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_a_session_fits_and_installs_the_stand_in_at_each_refit(case, monkeypatch) -> None:
    """At a re-fit the inverse is fitted to Jacobian + shift + the stand-in's diagonal, and the step the
    march is handed differentiates the stand-in."""
    _, coupled, state = case
    fitted: list[np.ndarray] = []
    original = MaterializedJacobianPreconditioner._shifted

    def recording(jacobian, shift_diagonal):
        fitted.append(np.asarray(shift_diagonal))
        return original(jacobian, shift_diagonal)

    monkeypatch.setattr(MaterializedJacobianPreconditioner, "_shifted", staticmethod(recording))
    session = open_session(MaterializedJacobian(_SPLIT), coupled, production_freeze_threshold=0.9)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    assert step.jacobian_residual is None
    beta = 1e-3
    active = eqx.tree_at(
        lambda s: s.relaxation_schedule, step, ConstantRelaxation(jnp.asarray(beta))
    )
    fitted.clear()
    updated = session.refresh_preconditioner(active, state)(active)

    stand_in = session._hook.standing_stand_in
    start, stop = _k_block(coupled)
    assert np.count_nonzero(stand_in.diagonal[start:stop]) > 0
    assert np.count_nonzero(stand_in.diagonal[:start]) == 0
    shift = np.asarray(frozen_shift_diagonal(step.shift_policy.base, beta, state))
    np.testing.assert_allclose(fitted[-1], shift + stand_in.diagonal, rtol=0, atol=1e-14)
    # The reported diagonal is the stand-in's own Jacobian change, on a frozen row and an unfrozen one.
    frozen_rows = np.flatnonzero(stand_in.diagonal)
    unfrozen_rows = np.setdiff1d(np.arange(start, stop), frozen_rows)
    for row in (frozen_rows[0], unfrozen_rows[0]):
        e = jnp.zeros_like(state).at[row].set(1.0)
        change = (
            jax.jvp(stand_in.residual, (state,), (e,))[1]
            - jax.jvp(coupled.residual, (state,), (e,))[1]
        )
        assert float(change[row]) == pytest.approx(stand_in.diagonal[row], rel=1e-10, abs=1e-15)
    assert updated.jacobian_residual is not None
    v = jax.random.normal(jax.random.PRNGKey(3), state.shape)
    assert jnp.allclose(
        jax.jvp(updated.jacobian_residual, (state,), (v,))[1],
        jax.jvp(stand_in.residual, (state,), (v,))[1],
    )


def test_without_a_threshold_a_refit_fits_the_plain_shifted_jacobian(case, monkeypatch) -> None:
    _, coupled, state = case
    fitted: list[np.ndarray] = []
    original = MaterializedJacobianPreconditioner._shifted

    def recording(jacobian, shift_diagonal):
        fitted.append(np.asarray(shift_diagonal))
        return original(jacobian, shift_diagonal)

    monkeypatch.setattr(MaterializedJacobianPreconditioner, "_shifted", staticmethod(recording))
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    active = eqx.tree_at(
        lambda s: s.relaxation_schedule, step, ConstantRelaxation(jnp.asarray(1e-3))
    )
    fitted.clear()
    update = session.refresh_preconditioner(active, state)
    shift = np.asarray(frozen_shift_diagonal(step.shift_policy.base, 1e-3, state))
    np.testing.assert_array_equal(fitted[-1], shift)
    assert session._hook.standing_stand_in is None
    assert update is None or update(active).jacobian_residual is None


def test_the_setting_is_refused_where_it_cannot_act(case) -> None:
    _, coupled, _ = case
    with pytest.raises(TypeError, match="materializes none"):
        open_session(BlockDiagonal(), coupled, production_freeze_threshold=0.1)
    with pytest.raises(ValueError, match="already freezes it in every cell"):
        open_session(
            MaterializedJacobian(_SPLIT),
            coupled,
            production_freeze_threshold=0.1,
            jacobian_production_viscosity=True,
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        open_session(MaterializedJacobian(_SPLIT), coupled, production_freeze_threshold=1.0)
    with pytest.raises(ValueError, match="set one"):
        CoupledMarch(production_freeze_threshold=0.1)


def test_the_setting_reaches_the_session_from_solve_coupled_and_the_reynolds_drivers(
    case, monkeypatch
) -> None:
    _, coupled, _ = case
    opened: list[dict] = []
    real = coupled_module.open_session

    def recording(preconditioner, assembler, **settings):
        opened.append(settings)
        return real(preconditioner, assembler, **settings)

    monkeypatch.setattr(coupled_module, "open_session", recording)
    monkeypatch.setattr(reynolds_module, "open_session", recording)
    coupled_module._continuation_source(
        coupled,
        None,
        coupled_module.RefreshPolicy(),
        MaterializedJacobian(_SPLIT),
        None,
        {"production_freeze_threshold": 0.2},
    )
    assert opened[-1] == {"production_freeze_threshold": 0.2}
    reynolds_module._shared_session(
        {"preconditioner": MaterializedJacobian(_SPLIT), "production_freeze_threshold": 0.3},
        coupled,
    )
    assert opened[-1] == {"production_freeze_threshold": 0.3}
    kept = reynolds_module._with_session(
        {"preconditioner": MaterializedJacobian(_SPLIT), "production_freeze_threshold": 0.3},
        object(),
    )
    assert "production_freeze_threshold" not in kept


def test_the_case_file_setting_reaches_the_session(case, monkeypatch) -> None:
    _, coupled, _ = case
    opened: list[dict] = []

    def recording(preconditioner, assembler, **settings):
        opened.append(settings)
        raise _Stop

    class _Stop(Exception):
        pass

    import aquaflux.case.solver as case_solver

    monkeypatch.setattr(case_solver, "open_session", recording)
    march = CoupledMarch(
        preconditioner=MaterializedJacobian(_SPLIT), production_freeze_threshold=0.15
    )
    assert "production_freeze_threshold" in march._owned()
    with pytest.raises(_Stop):
        march.solve(coupled)
    assert opened == [{"production_freeze_threshold": 0.15}]
