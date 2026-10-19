"""Moving near-zero diagonal pivots out of a materialized preconditioner's fit.

What is pinned: which pivots ``PivotFloor`` replaces and with what (its own row's base pseudo-time
diagonal, sign kept, healthy pivots untouched); that the refresh hook measures pivots on everything the fit
would otherwise see and against the base diagonal rather than the shift; that it composes after a
stand-in; that a session changes only the rows its problem names and leaves the step's operator alone; and
that the setting is refused where it could act on nothing and survives a case file.
"""

from __future__ import annotations

from types import SimpleNamespace

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.precondition import (
    DiagonalStandIn,
    FieldSplit,
    JacobianProbe,
    JacobiSmoothed,
    MaterializedJacobian,
    MaterializedProblem,
    MaterializedSession,
    PivotFloor,
    SimpleSmoothed,
    materialized_spec_from_mapping,
    materialized_spec_to_mapping,
)
from aquaflux.precondition.materialized_preconditioner import MaterializedJacobianPreconditioner
from aquaflux.precondition.materialized_session import BetaTrackingRefresh
from aquaflux.solve import CellFields, ConstantRelaxation, DualTimeLoop, FieldLayout, ShiftTerm
from aquaflux.turbulence import open_session
from aquaflux.turbulence.coupled import _CoupledProblem

from tests.unit.test_coupled_rans import _cavity, _healthy_state

_SPLIT = FieldSplit(SimpleSmoothed(), JacobiSmoothed())


# --- the replacement rule ---------------------------------------------------------------------------


def test_a_near_zero_pivot_takes_its_rows_scale_and_keeps_its_sign() -> None:
    """Below the band a pivot becomes ``sign(p) d``; a pivot of exactly zero takes ``+d``.

    Catches a dropped sign (every replaced pivot positive), a replacement by the threshold's edge rather
    than by the row's scale, and a zero pivot left at zero (a singular fit).
    """
    floor = PivotFloor(threshold=0.15)
    pivot = np.array([1.0e-6, -1.0e-6, 0.0])
    scale = np.array([2.0e-5, 4.0e-5, 3.0e-5])
    np.testing.assert_allclose(pivot + floor.correction(pivot, scale), [2.0e-5, -4.0e-5, 3.0e-5])


def test_only_pivots_inside_the_band_relative_to_their_own_row_are_replaced() -> None:
    """The band is ``threshold`` times each row's own scale, and healthy pivots of either sign stay.

    Rows 0 and 1 sit just inside and just outside ``0.15 d`` on the same scale (an absolute band, or one
    measured on the wrong row, misplaces one of them); rows 2 and 3 have the same pivot on different
    scales, so only the larger scale replaces it; rows 4 and 5 are ordinary pivots, the second negative,
    which a rule that floored every negative pivot would change.
    """
    floor = PivotFloor(threshold=0.15)
    pivot = np.array([2.9e-6, 3.1e-6, 5.0e-6, 5.0e-6, 5.0e-5, -7.0e-5])
    scale = np.array([2.0e-5, 2.0e-5, 2.0e-5, 4.0e-5, 2.0e-5, 2.0e-5])
    changed = np.flatnonzero(floor.correction(pivot, scale))
    np.testing.assert_array_equal(changed, [0, 3])


def test_rows_without_a_shift_or_outside_the_mask_are_never_changed() -> None:
    """A row with no pseudo-time diagonal (a pressure row) has no scale to measure against, and a row the
    caller's mask excludes is left whatever its pivot."""
    floor = PivotFloor(threshold=0.5)
    pivot = np.array([0.0, 0.0, 1.0e-9, 1.0e-9])
    scale = np.array([0.0, -1.0, 1.0, 1.0])
    rows = np.array([True, True, False, True])
    np.testing.assert_array_equal(
        floor.correction(pivot, scale, rows=rows), [0.0, 0.0, 0.0, 1.0 - 1.0e-9]
    )


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_a_threshold_outside_the_open_unit_interval_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        PivotFloor(threshold=threshold)


# --- the refresh hook ---------------------------------------------------------------------------------


class _LinearResidual(eqx.Module):
    """``R(u) = a * u``: its Jacobian's diagonal is ``a``, so a test chooses the pivots outright."""

    a: jnp.ndarray

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.a * state


class _FittingPreconditioner:
    """Records the diagonal a refresh would fit: the Jacobian's, the shift, and any correction."""

    def __init__(self, jacobian_diagonal: np.ndarray) -> None:
        self.jacobian_diagonal = np.asarray(jacobian_diagonal)
        self.fitted: np.ndarray | None = None
        self.asked_for_correction = False

    def refresh_in_place(
        self, matvec, plan, shift_diagonal, *, diagonal_correction=None, **_kwargs
    ):
        self.asked_for_correction = diagonal_correction is not None
        correction = (
            None if diagonal_correction is None else diagonal_correction(self.jacobian_diagonal)
        )
        self.fitted = self.jacobian_diagonal + np.asarray(shift_diagonal)
        if correction is not None:
            self.fitted = self.fitted + correction
        return ()


_D = 2.0  # every row's base pseudo-time diagonal
_FACTOR = 3.0  # the policy's per-row multiplier on beta, so shift / beta = 6 is NOT the scale
_BETA = 0.5
_SHIFT = _BETA * _FACTOR * _D  # 3.0
_PIVOTS = np.array([0.1, -0.1, 3.0, -0.25, 0.2])  # J + shift, as the fit would otherwise see it


def _step():
    diagonal = jnp.full(5, _D)
    term = ShiftTerm(
        diagonal, lambda _beta: None, row_shift_scale=lambda _beta: jnp.full(5, _FACTOR)
    )
    base = SimpleNamespace(shift_term=lambda _phi, _residual=None: term)
    return SimpleNamespace(
        relaxation_schedule=SimpleNamespace(beta=_BETA), shift_policy=SimpleNamespace(base=base)
    )


def _hook(preconditioner, **kwargs) -> BetaTrackingRefresh:
    probe = JacobianProbe(plan=object(), structure=object())
    return BetaTrackingRefresh(
        _LinearResidual(a=jnp.asarray(_PIVOTS - _SHIFT)), probe, preconditioner, **kwargs
    )


def test_the_hook_fits_near_zero_pivots_at_their_rows_base_diagonal() -> None:
    """Measured against ``d`` (2.0), rows 0, 1 and 3 are inside ``0.15 d`` and go to ``±d``; row 4 is
    inside too but outside the mask. Measured against ``shift / beta`` (6.0) instead, row 2 would join
    and every replacement would be ``±6``."""
    pc = _FittingPreconditioner(_PIVOTS - _SHIFT)
    rows = np.array([True, True, True, True, False])
    hook = _hook(pc, pivot_floor=PivotFloor(0.15), floorable_rows=rows)
    hook(_step(), jnp.ones(5))
    np.testing.assert_allclose(pc.fitted, [_D, -_D, 3.0, -_D, 0.2])
    np.testing.assert_array_equal(hook.floored_rows, [0, 1, 3])


def test_the_floor_is_measured_after_the_stand_in() -> None:
    """A stand-in that already lifts row 0 well clear of zero leaves nothing for the floor to do there.

    Measured on the Jacobian and shift alone, row 0 (0.1) would be floored on top of the stand-in's 5.0
    and fitted at 7.0 instead of 5.1.
    """
    pc = _FittingPreconditioner(_PIVOTS - _SHIFT)
    lift = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    hook = _hook(
        pc,
        pivot_floor=PivotFloor(0.15),
        stand_in=lambda _state, _jd, _shift: DiagonalStandIn(residual=None, diagonal=lift),
    )
    hook(_step(), jnp.ones(5))
    np.testing.assert_allclose(pc.fitted, [5.1, -_D, 3.0, -_D, _D])
    np.testing.assert_array_equal(hook.standing_stand_in.diagonal, lift)


def test_without_a_floor_the_hook_asks_for_no_correction() -> None:
    pc = _FittingPreconditioner(_PIVOTS - _SHIFT)
    hook = _hook(pc)
    hook(_step(), jnp.ones(5))
    assert not pc.asked_for_correction
    np.testing.assert_allclose(pc.fitted, _PIVOTS)
    assert hook.floored_rows.size == 0


# --- the session --------------------------------------------------------------------------------------


class _NamingProblem(MaterializedProblem):
    """Just enough of a problem to be opened: a layout and the fields it names for the floor."""

    def __init__(self, floorable_fields: tuple[str, ...]) -> None:
        self.floorable_fields = floorable_fields
        self._layout = FieldLayout(3, (CellFields("a", 1), CellFields("k", 1)))

    assembler = property(lambda self: None)
    layout = property(lambda self: self._layout)

    def with_assembler(self, assembler):
        return self

    def probe(self, settings, active_rows):
        raise NotImplementedError

    def groups(self):
        return None

    def bind_march(self, march):
        return march

    def shift_source(self, state, march):
        raise NotImplementedError

    def build_step(self, state, policy, march, base_regime):
        raise NotImplementedError


def test_a_floor_is_refused_for_a_problem_that_names_no_field() -> None:
    spec = MaterializedJacobian(SimpleSmoothed(), pivot_floor=PivotFloor())
    with pytest.raises(ValueError, match="names none"):
        MaterializedSession(spec, _NamingProblem(()))
    MaterializedSession(MaterializedJacobian(SimpleSmoothed()), _NamingProblem(()))


def test_the_floor_may_change_exactly_the_named_fields_rows() -> None:
    spec = MaterializedJacobian(SimpleSmoothed(), pivot_floor=PivotFloor())
    session = MaterializedSession(spec, _NamingProblem(("k",)))
    np.testing.assert_array_equal(
        session._floorable_row_mask(), [False, False, False, True, True, True]
    )


def test_the_coupled_rans_problem_names_k_alone() -> None:
    """The k row is the one its production can take through zero; a momentum row's pivot is small beside
    its pseudo-time diagonal without being near-singular, and floored would be changed for nothing."""
    assert _CoupledProblem.floorable_fields == ("k",)


@pytest.fixture(scope="module")
def case():
    mesh, coupled = _cavity()
    return mesh, coupled, _healthy_state(mesh, coupled)


def test_a_coupled_refit_floors_only_k_rows_and_leaves_the_operator_exact(
    case, monkeypatch
) -> None:
    """On the real coupled problem: every floored row is a k row, the fit sees exactly
    ``sign(p) d - p`` added there and nothing elsewhere, and the step keeps differentiating the residual."""
    _, coupled, state = case
    fitted: list[np.ndarray] = []
    original = MaterializedJacobianPreconditioner._shifted

    def recording(jacobian, shift_diagonal):
        fitted.append((np.asarray(jacobian.diagonal()), np.asarray(shift_diagonal)))
        return original(jacobian, shift_diagonal)

    monkeypatch.setattr(MaterializedJacobianPreconditioner, "_shifted", staticmethod(recording))
    # A wide band at a small shift, so the healthy cavity has k pivots inside it.
    session = open_session(MaterializedJacobian(_SPLIT, pivot_floor=PivotFloor(0.9)), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    beta = 1e-3
    active = eqx.tree_at(
        lambda s: s.relaxation_schedule, step, ConstantRelaxation(jnp.asarray(beta))
    )
    fitted.clear()
    update = session.refresh_preconditioner(active, state)
    rows = session._hook.floored_rows
    k = coupled.layout.slice_of("k")
    assert rows.size > 0
    assert np.all((rows >= k.start) & (rows < k.stop))
    jacobian_diagonal, fit_diagonal = fitted[-1]
    term = step.shift_policy.base.shift_term(state)
    shift, scale = np.asarray(term.shift(beta)), np.asarray(term.diagonal)
    pivot = jacobian_diagonal + shift
    expected = shift.copy()
    expected[rows] = np.where(pivot[rows] < 0, -1.0, 1.0) * scale[rows] - jacobian_diagonal[rows]
    np.testing.assert_allclose(fit_diagonal, expected, rtol=1e-12, atol=1e-14)
    assert update is None or update(active).jacobian_residual is None


def test_the_setting_survives_a_case_file_and_is_checked_there() -> None:
    spec = MaterializedJacobian(_SPLIT, pivot_floor=PivotFloor(0.2))
    assert materialized_spec_from_mapping(materialized_spec_to_mapping(spec)) == spec
    mapping = materialized_spec_to_mapping(spec)
    mapping["pivot_floor"] = {"kind": "PivotFloor", "threshold": 1.2}
    with pytest.raises(ValueError, match="pivot_floor"):
        materialized_spec_from_mapping(mapping)
