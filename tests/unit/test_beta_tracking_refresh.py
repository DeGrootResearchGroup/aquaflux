"""``BetaTrackingRefresh``: when it re-fits a materialized preconditioner, at which state, and at which shift.

The hook needs nothing from a residual but its Jacobian-vector product, so every test here runs on a
one-line scalar stand-in and a preconditioner that records what it was asked to build and builds nothing.
What is under test is only the decision -- when to rebuild, of what, at what shift -- not the inverse.
"""

from __future__ import annotations

from types import SimpleNamespace

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import (
    JacobianProbe,
    RefreshTiming,
    ShiftTerm,
)
from aquaflux.solve.materialized_session import (
    BetaTrackingRefresh,
)


class _ScalarResidual(eqx.Module):
    """``R(u) = gain u² / 2``, whose Jacobian ``gain u`` shows both the assembler and the state it is at."""

    gain: jnp.ndarray

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.gain * state**2


class _RecordingPreconditioner:
    """A frozen inverse that records what each refresh was asked to build, and builds nothing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def refresh_in_place(self, matvec, plan, shift_diagonal, **_kwargs):
        self.calls.append({"matvec": matvec, "shift": np.asarray(shift_diagonal)})
        return ()


def _stub_step(preconditioner, beta, diagonal):
    """The smallest Newton step the hook reads: a shift strength, a policy and its diagonal."""
    # Accepts the optional residual even though the hook does not pass one: a stand-in that is narrower
    # than the protocol breaks silently the day a caller starts supplying it.
    base = SimpleNamespace(
        shift_term=lambda _phi, _residual=None: ShiftTerm(diagonal, lambda _relaxation: None)
    )
    return SimpleNamespace(
        relaxation_schedule=SimpleNamespace(beta=beta),
        shift_policy=SimpleNamespace(preconditioner=preconditioner, base=base),
    )


def _hook(gain: float = 3.0, **kwargs) -> BetaTrackingRefresh:
    # The real probe (its plan and gather map are unused here), not a lookalike: the hook asks it which
    # assembler to differentiate, which only the class itself can answer.
    probe = JacobianProbe(plan=object(), structure=object())
    return BetaTrackingRefresh(_ScalarResidual(gain=jnp.asarray(gain)), probe, **kwargs)


_STATE = jnp.linspace(1.0, 2.0, 5)
_DIAGONAL = jnp.full(5, 2.0)
_TANGENT = jnp.ones(5)


def test_rebinding_the_refresh_swaps_the_case_and_forces_a_full_rebuild() -> None:
    """One refresh hook can serve a whole Reynolds ramp, which is what lets the ramp share one V-cycle.

    Nothing else in the hook watches for a rung boundary, so a hook that had stopped rebuilding would
    leave the next rung solving against a V-cycle fitted to the previous rung's viscosity. ``rebind``
    therefore does two things, and both are asserted: the Jacobian probe starts reporting the NEW
    companion's derivative, and the next refresh is a full re-materialize.
    """
    pc = _RecordingPreconditioner()
    step = _stub_step(pc, beta=0.5, diagonal=_DIAGONAL)
    # A full rebuild on the first call and after a rebind, none otherwise -- between those only the
    # dual-time loop's cost trigger rebuilds the V-cycle.
    refresh = _hook(gain=3.0)

    refresh(step, _STATE)  # the initializing call
    assert len(pc.calls) == 1
    assert np.allclose(pc.calls[0]["shift"], 0.5 * np.asarray(_DIAGONAL))
    assert np.allclose(pc.calls[0]["matvec"](_TANGENT), 3.0 * _STATE)

    refresh(step, _STATE)  # no rebuild between rebinds, as for the rest of a rung
    assert len(pc.calls) == 1

    companion = _ScalarResidual(gain=jnp.asarray(7.0))
    refresh.rebind(companion)
    assert refresh.assembler is companion
    refresh(step, _STATE)
    assert len(pc.calls) == 2  # forced by the rebind
    assert np.allclose(pc.calls[1]["matvec"](_TANGENT), 7.0 * _STATE)  # ...at the new companion

    refresh(step, _STATE)  # and the force is spent: one rebuild per rebind, not a stuck flag
    assert len(pc.calls) == 2


def test_the_inner_refresh_rebuilds_at_the_iterate_it_is_given_for_the_current_step() -> None:
    """``refresh_at`` re-materializes at the MID-STEP iterate, for the step the march last handed over.

    Two ways to get it wrong, both asserted: rebuilding at the state the step started from (the
    Jacobian would be the stale one the refresh exists to replace), and refreshing a step other than
    the current one -- the march replaces the step every iteration with one carrying a new ``β``.
    """
    pc = _RecordingPreconditioner()
    refresh = _hook(gain=3.0)

    refresh.refresh_at(_STATE)  # before any step: nothing to refresh, and nothing to fail on
    assert pc.calls == []

    refresh(_stub_step(pc, beta=0.5, diagonal=_DIAGONAL), _STATE)
    current = _stub_step(pc, beta=0.25, diagonal=_DIAGONAL)
    refresh(current, _STATE)  # no full rebuild after the first, but it is now the current step
    assert len(pc.calls) == 1

    iterate = 2.0 * _STATE
    refresh.refresh_at(iterate)
    assert len(pc.calls) == 2
    assert np.allclose(pc.calls[1]["matvec"](_TANGENT), 3.0 * iterate)
    assert np.allclose(pc.calls[1]["shift"], 0.25 * np.asarray(_DIAGONAL))


def test_the_refit_floor_bounds_the_preconditioners_shift_on_both_refreshes() -> None:
    """``refit_beta_floor`` floors the shift the inverse is fitted at, per-step and mid-step alike.

    A floor applied on only one of the two paths would leave the other fitting a V-cycle at a shift it
    inverts badly, which is the regime the floor exists to keep it out of.
    """
    pc = _RecordingPreconditioner()
    refresh = _hook(refit_beta_floor=0.05)

    refresh(_stub_step(pc, beta=0.01, diagonal=_DIAGONAL), _STATE)  # below the floor: floored
    refresh.refresh_at(_STATE)
    refresh(_stub_step(pc, beta=0.5, diagonal=_DIAGONAL), _STATE)  # no rebuild, but now current
    refresh.refresh_at(_STATE)  # above the floor: untouched
    assert [float(call["shift"][0]) for call in pc.calls] == [0.1, 0.1, 1.0]


def test_the_observer_is_told_which_branch_each_refresh_took() -> None:
    """A march log reads ``full`` / ``none`` / ``inner`` from here, so each must name its own branch."""
    timings: list[RefreshTiming] = []
    pc = _RecordingPreconditioner()
    refresh = _hook(observer=timings.append)

    step = _stub_step(pc, beta=0.5, diagonal=_DIAGONAL)
    refresh(step, _STATE)
    refresh(step, _STATE)
    refresh.refresh_at(_STATE)
    assert [timing.kind for timing in timings] == ["full", "none", "inner"]


def test_a_schedule_with_no_readable_shift_is_refused() -> None:
    """The hook fits at the step's ``β``, so a schedule that does not expose one as a constant is refused
    by name rather than fitted at some other shift."""
    refresh = _hook()
    step = _stub_step(_RecordingPreconditioner(), beta=0.5, diagonal=_DIAGONAL)
    step.relaxation_schedule = SimpleNamespace()
    with pytest.raises(ValueError, match="readable constant"):
        refresh(step, _STATE)
