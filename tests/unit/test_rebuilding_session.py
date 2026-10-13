"""A session with nothing to re-fit follows a change of problem by rebuilding the step it hands the march.

Mesh-free: the steps are real shifted steps over a scalar residual, and the session's build is a stub
whose only job is to say which problem a step was built on. What is pinned is what the march relies on:
the hook hands back nothing until the session is re-pointed, then exactly one update that carries the
new problem's shift and preconditioner while keeping the shift strength the march had set, and a step
that would stop for a re-fit is refused because nothing would take it.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    ConstantRelaxation,
    DualTimeStep,
    PseudoTransientStep,
    RebuildingSession,
    ShiftTerm,
)


class _ScaledShift(eqx.Module):
    """A shift policy whose diagonal says which problem it was built on."""

    scale: jnp.ndarray

    def shift_term(self, phi, residual=None):
        del residual
        diagonal = self.scale * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda beta: lambda r: r / (1.0 + beta * diagonal))


class _ScaledFactory(eqx.Module):
    """An adjoint preconditioner factory that, too, says which problem it was built on."""

    scale: jnp.ndarray

    def __call__(self, state):
        del state
        return lambda r: r / self.scale


class _Session(RebuildingSession):
    """Builds a step whose policy, adjoint factory and operator stand-in all carry its assembler."""

    def __init__(self, assembler: float, *, step_class=PseudoTransientStep) -> None:
        super().__init__(assembler)
        self.built: list[tuple[float, float]] = []
        self.held: list[object] = []
        self._step_class = step_class

    def _build(self, state, march):
        self.built.append((self.assembler, float(state[0])))
        scale = jnp.asarray(self.assembler)
        return self._step_class(
            _ScaledShift(scale),
            adjoint_preconditioner_factory=_ScaledFactory(scale),
            jacobian_residual=_ScaledFactory(scale),
            **march,
        )

    def _repointed(self, state, held, march):
        self.held.append(held)
        return self._build(state, march)

    def refresh(self, state, previous, **march):
        return self._build(state, march)


def _held(step: PseudoTransientStep, beta: float = 0.25) -> PseudoTransientStep:
    """``step`` as a march holds it: with the shift strength its step control set."""
    return eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(jnp.asarray(beta)))


def test_a_session_never_re_pointed_hands_the_march_nothing() -> None:
    session = _Session(1.0)
    step = session.build(jnp.ones(3))
    assert session.refresh_preconditioner(step, jnp.ones(3)) is None
    assert session.refresh_preconditioner(step, jnp.ones(3), due=True) is None
    assert session.built == [(1.0, 1.0)]


def test_a_re_pointed_session_rebuilds_once_on_the_new_problem_at_the_marchs_state() -> None:
    """One rebuild per re-pointing, on the new assembler, at the state the march hands the hook."""
    session = _Session(1.0)
    step = _held(session.build(jnp.ones(3)))
    session.rebind(5.0)
    update = session.refresh_preconditioner(step, jnp.full(3, 2.0))
    assert update is not None
    assert session.built == [(1.0, 1.0), (5.0, 2.0)]
    # The rebuild is handed the step the march holds, to derive the new one from.
    assert session.held == [step]
    # A second call before the next re-pointing (a retry of the same step) rebuilds nothing.
    assert session.refresh_preconditioner(step, jnp.full(3, 3.0)) is None
    assert session.built == [(1.0, 1.0), (5.0, 2.0)]


def test_a_build_after_a_re_pointing_is_already_on_the_new_problem() -> None:
    """A ramp re-points its session at the anchor before the first build; that build needs no rebuild.

    The step it makes is built on the anchor, so the first refresh of the march has nothing to hand
    back. A session that kept the re-pointing pending across the build would rebuild the step it had
    just built, at the start of every ramp.
    """
    session = _Session(1.0)
    session.rebind(5.0)
    step = _held(session.build(jnp.ones(3)))
    assert session.refresh_preconditioner(step, jnp.ones(3)) is None
    assert session.built == [(5.0, 1.0)]


def test_the_update_carries_the_new_problems_preconditioning_and_keeps_the_marchs_shift() -> None:
    """Shift policy, adjoint factory and operator stand-in come from the rebuild; the shift strength stays.

    A rebuilt step comes with the default schedule, so handing it to the march whole would throw away
    the shift strength the step control had walked to -- the one thing the march held that the rebuild
    cannot know.
    """
    session = _Session(1.0)
    step = _held(session.build(jnp.ones(3), line_search=4), beta=0.25)
    session.rebind(5.0)
    updated = session.refresh_preconditioner(step, jnp.ones(3))(step)
    assert float(updated.shift_policy.scale) == 5.0
    assert float(updated.adjoint_preconditioner_factory.scale) == 5.0
    assert float(updated.jacobian_residual.scale) == 5.0
    assert float(updated.relaxation_schedule.beta) == 0.25
    assert updated.line_search == 4
    assert updated.residual_norm is step.residual_norm


def test_the_update_reaches_a_dual_time_step_too() -> None:
    session = _Session(1.0, step_class=DualTimeStep)
    step = _held(session.build(jnp.ones(3), inner_steps=3))
    session.rebind(7.0)
    updated = session.refresh_preconditioner(step, jnp.ones(3))(step)
    assert float(updated.shift_policy.scale) == 7.0
    assert updated.inner_steps == 3
    assert float(updated.relaxation_schedule.beta) == 0.25


def test_a_rebuild_after_a_build_is_made_with_the_builds_march_keywords() -> None:
    session = _Session(1.0)
    step = session.build(jnp.ones(3), line_search=6)
    session.rebind(2.0)
    rebuilt = []
    original = session._build

    def recording(state, march):
        rebuilt.append(dict(march))
        return original(state, march)

    session._build = recording
    session.refresh_preconditioner(step, jnp.ones(3))
    assert rebuilt == [{"line_search": 6}]


def test_a_step_that_would_stop_for_a_re_fit_is_refused_when_built() -> None:
    """Nothing here re-fits, so such a step would end every hard step early for a re-fit never taken."""
    session = _Session(1.0, step_class=DualTimeStep)
    with pytest.raises(ValueError, match="refresh_on_cycles=3"):
        session.build(jnp.ones(3), inner_steps=3, refresh_on_cycles=3)


def test_preconditioned_as_replaces_exactly_the_three_problem_built_fields() -> None:
    """The fields built from the problem come from ``other``; every setting of the step itself stays."""
    base = _held(
        PseudoTransientStep(
            _ScaledShift(jnp.asarray(1.0)),
            line_search=3,
            adjoint_preconditioner_factory=None,
            jacobian_residual=None,
        ),
        beta=0.5,
    )
    other = PseudoTransientStep(
        _ScaledShift(jnp.asarray(9.0)),
        adjoint_preconditioner_factory=_ScaledFactory(jnp.asarray(9.0)),
        jacobian_residual=_ScaledFactory(jnp.asarray(9.0)),
    )
    swapped = base.preconditioned_as(other)
    assert float(swapped.shift_policy.scale) == 9.0
    assert float(swapped.adjoint_preconditioner_factory.scale) == 9.0
    assert float(swapped.jacobian_residual.scale) == 9.0
    assert float(swapped.relaxation_schedule.beta) == 0.5
    assert swapped.line_search == 3
    # And back: a field the other step leaves unset is unset in the result.
    assert base.preconditioned_as(base).jacobian_residual is None
    assert other.preconditioned_as(base).adjoint_preconditioner_factory is None
