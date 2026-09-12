"""The pseudo-transient continuation engine drives a non-flow residual.

``PseudoTransientStep`` is the residual-agnostic pseudo-transient continuation strategy: the
switched-evolution-relaxation schedule, the shifted solve, and the accept/escalate loop, with the
problem-specific choices (which DOFs shift, the shift magnitude, the shifted preconditioner) supplied
by an injected ``ShiftPolicy``. These tests exercise it on a small scalar nonlinear root with a
trivial shift policy — no mesh, no flow assembler, no block preconditioner — proving the engine is
reusable beyond the coupled flow it was first built for.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.solve import (
    ConstantRelaxation,
    DivergenceGuard,
    DualTimeStep,
    ImplicitNewtonSolver,
    MonotoneLineSearch,
    PseudoTransientStep,
    RelaxedFarFromRoot,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
    forward_march,
    positive_block_limit,
    positive_block_projection,
)
from aquaflux.solve.continuation import _shifted_solve
from aquaflux.solve.implicit import backtracking_line_search


class UniformShiftPolicy(eqx.Module):
    """A minimal non-flow shift policy: a uniform pseudo-time shift on every DOF, unpreconditioned.

    Attributes
    ----------
    strength : float
        The per-DOF base shift magnitude ``d`` (static); the engine scales it by the relaxation ``β``.
    """

    strength: float = eqx.field(static=True, default=1.0)

    def shift_term(self, phi: jnp.ndarray, residual=None) -> ShiftTerm:
        diagonal = self.strength * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda relaxation: None)


def _residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A nonlinear residual with root ``phi = cbrt(theta)`` (per component)."""
    return phi**3 - theta


class _RecordingShiftPolicy(eqx.Module):
    """``UniformShiftPolicy`` that records the ``(phi, residual)`` pairs it is asked about."""

    seen: list = eqx.field(static=True)

    def shift_term(self, phi: jnp.ndarray, residual=None) -> ShiftTerm:
        self.seen.append((phi, residual))
        return ShiftTerm(jnp.ones_like(phi), lambda relaxation: None)


class _BlockDampedShiftPolicy(eqx.Module):
    """A uniform shift with the SECOND component damped ``ratio`` times harder.

    ``through_the_diagonal`` picks which route the factor takes: folded into the base diagonal, or
    carried as a ``row_relaxation`` closure the step applies at the ``beta`` it actually uses. At a
    fixed ``beta`` the two are the same number, which is what makes them comparable.
    """

    ratio: float = eqx.field(static=True, default=4.0)
    through_the_diagonal: bool = eqx.field(static=True, default=False)

    def shift_term(self, phi: jnp.ndarray, residual=None) -> ShiftTerm:
        scale = jnp.array([1.0, self.ratio])
        if self.through_the_diagonal:
            return ShiftTerm(scale * jnp.ones_like(phi), lambda relaxation: None)
        return ShiftTerm(jnp.ones_like(phi), lambda relaxation: None, lambda relaxation: scale)


def test_the_step_hands_the_policy_the_residual_it_just_computed() -> None:
    """Why the protocol grew an argument rather than a policy re-deriving ``R(phi)`` itself.

    The step evaluates the residual on the line before it asks for the shift, so a policy whose shift
    depends on how far the state is from a root reads it for free; deriving the same quantity a second
    time inside the policy would be one value computed in two places.
    """
    theta = jnp.array([8.0, 27.0])
    seen: list = []
    step = PseudoTransientStep(
        _RecordingShiftPolicy(seen=seen),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
    )
    phi0 = jnp.array([0.7, 1.3])

    def residual_theta(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    # One step, driven directly: a whole march runs inside a `while_loop`, where a recorded array is a
    # tracer that cannot be read afterwards.
    step.stepper()(
        residual_theta, phi0, jnp.linalg.norm(residual_theta(phi0)), step.default_solver()
    )

    assert seen, "the policy was never asked for a shift"
    assert all(residual is not None for _phi, residual in seen)
    for phi, residual in seen:
        assert jnp.allclose(residual, residual_theta(phi))


def test_a_row_relaxation_reaches_the_SHIFTED_OPERATOR_not_just_the_policy() -> None:
    """The plumbing that decides whether a per-block pseudo-timestep exists at all.

    A wrapper policy that rebuilds a ``ShiftTerm`` from ``.diagonal`` alone drops the multiplier in
    silence: the march still runs and the damping simply never happens. So the check is that the two
    routes to the SAME shift give the same march bit for bit, and that damping one block really does
    change the trajectory -- if it did not, the first assertion would hold for the wrong reason.
    """
    theta = jnp.array([8.0, 27.0])

    def march(policy):
        step = PseudoTransientStep(policy, relaxation_schedule=ConstantRelaxation(beta=2.0))
        solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)
        return solver.solve(_residual, jnp.ones_like(theta), theta)

    by_row = march(_BlockDampedShiftPolicy(ratio=4.0))
    by_diagonal = march(_BlockDampedShiftPolicy(ratio=4.0, through_the_diagonal=True))
    undamped = march(_BlockDampedShiftPolicy(ratio=1.0))

    assert jnp.array_equal(by_row, by_diagonal)
    assert jnp.allclose(by_row, jnp.cbrt(theta), atol=1e-6)
    # The shift vanishes at the root, so damping moves the path and not the answer -- which is why the
    # trajectories, not the roots, are what separates a live multiplier from a dropped one.
    assert not jnp.array_equal(by_row, undamped)


def test_pseudo_transient_engine_runs_without_flow() -> None:
    """The engine converges a nonlinear root using only an injected scalar shift policy."""
    theta = jnp.array([8.0, 27.0, 64.0])
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_pseudo_transient_engine_is_differentiable() -> None:
    """Reverse-mode gradient through the engine's converged solve matches the closed form."""
    theta = jnp.array([8.0])
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(solver.solve(_residual, jnp.ones_like(t), t))

    grad = jax.grad(solved_sum)(theta)

    # d/dtheta cbrt(theta) = (1/3) theta^(-2/3); the IFT adjoint is independent of the shift.
    assert jnp.allclose(grad, (1.0 / 3.0) * theta ** (-2.0 / 3.0), atol=1e-6)


#: How far the stand-in below scales :func:`_residual`'s Jacobian.
#:
#: ⚠️ **NOT 0.5, and the reason is worth knowing before choosing a factor for a fixture like this.** A
#: stand-in scaled by ``s`` makes the step ``1/s`` times the Newton step, so the iteration's fixed-point
#: derivative at the root is ``1 - 1/s`` -- which at ``s = 0.5`` is exactly ``-1``. **A
#: factor-of-two-wrong Jacobian sits precisely on Newton's stability boundary**: it oscillates without
#: decaying, forever, once the pseudo-transient shift has eased to zero near the root. That is the
#: least harmless-looking factor available, not the most, and it made this suite hang against
#: ``max_steps`` before it was chosen deliberately. At ``0.8`` the derivative is ``-0.25`` and the
#: iteration converges linearly -- still an unmistakably wrong operator, and one that works.
_STAND_IN_SLOPE = 0.8


def _scaled_slope_residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A stand-in whose Jacobian is :data:`_STAND_IN_SLOPE` times :func:`_residual`'s, with a different root.

    Deliberately wrong in *both* ways a stand-in can be wrong -- the operator it supplies is off by a
    fixed factor, and its own root (``cbrt(theta / 0.8)``) is nowhere near the real one -- so a step
    that accidentally drove this residual to zero, or that quietly kept differentiating the real one,
    is distinguishable from a step that uses it for the operator alone.
    """
    return _STAND_IN_SLOPE * phi**3 - theta


def _jacobian_narrowed_step(**kwargs: object) -> PseudoTransientStep:
    return PseudoTransientStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        **kwargs,
    )


def test_a_stand_in_jacobian_leaves_the_root_where_the_residual_puts_it() -> None:
    """``jacobian_residual`` changes the operator and not the answer.

    The whole asymmetry the field exists for: the residual decides *which* equations are solved, the
    operator decides only how fast they are solved. A stand-in whose own root is ``cbrt(2*theta)``
    must therefore not move the converged state off ``cbrt(theta)``.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    solver = ImplicitNewtonSolver(
        rtol=1e-10,
        atol=1e-10,
        max_steps=400,
        forward_step=_jacobian_narrowed_step(
            jacobian_residual=lambda p: _scaled_slope_residual(p, theta)
        ),
    )

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)
    # ...and emphatically not the stand-in's own root, which a step that solved the wrong residual
    # would have found instead.
    assert not jnp.allclose(phi, jnp.cbrt(theta / _STAND_IN_SLOPE), atol=1e-2)


def test_a_stand_in_jacobian_genuinely_changes_the_step_it_produces() -> None:
    """The reachability half: the field must have teeth, or the test above proves nothing.

    An unwired ``jacobian_residual`` would leave the correction bit-identical, and every property
    asserted of it would then hold for a reason that has nothing to do with the field. So assert the
    corrections differ by exactly the factor the stand-in's Jacobian differs by: with no shift and an
    exact solve, ``J delta = -R`` against ``(s J) delta = -R`` scales the step by ``1/s``.
    """
    phi = jnp.array([1.4, 1.1, 0.6])
    theta = jnp.array([8.0, 27.0, 64.0])
    residual = jax.tree_util.Partial(_residual, theta=theta)
    solver = lx.GMRES(rtol=1e-12, atol=1e-12)
    shift = jnp.zeros_like(phi)

    exact, _ = _shifted_solve(residual, phi, residual(phi), shift, None, solver)
    passthrough, _ = _shifted_solve(
        residual, phi, residual(phi), shift, None, solver, jacobian_fn=residual
    )
    stand_in, _ = _shifted_solve(
        residual,
        phi,
        residual(phi),
        shift,
        None,
        solver,
        jacobian_fn=jax.tree_util.Partial(_scaled_slope_residual, theta=theta),
    )

    # Passing the residual itself is the default path exactly, so `None` cannot be a silent special
    # case that skips the substitution.
    assert jnp.array_equal(exact, passthrough)
    assert jnp.allclose(stand_in, exact / _STAND_IN_SLOPE, rtol=1e-8)


def test_a_stand_in_jacobian_leaves_the_adjoint_exact() -> None:
    """The gradient is unchanged, because the adjoint never consults the forward step.

    The implicit-function-theorem reverse rule differentiates the residual it was handed, at the
    converged state, so an approximate *forward* operator cannot reach it. That is the property that
    makes a cheaper forward Jacobian legitimate at all -- the sensitivity a user asks for stays exact
    however loosely the march got to the root.
    """
    theta = jnp.array([8.0])

    def solved_sum(t: jnp.ndarray, step: PseudoTransientStep) -> jnp.ndarray:
        solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=400, forward_step=step)
        return jnp.sum(solver.solve(_residual, jnp.ones_like(t), t))

    exact_step = _jacobian_narrowed_step()
    stand_in_step = _jacobian_narrowed_step(
        jacobian_residual=lambda p: _scaled_slope_residual(p, theta)
    )
    closed_form = (1.0 / 3.0) * theta ** (-2.0 / 3.0)

    for step in (exact_step, stand_in_step):
        assert jnp.allclose(jax.grad(solved_sum)(theta, step), closed_form, atol=1e-6)


def test_the_dual_time_step_carries_a_stand_in_jacobian_too() -> None:
    """Both shifted steps read the field, so a march does not lose it by running an inner loop.

    ``DualTimeStep`` is what a coupled march actually runs (``inner_steps > 1``), and it forms its own
    shifted solve rather than delegating to the single-step one -- so wiring the single-step branch
    alone would leave the field inert on the path that matters.
    """
    theta = jnp.array([8.0, 27.0])
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=4,
        inner_tol=1e-2,
        jacobian_residual=lambda p: _scaled_slope_residual(p, theta),
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=400, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_divergence_guard_accepts_below_cap_and_rejects_divergence() -> None:
    """The default acceptance policy is decidable from scalar norms alone (no solve, no mesh)."""
    guard = DivergenceGuard(divergence_cap=10.0)
    r0 = jnp.asarray(1.0)
    previous, attempt = jnp.asarray(2.0), jnp.asarray(0)  # unused by a pure divergence guard

    # Finite candidate below cap × ‖R₀‖ is accepted; at/above the cap, or non-finite, is rejected.
    assert bool(guard.accept(jnp.asarray(5.0), previous, r0, attempt))
    assert bool(guard.accept(jnp.asarray(9.999), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(10.0), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(50.0), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(jnp.inf), previous, r0, attempt))
    assert not bool(guard.accept(jnp.asarray(jnp.nan), previous, r0, attempt))

    # The cap scales with the initial residual and is tunable.
    assert bool(
        DivergenceGuard(divergence_cap=100.0).accept(jnp.asarray(50.0), previous, r0, attempt)
    )


def test_injected_acceptance_policy_is_honoured() -> None:
    """A custom acceptance policy is used by the engine — the seam is real, not just present.

    ``RejectFirstAttempt`` refuses the first (undamped-schedule) attempt of every step, forcing one
    escalation; the solve must still converge, proving the engine routes the accept/reject decision
    through the injected policy rather than a hardwired test.
    """

    class RejectFirstAttempt(eqx.Module):
        def accept(self, candidate_norm, residual_norm, residual_norm_0, attempt):
            finite_bounded = jnp.isfinite(candidate_norm) & (
                candidate_norm < 10.0 * residual_norm_0
            )
            return finite_bounded & (attempt > 0)

    theta = jnp.array([8.0, 27.0])
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        acceptance=RejectFirstAttempt(),
    )
    solver = ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=200, forward_step=step)

    phi = solver.solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_backtracking_line_search_picks_largest_descending_rung() -> None:
    """The shared backtracking helper keeps the largest step length that reduces the residual, and
    falls back to the smallest rung when none does. Physics-free: ``R(x) = x`` so ``||R|| = |x|``."""
    residual = lambda x: x  # noqa: E731
    phi = jnp.array([1.0])
    reference = jnp.asarray(1.0)  # ||R(phi)||

    # delta = -4: full step x = -3 (|R| = 3, overshoot); alpha = 1/2 -> x = -1 (|R| = 1, not < 1);
    # alpha = 1/4 -> x = 0 (|R| = 0 < 1). Largest descending rung is 1/4.
    out, alpha, _ = backtracking_line_search(residual, phi, jnp.array([-4.0]), reference, steps=4)
    assert jnp.allclose(out, 0.0)
    assert jnp.allclose(alpha, 0.25)  # the kept fraction is reported

    # steps = 0 takes the full (overshooting) step unchanged, and reports alpha = 1.
    full, full_alpha, _ = backtracking_line_search(
        residual, phi, jnp.array([-4.0]), reference, steps=0
    )
    assert jnp.allclose(full, -3.0)
    assert jnp.allclose(full_alpha, 1.0)

    # delta = +4: every rung increases the residual, so none is admissible and the search falls back
    # to the LONGEST finite rung -- the full step. Falling back to the shortest instead would return a
    # near-null step that changes nothing, which is a guaranteed stall rather than a slow step.
    fallback, fb_alpha, _ = backtracking_line_search(
        residual, phi, jnp.array([4.0]), reference, steps=4
    )
    assert jnp.allclose(fallback, 1.0 + 1.0 * 4.0)
    assert jnp.allclose(fb_alpha, 1.0)


def test_line_search_recovers_an_overshooting_step_without_escalation() -> None:
    """With the escalation fallback disabled, the line search alone rescues a step whose full shifted
    correction overshoots -- the stiff-first-step regime the coupled RANS solve hits.

    From ``phi = 1`` toward the root ``phi = 10`` (``theta = 1000``) with only a weak shift, the full
    Newton correction lands near ``phi ~ 334`` and the cubic residual explodes. A backtracking search
    scales it back to a descent; without it (and without escalation) the step is rejected every
    iteration and the solve never converges.
    """
    theta = jnp.array([1000.0])
    policy = UniformShiftPolicy(strength=1.0)

    searched = ImplicitNewtonSolver(
        rtol=1e-8,
        atol=1e-10,
        max_steps=200,
        forward_step=PseudoTransientStep(
            policy,
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=0.01),
            max_escalations=0,
            line_search=40,
        ),
    )
    phi = searched.solve(_residual, jnp.ones_like(theta), theta)
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-5)

    # No line search and no escalation: the overshoot is never tamed, so the solve cannot converge.
    unsearched = ImplicitNewtonSolver(
        rtol=1e-8,
        atol=1e-10,
        max_steps=50,
        forward_step=PseudoTransientStep(
            policy,
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=0.01),
            max_escalations=0,
            line_search=0,
        ),
    )
    with pytest.raises(Exception):  # noqa: B017  (EquinoxRuntimeError, raised at solve time)
        jax.block_until_ready(unsearched.solve(_residual, jnp.ones_like(theta), theta))


def test_stepper_returns_the_step_and_its_linear_solve_cycle_count() -> None:
    """``stepper()`` returns ``(phi_next, cycles)`` -- the step, and what its shifted solve cost.

    The count is the cost of the *accepted* attempt's shifted solve, the signal an observed march
    watches to decide a frozen preconditioner has gone stale. There is one stepper: a caller with no
    use for the count drops it, rather than there being a second count-free method to drift from.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    phi0 = jnp.ones_like(theta)
    step = PseudoTransientStep(
        UniformShiftPolicy(strength=1.0), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0)
    )
    residual_norm_0 = jnp.linalg.norm(_residual(phi0, theta))
    solver = step.default_solver()

    def residual_fn(phi):
        return _residual(phi, theta)

    outcome = step.stepper()(residual_fn, phi0, residual_norm_0, solver)
    phi_next, cycles, alpha = outcome.phi, outcome.cycles, outcome.alpha

    # A real shifted solve was taken: the iterate moved, and stayed finite. Deliberately not a
    # descent assertion -- the pseudo-transient march is non-monotone (which is why its acceptance
    # policy is a divergence guard rather than a descent test), so one step need not reduce ‖R‖.
    assert not jnp.allclose(phi_next, phi0)
    assert bool(jnp.all(jnp.isfinite(phi_next)))
    assert int(cycles) > 0
    assert cycles.dtype == jnp.int32  # invariant carry dtype for a lax.while_loop
    # This step has no line search (default line_search=0), so the full shifted step is taken: alpha=1.
    assert jnp.allclose(alpha, 1.0)


def test_monotone_growth_is_the_default_and_reproduces_strict_descent() -> None:
    """The default schedule is ``1`` everywhere, i.e. the classical strict-descent ladder."""
    schedule = MonotoneLineSearch()
    for ratio in (1.0, 1e-2, 1e-8):
        got = schedule.growth(jnp.asarray(ratio), jnp.asarray(1.0))
        assert float(got) == 1.0


def test_relaxed_growth_admits_growth_far_out_and_restores_descent_in_the_basin() -> None:
    """``RelaxedFarFromRoot`` relaxes monotonicity far from the root and restores it below ``basin``.

    The point of the schedule: a pseudo-time march is not a descent method, so strict descent vetoes
    correct steps while the transient is still being traversed -- but near the root monotonicity is
    what delivers the terminal quadratic phase, so it must come back.
    """
    schedule = RelaxedFarFromRoot(max_growth=2.0, basin=1e-2)
    far = float(schedule.growth(jnp.asarray(1.0), jnp.asarray(1.0)))
    edge = float(schedule.growth(jnp.asarray(3e-2), jnp.asarray(1.0)))
    basin = float(schedule.growth(jnp.asarray(1e-2), jnp.asarray(1.0)))
    deep = float(schedule.growth(jnp.asarray(1e-6), jnp.asarray(1.0)))
    assert far == 2.0  # fully relaxed far from the root
    assert 1.0 < edge < 2.0  # smooth transition, not a switch
    assert basin == 1.0 and deep == 1.0  # strict descent restored in the basin
    assert edge > basin  # monotone in the ratio


def test_the_line_search_takes_the_longest_admissible_step_not_the_best_one() -> None:
    """Largest admissible, not minimizing -- distance travelled beats residual depth on a march.

    A minimizing search reaches a lower residual per step but travels much less far. Measured on a
    stiff coupled case, it developed the recirculation nine times more slowly while reporting better
    residuals at every early step. Here the measure is minimized at ``alpha = 1/4`` but the full step
    is also admissible, and the full step is what must be taken.
    """

    def residual_fn(phi):
        return jnp.abs(phi - 0.75) + 0.1  # minimized at alpha = 1/4 along delta = -1 from phi = 1

    phi = jnp.array([1.0])
    delta = jnp.array([-1.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    alpha = backtracking_line_search(residual_fn, phi, delta, reference, 10, growth=2.0).alpha
    # alpha = 1 lands at 0.85, outside the 2x tolerance (0.7); alpha = 1/2 lands at 0.35 and is the
    # longest that fits. The MINIMIZER is alpha = 1/4 (0.1) -- a shorter, better step this must not take.
    assert float(alpha) == 0.5


def test_the_ladder_reaches_step_lengths_longer_than_the_full_step() -> None:
    """``grow`` rungs above one, because the admissible step is often longer than the full step.

    Measured on a developed state: the full step moved the reattachment not at all while ``alpha`` of
    about 5.7 moved it four times further, and that longer step already sat inside the tolerance the
    acceptance rule allowed -- it was simply unreachable from a ladder starting at one. Here every step
    length keeps reducing out to ``alpha = 4``, so that is what a largest-admissible search must take.
    """

    def residual_fn(phi):
        return jnp.abs(phi - 4.0) + 0.1  # still improving all the way out to alpha = 4

    phi = jnp.array([0.0])
    delta = jnp.array([1.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    without = backtracking_line_search(residual_fn, phi, delta, reference, 6).alpha
    with_growth = backtracking_line_search(residual_fn, phi, delta, reference, 6, grow=3).alpha
    assert float(without) == 1.0  # a one-sided ladder cannot express it
    assert float(with_growth) == 4.0


def test_when_nothing_is_admissible_the_search_moves_as_far_as_it_finitely_can() -> None:
    """The fallback must not be the shortest rung -- that is a null step and a guaranteed stall.

    With no admissible rung the old search returned its smallest, a step so short it changes nothing,
    which the divergence guard then accepted as finite: the march reported a step and stood still.
    Moving as far as the arithmetic allows at least leaves the basin, and whether that step is kept is
    the accept/escalate test's decision. Here every step length raises the norm beyond the tolerance,
    and the full step is finite, so the full step is what comes back.
    """

    def residual_fn(phi):
        return phi * 1.05  # any step along +delta raises the norm

    phi = jnp.array([1.0])
    delta = jnp.array([1.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    alpha = backtracking_line_search(residual_fn, phi, delta, reference, 10).alpha
    assert float(alpha) == 1.0  # the longest finite rung, not 0.5**10


def test_the_fallback_skips_step_lengths_that_overflow() -> None:
    """The longest *finite* rung -- an overflowing trial state is never returned.

    A non-finite trial compares False against the acceptance test, so without an explicit finiteness
    check the fallback could hand back a step that blew the state up.
    """

    def residual_fn(phi):
        return jnp.where(
            phi > 2.5, jnp.inf, phi * 1.05
        )  # all rungs inadmissible; long ones overflow

    phi = jnp.array([1.0])
    delta = jnp.array([4.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    alpha = backtracking_line_search(residual_fn, phi, delta, reference, 6).alpha
    assert float(alpha) < 1.0  # alpha = 1 would land at phi = 5 and overflow
    assert bool(jnp.isfinite(residual_fn(phi + float(alpha) * delta)).all())


def test_growth_factor_widens_what_the_line_search_accepts() -> None:
    """A growth factor above one accepts a step strict descent rejects.

    Insisting on a decrease every step is incompatible with pseudo-transient continuation: the shift
    makes the residual rise along the path out of a bad basin, so a strictly monotone test rejects
    exactly the steps that would escape it. Allowing controlled growth far from the root admits them.
    """

    def residual_fn(phi):
        return phi * 1.05

    phi = jnp.array([1.0])
    delta = jnp.array([1.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    monotone = backtracking_line_search(residual_fn, phi, delta, reference, 10).alpha
    relaxed = backtracking_line_search(residual_fn, phi, delta, reference, 10, growth=2.5).alpha
    # Strict descent finds nothing admissible and falls back to the longest finite rung; the relaxed
    # test *accepts* the full step outright. Same alpha here, but one is a fallback and one a choice.
    assert float(monotone) == 1.0
    assert float(relaxed) == 1.0


def test_a_growth_rung_is_never_reached_by_falling_back_onto_it() -> None:
    """The fallback is capped at the full step, so extending the ladder cannot license an excursion.

    Without the cap, adding rungs above one also moves the no-admissible-step fallback above one, and a
    step with no acceptable length quietly becomes a multiple of the full step. Measured on a real
    march with two growth rungs: it fell back onto ``alpha = 4`` and multiplied its residual measure by
    4.6 in a single step. Growth rungs must be reachable only by PASSING the acceptance test.
    """

    def residual_fn(phi):
        return phi * 1.05  # every step length raises the norm, so nothing is admissible

    phi = jnp.array([1.0])
    delta = jnp.array([1.0])
    reference = jnp.linalg.norm(residual_fn(phi))
    capped = backtracking_line_search(residual_fn, phi, delta, reference, 6, grow=3).alpha
    assert float(capped) == 1.0  # not 8.0, the longest rung on the extended ladder

    # ...while a growth rung IS taken when it genuinely passes the acceptance test.
    def improving(phi):
        return jnp.abs(phi - 4.0) + 0.1  # keeps improving out to alpha = 4

    taken = backtracking_line_search(
        improving,
        jnp.array([0.0]),
        jnp.array([1.0]),
        jnp.linalg.norm(improving(jnp.array([0.0]))),
        6,
        grow=3,
    ).alpha
    assert float(taken) == 4.0


class _NegativeRoot(eqx.Module):
    """A residual whose Newton step drives the first entry straight through zero."""

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        return phi - jnp.array([-5.0, 1.0])


def _unshifted_step(**overrides) -> PseudoTransientStep:
    """A pseudo-transient step at zero shift, so the correction is the plain Newton one."""
    return dataclasses.replace(
        PseudoTransientStep(
            UniformShiftPolicy(),
            relaxation_schedule=ConstantRelaxation(jnp.asarray(0.0)),
            line_search=6,
        ),
        **overrides,
    )


def test_the_positivity_cap_is_available_on_the_pseudo_transient_step() -> None:
    """The guard protects the *state*, so it cannot depend on which strategy is stepping it.

    It used to live only on :class:`~aquaflux.solve.DualTimeStep`, so choosing the single-step strategy
    silently gave up a guard whose absence is a recorded march death: two cells of 23040 took ``k``
    negative and NaN'd the whole residual through a bare ``sqrt``, with every field still finite and
    nothing in the ordinary stopping tests able to see it.
    """
    phi0 = jnp.array([1.0, 2.0])
    common = dict(max_steps=1, rtol=1e-12, atol=1e-14)

    unguarded = forward_march(_unshifted_step(), _NegativeRoot(), phi0, **common)
    guarded = forward_march(
        _unshifted_step(step_limit=positive_block_limit(0, 2, tau=0.99)),
        _NegativeRoot(),
        phi0,
        **common,
    )

    assert float(unguarded.state[0]) < 0.0  # the unconstrained step really does cross zero
    assert float(guarded.state[0]) > 0.0  # and the cap holds it back
    # `tau = 0.99` leaves the binding entry at 1% of its value -- the fraction-to-the-boundary rule.
    assert float(guarded.state[0]) == pytest.approx(0.01, rel=1e-6)


def test_the_per_entry_projection_is_available_too_and_leaves_the_cap_inactive() -> None:
    """The projection clips each entry's own correction, so the global cap then finds nothing binding.

    Same composition the dual-time step relies on, and the reason both are offered rather than one: a
    global cap lets a single dead entry throttle every other one, which is what the projection exists
    to avoid.
    """
    phi0 = jnp.array([1.0, 2.0])
    projected = forward_march(
        _unshifted_step(step_projection=positive_block_projection(0, 2, tau=0.99)),
        _NegativeRoot(),
        phi0,
        max_steps=1,
        rtol=1e-12,
        atol=1e-14,
    )

    assert float(projected.state[0]) > 0.0  # held positive
    # The second entry reaches its own root: the projection did not shorten the whole step for it.
    assert float(projected.state[1]) == pytest.approx(1.0, rel=1e-6)
    assert float(projected.reports[0].binding_limit) == pytest.approx(1.0)


def test_both_default_to_off_and_leave_the_step_untouched() -> None:
    """Off by default, and byte-identical off -- the guard is opt-in on both strategies."""
    plain = PseudoTransientStep(UniformShiftPolicy())
    assert plain.step_limit is None and plain.step_projection is None

    phi0 = jnp.array([1.0, 2.0])
    common = dict(max_steps=1, rtol=1e-12, atol=1e-14)
    without = forward_march(_unshifted_step(), _NegativeRoot(), phi0, **common)
    explicit_none = forward_march(
        _unshifted_step(step_limit=None, step_projection=None), _NegativeRoot(), phi0, **common
    )
    assert jnp.array_equal(without.state, explicit_none.state)
