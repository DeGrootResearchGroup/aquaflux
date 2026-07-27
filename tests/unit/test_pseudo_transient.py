"""The pseudo-transient continuation engine drives a non-flow residual.

``PseudoTransientStep`` is the residual-agnostic pseudo-transient continuation strategy: the
switched-evolution-relaxation schedule, the shifted solve, and the accept/escalate loop, with the
problem-specific choices (which DOFs shift, the shift magnitude, the shifted preconditioner) supplied
by an injected ``ShiftPolicy``. These tests exercise it on a small scalar nonlinear root with a
trivial shift policy — no mesh, no flow assembler, no block preconditioner — proving the engine is
reusable beyond the coupled flow it was first built for.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    DivergenceGuard,
    ImplicitNewtonSolver,
    MonotoneLineSearch,
    PseudoTransientStep,
    RelaxedFarFromRoot,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
)
from aquaflux.solve.implicit import backtracking_line_search


class UniformShiftPolicy(eqx.Module):
    """A minimal non-flow shift policy: a uniform pseudo-time shift on every DOF, unpreconditioned.

    Attributes
    ----------
    strength : float
        The per-DOF base shift magnitude ``d`` (static); the engine scales it by the relaxation ``β``.
    """

    strength: float = eqx.field(static=True, default=1.0)

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        diagonal = self.strength * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda relaxation: None)


def _residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A nonlinear residual with root ``phi = cbrt(theta)`` (per component)."""
    return phi**3 - theta


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
    out, alpha = backtracking_line_search(residual, phi, jnp.array([-4.0]), reference, steps=4)
    assert jnp.allclose(out, 0.0)
    assert jnp.allclose(alpha, 0.25)  # the kept fraction is reported

    # steps = 0 takes the full (overshooting) step unchanged, and reports alpha = 1.
    full, full_alpha = backtracking_line_search(
        residual, phi, jnp.array([-4.0]), reference, steps=0
    )
    assert jnp.allclose(full, -3.0)
    assert jnp.allclose(full_alpha, 1.0)

    # delta = +4: every rung increases the residual, so none is admissible and the search falls back
    # to the LONGEST finite rung -- the full step. Falling back to the shortest instead would return a
    # near-null step that changes nothing, which is a guaranteed stall rather than a slow step.
    fallback, fb_alpha = backtracking_line_search(
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

    phi_next, cycles, alpha = step.stepper()(residual_fn, phi0, residual_norm_0, solver)

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
    _, alpha = backtracking_line_search(residual_fn, phi, delta, reference, 10, growth=2.0)
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
    _, without = backtracking_line_search(residual_fn, phi, delta, reference, 6)
    _, with_growth = backtracking_line_search(residual_fn, phi, delta, reference, 6, grow=3)
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
    _, alpha = backtracking_line_search(residual_fn, phi, delta, reference, 10)
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
    _, alpha = backtracking_line_search(residual_fn, phi, delta, reference, 6)
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
    _, monotone = backtracking_line_search(residual_fn, phi, delta, reference, 10)
    _, relaxed = backtracking_line_search(residual_fn, phi, delta, reference, 10, growth=2.5)
    # Strict descent finds nothing admissible and falls back to the longest finite rung; the relaxed
    # test *accepts* the full step outright. Same alpha here, but one is a fallback and one a choice.
    assert float(monotone) == 1.0
    assert float(relaxed) == 1.0


class _SaddleShift(eqx.Module):
    """A shift that damps the first row and leaves the second unshifted -- a constraint row.

    This is the shape the coupled flow policy has: the momentum rows carry the operator diagonal while
    continuity, being an algebraic constraint with no time derivative, carries zero.
    """

    def shift_term(self, phi):
        del phi
        return ShiftTerm(jnp.array([1.0, 0.0]), lambda relaxation: None)


def test_the_descent_backoff_lowers_the_shift_until_the_correction_descends() -> None:
    """The shift is backed OFF, not escalated, when no step length along the direction can help.

    The shifted correction is not a descent direction by construction, and the reason is the *mixture*
    of shifted and unshifted rows. On a saddle system whose constraint row carries no shift, the
    directional derivative of the residual measure along the correction goes

        beta = 0 -> -2.3,  beta = 0.5 -> -1.15,  beta = 2 -> +2.3,  beta = 10 -> +20.7

    -- descent is lost above a critical shift strength and gets worse from there. Every step length
    then raises the measure, so the line search can only pick the least-harmful rung and the march
    reports steps while standing still. Escalating there makes it worse; the cure is less shift, which
    is what the backoff does. (Shift *every* row uniformly and the derivative stays negative at any
    beta, which is why this needs the constraint row to reproduce.)
    """
    matrix = jnp.array([[1.0, 1.0], [-1.0, 0.0]])
    target = jnp.array([1.0, 0.3])

    def residual_fn(phi):
        return matrix @ phi - target

    phi = jnp.array([2.0, -1.0])
    # The sign change above is in an L1 measure -- the shape the equilibrated residual measure takes.
    common = dict(
        shift_policy=_SaddleShift(),
        line_search=6,
        max_escalations=0,
        residual_norm=lambda r: jnp.sum(jnp.abs(r)),
    )
    # beta0 = 2 puts the starting shift past the sign change measured above.
    schedule = SwitchedEvolutionRelaxation(beta0=2.0, exponent=0.0)
    without = PseudoTransientStep(**common, relaxation_schedule=schedule)
    with_backoff = PseudoTransientStep(
        **common, relaxation_schedule=schedule, descent_backoff=4, descent_test=True
    )

    measure = common["residual_norm"]
    r0 = measure(residual_fn(phi))
    solver = without.default_solver()
    plain, _, _ = without.stepper()(residual_fn, phi, r0, solver)
    backed, _, _ = with_backoff.stepper()(residual_fn, phi, r0, solver)

    # Backing the shift off reaches a lower residual than stepping at the non-descent shift strength.
    assert float(measure(residual_fn(backed))) < float(measure(residual_fn(plain)))
    assert float(measure(residual_fn(backed))) < float(r0)


def test_the_descent_machinery_is_off_by_default() -> None:
    """Both are opt-in: each backoff costs a shifted solve, so nothing pays for it unasked."""
    step = PseudoTransientStep(shift_policy=_SaddleShift())
    assert step.descent_backoff == 0
    assert step.descent_test is False
