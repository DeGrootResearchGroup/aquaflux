"""The dual-time (backward-Euler) forward step drives a non-flow residual and keeps the IFT adjoint.

``DualTimeStep`` holds a reference ``phi^n`` and runs an inner Newton loop on the transient residual
``G = R + beta d (phi - phi^n)`` each outer timestep, so the shift sits in the residual (not only the
Jacobian) and the measured steady residual is the honest discrete time derivative rather than
``beta x travel``. These tests exercise it on a small scalar nonlinear root with a trivial shift
policy -- no mesh, no flow -- proving: it converges, its gradient is the exact
implicit-function-theorem sensitivity (independent of the iteration count), and with a single inner
step it reduces to the ordinary pseudo-transient step.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    DivergenceGuard,
    DualTimeStep,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
    positive_block_limit,
    positive_block_projection,
)
from aquaflux.solve.implicit import backtracking_line_search

#: Incremented on every *trace* of a residual below, so a test can assert that a rebuilt step
#: reuses the compiled march step instead of retracing it.
_TRACES: list[int] = []


class UniformShiftPolicy(eqx.Module):
    """A minimal non-flow shift policy: a uniform pseudo-time shift on every DOF, unpreconditioned."""

    strength: float = eqx.field(static=True, default=1.0)

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        diagonal = self.strength * jnp.ones_like(phi)
        return ShiftTerm(diagonal, lambda relaxation: None)


def _residual(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """A nonlinear residual with root ``phi = cbrt(theta)`` (per component)."""
    return phi**3 - theta


def _solver(step: DualTimeStep, max_steps: int = 200) -> ImplicitNewtonSolver:
    return ImplicitNewtonSolver(rtol=1e-10, atol=1e-10, max_steps=max_steps, forward_step=step)


def test_dual_time_converges_without_flow() -> None:
    """The dual-time step converges a nonlinear root using only an injected scalar shift policy."""
    theta = jnp.array([8.0, 27.0, 64.0])
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=4,
    )
    phi = _solver(step).solve(_residual, jnp.ones_like(theta), theta)

    assert jnp.all(jnp.isfinite(phi))
    assert jnp.allclose(phi, jnp.cbrt(theta), atol=1e-6)


def test_dual_time_is_differentiable() -> None:
    """Reverse-mode gradient through the converged dual-time solve matches the closed form."""
    theta = jnp.array([8.0])
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=4,
    )

    def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(_solver(step).solve(_residual, jnp.ones_like(t), t))

    grad = jax.grad(solved_sum)(theta)

    # d/dtheta cbrt(theta) = (1/3) theta^(-2/3); the IFT adjoint is independent of the shift and of the
    # inner/outer iteration structure.
    assert jnp.allclose(grad, (1.0 / 3.0) * theta ** (-2.0 / 3.0), atol=1e-6)


def test_dual_time_gradient_is_iteration_count_independent() -> None:
    """The adjoint is the transpose solve at the root, so it does not depend on how the march got there.

    Varying the outer step cap and the inner-loop depth changes the forward path but not the converged
    root, so the gradient must be identical -- the signature of an implicit-function-theorem adjoint
    rather than an unrolled iteration.
    """
    theta = jnp.array([8.0, 27.0])

    def grad_with(inner_steps: int, max_steps: int) -> jnp.ndarray:
        step = DualTimeStep(
            UniformShiftPolicy(strength=1.0),
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
            inner_steps=inner_steps,
        )

        def solved_sum(t: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(_solver(step, max_steps=max_steps).solve(_residual, jnp.ones_like(t), t))

        return jax.grad(solved_sum)(theta)

    reference = grad_with(inner_steps=2, max_steps=200)
    assert jnp.allclose(grad_with(inner_steps=5, max_steps=200), reference, atol=1e-8)
    assert jnp.allclose(grad_with(inner_steps=2, max_steps=500), reference, atol=1e-8)


def test_dual_time_inner_loop_iterates_and_sums_cost() -> None:
    """inner_steps > 1 runs multiple inner Newton steps: it converges G further and sums the cycles.

    Guards the per-step signals a StepControl reads: a broken accumulation (cycles overwritten instead
    of summed, or the inner loop not iterating) would still pass the convergence/gradient tests, which
    only check the root. Here the shared anchor is phi0 and beta = 1 (the residual ratio is 1 at the
    anchor), so the transient residual is G(p) = R(p) + (p - phi0); more inner iterations must drive it
    lower and cost more linear-solve cycles.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)
    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    def run(inner_steps: int):
        step = DualTimeStep(
            policy, relaxation_schedule=schedule, inner_steps=inner_steps, inner_tol=1e-6
        )
        return step.stepper()(residual_theta, phi0, r0, step.default_solver())

    def gnorm(p: jnp.ndarray) -> float:
        return float(jnp.linalg.norm(_residual(p, theta) + (p - phi0)))

    phi1, cyc1 = (lambda o: (o.phi, o.cycles))(run(1))
    phi4, cyc4 = (lambda o: (o.phi, o.cycles))(run(4))
    assert gnorm(phi4) < gnorm(phi1)  # more inner iterations converge the implicit step further
    assert int(cyc4) > int(cyc1)  # cycles are summed over the inner iterations, not overwritten


def test_dual_time_inner_observer_surfaces_the_trajectory_without_changing_the_step() -> None:
    """The opt-in inner_observer fires once per inner iteration with its G-norm before/after, cycle
    count and line-search factor -- and setting it leaves the computed step byte-identical."""
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)
    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    records: list[tuple[int, float, float, int, float]] = []
    iterates: list[jnp.ndarray] = []

    def observer(inner, g_before, g_after, cycles, alpha, iterate) -> None:
        records.append((int(inner), float(g_before), float(g_after), int(cycles), float(alpha)))
        iterates.append(jnp.asarray(iterate))

    observed = DualTimeStep(
        policy, relaxation_schedule=schedule, inner_steps=4, inner_tol=1e-8, inner_observer=observer
    )
    plain = DualTimeStep(policy, relaxation_schedule=schedule, inner_steps=4, inner_tol=1e-8)
    outcome = observed.stepper()(residual_theta, phi0, r0, observed.default_solver())
    phi_obs, n_inner = outcome.phi, outcome.inner_iterations
    phi_obs.block_until_ready()  # flush the ordered debug callbacks
    phi_plain = plain.stepper()(residual_theta, phi0, r0, plain.default_solver()).phi

    assert len(records) == int(n_inner) >= 1  # one record per inner iteration
    assert [r[0] for r in records] == list(range(len(records)))  # indices 0,1,2,... in order
    assert records[-1][2] < records[0][1]  # G-norm falls across the inner loop
    assert jnp.allclose(phi_obs, phi_plain)  # the observer does not perturb the step
    # The hook also carries each inner ITERATE, which is the only way to reach the states where a
    # march's expensive solves happen -- a checkpoint is written at the end of a step, after them.
    assert len(iterates) == len(records)
    assert jnp.allclose(iterates[-1], phi_obs)  # the last inner iterate IS the step's result


def test_dual_time_one_inner_step_is_a_single_shifted_step() -> None:
    """With one inner step the dual-time step is exactly one shifted Newton step.

    At the anchor the transient term is zero, so ``G(phi^n) = R(phi^n)`` and one inner solve is the
    same attempt ``PseudoTransientStep`` forms. Compared against a pseudo-transient step with its
    escalation disabled and a permissive guard (so it, too, takes its raw first attempt), the single
    inner iterate must match. Both strategies must also converge to the same root -- their overshoot
    handling differs (inner loop vs escalation) but the fixed point does not.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    schedule = SwitchedEvolutionRelaxation(beta0=1.0)
    policy = UniformShiftPolicy(strength=1.0)

    dual = DualTimeStep(policy, relaxation_schedule=schedule, inner_steps=1, line_search=0)
    # No escalation and a permissive guard: PseudoTransientStep then takes its raw first attempt, which
    # is the same single shifted step DualTimeStep takes when the inner loop is one step and unclipped.
    raw_shifted = PseudoTransientStep(
        policy,
        relaxation_schedule=schedule,
        line_search=0,
        max_escalations=0,
        acceptance=DivergenceGuard(divergence_cap=1e12),
    )

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    dual_next = dual.stepper()(residual_theta, phi0, r0, dual.default_solver()).phi
    shifted_next = raw_shifted.stepper()(residual_theta, phi0, r0, raw_shifted.default_solver()).phi
    assert jnp.allclose(dual_next, shifted_next, atol=1e-10)

    # And the full dual-time march reaches the same root as the escalating pseudo-transient march.
    pseudo = PseudoTransientStep(policy, relaxation_schedule=schedule)
    root_dual = _solver(dual).solve(_residual, phi0, theta)
    root_pseudo = ImplicitNewtonSolver(
        rtol=1e-10, atol=1e-10, max_steps=200, forward_step=pseudo
    ).solve(_residual, phi0, theta)
    assert jnp.allclose(root_dual, root_pseudo, atol=1e-8)


def test_dual_time_cycle_budget_caps_the_inner_loop() -> None:
    """A ``cycle_budget`` stops the inner loop once its accumulated linear-solve count reaches the budget.

    With ``inner_tol = 0`` the inner target is unreachable, so an unbounded step runs the full
    ``inner_steps`` and accumulates their summed solve count; the budgeted step instead stops after the
    first inner iteration that carries the running count to the budget -- the cost bailout that keeps a
    grinding primary solve from running every inner iteration into the restart cap. ``cycle_budget=None``
    leaves the step byte-identical.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20, inner_tol=0.0
    )
    unbounded = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    budgeted = DualTimeStep(UniformShiftPolicy(strength=1.0), cycle_budget=5, **common)

    out_u = unbounded.stepper()(residual_fn, phi0, r0, unbounded.default_solver())
    out_b = budgeted.stepper()(residual_fn, phi0, r0, budgeted.default_solver())
    cyc_u, inner_u = out_u.cycles, out_u.inner_iterations
    cyc_b, inner_b = out_b.cycles, out_b.inner_iterations

    assert int(inner_u) == 20  # unreachable target -> ran the full inner budget
    assert int(cyc_u) > 5  # accumulated more than the budget without a cap (sanity for the test)
    assert int(inner_b) < int(inner_u)  # the budget cut the loop short
    # Stops at the first inner iteration whose running total reaches the budget: at most one over-budget
    # solve's worth beyond it.
    assert 5 <= int(cyc_b) <= 5 + int(cyc_u) // int(inner_u) + 1


def test_dual_time_cycle_budget_none_is_the_unbounded_step() -> None:
    """The default ``cycle_budget=None`` reproduces the step with no budget field set, bit for bit."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=6)
    default = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    explicit_none = DualTimeStep(UniformShiftPolicy(strength=1.0), cycle_budget=None, **common)

    a = default.stepper()(residual_fn, phi0, r0, default.default_solver())
    b = explicit_none.stepper()(residual_fn, phi0, r0, explicit_none.default_solver())
    assert jnp.allclose(a[0], b[0]) and int(a[1]) == int(b[1]) and int(a[3]) == int(b[3])


def test_a_cut_short_step_reports_that_it_did_not_reach_its_target() -> None:
    """A cost-only escalation cannot tell an expensive success from a grind, and would bin the success.

    ``max_inner_cycles`` is reported alongside because the summed count is not an inner-count-invariant
    difficulty signal: it grows with how many times the step solved.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20)
    cut = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_tol=0.0, cycle_budget=5, **common)
    met = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_tol=1e-2, **common)

    cut_out = cut.stepper()(residual_fn, phi0, r0, cut.default_solver())
    met_out = met.stepper()(residual_fn, phi0, r0, met.default_solver())

    assert not bool(cut_out.reached_target)  # an unreachable target, stopped by the budget
    assert bool(met_out.reached_target)  # a loose target the inner loop actually met
    assert int(met_out.max_inner_cycles) <= int(met_out.cycles)  # one solve, not their sum


def test_positive_block_limit_keeps_a_constrained_field_off_zero() -> None:
    """Fraction-to-the-boundary: the largest step that keeps the block strictly positive.

    Rejecting violating rungs is not enough -- on the march that motivated this, the search was
    already at its shortest rung when the field crossed zero, so there was no shorter rung to take.
    Capping makes an admissible step reachable by construction.
    """
    limit = positive_block_limit(2, 5)
    phi = jnp.array([1.0, 1.0, 1.0e-5, 2.0, 3.0])
    delta = jnp.array([9.0, 9.0, -1.0e-3, 0.5, -1.0])

    alpha = limit(phi, delta)

    assert float(alpha) == pytest.approx(0.99 * 1.0e-5 / 1.0e-3)  # set by the one near-zero entry
    assert bool(((phi + alpha * delta)[2:5] > 0).all())
    # Entries outside the block are unconstrained: the cap ignores them however far they move.
    assert float(limit(phi, jnp.array([-1e3, 0.0, 1.0, 1.0, 1.0]))) == 1.0


def test_the_projection_holds_each_entry_off_zero_without_shortening_the_step() -> None:
    """The per-entry counterpart of the cap: a dead entry is held back ALONE.

    The failure it answers is the cap's one structural weakness -- it is a minimum over entries, so an
    entry whose value is numerically zero, and whose own equation is asking it to be zero, sets the
    step length for every other entry. Here the same near-zero entry is clipped on its own and the
    healthy entries keep their full corrections.
    """
    project = positive_block_projection(2, 5)
    phi = jnp.array([1.0, 1.0, 1.0e-5, 2.0, 3.0])
    delta = jnp.array([9.0, 9.0, -1.0e-3, 0.5, -1.0])

    clipped = project(phi, delta)

    # The dead entry is held to 0.99 of its own distance to zero...
    assert float(clipped[2]) == pytest.approx(-0.99 * 1.0e-5)
    # ...while every other entry, in and out of the block, keeps its correction exactly.
    assert [float(clipped[i]) for i in (0, 1, 3, 4)] == [9.0, 9.0, 0.5, -1.0]
    # A FULL step is now admissible, which is the whole point: the cap would have allowed 0.0099.
    assert bool(((phi + clipped)[2:5] > 0).all())


def test_the_projection_makes_the_cap_unbinding_so_the_two_compose() -> None:
    """Projection first, cap second: the cap then computes exactly 1 rather than being deleted.

    This is what lets the projection be added without tearing out the cap and everything keyed on it.
    After clipping, every decreasing entry satisfies ``|delta_i| <= tau (phi_i + floor)``, hence
    ``room_i >= 1 / tau`` and ``tau * min_i(room_i) >= 1`` -- so a step carrying both reports an
    unbinding cap and its diagnostics keep working, saying "nothing bound" rather than going silent.
    """
    phi = jnp.array([1.0e-12, 3.0e-2, 5.0, 0.25])
    delta = jnp.array([-7.0e-4, -1.0e-1, -9.0, -1.0e-3])
    for floor in (0.0, 1.0e-8):
        cap = positive_block_limit(0, 4, floor=floor)
        project = positive_block_projection(0, 4, floor=floor)
        # Unprojected, one near-zero entry throttles everything.
        assert float(cap(phi, delta)) < 1.0e-3
        assert float(cap(phi, project(phi, delta))) == pytest.approx(1.0)


def test_the_projection_is_inactive_at_a_root_and_is_a_compilation_cache_hit() -> None:
    """Two properties that keep it out of the converged state and out of the recompilation path.

    At a root the correction vanishes, so the projection returns it unchanged for any ``tau`` and
    ``floor`` -- which is what keeps it from perturbing the converged state, and therefore from
    reaching the implicit-function-theorem adjoint taken there. And like the cap it is a value object
    rather than a closure, so two built for the same block compare equal and a forward step carrying
    one stays a compilation-cache hit across rebuilds instead of retracing the whole solve.
    """
    phi = jnp.array([1.0, 2.0, 3.0])
    zero = jnp.zeros_like(phi)
    for floor in (0.0, 1.0e-8, 1.0e-3):
        assert bool((positive_block_projection(0, 3, floor=floor)(phi, zero) == zero).all())

    assert positive_block_projection(0, 3, floor=1e-8) == positive_block_projection(
        0, 3, floor=1e-8
    )
    assert positive_block_projection(0, 3, floor=1e-8) != positive_block_projection(
        0, 3, floor=1e-9
    )
    assert positive_block_projection(0, 3, floor=0.0) == positive_block_projection(0, 3)
    # ...and it is not interchangeable with the cap, which returns a scalar rather than a direction.
    assert positive_block_projection(0, 3) != positive_block_limit(0, 3)


def test_the_projection_does_not_reproduce_the_floors_ratchet() -> None:
    """A floored CAP postpones the collapse; the projection removes the coupling that caused it.

    Iterating the capped rule on a dead entry, ``(phi + floor)`` decays by exactly ``1 - tau`` per
    step whatever the floor is -- so the cap falls by that factor per step and every other entry pays.
    Under the projection the same entry decays identically, because that is what its own equation
    asks, but the admissible step for everything else stays 1.
    """
    tau, floor, correction = 0.99, 1.0e-8, -7.0e-4
    cap = positive_block_limit(0, 2, tau=tau, floor=floor)
    project = positive_block_projection(0, 2, tau=tau, floor=floor)

    capped, projected = jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])
    delta = jnp.array([correction, -1.0e-3])
    caps = []
    for _ in range(4):
        alpha = cap(capped, delta)
        caps.append(float(alpha))
        capped = capped + alpha * delta
        # The projection leaves a full step admissible at every iteration.
        assert float(cap(projected, project(projected, delta))) == pytest.approx(1.0)
        projected = projected + project(projected, delta)

    # The capped rule's admissible step falls by (1 - tau) per step, without bound.
    for earlier, later in itertools.pairwise(caps):
        assert later == pytest.approx((1.0 - tau) * earlier, rel=1e-6)
    # ...and the healthy entry has been dragged down with it, while the projected one is untouched.
    assert float(capped[1]) == pytest.approx(1.0, abs=1e-5)  # throttled: it never moved
    assert float(projected[1]) < float(capped[1])  # ...whereas this one made progress every step


def test_a_floored_limit_ignores_a_numerically_dead_entry_but_still_guards_a_live_one() -> None:
    """The floor exists so one entry that is numerically zero cannot set the step for all of them.

    The failure it answers: the cap is a minimum over entries, so taking ``tau`` of the distance to
    the boundary leaves the binding entry at ``1 - tau`` of its value, whose next room is smaller by
    the same factor. That is a geometric collapse -- on the march that motivated this, one cell of
    23040 drove the cap from 3.8e-03 to 1.1e-09 while its own correction never changed.
    """
    dead, live = 3.08e-22, 1.0e-2
    phi = jnp.array([dead, live])
    delta = jnp.array([-8.1e-14, -1.0e-2])

    plain = positive_block_limit(0, 2)
    floored = positive_block_limit(0, 2, floor=1.0e-8)

    # Unfloored, the dead entry sets the cap and it is catastrophic.
    assert float(plain(phi, delta)) == pytest.approx(0.99 * dead / 8.1e-14)
    # Floored, the dead entry is bought out of the minimum and the LIVE entry sets the cap -- the
    # limiter is still guarding, just not on an entry whose own equation is asking it to be zero.
    assert float(floored(phi, delta)) == pytest.approx(0.99 * (live + 1.0e-8) / 1.0e-2)


def test_a_floor_of_zero_is_the_plain_rule_and_a_floored_limit_is_inactive_at_a_root() -> None:
    """Two properties the floor must not break.

    ``floor=0`` has to be bit-identical, so adopting the parameter cannot move an existing march. And
    the limiter has to stay inactive at a root for **any** floor -- that is what keeps it out of the
    converged state, and therefore out of the implicit-function-theorem adjoint, which is taken at
    that state.
    """
    phi = jnp.array([1.0, 1.0e-5, 2.0])
    delta = jnp.array([-0.5, -1.0e-3, -1.0])

    assert float(positive_block_limit(0, 3, floor=0.0)(phi, delta)) == float(
        positive_block_limit(0, 3)(phi, delta)
    )
    # At a root the correction vanishes, so nothing decreases and the cap is 1 whatever the floor.
    for floor in (0.0, 1.0e-8, 1.0):
        assert float(positive_block_limit(0, 3, floor=floor)(phi, jnp.zeros_like(phi))) == 1.0


def test_a_floored_limiter_is_still_a_compilation_cache_hit() -> None:
    """The floor rides in the same static field, so it must not break value equality.

    A closure here would make every rebuild a fresh cache key and recompile the whole coupled solve;
    the limiter is a frozen dataclass of plain numbers precisely to avoid that, and adding a field
    must not change it.
    """
    assert positive_block_limit(0, 3, floor=1e-8) == positive_block_limit(0, 3, floor=1e-8)
    assert positive_block_limit(0, 3, floor=1e-8) != positive_block_limit(0, 3, floor=1e-9)
    assert positive_block_limit(0, 3, floor=0.0) == positive_block_limit(0, 3)


def test_a_capped_line_search_never_exceeds_the_cap() -> None:
    """The cap applies to every rung, including the growth rungs above one."""
    phi, delta = jnp.array([1.0]), jnp.array([-1.0])
    stepped, alpha = backtracking_line_search(
        lambda p: p * 0.0, phi, delta, jnp.asarray(1.0), steps=4, grow=2, max_alpha=0.1
    )

    assert float(alpha) <= 0.1
    assert float(stepped[0]) >= 0.9  # phi - alpha, with alpha capped


def test_the_inner_line_search_honours_an_injected_step_limit() -> None:
    """A step that would drive the constrained block negative is shortened, not taken."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=3)
    free = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    capped = DualTimeStep(
        UniformShiftPolicy(strength=1.0), step_limit=lambda p, d: jnp.asarray(0.01), **common
    )

    free_out = free.stepper()(residual_fn, phi0, r0, free.default_solver())
    capped_out = capped.stepper()(residual_fn, phi0, r0, capped.default_solver())

    assert float(capped_out.alpha) <= 0.01 < float(free_out.alpha)
    # The capped step still moves -- a cap shortens the step, it does not null it.
    assert not jnp.allclose(capped_out.phi, phi0)


def test_abort_above_inner_cycles_stops_a_doomed_attempt_early() -> None:
    """A solve costing more than the march's discard threshold ends the attempt there and then.

    ``retry_on_cycles`` is a PER-SOLVE quantity, so the moment one solve crosses it -- with the inner
    target still unmet -- the march is going to throw this attempt away and redo it at a larger shift.
    Every inner iteration after that point is work that is discarded. Measured on a three-dimensional
    coupled march, three such attempts ran 26, 56 and 59 cycles where the threshold was crossed at 14,
    17 and 16.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20, inner_tol=0.0
    )
    unbounded = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    # A threshold of 0 makes every solve "expensive", so the first one trips it.
    aborting = DualTimeStep(UniformShiftPolicy(strength=1.0), abort_above_inner_cycles=0, **common)

    out_u = unbounded.stepper()(residual_fn, phi0, r0, unbounded.default_solver())
    out_a = aborting.stepper()(residual_fn, phi0, r0, aborting.default_solver())

    assert int(out_u.inner_iterations) == 20  # unreachable target -> the full inner budget
    assert int(out_a.inner_iterations) == 1  # stopped as soon as one solve crossed the threshold
    assert int(out_a.cycles) < int(out_u.cycles)
    # Cut short, so the march must still see it as not having met its own criterion and escalate.
    assert not bool(out_a.reached_target)


def test_abort_above_inner_cycles_never_bins_an_expensive_success() -> None:
    """A costly solve that DOES reach the inner target exits normally and is kept.

    The loop tests the convergence target before the cost bailouts, so cost can only end an attempt
    that was going to be discarded anyway. Without this ordering the bailout would throw away a good
    iterate and replace it with a shorter step than the work already bought.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    # A target the FIRST inner iteration already meets, with a threshold of 0 so that same solve
    # also counts as over-cost -- the two conditions the ordering has to arbitrate between.
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=20, inner_tol=0.99
    )
    plain = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    aborting = DualTimeStep(UniformShiftPolicy(strength=1.0), abort_above_inner_cycles=0, **common)

    out_p = plain.stepper()(residual_fn, phi0, r0, plain.default_solver())
    out_a = aborting.stepper()(residual_fn, phi0, r0, aborting.default_solver())

    assert bool(out_p.reached_target)  # the target is reachable (sanity for the test)
    assert bool(out_a.reached_target)  # ...and the cost bailout did not prevent reaching it
    assert jnp.allclose(out_a.phi, out_p.phi)


def test_abort_above_inner_cycles_none_is_the_unbounded_step() -> None:
    """The default leaves the step byte-identical, so an unconfigured march is unchanged."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=6, inner_tol=1e-3
    )
    default = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    explicit = DualTimeStep(
        UniformShiftPolicy(strength=1.0), abort_above_inner_cycles=None, **common
    )

    out_d = default.stepper()(residual_fn, phi0, r0, default.default_solver())
    out_e = explicit.stepper()(residual_fn, phi0, r0, explicit.default_solver())
    assert jnp.array_equal(out_d.phi, out_e.phi)
    assert int(out_d.cycles) == int(out_e.cycles)


def test_the_march_pushes_its_discard_threshold_into_the_step() -> None:
    """``retry_on_cycles`` reaches the inner loop, so a doomed attempt can stop where it is detected.

    The threshold is a per-solve quantity and the march evaluates it only after the whole step returns,
    which means inner iterations keep running after the attempt is already destined to be discarded.
    Handing the number to the step lets it stop there. It must stay ONE number -- a step configured with
    its own copy would be a second spelling to keep in step with this one.
    """
    from aquaflux.solve.march import _with_inner_abort

    step = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_steps=3)
    assert step.abort_above_inner_cycles is None

    armed = _with_inner_abort(step, 10)
    assert armed.abort_above_inner_cycles == 10
    assert step.abort_above_inner_cycles is None  # the original is untouched


def test_a_step_with_no_inner_loop_is_returned_unchanged() -> None:
    """Only a step that runs inner solves has anything to abort; the rest must pass through untouched."""
    from aquaflux.solve.march import _with_inner_abort

    step = PseudoTransientStep(UniformShiftPolicy(strength=1.0))
    assert _with_inner_abort(step, 10) is step
    assert _with_inner_abort(step, None) is step


def test_no_discard_threshold_leaves_the_step_untouched() -> None:
    """``retry_on_cycles=None`` is the default, and must stay byte-identical."""
    from aquaflux.solve.march import _with_inner_abort

    step = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_steps=3)
    assert _with_inner_abort(step, None) is step


def test_a_rebuilt_step_limiter_is_a_compilation_cache_hit() -> None:
    """Two limiters built for the same block must be interchangeable in the compiled step's cache key.

    ``step_limit`` is a STATIC field, so it is part of that key, and static fields are compared by
    ``__eq__``. A closure compares by identity, so every rebuild was a fresh key and recompiled the
    whole coupled solve -- measured on a Reynolds ramp, ~100-150 s per rung at an identical cycle count.
    A value object compares by value, so a rebuild is a hit while a genuinely different block still
    retraces.
    """
    from aquaflux.solve import positive_block_limit
    from aquaflux.solve.march import _march_step

    assert positive_block_limit(1, 3) == positive_block_limit(1, 3)
    assert positive_block_limit(1, 3) != positive_block_limit(0, 3)

    # A unique state size, so a would-be recompile cannot be a hit from another test's compiled step.
    target = jnp.asarray([8.0, 27.0, 64.0, 125.0, 216.0])

    def residual_fn(
        phi: jnp.ndarray,
    ) -> jnp.ndarray:  # one stable object: never the thing that differs
        _TRACES.append(1)
        return phi**3 - target

    phi0 = jnp.ones(5)
    policy = UniformShiftPolicy(strength=1.0)

    def run(limit) -> int:
        step = DualTimeStep(policy, inner_steps=2, inner_tol=1e-3, step_limit=limit)
        before = len(_TRACES)
        _march_step(step, residual_fn, phi0, jnp.asarray(1.0), step.default_solver())
        return len(_TRACES) - before

    _TRACES.clear()
    assert run(positive_block_limit(1, 3)) > 0  # first build compiles
    for _ in range(2):  # two further "rungs", each rebuilding an equal limiter
        assert run(positive_block_limit(1, 3)) == 0
    assert run(positive_block_limit(0, 3)) > 0  # a genuinely different block must still retrace


def test_a_mid_step_refresh_buys_the_attempt_another_solve_before_the_abort() -> None:
    """Refreshing must FORGIVE the abort, or the rebuild is paid for and then thrown away with the step.

    ``abort_above_inner_cycles`` and the refresh trigger both fire after the same expensive solve, so
    without this the attempt aborts anyway and the march escalates the shift -- discarding the inner
    loop's progress *and* the pseudo-timestep, which is the cascade the refresh exists to prevent. So the
    abort counts cycles since the last refresh rather than since the step began, and the rebuilt
    preconditioner is judged on a solve of its own. A second expensive solve after it means the operator
    really is hard, and the abort then does its job.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(_residual(phi0, theta))

    def residual_theta(p: jnp.ndarray) -> jnp.ndarray:
        return _residual(p, theta)

    def build(**extra):
        return DualTimeStep(
            UniformShiftPolicy(strength=1.0),
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
            inner_steps=6,
            inner_tol=1e-14,  # never met, so only the cost bailouts can end the loop
            abort_above_inner_cycles=0,  # every solve looks over budget
            **extra,
        )

    refreshed: list[int] = []
    without = build()
    with_refresh = build(refresh_on_cycles=0, inner_refresh=lambda _it: refreshed.append(1))
    bare = without.stepper()(residual_theta, phi0, r0, without.default_solver())
    bare.phi.block_until_ready()
    forgiven = with_refresh.stepper()(residual_theta, phi0, r0, with_refresh.default_solver())
    forgiven.phi.block_until_ready()

    assert refreshed, "the refresh never fired"
    # The bare step is cut at its first over-budget solve; the refreshed one is given another.
    assert int(forgiven.inner_iterations) > int(bare.inner_iterations)


def test_abort_below_alpha_stops_an_attempt_that_can_no_longer_move() -> None:
    """A step length capped at essentially nothing ends the attempt, instead of iterating on in place.

    The failure this catches is invisible to the cost bailout, because the solves stay *cheap*: the
    correction simply cannot be followed. Here an injected limit admits 1e-12 of every step, which is
    the shape a positivity cap takes when a cell sits on the boundary -- measured on a three-dimensional
    coupled march at a limit of 4.4e-10, where four consecutive inner iterations moved ``‖G‖`` from
    4.442e-03 to 4.440e-03 and the loop ran to the end of its budget regardless.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=20,
        inner_tol=0.0,
        # Admit almost none of any correction, so the ladder cannot move the iterate.
        step_limit=lambda p, d: jnp.asarray(1e-12),
    )
    unbounded = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    aborting = DualTimeStep(UniformShiftPolicy(strength=1.0), abort_below_alpha=1e-6, **common)

    out_u = unbounded.stepper()(residual_fn, phi0, r0, unbounded.default_solver())
    out_a = aborting.stepper()(residual_fn, phi0, r0, aborting.default_solver())

    assert int(out_u.inner_iterations) == 20  # runs the full budget going nowhere
    assert int(out_a.inner_iterations) == 1  # stops as soon as the length collapses
    assert int(out_a.cycles) < int(out_u.cycles)
    # Cut short, so the march must still read it as not having met its own criterion and escalate.
    assert not bool(out_a.reached_target)


def test_abort_below_alpha_never_bins_a_step_that_reaches_its_target() -> None:
    """A clipped step that still brings ``‖G‖`` under the target exits normally and is kept.

    The loop tests convergence before either bailout, so a collapsed length can only end an attempt
    that was going to be discarded anyway.
    """
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    # inner_tol = 1.0 is met at the anchor itself, so the loop exits before any bailout can look.
    step = DualTimeStep(
        UniformShiftPolicy(strength=1.0),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        inner_steps=20,
        inner_tol=1.0,
        abort_below_alpha=1.0,  # would abort immediately if it were consulted first
    )
    out = step.stepper()(residual_fn, phi0, r0, step.default_solver())
    assert bool(out.reached_target)


def test_abort_below_alpha_none_is_the_unbounded_step() -> None:
    """The default leaves the step byte-identical -- the bailout is opt-in, like its cost sibling."""
    theta = jnp.array([8.0, 27.0, 64.0])

    def residual_fn(phi: jnp.ndarray) -> jnp.ndarray:
        return _residual(phi, theta)

    phi0 = jnp.ones_like(theta)
    r0 = jnp.linalg.norm(residual_fn(phi0))
    common = dict(
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0), inner_steps=6, inner_tol=1e-10
    )
    default = DualTimeStep(UniformShiftPolicy(strength=1.0), **common)
    explicit = DualTimeStep(UniformShiftPolicy(strength=1.0), abort_below_alpha=None, **common)

    out_d = default.stepper()(residual_fn, phi0, r0, default.default_solver())
    out_e = explicit.stepper()(residual_fn, phi0, r0, explicit.default_solver())
    assert jnp.array_equal(out_d.phi, out_e.phi)
    assert int(out_d.inner_iterations) == int(out_e.inner_iterations)


def test_the_march_pushes_both_discard_thresholds_into_the_step() -> None:
    """Cost and step-length thresholds both travel down to the inner loop, and stay ONE number each.

    Each is knowable inside the step -- a cost the moment a solve returns, a collapse the moment the
    ladder gives up -- while the march evaluates both only after the whole step is back. A step
    configured with its own copy of either would be a second spelling to keep in step with the march's.
    """
    from aquaflux.solve.march import _with_inner_abort

    step = DualTimeStep(UniformShiftPolicy(strength=1.0), inner_steps=3)
    assert step.abort_below_alpha is None

    armed = _with_inner_abort(step, 10, 0.01)
    assert armed.abort_above_inner_cycles == 10
    assert armed.abort_below_alpha == 0.01
    assert step.abort_below_alpha is None  # the original is untouched

    # Either alone arms only its own threshold.
    assert _with_inner_abort(step, None, 0.01).abort_above_inner_cycles is None
    assert _with_inner_abort(step, None, 0.01).abort_below_alpha == 0.01
    assert _with_inner_abort(step, 10).abort_below_alpha is None
    # Neither leaves the step untouched.
    assert _with_inner_abort(step, None, None) is step
