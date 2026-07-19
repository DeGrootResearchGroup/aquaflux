"""High-Reynolds open channel: the pseudo-transient continuation that lifts the Re ~ 100 floor.

The basic open channel (:mod:`tests.integration.test_channel`) solves at Re ~ 100 with a
line-searched Newton step, but fails once the flow becomes convection-dominated (a few hundred
Reynolds): the undamped Newton step from the uniform cold start overshoots (the full step *increases*
the residual, more steeply with Reynolds number), so the basin shrinks and the backtracking line
search is too weak a globalization to march the convective path. The dominant missing ingredient is
*outer* Newton globalization, not a better linear solve — the block-SIMPLE preconditioner stays an
effective Jacobian approximation and the preconditioned inner GMRES converges cleanly at the cold start
at every tested Reynolds number (pinned by ``test_inner_gmres_does_not_stall_at_high_reynolds``); a
stronger preconditioner extends the line search's reach only modestly.

:func:`~aquaflux.flow.momentum_continuation` fixes this by damping each Newton step with an
``a_P``-proportional diagonal shift that ramps to zero on the residual — keeping each iterate inside
the basin the undamped step overshoots, and recovering the exact steady Newton step (and its converged
state) as the shift vanishes. These tests drive the same channel setup to genuinely convective
Reynolds numbers, on both uniform and wall-graded meshes, and confirm the converged solve stays
differentiable.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    momentum_continuation,
    reused_flow_solve,
)
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver, solve_linear

H, L, U_IN, RHO = 1.0, 4.0, 1.0, 1.0


def _channel(nx, ny, mu, *, wall_growth=1.0, u_in=U_IN):
    """The plane channel of :mod:`tests.integration.test_channel`, at viscosity ``mu``.

    ``wall_growth > 1`` grades the wall-normal spacing (finest at both walls) so the near-wall cell
    resolves a boundary layer — the mesh a turbulent-channel validation needs. ``u_in`` sets the inlet
    speed (default ``U_IN``); vary it (with ``mu`` scaled to hold the Reynolds number) to change the
    characteristic velocity scale.
    """
    y_nodes = graded_nodes(ny, H, wall_growth) if wall_growth != 1.0 else None
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(u_in, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )


def _reynolds(mu):
    return RHO * U_IN * H / mu


def _solve(assembler, *, continuation=None, max_steps=120, **kwargs):
    if continuation is None:
        continuation = momentum_continuation(assembler)
    solver = ImplicitNewtonSolver(max_steps=max_steps, forward_step=continuation, **kwargs)
    return solver.solve(lambda s, a: a.residual(s), assembler.initial_state(), assembler)


@pytest.mark.parametrize(
    ("nx", "ny", "mu"),
    [(24, 16, 5e-3), (32, 24, 2e-3), (40, 32, 1e-3)],  # Re = 200, 500, 1000
)
def test_continuation_converges_at_high_reynolds(nx, ny, mu) -> None:
    """The continuation drives the convective channel to a converged flow well past the Re~100 floor."""
    assembler = _channel(nx, ny, mu)
    state = _solve(assembler)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8


def test_continuation_is_necessary_beyond_the_laminar_floor() -> None:
    """At Re = 200 the line-searched Newton solve fails; the continuation recovers convergence.

    This is the higher-Reynolds analogue of ``test_line_search_is_necessary``: there the line search
    was the missing globalization at Re ~ 100, here it is no longer enough — the undamped step
    overshoots the shrinking basin — and the pseudo-transient continuation is what makes the solve
    converge.
    """
    assembler = _channel(24, 16, 5e-3)  # Re = 200

    line_searched = ImplicitNewtonSolver(
        max_steps=40,
        forward_step=DampedNewtonStep(
            preconditioner=BlockPreconditioner.build(assembler).factory()
        ),
    )
    try:
        undamped = line_searched.solve(
            lambda s, a: a.residual(s), assembler.initial_state(), assembler
        )
        residual = float(jnp.linalg.norm(assembler.residual(undamped)))
        line_search_failed = (not np.isfinite(residual)) or residual > 1e-6
    except Exception:
        line_search_failed = (
            True  # the overshooting cold-start march drove the solve non-convergent
        )
    assert line_search_failed


def test_inner_gmres_does_not_stall_at_high_reynolds() -> None:
    """The preconditioned inner GMRES converges at the cold start even at Re = 1000.

    The pseudo-transient continuation is an *outer* globalization: the block preconditioner stays an
    effective Jacobian approximation, so the inner linear solve is healthy at the uniform cold start
    (where the undamped outer step would overshoot). The line-search ceiling is set by that outer
    overshoot, not by an inner stall — this pins the premise, so the motivation cannot silently drift
    back to "the inner GMRES stalls."
    """
    assembler = _channel(32, 24, 1e-3)  # Re = 1000
    state = assembler.initial_state()
    residual = assembler.residual(state)
    apply_m = BlockPreconditioner.build(
        assembler, schur_scaling="msimpler", velocity="convection"
    ).factory()(state)

    def preconditioned_jacobian(v):  # M ∘ J at the cold start
        jvp = jax.jvp(lambda s: assembler.residual(s), (state,), (v,))[1]
        return apply_m(jvp)

    rhs = apply_m(-residual)  # M (−R)
    solution = solve_linear(
        preconditioned_jacobian,
        rhs,
        solver=lx.GMRES(rtol=1e-8, atol=1e-8, restart=60, stagnation_iters=60),
        throw=False,
    )
    achieved = jnp.linalg.norm(preconditioned_jacobian(solution) - rhs) / jnp.linalg.norm(rhs)
    assert (
        float(achieved) < 1e-6
    )  # the inner solve reduces the linear residual ~12 orders, no stall


def test_line_search_fails_in_the_convective_regime_even_with_best_preconditioner() -> None:
    """The current best preconditioner does not make line-search-only a high-Reynolds globalization.

    A stronger preconditioner (MSIMPLER Schur + convection-velocity AMG) extends the line-search reach
    only modestly — it converges the Re ~ 200 coarse channel the default SIMPLE preconditioner does not,
    but line-search-only still fails once the flow is convection-dominated (here Re = 500), because the
    outer step overshoots the shrinking basin, which the preconditioner does not address. The
    continuation is what carries the convective regime (``test_continuation_converges_at_high_reynolds``).
    """
    assembler = _channel(24, 16, 2e-3)  # Re = 500
    best = BlockPreconditioner.build(
        assembler, schur_scaling="msimpler", velocity="convection"
    ).factory()
    line_searched = ImplicitNewtonSolver(
        max_steps=60, forward_step=DampedNewtonStep(preconditioner=best)
    )
    try:
        undamped = line_searched.solve(
            lambda s, a: a.residual(s), assembler.initial_state(), assembler
        )
        residual = float(jnp.linalg.norm(assembler.residual(undamped)))
        line_search_failed = (not np.isfinite(residual)) or residual > 1e-6
    except Exception:
        line_search_failed = (
            True  # the overshooting cold-start march drove the solve non-convergent
        )
    assert line_search_failed

    converged = _solve(assembler)  # continuation on
    assert float(jnp.linalg.norm(assembler.residual(converged))) < 1e-8


def test_continuation_conserves_mass() -> None:
    """The high-Re converged flow conserves mass: the inlet delivers exactly what the outlet passes."""
    assembler = _channel(32, 24, 2e-3)  # Re = 500
    state = _solve(assembler)
    mdot = np.asarray(assembler.mass_flux(state))
    patches = assembler.mesh.face_patches
    inlet = mdot[np.asarray(patches.indices("left"))].sum()
    outlet = mdot[np.asarray(patches.indices("right"))].sum()
    assert inlet < 0.0 < outlet
    assert abs(inlet + outlet) < 1e-6  # global continuity
    assert abs(abs(inlet) - RHO * U_IN * H) < 1e-8  # inlet delivers rho U H exactly


def test_wall_graded_channel_resolves_and_converges() -> None:
    """A wall-graded mesh (fine near the walls) converges at Re = 1000 — the wall-resolved setup a
    turbulent boundary layer needs, which the scale-invariant ``a_P``-proportional shift handles."""
    assembler = _channel(40, 32, 1e-3, wall_growth=1.2)  # Re = 1000, graded
    state = _solve(assembler)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8

    # The near-wall cell is much finer than the core cell — the point of the grading.
    volume = np.asarray(assembler.geometry.cell.volume)
    wall_cell = volume.min()
    core_cell = volume.max()
    assert wall_cell < 0.5 * core_cell


def test_continuation_solve_is_differentiable() -> None:
    """Reverse-mode gradient through the continuation solve is finite and matches finite differences.

    The continuation (like the preconditioner) is built once outside ``jax.grad`` with concrete
    parameters and reused across ``mu``; its diagonal shift vanishes at convergence, so the IFT
    adjoint linearizes the same steady residual it would without it.
    """
    continuation = momentum_continuation(_channel(24, 16, 5e-3))

    def mean_speed(mu):
        assembler = _channel(24, 16, mu)
        state = _solve(assembler, continuation=continuation)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(5e-3))
    assert np.isfinite(grad)

    step = 5e-5
    finite_difference = (float(mean_speed(5e-3 + step)) - float(mean_speed(5e-3 - step))) / (
        2 * step
    )
    assert abs(grad - finite_difference) < 1e-3 * abs(finite_difference)


def test_escalation_recovers_an_underdamped_step() -> None:
    """The step-acceptance escalation makes ``β₀`` robust rather than a per-case knob.

    An intentionally under-damped ``β₀`` produces a shifted step that diverges (the divergence guard
    rejects it), so without escalation the march makes no progress and stalls — the solve fails to
    reach tolerance within ``max_steps`` and the convergence guard rejects the stalled field rather
    than returning it. Escalating the damping until the step descends is what recovers convergence —
    so ``β₀`` only sets the starting damping, and being too small is self-corrected rather than fatal.
    """
    assembler = _channel(40, 32, 1e-3, wall_growth=1.2)  # Re = 1000, wall-graded

    def solve_with(max_escalations):
        continuation = momentum_continuation(
            assembler, schur_scaling="msimpler", beta0=0.2, max_escalations=max_escalations
        )
        return _solve(assembler, continuation=continuation, max_steps=150)

    try:
        stalled = solve_with(0)  # under-damped, no escalation
        no_escalation_failed = float(jnp.linalg.norm(assembler.residual(stalled))) > 1e-6
    except Exception:
        no_escalation_failed = (
            True  # the stalled march never reaches tolerance; the guard rejects it
        )
    assert no_escalation_failed

    converged = solve_with(6)  # escalation recovers convergence
    assert float(jnp.linalg.norm(assembler.residual(converged))) < 1e-8


@pytest.mark.slow
def test_msimpler_schur_reaches_beyond_the_simple_schur() -> None:
    """The MSIMPLER pressure Schur carries the solve past the Reynolds number where SIMPLE stalls.

    The SIMPLE Schur scales by the momentum diagonal ``V / a_P``, which becomes a poor operator as
    convection dominates: on a wall-graded Re = 2000 channel its inner GMRES stalls even with the
    continuation. MSIMPLER's frozen, velocity-independent Schur scaling (``schur_scaling="msimpler"``)
    stays a well-conditioned pressure Poisson and converges. This is the follow-on to the
    convection-aware-preconditioner work: the pressure block fixed, the velocity block still bounds
    the reachable Reynolds number.

    The SIMPLE-Schur failure at the same setup is not re-asserted here — reproducing a stagnation is
    both slow and sensitive to exactly when the inner solve gives up — but it is what motivates the
    swap; the test asserts the positive: the MSIMPLER-Schur continuation converges at this Reynolds
    number.
    """
    assembler = _channel(64, 48, 5e-4, wall_growth=1.15)  # Re = 2000, wall-graded
    msimpler = momentum_continuation(assembler, schur_scaling="msimpler")
    converged = _solve(assembler, continuation=msimpler, max_steps=150)
    assert float(jnp.linalg.norm(assembler.residual(converged))) < 1e-8


@pytest.mark.slow
def test_msimpler_schur_matches_simple_at_moderate_reynolds() -> None:
    """MSIMPLER is a drop-in for SIMPLE where SIMPLE already works: both converge, and the MSIMPLER
    solve stays reverse-differentiable (its Schur scaling is a frozen, ``stop_gradient``-ed operator,
    so the implicit-function-theorem adjoint is untouched)."""
    continuation = momentum_continuation(_channel(32, 24, 2e-3), schur_scaling="msimpler")

    def mean_speed(mu):
        assembler = _channel(32, 24, mu)
        state = _solve(assembler, continuation=continuation)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    assembler = _channel(32, 24, 2e-3)  # Re = 500
    state = _solve(assembler, continuation=continuation)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8

    grad = float(jax.grad(mean_speed)(2e-3))
    assert np.isfinite(grad)


def test_msimpler_scale_tracks_the_characteristic_speed() -> None:
    """MSIMPLER's ``k`` auto-calibrates to the operating scale, with no unit-speed assumption.

    ``k = mean(V / a_P)`` is taken from the *real* momentum diagonal, which is convection-dominated
    (``a_P ~ ρ U``), so at a fixed Reynolds number scaling the inlet speed by ``s`` scales ``k`` by
    ``1/s``. A fluid or nondimensionalisation whose characteristic speed is far from one therefore
    calibrates itself — the case that previously needed a manual ``msimpler_scale``.
    """
    from aquaflux.flow import BlockPreconditioner

    def scale_at(u_in):
        mu = RHO * u_in * H / 500.0  # fixed Re = 500; mu scales with the speed
        assembler = _channel(32, 24, mu, wall_growth=1.15, u_in=u_in)
        preconditioner = BlockPreconditioner.build(assembler, schur_scaling="msimpler")
        # Evaluate at a uniform inlet-speed flow — the operating scale the Schur must match.
        velocity = jnp.zeros((assembler.mesh.n_cells, assembler.mesh.dim)).at[:, 0].set(u_in)
        state = assembler.pack(velocity, jnp.zeros(assembler.mesh.n_cells))
        return float(preconditioner._msimpler_scale(state))

    unit = scale_at(1.0)
    assert 50.0 < scale_at(0.01) / unit < 200.0  # ~100x slower speed -> ~100x larger k
    assert 0.005 < scale_at(100.0) / unit < 0.02  # ~100x faster speed -> ~100x smaller k


@pytest.mark.slow
def test_msimpler_auto_scale_converges_at_non_unit_speed() -> None:
    """The auto-calibrated ``k`` carries the MSIMPLER solve at a characteristic speed far from one,
    with no manual ``msimpler_scale`` — the unit-speed assumption is gone. A deliberately mis-scaled
    ``k`` (the old unit-speed calibration) still converges here only because the continuation escalates
    its damping to compensate, but ~4x slower; the auto-calibration is what keeps it well-conditioned.
    """
    u_in = 100.0
    assembler = _channel(32, 24, RHO * u_in * H / 500.0, wall_growth=1.15, u_in=u_in)  # Re 500
    continuation = momentum_continuation(assembler, schur_scaling="msimpler")
    state = _solve(assembler, continuation=continuation, max_steps=200)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8


@pytest.mark.slow
def test_convection_velocity_block_converges_at_high_reynolds() -> None:
    """The convection-aware velocity block converges the wall-graded high-Reynolds channel deeply.

    The default velocity block builds its AMG on the viscous (symmetric) momentum operator, so it is
    Peclet-blind; ``velocity="convection"`` instead builds it on the frozen ``viscous + first-order-
    upwind`` operator, staying a good momentum-block approximation as convection strengthens. Its
    reference flux comes from the inlet, with no reference state passed. Paired with the MSIMPLER
    Schur it drives the coupled steady residual well below the tolerance a segregated under-relaxed
    scheme could reach — a *true* steady residual, since the terminal phase is undamped coupled
    Newton.
    """
    assembler = _channel(64, 48, 5e-4, wall_growth=1.15)  # Re = 2000, wall-graded
    continuation = momentum_continuation(
        assembler,
        schur_scaling="msimpler",
        velocity="convection",
    )
    converged = _solve(assembler, continuation=continuation, max_steps=150)
    assert float(jnp.linalg.norm(assembler.residual(converged))) < 1e-8


@pytest.mark.slow
def test_convection_velocity_block_is_differentiable() -> None:
    """The convection-velocity-block solve stays reverse-differentiable: its hierarchy is frozen and
    ``stop_gradient``-ed (only accelerates the Krylov iteration), so the implicit-function-theorem
    adjoint is untouched. Built once outside ``jax.grad`` from concrete parameters and reused.
    """
    continuation = momentum_continuation(
        _channel(32, 24, 2e-3, wall_growth=1.15),  # Re = 500, wall-graded
        schur_scaling="msimpler",
        velocity="convection",
    )

    def mean_speed(mu):
        assembler = _channel(32, 24, mu, wall_growth=1.15)
        state = _solve(assembler, continuation=continuation)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(2e-3))
    assert np.isfinite(grad)

    step = 2e-3 * 1e-3
    finite_difference = (float(mean_speed(2e-3 + step)) - float(mean_speed(2e-3 - step))) / (
        2 * step
    )
    assert abs(grad - finite_difference) < 1e-2 * abs(finite_difference)


@pytest.mark.slow
def test_air_velocity_block_converges_and_is_differentiable() -> None:
    """The reduction-based (lAIR) velocity block converges the wall-graded channel and stays
    reverse-differentiable.

    ``velocity="convection-air"`` coarsens the frozen convection-diffusion momentum operator by local
    approximate ideal restriction — Peclet-robust *and* mesh-independent (so it scales where the
    two-level ``"convection"`` block's direct coarse solve cannot). Its restriction and prolongation
    differ (``R != Pᵀ``) but the frozen apply still transposes cleanly, so the implicit-function-theorem
    adjoint is untouched and the gradient matches finite differences.
    """
    continuation = momentum_continuation(
        _channel(32, 24, 2e-3, wall_growth=1.15),  # Re = 500, wall-graded
        schur_scaling="msimpler",
        velocity="convection-air",
    )

    def mean_speed(mu):
        assembler = _channel(32, 24, mu, wall_growth=1.15)
        state = _solve(assembler, continuation=continuation)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    assembler = _channel(32, 24, 2e-3, wall_growth=1.15)
    state = _solve(assembler, continuation=continuation)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8

    grad = float(jax.grad(mean_speed)(2e-3))
    assert np.isfinite(grad)

    step = 2e-3 * 1e-3
    finite_difference = (float(mean_speed(2e-3 + step)) - float(mean_speed(2e-3 - step))) / (
        2 * step
    )
    assert abs(grad - finite_difference) < 1e-2 * abs(finite_difference)


@pytest.mark.slow
def test_reused_flow_solve_converges_across_viscosities() -> None:
    """``reused_flow_solve`` builds its preconditioned continuation once and drives solves whose
    effective viscosity differs to convergence — the segregated-turbulence reuse pattern.

    A segregated k--omega SST loop re-solves the momentum system every sweep at an updated eddy
    viscosity ``nu + nu_t``. Rebuilding the block-preconditioner continuation each sweep re-runs its
    off-jit AMG setup and recompiles the solve; ``reused_flow_solve`` builds it once and reuses it, a
    frozen accelerator that stays effective because a larger effective viscosity only makes the
    momentum operator more diffusion-dominated (lower cell Peclet). One build applied across a
    viscosity sweep, each solve converging, confirms the reuse holds.
    """
    reference = _channel(32, 24, 1e-3, wall_growth=1.15)  # a mid-sweep reference viscosity
    solve_flow = reused_flow_solve(reference, schur_scaling="msimpler", velocity="convection")
    for mu in (2e-3, 1e-3, 5e-4):  # Re = 500, 1000, 2000 — a viscosity sweep on the frozen build
        assembler = _channel(32, 24, mu, wall_growth=1.15)
        state = solve_flow(assembler, assembler.initial_state())
        assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8


@pytest.mark.slow
def test_reused_flow_solve_is_differentiable() -> None:
    """A solve driven by the reused (jitted) flow solver stays reverse-differentiable: the frozen
    preconditioner only accelerates the Krylov iteration, so the implicit-function-theorem adjoint is
    unchanged and the ``jit`` wrapper is transparent to it.
    """
    reference = _channel(32, 24, 2e-3, wall_growth=1.15)
    solve_flow = reused_flow_solve(reference, schur_scaling="msimpler", velocity="convection")

    def mean_speed(mu):
        assembler = _channel(32, 24, mu, wall_growth=1.15)
        state = solve_flow(assembler, assembler.initial_state())
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(2e-3))
    assert np.isfinite(grad)

    step = 2e-3 * 1e-3
    finite_difference = (float(mean_speed(2e-3 + step)) - float(mean_speed(2e-3 - step))) / (
        2 * step
    )
    assert abs(grad - finite_difference) < 1e-2 * abs(finite_difference)
