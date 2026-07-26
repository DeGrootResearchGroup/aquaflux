"""Unit: the monolithic coupled RANS residual -- layout, jit-safety, and Jacobian correctness.

Fast checks that do not run a full coupled solve: the state layout in isolation, that the residual
assembles under jit (the regression guard for the boundary-resolve fix), and that its automatic
Jacobian matches finite differences on a healthy (well-positive) state. The full coupled Newton
convergence, its agreement with the segregated loop, and the coupled adjoint are the slow integration
tests (:mod:`tests.integration.test_coupled_rans`).
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import DifferenceRow, FirstOrderUpwind, LogRatioRow
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import CycleGrowthTrigger, PseudoTransientStep, ShiftTerm
from aquaflux.turbulence import (
    DirectScalars,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    eddy_viscosity_drift,
)
from aquaflux.turbulence.coupled import (
    CoupledRANS,
    CoupledRANSLayout,
    LiveViscosityVelocityParts,
    _row_jacobian_scale,
    coupled_continuation,
    solve_coupled,
)

RHO, NU, U_LID = 1.0, 1e-2, 1.0
WALLS = ("top", "bottom", "left", "right")


def test_layout_round_trips_and_sizes() -> None:
    layout = CoupledRANSLayout(dim=2, n_cells=5)
    assert layout.flow_size == (2 + 1) * 5
    assert layout.size == (2 + 3) * 5
    flow = jnp.arange(layout.flow_size, dtype=float)
    k = 10.0 + jnp.arange(5, dtype=float)
    omega = 100.0 + jnp.arange(5, dtype=float)
    state = layout.pack(flow, k, omega)
    assert state.shape == (layout.size,)
    f, kk, oo = layout.unpack(state)
    assert jnp.array_equal(f, flow)
    assert jnp.array_equal(kk, k)
    assert jnp.array_equal(oo, omega)


def _cavity(n=6):
    mesh = structured_grid_2d(n, n, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=list(WALLS),
        k_boundary=BoundaryConditions({w: Dirichlet(0.0) for w in WALLS}),
        omega_boundary=BoundaryConditions({w: ZeroGradient() for w in WALLS}),
    )
    return mesh, CoupledRANS.build(momentum, turbulence)


def _healthy_state(mesh, coupled, seed=0):
    """A well-positive coupled state: modest random flow, k ~ 0.05, omega ~ 10 (floor inactive)."""
    n = mesh.n_cells
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    velocity = 0.1 * jax.random.normal(keys[0], (n, mesh.dim))
    pressure = 0.1 * jax.random.normal(keys[1], (n,))
    flow = coupled.momentum.pack(velocity, pressure)
    k = 0.05 + 0.01 * jax.random.uniform(keys[2], (n,))
    omega = 10.0 + jax.random.uniform(keys[3], (n,))
    return coupled.pack_state(flow, k, omega)


def test_coupled_build_resolves_boundaries_so_the_residual_jits() -> None:
    # Regression: the turbulence residual rebuilds its assembler each call; without the pre-resolved
    # boundaries (CoupledRANS.build) that rebuild re-runs a dynamic-shape nonzero on the mesh labels
    # and a jitted residual raises ConcretizationTypeError. jit + eval must succeed and stay finite.
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    residual = eqx.filter_jit(coupled.residual)(state)
    assert residual.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(residual)))


def test_residual_jacobian_matches_finite_difference() -> None:
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    direction = jax.random.normal(jax.random.PRNGKey(3), (state.shape[0],))
    direction = direction / jnp.linalg.norm(direction)
    jvp = jax.jvp(coupled.residual, (state,), (direction,))[1]
    assert bool(jnp.all(jnp.isfinite(jvp)))
    eps = 1e-5
    fd = (coupled.residual(state + eps * direction) - coupled.residual(state - eps * direction)) / (
        2 * eps
    )
    rel = float(jnp.linalg.norm(fd - jvp) / jnp.linalg.norm(jvp))
    assert rel < 1e-6


def test_scalar_variable_transforms() -> None:
    """DirectScalars is the identity; LogScalars is ``e^w`` with derivative ``e^w`` (physics-free)."""
    w = jnp.array([-3.0, 0.0, 2.5])
    direct = DirectScalars()
    assert jnp.array_equal(direct.to_physical(w), w)
    assert jnp.array_equal(direct.to_solved(w), w)
    assert jnp.array_equal(direct.jacobian_scale(w), jnp.ones_like(w))

    log_scalars = LogScalars()
    phi = log_scalars.to_physical(w)
    assert jnp.allclose(phi, jnp.exp(w))
    assert bool(jnp.all(phi > 0.0))  # positive for any real w -- the structural guarantee
    assert jnp.allclose(log_scalars.to_solved(phi), w)  # round trip
    assert jnp.allclose(log_scalars.jacobian_scale(phi), phi)  # d(e^w)/dw = e^w = phi


def test_log_omega_reparametrization_preserves_the_transport_residual() -> None:
    """omega-log reparametrizes the Newton *unknown*, not the physics.

    Every **transport** row of the coupled residual at the log-mapped state equals the direct
    residual at the same physical fields, and it stays differentiable through the ``e^w`` map.

    The near-wall **fixation** rows are deliberately *not* identical: each transform writes the
    fixation in its own solved variable (``omega - omega_wall`` directly, ``log(omega/omega_wall)``
    under the log map), which is what keeps that row linear in the unknown actually being stepped.
    Both vanish on exactly the same set, so the two forms still share a root -- which the companion
    test pins.
    """
    mesh, direct = _cavity()
    log_omega = CoupledRANS.build(direct.momentum, direct.turbulence, omega_transform=LogScalars())
    physical = _healthy_state(mesh, direct)
    flow, k, omega = direct.layout.unpack(physical)
    solved = log_omega.state_from_physical(flow, k, omega)

    _, _, direct_omega = direct.layout.unpack(direct.residual(physical))
    _, _, log_omega_rows = log_omega.layout.unpack(log_omega.residual(solved))
    interior = jnp.setdiff1d(jnp.arange(direct_omega.shape[0]), direct.turbulence.wall_cells)
    assert jnp.allclose(direct_omega[interior], log_omega_rows[interior], atol=1e-10)

    direction = jax.random.normal(jax.random.PRNGKey(4), (solved.shape[0],))
    jvp = jax.jvp(log_omega.residual, (solved,), (direction,))[1]
    assert bool(jnp.all(jnp.isfinite(jvp)))


def test_the_two_fixation_row_forms_share_a_root_and_the_log_form_is_linear() -> None:
    """The transform-matched fixation rows vanish together, and the log form is linear in ``w``.

    Linearity is the point: the difference row's Newton correction in the log variable is
    ``target/phi - 1`` (the linearization of an exponential, which overshoots by ``e**(r-1)`` at a
    target ratio ``r``), while the log-ratio row's is ``log(target/phi)`` -- exact at any ratio, so a
    full step lands on the constraint however far off it starts.
    """
    phi = jnp.array([1.0, 5.0, 1.0e4])
    target = jnp.array([1.0, 5.0, 5.0])  # first two already satisfied, the third far off
    difference = DifferenceRow().row(phi, target)
    log_ratio = LogRatioRow().row(phi, target)
    # Same root: both vanish exactly where phi == target, and nowhere else.
    assert jnp.array_equal(difference == 0.0, log_ratio == 0.0)
    assert bool(jnp.all(difference[:2] == 0.0)) and bool(jnp.all(log_ratio[:2] == 0.0))

    # The log row is exactly linear in w = log(phi): its derivative is 1 regardless of the ratio.
    slope = jax.grad(lambda w: jnp.sum(LogRatioRow().row(jnp.exp(w), target)))(jnp.log(phi))
    assert jnp.allclose(slope, jnp.ones_like(slope))


def _count_rhie_chow_assemblies(monkeypatch):
    """A mutable ``[count]`` incremented on each lagged-``a_P`` Rhie--Chow assembly (see the seam)."""
    calls = [0]
    original = MomentumContinuity.momentum_matrix_diagonal

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MomentumContinuity, "momentum_matrix_diagonal", counted)
    return calls


def test_residual_assembles_the_flow_fields_once(monkeypatch) -> None:
    """The coupled residual re-derives the Rhie--Chow flow fields a single time per evaluation.

    The residual, the mass flux the scalars advect on, and the velocity gradient the closure reads all
    come from one :meth:`~aquaflux.flow.MomentumContinuity.flow_fields` assembly (the gradient is the
    lightweight one that does no ``a_P`` work), so the expensive lagged-``a_P`` Rhie--Chow assembly runs
    exactly once -- not once each for the residual, the mass flux, and the gradient.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    calls = _count_rhie_chow_assemblies(monkeypatch)
    calls[0] = 0
    coupled.residual(state)
    assert calls[0] == 1


def test_segregated_prologues_match_the_eager_assembly() -> None:
    """The jitted sweep prologues equal the eager accessor expressions they replace.

    ``_sweep_eddy_viscosity`` is the pre-solve ``nu_t`` from the velocity gradient; ``_sweep_closure``
    is the post-solve ``(mdot, closure)`` from a single flow-field assembly. Jitting and fusing them
    must not change the numbers (the driver's per-sweep assembly savings come for free). That the
    fused path assembles the Rhie--Chow flow fields only once is pinned by the eager
    ``test_residual_assembles_the_flow_fields_once`` and the momentum seam tests.
    """
    from aquaflux.turbulence.driver import _sweep_closure, _sweep_eddy_viscosity

    mesh, coupled = _cavity()
    momentum, turbulence = coupled.momentum, coupled.turbulence
    flow, k, omega = coupled.layout.unpack(_healthy_state(mesh, coupled))

    nu_t = _sweep_eddy_viscosity(momentum, turbulence, flow, k, omega)
    expected_nu_t = turbulence.eddy_viscosity(momentum.velocity_fields(flow).gradient, k, omega)
    assert jnp.allclose(nu_t, expected_nu_t)

    mdot, closure = _sweep_closure(momentum, turbulence, flow, k, omega)
    assert jnp.allclose(mdot, momentum.mass_flux(flow))
    expected_closure = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega)
    assert jnp.allclose(closure.nu_t, expected_closure.nu_t)
    assert jnp.allclose(closure.strain_rate, expected_closure.strain_rate)


def test_layout_matches_the_assembler_dimensions() -> None:
    mesh, coupled = _cavity()
    assert coupled.layout.dim == mesh.dim
    assert coupled.layout.n_cells == mesh.n_cells
    assert coupled.pack_state(
        coupled.momentum.initial_state(),
        jnp.ones(mesh.n_cells),
        jnp.ones(mesh.n_cells),
    ).shape == (coupled.layout.size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_refresh_trigger_is_rejected_under_differentiation() -> None:
    """``refresh_trigger`` is a forward-only accelerator: it raises under ``jax.grad``, not leaks.

    The refresh re-derives the preconditioner from the mid-march state, which is a tracer when
    differentiating; the refreshed preconditioner would then capture that tracer and escape the
    converged solve's ``custom_vjp`` as an opaque ``UnexpectedTracerError``. A refresh also forbids an
    explicit (concrete) ``continuation``, so there is no way to build the preconditioner outside the
    trace -- the only honest behaviour is a clear up-front error. The guard fires before any solve, so
    this stays a fast test. Differentiating the single-stage solve (no trigger) remains the supported
    path and is exercised by the integration adjoint gate.
    """
    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        f, _, _ = solve_coupled(
            scaled, flow, k, omega, rtol=1e-2, refresh_trigger=CycleGrowthTrigger()
        )
        return jnp.sum(f**2)

    with pytest.raises(ValueError, match="forward-only eager march"):
        jax.grad(objective)(1.0)


class _TrivialShiftPolicy(eqx.Module):
    """A shift policy with a unit diagonal and no preconditioner -- enough to build a step object."""

    def shift_term(self, phi):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _relaxation: None)


def test_refresh_trigger_with_an_explicit_continuation_is_rejected() -> None:
    """A refresh must rebuild the continuation, so it cannot accept a pre-built one.

    The error names the two supported alternatives rather than silently ignoring either argument.
    The guard is on the argument combination and fires before the continuation is ever stepped, so a
    trivial step object is sufficient here -- no preconditioner needs to be built.
    """
    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    with pytest.raises(ValueError, match="needs solve_coupled to build the continuation"):
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            rtol=1e-2,
            continuation=PseudoTransientStep(_TrivialShiftPolicy()),
            refresh_trigger=CycleGrowthTrigger(),
        )


def test_refreshing_the_policy_rebuilds_transport_and_carries_the_coordinate_factor() -> None:
    """A ``reuse=`` refresh rebuilds the shift's transport time scale and carries its coordinate factor.

    The shift diagonal is ``transport_diagonal * jacobian_scale``. Rebuilding the *product* at a
    developed state over-damps and freezes the coupled log-omega march, because under ``LogScalars``
    ``jacobian_scale(omega) = omega`` drags the field's growing range into the shift. Storing the two
    factors separately (issue #156) lets a refresh rebuild the transport time scale -- physics that
    should track the flow -- while carrying the coordinate factor frozen, so the temporal ratio
    ``transport(state)/transport(reference)`` has that range cancel. Pinned here at the mechanism,
    without a full separating march. The flow block is carried too; the scalar AMG refresh itself is
    pinned in ``test_scalar_transport_preconditioner``. ``method=None`` and the symmetric viscous
    velocity block keep the policy build robust to the two synthetic states.
    """
    from aquaflux.turbulence.coupled import _coupled_shift_policy

    mesh, base_coupled = _cavity()
    coupled = CoupledRANS.build(
        base_coupled.momentum, base_coupled.turbulence, omega_transform=LogScalars()
    )
    cold = _healthy_state(mesh, coupled, seed=0)
    developed = _healthy_state(mesh, coupled, seed=1)  # a *different*, more-developed reference

    kw = dict(velocity="smoothed")
    base = _coupled_shift_policy(coupled, cold, None, **kw)
    refreshed = _coupled_shift_policy(coupled, developed, None, base, **kw)
    rebuilt = _coupled_shift_policy(coupled, developed, None, **kw)

    # The coordinate factor (jacobian_scale) is carried from `base` frozen ...
    assert jnp.array_equal(refreshed.k_jacobian_scale, base.k_jacobian_scale)
    assert jnp.array_equal(refreshed.omega_jacobian_scale, base.omega_jacobian_scale)
    # ... while the transport time scale is rebuilt at the developed state (== a from-scratch build
    # there, and genuinely different from the cold-reference one, so the refresh actually tracks it).
    assert jnp.array_equal(refreshed.k_shift_transport, rebuilt.k_shift_transport)
    assert jnp.array_equal(refreshed.omega_shift_transport, rebuilt.omega_shift_transport)
    assert not jnp.allclose(refreshed.k_shift_transport, base.k_shift_transport)
    assert not jnp.allclose(refreshed.omega_shift_transport, base.omega_shift_transport)
    # The flow block is carried over (the expensive half; measured no help to re-freeze).
    assert refreshed.flow_preconditioner is base.flow_preconditioner


def test_refresh_carries_the_block_scaled_progress_norm_fixed_at_the_initial_state() -> None:
    """A refresh reuses the initial residual measure, so block-scaled progress does not re-base (#156 s4).

    ``BlockScaledNorm`` is self-normalising: at the state its per-block scales were built at it returns
    ``sqrt(n_blocks)``. If a refresh rebuilt it at the developed state, every ``residual_ratio`` would
    jump back toward one and the convergence test become unreachable. ``coupled_continuation`` with an
    explicit ``residual_norm`` (what ``solve_coupled`` passes on every refresh) uses it verbatim instead
    of rebuilding, so the measure stays fixed at the state the global progress reference was measured at.
    """
    mesh, coupled = _cavity()
    cold = _healthy_state(mesh, coupled, seed=0)
    developed = _healthy_state(mesh, coupled, seed=1)
    kw = dict(method=None, block_scaled_norm=True, velocity="smoothed")

    base = coupled_continuation(coupled, cold, **kw)
    base_norm = base.norm()
    refreshed = coupled_continuation(
        coupled, developed, reuse=base.shift_policy, residual_norm=base_norm, **kw
    )
    # The refreshed continuation measures progress with the *same* norm object, not a re-based one.
    assert refreshed.norm() is base_norm
    # And that carry matters: a from-scratch rebuild at the developed state re-bases the per-block
    # scales, so it scores the same residual differently (it self-normalises to sqrt(n_blocks) there).
    rebuilt = coupled_continuation(coupled, developed, **kw)
    r_dev = coupled.residual(developed)
    assert not jnp.allclose(base_norm(r_dev), rebuilt.norm()(r_dev))


def test_fixation_rows_take_their_own_derivative_not_the_chain_factor() -> None:
    """The per-row Jacobian scale is ``phi`` on transport rows but **one** on the fixation rows.

    Regression test. The scalar block's frozen preconditioner is built for the physical operator and
    rescaled by ``1 / (d(row)/d(w))``; the frozen operator carries a unit identity row at each fixed
    cell. Rescaling those rows by the block-wide chain factor ``d(phi)/d(w) = phi`` instead of the
    fixation row's own unit derivative leaves the preconditioned operator with a cluster of ``1/phi``
    eigenvalues -- measured at ~1e-5 on a wall-resolved mesh -- which stalls the Krylov solve.
    """
    omega = jnp.array([10.0, 1.0e5, 3.0, 250.0])
    fixed = jnp.array([1, 3])
    transport = jnp.array([0, 2])

    scale = _row_jacobian_scale(LogScalars(), omega, fixed)

    assert jnp.allclose(scale[transport], omega[transport])
    assert jnp.allclose(scale[fixed], 1.0)


def test_row_jacobian_scale_is_all_ones_for_a_directly_solved_scalar() -> None:
    """The directly-solved path keeps a unit scale with or without fixed cells, so it is unchanged."""
    omega = jnp.array([10.0, 1.0e5, 3.0, 250.0])
    assert jnp.allclose(_row_jacobian_scale(DirectScalars(), omega), 1.0)
    assert jnp.allclose(_row_jacobian_scale(DirectScalars(), omega, jnp.array([1, 3])), 1.0)


def test_eddy_viscosity_drift_is_zero_at_its_reference_and_grows_away_from_it() -> None:
    """The staleness measure a drift trigger fires on: relative movement of ``nu_t``.

    Zero at the state the preconditioner was frozen at (nothing has gone stale yet) and positive
    once the turbulence field moves, which is what makes it a direct staleness signal rather than an
    inference from solver cost.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    drift = eddy_viscosity_drift(coupled, state)

    assert float(drift(state)) == pytest.approx(0.0, abs=1e-12)

    # Raise k, which raises nu_t = k / omega: the frozen scalar operators no longer describe this
    # state, and the measure must say so.
    flow, k, omega = coupled.physical_fields(state)
    moved = coupled.state_from_physical(flow, 1.5 * k, omega)
    assert float(drift(moved)) > 0.1


def test_eddy_viscosity_drift_matches_its_definition() -> None:
    """The measure is exactly the relative L2 movement of ``nu_t`` -- pinned against a direct compute.

    Stated as the definition rather than as an invariance: ``nu_t`` is **not** proportional to ``k``
    once the shear limiter ``a1 k / max(a1 omega, S F2)`` engages, so properties that assume
    homogeneity in ``k`` do not hold, and asserting one would be testing the closure rather than the
    measure.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)
    moved = coupled.state_from_physical(flow, 1.3 * k, 0.8 * omega)

    reference_nu_t = coupled.eddy_viscosity(state)
    expected = float(
        jnp.linalg.norm(coupled.eddy_viscosity(moved) - reference_nu_t)
        / jnp.linalg.norm(reference_nu_t)
    )
    assert float(eddy_viscosity_drift(coupled, state)(moved)) == pytest.approx(expected, rel=1e-10)
    assert expected > 0.0


def test_live_velocity_shift_parts_use_the_current_eddy_viscosity() -> None:
    """The injected live source reproduces the momentum diagonal at the state's own ``nu_t``.

    The velocity shift's buckets are a *local time scale* and must describe the operator being solved,
    so they have to see the current effective viscosity rather than one frozen at a reference state.
    Pinned by composing the closure by hand and comparing.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)

    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
    live_assembler = coupled.momentum.with_eddy_viscosity(
        closure.nu_t, coupled.turbulence.wall_face_eddy_viscosity(k)
    )
    velocity, _pressure = live_assembler.unpack(flow)
    expected = live_assembler.momentum_matrix_diagonal_parts(velocity)

    source = LiveViscosityVelocityParts(
        coupled.momentum, coupled.turbulence, coupled.k_transform, coupled.omega_transform
    )
    flow_block, k_solved, omega_solved = coupled.layout.unpack(state)
    got = source.parts(flow_block, k_solved, omega_solved)

    for mine, theirs in zip(got, expected, strict=True):
        assert jnp.allclose(mine, theirs)


def test_live_velocity_shift_parts_map_the_solved_unknown_back_to_physical() -> None:
    """The buckets are the same whether omega is solved directly or in log form.

    The source receives the block **as solved**, so under a log parametrization it must exponentiate
    before forming the closure. Without that it would build the shift from ``log(omega)`` as if it were
    ``omega`` -- a silent, badly wrong local time scale.
    """
    mesh, direct = _cavity()
    log_omega = CoupledRANS.build(direct.momentum, direct.turbulence, omega_transform=LogScalars())
    physical = _healthy_state(mesh, direct)
    flow, k, omega = direct.layout.unpack(physical)

    direct_parts = LiveViscosityVelocityParts(
        direct.momentum, direct.turbulence, direct.k_transform, direct.omega_transform
    ).parts(flow, k, omega)

    solved = log_omega.state_from_physical(flow, k, omega)
    flow_l, k_l, omega_l = log_omega.layout.unpack(solved)
    log_parts = LiveViscosityVelocityParts(
        log_omega.momentum, log_omega.turbulence, log_omega.k_transform, log_omega.omega_transform
    ).parts(flow_l, k_l, omega_l)

    for a, b in zip(direct_parts, log_parts, strict=True):
        assert jnp.allclose(a, b)
