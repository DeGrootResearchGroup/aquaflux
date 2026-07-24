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
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import CycleGrowthTrigger, PseudoTransientStep, ShiftTerm
from aquaflux.turbulence import DirectScalars, LogScalars, SSTModel, SSTTurbulence
from aquaflux.turbulence.coupled import (
    CoupledRANS,
    CoupledRANSLayout,
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


def test_log_omega_reparametrization_preserves_the_residual() -> None:
    """omega-log reparametrizes the Newton *unknown*, not the physics.

    The coupled residual at the log-mapped state equals the direct residual at the same physical
    fields (so the two forms share a root), and it stays differentiable through the ``e^w`` map.
    """
    mesh, direct = _cavity()
    log_omega = CoupledRANS.build(direct.momentum, direct.turbulence, omega_transform=LogScalars())
    physical = _healthy_state(mesh, direct)
    flow, k, omega = direct.layout.unpack(physical)
    solved = log_omega.state_from_physical(flow, k, omega)

    assert jnp.allclose(direct.residual(physical), log_omega.residual(solved), atol=1e-10)
    direction = jax.random.normal(jax.random.PRNGKey(4), (solved.shape[0],))
    jvp = jax.jvp(log_omega.residual, (solved,), (direction,))[1]
    assert bool(jnp.all(jnp.isfinite(jvp)))


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


def test_refreshing_the_policy_carries_the_shift_diagonals() -> None:
    """A ``reuse=`` refresh must re-derive only the AMGs and carry the shift diagonals unchanged.

    Rebuilding the shift diagonals at a developed state over-damps the pseudo-transient step and freezes
    the coupled log-omega march (the shift is ``transport_diagonal * jacobian_scale``, and under
    ``LogScalars`` ``jacobian_scale(omega) = omega`` grows with the field). So the refreshed policy's
    shift diagonals must be *identical* to the reused ones -- pinned here at the mechanism, without
    needing a full separating march to exhibit the freeze. The flow block is carried too; the scalar AMG
    refresh itself is pinned in ``test_scalar_transport_preconditioner``. ``method=None`` and the
    symmetric viscous velocity block keep the policy build robust to the two synthetic states.
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

    # The shift diagonals are carried from `base` verbatim ...
    assert jnp.array_equal(refreshed.k_shift_diagonal, base.k_shift_diagonal)
    assert jnp.array_equal(refreshed.omega_shift_diagonal, base.omega_shift_diagonal)
    # ... and rebuilding at the developed state genuinely WOULD have changed them (so the carry matters).
    assert not jnp.allclose(rebuilt.k_shift_diagonal, base.k_shift_diagonal)
    assert not jnp.allclose(rebuilt.omega_shift_diagonal, base.omega_shift_diagonal)
    # The flow block is carried over (the expensive half; measured no help to re-freeze).
    assert refreshed.flow_preconditioner is base.flow_preconditioner
