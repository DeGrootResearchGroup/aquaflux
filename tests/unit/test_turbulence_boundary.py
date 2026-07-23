"""Unit tests for the k-omega SST boundary values (pure formulas, no mesh, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.turbulence import (
    SSTModel,
    inlet_k,
    inlet_omega,
    k_wall_production,
    log_layer_shear_rate,
    nut_wall,
    omega_wall,
    omega_wall_value,
    wall_function_weight,
    wall_k_diffusivity,
    wall_shear_stress,
)

MODEL = SSTModel()


def _omega_log(k, d, model):
    """The log-layer near-wall omega branch, computed independently of the implementation."""
    return jnp.sqrt(k) / (model.beta_star**0.25 * model.kappa * d)


def _y_star(k, d, nu, model):
    """The k-based wall coordinate, computed independently of the implementation."""
    return model.beta_star**0.25 * jnp.sqrt(k) * d / nu


def test_omega_wall_value_solves_the_viscous_sublayer_balance() -> None:
    """The wall omega must *satisfy* the sublayer equation at the distance it is imposed at.

    As ``k -> 0`` the omega equation reduces to viscous diffusion against destruction,
    ``nu d2(omega)/dy2 = beta_1 omega**2``. Checking the residual of that balance -- rather than
    restating the closed-form coefficient -- is what distinguishes the analytical cell-centre value
    from the ten-times-larger wall-face surrogate, which leaves a residual of order the equation
    itself.
    """
    nu, d = 1e-5, jnp.logspace(-5.0, -3.0, 7)
    omega = omega_wall_value(jnp.full_like(d, nu), d, MODEL)
    # omega = A / d**2  =>  d2(omega)/dy2 = 6 A / d**4, evaluated from the returned field itself.
    a = omega * d**2
    residual = nu * 6.0 * a / d**4 - MODEL.beta_1 * omega**2
    assert jnp.max(jnp.abs(residual)) < 1e-8 * jnp.max(MODEL.beta_1 * omega**2)


def test_omega_wall_value_scales_with_viscosity_and_inverse_square_distance() -> None:
    nu, d = jnp.array([1e-5]), jnp.array([1e-3])
    base = omega_wall_value(nu, d, MODEL)
    assert jnp.allclose(omega_wall_value(2.0 * nu, d, MODEL), 2.0 * base)
    assert jnp.allclose(omega_wall_value(nu, 2.0 * d, MODEL), base / 4.0)


def test_omega_wall_reduces_to_the_viscous_value_as_k_vanishes() -> None:
    """With no turbulent energy the adaptive wall omega is exactly the sublayer value.

    This is the wall-resolved (``y+ -> 0``) limit: the log branch carries ``sqrt(k)``, so at ``k = 0``
    it drops out and ``omega_wall`` must equal :func:`omega_wall_value` bit-for-bit -- the property
    that makes the adaptive treatment a strict generalization of the wall-resolved one.
    """
    nu, d = jnp.full((5,), 1e-5), jnp.logspace(-5.0, -3.0, 5)
    assert jnp.array_equal(
        omega_wall(nu, d, jnp.zeros_like(d), MODEL), omega_wall_value(nu, d, MODEL)
    )


def test_omega_wall_is_the_quadrature_blend_of_the_two_branches() -> None:
    """omega_wall**2 == omega_vis**2 + omega_log**2 (the Menter automatic blend)."""
    nu, d, k = jnp.array([1e-5]), jnp.array([1e-3]), jnp.array([0.2])
    expected = jnp.sqrt(omega_wall_value(nu, d, MODEL) ** 2 + _omega_log(k, d, MODEL) ** 2)
    assert jnp.allclose(omega_wall(nu, d, k, MODEL), expected)


def test_omega_wall_recovers_the_log_branch_when_it_dominates() -> None:
    """Out in the log layer (large ``d``, finite ``k``) the viscous branch is negligible.

    ``omega_vis ~ 1/d**2`` decays faster than ``omega_log ~ 1/d``, so at a wall-function-scale first
    cell the blend is the log-layer value -- the value that raises the near-wall omega above the (too
    low) sublayer fixation on a ``y+ ~ 30`` mesh.
    """
    nu, d, k = jnp.array([1e-5]), jnp.array([2e-3]), jnp.array([0.3])
    omega_vis = omega_wall_value(nu, d, MODEL)
    omega_log = _omega_log(k, d, MODEL)
    assert bool(omega_log[0] > 5.0 * omega_vis[0])  # log branch genuinely dominates here
    assert jnp.allclose(omega_wall(nu, d, k, MODEL), omega_log, rtol=0.03)


def test_omega_wall_has_a_finite_k_derivative_at_zero() -> None:
    """The blend is differentiable in ``k`` at ``k = 0`` -- no ``sqrt(k)`` cone point.

    Writing the log branch as ``k`` (not ``sqrt(k)``) under a radicand kept positive by the viscous
    branch is what keeps ``d(omega_wall)/dk`` finite at ``k = 0``; a naive ``sqrt(k)`` would give an
    infinite (NaN) derivative there and poison the coupled Jacobian at a wall-adjacent cell whose
    ``k`` approaches zero.
    """
    nu, d = jnp.array([1e-5]), jnp.array([1e-3])
    grad_k = jax.grad(lambda k: jnp.sum(omega_wall(nu, d, k, MODEL)))(jnp.array([0.0]))
    assert bool(jnp.all(jnp.isfinite(grad_k)))
    # A negative k is off-solution and clamped out of the log term, so it has zero sensitivity there.
    grad_neg = jax.grad(lambda k: jnp.sum(omega_wall(nu, d, k, MODEL)))(jnp.array([-0.1]))
    assert jnp.allclose(grad_neg, 0.0)


def test_wall_y_star_lam_solves_the_laminar_log_crossover() -> None:
    """y*_lam is the fixed point of y = ln(E y)/kappa (~11 for the standard constants)."""
    y = MODEL.wall_y_star_lam
    assert jnp.allclose(y, jnp.log(MODEL.e_wall * y) / MODEL.kappa)
    assert 10.0 < float(y) < 13.0


def test_nut_wall_is_zero_in_the_viscous_sublayer() -> None:
    """A wall-resolved first cell (small ``y*``) gets no wall-function eddy viscosity.

    Below the laminar/log crossover the wall is resolved, so ``nu_t,wall = 0`` and the momentum wall
    shear is the plain molecular ``mu |U|/d`` -- the treatment reduces to a no-slip wall exactly.
    """
    nu = 1e-5
    # Pick k, d so y* is well inside the sublayer (< y*_lam ~ 11).
    d, k = jnp.array([1e-5]), jnp.array([0.01])
    assert float(_y_star(k, d, nu, MODEL)[0]) < MODEL.wall_y_star_lam
    assert jnp.array_equal(nut_wall(jnp.full_like(d, nu), d, k, MODEL), jnp.zeros_like(d))


def test_nut_wall_matches_the_log_law_value_out_in_the_log_layer() -> None:
    """In the log layer nu_t,wall = nu(y* kappa/ln(E y*) - 1), and it is strictly positive."""
    nu = 1e-5
    d, k = jnp.array([3e-3]), jnp.array([0.3])
    y = _y_star(k, d, nu, MODEL)
    assert float(y[0]) > MODEL.wall_y_star_lam  # genuinely in the log layer
    expected = nu * (y * MODEL.kappa / jnp.log(MODEL.e_wall * y) - 1.0)
    got = nut_wall(jnp.full_like(d, nu), d, k, MODEL)
    assert jnp.allclose(got, expected)
    assert float(got[0]) > 0.0


def test_nut_wall_is_velocity_independent_and_finite_everywhere() -> None:
    """No dependence on velocity (so no reattachment singularity) and finite/non-negative for all k.

    Sweeping ``k`` across the sublayer, the crossover, and the log layer, ``nu_t,wall`` is finite and
    ``>= 0`` throughout -- there is no ``ln`` singularity at the crossover and no blow-up (the failure
    mode a velocity-based wall law has where the near-wall velocity vanishes).
    """
    nu = 1e-5
    d = jnp.full((200,), 5e-4)
    k = jnp.linspace(0.0, 2.0, 200)  # spans y* from 0 through the crossover into the log layer
    got = nut_wall(jnp.full_like(d, nu), d, k, MODEL)
    assert bool(jnp.all(jnp.isfinite(got)))
    assert bool(jnp.all(got >= 0.0))


def test_nut_wall_is_differentiable_in_k() -> None:
    """Gradients flow through the wall function without NaNs (it is live in the coupled residual)."""
    nu = 1e-5
    d = jnp.array([3e-3])
    grad_log = jax.grad(lambda k: jnp.sum(nut_wall(jnp.full_like(d, nu), d, k, MODEL)))(
        jnp.array([0.3])
    )
    grad_sub = jax.grad(lambda k: jnp.sum(nut_wall(jnp.full_like(d, nu), d, k, MODEL)))(
        jnp.array([1e-4])
    )
    assert bool(jnp.all(jnp.isfinite(grad_log)))
    assert bool(jnp.all(jnp.isfinite(grad_sub)))


# --- the log-layer crossover and the closures blended over it --------------------------------


def test_wall_function_weight_spans_the_crossover_smoothly() -> None:
    """f -> 0 deep in the sublayer, -> 1 far into the log layer, monotone and differentiable between.

    Every near-wall closure is blended on this weight rather than switched: the residual is solved by
    an automatically-differentiated Newton method, which cannot converge through a jump.
    """
    nu = 1e-5
    d = jnp.full((400,), 5e-4)
    k = jnp.linspace(0.0, 4.0, 400)  # y* from 0 well past the crossover
    f = wall_function_weight(jnp.full_like(d, nu), d, k, MODEL)
    assert float(f[0]) == 0.0  # k = 0 is fully resolved
    assert float(f[-1]) > 0.999  # far into the log layer
    assert bool(jnp.all(jnp.diff(f) >= 0.0))  # monotone in k
    grad = jax.grad(lambda kk: jnp.sum(wall_function_weight(jnp.full_like(d, nu), d, kk, MODEL)))(k)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_log_layer_shear_rate_is_the_law_of_the_wall_gradient() -> None:
    """S_log = u_tau/(kappa d) with the k-based u_tau, and its k-derivative is finite at k = 0."""
    d, k = jnp.array([3e-3]), jnp.array([0.3])
    u_tau = MODEL.beta_star**0.25 * jnp.sqrt(k)
    assert jnp.allclose(log_layer_shear_rate(d, k, MODEL), u_tau / (MODEL.kappa * d))
    grad = jax.grad(lambda kk: jnp.sum(log_layer_shear_rate(d, kk, MODEL)))(jnp.zeros_like(k))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_wall_shear_stress_is_the_geometric_mean_of_the_two_friction_velocities() -> None:
    """tau_w = u_k u_log: the k-based and velocity-based friction velocities multiplied.

    This identity is what makes the near-wall k budget a genuine equation for ``k`` -- it is satisfied
    only where the two friction velocities agree -- rather than an identity every ``k`` solves.
    """
    nu = 1e-5
    d, k, shear = jnp.array([3e-3]), jnp.array([0.3]), jnp.array([900.0])
    y_star = _y_star(k, d, nu, MODEL)
    assert float(y_star[0]) > MODEL.wall_y_star_lam  # in the log layer, where the identity holds
    u_k = MODEL.beta_star**0.25 * jnp.sqrt(k)
    u_log = MODEL.kappa * shear * d / jnp.log(MODEL.e_wall * y_star)
    got = wall_shear_stress(jnp.full_like(d, nu), d, k, shear, MODEL)
    assert jnp.allclose(got, u_k * u_log)


def test_wall_shear_stress_reduces_to_the_molecular_stress_when_resolved() -> None:
    """Below the crossover nu_t,wall = 0, so the wall stress is exactly nu |dU/dn| -- a no-slip wall."""
    nu = 1e-5
    d, k, shear = jnp.array([1e-5]), jnp.array([0.01]), jnp.array([50.0])
    assert float(_y_star(k, d, nu, MODEL)[0]) < MODEL.wall_y_star_lam
    got = wall_shear_stress(jnp.full_like(d, nu), d, k, shear, MODEL)
    assert jnp.allclose(got, nu * shear)


def test_k_wall_production_balances_destruction_at_the_equilibrium_k() -> None:
    """At the log-layer equilibrium, P_k = beta_star k omega_log exactly -- and only there.

    Equilibrium is where the k-based friction velocity equals the velocity-based one. Picking the
    wall shear rate that makes them agree, production must equal the log-layer destruction exactly;
    perturbing ``k`` off it must break the balance in the restoring direction (so the wall cell has a
    genuine root, not an identity every ``k`` satisfies). The sign checks use the full
    :func:`omega_wall`, whose viscous branch adds a fraction of a percent here.
    """
    nu = 1e-5
    d, k = jnp.array([3e-3]), jnp.array([0.3])
    # The shear rate for which u_log == u_k, i.e. the equilibrium wall shear rate.
    y_star = _y_star(k, d, nu, MODEL)
    u_k = MODEL.beta_star**0.25 * jnp.sqrt(k)
    shear = u_k * jnp.log(MODEL.e_wall * y_star) / (MODEL.kappa * d)

    production = k_wall_production(jnp.full_like(d, nu), d, k, shear, MODEL)
    assert jnp.allclose(production, MODEL.beta_star * k * _omega_log(k, d, MODEL), rtol=1e-12)

    def imbalance(kk):
        p = k_wall_production(jnp.full_like(d, nu), d, kk, shear, MODEL)
        destruction = MODEL.beta_star * kk * omega_wall(jnp.full_like(d, nu), d, kk, MODEL)
        return float((p - destruction)[0])

    assert imbalance(0.7 * k) > 0.0  # too little k -> net production, k is driven back up
    assert imbalance(1.4 * k) < 0.0  # too much k -> net destruction


def test_k_wall_production_is_finite_and_differentiable_at_zero_k() -> None:
    """No NaN in the value or the k-derivative at k = 0 (the cold-start / stagnation corner)."""
    nu = 1e-5
    d, shear = jnp.full((3,), 3e-3), jnp.full((3,), 900.0)
    k = jnp.zeros(3)
    value = k_wall_production(jnp.full_like(d, nu), d, k, shear, MODEL)
    grad = jax.grad(
        lambda kk: jnp.sum(k_wall_production(jnp.full_like(d, nu), d, kk, shear, MODEL))
    )(k)
    assert bool(jnp.all(jnp.isfinite(value)))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_k_wall_production_vanishes_with_the_wall_shear() -> None:
    """A stagnation / reattachment point (zero wall shear) produces no k, with no singularity."""
    nu = 1e-5
    d, k = jnp.array([3e-3]), jnp.array([0.3])
    got = k_wall_production(jnp.full_like(d, nu), d, k, jnp.zeros_like(d), MODEL)
    assert jnp.allclose(got, 0.0)


def test_wall_k_diffusivity_fades_from_the_full_value_to_zero() -> None:
    """gamma_wall = gamma when resolved (Dirichlet-0 flux intact) and -> 0 out in the log layer."""
    nu = 1e-5
    gamma = jnp.array([2e-4, 2e-4])
    d = jnp.array([1e-5, 3e-3])  # sublayer, then log layer
    k = jnp.array([0.01, 0.3])
    got = wall_k_diffusivity(gamma, jnp.full_like(d, nu), d, k, MODEL)
    assert float(_y_star(k[:1], d[:1], nu, MODEL)[0]) < MODEL.wall_y_star_lam
    # Resolved wall: unchanged to ~1e-10 relative (the quartic tanh weight is small but not zero
    # below the crossover, which is the price of making the transition differentiable).
    assert jnp.allclose(got[0], gamma[0], rtol=1e-8)
    assert float(got[1]) < 1e-3 * float(gamma[1])  # log layer: no turbulent-energy flux to the wall


def test_wall_k_diffusivity_is_differentiable_in_k() -> None:
    """The fade is live in the coupled residual, so its k-derivative must be finite everywhere."""
    nu = 1e-5
    gamma = jnp.full((200,), 2e-4)
    d = jnp.full((200,), 5e-4)
    k = jnp.linspace(0.0, 2.0, 200)
    grad = jax.grad(
        lambda kk: jnp.sum(wall_k_diffusivity(gamma, jnp.full_like(d, nu), d, kk, MODEL))
    )(k)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_inlet_k_from_intensity() -> None:
    assert jnp.allclose(inlet_k(jnp.array([10.0]), 0.05), 1.5 * (10.0 * 0.05) ** 2)


def test_inlet_omega_from_length_scale() -> None:
    k = jnp.array([0.375])
    assert jnp.allclose(inlet_omega(k, 0.1, MODEL), jnp.sqrt(k) / (MODEL.beta_star**0.25 * 0.1))


def test_boundary_values_are_differentiable() -> None:
    """Gradients flow through the viscosity (wall omega) and the velocity (inlet k), no NaNs."""
    grad_nu = jax.grad(lambda nu: jnp.sum(omega_wall_value(nu, jnp.array([1e-3]), MODEL)))(
        jnp.array([1e-5])
    )
    grad_u = jax.grad(lambda u: jnp.sum(inlet_k(u, 0.05)))(jnp.array([10.0]))
    assert not bool(jnp.any(jnp.isnan(grad_nu)))
    assert not bool(jnp.any(jnp.isnan(grad_u)))
