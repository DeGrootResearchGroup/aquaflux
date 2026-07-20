"""Unit tests for the k-omega SST source operators.

Each operator is evaluated on a two-cell context with hand-chosen frozen fields and checked against
its closed form times the cell volume; the production limiter is checked on both sides of the cap,
and gradients are checked through a model constant and the solved field.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.discretization import FaceContext
from aquaflux.mesh import structured_grid_2d
from aquaflux.turbulence import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
    SSTModel,
)


def _context_and_volume():
    """A two-cell context (only its cell volume is read by a source) and that volume."""
    mesh = structured_grid_2d(2, 1)
    geometry = mesh.geometry()
    context = FaceContext(
        face_cells=mesh.face_cells,
        geometry=geometry,
        boundary_values=jnp.zeros(mesh.n_faces),
        gradient=jnp.zeros((mesh.n_cells, mesh.dim)),
        properties={},
    )
    return context, geometry.cell.volume


def _cell(*values):
    return jnp.array([float(v) for v in values])


MODEL = SSTModel()


def test_k_production_unlimited() -> None:
    """When ν_t S² is below the cap, production is ν_t S² times the volume."""
    context, volume = _context_and_volume()
    op = KProduction(
        nu_t=_cell(0.01, 0.01), strain_rate=_cell(1.0, 1.0), omega=_cell(1.0, 1.0), model=MODEL
    )
    assert jnp.allclose(op.source(_cell(1.0, 1.0), context), 0.01 * volume)


def test_k_production_limited() -> None:
    """When ν_t S² exceeds 10 β* k ω, production is capped at 10 β* k ω."""
    context, volume = _context_and_volume()
    op = KProduction(
        nu_t=_cell(10.0, 10.0), strain_rate=_cell(1.0, 1.0), omega=_cell(1.0, 1.0), model=MODEL
    )
    cap = 10.0 * MODEL.beta_star * 1.0 * 1.0  # 10 β* k ω
    assert jnp.allclose(op.source(_cell(1.0, 1.0), context), cap * volume)


def test_k_production_explicit_limiter_keeps_the_value_but_drops_the_k_derivative() -> None:
    """``explicit_limiter`` (a forward-solve stabilization) leaves the production *value* identical
    but removes its ``k``-derivative where the cap is active -- so the k-equation Jacobian loses the
    destabilizing feedback term and stays an M-matrix, while the residual is unchanged."""
    context, _ = _context_and_volume()
    args = dict(
        nu_t=_cell(10.0, 10.0), strain_rate=_cell(1.0, 1.0), omega=_cell(1.0, 1.0), model=MODEL
    )
    exact = KProduction(**args)
    explicit = KProduction(**args, explicit_limiter=True)
    k = _cell(1.0, 1.0)  # cap active (ν_t S² = 10 > 10 β* k ω)

    # Same forward value.
    assert jnp.allclose(exact.source(k, context), explicit.source(k, context))
    # Exact linearization is non-zero (the cap feeds k back); the explicit one is zero.
    d_exact = jax.jacobian(lambda f: exact.source(f, context))(k)
    d_explicit = jax.jacobian(lambda f: explicit.source(f, context))(k)
    assert float(jnp.max(jnp.abs(jnp.diag(d_exact)))) > 0.0
    assert float(jnp.max(jnp.abs(d_explicit))) == 0.0


def test_k_destruction_is_a_negative_sink() -> None:
    context, volume = _context_and_volume()
    op = KDestruction(omega=_cell(3.0, 3.0), model=MODEL)
    assert jnp.allclose(op.source(_cell(2.0, 2.0), context), -MODEL.beta_star * 2.0 * 3.0 * volume)


def _omega_production(s, f1, nu_t=None, k=None, omega=None):
    unit = _cell(1.0, 1.0)
    return OmegaProduction(
        strain_rate=s,
        nu_t=_cell(0.01, 0.01) if nu_t is None else nu_t,
        k=unit if k is None else k,
        omega=unit if omega is None else omega,
        f1=f1,
        model=MODEL,
    )


def test_omega_production_unlimited_blends_alpha() -> None:
    """Below the cap the production is the blended ``α S²`` (α by F1); here 10 β* k ω / ν_t = 90 > 4."""
    context, volume = _context_and_volume()
    s = _cell(2.0, 2.0)  # S² = 4, cap = 10 * 0.09 * 1 * 1 / 0.01 = 90 -> inactive
    inner = _omega_production(s, _cell(1.0, 1.0))
    outer = _omega_production(s, _cell(0.0, 0.0))
    assert jnp.allclose(inner.source(_cell(1.0, 1.0), context), MODEL.alpha_1 * 4.0 * volume)
    assert jnp.allclose(outer.source(_cell(1.0, 1.0), context), MODEL.alpha_2 * 4.0 * volume)


def test_omega_production_limited_at_the_destruction_scale() -> None:
    """When ``S²`` exceeds ``10 β* k ω / ν_t`` the production is capped there (times α)."""
    context, volume = _context_and_volume()
    s = _cell(3.0, 3.0)  # S² = 9
    nu_t, k, omega = _cell(1.0, 1.0), _cell(1.0, 1.0), _cell(1.0, 1.0)
    cap = 10.0 * MODEL.beta_star * 1.0 * 1.0 / 1.0  # 10 β* k ω / ν_t = 0.9 < 9 -> active
    op = _omega_production(s, _cell(1.0, 1.0), nu_t=nu_t, k=k, omega=omega)
    assert jnp.allclose(op.source(_cell(1.0, 1.0), context), MODEL.alpha_1 * cap * volume)


def test_omega_production_is_independent_of_the_solved_field() -> None:
    """The source reads only the frozen closure, so it has no derivative in the solved ``ω`` field
    (it adds no diagonal term to the ω-equation Jacobian)."""
    context, _ = _context_and_volume()
    op = _omega_production(_cell(3.0, 3.0), _cell(1.0, 1.0), nu_t=_cell(1.0, 1.0))
    d = jax.jacobian(lambda f: op.source(f, context))(_cell(1.0, 1.0))
    assert float(jnp.max(jnp.abs(d))) == 0.0


def test_omega_destruction_blends_beta_and_is_negative() -> None:
    context, volume = _context_and_volume()
    op = OmegaDestruction(f1=_cell(1.0, 1.0), model=MODEL)  # F1 = 1 -> beta_1
    assert jnp.allclose(op.source(_cell(2.0, 2.0), context), -MODEL.beta_1 * 4.0 * volume)


def test_omega_cross_diffusion_value_and_vanishes_at_wall() -> None:
    """Away from the wall (F1 = 0) the term is 2 σ_ω2 (∇k·∇ω)/ω; at the wall (F1 = 1) it vanishes."""
    context, volume = _context_and_volume()
    grad_k = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    grad_omega = jnp.array([[2.0, 0.0], [2.0, 0.0]])  # dot = 2
    outer = OmegaCrossDiffusion(
        omega=_cell(1.0, 1.0), grad_k=grad_k, grad_omega=grad_omega, f1=_cell(0.0, 0.0), model=MODEL
    )
    expected = 2.0 * MODEL.sigma_omega2 * 2.0 / 1.0
    assert jnp.allclose(outer.source(_cell(1.0, 1.0), context), expected * volume)
    at_wall = OmegaCrossDiffusion(
        omega=_cell(1.0, 1.0), grad_k=grad_k, grad_omega=grad_omega, f1=_cell(1.0, 1.0), model=MODEL
    )
    assert jnp.allclose(at_wall.source(_cell(1.0, 1.0), context), 0.0)


def test_sources_are_differentiable() -> None:
    """jax.grad flows through a model constant (destruction) and the solved field, no NaNs."""
    context, _ = _context_and_volume()
    k = _cell(2.0, 2.0)

    def destruction_total(beta_star):
        op = KDestruction(omega=_cell(3.0, 3.0), model=SSTModel(beta_star=beta_star))
        return jnp.sum(op.source(k, context))

    grad_const = jax.grad(destruction_total)(0.09)
    grad_field = jax.grad(
        lambda kk: jnp.sum(KDestruction(omega=_cell(3.0, 3.0), model=MODEL).source(kk, context))
    )(k)
    assert not bool(jnp.isnan(grad_const))
    assert not bool(jnp.any(jnp.isnan(grad_field)))
