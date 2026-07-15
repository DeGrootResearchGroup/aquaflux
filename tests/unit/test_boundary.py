"""Unit tests for the weak boundary face-value closures.

Each closure is exercised on a single boundary face with a known owner value, gradient, and
geometry, and checked against its closed form — no mesh, no solve (the seam the boundary rule
requires). Both an orthogonal face (displacement parallel to the normal, so the tangential
correction vanishes) and a non-orthogonal face (a genuine tangential offset) are used.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.boundary import Convective, Dirichlet, DirichletField, Neumann, ZeroGradient

# A non-orthogonal face: normal n = (1, 0), displacement d = (0.5, 0.3) so d.n = 0.5 and the
# tangential part is (0, 0.3). With gradient (1, 4), corr = 4 * 0.3 = 1.2.
N = jnp.array([[1.0, 0.0]])
D = jnp.array([[0.5, 0.3]])
GRAD = jnp.array([[1.0, 4.0]])
PHI = jnp.array([3.0])
GAMMA = jnp.array([2.0])
CORR = 1.2  # grad . tangential(d)
DN = 0.5  # d . n

# An orthogonal face: d parallel to n, so the tangential correction is exactly zero.
D_ORTHO = jnp.array([[0.5, 0.0]])

# Face centroid — ignored by every closure here except a position-dependent Dirichlet.
FC = jnp.array([[1.0, 0.2]])


def test_dirichlet_returns_value() -> None:
    bc = Dirichlet(value=7.5)
    assert float(bc.face_value(PHI, GRAD, D, N, GAMMA, FC)[0]) == 7.5


def test_dirichlet_field_evaluates_at_face_centroid() -> None:
    bc = DirichletField(field_fn=lambda x: 2.0 * x[..., 0] - x[..., 1])
    centroids = jnp.array([[1.0, 0.2], [0.5, 0.5]])
    vals = bc.face_value(PHI, GRAD, D, N, GAMMA, centroids)
    assert jnp.allclose(vals, jnp.array([2.0 * 1.0 - 0.2, 2.0 * 0.5 - 0.5]))


def test_zero_gradient_adds_only_tangential_correction() -> None:
    bc = ZeroGradient()
    assert float(bc.face_value(PHI, GRAD, D, N, GAMMA, FC)[0]) == float(PHI[0]) + CORR


def test_zero_gradient_orthogonal_equals_cell_value() -> None:
    bc = ZeroGradient()
    assert float(bc.face_value(PHI, GRAD, D_ORTHO, N, GAMMA, FC)[0]) == float(PHI[0])


def test_neumann_matches_closed_form() -> None:
    bc = Neumann(flux=1.5)
    expected = float(PHI[0]) + CORR - 1.5 / float(GAMMA[0]) * DN
    assert float(bc.face_value(PHI, GRAD, D, N, GAMMA, FC)[0]) == expected


def test_convective_matches_closed_form() -> None:
    bc = Convective(h=4.0, t_inf=1.0)
    beta = 4.0 / float(GAMMA[0]) * DN
    expected = (float(PHI[0]) + CORR + beta * 1.0) / (1.0 + beta)
    assert abs(float(bc.face_value(PHI, GRAD, D, N, GAMMA, FC)[0]) - expected) < 1e-12


def test_convective_enforces_robin_balance() -> None:
    """The returned face value must satisfy Gamma dphi/dn = h (Tinf - phi_ip) at the face."""
    h, t_inf = 4.0, 1.0
    phi_ip = float(Convective(h=h, t_inf=t_inf).face_value(PHI, GRAD, D, N, GAMMA, FC)[0])
    dphi_dn = (phi_ip - float(PHI[0]) - CORR) / DN  # one-sided normal derivative to the face
    assert abs(float(GAMMA[0]) * dphi_dn - h * (t_inf - phi_ip)) < 1e-12


def test_convective_high_biot_approaches_dirichlet() -> None:
    """As h -> infinity the convective closure drives the face value to the ambient value."""
    phi_ip = float(Convective(h=1e6, t_inf=0.25).face_value(PHI, GRAD, D_ORTHO, N, GAMMA, FC)[0])
    assert abs(phi_ip - 0.25) < 1e-4
