"""Prescribed boundary numbers are floating array leaves, so a pytree gradient reaches them.

``equinox.filter_grad`` -- and the implicit-function-theorem adjoint of a solve that carries its
assembler as the parameters -- differentiates floating array leaves only. A coefficient held as a
Python float, an integer, or in a static field receives no cotangent at all: the sensitivity is
absent rather than wrong, and a finiteness check cannot see it. These pin the storage rule on a
single face, with no mesh: every coefficient is a floating array whatever the caller passes, a
position function's coefficients are leaves when it is an ``equinox.Module``, and a plain function
is still accepted, held static.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import Convective, Dirichlet, DirichletField, Neumann
from aquaflux.boundary.conditions import StaticFunction
from aquaflux.flow import MovingWall, PressureOutlet, VelocityInlet

# One non-orthogonal face: n = (1, 0), d = (0.5, 0.3), so d.n = 0.5; gradient (1, 4) gives a
# tangential correction of 1.2, and the owner value 3 makes phi_P + corr = 4.2.
N = jnp.array([[1.0, 0.0]])
D = jnp.array([[0.5, 0.3]])
GRAD = jnp.array([[1.0, 4.0]])
PHI = jnp.array([3.0])
GAMMA = jnp.array([2.0])
FC = jnp.array([[1.0, 0.2]])
OWNER_PLUS_CORR, DN = 4.2, 0.5


class _LinearInY(eqx.Module):
    """``x -> a * y``, a position function whose coefficient is a field."""

    a: jnp.ndarray

    def __call__(self, x):
        return self.a * x[:, 1]


class _Plug(eqx.Module):
    """A uniform ``(speed, 0)`` velocity profile whose speed is a field."""

    speed: jnp.ndarray

    def __call__(self, x):
        return jnp.stack([self.speed * jnp.ones(x.shape[0]), jnp.zeros(x.shape[0])], axis=1)


def _face(bc):
    return bc.face_value(PHI, GRAD, D, N, GAMMA, FC)[0]


@pytest.mark.parametrize(
    ("closure", "fields"),
    [
        (Dirichlet(1), ("value",)),
        (Neumann(2), ("flux",)),
        (Convective(3, 4), ("h", "t_inf")),
        (PressureOutlet(pressure=0), ("pressure",)),
    ],
)
def test_numeric_coefficients_are_floating_array_leaves_even_when_given_integers(
    closure, fields
) -> None:
    """An integer is the hardest input: a Python float would at least be inexact once converted."""
    for name in fields:
        assert eqx.is_inexact_array(getattr(closure, name))
    assert len(jax.tree.leaves(closure)) == len(fields)


def test_filter_grad_reaches_every_scalar_closure_coefficient() -> None:
    """Each derivative against the closure's own closed form -- not merely non-zero."""
    assert float(eqx.filter_grad(_face)(Dirichlet(7.5)).value) == 1.0
    # phi_ip = phi_P + corr - (q / Gamma) (d.n)  =>  d/dq = -(d.n) / Gamma
    assert float(eqx.filter_grad(_face)(Neumann(1.5)).flux) == pytest.approx(-DN / 2.0)
    # phi_ip = (a + beta t) / (1 + beta), beta = h (d.n) / Gamma
    h, t_inf = 2.0, 5.0
    beta = h * DN / 2.0
    grad = eqx.filter_grad(_face)(Convective(h, t_inf))
    assert float(grad.t_inf) == pytest.approx(beta / (1.0 + beta))
    assert float(grad.h) == pytest.approx(
        (DN / 2.0) * (t_inf - OWNER_PLUS_CORR) / (1.0 + beta) ** 2
    )


def test_filter_grad_reaches_an_outlet_pressure_given_as_a_float() -> None:
    """The pressure the outlet imposes is its face value, so its derivative is exactly one."""
    grad = eqx.filter_grad(lambda bc: bc.pressure_face(PHI, GRAD, D, N, FC)[0])(
        PressureOutlet(pressure=0.5)
    )
    assert float(grad.pressure) == 1.0


def test_a_plain_position_function_is_held_static_and_still_evaluates() -> None:
    bc = DirichletField(field_fn=lambda x: 2.0 * x[..., 0])
    assert isinstance(bc.field_fn, StaticFunction)
    assert jax.tree.leaves(bc) == []
    assert float(_face(bc)) == 2.0


def test_a_module_position_function_keeps_its_coefficient_differentiable() -> None:
    grad = eqx.filter_grad(_face)(DirichletField(field_fn=_LinearInY(a=jnp.asarray(3.0))))
    assert float(grad.field_fn.a) == pytest.approx(float(FC[0, 1]))  # d(a y)/da = y


def test_a_non_callable_position_function_is_refused() -> None:
    with pytest.raises(TypeError, match="callable"):
        DirichletField(field_fn=1.0)


@pytest.mark.parametrize("cls", [VelocityInlet, MovingWall])
def test_a_constant_prescribed_velocity_is_a_floating_array_leaf(cls) -> None:
    bc = cls(velocity=(1, 0))
    assert eqx.is_inexact_array(bc.velocity)
    assert bc.velocity.shape == (2,)


@pytest.mark.parametrize("cls", [VelocityInlet, MovingWall])
def test_filter_grad_reaches_a_prescribed_velocity(cls) -> None:
    """Constant and module profiles both carry the imposed face speed, with derivative one."""

    def face_speed(bc):
        return bc.velocity_face(jnp.zeros((1, 2)), jnp.zeros((1, 2, 2)), D, N, FC)[0, 0]

    assert float(eqx.filter_grad(face_speed)(cls(velocity=(2.0, 0.0))).velocity[0]) == 1.0
    profile = cls(velocity=_Plug(speed=jnp.asarray(2.0)))
    assert float(eqx.filter_grad(face_speed)(profile).velocity.speed) == 1.0


def test_a_plain_velocity_profile_is_held_static() -> None:
    """Existing function profiles keep working, but carry nothing a gradient could reach."""
    bc = VelocityInlet(velocity=lambda x: jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1))
    assert jax.tree.leaves(bc) == []
    face = bc.velocity_face(jnp.zeros((1, 2)), jnp.zeros((1, 2, 2)), D, N, FC)
    assert jnp.allclose(face, jnp.array([[0.2, 0.0]]))


def test_closures_stack_across_partitions() -> None:
    """A distributed build stacks one closure per partition; array leaves and static functions both
    survive that, where a function held as a leaf would not."""

    def profile(x):
        return jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1)

    for make in (
        lambda: VelocityInlet(velocity=(1.0, 0.0)),
        lambda: VelocityInlet(velocity=profile),
        lambda: DirichletField(field_fn=_LinearInY(a=jnp.asarray(1.0))),
    ):
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), make(), make())
        assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(stacked))
