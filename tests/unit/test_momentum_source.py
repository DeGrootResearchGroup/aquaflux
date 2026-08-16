"""Unit tests for the momentum-source seam: :class:`MomentumSource` and :class:`UniformBodyForce`.

A momentum source is the vector counterpart of a :class:`~aquaflux.discretization.VolumeSource`:
it returns a volume-integrated force per cell, production positive, and the momentum residual
subtracts it. These check that sign, the volume integration, that several sources compose
additively, and that gradients flow through a source's own coefficients -- the property that makes
a source usable as a design parameter.

The contract's other two members are checked for what they promise rather than for a value: a
uniform force needs no mass-flux treatment (``face_force`` is ``None``) and adds no diagonal,
because it neither varies in space nor reads the velocity. Both are pinned against
differentiation of the source itself, so an implementation cannot drift from the term it describes.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    MomentumContinuity,
    MomentumSource,
    NoSlipWall,
    UniformBodyForce,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss


class _LinearDrag(MomentumSource):
    """A velocity-dependent sink ``-rate * u``, the shape a porous-media drag takes.

    Present so the contract is exercised by something that genuinely reads the velocity and
    therefore genuinely contributes a diagonal -- the members a uniform force answers trivially.
    """

    rate: float

    def source(self, fields, geometry, properties):
        from aquaflux.vectors import scale

        return scale(-self.rate * fields.velocity, geometry.cell.volume)

    def face_force(self, geometry, properties):
        return None

    def diagonal(self, fields, geometry, properties):
        # -d(source)/d(u), integrated on the volume: a drag opposing the flow is a positive diagonal.
        return jnp.broadcast_to((self.rate * geometry.cell.volume)[:, None], fields.velocity.shape)


@pytest.fixture
def case():
    """A small closed cavity, its geometry, and a kinematic state to evaluate sources at."""
    mesh = structured_grid_2d(3, 3, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(1e-3), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({name: NoSlipWall() for name in mesh.face_patches.names}),
        pressure_pin=0,
    )
    state = (
        momentum.initial_state()
        .at[: mesh.n_cells * mesh.dim]
        .set(jnp.linspace(-1.0, 2.0, mesh.n_cells * mesh.dim))
    )
    return momentum, geometry, momentum.velocity_fields(state)


def _properties(momentum):
    return momentum.properties.evaluate(momentum.mesh.cell_zones)


def test_a_uniform_force_is_the_force_times_the_cell_volume(case) -> None:
    """The source bakes in its own volume quadrature, as the scalar contract does."""
    momentum, geometry, fields = case
    force = jnp.array([0.35, -0.2])

    integrated = UniformBodyForce(force).source(fields, geometry, _properties(momentum))

    assert jnp.allclose(integrated, geometry.cell.volume[:, None] * force)


def test_a_uniform_force_needs_no_face_treatment_and_adds_no_diagonal(case) -> None:
    """Both are properties of being uniform and state-independent, not conveniences."""
    momentum, geometry, fields = case
    source = UniformBodyForce(jnp.array([0.35, -0.2]))

    assert source.face_force(geometry, _properties(momentum)) is None
    assert jnp.array_equal(
        source.diagonal(fields, geometry, _properties(momentum)), jnp.zeros_like(fields.velocity)
    )


def test_a_uniform_source_reproduces_the_inline_body_force_term(case) -> None:
    """``UniformBodyForce`` and the ``body_force`` leaf are the same term, so they must agree.

    They add rather than replace, so putting the force in one place or the other gives the same
    residual -- which is what will make migrating the leaf onto the source a behaviour-neutral
    change rather than a numerical one.
    """
    momentum, _, _ = case
    force = (0.35, -0.2)
    state = momentum.initial_state().at[0].set(0.4)

    via_leaf = momentum.build(
        momentum.mesh,
        momentum.geometry,
        momentum.properties,
        momentum.gradient_scheme,
        BoundaryConditions({name: NoSlipWall() for name in momentum.mesh.face_patches.names}),
        pressure_pin=0,
        body_force=force,
    )
    via_source = momentum.build(
        momentum.mesh,
        momentum.geometry,
        momentum.properties,
        momentum.gradient_scheme,
        BoundaryConditions({name: NoSlipWall() for name in momentum.mesh.face_patches.names}),
        pressure_pin=0,
        sources=(UniformBodyForce(jnp.asarray(force)),),
    )

    assert jnp.allclose(via_leaf.residual(state), via_source.residual(state))


def test_sources_are_subtracted_and_compose_additively(case) -> None:
    """Two sources sum, and each leaves the balance as a sink."""
    momentum, _, _ = case
    state = momentum.initial_state().at[0].set(0.4)
    boundary = BoundaryConditions({n: NoSlipWall() for n in momentum.mesh.face_patches.names})

    def built(sources):
        return momentum.build(
            momentum.mesh,
            momentum.geometry,
            momentum.properties,
            momentum.gradient_scheme,
            boundary,
            pressure_pin=0,
            sources=sources,
        )

    a = UniformBodyForce(jnp.array([0.35, 0.0]))
    b = UniformBodyForce(jnp.array([0.0, -0.2]))
    both = UniformBodyForce(jnp.array([0.35, -0.2]))

    assert jnp.allclose(built((a, b)).residual(state), built((both,)).residual(state))


def test_no_sources_is_the_unsourced_residual(case) -> None:
    """The default empty tuple must be exactly a no-op, not merely a small one."""
    momentum, _, _ = case
    state = momentum.initial_state().at[0].set(0.4)
    boundary = BoundaryConditions({n: NoSlipWall() for n in momentum.mesh.face_patches.names})

    plain = momentum.build(
        momentum.mesh,
        momentum.geometry,
        momentum.properties,
        momentum.gradient_scheme,
        boundary,
        pressure_pin=0,
    )
    empty = momentum.build(
        momentum.mesh,
        momentum.geometry,
        momentum.properties,
        momentum.gradient_scheme,
        boundary,
        pressure_pin=0,
        sources=(),
    )

    assert jnp.array_equal(plain.residual(state), empty.residual(state))


def test_a_source_is_differentiable_in_its_own_coefficient(case) -> None:
    """Gradients reach a source's leaves, which is what makes one usable as a design parameter."""
    momentum, geometry, fields = case
    properties = _properties(momentum)

    def total(force):
        return jnp.sum(UniformBodyForce(force).source(fields, geometry, properties))

    grad = jax.grad(total)(jnp.array([0.35, -0.2]))

    assert jnp.allclose(grad, jnp.sum(geometry.cell.volume))


def test_a_velocity_dependent_source_reports_the_diagonal_ad_gives(case) -> None:
    """The diagonal must equal ``-d(source)/d(u)`` from AD of the source's own term.

    This is the check that keeps the two from drifting. The momentum diagonal is assembled
    separately from the residual and feeds the Rhie--Chow damping, the frozen preconditioner and the
    pseudo-transient shift, so a source that misreports it leaves those built from an operator the
    solve is not running.
    """
    momentum, geometry, fields = case
    properties = _properties(momentum)
    drag = _LinearDrag(rate=2.5)

    def component_source(velocity):
        moved = type(fields)(velocity, fields.boundary_velocity, fields.gradient)
        return drag.source(moved, geometry, properties)

    jacobian = jax.jacobian(component_source)(fields.velocity)
    n_cells, dim = fields.velocity.shape
    ad_diagonal = jnp.stack(
        [jnp.stack([-jacobian[c, i, c, i] for i in range(dim)]) for c in range(n_cells)]
    )

    assert jnp.allclose(drag.diagonal(fields, geometry, properties), ad_diagonal)
