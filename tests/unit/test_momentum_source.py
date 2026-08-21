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

Declaring a diagonal is only half of it -- it also has to be *used*, so the rest of these check that
a source's diagonal reaches the assembled ``a_P``, that ``a_P`` is still the residual's own operator
diagonal once it does, and that the Rhie--Chow damping ``V / a_P`` moves with it (which is what makes
this a converged-answer property rather than a preconditioning one). ``face_force`` has no consumer
yet, so what is pinned there is the refusal: declaring one is an error rather than a silent drop.
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
from aquaflux.flow.rhie_chow import momentum_diagonal
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.vectors import scale


class _LinearDrag(MomentumSource):
    """A velocity-dependent sink ``-rate * u``, the shape a porous-media drag takes.

    Present so the contract is exercised by something that genuinely reads the velocity and
    therefore genuinely contributes a diagonal -- the members a uniform force answers trivially.
    """

    rate: float

    def source(self, fields, geometry, properties):
        return scale(-self.rate * fields.velocity, geometry.cell.volume)

    def face_force(self, geometry, properties):
        return None

    def diagonal(self, velocity, geometry, properties):
        # -d(source)/d(u), integrated on the volume: a drag opposing the flow is a positive diagonal.
        return jnp.broadcast_to((self.rate * geometry.cell.volume)[:, None], velocity.shape)


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
        source.diagonal(fields.velocity, geometry, _properties(momentum)),
        jnp.zeros_like(fields.velocity),
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

    assert jnp.allclose(drag.diagonal(fields.velocity, geometry, properties), ad_diagonal)


class _SpatiallyVaryingForce(MomentumSource):
    """A source that declares a face-normal force -- the case the mass flux cannot yet carry."""

    def source(self, fields, geometry, properties):
        return scale(jnp.ones_like(fields.velocity), geometry.cell.volume)

    def face_force(self, geometry, properties):
        return geometry.face.centroid[:, 0]

    def diagonal(self, velocity, geometry, properties):
        return jnp.zeros_like(velocity)


def _stokes_case(sources=()):
    """A closed Stokes cavity: no advection, so ``a_P`` is the viscous diagonal plus any source."""
    mesh = structured_grid_2d(6, 6, named_boundaries=True)
    geometry = mesh.geometry()
    return mesh, MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({side: NoSlipWall() for side in ("top", "bottom", "left", "right")}),
        pressure_pin=0,
        sources=sources,
    )


def test_a_source_diagonal_reaches_a_p_and_keeps_it_the_residual_operator_diagonal() -> None:
    """``a_P`` must still equal ``diag(J_momentum)`` once a velocity-dependent source is injected.

    This is the check the declared-but-unwired diagonal could not pass. A drag contributes
    ``rate * V`` to its own momentum row, so an ``a_P`` assembled from the face terms alone is short
    by exactly that -- and ``a_P`` is not solver-internal: it is the Rhie--Chow damping ``V / a_P``,
    which is differentiated and solution-affecting. Comparing against the residual's own Jacobian
    diagonal rather than against ``rate * V`` is what makes this an operator-consistency test instead
    of a transcription of the same formula twice. Stokes flow (no advection) isolates it: the residual
    is linear and ``a_P`` carries no lagged convective estimate.
    """
    mesh, asm = _stokes_case(sources=(_LinearDrag(rate=3.5),))

    velocity = jnp.zeros((mesh.n_cells, mesh.dim))
    a_p = asm.momentum_matrix_diagonal(velocity)
    state = asm.pack(velocity, jnp.zeros(mesh.n_cells))
    velocity_diag, _ = asm.unpack(jnp.diag(jax.jacfwd(asm.residual)(state)))

    assert jnp.allclose(a_p, velocity_diag, rtol=1e-9, atol=1e-9)


def test_the_drag_is_what_that_agreement_turns_on() -> None:
    """The control: without the wiring the same comparison is off by the drag, so it can fail.

    A test that only ever compares two things that agree cannot show it is measuring anything. Here
    the sourceless ``a_P`` is compared against the *sourced* residual's diagonal, which is what the
    unwired code produced -- and it must disagree by exactly ``rate * V``.
    """
    mesh, sourced = _stokes_case(sources=(_LinearDrag(rate=3.5),))
    _, plain = _stokes_case()

    velocity = jnp.zeros((mesh.n_cells, mesh.dim))
    state = sourced.pack(velocity, jnp.zeros(mesh.n_cells))
    velocity_diag, _ = sourced.unpack(jnp.diag(jax.jacfwd(sourced.residual)(state)))
    shortfall = velocity_diag - plain.momentum_matrix_diagonal(velocity)

    assert jnp.allclose(shortfall, 3.5 * sourced.geometry.cell.volume[:, None], rtol=1e-9)


def test_a_source_diagonal_moves_the_rhie_chow_mass_flux() -> None:
    """The consequence that makes this a converged-answer defect, not a preconditioning one.

    ``a_P`` enters the mass flux only through the damping coefficient ``V / a_P`` -- the source term
    itself is nowhere in :meth:`mass_flux` -- so a difference here isolates that path exactly. It is
    non-zero only where the pressure is non-linear, which is why the state carries a quadratic
    pressure rather than a uniform one.
    """
    mesh, sourced = _stokes_case(sources=(_LinearDrag(rate=40.0),))
    _, plain = _stokes_case()

    x, y = sourced.geometry.cell.centroid[:, 0], sourced.geometry.cell.centroid[:, 1]
    state = sourced.pack(jnp.zeros((mesh.n_cells, mesh.dim)), x**2 + 2.0 * y**2)

    interior = mesh.face_cells.interior
    with_drag = sourced.mass_flux(state)[interior]
    without = plain.mass_flux(state)[interior]

    assert not jnp.allclose(with_drag, without, rtol=1e-6, atol=1e-12)


def test_the_shift_buckets_still_sum_to_the_isotropic_diagonal_with_a_source() -> None:
    """``convective + dissipative`` is documented to be the all-faces ``a_P``; a source must not break it.

    The two buckets are what a :class:`~aquaflux.solve.ShiftBasis` combines into the pseudo-transient
    shift, so a source diagonal that reached the total and not the split would leave a march damping
    by a diagonal the solve is not running -- the same defect one level down.
    """
    mesh, asm = _stokes_case(sources=(_LinearDrag(rate=3.5),))
    velocity = jnp.zeros((mesh.n_cells, mesh.dim))

    convective, dissipative = asm.momentum_matrix_diagonal_parts(velocity)
    isotropic = jnp.mean(asm.momentum_matrix_diagonal(velocity, boundary_corrected=False), axis=1)

    assert jnp.allclose(convective + dissipative, isotropic, rtol=1e-12, atol=0.0)


def test_a_sourceless_assembler_is_bit_identical_to_the_bare_face_assembly() -> None:
    """Without sources the wiring must be exactly absent, not merely negligible.

    ``a_P`` is on the hot path of every flow residual, and adding a zero array is not free in floating
    point once the array exists. Compared against :func:`~aquaflux.flow.rhie_chow.momentum_diagonal`
    itself -- the face-term assembly the method wraps -- so this pins that a sourceless assembler
    returns that value untouched, rather than merely that two sourceless assemblers agree.
    """
    mesh, plain = _stokes_case()
    velocity = jnp.linspace(-1.0, 1.0, mesh.n_cells * mesh.dim).reshape(mesh.n_cells, mesh.dim)

    # The all-faces form, whose face assembly takes no boundary correction and so needs nothing of
    # the assembler beyond its public geometry and viscosity.
    bare = momentum_diagonal(
        mesh.face_cells,
        plain.geometry,
        plain.viscosity,
        mdot_lagged=None,  # Stokes: no convective term
    )

    assert jnp.array_equal(plain.momentum_matrix_diagonal(velocity, boundary_corrected=False), bare)


def test_a_declared_face_force_is_refused_rather_than_silently_dropped() -> None:
    """The mass flux carries no force term yet, so declaring one must be an error.

    The balanced-force face treatment arrives with buoyancy, which it is entangled with through
    variable density. Until then a source returning a face force would have it dropped without a
    word, reintroducing the pressure--velocity decoupling Rhie--Chow exists to suppress -- so the
    seam stays declared and the omission is made loud.
    """
    mesh = structured_grid_2d(3, 3, named_boundaries=True)

    with pytest.raises(NotImplementedError, match="face_force"):
        MomentumContinuity.build(
            mesh,
            mesh.geometry(),
            PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
            CompactGreenGauss(),
            BoundaryConditions({name: NoSlipWall() for name in mesh.face_patches.names}),
            pressure_pin=0,
            sources=(_SpatiallyVaryingForce(),),
        )
