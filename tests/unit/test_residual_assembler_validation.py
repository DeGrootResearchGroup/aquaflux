"""Unit tests for the build-time validation `ResidualAssembler.build` performs.

Two failure modes used to surface only deep inside a jitted residual evaluation: a flux operator
naming a property the model does not supply (a bare ``KeyError``), and an advection scheme whose
whole reconstruction depends on a gradient silently falling back to first order when no
``gradient_scheme`` is injected. Both are now build-time ``ValueError``s, driven by two operator
methods -- :meth:`~aquaflux.discretization.face_flux.FaceFluxOperator.requires` (named properties)
and :meth:`~aquaflux.discretization.face_flux.FaceFluxOperator.uses_gradient` -- that let an
operator declare what it needs rather than the assembler guessing from its type.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Convective, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    FirstOrderUpwind,
    LimitedUpwind,
    ResidualAssembler,
    VolumeSource,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss


class _NamedPropertySource(VolumeSource):
    """A stub source that names a property it reads, to exercise the ``requires`` seam."""

    name: str

    def requires(self):
        return (self.name,)

    def source(self, field, context):
        return context.properties[self.name]


def _build(
    *,
    flux_operators,
    properties=None,
    gradient_scheme=None,
    source_operators=(),
    boundary=None,
    coefficient="diffusivity",
):
    mesh = structured_grid_2d(2, 1)
    properties = PropertyModel({}) if properties is None else properties
    boundary = BoundaryConditions({"boundary": ZeroGradient()}) if boundary is None else boundary
    return ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        properties,
        flux_operators,
        boundary,
        source_operators=source_operators,
        gradient_scheme=gradient_scheme,
        coefficient=coefficient,
    )


# --- requires() -----------------------------------------------------------------------


def test_build_raises_when_a_flux_operator_names_a_missing_property() -> None:
    with pytest.raises(ValueError, match="conductivity"):
        _build(
            flux_operators=(DiffusionFlux(coefficient="conductivity"),),
            properties=PropertyModel({"diffusivity": Constant(1.0)}),
        )


def test_build_succeeds_when_the_named_property_is_present() -> None:
    _build(
        flux_operators=(DiffusionFlux(coefficient="conductivity"),),
        properties=PropertyModel({"conductivity": Constant(1.0)}),
    )  # no raise


def test_build_does_not_require_any_property_when_no_operator_names_one() -> None:
    """A coefficient-free residual (e.g. a wall-distance gradient reconstruction) needs nothing."""
    _build(flux_operators=(), properties=PropertyModel({}))  # no raise


def test_build_raises_when_a_volume_source_names_a_missing_property() -> None:
    with pytest.raises(ValueError, match="reaction_rate"):
        _build(
            flux_operators=(),
            source_operators=(_NamedPropertySource(name="reaction_rate"),),
            properties=PropertyModel({}),
        )


def test_build_lists_every_missing_property_from_every_operator() -> None:
    with pytest.raises(ValueError) as excinfo:
        _build(
            flux_operators=(DiffusionFlux(coefficient="conductivity"),),
            source_operators=(_NamedPropertySource(name="reaction_rate"),),
            properties=PropertyModel({}),
        )
    message = str(excinfo.value)
    assert "conductivity" in message and "reaction_rate" in message


# --- boundary-closure requires_coefficient() (#360) ------------------------------------


def test_build_raises_when_a_neumann_closure_needs_an_unset_coefficient() -> None:
    with pytest.raises(ValueError, match="diffusivity"):
        _build(
            flux_operators=(),
            boundary=BoundaryConditions({"boundary": Neumann(flux=2.0)}),
            properties=PropertyModel({}),
        )


def test_build_raises_when_a_convective_closure_needs_an_unset_coefficient() -> None:
    with pytest.raises(ValueError, match="diffusivity"):
        _build(
            flux_operators=(),
            boundary=BoundaryConditions({"boundary": Convective(h=10.0, t_inf=1.0)}),
            properties=PropertyModel({}),
        )


def test_build_succeeds_when_a_flux_type_closure_has_its_coefficient() -> None:
    _build(
        flux_operators=(),
        boundary=BoundaryConditions({"boundary": Neumann(flux=2.0)}),
        properties=PropertyModel({"diffusivity": Constant(1.0)}),
    )  # no raise


def test_build_does_not_require_a_coefficient_for_value_or_zero_gradient_closures() -> None:
    """The documented pure-advection case: no flux-type closure, so the fallback stays legal."""
    _build(
        flux_operators=(),
        boundary=BoundaryConditions({"boundary": ZeroGradient()}),
        properties=PropertyModel({}),
    )  # no raise
    _build(
        flux_operators=(),
        boundary=BoundaryConditions({"boundary": Dirichlet(1.0)}),
        properties=PropertyModel({}),
    )  # no raise


def test_a_mistyped_coefficient_name_is_now_caught_at_build_rather_than_NaNing() -> None:
    """Regression for #360's own reproduction: a typo'd ``coefficient=`` used to build cleanly and
    NaN (a divide by the zero fallback) inside the residual. It must now raise at build."""
    with pytest.raises(ValueError, match="difusivity"):
        _build(
            flux_operators=(DiffusionFlux(coefficient="diffusivity"),),
            boundary=BoundaryConditions({"boundary": Convective(h=10.0, t_inf=1.0)}),
            properties=PropertyModel({"diffusivity": Constant(5.0)}),
            coefficient="difusivity",
        )


def test_the_correctly_named_case_still_evaluates_to_a_finite_residual() -> None:
    """Same configuration as the regression above with the name fixed: the fix only adds a
    build-time check -- the boundary Gamma lookup itself is untouched."""
    asm = _build(
        flux_operators=(DiffusionFlux(coefficient="diffusivity"),),
        boundary=BoundaryConditions({"boundary": Convective(h=10.0, t_inf=1.0)}),
        properties=PropertyModel({"diffusivity": Constant(5.0)}),
    )
    residual = asm.residual(jnp.array([1.0, 3.0]))
    assert bool(jnp.all(jnp.isfinite(residual)))


# --- uses_gradient() --------------------------------------------------------------------


def test_build_raises_when_limited_upwind_has_no_gradient_scheme() -> None:
    """The message names the offending FLUX OPERATOR (AdvectionFlux), not its nested scheme --
    that is the granularity `uses_gradient` is declared at, the same as `requires`.
    """
    mesh = structured_grid_2d(2, 1)
    with pytest.raises(ValueError, match=r"AdvectionFlux.*gradient"):
        _build(
            flux_operators=(
                AdvectionFlux(mass_flux=jnp.zeros(mesh.n_faces), scheme=LimitedUpwind()),
            ),
            gradient_scheme=None,
        )


def test_build_succeeds_when_limited_upwind_has_a_gradient_scheme() -> None:
    mesh = structured_grid_2d(2, 1)
    _build(
        flux_operators=(AdvectionFlux(mass_flux=jnp.zeros(mesh.n_faces), scheme=LimitedUpwind()),),
        gradient_scheme=CompactGreenGauss(),
    )  # no raise


def test_build_does_not_require_a_gradient_scheme_for_first_order_upwind() -> None:
    mesh = structured_grid_2d(2, 1)
    _build(
        flux_operators=(
            AdvectionFlux(mass_flux=jnp.zeros(mesh.n_faces), scheme=FirstOrderUpwind()),
        ),
        gradient_scheme=None,
    )  # no raise


def test_build_does_not_require_a_gradient_scheme_for_diffusion_alone() -> None:
    """DiffusionFlux reads the gradient too, but degrades gracefully (Gate A), so it opts out."""
    _build(
        flux_operators=(DiffusionFlux(),),
        properties=PropertyModel({"diffusivity": Constant(1.0)}),
        gradient_scheme=None,
    )  # no raise
