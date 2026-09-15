"""Unit tests for the volume-source seam of the residual assembler.

The :class:`VolumeSource` contract is exercised with stub sources (no physics) so the
assembler plumbing is verified independently: a source is subtracted from the balance with the
right sign, several sources sum, a source composes additively with the flux terms, and a
field-dependent source integrates over the cell volume and stays differentiable in both the
solved field and its own coefficient.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler, VolumeSource
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel

STEP = 1e-6


class _ConstantSource(VolumeSource):
    """A uniform, already-integrated source per cell (production positive)."""

    value: float

    def source(self, field, context):
        return jnp.full(field.shape[0], self.value, dtype=field.dtype)


class _LinearSink(VolumeSource):
    """A first-order sink ``-rate * phi`` integrated over the cell volume.

    Exercises the two things a real source does: it depends on the solved ``field`` and it bakes
    in its own volume quadrature by reading ``context.mesh.geometry.cell.volume``.
    """

    rate: float

    def source(self, field, context):
        return -self.rate * field * context.mesh.geometry.cell.volume


def _assembler(source_operators, *, flux_operators=(), properties=None):
    """A source-only (by default) steady assembler on a two-cell grid."""
    mesh = structured_grid_2d(2, 1)
    properties = PropertyModel({}) if properties is None else properties
    asm = ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        properties,
        flux_operators,
        BoundaryConditions({"boundary": ZeroGradient()}),
        source_operators=source_operators,
    )
    return mesh, asm


def test_constant_source_is_subtracted_with_correct_sign() -> None:
    """With no flux, the steady residual is exactly minus the integrated source (a sink of +3)."""
    mesh, asm = _assembler((_ConstantSource(value=3.0),))
    residual = asm.residual(jnp.zeros(mesh.n_cells))
    assert jnp.allclose(residual, -3.0)


def test_multiple_sources_sum() -> None:
    """Injected sources accumulate: production 2 and 5 leave a combined sink of 7."""
    mesh, asm = _assembler((_ConstantSource(value=2.0), _ConstantSource(value=5.0)))
    residual = asm.residual(jnp.zeros(mesh.n_cells))
    assert jnp.allclose(residual, -7.0)


def test_source_composes_additively_with_flux() -> None:
    """Adding a source shifts the flux-only residual by exactly minus that source, nothing else."""
    properties = PropertyModel({"diffusivity": Constant(1.0)})
    _, asm_flux = _assembler((), flux_operators=(DiffusionFlux(),), properties=properties)
    _, asm_both = _assembler(
        (_ConstantSource(value=4.0),), flux_operators=(DiffusionFlux(),), properties=properties
    )
    phi = jnp.array([1.0, 4.0])  # a non-uniform field so the diffusion flux is non-zero
    assert jnp.allclose(asm_both.residual(phi), asm_flux.residual(phi) - 4.0)


def test_field_dependent_sink_integrates_over_volume() -> None:
    """A ``-rate * phi`` sink yields residual ``+rate * phi * V`` (no flux, steady)."""
    mesh, asm = _assembler((_LinearSink(rate=2.0),))
    volume = mesh.geometry().cell.volume
    phi = jnp.array([1.0, 3.0])
    assert jnp.allclose(asm.residual(phi), 2.0 * phi * volume)


def test_source_is_differentiable_in_field_and_coefficient() -> None:
    """``jax.grad`` through a source's solved field and its own coefficient matches a central
    finite difference -- not merely finite (a ``stop_gradient`` around the source term returns
    an exact zero here, which is finite and would pass a bare NaN check)."""
    mesh = structured_grid_2d(2, 1)
    geometry = mesh.geometry()

    def build(rate):
        return ResidualAssembler.build(
            mesh,
            geometry,
            PropertyModel({}),
            (),
            BoundaryConditions({"boundary": ZeroGradient()}),
            source_operators=(_LinearSink(rate=rate),),
        )

    phi = jnp.array([1.0, 3.0])

    def loss_field(p):
        return jnp.sum(build(2.0).residual(p) ** 2)

    def loss_rate(r):
        return jnp.sum(build(r).residual(phi) ** 2)

    grad_field = jax.grad(loss_field)(phi)
    for index in range(phi.shape[0]):
        finite_difference = float(
            (loss_field(phi.at[index].add(STEP)) - loss_field(phi.at[index].add(-STEP)))
            / (2.0 * STEP)
        )
        assert float(grad_field[index]) == pytest.approx(finite_difference, rel=1e-6)
    assert not bool(jnp.any(jnp.isnan(grad_field)))
    assert float(jnp.abs(grad_field).max()) > 1e-6, (
        "a severed adjoint would report zero and pass a finiteness check"
    )

    grad_rate = jax.grad(loss_rate)(2.0)
    finite_difference_rate = float((loss_rate(2.0 + STEP) - loss_rate(2.0 - STEP)) / (2.0 * STEP))
    assert grad_rate == pytest.approx(finite_difference_rate, rel=1e-6)
    assert abs(grad_rate) > 1e-6, "a severed adjoint would report zero and pass a finiteness check"
