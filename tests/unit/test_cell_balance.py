"""Unit tests for :class:`~aquaflux.discretization.CellBalance`, the operator half of a residual.

The point of the class is that it needs nothing but its operators and a
:class:`~aquaflux.discretization.FaceContext`: no boundary conditions, no property model, no
gradient scheme, and no assembler. So every test here hands it a context built by hand -- which is
also the seam a coupled system uses, since the momentum block forms its own context and drives a
balance directly.

Stub operators (no physics) isolate the plumbing: the owner-outward scatter and its sign, the
source subtraction, the accumulation, and the **summation order**, which is part of the arithmetic
because floating-point addition is not associative.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.discretization import (
    CellBalance,
    FaceContext,
    FaceFluxOperator,
    TransientTerm,
    VolumeSource,
)
from aquaflux.mesh import structured_grid_2d


class _ConstantFlux(FaceFluxOperator):
    """A prescribed owner-outward flux per face, independent of the transported field."""

    value: float

    def face_flux(self, field, context):
        return jnp.full(context.face_cells.n_faces, self.value, dtype=field.dtype)


class _ConstantSource(VolumeSource):
    """A uniform, already-integrated source per cell (production positive)."""

    value: float

    def source(self, field, context):
        return jnp.full(field.shape[0], self.value, dtype=field.dtype)


@pytest.fixture
def two_cells():
    """A two-cell grid and a context over it, with no boundary values and a zero gradient."""
    mesh = structured_grid_2d(2, 1)
    geometry = mesh.geometry()
    context = FaceContext(
        face_cells=mesh.face_cells,
        geometry=geometry,
        boundary_values=jnp.zeros(mesh.n_faces),
        gradient=jnp.zeros((mesh.n_cells, mesh.dim)),
        properties={},
    )
    return mesh, context


def test_a_balance_needs_no_mesh_no_boundary_and_no_properties(two_cells) -> None:
    """The whole object is its operators: everything else arrives on the context it is handed."""
    mesh, context = two_cells
    balance = CellBalance((_ConstantFlux(value=1.0),))

    residual = balance.residual(jnp.zeros(mesh.n_cells), context)

    assert residual.shape == (mesh.n_cells,)


def test_flux_scatters_owner_positive_and_neighbour_negative(two_cells) -> None:
    """A uniform owner-outward flux sums into the owner and out of the interior neighbour.

    Each cell of the 2x1 grid has four faces; the single interior face is owned by one of them, so
    the two cells' balances differ by exactly twice that face's flux and their sum is the net flux
    through the boundary (the interior face cancels -- the conservation statement).
    """
    mesh, context = two_cells
    balance = CellBalance((_ConstantFlux(value=1.0),))

    residual = balance.residual(jnp.zeros(mesh.n_cells), context)

    n_boundary = int(jnp.sum(~mesh.face_cells.interior))
    assert float(jnp.sum(residual)) == pytest.approx(n_boundary)
    assert float(residual[0] - residual[1]) == pytest.approx(2.0)


def test_sources_are_subtracted_and_sum(two_cells) -> None:
    """Volume sources leave the balance as sinks, and several compose additively."""
    mesh, context = two_cells
    phi = jnp.zeros(mesh.n_cells)
    one = CellBalance((), (_ConstantSource(value=3.0),))
    both = CellBalance((), (_ConstantSource(value=3.0), _ConstantSource(value=1.5)))

    assert jnp.allclose(one.residual(phi, context), -3.0)
    assert jnp.allclose(both.residual(phi, context), -4.5)


def test_the_transient_reads_cell_volumes_from_the_context(two_cells) -> None:
    """The accumulation term is added, integrated on the context's own cell volumes.

    A first (BDF1) step of ``d(phi)/dt`` from ``phi_old`` to ``phi`` contributes
    ``V (phi - phi_old) / dt`` per cell.
    """
    mesh, context = two_cells
    balance = CellBalance((), (), TransientTerm())
    phi = jnp.full(mesh.n_cells, 2.0)
    phi_old = jnp.full(mesh.n_cells, 1.0)

    residual = balance.residual(phi, context, phi_old, phi_old, dt=0.5, first_step=True)

    assert jnp.allclose(residual, context.geometry.cell.volume * (2.0 - 1.0) / 0.5)


def test_an_empty_balance_is_zero(two_cells) -> None:
    """No operators, no terms: the flux accumulator starts at zero and stays there."""
    mesh, context = two_cells

    residual = CellBalance(()).residual(jnp.ones(mesh.n_cells), context)

    assert jnp.array_equal(residual, jnp.zeros(mesh.n_cells))


def test_operators_are_summed_in_tuple_order(two_cells) -> None:
    """The tuple order is the summation order, and it is visible in the last bits.

    Floating-point addition is not associative, so reordering the operators of a balance perturbs
    the residual. That makes the order part of the arithmetic rather than a presentational choice —
    the momentum block relies on it to keep its viscous-pressure-advective sum unchanged. These
    magnitudes are chosen so the two orders differ by exactly one unit in the last place: adding
    both tiny values to 1.0 in turn absorbs each, while adding them to each other first does not.
    """
    mesh, context = two_cells
    phi = jnp.zeros(mesh.n_cells)
    tiny = _ConstantFlux(value=1e-16)
    big = _ConstantFlux(value=1.0)

    big_first = CellBalance((big, tiny, tiny)).residual(phi, context)
    tiny_first = CellBalance((tiny, tiny, big)).residual(phi, context)

    assert not jnp.array_equal(big_first, tiny_first)
    assert jnp.allclose(big_first, tiny_first)


def test_the_assembler_delegates_to_its_balance(two_cells) -> None:
    """A built assembler's residual is exactly its balance evaluated on the context it forms."""
    from aquaflux.boundary import BoundaryConditions, ZeroGradient
    from aquaflux.discretization import DiffusionFlux, ResidualAssembler
    from aquaflux.properties import Constant, PropertyModel

    mesh, _ = two_cells
    assembler = ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        BoundaryConditions({"boundary": ZeroGradient()}),
    )
    phi = jnp.array([1.0, 4.0])

    properties = assembler.properties.evaluate(mesh.cell_zones)
    gradient, boundary_values = assembler._gradient(phi, properties)
    context = assembler._context(gradient, boundary_values, properties)

    assert jnp.array_equal(assembler.residual(phi), assembler.balance.residual(phi, context))
