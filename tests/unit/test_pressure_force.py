"""Unit tests for :class:`~aquaflux.flow.PressureForce`, the momentum pressure term as a flux.

The pressure term of the momentum balance is the surface integral ``∮ p n dA``, so per face and
per component it is ``p_f n_i A``. Writing it as a
:class:`~aquaflux.discretization.FaceFluxOperator` is what lets each momentum component be
assembled by the same :class:`~aquaflux.discretization.CellBalance` as every other transport
equation. These check the closed form on a mesh whose normals and areas are known, that it does not
read the transported velocity component, and that it stays differentiable in the face pressure it
carries.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.discretization import FaceContext
from aquaflux.flow import PressureForce
from aquaflux.mesh import structured_grid_2d


@pytest.fixture
def context():
    """A context over a 2x2 unit grid; only its geometry is read by this operator."""
    mesh = structured_grid_2d(2, 2, lx=1.0, ly=1.0)
    geometry = mesh.geometry()
    return FaceContext(
        face_cells=mesh.face_cells,
        geometry=geometry,
        boundary_values=jnp.zeros(mesh.n_faces),
        gradient=jnp.zeros((mesh.n_cells, mesh.dim)),
        properties={},
    )


@pytest.mark.parametrize("component", [0, 1])
def test_face_flux_is_the_closed_form(context, component) -> None:
    """``p_f n_i A`` face by face, against the geometry's own normals and areas."""
    face = context.geometry.face
    face_pressure = jnp.linspace(0.5, 3.0, face.area.shape[0])

    flux = PressureForce(face_pressure, component).face_flux(
        jnp.zeros(context.face_cells.n_cells), context
    )

    assert jnp.array_equal(flux, face_pressure * face.normal[:, component] * face.area)


def test_the_force_does_not_read_the_transported_component(context) -> None:
    """Pressure is a different unknown, carried as a face value -- the velocity is not consulted.

    The pressure-velocity coupling is not lost by this: ``face_pressure`` is a differentiable
    function of the pressure unknowns, so the assembled residual still carries ``dR/dp``.
    """
    n_cells = context.face_cells.n_cells
    operator = PressureForce(jnp.linspace(0.5, 3.0, context.face_cells.n_faces), 0)

    zeros = operator.face_flux(jnp.zeros(n_cells), context)
    wild = operator.face_flux(jnp.linspace(-1e3, 1e3, n_cells), context)

    assert jnp.array_equal(zeros, wild)


def test_a_uniform_pressure_exerts_no_net_force_on_a_cell(context) -> None:
    """Scattered to cells, a constant face pressure cancels -- the closed-cell area-vector sum.

    The discrete statement of "uniform pressure produces no force": each cell's outward face area
    vectors sum to zero, so the scattered pressure force does too.
    """
    face_cells = context.face_cells
    uniform = jnp.full(face_cells.n_faces, 2.5)

    for component in range(context.geometry.face.normal.shape[1]):
        flux = PressureForce(uniform, component).face_flux(jnp.zeros(face_cells.n_cells), context)
        assert jnp.allclose(face_cells.scatter_conservative(flux), 0.0, atol=1e-12)


def test_the_force_is_differentiable_in_the_face_pressure(context) -> None:
    """Gradients flow through the carried face value, which is what makes ``dR/dp`` exact."""
    face = context.geometry.face
    n_cells = context.face_cells.n_cells

    def total(face_pressure):
        return jnp.sum(PressureForce(face_pressure, 0).face_flux(jnp.zeros(n_cells), context))

    grad = jax.grad(total)(jnp.linspace(0.5, 3.0, face.area.shape[0]))

    assert jnp.allclose(grad, face.normal[:, 0] * face.area)
