"""Unit tests for the flow coupling accessors used by scalar transport and turbulence.

``MomentumContinuity.mass_flux`` must return exactly the Rhie--Chow mass flux the continuity
residual is built from (so a transported scalar advects on the same conservative flux), and
``velocity_fields`` must return the kinematic bundle a turbulence model reads. Both are checked without a
solve -- they are functions of the state. Also the shared :meth:`MomentumContinuity.flow_fields`
seam: the accessors and the residual read off one assembly so a coupling caller re-derives the
boundary fields, gradients, and Rhie--Chow flux once rather than per accessor.

The reverse direction -- how a RANS closure's eddy viscosity enters the momentum block -- is
``with_eddy_viscosity`` / ``viscosity``, checked here on a fluid with ``rho != 1`` so a missing
density factor in ``mu_eff = mu + rho nu_t`` cannot pass.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss


def _count_rhie_chow_assemblies(monkeypatch):
    """Return a mutable ``[count]`` incremented on every lagged-``a_P`` Rhie--Chow assembly.

    ``momentum_matrix_diagonal`` is the expensive nonlinear diagonal only the full
    :meth:`~aquaflux.flow.MomentumContinuity.flow_fields` assembly forms, so its call count is a
    proxy for how many times the Rhie--Chow flux is assembled in an evaluation.
    """
    calls = [0]
    original = MomentumContinuity.momentum_matrix_diagonal

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MomentumContinuity, "momentum_matrix_diagonal", counted)
    return calls


def _uniform_inlet(centroid):
    n = centroid.shape[0]
    return jnp.stack([jnp.ones(n), jnp.zeros(n)], axis=1)


def _assembler():
    mesh = structured_grid_2d(4, 3, lx=3.0, ly=1.0, named_boundaries=True)
    geom = mesh.geometry()
    asm = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(0.1), "density": Constant(1.0)}),
        CorrectedGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=_uniform_inlet),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
    )
    return mesh, asm


def _arbitrary_state(mesh):
    """A deterministic non-trivial flat state (mass-flux consistency holds for any state)."""
    n = (mesh.dim + 1) * mesh.n_cells
    return 0.2 + 0.1 * jnp.sin(jnp.arange(n, dtype=jnp.float64))


def test_mass_flux_is_the_continuity_mdot() -> None:
    """mass_flux(state) is exactly the mdot the continuity residual scatters (no pin here)."""
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    _, pressure_residual = asm.unpack(asm.residual(state))
    mdot = asm.mass_flux(state)
    assert jnp.allclose(pressure_residual, mesh.face_cells.scatter_conservative(mdot))


def test_velocity_fields_shape_and_differentiable() -> None:
    """The bundle's arrays have the documented shapes, are finite, and grad flows through them."""
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    fields = asm.velocity_fields(state)
    assert fields.gradient.shape == (mesh.n_cells, mesh.dim, mesh.dim)
    assert fields.velocity.shape == (mesh.n_cells, mesh.dim)
    assert fields.boundary_velocity.shape == (mesh.n_faces, mesh.dim)
    assert not bool(jnp.any(jnp.isnan(fields.gradient)))
    g = jax.grad(lambda s: jnp.sum(asm.velocity_fields(s).gradient ** 2))(state)
    assert not bool(jnp.any(jnp.isnan(g)))


def test_flow_fields_accessors_agree_with_the_bundle() -> None:
    """mass_flux / velocity_fields / residual read the same values a single flow_fields yields."""
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    fields = asm.flow_fields(state)
    assert jnp.array_equal(asm.mass_flux(state), fields.mdot)
    assert jnp.array_equal(asm.velocity_fields(state).gradient, fields.velocity_fields.gradient)
    assert jnp.array_equal(asm.residual(state), asm.residual_from_fields(fields))


def test_velocity_fields_skips_the_rhie_chow_assembly(monkeypatch) -> None:
    """velocity_fields is the lightweight reconstruction: it does not build the lagged a_P / mdot.

    A segregated sweep needs the eddy viscosity (from the gradient) before the mass flux is even
    defined, so ``velocity_fields`` must not drag the whole Rhie--Chow assembly along; ``flow_fields``
    and ``mass_flux`` do exactly one such assembly.
    """
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    calls = _count_rhie_chow_assemblies(monkeypatch)

    calls[0] = 0
    asm.velocity_fields(state)
    assert calls[0] == 0  # kinematic only -- no a_P / Rhie--Chow work

    calls[0] = 0
    asm.flow_fields(state)
    assert calls[0] == 1

    calls[0] = 0
    asm.mass_flux(state)
    assert calls[0] == 1


def test_body_force_is_a_uniform_volume_source() -> None:
    """A uniform body force enters the momentum residual as ``-beta * volume`` per component and
    only there: at the zero state the streamwise block is exactly ``-beta * V`` and the rest is
    zero, and the force is a differentiable leaf (a mass-flow controller updates it)."""
    beta = 3.0
    mesh = structured_grid_2d(6, 4, periodic=("x",), named_boundaries=True)
    geom = mesh.geometry()
    asm = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CorrectedGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        pressure_pin=0,
        body_force=(beta, 0.0),
    )
    velocity_residual, _ = asm.unpack(asm.residual(asm.initial_state()))
    volume = jnp.asarray(geom.cell.volume)
    assert jnp.allclose(velocity_residual[:, 0], -beta * volume)
    assert jnp.allclose(velocity_residual[:, 1], 0.0)

    # The force is a differentiable leaf: sensitivity of the summed streamwise residual to beta is -V.
    def summed_streamwise_residual(b):
        assembler = MomentumContinuity.build(
            mesh,
            geom,
            PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
            CorrectedGreenGauss(),
            BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
            pressure_pin=0,
            body_force=jnp.array([b, 0.0]),
        )
        return jnp.sum(assembler.residual(asm.initial_state())[: mesh.n_cells])

    grad = jax.grad(summed_streamwise_residual)(beta)
    assert jnp.isclose(grad, -jnp.sum(volume))


# --- the turbulence coupling seam: nu_t -> the momentum diffusion coefficient -------------------

RHO_T, MU_T = 2.0, 0.6  # rho != 1, so a dropped density factor cannot pass


def _turbulent_assembler():
    """A closed box with ``rho != 1`` and a known molecular ``mu``."""
    mesh = structured_grid_2d(4, 3, lx=3.0, ly=1.0, named_boundaries=True)
    asm = MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(MU_T), "density": Constant(RHO_T)}),
        CorrectedGreenGauss(),
        BoundaryConditions({n: NoSlipWall() for n in ("left", "right", "bottom", "top")}),
    )
    return mesh, asm


def test_without_a_closure_the_viscosity_is_the_molecular_value() -> None:
    """No eddy viscosity supplied -- the flow is laminar and sees only the fluid's own ``mu``."""
    _, asm = _turbulent_assembler()
    assert asm.eddy_viscosity is None
    assert jnp.allclose(asm.viscosity, MU_T)


def test_eddy_viscosity_forms_the_effective_diffusion_coefficient() -> None:
    """``viscosity`` is ``mu + rho nu_t`` -- the closure supplies kinematic ``nu_t``, not ``mu_t``."""
    mesh, asm = _turbulent_assembler()
    nu_t = jnp.linspace(0.0, 5.0, mesh.n_cells)
    assert jnp.allclose(asm.with_eddy_viscosity(nu_t).viscosity, MU_T + RHO_T * nu_t)


def test_eddy_viscosity_is_idempotent_and_keeps_the_molecular_value() -> None:
    """Re-applying a closure must not accumulate: the molecular viscosity stays the material one.

    The eddy contribution rides on its own leaf rather than overwriting the material properties, so
    ``properties`` still describes the fluid after a swap and a second swap replaces rather than adds.
    """
    mesh, asm = _turbulent_assembler()
    nu_t = jnp.full(mesh.n_cells, 4.0)
    once = asm.with_eddy_viscosity(nu_t)
    twice = once.with_eddy_viscosity(nu_t)
    assert jnp.allclose(twice.viscosity, once.viscosity)
    assert jnp.allclose(once.properties.evaluate(mesh.cell_zones)["viscosity"], MU_T)
    assert jnp.allclose(asm.viscosity, MU_T)  # the original assembler is unchanged


def test_eddy_viscosity_is_a_differentiable_leaf() -> None:
    """A coupled residual computes ``nu_t`` from ``(k, omega)``, so gradients must flow through it."""
    mesh, asm = _turbulent_assembler()
    grad = jax.grad(lambda nu_t: jnp.sum(asm.with_eddy_viscosity(nu_t).viscosity))(
        jnp.full(mesh.n_cells, 1.0)
    )
    assert jnp.allclose(grad, RHO_T)  # d(mu + rho nu_t)/d(nu_t) = rho
