"""Tests for the k-omega SST transport-equation assembly.

The assembler is built on a small channel and each equation is solved with prescribed (frozen)
closure fields and no advection, checking that the equation is well-posed (the Newton residual
converges), the fields are finite, and the omega wall cells are fixed to the analytical value.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import VelocityFields
from aquaflux.mesh import CellZones, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel, ZoneConstant
from aquaflux.schemes import CorrectedGreenGauss, GradientScheme, ImposedGradient
from aquaflux.solve import RootSolver
from aquaflux.turbulence import (
    SSTClosureFields,
    SSTModel,
    SSTTurbulence,
    log_layer_shear_rate,
    omega_wall,
    omega_wall_gradient,
    production_and_limit,
)

NU = 1e-3


def _turbulence(*, explicit_production_limiter=False, gradient_scheme=None, properties=None):
    mesh = structured_grid_2d(6, 4, lx=3.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    turb = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        gradient_scheme or CorrectedGreenGauss(),
        FirstOrderUpwind(),
        (
            properties
            if properties is not None
            else PropertyModel({"viscosity": Constant(NU), "density": Constant(1.0)})
        ),
        wall_patches=["bottom", "top"],
        explicit_production_limiter=explicit_production_limiter,
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.01),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(10.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return mesh, turb


def _closure(turb):
    n = turb.mesh.n_cells
    return SSTClosureFields(
        nu_t=jnp.full(n, 0.01),
        strain_rate=jnp.full(n, 1.0),
        f1=jnp.full(n, 0.5),
        grad_k=jnp.zeros((n, 2)),
        grad_omega=jnp.zeros((n, 2)),
        omega=jnp.full(n, 1.0),
        k=jnp.full(n, 1.0),
        wall_shear_rate=jnp.full(turb.wall_cells.shape, 1.0),
        # A prescribed closure needs a prescribed imposition too; zero keeps this fixture's omega
        # gradient the flat field the rest of it is, without leaving the wall cells reconstructed.
        imposed_omega_gradient=ImposedGradient(
            turb.wall_cells, jnp.zeros((turb.wall_cells.shape[0], turb.mesh.dim))
        ),
    )


def _velocity(mesh, gradient):
    """A kinematic bundle carrying a prescribed gradient and a quiescent (zero) velocity."""
    return VelocityFields(
        velocity=jnp.zeros((mesh.n_cells, mesh.dim)),
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),
        gradient=gradient,
    )


def test_build_identifies_the_wall_adjacent_cells() -> None:
    """The bottom and top rows each contribute their cells to the omega fixation set."""
    mesh, turb = _turbulence()
    assert turb.wall_cells.shape[0] == 2 * 6  # bottom row + top row of a 6x4 grid
    assert turb.wall_distance.shape == (mesh.n_cells,)


def test_k_equation_solves_to_a_finite_bounded_field() -> None:
    """The equation is well-posed and solvable; the residual converges and the field is bounded.

    Strict positivity is *not* guaranteed by the raw solve (AD-Newton has no discrete maximum
    principle) -- it is secured by the realizability floor the driver applies between sweeps -- so
    this checks only convergence, finiteness, and a sensible magnitude.

    **This is the configuration the production limiter exists for, and the only measured case where
    it is load-bearing.** The solve is a bare ``RootSolver`` -- no preconditioner, no
    globalization -- and the cap is active in EVERY cell at the starting field (asserted below), so
    with the exact operator the k-Jacobian carries the cap's indefinite derivative everywhere and the
    unpreconditioned Newton stagnates rather than converging. Opting in drops that term and restores
    the M-matrix.

    The library default is the exact operator (``False``), because a *coupled* solve always carries a
    preconditioner and takes gradients, and freezing the cap there silently corrupts the adjoint
    wherever it binds at the root. Nothing about that default is contradicted here: this solve is
    unpreconditioned and differentiates nothing.
    """
    mesh, turb = _turbulence(explicit_production_limiter=True)
    exact_closure = _closure(turb)
    # The premise, pinned: the cap really is active at the starting field. If a future change makes
    # it inactive here, this test no longer needs the opt-in -- and this assertion is what will say so
    # rather than leaving the flag as cargo.
    production, limit = production_and_limit(
        exact_closure.nu_t,
        exact_closure.strain_rate,
        exact_closure.omega,
        jnp.full(mesh.n_cells, 0.01),
        turb.model,
    )
    assert bool(jnp.all(production > limit))

    residual = turb.k_residual(jnp.zeros(mesh.n_faces), _closure(turb))
    k = RootSolver(max_steps=30).solve(
        lambda phi, _: residual(phi), jnp.full(mesh.n_cells, 0.01), None
    )
    assert float(jnp.linalg.norm(residual(k))) < 1e-8  # the equation is solvable
    assert not bool(jnp.any(jnp.isnan(k)))
    assert float(jnp.max(jnp.abs(k))) < 0.1  # bounded near the inlet magnitude, no blow-up


def test_omega_equation_fixes_the_wall_cells_to_the_adaptive_value() -> None:
    """The wall cells are fixed to the adaptive (blended) near-wall omega, reading the closure ``k``.

    With the frozen closure's ``k = 1`` the log branch dominates the viscous one here, so the fixed
    value is the blend :func:`~aquaflux.turbulence.omega_wall`, not the bare viscous fixation.
    """
    mesh, turb = _turbulence()
    closure = _closure(turb)
    residual = turb.omega_residual(jnp.zeros(mesh.n_faces), closure)
    omega = RootSolver(max_steps=40).solve(
        lambda phi, _: residual(phi), jnp.full(mesh.n_cells, 10.0), None
    )
    assert float(jnp.linalg.norm(residual(omega))) < 1e-6
    expected = omega_wall(
        jnp.full(turb.wall_cells.shape[0], NU),
        turb.wall_distance[turb.wall_cells],
        closure.k[turb.wall_cells],
        SSTModel(),
    )
    assert jnp.allclose(omega[turb.wall_cells], expected)


def test_k_residual_is_differentiable_in_a_closure_field() -> None:
    """Gradient flows through the frozen eddy viscosity into the k residual, no NaNs."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k = jnp.full(n, 0.01)

    def loss(nu_t_scale):
        closure = _closure(turb)._replace(nu_t=nu_t_scale * jnp.full(n, 0.01))
        return jnp.sum(turb.k_residual(jnp.zeros(mesh.n_faces), closure)(k) ** 2)

    assert not bool(jnp.isnan(jax.grad(loss)(1.0)))


def _shear(n, gamma=2.0):
    """A uniform simple-shear velocity gradient (du_x/dy = gamma), so S = gamma."""
    return jnp.tile(jnp.array([[[0.0, gamma], [0.0, 0.0]]]), (n, 1, 1))


def test_eddy_viscosity_matches_the_model() -> None:
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k, omega = jnp.full(n, 0.01), jnp.full(n, 10.0)
    nu_t = turb.eddy_viscosity(_shear(n), k, omega)
    expected = SSTModel().eddy_viscosity(
        k, omega, jnp.full(n, 2.0), jnp.full(n, NU), turb.wall_distance
    )
    assert jnp.allclose(nu_t, expected)
    assert bool(jnp.all(nu_t > 0.0))


def test_closure_fields_are_well_formed() -> None:
    """The strain rate, blending function, gradients, and eddy viscosity are sensible."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k, omega = jnp.full(n, 0.01), jnp.full(n, 10.0)
    closure = turb.closure_fields(_velocity(mesh, _shear(n)), k, omega)
    interior = jnp.setdiff1d(jnp.arange(n), turb.wall_cells)
    assert jnp.allclose(
        closure.strain_rate[interior], 2.0
    )  # away from the wall, the reconstruction
    assert bool(jnp.all(closure.nu_t > 0.0))
    assert bool(jnp.all((closure.f1 >= 0.0) & (closure.f1 <= 1.0)))  # F1 = tanh(.) in [0, 1]
    assert closure.grad_k.shape == (n, mesh.dim)
    assert closure.grad_omega.shape == (n, mesh.dim)
    assert jnp.allclose(closure.omega, omega)


def test_the_wall_cells_omega_gradient_is_the_analytical_one() -> None:
    """Those cells' ``omega`` is imposed, so its gradient is a model quantity and not a reconstruction.

    ``omega_wall`` goes like ``1/d**2`` and the wall face's own ``omega`` comes from a zero-gradient
    closure, so a linear fit over that stencil is measured at about a quarter of the analytical
    magnitude. The equality here is exact because the analytical value is *imposed*, not approached.
    """
    mesh, turb = _turbulence()
    n = mesh.n_cells
    k, omega = jnp.full(n, 0.01), jnp.full(n, 10.0)
    closure = turb.closure_fields(_velocity(mesh, _shear(n)), k, omega)

    wall = turb.wall_cells
    expected = omega_wall_gradient(
        jnp.full(wall.shape[0], NU),
        turb.wall_distance[wall],
        k[wall],
        turb.wall_distance_gradient[wall],
        closure.grad_k[wall],
        SSTModel(),
    )
    assert jnp.array_equal(closure.grad_omega[wall], expected)


def test_the_imposed_wall_gradient_is_handed_to_the_scheme_not_applied_to_its_result() -> None:
    """Which is the whole point, for any scheme that consumes its own reconstructed gradient.

    :class:`~aquaflux.schemes.MultipleCorrectionGradient` differentiates its first estimate to build
    the Hessian that corrects the gradient it returns, so an imposition arriving afterwards would
    leave that Hessian -- and through it every other cell -- built on the near-wall value it
    replaces. Asserted at the seam rather than through a moved number, because the correction it
    feeds vanishes identically on a Cartesian grid and a numerical check here would be measuring the
    mesh.
    """
    handed = []

    class _Recording(GradientScheme):
        """Records the imposition it is handed, and reconstructs with the scheme it wraps."""

        inner: GradientScheme

        def _reconstruct_gradient(
            self,
            field,
            mesh,
            geometry,
            boundary_values,
            *,
            operator_hook=None,
            imposed=None,
            boundary_values_at=None,
            boundary_chain=None,
        ):
            handed.append(imposed)
            return self.inner.gradients(field, mesh, geometry, boundary_values)

    mesh, turb = _turbulence(gradient_scheme=_Recording(CorrectedGreenGauss()))
    n = mesh.n_cells
    turb.closure_fields(_velocity(mesh, _shear(n)), jnp.full(n, 0.01), jnp.full(n, 10.0))

    imposed = [one for one in handed if one is not None]
    assert len(imposed) == 1  # only omega's; every other field reconstructs freely
    assert any(one is None for one in handed)
    assert jnp.array_equal(imposed[0].cells, turb.wall_cells)


def test_eddy_viscosity_is_differentiable_in_k() -> None:
    """The k-derivative must be the RIGHT number, not merely a finite one.

    ``eddy_viscosity`` is used by the segregated driver's eddy-viscosity sweep, with no other
    gradient check anywhere -- a severed derivative along this path (the frozen strain-rate blend,
    the SST limiter) reports a finite ``0.0``, which an isfinite-only check cannot tell apart from
    the true, nonzero sensitivity checked here against a central finite difference.
    """
    mesh, turb = _turbulence()
    n = mesh.n_cells
    omega = jnp.full(n, 10.0)
    k0 = jnp.full(n, 0.01)

    def total(k):
        return jnp.sum(turb.eddy_viscosity(_shear(n), k, omega))

    def central_difference(k_point, step):
        return (total(k_point + step) - total(k_point - step)) / (2.0 * step)

    grad = jax.grad(total)(k0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(grad)) != 0.0  # a severed gradient reports 0.0, which is finite too
    eps = 1e-6
    assert float(jnp.sum(grad)) == pytest.approx(float(central_difference(k0, eps)), rel=1e-6)


# --- the adaptive near-wall seams the closure fields carry --------------------------------------


def test_wall_shear_rate_is_the_wall_normal_velocity_difference_over_the_distance() -> None:
    """|U_P - U_wall| / d at the wall-adjacent cells, measured against the patch's own velocity."""
    mesh, turb = _turbulence()
    speed = 3.0
    velocity = jnp.zeros((mesh.n_cells, mesh.dim)).at[:, 0].set(speed)
    fields = VelocityFields(
        velocity=velocity,
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),  # stationary no-slip walls
        gradient=_shear(mesh.n_cells),
    )
    got = turb.wall_shear_rate(fields)
    assert got.shape == turb.wall_cells.shape
    assert jnp.allclose(got, speed / turb.wall_distance[turb.wall_cells])


def test_wall_shear_rate_is_relative_to_a_moving_wall() -> None:
    """A cell moving with its wall has no wall-normal shear -- the difference is the relative one."""
    mesh, turb = _turbulence()
    speed = 3.0
    velocity = jnp.zeros((mesh.n_cells, mesh.dim)).at[:, 0].set(speed)
    fields = VelocityFields(
        velocity=velocity,
        boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)).at[:, 0].set(speed),
        gradient=_shear(mesh.n_cells),
    )
    assert jnp.allclose(turb.wall_shear_rate(fields), 0.0)


def test_wall_shear_rate_has_a_finite_derivative_at_zero_velocity() -> None:
    """A quiescent field sits exactly on the cone point of the vector magnitude -- no NaN there."""
    mesh, turb = _turbulence()
    zeros = jnp.zeros((mesh.n_cells, mesh.dim))

    def total(velocity):
        fields = VelocityFields(
            velocity=velocity,
            boundary_velocity=jnp.zeros((mesh.n_faces, mesh.dim)),
            gradient=_shear(mesh.n_cells),
        )
        return jnp.sum(turb.wall_shear_rate(fields))

    assert bool(jnp.all(jnp.isfinite(total(zeros))))
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(zeros))))


def test_strain_rate_blends_the_wall_cells_onto_the_log_layer_shear() -> None:
    """Resolved (tiny k): the reconstruction, untouched. Log layer (large k): the log-law shear.

    Only the wall-adjacent cells move; the interior keeps the reconstructed strain either way.
    """
    mesh, turb = _turbulence()
    n = mesh.n_cells
    wall = turb.wall_cells
    interior = jnp.setdiff1d(jnp.arange(n), wall)
    d = turb.wall_distance[wall]

    resolved = turb.strain_rate(_shear(n), jnp.full(n, 1e-12))
    assert jnp.allclose(resolved, 2.0, rtol=1e-6)  # entirely inside the sublayer: no change

    k = jnp.full(n, 30.0)  # far into the log layer
    got = turb.strain_rate(_shear(n), k)
    assert jnp.allclose(got[interior], 2.0)
    assert jnp.allclose(got[wall], log_layer_shear_rate(d, k[wall], turb.model), rtol=1e-3)


def test_strain_rate_is_differentiable_in_k() -> None:
    """The blend is live in the coupled residual, so its k-derivative must be finite -- also at k = 0."""
    mesh, turb = _turbulence()
    n = mesh.n_cells
    for k in (jnp.zeros(n), jnp.full(n, 30.0)):
        g = jax.grad(lambda kk: jnp.sum(turb.strain_rate(_shear(n), kk)))(k)
        assert bool(jnp.all(jnp.isfinite(g)))


# --- build() deriving density / molecular_viscosity from a PropertyModel (issue #353) ---------


def test_build_derives_kinematic_viscosity_from_the_property_model() -> None:
    """``molecular_viscosity = viscosity / density``, both read from one ``PropertyModel``."""
    _, turb = _turbulence(
        properties=PropertyModel({"viscosity": Constant(2.0), "density": Constant(2.0)})
    )
    assert jnp.allclose(turb.density, 2.0)
    assert jnp.allclose(turb.molecular_viscosity, 1.0)


def test_a_non_uniform_zoneconstant_density_raises_before_reaching_the_residual() -> None:
    """A per-cell density would silently mis-scale the volume flux -- caught here instead."""
    mesh = structured_grid_2d(6, 4, lx=3.0, ly=1.0, named_boundaries=True)
    n = mesh.n_cells
    zones = CellZones.from_dict(n, {"lo": [0], "hi": list(range(1, n))})
    mesh = eqx.tree_at(lambda m: m.cell_zones, mesh, zones)
    geometry = mesh.geometry()
    properties = PropertyModel(
        {
            "viscosity": Constant(NU),
            "density": ZoneConstant.from_dict(zones, {"lo": 1.0, "hi": 2.0}),
        }
    )
    with pytest.raises(ValueError, match=r"density.*must be uniform"):
        SSTTurbulence.build(
            SSTModel(),
            mesh,
            geometry,
            CorrectedGreenGauss(),
            FirstOrderUpwind(),
            properties,
            wall_patches=["bottom", "top"],
            k_boundary=BoundaryConditions(
                {
                    "left": Dirichlet(0.01),
                    "right": ZeroGradient(),
                    "bottom": Dirichlet(0.0),
                    "top": Dirichlet(0.0),
                }
            ),
            omega_boundary=BoundaryConditions(
                {
                    "left": Dirichlet(10.0),
                    "right": ZeroGradient(),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
        )


def test_with_scaled_molecular_viscosity_is_still_a_tree_at_target_with_a_nonzero_gradient() -> (
    None
):
    """``molecular_viscosity`` stays its own differentiable leaf, independent of ``properties``.

    After a homotopy ramp station it deliberately no longer equals the property model's own
    ``viscosity / density`` (see the class docstring), so the rescale must move the stored array
    directly rather than through a live property -- and stay reachable by ``jax.grad``, not merely
    finite.
    """
    _, turb = _turbulence()

    def total(factor):
        return jnp.sum(turb.with_scaled_molecular_viscosity(factor).molecular_viscosity)

    grad = jax.grad(total)(1.0)
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
