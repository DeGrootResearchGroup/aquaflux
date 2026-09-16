"""The convection-diffusion AMG preconditioner for a scalar transport equation.

Exercises :func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner` on a synthetic
convection-dominated advection-diffusion scalar (a uniform streamwise flux, no turbulence needed), the
same operator family the k/omega transport linearizes to. The preconditioner must reconstruct that
operator well enough that a single V-cycle strongly contracts the error, and the contraction must stay
bounded as the mesh refines (mesh-independent) -- the property that keeps the scalar solve's iteration
count from growing with problem size at high cell Peclet.
"""

from __future__ import annotations

import dataclasses

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    FirstOrderUpwind,
    ResidualAssembler,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, FieldProperty, PropertyModel
from aquaflux.turbulence.preconditioner import (
    AirAmgPreconditioner,
    ConvectionAmgPreconditioner,
    ScalarAir,
    ScalarTwoLevel,
    UnpreconditionedScalars,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)

U, GAMMA = 1.0, 1e-3  # cell Peclet ~ U dx / Gamma is large -> convection-dominated


def _transport(nx, ny):
    """A scalar advected by a uniform streamwise flux and diffusing with constant ``GAMMA``."""
    mesh = structured_grid_2d(nx, ny, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    volume_flux = U * geometry.face.normal[:, 0] * geometry.face.area  # uniform u = (U, 0)
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(GAMMA)}),
        (AdvectionFlux(volume_flux, FirstOrderUpwind()), DiffusionFlux()),
        BoundaryConditions(
            {
                "left": Dirichlet(1.0),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
    )
    return mesh, geometry, volume_flux, assembler.residual


def test_scalar_shift_diagonal_is_the_operator_diagonal_under_graded_diffusivity() -> None:
    """The scalar pseudo-time shift diagonal equals ``diag(J)`` for a pure-diffusion graded scalar (#154).

    For a pure diffusion operator (no advection, no reaction) the shift is built to be exactly the
    operator diagonal: the interior conductance scattered to both cells plus the ``J . 1`` boundary
    stiffness. That identity only holds when the interior conductance is the operator's own (the
    flux-continuous, harmonic-on-graded-``Gamma`` form) — the g-weighted arithmetic coefficient it
    replaced over-counts a graded face and breaks it. Constant ``Gamma`` is byte-identical either way,
    which is why no prior test caught this.
    """
    mesh = structured_grid_2d(6, 5, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    x = geometry.cell.centroid[:, 0]
    gamma = jnp.exp(2.5 * (x - x.min()) / (x.max() - x.min()))  # graded ~12x across the domain
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": FieldProperty(gamma)}),
        (DiffusionFlux(),),  # pure diffusion: shift diagonal must reproduce diag(J) exactly
        BoundaryConditions(
            {
                "left": Dirichlet(1.0),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
    )
    reference = jnp.zeros(mesh.n_cells)
    shift = scalar_transport_shift_diagonal(
        mesh, geometry, gamma, jnp.zeros(mesh.n_faces), assembler.residual, reference
    )
    diag_j = jnp.diag(jax.jacfwd(assembler.residual)(reference))
    assert jnp.allclose(shift, diag_j, rtol=1e-9, atol=1e-9)


def _contraction(nx, ny, scalar, *, seed=0):
    mesh, geometry, volume_flux, residual = _transport(nx, ny)
    reference = jnp.ones(mesh.n_cells)
    m = scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA),
        volume_flux,
        residual,
        reference,
        scalar=scalar,
    )(reference)

    def jacobian(v):
        return jax.jvp(residual, (reference,), (v,))[1]

    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(3):
        v = jnp.asarray(rng.standard_normal(mesh.n_cells))
        factors.append(float(jnp.linalg.norm(jacobian(m(v)) - v) / jnp.linalg.norm(v)))
    return float(np.mean(factors))


def test_preconditioner_strongly_contracts_the_convection_diffusion_error() -> None:
    """A single V-cycle brings ``J M v`` close to ``v`` -- i.e. ``M`` approximates ``J^{-1}`` well --
    on the convection-dominated operator, for both the two-level and the reduction-based hierarchy."""
    assert _contraction(32, 16, ScalarTwoLevel()) < 0.5
    assert _contraction(32, 16, ScalarAir()) < 0.2  # lAIR is near-exact per cycle


def test_preconditioner_contraction_is_mesh_independent() -> None:
    """The contraction stays bounded as the mesh refines -- the scalable-iteration property (an
    unpreconditioned convection solve would instead need O(N) iterations)."""
    coarse = _contraction(24, 12, ScalarTwoLevel())
    fine = _contraction(48, 24, ScalarTwoLevel())
    assert coarse < 0.5 and fine < 0.5
    assert fine < 1.6 * coarse  # bounded, not growing with size


def test_preconditioner_apply_is_linear() -> None:
    """The frozen V-cycle is a constant linear operator in its argument (a valid left preconditioner
    for a plain-GMRES step)."""
    mesh, geometry, volume_flux, residual = _transport(16, 8)
    reference = jnp.ones(mesh.n_cells)
    m = scalar_transport_preconditioner(
        mesh, geometry, jnp.full(mesh.n_cells, GAMMA), volume_flux, residual, reference
    )(reference)
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal(mesh.n_cells))
    b = jnp.asarray(rng.standard_normal(mesh.n_cells))
    assert jnp.allclose(m(2.0 * a - 3.0 * b), 2.0 * m(a) - 3.0 * m(b), atol=1e-9)


def _built(scalar, diffusivity_scale, flux_scale, *, reuse=None):
    """A preconditioner for the transport operator at a scaled diffusivity / flux (same mesh)."""
    mesh, geometry, volume_flux, residual = _transport(24, 12)
    reference = jnp.ones(mesh.n_cells)
    return scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA * diffusivity_scale),
        volume_flux * flux_scale,
        residual,
        reference,
        scalar=scalar,
        reuse=reuse,
    )


def test_refreshing_an_air_scalar_preconditioner_preserves_its_structure() -> None:
    """``reuse=`` re-derives an lAIR k/omega preconditioner's values on its frozen coarsening.

    lAIR's C/F split reads operator values, so rebuilding it at a state whose convection/diffusion
    balance has moved changes the shapes, and the refreshed preconditioner would then force a recompile
    of the solve it accelerates. Reusing the frozen coarsening keeps every shape, which is what makes a
    mid-march refresh affordable; a plain rebuild is checked to actually differ, so the test cannot pass
    vacuously.

    **The two comparisons deliberately use different operating points, and that is the finding rather
    than a convenience.** Scaling the diffusivity and the flux *together* on a uniform mesh leaves every
    relative strength where it was, so the C/F split does not move at all and a rebuild is shape-stable
    — measured identical at level 0 across a 400x swing in both. Only changing the **balance** far
    enough to reorder an edge's strength against its neighbours re-splits, which is what the
    diffusion-dominated third arm does. An earlier form of this test read the first pair as
    structure-changing; it was reading a `scipy` artifact, since the coarse operator's stored entry
    count then depended on which product entries came out exactly zero. That is fixed
    (`multigrid._galerkin_coarse`), so the shape difference has to come from the coarsening itself.
    """
    cold = _built(ScalarAir(), 1.0, 1.0)
    refreshed = _built(ScalarAir(), 0.05, 20.0, reuse=cold)
    # Diffusion-dominated rather than more convective: the balance has to move, not just the magnitudes.
    rebuilt = _built(ScalarAir(), 1.0, 1e-3)

    cold_shapes = [(lv.n, lv.n_coarse, lv.operator.data.shape) for lv in cold.hierarchy.levels]
    refreshed_shapes = [
        (lv.n, lv.n_coarse, lv.operator.data.shape) for lv in refreshed.hierarchy.levels
    ]
    rebuilt_shapes = [
        (lv.n, lv.n_coarse, lv.operator.data.shape) for lv in rebuilt.hierarchy.levels
    ]

    assert refreshed_shapes == cold_shapes  # the point: signature preserved
    assert rebuilt_shapes != cold_shapes  # ...and a rebuild genuinely would not preserve it
    # The refresh moved the values to the new operator.
    assert not np.allclose(
        np.asarray(cold.hierarchy.levels[0].operator.data),
        np.asarray(refreshed.hierarchy.levels[0].operator.data),
    )


def test_refreshing_a_twolevel_scalar_preconditioner_is_structure_preserving_anyway() -> None:
    """The aggregation path needs no ``reuse``: its coarsening reads only the graph.

    Pinned so the asymmetry with lAIR is explicit — for :class:`ScalarTwoLevel` a plain rebuild at a new
    state already reproduces the structure, so ``reuse`` is accepted but changes nothing.
    """
    cold = _built(ScalarTwoLevel(), 1.0, 1.0)
    rebuilt = _built(ScalarTwoLevel(), 0.05, 20.0)
    reused = _built(ScalarTwoLevel(), 0.05, 20.0, reuse=cold)

    shapes = [(lv.n, lv.n_coarse, lv.operator.data.shape) for lv in cold.hierarchy.levels]
    assert [
        (lv.n, lv.n_coarse, lv.operator.data.shape) for lv in rebuilt.hierarchy.levels
    ] == shapes
    assert [(lv.n, lv.n_coarse, lv.operator.data.shape) for lv in reused.hierarchy.levels] == shapes


def _scaled_transport(gamma_scale, flux_scale, nx=24, ny=12):
    """The same transport problem at a scaled diffusivity and flux — a 'developed' operating point."""
    mesh = structured_grid_2d(nx, ny, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    gamma = GAMMA * gamma_scale
    volume_flux = U * flux_scale * geometry.face.normal[:, 0] * geometry.face.area
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(gamma)}),
        (AdvectionFlux(volume_flux, FirstOrderUpwind()), DiffusionFlux()),
        BoundaryConditions(
            {
                "left": Dirichlet(1.0),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
    )
    return mesh, geometry, gamma, volume_flux, assembler.residual


def test_refreshed_air_preconditioner_tracks_the_new_operator() -> None:
    """A refreshed lAIR preconditioner must beat the stale one *on the developed operator*.

    Reusing the reference state's C/F split is a deliberate trade (any valid split preconditions), so
    preserving the structure is only worth anything if the recomputed values genuinely track the new
    coefficients. Judged by the residual left after one V-cycle applied to the developed operator's
    Jacobian — the thing the solve actually iterates on.
    """
    mesh, geometry, _, cold_flux, cold_residual = _scaled_transport(1.0, 1.0)
    _, _, dev_gamma, dev_flux, dev_residual = _scaled_transport(0.05, 20.0)
    reference = jnp.ones(mesh.n_cells)

    cold = scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA),
        cold_flux,
        cold_residual,
        reference,
        scalar=ScalarAir(),
    )
    refreshed = scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, dev_gamma),
        dev_flux,
        dev_residual,
        reference,
        scalar=ScalarAir(),
        reuse=cold,
    )

    def developed_jacobian(v):
        return jax.jvp(dev_residual, (reference,), (v,))[1]

    b = jnp.asarray(np.random.default_rng(0).normal(size=mesh.n_cells))

    def left_residual(preconditioner):
        x = preconditioner(reference)(b)
        return float(jnp.linalg.norm(developed_jacobian(x) - b) / jnp.linalg.norm(b))

    stale, fresh = left_residual(cold), left_residual(refreshed)
    assert fresh < stale, f"refreshed ({fresh:.3e}) did not beat stale ({stale:.3e})"


def test_refreshing_rejects_a_mismatched_scalar_block() -> None:
    """Refreshing an aggregation preconditioner as lAIR (or vice versa) raises rather than mis-reusing."""
    cold_twolevel = _built(ScalarTwoLevel(), 1.0, 1.0)
    with pytest.raises(ValueError, match="same kind of `scalar`"):
        _built(ScalarAir(), 0.05, 20.0, reuse=cold_twolevel)


@pytest.mark.parametrize(
    ("value", "preconditioner"),
    [(ScalarTwoLevel, ConvectionAmgPreconditioner), (ScalarAir, AirAmgPreconditioner)],
    ids=["two-level", "air"],
)
def test_each_scalar_value_names_exactly_its_preconditioner_s_own_settings(
    value, preconditioner
) -> None:
    """A value's fields are its preconditioner's fields, less the hierarchy the build supplies.

    A setting added to a preconditioner and not to its value -- or the reverse -- fails here rather than
    becoming unreachable, which is how ``v_cycles`` sat unreachable from a coupled spec behind the string.
    """
    settings = {field.name for field in dataclasses.fields(preconditioner)} - {"hierarchy"}
    assert {field.name for field in dataclasses.fields(value)} == settings


def test_leaving_the_blocks_unpreconditioned_has_no_settings() -> None:
    assert dataclasses.fields(UnpreconditionedScalars) == ()


@pytest.mark.parametrize("value", [ScalarTwoLevel, ScalarAir], ids=["two-level", "air"])
def test_a_set_v_cycle_count_reaches_the_preconditioner_and_an_unset_one_keeps_its_default(
    value,
) -> None:
    assert _built(value(v_cycles=3), 1.0, 1.0).v_cycles == 3
    assert _built(value(), 1.0, 1.0).v_cycles == 1


def test_unpreconditioned_scalars_build_nothing_and_assemble_no_operator() -> None:
    """The operator's ``J . 1`` row sum is a residual evaluation, which a value that builds nothing skips."""
    mesh, geometry, volume_flux, _ = _transport(8, 4)

    def residual(phi):
        raise AssertionError("the operator was assembled for a block that is not preconditioned")

    built = scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA),
        volume_flux,
        residual,
        jnp.ones(mesh.n_cells),
        scalar=UnpreconditionedScalars(),
    )
    assert built is None


def test_a_scalar_block_given_as_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="scalar must be a scalar-block value"):
        _built("air", 1.0, 1.0)
