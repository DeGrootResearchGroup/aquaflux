"""Unit tests for the SIMPLE-preconditioner numerics: the pressure Schur Laplacian and the fixed
damped-Jacobi inner solve. Both are tested in isolation from the Newton driver."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    damped_jacobi_solve,
    pressure_schur_laplacian,
)
from aquaflux.flow.block_preconditioner import (
    SmoothedAmgConvectionVelocity,
    SmoothedAmgVelocity,
    _characteristic_reference_state,
    _VelocityGeometry,
)
from aquaflux.flow.rhie_chow import momentum_diagonal
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss

from tests.support.meshes import perturbed_grid_2d

H, NY = 1.0, 6  # channel height and wall-normal count: the vertical interior faces have area H / NY


def _geometry(n, perturb=0.0):
    """A small closed-cavity assembler, only for its geometry (interp_factor, normal_distance)."""
    mesh = perturbed_grid_2d(n, n, perturb=perturb, named_boundaries=True)
    geom = mesh.geometry()
    walls = {side: NoSlipWall() for side in ("top", "bottom", "left", "right")}
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions(walls),
    )


def _channel(u_in, rho=1.0):
    """A small open channel driven by a uniform velocity inlet at speed ``u_in`` and density ``rho``."""
    mesh = structured_grid_2d(10, NY, lx=4.0, ly=H, named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(1e-2), "density": Constant(rho)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(u_in, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )


def _msimpler_at(asm, u_in):
    """The MSIMPLER preconditioner for ``asm`` and a uniform inlet-speed flow, and that state."""
    preconditioner = BlockPreconditioner.build(asm, schur_scaling="msimpler")
    velocity = jnp.zeros((asm.mesh.n_cells, asm.mesh.dim)).at[:, 0].set(u_in)
    state = asm.pack(velocity, jnp.zeros(asm.mesh.n_cells))
    return preconditioner, state


def test_msimpler_schur_scale_carries_density_like_simple() -> None:
    """MSIMPLER's Schur coefficient keeps its density factor for rho != 1, matching SIMPLE (issue #40).

    ``schur_face_coefficient`` applies ``rho_face`` itself, so the frozen MSIMPLER diagonal
    ``schur_a_P = Q_hat / k = rho V / k`` must be calibrated in ``rho V`` units (``k = mean(rho V /
    a_P)``); otherwise the density cancels and the coefficient comes out ``rho`` times too small. At
    water density (rho = 1000) the per-cell effective Schur coefficient ``rho V / schur_a_P`` must
    match SIMPLE's ``rho V / a_P`` in mean magnitude -- before the fix it was 1000x smaller.
    """
    rho = 1000.0
    asm = _channel(u_in=3.0, rho=rho)
    preconditioner, state = _msimpler_at(asm, u_in=3.0)

    a_p = preconditioner.frozen_momentum_diagonal(state)  # the real a_P (SIMPLE's schur diagonal)
    schur_a_p = preconditioner.schur_mass_diagonal / preconditioner._msimpler_scale(
        state
    )  # rho V / k

    rho_v = asm.density * asm.geometry.cell.volume
    eff_simple = float(jnp.mean(rho_v / a_p))  # mean(rho V / a_P) = k
    eff_msimpler = float(
        jnp.mean(rho_v / schur_a_p)
    )  # mean(rho V / (rho V / k)) = k iff k carries rho
    assert eff_msimpler == pytest.approx(eff_simple, rel=1e-6)


def test_msimpler_scale_does_not_leak_a_geometry_gradient() -> None:
    """The MSIMPLER scale is frozen: no live cell-volume gradient reaches the Schur diagonal (issue #40).

    ``k`` feeds ``schur_a_P`` (the Schur operator), so a live cell-volume dependence would leak a
    mesh-geometry gradient into the adjoint. The scale must be ``stop_gradient``-ed, like the momentum
    diagonal it is built from -- scaling the cell volumes must not move it.
    """
    asm = _channel(u_in=3.0, rho=1.2)
    preconditioner, state = _msimpler_at(asm, u_in=3.0)

    def scale_with_scaled_volumes(s):
        scaled = eqx.tree_at(lambda a: a.geometry.cell.volume, asm, asm.geometry.cell.volume * s)
        moved = eqx.tree_at(lambda p: p.assembler, preconditioner, scaled)
        return moved._msimpler_scale(state)

    assert float(jax.grad(scale_with_scaled_volumes)(1.0)) == 0.0


def test_boundary_a_p_drops_wall_convection() -> None:
    """A wall passes no fluid, so its a_P owner contribution ignores the convective estimate (issue #41).

    On a closed cavity (all no-slip walls) the boundary momentum diagonal must be the Dirichlet viscous
    term alone: feeding a huge convective mass flux must not change it, because the wall carries none.
    """
    asm = _geometry(6)  # closed cavity: every patch is a NoSlipWall
    n_faces = asm.mesh.n_faces
    viscous_only = asm.boundary_momentum_diagonal(asm.viscosity, None)
    with_huge_mdot = asm.boundary_momentum_diagonal(asm.viscosity, jnp.full(n_faces, 1e6))
    assert jnp.allclose(viscous_only, with_huge_mdot)  # convective dropped at every wall face
    assert float(jnp.sum(viscous_only)) > 0.0  # the Dirichlet viscous stiffness is still there


def test_boundary_a_p_drops_outlet_viscosity() -> None:
    """A zero-gradient outlet imposes no velocity, so it adds no viscous a_P (issue #41).

    At zero mass flux only the Dirichlet-velocity faces (walls, inlet) contribute their viscous term;
    the outlet contributes nothing, so the total is strictly less than the naive all-boundary-faces
    viscous sum (which the pre-fix code produced).
    """
    asm = _channel(u_in=1.0)  # velocity inlet + pressure outlet + no-slip walls
    fc = asm.mesh.face_cells
    boundary_a_p = asm.boundary_momentum_diagonal(asm.viscosity, jnp.zeros(asm.mesh.n_faces))
    all_faces_viscous = jnp.where(
        fc.interior, 0.0, asm.viscosity[fc.owner] * asm.geometry.face.area / asm.normal_distance
    )
    assert float(jnp.sum(boundary_a_p)) < float(
        jnp.sum(all_faces_viscous)
    )  # outlet viscous excluded
    assert float(jnp.sum(boundary_a_p)) > 0.0  # walls + inlet still contribute


@pytest.mark.parametrize("u_in", [0.01, 1.0, 100.0])
def test_reference_state_tracks_the_boundary_driven_velocity_scale(u_in) -> None:
    """The derived reference is a uniform flow at the prescribed inlet velocity, whatever its scale.

    This is what makes the convection-aware velocity block freeze its linearization at the operating
    cell Peclet without being handed a characteristic speed: the inlet already states it, at any
    magnitude (a slow-water or fast-air nondimensionalisation alike).
    """
    assembler = _channel(u_in)
    velocity, pressure = assembler.unpack(_characteristic_reference_state(assembler))
    assert jnp.allclose(velocity, jnp.array([u_in, 0.0]))
    assert jnp.allclose(pressure, 0.0)


def test_reference_state_convects_where_a_cold_state_does_not() -> None:
    """The reference state carries the inlet's convection into the *interior* faces, where a cold
    state carries none.

    The frozen momentum operator takes its upwind term from this mass flux, so a zero-flux state
    would leave the operator purely viscous — silently turning the convection-aware block back into
    the Peclet-blind one it exists to replace.
    """
    assembler = _channel(2.0)
    interior = np.asarray(assembler.mesh.face_cells.interior)
    reference = assembler.mass_flux(_characteristic_reference_state(assembler))[interior]
    cold = assembler.mass_flux(assembler.initial_state())[interior]

    assert jnp.allclose(cold, 0.0)
    # The streamwise faces carry the full inlet flux rho * u_in * (H / NY); the wall-normal ones,
    # whose normals are orthogonal to the uniform flow, carry none.
    assert abs(float(jnp.max(jnp.abs(reference))) - 2.0 * (H / NY)) < 1e-12


def test_reference_state_is_driven_by_a_moving_wall_too() -> None:
    """A driven cavity states its scale on a moving wall rather than an inlet, and is picked up the
    same way; a domain with no prescribed velocity at all drives no flow, so its reference is zero."""
    mesh = structured_grid_2d(6, 6, lx=1.0, ly=1.0, named_boundaries=True)
    walls = {side: NoSlipWall() for side in ("top", "bottom", "left", "right")}

    def _cavity(conditions):
        return MomentumContinuity.build(
            mesh,
            mesh.geometry(),
            PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
            CompactGreenGauss(),
            BoundaryConditions(conditions),
        )

    lid = _cavity({**walls, "top": MovingWall(velocity=(2.5, 0.0))})
    velocity, _ = lid.unpack(_characteristic_reference_state(lid))
    assert jnp.allclose(velocity, jnp.array([2.5, 0.0]))

    closed = _cavity(walls)
    velocity, _ = closed.unpack(_characteristic_reference_state(closed))
    assert jnp.allclose(velocity, 0.0)


def test_convection_velocity_operator_diagonal_is_the_momentum_diagonal() -> None:
    """The convection velocity block's frozen operator carries the momentum diagonal ``a_P`` exactly.

    The per-iterate rescaling ``sqrt(diag_ref / a_P)`` is only diagonal-exact if the assembled
    reference operator's diagonal *is* ``a_P`` at that reference. It is built to be: the interior
    upwind stencil supplies the interior part, the boundary-face owner coefficient supplies the rest,
    and both come from the same shared viscous coefficient and reference flux the momentum diagonal
    itself uses — so the two cannot drift, and nothing has to match a separately reconstructed
    diagonal. The plain all-faces form is the one checked, since that is what the frozen diagonal
    (``boundary_corrected=False``) the rescaling divides by uses.
    """
    asm = _channel(1.0)
    n_cells = asm.mesh.n_cells
    owner_e, nb_e, _ = asm.mesh.face_cells.interior_edges()
    interior = np.asarray(asm.mesh.face_cells.interior)
    reference_mdot = jax.lax.stop_gradient(asm.mass_flux(_characteristic_reference_state(asm)))
    block = SmoothedAmgConvectionVelocity.build(
        _VelocityGeometry.of(asm),
        owner_e,
        nb_e,
        interior,
        n_cells,
        1,
        reference_mdot,
        method="twolevel",
    )
    reference_a_p = jnp.mean(
        momentum_diagonal(
            asm.mesh.face_cells,
            asm.geometry,
            asm.viscosity,
            asm.normal_distance,
            asm.interp_factor,
            mdot_lagged=reference_mdot,
        ),
        axis=1,
    )
    assert jnp.allclose(block.hierarchy.levels[0].diagonal, reference_a_p, rtol=1e-12, atol=0.0)


def test_schur_laplacian_is_conservative_and_spd() -> None:
    """It is an M-matrix Laplacian: constant pressure -> zero, positive diagonal, symmetric, PSD."""
    asm = _geometry(8, perturb=0.15)
    a_p = 2.0 + jnp.arange(asm.mesh.n_cells, dtype=float)  # arbitrary positive, non-uniform
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
    )
    n = asm.mesh.n_cells
    assert jnp.allclose(matvec(jnp.ones(n)), 0.0, atol=1e-12)  # constant in the null space
    assert bool(jnp.all(diagonal > 0.0))
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal(n))
    q = jnp.asarray(rng.standard_normal(n))
    assert float(jnp.abs(jnp.dot(p, matvec(q)) - jnp.dot(q, matvec(p)))) < 1e-10  # symmetric
    assert float(jnp.dot(p, matvec(p))) > 0.0  # positive semi-definite (definite off the constant)


def test_schur_laplacian_pin_row_is_identity() -> None:
    """A pinned cell's row is the identity: its diagonal is 1 and its matvec returns its own value."""
    asm = _geometry(6)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    assert float(diagonal[0]) == 1.0
    p = jnp.asarray(np.random.default_rng(1).standard_normal(asm.mesh.n_cells))
    assert float(matvec(p)[0]) == float(p[0])


def test_schur_boundary_diagonal_removes_the_null_space() -> None:
    """A boundary (pressure-outlet) diagonal turns the singular pure-Neumann Schur into a definite
    operator: the constant leaves the null space and every eigenvalue is positive."""
    asm = _geometry(6)
    n = asm.mesh.n_cells
    a_p = jnp.ones(n)
    args = (
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
    )

    neumann, _ = pressure_schur_laplacian(*args)
    assert jnp.allclose(neumann(jnp.ones(n)), 0.0, atol=1e-12)  # constant is a null vector

    boundary = jnp.zeros(n).at[0].set(3.0).at[1].set(1.5)  # outlet stiffness on two cells
    stiffened, diagonal = pressure_schur_laplacian(*args, boundary_diagonal=boundary)
    ones = stiffened(jnp.ones(n))
    assert not jnp.allclose(ones, 0.0, atol=1e-9)  # constant no longer in the null space
    assert jnp.allclose(ones, boundary, atol=1e-9)  # Ŝ·1 = boundary diagonal (Laplacian part is 0)
    assert float(jnp.dot(jnp.ones(n), stiffened(jnp.ones(n)))) > 0.0  # positive on the constant
    # The extra diagonal lands exactly where the outlet coupling was placed.
    plain_diag = pressure_schur_laplacian(*args)[1]
    assert jnp.allclose(diagonal - plain_diag, boundary, atol=1e-12)


def test_pressure_schur_coefficient_only_from_pressure_outlet() -> None:
    """Only a pressure-fixing outlet contributes to the Schur boundary diagonal; a wall or a
    velocity inlet sets its mass flux independently of pressure and so contributes nothing."""
    d_coeff = jnp.array([2.0, 0.5])
    area = jnp.array([1.0, 1.0])
    normal_distance = jnp.array([0.25, 0.5])
    rho = jnp.array([1.0, 1.0])
    outlet = PressureOutlet(pressure=0.0).pressure_schur_coefficient(
        d_coeff, area, normal_distance, rho
    )
    assert jnp.allclose(outlet, rho * d_coeff * area / normal_distance)
    assert bool(jnp.all(outlet > 0.0))
    for closure in (NoSlipWall(), VelocityInlet(velocity=(1.0, 0.0))):
        contrib = closure.pressure_schur_coefficient(d_coeff, area, normal_distance, rho)
        assert jnp.allclose(contrib, 0.0)


def test_damped_jacobi_is_linear_in_rhs() -> None:
    """A fixed sweep count makes rhs -> x a linear operator (required for a plain-GMRES left PC)."""
    asm = _geometry(6)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    rng = np.random.default_rng(2)
    r1 = jnp.asarray(rng.standard_normal(asm.mesh.n_cells))
    r2 = jnp.asarray(rng.standard_normal(asm.mesh.n_cells))

    def solve(r):
        return damped_jacobi_solve(matvec, diagonal, r, sweeps=12, omega=0.7, pressure_pin=0)

    lhs = solve(2.5 * r1 - 1.5 * r2)
    rhs = 2.5 * solve(r1) - 1.5 * solve(r2)
    assert jnp.allclose(lhs, rhs, atol=1e-12)


def test_damped_jacobi_converges_toward_solution() -> None:
    """More sweeps drive the residual of the pinned Laplacian system down (it is a valid solver)."""
    asm = _geometry(8)
    a_p = jnp.ones(asm.mesh.n_cells)
    matvec, diagonal = pressure_schur_laplacian(
        asm.mesh.face_cells,
        asm.geometry,
        asm.interp_factor,
        asm.normal_distance,
        a_p,
        asm.density,
        pressure_pin=0,
    )
    x_true = jnp.asarray(np.random.default_rng(3).standard_normal(asm.mesh.n_cells))
    rhs = matvec(x_true)  # consistent RHS (pin row carries x_true[0])

    def residual_norm(sweeps):
        x = damped_jacobi_solve(matvec, diagonal, rhs, sweeps=sweeps, omega=0.7, pressure_pin=0)
        return float(jnp.linalg.norm(matvec(x) - rhs))

    assert residual_norm(40) < 0.3 * residual_norm(5)  # clearly decreasing with sweeps


def test_velocity_block_builds_from_a_narrow_geometry_seam() -> None:
    """A velocity-block strategy builds from a ``_VelocityGeometry`` bundle alone — mesh geometry only,
    no boundary conditions / property model / schemes — so it is unit-testable in isolation without a
    full flow assembler. This is the encapsulation the seam exists for: the strategy takes the narrow
    geometry it needs, not the whole ``MomentumContinuity``.
    """
    mesh = structured_grid_2d(4, 3)
    face_cells = mesh.face_cells
    n_cells, n_faces = mesh.n_cells, mesh.n_faces
    # SmoothedAmgVelocity reads only the mesh geometry (face area, normal distance, owner, dim); the
    # interp_factor / viscosity fields belong to the convection-aware sibling, so they are unit here.
    geometry = _VelocityGeometry(
        face_cells=face_cells,
        mesh_geometry=mesh.geometry(),
        interp_factor=jnp.ones(n_faces),
        normal_distance=jnp.ones(n_faces),
        viscosity=jnp.ones(n_cells),
        dim=mesh.dim,
    )
    owner_e, nb_e, _ = face_cells.interior_edges()
    interior = np.asarray(face_cells.interior)
    block = SmoothedAmgVelocity.build(geometry, owner_e, nb_e, interior, n_cells, v_cycles=1)

    solve = block.apply(jnp.full(n_cells, 2.0))  # the per-iterate velocity solve at a_P = 2
    ru = jnp.asarray(np.random.default_rng(0).standard_normal((n_cells, mesh.dim)))
    du = solve(ru)
    assert du.shape == (n_cells, mesh.dim)
    assert bool(jnp.all(jnp.isfinite(du)))
