"""Unit tests for the SIMPLE-preconditioner numerics: the pressure Schur Laplacian and the fixed
damped-Jacobi inner solve. Both are tested in isolation from the Newton driver."""

from __future__ import annotations

import warnings

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
    FrozenViscosityVelocityParts,
    MomentumContinuity,
    MomentumShiftPolicy,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    damped_jacobi_solve,
    pressure_schur_laplacian,
)
from aquaflux.flow.block_preconditioner import (
    FlowBlocks,
    SmoothedAmgConvectionVelocity,
    SmoothedAmgVelocity,
    _build_composition,
    _characteristic_reference_state,
    _per_component,
    _symmetric_rescaled,
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


def _mass_scaled_at(asm, u_in):
    """The mass-scaled-Schur preconditioner for ``asm`` and a uniform inlet-speed flow, and that state."""
    preconditioner = BlockPreconditioner.build(asm, schur_scaling="msimple")
    velocity = jnp.zeros((asm.mesh.n_cells, asm.mesh.dim)).at[:, 0].set(u_in)
    state = asm.pack(velocity, jnp.zeros(asm.mesh.n_cells))
    return preconditioner, state


def test_mass_scaled_schur_scale_carries_density_like_a_p_schur() -> None:
    """The mass-scaled Schur's coefficient keeps its density factor for rho != 1, matching SIMPLE (issue #40).

    ``schur_face_coefficient`` applies ``rho_face`` itself, so the frozen mass-matrix diagonal
    ``schur_a_P = Q_hat / k = rho V / k`` must be calibrated in ``rho V`` units (``k = mean(rho V /
    a_P)``); otherwise the density cancels and the coefficient comes out ``rho`` times too small. At
    water density (rho = 1000) the per-cell effective Schur coefficient ``rho V / schur_a_P`` must
    match SIMPLE's ``rho V / a_P`` in mean magnitude -- before the fix it was 1000x smaller.
    """
    rho = 1000.0
    asm = _channel(u_in=3.0, rho=rho)
    preconditioner, state = _mass_scaled_at(asm, u_in=3.0)

    a_p = preconditioner.frozen_momentum_diagonal(state)  # the real a_P (SIMPLE's schur diagonal)
    schur_a_p = preconditioner.schur_mass_diagonal / preconditioner._mass_scale(state)  # rho V / k

    rho_v = asm.density * asm.geometry.cell.volume
    eff_simple = float(jnp.mean(rho_v / a_p))  # mean(rho V / a_P) = k
    eff_mass_scaled = float(
        jnp.mean(rho_v / schur_a_p)
    )  # mean(rho V / (rho V / k)) = k iff k carries rho
    assert eff_mass_scaled == pytest.approx(eff_simple, rel=1e-6)


def test_mass_scale_does_not_leak_a_geometry_gradient() -> None:
    """The mass-matrix scale is frozen: no live cell-volume gradient reaches the Schur diagonal (issue #40).

    ``k`` feeds ``schur_a_P`` (the Schur operator), so a live cell-volume dependence would leak a
    mesh-geometry gradient into the adjoint. The scale must be ``stop_gradient``-ed, like the momentum
    diagonal it is built from -- scaling the cell volumes must not move it.
    """
    asm = _channel(u_in=3.0, rho=1.2)
    preconditioner, state = _mass_scaled_at(asm, u_in=3.0)

    def scale_with_scaled_volumes(s):
        scaled = eqx.tree_at(lambda a: a.geometry.cell.volume, asm, asm.geometry.cell.volume * s)
        moved = eqx.tree_at(lambda p: p.assembler, preconditioner, scaled)
        return moved._mass_scale(state)

    assert float(jax.grad(scale_with_scaled_volumes)(1.0)) == 0.0


def test_boundary_a_p_drops_wall_convection() -> None:
    """A wall passes no fluid, so its a_P owner contribution ignores the convective estimate (issue #41).

    On a closed cavity (all no-slip walls) the boundary momentum diagonal must be the Dirichlet viscous
    term alone: feeding a huge convective mass flux must not change it, because the wall carries none.
    """
    asm = _geometry(6)  # closed cavity: every patch is a NoSlipWall
    n_faces = asm.mesh.n_faces
    boundary_mu = asm.viscosity[asm.mesh.face_cells.owner]  # per-face (no wall model)
    viscous_only = asm.boundary_momentum_diagonal(boundary_mu, None)
    with_huge_mdot = asm.boundary_momentum_diagonal(boundary_mu, jnp.full(n_faces, 1e6))
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
    boundary_a_p = asm.boundary_momentum_diagonal(
        asm.viscosity[fc.owner], jnp.zeros(asm.mesh.n_faces)
    )
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


def test_a_convection_block_with_no_reference_flux_says_so_instead_of_degrading_in_silence() -> (
    None
):
    """A closed domain drives no flow, so ``velocity="convection"`` quietly becomes ``"smoothed"``.

    The build stays valid -- a zero convective linearization is still a usable viscous operator -- so
    this is a warning rather than a refusal. But it is the *only* signal that the Peclet-aware block
    the caller asked for is not the block they got, and the degradation is otherwise invisible: the
    preconditioner still applies, the solve still converges, and only the iteration count moves. A
    guard whose whole job is to be noticed needs a test that it is still emitted, or it can stop
    firing without anything changing colour.

    The paired silent case is what makes this a test of the *condition* rather than of the warning:
    the same build over a lid-driven cavity has a reference flux and must say nothing.
    """
    mesh = structured_grid_2d(6, 6, lx=1.0, ly=1.0, named_boundaries=True)
    walls = {side: NoSlipWall() for side in ("top", "bottom", "left", "right")}

    def _built(conditions):
        assembler = MomentumContinuity.build(
            mesh,
            mesh.geometry(),
            PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
            CompactGreenGauss(),
            BoundaryConditions(conditions),
        )
        return BlockPreconditioner.build(assembler, velocity="convection")

    with pytest.warns(RuntimeWarning, match="no mass flux"):
        _built(walls)  # every patch a stationary wall: nothing prescribes a velocity anywhere

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a spurious warning here would fail the build outright
        _built({**walls, "top": MovingWall(velocity=(2.5, 0.0))})


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
            mdot_lagged=reference_mdot,
        ),
        axis=1,
    )
    assert jnp.allclose(block.hierarchy.levels[0].diagonal, reference_a_p, rtol=1e-12, atol=0.0)


def test_momentum_diagonal_is_the_residual_operator_diagonal_under_graded_viscosity() -> None:
    """``a_P`` equals ``diag(J_momentum)`` even where the viscosity is graded (issue #154).

    ``a_P``'s viscous term is the diffusion operator's own diagonal contribution (harmonic on a graded
    face), so ``a_P`` *is* the true momentum-matrix diagonal — the coefficient the Rhie--Chow damping
    ``V / a_P`` (a differentiated, solution-affecting term) and the pseudo-time shift both assume. The
    g-weighted arithmetic viscosity it replaced over-estimated a graded face by ``(1+r)^2/(4r)`` and
    failed this at non-constant viscosity; at constant viscosity both forms are byte-identical, which is
    why every prior (constant-``mu``) test passed it. A Stokes flow (no advection) isolates the viscous
    diagonal; graded ``mu_eff = mu + rho nu_t`` is injected through the turbulence-closure seam.
    """
    mesh = structured_grid_2d(8, 8, named_boundaries=True)
    geom = mesh.geometry()
    asm0 = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({side: NoSlipWall() for side in ("top", "bottom", "left", "right")}),
        pressure_pin=0,  # closed domain
    )  # no advection_scheme -> Stokes: a_P is purely viscous, the residual is linear
    x = geom.cell.centroid[:, 0]
    nu_t = jnp.exp(3.0 * (x - x.min()) / (x.max() - x.min())) - 1.0  # 0 .. ~19, smoothly graded
    asm = asm0.with_eddy_viscosity(nu_t)

    velocity = jnp.zeros((mesh.n_cells, mesh.dim))
    a_p = asm.momentum_matrix_diagonal(velocity)  # (n_cells, dim), boundary-consistent form
    state = asm.pack(velocity, jnp.zeros(mesh.n_cells))
    velocity_diag, _ = asm.unpack(jnp.diag(jax.jacfwd(asm.residual)(state)))
    assert jnp.allclose(a_p, velocity_diag, rtol=1e-9, atol=1e-9)


def test_momentum_diagonal_matches_the_operator_at_active_wall_faces() -> None:
    """``a_P``'s wall-face term uses the wall-model boundary viscosity, matching the operator (#155).

    On a wall-function mesh the momentum ``DiffusionFlux`` uses the wall model's ``mu + rho nu_t,wall``
    at wall faces (its ``boundary_coefficient``), while the wall-adjacent cell's own ``mu_eff`` is the
    log-layer ``k/omega`` value — larger by a big factor. Building ``a_P``'s boundary term from the
    **owner-cell** viscosity (the bug) left ``a_P`` disagreeing with ``diag(J_momentum)`` at every wall
    cell; building it from the operator's own **per-face** boundary viscosity fixes it. A wall-resolved
    mesh has ``nu_t,wall = 0`` and is a no-op, which is why the wall-resolved law-of-the-wall test
    structurally cannot see this.
    """
    mesh = structured_grid_2d(8, 8, named_boundaries=True)
    geom = mesh.geometry()
    asm0 = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1e-3), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({side: NoSlipWall() for side in ("top", "bottom", "left", "right")}),
        pressure_pin=0,
    )  # Stokes closed cavity: a_P is purely viscous, the residual is linear
    # A large log-layer cell eddy viscosity with a much smaller wall-model value on the wall faces —
    # the wall-function regime where the owner-cell and wall-model viscosities genuinely differ (~7x).
    nu_t = jnp.full(mesh.n_cells, 1.6e-2)
    wall_nu_t = jnp.full(mesh.n_faces, 1.6e-3)
    asm = asm0.with_eddy_viscosity(nu_t, wall_nu_t)

    velocity = jnp.zeros((mesh.n_cells, mesh.dim))
    a_p = asm.momentum_matrix_diagonal(velocity)  # (n_cells, dim), boundary-consistent form
    state = asm.pack(velocity, jnp.zeros(mesh.n_cells))
    velocity_diag, _ = asm.unpack(jnp.diag(jax.jacfwd(asm.residual)(state)))
    assert jnp.allclose(a_p, velocity_diag, rtol=1e-9, atol=1e-12)


def test_momentum_shift_policy_injects_its_velocity_parts_source() -> None:
    """``MomentumShiftPolicy`` takes its shift buckets from an injected source (issue #156, seam 3).

    The flow-only shift policy now has the same ``velocity_shift_parts`` seam as the coupled one — which
    it could not reach before the :class:`~aquaflux.solve.VelocityShiftParts` protocol was moved out of
    ``turbulence`` (a flow→turbulence import cycle). The explicit
    :class:`~aquaflux.flow.FrozenViscosityVelocityParts` spelling equals the ``None`` default
    (bit-identical, the historical inline path), and a different injected source genuinely changes the
    shift diagonal, so the seam is live rather than cosmetic.
    """
    asm = _channel(u_in=1.0)
    block = BlockPreconditioner.build(asm)
    velocity = jnp.zeros((asm.mesh.n_cells, asm.mesh.dim)).at[:, 0].set(1.0)
    phi = asm.pack(velocity, jnp.zeros(asm.mesh.n_cells))

    default = MomentumShiftPolicy(block).shift_term(phi).diagonal
    explicit = (
        MomentumShiftPolicy(block, velocity_shift_parts=FrozenViscosityVelocityParts(block))
        .shift_term(phi)
        .diagonal
    )
    assert jnp.array_equal(default, explicit)  # the frozen spelling IS the None default

    class _Doubled:  # a stub source that doubles the buckets, so the shift must change if it is used
        def parts(self, flow, k_solved=None, omega_solved=None):
            convective, dissipative = block.frozen_momentum_diagonal_parts(flow)
            return 2.0 * convective, 2.0 * dissipative

    injected = MomentumShiftPolicy(block, velocity_shift_parts=_Doubled()).shift_term(phi).diagonal
    assert not jnp.allclose(injected, default)


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


def test_symmetric_rescaling_is_exact_for_a_diagonal_congruence() -> None:
    """The rescaling sandwich inverts ``A_cur`` exactly when ``A_cur = D A_ref D`` — the invariant it
    is built on. Every multigrid block here freezes a hierarchy at ``A_ref`` and tracks the current
    operator this way, so the property is pinned once against a dense operator whose exact inverse is
    known, independent of any multigrid.
    """
    rng = np.random.default_rng(11)
    n = 8
    root = rng.standard_normal((n, n))
    a_ref = jnp.asarray(root @ root.T + n * np.eye(n))  # SPD
    scale = jnp.asarray(rng.uniform(0.2, 5.0, n))  # the per-cell drift D = diag(scale)
    a_cur = scale[:, None] * a_ref * scale[None, :]

    rescaled = _symmetric_rescaled(
        lambda b: jnp.linalg.solve(a_ref, b), jnp.diag(a_ref), jnp.diag(a_cur)
    )
    b = jnp.asarray(rng.standard_normal(n))
    assert np.allclose(rescaled(b), jnp.linalg.solve(a_cur, b))

    # A zero drift (diag_cur == diag_ref) leaves the frozen solve untouched.
    identity = _symmetric_rescaled(
        lambda x: jnp.linalg.solve(a_ref, x), jnp.diag(a_ref), jnp.diag(a_ref)
    )
    assert np.allclose(identity(b), jnp.linalg.solve(a_ref, b))


def test_per_component_applies_the_scalar_solve_to_each_column() -> None:
    """The momentum block is block-diagonal across velocity components, so lifting a scalar solve to a
    vector field is exactly the same solve per column — no cross-component mixing."""
    rng = np.random.default_rng(12)
    n, dim = 5, 3
    weights = jnp.asarray(rng.uniform(1.0, 3.0, n))

    def scalar_solve(b):
        return weights * b

    ru = jnp.asarray(rng.standard_normal((n, dim)))
    du = _per_component(scalar_solve, dim)(ru)

    assert du.shape == (n, dim)
    for i in range(dim):
        assert np.allclose(du[:, i], scalar_solve(ru[:, i]))


def test_velocity_block_builds_from_a_narrow_geometry_seam() -> None:
    """A velocity-block strategy builds from a ``_VelocityGeometry`` bundle alone — mesh geometry only,
    no boundary conditions / property model / schemes — so it is unit-testable in isolation without a
    full flow assembler. This is the encapsulation the seam exists for: the strategy takes the narrow
    geometry it needs, not the whole ``MomentumContinuity``.
    """
    mesh = structured_grid_2d(4, 3)
    face_cells = mesh.face_cells
    n_cells = mesh.n_cells
    # SmoothedAmgVelocity reads only the mesh geometry (centroids/areas via the flux-continuous
    # conductance, owner, dim); the viscosity field belongs to the convection-aware sibling, so it is
    # unit here.
    geometry = _VelocityGeometry(
        face_cells=face_cells,
        mesh_geometry=mesh.geometry(),
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


# --- saddle compositions ---------------------------------------------------------------
#
# The composition family says how many times, and in what order, the velocity and Schur solves are
# applied. Its members are checked here against the property that *defines* each one, rather than
# against a transcription of its own code: exactness on a saddle system the composition's own
# approximations are exact for. That is the check a missing algorithm step cannot pass.


def _small_channel():
    """A 12-cell inlet/outlet channel — small enough to materialize, and *not* pressure-singular.

    A closed all-wall cavity fixes the pressure only up to a constant, so its saddle is singular and
    no composition could reproduce a state applied through it. The pressure outlet is what makes the
    exactness checks below well-posed.
    """
    mesh = structured_grid_2d(4, 3, lx=2.0, ly=H, named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(1e-2), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(1.0, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )


def _materialize(op, shape, out_shape):
    """Dense matrix of a linear operator, column by column (small meshes only)."""
    columns = []
    for i in range(int(np.prod(shape))):
        basis = jnp.zeros(int(np.prod(shape))).at[i].set(1.0).reshape(shape)
        columns.append(np.asarray(op(basis)).reshape(-1))
    return np.array(columns).T.reshape(int(np.prod(out_shape)), int(np.prod(shape)))


def _exact_saddle(asm, state, diagonal):
    """The blocks, and exact inner solves, of the saddle ``[[diag(d), G], [B, Ĉ]]``.

    The *velocity* block is replaced by the very diagonal ``F̃`` the compositions correct with, so
    the SIMPLE approximation ``F⁻¹ ≈ F̃⁻¹`` is exact for this system and the paper's algorithms must
    reproduce its inverse to solver tolerance. The gradient, divergence and pressure-coupling blocks
    are the real ones, so nothing about the saddle's structure is faked away.
    """
    blocks = FlowBlocks.of(asm, state)
    n_cells, dim = asm.mesh.n_cells, asm.mesh.dim
    gradient = _materialize(blocks.gradient, (n_cells,), (n_cells, dim))
    divergence = _materialize(blocks.divergence, (n_cells, dim), (n_cells,))
    coupling = _materialize(blocks.pressure_coupling, (n_cells,), (n_cells,))
    velocity_diagonal = np.repeat(np.asarray(diagonal), dim)
    saddle = np.block([[np.diag(velocity_diagonal), gradient], [divergence, coupling]])
    schur = coupling - divergence @ np.diag(1.0 / velocity_diagonal) @ gradient
    schur_inverse = np.linalg.inv(schur)
    return (
        blocks,
        saddle,
        lambda r: r / diagonal[:, None],
        lambda r: jnp.asarray(schur_inverse @ np.asarray(r)),
    )


def _apply_saddle(asm, saddle, vector):
    """``[[diag(d), G], [B, Ĉ]] v`` on a packed state vector."""
    n_cells, dim = asm.mesh.n_cells, asm.mesh.dim
    velocity, pressure = asm.unpack(vector)
    flat = np.concatenate([np.asarray(velocity).reshape(-1), np.asarray(pressure)])
    out = saddle @ flat
    return asm.pack(
        jnp.asarray(out[: n_cells * dim].reshape(n_cells, dim)), jnp.asarray(out[n_cells * dim :])
    )


@pytest.mark.parametrize("composition", ["simple", "simpler"])
def test_a_faithful_composition_inverts_the_saddle_its_approximations_are_exact_for(
    composition,
) -> None:
    """SIMPLE and SIMPLER are exact solvers once their two approximations are.

    Both are block factorizations of the saddle: with the momentum block equal to the diagonal they
    correct with, and the Schur solved exactly, applying the composition to ``A z`` must return
    ``z``. A dropped or mis-signed step breaks this outright — it is the property the missing
    pressure prediction failed.
    """
    asm = _small_channel()
    state = asm.initial_state()
    diagonal = jnp.asarray(np.linspace(1.0, 3.0, asm.mesh.n_cells))
    blocks, saddle, velocity_solve, schur_solve = _exact_saddle(asm, state, diagonal)
    solve = _build_composition(composition).apply(
        blocks, velocity_solve, schur_solve, 1.0 / diagonal
    )

    rng = np.random.default_rng(0)
    expected = jnp.asarray(rng.normal(size=asm.initial_state().shape))
    recovered = asm.pack(*solve(*asm.unpack(_apply_saddle(asm, saddle, expected))))

    assert np.allclose(np.asarray(recovered), np.asarray(expected), rtol=1e-8, atol=1e-8)


def test_the_block_triangular_composition_is_exact_in_pressure_but_not_in_velocity() -> None:
    """The default composition is deliberately *not* the full factorization, and this pins which half.

    It drops SIMPLE's closing velocity update, which leaves the preconditioned operator unipotent
    (the Murphy--Golub--Wathen two-iteration structure) rather than the identity. So on the same
    saddle the pressure comes back exact and the velocity does not — evidence the omission is the
    documented one and not a second missing step.
    """
    asm = _small_channel()
    state = asm.initial_state()
    diagonal = jnp.asarray(np.linspace(1.0, 3.0, asm.mesh.n_cells))
    blocks, saddle, velocity_solve, schur_solve = _exact_saddle(asm, state, diagonal)
    solve = _build_composition("triangular").apply(
        blocks, velocity_solve, schur_solve, 1.0 / diagonal
    )

    rng = np.random.default_rng(0)
    expected = jnp.asarray(rng.normal(size=asm.initial_state().shape))
    velocity, pressure = solve(*asm.unpack(_apply_saddle(asm, saddle, expected)))
    want_velocity, want_pressure = asm.unpack(expected)

    assert np.allclose(np.asarray(pressure), np.asarray(want_pressure), rtol=1e-8, atol=1e-8)
    assert not np.allclose(np.asarray(velocity), np.asarray(want_velocity), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("composition", ["triangular", "simple", "simpler"])
def test_every_composition_is_a_fixed_linear_map(composition) -> None:
    """``M`` must be linear in its argument, or non-flexible GMRES and the transposed adjoint are both
    invalid. Every member applies a fixed number of fixed-cycle inner solves, so this holds by
    construction — pinned because a future member could break it by iterating to a tolerance."""
    asm = _channel(1.0)
    preconditioner = BlockPreconditioner.build(asm, composition=composition)
    state = asm.initial_state()
    m = preconditioner.factory()(state)

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=state.shape))
    y = jnp.asarray(rng.normal(size=state.shape))

    combined = m(2.5 * x - 0.75 * y)
    separate = 2.5 * m(x) - 0.75 * m(y)
    assert np.allclose(np.asarray(combined), np.asarray(separate), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("composition", ["triangular", "simple", "simpler"])
def test_every_composition_transposes_for_the_adjoint(composition) -> None:
    """The adjoint applies ``Mᵀ`` via :func:`jax.linear_transpose`, so every member must satisfy
    ``<y, M x> = <Mᵀ y, x>`` — the identity that makes the preconditioned transpose solve legitimate."""
    asm = _channel(1.0)
    preconditioner = BlockPreconditioner.build(asm, composition=composition)
    state = asm.initial_state()
    m = preconditioner.factory()(state)

    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=state.shape))
    y = jnp.asarray(rng.normal(size=state.shape))
    (transposed,) = jax.linear_transpose(m, x)(y)

    assert float(jnp.vdot(y, m(x))) == pytest.approx(float(jnp.vdot(transposed, x)), rel=1e-9)


def test_the_default_composition_leaves_the_preconditioner_byte_identical() -> None:
    """The shipped default is the lower block-triangular pass the preconditioner has always applied,
    so introducing the family changed no existing behaviour."""
    asm = _channel(1.0)
    state = asm.initial_state()
    default = BlockPreconditioner.build(asm).factory()(state)
    triangular = BlockPreconditioner.build(asm, composition="triangular").factory()(state)

    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.normal(size=state.shape))
    assert np.array_equal(np.asarray(default(v)), np.asarray(triangular(v)))


def test_an_unknown_composition_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="unknown composition"):
        BlockPreconditioner.build(_channel(1.0), composition="simplest")
