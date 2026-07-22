"""The SIMPLE block-diagonal preconditioner inside the coupled p--U Newton solve.

It must (a) sharply reduce the outer GMRES iteration count on the saddle-point system, (b) be an
exact drop-in (same Newton update / converged solution as the unpreconditioned solve), and (c) leave
the solve reverse-mode differentiable. The inner Schur solve is a fixed damped-Jacobi sweep, so it is
a constant left preconditioner; a mesh-independent multigrid inner is the scalable upgrade.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import BlockPreconditioner, MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import permute_cells
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import newton_step

from tests.support.meshes import perturbed_grid_2d

RHO, MU = 1.0, 0.02


def _build(mesh, mu=MU, pin=0):
    """Build the lid-driven-cavity coupled p--U assembler on a given (possibly renumbered) mesh."""
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=pin,
    )


def _cavity(n, mu=MU, perm=None):
    mesh = perturbed_grid_2d(n, n, perturb=0.15, named_boundaries=True)
    pin = 0
    if perm is not None:
        mesh = permute_cells(mesh, perm)  # renumbered system P·J·Pᵀ
        pin = int(np.asarray(perm)[0])
    return _build(mesh, mu, pin)


def _newton_linear_solve(asm, state, preconditioned):
    """One Newton linear solve; returns (outer GMRES iterations, update)."""
    r = asm.residual(state)

    def jvp(v):
        return jax.jvp(asm.residual, (state,), (v,))[1]

    op = lx.FunctionLinearOperator(jvp, jax.ShapeDtypeStruct(r.shape, r.dtype))
    solver = lx.GMRES(rtol=1e-8, atol=1e-8)
    if not preconditioned:
        sol = lx.linear_solve(op, -r, solver=solver)
        return int(sol.stats["num_steps"]), sol.value
    m = BlockPreconditioner.build(asm).factory()(state)
    pop = lx.FunctionLinearOperator(lambda x: m(jvp(x)), jax.ShapeDtypeStruct(r.shape, r.dtype))
    sol = lx.linear_solve(pop, m(-r), solver=solver)
    return int(sol.stats["num_steps"]), sol.value


def test_preconditioner_reduces_outer_iterations() -> None:
    """The block-diagonal SIMPLE preconditioner cuts the outer GMRES count several-fold."""
    asm = _cavity(16)
    state = asm.initial_state()
    n_plain, _ = _newton_linear_solve(asm, state, preconditioned=False)
    n_prec, _ = _newton_linear_solve(asm, state, preconditioned=True)
    assert n_prec < n_plain / 4  # measured ~10x; guard a conservative factor


def test_preconditioner_is_a_drop_in() -> None:
    """Preconditioning changes only the Krylov path, not the Newton update it computes."""
    asm = _cavity(16)
    state = asm.initial_state()
    _, update_plain = _newton_linear_solve(asm, state, preconditioned=False)
    _, update_prec = _newton_linear_solve(asm, state, preconditioned=True)
    assert jnp.allclose(update_plain, update_prec, atol=1e-6)


def test_preconditioned_solve_converges_to_same_flow() -> None:
    """A full preconditioned Newton solve reaches the same converged field as the unpreconditioned."""
    asm = _cavity(16)
    precond = BlockPreconditioner.build(asm).factory()
    phi_plain = asm.initial_state()
    phi_prec = asm.initial_state()
    for _ in range(8):
        phi_plain = newton_step(asm.residual, phi_plain)
        phi_prec = newton_step(asm.residual, phi_prec, preconditioner=precond)
    assert float(jnp.linalg.norm(asm.residual(phi_prec))) < 1e-8
    assert jnp.allclose(phi_plain, phi_prec, atol=1e-6)


def _preconditioned_count(asm):
    return _newton_linear_solve(asm, asm.initial_state(), preconditioned=True)[0]


def test_preconditioned_solve_converges_under_any_ordering() -> None:
    """Convergence is order-robust: even a maximally-scrambled cell numbering keeps the outer
    GMRES count bounded and far below the unpreconditioned solve. The V-cycle smoother is
    permutation-invariant, so a bad ordering can only degrade the aggregation *coarse space*
    (its contraction factor) — which slows the inner rate but never breaks convergence, and no
    reordering is *required* for correctness.

    At this mesh size the block-triangular structure has enough slack that the outer count is
    essentially identical across orderings; the coarse-space penalty (see the RCM-restoration
    check in ``test_multigrid.py``) only reaches the *outer* count at large-mesh scale, which is
    why RCM is a large-mesh-pipeline step, not a correctness prerequisite."""
    n = 16
    asm_natural = _cavity(n)
    n_natural = _preconditioned_count(asm_natural)
    n_plain, _ = _newton_linear_solve(
        asm_natural, asm_natural.initial_state(), preconditioned=False
    )

    scramble = np.random.default_rng(0).permutation(n * n)
    n_scrambled = _preconditioned_count(_cavity(n, perm=scramble))

    assert n_scrambled < n_plain / 3  # still a strong preconditioner despite the worst ordering
    assert n_scrambled <= 2 * n_natural + 3  # bounded degradation, not divergence


def test_preconditioned_solve_is_differentiable() -> None:
    """Reverse-mode gradient through the preconditioned Newton solve is finite."""

    def mean_speed(mu):
        asm = _cavity(12, mu=mu)
        precond = BlockPreconditioner.build(asm).factory()
        state = asm.initial_state()
        for _ in range(8):
            state = newton_step(asm.residual, state, preconditioner=precond)
        velocity, _ = asm.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(MU))
    assert np.isfinite(grad)


def _dense_pressure_schur(assembler, state):
    """The true pressure Schur complement ``S = Ĉ - B F⁻¹ G``, densely, on a small problem."""
    ndof = state.shape[0]
    jacobian = np.zeros((ndof, ndof))
    for j in range(ndof):
        tangent = jnp.zeros(ndof).at[j].set(1.0)
        jacobian[:, j] = np.asarray(jax.jvp(assembler.residual, (state,), (tangent,))[1])
    n_cells = assembler.mesh.n_cells
    marker = assembler.pack(jnp.zeros((n_cells, assembler.mesh.dim)), jnp.arange(1.0, n_cells + 1))
    pressure = np.nonzero(np.asarray(marker))[0]
    velocity = np.setdiff1d(np.arange(ndof), pressure)
    momentum = jacobian[np.ix_(velocity, velocity)]
    gradient = jacobian[np.ix_(velocity, pressure)]
    divergence = jacobian[np.ix_(pressure, velocity)]
    coupling = jacobian[np.ix_(pressure, pressure)]
    return coupling - divergence @ np.linalg.solve(momentum, gradient)


def _dense_schur_preconditioner(assembler, state, scaling):
    """The Schur strategy's action as a dense matrix, for comparison against the true ``S⁻¹``."""
    from aquaflux.flow.block_preconditioner import FlowBlocks

    block = BlockPreconditioner.build(assembler, velocity="convection", schur_scaling=scaling)
    a_p = block.frozen_momentum_diagonal(state)
    schur_a_p = (
        a_p
        if block.schur_mass_diagonal is None
        else block.schur_mass_diagonal / block._msimpler_scale(state)
    )
    solve = block.schur.apply(schur_a_p, FlowBlocks.of(assembler, state))
    n_cells = assembler.mesh.n_cells
    dense = np.zeros((n_cells, n_cells))
    for j in range(n_cells):
        dense[:, j] = np.asarray(solve(jnp.zeros(n_cells).at[j].set(1.0)))
    return dense


def test_flow_saddle_pressure_block_is_positive_definite() -> None:
    """The sign convention every Schur strategy is written against, pinned.

    In this residual's signs the pressure--pressure block is *positive* definite and ``B F⁻¹ G`` is
    negative definite, so the Schur complement ``S = Ĉ - B F⁻¹ G`` is positive definite. The
    commutator Schur's formula comes from the literature's ``[[F, Bᵀ], [B, -C]]`` convention, whose
    ``Bᵀ`` is ``-G`` here — a sign flip that is invisible until the preconditioner diverges, so it is
    pinned rather than left to a comment.
    """
    assembler = _cavity(6)
    state = assembler.initial_state()
    schur = _dense_pressure_schur(assembler, state)
    eigenvalues = np.linalg.eigvalsh(0.5 * (schur + schur.T))
    assert eigenvalues[0] > 0.0


def test_stabilized_lsc_beats_the_scaled_laplacian_schur() -> None:
    """The commutator Schur is a closer approximation to ``S⁻¹`` than the scaled-Laplacian ones.

    Measured as ``‖I - S M‖``: the whole reason the commutator Schur exists is that the scaled
    Laplacian stops representing the Schur complement once convection matters, and no amount of extra
    accuracy in *inverting* it recovers that. The comparison is on the operator itself, so it is
    independent of the Krylov method wrapped around it.
    """
    assembler = _cavity(6)
    state = assembler.initial_state()
    schur = _dense_pressure_schur(assembler, state)
    identity = np.eye(schur.shape[0])

    errors = {
        scaling: np.linalg.norm(
            identity - schur @ _dense_schur_preconditioner(assembler, state, scaling), 2
        )
        for scaling in ("simple", "msimpler", "lsc")
    }
    assert errors["lsc"] < errors["simple"]
    assert errors["lsc"] < errors["msimpler"]


def test_stabilized_lsc_solves_the_saddle_system() -> None:
    """The commutator Schur is a valid preconditioner: it drives the Newton correction to solution.

    A sign or scaling error in the commutator composition still *runs* — it simply fails to converge —
    so correctness here means the preconditioned solve actually reaches the linear system's answer.
    """
    assembler = _cavity(10)
    state = assembler.initial_state()
    precond = BlockPreconditioner.build(
        assembler, velocity="convection", schur_scaling="lsc"
    ).factory()(state)

    def matvec(v):
        return jax.jvp(assembler.residual, (state,), (v,))[1]

    rhs = -assembler.residual(state)
    operator = lx.FunctionLinearOperator(lambda v: precond(matvec(v)), jax.eval_shape(lambda: rhs))
    solution = lx.linear_solve(
        operator, precond(rhs), lx.GMRES(rtol=1e-8, atol=1e-12, restart=40), throw=False
    )
    residual = float(jnp.linalg.norm(matvec(solution.value) - rhs) / jnp.linalg.norm(rhs))
    assert residual < 1e-6
