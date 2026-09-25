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
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
)
from aquaflux.mesh import permute_cells
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import newton_step

from tests.support.meshes import perturbed_grid_2d

RHO, MU = 1.0, 0.02

# Newton steps to drive a cavity solve to its root. The convergence is quadratic here -- the
# 12x12 and 16x16 cavities reach |R| ~ 1e-11 by the fourth step -- so this is convergence plus a
# step of margin. It is deliberately a named constant rather than a literal at each loop: the
# count is not a property any test here pins, and every step it runs past the root is a full
# preconditioned GMRES solve (and, in the differentiability check, another step on the tape).
_NEWTON_STEPS = 5


#: The pressure datum, named by a place: under a renumbering of the cells it still finds the same
#: physical cell, which an index would not.
_DATUM = PinnedPoint((0.0, 0.0))


def _build(mesh, mu=MU):
    """Build the lid-driven-cavity coupled p--U assembler on a given (possibly renumbered) mesh."""
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
        pressure_datum=_DATUM,
    )


def _cavity(n, mu=MU, perm=None):
    mesh = perturbed_grid_2d(n, n, perturb=0.15, named_boundaries=True)
    if perm is not None:
        mesh = permute_cells(mesh, perm)  # renumbered system P·J·Pᵀ
    return _build(mesh, mu)


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
    for _ in range(_NEWTON_STEPS):
        phi_plain = newton_step(asm.residual, phi_plain)
        phi_prec = newton_step(asm.residual, phi_prec, preconditioner=precond)
    assert float(jnp.linalg.norm(asm.residual(phi_prec))) < 1e-8
    assert jnp.allclose(phi_plain, phi_prec, atol=1e-6)


def test_the_pressure_prediction_is_a_drop_in_that_costs_fewer_outer_iterations() -> None:
    """SIMPLER converges the same Newton solve to the same flow, in fewer outer GMRES iterations.

    The pressure prediction changes only how ``M`` is assembled from its two inner solves, so it must
    leave the converged field alone (a preconditioner never moves the root) while buying iterations --
    which is the whole claim the prediction exists to make, and which cannot be checked at all while
    the step is missing. Cross-checked against the mass-scaled Schur, since the prediction's derivation
    assumes a Schur of the form it assembles.
    """
    asm = _cavity(16)
    triangular = BlockPreconditioner.build(asm, schur_scaling="msimple").factory()
    simpler = BlockPreconditioner.build(
        asm, schur_scaling="msimple", composition="simpler"
    ).factory()

    state = asm.initial_state()
    r = asm.residual(state)

    def jvp(v):
        return jax.jvp(asm.residual, (state,), (v,))[1]

    def count(precond):
        m = precond(state)
        op = lx.FunctionLinearOperator(lambda x: m(jvp(x)), jax.ShapeDtypeStruct(r.shape, r.dtype))
        sol = lx.linear_solve(op, m(-r), solver=lx.GMRES(rtol=1e-8, atol=1e-8))
        return int(sol.stats["num_steps"])

    assert count(simpler) < count(triangular)

    phi_triangular, phi_simpler = asm.initial_state(), asm.initial_state()
    for _ in range(_NEWTON_STEPS):
        phi_triangular = newton_step(asm.residual, phi_triangular, preconditioner=triangular)
        phi_simpler = newton_step(asm.residual, phi_simpler, preconditioner=simpler)
    assert float(jnp.linalg.norm(asm.residual(phi_simpler))) < 1e-8
    assert jnp.allclose(phi_triangular, phi_simpler, atol=1e-6)


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
    """Reverse-mode gradient through the preconditioned Newton solve is finite and not severed.

    The non-zero assertion is the load-bearing half. ``0.0`` is finite, so a ``stop_gradient`` left
    anywhere on this path -- in an operator, a property, a boundary closure -- satisfies a finiteness
    check and reports nothing at all. It does not make this a check of the gradient's *value*: that
    needs a finite difference, which costs two more solves of a test that already runs for about a
    minute, and it is covered against closed forms in the flow-adjoint suite instead.
    """

    def mean_speed(mu):
        asm = _cavity(12, mu=mu)
        precond = BlockPreconditioner.build(asm).factory()
        state = asm.initial_state()
        for _ in range(_NEWTON_STEPS):
            state = newton_step(asm.residual, state, preconditioner=precond)
        velocity, _ = asm.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(MU))
    assert np.isfinite(grad)
    assert grad != 0.0


def _dense_pressure_schur(assembler, state):
    """The true pressure Schur complement ``S = Ĉ - B F⁻¹ G``, densely, on a small problem."""
    # One batched forward-mode pass, not one dispatch per column: `jacfwd` pushes the whole
    # identity basis through in a single `vmap`, which is the same matrix an eager column loop
    # builds and is what keeps this dense construction affordable in the always-on tier.
    jacobian = np.asarray(jax.jacfwd(assembler.residual)(state))
    ndof = state.shape[0]
    n_cells = assembler.mesh.n_cells
    marker = assembler.pack(jnp.zeros((n_cells, assembler.mesh.dim)), jnp.arange(1.0, n_cells + 1))
    pressure = np.nonzero(np.asarray(marker))[0]
    velocity = np.setdiff1d(np.arange(ndof), pressure)
    momentum = jacobian[np.ix_(velocity, velocity)]
    gradient = jacobian[np.ix_(velocity, pressure)]
    divergence = jacobian[np.ix_(pressure, velocity)]
    coupling = jacobian[np.ix_(pressure, pressure)]
    return coupling - divergence @ np.linalg.solve(momentum, gradient)


def test_flow_saddle_pressure_block_is_positive_definite() -> None:
    """The sign convention every Schur strategy is written against, pinned.

    In this residual's signs the pressure--pressure block is *positive* definite and ``B F⁻¹ G`` is
    negative definite, so the Schur complement ``S = Ĉ - B F⁻¹ G`` is positive definite. A formula taken
    from the literature's ``[[F, Bᵀ], [B, -C]]`` convention, whose ``Bᵀ`` is ``-G`` here, picks up a
    sign flip that is invisible until the preconditioner diverges, so it is pinned rather than left to
    a comment.
    """
    assembler = _cavity(6)
    state = assembler.initial_state()
    schur = _dense_pressure_schur(assembler, state)
    eigenvalues = np.linalg.eigvalsh(0.5 * (schur + schur.T))
    assert eigenvalues[0] > 0.0
