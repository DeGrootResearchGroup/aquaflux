"""Pseudo-transient continuation for the SST scalar (k, omega) transport solves.

Exercises the scalar counterpart of the flow block's globalization: the transport-operator shift
diagonal (the ``a_P`` analogue), the :class:`~aquaflux.turbulence.ScalarShiftPolicy` that carries it
with the frozen AMG, and the continuation solve built from them. The behaviours pinned here are the
ones the driver's continuation mode relies on: the shift is the (positive part of the) true operator
diagonal, the continuation solve converges a stiff reactive scalar from a cold start where a
fixed-count Newton solve stalls, the converged field solves the *unshifted* residual (the shift
vanishes), and the underlying engine leaves a clean implicit-function-theorem adjoint.
"""

from __future__ import annotations

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
from aquaflux.properties import Constant, PropertyModel
from aquaflux.solve.continuation import PseudoTransientStep
from aquaflux.solve.implicit import ImplicitNewtonSolver
from aquaflux.solve.newton import newton_step
from aquaflux.solve.relaxation import SwitchedEvolutionRelaxation
from aquaflux.turbulence import ScalarShiftPolicy, scalar_pseudo_transient_solve
from aquaflux.turbulence.preconditioner import (
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)

U, GAMMA, REACTION = 1.0, 1e-3, 5.0  # convection-dominated + a nonlinear destruction sink


def _reactive_transport(nx, ny):
    """A convection-diffusion scalar with a nonlinear destruction sink ``-REACTION * phi**2``.

    The sink is the stiffness the k/omega destruction terms carry: it grows with the field, so a full
    Newton step from a large cold start overshoots -- the case continuation must globalize.
    """
    mesh = structured_grid_2d(nx, ny, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    volume_flux = U * geometry.face.normal[:, 0] * geometry.face.area  # uniform u = (U, 0)
    base = ResidualAssembler.build(
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
    volume = geometry.cell.volume

    def residual(phi):
        return base.residual(phi) + REACTION * phi**2 * volume

    return mesh, geometry, volume_flux, residual


def test_shift_diagonal_is_the_positive_operator_diagonal() -> None:
    """The shift base is the non-negative part of the true Jacobian diagonal -- the ``a_P`` analogue."""
    mesh, geometry, volume_flux, residual = _reactive_transport(20, 10)
    reference = jnp.full(mesh.n_cells, 0.5)
    shift = scalar_transport_shift_diagonal(
        mesh, geometry, jnp.full(mesh.n_cells, GAMMA), volume_flux, residual, reference
    )
    # Materialize the exact Jacobian diagonal with unit probes.
    eye = jnp.eye(mesh.n_cells)
    j_diagonal = jnp.array(
        [jax.jvp(residual, (reference,), (eye[i],))[1][i] for i in range(mesh.n_cells)]
    )
    assert bool(jnp.all(shift >= 0.0))
    assert float(jnp.max(jnp.abs(shift - jnp.maximum(j_diagonal, 0.0)))) < 1e-12


def test_fixed_cells_get_zero_shift() -> None:
    """A value-fixation cell carries no pseudo-time shift (a full Newton step converges it exactly)."""
    mesh, geometry, volume_flux, residual = _reactive_transport(16, 8)
    reference = jnp.full(mesh.n_cells, 0.5)
    fixed = jnp.array([0, 5, 11])
    shift = scalar_transport_shift_diagonal(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA),
        volume_flux,
        residual,
        reference,
        fixed_cells=fixed,
    )
    assert bool(jnp.all(shift[fixed] == 0.0))
    others = jnp.setdiff1d(jnp.arange(mesh.n_cells), fixed)
    assert bool(jnp.all(shift[others] > 0.0))


def test_shift_policy_carries_a_frozen_term() -> None:
    """``ScalarShiftPolicy.shift_term`` returns the frozen shift diagonal and the AMG preconditioner."""
    mesh, geometry, volume_flux, residual = _reactive_transport(16, 8)
    reference = jnp.full(mesh.n_cells, 0.5)
    shift = scalar_transport_shift_diagonal(
        mesh, geometry, jnp.full(mesh.n_cells, GAMMA), volume_flux, residual, reference
    )
    precond = scalar_transport_preconditioner(
        mesh, geometry, jnp.full(mesh.n_cells, GAMMA), volume_flux, residual, reference
    )
    term = ScalarShiftPolicy(shift, precond).shift_term(reference)
    assert bool(jnp.array_equal(term.diagonal, shift))
    # The preconditioner is beta-independent (the same frozen matvec at any relaxation).
    m_small, m_large = term.make_preconditioner(0.1), term.make_preconditioner(9.0)
    probe = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
    assert float(jnp.linalg.norm(m_small(probe) - m_large(probe))) == 0.0
    # A policy without a preconditioner yields an unpreconditioned (None) term.
    assert ScalarShiftPolicy(shift).shift_term(reference).make_preconditioner(1.0) is None


def test_continuation_globalizes_where_fixed_count_newton_stalls() -> None:
    """From a large cold start the continuation solve converges the stiff reactive scalar (staying
    positive), where the fixed-count Newton loop the scalar sub-solve used before does not."""
    mesh, geometry, volume_flux, residual = _reactive_transport(24, 12)
    reference = jnp.full(mesh.n_cells, 0.5)
    n = mesh.n_cells
    precond = scalar_transport_preconditioner(
        mesh, geometry, jnp.full(n, GAMMA), volume_flux, residual, reference, method="twolevel"
    )
    shift = scalar_transport_shift_diagonal(
        mesh, geometry, jnp.full(n, GAMMA), volume_flux, residual, reference
    )
    cold = jnp.full(n, 5.0)  # a poor, large initial guess

    solved = scalar_pseudo_transient_solve(max_steps=60)(
        residual, cold, ScalarShiftPolicy(shift, precond)
    )
    assert float(jnp.linalg.norm(residual(solved))) < 1e-8
    assert bool(jnp.all(solved > 0.0))

    # The baseline this is measured against: six *unglobalized* Newton steps, written out here
    # because a fixed count with no convergence test is the thing being shown to be insufficient.
    newton = cold
    for _ in range(6):
        newton = newton_step(residual, newton, preconditioner=precond)
    # The globalized solve is at least three orders of magnitude tighter than fixed-count Newton.
    assert float(jnp.linalg.norm(residual(solved))) < 1e-3 * float(
        jnp.linalg.norm(residual(newton))
    )


def test_none_policy_falls_back_to_plain_continuation() -> None:
    """A ``None`` policy still converges (unshifted, unpreconditioned continuation)."""
    mesh, _, _, residual = _reactive_transport(16, 8)
    solved = scalar_pseudo_transient_solve(max_steps=60)(
        residual, jnp.full(mesh.n_cells, 2.0), None
    )
    assert float(jnp.linalg.norm(residual(solved))) < 1e-8


def test_engine_leaves_a_clean_ift_adjoint() -> None:
    """The continuation engine is adjoint-transparent: with the residual parameters threaded as
    ``theta``, the reverse-mode gradient through the converged solve matches a finite difference (the
    frozen shift and preconditioner do not pollute it)."""
    mesh, geometry, volume_flux, _ = _reactive_transport(20, 10)
    n = mesh.n_cells
    volume = geometry.cell.volume
    base = ResidualAssembler.build(
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

    def residual_fn(phi, c):  # c (a reaction coefficient) is theta -- explicit and differentiable
        return base.residual(phi) + c * phi**2 * volume

    shift = scalar_transport_shift_diagonal(
        mesh,
        geometry,
        jnp.full(n, GAMMA),
        volume_flux,
        lambda p: residual_fn(p, REACTION),
        jnp.full(n, 0.5),
    )

    def solved_norm(c):
        engine = PseudoTransientStep(
            ScalarShiftPolicy(shift), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0)
        )
        solver = ImplicitNewtonSolver(max_steps=60, forward_step=engine)
        return jnp.sum(solver.solve(residual_fn, jnp.full(n, 1.0), c) ** 2)

    grad = float(jax.grad(solved_norm)(REACTION))
    fd = float((solved_norm(REACTION + 1e-5) - solved_norm(REACTION - 1e-5)) / 2e-5)
    assert abs(grad - fd) < 1e-6 * abs(fd)


def _sweep_traces(*, freeze):
    """Trace counts per simulated outer sweep, reusing (or rebuilding) the AMG preconditioner.

    Mimics the driver's sweep: a shift diagonal rebuilt from a changing diffusivity each sweep, with
    the preconditioner either carried across sweeps or rebuilt. Returns the per-sweep count of
    residual traces and the converged field.
    """
    mesh, geometry, volume_flux, residual = _reactive_transport(16, 8)
    n = mesh.n_cells
    reference, gamma = jnp.full(n, 0.5), jnp.full(n, GAMMA)
    traces = {"n": 0}

    def counting_residual(phi):
        traces["n"] += 1
        return residual(phi)

    solve = scalar_pseudo_transient_solve(max_steps=20)
    carried = scalar_transport_preconditioner(
        mesh, geometry, gamma, volume_flux, residual, reference
    )
    state = jnp.zeros(n) + 1.0  # not a weak-typed literal: avoids a spurious second compile
    per_sweep = []
    for sweep in range(4):
        diffusivity = gamma * (1.0 + 0.3 * sweep)
        shift = scalar_transport_shift_diagonal(
            mesh, geometry, diffusivity, volume_flux, residual, reference
        )
        precond = (
            carried
            if freeze
            else scalar_transport_preconditioner(
                mesh, geometry, diffusivity, volume_flux, residual, reference
            )
        )
        before = traces["n"]
        state = solve(counting_residual, state, ScalarShiftPolicy(shift, precond))
        per_sweep.append(traces["n"] - before)
    return per_sweep, state


def test_a_carried_preconditioner_compiles_the_scalar_solve_once() -> None:
    """Reusing one preconditioner across sweeps makes the jitted solve a compilation-cache hit.

    The preconditioner is a frozen, off-jit constant, so ``equinox.filter_jit`` keeps it on the
    static side, hashed by object identity: carrying **one** instance across sweeps reuses the
    compiled solve, while a freshly built one each sweep re-compiles the whole
    ``ImplicitNewtonSolver`` -- the escalation loop, the GMRES, and the V-cycle. Only the shift
    diagonal changes per sweep, and being an array it re-traces nothing.
    """
    frozen, frozen_state = _sweep_traces(freeze=True)
    rebuilt, rebuilt_state = _sweep_traces(freeze=False)

    assert frozen[0] > 0  # the first sweep compiles
    assert frozen[1:] == [0, 0, 0]  # and every later sweep is a cache hit
    assert all(count > 0 for count in rebuilt)  # rebuilding re-compiles every sweep
    # Freezing the preconditioner only changes how fast the Krylov iteration converges, never the
    # converged field -- so the two paths agree to solver tolerance.
    assert float(jnp.max(jnp.abs(frozen_state - rebuilt_state))) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
