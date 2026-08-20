"""Which SIMPLE-type composition should the flow block's preconditioner apply?

The block preconditioner is built from two inner solves -- an approximate momentum inverse and an
approximate pressure-Schur inverse -- and a *composition* that says how many times and in what order
to apply them. Klaij & Vuik (2013) give three, and their finding is that the one with a **pressure
prediction** before the velocity solve (SIMPLER, their Algorithm 2) needs far fewer linear iterations
per nonlinear iteration than the one without (SIMPLE, their Algorithm 1). That claim had never been
checked here, because the prediction had never been implemented.

This measures all six pairings of the two axes that are actually independent:

* **the Schur scaling** -- the momentum diagonal ``a_P``, or the frozen mass diagonal ``rho V / k``
  (the paper's ``M`` prefix);
* **the composition** -- ``triangular`` (the lower block-triangular pass, this project's long-standing
  default), ``simple`` (Algorithm 1, which adds the closing velocity update), and ``simpler``
  (Algorithm 2, which adds the pressure prediction and a second Schur solve).

So ``msimple`` x ``simpler`` is the paper's MSIMPLER and ``simple`` x ``simpler`` its SIMPLER.

Two things about the method are load-bearing, and getting either wrong makes the whole table
meaningless:

* **Right preconditioning.** GMRES stops on the residual of the system it is handed. Left-precondition
  it and that residual is ``M(Ax - b)``, a *different* measure for every arm -- so the arms are ranked
  on incomparable quantities, and an arm can stop five orders short of the tolerance it reports. With
  right preconditioning (which is also what the solver ships) the measured and reported residual is
  the true one. The true relative residual is recomputed independently at the end regardless, and an
  arm that did not reach the tolerance is flagged rather than tabulated as if it had.
* **A state that is developed but NOT converged.** At a machine-precision root the right-hand side
  ``-R`` is roundoff, so every arm is ranked on noise -- and does, uninformatively, tie. The march is
  stopped at a loose relative tolerance instead, which is where a real inexact-Newton solve lives.

Run: ``python3 validation/simple_type_composition.py [nx ny mu march_rtol]``
(defaults: a 64x32 channel at ``mu = 4e-4``, i.e. Reynolds number 2500, marched to rel 1e-3;
a few minutes.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Run directly (`python3 validation/simple_type_composition.py`): Python puts THIS file's directory on
# the path, not the repository root, so put the root there ourselves -- as the case harnesses do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    momentum_continuation,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import ImplicitNewtonSolver, relative_residual_gmres

#: The tolerance every arm is driven to, on the TRUE residual, and the Krylov subspace size before a
#: restart. The cap exists so a failing arm reports a failure instead of running for the afternoon.
RTOL = 1e-8
RESTART = 30
MAX_RESTARTS = 200

SCALINGS = ("simple", "msimple")
COMPOSITIONS = ("triangular", "simple", "simpler")


def channel(nx: int, ny: int, mu: float, u_in: float = 1.0) -> MomentumContinuity:
    """A plane channel driven by a uniform inlet, with a pressure outlet fixing the pressure level."""
    mesh = structured_grid_2d(nx, ny, lx=8.0, ly=1.0, named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(mu), "density": Constant(1.0)}),
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


def developing(assembler: MomentumContinuity, rtol: float) -> jnp.ndarray:
    """March to a developed but deliberately *unconverged* state -- see the module docstring."""
    continuation = momentum_continuation(assembler, schur_scaling="msimple", velocity="convection")
    return ImplicitNewtonSolver(
        max_steps=200, rtol=rtol, atol=0.0, forward_step=continuation
    ).solve(lambda state, asm: asm.residual(state), assembler.initial_state(), assembler)


def probe(
    assembler: MomentumContinuity,
    state: jnp.ndarray,
    scaling: str,
    composition: str,
    velocity: str = "convection",
) -> tuple[int, float, float]:
    """Solve the real Newton system with one arm; return (restart cycles, TRUE rel, wall)."""
    residual = assembler.residual(state)

    def jvp(v: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(assembler.residual, (state,), (v,))[1]

    m = BlockPreconditioner.build(
        assembler, schur_scaling=scaling, composition=composition, velocity=velocity
    ).factory()(state)
    operator = lx.FunctionLinearOperator(
        lambda y: jvp(m(y)), jax.ShapeDtypeStruct(residual.shape, residual.dtype)
    )
    solver = relative_residual_gmres(
        RTOL, restart=RESTART, stagnation_iters=40, max_restarts=MAX_RESTARTS
    )

    @jax.jit
    def solve(b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        solution = lx.linear_solve(operator, b, solver=solver, throw=False)
        return m(solution.value), solution.stats["num_steps"]

    value, cycles = solve(-residual)
    value.block_until_ready()  # compile before timing
    started = time.perf_counter()
    value, cycles = solve(-residual)
    value.block_until_ready()
    wall = time.perf_counter() - started
    true_rel = float(jnp.linalg.norm(jvp(value) + residual) / jnp.linalg.norm(residual))
    return int(cycles), true_rel, wall


def main(nx: int, ny: int, mu: float, march_rtol: float) -> None:
    assembler = channel(nx, ny, mu)
    reference = float(jnp.linalg.norm(assembler.residual(assembler.initial_state())))
    print(
        f"channel {nx}x{ny}, mu={mu:g} (Reynolds number {1.0 / mu:.0f}), {nx * ny} cells; "
        f"marched to rel {march_rtol:.0e}",
        flush=True,
    )
    state = developing(assembler, march_rtol)
    norm = float(jnp.linalg.norm(assembler.residual(state)))
    print(f"state |R| = {norm:.3e} (rel {norm / reference:.2e})")
    print(
        f"right-preconditioned GMRES to TRUE rtol {RTOL:.0e}, restart {RESTART}, "
        f"cap {MAX_RESTARTS} cycles\n"
    )
    print(f"{'schur':<10}{'composition':<14}{'cycles':>8}{'TRUE rel':>12}{'wall':>10}")
    for scaling in SCALINGS:
        for composition in COMPOSITIONS:
            cycles, true_rel, wall = probe(assembler, state, scaling, composition)
            reached = "" if true_rel <= 10 * RTOL else "   <-- did NOT converge"
            print(
                f"{scaling:<10}{composition:<14}{cycles:>8}{true_rel:>12.2e}{wall:>9.2f}s{reached}",
                flush=True,
            )


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(
        int(argv[0]) if len(argv) > 0 else 64,
        int(argv[1]) if len(argv) > 1 else 32,
        float(argv[2]) if len(argv) > 2 else 4e-4,
        float(argv[3]) if len(argv) > 3 else 1e-3,
    )
