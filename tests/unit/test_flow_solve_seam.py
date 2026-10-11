"""A flow solve returns the assembler with the state, whichever builder made it.

The segregated loop calls its flow solve as ``solve(momentum, state) -> (momentum, state)``, because a
bulk-velocity-constrained solve carries its converged body force out on the assembler. The
unconstrained builder returned the state alone, so a case handing it to the loop failed on the first
sweep with an unpacking error, and the tests of that handover replaced the loop with a recorder and
never noticed. This pins the shape at the builder, with the Newton solve replaced so no solve is run.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.flow import reused_flow_solve
from aquaflux.solve import RootSolveSettings

from tests.unit.test_coupled_rans import _cavity


class _Solver:
    """Stands in for the Newton solver: records what it is asked to solve and returns a marker state."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def solve(self, residual, state, theta):
        self.calls.append((state, theta))
        return self.answer


def test_the_unconstrained_flow_solve_hands_back_the_assembler_it_was_given(monkeypatch) -> None:
    _, coupled = _cavity(4)
    momentum = coupled.momentum
    start = momentum.initial_state()
    solver = _Solver(start + 1.0)
    monkeypatch.setattr(RootSolveSettings, "solver", lambda self, strategy, **fields: solver)

    # Built from a different viscosity, as a segregated loop builds it once at a reference and calls it
    # on each sweep's assembler: what comes back must be the one it was called with.
    reference = momentum.with_scaled_molecular_viscosity(2.0)
    returned, state = reused_flow_solve(reference)(momentum, start)

    # An unconstrained solve has nothing to change on the assembler, so it is the very object.
    assert returned is momentum
    assert jnp.array_equal(state, start + 1.0)
    # And the solve ran on the assembler it was called with, not on the reference it was built from.
    [(solved_from, theta)] = solver.calls
    assert solved_from is start
    assert theta is momentum
