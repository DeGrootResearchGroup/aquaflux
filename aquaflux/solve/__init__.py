"""Newton on the residual + implicitly-differentiated linear solve.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.
"""

from __future__ import annotations

from .continuation import (
    DivergenceGuard,
    PseudoTransientStep,
    ShiftPolicy,
    ShiftTerm,
    StepAcceptance,
)
from .implicit import DampedNewtonStep, ForwardStep, ImplicitNewtonSolver
from .linear import default_linear_solver, solve_linear
from .multigrid import (
    SmoothedHierarchy,
    build_smoothed_hierarchy,
    smoothed_multigrid_solve,
)
from .newton import NewtonSolver, newton_step

__all__ = [
    "DampedNewtonStep",
    "DivergenceGuard",
    "ForwardStep",
    "ImplicitNewtonSolver",
    "NewtonSolver",
    "PseudoTransientStep",
    "ShiftPolicy",
    "ShiftTerm",
    "SmoothedHierarchy",
    "StepAcceptance",
    "build_smoothed_hierarchy",
    "default_linear_solver",
    "newton_step",
    "smoothed_multigrid_solve",
    "solve_linear",
]
