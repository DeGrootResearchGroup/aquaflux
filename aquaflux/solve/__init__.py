"""Newton on the residual + implicitly-differentiated linear solve.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.
"""

from __future__ import annotations

from .implicit import DampedNewtonStep, ForwardStep, ImplicitNewtonSolver
from .linear import default_linear_solver, solve_linear
from .multigrid import (
    MultigridHierarchy,
    SmoothedHierarchy,
    build_hierarchy,
    build_smoothed_hierarchy,
    level_coefficients,
    multigrid_solve,
    smoothed_multigrid_solve,
    v_cycle,
)
from .newton import NewtonSolver, newton_step

__all__ = [
    "DampedNewtonStep",
    "ForwardStep",
    "ImplicitNewtonSolver",
    "MultigridHierarchy",
    "NewtonSolver",
    "SmoothedHierarchy",
    "build_hierarchy",
    "build_smoothed_hierarchy",
    "default_linear_solver",
    "level_coefficients",
    "multigrid_solve",
    "newton_step",
    "smoothed_multigrid_solve",
    "solve_linear",
    "v_cycle",
]
