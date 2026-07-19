"""Newton on the residual + implicitly-differentiated linear solve, and the AMG that preconditions it.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.

**This module is the package's API boundary: everything the rest of the library (or a user) may
consume from `solve` is re-exported here, and consumers import from `aquaflux.solve`, not from its
submodules.** A name absent from `__all__` is internal — reach for it only from that submodule's own
unit tests. The surface is three groups:

* **Newton drivers and the linear solve** — `NewtonSolver`, `ImplicitNewtonSolver`, `newton_step`,
  `solve_linear`, `default_linear_solver`.
* **Forward globalization** — the `ForwardStep` strategies `DampedNewtonStep` and
  `PseudoTransientStep`, with the `ShiftPolicy` / `ShiftTerm` / `StepAcceptance` seams a caller
  implements and the default `DivergenceGuard`.
* **Frozen algebraic multigrid** — the operator assembler `convection_diffusion_operator` (plus
  `decouple_dof` for a closed-domain pressure pin), the hierarchy builders
  `build_smoothed_hierarchy` / `build_convection_hierarchy` / `build_air_hierarchy`, and their
  matching fixed-cycle applies. Callers assemble an operator, build a hierarchy once off the jit
  path, and apply it as a frozen matrix-free V-cycle preconditioner.
"""

from __future__ import annotations

from .continuation import (
    DivergenceGuard,
    PseudoTransientStep,
    ShiftPolicy,
    ShiftTerm,
    StepAcceptance,
)
from .frozen_operator import convection_diffusion_operator, decouple_dof
from .implicit import DampedNewtonStep, ForwardStep, ImplicitNewtonSolver
from .linear import default_linear_solver, solve_linear
from .multigrid import (
    AirHierarchy,
    SmoothedHierarchy,
    air_multigrid_solve,
    build_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_solve,
    smoothed_multigrid_solve,
)
from .newton import NewtonSolver, newton_step

__all__ = [
    "AirHierarchy",
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
    "air_multigrid_solve",
    "build_air_hierarchy",
    "build_convection_hierarchy",
    "build_smoothed_hierarchy",
    "convection_diffusion_operator",
    "convection_multigrid_solve",
    "decouple_dof",
    "default_linear_solver",
    "newton_step",
    "smoothed_multigrid_solve",
    "solve_linear",
]
