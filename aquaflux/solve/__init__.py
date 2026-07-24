"""Newton on the residual + implicitly-differentiated linear solve, and the AMG that preconditions it.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.

**This module is the package's API boundary: everything the rest of the library (or a user) may
consume from `solve` is re-exported here, and consumers import from `aquaflux.solve`, not from its
submodules.** A name absent from `__all__` is internal — reach for it only from that submodule's own
unit tests. The surface is three groups:

* **The Newton driver, the single step, and the linear solve** — `ImplicitNewtonSolver` (the
  driver: converges, globalizes, and carries the implicit-function-theorem adjoint), `newton_step`
  (one matrix-free correction — exact in one call for a linear residual, and differentiable in both
  modes), `solve_linear` (returns the solution together with the solve's restart-cycle count —
  the staleness signal a mid-march preconditioner refresh triggers on), `default_linear_solver`.
* **Forward globalization** — the `ForwardStep` strategies `DampedNewtonStep` and
  `PseudoTransientStep`, with the `ShiftPolicy` / `ShiftTerm` / `StepAcceptance` seams a caller
  implements and the default `DivergenceGuard`, and the injected `ResidualNorm` the strategy judges
  progress by (default the Euclidean norm; `BlockScaledNorm` scales each block of a heterogeneous
  state by its own reference magnitude so no single large-magnitude block dominates the convergence
  test or the globalization). The pseudo-transient shift strength is itself an injected
  `RelaxationSchedule` — `SwitchedEvolutionRelaxation` (SER, the default) or `ConstantRelaxation`
  (a fixed β an external control sets) — a memoryless rule that stays on the differentiable path.
* **Observed-march step control (forward-only, experimental)** — a `StepControl` reshapes the eager
  march's step each iteration from the previous step's feedback (the line-search factor), where a
  memoryless schedule cannot. `AlphaTargetingControl` drives β toward the α=1 boundary; it beats SER
  on a stiff coupled march *with* a preconditioner refresh but does not yet converge standalone, so
  it is opt-in and never a default.
* **The observed forward march** — `forward_march`, an eager, forward-only march that applies the
  same `ForwardStep` as the Newton driver but reports each step (`StepReport`, `MarchResult`) and
  may stop early. It is what lets a driver rebuild a frozen preconditioner part way through a solve,
  on the evidence of the `RefreshTrigger` it injects (`CycleGrowthTrigger` watches the per-step
  linear-solve cost). It is an accelerator, not a solver: a real `ImplicitNewtonSolver` solve still
  produces the result.
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
from .march import (
    CycleGrowthTrigger,
    MarchResult,
    RefreshTrigger,
    StepControl,
    StepReport,
    forward_march,
)
from .multigrid import (
    AirHierarchy,
    SmoothedHierarchy,
    air_multigrid_solve,
    build_air_hierarchy,
    refresh_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_solve,
    smoothed_multigrid_solve,
)
from .newton import newton_step
from .norm import BlockScaledNorm, ResidualNorm
from .relaxation import ConstantRelaxation, RelaxationSchedule, SwitchedEvolutionRelaxation
from .step_control import AlphaTargetingControl

__all__ = [
    "AirHierarchy",
    "AlphaTargetingControl",
    "BlockScaledNorm",
    "ConstantRelaxation",
    "CycleGrowthTrigger",
    "DampedNewtonStep",
    "DivergenceGuard",
    "ForwardStep",
    "ImplicitNewtonSolver",
    "MarchResult",
    "PseudoTransientStep",
    "RefreshTrigger",
    "RelaxationSchedule",
    "ResidualNorm",
    "ShiftPolicy",
    "ShiftTerm",
    "SmoothedHierarchy",
    "StepAcceptance",
    "StepControl",
    "StepReport",
    "SwitchedEvolutionRelaxation",
    "air_multigrid_solve",
    "build_air_hierarchy",
    "build_convection_hierarchy",
    "build_smoothed_hierarchy",
    "convection_diffusion_operator",
    "convection_multigrid_solve",
    "decouple_dof",
    "default_linear_solver",
    "forward_march",
    "newton_step",
    "refresh_air_hierarchy",
    "smoothed_multigrid_solve",
    "solve_linear",
]
