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
  the staleness signal a mid-march preconditioner refresh triggers on), `default_linear_solver`, and
  `relative_residual_gmres` (a GMRES that stops on a *global* 2-norm relative residual — the robust
  inexact-Newton forward stop, immune to the near-zero-right-hand-side rows that make the stock
  componentwise test over-solve).
* **Forward globalization** — the `ForwardStep` strategies `DampedNewtonStep` and
  `PseudoTransientStep`, with the `ShiftPolicy` / `ShiftTerm` / `StepAcceptance` seams a caller
  implements and the default `DivergenceGuard`, and the injected `ResidualNorm` the strategy judges
  progress by (default the Euclidean norm; `BlockScaledNorm` scales each block of a heterogeneous
  state by its own reference magnitude so no single large-magnitude block dominates the convergence
  test or the globalization). The pseudo-transient shift strength is itself an injected
  `RelaxationSchedule` — `SwitchedEvolutionRelaxation` (SER, the default) or `ConstantRelaxation`
  (a fixed β an external control sets) — a memoryless rule that stays on the differentiable path.
* **Observed-march step control (forward-only, experimental)** — a `StepControl` reshapes the eager
  march's step each iteration from the previous step's feedback, where a memoryless schedule cannot.
  `AlphaTargetingControl` drives the single-step β toward the α=1 boundary from the line-search factor.
  `DualTimeControl` ramps a dual-time pseudo-timestep by that same inner-loop comfort — but growing on
  inner comfort alone is blind to the steady residual and can run the transient away.
  `ResidualRatioDualTimeControl` fixes that: it ramps the pseudo-timestep by the steady-residual
  reduction ratio (switched evolution relaxation / Kelley–Keyes pseudo-transient continuation), so a
  rising residual automatically shrinks the step — but keying growth on the residual alone stalls where
  the residual is flat while the flow develops. `CflResidualDualTimeControl` combines them: it grows on
  the inner-loop comfort α (fast on the flat-residual development) but brakes on a rising residual (safe
  on the overshoot), the two signals covering each other's blind spots. All are opt-in and never a
  default; the finishing solve owns the converged root and the adjoint.
* **The observed forward march** — `forward_march`, an eager, forward-only march that applies the
  same `ForwardStep` as the Newton driver but reports each step (`StepReport`, `MarchResult`) and
  may stop early. It is what lets a driver rebuild a frozen preconditioner part way through a solve,
  on the evidence of the `RefreshTrigger` it injects — `CoefficientDriftTrigger` watches how far the
  operator's own coefficients have moved since they were frozen (the direct staleness signal, fed by
  the march's `drift_measure`), while `CycleGrowthTrigger` infers it from the per-step linear-solve
  cost. It is an accelerator, not a solver: a real `ImplicitNewtonSolver` solve still produces the
  result.
* **Frozen algebraic multigrid** — the operator assembler `convection_diffusion_operator` (plus
  `decouple_dof` for a closed-domain pressure pin), the hierarchy builders
  `build_smoothed_hierarchy` / `build_convection_hierarchy` / `build_air_hierarchy`, and their
  matching fixed-cycle applies. Callers assemble an operator, build a hierarchy once off the jit
  path, and apply it as a frozen matrix-free V-cycle preconditioner.
"""

from __future__ import annotations

from .continuation import (
    DivergenceGuard,
    DualTimeStep,
    PseudoTransientStep,
    ShiftPolicy,
    ShiftTerm,
    StepAcceptance,
)
from .frozen_operator import convection_diffusion_operator, decouple_dof
from .amg_preconditioner import AmgVCycle, MonolithicAmgPreconditioner, build_amg_vcycle
from .ilut_preconditioner import MonolithicIlutPreconditioner
from .lu_preconditioner import MonolithicLuPreconditioner
from .implicit import (
    DampedNewtonStep,
    ForwardStep,
    ImplicitNewtonSolver,
    TransposedPreconditioner,
)
from .line_search_growth import (
    LineSearchGrowth,
    MonotoneLineSearch,
    RelaxedFarFromRoot,
)
from .linear import default_linear_solver, relative_residual_gmres, solve_linear
from .march import (
    CoefficientDriftTrigger,
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
from .norm import BlockScaledNorm, ResidualNorm, RowScaledNorm
from .relaxation import ConstantRelaxation, RelaxationSchedule, SwitchedEvolutionRelaxation
from .shift_basis import LocalCourantBasis, ShiftBasis, VelocityShiftParts
from .sparse_jacobian import (
    BlockColouring,
    block_stencil_colouring,
    jacobian_relative_error,
    materialize_block_jacobian,
)
from .step_control import (
    AlphaTargetingControl,
    CflResidualDualTimeControl,
    DualTimeControl,
    ResidualRatioDualTimeControl,
)

__all__ = [
    "AirHierarchy",
    "AlphaTargetingControl",
    "AmgVCycle",
    "BlockColouring",
    "BlockScaledNorm",
    "CflResidualDualTimeControl",
    "CoefficientDriftTrigger",
    "ConstantRelaxation",
    "CycleGrowthTrigger",
    "DampedNewtonStep",
    "DivergenceGuard",
    "DualTimeControl",
    "DualTimeStep",
    "ForwardStep",
    "ImplicitNewtonSolver",
    "LineSearchGrowth",
    "LocalCourantBasis",
    "MarchResult",
    "MonolithicAmgPreconditioner",
    "MonolithicIlutPreconditioner",
    "MonolithicLuPreconditioner",
    "MonotoneLineSearch",
    "PseudoTransientStep",
    "RefreshTrigger",
    "RelaxationSchedule",
    "RelaxedFarFromRoot",
    "ResidualNorm",
    "ResidualRatioDualTimeControl",
    "RowScaledNorm",
    "ShiftBasis",
    "ShiftPolicy",
    "ShiftTerm",
    "SmoothedHierarchy",
    "StepAcceptance",
    "StepControl",
    "StepReport",
    "SwitchedEvolutionRelaxation",
    "TransposedPreconditioner",
    "VelocityShiftParts",
    "air_multigrid_solve",
    "block_stencil_colouring",
    "build_air_hierarchy",
    "build_amg_vcycle",
    "build_convection_hierarchy",
    "build_smoothed_hierarchy",
    "convection_diffusion_operator",
    "convection_multigrid_solve",
    "decouple_dof",
    "default_linear_solver",
    "forward_march",
    "jacobian_relative_error",
    "materialize_block_jacobian",
    "newton_step",
    "refresh_air_hierarchy",
    "relative_residual_gmres",
    "smoothed_multigrid_solve",
    "solve_linear",
]
