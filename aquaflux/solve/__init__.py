"""Newton on the residual + implicitly-differentiated linear solve, and the AMG that preconditions it.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.

**This module is the package's API boundary: everything the rest of the library (or a user) may
consume from `solve` is re-exported here, and consumers import from `aquaflux.solve`, not from its
submodules.** A name absent from `__all__` is internal — reach for it only from that submodule's own
unit tests. The surface is five groups:

* **The Newton driver, the single step, and the linear solve** — `RootSolver` (the
  driver: converges, globalizes, and carries the implicit-function-theorem adjoint), `root_adjoint`
  (that adjoint on its own: attaches the derivative of a root to a root found by any means),
  `assembler_residual` (the two-argument residual of an assembler passed as the differentiated
  parameter — one shared object, so repeated solves reuse the compiled march step that a lambda
  written at the call site would rebuild), `newton_step`
  (one matrix-free correction — exact in one call for a linear residual, and differentiable in both
  modes), `solve_linear` (returns the solution together with the solve's restart-cycle count —
  the staleness signal a mid-march preconditioner refresh triggers on), `default_linear_solver`, and
  `relative_residual_gmres` (a GMRES that stops on a *global* relative residual in an injected norm —
  the robust inexact-Newton forward stop, immune to the near-zero-right-hand-side rows that make the
  stock componentwise test over-solve; the default is the Euclidean norm, and passing the row-scaled
  `RowScaledNorm` makes the stop weigh every field block comparably instead of letting the
  largest-magnitude block — `omega` on the coupled saddle — decide alone).
* **Forward globalization** — the `NewtonStrategy` strategies `DampedNewtonStep`,
  `PseudoTransientStep` and `DualTimeStep`, with the `ShiftPolicy` / `ShiftTerm` / `StepAcceptance`
  seams a caller
  implements and the default `DivergenceGuard`, and the injected `ResidualNorm` the strategy judges
  progress by (default the Euclidean norm; `BlockScaledNorm` scales each block of a heterogeneous
  state by its own reference magnitude so no single large-magnitude block dominates the convergence
  test or the globalization). The pseudo-transient shift strength is itself an injected
  `RelaxationSchedule` — `SwitchedEvolutionRelaxation` (SER, the default) or `ConstantRelaxation`
  (a fixed β an external control sets) — a memoryless rule that stays on the differentiable path.
* **Observed-march step control (forward-only, experimental)** — a `StepControl` reshapes the eager
  march's step each iteration from the previous step's feedback, where a memoryless schedule cannot.
  All three drive the pseudo-transient shift strength β and share one body, `ShiftStrengthControl`,
  supplying only their adaptation rule. `DualTimeControl` ramps a dual-time pseudo-timestep by the
  inner loop's comfort (its line-search factor α) — but growing on inner comfort alone is blind to the
  steady residual and can run the transient away. `ResidualRatioDualTimeControl` fixes that: it ramps
  the pseudo-timestep by the steady-residual reduction ratio (switched evolution relaxation /
  Kelley–Keyes pseudo-transient continuation), so a rising residual automatically shrinks the step — but
  keying growth on the residual alone stalls where the residual is flat while the flow develops.
  `CflResidualDualTimeControl` combines them: it grows on the inner-loop comfort α (fast on the
  flat-residual development) but brakes on a rising residual (safe on the overshoot), the two signals
  covering each other's blind spots, and reduces exactly to `DualTimeControl` at infinite ratio
  thresholds. A control changes only the path: the root is the residual's, and the adjoint is attached
  at it regardless.
* **The forward march** — `newton_march`, the eager, forward-only march every Newton solve in the
  package runs on, reporting each step (`StepReport`, `MarchResult`) and able to stop early. It is what lets a driver rebuild a frozen preconditioner part way through a solve,
  on the evidence of the `RefreshTrigger` it injects — `CoefficientDriftTrigger` watches how far the
  operator's own coefficients have moved since they were frozen (the direct staleness signal, fed by
  the march's `drift_measure`), while `CycleGrowthTrigger` infers it from the per-step linear-solve
  cost. It carries no convergence guard, so the driver that runs it reads `MarchResult.converged`
  before treating the state as an answer — `RootSolver` does exactly that, then attaches
  `root_adjoint`. Because it steps in Python it cannot run inside a traced program -- `jax.jit`,
  `jax.vmap`, or a traced loop such as `jax.lax.scan` -- and every
  driver refuses those up front with `refuse_a_transform_the_march_cannot_run_in`; `jax.grad` is
  unaffected, since the march runs on stopped copies and the derivative is attached at the root
  afterwards. When and how it redoes a bad step is one injected `RetryPolicy` — the three escalation
  triggers (a costly solve, a collapsed step length, a diverged correction), the shift factor and
  escalation limit, and the optional tighter linear solver that is the fallback for a step more
  damping cannot fix. The default policy retries nothing.
* **Frozen algebraic multigrid** — the operator assembler `convection_diffusion_operator` (plus
  `decouple_dof` for a closed-domain pressure pin and `symmetrically_equilibrate` for the
  square-root-diagonal rescaling a factorization or a coarsening may want), the hierarchy builders
  `build_smoothed_hierarchy` / `build_convection_hierarchy` / `build_air_hierarchy`, and their
  matching fixed-cycle applies. Callers assemble an operator, build a hierarchy once off the jit
  path, and apply it as a frozen matrix-free V-cycle preconditioner.
"""

from __future__ import annotations

from .continuation import (
    DEFAULT_GLOBALIZATION,
    DivergenceGuard,
    DualTimeLoop,
    DualTimeStep,
    Globalization,
    PseudoTransientStep,
    ShiftPolicy,
    ShiftTerm,
    StepAcceptance,
)
from .frozen_operator import (
    cell_major_permutation,
    convection_diffusion_operator,
    equilibrate_cell_major,
    decouple_dof,
    symmetrically_equilibrate,
)
from .amg_preconditioner import (
    AmgVCycle,
    MaterializedJacobianPreconditioner,
    MonolithicAmgPreconditioner,
    build_amg_vcycle,
)
from .field_split import (
    JacobiSmoothedInverse,
    BlockTriangularFieldSplit,
    FieldGroups,
    FieldSplitAmgPreconditioner,
    build_block_triangular_field_split,
)
from .block_inverse import AirReduction, BlockInverse, JacobiSmoothed, SimpleSmoothed
from .settings_mapping import SettingsMapping
from .settings_value import SettingsValue, filled_from
from .host_preconditioner import HostFactors, HostPreconditioner
from .hierarchy_inverse import HierarchyBlockInverse
from .refresh_timing import PhaseTimer, RefreshTiming
from .state import CellFields, FieldLayout, GlobalDofs, StateBlock, SubLayout
from .lu_preconditioner import MonolithicLuPreconditioner
from .strategy import (
    NewtonStrategy,
    ShiftedNewtonStrategy,
    StepControl,
    StepOutcome,
    StepReport,
)
from .root_adjoint import TransposedPreconditioner, root_adjoint, stop_array_gradients
from .implicit import (
    DampedNewtonStep,
    RootSolver,
    PositiveBlockLimit,
    PositiveBlockProjection,
    assembler_residual,
    positive_block_limit,
    positive_block_projection,
)
from .line_search_growth import (
    LineSearchGrowth,
    MonotoneLineSearch,
    RelaxedFarFromRoot,
)
from .linear import (
    default_linear_solver,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from .checkpoint import InnerIterateCheckpointer, StateCheckpointer
from .march import (
    CoefficientDriftTrigger,
    combine_observers,
    CycleGrowthTrigger,
    MarchResult,
    RefreshTrigger,
    ResidualHomotopy,
    newton_march,
    refuse_a_transform_the_march_cannot_run_in,
)
from .march_log import MarchLogger, combine_metrics, field_change_metrics
from .saddle_multigrid import (
    SimpleSmoothedInverse,
    block_approximate_inverse,
)
from .multigrid import (
    AirHierarchy,
    ShapeBudget,
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
from .refresh import NO_REFRESH, RefreshPolicy
from .retry import ESCALATING_REASONS, NO_RETRIES, RetryPolicy
from .shift_basis import LocalCourantBasis, ShiftBasis, VelocityShiftParts
from .sparse_jacobian import (
    BlockColouring,
    ColumnProbePlan,
    ProbeGather,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
    shifted_jacobian,
    jacobian_relative_error,
    materialize_block_jacobian,
)
from .step_control import (
    CflResidualDualTimeControl,
    default_dual_time_control,
    DualTimeControl,
    ResidualRatioDualTimeControl,
    ShiftStrengthControl,
)

__all__ = [
    "DEFAULT_GLOBALIZATION",
    "ESCALATING_REASONS",
    "NO_REFRESH",
    "NO_RETRIES",
    "AirHierarchy",
    "AirReduction",
    "AmgVCycle",
    "BlockColouring",
    "BlockInverse",
    "BlockScaledNorm",
    "BlockTriangularFieldSplit",
    "CellFields",
    "CflResidualDualTimeControl",
    "CoefficientDriftTrigger",
    "ColumnProbePlan",
    "ConstantRelaxation",
    "CycleGrowthTrigger",
    "DampedNewtonStep",
    "DivergenceGuard",
    "DualTimeControl",
    "DualTimeLoop",
    "DualTimeStep",
    "FieldGroups",
    "FieldLayout",
    "FieldSplitAmgPreconditioner",
    "GlobalDofs",
    "Globalization",
    "HierarchyBlockInverse",
    "HostFactors",
    "HostPreconditioner",
    "InnerIterateCheckpointer",
    "JacobiSmoothed",
    "JacobiSmoothedInverse",
    "LineSearchGrowth",
    "LocalCourantBasis",
    "MarchLogger",
    "MarchResult",
    "MaterializedJacobianPreconditioner",
    "MonolithicAmgPreconditioner",
    "MonolithicLuPreconditioner",
    "MonotoneLineSearch",
    "NewtonStrategy",
    "PhaseTimer",
    "PositiveBlockLimit",
    "PositiveBlockProjection",
    "ProbeGather",
    "PseudoTransientStep",
    "RefreshPolicy",
    "RefreshTiming",
    "RefreshTrigger",
    "RelaxationSchedule",
    "RelaxedFarFromRoot",
    "ResidualHomotopy",
    "ResidualNorm",
    "ResidualRatioDualTimeControl",
    "RetryPolicy",
    "RootSolver",
    "RowScaledNorm",
    "SettingsMapping",
    "SettingsValue",
    "ShapeBudget",
    "ShiftBasis",
    "ShiftPolicy",
    "ShiftStrengthControl",
    "ShiftTerm",
    "ShiftedNewtonStrategy",
    "SimpleSmoothed",
    "SimpleSmoothedInverse",
    "SmoothedHierarchy",
    "StateBlock",
    "StateCheckpointer",
    "StepAcceptance",
    "StepControl",
    "StepOutcome",
    "StepReport",
    "SubLayout",
    "SwitchedEvolutionRelaxation",
    "TransposedPreconditioner",
    "VelocityShiftParts",
    "air_multigrid_solve",
    "assembler_residual",
    "block_approximate_inverse",
    "block_stencil_colouring",
    "block_stencil_gather_map",
    "build_air_hierarchy",
    "build_amg_vcycle",
    "build_block_triangular_field_split",
    "build_convection_hierarchy",
    "build_smoothed_hierarchy",
    "cell_major_permutation",
    "column_probe_plan",
    "combine_metrics",
    "combine_observers",
    "convection_diffusion_operator",
    "convection_multigrid_solve",
    "decouple_dof",
    "default_dual_time_control",
    "default_linear_solver",
    "equilibrate_cell_major",
    "field_change_metrics",
    "filled_from",
    "jacobian_relative_error",
    "materialize_block_jacobian",
    "newton_march",
    "newton_step",
    "positive_block_limit",
    "positive_block_projection",
    "refresh_air_hierarchy",
    "refuse_a_transform_the_march_cannot_run_in",
    "relative_residual_gmres",
    "restart_cycles",
    "root_adjoint",
    "shifted_jacobian",
    "smoothed_multigrid_solve",
    "solve_linear",
    "stop_array_gradients",
    "symmetrically_equilibrate",
]
