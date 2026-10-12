"""Newton on the residual + implicitly-differentiated linear solve, and the march that drives it.

Drives `R(state, params) = 0` and exposes an exact adjoint via two-level implicit
differentiation (IFT on the converged state + `custom_vjp`/adjoint on each linear
solve) — no iteration is unrolled onto the tape. The residual and linear-solve
functions are injected, so the driver is testable on a trivial analytic residual.

**This module is the package's API boundary: everything the rest of the library (or a user) may
consume from `solve` is re-exported here, and consumers import from `aquaflux.solve`, not from its
submodules.** A name absent from `__all__` is internal — reach for it only from that submodule's own
unit tests. The surface is four groups:

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

The preconditioners those linear solves take live in :mod:`aquaflux.precondition`, which builds on
this package; nothing here imports it.
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
from .settings_mapping import SettingsMapping
from .settings_value import MergeableSettings, SettingsValue, filled_from
from .refresh_timing import (
    RefreshTiming,
)
from .state import CellFields, FieldLayout, GlobalDofs, StateBlock, SubLayout
from .strategy import (
    AbortsInnerLoop,
    CarriesRelaxationSchedule,
    NewtonStrategy,
    ReadableShift,
    ShiftCarryingControl,
    ShiftedNewtonStrategy,
    StepControl,
    StepOutcome,
    StepReport,
    shift_of,
)
from .root_adjoint import TransposedPreconditioner, root_adjoint, stop_array_gradients
from .implicit import (
    DEFAULT_ROOT_SOLVE,
    DampedNewtonStep,
    RootSolveSettings,
    RootSolver,
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
    residual_stop_gmres,
    restart_cycles,
    solve_linear,
)
from .checkpoint import (
    StateCheckpointer,
    find_checkpoint,
    report_record,
)
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
from .march_history import MarchRecorder, StepHistory
from .march_log import (
    MarchLogger,
    field_change_metrics,
)
from .newton import newton_step
from .norm import (
    BlockScaledNorm,
    ResidualNorm,
    NamedBlockMeasure,
    RowScaledNorm,
    block_reference_scales,
)
from .convergence import (
    BlockScaled,
    Convergence,
    Euclidean,
    MeasureBuilder,
    ResidualMeasure,
    ResidualMeasures,
    RowScaled,
)
from .relaxation import ConstantRelaxation, RelaxationSchedule, SwitchedEvolutionRelaxation
from .refresh import NO_REFRESH, RefreshPolicy
from .resumption import Resumption
from .driver import (
    ContinuationSource,
    SessionSource,
    StagedResult,
    explicit_source,
    staged_march,
)
from .session import PreconditionerSession
from .shifted_step import LinearSolveRegime, LinearSolveSettings, resolve_linear_solve, shifted_step
from .linear_solver_spec import DirectSolve, GmresSolve, LinearSolverSpec
from .ramp_schedule import RampPosition, RampSchedule, RampWalk
from .retry import NO_RETRIES, RetryPolicy
from .shift_basis import (
    DEFAULT_SHIFT_BASIS,
    LocalCourantBasis,
    ShiftBasis,
    ShiftSettings,
    VelocityShiftParts,
)
from .step_control import (
    CflResidualDualTimeControl,
    DualTimeControl,
    ReleaseMemory,
    ResidualRatioDualTimeControl,
    ShiftControlState,
    ShiftStrengthControl,
)

__all__ = [
    "DEFAULT_GLOBALIZATION",
    "DEFAULT_ROOT_SOLVE",
    "DEFAULT_SHIFT_BASIS",
    "NO_REFRESH",
    "NO_RETRIES",
    "AbortsInnerLoop",
    "BlockScaled",
    "BlockScaledNorm",
    "CarriesRelaxationSchedule",
    "CellFields",
    "CflResidualDualTimeControl",
    "CoefficientDriftTrigger",
    "ConstantRelaxation",
    "ContinuationSource",
    "Convergence",
    "CycleGrowthTrigger",
    "DampedNewtonStep",
    "DirectSolve",
    "DivergenceGuard",
    "DualTimeControl",
    "DualTimeLoop",
    "DualTimeStep",
    "Euclidean",
    "FieldLayout",
    "GlobalDofs",
    "Globalization",
    "GmresSolve",
    "LineSearchGrowth",
    "LinearSolveRegime",
    "LinearSolveSettings",
    "LinearSolverSpec",
    "LocalCourantBasis",
    "MarchLogger",
    "MarchRecorder",
    "MarchResult",
    "MeasureBuilder",
    "MergeableSettings",
    "MonotoneLineSearch",
    "NamedBlockMeasure",
    "NewtonStrategy",
    "PreconditionerSession",
    "PseudoTransientStep",
    "RampPosition",
    "RampSchedule",
    "RampWalk",
    "ReadableShift",
    "RefreshPolicy",
    "RefreshTiming",
    "RefreshTrigger",
    "RelaxationSchedule",
    "RelaxedFarFromRoot",
    "ReleaseMemory",
    "ResidualHomotopy",
    "ResidualMeasure",
    "ResidualMeasures",
    "ResidualNorm",
    "ResidualRatioDualTimeControl",
    "Resumption",
    "RetryPolicy",
    "RootSolveSettings",
    "RootSolver",
    "RowScaled",
    "RowScaledNorm",
    "SessionSource",
    "SettingsMapping",
    "SettingsValue",
    "ShiftBasis",
    "ShiftCarryingControl",
    "ShiftControlState",
    "ShiftPolicy",
    "ShiftSettings",
    "ShiftStrengthControl",
    "ShiftTerm",
    "ShiftedNewtonStrategy",
    "StagedResult",
    "StateBlock",
    "StateCheckpointer",
    "StepAcceptance",
    "StepControl",
    "StepHistory",
    "StepOutcome",
    "StepReport",
    "SubLayout",
    "SwitchedEvolutionRelaxation",
    "TransposedPreconditioner",
    "VelocityShiftParts",
    "assembler_residual",
    "block_reference_scales",
    "combine_observers",
    "default_linear_solver",
    "explicit_source",
    "field_change_metrics",
    "filled_from",
    "find_checkpoint",
    "newton_march",
    "newton_step",
    "positive_block_limit",
    "positive_block_projection",
    "refuse_a_transform_the_march_cannot_run_in",
    "relative_residual_gmres",
    "report_record",
    "residual_stop_gmres",
    "resolve_linear_solve",
    "restart_cycles",
    "root_adjoint",
    "shift_of",
    "shifted_step",
    "solve_linear",
    "staged_march",
    "stop_array_gradients",
]
