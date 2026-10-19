"""Preconditioners for the linear solves inside a Newton step, built off the jit path and held frozen.

A preconditioner only changes how quickly a Krylov solve converges, never the solution or its
derivative, so everything here may approximate: it is assembled from concrete arrays outside any traced
program, frozen, and applied as a matrix-vector product the solve in :mod:`aquaflux.solve` takes through
its ``preconditioner`` seam. This package depends on :mod:`aquaflux.solve` for the state layout, the
settings values and the shift-policy contract; :mod:`aquaflux.solve` depends on nothing here, so a
Newton march can be driven with any preconditioner, or none.

**This module is the package's API boundary: other packages import from ``aquaflux.precondition``, not
from its submodules.** A name absent from ``__all__`` is internal. The surface is five groups:

* **Frozen operators and algebraic multigrid** -- the operator description
  ``ConvectionDiffusionStencil``, which assembles itself and reports its diagonal (plus ``decouple_dof``
  for a closed-domain pressure pin), the hierarchy builders ``build_smoothed_hierarchy`` /
  ``build_convection_hierarchy`` / ``build_air_hierarchy`` and their matching fixed-cycle applies.
  Callers assemble an operator, build a hierarchy once, and apply it as a frozen V-cycle.
* **The frozen-inverse contract** -- ``FrozenInverse`` (a size and an ``apply`` with an optional
  transpose) and the two capabilities only some inverses have, ``RefactorableInverse`` and
  ``ReleasableInverse``; ``HostPreconditioner`` turns one into the matrix-vector product a jitted
  Krylov solve calls back to.
* **Block inverses and the field split** -- the settings values ``SimpleSmoothed``, ``JacobiSmoothed``
  and ``AirReduction`` (all ``BlockInverse``), the inverses they build, and ``FieldSplitInverse``, which
  inverts a state split into a leading saddle and trailing transported fields block-triangularly.
* **The materialized Jacobian** -- ``JacobianProbe`` and ``materialize_block_jacobian`` recover the
  assembled Jacobian by graph-coloured probing; ``MaterializedJacobian`` (with ``FieldSplit``) says how
  to precondition with it, and ``MaterializedSession`` keeps that preconditioner current across a march
  for any residual that supplies a ``MaterializedProblem``. ``MonolithicFactorShiftPolicy`` pairs a
  march's shift with such a frozen inverse.
* **Bordered systems** -- ``bordered_preconditioner`` extends any preconditioner to the same system
  bordered by one scalar unknown (a Lagrange multiplier), eliminating it through its 1x1 Schur
  complement; where the scalar sits is a ``ScalarBorder``'s to say.
"""

from __future__ import annotations

from .block_inverse import AirReduction, BlockInverse, JacobiSmoothed, SimpleSmoothed
from .bordered import ScalarBorder, bordered_preconditioner
from .field_split import (
    FieldGroups,
    FieldSplitInverse,
    FieldSplitPreconditioner,
    JacobiSmoothedInverse,
    field_split_inverse,
)
from .frozen_operator import ConvectionDiffusionStencil, decouple_dof
from .hierarchy_inverse import HierarchyBlockInverse
from .host_preconditioner import (
    FrozenInverse,
    HostPreconditioner,
    RefactorableInverse,
    ReleasableInverse,
)
from .jacobian_probe import JacobianProbe, jacobian_probe_plan
from .materialized_preconditioner import MaterializedJacobianPreconditioner
from .materialized_session import DiagonalStandIn, MaterializedProblem, MaterializedSession
from .materialized_spec import (
    MATERIALIZED_MAPPING,
    FieldSplit,
    JacobianProbeSpec,
    MaterializedJacobian,
    materialized_spec_from_mapping,
    materialized_spec_to_mapping,
)
from .monolithic_policy import MonolithicFactorShiftPolicy
from .pivot_floor import PivotFloor
from .multigrid import (
    AirHierarchy,
    SmoothedHierarchy,
    air_multigrid_cycles,
    build_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_cycles,
    refresh_air_hierarchy,
    smoothed_multigrid_cycles,
)
from .saddle_multigrid import SimpleSmoothedInverse
from .sparse_jacobian import materialize_block_jacobian
from .traced_cycle import OffersTracedCycle, TracedCycle

__all__ = [
    "MATERIALIZED_MAPPING",
    "AirHierarchy",
    "AirReduction",
    "BlockInverse",
    "ConvectionDiffusionStencil",
    "DiagonalStandIn",
    "FieldGroups",
    "FieldSplit",
    "FieldSplitInverse",
    "FieldSplitPreconditioner",
    "FrozenInverse",
    "HierarchyBlockInverse",
    "HostPreconditioner",
    "JacobiSmoothed",
    "JacobiSmoothedInverse",
    "JacobianProbe",
    "JacobianProbeSpec",
    "MaterializedJacobian",
    "MaterializedJacobianPreconditioner",
    "MaterializedProblem",
    "MaterializedSession",
    "MonolithicFactorShiftPolicy",
    "OffersTracedCycle",
    "PivotFloor",
    "RefactorableInverse",
    "ReleasableInverse",
    "ScalarBorder",
    "SimpleSmoothed",
    "SimpleSmoothedInverse",
    "SmoothedHierarchy",
    "TracedCycle",
    "air_multigrid_cycles",
    "bordered_preconditioner",
    "build_air_hierarchy",
    "build_convection_hierarchy",
    "build_smoothed_hierarchy",
    "convection_multigrid_cycles",
    "decouple_dof",
    "field_split_inverse",
    "jacobian_probe_plan",
    "materialize_block_jacobian",
    "materialized_spec_from_mapping",
    "materialized_spec_to_mapping",
    "refresh_air_hierarchy",
    "smoothed_multigrid_cycles",
]
