"""The `aquaflux.solve` and `aquaflux.precondition` boundaries: each `__init__` is a real surface.

Each ``__init__`` re-exports what the rest of the library (and a user) may consume. That curation is
only meaningful if consumers actually go through it. That library modules do is pinned for every
package at once in ``test_layering.py``; this file pins what is particular to these two, which were one
package until the preconditioners moved out of ``solve``: each surface resolves and holds only what a
user needs, the multigrid toolkit is exported whole, and the study harnesses in ``validation/`` keep to
the surfaces too.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
from aquaflux import precondition, solve

PACKAGE_ROOT = pathlib.Path(solve.__file__).resolve().parent.parent
SOLVE_ROOT = PACKAGE_ROOT / "solve"

#: The packages this file guards, by the name ``aquaflux`` gives each.
PACKAGES = {"solve": solve, "precondition": precondition}


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_every_exported_name_resolves(name: str) -> None:
    """`__all__` is honest: every advertised name is actually present on the package."""
    package = PACKAGES[name]
    missing = [export for export in package.__all__ if not hasattr(package, export)]
    assert missing == [], f"__all__ advertises names {name} does not define: {missing}"


def test_no_name_is_exported_by_both() -> None:
    """A name has one home; exported from both packages, an import could reach either."""
    shared = sorted(set(solve.__all__) & set(precondition.__all__))
    assert shared == [], f"exported by both solve and precondition: {shared}"


#: Exports that no other package imports and no user documentation names, each kept because a user
#: writes against it. Anything else that nothing outside its own package uses is plumbing that leaked
#: onto the surface: it belongs in its submodule, where tests and study harnesses can still reach it.
USER_IMPLEMENTS = frozenset(
    {
        # Protocols and family bases a user writes a new member of.
        "AbortsInnerLoop",
        "BlockInverse",
        "CarriesRelaxationSchedule",
        "FrozenInverse",
        "HierarchyBlockInverse",
        # The base of a preconditioner applied through a host callback, for an inverse with no traced form.
        "HostPreconditioner",
        "LineSearchGrowth",
        "MaterializedJacobianPreconditioner",
        "MeasureBuilder",
        "NamedBlockMeasure",
        "OffersTracedCycle",
        "ReadableShift",
        "RefactorableInverse",
        "RefreshTrigger",
        "RelaxationSchedule",
        "ReleasableInverse",
        "ResidualMeasure",
        "ResidualNorm",
        "ScalarBorder",
        "ShiftCarryingControl",
        "ShiftedNewtonStrategy",
        "StateBlock",
        "StepAcceptance",
        "StopsForRefresh",
        # The value an `OffersTracedCycle` returns, built by the block inverse that implements it.
        "TracedCycle",
    }
)
USER_CONFIGURES = frozenset(
    {
        # Values a solve is configured with: a trigger, a shift rule or basis, an acceptance or
        # line-search policy, a strategy.
        "CoefficientDriftTrigger",
        "ConstantRelaxation",
        "DivergenceGuard",
        "DualTimeStep",
        "LocalCourantBasis",
        # The policy a step is built around when it is preconditioned by a host materialized inverse.
        "MonolithicFactorShiftPolicy",
        "MonotoneLineSearch",
        "RelaxedFarFromRoot",
        "SwitchedEvolutionRelaxation",
    }
)
USER_CALLS = frozenset(
    {
        # Functions a user calls, and what public drivers hand back.
        "MarchResult",
        "RampWalk",
        # A shift control's carried state, as `MarchResult.control_state` hands it back, and its field.
        "ReleaseMemory",
        "ShiftControlState",
        "StagedResult",
        "StepOutcome",
        "default_linear_solver",
        "field_change_metrics",
        "filled_from",
        "materialize_block_jacobian",
        "materialized_spec_from_mapping",
        "materialized_spec_to_mapping",
        "newton_march",
    }
)
USER_FACING = USER_IMPLEMENTS | USER_CONFIGURES | USER_CALLS


def _imported_by_other_packages(name: str) -> set[str]:
    """Every name a module outside the package ``name`` imports from ``aquaflux.<name>``.

    Relative imports count: ``precondition`` takes what it needs from ``solve`` as ``from ..solve``.
    """
    own = PACKAGE_ROOT / name
    names = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if own in path.parents:
            continue
        package = ["aquaflux", *path.parent.relative_to(PACKAGE_ROOT).parts]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = package[: len(package) - (node.level - 1)] if node.level else []
            target = [*base, *(node.module.split(".") if node.module else [])]
            if target == ["aquaflux", name]:
                names.update(alias.name for alias in node.names)
    return names


def _named_in_user_documentation(name: str) -> set[str]:
    """Every export of ``name`` the hand-written documentation names (the generated API page aside)."""
    root = PACKAGE_ROOT.parent
    pages = [root / "README.md", *(root / "docs").glob("*.md")]
    text = "\n".join(
        page.read_text()
        for page in pages
        if page.is_file() and page.name not in ("api.md", "package_structure.md")
    )
    exports = PACKAGES[name].__all__
    return {export for export in exports if re.search(rf"\b{re.escape(export)}\b", text)}


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_every_export_is_used_elsewhere_or_declared_user_facing(name: str) -> None:
    """An export nothing else uses must say why a user needs it, and a stated reason must still hold.

    A name reaches ``__all__`` because some other module wanted it; once nothing does, it stays on the
    surface, on the documentation site and in every future compatibility promise unless something asks.
    The declared lists are that question, in both directions: an unlisted unused export fails, and so
    does a listed one that another package now imports or the documentation now names.
    """
    exports = set(PACKAGES[name].__all__)
    justified = _imported_by_other_packages(name) | _named_in_user_documentation(name)
    unexplained = sorted(exports - justified - USER_FACING)
    assert unexplained == [], (
        f"these exports have no importer outside {name}/ and no user documentation; say in "
        f"USER_FACING why a user needs each, or leave it in its submodule: {unexplained}"
    )
    stale = sorted(export for export in USER_FACING & exports if export in justified)
    assert stale == [], f"USER_FACING lists names {name} exports that need no reason: {stale}"


def test_every_declared_user_facing_name_is_exported() -> None:
    """A declared reason for a name neither package exports any more is a stale entry."""
    exported = set(solve.__all__) | set(precondition.__all__)
    assert sorted(USER_FACING - exported) == []


def test_the_multigrid_surface_is_complete() -> None:
    """The frozen-AMG toolkit is exported as a whole — assemble, build, apply.

    The boundary previously exported only the smoothed-aggregation third of it, which is what pushed
    consumers into deep imports in the first place; a partial surface is what re-creates the problem.
    """
    required = {
        "ConvectionDiffusionStencil",
        "decouple_dof",
        "build_smoothed_hierarchy",
        "build_convection_hierarchy",
        "build_air_hierarchy",
        "smoothed_multigrid_cycles",
        "convection_multigrid_cycles",
        "air_multigrid_cycles",
        "SmoothedHierarchy",
        "AirHierarchy",
    }
    assert required <= set(precondition.__all__), (
        f"missing from the surface: {sorted(required - set(precondition.__all__))}"
    )


#: Study harnesses in ``validation/`` that legitimately reach past the package surface, and what for.
#: A short, explicit list rather than a blanket exemption: each entry is a name the package does *not*
#: export, so reaching for it is a considered decision, and listing it here is what makes that decision
#: reviewable instead of invisible. Adding a row is cheap and deliberate; the guard below is what stops
#: the list growing by accident.
VALIDATION_INTERNAL_REACHES = {
    # One SIMPLE level's plan, fit and pieces, built by `schur_storage_probe.py` on a lattice saddle to
    # price holding the Schur complement against applying it from its pieces; `in_precision` rounds a
    # formed Schur as a single-precision split rounds its pieces.
    "_simple_pieces",
    "fit_simple_level",
    "simple_plan",
    "in_precision",
    # The probe's entry-by-entry order, which `pitzdaily_openfoam/probe_loop_probe.py` places responses
    # by on the host inside its looped arm: the session's own structure holds its order at cell level,
    # so the harness builds the entry-by-entry one over the same layout.
    "_dof_probe_gather",
    # The march's own single-step call, reached by the harness that replays one outer step of a recorded
    # march from its checkpoint (`bfs3d_openfoam/step_replay.py`). Reproducing the march's inner rows
    # digit for digit is what makes a replayed arm evidence about the march, so it must run the march's
    # step, not a re-implementation of it.
    "_march_step",
    # The shifted Newton solve one inner iteration makes, re-run by `step_descent_probe.py` to take a
    # step's first direction loose, tight and clipped, and judge each by its descent in the step's measure.
    "_shifted_solve",
    # The aggregation's internals, reached by the harness that measured whether equilibration changes
    # the graph the coarsening sees (it does not -- 0.03% of edges). Running the real `_cell_graph` /
    # `_aggregation_edges` / `_mis_aggregate` is the point: a re-implementation would have measured a
    # different coarsener and proved nothing about this one.
    "_cell_graph",
    "_aggregation_edges",
    "_mis_aggregate",
    # The level operator's own product, timed by `csr_kernel_precision.py` on a synthetic matrix beside
    # the CSR and segment-sum forms it replaced: the comparison is only worth making with the real kernel.
    "_PaddedRowOperator",
    # The strength graph and the Vanek aggregation, reached by the local-descent probe to build a
    # "line-like" block smoother out of the coarsening the preconditioner already uses. A line through
    # an anisotropic near-wall layer is a chain of strong couplings, so the strength filter is the
    # algebraic form of that construction -- and running the real one is the point, since a
    # re-implementation would answer the question about a different grouping than the solver's.
    # (What it measured: these aggregates are isotropic blobs, median size 7, so they are NOT a line
    # substitute -- which is itself a finding about `_aggregate`, and one only the real one can give.)
    "_strength_classical",
    "_aggregate",
    # The step ladder itself, reached by the closure probe so that "which step length is admissible
    # here" is answered by the search a march actually walks rather than by a second one written
    # beside it. That question is the whole of that harness, and a re-implementation would answer it
    # about the wrong ladder -- which rung is kept depends on the fallback rule and the growth cap,
    # neither of which is obvious from the outside.
    "backtracking_line_search",
    # The coloured probe's own plan, gather and matvecs, and the shift and comparison applied to its
    # output. The bfs3d and pitzDaily probe studies measure what a reach or a batch size costs and how
    # accurate the materialized Jacobian is; running the materialize the solve runs is the point. None of
    # these is something a case configures or a user implements, which is why they are not exported.
    "PROBE_BATCH_SIZE",
    "ColumnProbePlan",
    "ProbeGather",
    # The probed blocks' in-place shift, wrapped by `compact_jacobian_screen.py` to drop a probe's far
    # couplings just before every fit. Hooking the place both fit paths hand their values to the split
    # is what makes the screened split the one a march would build from such a Jacobian.
    "ProbedBlocks",
    # A refit's blocks as the probe hands them to the split -- a host pattern with device values --
    # rebuilt from captured matrices by `pitzdaily_openfoam/refit_profile.py`, so the profiled refit
    # takes the path a march's does rather than reading host blocks back in.
    "DeviceMatrix",
    "SparsePattern",
    "as_scipy",
    "batched_field_jacobian_matvec",
    "batched_jacobian_matvec",
    # The one place a fit and a re-fit choose their products, reached by the harness that compares the
    # per-column-field probe with the whole-state one entry for entry (`fieldwise_probe_identity.py`).
    "_probe_products",
    "block_stencil_colouring",
    "block_stencil_gather_map",
    "column_probe_plan",
    "jacobian_matvec",
    "jacobian_relative_error",
    "frozen_shift_diagonal",
    "shifted_jacobian",
    # The equilibration and reordering the field split applies before it coarsens, reached by the
    # conditioning studies to look at the operator the hierarchy is actually built on.
    "equilibrate_cell_major",
    "symmetrically_equilibrate",
    # Instruments of a march a study attaches beside the case's own: the expensive-inner-solve
    # checkpointer, the metrics merger and the measure a configured solver stops in.
    "InnerIterateCheckpointer",
    "combine_metrics",
    "in_progress_measure",
    # The re-fit hook, whose re-materialize the bfs3d memory census (`bfs3d_openfoam/memory_by_stage.py`)
    # wraps to read the resident set around each re-fit of the shipped march without restating it.
    "BetaTrackingRefresh",
    # The tolerance a configured solver stops at, read and re-set by the pitzDaily replay
    # (`pitzdaily_openfoam/replay.py`) so each replayed system is solved to the tolerance the march's
    # adaptive rule gave its own step. A fixed tolerance read 1-2 cycles low on 27 of 50 solves, so
    # reproducing the march means setting the same field of the same solver, not building a second one.
    "relative_tolerance",
    "with_relative_tolerance",
    # The traced form of a fitted host split and the host-callback detector, reached by the two
    # traced-split harnesses: one times the traced split against the host one, the other checks a
    # refit keeps its array shapes. Both measure the object the session itself builds, so they take it
    # from the same function rather than assembling a lookalike.
    "contains_host_callback",
    "traced_field_split",
    # A step with its traced inverse swapped for a refitted one, reached by the secant-update probe
    # (`pitzdaily_openfoam/secant_update_probe.py`) to solve each arm's Jacobian under the same step the
    # march builds; a step assembled by hand would put the inverse in one of its two places but not both.
    "with_traced_form",
}


def _validation_modules() -> list[pathlib.Path]:
    """Every study harness under ``validation/``, or an empty list if it is not present."""
    root = PACKAGE_ROOT.parent / "validation"
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _deep_imported_names(source: str) -> list[tuple[str, str]]:
    """``(package, name)`` for each ``from aquaflux.<package>.<submodule> import name`` of a guarded one."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        parts = (node.module or "").split(".")
        if len(parts) > 2 and parts[0] == "aquaflux" and parts[1] in PACKAGES:
            found.extend((parts[1], alias.name) for alias in node.names)
    return found


def test_the_study_harnesses_take_exported_names_from_the_package_surface() -> None:
    """The boundary guard above scans ``aquaflux/`` only, and the boundary eroded where it does not look.

    ``validation/`` held 12 deep imports of names ``__all__` already advertises -- ``restart_cycles``,
    ``MonolithicVCyclePreconditioner``, ``symmetrically_equilibrate`` -- so the design record's claim that
    the boundary "cannot erode silently" was true only of the directory under guard. These harnesses are
    the project's re-adjudication instruments, so a rename inside a preconditioner breaks a study rather
    than a test, which is the more expensive failure and the later-discovered one.

    The rule enforced is the one that needs no API decision: **if the package exports it, import it from
    the package.** A harness reaching for something genuinely internal is a separate judgement, and each
    such reach is listed in :data:`VALIDATION_INTERNAL_REACHES` with its reason.
    """
    offenders = [
        f"{path.name}: {package}.{name}"
        for path in _validation_modules()
        for package, name in _deep_imported_names(path.read_text())
        if name in PACKAGES[package].__all__
    ]
    assert offenders == [], (
        "these names are exported, so the harness should import them from the package rather than "
        f"from a submodule: {offenders}"
    )


def test_the_harnesses_internal_reaches_stay_the_listed_ones() -> None:
    """A new reach past the surface must be a deliberate entry, not an accident.

    Asserted in both directions: an unlisted reach fails, and so does a listed one that no longer
    happens -- because a stale exemption is how a list like this stops describing the code.
    """
    reached = {
        name
        for path in _validation_modules()
        for package, name in _deep_imported_names(path.read_text())
        if name not in PACKAGES[package].__all__
    }
    if not _validation_modules():  # a checkout without the harnesses has nothing to say
        return
    assert reached == VALIDATION_INTERNAL_REACHES, (
        "the harnesses' internal reaches changed; add the new one to VALIDATION_INTERNAL_REACHES with "
        f"its reason, or drop the stale entry. unlisted: {sorted(reached - VALIDATION_INTERNAL_REACHES)}; "
        f"listed but gone: {sorted(VALIDATION_INTERNAL_REACHES - reached)}"
    )


def test_the_strategy_contract_module_stays_a_leaf() -> None:
    """`strategy.py` may not import a module that imports it back.

    It exists to break a cycle: `StepControl` was declared in `march.py` with no implementations there,
    so `step_control.py` had to import `march`, which forbade the reverse -- and a defaulting rule about
    two `solve/` objects therefore could not be written in `solve/` at all. An import here pointing at
    any of its dependents re-creates that cycle, and the symptom would show up somewhere else entirely
    (a rule stranded in another package), which is why this is asserted rather than left to review.
    """
    contract = SOLVE_ROOT / "strategy.py"
    dependents = {
        "implicit",
        "march",
        "continuation",
        "retry",
        "step_control",
        "march_log",
        "march_history",
        "checkpoint",
    }
    tree = ast.parse(contract.read_text())
    siblings = {
        node.module.lstrip(".")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module
    }
    assert siblings.isdisjoint(dependents), (
        "strategy.py imports a module that depends on it, re-creating the cycle it removes: "
        f"{sorted(siblings & dependents)}"
    )
