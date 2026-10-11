"""The `aquaflux.solve` package boundary: `__init__` is the real API surface.

`solve/__init__` re-exports what the rest of the library (and a user) may consume. That curation is
only meaningful if consumers actually go through it. That library modules do is pinned for every
package at once in ``test_layering.py``; this file pins what is particular to ``solve``: the surface
resolves, it holds only what a user needs, the multigrid toolkit is exported whole, and the study
harnesses in ``validation/`` keep to the surface too.
"""

from __future__ import annotations

import ast
import pathlib
import re

from aquaflux import solve

PACKAGE_ROOT = pathlib.Path(solve.__file__).resolve().parent.parent
SOLVE_ROOT = PACKAGE_ROOT / "solve"


def test_every_exported_name_resolves() -> None:
    """`__all__` is honest: every advertised name is actually present on the package."""
    missing = [name for name in solve.__all__ if not hasattr(solve, name)]
    assert missing == [], f"__all__ advertises names the package does not define: {missing}"


#: Exports that no other package imports and no user documentation names, each kept because a user
#: writes against it. Anything else that nothing outside ``solve`` uses is plumbing that leaked onto the
#: surface: it belongs in its submodule, where tests and study harnesses can still reach it.
USER_IMPLEMENTS = frozenset(
    {
        # Protocols and family bases a user writes a new member of.
        "AbortsInnerLoop",
        "BlockInverse",
        "CarriesRelaxationSchedule",
        "FrozenInverse",
        "HierarchyBlockInverse",
        "LineSearchGrowth",
        "MaterializedJacobianPreconditioner",
        "MeasureBuilder",
        "NamedBlockMeasure",
        "ReadableShift",
        "RefactorableInverse",
        "RefreshTrigger",
        "RelaxationSchedule",
        "ReleasableInverse",
        "ResidualMeasure",
        "ResidualMeasures",
        "ResidualNorm",
        "ScalarBorder",
        "ShiftCarryingControl",
        "ShiftedNewtonStrategy",
        "StateBlock",
        "StepAcceptance",
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
        "MonotoneLineSearch",
        "RelaxedFarFromRoot",
        "SwitchedEvolutionRelaxation",
    }
)
USER_CALLS = frozenset(
    {
        # Functions a user calls, and what public drivers hand back.
        "MarchResult",
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


def _imported_by_other_packages() -> set[str]:
    """Every name a module outside ``solve/`` imports from ``aquaflux.solve``."""
    names = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if SOLVE_ROOT in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "aquaflux.solve":
                names.update(alias.name for alias in node.names)
    return names


def _named_in_user_documentation() -> set[str]:
    """Every export the hand-written documentation names (the generated API page aside)."""
    root = PACKAGE_ROOT.parent
    pages = [root / "README.md", *(root / "docs").glob("*.md")]
    text = "\n".join(
        page.read_text()
        for page in pages
        if page.is_file() and page.name not in ("api.md", "package_structure.md")
    )
    return {name for name in solve.__all__ if re.search(rf"\b{re.escape(name)}\b", text)}


def test_every_export_is_used_elsewhere_or_declared_user_facing() -> None:
    """An export nothing else uses must say why a user needs it, and a stated reason must still hold.

    A name reaches ``__all__`` because some other module wanted it; once nothing does, it stays on the
    surface, on the documentation site and in every future compatibility promise unless something asks.
    The declared lists are that question, in both directions: an unlisted unused export fails, and so
    does a listed one that another package now imports or the documentation now names.
    """
    justified = _imported_by_other_packages() | _named_in_user_documentation()
    unexplained = sorted(set(solve.__all__) - justified - USER_FACING)
    assert unexplained == [], (
        "these exports have no importer outside solve/ and no user documentation; say in "
        f"USER_FACING why a user needs each, or leave it in its submodule: {unexplained}"
    )
    stale = sorted(name for name in USER_FACING if name not in solve.__all__ or name in justified)
    assert stale == [], f"USER_FACING lists names that are not exported or need no reason: {stale}"


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
    assert required <= set(solve.__all__), (
        f"missing from the surface: {sorted(required - set(solve.__all__))}"
    )


#: Study harnesses in ``validation/`` that legitimately reach past the package surface, and what for.
#: A short, explicit list rather than a blanket exemption: each entry is a name the package does *not*
#: export, so reaching for it is a considered decision, and listing it here is what makes that decision
#: reviewable instead of invisible. Adding a row is cheap and deliberate; the guard below is what stops
#: the list growing by accident.
VALIDATION_INTERNAL_REACHES = {
    # The march's lock-up predicate, replayed over archived march logs to check a candidate rule fires
    # on the runs that stalled and on nothing that recovered. Replaying a private predicate is the
    # whole point of that harness, so this one is unlikely ever to become public.
    "_limit_collapsing",
    # The aggregation's internals, reached by the harness that measured whether equilibration changes
    # the graph the coarsening sees (it does not -- 0.03% of edges). Running the real `_cell_graph` /
    # `_aggregation_edges` / `_mis_aggregate` is the point: a re-implementation would have measured a
    # different coarsener and proved nothing about this one.
    "_cell_graph",
    "_aggregation_edges",
    "_mis_aggregate",
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
    "batched_jacobian_matvec",
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
    # The shift a step will run at, read by the aggressive-continuation harness to log the shift each
    # rung starts from; the step control's carried state is not the bare shift, so it asks the step.
    "shift_of",
}


def _validation_modules() -> list[pathlib.Path]:
    """Every study harness under ``validation/``, or an empty list if it is not present."""
    root = PACKAGE_ROOT.parent / "validation"
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _deep_imported_names(source: str) -> list[str]:
    """Every name imported via ``from aquaflux.solve.<submodule> import ...``."""
    tree = ast.parse(source)
    return [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and (node.module or "").startswith("aquaflux.solve.")
        for alias in node.names
    ]


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
        f"{path.name}: {name}"
        for path in _validation_modules()
        for name in _deep_imported_names(path.read_text())
        if name in solve.__all__
    ]
    assert offenders == [], (
        "these names are exported, so the harness should import them from `aquaflux.solve` rather "
        f"than from a submodule: {offenders}"
    )


def test_the_harnesses_internal_reaches_stay_the_listed_ones() -> None:
    """A new reach past the surface must be a deliberate entry, not an accident.

    Asserted in both directions: an unlisted reach fails, and so does a listed one that no longer
    happens -- because a stale exemption is how a list like this stops describing the code.
    """
    reached = {
        name
        for path in _validation_modules()
        for name in _deep_imported_names(path.read_text())
        if name not in solve.__all__
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
