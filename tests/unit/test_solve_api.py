"""The `aquaflux.solve` package boundary: `__init__` is the real API surface.

`solve/__init__` re-exports what the rest of the library (and a user) may consume. That curation is
only meaningful if consumers actually go through it, so these tests pin both halves: the surface
resolves, and no library module reaches past it into a submodule. Without the second check the
boundary erodes silently — a deep import works fine at runtime, so nothing else would catch it.
"""

from __future__ import annotations

import ast
import pathlib

from aquaflux import solve

PACKAGE_ROOT = pathlib.Path(solve.__file__).resolve().parent.parent
SOLVE_ROOT = PACKAGE_ROOT / "solve"


def _library_modules() -> list[pathlib.Path]:
    """Every shipped module outside `solve/` itself (which imports its own siblings relatively)."""
    return [p for p in PACKAGE_ROOT.rglob("*.py") if SOLVE_ROOT not in p.parents]


def _absolute_imports(source: str) -> list[str]:
    """Every ``from <module> import ...`` target in ``source`` (absolute imports only)."""
    tree = ast.parse(source)
    return [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    ]


def test_every_exported_name_resolves() -> None:
    """`__all__` is honest: every advertised name is actually present on the package."""
    missing = [name for name in solve.__all__ if not hasattr(solve, name)]
    assert missing == [], f"__all__ advertises names the package does not define: {missing}"


def test_no_library_module_imports_past_the_boundary() -> None:
    """Library code imports from ``aquaflux.solve``, never from ``aquaflux.solve.<submodule>``.

    Deep-importing a submodule bypasses the curated surface, so `__init__` stops describing what the
    package offers — the state this guard exists to prevent (a consumer once pulled nine names
    straight out of ``solve.multigrid``). Tests are exempt: a unit test of a submodule's internals
    legitimately reaches for its private helpers.
    """
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT)}: {module}"
        for path in _library_modules()
        for module in _absolute_imports(path.read_text())
        if module.startswith("aquaflux.solve.")
    ]
    assert offenders == [], (
        "library modules must import from the aquaflux.solve package surface, not its submodules; "
        f"offenders: {offenders}"
    )


def test_the_multigrid_surface_is_complete() -> None:
    """The frozen-AMG toolkit is exported as a whole — assemble, build, apply.

    The boundary previously exported only the smoothed-aggregation third of it, which is what pushed
    consumers into deep imports in the first place; a partial surface is what re-creates the problem.
    """
    required = {
        "convection_diffusion_operator",
        "decouple_dof",
        "build_smoothed_hierarchy",
        "build_convection_hierarchy",
        "build_air_hierarchy",
        "smoothed_multigrid_solve",
        "convection_multigrid_solve",
        "air_multigrid_solve",
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
    # The fixed-pattern shift/equilibrate/reorder assembler. It holds no PETSc, no V-cycle and no jax,
    # so it is not really the AMG's -- but it has no better home yet, so it is not exported either.
    "ShiftedCellMajorOperator",
    # The march's lock-up predicate, replayed over archived march logs to check a candidate rule fires
    # on the runs that stalled and on nothing that recovered. Replaying a private predicate is the
    # whole point of that harness, so this one is unlikely ever to become public.
    "_limit_collapsing",
    # The square-root-diagonal scale alone, without the reorder `equilibrate_cell_major` pairs it with.
    "equilibration_scale",
    # The aggregation's internals, reached by the harness that measured whether equilibration changes
    # the graph the coarsening sees (it does not -- 0.03% of edges). Running the real `_cell_graph` /
    # `_aggregation_edges` / `_mis_aggregate` is the point: a re-implementation would have measured a
    # different coarsener and proved nothing about this one.
    "_cell_graph",
    "_aggregation_edges",
    "_mis_aggregate",
    # The level operator and the SIMPLE splitting, reached by the field-split probe to build candidate
    # block inverses out of the same pieces the shipped V-cycle uses.
    "_CsrOperator",
    "_diagonal_approximate_inverse",
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
    ``MonolithicAmgPreconditioner``, ``symmetrically_equilibrate`` -- so the design record's claim that
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
