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
