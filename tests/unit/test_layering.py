"""The package's layering: what may import what, and how large a physics module may grow.

Two failures this pins, both of which happened. The staged solve driver, the preconditioner sessions, the
shifted-step tail and the residual measures were written for the k--omega SST solve, inside
``turbulence/coupled.py``, and none of them mentions turbulence. A laminar flow problem could not use
them, so it could not be marched by the machinery a turbulent one is, and a control experiment that
asked whether a solver behaviour needs turbulence had nothing to run. The file reached ~4000 lines
before anyone moved a line of it out, because no check looked at how large a physics module had become.

* :func:`test_the_solve_package_imports_nothing_outside_itself` -- ``solve/`` is the residual-agnostic
  layer. It holds no mesh, field or physics import, which is what lets every residual run on it.
* :func:`test_a_physics_module_stays_below_the_size_at_which_generic_machinery_hides_in_it` -- a size
  ratchet on the physics packages, with each over-size module's current size as its budget.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "aquaflux"

#: Dependency-free leaf modules of the package, which any layer may import.
NEUTRAL_LEAVES = ("text_table", "vectors")

#: The packages that hold a specific physical model rather than machinery. A module here that grows large
#: is the place residual-agnostic code hides, because it was written for the first consumer.
PHYSICS_PACKAGES = ("flow", "turbulence", "transport", "radiation")

#: The size past which a physics module must justify itself. Tripping it is a prompt to ask which of the
#: module's contents would be identical for another residual, and to move those to ``solve/``.
LINE_LIMIT = 1500

#: Modules over the limit, each with the size it may not exceed. **A ratchet, not a licence**: a budget
#: is lowered as code moves out, never raised to make a change fit. Add an entry only with the reason.
BUDGETS = {
    # Holds the k--omega residual and the coupled march's preconditioner sessions, probe and mass-flow
    # borders. The residual-agnostic driver and step tail have moved to ``solve/``; the sessions and the
    # coloured probe are still to move (they take a ``CoupledRANS`` only for its residual and layout).
    "turbulence/coupled.py": 3295,
}


def _imports(path: pathlib.Path) -> list[tuple[int, str | None, int]]:
    """``(line, module, level)`` for every ``from ... import`` and ``import`` in ``path``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            found.append((node.lineno, node.module, node.level))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name, 0) for alias in node.names)
    return found


def test_the_solve_package_imports_nothing_outside_itself() -> None:
    offenders = []
    for path in sorted((PACKAGE / "solve").glob("*.py")):
        for line, module, level in _imports(path):
            if level >= 2:
                target = (module or "").split(".")[0]
            elif level == 0 and module is not None and module.split(".")[0] == "aquaflux":
                target = [*module.split("."), ""][1]
            else:
                continue
            leaves_solve = target not in ("solve", *NEUTRAL_LEAVES)
            if leaves_solve:
                offenders.append(f"{path.name}:{line} imports {'.' * level}{module}")
    assert not offenders, (
        "solve/ is the residual-agnostic layer and must not import a physics or discretization "
        "package -- a residual-agnostic solver that names one is not residual-agnostic:\n"
        + "\n".join(offenders)
    )


def test_a_physics_module_stays_below_the_size_at_which_generic_machinery_hides_in_it() -> None:
    over = {}
    for package in PHYSICS_PACKAGES:
        for path in sorted((PACKAGE / package).glob("*.py")):
            lines = len(path.read_text().splitlines())
            if lines > LINE_LIMIT:
                over[f"{package}/{path.name}"] = lines
    unbudgeted = {name: n for name, n in over.items() if name not in BUDGETS}
    assert not unbudgeted, (
        f"{sorted(unbudgeted)} exceed {LINE_LIMIT} lines. Before splitting them by topic, ask which of "
        "their contents would be identical for another residual (a driver, a step assembly, a measure, a "
        "settings value): those belong in solve/, where every residual reaches them."
    )
    grown = {name: (n, BUDGETS[name]) for name, n in over.items() if n > BUDGETS[name]}
    assert not grown, f"{grown} grew past their budget; move code out instead of raising it."
    stale = sorted(name for name in BUDGETS if name not in over)
    assert not stale, f"{stale} are back under {LINE_LIMIT} lines; delete their budget."
