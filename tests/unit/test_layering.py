"""The package's layering: what may import what, and how large a physics module may grow.

Two failures this pins, both of which happened. The staged solve driver, the preconditioner sessions, the
shifted-step tail and the residual measures were written for the k--omega SST solve, inside
``turbulence/coupled.py``, and none of them mentions turbulence. A laminar flow problem could not use
them, so it could not be marched by the machinery a turbulent one is, and a control experiment that
asked whether a solver behaviour needs turbulence had nothing to run. The file reached ~4000 lines
before anyone moved a line of it out, because no check looked at how large a physics module had become.

* :func:`test_a_generic_package_imports_nothing_outside_itself` -- ``solve/`` is the residual-agnostic
  layer. It holds no mesh, field or physics import, which is what lets every residual run on it. The
  same holds for ``solids/``: its bodies were written for the radiation model's shadows and lived
  inside it, and a solid body names no physics, so a computer-aided design (CAD) reader or any other
  consumer can use them only while they import nothing from a physics package.
* :func:`test_a_physics_module_stays_below_the_size_at_which_generic_machinery_hides_in_it` -- a size
  ratchet on the physics packages, with each over-size module's current size as its budget.
* :func:`test_nothing_below_the_case_layer_imports_it` -- ``case/`` is the opposite end: it composes
  every physics package into a described case, so a package it composes that imported it back would
  make the description a dependency of the thing described.
* :func:`test_one_package_reaches_another_only_through_its_public_surface` -- a package's ``__all__``
  is what other packages may use. Seven imports once reached past it, three of them of private names,
  so each could be renamed inside its own module with nothing in the importing package failing.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "aquaflux"

#: Dependency-free leaf modules of the package, which any layer may import.
NEUTRAL_LEAVES = ("morton", "ragged", "text_table", "vectors")

#: The packages that hold a specific physical model rather than machinery. A module here that grows large
#: is the place residual-agnostic code hides, because it was written for the first consumer.
PHYSICS_PACKAGES = ("flow", "turbulence", "transport", "radiation")

#: The size past which a physics module must justify itself. Tripping it is a prompt to ask which of the
#: module's contents would be identical for another residual, and to move those to ``solve/``.
LINE_LIMIT = 1500

#: Modules over the limit, each with the size it may not exceed. **A ratchet, not a licence**: a budget
#: is lowered as code moves out, never raised to make a change fit. Add an entry only with the reason.
BUDGETS = {
    # Holds the k--omega residual and the coupled march's preconditioner sessions and probe. The
    # residual-agnostic driver and step tail have moved to ``solve/``, and the mass-flow border --
    # layout, seed, constraint vectors and bulk-velocity average -- to ``flow/drive.py``, and the
    # row-scaled measure with its equation names to ``turbulence/measures.py``; the sessions and the
    # coloured probe are still to move (they take a ``CoupledRANS`` only for its residual and layout).
    "turbulence/coupled.py": 3196,
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


def _absolute(path: pathlib.Path, module: str | None, level: int) -> list[str]:
    """The dotted parts of the module an import in ``path`` names, relative imports resolved."""
    if level == 0:
        return (module or "").split(".")
    package = ["aquaflux", *path.parent.relative_to(PACKAGE).parts]
    return [*package[: len(package) - (level - 1)], *(module.split(".") if module else [])]


#: What may import the case layer: itself, and the command-line entry point, which runs a case file and so
#: sits above everything. Nothing else is above it.
ABOVE_THE_CASE_LAYER = frozenset({"case", "__main__.py"})


def test_nothing_below_the_case_layer_imports_it() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE)}:{line} imports {'.' * level}{module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).parts[0] not in ABOVE_THE_CASE_LAYER
        for line, module, level in _imports(path)
        if _absolute(path, module, level)[:2] == ["aquaflux", "case"]
    ]
    assert not offenders, (
        "case/ is the top layer: it reads a case file into the physics packages' own values, so none "
        "of them may import it:\n" + "\n".join(offenders)
    )


#: Packages that must import nothing of aquaflux but themselves and the neutral leaves, each with
#: what it is. A generic package that names a physics one has stopped being generic.
GENERIC_PACKAGES = {
    "solve": "the residual-agnostic layer, and a residual-agnostic solver that names one is not "
    "residual-agnostic",
    "solids": "plain geometry, and a solid body that needs a physics package to be described is in "
    "the wrong package",
}


@pytest.mark.parametrize("package", sorted(GENERIC_PACKAGES))
def test_a_generic_package_imports_nothing_outside_itself(package: str) -> None:
    offenders = []
    paths = sorted((PACKAGE / package).glob("*.py"))
    assert paths, f"{package}/ holds no modules, so this check would see nothing"
    for path in paths:
        for line, module, level in _imports(path):
            if level >= 2:
                target = (module or "").split(".")[0]
            elif level == 0 and module is not None and module.split(".")[0] == "aquaflux":
                target = [*module.split("."), ""][1]
            else:
                continue
            if target not in (package, *NEUTRAL_LEAVES):
                offenders.append(f"{path.name}:{line} imports {'.' * level}{module}")
    assert not offenders, (
        f"{package}/ is {GENERIC_PACKAGES[package]}, so it must not import a physics or "
        "discretization package:\n" + "\n".join(offenders)
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


def _package_of(parts: list[str]) -> list[str]:
    """The deepest package along the dotted ``parts``: the module path with any submodule dropped."""
    depth = max(
        n
        for n in range(1, len(parts) + 1)
        if (PACKAGE.parent.joinpath(*parts[:n]) / "__init__.py").exists()
    )
    return parts[:depth]


@functools.cache
def _exports(package: tuple[str, ...]) -> frozenset[str]:
    """The names ``package``'s ``__init__`` lists in ``__all__``, read without importing it."""
    tree = ast.parse((PACKAGE.parent.joinpath(*package) / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


def _surface_breaches(path: pathlib.Path) -> list[str]:
    """Each import in ``path`` that reaches another package past its ``__init__`` or its ``__all__``."""
    own = path.relative_to(PACKAGE).parts
    own_package = own[0] if len(own) > 1 else None
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            targets = [(_absolute(path, node.module, node.level), [a.name for a in node.names])]
        elif isinstance(node, ast.Import):
            targets = [(alias.name.split("."), []) for alias in node.names]
        else:
            continue
        for target, names in targets:
            if target[0] != "aquaflux" or len(target) < 2 or target[1] == own_package:
                continue
            if not (PACKAGE / target[1]).is_dir():
                continue  # a neutral leaf module, which has no package surface to go through
            package = _package_of(target)
            where = f"{path.relative_to(PACKAGE)}:{node.lineno}"
            if len(package) < len(target):
                found.append(f"{where} imports from the submodule {'.'.join(target)}")
                continue
            hidden = sorted(set(names) - _exports(tuple(package)))
            if hidden:
                found.append(f"{where} imports {hidden}, absent from {'.'.join(package)}.__all__")
    return found


def test_one_package_reaches_another_only_through_its_public_surface() -> None:
    offenders = [
        breach for path in sorted(PACKAGE.rglob("*.py")) for breach in _surface_breaches(path)
    ]
    assert not offenders, (
        "a package is used through its __init__, and only for what its __all__ lists. Export the name "
        "(and import it from the package), or move it to the package that needs it; a submodule or a "
        "name outside __all__ can change with nothing in the importing package failing:\n"
        + "\n".join(offenders)
    )
