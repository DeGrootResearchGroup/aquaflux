"""The package's layering: what may import what, and how large a module may grow.

Two failures this pins, both of which happened. The staged solve driver, the preconditioner sessions, the
shifted-step tail and the residual measures were written for the k--omega SST solve, inside
``turbulence/coupled.py``, and none of them mentions turbulence. A laminar flow problem could not use
them, so it could not be marched by the machinery a turbulent one is, and a control experiment that
asked whether a solver behaviour needs turbulence had nothing to run. The file reached ~4000 lines
before anyone moved a line of it out, because no check looked at how large a physics module had become.
The generic packages had the same blind spot later: ``schemes/gradient.py`` and
``precondition/multigrid.py`` each passed 3000 lines with nothing measuring that they were growing.

* :func:`test_a_generic_package_imports_nothing_outside_itself` -- ``solve/`` is the residual-agnostic
  layer. It holds no mesh, field or physics import, which is what lets every residual run on it.
  ``precondition/`` sits directly on it and may import ``solve/`` and nothing else, so the march never
  depends on a concrete preconditioner. The same holds for ``solids/``: its bodies were written for the
  radiation model's shadows and lived inside it, and a solid body names no physics, so a computer-aided
  design (CAD) reader or any other consumer can use them only while they import nothing from a physics
  package. ``octree/`` holds the octree and samples a field over a region on an adaptive one; the
  sampler was written for the fluence rate and the tree for mesh generation, and it takes the field as
  a callable and the region as a solid, so neither consumer waits on the other's package.
* :func:`test_a_module_stays_below_the_size_at_which_misplaced_code_hides_in_it` -- a size ratchet on
  every module of the package, physics and generic alike, with each over-size module's current size as
  its budget.
* :func:`test_nothing_below_the_case_layer_imports_it` -- ``case/`` is the opposite end: it composes
  every physics package into a described case, so a package it composes that imported it back would
  make the description a dependency of the thing described.
* :func:`test_each_geometry_kernel_binding_has_one_importer` and its neighbours -- OCP and Netgen
  each bring their own build of one geometry kernel, and a process that loads both aborts at exit, so
  each is imported by one module only, and Netgen's only from a script that runs in a child process.
* :func:`test_one_package_reaches_another_only_through_its_public_surface` -- a package's ``__all__``
  is what other packages may use. Seven imports once reached past it, three of them of private names,
  so each could be renamed inside its own module with nothing in the importing package failing.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import sys

import pytest

from tests.support.tools import load_tool

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "aquaflux"

#: Dependency-free leaf modules of the package, which any layer may import.
NEUTRAL_LEAVES = (
    "morton",
    "ragged",
    "text_table",
    "triangle_grid",
    "triangle_grid_walk",
    "vectors",
    "warn_once",
)

#: The packages that hold a specific physical model rather than machinery. A module here that grows large
#: is the place residual-agnostic code hides, because it was written for the first consumer.
PHYSICS_PACKAGES = ("flow", "turbulence", "transport", "radiation")

#: The size past which a module must justify itself. Tripping it is a prompt to sort the module's
#: contents, not to split it by topic: in a physics package, ask which of them would be identical for
#: another residual and move those to ``solve/``; in a generic one, ask which of them do a different job
#: (host-side setup beside traced code, one strategy family beside another) and give each its own module.
LINE_LIMIT = 1500

#: Modules over the limit, each with the size it may not exceed. **A ratchet, not a licence**: a budget
#: is lowered as code moves out, never raised to make a change fit. Add an entry only with the reason.
BUDGETS = {
    # Holds the k--omega residual, its shift policy and damping, the k-positivity guards and the coupled
    # solve drivers. Everything residual-agnostic has moved out: the staged driver and step tail to
    # ``solve/``; the materialized-Jacobian session, the coloured probe and the monolithic shift policy
    # to ``precondition/``; the mass-flow border -- layout, seed, constraint vectors and bulk-velocity
    # average -- to ``flow/drive.py``, and its bordered residual, shift policy and measures to
    # ``flow/bordered.py`` (which also let the separate mass-flow solve and builder go); the row-scaled
    # measure with its equation names to ``turbulence/measures.py``.
    # What stays of the session and the probe are their RANS adapters (``_CoupledProblem``,
    # ``_CoupledNarrowing``) and the block-diagonal session, a preconditioner family of its own.
    "turbulence/coupled.py": 2886,
    # Every algebraic multigrid family -- smoothed aggregation, its two-level convection variant and
    # lAIR -- in one module: the host-side coarsening (orderings, aggregation, strength and splitting,
    # structural patterns, Galerkin products) beside the traced levels, smoothers and cycle drivers
    # applied under jit. That is two jobs, on either side of the build-once, apply-traced seam.
    "precondition/multigrid.py": 2785,
    # Several gradient-scheme families -- compact and corrected Green--Gauss, the Hessian-corrected
    # gradient with its Hessian solves and boundary closures -- beside the machinery that solves for a
    # gradient: the gradient and cell preconditioners, the swept and Krylov gradient solves, and the
    # sweep-count calibration. Lowered from 3637 when the nested Hessian solve was deleted.
    "schemes/gradient.py": 3444,
    # The body contract and every solid in one module: the interval constraints the convex primitives
    # are built from, the primitives, the torus with its quartic root-finding (cubic resolvent,
    # bracketed polish) beside them, and the constructive combinations and ``Outside``. The torus pushed
    # it over: its root-finding is numerics no other primitive uses.
    "solids/bodies.py": 1660,
}


#: The selector's import walk, so the layering and the scoped test tier read imports one way.
_scope = load_tool("scope_tests")


def _imports(path: pathlib.Path) -> list[tuple[int, str | None, int]]:
    """``(line, module, level)`` for every ``from ... import`` and ``import`` in ``path``."""
    return [(s.line, s.module, s.level) for s in _scope.imports(path)]


def _absolute(path: pathlib.Path, module: str | None, level: int) -> list[str]:
    """The dotted parts of the module an import in ``path`` names, relative imports resolved."""
    return _scope.absolute(["aquaflux", *path.parent.relative_to(PACKAGE).parts], module, level)


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


#: Packages that must import nothing of aquaflux but themselves, the neutral leaves and the generic
#: packages they are listed as built on, each with what it is. A generic package that names a physics
#: one has stopped being generic.
GENERIC_PACKAGES = {
    "solve": "the residual-agnostic layer, and a residual-agnostic solver that names one is not "
    "residual-agnostic",
    "precondition": "the residual-agnostic preconditioners the solve layer's linear solves take, and "
    "one that needs a physics package to be built is not residual-agnostic",
    "solids": "plain geometry, and a solid body that needs a physics package to be described is in "
    "the wrong package",
    "octree": "the octree the mesher and the sampler share, and adaptive sampling of any field over any "
    "region; a tree or a sampler that needs a physics package is not generic",
}

#: The generic packages each generic package is built on. ``solve`` is on nothing, so the march can be
#: driven with any preconditioner or none; ``precondition`` takes its state layout, settings values and
#: shift-policy contract from ``solve``.
BUILT_ON = {"precondition": ("solve",)}


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
            if target not in (package, *BUILT_ON.get(package, ()), *NEUTRAL_LEAVES):
                offenders.append(f"{path.name}:{line} imports {'.' * level}{module}")
    assert not offenders, (
        f"{package}/ is {GENERIC_PACKAGES[package]}, so it may import only "
        f"{sorted(BUILT_ON.get(package, ()))} of the other packages:\n" + "\n".join(offenders)
    )


def test_a_module_stays_below_the_size_at_which_misplaced_code_hides_in_it() -> None:
    over = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        lines = len(path.read_text().splitlines())
        if lines > LINE_LIMIT:
            over[path.relative_to(PACKAGE).as_posix()] = lines
    unbudgeted = {name: n for name, n in over.items() if name not in BUDGETS}
    physics = sorted(name for name in unbudgeted if name.split("/")[0] in PHYSICS_PACKAGES)
    generic = sorted(name for name in unbudgeted if name not in physics)
    assert not unbudgeted, (
        f"{sorted(unbudgeted)} exceed {LINE_LIMIT} lines. Before splitting them by topic, sort their "
        "contents."
        + (
            f" In {physics}, ask which would be identical for another residual (a driver, a step "
            "assembly, a measure, a settings value): those belong in solve/, where every residual "
            "reaches them."
            if physics
            else ""
        )
        + (
            f" In {generic}, ask which do a different job from the rest (host-side setup beside traced "
            "code, one strategy family beside another): each such job belongs in a module of its own."
            if generic
            else ""
        )
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
    for statement in _scope.imports(path):
        if statement.names is not None:
            targets = [(_absolute(path, statement.module, statement.level), list(statement.names))]
        else:
            targets = [(statement.module.split("."), [])]
        for target, names in targets:
            if target[0] != "aquaflux" or len(target) < 2 or target[1] == own_package:
                continue
            if not (PACKAGE / target[1]).is_dir():
                continue  # a neutral leaf module, which has no package surface to go through
            package = _package_of(target)
            where = f"{path.relative_to(PACKAGE)}:{statement.line}"
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


#: Libraries that may each be imported by exactly one module of the package, and which one. OCP and
#: Netgen each carry their own build of the OpenCASCADE geometry kernel, and a process that loads both
#: aborts at exit: so OCP stays behind the CAD reader's kernel module, and Netgen behind the script the
#: mesher runs in a child process, which nothing imports.
ONE_IMPORTER = {
    "OCP": "io/cad/kernel.py",
    "netgen": "meshing/_netgen_child.py",
    "pyngcore": "meshing/_netgen_child.py",
}

#: The script the mesher runs in a child interpreter, by path.
NETGEN_CHILD = ["aquaflux", "meshing", "_netgen_child"]

TESTS = PACKAGE.parent / "tests"


def _top_level_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """``(line, first dotted part)`` of every absolute import in ``path``."""
    return [
        (line, module.split(".")[0])
        for line, module, level in _imports(path)
        if level == 0 and module is not None
    ]


def test_each_geometry_kernel_binding_has_one_importer() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE)}:{line} imports {name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line, name in _top_level_imports(path)
        if name in ONE_IMPORTER and path.relative_to(PACKAGE).as_posix() != ONE_IMPORTER[name]
    ]
    assert not offenders, (
        "OCP and Netgen bring two builds of one geometry kernel, which abort a process that loads "
        f"both; each stays behind its one importer {ONE_IMPORTER}:\n" + "\n".join(offenders)
    )


def test_no_test_imports_netgen() -> None:
    # The test process holds OCP, so a test that imported Netgen would abort its worker at exit.
    offenders = [
        f"{path.relative_to(TESTS)}:{line} imports {name}"
        for path in sorted(TESTS.rglob("*.py"))
        for line, name in _top_level_imports(path)
        if name in ("netgen", "pyngcore")
    ]
    assert not offenders, "drive Netgen through aquaflux.meshing.NetgenProcess:\n" + "\n".join(
        offenders
    )


def _imported_modules(root: pathlib.Path, path: pathlib.Path) -> list[tuple[int, list[str]]]:
    """``(line, dotted parts)`` of every module an import in ``path`` could load.

    A ``from a.b import c`` may load the submodule ``a.b.c``, so it yields both ``a.b`` and ``a.b.c``.
    """
    found = []
    for statement in _scope.imports(path):
        if root == PACKAGE:
            module = _absolute(path, statement.module, statement.level)
        else:
            module = (statement.module or "").split(".")
        found.append((statement.line, module))
        found += [(statement.line, [*module, name]) for name in statement.names or ()]
    return found


def test_nothing_imports_the_netgen_child_script() -> None:
    offenders = [
        f"{path}:{line}"
        for root in (PACKAGE, TESTS)
        for path in sorted(root.rglob("*.py"))
        for line, module in _imported_modules(root, path)
        if module[:3] == NETGEN_CHILD
    ]
    assert not offenders, (
        "the Netgen child script runs only as its own process; importing it would load Netgen "
        "here:\n" + "\n".join(offenders)
    )


def test_the_netgen_child_imports_only_the_standard_library_numpy_and_netgen() -> None:
    # It runs by path in a fresh interpreter: an import of aquaflux would load the package (and
    # through it, possibly the CAD kernel) into the very process kept apart from it.
    allowed = {*sys.stdlib_module_names, "numpy", "netgen", "pyngcore"}
    child = PACKAGE / "meshing" / "_netgen_child.py"
    found = {name for _, name in _top_level_imports(child)}
    relative = [line for line, _, level in _imports(child) if level > 0]
    assert found, "the scan found no imports in the child script, so it would see nothing"
    assert not (found - allowed) and not relative, (
        f"the Netgen child imports {sorted(found - allowed)} (relative at lines {relative})"
    )
