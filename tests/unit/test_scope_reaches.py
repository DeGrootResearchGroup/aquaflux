"""Every test that reaches a file without importing it is declared to the scoped tier, or this fails.

``tools/fastgate.sh --changed`` runs the tests whose imports reach what changed. A test that reads the
package source as text, runs a script, loads a case file or starts a subprocess reaches files no import
walk can see, so the selector would leave it out of a change it covers -- and a run that leaves a test
out is green in exactly the way a run that passed it is. The selector's ``REACHES`` table names each
such test with the paths it reaches. This module is what keeps that table from falling behind the
suite, in the same shape as the optional-dependency census: conditional coverage is declared, not
discovered.

**What counts as a reach.** A test module (or a ``tests/support`` module it imports) that builds a path
from ``__file__``, names a top-level entry of the repository in a string (``"validation"``,
``"tools/fastgate.sh"``), imports ``subprocess``, touches ``sys.path``, imports by name
(``importlib.import_module``, ``spec_from_file_location``) or reads the mesh fixtures. Each is a sign,
not a proof: an entry may declare that the reach is one its imports already cover, with the reason.

**What it cannot see.** A reach made some other way -- a path assembled from pieces that name nothing,
an environment variable pointing at a file. The full gate in continuous integration covers those.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.support.tools import load_tool

scope = load_tool("scope_tests")

REPOSITORY = Path(__file__).resolve().parents[2]

#: Top-level entries that are not paths a test could reach: version control and generated caches.
_NOT_REACHES = {".git", "tests", ".pytest_cache", "aquaflux.egg-info", "build", "dist", ".venv"}

#: Names under which the mesh fixtures directory reaches a test.
_FIXTURE_NAMES = {"FIXTURES", "fixtures_dir"}


def reach_signs(source: str, top_level: set[str]) -> set[str]:
    """The signs in ``source`` that it reaches a file without importing it."""
    signs = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and any(
            isinstance(a, ast.Name) and a.id == "__file__" for a in node.args
        ):
            signs.add("a path built from __file__")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            head = node.value.split("/")[0]
            if head in top_level and node.value.strip():
                signs.add(f"names {node.value!r}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(m.split(".")[0] == "subprocess" for m in modules):
                signs.add("starts a subprocess")
        elif isinstance(node, ast.Attribute):
            if node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys":
                signs.add("touches sys.path")
            elif node.attr in ("import_module", "spec_from_file_location"):
                signs.add(f"imports by name ({node.attr})")
        elif (isinstance(node, ast.Name) and node.id in _FIXTURE_NAMES) or (
            isinstance(node, ast.arg) and node.arg in _FIXTURE_NAMES
        ):
            signs.add("reads the mesh fixtures")
    return signs


def _top_level() -> set[str]:
    return {p.name for p in REPOSITORY.iterdir()} - _NOT_REACHES


def _undeclared_signs(declared: dict) -> dict[str, set[str]]:
    """Each test module's signs, with those of the helpers it imports, unless declared in ``declared``."""
    top = _top_level()
    graph = scope.ImportGraph(REPOSITORY)
    relative = {
        module: path.relative_to(REPOSITORY).as_posix() for module, path in graph.paths.items()
    }
    own = {
        module: reach_signs(graph.paths[module].read_text(), top)
        for module in graph.paths
        if module.startswith("tests.") and relative[module] not in declared
    }
    found = {}
    for module in graph.paths:
        if not scope.is_test_file(relative[module]) or relative[module] in declared:
            continue  # a test's own entry speaks for the helpers it imports as well
        signs = set(own.get(module, ()))
        for helper in _test_tree_dependencies(graph, module):
            signs |= {f"{sign}, through {helper}" for sign in own.get(helper, ())}
        if signs:
            found[relative[module]] = signs
    return found


def _test_tree_dependencies(graph, module: str) -> set[str]:
    """The ``tests.support`` modules ``module`` imports, transitively.

    Not a ``conftest.py``: what one *provides* reaches a test only when the test asks for it by name,
    which its own source shows (``fixtures_dir`` as a parameter), so charging every test below a conftest
    with the conftest's own signs would bury the census in entries that declare nothing.
    """
    seen, stack = set(), [module]
    while stack:
        for target in graph.edges.get(stack.pop(), ()):
            if target.startswith("tests.support") and target in graph.paths and target not in seen:
                seen.add(target)
                stack.append(target)
    return {
        m for m in seen if not scope.is_test_file(graph.paths[m].relative_to(REPOSITORY).as_posix())
    }


def _repository_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPOSITORY), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.splitlines()


def test_every_test_that_reaches_a_file_without_importing_it_is_declared() -> None:
    undeclared = {test: sorted(signs) for test, signs in _undeclared_signs(scope.REACHES).items()}
    assert not undeclared, (
        "these test modules reach files their imports do not show, so `fastgate.sh --changed` cannot "
        "select them for a change to those files. Add each to REACHES in tools/scope_tests.py with the "
        "paths it reaches -- or the helper it reaches them through -- or `()` and the reason, if its imports already cover them:\n"
        + "\n".join(f"  {test}: {'; '.join(signs)}" for test, signs in sorted(undeclared.items()))
    )


def test_every_declared_test_exists() -> None:
    missing = [test for test in scope.REACHES if not (REPOSITORY / test).is_file()]
    assert not missing, f"REACHES names test modules that do not exist: {missing}"


def _stale(globs: set[tuple[str, str]], files: list[str]) -> list[str]:
    """Each ``(owner, glob)`` matching none of ``files``."""
    return sorted(
        f"{owner}: {glob}"
        for owner, glob in globs
        if not any(scope.matches(f, (glob,)) for f in files)
    )


@pytest.mark.parametrize("table", ["REACHES", "REACHES_NO_TEST"])
def test_every_declared_glob_still_matches_a_file(table: str) -> None:
    if table == "REACHES":
        globs = {(test, g) for test, (gs, _) in scope.REACHES.items() for g in gs}
    else:
        globs = {("REACHES_NO_TEST", g) for g in scope.REACHES_NO_TEST}
    stale = _stale(globs, _repository_files())
    assert not stale, f"{table} globs that match no file -- stale, or misspelled:\n" + "\n".join(
        stale
    )


def test_a_glob_that_matches_nothing_is_reported_stale() -> None:
    assert _stale({("t", "validation/gone/*"), ("t", "tools/*")}, ["tools/run.sh"]) == [
        "t: validation/gone/*"
    ]


@pytest.mark.parametrize(
    ("source", "sign"),
    [
        (
            "from pathlib import Path\nROOT = Path(__file__).parents[2]\n",
            "a path built from __file__",
        ),
        (
            "CASE = 'validation/pitzdaily_openfoam/case.yaml'\n",
            "names 'validation/pitzdaily_openfoam/case.yaml'",
        ),
        ("import subprocess\n", "starts a subprocess"),
        ("import sys\nsys.path.insert(0, 'x')\n", "touches sys.path"),
        (
            "import importlib\nimportlib.import_module('aquaflux.solve')\n",
            "imports by name (import_module)",
        ),
        ("def test_it(fixtures_dir):\n    pass\n", "reads the mesh fixtures"),
    ],
)
def test_each_kind_of_reach_is_seen(source: str, sign: str) -> None:
    # The census is only as good as its detector; a sign it has stopped seeing reads as a clean suite.
    assert sign in reach_signs(source, {"validation", "aquaflux"})


def test_a_test_that_only_runs_itself_under_pytest_main_is_not_a_reach() -> None:
    assert reach_signs("import pytest\npytest.main([__file__, '-v'])\n", {"validation"}) == set()


def test_a_reach_through_a_helper_is_charged_to_the_tests_that_import_it() -> None:
    # tests/support/polymesh.py builds meshes from a fixture; with its entry gone, its importers owe one.
    without = {k: v for k, v in scope.REACHES.items() if k != "tests/support/polymesh.py"}
    signs = _undeclared_signs(without)
    assert any(
        "through tests.support.polymesh" in s for s in signs["tests/unit/test_vtk_topology.py"]
    )
    assert "tests/unit/test_vtk_topology.py" not in _undeclared_signs(scope.REACHES)
