"""Every test hidden behind an optional dependency is declared here, or this fails.

``pytest.importorskip`` is invisible in an exit status: a module that is skipped and a module that
passed are the same green run. That is not a hypothetical cost. Two integration modules open with
``pytest.importorskip("petsc4py")`` and the continuous-integration workflow installs the ``test``
extra, which does not carry ``petsc`` -- so those modules have never run there. Three of their tests
failed for four days while every required check stayed green, and the failure surfaced only on the one
machine that happens to have PETSc, where it read as "broken on my machine" when the truth was the
opposite: that machine was the only place they were checked at all.

This module is the standing answer to "which coverage is conditional, and where does it disappear?".
It is the same shape as :mod:`tests.unit.test_check_hooks` and :mod:`tests.unit.test_sibling_builders`
-- a check that has quietly stopped seeing anything looks exactly like a clean tree, so the check gets
a test of its own.

**What it pins.** A module-level dependency gate must be listed in :data:`GATED_MODULES` together with
the distribution that supplies it and the tier the module's tests sit in. Adding a new gate without
adding the entry fails, which is the point: the decision "this coverage is now conditional" should be
made deliberately and written down, not arrived at by importing something.

A second, wider census covers everything else that can skip -- a data-gated fixture, a
``skipif`` on one test, a probe for a command-line tool. Those remove less, but they remove it just
as quietly, and the same rule applies: conditional coverage is declared, not discovered.

**What it deliberately does not pin.** Not the number of tests behind each gate -- that changes
whenever a test is added and would be pure churn. Not whether the dependency is present: these gates
exist precisely so the suite runs without it. What it makes impossible is a *new* gate appearing
unannounced, and a reader who wants the current picture can run this module with ``-s`` to print it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]

#: Every module-level optional-dependency gate in the suite: module path -> (import name, why).
#:
#: The tier matters as much as the dependency. A gate on a ``slow``-tier module removes coverage from
#: a tier that already runs only on a merge; a gate on a fast-tier module removes it from the required
#: check itself, which is the more surprising of the two and is where most of the hidden tests are.
GATED_MODULES: dict[str, tuple[str, str]] = {
    "unit/test_amg_preconditioner.py": ("petsc4py", "PCGAMG V-cycle; fast tier"),
    "unit/test_field_split_vcycle.py": ("petsc4py", "per-field GAMG blocks; fast tier"),
    "integration/test_coupled_amg.py": ("petsc4py", "AMG-preconditioned coupled solve; slow tier"),
    "integration/test_coupled_field_split.py": (
        "petsc4py",
        "field-split coupled solve; mostly fast tier",
    ),
}

#: Distributions that CI does not install, so every gate above naming one is dead there.
#:
#: ``petsc4py`` has no wheels -- it builds PETSc from source -- which is why the ``petsc`` extra is
#: kept out of ``test`` rather than merged into it. That is a defensible cost decision; what is not
#: defensible is making it without recording that the coverage goes with it.
NOT_INSTALLED_BY_CI: frozenset[str] = frozenset({"petsc4py", "scotchpy"})


def _module_level_gates(path: Path) -> set[str]:
    """Import names guarded by a module-level ``pytest.importorskip(...)`` in ``path``.

    Module level only: a gate inside a function skips one test, while a gate at module level removes
    the whole file, and it is the second that goes unnoticed.
    """
    tree = ast.parse(path.read_text())
    gates: set[str] = set()
    for node in tree.body:
        call = node.value if isinstance(node, ast.Expr) else None
        if isinstance(node, ast.Assign):
            call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "importorskip" or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            gates.add(first.value)
    return gates


def _discovered() -> dict[str, set[str]]:
    """Every module-level gate in the suite, keyed by path relative to ``tests/``."""
    found: dict[str, set[str]] = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        gates = _module_level_gates(path)
        if gates:
            found[path.relative_to(_TESTS).as_posix()] = gates
    return found


def test_no_test_module_is_hidden_behind_an_undeclared_dependency() -> None:
    """A new module-level ``importorskip`` must be declared, because nothing else reports one.

    Failing here is not a request to delete the gate. It is a request to decide, in the same change,
    whether the coverage it removes is coverage anyone is counting on -- and if it is, to install the
    dependency in the workflow rather than to let a green run stand for it.
    """
    discovered = _discovered()
    declared = {path: {name} for path, (name, _) in GATED_MODULES.items()}

    assert discovered == declared, (
        "the suite's module-level optional-dependency gates have changed.\n"
        f"  discovered: {discovered}\n"
        f"  declared:   {declared}\n"
        "Update GATED_MODULES in this file, and say whether CI should install the dependency."
    )


def test_every_declared_gate_names_a_module_that_exists() -> None:
    """The declaration must track the tree, or it decays into a list of files that used to exist."""
    missing = [path for path in GATED_MODULES if not (_TESTS / path).is_file()]
    assert missing == [], f"GATED_MODULES names files that are gone: {missing}"


def test_the_scanner_finds_a_gate_it_is_shown() -> None:
    """The scanner must actually see an ``importorskip``, or a green run above means nothing.

    Written against synthetic source rather than a real module: pinning it to a real file would make
    it fail whenever that file is legitimately edited, and what is under test is the scanner. The
    negative half matters as much as the positive one -- a scanner that reported every module as
    gated would also pass the equality check above, by making both sides wrong together.
    """
    import tempfile

    def gates_in(source: str) -> set[str]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(source)
            written = Path(handle.name)
        try:
            return _module_level_gates(written)
        finally:
            written.unlink()

    assert gates_in("import pytest\npytest.importorskip('somepkg')\n") == {"somepkg"}
    assert gates_in("import pytest\nmod = pytest.importorskip('somepkg')\n") == {"somepkg"}
    # A gate inside a function body is a per-test skip, not a whole-module one, and is out of scope.
    assert gates_in("import pytest\ndef test_x():\n    pytest.importorskip('somepkg')\n") == set()
    assert gates_in("import pytest\n") == set()


@pytest.mark.parametrize("path", sorted(GATED_MODULES))
def test_a_gate_on_a_dependency_ci_does_not_install_is_recorded_as_such(path: str) -> None:
    """Every gate naming an uninstalled distribution is one whose tests never run in the workflow.

    This does not fail on that state -- it is the state the project has chosen, for a dependency that
    builds from source. It fails if such a gate carries no explanation, so the next reader of a green
    slow tier can find out in one grep what that tier was not able to tell them.
    """
    name, why = GATED_MODULES[path]
    if name in NOT_INSTALLED_BY_CI:
        assert why.strip(), f"{path} is gated on {name}, which CI does not install, with no note"


#: Every OTHER file that can skip something, and what the skip is conditional on.
#:
#: These are narrower than the module-level gates above -- most remove a single test -- with one
#: exception worth naming, because it is the same whole-module shape by a different mechanism:
#: ``integration/test_bfs3d_species.py`` skips every one of its tests through a module-scoped fixture
#: when the case data is absent, and that data is deliberately not in the repository. It is therefore
#: skipped in the workflow exactly as the solver-library modules are, and it is the only test that
#: reaches into the validation cases at all.
CONDITIONALLY_SKIPPED: dict[str, str] = {
    "integration/test_bfs3d_species.py": "the whole module: the case data is not in the repository",
    "unit/test_ilu0.py": "the compiled-vs-reference agreement, when the extension is not built",
    "unit/test_lu_preconditioner.py": "the UMFPACK backend's refactor, when PETSc is absent",
    "unit/test_openfoam_fields.py": "one ordering case, when the generated grid is already ordered",
    "unit/test_partitioner.py": "the Scotch binding and command-line partitioners",
    "unit/test_sibling_builders.py": "the tool is in the repository, not the installed package",
    "unit/test_validation_api.py": "a checkout carrying no validation cases",
}


def _skipping_modules() -> set[str]:
    """Every module that can skip some of its coverage, keyed by path relative to ``tests/``.

    Syntactic rather than textual: all three mechanisms are calls or decorators
    (``pytest.importorskip``, ``pytest.skip``, ``pytest.mark.skipif``), and reading the source as
    text instead would count a skip that appears inside a string literal -- which one module here
    genuinely contains, as the body of a synthetic test tree it writes to a temporary directory.
    This module names itself, and is excluded.
    """
    wanted = {"importorskip", "skip", "skipif"}
    found = set()
    for path in sorted(_TESTS.rglob("test_*.py")):
        if path == Path(__file__):
            continue
        tree = ast.parse(path.read_text())
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        if names & wanted:
            found.add(path.relative_to(_TESTS).as_posix())
    return found


def test_every_conditional_skip_in_the_suite_is_declared() -> None:
    """Coverage that switches itself off must be written down, whichever mechanism does it.

    The module-level gates above are the loud version of this; a ``skipif`` buried on one test is the
    quiet one, and quiet is the property that matters -- a suite reports the same green either way.
    Declaring them is not an objection to any of them. It is so that "what does this suite not check
    on this machine?" has an answer that does not depend on someone thinking to look.
    """
    declared = set(GATED_MODULES) | set(CONDITIONALLY_SKIPPED)

    assert _skipping_modules() == declared, (
        "the set of modules that can skip has changed.\n"
        f"  undeclared: {sorted(_skipping_modules() - declared)}\n"
        f"  stale:      {sorted(declared - _skipping_modules())}\n"
        "Add it to CONDITIONALLY_SKIPPED (or GATED_MODULES) with what the skip is conditional on."
    )
