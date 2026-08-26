"""The validation cases are checked against the API they call, because nothing else checks them.

The scientific cases under ``validation/`` are not part of any test tier -- they take tens of minutes
each, so they cannot run in CI or on a routine gate. That leaves them in a bad position: they are the
project's re-adjudication instruments, they call deep into the solver, and **a refactor can break them
without failing anything**. That is not hypothetical. In one session a single case was found to have
been broken three separate ways at once:

* a settings object was introduced (``RefreshPolicy``) and the case still passed a bare callable, so
  the driver raised ``AttributeError: 'function' object has no attribute 'observes'`` before its first
  step -- for every configuration, under every preconditioner;
* a guard was tightened to validate a step the case does not hand it, so the march refused to start
  with a ``TypeError`` whose own message named the step type it was rejecting as acceptable;
* the case had no ``sys.path`` bootstrap at all, so it could not be launched through the case runner.

Every one of those is cheap to detect and none of them was detected, because the suite was green: no
tier drives these files.

**What this module checks, and what it deliberately does not.** It is a STATIC check -- it imports the
API and reads the cases with :mod:`ast`, and it never builds a mesh or solves anything, so it costs
milliseconds and can live in the always-on gate:

* every name a case imports from ``aquaflux`` still exists;
* every **literal keyword argument** a case passes to an ``aquaflux`` callable is one that callable
  accepts;
* no function in a case reads a module global that only ONE branch of an ``if``/``try`` binds -- the
  shape a settings module falls into when it configures itself one block per arm and then reads the
  arms back through a branch of its own. That is a ``NameError`` on whichever arm nobody re-checked,
  and it is invisible to the lint gate (the name IS bound at module level), to an import (the read is
  in a function body) and to every tier. It shipped: a rename left one arm naming the other's
  settings, and the 3D backward-facing-step case could not start at its DEFAULT configuration for
  five days.

⚠️ **The main entry point is partly inside the blind spot.** ``solve_coupled`` takes
``**continuation_kwargs`` and forwards them to whichever continuation builder it is given, so *every*
keyword is "accepted" there and none is checked here -- which builder receives them is not knowable
statically. A misspelled or retired setting passed through that door reaches the builder and raises at
run time, not here. What ``solve_coupled`` now *does* catch itself is the case where there is no builder
to reach: given an explicit ``continuation`` or a ``RefreshPolicy(builder=...)``, a continuation setting
is refused rather than dropped in silence, which is how a ``precondition_step=`` meant for a
``RefreshPolicy`` used to disappear.

It cannot see a *semantic* break -- a parameter that still exists but now means something else, or a
type that changed under a name that did not (the ``RefreshPolicy`` case above is exactly this, and
this module would NOT have caught it). That gap is why the pre-commit reminder exists beside this: the
static half is automated here, and the half that needs judgement is a human obligation the hook
raises. Do not read a green run here as "the cases still work".
"""

from __future__ import annotations

import ast
import importlib
import inspect
import symtable
from pathlib import Path

import pytest

#: Callables whose signature says nothing useful, so a keyword check against them is noise. Keep this
#: list short and justified: every entry is a place this guard is blind.
_UNCHECKABLE: frozenset[str] = frozenset()

_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION = _ROOT / "validation"


def _cases() -> list[Path]:
    """Every validation script. Empty in a checkout without them, which is not a failure."""
    return sorted(_VALIDATION.rglob("*.py")) if _VALIDATION.is_dir() else []


def _aquaflux_imports(tree: ast.AST) -> dict[str, tuple[str, str]]:
    """Map each locally-bound name to the ``(module, attribute)`` it was imported from.

    Only ``from aquaflux... import name`` forms: a bare ``import aquaflux`` gives attribute access
    this does not try to resolve, and a name bound some other way is not an API call site.
    """
    bound: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("aquaflux"):
            for alias in node.names:
                bound[alias.asname or alias.name] = (node.module or "", alias.name)
    return bound


def _resolve(module: str, attribute: str):
    """The live object a case's import refers to, or ``None`` if it no longer exists."""
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError):
        return None


def _accepted_keywords(obj) -> frozenset[str] | None:
    """The keyword names ``obj`` accepts, or ``None`` when every keyword is accepted or unknowable.

    ``None`` covers three cases that must not be reported: a callable taking ``**kwargs`` (anything
    goes), one whose signature cannot be read (builtins, C extensions), and a non-callable.
    """
    if not callable(obj):
        return None
    try:
        parameters = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return frozenset(
        name
        for name, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _missing_names(path: Path) -> list[str]:
    """Names the case imports from ``aquaflux`` that the package no longer provides."""
    tree = ast.parse(path.read_text())
    return [
        f"{path.relative_to(_ROOT)}: `from {module} import {attribute}` -- no such name"
        for _local, (module, attribute) in _aquaflux_imports(tree).items()
        if _resolve(module, attribute) is None
    ]


def _called(node: ast.Call, bound: dict[str, tuple[str, str]]) -> tuple[str, object] | None:
    """The ``(display name, live callable)`` a call resolves to, or ``None`` if it is out of scope.

    Two forms are in scope. A bare imported name (``RefreshPolicy(...)``) is the obvious one. The
    second is a call on an imported name -- ``CoupledRANS.build(...)``,
    ``SSTTurbulence.build(...)`` -- which is how every case constructs its assemblers, and which a
    check that looked only at bare names could not see at all: those constructors carry most of the
    keywords a case passes, so the guard's own coverage was the shape of its blind spot.

    A call on anything else (an instance the case built earlier, a module attribute) is not resolvable
    without knowing that object's type, and is left alone.
    """
    if isinstance(node.func, ast.Name):
        origin = bound.get(node.func.id)
        return (node.func.id, _resolve(*origin)) if origin else None
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        origin = bound.get(node.func.value.id)
        if origin is None:
            return None
        owner = _resolve(*origin)
        name = f"{node.func.value.id}.{node.func.attr}"
        return (name, getattr(owner, node.func.attr, None)) if owner is not None else (name, None)
    return None


def _rejected_keywords(path: Path) -> list[str]:
    """Literal keyword arguments a case passes that the callable does not accept."""
    tree = ast.parse(path.read_text())
    bound = _aquaflux_imports(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _called(node, bound)
        if resolved is None:
            continue
        display, obj = resolved
        if display in _UNCHECKABLE:
            continue
        if obj is None:
            offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}: {display} -- no such name")
            continue
        accepted = _accepted_keywords(obj)
        if accepted is None:
            continue
        for keyword in node.keywords:
            # `**something` carries no name and cannot be checked; the literals beside it still can.
            if keyword.arg is not None and keyword.arg not in accepted:
                offenders.append(
                    f"{path.relative_to(_ROOT)}:{node.lineno}: {display}"
                    f"({keyword.arg}=...) -- accepts {sorted(accepted)}"
                )
    return offenders


#: Statements after which control does not reach the rest of the block, so a branch ending in one
#: constrains nothing about what is bound when the block is left.
_LEAVES = (ast.Raise, ast.Return, ast.Continue, ast.Break)


def _assigned(target: ast.AST):
    """The plain names an assignment target binds; subscript and attribute targets bind none."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _assigned(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assigned(element)


def _binds(node: ast.AST):
    """The names one statement binds in the scope it sits in, ignoring its nested blocks."""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            yield from _assigned(target)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        yield from _assigned(node.target)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            yield (alias.asname or alias.name).split(".")[0]
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        yield from _assigned(node.target)
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                yield from _assigned(item.optional_vars)


def _reached(body: list[ast.stmt]) -> bool:
    """Whether control can fall out of ``body``; a branch that raises cannot."""
    return bool(body) and not any(isinstance(statement, _LEAVES) for statement in body)


def _certainly_bound(body: list[ast.stmt]) -> set[str]:
    """The names ``body`` binds on **every** path through it.

    A two-armed ``if`` binds what both arms bind; a ``try`` binds what its body and every handler that
    can fall through all bind. A handler that re-raises is excluded, which is what makes the common
    ``try: x = f() / except ValueError: raise SystemExit(...)`` shape count as a definite binding.
    Loops are treated as bodies rather than as branches -- a module-level loop that binds a name and
    then reads it in the same block is idiom, not a hazard, and treating it as one is pure noise.
    """
    bound: set[str] = set()
    for node in body:
        bound |= set(_binds(node))
        if isinstance(node, ast.If):
            arms = [arm for arm in (node.body, node.orelse) if _reached(arm)]
            bound |= set.intersection(*map(_certainly_bound, arms)) if len(arms) == 2 else set()
        elif isinstance(node, ast.Try):
            handlers = [handler.body for handler in node.handlers if _reached(handler.body)]
            attempted = _certainly_bound(node.body) | _certainly_bound(node.orelse)
            bound |= (
                attempted
                if not handlers
                else set.intersection(attempted, *map(_certainly_bound, handlers))
            )
            bound |= _certainly_bound(node.finalbody)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With)):
            bound |= _certainly_bound(node.body)
        if isinstance(node, _LEAVES):
            break
    return bound


def _branch_bound(
    body: list[ast.stmt], guard: ast.stmt | None, found: dict[str, ast.stmt]
) -> dict[str, ast.stmt]:
    """Map each name bound inside an ``if``/``try`` branch to the outermost branch that binds it."""
    for node in body:
        if guard is not None:
            for name in _binds(node):
                found.setdefault(name, guard)
        if isinstance(node, ast.If):
            _branch_bound(node.body, guard or node, found)
            _branch_bound(node.orelse, guard or node, found)
        elif isinstance(node, ast.Try):
            _branch_bound(node.body, guard or node, found)
            for handler in node.handlers:
                _branch_bound(handler.body, guard or node, found)
            _branch_bound(node.orelse, guard or node, found)
            _branch_bound(node.finalbody, guard, found)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With)):
            _branch_bound(node.body, guard, found)
            _branch_bound(getattr(node, "orelse", []), guard, found)
    return found


def _function_scopes(table: symtable.SymbolTable, prefix: str = ""):
    """Every function scope in the module, with the dotted name a reader can find it by."""
    for child in table.get_children():
        name = f"{prefix}{child.get_name()}"
        if child.get_type() == "function":
            yield name, child
        yield from _function_scopes(child, f"{name}.")


def _conditional_reads(source: str, label: str) -> list[str]:
    """Functions that read a module global bound only inside one branch of an ``if``/``try``.

    The failure this names is a settings module that configures itself by branching -- one block per
    arm, each binding its own per-arm name -- and is then read back by a function that picks the name
    with a branch of its own. Once the two branches disagree, the arm nobody re-checked reads a name
    that does not exist, and the case dies with a ``NameError`` before its first step.

    Nothing else sees it. The name IS bound at module level as far as any static tool is concerned, so
    the lint gate is quiet; the read is inside a function body, so importing the module is quiet too;
    and no test tier runs these files. It shipped exactly this way: a rename left one arm of such a
    ternary naming the other arm's settings, and the case's DEFAULT configuration could not start for
    five days.

    A read from a scope that is itself defined inside the binding branch is fine and is not reported.

    Parameters
    ----------
    source : str
        The module source to check.
    label : str
        How to name the module in a message -- normally its path relative to the repository root.

    Returns
    -------
    list of str
        One message per offending read, empty when there are none.
    """
    tree = ast.parse(source)
    risky = {
        name: guard
        for name, guard in _branch_bound(tree.body, None, {}).items()
        if name not in _certainly_bound(tree.body)
    }
    if not risky:
        return []
    offenders = []
    for name, scope in _function_scopes(symtable.symtable(source, label, "exec")):
        for symbol in scope.get_symbols():
            guard = risky.get(symbol.get_name())
            if guard is None or symbol.is_local() or not symbol.is_referenced():
                continue
            if guard.lineno <= scope.get_lineno() <= (guard.end_lineno or guard.lineno):
                continue  # the reader is itself defined inside the branch that binds it
            offenders.append(
                f"{label}: {name}() reads `{symbol.get_name()}`, which is bound only inside the "
                f"branch at line {guard.lineno}"
            )
    return offenders


@pytest.mark.skipif(not _cases(), reason="this checkout carries no validation cases")
def test_the_cases_import_names_that_still_exist() -> None:
    """A rename that misses these files breaks a study rather than a test -- which is found later.

    The cases are the instruments the design record's findings were measured with, so a broken one is
    not merely an inconvenience: it makes a recorded number un-re-adjudicable, and this project treats
    an unfalsifiable finding as worse than a wrong one.
    """
    offenders = [problem for path in _cases() for problem in _missing_names(path)]

    assert offenders == [], "validation cases import names that no longer exist:\n  " + "\n  ".join(
        offenders
    )


@pytest.mark.skipif(not _cases(), reason="this checkout carries no validation cases")
def test_the_cases_pass_keywords_the_api_accepts() -> None:
    """A removed or renamed parameter is caught here rather than an hour into a case.

    Only literal keywords against a resolvable signature; a callable taking ``**kwargs`` accepts
    anything and is skipped, as is any signature that cannot be read.
    """
    offenders = [problem for path in _cases() for problem in _rejected_keywords(path)]

    assert offenders == [], (
        "validation cases pass keywords the API does not accept:\n  " + "\n  ".join(offenders)
    )


def test_the_checker_actually_catches_a_break() -> None:
    """The guard must fail on a broken case, or a green run means nothing.

    Written against synthetic source rather than a real case: pinning it to a real one would make this
    test fail whenever that case is legitimately edited, and what is under test is the checker.
    """
    source = "from aquaflux.solve import RetryPolicy\nRetryPolicy(no_such_parameter=1)\n"
    tree = ast.parse(source)
    bound = _aquaflux_imports(tree)

    assert bound == {"RetryPolicy": ("aquaflux.solve", "RetryPolicy")}
    accepted = _accepted_keywords(_resolve(*bound["RetryPolicy"]))
    assert accepted is not None and "no_such_parameter" not in accepted
    assert "abort_above_cycles" in accepted  # and it reads the real signature, not an empty set


def test_the_checker_reaches_a_call_on_an_imported_CLASS_not_only_a_bare_name() -> None:
    """``CoupledRANS.build(...)`` must be checked, because that is how every case is wired.

    The bare-name form above is the easy half. The cases construct their assemblers through class
    methods, so a guard that resolved only bare names would report clean over every constructor call
    in every case -- coverage exactly where the cases spend their keywords, and it would look
    identical to coverage that worked.

    Both directions are pinned. A wrong keyword and a method that no longer exists are found; a
    correct call is left alone, so the check cannot be passing by objecting to everything.
    """
    module = "from aquaflux.turbulence import CoupledRANS\n"
    bound = _aquaflux_imports(ast.parse(module))

    def called(source: str):
        node = next(n for n in ast.walk(ast.parse(module + source)) if isinstance(n, ast.Call))
        return _called(node, bound)

    display, obj = called("CoupledRANS.build(m, t, omega_transform=None)\n")
    assert display == "CoupledRANS.build"
    accepted = _accepted_keywords(obj)
    assert accepted is not None
    assert "omega_transform" in accepted  # the real signature, not an empty set
    assert "no_such_parameter" not in accepted

    assert called("CoupledRANS.no_such_method(x=1)\n") == ("CoupledRANS.no_such_method", None)
    # A call on something the case built itself is not resolvable and must be left alone.
    assert called("solver.step(x=1)\n") is None


@pytest.mark.skipif(not _cases(), reason="this checkout carries no validation cases")
def test_the_cases_do_not_read_a_global_that_only_one_branch_binds() -> None:
    """A case that configures itself by branching must not read one arm's name from the other's.

    This is the one break in this module that neither of the checks above can see and that importing
    the module cannot see either: the name exists at module level, and the read is inside a function.
    """
    offenders = [
        problem
        for path in _cases()
        for problem in _conditional_reads(path.read_text(), str(path.relative_to(_ROOT)))
    ]

    assert offenders == [], (
        "validation cases read a global that only one branch binds:\n  " + "\n  ".join(offenders)
    )


def test_the_branch_checker_separates_a_real_hazard_from_the_idioms_around_it() -> None:
    """Both directions, because a check that objects to everything and one that sees nothing agree.

    The hazard is a name bound in only one arm and read outside it. The idioms it must stay quiet
    about are the ones this repository's cases are written in: a name bound by both arms, a name whose
    other arm raises instead of binding, and a helper defined inside the branch that binds it.
    """
    hazard = (
        "import os\nif os.environ.get('X'):\n    SETTINGS = {}\ndef run():\n    return SETTINGS\n"
    )
    (problem,) = _conditional_reads(hazard, "case.py")
    assert "run() reads `SETTINGS`" in problem and "line 2" in problem

    both_arms = (
        "import os\nif os.environ.get('X'):\n    SETTINGS = {}\nelse:\n    SETTINGS = {'a': 1}\n"
        "def run():\n    return SETTINGS\n"
    )
    assert _conditional_reads(both_arms, "case.py") == []

    raising_arm = (
        "import os\ntry:\n    SETTINGS = int(os.environ['X'])\nexcept ValueError:\n"
        "    raise SystemExit('bad X')\ndef run():\n    return SETTINGS\n"
    )
    assert _conditional_reads(raising_arm, "case.py") == []

    reader_inside = (
        "import os\nif os.environ.get('X'):\n    SETTINGS = {}\n\n    def run():\n"
        "        return SETTINGS\n"
    )
    assert _conditional_reads(reader_inside, "case.py") == []

    # A local of the same name is not a read of the global, or every case would report.
    shadowed = (
        "import os\nif os.environ.get('X'):\n    SETTINGS = {}\ndef run():\n    SETTINGS = {}\n"
        "    return SETTINGS\n"
    )
    assert _conditional_reads(shadowed, "case.py") == []
