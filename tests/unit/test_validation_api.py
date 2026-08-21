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
  accepts.

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


def _rejected_keywords(path: Path) -> list[str]:
    """Literal keyword arguments a case passes that the callable does not accept."""
    tree = ast.parse(path.read_text())
    bound = _aquaflux_imports(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        origin = bound.get(node.func.id)
        if origin is None or node.func.id in _UNCHECKABLE:
            continue
        obj = _resolve(*origin)
        accepted = _accepted_keywords(obj)
        if accepted is None:
            continue
        for keyword in node.keywords:
            # `**something` carries no name and cannot be checked; the literals beside it still can.
            if keyword.arg is not None and keyword.arg not in accepted:
                offenders.append(
                    f"{path.relative_to(_ROOT)}:{node.lineno}: {node.func.id}"
                    f"({keyword.arg}=...) -- accepts {sorted(accepted)}"
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
