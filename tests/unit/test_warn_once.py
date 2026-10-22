"""``warn_once``: one emission per kind per process, a separate record per kind, and a stack level
counted from its caller."""

from __future__ import annotations

import inspect
import warnings

import pytest
from aquaflux.warn_once import forget_warnings, warn_once

FIRST, SECOND = "test_warn_once.first", "test_warn_once.second"


@pytest.fixture(autouse=True)
def _fresh_record():
    forget_warnings(FIRST, SECOND)
    yield
    forget_warnings(FIRST, SECOND)


def _emitted(*calls) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for call in calls:
            call()
    return caught


def test_a_kind_is_said_once_and_does_not_silence_another() -> None:
    # The second kind is the case a single shared flag gets wrong: a routine first report would
    # silence a later, graver one.
    caught = _emitted(
        lambda: warn_once(FIRST, "first"),
        lambda: warn_once(FIRST, "first again, with different numbers"),
        lambda: warn_once(SECOND, "second"),
    )
    assert [str(w.message) for w in caught] == ["first", "second"]
    assert all(w.category is UserWarning for w in caught)


def test_forgetting_a_kind_lets_only_that_kind_be_said_again() -> None:
    _emitted(lambda: warn_once(FIRST, "a"), lambda: warn_once(SECOND, "b"))
    forget_warnings(FIRST)
    caught = _emitted(lambda: warn_once(FIRST, "a"), lambda: warn_once(SECOND, "b"))
    assert [str(w.message) for w in caught] == ["a"]

    forget_warnings()
    caught = _emitted(lambda: warn_once(FIRST, "a"), lambda: warn_once(SECOND, "b"))
    assert [str(w.message) for w in caught] == ["a", "b"]


def test_the_category_is_the_one_asked_for() -> None:
    caught = _emitted(lambda: warn_once(FIRST, "a", category=RuntimeWarning))
    assert caught[0].category is RuntimeWarning


def _reports(stacklevel: int) -> int:
    """Warn from here, returning the line that called this function."""
    warn_once(FIRST, "located", stacklevel=stacklevel)  # line of the call
    return inspect.currentframe().f_back.f_lineno


def test_the_stack_level_is_counted_from_the_code_that_calls_it() -> None:
    call_line = inspect.getsourcelines(_reports)[1] + 2
    caught = _emitted(lambda: _reports(1))
    assert caught[0].lineno == call_line  # the warn_once call itself, not a line inside it

    forget_warnings(FIRST)
    lines = []
    caught = _emitted(lambda: lines.append(_reports(2)))
    assert caught[0].lineno == lines[0]  # the caller's caller
