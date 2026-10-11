"""The history a resumed march is handed: what it accepts, and which anchor the damping reads."""

from __future__ import annotations

import math

import aquaflux  # noqa: F401  (enables x64)
import pytest
from aquaflux.solve import Resumption


def test_the_damping_anchor_is_the_reference_unless_one_is_given() -> None:
    assert Resumption(reference_residual=0.25).anchor == 0.25
    assert Resumption(reference_residual=0.25, damping_reference=0.5).anchor == 0.5


@pytest.mark.parametrize("name", ["reference_residual", "damping_reference", "shift"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_each_number_must_be_positive_and_finite(name, bad) -> None:
    given = {"reference_residual": 1.0} | {name: bad}
    with pytest.raises(ValueError, match=f"Resumption.{name} must be a positive finite number"):
        Resumption(**given)


def test_the_optional_numbers_may_be_left_unset() -> None:
    history = Resumption(reference_residual=1.0)
    assert (history.damping_reference, history.shift) == (None, None)
