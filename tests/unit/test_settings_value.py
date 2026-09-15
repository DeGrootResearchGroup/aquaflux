"""A settings value: set fields are its settings, and an abstract family base cannot be constructed."""

from __future__ import annotations

import abc
import copy
import dataclasses
import pickle

import pytest
from aquaflux.solve import SettingsValue


@dataclasses.dataclass(frozen=True)
class _Family(SettingsValue, abc.ABC):
    @abc.abstractmethod
    def kind(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class _PrivateBranch(_Family, abc.ABC):
    """An abstract, private intermediate base: neither offered nor constructible."""


@dataclasses.dataclass(frozen=True)
class Alpha(_PrivateBranch):
    sweeps: int | None = None

    def kind(self) -> str:
        return "alpha"


@dataclasses.dataclass(frozen=True)
class Beta(_Family):
    omega: float | None = None

    def kind(self) -> str:
        return "beta"


@dataclasses.dataclass(frozen=True)
class _PrivateLeaf(_Family):
    def kind(self) -> str:
        return "private"


def test_only_the_fields_that_are_set_are_settings() -> None:
    assert Alpha().settings() == {}
    assert Alpha(sweeps=2).settings() == {"sweeps": 2}


@pytest.mark.parametrize("base", [_Family, _PrivateBranch], ids=["family", "intermediate"])
def test_an_abstract_base_is_refused_naming_its_public_concrete_members(base) -> None:
    """A private concrete class is not offered, and neither is an abstract one."""
    with pytest.raises(TypeError, match=rf"^{base.__name__} is abstract; construct ") as raised:
        base()
    offered = str(raised.value)
    assert "Alpha()" in offered
    assert ("Beta()" in offered) is (base is _Family)
    assert "_PrivateLeaf" not in offered and "_PrivateBranch()" not in offered


def test_a_concrete_value_still_constructs_copies_and_pickles() -> None:
    value = Beta(omega=0.5)
    assert value == Beta(0.5) and hash(value) == hash(Beta(omega=0.5))
    assert copy.deepcopy(value) == value
    assert dataclasses.replace(value, omega=0.7) == Beta(omega=0.7)
    assert pickle.loads(pickle.dumps(_PrivateLeaf())) == _PrivateLeaf()
