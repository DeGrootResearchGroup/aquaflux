"""The sibling-builder report must fire on a duplicated builder pair — and stay quiet otherwise.

Tested for the same reason ``tools/check_hooks.sh`` is: this check reports **nothing** on a clean tree,
and a broken version reports nothing too. The two states are indistinguishable from the output, so the
only way to know the check still works is to hand it something it must object to.

The defect it exists for is a coupled march whose k-positivity limit — the fix for a solve that went
non-finite from ``k < 0`` in two cells of 23040 — was wired on one of four builders of the same step and
reached none of the others.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "sibling_builders.py"

#: Two builders of one class whose surfaces overlap in every parameter but one — the shape the report
#: exists to name. `slow` is the drift: present on one, absent on the other.
_SIBLINGS = """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass


def build_one(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f)


def build_two(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)
"""

#: Three solvers that share a vocabulary but are different methods with different smoother families —
#: the false positive the rule's carve-out protects, and which the report must not raise.
_DIFFERENT_METHODS = """
def smoothed_solve(hierarchy, b, *, cycles=1, omega=1.0, sweeps=2):
    return Result(hierarchy, b, cycles)


def air_solve(hierarchy, b, *, cycles=1, f_iters=2, c_iters=1):
    return Result(hierarchy, b, cycles)
"""

#: The same drifted pair, after the duplicated *body* has been extracted into a shared private tail —
#: which is the right repair for the body and, on its own, hides the surfaces above it. Neither builder
#: constructs `Step` any more, so a check that looks only at direct construction reports nothing here
#: while `slow` still sits on one of them. Two levels of delegation, because a real builder reaches the
#: shared step through a per-family seam.
_SHARED_TAIL = """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass


def _tail(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)


def _family_seam(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return _tail(policy, a=a, b=b, c=c, d=d, e=e, f=f)


def build_one(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return _family_seam(policy, a=a, b=b, c=c, d=d, e=e, f=f)


def build_two(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return _tail(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)
"""


#: Two classmethod factories on different classes, each returning ``cls(...)`` over one shared private
#: tail — the shape a scheme-level factory takes. Nothing here is a call to a capitalized name, so a
#: check keyed on naming convention alone credits both with constructing nothing and drops them from
#: the report entirely: not a quiet pair, but no pair at all, which reads identically to a clean tree.
_CLASSMETHOD_FACTORIES = """
class Solve:
    def __init__(self, count, warn=None):
        pass


def _calibrated(system, *, a=0, b=0, c=0, d=0, e=0, warn=None):
    return Solve(system, warn=warn)


class SchemeOne:
    @classmethod
    def calibrated(cls, mesh, *, a=0, b=0, c=0, d=0, e=0):
        return cls(solver=_calibrated(mesh, a=a, b=b, c=c, d=d, e=e))


class SchemeTwo:
    @classmethod
    def calibrated(cls, mesh, *, a=0, b=0, c=0, d=0, e=0, slow=None):
        return cls(solver=_calibrated(mesh, a=a, b=b, c=c, d=d, e=e), slow=slow)
"""


# A classmethod factory whose name is in no convention list -- the blind spot the structural test
# closes. Both build the same class from the same surface; only `slow` differs.
_UNCONVENTIONALLY_NAMED_FACTORIES = """
class Solve:
    def __init__(self, count, warn=None):
        pass


def _fitted(system, *, a=0, b=0, c=0, d=0, e=0, warn=None):
    return Solve(system, warn=warn)


class SchemeOne:
    @classmethod
    def by_counts(cls, mesh, *, a=0, b=0, c=0, d=0, e=0):
        return cls(solver=_fitted(mesh, a=a, b=b, c=c, d=d, e=e))


class SchemeTwo:
    @classmethod
    def by_counts(cls, mesh, *, a=0, b=0, c=0, d=0, e=0, slow=None):
        return cls(solver=_fitted(mesh, a=a, b=b, c=c, d=d, e=e), slow=slow)
"""


#: The same drifted pair again, with the shared tail extracted **onto another object in another
#: module** -- a configuration value object whose method builds the step. Nothing here is spelled like
#: a construction from the builders' side: the call is ``settings.step(...)``, lowercase and reached
#: through an attribute, and it crosses a file. Resolution that stopped at same-module private
#: functions saw neither, so both builders dropped out of the report entirely while `slow` still sat
#: on one of them -- a whole family reported as a clean tree.
_METHOD_TAIL_IN_ANOTHER_MODULE = {
    "config.py": """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass


class Settings:
    def step(self, policy, **fields):
        return Step(policy, **fields)
""",
    "builders.py": """
from .config import Settings


def build_one(policy, *, settings=Settings(), a=0, b=0, c=0, d=0, e=0, f=0):
    return settings.step(policy, a=a, b=b, c=c, d=d, e=e, f=f)


def build_two(policy, *, settings=Settings(), a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return settings.step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)
""",
}


#: The drifted pair split across two SUBPACKAGES of one tree -- a flow-only builder and a coupled one,
#: which is where six builders of one march actually lived. A same-directory rule treated them as
#: unrelated namesakes and discarded the pair before comparing a single parameter, which is how that
#: family went unreported with six shared parameters against a threshold of five.
_SIBLINGS_IN_TWO_SUBPACKAGES = {
    "solve.py": """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass
""",
    "flow/__init__.py": "",
    "flow/continuation.py": """
from ..solve import Step


def build_one(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f)
""",
    "turbulence/__init__.py": "",
    "turbulence/coupled.py": """
from ..solve import Step


def build_two(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)
""",
}

#: A public builder that delegates to another public builder, forwarding the options it was given. The two
#: share that surface by construction -- the delegation working, not two copies drifting -- so pairing
#: them reports every wrapper beside its callee and buries the pairs a reader has to judge.
_PUBLIC_WRAPPER = """
class VCycle:
    def __init__(self, matrix, a=0, b=0, c=0, d=0, e=0):
        pass


def build_vcycle(matrix, *, a=0, b=0, c=0, d=0, e=0):
    return VCycle(matrix, a=a, b=b, c=c, d=d, e=e)


class Preconditioner:
    def __init__(self, cycle):
        pass

    @classmethod
    def build(cls, matvec, *, a=0, b=0, c=0, d=0, e=0, probe=None):
        return cls(build_vcycle(matvec, a=a, b=b, c=c, d=d, e=e))
"""


#: The drifted pair again, with one builder reaching the shared tail through a STRATEGY OBJECT it
#: opens: ``session = open_session(...)`` then ``return session._build(...)``. ``_build`` is defined on
#: three classes, so the name alone is ambiguous -- and one of the three, ``Unrelated``, builds a
#: different class entirely. The two sessions then reach the tail by the two other shapes a method
#: body takes: ``self._step(...)`` (also defined on ``Unrelated``) and a local bound to a call and
#: returned. ``build_noise`` is the precision check: it builds only what ``Unrelated`` builds, so it
#: pairs with ``build_one`` exactly when resolution unions every ``_build`` instead of following the
#: receiver.
_SESSION_RECEIVER = {
    "steps.py": """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass


class Noise:
    def __init__(self):
        pass


def _tail(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)
""",
    "sessions.py": """
from .steps import Noise, _tail


class _OneSession:
    def _build(self, policy, fields):
        return self._finish(self._step(policy, fields))

    def _step(self, policy, fields):
        return _tail(policy, **fields)

    def _finish(self, step):
        return step


class _OtherSession:
    def _build(self, policy, fields):
        step = _tail(policy, **fields)
        return step if fields else None


class Unrelated:
    def _build(self):
        return Noise()

    def _step(self):
        return Noise()


def open_session(kind):
    if kind:
        return _OneSession()
    return _OtherSession()
""",
    "builders.py": """
from .sessions import open_session
from .steps import Noise, _tail


def build_one(policy, *, kind=None, a=0, b=0, c=0, d=0, e=0, f=0):
    session = open_session(kind)
    return session._build(policy, dict(a=a, b=b, c=c, d=d, e=e, f=f))


def build_two(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return _tail(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)


def build_noise(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return Noise()
""",
}


def _run(source: str, tmp_path: Path) -> str:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text(source)
    result = subprocess.run(
        [sys.executable, str(TOOL), str(package)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"the report must always exit 0, got {result.returncode}"
    return result.stdout


def _run_package(sources: dict[str, str], tmp_path: Path) -> str:
    """Run the report over a multi-module package, for the delegations that cross a file."""
    package = tmp_path / "pkg"
    package.mkdir()
    for name, source in sources.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    result = subprocess.run(
        [sys.executable, str(TOOL), str(package)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"the report must always exit 0, got {result.returncode}"
    return result.stdout


def test_it_reports_a_duplicated_builder_pair_and_names_the_drift(tmp_path: Path) -> None:
    out = _run(_SIBLINGS, tmp_path)
    assert "1 sibling-builder pair" in out
    assert "build_one" in out and "build_two" in out
    assert "Step" in out
    # The drift itself, which is the actionable half: `slow` is on one side only.
    assert "'slow'" in out


def test_it_stays_quiet_for_siblings_that_only_share_a_vocabulary(tmp_path: Path) -> None:
    """Different methods taking similarly-named arguments are not one builder written twice."""
    assert "no sibling-builder pairs" in _run(_DIFFERENT_METHODS, tmp_path)


def test_it_sees_through_a_shared_private_tail(tmp_path: Path) -> None:
    """Extracting the duplicated body must not take the drift signal with it.

    This is the case that went undetected: the shared tail was extracted — the right repair — and the
    two builders stopped constructing a common class directly, so the report went quiet while their
    public surfaces stayed hand-copied and one kept a keyword the other never got. The report has to
    follow the delegation, transitively, and name the same drift it would have named before.
    """
    out = _run(_SHARED_TAIL, tmp_path)
    assert "build_one" in out and "build_two" in out
    assert "Step" in out, "the tail's constructed class must be credited to its callers"
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"


def test_it_sees_through_a_tail_that_is_a_METHOD_IN_ANOTHER_MODULE(tmp_path: Path) -> None:
    """The shared tail need not be a private function, and need not be in the same file.

    A configuration object whose method builds the step is the natural place for a surface several
    builders share -- it is one object rather than one keyword list copied N times, so the settings
    *cannot* drift. What can still drift is everything above it, and that is what this report is for.
    Both spellings the extraction introduces defeat a same-module private-function rule: the callee is
    an attribute, so its owning class is not in the syntax tree, and it lives in another module.
    """
    out = _run_package(_METHOD_TAIL_IN_ANOTHER_MODULE, tmp_path)
    assert "build_one" in out and "build_two" in out, (
        f"a tail on another object in another module hid the pair entirely:\n{out}"
    )
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"


def test_it_pairs_siblings_that_live_in_different_subpackages(tmp_path: Path) -> None:
    """A directory boundary is not evidence that two builders are unrelated.

    The builders of one engine naturally live beside the physics they configure, which is several
    subpackages. What separates siblings from namesakes is a shared constructed class and a shared
    surface -- both already required -- not where the files sit.
    """
    out = _run_package(_SIBLINGS_IN_TWO_SUBPACKAGES, tmp_path)
    assert "build_one" in out and "build_two" in out, (
        f"a pair split across two subpackages was discarded as namesakes:\n{out}"
    )
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"


def test_it_does_not_pair_a_public_builder_with_the_builder_it_delegates_to(tmp_path: Path) -> None:
    """A wrapper shares its callee's surface by construction, exactly as a private tail does.

    Following delegation into public functions and methods -- which a tail extracted onto a
    configuration object needs -- would otherwise report every wrapper beside the builder it calls.
    """
    out = _run(_PUBLIC_WRAPPER, tmp_path)
    assert "no sibling-builder pairs" in out, out


def test_it_does_not_pair_a_private_tail_with_its_own_callers(tmp_path: Path) -> None:
    """A tail shares most of its surface with every builder that delegates to it, by construction.

    Reporting those pairs is the extraction working, not drift, and at four builders and two seams it
    buries the pairs a reader has to judge. Only public surfaces are compared.
    """
    out = _run(_SHARED_TAIL, tmp_path)
    assert "_tail" not in out and "_family_seam" not in out
    assert "1 sibling-builder pair" in out


def test_it_reaches_classmethod_factories_that_return_cls(tmp_path: Path) -> None:
    """A ``@classmethod`` factory building its own class is invisible to a naming convention.

    ``return cls(...)`` names no class, so crediting construction by capitalization alone finds
    nothing to credit and the factory never enters the report — which looks exactly like a tree with
    no drift in it. Two schemes calibrating themselves from one shared tail are one configuration
    surface written twice, and the report has to say so: here ``slow`` sits on one side only.
    """
    out = _run(_CLASSMETHOD_FACTORIES, tmp_path)
    assert "SchemeOne.calibrated" in out and "SchemeTwo.calibrated" in out
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"


def test_it_reaches_a_classmethod_factory_no_naming_convention_covers(tmp_path: Path) -> None:
    """The name list cannot be the primary test, because it is blind to every name nobody added.

    That blindness has cost twice, and each time the fix was to teach it one more name -- which leaves
    the next naming style just as invisible, and a factory it cannot see reports as a clean tree rather
    than as a gap. A ``@classmethod`` whose body returns ``cls(...)`` is a factory whatever it is
    called, so the shape is what qualifies it and the name list is only the fallback for factories
    whose construction the syntax tree cannot follow.
    """
    out = _run(_UNCONVENTIONALLY_NAMED_FACTORIES, tmp_path)
    assert "SchemeOne.by_counts" in out and "SchemeTwo.by_counts" in out
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"


def test_it_follows_a_method_through_a_receiver_whose_class_is_knowable(tmp_path: Path) -> None:
    """A builder that opens a strategy object and builds through it is still a builder.

    ``_build`` is defined on several classes, and a name defined more than once was never followed, so
    a builder written this way was credited with constructing nothing and left the report entirely --
    the coupled march's one remaining builder did exactly that, reading as a clean tree. The method is
    resolved on the classes ``open_session`` returns, and on no other class that defines ``_build``.
    """
    out = _run_package(_SESSION_RECEIVER, tmp_path)
    assert "build_one" in out and "build_two" in out, (
        f"a builder reaching its tail through an opened session dropped out of the report:\n{out}"
    )
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"
    assert "build_noise" not in out, (
        f"the receiver's method was resolved on every class defining it, not on the receiver:\n{out}"
    )
    assert "1 sibling-builder pair" in out, out


#: A shared step class and a decoy, used by the receiver fixtures below. Each builder pair in them
#: shares six parameters with ``build_two``, which reaches ``Step`` directly, and ``slow`` is the drift.
_STEPS = """
class Step:
    def __init__(self, policy, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
        pass


class Noise:
    def __init__(self):
        pass


def _tail(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return Step(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)


def build_two(policy, *, a=0, b=0, c=0, d=0, e=0, f=0, slow=None):
    return _tail(policy, a=a, b=b, c=c, d=d, e=e, f=f, slow=slow)


def build_noise(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return Noise()
"""

_BUILD_ONE_THROUGH_A_SESSION = """
from .sessions import open_session


def build_one(policy, *, a=0, b=0, c=0, d=0, e=0, f=0):
    session = open_session()
    return session._build(policy, dict(a=a, b=b, c=c, d=d, e=e, f=f))
"""

#: The session reaches the tail ONLY through ``self._step``, and ``_step`` is defined on two classes --
#: one of which builds the decoy. The pair is reported only if ``self.m`` resolves on the owning class.
_SELF_METHOD = {
    "steps.py": _STEPS,
    "builders.py": _BUILD_ONE_THROUGH_A_SESSION,
    "sessions.py": """
from .steps import Noise, _tail


class Session:
    def _build(self, policy, fields):
        return self._step(policy, fields)

    def _step(self, policy, fields):
        return _tail(policy, **fields)


class Unrelated:
    def _step(self):
        return Noise()


def open_session():
    return Session()
""",
}

#: The session reaches the tail ONLY through a local bound to a call and returned as one arm of a
#: conditional expression.
_RETURNED_LOCAL = {
    "steps.py": _STEPS,
    "builders.py": _BUILD_ONE_THROUGH_A_SESSION,
    "sessions.py": """
from .steps import _tail


class Session:
    def _build(self, policy, fields):
        step = _tail(policy, **fields)
        return step if fields else None


def open_session():
    return Session()
""",
}

#: The session reaches the tail ONLY through ``self._step``, and ``_step`` is INHERITED -- defined on a
#: base class, not on ``Session`` -- and defined exactly once in the package.
_INHERITED_SELF_METHOD = {
    "steps.py": _STEPS,
    "builders.py": _BUILD_ONE_THROUGH_A_SESSION,
    "sessions.py": """
from .steps import _tail


class Base:
    def _step(self, policy, fields):
        return _tail(policy, **fields)


class Session(Base):
    def _build(self, policy, fields):
        return self._step(policy, fields)


def open_session():
    return Session()
""",
}

#: A receiver whose producer builds a SECOND class as an argument: ``Session(Helper())``. ``Helper``
#: also defines ``_build``, returning the decoy. A receiver typed by everything the producer's return
#: mentions resolves ``session._build`` on ``Helper`` as well, and pairs ``build_one`` with the decoy.
_PRODUCER_ARGUMENT = {
    "steps.py": _STEPS,
    "builders.py": _BUILD_ONE_THROUGH_A_SESSION,
    "sessions.py": """
from .steps import Noise, _tail


class Session:
    def __init__(self, helper):
        pass

    def _build(self, policy, fields):
        return _tail(policy, **fields)


class Helper:
    def _build(self):
        return Noise()


def open_session():
    return Session(Helper())
""",
}

#: Locals a return READS but does not return: ``m.value``, and ``m`` as a call's argument. Neither
#: builder builds a ``Measurement``, so neither may pair with the decoy that does.
_READ_BUT_NOT_RETURNED = """
class Measurement:
    pass


class Result:
    pass


def measure(a):
    return Measurement()


def build_from_attribute(p, *, a=0, b=0, c=0, d=0, e=0, f=0):
    m = measure(a)
    return Result(m.value, p)


def build_from_argument(p, *, a=0, b=0, c=0, d=0, e=0, f=0):
    m = measure(a)
    return Result(m, p)


def build_measurement(p, *, a=0, b=0, c=0, d=0, e=0, f=0):
    return Measurement()
"""

#: A public builder that reaches a public factory through a TYPED RECEIVER, so the delegation is
#: recorded as ``Preconditioner.build`` rather than as the bare ``build``. The two share the factory's
#: surface by construction and must not pair.
_WRAPPER_THROUGH_A_RECEIVER = {
    "pc.py": """
class VCycle:
    def __init__(self, matrix, a=0, b=0, c=0, d=0, e=0):
        pass


class Preconditioner:
    def __init__(self, cycle):
        pass

    def build(self, matrix, *, a=0, b=0, c=0, d=0, e=0):
        return Preconditioner(VCycle(matrix, a=a, b=b, c=c, d=d, e=e))


class Other:
    def build(self):
        return VCycle(None)


def open_preconditioner():
    return Preconditioner(None)
""",
    "builders.py": """
from .pc import open_preconditioner


def build_preconditioner(matrix, *, a=0, b=0, c=0, d=0, e=0):
    pc = open_preconditioner()
    return pc.build(matrix, a=a, b=b, c=c, d=d, e=e)
""",
}


@pytest.mark.parametrize(
    "sources",
    [_SELF_METHOD, _RETURNED_LOCAL, _INHERITED_SELF_METHOD],
    ids=["self-method-on-its-own-class", "returned-local", "inherited-self-method"],
)
def test_each_way_a_session_method_reaches_its_tail_is_followed(sources, tmp_path: Path) -> None:
    """Each fixture reaches the shared tail by exactly ONE resolution rule, so each rule is pinned alone.

    A single fixture exercising all three at once let any one of them be deleted with the tests still
    green, because the others still carried the builder into the report.
    """
    out = _run_package(sources, tmp_path)
    assert "build_one" in out and "build_two" in out, (
        f"a builder reaching its tail through this route dropped out of the report:\n{out}"
    )
    assert "'slow'" in out, "the drifted keyword is the actionable half of the report"
    assert "build_noise" not in out, f"a method was resolved on a class it is not called on:\n{out}"


def test_a_receiver_is_typed_by_what_its_producer_returns_not_by_its_arguments(
    tmp_path: Path,
) -> None:
    """``return Session(Helper())`` returns a ``Session``; the ``Helper`` is only an argument."""
    out = _run_package(_PRODUCER_ARGUMENT, tmp_path)
    assert "build_noise" not in out, (
        f"a receiver's method was resolved on a class its producer only passes as an argument:\n{out}"
    )
    assert "build_one" in out and "'slow'" in out, out


def test_a_local_the_return_only_reads_is_not_credited(tmp_path: Path) -> None:
    """``rate = measure(...)`` then ``return cls(sweeps=rate.value)`` does not build what ``measure`` built.

    Crediting every mention of a bound local put six invented pairs into the package report, all of
    them sharing nothing but a measurement a calibration helper returns.
    """
    out = _run(_READ_BUT_NOT_RETURNED, tmp_path)
    assert "build_measurement" not in out, (
        f"a builder was credited with a class its return only reads from:\n{out}"
    )


def test_a_wrapper_reached_through_a_typed_receiver_is_not_paired_with_its_callee(
    tmp_path: Path,
) -> None:
    """The delegation is recorded as ``Class.method``, so the exclusion must compare the qualified label."""
    out = _run_package(_WRAPPER_THROUGH_A_RECEIVER, tmp_path)
    assert "no sibling-builder pairs" in out, out


@pytest.mark.skipif(not TOOL.exists(), reason="the tool is part of the repository, not the package")
def test_the_package_report_still_reaches_the_coupled_builders(tmp_path: Path) -> None:
    """Run it where it matters, and check it has not gone blind to the family it exists for.

    This is deliberately **not** an assertion that the package reports zero pairs. It once was, and that
    made a report into a gate — which is wrong twice over: whether a pair is one builder or two
    genuinely different methods is a judgement no script can make, and a green gate here was
    indistinguishable from a check that had stopped seeing anything at all. What is worth pinning is
    that the coupled march's builders are still *visible* to it. There are two: ``coupled_step``, which
    reaches its step through a preconditioner session's ``_build`` -- a method several classes define --
    and the bordered mass-flow builder, which shares nearly all of its surface. The tool was once blind
    to the first, and the family dropped out of the report entirely.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        text=True,
        check=False,
        cwd=TOOL.parent.parent,
    )
    assert result.returncode == 0, "the report must always exit 0"
    assert "sibling-builder pair(s)" in result.stdout, (
        f"the report found no pairs at all, which is a check that has stopped seeing:\n{result.stdout}"
    )
    pairs = result.stdout.split("\n\n")
    assert any(
        " coupled_step\n" in pair and " mass_flow_coupled_continuation\n" in pair for pair in pairs
    ), (
        "the coupled march's two builders are not reported as a pair, so the tool has stopped seeing "
        f"the family it exists for:\n{result.stdout}"
    )
