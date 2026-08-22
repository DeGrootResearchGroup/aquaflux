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


def _run(source: str, tmp_path: Path) -> str:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text(source)
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


@pytest.mark.skipif(not TOOL.exists(), reason="the tool is part of the repository, not the package")
def test_the_package_report_still_reaches_the_coupled_builders(tmp_path: Path) -> None:
    """Run it where it matters, and check it has not gone blind to the family it exists for.

    This is deliberately **not** an assertion that the package reports zero pairs. It once was, and that
    made a report into a gate — which is wrong twice over: whether a pair is one builder or two
    genuinely different methods is a judgement no script can make, and a green gate here was
    indistinguishable from a check that had stopped seeing anything at all. What is worth pinning is
    that the coupled march's builders, which reach their shared step through a private tail, are still
    *visible* to it. If they are ever genuinely unified into one builder, updating this is part of that
    change.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        text=True,
        check=False,
        cwd=TOOL.parent.parent,
    )
    assert result.returncode == 0, "the report must always exit 0"
    assert (
        "coupled_continuation" in result.stdout and "coupled_amg_continuation" in result.stdout
    ), (
        "the coupled builders delegate to a shared private tail; the report must follow it, or it is "
        f"blind to exactly the drift it was written for:\n{result.stdout}"
    )
