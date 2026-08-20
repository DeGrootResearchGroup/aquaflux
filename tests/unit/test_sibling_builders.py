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


@pytest.mark.skipif(not TOOL.exists(), reason="the tool is part of the repository, not the package")
def test_the_package_itself_is_clean(tmp_path: Path) -> None:
    """The check that would have caught the coupled-continuation split, run where it matters.

    A failure here is not necessarily a defect — two genuinely different methods can trip it — but it is
    always something to look at, which is why the message says so rather than asserting a bare count.
    """
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        text=True,
        check=False,
        cwd=TOOL.parent.parent,
    )
    assert result.returncode == 0
    assert "no sibling-builder pairs" in result.stdout, (
        "sibling builders reappeared; each 'only here' entry must be a genuine property of that "
        f"path, not drift:\n{result.stdout}"
    )
