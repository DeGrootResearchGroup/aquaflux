"""``tools/fastgate.sh`` must report what the run actually said.

It is the blessed way to run a test tier, and it exists because the obvious hand-rolled version --
``pytest ... | tail -n`` -- reports the exit status of ``tail``, which is ``0`` however the run went.
Everything the script does is therefore a claim about honesty rather than about testing: it exits
with pytest's own status, it finds the summary line by shape because this suite prints solver-library
shutdown chatter after it, and it refuses a tier it does not recognize instead of quietly running a
different one.

None of that can report its own failure. A runner that had stopped propagating a non-zero status
would look exactly like a suite that passes, which is the same reason
:mod:`tests.unit.test_check_hooks` and :mod:`tests.unit.test_sibling_builders` exist -- and it is the
more dangerous of the three, because every other check in the project is read through this one.

Each case runs the script over a throwaway test tree, so nothing here depends on the state of the
real suite.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

FASTGATE = Path(__file__).resolve().parents[2] / "tools" / "fastgate.sh"

#: A module printing shutdown chatter AFTER pytest's summary, which is what makes a positional
#: `tail -n` unable to find the result. Written at exit, so it lands past the summary line.
_CHATTER = """
import atexit, sys

@atexit.register
def _noise():
    for i in range(40):
        print(f"library shutdown chatter line {i}", file=sys.stderr)

def test_passes():
    assert True

def test_skipped():
    import pytest
    pytest.skip("an optional dependency is absent")
"""

_FAILING = """
def test_fails():
    assert 1 == 2, "a deliberate failure"
"""


def _tree(tmp_path: Path, **modules: str) -> Path:
    """A directory holding the given test modules and nothing else."""
    root = tmp_path / "tree"
    root.mkdir(exist_ok=True)
    for name, source in modules.items():
        (root / f"{name}.py").write_text(source)
    return root


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the gate in ``cwd``. ``CI`` is set so the hooks warning cannot colour the output."""
    environment = dict(os.environ, CI="1")
    return subprocess.run(
        [str(FASTGATE), *args], cwd=cwd, capture_output=True, text=True, env=environment
    )


def test_a_passing_run_exits_zero_and_reports_its_summary(tmp_path: Path) -> None:
    """The ordinary case, and the baseline the failing case below is meaningful against."""
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, "fast", str(tree))

    assert result.returncode == 0
    assert "1 passed" in result.stdout


def test_a_FAILING_run_exits_NON_ZERO_and_names_the_failure(tmp_path: Path) -> None:
    """The property the whole script exists for: a red run must be red in the exit status.

    A pipeline through ``tail`` gets this wrong and reports success, and a caller -- a hook, a shell
    ``&&`` chain, a person reading the last line -- has no other way to find out.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER, test_bad=_FAILING)

    result = _run(tree, "fast", str(tree))

    assert result.returncode != 0
    assert "FAILED" in result.stdout
    assert "test_fails" in result.stdout


def test_the_summary_is_found_past_the_shutdown_chatter(tmp_path: Path) -> None:
    """The summary is matched by shape, so output printed after it cannot bury it.

    The synthetic tree prints forty lines of chatter at exit -- more than any fixed ``tail`` window --
    so a positional reader would report the chatter and call it the result.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, "fast", str(tree))

    reported = [line for line in result.stdout.splitlines() if line.startswith("result:")]
    assert len(reported) == 1
    assert "passed" in reported[0]
    assert "chatter" not in reported[0]


def test_the_reported_summary_carries_the_SKIP_count(tmp_path: Path) -> None:
    """A skipped test must be visible in what the runner reports, not only in the raw log.

    A skip and a pass are the same exit status, so the count is the only thing that distinguishes
    "these tests ran and passed" from "these tests were never checked here". That distinction is not
    academic: the modules gated on an optional solver library are skipped wherever it is absent, and
    a run that does not say so reads as coverage it did not provide.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, "fast", str(tree))

    reported = next(line for line in result.stdout.splitlines() if line.startswith("result:"))
    assert "1 skipped" in reported


def test_the_log_name_says_which_CHECKOUT_the_run_came_from(tmp_path: Path) -> None:
    """Several worktrees share one temporary directory, so a log name must identify its own tree.

    Without the checkout in the name, two concurrent runs write indistinguishable files into the same
    directory and the newest one is not necessarily yours. Reading another tree's log as your own
    makes a run look like it restarted, stalled or died -- and every conclusion drawn from it is about
    somebody else's code. This is the question the case runner answers with a run-file, asked of the
    test tiers instead.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, "fast", str(tree))

    checkout = FASTGATE.resolve().parents[1].name
    announced = next(line for line in result.stdout.splitlines() if line.startswith("running the"))
    assert checkout in announced
    log = Path(announced.rsplit(" ", 1)[-1])
    assert checkout in log.name and "fast" in log.name
    assert log.is_file()  # it announces the file it actually wrote


def test_an_unknown_tier_is_refused_rather_than_run_as_the_default(tmp_path: Path) -> None:
    """A mistyped tier must not silently run a different one.

    ``fastgate.sh slwo`` running the fast tier and reporting green would be read as the slow tier
    passing -- the same class of mistake the tier markers themselves are guarded against.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, "slwo", str(tree))

    assert result.returncode == 2
    assert "unknown tier" in result.stderr


@pytest.mark.parametrize("tier", ["fast", "slow", "validation", "all"])
def test_every_tier_name_is_accepted_and_reaches_pytest(tmp_path: Path, tier: str) -> None:
    """Each documented tier runs and reports a summary -- the negative case above is not vacuous.

    ``slow`` and ``validation`` select nothing from this tree, which pytest exits ``5`` for; that is
    a real distinction the gate passes through rather than hides, so it is asserted rather than
    smoothed over.
    """
    tree = _tree(tmp_path, test_ok=_CHATTER)

    result = _run(tree, tier, str(tree))

    assert "unknown tier" not in result.stderr
    assert result.returncode == (0 if tier in ("fast", "all") else 5)
