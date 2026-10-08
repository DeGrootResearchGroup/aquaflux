"""Tests for :class:`~aquaflux_ui.run_process.CaseRun`, the ``aquaflux run`` process the Run section drives.

A short Python program stands in for the solver: it prints what it was asked, and exits or waits as
the test needs. What is pinned is how the page starts, watches and stops a run, not the solve.
"""

from __future__ import annotations

import sys
import textwrap
import time

import pytest

pytest.importorskip("trame")

from aquaflux_ui.run_process import CONVERGED, NOT_CONVERGED, CaseRun

#: Prints its arguments and exits with the status its case file's name ends in (``case-1`` exits 1, a
#: name with none exits 0); a case named ``wait`` waits for an interrupt instead, and exits 1 when it
#: gets one, as ``aquaflux run`` records an interrupted run.
_STAND_IN = textwrap.dedent(
    """
    import os, sys, time
    print("asked:", " ".join(sys.argv[1:]), flush=True)
    name = os.path.basename(sys.argv[2])
    if name == "wait":
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            sys.exit(1)
    sys.exit(int(name.split("-")[-1]) if "-" in name else 0)
    """
)


@pytest.fixture
def stand_in(tmp_path):
    script = tmp_path / "stand_in.py"
    script.write_text(_STAND_IN)
    return [sys.executable, str(script)]


def _wait(run: CaseRun, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while run.running and time.monotonic() < deadline:
        time.sleep(0.05)


def test_a_run_reports_its_exit_status_and_what_it_printed(tmp_path, stand_in):
    run = CaseRun(tmp_path / "case-1", tmp_path / "console.txt", command=stand_in)
    assert run.status is None and not run.running
    run.start()
    _wait(run)
    assert run.status == NOT_CONVERGED
    assert run.console_tail() == f"asked: run {tmp_path / 'case-1'}"


def test_replacing_earlier_results_is_asked_for_only_when_told_to(tmp_path, stand_in):
    replacing = CaseRun(tmp_path / "case", tmp_path / "a.txt", overwrite=True, command=stand_in)
    keeping = CaseRun(tmp_path / "case", tmp_path / "b.txt", command=stand_in)
    assert replacing.arguments[-1] == "--overwrite"
    assert "--overwrite" not in keeping.arguments
    replacing.start()
    _wait(replacing)
    assert replacing.status == CONVERGED and replacing.console_tail().endswith("--overwrite")


def test_stop_interrupts_the_run_so_it_can_record_how_it_ended(tmp_path, stand_in):
    run = CaseRun(tmp_path / "wait", tmp_path / "console.txt", command=stand_in)
    run.start()
    deadline = time.monotonic() + 20.0
    while "asked" not in run.console_tail() and time.monotonic() < deadline:
        time.sleep(0.05)
    started = time.monotonic()
    run.stop(grace=20.0)
    # Interrupted, not killed: it had the chance to say so and exit with its own status.
    assert run.stopped and not run.running
    assert run.console_tail().splitlines()[-1] == "interrupted"
    assert run.status == NOT_CONVERGED
    assert time.monotonic() - started < 10.0


def test_a_run_is_started_once(tmp_path, stand_in):
    run = CaseRun(tmp_path / "case", tmp_path / "console.txt", command=stand_in)
    run.start()
    with pytest.raises(RuntimeError, match="started already"):
        run.start()
    _wait(run)
