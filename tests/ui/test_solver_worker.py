"""Tests for the long-lived solver process: one process for many requests, and recovery when it fails.

A stand-in for ``aquaflux serve`` -- a few lines of Python speaking the same one-line-per-request
protocol -- lets a test make the process hang or die on cue; one test runs the real one.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("trame")

from aquaflux_ui.solver_worker import CommandResult, SolverWorker

#: Answers ``echo`` with its standard input, ``pid`` with its process identifier; ``hang`` never
#: answers, and ``die`` exits mid-request.
STAND_IN = """
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    command = request["arguments"][0]
    if command == "hang":
        time.sleep(60)
    if command == "die":
        sys.exit(3)
    output = request["stdin"] if command == "echo" else str(os.getpid())
    print(json.dumps({"status": 0, "output": output, "errors": ""}), flush=True)
"""


@pytest.fixture
def worker():
    worker = SolverWorker([sys.executable, "-c", STAND_IN], timeout=2.0)
    yield worker
    worker.close()


def test_every_request_is_answered_by_the_same_process(worker):
    first, second = worker(["pid"], None), worker(["pid"], None)
    assert first.status == 0 and first.output == second.output == str(worker.pid)
    assert worker(["echo"], "line one\nline two") == CommandResult(0, "line one\nline two", "")


def test_a_request_that_takes_too_long_stops_the_process_and_the_next_starts_another(worker):
    before = worker(["pid"], None).output
    hung = worker(["hang"], None)
    assert hung.status != 0 and "longer than 2 s" in hung.errors
    assert worker.pid is None  # stopped, not left running
    assert worker(["pid"], None).output not in ("", before)


def test_a_process_that_dies_during_a_request_is_reported_and_replaced(worker):
    before = worker(["pid"], None).output
    died = worker(["die"], None)
    assert died.status != 0 and "exited with status 3" in died.errors
    assert worker(["pid"], None).output not in ("", before)


def test_a_process_that_exited_between_requests_is_restarted_without_a_failure(worker):
    before = worker(["pid"], None).output
    worker._process.kill()
    worker._process.wait()
    after = worker(["pid"], None)
    assert after.status == 0 and after.output != before


def test_a_process_that_cannot_be_started_says_why():
    result = SolverWorker(["/nonexistent/aquaflux-serve"])(["schema"], None)
    assert result.status != 0 and "could not be started" in result.errors


def test_the_real_solver_process_answers_and_keeps_its_replies_apart():
    worker = SolverWorker()
    try:
        schema, unserved = worker(["schema"], None), worker(["run", "case.yaml"], None)
    finally:
        worker.close()
    assert schema.status == 0 and '"root": "CaseSpec"' in schema.output
    assert unserved.status == 2 and "answers only" in unserved.errors


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are a POSIX notion")
def test_the_process_is_out_of_reach_of_the_terminals_ctrl_c(worker):
    """Ctrl-C signals the terminal's foreground process group; the solver process is not in it."""
    worker.start()
    assert os.getpgid(worker.pid) != os.getpgid(0)
