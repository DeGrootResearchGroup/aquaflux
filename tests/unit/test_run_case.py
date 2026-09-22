"""``validation/run_case.sh --wait`` must exit with the case's own status.

The script launches a case detached -- it has to survive its caller, and ``--wait`` on an
already-running case is not its parent -- so nothing ever collected the case's exit status, and both
``--wait`` forms exited 0 however the case ended. That was observed on a case that died of a
``ModuleNotFoundError`` and on one killed mid-script with no traceback at all: ``run_case.sh ... --wait
&& next-step`` went on to the next step after a crash. A launch wrapper now records the status beside
the log, and the waiters exit with it.

Like :mod:`tests.unit.test_fastgate`, this pins a runner's honesty: a waiter that had gone back to
reporting success for everything would look exactly like a string of successful cases.

Every case here points ``TMPDIR`` at a throwaway directory, so the machine-global run-file that real
cases use is never read or written, and passes ``--force`` so the memory and load pre-flight (which
judges the machine, not the script) cannot refuse a launch that the test is about.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

RUN_CASE = Path(__file__).resolve().parents[2] / "validation" / "run_case.sh"

#: A case that ends normally with a chosen non-zero status.
_EXITS_3 = "import sys\nprint('about to exit 3')\nsys.exit(3)\n"

#: A case that dies of an uncaught exception, the way the observed ``ModuleNotFoundError`` did.
_RAISES = "import no_such_module_anywhere  # noqa: F401\n"

#: A case killed by a signal it cannot handle, standing in for an out-of-memory kill: no traceback,
#: nothing in the log to say it died.
_KILLED = "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"

#: A case that runs until stopped, recording its own pid so a test can check it is really gone. It
#: shuts down on SIGTERM the way a case writing a checkpoint would: slowly, and with a status of its
#: own (7) that differs from the 143 a bare SIGTERM death gives -- so the status recorded can only be
#: right if the wrapper waited for the case to finish, rather than reporting the moment it forwarded.
_LONG = (
    "import os, pathlib, signal, sys, time\n"
    "def _shut_down(*_):\n"
    "    time.sleep(0.5)\n"
    "    sys.exit(7)\n"
    "signal.signal(signal.SIGTERM, _shut_down)\n"
    "pathlib.Path(sys.argv[0]).with_suffix('.pid').write_text(str(os.getpid()))\n"
    "time.sleep(120)\n"
)


def _script(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    """An isolated run-file and a fast poll."""
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir(exist_ok=True)
    return {**os.environ, "TMPDIR": str(tmpdir), "AQUAFLUX_CASE_POLL_SECONDS": "0.1"}


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUN_CASE), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
        timeout=120,
    )


def _launch_and_wait(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    return _run(tmp_path, str(_script(tmp_path, "case", source)), "--force", "--wait")


def _recorded_pid(result: subprocess.CompletedProcess[str]) -> int:
    for line in result.stdout.splitlines():
        if line.startswith("launched pid "):
            return int(line.removeprefix("launched pid "))
    raise AssertionError(f"no pid in the launch output:\n{result.stdout}")


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_case_that_EXITS_NON_ZERO_makes_wait_exit_with_that_status(tmp_path: Path) -> None:
    result = _launch_and_wait(tmp_path, _EXITS_3)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "exit status: 3" in result.stdout


def test_a_case_that_RAISES_makes_wait_exit_non_zero(tmp_path: Path) -> None:
    result = _launch_and_wait(tmp_path, _RAISES)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ModuleNotFoundError" in result.stdout  # the log's tail, shown by the waiter


def test_a_case_KILLED_BY_A_SIGNAL_exits_128_plus_N_and_says_so_in_its_log(tmp_path: Path) -> None:
    """An out-of-memory kill leaves no traceback; the status is the only record that it died."""
    result = _launch_and_wait(tmp_path, _KILLED)
    assert result.returncode == 128 + signal.SIGKILL, result.stdout + result.stderr
    (log,) = tmp_path.glob("run-*.log")
    assert "exited with status 137 (killed by signal 9)" in log.read_text()


def test_a_case_that_SUCCEEDS_still_exits_zero(tmp_path: Path) -> None:
    """The other direction: a waiter that reported failure for everything would also pass the above."""
    result = _launch_and_wait(tmp_path, "print('fine')\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exit status: 0" in result.stdout


def test_WAIT_ON_AN_ALREADY_RUNNING_case_exits_with_its_status(tmp_path: Path) -> None:
    """The second `--wait` form is not the case's parent, so it can only read what the wrapper wrote."""
    source = "import sys, time\ntime.sleep(1.5)\nsys.exit(3)\n"
    launched = _run(tmp_path, str(_script(tmp_path, "case", source)), "--force")
    assert launched.returncode == 0, launched.stdout + launched.stderr
    waited = _run(tmp_path, "--wait")
    assert "waiting on pid" in waited.stdout, "the case finished before the waiter attached"
    assert waited.returncode == 3, waited.stdout + waited.stderr


def _start_long_case(tmp_path: Path) -> tuple[int, int, subprocess.Popen[str]]:
    """Launch :data:`_LONG` and attach a `--wait` to it: (wrapper pid, case pid, waiter)."""
    script = _script(tmp_path, "case", _LONG)
    wrapper = _recorded_pid(_run(tmp_path, str(script), "--force"))
    pid_file = script.with_suffix(".pid")
    deadline = time.monotonic() + 30
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    case = int(pid_file.read_text())
    # Attach the waiter BEFORE anything is killed: attached after, it could find the run already over
    # and report "no case is running", which says nothing about the status.
    waiter = subprocess.Popen(
        [str(RUN_CASE), "--wait"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_environment(tmp_path),
    )
    assert waiter.stdout is not None
    assert waiter.stdout.readline().startswith(f"waiting on pid {wrapper}")
    return wrapper, case, waiter


def _wait_until_gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _gone(pid)


def test_killing_the_RECORDED_pid_stops_the_case_and_reports_ITS_status(tmp_path: Path) -> None:
    """The recorded pid is the wrapper's; `kill <pid>` must reach the case, not orphan it.

    Were the signal to kill only the wrapper, the case would run on unseen while `--status` reported
    nothing running -- the liveness test answering for a process that is no longer the case. And the
    status must be the one the case finished with (7), not the 143 of the interrupted `wait`.
    """
    wrapper, case, waiter = _start_long_case(tmp_path)
    os.kill(wrapper, signal.SIGTERM)
    output, _ = waiter.communicate(timeout=60)
    orphaned = not _wait_until_gone(case)
    if orphaned:
        os.kill(case, signal.SIGKILL)  # do not leave a failing run's case sleeping on the machine

    assert not orphaned, "the case outlived the pid the run-file names"
    assert waiter.returncode == 7, output


def test_a_run_with_NO_RECORDED_STATUS_is_reported_as_a_failure(tmp_path: Path) -> None:
    """If the wrapper itself dies, nothing is known about how the case ended -- which is not success."""
    wrapper, case, waiter = _start_long_case(tmp_path)
    try:
        os.kill(wrapper, signal.SIGKILL)
        output, _ = waiter.communicate(timeout=60)
    finally:
        os.kill(case, signal.SIGKILL)  # orphaned by the wrapper's death; nothing else will stop it

    assert waiter.returncode != 0, output
    assert "NO exit status was recorded" in output
