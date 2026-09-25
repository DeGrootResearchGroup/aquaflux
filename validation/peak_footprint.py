"""Run a command as a child process and read its peak memory footprint, without orphaning it.

Several harnesses re-execute themselves as a child under ``/usr/bin/time -l`` because the number
that decides their question is the child's peak memory *footprint* -- macOS's "peak memory
footprint", which counts the compressed pages a resident-set figure leaves out -- and a footprint
can only be read per process.

**Why this is not a bare ``subprocess.run``.** Stopping the parent with a plain ``kill <pid>`` (the
signal ``validation/run_case.sh`` forwards to the case it launched) kills only the parent: the
``/usr/bin/time`` child and the Python process under it are reparented to launchd and run on,
holding their whole working set, with nothing left that records them. So the child is started in a
session of its own, and a SIGTERM, SIGHUP or SIGINT reaching the parent is forwarded to that whole
process group -- which holds both ``/usr/bin/time`` and the Python beneath it. The parent then dies
by the same signal once the group is gone, so whoever sent it sees the exit status they asked for.
SIGINT is on the list because a session of its own also leaves the terminal's foreground group, so
a Ctrl-C would otherwise no longer reach the child at all.

A SIGKILL cannot be caught and still orphans the child; stop a run with ``kill <pid>``, not
``kill -9``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: The signals forwarded to the child's process group.
FORWARDED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)

_PEAK = re.compile(r"(\d+)\s+peak memory footprint")


@dataclass(frozen=True)
class Measured:
    """What a child run under :func:`run_with_footprint` returned.

    Attributes
    ----------
    returncode : int
        The child's exit status.
    peak_footprint_bytes : int or None
        Its peak memory footprint, or ``None`` if ``/usr/bin/time`` did not report one.
    stdout, stderr : str or None
        The child's output when captured, else ``None`` (it went to the parent's streams).
    """

    returncode: int
    peak_footprint_bytes: int | None
    stdout: str | None = None
    stderr: str | None = None

    @property
    def peak_footprint_gb(self) -> float | None:
        """The peak footprint in gigabytes (10^9 bytes), or ``None``."""
        return None if self.peak_footprint_bytes is None else self.peak_footprint_bytes / 1e9


def run_with_footprint(command: list[str], *, capture_output: bool = False) -> Measured:
    """Run ``command`` under ``/usr/bin/time -l`` and return its status and peak footprint.

    Parameters
    ----------
    command : list of str
        The program and its arguments, without the ``/usr/bin/time`` prefix.
    capture_output : bool
        Capture the child's stdout and stderr as text instead of letting them stream to the
        parent's. The footprint is read from a file of its own either way, so stderr is the child's
        alone.

    Returns
    -------
    Measured
        The exit status, the peak footprint, and the captured output if asked for.
    """
    received: list[int] = []
    child: subprocess.Popen | None = None

    def forward(signum: int, _frame: object) -> None:
        received.append(signum)
        if child is not None:
            _signal_group(child, signum)

    previous = {sig: signal.signal(sig, forward) for sig in FORWARDED}
    try:
        with tempfile.NamedTemporaryFile(suffix=".time", delete=False) as timing:
            record = Path(timing.name)
        try:
            pipe = subprocess.PIPE if capture_output else None
            child = subprocess.Popen(
                ["/usr/bin/time", "-l", "-o", str(record), *command],
                stdout=pipe,
                stderr=pipe,
                text=True,
                start_new_session=True,
            )
            if received:  # a signal landed before the child existed to forward it to
                _signal_group(child, received[0])
            try:
                stdout, stderr = child.communicate()
            finally:
                if child.poll() is None:  # the parent is leaving for some other reason
                    _signal_group(child, signal.SIGTERM)
                    child.wait()
            found = _PEAK.search(record.read_text())
        finally:
            record.unlink(missing_ok=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if received:
        # The child is gone; leave the way the sender asked, with the default action of its signal.
        signal.signal(received[0], signal.SIG_DFL)
        os.kill(os.getpid(), received[0])
    return Measured(child.returncode, int(found.group(1)) if found else None, stdout, stderr)


def _signal_group(child: subprocess.Popen, signum: int) -> None:
    """Send ``signum`` to the child's whole process group, if it still has one."""
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass
