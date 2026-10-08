"""One solver process, started once and asked every case-file command in turn.

Starting Python and importing the solver takes seconds -- most of it importing JAX -- while reading a
case file takes a fraction of a second. So rather than a new process per command, the page keeps one
``aquaflux serve`` process: it reads a request per line on its standard input and answers each with a
line on its standard output. The page still imports neither the solver nor JAX; it only talks to the
process.

Requests are answered one at a time, in the order asked. A request that takes longer than its time
limit stops the process, as does one the process dies during; the next request starts a new one.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import subprocess
import sys
import threading
from collections.abc import Sequence

__all__ = ["OWN_PROCESS_GROUP", "TIMEOUT", "CommandResult", "SolverWorker"]

#: Seconds a command may take before it is given up on: ``check`` and ``mesh`` read the case's mesh,
#: and a large one takes a while.
TIMEOUT = 600.0


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """What one command did.

    Attributes
    ----------
    status : int
        Its exit status: 0 for success.
    output : str
        What it printed on standard output.
    errors : str
        What it printed on standard error.
    """

    status: int
    output: str
    errors: str = ""


#: Starts the process in a process group of its own, so Ctrl-C in the terminal running the page reaches
#: the page alone; the process stops when the page's end closes its input.
OWN_PROCESS_GROUP = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32"
    else {"start_new_session": True}
)

#: Put on a process's reply queue when its standard output closes: it has exited.
_EXITED = None


class SolverWorker:
    """Runs case-file commands in one long-lived ``aquaflux serve`` process.

    Called as ``worker(arguments, stdin)``, it is the runner :class:`SolverCommands` takes.

    Parameters
    ----------
    command : sequence of str, optional
        What starts the process; unset, ``aquaflux serve`` run by this interpreter, so it is the
        solver installed beside the page.
    timeout : float, optional
        Seconds one request may take.
    """

    def __init__(self, command: Sequence[str] | None = None, timeout: float = TIMEOUT) -> None:
        self.command = (
            list(command) if command is not None else [sys.executable, "-m", "aquaflux", "serve"]
        )
        self.timeout = timeout
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue | None = None

    def start(self) -> None:
        """Start the process now, if it is not running, rather than at the first request."""
        with self._lock:
            self._running()

    def __call__(self, arguments: Sequence[str], stdin: str | None = None) -> CommandResult:
        """Run ``aquaflux <arguments>`` with ``stdin`` as its standard input, and return what it did."""
        request = json.dumps({"arguments": list(arguments), "stdin": stdin}) + "\n"
        with self._lock:
            try:
                process, replies = self._send(request)
            except OSError as error:
                self._stop()
                return CommandResult(1, "", f"the solver process could not be started: {error}")
            try:
                line = replies.get(timeout=self.timeout)
            except queue.Empty:
                self._stop()
                return CommandResult(
                    1, "", f"the solver took longer than {self.timeout:g} s and was stopped"
                )
            if line is _EXITED:
                status = process.wait()
                self._stop()
                return CommandResult(
                    1, "", f"the solver process exited with status {status} during the request"
                )
        reply = json.loads(line)
        return CommandResult(reply["status"], reply["output"], reply["errors"])

    def close(self) -> None:
        """Stop the process; a later request starts a new one."""
        with self._lock:
            self._stop()

    @property
    def pid(self) -> int | None:
        """The running process's identifier, or ``None`` when none is running."""
        return self._process.pid if self._process is not None else None

    def _send(self, request: str) -> tuple[subprocess.Popen, queue.Queue]:
        """Write ``request`` to a running process -- a new one if the last has gone."""
        for attempt in range(2):
            process, replies = self._running()
            try:
                process.stdin.write(request)
                process.stdin.flush()
                return process, replies
            except BrokenPipeError:
                # It exited between requests: start another and ask that one.
                self._stop()
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def _running(self) -> tuple[subprocess.Popen, queue.Queue]:
        """The process, started if it is not running."""
        if self._process is not None and self._process.poll() is not None:
            self._stop()
        if self._process is None:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                **OWN_PROCESS_GROUP,
            )
            replies: queue.Queue = queue.Queue()
            # Read on a thread of its own, so a reply can be waited for with a time limit on every
            # platform; each process gets its own queue, so a stopped one's lines are never read.
            threading.Thread(
                target=_read_lines, args=(process.stdout, replies), daemon=True
            ).start()
            self._process, self._replies = process, replies
        return self._process, self._replies

    def _stop(self) -> None:
        process, self._process, self._replies = self._process, None, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.close()  # it stops at the end of its input
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()


def _read_lines(stream, replies: queue.Queue) -> None:
    for line in stream:
        replies.put(line)
    replies.put(_EXITED)
