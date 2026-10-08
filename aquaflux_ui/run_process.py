"""A case being solved: one ``aquaflux run`` process, started, watched and stopped from the page.

The solve runs in a process of its own -- this package does not import the solver -- with the same
Python interpreter as the page, so it is the solver installed beside it. What it prints, its log and
any refusal, goes to a file the page reads back; how the march is going is read from the history it
writes, a row per step.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .solver_worker import OWN_PROCESS_GROUP

__all__ = ["CONVERGED", "NOT_CONVERGED", "REFUSED", "STOP_GRACE", "CaseRun"]

#: ``aquaflux run``'s exit statuses: converged, stopped short of its stopping test, file refused.
CONVERGED, NOT_CONVERGED, REFUSED = 0, 1, 2

#: Seconds a stopped run is given to close its files before it is killed. An interrupt lands only when
#: the solve returns to Python between compiled steps, which on a large case is tens of seconds.
STOP_GRACE = 60.0


class CaseRun:
    """One solve of a case file, in a process of its own.

    Parameters
    ----------
    case : path-like
        The case file.
    console : path-like
        Where what the run prints is written: its log as it goes, and the reason if it is refused.
    overwrite : bool
        Replace an earlier run's results; without it a case whose output directory holds results is
        refused, as ``aquaflux run`` refuses it.
    command : sequence of str, optional
        The command before the case file's arguments; unset, ``python -m aquaflux`` with this
        interpreter. A test passes another program.
    """

    def __init__(
        self,
        case: str | os.PathLike[str],
        console: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        command: Sequence[str] | None = None,
    ) -> None:
        self.case = Path(case)
        self.console = Path(console)
        base = list(command) if command is not None else [sys.executable, "-m", "aquaflux"]
        self.arguments = [*base, "run", str(self.case), *(["--overwrite"] if overwrite else [])]
        self._process: subprocess.Popen | None = None
        self.stopped = False

    def start(self) -> None:
        """Start the solve.

        Raises
        ------
        RuntimeError
            If it was started already.
        """
        if self._process is not None:
            raise RuntimeError("this run was started already.")
        self.console.parent.mkdir(parents=True, exist_ok=True)
        with self.console.open("w", encoding="utf-8") as console:
            # Unbuffered, so the console reads line by line as the run prints.
            self._process = subprocess.Popen(
                self.arguments,
                stdout=console,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                **OWN_PROCESS_GROUP,
            )

    @property
    def running(self) -> bool:
        """Whether it has been started and has not ended."""
        return self._process is not None and self._process.poll() is None

    @property
    def status(self) -> int | None:
        """Its exit status once it has ended (see :data:`CONVERGED`), else ``None``."""
        return None if self._process is None else self._process.poll()

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Interrupt the solve, as Ctrl-C would, and kill it if it has not ended within ``grace``."""
        if not self.running:
            return
        self.stopped = True
        process = self._process
        try:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=grace)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()

    def console_tail(self, lines: int = 200) -> str:
        """The last ``lines`` lines it printed."""
        try:
            text = self.console.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return "\n".join(text.splitlines()[-lines:])
