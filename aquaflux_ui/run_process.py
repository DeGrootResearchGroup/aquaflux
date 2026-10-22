"""A long solver command -- a case being solved, a mesh being made -- in a process started, watched
and stopped from the page.

The command runs in a process of its own -- this package does not import the solver -- with the same
Python interpreter as the page, so it is the solver installed beside it. What it prints goes to a
file the page reads back: a solve's log and any refusal, a meshing's stages as they start. How a
march is going is read from the history it writes, a row per step.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .solver_worker import OWN_PROCESS_GROUP

__all__ = ["CONVERGED", "NOT_CONVERGED", "REFUSED", "STOP_GRACE", "SolverRun"]

#: ``aquaflux run``'s exit statuses: converged, stopped short of its stopping test, file refused.
CONVERGED, NOT_CONVERGED, REFUSED = 0, 1, 2

#: Seconds a stopped run is given to close its files before it is killed. An interrupt lands only when
#: the solve returns to Python between compiled steps, which on a large case is tens of seconds.
STOP_GRACE = 60.0


class SolverRun:
    """One ``aquaflux <subcommand> <file>`` -- a solve, a meshing -- in a process of its own.

    Parameters
    ----------
    subcommand : str
        ``run`` to solve a case file, ``mesh`` to make a mesh file's mesh.
    file : path-like
        The case file, or the mesh file.
    console : path-like
        Where what it prints is written as it goes, and the reason if it is refused.
    overwrite : bool
        Replace what an earlier one wrote; without it, a run whose output directory holds results,
        or a meshing whose mesh is already there, is refused, as the command refuses it.
    command : sequence of str, optional
        The command before the subcommand; unset, ``python -m aquaflux`` with this interpreter. A
        test passes another program.
    """

    def __init__(
        self,
        subcommand: str,
        file: str | os.PathLike[str],
        console: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        command: Sequence[str] | None = None,
    ) -> None:
        self.file = Path(file)
        self.console = Path(console)
        base = list(command) if command is not None else [sys.executable, "-m", "aquaflux"]
        self.arguments = [
            *base, subcommand, str(self.file), *(["--overwrite"] if overwrite else [])
        ]  # fmt: skip
        self._process: subprocess.Popen | None = None
        self.stopped = False

    def start(self) -> None:
        """Start the command.

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
        """Its exit status once it has ended (for a solve, see :data:`CONVERGED`), else ``None``."""
        return None if self._process is None else self._process.poll()

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Interrupt it, as Ctrl-C would, and kill it if it has not ended within ``grace``."""
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
