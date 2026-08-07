"""Periodic state checkpointing for a long march -- the persistence half of the ``on_checkpoint`` seam.

A multi-hour march that raises at its last step loses everything: the exception propagates out of the
solve, so a driver that writes the state only on a successful return has nothing to show for the hours
before the failure. That is the expensive case, not a rare one -- a solve that fails late is exactly the
solve worth restarting from, probing, or comparing against.

:class:`StateCheckpointer` plugs into the same ``on_checkpoint(report, state)`` callback the march
already offers, writes the state every ``every`` steps, and keeps the most recent ``keep`` files. So a
failure costs at most ``every`` steps of work, and disk usage is bounded no matter how long the run is.

Two deliberate safety properties, both of which matter more than they look for a job that may be killed:

- **A checkpoint is written to a temporary name and then renamed.** A process killed mid-write would
  otherwise leave a truncated file that reads as a checkpoint, and with a small ``keep`` that corrupt
  file could be the only one left.
- **Retention only ever deletes files this object wrote.** It does not glob the directory, so it cannot
  remove a checkpoint from another run, a hand-saved reference state, or anything else that happens to
  match the naming pattern.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .march import StepReport

__all__ = ["StateCheckpointer"]


def _save_array(path: Path, state: Any, report: StepReport) -> None:
    """Default serializer: the state as one array, with the step's own numbers beside it.

    The metadata is what makes a checkpoint self-describing -- a bare array file cannot say which step
    it came from or how converged it was, so a directory of them is unusable without the log.
    """
    # A file object, not the path: `np.savez` APPENDS ".npz" to any path that lacks it, which would
    # silently write somewhere other than where it was asked to -- and the staging name below does not
    # end in ".npz". The serializer contract is "write exactly to this path".
    with open(path, "wb") as handle:
        np.savez(
            handle,
            state=np.asarray(state),
            step=report.step,
            residual_norm=report.residual_norm,
            residual_ratio=report.residual_ratio,
            shift=report.shift,
            alpha=report.alpha,
            cycles=report.cycles,
        )


class StateCheckpointer:
    """Write the march state periodically, keeping a bounded number of recent files.

    Not an :class:`equinox.Module`: it holds mutable bookkeeping and touches the filesystem, so it is a
    host-side observer rather than a pytree the solve carries. **Forward-only** -- it materializes the
    state, so do not attach it to a differentiated solve.

    Parameters
    ----------
    directory : path-like
        Where checkpoints are written. Created if it does not exist.
    prefix : str
        Filename stem; files are ``<prefix>-<step>.npz``. Default ``"state"``. Note that a second run
        sharing a directory *and* prefix writes the same names and so **overwrites** the first run's
        checkpoints -- give each run its own directory (or prefix) when both are worth keeping.
    every : int
        Write on every ``every``-th observed step. ``1`` (the default) checkpoints every step, which
        bounds the loss from a failure at one step's work. Must be at least 1.
    keep : int
        How many recent checkpoints to retain; older ones **this object wrote** are deleted as new ones
        appear. Must be at least 1. Keep at least 2 if you intend to restart from a checkpoint while the
        run continues, so the file you are reading is never the one being replaced.
    save : callable, optional
        ``(path, state, report) -> None``, writing **exactly** to ``path``, overriding the default
        ``numpy`` serializer. The seam for a state that is a pytree rather than one array, or for a
        different file format.

    Attributes
    ----------
    latest : pathlib.Path or None
        The most recently written checkpoint, or ``None`` before the first write.

    Raises
    ------
    ValueError
        If ``every`` or ``keep`` is less than 1.

    Examples
    --------
    >>> checkpoints = StateCheckpointer("runs/bfs3d", every=5, keep=3)  # doctest: +SKIP
    >>> solve_coupled(coupled, on_checkpoint=checkpoints.on_checkpoint)  # doctest: +SKIP
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        prefix: str = "state",
        every: int = 1,
        keep: int = 3,
        save: Callable[[Path, Any, StepReport], None] | None = None,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}")
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._every = every
        self._save = _save_array if save is None else save
        # Only paths THIS object wrote, so retention can never delete another run's checkpoint or a
        # hand-saved reference state that happens to match the naming pattern.
        self._written: deque[Path] = deque(maxlen=keep)
        self._steps = 0

    @property
    def latest(self) -> Path | None:
        """The most recently written checkpoint, or ``None`` if nothing has been written yet."""
        return self._written[-1] if self._written else None

    def on_checkpoint(self, report: StepReport, state: Any) -> None:
        """``on_checkpoint`` callback: write this step's state if it falls on the interval."""
        self._steps += 1
        if self._steps % self._every:
            return
        path = self._directory / f"{self._prefix}-{self._steps:05d}.npz"
        # Write-then-rename: a process killed mid-write would otherwise leave a truncated file that
        # still reads as a checkpoint, and with a small `keep` it could be the only one left.
        staging = path.with_suffix(".npz.partial")
        self._save(staging, state, report)
        os.replace(staging, path)
        evicted = self._written[0] if len(self._written) == self._written.maxlen else None
        self._written.append(path)
        if evicted is not None and evicted != path:
            evicted.unlink(missing_ok=True)
