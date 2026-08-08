"""Unit tests for :class:`~aquaflux.solve.StateCheckpointer`.

Driven by synthetic :class:`~aquaflux.solve.StepReport`s against ``tmp_path``, so no solve is needed.
What is pinned is what a long run depends on: the interval, the bounded retention, that retention
cannot reach a file this object did not write, and that a checkpoint is self-describing.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.solve import StateCheckpointer, StepReport, combine_observers


def _report(**kwargs) -> StepReport:
    fields = dict(step=0, cycles=10, residual_norm=2.0e-2, residual_ratio=4.0e-2, alpha=1.0)
    return StepReport(**(fields | kwargs))


def _names(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_writes_on_the_interval_not_on_every_step(tmp_path) -> None:
    """The interval is what trades checkpoint cost against how much a failure can lose."""
    checkpoints = StateCheckpointer(tmp_path, every=3, keep=10)
    for _ in range(7):
        checkpoints.on_checkpoint(_report(), np.zeros(4))

    assert _names(tmp_path) == ["state-00003.npz", "state-00006.npz"]


def test_retention_bounds_disk_use_over_a_long_run(tmp_path) -> None:
    """A multi-hour march must not fill the disk with states nobody will read."""
    checkpoints = StateCheckpointer(tmp_path, keep=2)
    for _ in range(5):
        checkpoints.on_checkpoint(_report(), np.zeros(4))

    assert _names(tmp_path) == ["state-00004.npz", "state-00005.npz"]
    assert checkpoints.latest.name == "state-00005.npz"


def test_retention_never_deletes_a_file_it_did_not_write(tmp_path) -> None:
    """Retention tracks its own writes rather than globbing.

    Globbing would delete a previous run's checkpoints, or a hand-saved reference state, purely for
    matching the naming pattern -- and the whole point of this class is not losing expensive states.
    """
    (tmp_path / "state-09999.npz").write_text("a reference state from another run")
    checkpoints = StateCheckpointer(tmp_path, keep=1)
    for _ in range(3):
        checkpoints.on_checkpoint(_report(), np.zeros(4))

    assert "state-09999.npz" in _names(tmp_path)
    assert (tmp_path / "state-09999.npz").read_text().startswith("a reference")


def test_a_checkpoint_carries_the_step_that_produced_it(tmp_path) -> None:
    """A bare array cannot say which step it came from or how converged it was.

    Without the metadata a directory of checkpoints is unusable unless the log survived alongside it.
    """
    checkpoints = StateCheckpointer(tmp_path)
    checkpoints.on_checkpoint(_report(residual_norm=3.5e-4, shift=0.078), np.arange(4.0))

    saved = np.load(checkpoints.latest)
    assert np.array_equal(saved["state"], np.arange(4.0))
    assert saved["residual_norm"] == pytest.approx(3.5e-4)
    assert saved["shift"] == pytest.approx(0.078)


def test_no_partial_file_is_left_behind(tmp_path) -> None:
    """Written to a staging name then renamed, so a kill mid-write cannot leave a truncated file
    that still reads as a checkpoint -- which, with a small ``keep``, could be the only one left."""
    checkpoints = StateCheckpointer(tmp_path, keep=1)
    checkpoints.on_checkpoint(_report(), np.zeros(4))

    assert not list(tmp_path.glob("*.partial"))


def test_a_custom_serializer_handles_a_state_that_is_not_one_array(tmp_path) -> None:
    """The default assumes a single array; a pytree state injects its own writer."""
    written = []
    checkpoints = StateCheckpointer(
        tmp_path,
        save=lambda path, state, report: (path.write_text(str(state)), written.append(state)),
    )
    checkpoints.on_checkpoint(_report(), {"u": 1, "k": 2})

    assert written == [{"u": 1, "k": 2}]
    assert checkpoints.latest.exists()


@pytest.mark.parametrize("kwargs", [{"every": 0}, {"keep": 0}])
def test_a_meaningless_cadence_raises(tmp_path, kwargs) -> None:
    """``every=0`` would never checkpoint and ``keep=0`` would delete every write immediately -- both
    silently defeat the feature, so they fail at construction rather than at the end of a long run."""
    with pytest.raises(ValueError):
        StateCheckpointer(tmp_path, **kwargs)


def test_logging_and_checkpointing_share_the_one_callback(tmp_path) -> None:
    """The march takes a single ``on_checkpoint``; a run wants to log AND persist each step."""
    checkpoints = StateCheckpointer(tmp_path)
    logged = []
    observer = combine_observers(lambda r, s: logged.append(r.step), checkpoints.on_checkpoint)
    observer(_report(step=7), np.zeros(4))

    assert logged == [7]
    assert checkpoints.latest is not None


def test_inner_iterate_checkpointer_keeps_only_the_expensive_iterations(tmp_path):
    """A march that behaves should cost nothing; only the solves worth probing get written."""
    from aquaflux.solve import InnerIterateCheckpointer

    keeper = InnerIterateCheckpointer(tmp_path, above=5)
    # restart_cycles strips a +2 offset per solve, so raw 3 is one cycle and raw 17 is fifteen.
    keeper.on_inner(0, 1.0, 0.5, 3, 1.0, np.array([1.0, 2.0]))
    keeper.on_inner(1, 0.5, 0.4, 17, 0.01, np.array([3.0, 4.0]))
    assert len(keeper.written) == 1
    saved = np.load(keeper.written[0])
    assert saved["cycles"] == 15
    assert saved["inner"] == 1
    np.testing.assert_array_equal(saved["state"], [3.0, 4.0])


def test_inner_iterate_checkpointer_numbers_retry_attempts_apart(tmp_path):
    """A redone step restarts its inner loop, and the REJECTED attempt is usually the hard one."""
    from aquaflux.solve import InnerIterateCheckpointer

    keeper = InnerIterateCheckpointer(tmp_path, above=1)
    keeper.on_inner(0, 1.0, 1.0, 17, 0.0, np.array([1.0]))  # attempt 1, then the step is redone
    keeper.on_inner(0, 1.0, 0.2, 17, 1.0, np.array([2.0]))  # attempt 2 from the same step
    attempts = [int(np.load(path)["attempt"]) for path in keeper.written]
    assert attempts == [1, 2]
    assert len({path.name for path in keeper.written}) == 2  # distinct files, neither overwritten


def test_inner_iterate_checkpointer_rejects_a_meaningless_threshold(tmp_path):
    from aquaflux.solve import InnerIterateCheckpointer

    with pytest.raises(ValueError, match="above"):
        InnerIterateCheckpointer(tmp_path, above=0)
