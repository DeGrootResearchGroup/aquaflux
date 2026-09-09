"""The simulated-device helper gives the child its devices *without* dropping the ambient flags.

``XLA_FLAGS`` is a single string holding several entries, so the only way to add the device-count
entry is to append to whatever is already there. A child that assigns it instead still gets its
devices and still passes every assertion it makes -- what it loses is the *other* entry, the test
workflow's ``--xla_cpu_multi_thread_eigen=false``, and the cost of that lands on a different process
entirely: the child starts a thread pool over every core, and a co-scheduled test worker is starved
until it is killed. Nothing about that failure names the child, so the appending behaviour is pinned
here rather than left to be re-derived from a red build.

The last case covers the other half of the contract: the helper's ``ok`` sentinel, which is what
stops a child that quietly did nothing from reading as a pass.
"""

from __future__ import annotations

import pytest

from tests.support.devices import run_on_simulated_devices

# Every test in this module spawns a fresh interpreter that compiles a multi-device program,
# so the marker collects them into a CI job of their own. Run alongside the rest of the tier
# they oversubscribe the runner -- each worker forks a child that wants the whole machine --
# and a test that costs ~4 minutes on its own has exceeded a 15-minute per-test timeout there.
pytestmark = pytest.mark.distributed

# One child does both checks: importing JAX is the expensive part, so it is paid once.
_DEVICES_AND_FLAGS = r"""
import os
ambient = os.environ["XLA_FLAGS"]
assert "--xla_force_host_platform_device_count=3" in ambient, ambient
assert "--xla_cpu_multi_thread_eigen=false" in ambient, ("ambient entry dropped", ambient)
import jax
assert jax.device_count() == 3, jax.device_count()
print("ok")
"""


def test_the_child_gets_its_devices_and_keeps_the_ambient_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device entry is appended to the inherited ``XLA_FLAGS``, not written over it."""
    # The entry the test workflow sets for the whole job, and the one a child must not drop.
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    run_on_simulated_devices(_DEVICES_AND_FLAGS, count=3)


def test_a_child_that_never_reports_ok_is_a_failure() -> None:
    """A child can exit zero having checked nothing; the sentinel is what catches that."""
    with pytest.raises(AssertionError):
        run_on_simulated_devices('print("done")\n', count=1)
