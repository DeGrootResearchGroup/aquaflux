"""Run a snippet of test source in a fresh interpreter that sees several simulated CPU devices.

JAX fixes its device list when it initializes, so a test of sharded (multi-device) code cannot ask
for more devices from inside the test process -- the count comes from an ``XLA_FLAGS`` entry that
must already be set when JAX is first imported. Every such test therefore hands its body to a fresh
interpreter, which is what :func:`run_on_simulated_devices` does: it prepends the environment
preamble, runs the source, and asserts the child both exited cleanly and printed ``ok`` last.

This lives in one place because the preamble carries a trap that is invisible at the call site, and
that was for three days fixed in one copy of it and not the other four. The test workflow sets
``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`` for the whole job so that its parallel workers scale: without it every JAX process starts a thread
pool over all cores, and N workers x M cores thrash instead of getting a core each. A child that
*assigns* ``XLA_FLAGS`` silently drops that flag, and the child is spawned from inside a worker, so
the process that loses it is exactly the one the flag was set for. The preamble below therefore
**appends**. The symptom of getting it wrong is not a failing assertion but a co-scheduled worker
being killed -- pytest reports ``node down: Not properly terminated`` against whichever test that
worker happened to be running, which is usually not the test that dropped the flag.
"""

from __future__ import annotations

import subprocess
import sys

_PREAMBLE = """\
import os
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count={count}"
).strip()
"""


def simulated_device_preamble(count: int = 4) -> str:
    """Return source that gives a fresh interpreter ``count`` simulated CPU devices.

    Parameters
    ----------
    count : int, optional
        Number of simulated devices the child should see.

    Returns
    -------
    str
        Python source to place ahead of any ``jax`` import in the child.
    """
    return _PREAMBLE.format(count=count)


def run_on_simulated_devices(source: str, count: int = 4) -> None:
    """Run ``source`` in a fresh interpreter with ``count`` simulated CPU devices.

    The child is expected to end by printing ``ok``, so a body that returns early -- or that
    prints a diagnostic and exits zero -- fails here rather than passing silently.

    Parameters
    ----------
    source : str
        Python source for the child. The device preamble is prepended, so the source must not
        set ``XLA_FLAGS`` itself.
    count : int, optional
        Number of simulated devices the child should see.

    Raises
    ------
    AssertionError
        If the child exits non-zero (its ``stderr`` is the message) or its last line is not ``ok``.
    """
    result = subprocess.run(
        [sys.executable, "-c", simulated_device_preamble(count) + source],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
