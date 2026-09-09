"""Every test that spawns a simulated-device interpreter carries the ``distributed`` marker.

Those tests are routed to a CI job of their own, because run alongside the rest of the tier they
oversubscribe the runner: each xdist worker forks a child that compiles a multi-device program and
wants the whole machine, so four workers and their four children share four cores. Under that
contention a test costing ~4 minutes alone has run past a 15-minute per-test timeout -- which kills
the worker and reports ``node down: Not properly terminated`` against it, with no mention of a
timeout anywhere, so the cause is not visible from the failure.

The routing is by marker, and a marker is applied by hand, so it can be forgotten. That failure is
silent in the direction that matters: a new distributed test without the marker does not fail, it
simply rejoins the contended job and makes the timeout kills more likely again, weeks after the
change that caused it. This census makes forgetting it fail here instead.
"""

from __future__ import annotations

from pathlib import Path

_TESTS = Path(__file__).resolve().parent.parent
_HELPER_MODULE = "tests.support.devices"
_MARKER = "pytestmark = pytest.mark.distributed"


def _modules_spawning_simulated_devices() -> list[Path]:
    """Return every test module that imports the simulated-device helper.

    Returns
    -------
    list of pathlib.Path
        Test modules importing from the helper, excluding the helper itself and this census.
    """
    here = Path(__file__).resolve()
    found = []
    for path in sorted(_TESTS.rglob("test_*.py")):
        if path.resolve() == here:
            continue
        if f"from {_HELPER_MODULE} import" in path.read_text():
            found.append(path)
    return found


def test_the_helper_is_still_the_way_these_tests_spawn_devices() -> None:
    """Guard the census's own premise: it finds modules by that import, so some must exist.

    If the helper is renamed or inlined, every check below passes vacuously -- a census that has
    stopped seeing anything looks exactly like a clean tree.
    """
    assert _modules_spawning_simulated_devices(), (
        f"no test module imports from {_HELPER_MODULE}, so this census is checking nothing. "
        "If the helper moved, point this file at its new home."
    )


def test_every_simulated_device_module_is_marked_distributed() -> None:
    """A module that spawns simulated devices must be routed to the distributed job."""
    unmarked = [
        str(p.relative_to(_TESTS))
        for p in _modules_spawning_simulated_devices()
        if _MARKER not in p.read_text()
    ]
    assert not unmarked, (
        "these modules spawn simulated-device interpreters but are not marked distributed, so "
        f"they run in the contended tier: {unmarked}. Add `{_MARKER}` at module level."
    )
