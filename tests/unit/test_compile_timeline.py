"""The compilation timeline harness reads a march log the way the logger and JAX write it.

``validation/pitzdaily_openfoam/compile_timeline.py`` is what the compile-cost measurements of a march
are taken with, so a misreading of the log is a wrong measurement rather than a crash. These lines are
the current step-table format (with its ``t(s)`` column) and JAX's compile-log lines, including a
program read from the persistent compilation cache.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HARNESS = Path(__file__).parents[2] / "validation" / "pitzdaily_openfoam" / "compile_timeline.py"


def _harness():
    spec = importlib.util.spec_from_file_location("compile_timeline", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOG = """\
Finished XLA compilation of jit(add) in 0.020000000 sec
Persistent compilation cache hit for 'jit__march_step' with key 'jit__march_step-0123abcd'
Finished XLA compilation of jit(_march_step) in 3.000000000 sec
    |     0 |    1 |  3.568e-02 |  7.531e-03 |  0.211 | 1.000 |
|    1 |    116 | 0.5000 |  3 |   3 | 3.180e-02 | 1.000 | 0.21 |     |
Finished XLA compilation of jit(batched_jacobian_matvec) in 7.000000000 sec
|    2 |    121 | 0.1667 |  3 |   3 | 2.648e-02 | 1.000 | 0.23 |     |
aquaflux coupled solve: 219s, Ux in [-2.693, 10.225]
Finished XLA compilation of jit(_field_values) in 0.500000000 sec
"""


def test_each_compilation_is_attributed_to_the_step_it_followed() -> None:
    phases, seconds, counts, compiles, _ = _harness().timeline(LOG.splitlines())
    assert phases == ["before step 1", "after step 1", "after the march"]
    assert counts == {"before step 1": 1, "after step 1": 1, "after the march": 1}
    assert seconds["after step 1"] == 7.0
    assert [name for _, _, name in compiles] == [
        "jit(add)",
        "jit(batched_jacobian_matvec)",
        "jit(_field_values)",
    ]


def test_a_program_read_from_the_persistent_cache_is_a_hit_not_a_compilation() -> None:
    _, seconds, counts, _, hits = _harness().timeline(LOG.splitlines())
    assert hits == {"before step 1": 1}
    # The 3 s logged for the cached program is its read, not a compilation.
    assert seconds["before step 1"] == 0.02
    assert counts["before step 1"] == 1
