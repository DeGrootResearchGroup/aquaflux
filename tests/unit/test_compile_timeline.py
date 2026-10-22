"""The compilation timeline harness reads a march log the way the logger and JAX write it.

``validation/pitzdaily_openfoam/compile_timeline.py`` is what the compile-cost measurements of a march
are taken with, so a misreading of the log is a wrong measurement rather than a crash. These lines are
the current step-table format (with its ``t(s)`` column) and JAX's compile-log lines, including a
program read from the persistent compilation cache.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HARNESS = Path(__file__).parents[2] / "validation" / "pitzdaily_openfoam" / "compile_timeline.py"


def _harness():
    spec = importlib.util.spec_from_file_location("compile_timeline", HARNESS)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: a dataclass looks its module up by name while it is being defined.
    sys.modules[spec.name] = module
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


STEPPED_LOG = """\
Finished tracing _march_step for jit in 4.000000000 sec
Finished jaxpr to MLIR module conversion jit(_march_step) in 2.000000000 sec
Persistent compilation cache hit for 'jit__march_step' with key 'jit__march_step-0123abcd'
Finished XLA compilation of jit(_march_step) in 1.500000000 sec
|    1 |     61 | 0.5000 |  3 |   3 | 3.180e-02 | 1.000 | 0.21 |     |
| pc full 4.3s (probe 2.2 assemble 0.2 refactor 1.9)                 |
Finished tracing norm_of_residual for jit in 0.050000000 sec
Finished jaxpr to MLIR module conversion jit(norm_of_residual) in 0.040000000 sec
Finished XLA compilation of jit(norm_of_residual) in 0.300000000 sec
    |     0 |    1 |  2.574e-02 |  1.132e-02 |  0.440 | 1.000 |
Finished XLA compilation of jit(add) in 0.020000000 sec
  -> redo step 2 (attempt 2): cycles, beta 0.1667 unchanged
Finished XLA compilation of jit(_padded_product) in 0.400000000 sec
    |     0 |    1 |  2.574e-02 |  1.132e-02 |  0.440 | 1.000 |
|    2 |    107 | 0.1667 |  3 |   3 | 2.648e-02 | 1.000 | 0.23 |     |
| pc full cost 2x 4.1s (probe 1.8 assemble 0.3 refactor 2.0)         |
"""


def test_the_breakdown_splits_a_step_where_its_own_work_and_each_redo_begin() -> None:
    phases = {phase.name: phase for phase in _harness().breakdown(STEPPED_LOG.splitlines())}
    assert list(phases) == [
        "before step 1",
        "step 2 set-up",
        "step 2 inner",
        "step 2 redo 2 (cycles)",
    ]
    assert phases["step 2 set-up"].compiled == [1, 0.3]
    assert phases["step 2 inner"].compiled == [1, 0.02]
    assert phases["step 2 redo 2 (cycles)"].compiled == [1, 0.4]
    # The refresh a step row reports is attached to every phase of that step.
    assert {phases[name].refresh for name in phases if name.startswith("step 2")} == {
        "pc full cost 2x"
    }
    assert phases["before step 1"].refresh == "pc full"


def test_the_breakdown_counts_tracing_and_lowering_paid_on_a_cache_hit() -> None:
    before = _harness().breakdown(STEPPED_LOG.splitlines())[0]
    assert before.hits == [1, 1.5]
    assert before.compiled == [0, 0.0]
    # The persistent cache skips only the XLA compilation: the hit's program was still traced and lowered.
    assert before.traced == [1, 4.0]
    assert before.lowered == [1, 2.0]
    assert [kind for _, kind, _ in sorted(before.large, reverse=True)] == ["trace", "lower", "hit"]
