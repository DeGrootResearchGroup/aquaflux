"""Where a pitzDaily march spends its wall clock, read from a ``py-spy`` capture of the whole process.

Record the case's Python process with ``py-spy record --pid <pid> -r 25 --idle --threads --format raw
-o <file>`` (``--idle`` so a main thread waiting on a compiled program is counted: that wait is the
compiled program's time). Each line of the capture is one collapsed stack, root first, and the number of
samples it took. This sorts the main thread's samples into the buckets the open-directions record uses,
by the FIRST rule (in the order below) any frame of the stack satisfies, so a stack compiling a program
called from inside a refit counts as compilation, and one executing a step program counts as the step:

1. ``compilation and tracing`` -- any tracing, lowering or compilation frame of JAX or equinox;
2. ``refit: coloured probe`` -- the batched JVPs that compute a refit's Jacobian values;
3. ``refit: hierarchy and other build`` -- anything else under the preconditioner's build or refresh;
4. ``the compiled step program`` -- the march's step, executing;
5. ``the stopping test's residual`` and ``march-log field extraction`` -- the two programs read outside
   the step;
6. ``the step's measure rebuilt`` -- the row-scaled measure built again from each step's state;
7. ``driver, setup, report`` -- everything else on the main thread.

Usage
-----
    python validation/pitzdaily_openfoam/profile_buckets.py <capture.txt> [--rate 25] [--show-other 15]

``--show-other`` prints the commonest innermost frames of the last bucket, which is how a rule that has
stopped matching (a renamed function) shows up instead of quietly moving time into ``driver``.
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

MARCH = Path(__file__).resolve().parents[2] / "aquaflux" / "solve" / "march.py"


def _step_call_lines() -> str:
    """The lines of ``newton_march``'s call of the compiled step, as a regex alternation.

    The step is a ``filter_jit`` program, so while it executes the innermost Python frame of the march is
    ``newton_march`` at that call, with nothing of the march below it; the call is the only way to see it.
    Python reports a multi-line call at any of its lines, so all of them are taken.
    """
    lines = MARCH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "= _march_step(" in line)
    end = next(i for i in range(start, len(lines)) if lines[i].strip().endswith(")"))
    return "|".join(str(n + 1) for n in range(start, end + 1))


#: Each bucket and the frame patterns that put a stack in it, matched against ``"function (file:line)"``.
RULES = (
    (
        "compilation and tracing",
        re.compile(
            r"trace_to_jaxpr|_trace_for_jit|_infer_params|jit_trace|eval_shape \(|lower_jaxpr|"
            r"_cached_lowering|lower_sharding_computation|compile_or_get_cached|backend_compile|"
            r"_cached_compilation|from_hlo|_pjit_lower|\blower \(|\bcompile \(jax"
        ),
    ),
    ("refit: coloured probe", re.compile(r"sparse_jacobian\.py|batched_jacobian_matvec|probe")),
    (
        "refit: hierarchy and other build",
        re.compile(r"refresh_preconditioner|aquaflux/precondition/|scipy/sparse"),
    ),
    (
        "the compiled step program",
        re.compile(rf"newton_march \(aquaflux/solve/march\.py:({_step_call_lines()})\)$"),
    ),
    ("the stopping test's residual", re.compile(r"norm_of_residual|^judged \(")),
    ("march-log field extraction", re.compile(r"coupled_fields|march_log\.py|diagnostics\.py")),
    ("the step's measure rebuilt (row scales)", re.compile(r"row_scaled|coupled_scaled_norm")),
)
OTHER = "driver, setup, report"


def main_thread_stacks(path: str):
    """``(frames, samples)`` for each of the capture's main-thread stacks."""
    # `py-spy --nonblocking` can read a frame name mid-write, leaving bytes that are not UTF-8.
    with open(path, errors="replace") as capture:
        for line in capture:
            stack, _, count = line.rstrip("\n").rpartition(" ")
            if not count.isdigit():
                continue
            frames = stack.split(";")
            # With --threads the root frame names the thread: `thread (0x...)` or `process ...`.
            if not any("MainThread" in f for f in frames[:2]):
                continue
            yield frames, int(count)


def classify(frames) -> str:
    for name, pattern in RULES:
        if any(pattern.search(frame) for frame in frames):
            return name
    return OTHER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--show-other", type=int, default=0)
    args = parser.parse_args()
    totals = collections.Counter()
    leaves = collections.Counter()
    for frames, count in main_thread_stacks(args.capture):
        bucket = classify(frames)
        totals[bucket] += count
        if bucket == OTHER:
            leaves[";".join(frames[-3:])] += count
    total = sum(totals.values())
    if not total:
        raise SystemExit(f"{args.capture}: no main-thread samples")
    print(f"{total} main-thread samples = {total / args.rate:.0f} s at {args.rate:g} Hz")
    print("| bucket | s | share |\n|---|---|---|")
    for name in [r[0] for r in RULES] + [OTHER]:
        print(f"| {name} | {totals[name] / args.rate:.0f} | {totals[name] / total:.0%} |")
    for stack, count in leaves.most_common(args.show_other):
        print(f"{count / args.rate:7.1f} s  {stack}")


if __name__ == "__main__":
    main()
