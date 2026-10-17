"""What does an adaptive inexact-Newton tolerance buy the marches the residual-only stop slows or breaks?

``LinearSolveSettings(stop="residual")`` stops each Krylov solve as soon as its residual meets the
relative tolerance (0.3 on the coupled regimes), where lineax's stop over-solves. Forced as the default,
it failed three slow-tier tests that pass under lineax's stop, in two different ways:

* **a linear ending** -- the two channel marches of ``tests/integration/test_coupled_field_split.py``
  (20 x 14 graded turbulent channel, k-omega SST, default ``Convergence``: ``rtol`` 1e-10, ``atol``
  1e-12, ``max_steps`` 40 in the tests). Once Newton is otherwise exact, a step solved to 0.3 removes
  about 0.3 of the residual, so the end of the march converges linearly and needs more steps than the
  tests allow;
* **an erratic march** -- the Reynolds-continuation adjoint test of
  ``tests/integration/test_reynolds_continuation.py`` (28 x 20 channel, ``n_points=1``, ``max_steps``
  60), whose target march took accepted steps that raised the residual 30-fold and stalled.

This runs each of those marches forward under three arms -- lineax's stop, the residual stop, and the
residual stop with ``AdaptiveLinearTolerance()`` -- and prints steps, whether it converged, the final
residual, restart cycles and wall clock, plus each step's tolerance, line-search factor and residual
ratio for the adaptive arm. The budget is 60 steps everywhere, so a march the tests would fail at 40
still shows how far it is from converging.

Cases: ``block`` (the field-split test file's channel under ``BlockDiagonal(ScalarTwoLevel,
ConvectionTwoLevel)``), ``split`` (the same channel under the shipped ``MaterializedJacobian(FieldSplit(
SimpleSmoothed, JacobiSmoothed))``), ``reynolds`` (the continuation test's forward march at
``n_points=1``).

Run: ``validation/run_case.sh validation/adaptive_tolerance_probe.py`` (all cases), or
``python3 validation/adaptive_tolerance_probe.py [block|split|reynolds] ...``.

Recorded 2026-10-15 on the branch that built the setting (on main ``a5d61d4``): jax 0.11.2, CPU, Linux
x86_64, 4 cores, one run per arm. Outer steps (all marches of a case, both rungs for ``reynolds``) and
restart cycles; every arm converged except the one marked:

===========  ==============  =========================  ======================
case         lineax stop     residual stop              residual + adaptive
===========  ==============  =========================  ======================
block        27 / 19         41 / 41                    31 / 31
split        28 / 45         41 / 58                    30 / 52
reynolds     26 / 19         FAILED at 60 target steps  28 / 28 (16 + 12)
                             (``|R|`` 4.8e-2)
===========  ==============  =========================  ======================

Cycles are not comparable across the two stops (the residual stop counts the cycles it ran, ``lineax``'s
one fewer), and the wall clocks this prints include compilation, so they are not quoted.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
from aquaflux.initialization import hybrid_initialize
from aquaflux.solve import (
    AdaptiveLinearTolerance,
    LinearSolveSettings,
    restart_cycles,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CoupledRANS,
    ScalarTwoLevel,
    coupled_step,
    solve_coupled,
    solve_reynolds_continuation,
    sst_initial_fields,
)

MAX_STEPS = 60
ARMS = {
    "lineax stop": ("lineax", None),
    "residual stop": ("residual", None),
    "residual + adaptive": ("residual", AdaptiveLinearTolerance()),
}


def _field_split_channel():
    from tests.integration import test_coupled_field_split as split_tests

    momentum, turbulence = split_tests._channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    return coupled, split_tests


def _march_block(stop, adaptive, on_step):
    coupled, split_tests = _field_split_channel()
    flow, k, omega = sst_initial_fields(coupled.momentum, coupled.turbulence)
    solve_coupled(
        coupled,
        flow,
        k,
        omega,
        max_steps=MAX_STEPS,
        on_step=on_step,
        preconditioner=BlockDiagonal(scalar=ScalarTwoLevel(), **split_tests.PRECONDITIONER),
        linear_solve=LinearSolveSettings(stop=stop),
        adaptive_tolerance=adaptive,
    )


def _march_split(stop, adaptive, on_step):
    coupled, split_tests = _field_split_channel()
    flow, k, omega = sst_initial_fields(coupled.momentum, coupled.turbulence)
    step = coupled_step(
        coupled,
        coupled.pack_state(flow, k, omega),
        preconditioner=split_tests._shipped_split(),
        linear_solve=LinearSolveSettings(stop=stop),
    )
    solve_coupled(
        coupled,
        flow,
        k,
        omega,
        strategy=step,
        max_steps=MAX_STEPS,
        on_step=on_step,
        adaptive_tolerance=adaptive,
    )


def _march_reynolds(stop, adaptive, on_step):
    from tests.integration.test_coupled_rans import PRECONDITIONER, _channel

    _, momentum, turbulence, _, _ = _channel()
    channel = CoupledRANS.build(momentum, turbulence)
    preconditioner = BlockDiagonal(scalar=ScalarTwoLevel(), **PRECONDITIONER)
    step = coupled_step(
        channel,
        channel.pack_state(*hybrid_initialize(channel)),
        preconditioner=preconditioner,
        linear_solve=LinearSolveSettings(stop=stop),
    )
    solve_reynolds_continuation(
        channel,
        n_points=1,
        max_steps=MAX_STEPS,
        strategy=step,
        preconditioner=preconditioner,
        # The anchor rung builds its own step from the march keywords, so it takes the stop here; the
        # target runs on `step`, which already carries it.
        linear_solve=LinearSolveSettings(stop=stop),
        on_step=on_step,
        adaptive_tolerance=adaptive,
    )


CASES = {"block": _march_block, "split": _march_split, "reynolds": _march_reynolds}


def main(cases: list[str]) -> None:
    for case in cases:
        for arm, (stop, adaptive) in ARMS.items():
            reports = []
            started = time.perf_counter()
            try:
                CASES[case](stop, adaptive, reports.append)
                verdict = "converged"
            except Exception as error:  # a march that does not converge raises; record and go on
                verdict = f"FAILED ({type(error).__name__})"
            seconds = time.perf_counter() - started
            cycles = sum(restart_cycles(r.raw_cycles, r.inner_iterations) for r in reports)
            last = reports[-1].residual_norm if reports else math.nan
            print(
                f"{case:9s} {arm:20s} {len(reports):3d} steps  {verdict:24s} |R| {last:9.3e}  "
                f"{cycles:4d} cycles  {seconds:6.0f} s",
                flush=True,
            )
            if adaptive is not None:
                print("    step  tolerance  alpha   ratio", flush=True)
                for r in reports:
                    print(
                        f"    {r.step:4d}  {r.linear_tolerance:9.3g}  {r.alpha:5.3f}  "
                        f"{r.residual_ratio:9.3e}",
                        flush=True,
                    )


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
