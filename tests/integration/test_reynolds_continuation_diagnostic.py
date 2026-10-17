"""TEMPORARY CI diagnostic: which march default breaks the ramped Reynolds-continuation gradient.

``test_adjoint_matches_a_direct_solve_and_is_point_count_independent`` fails on the hosted CI runners
since the residual stop and the adaptive linear tolerance became the march defaults, and passes on
other machines with identical package versions. Each arm below runs that test's failing call -- the
gradient through a one-point continuation -- with one combination of the two defaults switched off.
An arm that fails re-runs the same march forward with a step observer and writes its trajectory to
the job log, so the log shows how the march stalled, not only that it did.

Delete this module once the cause is fixed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import LinearSolveSettings
from aquaflux.turbulence import (
    BlockDiagonal,
    ScalarTwoLevel,
    coupled_step,
    solve_reynolds_continuation,
)
from aquaflux.turbulence.coupled import CoupledRANS

from tests.integration.test_coupled_rans import PRECONDITIONER, _channel
from tests.integration.test_reynolds_continuation import MAX_STEPS, _hybrid

# Each arm: the keywords it adds to the march. `adaptive_tolerance=None` turns the adaptive tolerance
# off; `stop="lineax"` restores lineax's own GMRES stopping rule. Both off is the march as it ran before
# the two became defaults.
ARMS = {
    "defaults": {},
    "no_adaptive_tolerance": {"adaptive_tolerance": None},
    "lineax_stop": {"linear_solve": LinearSolveSettings(stop="lineax")},
    "neither": {
        "adaptive_tolerance": None,
        "linear_solve": LinearSolveSettings(stop="lineax"),
    },
}


@pytest.fixture(scope="module")
def channel():
    _, momentum, turbulence, _, _ = _channel()
    return CoupledRANS.build(momentum, turbulence)


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_ramped_gradient_converges(channel, arm) -> None:
    extra = ARMS[arm]
    step_keywords = {k: v for k, v in extra.items() if k == "linear_solve"}
    continuation = coupled_step(
        channel,
        channel.pack_state(*_hybrid(channel)),
        preconditioner=BlockDiagonal(scalar=ScalarTwoLevel(), **PRECONDITIONER),
        **step_keywords,
    )

    def solve(nu_scale, **observe):
        return solve_reynolds_continuation(
            channel.with_scaled_molecular_viscosity(nu_scale),
            n_points=1,
            max_steps=MAX_STEPS,
            strategy=continuation,
            preconditioner=BlockDiagonal(scalar=ScalarTwoLevel(), **PRECONDITIONER),
            **extra,
            **observe,
        )

    def objective(nu_scale):
        _, k, _ = solve(nu_scale)
        return jnp.sum(k**2)

    # The job runs under xdist, which drops a worker's raw output, so each arm's record goes out as a
    # warning (listed in the warnings summary, pass or fail) and as captured stdout (shown on failure).
    lines = [f"[diagnostic] arm={arm} cpu={_cpu()}"]

    def publish() -> None:
        text = "\n".join(lines)
        print(text)
        warnings.warn(text, UserWarning, stacklevel=1)

    try:
        gradient = float(jax.grad(objective)(1.0))
    except RuntimeError as failure:
        lines.append(f"  FAILED: {str(failure)[:300]}")
        lines.append("  forward trajectory of the same march:")

        def report(r):
            lines.append(
                f"  step {r.step:3d}  |R| {r.residual_norm:.3e}  alpha {r.alpha:.3f}"
                f"  shift {r.shift:.3e}  cycles {r.raw_cycles:3d}  tol {r.linear_tolerance:.2e}"
                f"  stop {r.stopping_norm:.3e}"
            )

        try:
            solve(1.0, on_step=report)
        except RuntimeError as exc:
            lines.append(f"  forward re-run also failed: {str(exc)[:300]}")
        else:
            lines.append("  forward re-run CONVERGED (the failure is specific to the traced call)")
        publish()
        raise
    lines.append(f"  converged, gradient {gradient:.10e}")
    publish()
