"""What damping ratio did a tapered march ACTUALLY apply, step by step?

A per-block damping ratio is a multiplier on the shift strength, and nothing in a march log reports
it. That is not a cosmetic gap: a taper whose signal never moves runs as a *constant*, which is a
different experiment from the one that was launched, and the log of the two is identical. This probe
closes that by replaying a run's checkpoints.

For each checkpointed state it reports the closure residual ``|R_turb|`` (the k and omega rows'
unscaled Euclidean norm -- the quantity :class:`~aquaflux.turbulence.ResidualTaperedDamping` keys on)
and the ratio that taper would have applied there. Read it against the step's ``beta`` from the march
log to see *when* the release happened relative to the phase the damping was meant for.

⚠️ **Only valid for a momentum-only viscosity ramp** (``PITZ_RAMP_SCALE=flow``). The k and omega rows
read the closure's own molecular viscosity, which a momentum-only station leaves untouched -- so the
same assembler evaluates them correctly at every station. Under ``both``-blocks scaling each station
has its own closure viscosity and this replay would evaluate the ramp's states under the wrong one.

⚠️ **Checkpoints are a rolling window** (``PITZ_CHECKPOINT_KEEP``) and the next march evicts them.
Copy a run's states under a stable directory name before measuring anything you intend to write down.

From the repository root, with no march running::

    TAPER_TRACE_DIR=checkpoints/taper-residual-g10 \
        validation/run_case.sh validation/pitzdaily_openfoam/damping_taper_trace.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.turbulence import hybrid_initialize, turbulence_residual_norm  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _DEFAULT_SHIFT_BASIS,
    _monolithic_shift_source,
    coupled_scaled_norm,
)

#: Directory of `state-000NN.npz` checkpoints to replay, relative to this case directory.
TRACE_DIR = CASE / os.environ.get("TAPER_TRACE_DIR", "checkpoints")

#: The initial ratio the taper opened at, so the reported factor is the one that march applied.
INITIAL = float(os.environ.get("TAPER_TRACE_INITIAL", "10"))

#: The taper's exponent, as the march ran it.
EXPONENT = float(os.environ.get("TAPER_TRACE_EXPONENT", "1"))


def main() -> int:
    coupled = compare.build_case()["coupled"]
    states = sorted(TRACE_DIR.glob("state-*.npz"))
    if not states:
        print(f"no checkpoints under {TRACE_DIR}")
        return 1

    # The reference the case takes: |R_turb| at the state the march opens from. The checkpointer
    # writes AFTER a step, so the first checkpoint is already one step in -- the true reference is the
    # hybrid start, rebuilt here the way `solve_reynolds_ramp` builds it.
    seed = coupled.state_from_physical(*hybrid_initialize(coupled.momentum, coupled.turbulence))
    # The row scales the march itself steers by, frozen at the seed so every step is reported in
    # ONE measure -- a per-state rebuild would mix progress with a change of measure.
    shift_policy = _monolithic_shift_source(coupled, seed, _DEFAULT_SHIFT_BASIS)
    reference = float(turbulence_residual_norm(coupled.layout, coupled.residual(seed)))
    print(f"reference |R_turb| at the hybrid seed: {reference:.4e}")
    print(f"initial ratio {INITIAL:g}, exponent {EXPONENT:g}, {len(states)} checkpoints\n")
    # `shift` and `cycles` ride in the checkpoint beside the state, so the ratio is reported against
    # the step's own beta and cost without cross-referencing a log that may have been rotated away.
    #
    # ⚠️ The checkpoint's OWN `step` is authoritative, not the file name: the directory is a rolling
    # window, so a shorter run leaves a longer one's high-numbered files in place and the names then
    # span two marches. Rows are ordered by the recorded step and a backwards jump is flagged.
    print(
        "| step |    beta | cyc |   |R_turb| |  share | gamma | R_turb/R_flow | scaled k+w/flow |"
    )
    print(
        "|------|---------|-----|------------|--------|-------|---------------|-----------------|"
    )
    previous = -1
    for path in states:
        with np.load(path) as data:
            state = jnp.asarray(data["state"])
            step, beta, cycles = int(data["step"]), float(data["shift"]), int(data["cycles"])
        residual = coupled.residual(state)
        norm = float(turbulence_residual_norm(coupled.layout, residual))
        share = min(1.0, max(0.0, norm / reference))
        gamma = 1.0 + (INITIAL - 1.0) * share**EXPONENT

        # Two candidate signals that are RATIOS rather than levels, so overall progress divides out
        # and what is left is how far the closure trails the mean flow -- which is the pathology
        # damping causes, and so the thing a release ought to key on.
        flow_rows, _k, _omega = coupled.layout.unpack(residual)
        flow_norm = float(jnp.linalg.norm(flow_rows))
        per_block = coupled_scaled_norm(coupled, shift_policy, state).per_block(residual)
        scaled_flow = float(jnp.linalg.norm(per_block[:-2]))
        scaled_closure = float(jnp.linalg.norm(per_block[-2:]))
        flag = "  <-- step went BACKWARDS (a different run's file)" if step <= previous else ""
        previous = step
        print(
            f"| {step:4d} | {beta:7.4f} | {cycles:3d} | {norm:10.4e} | {share:6.4f} | {gamma:5.2f} "
            f"| {norm / max(flow_norm, 1e-300):13.4e} "
            f"| {scaled_closure / max(scaled_flow, 1e-300):15.4f} |{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
