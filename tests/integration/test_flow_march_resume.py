"""A dual-time march resumed with the shift its step control had reached.

Its own module, deliberately: the dual-time step is a large compile, and the fast tier clears JAX's
compiled programs only between modules. Run after the other march tests of one process it was the
compile that aborted the process (an XLA abort with no message), so it starts from a clean slate here.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import pytest
from aquaflux.flow import flow_march_step
from aquaflux.solve import DualTimeLoop, Resumption

from .test_channel_high_reynolds import _channel
from .test_flow_march import MU, _march_with_states


@pytest.fixture(scope="module")
def channel():
    return _channel(24, 16, MU)


@pytest.fixture(scope="module")
def dual_time_step(channel):
    """A dual-time step, run under the default Courant shift control, shared by the tests below."""
    assembler = channel
    return flow_march_step(
        assembler, assembler.initial_state(), dual_time=DualTimeLoop(inner_steps=3)
    )


def test_a_dual_time_march_resumed_with_its_shift_does_not_walk_the_ramp_again(
    channel, dual_time_step
) -> None:
    """A Courant ramp comes down step by step, and a resume must continue it from where it had got to.

    Measured without the shift (24 x 16 channel, ``mu = 5e-3``, ``DualTimeLoop(inner_steps=3)``, the
    default control, row-scaled tolerance 1e-9): a march resumed at its tenth step opened at the ramp's
    starting shift again and took 14 steps against the 7 the interrupted march had left. With the
    shift it opens at the shift the tenth step ran at.
    """
    assembler = channel
    fresh, states = _march_with_states(assembler, dual_time_step)
    resumed_at = 10
    last = fresh[resumed_at - 1]
    reference = float(last.residual_norm) / float(last.residual_ratio)
    start = states[resumed_at - 1]
    assert last.shift < 0.1 * fresh[0].shift  # the ramp has come a long way down by here

    without, _ = _march_with_states(
        assembler, dual_time_step, start, resume=Resumption(reference_residual=reference)
    )
    resumed, _ = _march_with_states(
        assembler,
        dual_time_step,
        start,
        resume=Resumption(reference_residual=reference, shift=last.shift),
    )

    assert without[0].shift == pytest.approx(fresh[0].shift)  # control: reopens at the start
    # The first step holds the shift it resumes from, since there is no previous step to adapt it from.
    assert resumed[0].shift == pytest.approx(last.shift)
    # The ramp then continues as the interrupted march's would have, a step behind for that hold.
    following = [r.shift for r in fresh[resumed_at : resumed_at + 2]]
    assert [r.shift for r in resumed[1:3]] == pytest.approx(following)
    assert len(resumed) <= len(fresh) - resumed_at + 2
    assert len(without) > len(resumed) + 3
