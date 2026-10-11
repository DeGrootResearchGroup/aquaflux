"""Does a viscosity ramp that ENDS ON THE SETTLE SIGNAL need a station count? A prototype, one arm per run.

The ``grow`` x ``stations`` sweep (2026-10-11, the release on) found that a ramp's best length is not a
property of the viscosity step: the line search's ``alpha`` history was identical at 1.33x and 2x per
step, the march spends a near-fixed 6-9 steps at the shift floor ``beta_min`` before its first
full-length step there, and the cheapest ramp arrives one step after that. That suggests a ramp with no
station count at all -- walk the viscosity down one geometric station per outer step, and move to the
case's own viscosity on the first full-length step taken at the floor, the same signal the step
control's ``release_floor`` already keys on at the target.

This is the measurement before the library is changed. ``ResidualHomotopy`` is keyed on the step index
alone and never sees a ``StepReport``, so the prototype stands in for ``ViscosityRampHomotopy`` (the
class ``solve_reynolds_ramp`` constructs is replaced for this process only) and hears each step through
the case's ``on_checkpoint`` seam, which runs after every accepted step and before the next one enters
its station. Nothing else about the case's march changes: the ramp's per-station viscosity, its
re-pointing of the preconditioner and its anchor are the parent class's.

What it measures. ``PITZ_RAMP_STATIONS`` is no longer the ramp's length, only its pace -- the anchor
walked down in that many stations if the settle never came -- so two runs at different values ask
whether the pace still matters once the end is the settle. The larger the count, the larger the
viscosity jump the settle takes at the floor; that jump is the unmeasured risk.

Pre-registered (``.claude/notes/solve-open-directions.md`` entry 12): against the shipped
``grow`` 3 x 12 stations (111 restart cycles, 16 steps, ``x_r/h`` 8.07), a pass is the same cycles within
a few at the same ``x_r/h``, with the ramp ending by itself.

Usage
-----
    PITZ_RAMP_STATIONS=16 validation/run_case.sh validation/pitzdaily_openfoam/settle_ramp_probe.py
    PITZ_RAMP_STATIONS=24 validation/run_case.sh validation/pitzdaily_openfoam/settle_ramp_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import aquaflux.turbulence.reynolds as reynolds  # noqa: E402
import compare  # noqa: E402


class SettleEndedRamp(reynolds.ViscosityRampHomotopy):
    """A viscosity ramp at one station per step that moves to the target once the march has settled.

    "Settled" is the first step that was not on the target, ran at or below the control's ``beta_min``
    and took a full-length step (``alpha >= release_alpha``) -- the gate ``release_floor`` applies on
    the target, read here from the same report fields. Until then station ``s`` runs at step ``s``, as the
    parent's does at one step per station; from the next step on, every step runs the target. A ramp
    whose pace reaches the target before the settle simply arrives, as the parent's does.

    The station each step ran is recorded when the march first asks for it, so the march's look-backs
    (``station(step - 1)``, ``station(step - 2)``) read what that step actually ran.
    """

    #: The live instance, so the probe's step observer can reach the one ``solve_reynolds_ramp`` built.
    active: SettleEndedRamp | None = None

    def __init__(self, *args, beta_min: float, release_alpha: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.steps_per_station != 1:
            raise ValueError("the prototype walks one station per step; set PITZ_RAMP_STEPS=1")
        self.beta_min = float(beta_min)
        self.release_alpha = float(release_alpha)
        self.settled_after: int | None = None
        self._ran: dict[int, int] = {}
        SettleEndedRamp.active = self

    def station(self, step: int) -> int:
        if step not in self._ran:
            ended = self.settled_after is not None and step > self.settled_after
            self._ran[step] = self.stations if ended else min(step, self.stations)
        return self._ran[step]

    def observe(self, report, state) -> None:
        """``on_checkpoint``: mark the settle on the first full-length step at the floor off the target."""
        del state
        if self.settled_after is not None or report.arrived:
            return
        if float(report.shift) <= self.beta_min and float(report.alpha) >= self.release_alpha:
            self.settled_after = int(report.step)
            remaining = self.scale(self._ran[int(report.step)])
            print(
                f"[settle] step {int(report.step) + 1} was a full-length step at the floor "
                f"(shift {float(report.shift):.4g}, alpha {float(report.alpha):.3f}) at station "
                f"{self._ran[int(report.step)]} of {self.stations}; the next step runs the target, a "
                f"{remaining:.3g}x viscosity jump",
                flush=True,
            )


def main() -> None:
    control = compare.CONTROL
    ramp = compare.SOLVER.continuation
    print(
        f"settle-ended ramp prototype: pace {ramp.stations} stations from anchor {ramp.anchor:g} "
        f"({ramp.anchor ** (1 / ramp.stations):.3f}x per step), settle at shift <= {control.beta_min:g} "
        f"with alpha >= {control.release_alpha:g}; grow {control.grow}, release {control.release_floor}",
        flush=True,
    )

    def built(*args, **kwargs):
        return SettleEndedRamp(
            *args, beta_min=control.beta_min, release_alpha=control.release_alpha, **kwargs
        )

    reynolds.ViscosityRampHomotopy = built  # this process only: `solve_reynolds_ramp` builds this
    started = time.time()
    aq = compare.solve_aquaflux(
        checkpoint_dir=HERE / "checkpoints",
        on_checkpoint=lambda report, state: SettleEndedRamp.active.observe(report, state),
    )
    homotopy = SettleEndedRamp.active
    ran = [homotopy._ran[s] for s in sorted(homotopy._ran)]
    arrival = next((s for s, station in enumerate(ran) if station >= homotopy.stations), None)
    print(
        f"[result] {len(ran)} steps, {time.time() - started:.0f} s; settled after step "
        f"{'never' if homotopy.settled_after is None else homotopy.settled_after + 1}; first target step "
        f"{'never' if arrival is None else arrival + 1}; stations run {ran}; x_r/h "
        f"{compare.reattachment_length(aq['centroid'], aq['U'][:, 0]):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
