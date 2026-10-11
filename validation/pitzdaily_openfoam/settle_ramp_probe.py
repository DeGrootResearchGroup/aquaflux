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
    PITZ_RAMP_STATIONS=24 PITZ_SETTLE_FINISH=3 validation/run_case.sh validation/pitzdaily_openfoam/settle_ramp_probe.py
    PITZ_RAMP_STATIONS=16 PITZ_FULL_STEP_TRACE=1 validation/run_case.sh validation/pitzdaily_openfoam/settle_ramp_probe.py

``PITZ_SETTLE_FINISH`` (default 1, a single jump) is how many steps the remaining viscosity span is
walked in once the settle comes, geometrically -- the question being whether smaller final steps keep
the settle that one large jump undoes (measured 2026-10-11: after a 21.5x jump the target took five
steps to settle again).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import os  # noqa: E402

import aquaflux  # noqa: E402,F401  (enables x64)
import aquaflux.solve.continuation as continuation  # noqa: E402
import aquaflux.turbulence.reynolds as reynolds  # noqa: E402
import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

#: ``PITZ_FULL_STEP_TRACE=1``: record, for every inner iteration, the FULL-STEP trial ratio
#: ``rho1 = |G(phi + delta)| / |G(phi)|`` -- the number the line search compares with one to decide
#: whether the full step is accepted, and discards when it is not. The settle is the first step whose
#: first inner iteration has ``rho1 < 1``; whether ``rho1`` trends toward one over the preceding steps
#: is what decides if the settle can be predicted. Costs one extra residual evaluation per inner
#: iteration and changes nothing the march computes.
FULL_STEP_TRACE = os.environ.get("PITZ_FULL_STEP_TRACE", "") == "1"
_FULL_STEP_RATIOS: list[float] = []
_line_search = continuation.backtracking_line_search


def _traced_line_search(residual_fn, phi, delta, reference_norm, steps, **kwargs):
    """The dual-time inner loop's line search, unchanged, after recording its full step's ratio."""
    norm = kwargs.get("norm", jnp.linalg.norm)
    full = jnp.minimum(1.0, kwargs.get("max_alpha", jnp.inf))
    ratio = norm(residual_fn(phi + full * delta)) / reference_norm
    jax.debug.callback(lambda r: _FULL_STEP_RATIOS.append(float(r)), ratio, ordered=True)
    return _line_search(residual_fn, phi, delta, reference_norm, steps, **kwargs)


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
        self.finish = int(os.environ.get("PITZ_SETTLE_FINISH", "1"))
        if self.finish < 1:
            raise ValueError(f"PITZ_SETTLE_FINISH must be >= 1, got {self.finish}")
        self.settled_after: int | None = None
        self._ran: dict[int, int] = {}
        # The finishing stations' own viscosity multipliers, by station id. Their ids continue the
        # ramp's (the station the settle came at, plus one per finishing step); the ramp's own stations
        # are never visited again once the finish starts, so the ids cannot collide with a ramp scale.
        self._finish_scales: dict[int, float] = {}
        SettleEndedRamp.active = self

    def station(self, step: int) -> int:
        if step not in self._ran:
            if self.settled_after is None or step <= self.settled_after:
                self._ran[step] = min(step, self.stations)
            else:
                into = step - self.settled_after  # 1 on the first step after the settle
                settled_at = self._ran[self.settled_after]
                self._ran[step] = self.stations if into >= self.finish else settled_at + into
        return self._ran[step]

    def scale(self, station: int) -> float:
        if station in self._finish_scales:
            return self._finish_scales[station]
        return super().scale(station)

    def observe(self, report, state) -> None:
        """``on_checkpoint``: mark the settle on the first full-length step at the floor off the target."""
        del state
        if FULL_STEP_TRACE:
            print(
                f"[full-step] step {int(report.step) + 1}: rho1 per inner "
                f"{' '.join(f'{r:.3f}' for r in _FULL_STEP_RATIOS)}",
                flush=True,
            )
            _FULL_STEP_RATIOS.clear()
        if self.settled_after is not None or report.arrived:
            return
        if float(report.shift) <= self.beta_min and float(report.alpha) >= self.release_alpha:
            self.settled_after = int(report.step)
            settled_at = self._ran[int(report.step)]
            remaining = self.scale(settled_at)
            # Walk the remaining span in `finish` equal geometric steps; the last is the target itself.
            for into in range(1, self.finish):
                self._finish_scales[settled_at + into] = remaining ** (1.0 - into / self.finish)
            print(
                f"[settle] step {int(report.step) + 1} was a full-length step at the floor "
                f"(shift {float(report.shift):.4g}, alpha {float(report.alpha):.3f}) at station "
                f"{settled_at} of {self.stations}; the remaining {remaining:.3g}x is walked in "
                f"{self.finish} step(s), {remaining ** (1 / self.finish):.3g}x each",
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
    if FULL_STEP_TRACE:
        continuation.backtracking_line_search = _traced_line_search  # read at trace time
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
