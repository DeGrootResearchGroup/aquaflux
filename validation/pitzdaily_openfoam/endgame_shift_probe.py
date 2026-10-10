"""Does the shift FLOOR set the target station's convergence rate? One outer step per shift, from the
march's own states.

The shipped pitzDaily control walks the shift ``beta`` down by ``1/grow`` per comfortable step from
``beta_start`` and clamps it at ``beta_min = 0.005``, which it reaches inside the viscosity ramp -- so
every step of the target station runs at that one fixed shift. Near a root a dual-time step at a fixed
shift contracts each mode of the error by ``beta / (lambda + beta)``, where ``lambda`` is the mode's
generalized eigenvalue against the shift diagonal; the slowest mode therefore sets a LINEAR rate that
no number of steps turns quadratic, and only ``beta -> 0`` removes it. The march's own tail rate (about
0.64-0.82 per step at the floor) then implies ``lambda ~ 0.4 beta``, and predicts the rate at every
other shift. Whether that is what limits the station is the question; nothing on record measures a
shift below the floor at a settled target state (every recorded low-shift failure was taken while the
ramp was still moving the problem).

**The probe.** From a checkpointed state of the target station -- the state the ramp's arrival step
starts from, one mid-station and one in the linear tail -- take ONE outer dual-time step at each of
several shifts, with the preconditioner re-fitted at that state and shift, and record the contraction of
the row-equilibrated residual in the measure the march judged that step by (built at the starting
state and held fixed, as the march holds it within an iteration). Two preconditioner floors: the
shipped ``refit_beta_floor`` (the inverse fitted no lower than 0.05, a 10x mismatch at the floor) and
a tracking one (fitted at the shift itself, zero included -- the operator the adjoint's transpose solve
already runs on). Two inner-loop settings: the file's own (``inner_steps`` / ``inner_tol``), which is
what the march would get, and a tight one, which isolates the shift's contraction from the inner
loop's.

**The control that validates the harness.** Before the arms, the march's own step is re-taken from
the same state at its recorded shift, with the preconditioner walked to where the march had it, and its
residual and cycle count must reproduce the next checkpoint's record; a harness that does not reproduce
the march is measuring some other configuration (the carried-protocol trap).

Prediction per state, printed beside every measurement: with ``lambda`` fitted from the control's own
contraction ``rho_c`` at ``beta_c`` (``lambda = beta_c (1 / rho_c - 1)``), the contraction at ``beta`` is
``beta / (lambda + beta)``. A measured ratio far below the prediction at small ``beta`` is Newton taking
over (quadratic, not a fixed rate); one far above it says something other than the shift -- a limiter
binding, a stale or failing inverse, the inner loop's own tolerance -- is holding the step.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/endgame_shift_probe.py

``PITZ_ENDGAME_STATES`` (default: the arrival state, a mid-station one and a tail one -- file indices,
i.e. steps completed), ``PITZ_ENDGAME_BETAS`` (default ``0.005,0.0015,0.0005,0``),
``PITZ_ENDGAME_FLOORS`` (``shipped,tracking``) and ``PITZ_ENDGAME_INNER`` (``shipped,tight``) choose
the arms.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402
from replay import MarchReplay  # noqa: E402

BETAS = tuple(
    float(b) for b in os.environ.get("PITZ_ENDGAME_BETAS", "0.005,0.0015,0.0005,0").split(",")
)
FLOORS = tuple(os.environ.get("PITZ_ENDGAME_FLOORS", "shipped,tracking").split(","))
INNER = tuple(os.environ.get("PITZ_ENDGAME_INNER", "shipped,tight").split(","))
#: The tight inner loop: enough iterations at a hundredth of the file's tolerance that the inner loop
#: is not what bounds the step's contraction.
TIGHT_INNER_STEPS, TIGHT_INNER_TOL = 12, 1e-4


@eqx.filter_jit
def one_step(strategy, residual_fn, phi, residual_norm_0, solver):
    """One outer step, compiled as the march compiles it (strategy and residual as arguments)."""
    return strategy.stepper()(residual_fn, phi, residual_norm_0, solver)


@eqx.filter_jit
def measured(measure, residual):
    return measure(residual)


def default_states(replay: MarchReplay) -> tuple[int, ...]:
    """The arrival state, one a third of the way into the target station, and one two thirds in."""
    first, last = replay.stations, max(replay.states) - 1
    span = last - first
    return (first, first + span // 3, first + 2 * span // 3)


def main() -> None:
    replay = MarchReplay()
    states = (
        tuple(int(s) for s in os.environ["PITZ_ENDGAME_STATES"].split(","))
        if os.environ.get("PITZ_ENDGAME_STATES")
        else default_states(replay)
    )
    solver_settings = replay.solver
    floor_values = {
        "shipped": float(solver_settings.preconditioner.refit_beta_floor),
        "tracking": 0.0,
    }
    inner_values = {
        "shipped": (replay.step.inner_steps, replay.step.inner_tol),
        "tight": (TIGHT_INNER_STEPS, TIGHT_INNER_TOL),
    }
    print(
        f"{replay.describe()}; states {states}; betas {BETAS}; floors "
        f"{ {f: floor_values[f] for f in FLOORS} }; inner {{ {', '.join(f'{i}: {inner_values[i]}' for i in INNER)} }}",
        flush=True,
    )
    coupled = replay.coupled
    residual_fn = coupled.residual
    session = replay.session
    hook = session._refresh_hook()  # the one β-tracking refresh every step built here shares
    shipped_floor = hook._refit_beta_floor

    # One step object per inner setting; the shift rides as a dynamic leaf, so every β is a cache hit.
    steps = {}
    for name in INNER:
        inner_steps, inner_tol = inner_values[name]
        march = dict(replay.march)
        march["dual_time"] = dataclasses.replace(
            march["dual_time"], inner_steps=inner_steps, inner_tol=inner_tol
        )
        steps[name] = session.build(replay.seed, **march)

    def controlled(step, beta, measure):
        return eqx.tree_at(
            lambda s: s.relaxation_schedule, step, ConstantRelaxation(jnp.asarray(beta))
        ).with_norm(measure)

    def take(step, phi_n, reference_norm):
        started = time.perf_counter()
        outcome = one_step(step, residual_fn, phi_n, reference_norm, step.linear_solver())
        jax.block_until_ready(outcome.phi)
        return outcome, time.perf_counter() - started

    header = (
        f"{'state':>5} {'arm':>16} {'beta':>7} | {'ratio':>9} {'predict':>8} {'inner':>5} "
        f"{'cycles':>6} {'max':>4} {'alpha':>6} {'limit':>6} {'met':>3} {'s':>6}"
    )
    for k in states:
        phi_n = jnp.asarray(replay.states[k]["state"])
        record = replay.states[k + 1]
        beta_c = float(record["shift"])
        # The march's preconditioner where the march had it when it built step k+1's measure: walking
        # the replay's systems re-fits in full at a station change and mid-step where the march did.
        # The measure's velocity row scale reads the frozen inverse's `a_p`, and the march builds the
        # measure BEFORE the step's own pre-step refresh -- so at the arrival step it is built on the
        # ramp's last inverse, and the arrival's full re-fit at the target follows it.
        hook._refit_beta_floor = shipped_floor
        step = steps["shipped"] if "shipped" in INNER else steps[INNER[0]]
        if k == replay.stations:
            for _ in replay.systems(k, k):  # the ramp's last step, a station change itself
                pass
            measure = coupled_scaled_norm(coupled, step.shift_policy, phi_n)
            replay.refit(coupled, beta_c, phi_n)
        else:
            for _ in replay.systems(replay.stations + 1, k):
                pass
            measure = coupled_scaled_norm(coupled, step.shift_policy, phi_n)
        reference_norm = measured(measure, residual_fn(phi_n))
        session.refresh_preconditioner(
            controlled(step, beta_c, measure), phi_n
        )  # binds the hook's step
        outcome, seconds = take(controlled(step, beta_c, measure), phi_n, reference_norm)
        rho_c = float(outcome.residual_norm) / float(reference_norm)
        recorded = float(record["residual_norm"]) / float(reference_norm)
        print(
            f"\nstate {k} (step {k + 1} starts here): |R| {float(reference_norm):.4e} in its own measure; "
            f"recorded step {k + 1}: shift {beta_c:.4g}, |R| {float(record['residual_norm']):.4e} "
            f"(ratio {recorded:.4f}), cycles {int(record['cycles'])}, inner {int(record['inner_iterations'])}, "
            f"alpha {float(record['alpha']):.3f}",
            flush=True,
        )
        same = abs(rho_c - recorded) <= 1e-3 * max(recorded, 1e-12) and int(outcome.cycles) == int(
            record["cycles"]
        )
        print(
            f"control (march's own step re-taken): ratio {rho_c:.4f}, cycles {int(outcome.cycles)}, inner "
            f"{int(outcome.inner_iterations)}, alpha {float(outcome.alpha):.3f}, {seconds:.1f} s -- "
            f"{'reproduces the record' if same else '⚠️ DOES NOT reproduce the record'}",
            flush=True,
        )
        lam = beta_c * (1.0 / rho_c - 1.0)
        print(f"fitted slowest mode lambda = {lam:.3e} ({lam / beta_c:.2f} beta_c)", flush=True)
        print(header, flush=True)
        for floor in FLOORS:
            for inner in INNER:
                step = steps[inner]
                for beta in BETAS:
                    hook._refit_beta_floor = floor_values[floor]
                    arm = controlled(step, beta, measure)
                    hook.rebind(coupled)  # force the next refresh to be a full re-fit at this state
                    session.refresh_preconditioner(arm, phi_n)
                    outcome, seconds = take(arm, phi_n, reference_norm)
                    ratio = float(outcome.residual_norm) / float(reference_norm)
                    predicted = beta / (lam + beta)
                    print(
                        f"{k:5d} {floor + '/' + inner:>16} {beta:7.4g} | {ratio:9.4f} {predicted:8.4f} "
                        f"{int(outcome.inner_iterations):5d} {int(outcome.cycles):6d} "
                        f"{int(outcome.max_inner_cycles):4d} {float(outcome.alpha):6.3f} "
                        f"{float(outcome.binding_limit):6.3f} {'yes' if bool(outcome.reached_target) else 'no':>3} "
                        f"{seconds:6.1f}",
                        flush=True,
                    )
        hook._refit_beta_floor = shipped_floor


if __name__ == "__main__":
    main()
