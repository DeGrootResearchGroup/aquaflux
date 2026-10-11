"""Rebuild the linear systems a pitzDaily march solved, one at a time, from its own checkpoints.

A probe of the inner linear solve (a Krylov variant, a preconditioner change) is only evidence about the
march if it runs on the march's own systems. ``compare.py`` run with ``PITZ_CHECKPOINT_KEEP=500
PITZ_INNER_DUMP_ABOVE=1`` keeps the state every step starts from and every inner iterate; from those this
module rebuilds each system exactly. Inner solve ``i`` of step ``k`` linearizes at ``p_i`` (``p_0`` the
step's start ``phi_n``, ``p_{i+1}`` the dumped iterate of inner ``i``) and solves

    (J(p_i) + s) delta = -(R(p_i) + s (p_i - phi_n)),     s = beta_k * row_relaxation * d(phi_n),

with ``beta_k`` the step's recorded shift and ``d`` the step's own shift policy. That policy is built once,
at the hybrid start of the anchor station, exactly as the ramp builds it, and the preconditioner is
refitted where the march refitted it: in full at a station change, and at the iterate a solve reached
``refresh_on_cycles`` restart cycles on, once per step, at ``max(beta, refit_beta_floor)``.

A replay that does not reproduce the march's recorded cycle counts is measuring some other sequence, so
every consumer should check it: :meth:`MarchReplay.march_solver` is the march's own solver, bound to the
step's measure.

⚠️ Any later run of ``compare.py`` writes its own rolling checkpoints into the same directory and evicts
the capture's step states (the inner dumps survive, the states do not); re-run the capture first. A
capture made under different march settings leaves inner dumps the new one may not overwrite, so start a
capture from an empty ``checkpoints`` directory.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import compare
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.initialization import hybrid_initialize
from aquaflux.solve import ConstantRelaxation, in_progress_measure, jacobian_matvec, solve_linear
from aquaflux.turbulence import coupled_step, open_session, scale_both_blocks, scale_momentum_only
from aquaflux.turbulence.coupled import coupled_scaled_norm

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"
_COMPANIONS = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}


def load(directory: Path = CHECKPOINTS):
    """The capture's step states and inner iterates, keyed by step and by ``(step, inner)``."""
    states = {}
    for path in sorted(directory.glob("state-*.npz")):
        # Keyed by the FILE's index, the number of steps completed: the record's own `step` field
        # counts from zero, so `state-00017` holds step 16 -- the state step 18 (one-based) starts from.
        with np.load(path) as data:
            states[int(path.stem.split("-")[1])] = {k: np.asarray(data[k]) for k in data.files}
    inner = {}
    for path in sorted(directory.glob("inner-*.npz")):
        with np.load(path) as data:
            inner[(int(data["attempt"]), int(data["inner"]))] = {
                k: np.asarray(data[k]) for k in data.files
            }
    return states, inner


@dataclass(frozen=True)
class System:
    """One inner linear system of the march, ready to solve.

    ``b`` is the right-hand side, ``shift`` the diagonal ``s``, ``p`` the linearization point and
    ``measure`` the step's progress measure, which the march's solve stops in. ``recorded`` is the cycle
    count the march reported for it.
    """

    step: int
    inner: int
    assembler: object
    p: jnp.ndarray
    phi_n: jnp.ndarray
    shift: jnp.ndarray
    measure: object
    b: jnp.ndarray
    recorded: int


class MarchReplay:
    """The march's systems in order, with its preconditioner refitted where the march refitted it.

    ``preconditioner`` is the march's own (a host object behind a callback), so a probe may change how
    it applies between solves; ``apply_pc`` is its matvec and :meth:`preconditioned` the operator
    ``(J + s) M`` the solve sees.
    """

    def __init__(self, directory: Path = CHECKPOINTS):
        self.solver = compare.SOLVER
        self.ramp = self.solver.continuation
        self.refresh_on = self.solver.dual_time.refresh_on_cycles
        self.states, self.inner = load(directory)
        self.coupled = compare.build_case()["coupled"]
        self._companion = _COMPANIONS[self.ramp.scale]
        # The step and its shift policy, built where the ramp builds them: at the anchor's hybrid start.
        anchor = self.assembler_for(1)
        self.seed = anchor.state_from_physical(*hybrid_initialize(anchor))
        allowed = inspect.signature(coupled_step).parameters
        march = {
            k: v
            for k, v in self.solver.settings().items()
            if k in allowed and k != "preconditioner"
        }
        self.session = open_session(self.solver.settings()["preconditioner"], self.coupled)
        self.session.rebind(anchor)
        self.march = march
        self.step = self.session.build(self.seed, **march)
        self.policy = self.step.shift_policy
        self.preconditioner = self.policy.preconditioner
        self.apply_pc = self.preconditioner.matvec()
        apply_pc = self.apply_pc

        @eqx.filter_jit
        def preconditioned(assembler, p, shift, v):
            z = apply_pc(v)
            return jacobian_matvec(assembler, p, z) + shift * z

        @eqx.filter_jit
        def solve(assembler, p, shift, b, solver):
            return solve_linear(
                lambda v: jacobian_matvec(assembler, p, v) + shift * v,
                b,
                solver=solver,
                preconditioner=apply_pc,
                throw=False,
            )

        self._preconditioned = preconditioned
        self._solve = solve

    @property
    def stations(self) -> int:
        """The ramp's steps; the target station starts at the step after."""
        return self.ramp.stations * self.ramp.steps_per_station

    def assembler_for(self, step: int):
        """The residual the march solved at ``step`` (one-based), its viscosity station's companion."""
        station = min((step - 1) // self.ramp.steps_per_station, self.ramp.stations)
        if station == self.ramp.stations:
            return self.coupled
        return self._companion(
            self.coupled, self.ramp.anchor ** (1.0 - station / self.ramp.stations)
        )

    def march_solver(self, measure):
        """The march's own Krylov solver, stopping in ``measure`` as the step binds it."""
        return in_progress_measure(self.step.linear_solver(), measure)

    def solve(self, system: System, solver):
        """``solver`` on ``system`` with the current preconditioner: ``(delta, raw_steps)``."""
        return self._solve(system.assembler, system.p, system.shift, system.b, solver)

    def preconditioned(self, system: System, v):
        """``(J + s) M v`` at ``system``."""
        return self._preconditioned(system.assembler, system.p, system.shift, v)

    def refit(self, assembler, beta, state):
        """Re-fit the preconditioner in full at ``state`` and ``beta`` (floored as the march floors it)."""
        self.session.rebind(assembler)
        self.session.refresh_preconditioner(
            eqx.tree_at(lambda s: s.relaxation_schedule, self.step, ConstantRelaxation(beta)), state
        )

    def systems(self, first: int | None = None, last: int | None = None):
        """Yield every inner system of steps ``first..last`` in the march's order.

        The default range is the target station to the last captured step, or ``PITZ_REPLAY_FROM`` /
        ``PITZ_REPLAY_TO``. Starting anywhere but at a station change is NOT faithful: the march's
        inverse there dates from a mid-step refresh. The preconditioner is refitted after a system is
        consumed, so a consumer sees each system with the inverse the march used for it.
        """
        if first is None:
            first = int(os.environ.get("PITZ_REPLAY_FROM", self.stations + 1))
        if last is None:
            last = int(os.environ.get("PITZ_REPLAY_TO", max(self.states)))
        previous = None
        for k in range(first, last + 1):
            phi_n = jnp.asarray(self.states[k - 1]["state"]) if k > 1 else self.seed
            beta = float(self.states[k]["shift"])
            assembler = self.assembler_for(k)
            r_n = assembler.residual(phi_n)
            shift = jax.lax.stop_gradient(self.policy.shift_term(phi_n, r_n).shift(beta))
            measure = coupled_scaled_norm(self.coupled, self.policy, phi_n)
            # A station change re-fits in full at the step's start.
            if assembler is not previous:
                self.refit(assembler, beta, phi_n)
            previous = assembler
            refreshed = False
            i, p = 0, phi_n
            while (k, i) in self.inner:
                record = self.inner[(k, i)]
                recorded = int(record["cycles"])
                b = -(assembler.residual(p) + shift * (p - phi_n))
                yield System(k, i, assembler, p, phi_n, shift, measure, b, recorded)
                # The march's mid-step refresh: once per step, at the iterate the expensive solve reached.
                if self.refresh_on is not None and recorded >= self.refresh_on and not refreshed:
                    self.refit(assembler, beta, jnp.asarray(record["state"]))
                    refreshed = True
                p = jnp.asarray(record["state"])
                i += 1

    def describe(self) -> str:
        """The configuration line every consumer prints first."""
        linear = self.solver.linear_solve
        return (
            f"[configuration] states 0..{max(self.states)}; stop {linear.stop}, rtol {linear.rtol}, "
            f"restart {linear.restart}, max_restarts {linear.max_restarts}; refresh on "
            f"{self.refresh_on} cycles; jax {jax.__version__}, {jax.default_backend()}"
        )
