"""Does ``jax.grad`` actually run on the 3D case, and is the gradient right?

The whole coupled-preconditioner programme is justified by the adjoint: the transpose solve behind every
gradient meets the **unshifted** operator, with no pseudo-transient floor to soften it, which is why every
preconditioner comparison in this case is probed at ``beta = 0``. That argument had never been checked
end to end here -- no gradient had ever been taken through this case at all -- so a preconditioner could
have been tuned for an operating point nothing reached.

This asks two questions, cheapest first, and reports them separately because they fail for different
reasons:

1. **Does it execute, and is the result finite?** ``jax.grad`` through a converged coupled solve, on a
   real 3D state.
2. **Is it correct?** Against a central finite difference on the same objective (``BFS3D_ADJOINT_FD=1``).

⚠️ **What this does NOT yet check: iteration-count independence.** A gradient that matches a finite
difference can still have been taken by taping the forward march, and the two are told apart only by
showing the adjoint's cost does not scale with how many forward steps the solve happened to take. The
way to add it here is to repeat the same gradient from a *different* starting iterate -- which changes
the step count while leaving the root, and therefore the gradient, alone -- and check the gradient agrees
while the adjoint's cycle count does not move. Until that exists, treat a passing finite-difference
check as evidence the derivative is right, not as evidence of how it was obtained.

The objective is deliberately dull -- a sum of squares over ``k`` -- because the point is the derivative
machinery, not the functional. A reattachment length would drag a root-finder into the tape.

Usage::

    BFS3D_PROBE_STATE=state-00067 validation/run_case.sh validation/bfs3d_openfoam/adjoint_probe.py

``BFS3D_ADJOINT_FD=1`` adds part 2, which costs two more full solves and is off by default.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
from aquaflux.turbulence import solve_coupled  # noqa: E402
from aquaflux.turbulence.coupled import coupled_amg_continuation  # noqa: E402
from field_split_probe import STATES, load_state  # noqa: E402

#: Newton steps allowed per objective evaluation. The start is an already-converged root, so a solve at
#: the unperturbed viscosity should take one or two; the perturbed ones are what need the headroom.
MAX_STEPS = int(os.environ.get("BFS3D_ADJOINT_MAX_STEPS", "30"))

#: The finite-difference step for part 2. Large enough that the solve's own tolerance does not dominate
#: the difference, small enough to stay in the linear regime -- and the two constraints are why the
#: forward tolerance below is far tighter than a march would use.
FD_EPS = float(os.environ.get("BFS3D_ADJOINT_FD_EPS", "1e-4"))

#: Both the gradient and the finite difference are only as good as the root each solve lands on, so this
#: is far tighter than a march would use. But it is bounded from the other side too: the start is already
#: a converged root at |R| ~ 3.6e-06, and a RELATIVE tolerance is measured against that, so 1e-10 would
#: ask for ~3.6e-16 absolute -- unreachable in double precision, and the solve then spends its whole step
#: budget without stopping (observed: over four minutes and still going). 1e-8 asks for ~3.6e-14, which
#: is tight enough that a 1e-4 parameter perturbation is far above the noise it leaves.
RTOL = float(os.environ.get("BFS3D_ADJOINT_RTOL", "1e-8"))

#: How tightly each Newton step's LINEAR solve is driven. The march ships 0.3 because an inexact step is
#: cheap and globalization does the rest -- but that makes the nonlinear rate inexact-Newton-linear, and
#: measured here it is ~0.86 per step (each solve taking a single restart cycle), which cannot reach
#: `RTOL` in any affordable number of steps. From a converged root a near-exact linear solve should give
#: a Newton step that converges in a couple of iterations instead, so this probe pays for tight solves.
#: ⚠️ Nothing is free: at beta = 0 a tight solve is the EXPENSIVE regime (11 restart cycles on record,
#: against the 1 an `forward_rtol` of 0.3 buys), which is precisely the operator this whole campaign is
#: about.
FORWARD_RTOL = float(os.environ.get("BFS3D_ADJOINT_FORWARD_RTOL", "1e-6"))


def build():
    """The case, its converged state, and a preconditioner built ONCE on concrete parameters.

    The continuation must be constructed outside ``jax.grad``: it holds a factorized operator built from
    concrete numbers, and building it inside the traced objective would capture a tracer and escape the
    converged solve's ``custom_vjp`` -- the failure is a leaked-tracer error, not a wrong number.
    """
    name = os.environ.get("BFS3D_PROBE_STATE", "state-00067")
    if name not in STATES:
        raise SystemExit(f"BFS3D_PROBE_STATE={name!r} is not one of {list(STATES)}")
    coupled = compare.build_case()["coupled"]
    state = load_state(name)
    # PHYSICAL fields, not `layout.unpack`. The checkpoint stores the SOLVED-variable state, whose
    # scalar blocks are log(omega) under the log-variable transport this case runs; `solve_coupled`
    # takes a PHYSICAL initial condition and maps it into the solved space itself. Handing it the
    # unpacked blocks applies the log transform a second time -- which does not fail, it just starts
    # the solve a thousandfold away from the root it was handed (observed: |R| ~1.4 from a state whose
    # own residual is 1.07e-03). Caught only because the per-step log showed the first residual.
    flow, k, omega = coupled.physical_fields(state)
    print(
        f"\n{'=' * 78}\nadjoint probe on {name}: {STATES[name].description}\n{'=' * 78}", flush=True
    )
    return coupled, state, (flow, k, omega)


def _step_logger(label: str):
    """One line per outer step, flushed.

    Without this the probe prints nothing between "building" and its final answer, and a solve that is
    going nowhere costs its whole budget before saying so -- which is exactly what happened on the first
    attempt at this run. A gradient evaluation is several solves deep, so the label says which one.
    """

    def number(value, spec: str) -> str:
        """Format a step field, surviving a value that is a tracer rather than a number.

        Under ``jax.grad`` the forward march may run on traced values, and formatting a tracer with a
        numeric spec raises -- which would turn a diagnostic into the thing that kills the run. Falling
        back to the tracer's own repr keeps the log useful and the run alive.
        """
        try:
            return format(float(value), spec)
        except Exception:
            return f"<{type(value).__name__}>"

    def observe(report) -> None:
        print(
            f"    [{label}] step {report.step:3d}  |R| {number(report.residual_norm, '.4e')}  "
            f"ratio {number(report.residual_ratio, '.3f')}  beta {number(report.shift, '.4g')}  "
            f"alpha {number(report.alpha, '.3f')}  cycles {report.restart_cycles}",
            flush=True,
        )

    return observe


def make_objective(coupled, start, continuation, *, observe: bool = True):
    """``nu_scale -> sum(k**2)`` at the converged root, differentiable in ``nu_scale``.

    Scaling the molecular viscosity is the same differentiable parameter the 2D coupled adjoint tests
    use, and it reaches every term in the residual rather than only a boundary closure.
    """
    flow, k, omega = start
    label = {"n": 0}

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        label["n"] += 1
        # ⚠️ `on_step` MUST BE DROPPED under `jax.grad`, and `solve_coupled` raises rather than letting
        # it through: the observer drives a forward-only EAGER march that steps in Python on concrete
        # residual norms, which a differentiation tracer cannot flow through. The adjoint is
        # refresh-independent, so the single-stage solve gives the identical gradient -- the cost is
        # only that a gradient evaluation is silent. Hence the flag: the forward-only pass watches, the
        # differentiated pass does not.
        _, k_out, _ = solve_coupled(
            scaled,
            flow,
            k,
            omega,
            continuation=continuation,
            max_steps=MAX_STEPS,
            rtol=RTOL,
            **({"on_step": _step_logger(f"solve {label['n']}")} if observe else {}),
        )
        return jnp.sum(k_out**2)

    return objective


def main() -> None:
    coupled, state, start = build()
    print("  building the preconditioner (materialize + factorize)...", flush=True)
    started = time.time()
    continuation = coupled_amg_continuation(
        coupled,
        state,
        stencil_reach=3,
        column_reach=compare.COLUMN_REACH,
        smoother_fill_levels=0,
        smoother_sweeps=4,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        forward_rtol=FORWARD_RTOL,
        # THE CASE'S positivity settings, not the library's. Every step is capped by the k-positivity
        # rule whether or not one asks for it (`step_limit` is unconditional), and unfloored that cap
        # ratchets toward zero on this case -- observed here as a solve that decays geometrically at
        # ~0.855 per step from a converged root, identically at `forward_rtol` 0.3 and 1e-6, which is
        # what proved the linear solve was not what limited it. A probe that builds its own
        # continuation silently gets library defaults; this case's floor is load-bearing.
        positivity_floor=compare.K_POSITIVITY_FLOOR,
        positivity_projection=compare.K_POSITIVITY_PROJECTION,
        field_split=True,
        trailing_inverse=compare.TRAILING_INVERSE,
        leading_inverse=compare.LEADING_INVERSE,
    )
    print(f"  preconditioner built in {time.time() - started:.0f}s", flush=True)
    objective = make_objective(coupled, start, continuation, observe=True)
    # The same objective without the observer, for everything that runs under a JAX transform.
    silent = make_objective(coupled, start, continuation, observe=False)

    print(
        f"\n  -- forward only, rtol {RTOL:g}, forward_rtol {FORWARD_RTOL:g}, max_steps {MAX_STEPS}",
        flush=True,
    )
    started = time.time()
    value = float(objective(1.0))
    print(f"  forward only: objective {value:.9e}  in {time.time() - started:.0f}s", flush=True)

    print("\n  -- part 1: does the gradient run, and is it finite?", flush=True)
    started = time.time()
    gradient = float(jax.grad(silent)(1.0))
    elapsed = time.time() - started
    finite = np.isfinite(gradient)
    print(
        f"  d(sum k^2)/d(nu_scale) = {gradient:.9e}   finite={finite}   "
        f"forward+adjoint in {elapsed:.0f}s",
        flush=True,
    )
    if not finite:
        raise SystemExit("the gradient is not finite; nothing below is worth running")

    if os.environ.get("BFS3D_ADJOINT_FD") in (None, "", "0"):
        print(
            "\n  part 2 (finite-difference check) SKIPPED -- set BFS3D_ADJOINT_FD=1 to run it; it "
            "costs two more full solves.",
            flush=True,
        )
        return

    print(f"\n  -- part 2: central finite difference, eps={FD_EPS:g}", flush=True)
    started = time.time()
    plus = float(silent(1.0 + FD_EPS))
    minus = float(silent(1.0 - FD_EPS))
    difference = (plus - minus) / (2 * FD_EPS)
    relative = abs(gradient - difference) / max(abs(difference), 1e-300)
    print(
        f"  finite difference {difference:.9e}   adjoint {gradient:.9e}   "
        f"relative gap {relative:.3e}   in {time.time() - started:.0f}s",
        flush=True,
    )
    # A loose bar, deliberately. The finite difference is a difference of two independently converged
    # 3D solves, so its own accuracy is set by how tightly each landed -- it is the noisier of the two
    # quantities here, not the reference.
    print(
        "  VERDICT: "
        + (
            "adjoint agrees with the finite difference"
            if relative < 1e-3
            else "MISMATCH -- adjoint and finite difference disagree; suspect the solve tolerance "
            "before the adjoint, then check whether the two solves landed on the same root"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
