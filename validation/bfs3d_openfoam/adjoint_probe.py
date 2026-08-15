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
from aquaflux.solve import relative_residual_gmres  # noqa: E402
from aquaflux.turbulence import solve_coupled  # noqa: E402
from aquaflux.turbulence.coupled import coupled_amg_continuation  # noqa: E402
from field_split_probe import STATES, load_state  # noqa: E402

#: Newton steps allowed per objective evaluation. ⚠️ NOT "one or two": from a converged root at beta = 0
#: this solve contracts GEOMETRICALLY at ~0.83 per step rather than quadratically (cause undiagnosed), so
#: reaching `RTOL` takes tens of steps -- 22 to reach 2e-2, and the whole of this budget to reach 1e-4.
MAX_STEPS = int(os.environ.get("BFS3D_ADJOINT_MAX_STEPS", "90"))

#: The finite-difference step for part 2. Large enough that the solve's own tolerance does not dominate
#: the difference, small enough to stay in the linear regime -- and the two constraints are why the
#: forward tolerance below is far tighter than a march would use.
FD_EPS = float(os.environ.get("BFS3D_ADJOINT_FD_EPS", "1e-4"))

#: How tight a root each solve must reach, RELATIVE to its own starting residual. This is the single most
#: consequential setting here and the default is a MEASURED one, not a guess.
#:
#: ⚠️ A LOOSE ROOT SILENTLY BREAKS PART 2, AND BLAMES THE ADJOINT FOR IT. The implicit-function-theorem
#: gradient is only valid at `R = 0`, while a finite difference differentiates "parameter -> wherever the
#: solve stopped"; away from a root those are different functions. Measured on this case, against an
#: adjoint that moves by 0.001 % across the whole range:
#:
#:     root rtol 2e-2  ->  finite difference disagrees by 23 %
#:     root rtol 1e-3  ->  0.53 %
#:     root rtol 1e-4  ->  1.9e-04, agrees
#:
#: ⚠️ AND A BIGGER `FD_EPS` MAKES IT WORSE, NOT BETTER (23 % -> 64 % at 1e-3), so "the difference is just
#: noisy" is the wrong diagnosis however well the arithmetic seems to fit -- fix the root, not the step.
#: 1e-4 is the loosest value measured to agree. Tighter is bounded from the other side: the start is
#: already a converged root at |R| ~ 3.6e-06 and this is RELATIVE to it, so 1e-10 would ask for ~3.6e-16
#: absolute -- unreachable in double precision, and the solve then spends its whole budget without
#: stopping. 1e-8 is not demonstrated reachable at all.
RTOL = float(os.environ.get("BFS3D_ADJOINT_RTOL", "1e-4"))

#: How tightly each Newton step's LINEAR solve is driven. The march ships 0.3, and so does this probe.
#:
#: ⚠️ TIGHTENING IT IS MEASURED USELESS HERE, and an earlier version of this default (1e-6) rested on a
#: prediction that turned out to be wrong. The reasoning was: an inexact step makes the nonlinear rate
#: inexact-Newton-linear, so a near-exact linear solve should restore a couple-of-iterations Newton. It
#: does not. Two controlled arms sharing one trajectory show `forward_rtol` 0.3 against 1e-6 costs 4-8x
#: the Krylov work per step and moves the nonlinear trajectory **not at all** -- identical to five
#: significant figures. The geometric ~0.83 contraction is set by the Jacobian or the residual, not by
#: how well each step is solved, and it remains undiagnosed. `rtol` 1e-4 is reachable AT 0.3, which is
#: what the validated gradient was measured with.
FORWARD_RTOL = float(os.environ.get("BFS3D_ADJOINT_FORWARD_RTOL", "0.3"))

#: The adjoint's transpose solve gets its OWN Krylov settings, because it meets a different operator from
#: the forward march: no pseudo-transient shift on the diagonal, once, at the converged state. Left to
#: itself `solve_coupled` hands it `default_linear_solver()` -- a GMRES at lineax's stock restart and
#: stagnation budget -- which on this case stagnates and raises, naming a remedy that was unreachable from
#: here until `adjoint_solver` was threaded through `solve_coupled`.
#:
#: ⚠️ THESE DEFAULTS ARE THE MEASURED WORKING ONES. An earlier version defaulted to the FORWARD path's
#: choices (restart 15, a 60-restart cap) on the reasoning that they were what this case's linear algebra
#: was tuned at. That reasoning was unsound and the configuration FAILS: the transpose solve needs ~1450
#: preconditioner applications and restart-15 x 60 allows only ~900, so it dies on "The maximum number of
#: solver steps was reached". The operator was never intractable, it was under-resourced -- and the budget
#: had been sized from this file's own zero-shift figures, which are LINEAR-PROBE numbers at a different
#: right-hand side (`-R`, where the adjoint's is the cotangent, localized in one field block).
#: **Do not re-derive an adjoint budget from a forward or linear-probe cycle count. Measure it.**
#: At restart 120 the solve converges in ~12 restart cycles; the 150 cap leaves an order of headroom so it
#: does not bind, which is what lets the cost be MEASURED rather than truncated.
ADJOINT_RESTART = int(os.environ.get("BFS3D_ADJOINT_SOLVER_RESTART", "120"))
ADJOINT_STAGNATION_ITERS = int(os.environ.get("BFS3D_ADJOINT_SOLVER_STAGNATION_ITERS", "40"))
ADJOINT_MAX_RESTARTS = int(os.environ.get("BFS3D_ADJOINT_SOLVER_MAX_RESTARTS", "150"))

#: How tightly the transpose solve is driven. This single solve sets the gradient's accuracy directly.
#: 1e-6 is chosen against what the check can actually resolve -- a central difference between two
#: independently converged 3D solves -- and is measured sufficient for the 1.9e-04 agreement above.
ADJOINT_RTOL = float(os.environ.get("BFS3D_ADJOINT_SOLVER_RTOL", "1e-6"))

#: Print a line every this many transposed applications, so the transpose solve is not a silent black
#: box. `0` disables it. Cheap: one modulo per application, against a preconditioner apply.
ADJOINT_HEARTBEAT = int(os.environ.get("BFS3D_ADJOINT_HEARTBEAT", "100"))

#: Skip the watched forward-only evaluation, which exists to show the objective and its per-step march
#: and costs a whole solve. A gradient evaluation re-runs the forward solve anyway, so when the
#: question is the ADJOINT -- sweeping its Krylov settings, say -- that first pass is pure duplicate
#: cost: it halves the price of every arm to drop it. Keep it for the first run of a configuration,
#: where the per-step trajectory is what shows the solve reached a root at all.
SKIP_FORWARD = os.environ.get("BFS3D_ADJOINT_SKIP_FORWARD", "") not in ("", "0")


def adjoint_solver():
    """The transpose solve's own GMRES, at a relative-residual stop rather than a componentwise one.

    ``relative_residual_gmres`` stops on ``|A x - b| <= rtol |b|`` in a global two-norm. That matters
    on this system for the same reason the forward path uses it: the coupled state's blocks differ by
    orders of magnitude, and lineax's stock componentwise test lets a handful of near-zero right-hand
    side rows collapse onto their absolute floor and hold the whole solve far past the tolerance asked
    for.
    """
    return relative_residual_gmres(
        ADJOINT_RTOL,
        restart=ADJOINT_RESTART,
        stagnation_iters=ADJOINT_STAGNATION_ITERS,
        max_restarts=ADJOINT_MAX_RESTARTS,
    )


class TransposeApplyCounter:
    """Counts the preconditioner applications the ADJOINT spends, by counting transposed ones.

    The transpose solve's cost is what a preconditioner comparison at zero shift turns on, and nothing
    reports it: the restart-cycle count the linear solver returns is discarded inside the reverse rule,
    which has no observer. What *is* reachable is the preconditioner itself. It is a host object whose
    ``matvec`` reads ``self.factors`` at callback time, so replacing that attribute with a delegating
    proxy counts every application without touching the compiled solve.

    Counting the **transposed** applications is what makes the split exact and free. Forward and adjoint
    share one factorization, so a plain counter would mix them -- and a calibration run to subtract the
    forward share would cost a whole extra solve. But the forward march never applies the transpose, so
    the ``transpose=True`` applications are the adjoint's alone.

    An application is one right-preconditioned Krylov matrix-vector product (plus one recovery apply per
    restart cycle), so it is the honest unit here: restart cycles are not directly observable from
    outside, and dividing applications by the restart length only estimates them.

    Attributes
    ----------
    inner : object
        The real factors object, supplying ``n_dofs`` and ``apply(residual, transpose=)``.
    transposed : int
        Applications taken with ``transpose=True`` so far -- the adjoint's.
    forward : int
        Applications taken with ``transpose=False`` so far, kept as the control: it must not move
        across a gradient evaluation's transpose solve.
    """

    def __init__(self, inner: object, heartbeat: int = 0) -> None:
        self.inner = inner
        self.transposed = 0
        self.forward = 0
        self.heartbeat = heartbeat
        self.started = time.time()

    @property
    def n_dofs(self) -> int:
        return self.inner.n_dofs

    def apply(self, residual, *, transpose: bool = False):
        if transpose:
            self.transposed += 1
            # A HEARTBEAT, because the transpose solve is otherwise a silent black box for however
            # long it runs -- it is one linear solve, so it has no steps to report, and a run that is
            # going nowhere looks exactly like one that is nearly done. The application count is the
            # analogue of a per-step line: it says the solve is alive, how fast it is spending its
            # budget, and -- read against the restart length -- roughly which cycle it is in.
            # ⚠️ It does NOT say whether the residual is falling, so it distinguishes running from
            # hung, not converging from stagnating.
            if self.heartbeat and self.transposed % self.heartbeat == 0:
                print(
                    f"    [adjoint] {self.transposed} applications "
                    f"(~cycle {self.transposed / (ADJOINT_RESTART + 1):.1f}) "
                    f"in {time.time() - self.started:.0f}s",
                    flush=True,
                )
        else:
            self.forward += 1
        return self.inner.apply(residual, transpose=transpose)

    def __getattr__(self, name: str):
        # Anything else -- a refresh, a teardown -- goes straight through to the real object. Reached
        # only for attributes this class does not define, so the two counted ones cannot be bypassed.
        return getattr(self.inner, name)


def _adjoint_cost(counter: TransposeApplyCounter, forward_before: int) -> str:
    """The adjoint's measured cost, with the forward applications beside it as the control.

    The restart-cycle figure is derived, not measured, and is labelled as such: a right-preconditioned
    restarted GMRES applies the preconditioner once per Krylov matrix-vector product and once more to
    recover the solution at the end of each cycle, so ``restart + 1`` applications go into a full cycle
    -- but only the applications are counted here, and a partial final cycle makes the division an
    estimate rather than a count.
    """
    return (
        f"{counter.transposed} preconditioner applications "
        f"(~{counter.transposed / (ADJOINT_RESTART + 1):.1f} restart cycles at restart "
        f"{ADJOINT_RESTART}; the applications are measured, the cycles derived) -- forward "
        f"applications over the same evaluation: {counter.forward - forward_before}"
    )


def count_adjoint_applies(continuation, heartbeat: int = 0) -> TransposeApplyCounter:
    """Install a :class:`TransposeApplyCounter` on ``continuation``'s adjoint preconditioner.

    The adjoint factory is a ``TransposedPreconditioner`` wrapping a frozen-transpose factory that holds
    the host preconditioner, so the path to it is explicit rather than guessed. Mutating the host
    object's ``factors`` attribute is what the in-place refresh already does, and is seen by an
    already-compiled solve for the same reason: the callback reads the attribute rather than capturing
    it.
    """
    preconditioner = continuation.adjoint_preconditioner_factory.factory.preconditioner
    counter = TransposeApplyCounter(preconditioner.factors, heartbeat=heartbeat)
    preconditioner.factors = counter
    return counter


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
            adjoint_solver=adjoint_solver(),
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
    counter = count_adjoint_applies(continuation, heartbeat=ADJOINT_HEARTBEAT)
    # Every number below is only interpretable against the arm it was taken on, and the flow block is
    # the whole point of the comparison -- so state the bundle here rather than leaving it to be
    # reconstructed from the environment afterwards.
    print(
        f"  flow inverse {compare.FLOW_INVERSE}"
        f"   trailing {compare.TURBULENCE_INVERSE}"
        f"   column reach "
        f"{'uniform 3' if compare.COLUMN_REACH is None else '/'.join(map(str, compare.COLUMN_REACH))}",
        flush=True,
    )
    print(
        f"  adjoint solver: relative-residual GMRES rtol {ADJOINT_RTOL:g}, restart "
        f"{ADJOINT_RESTART}, stagnation_iters {ADJOINT_STAGNATION_ITERS}, max_restarts "
        f"{ADJOINT_MAX_RESTARTS}",
        flush=True,
    )
    objective = make_objective(coupled, start, continuation, observe=True)
    # The same objective without the observer, for everything that runs under a JAX transform.
    silent = make_objective(coupled, start, continuation, observe=False)

    print(
        f"\n  -- forward only, rtol {RTOL:g}, forward_rtol {FORWARD_RTOL:g}, max_steps {MAX_STEPS}",
        flush=True,
    )
    if SKIP_FORWARD:
        print(
            "  SKIPPED (BFS3D_ADJOINT_SKIP_FORWARD) -- the gradient re-runs it anyway", flush=True
        )
    else:
        started = time.time()
        value = float(objective(1.0))
        print(f"  forward only: objective {value:.9e}  in {time.time() - started:.0f}s", flush=True)

    print("\n  -- part 1: does the gradient run, and is it finite?", flush=True)
    forward_applies_before = counter.forward
    started = time.time()
    # ⚠️ REPORT THE COST EVEN WHEN THE SOLVE RAISES. A failing adjoint is the interesting case here,
    # and the count is the only thing that says WHICH solve failed and how far it got: a transposed
    # count of zero means the run never reached the transpose solve at all (so the forward march is
    # what raised), while a count sitting at the restart cap says the transpose solve ran and did not
    # converge. Without this the first failure threw the number away and left both readings open.
    try:
        gradient = float(jax.grad(silent)(1.0))
    except Exception:
        print(
            f"  ADJOINT COST AT FAILURE: {_adjoint_cost(counter, forward_applies_before)}",
            flush=True,
        )
        raise
    elapsed = time.time() - started
    finite = np.isfinite(gradient)
    print(
        f"  d(sum k^2)/d(nu_scale) = {gradient:.9e}   finite={finite}   "
        f"forward+adjoint in {elapsed:.0f}s",
        flush=True,
    )
    print(f"  ADJOINT COST: {_adjoint_cost(counter, forward_applies_before)}", flush=True)
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
