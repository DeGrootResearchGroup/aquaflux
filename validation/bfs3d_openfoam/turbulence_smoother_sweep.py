"""Which smoother should the turbulence half of the field split use, judged on TIME?

The shipped field split runs an incomplete-LU sweep on both halves. That choice was inherited from the
monolithic bundle, where it was tuned against a six-field indefinite saddle -- but the trailing half is
not a saddle. It is a two-field advection-diffusion-reaction pair with a genuine diagonal, and a
zero-fill incomplete factorization may simply be more work than it needs. This sweeps the trailing
smoother with the leading one **held fixed**, so the only variable is what preconditions ``[k, omega]``.

**The measure is seconds, and that is the whole point of the exercise.** A restart-cycle count is a valid
proxy for cost only between candidates that share a per-application price, and these deliberately do not:
an incomplete-LU sweep, a batch of 2x2 block inverses and a diagonal scaling differ several-fold in what
one application costs. A candidate that adds a cycle and halves the price of every cycle is a win that no
cycle count can show -- which is exactly how the field split itself was nearly abandoned. So each arm is
reported as:

``setup``
    The whole split's build, and the trailing hierarchy's build alone. A refresh pays this, and on this
    case refresh is about a sixth of the march.
``apply``
    One application of the assembled preconditioner, timed directly rather than inferred. This is paid
    once per Krylov matrix-vector product and is where a cheaper smoother earns its keep.
``march``
    Wall time to satisfy the march's **own** stopping rule -- the loose row-scaled inexact-Newton stop at
    the shipped restart, not a tight one. This is the number the arms are ranked by, because it is the
    one a march would actually pay.
``tight``
    Cycles and the TRUE relative residual at an adjoint-grade stop. This is the **screen**, not the
    ranking: it says whether an arm converges at all on a hard operator, which a loose stop reached in
    one cycle cannot tell you. An arm that fails here is out regardless of how fast it looked.

Ranking on the loose stop and screening on the tight one are separate jobs, and an arm has to pass both.

**A HARD state cannot rank candidates either, and that is the less obvious half.** The familiar trap is
the benign operating point, where every candidate converges in a cycle or two and the sweep reports no
difference -- so one reaches for the march's worst iterate instead. But the worst iterate is not what a
march mostly pays for: on the shipped run **139 of 194 inner solves cost one restart cycle and only 7
exceeded three**. A candidate that is cheaper per application and buys an extra cycle *only where the
operator is hard* pays that penalty a handful of times and banks the saving on every other solve, which
a hard-state ranking scores exactly backwards. So the two classes do different jobs:

* the **hard** iterate is the screen -- does this arm survive when the operator is not kind?
* the **step-initial** states are the ranking -- this is the solve the march actually repeats.

The blended estimate reported at the end weights them by how often each occurs in the shipped march.
It is a crude model and is labelled as one: a different preconditioner shifts that distribution, which
is precisely why the blend decides who earns a march rather than deciding the winner.

**A screen is not a verdict.** A preconditioner's effect on a march runs through the step control --
cheaper steps hold the line search higher, which changes the trajectory, the refresh pattern and the
step count. Only a full march settles an arm; this exists to decide which arms are worth one.

Usage -- states are measured one at a time, since each materializes a Jacobian of some gigabytes::

    python3 -u validation/bfs3d_openfoam/turbulence_smoother_sweep.py state-00064 inner-00050-03
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    build_amg_vcycle,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)
from field_split_probe import (  # noqa: E402
    FLOOR,
    SMOOTHERS,
    STATES,
    _trailing_inverse,
    field_split,
    load_state,
    march_solver,
    materialize,
)


def _replaces_hierarchy(arm: str) -> bool:
    """Does this arm supply a WHOLE block inverse rather than a smoother inside a V-cycle?

    The two kinds are measured differently: a smoother's setup cost is a hierarchy build, while a
    whole-block inverse has no hierarchy to build at all -- which is most of what makes it interesting.
    """
    return arm not in SMOOTHERS


#: The leading half is held at the shipped incomplete-LU sweep for every arm, so a difference between
#: rows is attributable to the trailing half and nothing else.
FLOW_SMOOTHER = "ilu0"

#: The trailing smoothers to try, in the order they are reported. ``ilu0`` is the shipped choice and the
#: control. Everything after it is cheaper per application in some way -- fewer sweeps, no factorization,
#: or no sequential dependency at all -- which is the hypothesis under test.
#:
#: ``chebyshev`` is deliberately absent. It does not converge on this block -- a polynomial built for a
#: bounded positive real spectrum, applied to an advection-dominated nonsymmetric pair whose spectrum is
#: complex -- and a non-converging arm is the most expensive kind to measure, because running to the
#: restart cap is what failing costs. Name it explicitly with ``--arms=chebyshev`` to re-confirm it.
ARMS = (
    "ilu0",
    "ilu0x2",
    "ilu0x1",
    "pbjacobi",
    "pbjacobix2",
    "sor",
    "jacobi",
    "jacobix2",
    "jacobix1",
    "jacobix8",
)

#: Adjoint-grade, for the screen. Far past the march's own stop, so an arm that is merely weak is
#: distinguishable from one that cannot converge.
TIGHT_RTOL = 1e-8

#: Applications to time. One application is milliseconds, so a single sample is dominated by whatever
#: else the machine is doing; the minimum over a batch is the estimator least sensitive to that.
APPLY_SAMPLES = 40

#: Repeats of the ranked (march-tolerance) solve. Same reasoning -- the minimum is reported.
SOLVE_REPEATS = 3

#: How the blended estimate weights a state, by the class it belongs to. Taken from the shipped march's
#: own inner-solve distribution: 139 of 194 solves took one restart cycle and 173 took at most two, so
#: roughly nine in ten are the cheap step-initial kind and the rest are the tail the hard iterates stand
#: for. A state whose name is not listed falls back to the cheap weight.
STATE_WEIGHTS = {"inner": 0.11, "step": 0.89}


def state_class(name: str) -> str:
    """``"inner"`` for a captured mid-loop iterate, ``"step"`` for a step-initial checkpoint."""
    return "inner" if name.startswith("inner-") else "step"


def time_apply(preconditioner, n_dofs: int, generator: np.random.Generator) -> float:
    """Seconds for one application of the assembled preconditioner, as a minimum over samples.

    Timed on the host, against the object's own ``apply``, so it measures the preconditioner rather than
    the callback that carries it into the traced solve. The right-hand side is random rather than a real
    residual because the cost of a V-cycle is set by the operator's structure, not by what is being
    smoothed -- and a random vector cannot accidentally be easy.
    """
    vector = generator.standard_normal(n_dofs)
    preconditioner.factors.apply(vector)  # once to fault in whatever the first call allocates
    best = float("inf")
    for _ in range(APPLY_SAMPLES):
        started = time.perf_counter()
        preconditioner.factors.apply(vector)
        best = min(best, time.perf_counter() - started)
    return best


def time_solve(preconditioner, coupled, state, rhs, op_shift, solver, repeats: int):
    """``(seconds, cycles, true relative residual)`` for the real system, seconds as a minimum.

    The first call compiles, so it is run once and discarded before timing: a compile charged to
    whichever arm happens to be first would swamp the differences this is looking for.
    """

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    matvec = preconditioner.matvec()
    solution, raw = solve_linear(operator, rhs, solver, preconditioner=matvec, throw=False)
    jnp.asarray(solution).block_until_ready()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        solution, raw = solve_linear(operator, rhs, solver, preconditioner=matvec, throw=False)
        jnp.asarray(solution).block_until_ready()
        best = min(best, time.perf_counter() - started)
    true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
    return best, restart_cycles(int(raw)), true


def trailing_setup(shifted, groups, arm: str) -> float:
    """Seconds to build the trailing hierarchy ALONE, so the shared leading half does not mask it.

    The full split's build is dominated by the four-field saddle, which every arm shares; a difference of
    a few seconds in the two-field block would be invisible inside it. This is the part a refresh would
    actually save.
    """
    block = shifted[groups.trailing, :][:, groups.trailing]
    started = time.perf_counter()
    if _replaces_hierarchy(arm):
        # No hierarchy to build at all -- this is the setup the arm exists to avoid paying.
        inverse = _trailing_inverse(arm)(block, groups.n_trailing_fields)
    else:
        inverse = build_amg_vcycle(
            block,
            groups.n_trailing_fields,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            extra_options=SMOOTHERS[arm] or None,
        )
    elapsed = time.perf_counter() - started
    inverse.destroy()
    return elapsed


def run(arm, shifted, groups, n_fields, coupled, state, rhs, op_shift, loose, generator):
    """Build one arm and report it, surviving a failure so the arms queued behind it still run."""
    preconditioner = None
    try:
        started = time.perf_counter()
        preconditioner = field_split(shifted, groups, n_fields, FLOW_SMOOTHER, arm, flow_first=True)
        setup = time.perf_counter() - started
        alone = trailing_setup(shifted, groups, arm)
        apply_s = time_apply(preconditioner, groups.n_dofs, generator)
        march_s, march_cyc, march_true = time_solve(
            preconditioner, coupled, state, rhs, op_shift, loose, SOLVE_REPEATS
        )
        tight = relative_residual_gmres(
            TIGHT_RTOL, restart=15, stagnation_iters=40, max_restarts=60
        )
        _, tight_cyc, tight_true = time_solve(
            preconditioner, coupled, state, rhs, op_shift, tight, 1
        )
        row = {
            "arm": arm,
            "setup": setup,
            "alone": alone,
            "apply": apply_s,
            "march_s": march_s,
            "march_cyc": march_cyc,
            "march_true": march_true,
            "tight_cyc": tight_cyc,
            "tight_true": tight_true,
        }
        # `march_true` is the ACHIEVED true residual at the march's own loose stop, and it is the
        # number this report exists to surface. The stop asks for 0.3; a strong preconditioner blows
        # through that by orders of magnitude inside one restart cycle while a weak one lands near it,
        # and BOTH report one cycle. Since the march takes an inexact-Newton step from whatever the
        # solve returns, that achieved residual -- not the requested tolerance and not the cycle count
        # -- is what sets the quality of the direction the step control then has to work with.
        print(
            f"    {arm:<12} setup {setup:5.1f}s (block {alone:4.1f}s)  apply {1e3 * apply_s:6.1f}ms  "
            f"march {march_s:6.2f}s / {march_cyc} cyc, achieved {march_true:.1e}  "
            f"tight {tight_cyc:>3} cyc {tight_true:.1e}",
            flush=True,
        )
        return row
    except Exception as failure:
        print(f"    {arm:<12} FAILED  {type(failure).__name__}: {failure}", flush=True)
        return None
    finally:
        if preconditioner is not None:
            preconditioner.factors.destroy()
        del preconditioner
        gc.collect()


def report(measured, baseline_arm: str) -> None:
    """Per-state tables, then a ranking blended by how often each class of solve occurs.

    Parameters
    ----------
    measured : dict
        ``state name -> list of row dicts`` (a row is ``None`` where that arm failed to build).
    baseline_arm : str
        The control every ratio is taken against.
    """
    states = list(measured)
    arms = [r["arm"] for r in measured[states[0]] if r is not None]
    # An arm is disqualified by failing the tight screen ANYWHERE. Surviving the easy states says
    # nothing -- the screen exists for the state that is not easy.
    failures = {
        row["arm"]: (name, row)
        for name in states
        for row in measured[name]
        if row is not None and not (np.isfinite(row["tight_true"]) and row["tight_true"] <= 1e-6)
    }

    for name in states:
        weight = STATE_WEIGHTS[state_class(name)]
        print(f"\n{'=' * 92}\n{name}  (class {state_class(name)!r}, blend weight {weight:.2f})")
        print(
            f"{'arm':<12}{'march s':>10}{'vs base':>10}{'cyc':>6}{'achieved':>11}"
            f"{'apply ms':>11}{'tight cyc':>11}"
        )
        rows = [r for r in measured[name] if r is not None]
        base = next((r for r in rows if r["arm"] == baseline_arm), None)
        for row in sorted(rows, key=lambda r: r["march_s"]):
            against = f"{row['march_s'] / base['march_s']:.2f}x" if base else "--"
            print(
                f"{row['arm']:<12}{row['march_s']:>10.2f}{against:>10}{row['march_cyc']:>6}"
                f"{row['march_true']:>11.1e}{1e3 * row['apply']:>11.1f}{row['tight_cyc']:>11}"
            )

    # The blend. Weights are per CLASS, so several states of one class share that class's weight
    # between them rather than each carrying it -- otherwise adding a second cheap state would silently
    # double the cheap class's influence.
    per_class = {}
    for name in states:
        per_class.setdefault(state_class(name), []).append(name)
    blended = {}
    for arm in arms:
        total = 0.0
        for klass, names in per_class.items():
            share = STATE_WEIGHTS[klass] / len(names)
            for name in names:
                row = next((r for r in measured[name] if r is not None and r["arm"] == arm), None)
                if row is None:
                    total = float("nan")
                    break
                total += share * row["march_s"]
        blended[arm] = total

    print(f"\n{'=' * 92}\nBLENDED ESTIMATE -- expected seconds per solve over the march's own mix")
    print(f"{'=' * 92}\n{'arm':<12}{'blended s':>12}{'vs base':>10}   screen")
    base_blend = blended.get(baseline_arm)
    for arm in sorted(arms, key=lambda a: blended[a]):
        against = f"{blended[arm] / base_blend:.2f}x" if base_blend else "--"
        if arm in failures:
            name, row = failures[arm]
            verdict = f"FAILS at {name} (TRUE rel {row['tight_true']:.1e})"
        else:
            verdict = "passes"
        print(f"{arm:<12}{blended[arm]:>12.2f}{against:>10}   {verdict}")
    print(
        "\nRanked on the loose stop because that is what a march pays; screened on the tight one\n"
        "because a loose stop reached in one cycle cannot tell a strong preconditioner from a lucky\n"
        "one. 'apply ms' and 'block setup' are the two places a cheaper smoother pays off -- per\n"
        "matrix-vector product and per refresh. The blend is a model, not a measurement: a different\n"
        "preconditioner shifts the very distribution the weights come from, and cheaper steps hold the\n"
        "line search higher, which changes the trajectory and the step count. Arms that lead here, and\n"
        "arms merely CLOSE here, earn a full march -- nothing here settles a default."
    )


def sweep_state(name: str, selected, coupled, groups, n_fields, colouring, structure):
    """Measure every selected arm at one state, then release that state's Jacobian.

    One state per call, and its several-gigabyte operator is dropped before the next: two live copies of
    a materialized 3D coupled Jacobian is enough to exhaust a workstation, and a sweep that fits in
    memory only when nothing else runs is a sweep that will eventually be run alongside something else.
    """
    march_beta, _, description = STATES[name]
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0
    print(
        f"\n{'=' * 92}\n{name}: {description}\noperator beta {march_beta}, "
        f"preconditioner beta {pc_beta}\n{'=' * 92}",
        flush=True,
    )
    state = load_state(name)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    op_shift = _frozen_shift_diagonal(base, march_beta, state) if march_beta > 0 else 0.0
    jacobian = materialize(coupled, state, colouring, structure, n_fields)
    pc_shift = (
        _frozen_shift_diagonal(base, pc_beta, state) if pc_beta > 0 else np.zeros(groups.n_dofs)
    )
    shifted = MonolithicAmgPreconditioner._shifted(jacobian, pc_shift)
    del jacobian
    gc.collect()

    loose = march_solver(coupled, base, state)
    generator = np.random.default_rng(0)
    rows = [
        run(arm, shifted, groups, n_fields, coupled, state, rhs, op_shift, loose, generator)
        for arm in selected
    ]
    del shifted
    gc.collect()
    return rows


def main() -> None:
    # `--arms=` restricts the ladder. Adding one candidate should not cost a rerun of the arms whose
    # answer is already on the log -- and the arms that FAIL are the expensive ones, since running to the
    # restart cap is what failing means here. The control is always kept: a subset with nothing to
    # compare against cannot be ranked.
    chosen = [a for a in sys.argv[1:] if a.startswith("--arms=")]
    names = [a for a in sys.argv[1:] if not a.startswith("--arms=")]
    if not names or any(n not in STATES for n in names):
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <state> [state ...] [--arms=key,key]\n"
            f"states: {', '.join(STATES)}\n"
            "Give at least one step-initial state (the ranking) and one inner iterate (the screen)."
        )
    only = set(chosen[-1].split("=", 1)[1].split(",")) if chosen else None
    # An arm is either a smoother recipe (a V-cycle configured by PETSc options) or a whole-block
    # inverse that REPLACES the hierarchy, which the trailing-inverse seam resolves by name.
    unknown = {a for a in (only or set()) if a not in SMOOTHERS and not _replaces_hierarchy(a)}
    if unknown:
        raise SystemExit(
            f"unknown arm(s) {sorted(unknown)}; smoothers: {sorted(SMOOTHERS)}, "
            "or blockjacobi<N> to replace the hierarchy entirely"
        )
    selected = tuple(a for a in ARMS if only is None or a == ARMS[0] or a in only)
    # An arm named on the command line but absent from ARMS is still a legitimate request -- ARMS is a
    # default ladder, not the set of things that exist.
    selected += tuple(a for a in sorted(only or ()) if a not in selected)

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=2,
    )
    print(
        f"{'=' * 92}\nturbulence-smoother sweep: leading half held at {FLOW_SMOOTHER!r}, "
        f"trailing half varied\narms: {', '.join(selected)}\n"
        f"plain aggregation, coarse_eq_limit {compare.COARSE_EQ_LIMIT}, reach 3, "
        f"preconditioner beta floor {FLOOR}\n{'=' * 92}",
        flush=True,
    )
    colouring = _coupled_jacobian_colouring(coupled, 3)
    structure = block_stencil_gather_map(colouring, n_fields)
    measured = {
        name: sweep_state(name, selected, coupled, groups, n_fields, colouring, structure)
        for name in names
    }
    report(measured, ARMS[0])


if __name__ == "__main__":
    main()
