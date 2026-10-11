"""Differentiable matrix-free linear solve, with optional left/right preconditioning.

A thin wrapper over ``lineax`` that solves ``A x = b`` given only a matrix-vector product
``matvec(x) = A x`` — never a materialized matrix. ``lineax`` differentiates the solve by
**implicit differentiation** (it differentiates the solution of ``A x = b`` directly, rather
than unrolling the iterative solver onto the tape), so a gradient taken through
:func:`solve_linear` costs one extra solve and is independent of the iteration count. This is
the linear-solve primitive the Newton driver and the gradient schemes build on.

An optional **preconditioner** ``M`` (a matvec approximating ``A^{-1}``) is applied on a caller-chosen
side (``preconditioner_side``), defaulting to the **right**: the solver is handed ``A M`` and ``b``,
solves for ``y``, and recovers ``x = M y``. The Krylov residual is then ``b - A M y = b - A x`` — the
*true* residual — so the stopping test stays honest even when ``M`` is a poor inverse (a left
preconditioner would instead stop on the *preconditioned* residual ``M(A x - b)``, which a weak ``M``
can drive small while the true residual is large — the failure mode on the shifted coupled saddle at
low pseudo-transient shift). The **left** form (``M A`` and ``M b``, stopping on ``‖M r‖``) is the right
choice for the opposite regime — a strong ``M`` on a well-behaved SPD operator, where the preconditioned
residual measures the error and reaches tolerance that the true residual, on a badly conditioned
operator, cannot in a bounded number of steps. The converged solution is identical either way, and since
``M``'s coefficients are treated as constant (the caller ``stop_gradient``s them), preconditioning
changes only the Krylov convergence, not the solution or its gradient — it is implicit-diff-transparent.

**That transparency is a property of a CONVERGED solve.** At a finite tolerance the returned ``x``
is whatever the iteration reached, which does depend on ``M``; a caller running a deliberately
inexact solve (a loose ``rtol``, ``throw=False``, or one that stagnates) can and does see the step
change when the preconditioner changes. Rely on ``M``-independence only where the solve actually
meets its tolerance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx


def default_linear_solver() -> lx.AbstractLinearSolver:
    """A general-purpose matrix-free solver (restarted GMRES) with tight tolerances.

    It keeps ``lineax``'s own restart length (20) and stagnation budget (20 cycles). It is the fallback
    for every adjoint transpose solve, and no single restart length is cheaper across the operators
    those solves meet: ``lineax`` always completes a restart cycle before testing convergence, and needs
    one further cycle to see the solution stop moving, so a longer restart is a higher floor on an easy
    solve and pays only on a hard one. Measured with ``validation/adjoint_gmres_restart.py`` (restart 10,
    20, 40, 80): on a skewed lid-driven cavity of 432 to 3072 unknowns, restart 20 was the cheapest at
    every size, and 80 cost 1.3 to 2.1 times as long; on a 2800-unknown turbulent
    channel, 40 and 80 were about 10 % and 30 % faster than 20, and 10 stagnated. Every converged arm
    gave the same gradient. An operator that needs a far larger budget (a three-dimensional coupled
    saddle) passes its own ``adjoint_solver``; raising this default would not reach it and would slow
    every easy solve.
    """
    return lx.GMRES(rtol=1e-10, atol=1e-10)


# ``lineax``'s GMRES counts a start-up pass as a step (it forms the initial residual and runs no Arnoldi
# cycle), and it stops only once the solution moved by less than the tolerance over a whole restart
# cycle, so a solve whose residual met the tolerance inside its first cycle still runs a second and
# reports 3. Subtracting 2 therefore leaves the Arnoldi cycles run MINUS ONE: a solve counted as "1
# cycle" ran two, ``2 * (restart + 1) + 1`` operator applications. The convention is kept because every
# recorded count is in it. The offset is per *solve*, which is why a multi-solve step must divide it out
# by its solve count rather than subtract a flat 2 (see `restart_cycles`).
_LINEAX_STEP_OFFSET = 2


def corrected_cycles(raw_count, solves=1):
    """The traced counterpart of :func:`restart_cycles`, for use inside a jitted loop.

    Same correction, expressed in ``jax.numpy`` so it can run on a tracer. Kept beside
    :func:`restart_cycles` so the offset has exactly one definition.
    """
    return jnp.maximum(raw_count - _LINEAX_STEP_OFFSET * solves, 0)


def restart_cycles(raw_count: int, solves: int = 1) -> int:
    """Strip the fixed per-solve offset from a raw ``lineax`` iteration count.

    :func:`solve_linear` returns ``lineax``'s ``num_steps``, which carries a constant ``+2`` per solve:
    one for a start-up pass that runs no Arnoldi cycle, and one for the cycle its stopping test needs
    after the residual has met the tolerance (it also demands that the solution stop moving). The
    corrected count is therefore one less than the Arnoldi cycles run, and it is the convention every
    count is reported in: a solve reported as one cycle ran two. Correcting it matters most where the
    count is smallest: at ``num_steps = 6`` over two solves the corrected count is two, against six raw.

    Parameters
    ----------
    raw_count : int
        The summed ``num_steps`` reported over ``solves`` solves.
    solves : int
        How many separate solves ``raw_count`` covers (e.g. the inner Newton iterations of one implicit
        timestep). Default ``1``.

    Returns
    -------
    int
        The offset-corrected restart-cycle count, clamped at ``0`` so a ``raw_count`` of ``0`` (the
        convention for "not measured") stays ``0`` rather than going negative.

    Examples
    --------
    >>> restart_cycles(3)
    1
    >>> restart_cycles(6, solves=2)
    2
    >>> restart_cycles(0, solves=1)
    0
    """
    return max(int(raw_count) - _LINEAX_STEP_OFFSET * int(solves), 0)


def _global_two_norm(pytree: Any) -> jnp.ndarray:
    """The Euclidean (2-)norm over *all* leaves of ``pytree`` as one flat vector."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _require_measure(solver: Any) -> None:
    """Refuse a solver configured to follow a step's measure but run outside one."""
    if solver.norm is None:
        raise TypeError(
            "this GMRES stops in the progress measure of the Newton step that runs it (norm=None), and "
            "it was run outside one. Pass the measure as `norm`, or hand the solver to a step."
        )


class _RelativeResidualGMRES(lx.GMRES):
    """GMRES whose stopping test is a *global* relative residual in an injected norm, not componentwise.

    ``lineax``'s stock GMRES applies ``rtol``/``atol`` **componentwise** -- it stops once, for every
    entry ``i``, ``|r_i| <= atol + rtol*|b_i|`` (default ``max_norm``, so the single worst entry
    decides). On a coupled saddle system a handful of entries have a near-zero right-hand side (e.g.
    wall-fixation rows that start satisfied, whose ``|b_i|`` collapses to zero), and there the scale
    degenerates to ``atol`` alone -- an *absolute* demand. Those few entries then hold the whole solve
    to ``~atol`` and force it to converge orders of magnitude past the relative tolerance that was
    actually requested.

    This subclass sidesteps that by scaling the right-hand side to unit ``self.norm`` inside
    :meth:`compute` and deferring to a stock GMRES configured with ``rtol = 0`` (so the componentwise
    ``rtol*|b_i|`` term vanishes and the scale is the uniform ``atol``), ``self.norm``, and ``atol`` set
    to the desired relative tolerance. The residual half of the stop is then ``norm(r) <= atol * norm(b)``
    -- one global relative test in whatever measure ``self.norm`` is, immune to the near-zero entries.
    It is not the whole stop: the stock GMRES also requires the change in the solution over the last
    restart cycle to be at most ``atol`` in the same norm (with the right-hand side at unit norm), and
    the first cycle it counts is a start-up pass that does not reduce the residual. So a solve whose
    residual meets the tolerance can still run another cycle because its solution is still moving. The
    measure is injected (:func:`relative_residual_gmres`): the default ``_global_two_norm`` gives the
    plain 2-norm stop, and a **row-scaled** measure (:class:`~aquaflux.solve.RowScaledNorm`, the coupled
    march's own progress measure) gives a stop that weights every field block comparably rather than
    letting the largest-magnitude block (``omega`` on the coupled saddle) decide alone. Any injected
    ``norm`` must be positively homogeneous of degree one (``norm(c v) = |c| norm(v)``) so the unit-norm
    scaling acts as a relative tolerance; the 2-norm and ``RowScaledNorm`` both are.
    """

    def compute(
        self, state: Any, vector: Any, options: dict[str, Any]
    ) -> tuple[Any, Any, dict[str, Any]]:
        _require_measure(self)
        # Scale the right-hand side to unit ``self.norm`` so the (absolute) ``atol`` floor acts as a
        # relative tolerance; undo the scaling on the returned solution (the map ``b -> x`` is linear, so
        # a constant factor passes straight through). ``jnp.where`` guards a zero right-hand side.
        scale = self.norm(vector)
        scale = jnp.where(scale > 0.0, scale, 1.0)
        scaled = jax.tree_util.tree_map(lambda v: v / scale, vector)
        solution, result, stats = super().compute(state, scaled, options)
        return jax.tree_util.tree_map(lambda x: x * scale, solution), result, stats


def relative_residual_gmres(
    rtol: float,
    *,
    norm: Callable[[Any], jnp.ndarray] | None = _global_two_norm,
    restart: int = 120,
    stagnation_iters: int = 40,
    max_restarts: int | None = None,
) -> lx.AbstractLinearSolver:
    """A GMRES that stops at a **global** relative residual ``norm(Ax - b) <= rtol*norm(b)``.

    The robust termination for an inexact-Newton *forward* solve: it stops when the linear residual
    has fallen by the factor ``rtol`` in ``norm``, rather than by ``lineax``'s stock componentwise
    ``max_norm`` test -- which a few near-zero-right-hand-side entries of a coupled saddle system
    quietly convert into an absolute ``atol`` demand, forcing the solve orders of magnitude past the
    tolerance asked for (see :class:`_RelativeResidualGMRES`).

    Parameters
    ----------
    rtol : float
        The relative-residual target ``norm(r) / norm(b)`` at which to stop.
    norm : callable
        The measure the relative stop is taken in, ``v -> scalar``, positively homogeneous of degree one
        (``norm(c v) = |c| norm(v)``). The default ``_global_two_norm`` gives the plain Euclidean stop.
        Passing the coupled march's **row-scaled** progress measure
        (:class:`~aquaflux.solve.RowScaledNorm`) makes the forward solve stop when *every* field block
        has fallen by ``rtol`` in that measure, rather than when the largest-magnitude block alone has
        (``omega`` on the coupled saddle, whose residual is orders above the flow) -- so the flow
        correction is never left blind. Pairing it with a *loose* ``rtol`` gives a cheap yet flow-aware
        inexact-Newton stop. The measure is a fixed, physically row-scaled one (state row-diagonals),
        not a per-solve right-hand-side normalization, so the stop is problem-independent.

        ``None`` takes the measure from the Newton step that runs the solver, **at each step**
        (:func:`in_progress_measure`): a march that rebuilds its measure every outer iteration then
        stops its linear solves in the rebuilt one, rather than in whichever measure was current when
        the solver was configured. Such a solver cannot be run outside a step.
    restart : int
        The Krylov subspace size before a restart (default ``120``).
    stagnation_iters : int
        Restart cycles without progress after which the solve gives up (default ``40``).
    max_restarts : int or None
        A hard cap on the number of restart cycles, as an inexact-Newton safety bound; ``None``
        (default) leaves ``lineax``'s own generous cap in place and relies on ``rtol``.

    Returns
    -------
    lineax.AbstractLinearSolver
        A solver realizing the relative-residual stop, for injection as a forward solver.

    Notes
    -----
    The residual it measures is the **preconditioned** one when the solve is left-preconditioned
    (``solve_linear`` folds the preconditioner into the operator and right-hand side), so the test is
    ``norm(M(Ax - b)) <= rtol*norm(M b)``. That is the standard, and adequate, inexact-Newton stopping
    quantity for a globalized march; the converged root and its adjoint are unaffected either way,
    since the shift vanishes at the root and the adjoint is a separate transpose solve.
    """
    return _RelativeResidualGMRES(
        rtol=0.0,
        atol=rtol,
        norm=norm,
        restart=restart,
        stagnation_iters=stagnation_iters,
        max_steps=max_restarts,
    )


class _ResidualStopGMRES(_RelativeResidualGMRES):
    """Restarted GMRES that stops the moment its residual meets the tolerance, tested every iteration.

    :class:`_RelativeResidualGMRES` inherits ``lineax``'s stopping rule, which is tested only at a
    restart boundary and also demands that the solution has stopped moving over the last whole cycle:
    every solve then runs at least two full cycles, ``2 (restart + 1) + 1`` operator applications, even
    when one cycle's first few iterations already met the tolerance. An inexact-Newton forcing term is a
    bound on the linear residual alone, so this solver tests that and nothing else, after every
    iteration, in the same injected measure: ``norm(b - A x) <= rtol norm(b)``.

    Within a cycle the residual is formed from the Arnoldi relation, ``r = V_{j+1} (beta e_1 - H_j y_j)``,
    so testing it costs no operator application; at the end of each cycle the true residual ``b - A x``
    is recomputed (one application) and is what the next cycle starts from and the final verdict reads,
    because the recurrence drifts from it as the basis loses orthogonality. The basis is orthogonalized by classical Gram--Schmidt applied twice,
    and the small least-squares problem is re-solved at every iteration (its size is at most the restart
    length, so this is negligible beside one application of the operator).

    The reported count is the restart cycles RUN, a partial one included, plus ``lineax``'s fixed offset
    of two, so :func:`restart_cycles` reads it as the number of cycles run. That is one MORE than the
    same number means for :class:`_RelativeResidualGMRES`, whose corrected count is the cycles run minus
    one; a cost trigger keyed on the count fires at a different difficulty under the two solvers.

    The forward march is the only consumer, so only :meth:`compute` is replaced; transposition and the
    rest are ``lineax``'s, and the adjoint's transpose solve is configured separately.

    Attributes
    ----------
    on_solve : callable or None
        ``(applications, cycles) -> None``, called on the host after each solve with the operator
        applications it made (one per Krylov iteration plus one per cycle for its true residual) and
        the cycles it ran. ``None`` (default) elides it. An observer for
        studies that count work: the cycle count alone cannot, since a cycle may stop part way.
    """

    on_solve: Callable[[Any, Any], None] | None = eqx.field(static=True, default=None)

    def compute(
        self, state: Any, vector: Any, options: dict[str, Any]
    ) -> tuple[Any, Any, dict[str, Any]]:
        _require_measure(self)
        del options  # ``solve_linear`` folds any preconditioner into the operator
        operator, norm, size = state, self.norm, self.restart
        target = self.atol * norm(vector)
        rows = jnp.arange(size + 1)

        def iterate(inner):
            j, basis, hessenberg, _, _, beta = inner
            w = operator.mv(basis[j])
            coefficients = jnp.zeros(size + 1, dtype=w.dtype)
            for _ in range(
                2
            ):  # classical Gram--Schmidt, twice: once loses orthogonality in float64
                projection = jnp.where(rows <= j, basis @ w, 0.0)
                w = w - projection @ basis
                coefficients = coefficients + projection
            length = jnp.linalg.norm(w)
            hessenberg = hessenberg.at[:, j].set(coefficients.at[j + 1].set(length))
            basis = basis.at[j + 1].set(w / jnp.where(length > 0.0, length, 1.0))
            # Least squares over the columns built so far: zero columns are left out of the
            # minimum-norm solution, so the unbuilt ones contribute nothing.
            built = jnp.where(jnp.arange(size) <= j, hessenberg, 0.0)
            rhs = jnp.zeros(size + 1, dtype=w.dtype).at[0].set(beta)
            y = jnp.linalg.lstsq(built, rhs)[0]
            residual = (rhs - built @ y) @ basis
            done = (norm(residual) <= target) | (length <= jnp.finfo(w.dtype).eps * beta)
            return j + 1, basis, hessenberg, y, done, beta

        def cycle(outer):
            x, r, cycles, applications, _ = outer
            beta = jnp.linalg.norm(r)
            basis = jnp.zeros((size + 1, r.size), dtype=r.dtype).at[0].set(r / beta)
            hessenberg = jnp.zeros((size + 1, size), dtype=r.dtype)
            inner = (0, basis, hessenberg, jnp.zeros(size, dtype=r.dtype), jnp.asarray(False), beta)
            j, basis, hessenberg, y, done, beta = jax.lax.while_loop(
                lambda inner: (inner[0] < size) & jnp.logical_not(inner[4]), iterate, inner
            )
            x = x + y @ basis[:size]
            # The true residual, not the Arnoldi one: the recurrence drifts from b - A x as the basis
            # loses orthogonality, and near a tight target that drift is the whole residual.
            r = vector - operator.mv(x)
            return x, r, cycles + 1, applications + j + 1, done

        def keep_going(outer):
            _, r, cycles, _, _ = outer
            return (cycles < self.max_steps) & (norm(r) > target)

        start = (jnp.zeros_like(vector), vector, 0, 0, norm(vector) <= target)
        x, r, cycles, applications, _ = jax.lax.while_loop(keep_going, cycle, start)
        if self.on_solve is not None:
            jax.debug.callback(self.on_solve, applications, cycles)
        converged = norm(r) <= target
        result = lx.RESULTS.where(converged, lx.RESULTS.successful, lx.RESULTS.max_steps_reached)
        return x, result, {"num_steps": cycles + _LINEAX_STEP_OFFSET, "max_steps": self.max_steps}


def residual_stop_gmres(
    rtol: float,
    *,
    norm: Callable[[Any], jnp.ndarray] | None = None,
    restart: int = 15,
    max_restarts: int = 14,
    on_solve: Callable[[Any, Any], None] | None = None,
) -> lx.AbstractLinearSolver:
    """A restarted GMRES that stops on ``norm(b - A x) <= rtol norm(b)`` alone, tested every iteration.

    The stop an inexact-Newton forcing term asks for, without the restart-boundary granularity and the
    solution-change test of :func:`relative_residual_gmres` (see :class:`_ResidualStopGMRES`).

    Parameters
    ----------
    rtol : float
        The relative-residual target.
    norm : callable or None
        The measure the stop is taken in, positively homogeneous of degree one. ``None`` (default) takes
        it from the Newton step that runs the solver (:func:`in_progress_measure`).
    restart : int
        The Krylov subspace size before a restart (default ``15``).
    max_restarts : int
        The most restart cycles a solve may run (default ``14``).
    on_solve : callable or None
        ``(applications, cycles) -> None``, told each solve's operator applications and cycles run.

    Returns
    -------
    lineax.AbstractLinearSolver
        The solver, for injection as a forward solver (``linear_solve=``).
    """
    return _ResidualStopGMRES(
        rtol=0.0,
        atol=rtol,
        norm=norm,
        restart=restart,
        stagnation_iters=max_restarts,
        max_steps=max_restarts,
        on_solve=on_solve,
    )


def in_progress_measure(
    solver: lx.AbstractLinearSolver, norm: Callable[[Any], jnp.ndarray]
) -> lx.AbstractLinearSolver:
    """``solver``, stopping in ``norm`` if it was configured to follow the running step's measure.

    A Newton step calls this with its own progress measure before each linear solve, so a solver built
    with ``relative_residual_gmres(norm=None)`` stops in the measure the march is judging that step by.
    Any other solver -- one given an explicit ``norm``, or a stock ``lineax`` solver -- is returned
    unchanged.

    Parameters
    ----------
    solver : lineax.AbstractLinearSolver
        The linear solver the step was given.
    norm : callable
        The step's progress measure, ``v -> scalar``.

    Returns
    -------
    lineax.AbstractLinearSolver
        The solver to run.
    """
    if isinstance(solver, _RelativeResidualGMRES) and solver.norm is None:
        return eqx.tree_at(lambda s: s.norm, solver, norm, is_leaf=lambda leaf: leaf is None)
    return solver


def solve_linear(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    *,
    preconditioner_side: str = "right",
    throw: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve ``A x = b`` for ``x`` given the linear map ``matvec(x) = A x``, with the solve's cost.

    Returns the solution **and** the solver's reported iteration count. The count is the *cost* of
    the solve rather than part of its result — how hard the preconditioned system was this time — so
    a caller that only wants the answer drops it at the call site (``x, _ = solve_linear(...)``).
    There is deliberately no count-free variant to wrap this one: a second entry point would mean a
    second signature and a second parameter docstring to keep in step.

    **The count is restart cycles, not matrix-vector products (binding — the easy misreading).** For a
    restarted GMRES ``stats["num_steps"]`` counts *cycles*, each of which is up to ``restart``
    matvecs, so a "17" is ~17x``restart`` matvecs. A solver that reports no iteration count (a direct
    factorization) yields ``0``.

    **Why the count is worth returning:** a frozen preconditioner going stale shows up first as a
    *rising cycle count* on an otherwise-unchanged system, well before it shows up in the residual
    history. That makes the cycle count the honest trigger for re-freezing the preconditioner
    mid-march — and a robust one, unlike wall-clock time, which a suspended or loaded machine
    perturbs without the linear algebra having changed at all.

    Parameters
    ----------
    matvec : callable
        The linear operator, mapping ``x`` of shape ``b.shape`` to ``A x`` of the same shape.
        Must be linear in its argument.
    b : jnp.ndarray
        Right-hand side.
    solver : lineax.AbstractLinearSolver, optional
        The linear solver; defaults to :func:`default_linear_solver`.
    preconditioner : callable, optional
        A preconditioner ``M`` (a matvec approximating ``A^{-1}``), applied on the side given by
        ``preconditioner_side``. ``M``'s internal coefficients must be constant with respect to any
        outer differentiation (``stop_gradient``-ed by the caller), so that preconditioning accelerates
        convergence without perturbing the solution or its gradient.
    preconditioner_side : str
        ``"right"`` (default) or ``"left"``. **Right** hands the solver ``x -> A(M(x))`` and ``b`` and
        recovers ``x = M(y)``, so the Krylov residual is the *true* residual ``b - A x`` — the honest
        stop when ``M`` is a **poor** inverse (a weak ``M`` cannot report convergence while the true
        residual is large, the failure mode on the shifted coupled saddle at low pseudo-transient shift).
        **Left** hands the solver ``x -> M(A(x))`` and ``M b``, so the Krylov residual is the
        *preconditioned* residual ``M(b - A x)``. That is the right choice when ``M`` is a **strong**
        inverse of a well-behaved (symmetric positive-definite) operator — there ``‖M r‖`` measures the
        solution error directly and reaches tolerance where the true residual, for a badly conditioned
        operator, cannot in a bounded number of steps (an anisotropy-stiff Poisson solve with a
        multigrid ``M``). The converged solution is identical either way; only the stopping quantity
        (hence which regime converges in ``max_steps``) differs. Ignored when ``preconditioner`` is ``None``.
    throw : bool
        If ``True`` (default), a non-convergent solve raises. If ``False``, it instead returns the
        solver's last iterate without raising — for a caller that tests the result and recovers (an
        adaptive continuation that escalates damping when the shifted solve fails to converge). The
        returned iterate may not solve the system; the caller must check it.

    Returns
    -------
    x : jnp.ndarray
        The solution, of shape ``b.shape``.
    cycles : jnp.ndarray
        The solver's iteration count (restart **cycles** for a restarted GMRES), an ``int32`` scalar.
        The dtype is pinned so a caller can carry it through a ``lax.while_loop`` whose carry
        structure must be invariant (the escalation loop in the pseudo-transient step does).
    """
    if solver is None:
        solver = default_linear_solver()
    if preconditioner is None:
        preconditioned_matvec, rhs, recover = matvec, b, (lambda y: y)
    elif preconditioner_side == "left":
        # LEFT preconditioning: solve ``(M A) x = M b`` directly for ``x``. The Krylov residual is the
        # *preconditioned* residual ``M(b - A x)`` -- the appropriate stop when ``M`` is a strong inverse
        # of a well-behaved (SPD) operator, where ``‖M r‖`` measures the error and reaches tolerance that
        # the true residual cannot on a badly conditioned operator (see the docstring). ``M``'s
        # coefficients are constant w.r.t. any outer differentiation, so the solution and its gradient are
        # unchanged; only the stopping quantity differs.
        def preconditioned_matvec(x):
            return preconditioner(matvec(x))

        rhs, recover = preconditioner(b), (lambda y: y)
    else:
        # RIGHT preconditioning (default): solve ``(A M) y = b`` for ``y`` and recover ``x = M y``. The
        # Krylov residual is then ``b - A M y = b - A x`` -- the *true* residual -- so the relative-residual
        # stop is honest even when ``M`` is a poor inverse (the shifted coupled saddle at low ``beta``,
        # where a left-preconditioned solve would report convergence while returning a step that does not
        # solve the system). The solution ``x`` is identical to the left form (both solve ``A x = b``); only
        # the honesty of the stopping test differs.
        def preconditioned_matvec(x):
            return matvec(preconditioner(x))

        rhs, recover = b, preconditioner
    operator = lx.FunctionLinearOperator(
        preconditioned_matvec, jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    solution = lx.linear_solve(operator, rhs, solver=solver, throw=throw)
    return recover(solution.value), jnp.asarray(solution.stats.get("num_steps", 0), dtype=jnp.int32)
