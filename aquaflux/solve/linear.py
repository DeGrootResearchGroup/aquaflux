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

import jax
import jax.numpy as jnp
import lineax as lx


def default_linear_solver() -> lx.AbstractLinearSolver:
    """A general-purpose matrix-free solver (restarted GMRES) with tight tolerances."""
    return lx.GMRES(rtol=1e-10, atol=1e-10)


# A converged ``lineax`` GMRES reports two `num_steps` beyond the restart cycles it actually ran, so a
# solve that converged inside a single cycle reads as 3. The offset is per *solve*, which is why a
# multi-solve step must divide it out by its solve count rather than subtract a flat 2 (see
# `restart_cycles`).
_LINEAX_STEP_OFFSET = 2


def corrected_cycles(raw_count, solves=1):
    """The traced counterpart of :func:`restart_cycles`, for use inside a jitted loop.

    Same correction, expressed in ``jax.numpy`` so it can run on a tracer. Kept beside
    :func:`restart_cycles` so the offset has exactly one definition.
    """
    return jnp.maximum(raw_count - _LINEAX_STEP_OFFSET * solves, 0)


def restart_cycles(raw_count: int, solves: int = 1) -> int:
    """Strip the fixed per-solve offset from a raw ``lineax`` iteration count.

    :func:`solve_linear` returns ``lineax``'s ``num_steps``, which carries a constant ``+2`` per solve,
    so an ideal one-cycle solve reports ``3`` rather than ``1``. Correcting it matters most where the
    count is smallest: at ``num_steps = 6`` over two solves, the raw number is *entirely* offset and the
    real cost is two single-cycle solves -- reading it as "six cycles" overstates the work threefold.

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
    to the desired relative tolerance. Termination is then exactly ``norm(r) <= atol * norm(b)`` -- one
    global relative test in whatever measure ``self.norm`` is, immune to the near-zero entries. The
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
    norm: Callable[[Any], jnp.ndarray] = _global_two_norm,
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
