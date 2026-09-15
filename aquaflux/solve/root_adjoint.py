"""The implicit-function-theorem (IFT) adjoint of a root, attached however the root was found.

A converged state ``phi*(theta)`` is defined implicitly by ``R(phi*, theta) = 0``, so its derivative
follows from the implicit function theorem rather than from the iteration that produced it:

    dphi*/dtheta = -(dR/dphi)^{-1} (dR/dtheta),

and the reverse-mode gradient of a loss ``L(phi*)`` with cotangent ``v = dL/dphi*`` is

    dL/dtheta = -(dR/dtheta)^T lambda,   where   (dR/dphi)^T lambda = v.

That is **one transpose linear solve** at the root, independent of how many iterations reached it.
Nothing here iterates: :func:`root_adjoint` takes a root found by any means and returns it unchanged,
carrying this derivative. The Newton iteration that finds the root is therefore never on the tape and
is free to stop on a data-dependent test, observe itself, or rebuild a preconditioner part way through,
since none of that reaches the gradient.

The adjoint is defined for reverse mode only (``jax.grad`` / ``jax.vjp``), which is what a scalar
objective through a solve needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx

from .linear import default_linear_solver, solve_linear

__all__ = ["TransposedPreconditioner", "root_adjoint"]


@dataclasses.dataclass(frozen=True)
class TransposedPreconditioner:
    """An adjoint-preconditioner factory whose output is **already** the transpose ``M^T``.

    The generic adjoint machinery derives the transpose preconditioner from the forward one with
    :func:`jax.linear_transpose`, which works only when the forward preconditioner is a traceable
    JAX operation (an algebraic-multigrid V-cycle is). A preconditioner applied through a host
    callback -- the monolithic incomplete-LU factorization, whose triangular solve runs in ``scipy``
    via :func:`jax.pure_callback` -- cannot be transposed that way; instead it supplies its own
    transpose directly (the same factorization applied with a transposed triangular solve). Wrapping
    the factory in this marker tells :func:`root_adjoint` to apply its output as-is rather than
    transpose it.

    Parameters
    ----------
    factory : callable
        The ``state -> M^T`` factory, returning the transpose preconditioner matvec directly.

    Notes
    -----
    A **frozen dataclass**, so two wrappers around the same factory compare equal. This rides in a
    forward step's ``adjoint_preconditioner_factory``, a *static* field and therefore part of the
    compiled step's cache key; identity comparison there means every rebuild recompiles the whole
    coupled solve. Equality is only as good as the wrapped factory's -- pass a value object, not a
    lambda (see :class:`~aquaflux.turbulence.coupled.FrozenTransposeFactory`).
    """

    factory: Callable[[Any], Callable[[Any], Any]]

    def __call__(self, state: Any) -> Callable[[Any], Any]:
        return self.factory(state)


def root_adjoint(
    residual_fn: Callable[[jnp.ndarray, object], jnp.ndarray],
    root: jnp.ndarray,
    theta: object,
    *,
    adjoint_solver: lx.AbstractLinearSolver | None = None,
    adjoint_preconditioner: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]
    | None = None,
) -> jnp.ndarray:
    """``root``, unchanged, carrying the derivative of the root of ``residual_fn`` with respect to ``theta``.

    Differentiating the result with respect to ``theta`` gives the implicit-function-theorem
    derivative of the root: one transpose linear solve at ``root``, independent of the iteration that
    produced it. Whatever derivative ``root`` itself carries is discarded -- its dependence on
    ``theta`` is supplied here instead -- so the iteration may run on ``stop_gradient`` copies, and
    should, since it is never differentiated.

    **It does not check that ``root`` is a root.** The derivative is valid only where
    ``residual_fn(root, theta) = 0``; at any other state the transpose solve is still well posed and
    returns a wrong gradient with no ``NaN`` to flag it. The caller owns the convergence test, because
    only the caller knows the tolerance and the measure the iteration was judged by.

    Parameters
    ----------
    residual_fn : callable
        ``(phi, theta) -> R``, the residual whose root ``root`` is. Treated as a constant: it must not
        close over values being differentiated, which belong in ``theta``.
    root : jnp.ndarray
        The converged state, shape ``(n,)``.
    theta : pytree
        The differentiable parameters ``residual_fn`` depends on. Every leaf must be a JAX type.
    adjoint_solver : lineax.AbstractLinearSolver or None
        The solver for the transpose linear solve. ``None`` uses the tight
        :func:`~aquaflux.solve.default_linear_solver`: this one solve sets the gradient's accuracy.
    adjoint_preconditioner : callable or None
        A factory ``state -> M`` for the **forward** preconditioner, ``M`` approximating
        ``(dR/dphi)^{-1}`` at ``state``. It is built at ``root`` and transposed with
        :func:`jax.linear_transpose`; wrap the factory in :class:`TransposedPreconditioner` when it
        already returns ``M^T`` (a host-callback preconditioner, which cannot be transposed that way).
        ``None`` solves unpreconditioned. It changes how fast the transpose solve converges, not the
        gradient it converges to.

    Returns
    -------
    jnp.ndarray
        ``root``, shape ``(n,)``.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from aquaflux.solve import root_adjoint
    >>> def residual(x, theta):
    ...     return x**2 - theta
    >>> theta = jnp.array([4.0])
    >>> root = jnp.sqrt(jax.lax.stop_gradient(theta))  # found by any means
    >>> jax.grad(lambda t: jnp.sum(root_adjoint(residual, root, t)))(theta)  # 1 / (2 sqrt(theta))
    Array([0.25], dtype=float64)
    """
    solver = adjoint_solver if adjoint_solver is not None else default_linear_solver()
    return _root_adjoint(residual_fn, root, theta, solver, adjoint_preconditioner)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def _root_adjoint(residual_fn, root, theta, adjoint_solver, adjoint_preconditioner):
    del residual_fn, theta, adjoint_solver, adjoint_preconditioner
    return root


def _root_adjoint_fwd(residual_fn, root, theta, adjoint_solver, adjoint_preconditioner):
    del residual_fn, adjoint_solver, adjoint_preconditioner
    return root, (root, theta)


def _transposed_preconditioner(preconditioner, root, example):
    """Transpose ``M^T`` of the forward preconditioner built at ``root``, for the transpose solve.

    The forward ``M = preconditioner(root)`` approximates ``J^{-1}``; the adjoint solves the
    transpose system ``J^T lambda = v``, for which ``M^T ~ J^{-T}`` is the consistent
    preconditioner, obtained by transposing the (linear) preconditioner matvec with
    :func:`jax.linear_transpose`. It is applied on whichever side
    :func:`~aquaflux.solve.linear.solve_linear` defaults to (the right), which is a different
    bracketing from transposing the forward *preconditioned operator* — the two have the same
    spectrum, so this changes the Krylov residual measured, not the converged gradient.
    It is mesh-independent wherever ``M`` is -- the adjoint GMRES iteration count stays flat under
    refinement instead of growing with the system size. ``None`` in, ``None`` out. A
    :class:`TransposedPreconditioner` factory already returns ``M^T`` (a callback preconditioner that
    :func:`jax.linear_transpose` cannot handle), so it is applied directly.
    """
    if preconditioner is None:
        return None
    if isinstance(preconditioner, TransposedPreconditioner):
        return preconditioner(root)
    m = preconditioner(root)
    transpose = jax.linear_transpose(m, example)
    return lambda u: transpose(u)[0]


def _root_adjoint_bwd(residual_fn, adjoint_solver, adjoint_preconditioner, residuals, cotangent):
    root, theta = residuals
    # Transpose Jacobian solve: (dR/dphi)^T lambda = cotangent, preconditioned by M^T so the adjoint
    # solve is mesh-independent (unpreconditioned it grows with the system size).
    _, vjp_phi = jax.vjp(lambda p: residual_fn(p, theta), root)
    preconditioner = _transposed_preconditioner(adjoint_preconditioner, root, cotangent)
    lam, _ = solve_linear(
        lambda u: vjp_phi(u)[0], cotangent, solver=adjoint_solver, preconditioner=preconditioner
    )
    # Parameter cotangent -(dR/dtheta)^T lambda: negate lambda so no pytree (float0) negation. The
    # root's own cotangent is zero: its dependence on theta is the one supplied above.
    _, vjp_theta = jax.vjp(lambda th: residual_fn(root, th), theta)
    (theta_cotangent,) = vjp_theta(-lam)
    return jnp.zeros_like(root), theta_cotangent


_root_adjoint.defvjp(_root_adjoint_fwd, _root_adjoint_bwd)
