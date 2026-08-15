"""The contract every frozen host preconditioner satisfies, and the JAX wrapper it shares.

Three preconditioners in this package hand a jitted Krylov solve an approximate ``A^-1`` computed on the
**host**: the threshold-ILU (:mod:`~aquaflux.solve.ilut_preconditioner`), the complete LU
(:mod:`~aquaflux.solve.lu_preconditioner`) and the algebraic-multigrid V-cycle
(:mod:`~aquaflux.solve.amg_preconditioner`). They differ entirely in how the inverse is *fitted* to the
matrix and not at all in how it is *applied*: each holds a frozen factorization, exposes it as a
``residual -> M residual`` callable through :func:`jax.pure_callback`, and reads that factorization at
call time so an in-place refresh re-preconditions the already-compiled solve.

**The contract was real and unnamed.** ``matvec`` needs exactly two things of whatever it wraps -- how
many degrees of freedom it spans, and how to apply it (or its transpose) to a host vector -- and seven
classes in this package already provide precisely that pair: the three factorizations above, the
V-cycle itself, the framework-native hierarchy inverse, both block-triangular field splits, and the
Vanka smoother. Nothing declared it, so each wrapper re-derived it: ``matvec`` was written out three
times, byte for byte, and the field split obtained it by *subclassing a concrete sibling* rather than a
contract. :class:`HostFactors` is that pair, written down.

**Naming it also closes a class of silent failure.** A base that reads anything off ``self.factors``
beyond this pair is making an assumption only some factorizations satisfy -- which is how a
``has_native_solve`` lookup came to raise on the field split while a ``getattr`` default at the call
site quietly turned the exception into ``False``. If a capability is not in :class:`HostFactors`, do
not reach for it through ``self.factors``; give the subclass an explicit answer instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np


@runtime_checkable
class HostFactors(Protocol):
    """A frozen inverse living on the host: how big it is, and how to apply it.

    Deliberately the smallest pair that :meth:`HostPreconditioner.matvec` needs, so that everything able
    to serve as a preconditioner's frozen inverse can satisfy it -- a triangular factorization, a
    complete factorization, a multigrid V-cycle, a block-triangular composition of any of those, or a
    patch smoother. Anything richer belongs on the concrete class, not here.
    """

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom the inverse spans -- the length of the vectors it maps."""
        ...

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply ``M ~= A^-1`` (or ``M^T`` when ``transpose``) to a host vector.

        The transpose is what the adjoint's transpose linear solve calls, so every implementation owes
        one; a factorization that cannot supply it cheaply is not usable as an adjoint preconditioner.
        """
        ...


class HostPreconditioner:
    """A frozen host inverse exposed to a jitted Krylov solve, shared by the whole family.

    Not an :class:`equinox.Module`: the factorization is a host ``scipy``/PETSc object, so an instance is
    held by the caller and captured in the :func:`jax.pure_callback` closure rather than threaded through
    the jit as a traced argument. It rides as a **static** field of the shift policy, which is what makes
    an in-place refresh a compilation-cache hit rather than a recompile.

    A subclass supplies the construction and the refresh -- how the inverse is fitted, and what a refresh
    re-fits -- and inherits the application. Those genuinely differ: an incomplete factorization, a
    complete one and a multigrid hierarchy are built from different inputs and refreshed at different
    costs, so ``build`` and ``refresh_in_place`` stay per-class rather than being unified behind a
    signature that would be the union of three.

    Attributes
    ----------
    factors : HostFactors
        The frozen inverse. **Rebind it in place** to refresh -- never replace the preconditioner object,
        whose identity is part of the compiled solve's pytree structure.
    """

    factors: HostFactors

    def __init__(self, factors: HostFactors) -> None:
        self.factors = factors

    def matvec(self, *, transpose: bool = False) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner as a JAX callable ``residual -> M residual`` (or ``M^T``).

        Parameters
        ----------
        transpose : bool
            Return ``M^T`` (for the adjoint transpose solve) instead of ``M``.

        Returns
        -------
        callable
            A :func:`jax.pure_callback` matvec applying the current inverse on the host.

        Notes
        -----
        The callback reads :attr:`factors` **at call time** rather than capturing it, so a
        ``refresh_in_place`` between two calls of the returned matvec is picked up without rebuilding the
        callback -- that indirection is the whole reason a mid-march refresh does not recompile the solve.
        The degree-of-freedom count is fixed by the mesh, so the output shape is stable across a refresh
        and the callback's result shape can be resolved once, here.
        """
        shape = jax.ShapeDtypeStruct((self.factors.n_dofs,), jnp.float64)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda r: self.factors.apply(r, transpose=transpose), shape, residual
            )

        return apply
