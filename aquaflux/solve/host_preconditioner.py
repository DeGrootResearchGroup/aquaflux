"""The contract every frozen host preconditioner satisfies, and the JAX wrapper it shares.

The materialized-Jacobian preconditioners (:mod:`~aquaflux.solve.materialized_preconditioner`) hand a
jitted Krylov solve an approximate ``A^-1`` computed on the **host**. They differ entirely in how the
inverse is *fitted* to the matrix and not at all in how it is *applied*: each holds a frozen inverse, exposes it as a
``residual -> M residual`` callable through :func:`jax.pure_callback`, and reads that factorization at
call time so an in-place refresh re-preconditions the already-compiled solve.

**The contract is real and named.** ``matvec`` needs exactly two things of whatever it wraps -- how
many degrees of freedom it spans, and how to apply it (or its transpose) to a host vector -- and the
traced hierarchy inverses and the block-triangular field split already provide precisely that pair. :class:`FrozenInverse` is that pair, written down, rather than each wrapper re-deriving
``matvec`` on its own.

**Naming it also closes a class of silent failure.** A base that reads anything off ``self.inverse``
beyond this pair is making an assumption only some factorizations satisfy, and on the others the
lookup raises -- which a ``getattr`` default at the call site quietly turns into a plausible value. If a
capability is not in :class:`FrozenInverse`, do not reach for it through ``self.inverse``; give the
subclass an explicit answer instead.

**The capabilities only some inverses have are declared too, one protocol each.** Re-fitting in place to
a new operator (:class:`RefactorableInverse`) and releasing held resources (:class:`ReleasableInverse`)
are each offered by some members of the family and not others; running as a traced cycle inside an outer
trace is the third, declared beside its value type in :mod:`~aquaflux.solve.traced_cycle`. A consumer asks with ``isinstance`` against the protocol,
never with ``getattr(inverse, name, default)``: the protocol is where the capability's signature is
written down, so an inverse that offers it under another name or another signature is reported as not
offering it, rather than being probed for and silently answered "no".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


@runtime_checkable
class FrozenInverse(Protocol):
    """A frozen inverse living on the host: how big it is, and how to apply it.

    Deliberately the smallest pair that :meth:`HostPreconditioner.matvec` needs, so that everything able
    to serve as a preconditioner's frozen inverse can satisfy it -- a triangular factorization, a
    complete factorization, a multigrid V-cycle, or a block-triangular composition of any of those.
    Anything richer belongs on the concrete class, not here.
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


@runtime_checkable
class RefactorableInverse(FrozenInverse, Protocol):
    """A frozen inverse that can be re-fitted to a new operator in place.

    What a mid-march refresh needs: the preconditioner object rides as a static field of a compiled
    solve, so a refresh must mutate the inverse it holds rather than replace it, and an inverse that
    cannot do so cannot be refreshed at all -- only rebuilt, which recompiles the solve.
    """

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-fit to ``block``, IN PLACE, keeping this object's identity.

        Parameters
        ----------
        block : scipy.sparse matrix
            The new operator in the raw field-major form the inverse was built from, of the shape it was
            built at, shape ``(n_dofs, n_dofs)``.
        """
        ...


@runtime_checkable
class ReleasableInverse(Protocol):
    """An inverse holding resources (a host solver's handles) that it can release on request.

    Garbage collection releases them eventually, but a caller building several preconditioners in turn
    -- each holding a copy of a large coupled operator and its factors -- needs the release to happen on
    its own schedule.
    """

    def destroy(self) -> None:
        """Release the held resources. The object must not be used afterwards."""
        ...


def release(inverse: object) -> None:
    """Release ``inverse``' held resources if it holds any; do nothing otherwise.

    Parameters
    ----------
    inverse : object
        A frozen inverse. Released when it is a :class:`ReleasableInverse`.
    """
    if isinstance(inverse, ReleasableInverse):
        inverse.destroy()


def require_refactorable(inverse: object, owner: str) -> RefactorableInverse:
    """``inverse``, checked to be re-fittable in place, for a refresh that is about to re-fit it.

    Parameters
    ----------
    inverse : object
        The inverse a refresh is about to re-fit.
    owner : str
        What is being refreshed, named in the error.

    Returns
    -------
    RefactorableInverse
        ``inverse`` itself.

    Raises
    ------
    TypeError
        If ``inverse`` offers no ``refactor_block``. An injected inverse need not be refreshable, so this
        is raised when a refresh is attempted rather than when the inverse is built.
    """
    if not isinstance(inverse, RefactorableInverse):
        raise TypeError(
            f"{type(inverse).__name__} offers no refactor_block, so {owner} cannot be refreshed "
            "mid-march; rebuild it instead, or inject an inverse that can re-fit in place."
        )
    return inverse


class HostPreconditioner:
    """A frozen host inverse exposed to a jitted Krylov solve, shared by the whole family.

    Not an :class:`equinox.Module`: the inverse is a host object, so an instance is
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
    inverse : FrozenInverse
        The frozen inverse. **Rebind it in place** to refresh -- never replace the preconditioner object,
        whose identity is part of the compiled solve's pytree structure.
    """

    inverse: FrozenInverse

    def __init__(self, inverse: FrozenInverse) -> None:
        self.inverse = inverse

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
        The callback reads :attr:`inverse` **at call time** rather than capturing it, so a
        ``refresh_in_place`` between two calls of the returned matvec is picked up without rebuilding the
        callback -- that indirection is the whole reason a mid-march refresh does not recompile the solve.
        The degree-of-freedom count is fixed by the mesh, so the output shape is stable across a refresh
        and the callback's result shape can be resolved once, here.
        """
        shape = jax.ShapeDtypeStruct((self.inverse.n_dofs,), jnp.float64)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                lambda r: self.inverse.apply(r, transpose=transpose), shape, residual
            )

        return apply

    def destroy(self) -> None:
        """Release the frozen inverse's held resources, if it holds any (:func:`release`)."""
        release(self.inverse)
