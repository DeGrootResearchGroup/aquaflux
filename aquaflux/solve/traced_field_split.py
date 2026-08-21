"""The block-triangular field split as a **traced** map, so a bundle of traced blocks stays on device.

:class:`~aquaflux.solve.BlockTriangularFieldSplit` composes its two block inverses in ``numpy`` on the
host, and is reached from the jitted Krylov solve through a :func:`jax.pure_callback`. That is the right
shape when a block inverse is a host factorization (an incomplete or complete LU), because the work
itself is on the host and the vector has to travel anyway.

It is the wrong shape when **both** blocks are traced multigrid cycles, which is what the shipped
flow-plus-turbulence bundle now is. There the composition layer is the only thing on the host, and it
forces the round trip anyway::

    device -> host          the callback
    host -> device -> host  the leading cycle, entered and left through ``jnp.asarray``/``np.asarray``
    host                    the coupling product, in ``scipy``
    host -> device -> host  the trailing cycle, likewise
    host -> device          the callback returns

Six transfers of the whole state vector per preconditioner application, each a synchronization point
that drains the device pipeline -- on an accelerator that cost is paid against a cycle that got faster,
so it grows as a share of the solve rather than shrinking.

This module removes it. The algebra is identical to the host split's; what changes is that the vector is
a traced array throughout, the coupling is a :class:`~aquaflux.solve.multigrid._CsrOperator` rather than
a ``scipy`` matrix, and the two inverses are entered through their traced cycle rather than through
their host-shaped ``apply``. Nothing crosses the device boundary between the Krylov solve and the
preconditioned vector it gets back.

**Both blocks must be traced.** An :class:`~aquaflux.solve.IluSmoothedInverse` is a sequential
triangular solve and has no traced cycle to offer; a bundle including one keeps the host split, which is
correct rather than a limitation -- that work genuinely belongs on a CPU.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.sparse as sp

from .field_split import FieldGroups
from .multigrid import _CsrOperator

__all__ = ["TracedFieldSplit", "traced_field_split"]


class TracedFieldSplit(eqx.Module):
    """A block-triangular approximate inverse over a two-group field partition, applied on device.

    The traced counterpart of :class:`~aquaflux.solve.BlockTriangularFieldSplit`, with the same
    approximation and the same ordering convention: one group is solved first, its effect on the other
    group's equations is subtracted through the retained coupling block, and the other group is solved
    against that corrected right-hand side. The discarded triangle is the same one.

    An ``equinox.Module`` rather than a plain object, unlike the host split: every array it holds is a
    traced leaf, so the whole split rides into a jitted solve as an argument. Re-deriving it at a new
    operator on unchanged shapes is then a compilation-cache hit rather than a retrace -- the same
    property, and for the same reason, that :class:`~aquaflux.solve.multigrid._CsrOperator` has.

    Attributes
    ----------
    leading_cycle, trailing_cycle : callable
        Each group's traced solve, ``jnp.ndarray -> jnp.ndarray``. Static, because they are functions.
    coupling : _CsrOperator
        The retained coupling block. No transpose is stored: ``jax.linear_transpose`` derives ``M^T``
        from the forward code, so there is nothing to keep in step with it.
    groups : FieldGroups
        The partition. Static -- it carries only sizes.
    leading_first : bool
        Which group is solved first. Static, so :meth:`apply` never branches on it at run time.
    """

    leading_cycle: object = eqx.field(static=True)
    trailing_cycle: object = eqx.field(static=True)
    coupling: _CsrOperator
    groups: FieldGroups = eqx.field(static=True)
    leading_first: bool = eqx.field(static=True)

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom this inverse is defined over."""
        return self.groups.n_dofs

    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        """Apply the block-triangular inverse ``M``, entirely on device.

        Parameters
        ----------
        residual : jnp.ndarray
            The field-major right-hand side, shape ``(n_dofs,)``.

        Returns
        -------
        jnp.ndarray
            The preconditioned vector, shape ``(n_dofs,)``.

        Notes
        -----
        There is deliberately **no** ``transpose`` argument, unlike
        :meth:`~aquaflux.solve.BlockTriangularFieldSplit.apply`. That one needs it because it is
        ``numpy``: transposing a block-triangular inverse means reversing the solve order, transposing
        the coupling, *and* transposing each block inverse, and on the host each of those has to be
        arranged by hand. This map is traced and linear, so :func:`jax.linear_transpose` produces
        ``M^T`` from this code exactly -- which is how the adjoint already obtains it
        (:func:`aquaflux.solve.implicit._adjoint_preconditioner`), and how
        :class:`~aquaflux.solve.HierarchyBlockInverse` obtains its own.

        The two groups are **contiguous** ranges of a field-major vector (see
        :class:`~aquaflux.solve.FieldGroups`), so this slices rather than gathers.
        """
        split = self.groups.n_leading_dofs
        lead, trail = residual[:split], residual[split:]
        # `leading_first` is static, so only one of these bodies is ever traced.
        if self.leading_first:
            first = self.leading_cycle(lead)
            return jnp.concatenate([first, self.trailing_cycle(trail - self.coupling.apply(first))])
        first = self.trailing_cycle(trail)
        return jnp.concatenate([self.leading_cycle(lead - self.coupling.apply(first)), first])

    def matvec(self):
        """The preconditioner matvec, as the ``residual -> M residual`` callable a Krylov solve takes.

        The traced counterpart of :meth:`~aquaflux.solve.HostPreconditioner.matvec`, and deliberately
        the same shape -- so a caller selects between a host bundle and a traced one by construction
        rather than by branching. Where that one returns a :func:`jax.pure_callback`, this returns the
        cycle itself, so the Krylov solve calls straight into it.

        Returned as a plain closure rather than as the bound :meth:`apply`. ``jax.jit`` keys its cache
        on the callable, and a bound method of an ``equinox.Module`` carries the module into that key --
        which raises here, since this one holds array leaves and is deliberately not hashable.
        """

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            return self.apply(residual)

        return apply


def traced_field_split(
    matrix: sp.spmatrix,
    groups: FieldGroups,
    leading,
    trailing,
    *,
    flow_first: bool = True,
) -> TracedFieldSplit:
    """Build a :class:`TracedFieldSplit` from an assembled operator and two traced block inverses.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major operator, shape ``(n_dofs, n_dofs)``. Only the retained off-diagonal
        block is read here; each inverse was fitted to its own diagonal block by its own builder.
    groups : FieldGroups
        The partition.
    leading, trailing : object
        The two block inverses. Each must expose a **traced** cycle -- ``_solve(vector)`` returning a
        traced array -- which :class:`~aquaflux.solve.HierarchyBlockInverse` and
        :class:`~aquaflux.solve.AirBlockInverse` both do.
    flow_first : bool
        Solve the leading group first (the default), retaining the trailing-by-leading coupling.

    Returns
    -------
    TracedFieldSplit

    Raises
    ------
    AttributeError
        If either inverse offers no traced cycle -- a host factorization cannot be composed on device,
        and silently falling back to the host split would hide the round trip this exists to remove.
    """
    for name, inverse in (("leading", leading), ("trailing", trailing)):
        if not hasattr(inverse, "_solve"):
            raise AttributeError(
                f"the {name} inverse {type(inverse).__name__} offers no traced cycle, so this bundle "
                "cannot be composed on device; use BlockTriangularFieldSplit for a host inverse."
            )
    _, leading_trailing, trailing_leading, _ = groups.blocks(matrix)
    retained = trailing_leading if flow_first else leading_trailing
    return TracedFieldSplit(
        leading_cycle=leading._solve,
        trailing_cycle=trailing._solve,
        coupling=_CsrOperator.from_scipy(sp.csr_matrix(retained)),
        groups=groups,
        leading_first=flow_first,
    )


def contains_host_callback(fn, example: jnp.ndarray) -> bool:
    """Whether tracing ``fn`` at ``example`` emits a host callback -- the property this module is for.

    A round trip to the host is invisible in a result and shows up only as time, so the absence of one
    is worth asserting rather than believing.

    Parameters
    ----------
    fn : callable
        A traced function of one array.
    example : jnp.ndarray
        An input of the shape and dtype to trace at.

    Returns
    -------
    bool
        ``True`` if a callback primitive appears anywhere in the jaxpr, nested calls included. The
        printed form is read rather than the equation tree walked, because a callback can sit inside a
        closed call, a scan or a cond, and the text carries all of them.
    """
    return "callback" in str(jax.make_jaxpr(fn)(example))
