"""Hybrid initialization of any problem: one entry point, registered per problem type.

A globalized Newton solve is a local method, and what it converges from matters. The *hybrid* initial
condition is the cheap physical start a few linear solves buy -- a potential-flow velocity, a harmonic
interpolant of a scalar's boundary values, a closure's equilibrium levels -- that lands the solve in its
basin instead of at a raw cold start.

What such a start *is* depends entirely on the problem, so this module owns none of it. It owns the one
public name, :func:`hybrid_initialize`, and the dispatch on the problem's type; each package registers what
it knows how to initialize, beside the class it belongs to::

    @hybrid_initialize.register(MomentumContinuity)
    def _(momentum, **settings): ...

so adding a problem type is one registration in its own package, with no central function to edit and no
physics imported here. A problem with nothing registered is refused by name rather than handed a
guess. What comes back is the problem's own physical starting fields -- a flat flow state for a flow, one
field for a scalar, ``(flow, k, omega)`` for a coupled turbulent flow -- in the form its solve takes.
"""

from __future__ import annotations

import functools

__all__ = ["hybrid_initialize"]


@functools.singledispatch
def hybrid_initialize(problem: object, **settings: object):
    """A cheap, physical initial condition for ``problem``, from the initializer its type registered.

    Parameters
    ----------
    problem : object
        The assembled problem to initialize: a :class:`~aquaflux.flow.MomentumContinuity` (potential
        flow), a :class:`~aquaflux.transport.ScalarTransport` (the harmonic interpolant of its boundary
        values), a :class:`~aquaflux.turbulence.CoupledRANS` (potential flow plus the closure's fields), or
        any type a package has registered.
    **settings
        Settings of the registered initializer, if it has any (a turbulence closure's floors, say). A
        type that takes none refuses any given.

    Returns
    -------
    object
        The problem's starting fields, in the form its solve takes.

    Raises
    ------
    TypeError
        If nothing is registered for the type of ``problem``, or a setting is given the registered
        initializer does not take.
    """
    raise TypeError(
        f"no hybrid initializer is registered for {type(problem).__name__}. Each package registers the "
        "types it owns when it is imported (`import aquaflux.flow`, `aquaflux.transport`, "
        "`aquaflux.turbulence`); a new problem type registers its own with "
        "`@hybrid_initialize.register(TheType)`."
    )
