"""The hybrid initial condition of a transported scalar: the harmonic interpolant of its boundary values.

A species, a temperature or a passive tracer has no analogue of a potential flow to start from, but it has
the same need: a smooth interior consistent with what is imposed at the boundary, so a globalized solve
does not start from a uniform guess that disagrees with an inlet. The harmonic field -- the solution of
``div(grad c) = 0`` for the scalar's own boundary data -- is that interpolant
(:func:`~aquaflux.flow.laplace_field`), the same one the turbulence fields are seeded with. By the
maximum principle it stays between the smallest and largest prescribed boundary values, so it cannot
start a bounded scalar outside its physical range.

It is reconstructed with the scalar's own gradient scheme, taken from the assembler that owns it: an
initializer that chose its own would put one discretization in the initial condition and another in the
equation being solved.
"""

from __future__ import annotations

import jax.numpy as jnp

from aquaflux.boundary import Dirichlet, DirichletField
from aquaflux.flow.initialization import laplace_field
from aquaflux.initialization import hybrid_initialize

from .scalar import ScalarTransport


@hybrid_initialize.register(ScalarTransport)
def _initialize_scalar(transport: ScalarTransport, **settings: object) -> jnp.ndarray:
    """The harmonic interpolant of ``transport``'s Dirichlet boundary values, shape ``(n_cells,)``.

    A scalar with **no** prescribed value anywhere (every patch zero-gradient or a flux) has nothing to
    interpolate -- the pure-Neumann Laplacian is singular and any constant solves it -- so it starts at
    zero, the neutral level, rather than at a value the solve made up.

    Parameters
    ----------
    transport : ScalarTransport
        The scalar's assembler, supplying its mesh, boundary conditions and gradient scheme.

    Returns
    -------
    jnp.ndarray
        The initial field.

    Raises
    ------
    TypeError
        If any setting is given; this initializer takes none.
    """
    if settings:
        raise TypeError(
            f"the scalar initializer takes no settings, got {sorted(settings)}. The harmonic field "
            "reads everything it needs from the assembler."
        )
    prescribes_a_value = any(
        isinstance(closure, Dirichlet | DirichletField)
        for closure in transport.boundary.conditions.values()
    )
    if not prescribes_a_value:
        return jnp.zeros(transport.mesh.n_cells)
    field, _ = laplace_field(
        transport.mesh,
        transport.geometry,
        transport.boundary,
        gradient_scheme=transport.gradient_scheme,
    )
    return field
