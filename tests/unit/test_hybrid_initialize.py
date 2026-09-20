"""``hybrid_initialize``: one entry point, dispatched on the problem's type.

It used to be the k--omega SST initializer under a general name: it took ``(momentum, turbulence)`` and so
could not start a laminar flow or a species at all. Now each package registers what it knows how to
initialize, and these tests pin the dispatch, each registration, and that a problem with nothing
registered is refused by name instead of being handed a guess.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import laplace_field, potential_flow
from aquaflux.initialization import hybrid_initialize
from aquaflux.schemes import CompactGreenGauss
from aquaflux.transport import ScalarTransport, effective_diffusivity
from aquaflux.turbulence import sst_initial_fields

from .test_coupled_rans import _cavity
from .test_initialization import _channel


def test_a_type_with_nothing_registered_is_refused_by_name() -> None:
    class Unregistered:
        pass

    with pytest.raises(TypeError, match="no hybrid initializer is registered for Unregistered"):
        hybrid_initialize(Unregistered())


def test_a_new_problem_type_registers_its_own_initializer() -> None:
    """Adding a problem is one registration, in its own package; nothing central is edited."""

    class Reactor:
        pass

    @hybrid_initialize.register(Reactor)
    def _(reactor, **settings):
        return ("initialized", settings)

    assert hybrid_initialize(Reactor(), gain=2.0) == ("initialized", {"gain": 2.0})


def test_a_laminar_flow_starts_from_potential_flow_exactly() -> None:
    """The flow's initializer IS ``potential_flow``, so a laminar problem no longer needs a turbulence model."""
    _, _, momentum = _channel(12, 8)
    assert bool(jnp.all(hybrid_initialize(momentum) == potential_flow(momentum)))
    with pytest.raises(TypeError, match="takes no settings"):
        hybrid_initialize(momentum, k_floor=1e-8)


def _scalar(boundary):
    mesh = _channel(16, 8)[0]
    return ScalarTransport.build(
        mesh,
        mesh.geometry(),
        effective_diffusivity(jnp.full(mesh.n_cells, 1e-2)),
        BoundaryConditions(boundary),
        FirstOrderUpwind(),
        gradient_scheme=CompactGreenGauss(),
    )


def test_a_species_starts_from_the_harmonic_interpolant_of_its_boundary_values() -> None:
    """Inlet 1, outlet 0: a smooth field between them, equal to the Laplace solve and inside [0, 1].

    The bounds are the maximum principle, and are what stops a bounded scalar starting out of range. The
    field varying across the domain is what stops this passing on a uniform stand-in.
    """
    transport = _scalar(
        {
            "left": Dirichlet(1.0),
            "right": Dirichlet(0.0),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    field = hybrid_initialize(transport)
    expected, _ = laplace_field(
        transport.mesh,
        transport.geometry,
        transport.boundary,
        gradient_scheme=transport.gradient_scheme,
    )
    assert bool(jnp.all(field == expected))
    assert float(field.min()) >= -1e-9 and float(field.max()) <= 1.0 + 1e-9
    assert float(field.max() - field.min()) > 0.5  # it interpolates, it is not a constant
    with pytest.raises(TypeError, match="takes no settings"):
        hybrid_initialize(transport, floor=0.0)


def test_a_species_with_no_prescribed_value_starts_at_zero_rather_than_singular() -> None:
    """A pure-Neumann Laplacian is singular; the neutral start is zero, not a value the solve invented.

    A prescribed FLUX is what makes the singular solve misbehave (zero-gradient patches leave a zero
    right-hand side, which any solve returns as zero), so the case uses one -- without the guard the
    inconsistent singular system returns a meaningless field.
    """
    transport = _scalar(
        {
            "left": Neumann(flux=-1.0),
            "right": ZeroGradient(),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    field = hybrid_initialize(transport)
    assert np.array_equal(np.asarray(field), np.zeros(transport.mesh.n_cells))


def test_a_coupled_turbulent_flow_starts_exactly_as_the_sst_initializer_does() -> None:
    """``hybrid_initialize(coupled)`` is the SST initializer, settings and all, behind the general name."""
    _, coupled = _cavity()
    flow, k, omega = hybrid_initialize(coupled)
    flow0, k0, omega0 = sst_initial_fields(coupled.momentum, coupled.turbulence)
    assert bool(jnp.all(flow == flow0) & jnp.all(k == k0) & jnp.all(omega == omega0))
    # A setting reaches the closure's initializer: a huge k floor lifts k, so it cannot have been dropped.
    _, lifted, _ = hybrid_initialize(coupled, k_floor=1e3)
    assert float(lifted.min()) >= 1e3 > float(k.min())
