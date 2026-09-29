"""Unit tests for the pseudo-transient shift-basis strategy (pure per-cell combination, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.solve import LocalCourantBasis


def test_full_weight_is_the_operator_diagonal() -> None:
    """``w = 1`` sums both buckets, i.e. the full operator diagonal (uniform-relaxation default)."""
    convective = jnp.array([1.0, 2.0, 3.0])
    dissipative = jnp.array([0.5, 4.0, 0.25])
    got = LocalCourantBasis().local_diagonal(convective, dissipative)
    assert jnp.array_equal(got, convective + dissipative)


def test_zero_weight_is_the_pure_convective_time_step() -> None:
    """``w = 0`` drops the dissipative bucket -- the pure convective local time step."""
    convective = jnp.array([1.0, 2.0, 3.0])
    dissipative = jnp.array([0.5, 4.0, 0.25])
    got = LocalCourantBasis(dissipative_weight=0.0).local_diagonal(convective, dissipative)
    assert jnp.array_equal(got, convective)


def test_intermediate_weight_down_weights_the_dissipative_bucket() -> None:
    """An intermediate ``w`` keeps a fraction of the dissipative stiffness."""
    convective = jnp.array([1.0, 2.0])
    dissipative = jnp.array([4.0, 8.0])
    got = LocalCourantBasis(dissipative_weight=0.25).local_diagonal(convective, dissipative)
    assert jnp.allclose(got, convective + 0.25 * dissipative)


def test_the_basis_is_non_negative_for_non_negative_buckets() -> None:
    """Both buckets are ``>= 0``, so the base shift diagonal is too (a valid pseudo-time scale)."""
    convective = jnp.array([0.0, 3.0, 1.5])
    dissipative = jnp.array([2.0, 0.0, 0.5])
    for weight in (0.0, 0.5, 1.0):
        got = LocalCourantBasis(dissipative_weight=weight).local_diagonal(convective, dissipative)
        assert bool(jnp.all(got >= 0.0))


def test_every_builder_that_shifts_a_block_defaults_to_the_one_shift_basis() -> None:
    """The default shift basis is one object, and every default that names a basis is that object.

    Each module used to construct its own ``LocalCourantBasis()``, so changing the default meant
    changing it in each, with nothing relating the copies. An equal-but-separate instance would still
    pass an equality check, so the defaults are compared by identity.
    """
    import dataclasses
    import inspect

    from aquaflux.flow.continuation import MomentumShiftPolicy
    from aquaflux.solve import DEFAULT_SHIFT_BASIS
    from aquaflux.turbulence import SSTTurbulence
    from aquaflux.turbulence.coupled import (
        CoupledShiftPolicy,
        _coupled_shift_policy,
        _resolved_shift,
    )

    def field_default(module_class):
        (field,) = (f for f in dataclasses.fields(module_class) if f.name == "shift_basis")
        return field.default

    defaults = {
        "SSTTurbulence.k_shift_policy": inspect.signature(SSTTurbulence.k_shift_policy)
        .parameters["shift_basis"]
        .default,
        "SSTTurbulence.omega_shift_policy": inspect.signature(SSTTurbulence.omega_shift_policy)
        .parameters["shift_basis"]
        .default,
        "_coupled_shift_policy": inspect.signature(_coupled_shift_policy)
        .parameters["shift_basis"]
        .default,
        "_resolved_shift": _resolved_shift(None)[0],
        "CoupledShiftPolicy.shift_basis": field_default(CoupledShiftPolicy),
        "MomentumShiftPolicy.shift_basis": field_default(MomentumShiftPolicy),
    }
    assert isinstance(DEFAULT_SHIFT_BASIS, LocalCourantBasis)
    assert DEFAULT_SHIFT_BASIS == LocalCourantBasis(), "the default is the full operator diagonal"
    assert {name for name, value in defaults.items() if value is not DEFAULT_SHIFT_BASIS} == set()
