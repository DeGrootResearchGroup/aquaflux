"""Unit tests for the coupled block-state flat-vector layout — mesh-free (Principle 1 seam)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.flow.state import BlockStateLayout


def test_pack_unpack_round_trips() -> None:
    layout = BlockStateLayout(dim=2, n_cells=3)
    velocity = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    pressure = jnp.array([7.0, 8.0, 9.0])
    v2, p2 = layout.unpack(layout.pack(velocity, pressure))
    assert jnp.allclose(v2, velocity)
    assert jnp.allclose(p2, pressure)


def test_flat_layout_is_component_blocked() -> None:
    """The flat vector is ``[u_0..u_{n-1}, v_0..v_{n-1}, p_0..p_{n-1}]`` (each block n_cells long)."""
    layout = BlockStateLayout(dim=2, n_cells=3)
    state = layout.pack(jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]), jnp.array([7.0, 8.0, 9.0]))
    np.testing.assert_array_equal(np.asarray(state), [1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_size_and_zeros() -> None:
    layout = BlockStateLayout(dim=3, n_cells=4)
    assert layout.size == 16  # (3 + 1) * 4
    z = layout.zeros()
    assert z.shape == (16,)
    assert bool(jnp.all(z == 0.0))


def test_works_in_3d() -> None:
    layout = BlockStateLayout(dim=3, n_cells=2)
    velocity = jnp.arange(6.0).reshape(2, 3)
    pressure = jnp.array([10.0, 11.0])
    v2, p2 = layout.unpack(layout.pack(velocity, pressure))
    assert jnp.allclose(v2, velocity)
    assert jnp.allclose(p2, pressure)
