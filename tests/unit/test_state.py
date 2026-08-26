"""Unit tests for the shared flat field-major state layout — mesh-free, array-free."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import CellFields, FieldLayout, GlobalDofs, SubLayout


def _flow(n_cells: int = 3, dim: int = 2) -> FieldLayout:
    return FieldLayout.cell_fields(n_cells, velocity=dim, pressure=1)


def _coupled(n_cells: int = 3, dim: int = 2) -> FieldLayout:
    return FieldLayout(
        n_cells,
        (SubLayout("flow", _flow(n_cells, dim)), CellFields("k", 1), CellFields("omega", 1)),
    )


class TestShape:
    def test_sizes_offsets_and_total_agree(self) -> None:
        layout = _coupled(n_cells=5, dim=3)
        assert layout.sizes == (4 * 5, 5, 5)
        assert layout.offsets == (0, 20, 25)
        assert layout.size == 30
        assert layout.n_fields == 6  # u, v, w, p, k, omega

    def test_a_nested_layout_contributes_its_own_fields_not_one(self) -> None:
        """A nested block counts the fields it holds, not one -- what `field_offset` sums over."""
        assert _coupled(dim=2).n_fields == 5  # u, v, p, k, omega
        assert _coupled(dim=3).n_fields == 6

    def test_slices_name_each_blocks_own_range(self) -> None:
        layout = _coupled(n_cells=4, dim=2)
        assert layout.slice_of("flow") == slice(0, 12)
        assert layout.slice_of("k") == slice(12, 16)
        assert layout.slice_of("omega") == slice(16, 20)

    def test_field_offset_counts_the_fields_before_a_block(self) -> None:
        layout = _coupled(dim=3)
        assert layout.field_offset("flow") == 0
        assert layout.field_offset("k") == 4  # u, v, w, p
        assert layout.field_offset("omega") == 5

    def test_field_dofs_is_the_field_major_stride(self) -> None:
        layout = _coupled(n_cells=7, dim=2)
        assert layout.field_dofs(3) == 21

    def test_an_unknown_block_is_named_in_the_error(self) -> None:
        with pytest.raises(KeyError, match="velocity"):
            _flow().slice_of("temperature")

    def test_the_block_names_are_the_flat_vector_order(self) -> None:
        assert _coupled().names == ("flow", "k", "omega")


class TestPacking:
    def test_pack_unpack_round_trips(self) -> None:
        layout = _coupled(n_cells=5, dim=2)
        flow = jnp.arange(15.0)
        k = 10.0 + jnp.arange(5.0)
        omega = 100.0 + jnp.arange(5.0)
        state = layout.pack(flow, k, omega)
        assert state.shape == (layout.size,)
        f, kk, oo = layout.unpack(state)
        assert jnp.array_equal(f, flow)
        assert jnp.array_equal(kk, k)
        assert jnp.array_equal(oo, omega)

    def test_a_multi_field_block_reads_out_component_last(self) -> None:
        layout = _flow(n_cells=3, dim=2)
        velocity = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        pressure = jnp.array([7.0, 8.0, 9.0])
        state = layout.pack(velocity, pressure)
        # Field-major within the block: [u_0..u_2, v_0..v_2, p_0..p_2].
        np.testing.assert_array_equal(np.asarray(state), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        v2, p2 = layout.unpack(state)
        assert v2.shape == (3, 2)
        assert p2.shape == (3,)
        assert jnp.array_equal(v2, velocity)

    def test_a_single_field_block_is_not_wrapped_in_a_length_one_axis(self) -> None:
        (pressure,) = FieldLayout.cell_fields(4, pressure=1).unpack(jnp.arange(4.0))
        assert pressure.shape == (4,)

    def test_a_nested_block_reads_out_flat_for_its_owner_to_unpack(self) -> None:
        """The point of nesting: the sub-vector comes back whole, for its own layout to split."""
        layout = _coupled(n_cells=4, dim=3)
        flow, _, _ = layout.unpack(jnp.arange(float(layout.size)))
        assert flow.shape == (16,)
        velocity, pressure = _flow(n_cells=4, dim=3).unpack(flow)
        assert velocity.shape == (4, 3)
        assert pressure.shape == (4,)

    def test_packing_the_wrong_number_of_values_is_refused(self) -> None:
        layout = _coupled()
        with pytest.raises(ValueError, match="3 blocks"):
            layout.pack(jnp.zeros(9), jnp.zeros(3))

    def test_zeros_has_the_layouts_length(self) -> None:
        z = _coupled(n_cells=4, dim=3).zeros()
        assert z.shape == (24,)  # (3 + 3) fields over 4 cells
        assert bool(jnp.all(z == 0.0))


class TestBorderedState:
    def test_an_appended_global_block_extends_the_layout_without_touching_the_rest(self) -> None:
        coupled = _coupled(n_cells=5, dim=2)
        bordered = coupled.appended(GlobalDofs("mass_flow", 1))
        assert bordered.sizes == (*coupled.sizes, 1)
        assert bordered.size == coupled.size + 1
        assert bordered.slice_of("k") == coupled.slice_of("k")

    def test_a_global_block_belongs_to_no_field(self) -> None:
        bordered = _coupled().appended(GlobalDofs("mass_flow", 1))
        assert bordered.n_fields == _coupled().n_fields
        assert not bordered.is_cell_major
        assert _coupled().is_cell_major

    def test_the_border_round_trips_with_the_rest_of_the_state(self) -> None:
        layout = _coupled(n_cells=3, dim=2).appended(GlobalDofs("mass_flow", 1))
        parts = (jnp.arange(9.0), jnp.zeros(3), jnp.ones(3), jnp.array([2.0]))
        flow, k, omega, beta = layout.unpack(layout.pack(*parts))
        assert beta.shape == (1,)
        assert float(beta[0]) == 2.0
        assert jnp.array_equal(flow, parts[0])
        assert jnp.array_equal(k, parts[1])
        assert jnp.array_equal(omega, parts[2])


class TestConstruction:
    def test_duplicate_block_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            FieldLayout(3, (CellFields("k", 1), CellFields("k", 2)))

    def test_an_empty_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one block"):
            FieldLayout(3, ())

    def test_a_nested_layout_over_a_different_mesh_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sits in a layout over"):
            FieldLayout(3, (SubLayout("flow", _flow(n_cells=4)),))

    def test_a_non_positive_block_width_is_refused_at_the_block(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            CellFields("k", 0)
        with pytest.raises(ValueError, match="at least one dof"):
            GlobalDofs("beta", 0)

    def test_a_non_positive_cell_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="n_cells must be positive"):
            FieldLayout.cell_fields(0, k=1)


class TestPytree:
    def test_a_layout_carries_no_leaves_so_it_never_reaches_the_tape(self) -> None:
        assert jax.tree.leaves(_coupled()) == []

    def test_a_layout_is_hashable_so_it_can_ride_as_a_static_field(self) -> None:
        assert hash(_coupled()) == hash(_coupled())
        assert _coupled() == _coupled()
        assert _coupled(dim=3) != _coupled(dim=2)
