"""The flat field-major state layout: named, variable-length blocks over one vector.

Every coupled system in this package stores its unknowns as **one flat vector** laid out
field-major: the whole of the first field, then the whole of the second, and so on, so degree of
freedom ``(cell i, field f)`` sits at ``f * n_cells + i``. That layout is what makes the numerics
cheap -- a group of whole fields is a contiguous *slice* rather than a gather, and an operator's
blocks under such a group are contiguous submatrices -- and it is what every consumer has to agree
on: the residual assembler that packs it, the shift policy that builds a diagonal over it, the
preconditioner that splits it, the residual measure that weights it block by block.

:class:`FieldLayout` is that agreement as one object. It holds no vector and no matrix, only the
cell count and an ordered tuple of named :class:`StateBlock`s, and it owns every piece of index
arithmetic over them, so no consumer re-derives ``f * n_cells + i`` or a block's offset inline.
Three kinds of block cover what the solvers need:

* :class:`CellFields` -- ``n_fields`` whole fields, one value per cell each. A scalar unknown is
  one field; a velocity is ``dim`` of them.
* :class:`SubLayout` -- a named sub-state that is itself a layout. Unpacking one yields the flat
  sub-vector, which is then handed to the assembler that owns it and unpacked with *its* layout.
  This is how a coupled turbulence state carries the flow assembler's own layout verbatim instead
  of restating its widths and drifting from them.
* :class:`GlobalDofs` -- degrees of freedom not attached to cells at all, such as the scalar
  multiplier a constraint borders the state with. A bordered system is then an ordinary layout
  with one more block rather than a special case threaded through every consumer.

The whole module is mesh-free and array-free: a layout is built from integers and names, so it is
testable on its own, and every layout is a pytree with **no** leaves (all fields are static), so it
rides inside a differentiated module without contributing to the tape and may be held as a static
field wherever a hashable descriptor is wanted.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

__all__ = [
    "CellFields",
    "FieldLayout",
    "GlobalDofs",
    "StateBlock",
    "SubLayout",
]


class StateBlock(eqx.Module):
    """One named, contiguous run of degrees of freedom in a flat state vector.

    A block knows only how long it is and how its own degrees of freedom are shaped when read out
    of, or written into, the flat vector. Everything positional -- where it starts, which slice it
    occupies -- belongs to the :class:`FieldLayout` holding it, so the same block description can
    sit at any position in any layout.

    Attributes
    ----------
    name : str
        The block's name, unique within its layout, static.
    """

    name: str = eqx.field(static=True)

    @abc.abstractmethod
    def size(self, n_cells: int) -> int:
        """Degrees of freedom this block occupies on a mesh of ``n_cells`` cells."""

    @property
    @abc.abstractmethod
    def n_cell_fields(self) -> int:
        """Whole per-cell fields in this block (``0`` for degrees of freedom not tied to cells)."""

    @abc.abstractmethod
    def unflatten(self, part: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """Shape this block's flat sub-vector into the form its owner works with.

        Parameters
        ----------
        part : jnp.ndarray
            This block's degrees of freedom, shape ``(size(n_cells),)``.
        n_cells : int
            Cells in the mesh, supplied by the enclosing layout -- a block does not carry a second
            copy of it.

        Returns
        -------
        jnp.ndarray
            The shaped value; each concrete block documents its own convention.
        """

    @abc.abstractmethod
    def flatten(self, value: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """The inverse of :meth:`unflatten`: this block's value back as a flat sub-vector."""


class CellFields(StateBlock):
    """``n_fields`` whole fields, one value per cell each, stored field-major within the block.

    A single field reads out as ``(n_cells,)`` and several as ``(n_cells, n_fields)`` -- the
    component-last form the rest of the package uses for a per-cell vector -- so a scalar unknown
    is never wrapped in a length-one axis a consumer would have to squeeze out again.

    Attributes
    ----------
    name : str
        The block's name, static.
    n_fields : int
        Fields in the block: ``1`` for a scalar unknown, ``dim`` for a velocity. Static.
    """

    n_fields: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.n_fields <= 0:
            raise ValueError(
                f"block {self.name!r} must hold at least one field, got {self.n_fields}"
            )

    @property
    def n_cell_fields(self) -> int:
        """Whole per-cell fields, ``n_fields``."""
        return self.n_fields

    def size(self, n_cells: int) -> int:
        """Degrees of freedom, ``n_fields * n_cells``."""
        return self.n_fields * n_cells

    def unflatten(self, part: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """``(n_cells,)`` for one field, ``(n_cells, n_fields)`` for several."""
        if self.n_fields == 1:
            return part
        return part.reshape(self.n_fields, n_cells).T

    def flatten(self, value: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """Field-major flat form of a ``(n_cells,)`` or ``(n_cells, n_fields)`` value."""
        if self.n_fields == 1:
            return value
        return value.T.reshape(-1)


class SubLayout(StateBlock):
    """A named sub-state that is itself a :class:`FieldLayout`.

    Unpacking one yields the **flat** sub-vector rather than the sub-layout's own blocks, because
    that is what its owner wants: the sub-state is handed on whole to the assembler that defines it
    (a coupled turbulence residual hands its flow sub-vector to the momentum assembler, which
    unpacks it with the very layout carried here). Nesting is what lets the outer layout say it
    carries the inner one verbatim, instead of restating its widths.

    Attributes
    ----------
    name : str
        The block's name, static.
    layout : FieldLayout
        The sub-state's own layout, static.
    """

    layout: FieldLayout = eqx.field(static=True)

    @property
    def n_cell_fields(self) -> int:
        """Whole per-cell fields, summed over the sub-layout's own blocks."""
        return self.layout.n_fields

    def size(self, n_cells: int) -> int:
        """Degrees of freedom, the sub-layout's own total."""
        return self.layout.size

    def unflatten(self, part: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """The flat sub-vector, unchanged."""
        return part

    def flatten(self, value: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """The flat sub-vector, unchanged."""
        return value


class GlobalDofs(StateBlock):
    """``count`` degrees of freedom not attached to any cell.

    The scalar multiplier a constraint borders a state with (a mass-flow forcing, a bulk unknown)
    is one of these, so a bordered system is an ordinary layout with one more block: its size, its
    slice and its share of a block-wise residual measure all come from the same arithmetic as every
    cell field, instead of being appended by hand at each consumer.

    Attributes
    ----------
    name : str
        The block's name, static.
    count : int
        Degrees of freedom in the block, static.
    """

    count: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"block {self.name!r} must hold at least one dof, got {self.count}")

    @property
    def n_cell_fields(self) -> int:
        """Whole per-cell fields: none, these degrees of freedom are not tied to cells."""
        return 0

    def size(self, n_cells: int) -> int:
        """Degrees of freedom, ``count`` -- independent of the mesh."""
        return self.count

    def unflatten(self, part: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """The block's degrees of freedom, shape ``(count,)``."""
        return part

    def flatten(self, value: jnp.ndarray, n_cells: int) -> jnp.ndarray:
        """The block's degrees of freedom, shape ``(count,)``."""
        return value


class FieldLayout(eqx.Module):
    """A flat field-major state vector as an ordered tuple of named blocks.

    Attributes
    ----------
    n_cells : int
        Cells in the mesh, static.
    blocks : tuple of StateBlock
        The blocks in flat-vector order, static. Names must be unique, and a nested
        :class:`SubLayout` must be over the same cell count as this layout.

    Examples
    --------
    The momentum-continuity state, and a coupled turbulence state carrying it verbatim:

    >>> flow = FieldLayout.cell_fields(4, velocity=2, pressure=1)
    >>> flow.size
    12
    >>> coupled = FieldLayout(4, (SubLayout("flow", flow), CellFields("k", 1)))
    >>> coupled.sizes
    (12, 4)
    >>> coupled.n_fields
    4
    >>> coupled.slice_of("k")
    slice(12, 16, None)
    """

    n_cells: int = eqx.field(static=True)
    blocks: tuple[StateBlock, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {self.n_cells}")
        if not self.blocks:
            raise ValueError("a layout must hold at least one block")
        names = [block.name for block in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique within a layout, got {names}")
        for block in self.blocks:
            if isinstance(block, SubLayout) and block.layout.n_cells != self.n_cells:
                raise ValueError(
                    f"block {block.name!r} is a layout over {block.layout.n_cells} cells but sits "
                    f"in a layout over {self.n_cells}."
                )

    @classmethod
    def cell_fields(cls, n_cells: int, **widths: int) -> FieldLayout:
        """A layout of nothing but per-cell fields, in the order the keywords are given.

        Parameters
        ----------
        n_cells : int
            Cells in the mesh.
        **widths : int
            One keyword per block, its value the block's field count -- ``velocity=dim,
            pressure=1`` for a momentum-continuity state. Keyword order is the flat-vector order.

        Returns
        -------
        FieldLayout
            The layout.
        """
        return cls(n_cells, tuple(CellFields(name, n) for name, n in widths.items()))

    # --- shape -------------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """The block names, in flat-vector order."""
        return tuple(block.name for block in self.blocks)

    @property
    def sizes(self) -> tuple[int, ...]:
        """Each block's degrees of freedom, in flat-vector order."""
        return tuple(block.size(self.n_cells) for block in self.blocks)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Each block's first index in the flat vector, in flat-vector order."""
        starts, total = [], 0
        for size in self.sizes:
            starts.append(total)
            total += size
        return tuple(starts)

    @property
    def size(self) -> int:
        """Length of the flat state vector."""
        return sum(self.sizes)

    @property
    def n_fields(self) -> int:
        """Whole per-cell fields across every block; blocks not tied to cells contribute none."""
        return sum(block.n_cell_fields for block in self.blocks)

    @property
    def is_cell_major(self) -> bool:
        """Whether every degree of freedom belongs to a whole per-cell field.

        ``False`` once the layout carries a :class:`GlobalDofs` block, which is what a partition
        into whole fields (:class:`~aquaflux.solve.FieldGroups`) needs to know: a bordered state's
        trailing multiplier belongs to no field, so it belongs to no field group either.
        """
        return self.size == self.n_fields * self.n_cells

    # --- addressing --------------------------------------------------------------------

    def _index(self, name: str) -> int:
        """Position of the named block in :attr:`blocks`.

        Raises
        ------
        KeyError
            If no block carries that name.
        """
        for position, block in enumerate(self.blocks):
            if block.name == name:
                return position
        raise KeyError(f"no block named {name!r} in this layout; it holds {list(self.names)}")

    def slice_of(self, name: str) -> slice:
        """The named block's degrees of freedom, as a slice into the flat vector.

        Raises
        ------
        KeyError
            If no block carries that name.
        """
        position = self._index(name)
        start = self.offsets[position]
        return slice(start, start + self.sizes[position])

    def field_dofs(self, n_fields: int) -> int:
        """Degrees of freedom occupied by ``n_fields`` whole per-cell fields, ``n_fields * n_cells``.

        The one home of the field-major stride: a consumer that partitions the vector on a field
        boundary asks here rather than multiplying by ``n_cells`` itself.
        """
        return n_fields * self.n_cells

    def field_offset(self, name: str) -> int:
        """Whole per-cell fields preceding the named block.

        Where a field-boundary partition (:class:`~aquaflux.solve.FieldGroups`) splits when it is
        asked to split before a named block.
        """
        return sum(block.n_cell_fields for block in self.blocks[: self._index(name)])

    def appended(self, *blocks: StateBlock) -> FieldLayout:
        """This layout with further blocks after its last -- how a state acquires a border."""
        return FieldLayout(self.n_cells, self.blocks + blocks)

    # --- packing -----------------------------------------------------------------------

    def unpack(self, state: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Split the flat state into one value per block, in flat-vector order.

        Parameters
        ----------
        state : jnp.ndarray
            The flat state vector, shape ``(size,)``.

        Returns
        -------
        tuple of jnp.ndarray
            Each block's value, shaped by its own :meth:`StateBlock.unflatten`.
        """
        return tuple(
            block.unflatten(state[start : start + size], self.n_cells)
            for block, start, size in zip(self.blocks, self.offsets, self.sizes, strict=True)
        )

    def pack(self, *parts: jnp.ndarray) -> jnp.ndarray:
        """Assemble one value per block into the flat state vector.

        Parameters
        ----------
        *parts : jnp.ndarray
            One value per block, in flat-vector order, each shaped as :meth:`unpack` returns it.

        Returns
        -------
        jnp.ndarray
            The flat state vector, shape ``(size,)``.

        Raises
        ------
        ValueError
            If the number of values does not match the number of blocks.
        """
        if len(parts) != len(self.blocks):
            raise ValueError(
                f"this layout has {len(self.blocks)} blocks {list(self.names)}, got {len(parts)} "
                "values to pack."
            )
        return jnp.concatenate(
            [
                block.flatten(part, self.n_cells)
                for block, part in zip(self.blocks, parts, strict=True)
            ]
        )

    def zeros(self) -> jnp.ndarray:
        """A zero flat state vector, shape ``(size,)``."""
        return jnp.zeros(self.size)
