"""The material model: the named collection of properties a set of physics models requires."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from .property import MaterialProperty

if TYPE_CHECKING:
    from aquaflux.mesh import CellZones


class MaterialModel(eqx.Module):
    """The named material properties a problem's models consume (density, viscosity, ...).

    Each entry is a single :class:`~aquaflux.materials.property.MaterialProperty`; operators name the
    property they need (e.g. a diffusion coefficient), and the model resolves the name to a per-cell
    array via :meth:`evaluate`. Adding a property is adding an entry — the abstraction and the
    consumers' context shape do not change.

    Attributes
    ----------
    properties : dict of {str: MaterialProperty}
        The property object for each name.
    """

    properties: dict[str, MaterialProperty]

    def evaluate(
        self, cell_zones: CellZones, fields: Mapping[str, jnp.ndarray] | None = None
    ) -> dict[str, jnp.ndarray]:
        """Evaluate every property to a per-cell array, returning ``{name: (n_cells,) array}``.

        Parameters
        ----------
        cell_zones : CellZones
            The cell partition.
        fields : mapping of {str: jnp.ndarray}, optional
            Named per-cell state fields for state-dependent properties; omit when every property is
            state-independent.
        """
        fields = {} if fields is None else fields
        return {name: prop.evaluate(cell_zones, fields) for name, prop in self.properties.items()}

    def require(self, *names: str) -> None:
        """Raise ``ValueError`` unless the model supplies every property in ``names``.

        A fail-fast check for an assembler: an operator that names a coefficient it consumes can
        assert the model provides it up front, rather than surfacing a ``KeyError`` mid-residual.
        """
        missing = [n for n in names if n not in self.properties]
        if missing:
            raise ValueError(
                f"material model is missing required propert{'y' if len(missing) == 1 else 'ies'} "
                f"{missing}; it supplies {sorted(self.properties)}"
            )
