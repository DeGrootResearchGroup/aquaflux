"""The format-agnostic mesh-reader contract.

A :class:`MeshReader` turns some external mesh source into an aquaflux
:class:`~aquaflux.mesh.Mesh`. The concrete strategy varies by file format (OpenFOAM now; other
formats later), but every reader exposes the same one-method interface, so callers depend on the
contract rather than a format. This mirrors the strategy interfaces elsewhere in the package (an
``equinox.Module`` with an abstract method).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh


class MeshReader(eqx.Module):
    """Strategy interface: read a mesh from an external source into an aquaflux ``Mesh``."""

    @abc.abstractmethod
    def read(self) -> Mesh:
        """Read the source and return the assembled mesh.

        Returns
        -------
        Mesh
            The mesh built from the reader's source.
        """
