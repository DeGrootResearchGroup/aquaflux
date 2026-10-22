"""The format-agnostic mesh-writer contract.

A :class:`MeshWriter` serializes an aquaflux :class:`~aquaflux.mesh.Mesh` to some external format --
the counterpart of :class:`~aquaflux.io.MeshReader`. The concrete strategy varies by format, and
each holds where it writes as its own state, so a caller that only has a mesh to hand off depends on
the one-method contract rather than on a format.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    from pathlib import Path

    from aquaflux.mesh import Mesh


class MeshWriter(eqx.Module):
    """Strategy interface: write an aquaflux ``Mesh`` to an external destination."""

    @abc.abstractmethod
    def write(self, mesh: Mesh) -> Path:
        """Write ``mesh`` and return where it went.

        Parameters
        ----------
        mesh : Mesh
            The mesh to write.

        Returns
        -------
        Path
            The file or directory written.
        """

    @abc.abstractmethod
    def replaces(self) -> tuple[Path, ...]:
        """The files writing would replace: those already where the mesh would go.

        Without leave to overwrite, :meth:`write` refuses to write while any is there, so a caller
        about to spend long making a mesh can ask first.

        Returns
        -------
        tuple of Path
            Empty when nothing is there.
        """
