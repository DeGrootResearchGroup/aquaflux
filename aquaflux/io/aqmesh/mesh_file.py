"""aquaflux's own mesh file (``.aqmesh``): a :class:`~aquaflux.mesh.Mesh`'s arrays as the mesh holds them.

A mesh written here is read back exactly -- the same arrays, the same dtypes, the same patches,
groups and zones -- with nothing reordered, reconstructed or searched for on the way: the faces as
compressed sparse rows (CSR, a row-pointer array beside one flat array of node indices), the owner
and neighbour of each face, each face's patch and each cell's zone as labels, and, for a periodic
mesh, the neighbour's offset across the seam. A two-dimensional mesh is stored as two-dimensional,
and a periodic one as periodic: neither has to be extruded or have its seam matched up again, as an
OpenFOAM polyMesh would.

The file is a NumPy ``.npz`` archive (uncompressed, so reading it costs little beyond the reading)
with one more member, ``header``: a small JSON document holding the format's name and version, the
mesh's sizes, its patch and zone names, patch types and groups, the mesh's digest
(:func:`~aquaflux.mesh.mesh_digest`), and whatever record of its making the writer was given. The
reader refuses a file of another format or a newer version, and one whose mesh no longer has the
digest it was written with.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh, mesh_digest

from ..reader import MeshReader
from ..writer import MeshWriter

__all__ = [
    "AQMESH_SUFFIX",
    "AQMESH_VERSION",
    "AquafluxMeshHeader",
    "AquafluxMeshReader",
    "AquafluxMeshWriter",
    "read_aqmesh",
    "read_aqmesh_header",
    "write_aqmesh",
]

#: The file's suffix.
AQMESH_SUFFIX = ".aqmesh"

#: The format's version: a reader reads this version and the ones before it, and refuses a newer one.
AQMESH_VERSION = 1

#: The format's name, the header's first word.
_FORMAT = "aquaflux-mesh"

#: The arrays every file holds, by their names in the archive.
_ARRAYS = (
    "node_coords",
    "face_node_offsets",
    "face_node_indices",
    "owner",
    "neighbour",
    "patch_label",
    "zone_label",
)

#: The array a periodic mesh holds too.
_OFFSET = "neighbour_offset"


class AquafluxMeshHeader(eqx.Module):
    """What a ``.aqmesh`` file says about its mesh, read without reading the mesh.

    Attributes
    ----------
    version : int
        The format version the file was written in.
    dim : int
        The mesh's spatial dimension.
    n_nodes, n_faces, n_cells : int
        Its sizes.
    patches : tuple of str
        Its named boundary patches, in the mesh's order.
    patch_types : tuple of (str, str)
        The types declared for its patches, as the mesh carries them.
    patch_groups : tuple of (str, tuple of str)
        Its patch groups and their members.
    zones : tuple of str
        Its cell zones, the default zone first.
    periodic : bool
        Whether the mesh has a periodic seam.
    digest : str
        The mesh's digest when it was written (:func:`~aquaflux.mesh.mesh_digest`).
    provenance : mapping
        Whatever record of the mesh's making the writer was given; empty if none.
    """

    version: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    n_faces: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)
    patches: tuple[str, ...] = eqx.field(static=True)
    patch_types: tuple[tuple[str, str], ...] = eqx.field(static=True)
    patch_groups: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    zones: tuple[str, ...] = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    digest: str = eqx.field(static=True)
    provenance: Mapping[str, Any] = eqx.field(static=True)

    @classmethod
    def of(cls, mesh: Mesh, provenance: Mapping[str, Any] | None = None) -> AquafluxMeshHeader:
        """The header of ``mesh``, carrying ``provenance`` (JSON-serializable)."""
        patches = mesh.face_patches
        return cls(
            version=AQMESH_VERSION,
            dim=mesh.dim,
            n_nodes=mesh.n_nodes,
            n_faces=mesh.n_faces,
            n_cells=mesh.n_cells,
            patches=tuple(n for n in patches.names if n not in RESERVED_PATCH_NAMES),
            patch_types=tuple(patches.patch_types),
            patch_groups=tuple((g, tuple(m)) for g, m in patches.patch_groups),
            zones=tuple(mesh.cell_zones.names),
            periodic=mesh.face_cells.neighbour_offset is not None,
            digest=mesh_digest(mesh),
            provenance=dict(provenance or {}),
        )

    def to_json(self) -> str:
        """The header as the JSON document the file holds."""
        return json.dumps(
            {
                "format": _FORMAT,
                "version": self.version,
                "dim": self.dim,
                "n_nodes": self.n_nodes,
                "n_faces": self.n_faces,
                "n_cells": self.n_cells,
                "patches": list(self.patches),
                "patch_types": [list(pair) for pair in self.patch_types],
                "patch_groups": [[group, list(members)] for group, members in self.patch_groups],
                "zones": list(self.zones),
                "periodic": self.periodic,
                "digest": self.digest,
                "provenance": dict(self.provenance),
            },
            sort_keys=False,
        )

    @classmethod
    def from_json(cls, text: str, source: str) -> AquafluxMeshHeader:
        """The header a file's JSON document describes; ``source`` names the file in a refusal.

        Raises
        ------
        ValueError
            If the document is not an aquaflux mesh header, or is of a newer version.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}: its header is not JSON ({error}).") from None
        if not isinstance(data, dict) or data.get("format") != _FORMAT:
            raise ValueError(f"{source} is not an aquaflux mesh file.")
        version = data.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"{source}: its format version {version!r} is not a version.")
        if version > AQMESH_VERSION:
            raise ValueError(
                f"{source} was written in format version {version}, newer than this aquaflux "
                f"reads ({AQMESH_VERSION}); read it with a newer aquaflux."
            )
        try:
            return cls(
                version=version,
                dim=int(data["dim"]),
                n_nodes=int(data["n_nodes"]),
                n_faces=int(data["n_faces"]),
                n_cells=int(data["n_cells"]),
                patches=tuple(data["patches"]),
                patch_types=tuple((str(p), str(t)) for p, t in data["patch_types"]),
                patch_groups=tuple((str(g), tuple(m)) for g, m in data["patch_groups"]),
                zones=tuple(data["zones"]),
                periodic=bool(data["periodic"]),
                digest=str(data["digest"]),
                provenance=dict(data.get("provenance") or {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}: its header is incomplete ({error!r}).") from None


class AquafluxMeshWriter(MeshWriter):
    """Write a mesh as a ``.aqmesh`` file.

    Attributes
    ----------
    path : str
        The file. Written whole or not at all: the mesh goes to a file beside it first, which then
        takes its place.
    overwrite : bool
        Replace a file already there; without it, an existing file is refused.
    provenance : mapping
        A record of the mesh's making for the header (JSON-serializable): the settings it was made
        with, a version, a time.
    """

    path: str = eqx.field(static=True)
    overwrite: bool = eqx.field(static=True, default=False)
    provenance: Mapping[str, Any] = eqx.field(static=True, default_factory=dict)

    def __init__(self, path, *, overwrite: bool = False, provenance=None):
        """Construct a writer for one file.

        Parameters
        ----------
        path : str or path-like
            The file, ending in :data:`AQMESH_SUFFIX`.
        overwrite : bool
            Replace a file already there.
        provenance : mapping, optional
            A record of the mesh's making, for the header.

        Raises
        ------
        ValueError
            If ``path`` does not end in :data:`AQMESH_SUFFIX`, or ``provenance`` is not
            JSON-serializable.
        """
        if Path(path).suffix != AQMESH_SUFFIX:
            raise ValueError(f"an aquaflux mesh file ends in {AQMESH_SUFFIX}, got {str(path)!r}.")
        provenance = dict(provenance or {})
        try:
            json.dumps(provenance)
        except TypeError as error:
            raise ValueError(
                f"a mesh file's provenance must be JSON-serializable: {error}"
            ) from None
        self.path = str(path)
        self.overwrite = overwrite
        self.provenance = provenance

    def replaces(self) -> tuple[Path, ...]:
        """The file itself, if it is already there."""
        return (Path(self.path),) if Path(self.path).exists() else ()

    def write(self, mesh: Mesh) -> Path:
        """Write ``mesh`` and return the file.

        Raises
        ------
        FileExistsError
            If the file exists and :attr:`overwrite` is not set.
        ValueError
            If the mesh carries the padding patch of a distributed partition, which is not a mesh
            anyone reads.
        """
        path = Path(self.path)
        if self.replaces() and not self.overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite to replace it.")
        if any(name not in ("interior", "boundary") and name in RESERVED_PATCH_NAMES
               for name in mesh.face_patches.names):  # fmt: skip
            raise ValueError("a partition's padded mesh is not written; write the whole mesh.")
        header = AquafluxMeshHeader.of(mesh, self.provenance)
        arrays = {
            "node_coords": np.asarray(mesh.node_coords),
            "face_node_offsets": np.asarray(mesh.face_nodes.offsets),
            "face_node_indices": np.asarray(mesh.face_nodes.face_node_indices),
            "owner": np.asarray(mesh.face_cells.owner),
            "neighbour": np.asarray(mesh.face_cells.neighbour),
            "patch_label": np.asarray(mesh.face_patches.label),
            "zone_label": np.asarray(mesh.cell_zones.label),
            "header": np.frombuffer(header.to_json().encode(), dtype=np.uint8),
        }
        if header.periodic:
            arrays[_OFFSET] = np.asarray(mesh.face_cells.neighbour_offset)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as file:
                np.savez(file, **arrays)
            # A temporary file is made readable by its owner alone; the mesh is an ordinary file.
            os.chmod(partial, 0o666 & ~_umask())
            os.replace(partial, path)
        except BaseException:
            Path(partial).unlink(missing_ok=True)
            raise
        return path


class AquafluxMeshReader(MeshReader):
    """Read a ``.aqmesh`` file back into the mesh it was written from.

    Attributes
    ----------
    path : str
        The file.
    """

    path: str = eqx.field(static=True)

    def __init__(self, path):
        """Construct a reader for one file.

        Parameters
        ----------
        path : str or path-like
            The file.
        """
        self.path = str(path)

    def header(self) -> AquafluxMeshHeader:
        """The file's header alone, without reading its mesh.

        Raises
        ------
        FileNotFoundError
            If there is no such file.
        ValueError
            If it is not an aquaflux mesh file, or of a newer version.
        """
        with self._archive() as archive:
            return self._header(archive)

    def read(self) -> Mesh:
        """The mesh, exactly as it was written.

        Raises
        ------
        FileNotFoundError
            If there is no such file.
        ValueError
            If it is not an aquaflux mesh file, of a newer version, missing an array, or holds a
            mesh whose digest is not the one it was written with (changed since, or damaged).
        """
        with self._archive() as archive:
            header = self._header(archive)
            missing = [name for name in _ARRAYS if name not in archive.files]
            if header.periodic and _OFFSET not in archive.files:
                missing.append(_OFFSET)
            if missing:
                raise ValueError(f"{self.path} is missing its {', '.join(missing)}.")
            arrays = {name: archive[name] for name in (*_ARRAYS, _OFFSET) if name in archive.files}
        names = ("interior", "boundary", *header.patches)
        patch_label = arrays["patch_label"]
        face_patches = {
            name: np.flatnonzero(patch_label == index)
            for index, name in enumerate(names)
            if name in header.patches
        }
        zone_label = arrays["zone_label"]
        cell_zones = {
            name: np.flatnonzero(zone_label == index)
            for index, name in enumerate(header.zones)
            if index > 0
        }
        mesh = Mesh.from_csr(
            arrays["node_coords"],
            arrays["face_node_offsets"],
            arrays["face_node_indices"],
            arrays["owner"],
            arrays["neighbour"],
            header.n_cells,
            cell_zones=cell_zones or None,
            face_patches=face_patches,
            patch_types=dict(header.patch_types),
            patch_groups={group: list(members) for group, members in header.patch_groups},
            neighbour_offset=arrays.get(_OFFSET),
        )
        if mesh_digest(mesh) != header.digest:
            raise ValueError(
                f"{self.path} holds a mesh other than the one it was written with (its digest "
                "differs): it has been changed or damaged since."
            )
        return mesh

    def _archive(self):
        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"no aquaflux mesh file at {path}.")
        try:
            return np.load(path, allow_pickle=False)
        except (ValueError, OSError) as error:
            raise ValueError(f"{path} is not an aquaflux mesh file ({error}).") from None

    def _header(self, archive) -> AquafluxMeshHeader:
        if "header" not in archive.files:
            raise ValueError(f"{self.path} is not an aquaflux mesh file (it has no header).")
        return AquafluxMeshHeader.from_json(archive["header"].tobytes().decode(), self.path)


def _umask() -> int:
    """The process's file-creation mask, read without changing it."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def write_aqmesh(mesh: Mesh, path, *, overwrite: bool = False, provenance=None) -> Path:
    """Write ``mesh`` as the ``.aqmesh`` file at ``path`` (see :class:`AquafluxMeshWriter`).

    Parameters
    ----------
    mesh : Mesh
    path : str or path-like
        The file, ending in :data:`AQMESH_SUFFIX`.
    overwrite : bool
        Replace a file already there.
    provenance : mapping, optional
        A record of the mesh's making, for the header (JSON-serializable).

    Returns
    -------
    Path
        The file. :func:`read_aqmesh` reads the same mesh back.
    """
    return AquafluxMeshWriter(path, overwrite=overwrite, provenance=provenance).write(mesh)


def read_aqmesh(path) -> Mesh:
    """The mesh in the ``.aqmesh`` file at ``path`` (see :class:`AquafluxMeshReader`)."""
    return AquafluxMeshReader(path).read()


def read_aqmesh_header(path) -> AquafluxMeshHeader:
    """The header of the ``.aqmesh`` file at ``path``, without reading its mesh."""
    return AquafluxMeshReader(path).header()
