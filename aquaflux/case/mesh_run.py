"""Making a mesh from a mesh file: what ``aquaflux mesh mesh.yaml`` does.

The mesh file is read and checked, the places the mesh would be written are checked before anything
is meshed -- meshing can take minutes, and a refusal should come first -- and then the region is
meshed and written: as ``.aqmesh`` always, and as an OpenFOAM polyMesh when the file asks. The
``.aqmesh`` file's header records how the mesh was made: the mesh file's settings as they ran, the
aquaflux version and commit, when, how long it took, and a summary of the mesh's quality
(:class:`~aquaflux.mesh.QualitySummary`), so a mesh can be judged without being read.
"""

from __future__ import annotations

import dataclasses
import datetime
import time
from pathlib import Path

import aquaflux
from aquaflux.io import AquafluxMeshWriter, OpenFOAMWriter
from aquaflux.mesh import Mesh, QualitySummary
from aquaflux.meshing import Stage

from .case_file import MeshFile, read_mesh_file
from .run import _checkout_state
from .spec import mesh_spec_to_mapping

__all__ = ["MeshRecord", "make_mesh"]


@dataclasses.dataclass(frozen=True)
class MeshRecord:
    """What making a mesh did.

    Attributes
    ----------
    mesh : Mesh
        The mesh made.
    aqmesh : pathlib.Path
        The ``.aqmesh`` file written.
    polymesh : pathlib.Path or None
        The OpenFOAM polyMesh directory written, if one was asked for.
    seconds : float
        How long the meshing took, writing excluded.
    quality : QualitySummary
        The mesh's quality, as its header records it: before its periodic patches, if any, are
        joined.
    """

    mesh: Mesh
    aqmesh: Path
    polymesh: Path | None
    seconds: float
    quality: QualitySummary


def make_mesh(
    path: str | Path, *, overwrite: bool = False, on_stage: Stage | None = None
) -> MeshRecord:
    """Mesh the region a mesh file describes and write it where the file says.

    Parameters
    ----------
    path : str or path-like
        The mesh file.
    overwrite : bool
        Replace a mesh already where this one would be written.
    on_stage : callable, optional
        Told the name of each stage as it starts: the meshing's (see
        :meth:`~aquaflux.case.MeshSpec.mesh`), then measuring the quality and writing.

    Returns
    -------
    MeshRecord

    Raises
    ------
    ValueError, TypeError
        If the mesh file is refused.
    FileExistsError
        If a mesh is already where this one would go and ``overwrite`` is not set -- raised before
        anything is meshed.
    """
    mesh_file = read_mesh_file(path)
    writers = _writers(mesh_file, overwrite)
    if not overwrite:
        occupied = [str(target) for writer in writers for target in writer.replaces()]
        if occupied:
            raise FileExistsError(
                f"{', '.join(occupied)} already there; pass overwrite to replace "
                f"{'it' if len(occupied) == 1 else 'them'}."
            )
    started, clock = datetime.datetime.now(datetime.UTC), time.perf_counter()
    mesh = mesh_file.spec.mesh(mesh_file.directory, on_stage=on_stage)
    seconds = time.perf_counter() - clock
    _tell(on_stage, "measuring the mesh's quality")
    # Measured before any periodic join: across a seam the concave-cell test would misread.
    quality = QualitySummary.of(mesh)
    if mesh_file.spec.periodic is not None:
        _tell(on_stage, "joining the periodic patches")
        mesh = mesh_file.spec.joined(mesh)
    _tell(on_stage, "writing the mesh")
    commit, modified = _checkout_state()
    provenance = {
        "mesh_file": str(Path(path).resolve()),
        "settings": mesh_spec_to_mapping(mesh_file.spec),
        "aquaflux": {"version": aquaflux.__version__, "commit": commit, "modified": modified},
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(seconds, 3),
        "quality": quality.to_mapping(),
    }
    aqmesh_writer, *others = writers
    aqmesh = AquafluxMeshWriter(
        aqmesh_writer.path, overwrite=overwrite, provenance=provenance
    ).write(mesh)
    polymesh = others[0].write(mesh) if others else None
    return MeshRecord(mesh=mesh, aqmesh=aqmesh, polymesh=polymesh, seconds=seconds, quality=quality)


def _tell(on_stage: Stage | None, stage: str) -> None:
    """Tell ``on_stage``, if anyone follows the stages, that ``stage`` starts."""
    if on_stage is not None:
        on_stage(stage)


def _writers(mesh_file: MeshFile, overwrite: bool) -> list:
    """The writers of the mesh file's outputs: the ``.aqmesh`` file's first, then any polyMesh."""
    output = mesh_file.spec.output
    writers = [AquafluxMeshWriter(mesh_file.directory / output.path, overwrite=overwrite)]
    if output.openfoam is not None:
        writers.append(OpenFOAMWriter(mesh_file.directory / output.openfoam, overwrite=overwrite))
    return writers
