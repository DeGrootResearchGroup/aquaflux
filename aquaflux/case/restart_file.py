"""The file a case's checkpoints are written as, and read back from: physical fields and a header.

A march's own state is a flat vector of its *solved* variables, which for a Reynolds-averaged case may
hold ``log(omega)`` rather than ``omega``. A file of that vector means something only to a case that
solves in the same variables, and a case that does not would read it without complaint: a log read as
the field has the right length, so no size check can tell. A checkpoint is therefore written as the
**physical** fields of the case -- ``U``, ``p`` and, when the case is Reynolds-averaged (RANS), ``k`` and
``omega`` -- which any case of the same physics and mesh can start from, whatever it solves in.

Beside the fields goes a header saying what they belong to: the physics, the cell count, the
dimension, and a digest of the mesh. The cell count alone would let a renumbered mesh, or one of the
same size but different shape, load as an unrelated field of the right length; the digest is what
catches that. The case the file came from is recorded too, as provenance for the run that starts from
it -- it is not compared, since a restart is often a case with something changed. A second digest,
of the problem the case solves and the measure it judged a residual in, is compared for one purpose:
a restart of the same problem continues the stopped march (its damping and stopping bar, which are
measured against residuals the march took, and its step control's shift), and one of a different
problem does not.

The header and the fields are written and read here and nowhere else, so the two cannot disagree.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aquaflux.solve import Resumption, report_record

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh
    from aquaflux.solve import StepReport

    from .physics import Physics
    from .spec import CaseSpec

__all__ = [
    "RestartFile",
    "RestartHeader",
    "checkpoint_writer",
    "mesh_digest",
    "read_restart",
    "refuse_non_finite",
]

#: How finely a node's position is resolved when a mesh is digested, as a fraction of the mesh's
#: largest extent. Coarse enough that rounding in the last bits of a coordinate -- the same mesh read
#: on another machine -- lands in the same bin, fine enough that no mesh anyone would call the same
#: shares a digest with a moved one.
_DIGEST_RESOLUTION = 1.0e-6

#: The prefix a field's array carries in the file, so a field named like a header entry cannot shadow it.
_FIELD_PREFIX = "field_"


def mesh_digest(mesh: Mesh) -> str:
    """A digest of a mesh's topology and node positions, the same for the same mesh wherever it is read.

    Parameters
    ----------
    mesh : Mesh
        The mesh. Only its face-to-cell connectivity and node coordinates are read, so no geometry
        has to have been computed.

    Returns
    -------
    str
        A hexadecimal digest. Two meshes with the same cell numbering and nodes within a millionth of
        the mesh's extent of each other share one.
    """
    owner = np.asarray(mesh.face_cells.owner, dtype=np.int64)
    neighbour = np.asarray(mesh.face_cells.neighbour, dtype=np.int64)
    nodes = np.asarray(mesh.node_coords, dtype=np.float64)
    extent = float(np.max(nodes.max(axis=0) - nodes.min(axis=0)))
    resolution = _DIGEST_RESOLUTION * (extent if extent > 0.0 else 1.0)
    binned = np.round(nodes / resolution).astype(np.int64)
    digest = hashlib.sha256()
    for array in (owner, neighbour, binned):
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class RestartHeader:
    """What a checkpoint's fields belong to.

    Attributes
    ----------
    physics : str
        The kind of physics the fields are of -- ``"Laminar"`` or ``"RANS"`` -- which decides which
        fields there are.
    n_cells : int
        The number of cells of the mesh the fields are on.
    dim : int
        The mesh's spatial dimension, which sets the width of ``U``.
    mesh_digest : str
        The mesh's :func:`mesh_digest`.
    case_digest : str
        A digest of the case that wrote the file. Recorded, never compared.
    problem_digest : str
        A digest of what that case solves and the measure it judged a residual in
        (:meth:`~aquaflux.case.CaseSpec.problem_digest`). It is what decides whether a restart may
        carry the stopped run's march history, which holds only for the same problem.
    """

    physics: str
    n_cells: int
    dim: int
    mesh_digest: str
    case_digest: str
    problem_digest: str

    @classmethod
    def of(cls, spec: CaseSpec, mesh: Mesh) -> RestartHeader:
        """The header of fields of ``spec``'s physics on ``mesh``.

        Parameters
        ----------
        spec : CaseSpec
            The case. Its physics names the fields and its digests are recorded.
        mesh : Mesh
            The case's mesh.

        Returns
        -------
        RestartHeader
        """
        return cls(
            physics=type(spec.physics).__name__,
            n_cells=int(mesh.n_cells),
            dim=int(mesh.dim),
            mesh_digest=mesh_digest(mesh),
            case_digest=spec.digest(),
            problem_digest=spec.problem_digest(),
        )

    def refuse_unless_fits(self, expected: RestartHeader, path: Path) -> None:
        """Refuse fields that do not belong to the physics and mesh ``expected`` describes.

        Parameters
        ----------
        expected : RestartHeader
            The header of the case about to start from the fields; its digests of the case are not
            read.
        path : pathlib.Path
            The file the fields were read from, for the message.

        Raises
        ------
        ValueError
            Naming every way the two differ. The digest is reported only when the cell count and
            dimension agree, since it differs whenever they do not and says nothing more then.
        """
        problems = []
        if self.physics != expected.physics:
            problems.append(
                f"its fields are of a {self.physics} case, but this case is {expected.physics}"
            )
        if self.n_cells != expected.n_cells:
            problems.append(
                f"it holds {self.n_cells} cells, but this case's mesh has {expected.n_cells}"
            )
        if self.dim != expected.dim:
            problems.append(
                f"it is of a {self.dim}-dimensional mesh, but this case's is {expected.dim}-dimensional"
            )
        same_size = self.n_cells == expected.n_cells and self.dim == expected.dim
        if same_size and self.mesh_digest != expected.mesh_digest:
            problems.append(
                "its mesh has the same number of cells as this case's but is not the same mesh "
                "(the cell numbering or the node positions differ)"
            )
        if problems:
            raise ValueError(f"{path} cannot start this case: " + "; ".join(problems) + ".")


@dataclasses.dataclass(frozen=True)
class RestartFile:
    """A checkpoint as read: its header, its physical fields and how converged they were.

    Attributes
    ----------
    header : RestartHeader
        What the fields belong to.
    fields : mapping of {str: np.ndarray}
        The physical fields by name -- a vector ``(n_cells, dim)``, a scalar ``(n_cells,)``.
    residual : float
        The march's residual at the step that wrote the file, in the measure it was steered by.
    resumption : Resumption or None
        What the step's record says of the march that wrote the file, for a march resuming it: the
        residual norm the march took at **its** first state (the recorded residual and residual ratio
        give it back as their quotient), the one its last segment's damping was anchored at, and the
        shift its last step ran at when a step control set one. ``None`` when the record has no usable
        ratio (a march that started at an exact root).
    path : pathlib.Path
        The file it was read from.
    """

    header: RestartHeader
    fields: Mapping[str, np.ndarray]
    residual: float
    resumption: Resumption | None
    path: Path

    def refuse_if_not_finite(self) -> None:
        """Refuse fields holding a value that is not a number.

        A checkpointer writes whatever the march reports, including a step that diverged, so the
        newest file can hold the state the march died in. Starting from it would only reproduce the
        failure, and say nothing about why.

        Raises
        ------
        ValueError
            Naming the fields, and pointing at an earlier step.
        """
        refuse_non_finite(
            self.fields,
            f"{self.path} holds values that are not finite in {{bad}}: the march had diverged when "
            "it wrote it. Start from an earlier checkpoint (set the step) if one was kept.",
        )


def refuse_non_finite(fields: Mapping[str, np.ndarray], message: str) -> None:
    """Refuse fields holding a value that is not a number, naming the fields that do.

    Parameters
    ----------
    fields : mapping of {str: np.ndarray}
        The fields by name.
    message : str
        What to raise, with ``{bad}`` standing for the sorted names of the offending fields.

    Raises
    ------
    ValueError
        With ``message``, if any field holds a value that is not finite.
    """
    bad = sorted(name for name, values in fields.items() if not np.all(np.isfinite(values)))
    if bad:
        raise ValueError(message.format(bad=bad))


def checkpoint_writer(
    physics: Physics, problem: object, header: RestartHeader
) -> Callable[[Path, Any, StepReport], None]:
    """The serializer a case's checkpointer writes with, in place of a bare solved-state array.

    Parameters
    ----------
    physics : Physics
        The case's physics, which turns a march state into physical fields.
    problem : object
        What the case built; the march states it is handed are states of it.
    header : RestartHeader
        What every file written belongs to.

    Returns
    -------
    callable
        ``(path, state, report) -> None``, the form :class:`~aquaflux.solve.StateCheckpointer` takes
        as ``save``. It writes exactly to ``path``.
    """

    def save(path: Path, state: Any, report: StepReport) -> None:
        fields = physics.restart_fields(problem, state)
        # A file object, not the path: `np.savez` appends ".npz" to a path that lacks it, and the
        # checkpointer's staging name does not end in one.
        with open(path, "wb") as handle:
            np.savez(
                handle,
                **{f"{_FIELD_PREFIX}{name}": np.asarray(values) for name, values in fields.items()},
                **dataclasses.asdict(header),
                **report_record(report),
            )

    return save


def _positive_or_none(value: np.ndarray) -> float | None:
    """A recorded number as a float, or ``None`` when it is not a usable positive one.

    A step record writes ``0.0`` for "not recorded" (a march with no step control has no shift, and a
    report from before the damping anchor was kept has no anchor), and a diverged step may write a
    non-finite one; neither says anything a march can resume from.
    """
    number = float(value)
    return number if np.isfinite(number) and number > 0.0 else None


def read_restart(path: str | Path) -> RestartFile:
    """Read a checkpoint written by :func:`checkpoint_writer`.

    Parameters
    ----------
    path : str or path-like
        The checkpoint file.

    Returns
    -------
    RestartFile

    Raises
    ------
    ValueError
        If the file has no header: it is a bare solved state, as a library checkpointer writes by
        default, which says nothing of the mesh or the variables it is in and so cannot start a case.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    header_names = {field.name for field in dataclasses.fields(RestartHeader)}
    with np.load(path) as data:
        missing = sorted(header_names - set(data.files))
        if missing:
            raise ValueError(
                f"{path} is not a case checkpoint: it has no {', '.join(missing)}. A file of a bare "
                "solved state names neither its mesh nor the variables it is in, so it cannot start a "
                "case; one written by a run of a case file can."
            )
        header = RestartHeader(
            physics=str(data["physics"]),
            n_cells=int(data["n_cells"]),
            dim=int(data["dim"]),
            mesh_digest=str(data["mesh_digest"]),
            case_digest=str(data["case_digest"]),
            problem_digest=str(data["problem_digest"]),
        )
        residual, ratio = float(data["residual_norm"]), float(data["residual_ratio"])
        fields = {
            name[len(_FIELD_PREFIX) :]: np.asarray(data[name])
            for name in data.files
            if name.startswith(_FIELD_PREFIX)
        }
        usable = np.isfinite(residual) and np.isfinite(ratio) and residual > 0.0 and ratio > 0.0
        return RestartFile(
            header=header,
            fields=fields,
            residual=residual,
            resumption=(
                Resumption(
                    reference_residual=residual / ratio,
                    damping_reference=_positive_or_none(data["damping_reference"]),
                    shift=_positive_or_none(data["shift"]),
                )
                if usable
                else None
            ),
            path=path,
        )
