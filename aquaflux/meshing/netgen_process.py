"""Netgen's surface triangulation and tetrahedral fill, run in a child process and reached by files.

Netgen does two jobs for the mesher: it triangulates the faces of a solid read from a computer-aided
design (CAD) file, each triangle tagged by the face it lies on; and it fills the region a supplied
closed triangle surface bounds with tetrahedra, *without changing that surface*, which is what lets
the pieces of a mesh be joined conformally. Its Python package carries its own build of the
OpenCASCADE geometry kernel, which cannot share an interpreter with the one the CAD reader loads
through OCP: a process holding both aborts at exit. So :class:`NetgenProcess` never imports Netgen.
Each call writes its inputs to a temporary directory, runs the one script that does import it in a
fresh interpreter, and reads the arrays back -- a CAD shape crosses as a file, never as an object.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

#: The script run in the child process; the only module of the package that imports Netgen.
CHILD_SCRIPT = Path(__file__).with_name("_netgen_child.py")

#: How many trailing lines of the child's output a failure reports. Netgen prints its own reason for
#: giving up (for example "Stop meshing since too many attempts") to its standard output.
FAILURE_LINES = 20


class MeshingError(RuntimeError):
    """A mesh generator ran and could not produce a mesh, with what it reported."""


def netgen_available() -> bool:
    """Whether the Netgen package is installed, found without importing it.

    Importing it here would load its geometry kernel into this process, which is the thing to avoid.
    """
    return importlib.util.find_spec("netgen") is not None


@dataclasses.dataclass(frozen=True)
class SurfaceTriangulation:
    """Triangles on the faces of a solid.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Vertex coordinates, every one on the solid's boundary.
    triangles : np.ndarray of int, shape ``(n_triangles, 3)``
        Vertex indices, wound so each triangle's right-hand normal points out of the solid.
    face : np.ndarray of int, shape ``(n_triangles,)``
        The face each triangle lies on: its position in the order a topology explorer visits the
        solid's faces in the file that was read.
    """

    points: np.ndarray
    triangles: np.ndarray
    face: np.ndarray


@dataclasses.dataclass(frozen=True)
class TetrahedralMesh:
    """Tetrahedra filling a region, with the boundary that was supplied for it kept exactly.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Vertex coordinates: the supplied surface's points first, at their indices and coordinates,
        then the points the fill added inside the region.
    tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
        Vertex indices, positively oriented: the fourth vertex lies on the side of the first three
        that their right-hand normal points to.
    """

    points: np.ndarray
    tetrahedra: np.ndarray


@dataclasses.dataclass(frozen=True)
class LocalSizes:
    """Sizes a mesh is to be no coarser than at some points, finer than its largest size.

    Netgen keeps the size near each point within the one given there, and lets it grow away from
    the point no faster than its grading allows. A part of a region is meshed finer by covering it
    with points spaced about its size apart.

    Attributes
    ----------
    points : np.ndarray, shape ``(n, 3)``
        The points.
    sizes : np.ndarray, shape ``(n,)``
        The size wanted at each, positive.
    """

    points: np.ndarray
    sizes: np.ndarray

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        sizes = np.ascontiguousarray(self.sizes, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or sizes.shape != points.shape[:1]:
            raise ValueError(
                f"local sizes take points of shape (n, 3) and one size each, got {points.shape} "
                f"and {sizes.shape}"
            )
        if not (np.isfinite(points).all() and np.isfinite(sizes).all() and np.all(sizes > 0.0)):
            raise ValueError("local sizes take finite points and positive, finite sizes")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "sizes", sizes)

    @classmethod
    def joined(cls, parts: Sequence[LocalSizes]) -> LocalSizes:
        """The points and sizes of every part, one after another."""
        return cls(
            np.concatenate([part.points for part in parts]).reshape(-1, 3),
            np.concatenate([part.sizes for part in parts]),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays the child process reads them from."""
        return {"size_points": self.points, "sizes": self.sizes}


@dataclasses.dataclass(frozen=True)
class NetgenProcess:
    """Netgen's meshing operations, each run in a fresh child interpreter.

    Attributes
    ----------
    python : str
        The interpreter the child runs on. It must have Netgen installed; by default it is this one.
    """

    python: str = sys.executable

    def triangulate(
        self,
        shape,
        *,
        max_size: float,
        min_size: float = 0.0,
        grading: float = 0.3,
        edge_divisions: Mapping[int, int] | None = None,
        local_sizes: LocalSizes | None = None,
    ) -> SurfaceTriangulation:
        """Triangulate the faces of the one solid in a BREP or STEP file.

        Parameters
        ----------
        shape : str or path-like
            The file. The solid should be the one the mesh is for, written by the CAD reader; a
            drawing of several solids is refused rather than meshed in part.
        max_size, min_size : float
            Bounds on the triangles' size, in the file's units.
        grading : float
            How fast the size may change between neighbouring triangles, in ``(0, 1]``; Netgen's own
            default is ``0.3``, and a smaller value grades more gently.
        edge_divisions : mapping of {int: int}, optional
            Edges to divide into exactly that many segments of equal length, in place of the
            division the sizes would give; the faces either side meet them at those points. An edge
            is numbered by its place among the solid's edges, each counted once where first met in
            a topology explorer's order -- the order :meth:`~aquaflux.io.cad.CadModel.topology`
            numbers them in.
        local_sizes : LocalSizes, optional
            Points near which the triangles are to be finer than ``max_size``.

        Returns
        -------
        SurfaceTriangulation

        Raises
        ------
        MeshingError
            If Netgen fails, with the end of its output.
        """
        request = {
            "operation": "triangulate",
            "shape": str(Path(shape).resolve()),
            "max_size": float(max_size),
            "min_size": float(min_size),
            "grading": float(grading),
            "edge_divisions": [
                [int(edge), int(segments)] for edge, segments in (edge_divisions or {}).items()
            ],
            "local_sizes": local_sizes is not None,
        }
        result = self._run(request, {} if local_sizes is None else local_sizes.arrays())
        return SurfaceTriangulation(result["points"], result["triangles"], result["face"])

    def fill(
        self,
        points,
        triangles,
        *,
        max_size: float,
        grading: float = 0.3,
        local_sizes: LocalSizes | None = None,
    ) -> TetrahedralMesh:
        """Fill the region a closed triangle surface bounds with tetrahedra, keeping the surface.

        Parameters
        ----------
        points : array-like, shape ``(n_points, 3)``
            The surface's vertices. Every one must be used by some triangle: Netgen renumbers the
            points when one is not, which would break the promise that they come back unchanged.
        triangles : array-like of int, shape ``(n_triangles, 3)``
            The surface, wound so each triangle's right-hand normal points out of the region. It may
            have several closed parts -- an outer boundary and the walls of holes, the walls of a hole
            wound into the hole -- but every edge must be shared by exactly two triangles, traversed
            in opposite directions. It must not intersect itself, which is not checked: two
            overlapping closed surfaces are each filled, and the fills overlap.
        max_size : float
            Upper bound on the tetrahedra's size.
        grading : float
            As in :meth:`triangulate`.
        local_sizes : LocalSizes, optional
            Points near which the tetrahedra are to be finer than ``max_size``. The surface is kept
            as it is supplied, so where a point lies near it the tetrahedra grade from the
            triangles' size to the point's.

        Returns
        -------
        TetrahedralMesh

        Raises
        ------
        ValueError
            If the surface is not one Netgen can fill unchanged (see :func:`check_closed_surface`).
        MeshingError
            If Netgen fails, with the end of its output -- including when its tetrahedra are not
            bounded by exactly the supplied triangles or do not hold the volume they enclose, which
            is how a hole whose walls are wound the wrong way shows (Netgen fills through it).
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        triangles = np.ascontiguousarray(triangles, dtype=np.int64)
        check_closed_surface(points, triangles)
        request = {
            "operation": "fill",
            "max_size": float(max_size),
            "grading": float(grading),
            "local_sizes": local_sizes is not None,
        }
        arrays = {"points": points, "triangles": triangles}
        if local_sizes is not None:
            arrays.update(local_sizes.arrays())
        result = self._run(request, arrays)
        return TetrahedralMesh(result["points"], result["tetrahedra"])

    def _run(self, request: dict, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Write the request, run the child on it, and return the arrays it wrote."""
        with tempfile.TemporaryDirectory(prefix="aquaflux-netgen-") as directory:
            work = Path(directory)
            (work / "request.json").write_text(json.dumps(request))
            if arrays:
                np.savez(work / "input.npz", **arrays)
            # `-P`: the script's own directory is not put on the module path, so a module of this
            # package can never shadow one the child imports.
            ran = subprocess.run(
                [self.python, "-P", str(CHILD_SCRIPT), str(work)],
                capture_output=True,
                text=True,
                check=False,
            )
            if ran.returncode != 0:
                output = (ran.stdout + ran.stderr).strip().splitlines()[-FAILURE_LINES:]
                raise MeshingError(
                    f"Netgen's {request['operation']} failed (exit status {ran.returncode}):\n"
                    + "\n".join(output)
                )
            with np.load(work / "result.npz") as result:
                return {name: result[name] for name in result.files}


def check_closed_surface(points: np.ndarray, triangles: np.ndarray) -> None:
    """Refuse a triangle surface that does not bound a region consistently, naming the defect.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    triangles : np.ndarray of int, shape ``(n_triangles, 3)``

    Raises
    ------
    ValueError
        If an array has the wrong shape, a coordinate is not finite, an index is out of range, a
        point is used by no triangle, an edge is not shared by exactly one triangle in each direction
        (an open, non-manifold or inconsistently wound surface), or the surface encloses a negative
        volume (wound into the region rather than out of it).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3); got {points.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"triangles must have shape (m, 3) with m > 0; got {triangles.shape}")
    if not np.isfinite(points).all():
        raise ValueError("a point has a non-finite coordinate")
    if triangles.min() < 0 or triangles.max() >= points.shape[0]:
        raise ValueError(f"a triangle names a point outside [0, {points.shape[0]})")
    unused = points.shape[0] - np.unique(triangles).size
    if unused:
        raise ValueError(f"{unused} point(s) are used by no triangle")
    # Each directed edge as one integer, so matching an edge to its reverse is a sorted lookup.
    n_points = points.shape[0]
    tail, head = triangles.reshape(-1), np.roll(triangles, -1, axis=1).reshape(-1)
    forward, backward = tail * n_points + head, head * n_points + tail
    if np.unique(forward).size != forward.size:
        raise ValueError(
            "an edge is traversed twice in the same direction: the surface is wound inconsistently "
            "or is not manifold there"
        )
    unmatched = int(np.count_nonzero(~np.isin(backward, forward)))
    if unmatched:
        raise ValueError(
            f"{unmatched} edge(s) have no triangle on their other side: the surface is not closed"
        )
    corner = points[triangles]
    volume = np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6.0
    if volume <= 0.0:
        raise ValueError(
            f"the surface encloses a volume of {volume:.6g}: it is wound into the region, not out of "
            "it"
        )
