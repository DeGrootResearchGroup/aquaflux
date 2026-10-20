"""A polyhedral mesh of a region of a computer-aided design (CAD) model.

:class:`CadMesher` meshes a region (some solids' union, less others) in four steps: the region is
written as a boundary-representation (BREP) file; Netgen triangulates its faces, each triangle
tagged by its face, and fills the surface with tetrahedra; the tetrahedra become a mesh whose
boundary faces are in patches chosen by face (:class:`FaceSelection`); and that mesh's polyhedral
dual is the result, the region's sharp edges kept as features.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aquaflux.io.cad import CadModel, CadRegion
from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh

from .dual import PolyhedralDual
from .netgen_process import NetgenProcess, SurfaceTriangulation
from .tetrahedra import tetrahedral_mesh

__all__ = ["CadMesher", "FaceSelection", "FacesNear", "FacesOn", "RegionFaces"]


@dataclasses.dataclass(frozen=True)
class RegionFaces:
    """A region's faces as a selection sees them.

    Attributes
    ----------
    model : CadModel
    region : CadRegion
    samples : np.ndarray, shape ``(n_faces, 3)``
        One point on each face, in the order :meth:`~aquaflux.io.cad.CadModel.topology` numbers
        them.
    """

    model: CadModel
    region: CadRegion
    samples: np.ndarray

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return self.samples.shape[0]


@dataclasses.dataclass(frozen=True)
class FaceSelection(abc.ABC):
    """Some faces of a region, for a boundary patch."""

    @abc.abstractmethod
    def faces(self, region: RegionFaces) -> np.ndarray:
        """The faces selected, as indices into the region's faces, ascending and distinct.

        Raises
        ------
        ValueError
            If the selection finds no face.
        """


@dataclasses.dataclass(frozen=True)
class FacesNear(FaceSelection):
    """The face of the region nearest each of some points: a pipe's end, by a point on it.

    Attributes
    ----------
    points : tuple of tuple of float
        The points, three coordinates each, in the model's frame.
    """

    points: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"FacesNear takes points of three coordinates, got {self.points!r}.")

    def faces(self, region: RegionFaces) -> np.ndarray:
        """The nearest face to each point -- see :meth:`FaceSelection.faces`."""
        landed = region.model.project(region.region, np.asarray(self.points, dtype=float))
        return np.unique(landed.face)


@dataclasses.dataclass(frozen=True)
class FacesOn(FaceSelection):
    """The faces of the region lying on the surface of some solids: a lamp taken out of the water.

    Attributes
    ----------
    solids : tuple of str
        The solids, by name in the model.
    tolerance : float or None
        How far a point of a face may lie from the solids' surface and still be on it, in the
        model's units. Unset, a millionth of the region's extent.
    """

    solids: tuple[str, ...]
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("FacesOn names the solids whose surface the faces lie on.")
        if self.tolerance is not None and not (
            math.isfinite(self.tolerance) and self.tolerance > 0
        ):
            raise ValueError(
                f"FacesOn.tolerance must be a positive, finite number, got {self.tolerance!r}."
            )

    def faces(self, region: RegionFaces) -> np.ndarray:
        """The faces on the solids' surface -- see :meth:`FaceSelection.faces`."""
        tolerance = self.tolerance
        if tolerance is None:
            extent = region.samples.max(axis=0) - region.samples.min(axis=0)
            tolerance = 1e-6 * float(np.linalg.norm(extent))
        distance = region.model.project(CadRegion(self.solids), region.samples).distance
        found = np.flatnonzero(distance <= tolerance)
        if not found.size:
            raise ValueError(f"no face of the region lies on the surface of {list(self.solids)}")
        return found


@dataclasses.dataclass(frozen=True)
class CadMesher:
    """Mesh a region of a CAD model as polyhedra: the dual of a Netgen tetrahedral fill.

    Each cell is the dual cell of one of the tetrahedra's points (cut in pieces at a sharp concave
    edge), so the cells come out larger than the tetrahedra; ``max_size`` bounds the tetrahedra.

    Attributes
    ----------
    max_size : float
        The largest tetrahedron edge wanted, in the model's units: Netgen's target, which its
        triangles and tetrahedra exceed somewhat.
    feature_angle : float
        The fold, in radians, beyond which an edge of the region, or of the triangulated surface,
        is kept sharp (see :class:`~aquaflux.meshing.PolyhedralDual`).
    min_size : float or None
        The smallest triangle wanted; unset, Netgen's.
    grading : float or None
        How fast the size may change between neighbours, in ``(0, 1]``; unset, Netgen's.
    netgen : NetgenProcess
        The Netgen it runs.
    """

    max_size: float
    feature_angle: float
    min_size: float | None = None
    grading: float | None = None
    netgen: NetgenProcess = dataclasses.field(default_factory=NetgenProcess)

    def __post_init__(self) -> None:
        for name in ("max_size", "feature_angle", "min_size", "grading"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"CadMesher.{name} must be a positive, finite number, got {value!r}."
                )

    def mesh(
        self,
        model: CadModel,
        region: CadRegion,
        patches: Mapping[str, FaceSelection] | None = None,
        others: str = "walls",
    ) -> Mesh:
        """The polyhedral mesh of ``region``.

        Parameters
        ----------
        model : CadModel
        region : CadRegion
        patches : mapping of {str: FaceSelection}, optional
            The boundary patches named for the faces selected.
        others : str
            The patch for the faces no selection names.

        Returns
        -------
        Mesh
            Its boundary in the patches named, and in ``others`` where any face is left.

        Raises
        ------
        ValueError
            If a patch name is reserved or given twice, a selection finds no face, or two
            selections share one.
        MeshingError
            If Netgen fails.
        """
        patches = dict(patches or {})
        _check_names(patches, others)
        topology = model.topology(region)
        with tempfile.TemporaryDirectory() as directory:
            surface = self.netgen.triangulate(
                model.write_brep(region, Path(directory) / "region.brep"),
                max_size=self.max_size,
                **self._netgen_options(min_size=self.min_size),
            )
        patch_of_face = _patch_of_face(model, region, surface, topology.n_faces, patches, others)
        filled = self.netgen.fill(
            surface.points, surface.triangles, max_size=self.max_size, **self._netgen_options()
        )
        names = [*patches, others]
        primal = tetrahedral_mesh(
            filled.points,
            filled.tetrahedra,
            {
                name: surface.triangles[patch_of_face[surface.face] == index]
                for index, name in enumerate(names)
                if np.any(patch_of_face == index)
            },
        )
        feature_pairs = {
            tuple(sorted(topology.faces_of_edge(edge)[:2].tolist()))
            for edge in topology.feature_edges(self.feature_angle)
            if topology.faces_of_edge(edge).size >= 2
        }
        dual = PolyhedralDual(
            self.feature_angle, feature_edges=_edges_between(surface, feature_pairs)
        )
        return dual.build(primal)

    def _netgen_options(self, **extra) -> dict:
        """The Netgen settings that are set, to pass through."""
        options = {"grading": self.grading, **extra}
        return {name: value for name, value in options.items() if value is not None}


def _check_names(patches, others: str) -> None:
    """Refuse a reserved patch name, or ``others`` naming a selection too."""
    reserved = sorted(({*patches, others}) & set(RESERVED_PATCH_NAMES))
    if reserved:
        raise ValueError(f"patch name(s) {reserved} are reserved for the mesh's own groups")
    if others in patches:
        raise ValueError(f"{others!r} names both a selection and the faces no selection names")


def _patch_of_face(model, region, surface: SurfaceTriangulation, n_faces, patches, others):
    """Each face's patch, as its position among the selections (``others`` last)."""
    # A point on each face: the centroid of one of its triangles, moved onto the face.
    first = np.full(n_faces, -1, dtype=np.int64)
    first[surface.face[::-1]] = np.arange(surface.face.size)[::-1]
    if np.any(first < 0):
        raise ValueError(f"{int(np.count_nonzero(first < 0))} face(s) were given no triangle")
    centroids = surface.points[surface.triangles[first]].mean(axis=1)
    samples = model.project(region, centroids, faces=np.arange(n_faces)).points
    faces = RegionFaces(model, region, samples)
    patch_of_face = np.full(n_faces, len(patches), dtype=np.int64)
    taken = np.zeros(n_faces, dtype=bool)
    for index, (name, selection) in enumerate(patches.items()):
        chosen = selection.faces(faces)
        if np.any(taken[chosen]):
            raise ValueError(f"patch {name!r} selects a face another patch has already")
        taken[chosen] = True
        patch_of_face[chosen] = index
    return patch_of_face


def _edges_between(surface: SurfaceTriangulation, pairs: set[tuple[int, int]]) -> np.ndarray:
    """The triangulation's edges between two faces whose pair is in ``pairs``, as point pairs."""
    triangles = surface.triangles
    edges = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2).reshape(-1, 2)
    face = np.repeat(surface.face, 3)
    keys = np.sort(edges, axis=1)
    order = np.lexsort(keys.T[::-1])
    keys, face = keys[order], face[order]
    # A closed surface has each edge on exactly two triangles, so sorted keys come in pairs.
    first, second = face[0::2], face[1::2]
    between = first != second
    bound = int(surface.face.max()) + 1
    pair_keys = np.array([a * bound + b for a, b in pairs], dtype=np.int64)
    wanted = np.isin(np.minimum(first, second) * bound + np.maximum(first, second), pair_keys)
    return keys[0::2][between & wanted]
