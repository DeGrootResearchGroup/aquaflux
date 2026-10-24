"""How a computer-aided design (CAD) region's faces, edges and vertices connect, as plain arrays.

A mesher needs more of a boundary-representation (B-rep) solid than recognition does: which faces
meet along which edge (so a mesh can keep that edge sharp, and a pipe can be found as the cylinder
it is joined to the vessel by), whether the fold at an edge is convex or concave (a wall layer
grows outward at one and must shrink at the other), the face each triangle of the surface lies on,
and the closest point of a face to a point near it. These records carry those answers without the
kernel that computed them, as :mod:`aquaflux.io.cad.faces` does for the face records.

**Faces are numbered in the kernel's exploration order** -- the order a topology explorer first
visits them in the region's shape -- and every record here uses that one numbering. It is also the
order a mesher reading the region's file numbers them in, so a face index passes between the two
unchanged. Edges and vertices are numbered the same way, each counted once however many faces share
it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["CadRegion", "CadTopology", "FaceTriangles", "Projection"]


@dataclasses.dataclass(frozen=True)
class CadRegion:
    """A region of space bounded by a CAD model's solids: the union of some, less the union of others,
    with sheets standing in it.

    A plain description, not a shape: :class:`~aquaflux.io.cad.CadModel` builds the region from it
    and answers questions about it, so the kernel's shape never leaves the model.

    Attributes
    ----------
    solids : tuple of str
        The solids whose union is the region: a vessel and the pipes joining it, say.
    without : tuple of str
        The solids taken out of it: a lamp sleeve standing in the vessel, say. Default: none.
    baffles : tuple of str, or None
        The sheets standing in it -- surfaces of no thickness, the region on both sides, such as a
        baffle plate -- each cut to the part inside the region. Unset, every sheet of the model that
        stands in the region; empty, none.

    Examples
    --------
    The water in the Sozzi reactor::

        CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    """

    solids: tuple[str, ...]
    without: tuple[str, ...] = ()
    baffles: tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "solids", tuple(self.solids))
        object.__setattr__(self, "without", tuple(self.without))
        if self.baffles is not None:
            object.__setattr__(self, "baffles", tuple(self.baffles))
        if not self.solids:
            raise ValueError("a region needs at least one solid")
        both = set(self.solids) & set(self.without)
        if both:
            raise ValueError(f"{sorted(both)} are both in the region and taken out of it")


@dataclasses.dataclass(frozen=True)
class CadTopology:
    """A region's faces, edges and vertices, and how they connect.

    Attributes
    ----------
    faces : tuple
        One face record (:mod:`aquaflux.io.cad.faces`) per face, in exploration order: what surface
        it lies on and which side of it the region is on.
    face_area : np.ndarray, shape ``(n_faces,)``
        Each face's area.
    vertices : np.ndarray, shape ``(n_vertices, 3)``
        The topological vertices, where edges end.
    edge_vertices : np.ndarray of int, shape ``(n_edges, 2)``
        Each edge's two end vertices; equal for a closed edge (a circle) or a degenerate one (a
        sphere's pole).
    edge_length : np.ndarray, shape ``(n_edges,)``
        Each edge's length; zero for a degenerate edge.
    edge_face_offsets : np.ndarray of int, shape ``(n_edges + 1,)``
        Compressed-sparse-row (CSR) row pointers into :attr:`edge_face_indices`.
    edge_face_indices : np.ndarray of int
        The distinct faces each edge bounds, concatenated: two on a closed region's ordinary edge,
        one on a seam (a face meeting itself), a degenerate edge or a sheet's free edge, and more
        where a sheet meets a surface.
    edge_angle : np.ndarray, shape ``(n_edges,)``
        The fold at an edge between two faces: the angle, in radians, between their outward normals
        at the edge's midpoint, **positive where the region is convex** (a box's edges, ``pi/2``)
        and **negative where it is concave** (the inside corner of an L). ``0`` where the faces
        meet smoothly, and ``nan`` on an edge that does not bound exactly two faces or bounds a
        sheet.
    seam : np.ndarray of bool, shape ``(n_edges,)``
        Whether the edge is a seam: a face's boundary running along itself, where a closed surface
        (a cylinder, a sphere) is cut open to be parametrized. Not a feature of the region.
    sheet : np.ndarray of bool, shape ``(n_faces,)``
        Whether the face is part of a sheet standing in the region (see :class:`CadRegion`), the
        region on both sides of it rather than one. Every edge of a sheet is a feature. Unset,
        no face is.
    """

    faces: tuple
    face_area: np.ndarray
    vertices: np.ndarray
    edge_vertices: np.ndarray
    edge_length: np.ndarray
    edge_face_offsets: np.ndarray
    edge_face_indices: np.ndarray
    edge_angle: np.ndarray
    seam: np.ndarray
    sheet: np.ndarray | None = None

    def __post_init__(self):
        if self.sheet is None:
            object.__setattr__(self, "sheet", np.zeros(len(self.faces), dtype=bool))

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.edge_vertices.shape[0]

    def faces_of_edge(self, edge: int) -> np.ndarray:
        """The distinct faces ``edge`` bounds."""
        return self.edge_face_indices[
            self.edge_face_offsets[edge] : self.edge_face_offsets[edge + 1]
        ]

    def edges_of_face(self, face: int) -> np.ndarray:
        """The edges bounding ``face``, ascending."""
        members = np.repeat(np.arange(self.n_edges), np.diff(self.edge_face_offsets))
        return np.unique(members[self.edge_face_indices == face])

    def face_neighbours(self) -> np.ndarray:
        """Every pair of distinct faces sharing an edge, shape ``(n_pairs, 2)``, each pair ascending.

        Each pair appears once however many edges the two faces share.
        """
        pairs = []
        for edge in range(self.n_edges):
            faces = np.sort(self.faces_of_edge(edge))
            pairs += [(a, b) for i, a in enumerate(faces) for b in faces[i + 1 :]]
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.unique(np.array(pairs, dtype=np.int64), axis=0)

    def feature_edges(self, angle: float) -> np.ndarray:
        """The edges folded by more than ``angle`` radians, either way, ascending.

        Parameters
        ----------
        angle : float
            The smallest fold, in radians, that counts as a feature a mesh must keep.

        Returns
        -------
        np.ndarray of int
            Edges whose :attr:`edge_angle` exceeds ``angle`` in magnitude, together with every edge
            that does not bound exactly two faces (where a fold is not defined, the edge is a
            feature by construction) other than a seam.
        """
        undefined = np.isnan(self.edge_angle) & ~self.seam & (self.edge_length > 0.0)
        folded = np.abs(np.nan_to_num(self.edge_angle)) > angle
        return np.flatnonzero(folded | undefined)


@dataclasses.dataclass(frozen=True)
class FaceTriangles:
    """A region's surface as triangles, each tagged by the face it lies on.

    Attributes
    ----------
    triangles : np.ndarray, shape ``(n_triangles, 3, 3)``
        Triangle corners, wound outward from the region, every corner on the true surface.
    face : np.ndarray of int, shape ``(n_triangles,)``
        The face (in exploration order) each triangle lies on.
    """

    triangles: np.ndarray
    face: np.ndarray


@dataclasses.dataclass(frozen=True)
class Projection:
    """Points moved onto a region's boundary.

    Attributes
    ----------
    points : np.ndarray, shape ``(n, 3)``
        The closest point of the boundary (or of the face asked for) to each point given.
    face : np.ndarray of int, shape ``(n,)``
        The face each lands on. A point closest to an edge or a vertex lands on every face meeting
        there at once; it is given the first of them in exploration order.
    normal : np.ndarray, shape ``(n, 3)``
        The unit normal pointing out of the region at each landing point, on that face.
    distance : np.ndarray, shape ``(n,)``
        How far each point moved.
    """

    points: np.ndarray
    face: np.ndarray
    normal: np.ndarray
    distance: np.ndarray
