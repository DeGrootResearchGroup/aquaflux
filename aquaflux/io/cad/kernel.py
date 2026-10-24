"""The CAD kernel: OpenCASCADE Technology (OCCT), through its Python binding OCP.

**This is the only module that imports the kernel.** It reads a STEP file (the ISO 10303 exchange
format) into named boundary-representation (B-rep) solids, and answers the handful of questions the
rest of :mod:`aquaflux.io.cad` asks of them — what each face is, how far one boundary is from
another, what a solid looks like as triangles — as plain numbers. Nothing it holds is handed further out than
:class:`~aquaflux.io.cad.model.CadModel`, so the bodies and triangles a user receives carry no
kernel object and the rest of the package never needs it installed.

The binding is ``cadquery-ocp`` on the Python Package Index (installed by ``pip install
aquaflux[cad]``), and its API moves between releases — collection types have been renamed and
static methods have lost their suffix — which is the reason every call is kept here, against one
pinned version.

Three details are load-bearing, each found by reading a real file:

* **Units are set before transfer, to metres.** The reader otherwise converts every length to
  millimetres whatever the file says, silently, and every body read would be a thousand times too
  large. Setting the reader's global unit parameter does not reach the document-based reader used
  here; the document's own length unit does.
* **Which side of a face the solid is on comes from the face's orientation**, not from classifying a
  point: in a valid closed solid the surface normal, flipped when the face is reversed, points out
  of the solid everywhere on that surface. A classifier needs a point that is certainly on the face,
  which a face with a hole does not provide at its parameter midpoint.
* **Edge types are not read at all.** The kernel reports straight seam lines as spline curves, so
  nothing here recognizes a solid from its edges; surfaces are reported faithfully. An edge's
  *topology* -- which faces it joins, how sharply they fold there -- is read, for meshing; only what
  kind of curve it is, is not.

Faces, edges and vertices are numbered in **exploration order**: the order a topology explorer first
visits them in a shape, each counted once. A mesher reading the shape's file numbers faces the same
way, so the face indices answered here are the ones it reports.
"""

from __future__ import annotations

import math

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Splitter
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_Transform,
)
from OCP.BRepExtrema import BRepExtrema_DistShapeShape, BRepExtrema_IsInFace
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.BRepTools import BRepTools
from OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    List_TopoDS_Shape,
    Sequence_TDF_Label,
)
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_INTERNAL,
    TopAbs_REVERSED,
    TopAbs_SHAPE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

from aquaflux.io.cad.faces import (
    ConeFace,
    CylinderFace,
    OtherFace,
    PlaneFace,
    SolidDescription,
    SphereFace,
)
from aquaflux.io.cad.placement import Placement
from aquaflux.io.cad.topology import CadTopology, Projection
from aquaflux.solids import Cone, Cylinder, Solid, Sphere, Union

__all__ = ["OpenCascade"]

#: The document length unit that makes every length read come out in metres.
_METRES = 1.0


def _faces(shape) -> list:
    """The shape's faces, in the kernel's own order."""
    found, explorer = [], TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        found.append(TopoDS.Face(explorer.Current()))
        explorer.Next()
    return found


def _indexed(shape, kind):
    """The shape's sub-shapes of one kind, each once, numbered from 1 in exploration order."""
    found = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, kind, found)
    return found


def _coords(entity) -> np.ndarray:
    """A kernel point or direction as a length-3 array."""
    return np.array(entity.Coord(), dtype=float)


class OpenCascade:
    """The kernel's answers, as plain numbers.

    Holds no state of its own; every method takes the shapes it works on. A shape is an opaque
    handle that only this class interprets.
    """

    # -----------------------------------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------------------------------

    def read_step(self, path, placement: Placement) -> tuple[dict, dict]:
        """The named solids in a STEP file, and its sheets, placed, each keyed by name.

        A body's name is its product name in the file. An unnamed one is called ``solid<k>``; a
        name that repeats gets ``[k]`` appended to every occurrence, so no body is dropped for
        sharing a name. An assembly is walked into its components, each at its own location. A
        part holding faces but no solid is a sheet: a surface of no thickness, such as a baffle.

        Returns
        -------
        solids : dict of str to kernel solid
        sheets : dict of str to kernel shape

        Raises
        ------
        OSError
            If the file cannot be read as STEP.
        ValueError
            If it holds no solid.
        """
        document = TDocStd_Document(TCollection_ExtendedString("aquaflux"))
        XCAFDoc_DocumentTool.SetLengthUnit_s(document, _METRES)
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            msg = f"{path} could not be read as a STEP file"
            raise OSError(msg)
        reader.Transfer(document)
        tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())

        named: list[tuple[str, object, bool]] = []
        roots = Sequence_TDF_Label()
        tool.GetFreeShapes(roots)
        for i in range(1, roots.Length() + 1):
            self._collect(tool, roots.Value(i), TopLoc_Location(), named)
        if not any(not sheet for _, _, sheet in named):
            msg = f"{path} holds no solid"
            raise ValueError(msg)

        transform = self._transform(placement)
        counts: dict[str, int] = {}
        for name, _, _ in named:
            counts[name] = counts.get(name, 0) + 1
        seen: dict[str, int] = {}
        solids, sheets = {}, {}
        for name, shape, sheet in named:
            if counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                name = f"{name}[{seen[name] - 1}]"
            placed = BRepBuilderAPI_Transform(shape, transform, True).Shape()
            (sheets if sheet else solids)[name] = placed
        return solids, sheets

    def _collect(self, tool, label, location, named: list, fallback: str = "") -> None:
        """Every solid under ``label``, or the label's sheet when it holds faces but no solid, placed
        by ``location``, with the name it is best known by and whether it is a sheet.

        An assembly's components each carry a location relative to it, so the location is
        accumulated on the way down rather than read from any one level. A part's own name is
        preferred to the name of the instance of it, which exporters often leave generic.
        """
        if tool.IsAssembly_s(label):
            components = Sequence_TDF_Label()
            tool.GetComponents_s(label, components)
            for i in range(1, components.Length() + 1):
                component = components.Value(i)
                part = TDF_Label()
                tool.GetReferredShape_s(component, part)
                placed = location.Multiplied(XCAFDoc_ShapeTool.GetLocation_s(component))
                self._collect(tool, part, placed, named, self._name(component))
            return
        base = self._name(label) or fallback
        shape = tool.GetShape_s(label).Moved(location)
        pieces, explorer = [], TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            pieces.append(TopoDS.Solid(explorer.Current()))
            explorer.Next()
        for k, piece in enumerate(pieces):
            name = base or f"solid{len(named)}"
            named.append((name if len(pieces) == 1 else f"{name}.{k}", piece, False))
        if not pieces and TopExp_Explorer(shape, TopAbs_FACE).More():
            named.append((base or f"solid{len(named)}", shape, True))

    @staticmethod
    def _name(label) -> str:
        attribute = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            return attribute.Get().ToExtString()
        return ""

    @staticmethod
    def _transform(placement: Placement) -> gp_Trsf:
        m, t = placement.matrix, placement.offset
        transform = gp_Trsf()
        # Row-major 3x4; the kernel accepts a reflection here and keeps faces outward.
        transform.SetValues(*m[0], t[0], *m[1], t[1], *m[2], t[2])
        return transform

    # -----------------------------------------------------------------------------------------
    # Describing
    # -----------------------------------------------------------------------------------------

    def describe(self, name: str, shape) -> SolidDescription:
        """Every face of a solid as a record, with its vertices and its size."""
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        extent = float(np.linalg.norm(_coords(box.CornerMax()) - _coords(box.CornerMin())))
        corners, explorer = [], TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            corners.append(_coords(BRep_Tool.Pnt_s(TopoDS.Vertex(explorer.Current()))))
            explorer.Next()
        return SolidDescription(
            name=name,
            faces=tuple(self._describe_face(face) for face in _faces(shape)),
            vertices=np.array(corners).reshape(-1, 3),
            extent=extent,
        )

    @staticmethod
    def _outward(surface, face, u, v) -> tuple[np.ndarray, np.ndarray]:
        """A point on the surface and the unit normal pointing out of the solid there."""
        point, d_u, d_v = gp_Pnt(), gp_Vec(), gp_Vec()
        surface.D1(u, v, point, d_u, d_v)
        normal = _coords(d_u.Crossed(d_v))
        normal /= np.linalg.norm(normal)
        if face.Orientation() == TopAbs_REVERSED:
            normal = -normal
        return _coords(point), normal

    def _describe_face(self, face):
        surface = BRepAdaptor_Surface(face)
        kind = surface.GetType()
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        point, outward = self._outward(surface, face, 0.5 * (u0 + u1), 0.5 * (v0 + v1))

        if kind == GeomAbs_Plane:
            return PlaneFace(point=point, outward_normal=outward)
        if kind == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            origin = _coords(cylinder.Axis().Location())
            axis = _coords(cylinder.Axis().Direction())
            radial = (point - origin) - ((point - origin) @ axis) * axis
            return CylinderFace(
                origin=origin,
                axis=axis,
                radius=float(cylinder.Radius()),
                axial_range=(float(v0), float(v1)),
                angle=float(u1 - u0),
                solid_inside=bool(outward @ radial > 0.0),
            )
        if kind == GeomAbs_Cone:
            # Along the generator ``v``, the axial position is ``v cos a`` and the radius
            # ``R + v sin a``, with ``R`` the radius where ``v`` is zero and ``a`` the half-angle.
            cone = surface.Cone()
            origin = _coords(cone.Axis().Location())
            axis = _coords(cone.Axis().Direction())
            half_angle, reference = float(cone.SemiAngle()), float(cone.RefRadius())
            radial = (point - origin) - ((point - origin) @ axis) * axis
            return ConeFace(
                origin=origin,
                axis=axis,
                axial_range=(v0 * math.cos(half_angle), v1 * math.cos(half_angle)),
                radii=(
                    reference + v0 * math.sin(half_angle),
                    reference + v1 * math.sin(half_angle),
                ),
                angle=float(u1 - u0),
                solid_inside=bool(outward @ radial > 0.0),
            )
        if kind == GeomAbs_Sphere:
            sphere = surface.Sphere()
            centre = _coords(sphere.Location())
            return SphereFace(
                centre=centre,
                radius=float(sphere.Radius()),
                solid_inside=bool(outward @ (point - centre) > 0.0),
            )
        return OtherFace(kind=str(kind).rsplit(".", 1)[-1].removeprefix("GeomAbs_"))

    # -----------------------------------------------------------------------------------------
    # Regions: building bodies back into the kernel, and comparing them
    # -----------------------------------------------------------------------------------------

    def shape_of(self, body: Solid):
        """A body from :mod:`aquaflux.solids` as a kernel solid, so it can be compared with one.

        Raises
        ------
        TypeError
            For a body this does not build: a half-space has no finite counterpart, and a convex
            polyhedron needs none, because its recognition is proven rather than checked.
        """
        if isinstance(body, Cylinder):
            base = np.asarray(body.centre) - float(body.half_length) * np.asarray(body.axis)
            return BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*base), gp_Dir(*np.asarray(body.axis))),
                float(body.radius),
                2.0 * float(body.half_length),
            ).Shape()
        if isinstance(body, Cone):
            base = np.asarray(body.centre) - float(body.half_length) * np.asarray(body.axis)
            return BRepPrimAPI_MakeCone(
                gp_Ax2(gp_Pnt(*base), gp_Dir(*np.asarray(body.axis))),
                float(body.base_radius),
                float(body.tip_radius),
                2.0 * float(body.half_length),
            ).Shape()
        if isinstance(body, Sphere):
            return BRepPrimAPI_MakeSphere(
                gp_Pnt(*np.asarray(body.centre)), float(body.radius)
            ).Shape()
        if isinstance(body, Union):
            return self.fuse([self.shape_of(part) for part in body.bodies])
        msg = (
            f"a {type(body).__name__} has no finite counterpart to build in the CAD kernel, so it "
            "cannot be checked against a solid this way"
        )
        raise TypeError(msg)

    @staticmethod
    def fuse(shapes):
        """The union of several kernel solids."""
        if len(shapes) == 1:
            return shapes[0]
        arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
        arguments.Append(shapes[0])
        for shape in shapes[1:]:
            tools.Append(shape)
        fuse = BRepAlgoAPI_Fuse()
        fuse.SetArguments(arguments)
        fuse.SetTools(tools)
        fuse.Build()
        return fuse.Shape()

    @staticmethod
    def tolerance(shapes) -> float:
        """The largest geometric tolerance the file declares on any of ``shapes``, in metres.

        Every edge, vertex and face of a boundary representation carries the distance within which
        it is known — the precision the drawing was exported at — and nothing can be compared with
        it more finely than that.
        """
        analysis = ShapeAnalysis_ShapeTolerance()
        return max(float(analysis.Tolerance(shape, 1, TopAbs_SHAPE)) for shape in shapes)

    @staticmethod
    def distances(points, shape) -> np.ndarray:
        """The distance from each of ``points`` to the nearest point of ``shape``'s boundary."""
        out = np.empty(len(points))
        for i, point in enumerate(np.asarray(points, dtype=float)):
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            out[i] = BRepExtrema_DistShapeShape(vertex, shape).Value()
        return out

    # -----------------------------------------------------------------------------------------
    # Meshing: a region built from solids, its topology, its file, and points moved onto it
    # -----------------------------------------------------------------------------------------

    def region(self, solids, without, sheets=()):
        """The union of the kernel solids ``solids``, less the union of ``without``, with the parts
        of ``sheets`` inside it standing in it as faces with the region on both sides.

        Returns
        -------
        shape
            The region.
        embedded : list of bool
            For each sheet, whether any of it stands in the region.

        Raises
        ------
        ValueError
            If the sheets cut a solid of the region in two: a sheet must leave a way round it.
        """
        shape = self.fuse(list(solids))
        if without:
            arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
            arguments.Append(shape)
            for tool in without:
                tools.Append(tool)
            cut = BRepAlgoAPI_Cut()
            cut.SetArguments(arguments)
            cut.SetTools(tools)
            cut.Build()
            shape = cut.Shape()
        if not sheets:
            return shape, []
        arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
        arguments.Append(shape)
        for sheet in sheets:
            tools.Append(sheet)
        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(arguments)
        splitter.SetTools(tools)
        splitter.Build()
        before, shape = _indexed(shape, TopAbs_SOLID).Size(), splitter.Shape()
        pieces = _indexed(shape, TopAbs_SOLID).Size()
        if pieces != before:
            msg = (
                f"the sheets cut the region's {before} solid(s) into {pieces}; a sheet standing in "
                "a region must leave a way round it"
            )
            raise ValueError(msg)
        inside = [face for face in _faces(shape) if face.Orientation() == TopAbs_INTERNAL]
        embedded = []
        for sheet in sheets:
            images = []
            for face in _faces(sheet):
                modified = splitter.Modified(face)
                images += list(modified) if modified.Size() else [face]
            embedded.append(any(image.IsSame(face) for image in images for face in inside))
        return shape, embedded

    def topology(self, shape) -> CadTopology:
        """How a shape's faces, edges and vertices connect, each numbered in exploration order."""
        faces, edges, vertices = (
            _indexed(shape, kind) for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX)
        )
        ancestors = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)

        inside = [face for face in _faces(shape) if face.Orientation() == TopAbs_INTERNAL]
        records, areas, sheet = [], [], []
        for index in range(1, faces.Size() + 1):
            face = TopoDS.Face(faces.FindKey(index))
            records.append(self._describe_face(face))
            sheet.append(any(face.IsSame(other) for other in inside))
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, properties)
            areas.append(properties.Mass())

        ends, lengths, offsets, members, angles, seams = [], [], [0], [], [], []
        for index in range(1, edges.Size() + 1):
            edge = TopoDS.Edge(edges.FindKey(index))
            ends.append(
                [
                    vertices.FindIndex(TopExp.FirstVertex_s(edge)) - 1,
                    vertices.FindIndex(TopExp.LastVertex_s(edge)) - 1,
                ]
            )
            degenerate = BRep_Tool.Degenerated_s(edge)
            lengths.append(
                0.0 if degenerate else GCPnts_AbscissaPoint.Length_s(BRepAdaptor_Curve(edge))
            )
            touching = sorted({faces.FindIndex(face) - 1 for face in ancestors.FindFromKey(edge)})
            members += touching
            offsets.append(len(members))
            first = TopoDS.Face(faces.FindKey(touching[0] + 1))
            seams.append(bool(BRep_Tool.IsClosed_s(edge, first)))
            # A fold is not defined where a sheet meets a face: the region is on both sides of it.
            if len(touching) == 2 and not degenerate and not any(sheet[f] for f in touching):
                second = TopoDS.Face(faces.FindKey(touching[1] + 1))
                angles.append(self._fold(edge, first, second))
            else:
                angles.append(math.nan)

        corners = [
            _coords(BRep_Tool.Pnt_s(TopoDS.Vertex(vertices.FindKey(i))))
            for i in range(1, vertices.Size() + 1)
        ]
        return CadTopology(
            faces=tuple(records),
            face_area=np.array(areas),
            vertices=np.array(corners).reshape(-1, 3),
            edge_vertices=np.array(ends, dtype=np.int64).reshape(-1, 2),
            edge_length=np.array(lengths),
            edge_face_offsets=np.array(offsets, dtype=np.int64),
            edge_face_indices=np.array(members, dtype=np.int64),
            edge_angle=np.array(angles),
            seam=np.array(seams, dtype=bool),
            sheet=np.array(sheet, dtype=bool),
        )

    def _fold(self, edge, first, second) -> float:
        """The signed angle between two faces' outward normals at an edge's midpoint.

        Positive where the solid is convex. The direction from the edge into ``first``, along
        ``first``, is its outward normal crossed with the edge's tangent as ``first`` traverses it
        (a face lies to the left of its boundary seen from outside); the fold is convex exactly when
        ``second``'s outward normal points against that direction.
        """
        curve = BRepAdaptor_Curve(edge)
        middle = 0.5 * (curve.FirstParameter() + curve.LastParameter())
        point, tangent = gp_Pnt(), gp_Vec()
        curve.D1(middle, point, tangent)
        normals = []
        for face in (first, second):
            uv = BRepAdaptor_Curve2d(edge, face).Value(middle)
            normals.append(self._outward(BRepAdaptor_Surface(face), face, uv.X(), uv.Y())[1])
        along = _coords(tangent)
        # Exploring a face composes its own orientation into its edges', so this is the direction
        # the face as it sits in the solid traverses the edge.
        if self._orientation_in(edge, first) == TopAbs_REVERSED:
            along = -along
        into_first = np.cross(normals[0], along)
        magnitude = math.acos(float(np.clip(normals[0] @ normals[1], -1.0, 1.0)))
        return magnitude if normals[1] @ into_first < 0.0 else -magnitude

    @staticmethod
    def _orientation_in(edge, face):
        """How ``face``'s boundary traverses ``edge``: forward or reversed."""
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            if explorer.Current().IsSame(edge):
                return explorer.Current().Orientation()
            explorer.Next()
        msg = "an edge's face does not list the edge"
        raise ValueError(msg)

    @staticmethod
    def write_brep(shape, path) -> None:
        """Write ``shape`` in the kernel's own boundary-representation (BREP) file format.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        if not BRepTools.Write_s(shape, str(path)):
            msg = f"could not write {path}"
            raise OSError(msg)

    def project(self, shape, points, faces=None, edges=None) -> Projection:
        """Each point moved to the closest point of ``shape``'s boundary, or of a face or edge of it.

        Parameters
        ----------
        points : array-like, shape ``(n, 3)``
        faces : array-like of int, shape ``(n,)``, optional
            The face (in exploration order) to move each point onto.
        edges : array-like of int, shape ``(n,)``, optional
            The edge (in exploration order) to move each point onto, in place of ``faces``; the face
            reported is the first meeting there. With neither, the closest point of the whole
            boundary.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        indexed = _indexed(shape, TopAbs_FACE)
        indexed_edges = _indexed(shape, TopAbs_EDGE) if edges is not None else None
        # The boundary, not the solid: a point inside a solid is at distance zero from it.
        boundary = self._boundary(shape)
        ancestors = None
        landed = np.empty_like(points)
        on_face = np.empty(len(points), dtype=np.int64)
        normals = np.empty_like(points)
        distance = np.empty(len(points))
        for i, point in enumerate(points):
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            if edges is not None:
                edge = indexed_edges.FindKey(int(edges[i]) + 1)
                extrema = BRepExtrema_DistShapeShape(vertex, edge)
                if ancestors is None:
                    ancestors = self._faces_around(shape)
                face_index = min(
                    indexed.FindIndex(face) - 1 for face in ancestors.FindFromKey(edge)
                )
            elif faces is None:
                extrema = BRepExtrema_DistShapeShape(vertex, boundary)
                support = extrema.SupportOnShape2(1)
                if extrema.SupportTypeShape2(1) == BRepExtrema_IsInFace:
                    face_index = indexed.FindIndex(support) - 1
                else:
                    if ancestors is None:
                        ancestors = self._faces_around(shape)
                    face_index = min(
                        indexed.FindIndex(face) - 1 for face in ancestors.FindFromKey(support)
                    )
            else:
                face_index = int(faces[i])
                extrema = BRepExtrema_DistShapeShape(vertex, indexed.FindKey(face_index + 1))
            face = TopoDS.Face(indexed.FindKey(face_index + 1))
            nearest = extrema.PointOnShape2(1)
            u, v = GeomAPI_ProjectPointOnSurf(
                nearest, BRep_Tool.Surface_s(face)
            ).LowerDistanceParameters()
            landed[i] = _coords(nearest)
            on_face[i] = face_index
            normals[i] = self._outward(BRepAdaptor_Surface(face), face, u, v)[1]
            distance[i] = extrema.Value()
        return Projection(points=landed, face=on_face, normal=normals, distance=distance)

    @staticmethod
    def _boundary(shape):
        """The shape's boundary shells, as one compound."""
        compound, builder = TopoDS_Compound(), BRep_Builder()
        builder.MakeCompound(compound)
        explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        while explorer.More():
            builder.Add(compound, explorer.Current())
            explorer.Next()
        return compound

    @staticmethod
    def _faces_around(shape):
        """The faces meeting at each edge and at each vertex of ``shape``."""
        around = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, around)
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_VERTEX, TopAbs_FACE, around)
        return around

    # -----------------------------------------------------------------------------------------
    # Triangulating
    # -----------------------------------------------------------------------------------------

    def triangulate(self, shape, chord: float, angle: float, facet_size: float | None):
        """A solid's surface as triangles wound outward, every vertex on the true surface.

        Parameters
        ----------
        chord : float
            The largest distance, in metres, a triangle may lie from the true surface.
        angle : float
            The largest angle, in radians, between the normals at a triangle's ends.
        facet_size : float or None
            If given, the surface is first cut by three families of planes this far apart, one
            family normal to each axis, so every piece fits inside a cube of this side and no
            triangle meshed on it is longer than the cube's diagonal. The kernel's mesher bounds
            only the chord, so without this a straight cylinder comes out as slivers running its
            whole length. Only the boundary shells are cut -- cutting the solid would mesh the
            cutting planes' interior faces as well.

        Returns
        -------
        triangles : np.ndarray, shape ``(n_triangles, 3, 3)``
            Triangles of positive area only: a mesher can emit a degenerate one at a pole, and a
            zero-area facet is not a small emitter but a point, which an emitting surface must
            not contain by accident.
        face : np.ndarray of int, shape ``(n_triangles,)``
            The face of ``shape``, in exploration order, each triangle lies on -- the face it was
            cut from, when the surface was cut.
        """
        original = _indexed(shape, TopAbs_FACE)
        if facet_size is None:
            surface, origin = shape, None
        else:
            surface, splitter = self._cut(shape, facet_size)
            origin = _indexed(surface, TopAbs_FACE)
            source = np.full(origin.Size(), -1, dtype=np.int64)
            for index in range(1, original.Size() + 1):
                face = original.FindKey(index)
                pieces = list(splitter.Modified(face)) or [face]
                for piece in pieces:
                    position = origin.FindIndex(piece)
                    if position:
                        source[position - 1] = index - 1
        BRepMesh_IncrementalMesh(surface, chord, False, angle, True)
        triangles, faces = [], []
        for face in _faces(surface):
            location = TopLoc_Location()
            mesh = BRep_Tool.Triangulation_s(face, location)
            if mesh is None:
                msg = "the CAD kernel could not triangulate a face; try a larger chord tolerance"
                raise ValueError(msg)
            placed = location.Transformation()
            points = np.array(
                [mesh.Node(i).Transformed(placed).Coord() for i in range(1, mesh.NbNodes() + 1)]
            )
            corners = (
                np.array([mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles() + 1)]) - 1
            )
            if face.Orientation() == TopAbs_REVERSED:
                corners = corners[:, ::-1]
            index = (original if origin is None else origin).FindIndex(face) - 1
            triangles.append(points[corners])
            faces.append(np.full(len(corners), index if origin is None else source[index]))
        triangles, faces = np.concatenate(triangles), np.concatenate(faces)
        doubled = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        if np.any(faces < 0):
            msg = "the CAD kernel cut a face into pieces it could not trace back to the face"
            raise ValueError(msg)
        return triangles[doubled > 0.0], faces[doubled > 0.0]

    @staticmethod
    def _cut(shape, spacing: float):
        """The solid's boundary shells, cut by planes normal to each axis ``spacing`` apart.

        Returns the cut shells and the splitter, whose history traces each piece to its face.
        """
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        low, high = _coords(box.CornerMin()), _coords(box.CornerMax())
        reach = 2.0 * float(np.linalg.norm(high - low)) + 1.0
        tools = List_TopoDS_Shape()
        for axis in range(3):
            pieces = math.ceil((high[axis] - low[axis]) / spacing)
            normal = np.eye(3)[axis]
            for k in range(1, pieces):
                point = low + normal * (high[axis] - low[axis]) * k / pieces
                plane = gp_Pln(gp_Pnt(*point), gp_Dir(*normal))
                tools.Append(BRepBuilderAPI_MakeFace(plane, -reach, reach, -reach, reach).Face())
        shells, explorer = List_TopoDS_Shape(), TopExp_Explorer(shape, TopAbs_SHELL)
        while explorer.More():
            shells.Append(explorer.Current())
            explorer.Next()
        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(shells)
        splitter.SetTools(tools)
        splitter.Build()
        return splitter.Shape(), splitter
