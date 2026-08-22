"""Face geometry strategies: area, centroid, and owner-outward unit normal per face.

Face geometry specializes by spatial dimension, so it is a strategy hierarchy —
:class:`EdgeFaceGeometry` (2D) and :class:`PolygonFaceGeometry` (3D), selected by
:func:`face_geometry_scheme` — each an ``equinox.Module``. A scheme computes the *raw*
(node-order) area/centroid/normal for all faces at once; the shared, dimension-agnostic
:meth:`FaceGeometryScheme.orient_owner_outward` fixes the normal to point out of the
owner cell.

**2D** — a face is an edge with endpoints ``p0``, ``p1`` and displacement ``d = p1 - p0``:

    area     = |d|
    centroid = (p0 + p1) / 2
    normal   = (d_y, -d_x) / |d|              # rotate the edge -90 degrees, normalize

**3D** — a face is an arbitrary polygon, decomposed by a **centre fan**: triangles
``(c, v_i, v_{i+1})`` around the perimeter, with apex ``c`` the face's vertex mean. Each
triangle has directed area ``d_i = 0.5 (v_i - c) x (v_{i+1} - c)`` and centroid
``(c + v_i + v_{i+1}) / 3``. The **vector area** ``S = sum_i d_i`` is independent of the
apex and of which vertex is listed first (it is the polygon boundary integral). Then:

    area     = |S|                            # projected area (the quantity fluxes use)
    normal   = S / |S|                        # unit, for warped faces too
    centroid = (sum_i (d_i . n) c_i) / |S|    # signed projected-area weights

The centre fan is robust where simpler decompositions are not. A node-0 fan with
``area = sum|d_i|`` yields a sub-unit normal and a vertex-order-dependent area on warped
faces; averaging both quad diagonals recovers ``|S|`` + a unit normal but only for quads.
The centre fan gives ``|S|`` + unit normal for *any* polygon and a symmetric,
vertex-order-independent centroid. The centroid weights each fan triangle by its **signed**
projected area ``d_i . n`` (not the unsigned ``|d_i|``), so it stays exact for **non-convex**
faces, where a reflex vertex makes one fan triangle wind against the face normal. See
``quality.py`` for the planarity and centroid-iteration diagnostics that confirm one pass
suffices for realistic warp.

**Node ordering is a precondition.** Each face's node list must trace the face *perimeter*
(consecutive nodes share an edge); the winding *direction* is free — the owner-outward flip
fixes the normal's sign — but a non-perimeter order (e.g. listing a quad's nodes across its
diagonal) makes the centre-fan triangles span a self-intersecting polygon and the geometry
is silently meaningless. This is not checked (it is not a topological property); it is the
caller's contract, stated on :meth:`~aquaflux.mesh.Mesh.from_faces`.

A degenerate face — zero length (2D) or zero vector area (3D), from coincident or collinear
nodes — has no well-defined normal. The normalization here is zero-safe (a degenerate face
yields a zero normal and a finite gradient, never a NaN that silently poisons the solver),
and such faces are rejected up front by :meth:`~aquaflux.mesh.Mesh.validate` (repeated node)
or surfaced by the ``quality.py`` diagnostics (collinear nodes).

Face node-lists are stored ragged, in compressed-sparse-row (CSR) form. The traversal — which node follows which around a
face's perimeter, and how to sum per-triangle contributions into faces — is owned by
:class:`~aquaflux.mesh.connectivity.FaceNodeConnectivity`, so a scheme here writes only the
polygon math. That traversal is enumerated once at build time (its per-face count is
data-dependent); ``unoriented_geometry`` itself is plain array arithmetic over the already-
enumerated traversal, and :meth:`~aquaflux.mesh.Mesh.geometry` runs it as part of one
compiled call.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, norm_squared, scale

from .connectivity import FaceNodeConnectivity


def _safe_magnitude(vectors: jnp.ndarray) -> jnp.ndarray:
    """Euclidean magnitude along the last axis, with a zero-safe gradient at the zero vector.

    ``jnp.linalg.norm`` returns a NaN *gradient* at exactly the zero vector (``d/dx √(x·x)`` is
    ``0/0`` there). Geometry is differentiable w.r.t. node coordinates, and a degenerate face —
    or even a single collinear fan triangle on an otherwise valid face — can make an intermediate
    vector exactly zero, so the plain norm would inject a NaN into ``grad`` with no NaN in the
    forward value. This masks the square root's argument so the taken branch never differentiates
    ``√0``: the magnitude is ``0`` at the zero vector and exact elsewhere.

    Parameters
    ----------
    vectors : jnp.ndarray
        Array of vectors, shape ``(..., dim)``.

    Returns
    -------
    jnp.ndarray
        Magnitudes, shape ``(...,)``.
    """
    sq = norm_squared(vectors)
    return jnp.where(sq > 0.0, jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)


def _safe_unit(vectors: jnp.ndarray) -> jnp.ndarray:
    """Unit vectors along the last axis, zero-safe in value and gradient (see :func:`_safe_magnitude`).

    A zero input vector maps to a zero output vector (a degenerate face has no unit normal), with a
    finite gradient rather than a ``0/0`` NaN.

    Parameters
    ----------
    vectors : jnp.ndarray
        Array of vectors, shape ``(..., dim)``.

    Returns
    -------
    jnp.ndarray
        Unit vectors (or zero where the input is zero), same shape.
    """
    sq = jnp.sum(vectors * vectors, axis=-1, keepdims=True)
    return jnp.where(sq > 0.0, vectors / jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)


class FaceGeometry(eqx.Module):
    """Per-face geometric quantities, with an owner-outward unit normal.

    Attributes
    ----------
    area : jnp.ndarray
        Face areas, shape ``(n_faces,)`` (edge lengths in 2D, polygon areas in 3D).
    centroid : jnp.ndarray
        Face centroids, shape ``(n_faces, dim)``.
    normal : jnp.ndarray
        Unit face normals oriented out of the owner cell, shape ``(n_faces, dim)``.
    """

    area: jnp.ndarray
    centroid: jnp.ndarray
    normal: jnp.ndarray


class FaceGeometryScheme(eqx.Module):
    """Strategy interface for computing face geometry (dimension-specialized).

    A scheme returns area and centroid in final form, but a normal whose *sign* follows the
    node-winding order — the "unoriented" normal. :meth:`orient_owner_outward` then flips it to
    point out of the owner cell (a separate step because the owner-outward test needs the
    approximate cell centroids, which in turn need the face centroids computed here first).
    """

    @abc.abstractmethod
    def unoriented_geometry(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(area, centroid, node_order_normal)`` for every face.

        Area and centroid are final; the normal's sign follows the node-winding order and is
        fixed to owner-outward by :meth:`orient_owner_outward`.

        Parameters
        ----------
        node_coords : jnp.ndarray
            Node coordinates, shape ``(n_nodes, dim)``.
        face_nodes : FaceNodeConnectivity
            The face→node gather/reduce operators (perimeter traversal).
        """

    @abc.abstractmethod
    def warp_first_moment(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
        centroid: jnp.ndarray,
        normal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return ``P_jk = integral over the face of (x - x_ip)_j n_k dS``, shape ``(n_faces, dim, dim)``.

        **This is identically zero for a planar face, and it is the exact measure of how much a
        warped one departs from one.** A planar face has a constant normal and its centroid
        satisfies ``integral (x - x_ip) dS = 0``, so the two factorize and vanish; on a warped face
        the normal varies over the surface and the integral does not.

        It exists because a second-order Green--Gauss reconstruction silently assumes this quantity
        is zero: the standard derivation replaces ``integral x (x) n dS`` with ``x_ip (x) S``, which
        is the same assumption. Schemes that carry a Hessian need the true value, since the term it
        multiplies is exactly the curvature correction they exist to apply.

        Parameters
        ----------
        node_coords : jnp.ndarray
            Node coordinates, shape ``(n_nodes, dim)``.
        face_nodes : FaceNodeConnectivity
            The face->node gather/reduce operators (perimeter traversal).
        centroid : jnp.ndarray
            Face centroids, shape ``(n_faces, dim)`` -- the point the moment is taken about, and
            the same centroid the face's own geometry was built with.
        normal : jnp.ndarray
            The face's **oriented** unit normal, shape ``(n_faces, dim)``. Required because the
            moment carries the normal's sign, and the triangulation's own winding is not the
            orientation a caller uses: the result must point the same way as the stored normal, or
            it is negated on every face whose nodes happen to wind the other way.
        """

    def orient_owner_outward(
        self,
        node_order_normals: jnp.ndarray,
        face_centroids: jnp.ndarray,
        owner_approx_centroid: jnp.ndarray,
    ) -> jnp.ndarray:
        """Flip node-order normals so each points out of its owner cell.

        A normal points out of the owner when the vector from the owner's (approximate)
        centroid to the face centroid has a non-negative projection onto it. This flip is
        applied once here so the stored normal is owner-outward. Dimension-agnostic.
        """
        outward = face_centroids - owner_approx_centroid
        projection = jnp.sum(outward * node_order_normals, axis=1, keepdims=True)
        sign = jnp.where(projection < 0.0, -1.0, 1.0)
        return node_order_normals * sign


class EdgeFaceGeometry(FaceGeometryScheme):
    """2D strategy: a face is an edge with exactly two nodes."""

    def unoriented_geometry(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # A 2D face is an edge with exactly two nodes (enforced by ``Mesh.validate``), so the
        # per-incidence node gather reshapes cleanly to the two endpoints of each face.
        verts = face_nodes.gather_node_coords(node_coords).reshape(face_nodes.n_faces, 2, -1)
        p0, p1 = verts[:, 0], verts[:, 1]
        d = p1 - p0
        area = _safe_magnitude(d)
        centroid = 0.5 * (p0 + p1)
        normal = _safe_unit(jnp.stack([d[:, 1], -d[:, 0]], axis=1))
        return area, centroid, normal

    def warp_first_moment(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
        centroid: jnp.ndarray,
        normal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Exactly zero: a 2D face is a straight segment, which cannot be warped.

        Returned as a real array of the right shape rather than as a special case for the caller to
        branch on, so a scheme applies the same formula in both dimensions and pays only a multiply
        by zero in 2D (which the compiler folds away against a constant).
        """
        return jnp.zeros((face_nodes.n_faces, centroid.shape[-1], centroid.shape[-1]))


class PolygonFaceGeometry(FaceGeometryScheme):
    """3D strategy: an arbitrary polygon face, decomposed by a centre fan.

    Gives the vector-area unit normal and physical area ``|S|`` for any polygon, warped or
    not, with a vertex-order-independent centroid.
    """

    def unoriented_geometry(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        area, centroid, normal, _ = self.centre_fan(node_coords, face_nodes)
        return area, centroid, normal

    def warp_first_moment(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
        centroid: jnp.ndarray,
        normal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sum ``(g_t - x_ip) (x) S_t`` over the centre-fan triangles -- exact, with no quadrature.

        The integrand is *linear* in ``x`` over each triangle, and a triangle is planar with a
        constant normal, so ``integral over t of (x - x_ip) n dS`` is exactly
        ``(g_t - x_ip) (x) S_t`` for its centroid ``g_t`` and vector area ``S_t``. No quadrature
        rule is involved and none is approximated: the fan reproduces the polygon exactly, so the
        sum is the true surface integral over the warped face.
        """
        pb = face_nodes.gather_node_coords(node_coords)
        pc = face_nodes.perimeter_next(pb)
        apex = face_nodes.vertex_mean(node_coords)[face_nodes.face_of_incidence]
        directed = 0.5 * jnp.cross(pb - apex, pc - apex, axis=-1)  # S_t = A_t * n_t
        offset = (apex + pb + pc) / 3.0 - centroid[face_nodes.face_of_incidence]
        moment = face_nodes.reduce_to_faces(offset[:, :, None] * directed[:, None, :])
        # ⚠️ ORIENTED TO MATCH THE SUPPLIED NORMAL. The fan's triangle normals follow the node
        # winding, which is the "unoriented" convention `orient_owner_outward` exists to fix; a
        # moment left in that convention is sign-flipped on every face whose nodes wind away from
        # the caller's normal, which is roughly half of them on a real mesh.
        winding = face_nodes.reduce_to_faces(directed)
        sign = jnp.where(jnp.sum(winding * normal, axis=-1) < 0.0, -1.0, 1.0)
        return moment * sign[:, None, None]

    def centre_fan(
        self,
        node_coords: jnp.ndarray,
        face_nodes: FaceNodeConnectivity,
        apex: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Centre-fan decomposition of every face.

        Triangles ``(c, v_i, v_{i+1})`` around each face's perimeter, apex ``c`` the face's
        vertex mean (or the supplied ``apex`` per face — used by the centroid-iteration
        diagnostic in ``quality.py``). The perimeter traversal (which vertex is next, summing
        triangles into faces) is provided by ``face_nodes``; the triangle math is here.

        Parameters
        ----------
        node_coords : jnp.ndarray
            Node coordinates, shape ``(n_nodes, 3)``.
        face_nodes : FaceNodeConnectivity
            The face→node gather/reduce operators.
        apex : jnp.ndarray, optional
            Per-face apex, shape ``(n_faces, 3)``; defaults to each face's vertex mean.

        Returns
        -------
        (area, centroid, normal, total_triangle_area)
            ``area = |S|``; ``normal = S/|S|`` (unit); ``total_triangle_area = sum_i|d_i|``
            (``>= |S|``, equal iff planar) — its ratio to ``area`` is the planarity metric.

        Notes
        -----
        The centroid weights each fan triangle by its **signed** projected area ``d_i . n``
        (which sums to ``|S|``), not the unsigned ``|d_i|``. The two agree for a convex face —
        every fan triangle winds with the face normal — but on a non-convex face a reflex
        vertex makes one triangle wind against it, and only the signed weighting gives the true
        area-weighted centroid. ``total_triangle_area`` keeps the unsigned sum, purely for the
        planarity diagnostic.
        """
        pb = face_nodes.gather_node_coords(node_coords)  # edge start vertex per triangle
        pc = face_nodes.perimeter_next(pb)  # edge end vertex (perimeter-wrapped)
        centre = apex if apex is not None else face_nodes.vertex_mean(node_coords)
        pa = centre[face_nodes.face_of_incidence]  # apex per edge-triangle

        directed = 0.5 * jnp.cross(pb - pa, pc - pa, axis=-1)  # (n_triangles, 3)
        tri_centroid = (pa + pb + pc) / 3.0

        vector_area = face_nodes.reduce_to_faces(directed)  # S
        area = _safe_magnitude(vector_area)  # |S|
        normal = _safe_unit(vector_area)
        # Signed projected area of each fan triangle onto the (unit) face normal. Summed per face
        # this is exactly |S|, so it is the correct area-weight — and it is negative for a
        # reflex-vertex triangle, which the unsigned |d_i| would wrongly add.
        signed = dot(directed, normal[face_nodes.face_of_incidence])
        safe_area = jnp.where(area > 0.0, area, 1.0)[
            :, None
        ]  # degenerate face -> 0 centroid, no NaN
        centroid = face_nodes.reduce_to_faces(scale(tri_centroid, signed)) / safe_area
        total_tri_area = face_nodes.reduce_to_faces(_safe_magnitude(directed))
        return area, centroid, normal, total_tri_area


def face_geometry_scheme(dim: int) -> FaceGeometryScheme:
    """Factory: the face-geometry strategy for a given spatial dimension."""
    if dim == 2:
        return EdgeFaceGeometry()
    if dim == 3:
        return PolygonFaceGeometry()
    raise ValueError(f"Unsupported spatial dimension: {dim} (expected 2 or 3)")
