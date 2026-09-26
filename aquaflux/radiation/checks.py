"""Build-time geometry checks that must pass before a surface set is used.

Each check here exists because the failure it catches is *silent*. A gather produces a
plausible field from bad geometry — dimmer here, brighter there — and nothing in the numbers
says so. These run once, on the host, while the surface set is being built, where a raised
exception costs a second and a wrong answer costs an afternoon.

The one that fires most often in practice is inconsistent winding. A triangulated surface
carries no orientation of its own; the outward direction is inferred from the order the
vertices are stored in, and exporters, boolean operations and hand edits all produce files in
which some triangles disagree with their neighbours. Those facets end up with normals pointing
into the solid, where the source-side visibility clamp discards them — so the surface simply
emits less than it should, over whatever patch happens to be reversed, with no error anywhere.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from aquaflux.radiation.solid_angle import signed_solid_angle
from aquaflux.radiation.triangles import padded_length
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = [
    "WindingReport",
    "check_points_outside",
    "check_profiles",
    "check_winding",
    "enclosure_winding",
    "open_facets",
    "stored_normal_disagreement",
    "winding_report",
]

#: Largest winding number, in magnitude, that a closed surface plausibly reports at a point it
#: does not enclose. A closed box measures around 1e-16 there; a bare disc measures 0.45, which
#: is what this is set to catch. Nothing in between is expected, so the value is not delicate.
_OPEN_SURFACE = 0.01


@dataclasses.dataclass(frozen=True)
class WindingReport:
    """What a winding check found, whether or not it was fatal.

    Attributes
    ----------
    conflicting_edges : np.ndarray of int, shape ``(n_conflicts, 2)``
        Unique-vertex index pairs traversed in the *same* direction by two triangles, which is
        what a reversed triangle looks like to its neighbour.
    conflicting_facets : np.ndarray of int, shape ``(n_involved,)``
        Facets touching at least one conflicting edge.
    boundary_edges : int
        Edges belonging to exactly one triangle. Normal for an open surface; a closed body with
        any is cracked.
    nonmanifold_edges : int
        Edges shared by three or more triangles. Not fatal for a gather, which never has to
        decide which side of a surface it is on, but usually a modelling mistake.
    merged_vertices : int
        Distinct vertex positions after coordinates were merged within tolerance. A count far
        below three times the facet count means the surface is well connected.
    """

    conflicting_edges: np.ndarray
    conflicting_facets: np.ndarray
    boundary_edges: int
    nonmanifold_edges: int
    merged_vertices: int

    @property
    def consistent(self) -> bool:
        """Whether every shared edge is traversed in opposite directions by its two triangles."""
        return len(self.conflicting_edges) == 0


def _merge_vertices(vertices: np.ndarray, tolerance: float | None) -> tuple[np.ndarray, int]:
    """Label coincident vertex positions with one shared index.

    An STL has no vertex table, so two triangles meeting along an edge repeat its endpoints as
    separate coordinate triples that are only *approximately* equal — they went through a
    file, and often through single precision on the way. Positions are therefore snapped to a
    grid before being matched. The default grid is relative to the model's own size rather than
    absolute, so the same tolerance is meaningful for a reactor in metres and a lamp in
    millimetres.
    """
    flat = vertices.reshape(-1, 3)
    if tolerance is None:
        extent = (
            float(np.max(flat, axis=0).max() - np.min(flat, axis=0).min()) if len(flat) else 0.0
        )
        tolerance = max(extent, 1.0) * 1e-9
    if tolerance <= 0.0:
        msg = f"tolerance must be positive; got {tolerance}"
        raise ValueError(msg)
    snapped = np.round(flat / tolerance).astype(np.int64)
    _, labels = np.unique(snapped, axis=0, return_inverse=True)
    labels = labels.reshape(-1)
    return labels.reshape(-1, 3), int(labels.max(initial=-1) + 1)


@dataclasses.dataclass(frozen=True)
class _EdgeUses:
    """Every undirected edge of a triangle set, and which triangles use it in which direction.

    Attributes
    ----------
    pairs : np.ndarray of int, shape ``(n_edges, 2)``
        Unique-vertex index pairs, lower index first.
    edge_of_use : np.ndarray of int, shape ``(3 * n_facets,)``
        For each directed edge of each triangle, in facet-major order, the undirected edge it is.
    direction : np.ndarray of int, shape ``(3 * n_facets,)``
        ``+1`` where that triangle traverses its edge from the lower index to the higher.
    uses : np.ndarray of int, shape ``(n_edges,)``
        How many triangles use each edge.
    degenerate : np.ndarray of bool, shape ``(n_edges,)``
        Edges whose two endpoints merged into one vertex, which have no direction to disagree
        about and bound nothing.
    merged_vertices : int
        Distinct vertex positions after merging.
    """

    pairs: np.ndarray
    edge_of_use: np.ndarray
    direction: np.ndarray
    uses: np.ndarray
    degenerate: np.ndarray
    merged_vertices: int

    @property
    def facet_of_use(self) -> np.ndarray:
        """The triangle each directed edge belongs to."""
        return np.repeat(np.arange(len(self.edge_of_use) // 3), 3)

    @property
    def boundary(self) -> np.ndarray:
        """Edges used by exactly one triangle -- a free edge, the rim of an open piece."""
        return (self.uses == 1) & ~self.degenerate


def _edge_uses(vertices: np.ndarray, tolerance: float | None) -> _EdgeUses:
    """Match the triangles' edges to one another through their merged vertices."""
    labels, merged = _merge_vertices(vertices, tolerance)
    # Every triangle contributes its three directed edges. An undirected edge is the sorted
    # pair; the sign says which way round that triangle traversed it. Two triangles sharing an
    # edge correctly traverse it in opposite directions, so their signs cancel.
    starts = labels
    ends = np.roll(labels, -1, axis=1)
    low = np.minimum(starts, ends).ravel()
    high = np.maximum(starts, ends).ravel()
    pairs, inverse, uses = np.unique(
        np.stack([low, high], axis=1).reshape(-1, 2),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return _EdgeUses(
        pairs=pairs,
        edge_of_use=inverse.reshape(-1),
        direction=np.where(starts.ravel() < ends.ravel(), 1, -1),
        uses=uses,
        degenerate=pairs[:, 0] == pairs[:, 1],
        merged_vertices=merged,
    )


def winding_report(vertices, *, tolerance: float | None = None) -> WindingReport:
    """Examine a triangle set's edge topology without raising.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        Triangle vertices in winding order.
    tolerance : float, optional
        Distance within which two vertex positions are the same point. Defaults to ``1e-9`` of
        the model's overall extent.

    Returns
    -------
    WindingReport
    """
    edges = _edge_uses(np.asarray(vertices, dtype=float), tolerance)
    net = np.bincount(edges.edge_of_use, weights=edges.direction, minlength=len(edges.pairs))
    conflicted = (edges.uses == 2) & (net != 0) & ~edges.degenerate
    involved = np.unique(edges.facet_of_use[conflicted[edges.edge_of_use]])

    return WindingReport(
        conflicting_edges=edges.pairs[conflicted],
        conflicting_facets=involved,
        boundary_edges=int(np.count_nonzero(edges.boundary)),
        nonmanifold_edges=int(np.count_nonzero((edges.uses > 2) & ~edges.degenerate)),
        merged_vertices=edges.merged_vertices,
    )


def open_facets(vertices, *, tolerance: float | None = None) -> np.ndarray:
    """Which facets belong to a connected piece of surface that has a free edge.

    A piece is the set of triangles reachable from one another across edges shared by exactly
    two triangles, and it is **open** if any edge of it is used by only one. A closed piece
    bounds a solid, so a sight line blocked by it crosses it twice -- in through one face and
    out through another -- whereas an open one is a sheet with the same medium on both sides,
    crossed once.

    ⚠️ Topology cannot see every sheet. A sheet whose rim is welded to another surface all the
    way round has no free edge: its rim edges are used three times, which neither joins it to
    the surface it is welded to nor marks it open, so it reads as closed. And a surface left open
    by accident -- a duct whose inlet and outlet caps were not exported -- reads as a sheet.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        Triangle vertices.
    tolerance : float, optional
        Distance within which two vertex positions are the same point. Defaults to ``1e-9`` of
        the model's overall extent.

    Returns
    -------
    np.ndarray of bool, shape ``(n_facets,)``
    """
    piece, open_piece = _surface_pieces(vertices, tolerance=tolerance)
    return open_piece[piece]


def _surface_pieces(vertices, *, tolerance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The connected pieces of a triangle set, and which of them are open.

    The pieces :func:`open_facets` reads: triangles reachable from one another across edges
    shared by exactly two triangles. Exposed for a caller that needs to treat each piece on its
    own -- a closed piece bounds a solid on one side, an open one is a sheet -- with the same
    caveats as :func:`open_facets` about what topology cannot see.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        Triangle vertices.
    tolerance : float, optional
        Distance within which two vertex positions are the same point. Defaults to ``1e-9`` of
        the model's overall extent.

    Returns
    -------
    piece : np.ndarray of int, shape ``(n_facets,)``
        Which piece each facet belongs to, numbered from zero.
    open_piece : np.ndarray of bool, shape ``(n_pieces,)``
        Whether each piece has a free edge.
    """
    vertices = np.asarray(vertices, dtype=float)
    n_facets = len(vertices)
    if n_facets == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    edges = _edge_uses(vertices, tolerance)
    piece = _pieces(edges)
    on_rim = np.zeros(n_facets, dtype=bool)
    on_rim[edges.facet_of_use[edges.boundary[edges.edge_of_use]]] = True
    open_piece = np.zeros(int(piece.max()) + 1, dtype=bool)
    open_piece[piece[on_rim]] = True
    return piece, open_piece


def _pieces(edges: _EdgeUses) -> np.ndarray:
    """Label each facet with its connected piece, as :func:`open_facets` defines one.

    A piece is the set of triangles reachable from one another across edges shared by exactly
    two triangles. Returns one label per facet, ``(n_facets,)``, numbered from zero.
    """
    n_facets = len(edges.edge_of_use) // 3
    facet = edges.facet_of_use
    shared = (edges.uses == 2)[edges.edge_of_use]
    # The two triangles on each two-use edge, found by sorting the uses of those edges by edge.
    order = np.argsort(edges.edge_of_use[shared], kind="stable")
    ends = facet[shared][order].reshape(-1, 2)
    links = coo_matrix((np.ones(len(ends)), (ends[:, 0], ends[:, 1])), shape=(n_facets, n_facets))
    _, piece = connected_components(links, directed=False)
    return piece


def _closed_pieces(edges: _EdgeUses, piece: np.ndarray) -> np.ndarray:
    """Which pieces are closed, consistently wound surfaces, one flag per piece.

    The test is that every edge's traversals **within the piece** cancel -- each edge is walked
    as often one way as the other by the piece's own triangles. That is exactly the condition
    under which the piece's summed signed solid angle is an integer everywhere off the surface
    and zero at infinity, so it is stricter than :func:`open_facets` on purpose: a sheet whose
    rim is welded to another surface has no free edge, and reads as closed there, but its rim
    edges are each walked once by its own triangles, so it fails this. A reversed triangle
    walks a shared edge the same way as its neighbour and fails it too.
    """
    n_pieces = int(piece.max(initial=-1)) + 1
    live = ~edges.degenerate[edges.edge_of_use]
    # One key per (piece, edge) combination, flattened to a single integer so a 1-D unique does
    # the grouping.
    key = piece[edges.facet_of_use[live]] * len(edges.pairs) + edges.edge_of_use[live]
    keys, inverse = np.unique(key, return_inverse=True)
    net = np.bincount(inverse.reshape(-1), weights=edges.direction[live], minlength=len(keys))
    closed = np.ones(n_pieces, dtype=bool)
    closed[keys[net != 0] // len(edges.pairs)] = False
    return closed


def check_winding(vertices, *, tolerance: float | None = None) -> WindingReport:
    """Raise unless every shared edge is traversed in opposite directions by its two triangles.

    Returns the report when the surface passes, so a caller that wants the boundary and
    non-manifold counts does not have to run the analysis twice.

    Raises
    ------
    ValueError
        If any shared edge is traversed the same way twice, which means one of its two
        triangles is wound backwards relative to the other.
    """
    report = winding_report(vertices, tolerance=tolerance)
    if not report.consistent:
        sample = report.conflicting_facets[:8].tolist()
        msg = (
            f"inconsistent triangle winding: {len(report.conflicting_edges)} shared edge(s) are "
            f"traversed in the same direction by both of their triangles, involving "
            f"{len(report.conflicting_facets)} facet(s) (first few: {sample}). Those facets' "
            "normals point the wrong way, and a gather discards whatever they would have "
            "emitted rather than reporting an error. Repair the winding in the surface file."
        )
        raise ValueError(msg)
    return report


def stored_normal_disagreement(vertices, stored_normal, *, cosine_tolerance: float = 1e-3):
    """Facets whose recorded normal disagrees with the one their winding implies.

    The recorded normal is advisory — the format lets a writer emit zeros, and many do, which
    is why this reports rather than raises and why an all-zero record is not a disagreement.
    A *non-zero* record pointing the other way is worth knowing about: it usually means the
    file was edited by something that moved vertices without updating the normals, and it is
    corroborating evidence when :func:`check_winding` has already objected.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
    stored_normal : array_like, shape ``(n_facets, 3)``
        Normals as read from the file, not normalized.
    cosine_tolerance : float, optional
        How far the two may point apart, as ``1 - cos(angle)``.

    Returns
    -------
    np.ndarray of int
        Indices of the disagreeing facets.
    """
    vertices = np.asarray(vertices, dtype=float)
    stored_normal = np.asarray(stored_normal, dtype=float)
    twice_vector_area = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    derived_length = np.linalg.norm(twice_vector_area, axis=1)
    stored_length = np.linalg.norm(stored_normal, axis=1)
    comparable = (derived_length > 0.0) & (stored_length > 0.0)
    cosine = np.zeros(len(vertices))
    cosine[comparable] = np.sum(
        twice_vector_area[comparable] * stored_normal[comparable], axis=1
    ) / (derived_length[comparable] * stored_length[comparable])
    return np.flatnonzero(comparable & (cosine < 1.0 - cosine_tolerance))


def check_profiles(surfaces) -> None:
    """Raise unless every facet's angular distribution suits the kind of source it is.

    The two kinds of source in a surface set are distinguished by area, and each admits only
    one family of distribution:

    * A **point source** is a zero-area facet carrying radiant power. It has no surface and so
      no normal, which means no direction-dependent distribution has anything to measure an
      angle against; it must be isotropic.
    * An **areal facet** carries an exitance over a surface that does have a normal, and an
      isotropic distribution on one is not a surface emitter — asked for a radiance it would
      divide by a vanishing cosine at grazing incidence.

    Both mismatches are silent if they are let through. A directional profile on a point source
    reads a zero normal as a right angle and returns zero intensity, so the source simply does
    not appear in the field. Checked here, at build time, where the profile is a concrete object
    and the message can name the facet.

    Raises
    ------
    ValueError
        If any facet carries a distribution that does not suit its kind.
    """
    from aquaflux.radiation.profiles import Isotropic

    # The kind of a source is read from its label, never from its area: the area is a quantity
    # a gradient may flow through, and the label is what decides a code path.
    is_point = surfaces.is_point_source
    index = np.asarray(surfaces.profile_index)
    for kind, profile in enumerate(surfaces.profiles):
        selected = index == kind
        isotropic = isinstance(profile, Isotropic)
        offenders = (
            np.flatnonzero(selected & ~is_point)
            if isotropic
            else np.flatnonzero(selected & is_point)
        )
        if len(offenders) == 0:
            continue
        if isotropic:
            msg = (
                f"{len(offenders)} surface facet(s) carry an Isotropic profile "
                f"(first few: {offenders[:8].tolist()}). Isotropic describes a point source; "
                "give an emitting surface Lambertian or CosinePower."
            )
        else:
            msg = (
                f"{len(offenders)} point source(s) carry a "
                f"{type(profile).__name__} profile (first few: {offenders[:8].tolist()}). A "
                "point source has no normal for a directional distribution to be measured "
                "against, and would silently contribute nothing; use Isotropic."
            )
        raise ValueError(msg)


def enclosure_winding(
    vertices,
    points,
    *,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
    tolerance: float | None = None,
) -> np.ndarray:
    """Winding number of a closed triangulated surface about each point.

    The signed solid angles of every facet, summed at a point and divided by ``4 pi``. For a
    closed, consistently wound surface that sum is ``±1`` at a point the surface encloses and
    ``0`` at one outside it, the overall sign fixed by whether the surface is wound outward or
    inward — so it is the **magnitude** that answers the question.

    Exact rather than asymptotic: on a unit box it reads ``1`` to within about ``1e-14`` a
    thousandth of a box-width from a wall, at every refinement, with no tolerance to tune.

    **On an open surface it reads somewhere in between, and that is the useful behaviour.** A
    bare disc measures ``±0.45`` just off its face. Open surfaces are legal here — a gather
    never has to decide which side of one it is on — so a test that answered a confident bit
    would be answering a question the geometry has not got. A value far from both ``0`` and
    ``±1`` means the surface is not closed, and that is worth reporting rather than rounding.

    **A closed piece is not summed at points outside its bounding box.** The surface is split
    into connected pieces, and a piece whose own triangles walk every edge as often one way as
    the other bounds a region: its winding number is an integer, constant off the surface and
    zero far away, so it is exactly ``0`` everywhere outside the box that holds it. Such a
    piece contributes that ``0`` there without being evaluated. Any other piece — open, wound
    inconsistently, or welded to another along its rim — is summed at every point, so the
    in-between readings above are unaffected.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        The surface's triangles.
    points : array_like, shape ``(n_points, 3)``
        Where to ask — cell centres, usually.
    pair_limit : int, optional
        Point-by-facet pairs per pass. The product is the whole memory cost, so it is cut into
        chunks of points; the answer is the same either way, to rounding.
    tolerance : float, optional
        Distance within which two vertex positions are the same point, when the surface is
        split into pieces. Defaults to ``1e-9`` of the model's overall extent, as in
        :func:`open_facets`. A piece whose seams close only to within it is skipped as though
        they closed exactly; what that leaves out is the solid angle of the gaps.

    Returns
    -------
    numpy.ndarray, shape ``(n_points,)``
    """
    vertices = np.asarray(vertices, dtype=float)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        msg = f"points must be (n_points, 3); got {tuple(points.shape)}"
        raise ValueError(msg)
    n_points, n_facets = points.shape[0], vertices.shape[0]
    if n_points == 0 or n_facets == 0:
        return np.zeros(n_points)

    edges = _edge_uses(vertices, tolerance)
    piece = _pieces(edges)
    closed = _closed_pieces(edges, piece)

    winding = np.zeros(n_points)
    everywhere = ~closed[piece]
    if np.any(everywhere):
        winding += _signed_total(points, vertices[everywhere], pair_limit)
    # Each closed piece's facets, grouped by one sort rather than a scan of every facet per piece.
    in_closed = np.flatnonzero(~everywhere)
    in_closed = in_closed[np.argsort(piece[in_closed], kind="stable")]
    groups = np.split(in_closed, np.flatnonzero(np.diff(piece[in_closed])) + 1)
    for facets in groups:
        if len(facets) == 0:
            continue
        corners = vertices[facets].reshape(-1, 3)
        # Inclusive at the box's faces: a point on one may lie on the surface itself.
        near = np.flatnonzero(
            np.all((points >= corners.min(axis=0)) & (points <= corners.max(axis=0)), axis=1)
        )
        if len(near):
            winding[near] += _signed_total(points[near], vertices[facets], pair_limit)
    return winding / (4.0 * np.pi)


def _signed_total(points: np.ndarray, vertices: np.ndarray, pair_limit: int) -> np.ndarray:
    """The signed solid angle of all of ``vertices`` summed at each point, in bounded passes.

    Each pass is one compiled call of :func:`_summed_signed_solid_angle`. A pass shorter than
    the full one — the last, or the only one — is padded to a power of two by repeating its last
    point, so pieces with different numbers of nearby points compile a few shapes rather than
    one each; the padding's answers are dropped.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    vertices : np.ndarray, shape ``(n_facets, 3, 3)``
    pair_limit : int

    Returns
    -------
    np.ndarray, shape ``(n_points,)``
    """
    n_points = len(points)
    chunk = receivers_per_pass(pair_limit, len(vertices))
    facets = jnp.asarray(vertices)
    totals = []
    for first in range(0, n_points, chunk):
        count = min(chunk, n_points - first)
        size = chunk if count == chunk else min(chunk, padded_length(count))
        index = first + np.minimum(np.arange(size), count - 1)
        totals.append(np.asarray(_summed_signed_solid_angle(points[index], facets))[:count])
    return np.concatenate(totals)


@jax.jit
def _summed_signed_solid_angle(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    """The signed solid angle of every facet summed at each point, as one compiled program.

    Compiled rather than run operation by operation because the kernel's intermediates are one
    entry per point-by-facet pair: eagerly, each is written out to memory in full and read back
    by the next operation, where compiled they fuse into one pass over the pairs.

    Parameters
    ----------
    points : jnp.ndarray, shape ``(n_points, 3)``
    vertices : jnp.ndarray, shape ``(n_facets, 3, 3)``

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
    """
    return jnp.sum(signed_solid_angle(points[:, None, :], vertices[None, ...]), axis=-1)


def check_points_outside(
    vertices,
    points,
    *,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
    tolerance: float | None = None,
) -> np.ndarray:
    """Refuse points the surface encloses — a cell centre embedded in the solid.

    A receiver inside the metal is a meshing error, not a dark corner: it is not shadowed by
    the geometry, it is *in* it, and the fluence rate computed there is a number with no
    physical referent. It reads as a plausible dim value, which is why this is a check rather
    than something a reader would notice in the output.

    The test is ``|winding| > 0.5`` — a threshold with nothing to tune behind it, since the
    quantity it cuts takes the values ``0`` and ``1`` and, on a closed surface, nothing in
    between.

    An **open** surface cannot answer the question, and this does not pretend otherwise. Open
    surfaces are legal here, so one is warned about rather than refused: a clean pass from a
    surface with no inside is a weaker statement than it reads as, and silence is how that goes
    unnoticed. Run :func:`check_winding` first — a surface whose facets disagree about which way
    is out has no consistent inside for this to find.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        The surface's triangles.
    points : array_like, shape ``(n_points, 3)``
        Where the field is wanted.
    pair_limit, tolerance : optional
        Passed through to :func:`enclosure_winding`.

    Returns
    -------
    numpy.ndarray, shape ``(n_points,)``
        The winding numbers, so a caller that wants to look rather than raise can.

    Raises
    ------
    ValueError
        If any point is enclosed by the surface.

    Warns
    -----
    UserWarning
        If nothing is enclosed but the surface does not appear to be closed, so the pass
        establishes less than it seems to.
    """
    winding = enclosure_winding(vertices, points, pair_limit=pair_limit, tolerance=tolerance)
    inside = np.flatnonzero(np.abs(winding) > 0.5)
    if len(inside):
        msg = (
            f"{len(inside)} of {len(winding)} point(s) lie inside the surface "
            f"(first few: {inside[:8].tolist()}, winding "
            f"{np.round(winding[inside[:8]], 3).tolist()}). A point inside the solid is "
            "embedded in it, not shadowed by it, and the fluence rate there means nothing. "
            "Move the points, or check that the surface is the one you meant."
        )
        raise ValueError(msg)
    worst = float(np.max(np.abs(winding), initial=0.0))
    if worst > _OPEN_SURFACE:
        # Warned, not raised. An open surface is legal here -- a gather never has to decide
        # which side of one it is on -- so refusing would reject a geometry the rest of the
        # package accepts. But a clean pass on a surface with no inside is a weaker statement
        # than it reads as, and silence is how that goes unnoticed.
        warnings.warn(
            f"no point is enclosed, but the largest winding number is {worst:.3f} rather than "
            "about 0, so this surface does not appear to be closed. That is legal, and it means "
            "this check found nothing because there was nothing to find, not because the points "
            "are known to be clear. Read enclosure_winding directly if you need the numbers.",
            UserWarning,
            stacklevel=2,
        )
    return winding
