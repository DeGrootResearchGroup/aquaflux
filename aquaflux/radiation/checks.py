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

import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.solid_angle import signed_solid_angle

__all__ = [
    "WindingReport",
    "check_points_outside",
    "check_profiles",
    "check_winding",
    "enclosure_winding",
    "stored_normal_disagreement",
    "winding_report",
]

#: Point-by-facet entries one pass may form, matching the intersection test's own budget in
#: :mod:`aquaflux.radiation.triangles` — the same product, formed for the same reason, so the
#: same bound applies.
_WORK_LIMIT = 4_000_000

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
    vertices = np.asarray(vertices, dtype=float)
    labels, merged = _merge_vertices(vertices, tolerance)
    n_facets = len(labels)

    # Every triangle contributes its three directed edges. An undirected edge is the sorted
    # pair; the sign says which way round that triangle traversed it. Two triangles sharing an
    # edge correctly traverse it in opposite directions, so their signs cancel.
    starts = labels
    ends = np.roll(labels, -1, axis=1)
    low = np.minimum(starts, ends).ravel()
    high = np.maximum(starts, ends).ravel()
    sign = np.where(starts.ravel() < ends.ravel(), 1, -1)
    facet_of_edge = np.repeat(np.arange(n_facets), 3)

    pairs = np.stack([low, high], axis=1)
    unique_pairs, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    net = np.bincount(inverse, weights=sign, minlength=len(unique_pairs))

    # A degenerate edge (both endpoints merged to one vertex) has no direction to disagree
    # about, so it is excluded rather than counted as a conflict.
    degenerate = unique_pairs[:, 0] == unique_pairs[:, 1]
    conflicted = (counts == 2) & (net != 0) & ~degenerate
    involved = np.unique(facet_of_edge[conflicted[inverse]])

    return WindingReport(
        conflicting_edges=unique_pairs[conflicted],
        conflicting_facets=involved,
        boundary_edges=int(np.count_nonzero((counts == 1) & ~degenerate)),
        nonmanifold_edges=int(np.count_nonzero((counts > 2) & ~degenerate)),
        merged_vertices=merged,
    )


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


def enclosure_winding(vertices, points, *, work_limit: int = _WORK_LIMIT) -> np.ndarray:
    """Winding number of a closed triangulated surface about each point.

    The signed solid angles of every facet, summed at a point and divided by ``4 pi``. For a
    closed, consistently wound surface that sum is ``±1`` at a point the surface encloses and
    ``0`` at one outside it, the overall sign fixed by whether the surface is wound outward or
    inward — so it is the **magnitude** that answers the question.

    Exact rather than asymptotic: on a unit box it reads ``1.0`` to the last bit a thousandth of
    a box-width from a wall, at every refinement, with no tolerance to tune.

    **On an open surface it reads somewhere in between, and that is the useful behaviour.** A
    bare disc measures ``±0.45`` just off its face. Open surfaces are legal here — a gather
    never has to decide which side of one it is on — so a test that answered a confident bit
    would be answering a question the geometry has not got. A value far from both ``0`` and
    ``±1`` means the surface is not closed, and that is worth reporting rather than rounding.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        The surface's triangles.
    points : array_like, shape ``(n_points, 3)``
        Where to ask — cell centres, usually.
    work_limit : int, optional
        Point-by-facet entries per pass. The product is the whole memory cost, so it is cut
        into chunks of points; the arithmetic is identical either way.

    Returns
    -------
    numpy.ndarray, shape ``(n_points,)``

    Notes
    -----
    The chunking here is a plain host-side loop, deliberately **not** the padded
    :func:`~aquaflux.radiation.gather` scan: that one exists so a traced body compiles once for
    a gather that runs on every solve, while this runs once, outside any trace, where the
    padding would be cost with nothing to buy.
    """
    vertices = jnp.asarray(vertices, dtype=float)
    points = jnp.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        msg = f"points must be (n_points, 3); got {tuple(points.shape)}"
        raise ValueError(msg)
    n_points, n_facets = points.shape[0], vertices.shape[0]
    if n_points == 0 or n_facets == 0:
        return np.zeros(n_points)

    chunk = max(1, work_limit // max(1, n_facets))
    totals = [
        np.asarray(
            jnp.sum(
                signed_solid_angle(points[first : first + chunk, None, :], vertices[None, ...]),
                axis=-1,
            )
        )
        for first in range(0, n_points, chunk)
    ]
    return np.concatenate(totals) / (4.0 * np.pi)


def check_points_outside(vertices, points, *, work_limit: int = _WORK_LIMIT) -> np.ndarray:
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
    work_limit : int, optional
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
    winding = enclosure_winding(vertices, points, work_limit=work_limit)
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
