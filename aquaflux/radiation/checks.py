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

import numpy as np

__all__ = ["WindingReport", "check_winding", "stored_normal_disagreement", "winding_report"]


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
