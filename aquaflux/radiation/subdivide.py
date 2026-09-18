"""Refining emitting facets until they are small compared with their distance to a receiver.

The closed-form solid angle of a triangle is exact however large the triangle looks, so
refinement here is not about the geometry term. It is about everything the model assumes to be
*constant across a facet*: one emission value, one reflectance, one outgoing radiance. A facet
that fills a large part of a nearby receiver's sky is being treated as uniformly bright when
the physics varies across it, and no amount of exactness in the solid angle repairs that.

The criterion is the one lighting simulation has used for decades — refine until a source's
width is small relative to its distance to the nearest point being illuminated. Radiance
exposes it directly, subdividing until width over distance falls below 0.25, with 0.15 and
0.05 recommended when accuracy matters. The same ratio is used here, and the realized
distribution is reported so it is a measured property of a build rather than an assumption.

Refinement is four-way: each triangle splits at its edge midpoints into four similar
triangles, halving every edge. Similar children keep the shape quality of the parent, which
repeated bisection of one edge would not.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from aquaflux.radiation.surfaces import Surfaces

__all__ = ["Subdivision", "refine_for_receivers", "subdivide_to_width"]


@dataclasses.dataclass(frozen=True)
class Subdivision:
    """The refined triangles, and what the refinement achieved.

    Attributes
    ----------
    vertices : np.ndarray, shape ``(n_refined, 3, 3)``
        The refined triangles. Winding is inherited from the parent, so a consistently wound
        input stays consistently wound.
    origin : np.ndarray of int, shape ``(n_refined,)``
        Which original facet each refined triangle came from — the map that carries per-facet
        properties onto the children.
    level : np.ndarray of int, shape ``(n_refined,)``
        How many times each triangle's ancestry was split.
    realized_ratio : np.ndarray, shape ``(n_refined,)``
        Final width over distance-to-nearest-receiver, per refined triangle. The criterion is
        met where this is below the requested maximum; entries above it are facets that ran
        into ``max_levels`` first.
    """

    vertices: np.ndarray
    origin: np.ndarray
    level: np.ndarray
    realized_ratio: np.ndarray

    @property
    def n_facets(self) -> int:
        """Number of triangles after refinement."""
        return int(self.vertices.shape[0])

    @property
    def unmet(self) -> np.ndarray:
        """Indices of triangles that still exceed the requested ratio."""
        return np.flatnonzero(self.realized_ratio > self._requested)

    _requested: float = np.inf


def _width(vertices: np.ndarray) -> np.ndarray:
    """Longest edge of each triangle — the dimension the criterion compares against distance."""
    edges = np.roll(vertices, -1, axis=1) - vertices
    return np.max(np.linalg.norm(edges, axis=2), axis=1)


def _distance_to_nearest_receiver(vertices: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Conservative distance from each triangle to the nearest receiver.

    The minimum over the triangle's three vertices and its centroid, rather than the centroid
    alone. A receiver sitting just off one corner of a large facet is the case that most needs
    refining and the one a centroid distance most overstates, so the cheap four-point minimum
    is used instead: it never reports a larger distance than the centroid would, so it never
    refines less.
    """
    probes = np.concatenate([vertices.reshape(-1, 3), vertices.mean(axis=1)], axis=0)
    distance, _ = tree.query(probes, k=1)
    per_vertex = distance[: len(vertices) * 3].reshape(-1, 3)
    per_centroid = distance[len(vertices) * 3 :]
    return np.minimum(per_vertex.min(axis=1), per_centroid)


def _split_four(vertices: np.ndarray) -> np.ndarray:
    """Split each triangle at its edge midpoints into four similar triangles."""
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
    children = np.stack(
        [
            np.stack([a, ab, ca], axis=1),
            np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1),
            # The middle child's winding must match its siblings', or refining a surface would
            # quietly reverse a quarter of it -- which the winding check would then report as a
            # defect of the input file.
            np.stack([ab, bc, ca], axis=1),
        ],
        axis=1,
    )
    return children.reshape(-1, 3, 3)


def subdivide_to_width(
    vertices,
    receivers,
    *,
    max_ratio: float = 0.25,
    max_levels: int = 6,
) -> Subdivision:
    """Split triangles until each is small compared with its distance to the nearest receiver.

    Parameters
    ----------
    vertices : array_like, shape ``(n_facets, 3, 3)``
        Triangles to refine, in winding order.
    receivers : array_like, shape ``(n_receivers, 3)``
        The points that will be illuminated. Only their positions matter, so this takes a bare
        array rather than a mesh.
    max_ratio : float, optional
        Target for longest edge over distance to the nearest receiver. The default 0.25 is
        Radiance's; 0.15 and 0.05 are its recommendations for accurate work.
    max_levels : int, optional
        Cap on the number of splits, since the criterion cannot be met at all for a receiver
        lying *on* a facet -- the distance goes to zero faster than the width does. Each level
        multiplies the facet count by four, so the cap is also the memory bound: six levels is
        4096 children per original facet.

    Returns
    -------
    Subdivision

    Raises
    ------
    ValueError
        If ``max_ratio`` is not positive, or ``receivers`` is empty.
    """
    if max_ratio <= 0.0:
        msg = f"max_ratio must be positive; got {max_ratio}"
        raise ValueError(msg)
    receivers = np.asarray(receivers, dtype=float)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        msg = f"receivers must have shape (n_receivers, 3); got {receivers.shape}"
        raise ValueError(msg)
    if len(receivers) == 0:
        msg = "receivers is empty; there is nothing for the criterion to measure against"
        raise ValueError(msg)

    current = np.asarray(vertices, dtype=float)
    origin = np.arange(len(current))
    level = np.zeros(len(current), dtype=np.int32)
    tree = cKDTree(receivers)

    for _ in range(max_levels):
        ratio = _width(current) / np.maximum(
            _distance_to_nearest_receiver(current, tree), np.finfo(float).tiny
        )
        too_wide = ratio > max_ratio
        if not np.any(too_wide):
            break
        keep = ~too_wide
        split = _split_four(current[too_wide])
        current = np.concatenate([current[keep], split], axis=0)
        origin = np.concatenate([origin[keep], np.repeat(origin[too_wide], 4)])
        level = np.concatenate([level[keep], np.repeat(level[too_wide], 4) + 1])

    final_ratio = _width(current) / np.maximum(
        _distance_to_nearest_receiver(current, tree), np.finfo(float).tiny
    )
    return Subdivision(
        vertices=current,
        origin=origin,
        level=level,
        realized_ratio=final_ratio,
        _requested=max_ratio,
    )


def refine_for_receivers(
    surfaces: Surfaces,
    receivers,
    *,
    max_ratio: float = 0.25,
    max_levels: int = 6,
) -> tuple[Surfaces, Subdivision]:
    """Refine a surface set for a set of receiver positions, carrying its properties along.

    Every per-facet property is inherited unchanged by a facet's children: emission and
    reflectance are intensive, so splitting a facet does not divide them. Radiant power is the
    exception and is **not** carried, because it is extensive -- a point source has no area to
    split and never meets the criterion anyway, so a surface set carrying point sources should
    be refined before they are added rather than after.

    Returns
    -------
    tuple of (Surfaces, Subdivision)
        The refined set, and the record of what refinement did -- which is what a build should
        report, since the realized ratio is the accuracy actually obtained rather than the one
        asked for.

    Raises
    ------
    ValueError
        If any facet carries non-zero radiant power.
    """
    if bool(np.any(np.asarray(surfaces.power) != 0.0)):
        msg = (
            "refine_for_receivers cannot carry radiant power: power is extensive, so it would "
            "have to be divided among a facet's children, and a zero-area point source never "
            "meets the width criterion in any case. Refine the areal facets first, then add "
            "the point sources."
        )
        raise ValueError(msg)

    division = subdivide_to_width(
        np.asarray(surfaces.vertices),
        receivers,
        max_ratio=max_ratio,
        max_levels=max_levels,
    )
    inherited = division.origin
    refined = Surfaces.from_triangles(
        division.vertices,
        solid_id=np.asarray(surfaces.solid_id)[inherited],
        solid_names=surfaces.solid_names,
        emission=np.asarray(surfaces.emission)[inherited],
        reflectance=np.asarray(surfaces.reflectance)[inherited],
        profiles=surfaces.profiles,
        profile_index=np.asarray(surfaces.profile_index)[inherited],
    )
    return refined, division
