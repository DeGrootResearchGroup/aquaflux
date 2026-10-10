"""The direct gather through transparent solids: each source seen along its refracted path.

Where a source and a receiver are in different media -- a lamp's arc inside its sleeve's air gap and
a point in the water outside it -- the light between them crosses surfaces that bend it, and a
straight line from the receiver to the source is the wrong path. This gathers those pairs along the
true ones (:func:`~aquaflux.radiation.refraction.solve_paths`).

**The solid angle a triangle fills, seen through the surfaces.** The path to each of a source
triangle's three vertices arrives at the receiver from some direction; those three directions span
a triangle on the receiver's unit sphere, and its area is the closed-form solid angle the direct
gather already evaluates -- given the arrival directions as the vertices. With nothing in the way
the arrival directions are the straight ones and this *is* the direct gather's kernel; through a
curved surface it is the image of the triangle, with its edges taken as great circles. The image's
edges are curved, so this is exact only in the limit of small triangles, which is the limit the
refinement criterion already drives an emitter towards. The plain and the cosine-weighted kernels
both take the arrival directions, so the fluence rate and the irradiance come the same way.

**What scales it.** The radiance leaving the source in each path's departure direction (its angular
distribution, asked about that direction), the share of light the path gets across
(:class:`~aquaflux.radiation.refraction.Paths`), each averaged over the three vertices; and the
square of the ratio of the receiver's index to the source's, because radiance divided by the
square of the index is what a lossless crossing conserves. A triangle any of whose vertices has no
transmitted path -- past the critical angle, or off the edge of a face -- carries nothing.

**What stands in the way** of a path is a frozen mask (:func:`build_refracted_visibility`): every
leg of the path to each facet's centroid tested against the bodies and the surface's own triangles,
as the direct gather's mask tests the straight segment.

Pairs whose source and receiver are in the same medium have no crossing, and are the direct
gather's; this sums only the others.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.culling import _body_blocks
from aquaflux.radiation.gather import _groups
from aquaflux.radiation.refraction import Chain, Media, _pairs_meet, solve_paths
from aquaflux.radiation.self_occlusion import RayCastOcclusion, SelfOcclusion
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import (
    refuse_points_inside,
    same_receivers,
    surviving_fraction,
)
from aquaflux.radiation.work import (
    DEFAULT_PAIR_LIMIT,
    in_passes,
    receivers_per_pass,
    receivers_per_step,
)

__all__ = [
    "RefractedVisibility",
    "build_refracted_visibility",
    "refracted_fluence_rate",
    "refracted_irradiance",
]


class RefractedVisibility(eqx.Module):
    """A frozen record of what stands across each refracted path.

    Its own type rather than a bare :class:`~aquaflux.radiation.visibility.Visibility`, because a
    mask of straight segments and a mask of refracted paths mean different things: one passed for
    the other would put every shadow where the light does not go.

    Attributes
    ----------
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        The receivers it was built for.
    blocked : tuple of jnp.ndarray of bool
        Per route (in the order the gather works them out), ``(n_starts, n_occluders, n_rows,
        n_facets)``: true where the body lies across any leg of the path from the facet's centroid to
        the receiver, for each starting point of the route.
    hidden : tuple of (jnp.ndarray of bool or None)
        Per route, ``(n_starts, n_rows, n_facets)``: where the surface's own triangles do; ``None``
        where they shadow nothing.
    """

    receivers: jnp.ndarray
    blocked: tuple
    hidden: tuple

    @property
    def n_occluders(self) -> int:
        """How many bodies the mask covers."""
        return int(self.blocked[0].shape[1]) if self.blocked else 0


@dataclasses.dataclass(frozen=True)
class _Group:
    """Facets in one medium and the receivers one route reaches from them."""

    chain: Chain
    facets: np.ndarray
    rows: np.ndarray
    profile: object


def _plan(surfaces: Surfaces, media: Media, points: np.ndarray, pair_limit: int) -> tuple:
    """Every group of pairs with a crossing, worked out on the host from concrete positions.

    Every route is taken for every receiver: a route through a region carries light to receivers
    whose straight segments miss the region altogether, because an air gap near the critical angle
    turns light by tens of degrees. No bound on that turn is used to leave any out.
    """
    facet_region = media.region_of_facets(surfaces)
    receiver_region = media.region_of(points, "receiver")
    parents = media.parents
    centroid = np.asarray(surfaces.centroid, dtype=float)
    groups = []
    for profile, areal, point in _groups(surfaces):
        for source in np.unique(facet_region[point]):
            facets = point[facet_region[point] == source]
            for receiver in np.unique(receiver_region):
                rows = np.flatnonzero(receiver_region == receiver)
                if receiver != source or _reaches_through(
                    media, parents, int(source), centroid[facets], points[rows], pair_limit
                ):
                    msg = (
                        "a point source's light would cross the surface of a transparent region; "
                        "a point source is seen along one path with no area to spread over, and "
                        "is not gathered through a transparent surface. Describe the source as an "
                        "areal one."
                    )
                    raise ValueError(msg)
        for source in np.unique(facet_region[areal]):
            facets = areal[facet_region[areal] == source]
            for receiver in np.unique(receiver_region):
                rows = np.flatnonzero(receiver_region == receiver)
                for chain in Chain.routes(parents, int(source), int(receiver)):
                    groups.append(_Group(chain, facets, rows, profile))
    return tuple(groups)


def _reaches_through(media, parents, medium, origins, targets, pair_limit) -> bool:
    """Whether any segment from ``origins`` to ``targets``, in one medium, meets a region."""
    beside = Chain.between(parents, medium, medium).beside[0]
    return bool(np.any(_pairs_meet(media, beside, origins, targets, pair_limit=pair_limit)))


def build_refracted_visibility(
    occluders,
    surfaces: Surfaces,
    media: Media,
    points,
    *,
    receiver_facet=None,
    self_occlusion: SelfOcclusion | None = None,
    offset_scale: float = 1e-6,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> RefractedVisibility:
    """Work out, once, what stands across the refracted path from each facet to each receiver.

    The path is the one to the facet's centroid, solved here at the geometry and indices given and
    then frozen, as every shadow is: a derivative with respect to an index moves the paths the
    gather solves, and not this mask.

    Parameters
    ----------
    occluders : sequence of aquaflux.solids.Body
        The opaque or partly transmitting bodies. The transparent regions are not among them: a
        region's surface is crossed, not tested.
    surfaces : Surfaces
        The emitting set; paths start at facet centroids.
    media : Media
    points : array_like, shape ``(n_receivers, 3)``
        Receiver positions.
    receiver_facet : array_like of int, shape ``(n_receivers,)`` or ``(n_receivers, k)``, optional
        The facets of ``surfaces`` each receiver lies on, ``-1`` for none, left out of the test of
        the last leg, which ends in them. As for
        :func:`~aquaflux.radiation.visibility.build_visibility`: omitted where it applies, a
        receiver on a facet reads that facet as blocking every path to it.
    self_occlusion : SelfOcclusion, optional
        How the surface's own triangles are tested; each leg is one ray whatever the strategy, as
        for a mirror's two legs. Defaults to
        :class:`~aquaflux.radiation.self_occlusion.RayCastOcclusion`.
    offset_scale : float, optional
        How far from each leg's start a hit must be before it counts, as a fraction of the source
        facet's square-root area, as in :func:`~aquaflux.radiation.visibility.build_visibility`.
    pair_limit : int, optional
        Receiver-by-facet pairs whose paths are solved at once.

    Returns
    -------
    RefractedVisibility

    Raises
    ------
    ValueError
        If a facet centroid or receiver lies inside a body, or a point's medium is not defined.
    """
    occluders = tuple(occluders)
    host_points = np.asarray(points, dtype=float)
    refuse_points_inside(occluders, surfaces, host_points)
    strategy = RayCastOcclusion() if self_occlusion is None else self_occlusion
    n_points = host_points.shape[0]
    centroid = np.asarray(surfaces.centroid, dtype=float)
    near = offset_scale * np.sqrt(np.asarray(surfaces.area, dtype=float))
    own = (
        np.full((n_points, 1), -1, dtype=int)
        if receiver_facet is None
        else np.asarray(receiver_facet, dtype=int).reshape(n_points, -1)
    )
    blocked_by_route, hidden_by_route = [], []
    for group in _plan(surfaces, media, host_points, pair_limit):
        starts = group.chain.n_starts
        blocked = np.zeros((starts, len(occluders), len(group.rows), len(group.facets)), bool)
        hidden = None
        per_pass = receivers_per_pass(pair_limit, starts * len(group.facets))
        for first in range(0, len(group.rows), per_pass):
            part = slice(first, first + per_pass)
            rows = group.rows[part]
            # (starts, rows, facets, crossings, 3)
            paths = _centroid_paths(
                media, group.chain, centroid[group.facets], host_points[rows], pair_limit
            )
            shape = paths.shape[:3]
            corners = np.concatenate(
                [
                    np.broadcast_to(centroid[group.facets][None, None, :, None], (*shape, 1, 3)),
                    paths,
                    np.broadcast_to(host_points[rows][None, :, None, None], (*shape, 1, 3)),
                ],
                axis=3,
            )
            origin = corners[:, :, :, :-1].reshape(-1, 3)
            target = corners[:, :, :, 1:].reshape(-1, 3)
            n_legs = corners.shape[3] - 1
            leg_near = np.broadcast_to(
                near[group.facets][None, None, :, None], (*shape, n_legs)
            ).reshape(-1)
            for index, body in enumerate(occluders):
                crossed = np.asarray(
                    _body_blocks(
                        body, jnp.asarray(origin), jnp.asarray(target), jnp.asarray(leg_near)
                    )
                ).reshape(*shape, n_legs)
                blocked[:, index, part] = np.any(crossed, axis=3)
            # The first leg leaves the source facet, and the last ends in the receiver's own.
            exclude = np.full((*shape, n_legs, 1 + own.shape[1]), -1, dtype=int)
            exclude[..., 0, 0] = group.facets[None, None, :]
            exclude[..., -1, 1:] = own[rows][None, :, None, :]
            exclude = exclude.reshape(-1, exclude.shape[-1])
            cut = strategy.segments_hidden(surfaces, origin, target, leg_near, exclude)
            if cut is not None:
                if hidden is None:
                    hidden = np.zeros((starts, len(group.rows), len(group.facets)), dtype=bool)
                hidden[:, part] = np.any(np.asarray(cut).reshape(*shape, n_legs), axis=3)
        blocked_by_route.append(jnp.asarray(blocked))
        hidden_by_route.append(None if hidden is None else jnp.asarray(hidden))
    return RefractedVisibility(
        receivers=jnp.asarray(host_points),
        blocked=tuple(blocked_by_route),
        hidden=tuple(hidden_by_route),
    )


def _centroid_paths(media: Media, chain: Chain, sources, receivers, pair_limit) -> np.ndarray:
    """``(n_starts, n_receivers, n_sources, n_crossings, 3)`` crossing points, on the host.

    Solved a step of receivers at a time: a path's solve holds a few kilobytes of intermediates,
    more the more surfaces it crosses, so a whole pass of paths at once would hold gigabytes. Every
    step has one shape -- the last is filled out by repeating its last receiver, whose answers are
    dropped -- so the solve compiles once.
    """
    sources, receivers = np.asarray(sources), np.asarray(receivers)
    step = receivers_per_step(pair_limit, chain.n_starts * len(sources))
    step = min(step, len(receivers))
    found = []
    for first in range(0, len(receivers), step):
        rows = np.minimum(np.arange(first, first + step), len(receivers) - 1)
        paths = solve_paths(
            media, chain, jnp.asarray(sources)[None], jnp.asarray(receivers[rows])[:, None]
        )
        points = np.asarray(paths.points)
        points = points if chain.passing else points[None]
        found.append(points[:, : len(receivers) - first])
    return np.concatenate(found, axis=1)


def refracted_fluence_rate(
    surfaces: Surfaces,
    media: Media,
    points,
    *,
    visibility: RefractedVisibility | None = None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> jnp.ndarray:
    """Fluence rate at each point from every source in another medium, along refracted paths.

    The counterpart of :func:`~aquaflux.radiation.gather.direct_fluence_rate` for the pairs that
    cross a transparent surface; the pairs in one medium are that function's and contribute
    nothing here.

    Parameters
    ----------
    surfaces : Surfaces
        The sources. Its emission, profile parameters and vertices are live; which facets are in
        which medium is read from it on the host.
    media : Media
        The indices and absorptions are live.
    points : array_like, shape ``(n_points, 3)``
        Receiver positions, concrete: which medium each is in decides the program.
    visibility : RefractedVisibility, optional
        What stands across each path, built for these points. Without one nothing does.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What each body in the mask lets through. Defaults to opaque; refused without a mask.
    pair_limit : int, optional
        Receiver-by-vertex pairs one pass solves paths for.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If a point source is in another medium than a receiver, a facet straddles a region's
        surface, or a point's medium is not defined.
    """
    return _refracted(surfaces, media, points, None, visibility, transmittance, pair_limit)


def refracted_irradiance(
    surfaces: Surfaces,
    media: Media,
    points,
    normals,
    *,
    visibility: RefractedVisibility | None = None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> jnp.ndarray:
    """Irradiance on surfaces at each point from every source in another medium.

    As :func:`refracted_fluence_rate`, each direction weighted by its cosine to ``normals``.

    Parameters
    ----------
    surfaces, media, points
        As for :func:`refracted_fluence_rate`.
    normals : array_like, shape ``(n_points, 3)``
        Unit normals of the receiving surfaces, pointing the way the light arrives from.
    visibility, transmittance, pair_limit
        As for :func:`refracted_fluence_rate`.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Irradiance in W/m².
    """
    normals = jnp.asarray(normals, dtype=float)
    return _refracted(surfaces, media, points, normals, visibility, transmittance, pair_limit)


def _refracted(surfaces, media, points, normals, visibility, transmittance, pair_limit):
    """The sum over every route of every group of pairs; the kernel by whether there are normals."""
    host_points = np.asarray(points, dtype=float)
    points = jnp.asarray(points, dtype=float)
    plan = _plan(surfaces, media, host_points, pair_limit)
    masks = _route_masks(visibility, transmittance, host_points, plan)
    total = jnp.zeros(points.shape[0])
    for group, mask in zip(plan, masks, strict=True):
        total = total.at[group.rows].add(
            _group_field(surfaces, media, group, points, normals, mask, pair_limit)
        )
    return total


def _route_masks(visibility, transmittance, points, plan):
    """Per route, its mask's arrays and the bodies' transmittance, or ``None`` with no mask."""
    if visibility is None:
        if transmittance is not None:
            msg = "transmittance was given without a visibility mask to apply it to"
            raise ValueError(msg)
        return [None] * len(plan)
    if not isinstance(visibility, RefractedVisibility):
        msg = (
            "visibility must be a RefractedVisibility, from build_refracted_visibility; got "
            f"{type(visibility).__name__}. A mask of straight segments marks where light does "
            "not go along a refracted path."
        )
        raise TypeError(msg)
    same_receivers(visibility.receivers, points, "refracted visibility mask")
    shapes = [(g.chain.n_starts, len(g.rows), len(g.facets)) for g in plan]
    built = [(b.shape[0], b.shape[2], b.shape[3]) for b in visibility.blocked]
    if shapes != built:
        msg = "this refracted visibility mask was built for other sources or other media"
        raise ValueError(msg)
    if transmittance is None:
        transmittance = jnp.zeros(visibility.n_occluders)
    return [
        (blocked, hidden, transmittance)
        for blocked, hidden in zip(visibility.blocked, visibility.hidden, strict=True)
    ]


def _group_field(surfaces, media, group: _Group, points, normals, mask, pair_limit):
    """What one group's receivers get from its facets along one route, ``(n_rows,)``.

    On a route through a region every starting point's path is a path of its own -- they are
    distinct where they are valid -- and each source triangle is seen once along each.
    """
    facets = group.facets
    chain = group.chain
    corners, shared = _shared_corners(surfaces, facets)
    normal = jnp.take(surfaces.normal, facets, axis=0)
    emission = jnp.take(surfaces.emission, facets)
    ratio = (media.index_of(chain.legs[-1]) / media.index_of(chain.legs[0])) ** 2
    receivers = jnp.take(points, group.rows, axis=0)
    arrays = [(receivers, 0)]
    if normals is not None:
        arrays.append((jnp.take(normals, group.rows, axis=0), 0))
    if mask is not None:
        blocked, hidden, _ = mask
        arrays.append((blocked, 2))
        if hidden is not None:
            arrays.append((hidden, 1))

    def at(chunk, *rest):
        rest = list(rest)
        chunk_normals = rest.pop(0) if normals is not None else None
        paths = solve_paths(media, chain, corners[None], chunk[:, None])
        if not chain.passing:
            paths = jax.tree.map(lambda leaf: leaf[None], paths)
        arrival = paths.arrival[:, :, shared]
        apparent = chunk[None, :, None, None, :] + arrival
        if chunk_normals is None:
            omega = solid_angle(chunk[None, :, None, :], apparent)
        else:
            omega = projected_solid_angle(
                chunk[None, :, None, :], chunk_normals[None, :, None, :], apparent
            )
        radiance = group.profile.radiance_per_exitance(
            paths.departure[:, :, shared], normal[None, None, :, None, :]
        )
        carried = jnp.mean(radiance * paths.transmittance[:, :, shared], axis=-1)
        seen = jnp.all(paths.valid[:, :, shared], axis=-1)
        weight = jnp.where(seen, emission * ratio * carried * omega, 0.0)
        if mask is not None:
            chunk_blocked = rest.pop(0)
            chunk_hidden = rest.pop(0) if mask[1] is not None else None
            # The bodies' axis first, as the fraction takes it; the starts ride with the pairs.
            weight = weight * surviving_fraction(
                jnp.moveaxis(chunk_blocked, 1, 0), chunk_hidden, mask[2]
            )
        return jnp.sum(weight, axis=(0, 2))

    return in_passes(arrays, pair_limit, chain.n_starts * corners.shape[0], at)


def _shared_corners(surfaces: Surfaces, facets: np.ndarray):
    """The distinct corners of ``facets``, and each facet's three as indices into them.

    Neighbouring triangles share corners, so solving a path per distinct corner rather than per
    corner of each triangle costs about a sixth as much on a closed triangulation. Which corners
    coincide is read from concrete vertices; traced ones are taken corner by corner.
    """
    vertices = jnp.take(surfaces.vertices, facets, axis=0).reshape(-1, 3)
    if isinstance(surfaces.vertices, jax.core.Tracer):
        return vertices, np.arange(3 * len(facets)).reshape(-1, 3)
    _, first, inverse = np.unique(
        np.asarray(vertices), axis=0, return_index=True, return_inverse=True
    )
    return jnp.take(vertices, first, axis=0), np.asarray(inverse).reshape(-1, 3)
