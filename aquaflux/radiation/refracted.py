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
transmitted path -- past the critical angle, or off the edge of a face -- carries nothing, with one
exception.

**Round a sleeve's rim, the part of a triangle that has a path is counted.** The light a route
through a sleeve's wall alone carries comes from a small patch of the source (the band of paths that
skim the rim), so many of the patch's triangles straddle its edge, and dropping them loses a large
share of the route's light at any practical facet size. On such a route (shot, see
:func:`~aquaflux.radiation.refraction._shot`) each edge from a corner with a path to one without is
bisected for where the path stops, and the triangle is cut there into the one or two triangles that
have a path at every corner; those are counted as a whole triangle would be. Where the patch's edge
crosses an edge once, as it does for a patch larger than a facet, this is exact in the same limit
of small triangles; a patch narrower than a facet that misses all three corners is still missed.
The cut is found on the host, from the geometry and indices as given, and then frozen -- as every
shadow is -- so a derivative with respect to an index moves the paths to the cut triangles' corners
but not where the cut lies. Under a trace of the receivers, the source geometry or the media, the
cut cannot be found and the triangle is dropped as on any other route.

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
from aquaflux.radiation.refraction import (
    Chain,
    Media,
    _only_saddles,
    _pairs_meet,
    _path_count,
    _shot,
    _shot_paths,
    _traced,
    solve_paths,
)
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
        Per route (in the order the gather works them out), ``(n_paths, n_occluders, n_rows,
        n_facets)``: true where the body lies across any leg of the path from the facet's centroid to
        the receiver, for each path of the route per pair (``refraction._path_count``).
    hidden : tuple of (jnp.ndarray of bool or None)
        Per route, ``(n_paths, n_rows, n_facets)``: where the surface's own triangles do; ``None``
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
    turns light by tens of degrees. No bound on that turn is used to leave any out. The one route left
    out is one whose paths no solve here can find -- through a sleeve's wall alone where they cannot be
    shot (:func:`~aquaflux.radiation.refraction._only_saddles`): solving it by descent would cost as
    much as any other route and carry nothing.
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
                    if _shot(media, chain) or not _only_saddles(media, chain):
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
        starts = _path_count(media, group.chain)
        blocked = np.zeros((starts, len(occluders), len(group.rows), len(group.facets)), bool)
        hidden = None
        per_pass = receivers_per_pass(pair_limit, starts * len(group.facets))
        for first in range(0, len(group.rows), per_pass):
            part = slice(first, first + per_pass)
            rows = group.rows[part]
            # (starts, rows, facets, crossings, 3)
            paths, found = _host_paths(
                media, group.chain, centroid[group.facets], host_points[rows], pair_limit
            )
            shape = paths.shape[:3]
            sources = np.broadcast_to(centroid[group.facets][None, None], (*shape, 3))
            if _shot(media, group.chain):
                sources, paths = _from_a_reached_point(
                    media,
                    group.chain,
                    surfaces,
                    group.facets,
                    host_points[rows],
                    sources,
                    paths,
                    found,
                    pair_limit,
                )
            corners = np.concatenate(
                [
                    sources[:, :, :, None],
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


def _host_paths(media: Media, chain: Chain, sources, receivers, pair_limit):
    """Paths between every source and receiver, on the host: crossing points and whether each is one.

    Returns ``(n_paths, n_receivers, n_sources, n_crossings, 3)`` points and an
    ``(n_paths, n_receivers, n_sources)`` flag. Solved a step of receivers at a time: a path's solve
    holds a few kilobytes of intermediates, more the more surfaces it crosses, so a whole pass of
    paths at once would hold gigabytes. Every step has one shape -- the last is filled out by
    repeating its last receiver, whose answers are dropped -- so the solve compiles once.
    """
    sources, receivers = np.asarray(sources), np.asarray(receivers)
    step = receivers_per_step(pair_limit, _path_count(media, chain) * len(sources))
    step = min(step, len(receivers))
    points, valid = [], []
    for first in range(0, len(receivers), step):
        rows = np.minimum(np.arange(first, first + step), len(receivers) - 1)
        paths = solve_paths(
            media, chain, jnp.asarray(sources)[None], jnp.asarray(receivers[rows])[:, None]
        )
        if not chain.passing:
            paths = jax.tree.map(lambda leaf: leaf[None], paths)
        points.append(np.asarray(paths.points)[:, : len(receivers) - first])
        valid.append(np.asarray(paths.valid)[:, : len(receivers) - first])
    return np.concatenate(points, axis=1), np.concatenate(valid, axis=1)


def _from_a_reached_point(
    media, chain, surfaces, facets, receivers, sources, paths, found, pair_limit
):
    """Where a facet's centroid has no path round a rim, the path from a corner that has one instead.

    The gather counts the part of a straddling facet that has a path
    (:func:`_reached_triangles`), and that part need not hold the centroid; the shadow of the path
    to the centroid is then the shadow of a straight line the solve fell back to. Each such facet's
    shadow is taken along the path from its first corner with one -- as near a path of the part
    counted as the facet offers. Which corners have one is :func:`_continued_paths`', as for the
    gather. A facet with none keeps the centroid's, and carries nothing.
    """
    corners, shared = _shared_corners(surfaces, facets)
    corners = np.asarray(corners, dtype=float)
    corner_paths, corner_found, _ = _continued_paths(
        media, chain, corners, receivers, _corner_edges(shared), pair_limit
    )
    # (paths, rows, facets, 3): which of each facet's corners has a path.
    reached = corner_found[:, :, shared]
    first = np.argmax(reached, axis=-1)
    swap = ~found & np.any(reached, axis=-1)
    which = shared[np.arange(len(facets))[None, None], first]
    taken = np.take_along_axis(corner_paths, which[..., None, None], axis=2)
    sources = np.where(swap[..., None], corners[which], sources)
    paths = np.where(swap[..., None, None], taken, paths)
    return sources, paths


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
    masks = _route_masks(media, visibility, transmittance, host_points, plan)
    total = jnp.zeros(points.shape[0])
    for group, mask in zip(plan, masks, strict=True):
        total = total.at[group.rows].add(
            _group_field(surfaces, media, group, points, normals, mask, pair_limit)
        )
    return total


def _route_masks(media, visibility, transmittance, points, plan):
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
    shapes = [(_path_count(media, g.chain), len(g.rows), len(g.facets)) for g in plan]
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


#: Halvings that find where a path stops along a triangle's edge, round a sleeve's rim.
_CUT_HALVINGS = 12


def _seen_weight(receiver, receiver_normal, paths, facet_normal, emission, ratio, profile):
    """What a triangle sends a receiver along the paths to its three corners, 0 where any is missing.

    ``paths`` is shaped ``(..., 3)`` over the corners; ``receiver`` and ``receiver_normal`` (``None``
    for a point in a volume) are ``(..., 3)`` broadcast against the triangles, and ``facet_normal``
    likewise. The solid angle is the closed form on the corners' arrival directions; the radiance in
    each departure direction and what each path gets across are averaged over the corners.
    """
    apparent = receiver[..., None, :] + paths.arrival
    if receiver_normal is None:
        omega = solid_angle(receiver, apparent)
    else:
        omega = projected_solid_angle(receiver, receiver_normal, apparent)
    radiance = profile.radiance_per_exitance(paths.departure, facet_normal[..., None, :])
    carried = jnp.mean(radiance * paths.transmittance, axis=-1)
    seen = jnp.all(paths.valid, axis=-1)
    return jnp.where(seen, emission * ratio * carried * omega, 0.0)


def _group_field(surfaces, media, group: _Group, points, normals, mask, pair_limit):
    """What one group's receivers get from its facets along one route, ``(n_rows,)``.

    On a route through a region every starting point's path is a path of its own -- they are
    distinct where they are valid -- and each source triangle is seen once along each.
    """
    if _shot(media, group.chain) and not _traced(
        (points, surfaces.vertices, jax.lax.stop_gradient(media))
    ):
        return _clipped_field(surfaces, media, group, points, normals, mask, pair_limit)
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
        paths = jax.tree.map(lambda leaf: leaf[:, :, shared], paths)
        weight = _seen_weight(
            chunk[None, :, None, :],
            None if chunk_normals is None else chunk_normals[None, :, None, :],
            paths,
            normal[None, None],
            emission,
            ratio,
            group.profile,
        )
        if mask is not None:
            chunk_blocked = rest.pop(0)
            chunk_hidden = rest.pop(0) if mask[1] is not None else None
            # The bodies' axis first, as the fraction takes it; the starts ride with the pairs.
            weight = weight * surviving_fraction(
                jnp.moveaxis(chunk_blocked, 1, 0), chunk_hidden, mask[2]
            )
        return jnp.sum(weight, axis=(0, 2))

    return in_passes(arrays, pair_limit, _path_count(media, chain) * corners.shape[0], at)


def _clipped_field(surfaces, media, group: _Group, points, normals, mask, pair_limit):
    """:func:`_group_field` on a shot route, each straddling triangle cut to the part with a path.

    Which corners have a path, and where along each edge a path stops, are found on the host from
    the geometry and indices as given (:func:`_reached_triangles`); the paths to the resulting
    triangles' corners are then solved live, each polished from the unknowns the host found it by
    (:func:`~aquaflux.radiation.shooting.rim_crossings`), so they carry every derivative but the
    cut's.
    """
    chain, facets = group.chain, group.facets
    still = jax.lax.stop_gradient(media)
    side, row, facet, vertices, seeds = _reached_triangles(
        surfaces, still, chain, facets, np.asarray(points)[group.rows], pair_limit
    )
    if len(side) == 0:
        return jnp.zeros(len(group.rows))
    normal = jnp.take(surfaces.normal, facets[facet], axis=0)
    emission = jnp.take(surfaces.emission, facets[facet])
    ratio = (media.index_of(chain.legs[-1]) / media.index_of(chain.legs[0])) ** 2
    receivers = jnp.take(points, group.rows[row], axis=0)
    arrays = [
        (receivers, 0),
        (jnp.asarray(vertices), 0),
        (jnp.asarray(seeds), 0),
        (jnp.asarray(np.where(side == 0, 1.0, -1.0)), 0),
        (normal, 0),
        (emission, 0),
    ]
    if normals is not None:
        arrays.append((jnp.take(normals, group.rows[row], axis=0), 0))
    if mask is not None:
        blocked, hidden, _ = mask
        # Each triangle takes its facet's shadow along its side's path.
        arrays.append((blocked[side, :, row, facet], 0))
        if hidden is not None:
            arrays.append((hidden[side, row, facet], 0))

    def at(chunk, chunk_vertices, chunk_seeds, chunk_sides, chunk_normal, chunk_emission, *rest):
        rest = list(rest)
        chunk_normals = rest.pop(0) if normals is not None else None
        n = chunk.shape[0]
        paths, _ = _shot_paths(
            media,
            chain,
            chunk_vertices.reshape(-1, 3),
            jnp.repeat(chunk, 3, axis=0),
            jnp.repeat(chunk_sides, 3),
            chunk_seeds.reshape(-1, 2),
        )
        paths = jax.tree.map(lambda leaf: leaf.reshape(n, 3, *leaf.shape[1:]), paths)
        weight = _seen_weight(
            chunk,
            chunk_normals,
            paths,
            chunk_normal,
            chunk_emission,
            ratio,
            group.profile,
        )
        if mask is not None:
            chunk_blocked = rest.pop(0)
            chunk_hidden = rest.pop(0) if mask[1] is not None else None
            weight = weight * surviving_fraction(chunk_blocked.T, chunk_hidden, mask[2])
        return weight

    weight = in_passes(arrays, pair_limit, 3, at)
    return jax.ops.segment_sum(weight, jnp.asarray(row), num_segments=len(group.rows))


def _reached_triangles(surfaces, media, chain, facets, receivers, pair_limit):
    """The triangles of ``facets`` with a path at every corner, round each side of a shell.

    For each receiver and side, a facet with a path at all three corners is itself; one with a path
    at one or two is cut along each edge from a corner with a path to one without, at the last point
    of :data:`_CUT_HALVINGS` halvings that still has one, into one triangle (one corner with a
    path) or two (two); one with none gives nothing. Which corners have a path is
    :func:`_continued_paths`'. All on the host, at concrete values.

    Returns
    -------
    side, row, facet : np.ndarray of int, shape ``(n,)``
        Each triangle's side (0 for ``+1``, 1 for ``-1``), receiver (an index into ``receivers``) and
        facet (an index into ``facets``).
    vertices : np.ndarray, shape ``(n, 3, 3)``
    seeds : np.ndarray, shape ``(n, 3, 2)``
        The unknowns each vertex's path was found by, to polish it from in the traced pass.
    """
    corners, shared = _shared_corners(surfaces, facets)
    corners = np.asarray(corners, dtype=float)
    n_rows, n_corners = len(receivers), len(corners)
    _, reach, unknowns = _continued_paths(
        media, chain, corners, receivers, _corner_edges(shared), pair_limit
    )

    # (2, rows, facets, 3): which of each triangle's corners has a path.
    reached = reach[:, :, shared]
    count = reached.sum(axis=-1)
    whole = np.argwhere(count == 3)
    mixed = np.argwhere((count == 1) | (count == 2))

    # Every edge of a mixed triangle from a corner with a path to one without, once per side and
    # receiver however many triangles share it.
    s, r, f = mixed.T
    ends = shared[f]
    lit = reached[s, r, f]
    keys = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            edge = lit[:, a] & ~lit[:, b]
            keys.append(np.stack([s, r, ends[:, a], ends[:, b]], axis=1)[edge])
    keys = np.concatenate(keys) if keys else np.zeros((0, 4), dtype=int)
    keys = np.unique(keys, axis=0)

    def code(table):
        """Each ``(side, receiver, corner, corner)`` row as one integer, in the rows' sort order."""
        side_, row_, from_, to_ = (table[:, k].astype(np.int64) for k in range(4))
        return ((side_ * n_rows + row_) * n_corners + from_) * n_corners + to_

    cuts, cut_seeds = _cut_edges(
        media,
        chain,
        corners,
        receivers,
        keys,
        unknowns[keys[:, 0], keys[:, 1], keys[:, 2]],
        pair_limit,
    )

    def at_cut(index, a, b):
        """Where the path stops along each listed triangle's edge from corner ``a`` to ``b``."""
        wanted = np.stack([s[index], r[index], ends[index, a], ends[index, b]], axis=1)
        found = np.searchsorted(code(keys), code(wanted))
        return cuts[found], cut_seeds[found]

    def corner(index, a):
        """Each listed triangle's corner ``a``, and the unknowns its path was found by."""
        return corners[ends[index, a]], unknowns[s[index], r[index], ends[index, a]]

    sides, rows_out, facet_out = [whole[:, 0]], [whole[:, 1]], [whole[:, 2]]
    vertices = [corners[shared[whole[:, 2]]]]
    seeds = [unknowns[whole[:, :1], whole[:, 1:2], shared[whole[:, 2]]]]
    one = np.flatnonzero(lit.sum(axis=1) == 1)
    two = np.flatnonzero(lit.sum(axis=1) == 2)
    pieces = []
    if len(one):
        a = np.argmax(lit[one], axis=1)
        b, c = (a + 1) % 3, (a + 2) % 3
        pieces.append((one, [corner(one, a), at_cut(one, a, b), at_cut(one, a, c)]))
    if len(two):
        c = np.argmin(lit[two], axis=1)
        a, b = (c + 1) % 3, (c + 2) % 3
        cut_a, cut_b = at_cut(two, a, c), at_cut(two, b, c)
        pieces.append((two, [corner(two, a), corner(two, b), cut_b]))
        pieces.append((two, [corner(two, a), cut_b, cut_a]))
    for index, triangle in pieces:
        sides.append(s[index])
        rows_out.append(r[index])
        facet_out.append(f[index])
        vertices.append(np.stack([point for point, _ in triangle], axis=1))
        seeds.append(np.stack([seed for _, seed in triangle], axis=1))
    return (
        np.concatenate(sides),
        np.concatenate(rows_out),
        np.concatenate(facet_out),
        np.concatenate(vertices).reshape(-1, 3, 3),
        np.concatenate(seeds).reshape(-1, 3, 2),
    )


def _corner_edges(shared):
    """``(n_edges, 2)`` the distinct edges of the triangles ``shared`` lists, as pairs of corners."""
    edges = np.concatenate([shared[:, [0, 1]], shared[:, [1, 2]], shared[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _continued_paths(media, chain, sources, receivers, edges, pair_limit):
    """Paths round each side of a shell between every source and receiver, on a shot route.

    Each pair is first shot from the grid (:func:`~aquaflux.radiation.shooting.rim_crossings`). A
    grid can miss a path whose run of rays is narrower than its spacing, but the unknowns of a path
    fix it whatever its source -- the band they range over depends only on the shell, its core and
    ``beta`` -- so a source joined by an edge to one with a path is tried again by polishing from
    that path's unknowns. Repeated from every source that gains a path until none does: the paths
    spread along the surface from wherever the grid found one.

    Parameters
    ----------
    sources : np.ndarray, shape ``(n_sources, 3)``
    receivers : np.ndarray, shape ``(n_receivers, 3)``
    edges : np.ndarray of int, shape ``(n_edges, 2)``
        Pairs of sources near enough for one's path to start the other's.

    Returns
    -------
    points : np.ndarray, shape ``(2, n_receivers, n_sources, n_crossings, 3)``
    valid : np.ndarray of bool, shape ``(2, n_receivers, n_sources)``
    unknowns : np.ndarray, shape ``(2, n_receivers, n_sources, 2)``
        Each side's (0 for ``+1``, 1 for ``-1``).
    """
    sources, receivers = np.asarray(sources, dtype=float), np.asarray(receivers, dtype=float)
    n_sources = len(sources)
    shape = (2, len(receivers), n_sources)
    side, row, source = (index.reshape(-1) for index in np.indices(shape))
    points, valid, unknowns = _stepped_shots(
        media, chain, sources[source], receivers[row], side, None, pair_limit
    )
    directed = np.concatenate([edges, edges[:, ::-1]])
    fresh = valid.reshape(-1, n_sources)
    while True:
        reached = valid.reshape(-1, n_sources)
        pair, edge = np.nonzero(fresh[:, directed[:, 0]] & ~reached[:, directed[:, 1]])
        if len(pair) == 0:
            break
        tried = pair * n_sources + directed[edge, 1]
        found_points, found, found_unknowns = _stepped_shots(
            media,
            chain,
            sources[source[tried]],
            receivers[row[tried]],
            side[tried],
            unknowns[pair * n_sources + directed[edge, 0]],
            pair_limit,
        )
        # A source reached from several neighbours keeps the first path found.
        gained, first = np.unique(tried[found], return_index=True)
        points[gained] = found_points[found][first]
        unknowns[gained] = found_unknowns[found][first]
        valid[gained] = True
        fresh = np.zeros_like(reached)
        fresh.reshape(-1)[gained] = True
    return (
        points.reshape(*shape, *points.shape[1:]),
        valid.reshape(shape),
        unknowns.reshape(*shape, 2),
    )


def _stepped_shots(media, chain, sources, receivers, side, starts, pair_limit):
    """:func:`~aquaflux.radiation.refraction._shot_paths` on the host, a step of pairs at a time.

    ``side`` is ``(n,)`` of 0 (``+1``) or 1 (``-1``) and ``starts`` ``(n, 2)`` or ``None`` (the
    grid). A step's paths each scan a grid, so it is sized as a pass of that many pairs, and padded
    to one shape -- repeating its last pair, whose answers are dropped -- so it compiles once.
    Returns the crossing points, whether each path exists, and its unknowns, as numpy arrays.
    """
    n = len(sources)
    signs = np.where(np.asarray(side) == 0, 1.0, -1.0)
    step = max(1, min(receivers_per_step(pair_limit, 1), n))
    points, valid, unknowns = [], [], []
    for first in range(0, n, step):
        rows = np.minimum(np.arange(first, first + step), n - 1)
        paths, found = _shot_paths(
            media,
            chain,
            jnp.asarray(sources[rows]),
            jnp.asarray(receivers[rows]),
            jnp.asarray(signs[rows]),
            None if starts is None else jnp.asarray(starts[rows]),
        )
        keep = slice(0, n - first)
        points.append(np.asarray(paths.points)[keep])
        valid.append(np.asarray(paths.valid)[keep])
        unknowns.append(np.asarray(found)[keep])
    if n == 0:
        return np.zeros((0, len(chain.crossings), 3)), np.zeros(0, bool), np.zeros((0, 2))
    return np.concatenate(points), np.concatenate(valid), np.concatenate(unknowns)


def _cut_edges(media, chain, corners, receivers, keys, inside_unknowns, pair_limit):
    """Where along each edge the path round one side of a shell stops, and its unknowns there.

    Each row of ``keys`` is ``(side, receiver, from corner, to corner)``: the path exists at the
    first corner, found by ``inside_unknowns``, and not at the second. The edge is halved
    :data:`_CUT_HALVINGS` times, each middle polished from the unknowns of the nearest point found to
    have a path, keeping the half across which the path stops; the last point found to have one is
    returned -- so the triangle cut there has a path at every corner. Returns ``(n_edges, 3)``
    points and ``(n_edges, 2)`` unknowns.
    """
    if len(keys) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))
    inside, outside = corners[keys[:, 2]], corners[keys[:, 3]]
    seeds = np.array(inside_unknowns, dtype=float)
    targets = receivers[keys[:, 1]]
    for _ in range(_CUT_HALVINGS):
        middle = 0.5 * (inside + outside)
        _, found, unknowns = _stepped_shots(
            media, chain, middle, targets, keys[:, 0], seeds, pair_limit
        )
        inside = np.where(found[:, None], middle, inside)
        seeds = np.where(found[:, None], unknowns, seeds)
        outside = np.where(found[:, None], outside, middle)
    return inside, seeds


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
