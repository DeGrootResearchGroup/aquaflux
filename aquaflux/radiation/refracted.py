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
from aquaflux.radiation.refraction import Chain, Media, solve_paths
from aquaflux.radiation.self_occlusion import RayCastOcclusion, SelfOcclusion
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import (
    Visibility,
    refuse_points_inside,
    same_receivers,
    surviving_fraction,
)
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, in_passes, receivers_per_pass

__all__ = [
    "RefractedVisibility",
    "build_refracted_visibility",
    "refracted_fluence_rate",
    "refracted_irradiance",
]


class RefractedVisibility(eqx.Module):
    """A frozen record of what stands across each refracted path.

    Its own type rather than a bare :class:`~aquaflux.radiation.visibility.Visibility`, because a
    mask of straight segments and a mask of refracted paths are laid out alike and mean different
    things: one passed for the other would put every shadow where the light does not go.

    Attributes
    ----------
    mask : Visibility
        Laid out as the direct gather's mask: ``blocked`` per body, receiver and facet, true where
        the body lies across any leg of the path from the facet's centroid to the receiver; and
        ``hidden_by_geometry`` where the surface's own triangles do. Pairs in one medium are
        recorded clear and never read.
    """

    mask: Visibility


@dataclasses.dataclass(frozen=True)
class _Group:
    """Facets in one medium and receivers in another, with the crossings between them."""

    chain: Chain
    facets: np.ndarray
    rows: np.ndarray
    profile: object


def _plan(surfaces: Surfaces, media: Media, points: np.ndarray) -> tuple[_Group, ...]:
    """Every group of pairs with a crossing, worked out on the host from concrete positions."""
    facet_region = _facet_regions(surfaces, media)
    receiver_region = media.region_of(points, "receiver")
    parents = media.parents
    groups = []
    for profile, areal, point in _groups(surfaces):
        for source in np.unique(facet_region[point]):
            if np.any(receiver_region != source):
                msg = (
                    "a point source and a receiver are in different media; a point source is seen "
                    "along one path with no area to spread over, and is not gathered through a "
                    "transparent surface. Describe the source as an areal one."
                )
                raise ValueError(msg)
        for source in np.unique(facet_region[areal]):
            facets = areal[facet_region[areal] == source]
            for receiver in np.unique(receiver_region):
                if receiver == source:
                    continue
                groups.append(
                    _Group(
                        chain=Chain.between(parents, int(source), int(receiver)),
                        facets=facets,
                        rows=np.flatnonzero(receiver_region == receiver),
                        profile=profile,
                    )
                )
    return tuple(groups)


def _facet_regions(surfaces: Surfaces, media: Media) -> np.ndarray:
    """The medium each facet emits into, refusing a facet whose corners are in different ones."""
    vertices = np.asarray(surfaces.vertices, dtype=float)
    corners = media.region_of(vertices.reshape(-1, 3), "facet vertex").reshape(-1, 3)
    centres = media.region_of(np.asarray(surfaces.centroid), "facet centroid")
    split = np.flatnonzero(np.any(corners != centres[:, None], axis=1))
    if len(split):
        msg = (
            f"{len(split)} facet(s) cross the surface of a transparent region (first few: "
            f"{split[:8].tolist()}); a facet must lie in one medium."
        )
        raise ValueError(msg)
    return centres


def build_refracted_visibility(
    occluders,
    surfaces: Surfaces,
    media: Media,
    points,
    *,
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
        Receiver positions, in the volume.
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
    n_points, n_facets = host_points.shape[0], surfaces.n_facets
    blocked = np.zeros((len(occluders), n_points, n_facets), dtype=bool)
    hidden = None
    centroid = np.asarray(surfaces.centroid, dtype=float)
    near = offset_scale * np.sqrt(np.asarray(surfaces.area, dtype=float))
    for group in _plan(surfaces, media, host_points):
        per_pass = receivers_per_pass(pair_limit, len(group.facets))
        for start in range(0, len(group.rows), per_pass):
            rows = group.rows[start : start + per_pass]
            paths = _centroid_paths(media, group.chain, centroid[group.facets], host_points[rows])
            corners = np.concatenate(
                [
                    np.broadcast_to(
                        centroid[group.facets][None, :, None], (*paths.shape[:2], 1, 3)
                    ),
                    paths,
                    np.broadcast_to(host_points[rows][:, None, None], (*paths.shape[:2], 1, 3)),
                ],
                axis=2,
            )
            origin = corners[:, :, :-1].reshape(-1, 3)
            target = corners[:, :, 1:].reshape(-1, 3)
            n_legs = corners.shape[2] - 1
            pair_near = np.broadcast_to(
                near[group.facets][None, :, None], (*paths.shape[:2], n_legs)
            )
            leg_near = pair_near.reshape(-1)
            cell = np.ix_(rows, group.facets)
            for index, body in enumerate(occluders):
                crossed = np.asarray(
                    _body_blocks(
                        body, jnp.asarray(origin), jnp.asarray(target), jnp.asarray(leg_near)
                    )
                ).reshape(len(rows), len(group.facets), n_legs)
                blocked[index][cell] = np.any(crossed, axis=2)
            source = np.broadcast_to(group.facets[None, :, None], (*paths.shape[:2], n_legs))
            first = np.zeros(n_legs, dtype=bool)
            first[0] = True
            exclude = np.where(first[None, None, :], source, -1).reshape(-1, 1)
            cut = strategy.segments_hidden(surfaces, origin, target, leg_near, exclude)
            if cut is not None:
                if hidden is None:
                    hidden = np.zeros((n_points, n_facets), dtype=bool)
                hidden[cell] = np.any(
                    np.asarray(cut).reshape(len(rows), len(group.facets), n_legs), axis=2
                )
    return RefractedVisibility(
        mask=Visibility(
            blocked=jnp.asarray(blocked),
            receivers=jnp.asarray(host_points),
            hidden_by_geometry=None if hidden is None else jnp.asarray(hidden),
            overlapping=None,
        )
    )


def _centroid_paths(media: Media, chain: Chain, sources, receivers) -> np.ndarray:
    """``(n_receivers, n_sources, n_crossings, 3)`` crossing points, on the host."""
    paths = solve_paths(media, chain, jnp.asarray(sources)[None], jnp.asarray(receivers)[:, None])
    return np.asarray(paths.points)


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
    """The sum over every group of pairs with a crossing; the kernel by whether there are normals."""
    host_points = np.asarray(points, dtype=float)
    points = jnp.asarray(points, dtype=float)
    layers = _mask_layers(visibility, transmittance, host_points)
    total = jnp.zeros(points.shape[0])
    for group in _plan(surfaces, media, host_points):
        total = total.at[group.rows].add(
            _group_field(surfaces, media, group, points, normals, layers, pair_limit)
        )
    return total


def _mask_layers(visibility, transmittance, points):
    """The mask's arrays and the bodies' transmittance, or ``None`` with no mask."""
    if visibility is None:
        if transmittance is not None:
            msg = "transmittance was given without a visibility mask to apply it to"
            raise ValueError(msg)
        return None
    if not isinstance(visibility, RefractedVisibility):
        msg = (
            "visibility must be a RefractedVisibility, from build_refracted_visibility; got "
            f"{type(visibility).__name__}. A mask of straight segments marks where light does "
            "not go along a refracted path."
        )
        raise TypeError(msg)
    mask = visibility.mask
    same_receivers(mask.receivers, points, "refracted visibility mask")
    if transmittance is None:
        transmittance = jnp.zeros(mask.n_occluders)
    return mask.blocked, mask.hidden_by_geometry, transmittance


def _group_field(surfaces, media, group: _Group, points, normals, layers, pair_limit):
    """What one group's receivers get from its facets, ``(n_rows,)``."""
    facets = group.facets
    corners, shared = _shared_corners(surfaces, facets)
    normal = jnp.take(surfaces.normal, facets, axis=0)
    emission = jnp.take(surfaces.emission, facets)
    ratio = (media.index_of(group.chain.legs[-1]) / media.index_of(group.chain.legs[0])) ** 2
    receivers = jnp.take(points, group.rows, axis=0)
    arrays = [(receivers, 0)]
    if normals is not None:
        arrays.append((jnp.take(normals, group.rows, axis=0), 0))
    if layers is not None:
        blocked, hidden, _ = layers
        cell = np.ix_(group.rows, facets)
        arrays.append((blocked[:, cell[0], cell[1]], 1))
        if hidden is not None:
            arrays.append((hidden[cell], 0))

    def at(chunk, *rest):
        rest = list(rest)
        chunk_normals = rest.pop(0) if normals is not None else None
        paths = solve_paths(media, group.chain, corners[None], chunk[:, None])
        arrival = paths.arrival[:, shared]
        apparent = chunk[:, None, None, :] + arrival
        if chunk_normals is None:
            omega = solid_angle(chunk[:, None, :], apparent)
        else:
            omega = projected_solid_angle(chunk[:, None, :], chunk_normals[:, None, :], apparent)
        radiance = group.profile.radiance_per_exitance(
            paths.departure[:, shared], normal[None, :, None, :]
        )
        carried = jnp.mean(radiance * paths.transmittance[:, shared], axis=-1)
        seen = jnp.all(paths.valid[:, shared], axis=-1)
        weight = jnp.where(seen, emission * ratio * carried * omega, 0.0)
        if layers is not None:
            chunk_blocked = rest.pop(0)
            chunk_hidden = rest.pop(0) if layers[1] is not None else None
            weight = weight * surviving_fraction(chunk_blocked, chunk_hidden, layers[2])
        return jnp.sum(weight, axis=1)

    return in_passes(arrays, pair_limit, corners.shape[0], at)


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
