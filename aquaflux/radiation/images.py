"""One specular bounce, gathered into the volume as the sources' mirror images.

Light leaving a source, reflecting once off a flat specular facet and arriving at a point in the
fluid travels a broken path; reflected in the facet's plane, the second leg straightens it out,
so the point receives that light along a straight line from the source's **mirror image**. The
image is an ordinary surface set (:meth:`~aquaflux.radiation.mirrors.Mirror.image`): reflected
triangles carrying the source's own emission and its distribution's mirror image. So a specular
bounce is a direct gather of images, with three things the direct gather does not have.

- **The image is seen only through the mirror.** A receiver sees the part of an image whose
  directions pass through one of the mirror's facets -- its aperture -- and nothing past the
  aperture's edge. That part is the image clipped to the aperture triangle's cone, which is the
  intersection :func:`~aquaflux.radiation.silhouette.covered_by` computes for a blocker in front
  of a source; the aperture triangles tile their plane without overlapping, so their shares add
  exactly. The clip's depth cut keeps only the part of an image beyond the mirror, which is what
  makes a source straddling the mirror's plane need no special case: only its part in front of
  the mirror has an image there.
- **Each aperture facet weights by its own specular reflectance**, which is live: it multiplies
  the clipped geometry.
- **The medium is crossed along the two real legs** -- receiver to the point where the unfolded
  path meets the mirror, and on to the source -- so a graded medium is integrated where the light
  actually goes. For a uniform medium the two legs sum to the unfolded distance exactly.

A point source's image is a point: it is seen through the one aperture facet the line from the
receiver to it crosses, and a line through an edge shared by two facets is credited to one of
them, so a symmetric layout is not counted twice.

What is **not** here: anything standing in the way of either leg. The image is gathered as though
the receiver saw the mirror and the mirror the source unobstructed -- as
:func:`~aquaflux.radiation.gather.direct_fluence_rate` gathers without a mask -- and the medium
and the aperture are the only things between them.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.clipping import decidable_heights, spanning_plane
from aquaflux.radiation.gather import (
    _areal_groups,
    _emitter_direction,
    _groups,
    _one_geometry,
)
from aquaflux.radiation.mirrors import Mirror
from aquaflux.radiation.silhouette import SourceView, _orientation, covered_by, source_view
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, in_passes
from aquaflux.vectors import dot

__all__ = [
    "PlaneExchange",
    "mirrored_fluence_rate",
    "mirrored_irradiance",
    "plane_exchange",
    "summed_mirrored_fluence_rate",
]


def mirrored_fluence_rate(
    surfaces: Surfaces,
    mirrors: Sequence[Mirror],
    points,
    *,
    absorption: Absorption | None = None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
):
    """Fluence rate at each point from one specular bounce of every source off every mirror.

    The specular counterpart of :func:`~aquaflux.radiation.gather.direct_fluence_rate`: each
    source's mirror image in each mirror, seen through that mirror's facets and weighted by each
    facet's ``specular_reflectance``. One bounce only -- light reflecting off two mirrors in turn
    is not included -- and nothing occludes either leg of the path.

    Parameters
    ----------
    surfaces : Surfaces
        The sources, and the facets the mirrors were found in: each mirror's facets index this
        set, and their ``specular_reflectance`` is read from it.
    mirrors : sequence of Mirror
        From :func:`~aquaflux.radiation.mirrors.planar_mirrors` on this set.
    points : array_like, shape ``(n_points, 3)``
        Where the fluence rate is wanted.
    absorption : Absorption, optional
        The medium, crossed along both legs of each path. Vacuum if omitted.
    pair_limit : int, optional
        Receiver-by-source-by-aperture-facet triples one pass may form.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If a mirror names a facet the surface set does not have.
    """
    return summed_mirrored_fluence_rate(
        (surfaces,), mirrors, points, absorption=absorption, pair_limit=pair_limit
    )


def summed_mirrored_fluence_rate(
    sets,
    mirrors: Sequence[Mirror],
    points,
    *,
    absorption: Absorption | None = None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
):
    """The mirrored fluence rate of several surface sets **on one geometry**, summed.

    What :func:`mirrored_fluence_rate` gives for each set, added, with the geometry -- the clipped
    images, the paths and their attenuation -- formed once from the first set and shared, as
    :func:`~aquaflux.radiation.gather.summed_fluence_rate` does for the direct field. The specular
    reflectance is read from the first set too: it belongs to the mirrors, not to the light.

    Parameters
    ----------
    sets : sequence of Surfaces
        Sets sharing one geometry -- ``with_optics(...)`` of one set -- whose emission, power and
        profiles are each gathered.
    mirrors, points, absorption, pair_limit
        As for :func:`mirrored_fluence_rate`.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``

    Raises
    ------
    ValueError
        If no set is given, the sets do not share their geometry, or a mirror names a facet the
        set does not have.
    """
    return _mirrored(sets, mirrors, points, None, absorption, pair_limit, point_sources_only=False)


def mirrored_irradiance(
    surfaces: Surfaces,
    mirrors: Sequence[Mirror],
    points,
    normals,
    *,
    absorption: Absorption | None = None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
    point_sources_only: bool = False,
):
    """Irradiance on oriented points from one specular bounce of every source off every mirror.

    The specular counterpart of :func:`~aquaflux.radiation.gather.direct_irradiance`, as
    :func:`mirrored_fluence_rate` is of the direct fluence rate: the same images through the same
    apertures, each weighted by the cosine of its direction from the receiver's normal rather than
    counted from every direction alike, and nothing arriving from behind the receiver.

    Parameters
    ----------
    surfaces, mirrors, absorption, pair_limit
        As for :func:`mirrored_fluence_rate`.
    points : array_like, shape ``(n_points, 3)``
    normals : array_like, shape ``(n_points, 3)``
        Unit normal at each point, on the side it receives light from.
    point_sources_only : bool, optional
        Gather only the point sources' images -- what the facet-to-facet transfer cannot carry,
        since a point source has no area in it.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Irradiance in W/m².
    """
    return _mirrored(
        (surfaces,),
        mirrors,
        points,
        jnp.asarray(normals, dtype=float),
        absorption,
        pair_limit,
        point_sources_only=point_sources_only,
    )


def _mirrored(sets, mirrors, points, normals, absorption, pair_limit, *, point_sources_only):
    """The mirrored field of several sets on one geometry: fluence rate, or irradiance given normals."""
    sets = tuple(sets)
    geometry = _one_geometry(sets)
    # Read before conversion: inside a trace a concrete array becomes a tracer once it passes
    # through jnp, and which receivers lie in front of a mirror is only worth knowing when it can
    # be read.
    readable = not any(isinstance(array, jax.core.Tracer) for array in (points, geometry.vertices))
    host_points = np.asarray(points, dtype=float) if readable else None
    points = jnp.asarray(points, dtype=float)
    total = jnp.zeros(points.shape[0])
    for mirror in mirrors:
        facets = np.asarray(mirror.facets)
        if facets.size and (facets.min() < 0 or facets.max() >= geometry.n_facets):
            msg = (
                f"a mirror names facets {facets.min()} to {facets.max()}, outside this surface "
                f"set's {geometry.n_facets}; it was found in a different set"
            )
            raise ValueError(msg)
        path = _Path(mirror, geometry, absorption)
        receivers = _in_front(mirror, host_points, points.shape[0], readable)
        if not receivers.size:
            continue
        at = (points[receivers], None if normals is None else normals[receivers])
        sources = _sources_in_front(mirror, geometry, readable)
        images = tuple(mirror.image(surfaces) for surfaces in sets)
        received = _point_images(sets, images, sources, path, at, pair_limit)
        if not point_sources_only:
            received = received + _areal_images(sets, images, sources, path, at, pair_limit)
        total = total.at[receivers].add(received)
    return total


class _Path:
    """What every path through one mirror shares: its plane, its aperture and the medium.

    Host-side bookkeeping for one mirror's gather, never traced as a whole, so a plain object.
    """

    def __init__(self, mirror: Mirror, geometry: Surfaces, absorption):
        self.mirror = mirror
        self.aperture = jnp.take(geometry.vertices, jnp.asarray(mirror.facets), axis=0)
        self.reflectance = jnp.take(
            jnp.asarray(geometry.specular_reflectance, dtype=float), jnp.asarray(mirror.facets)
        )
        self.absorption = absorption

    def transmittance(self, image_centroid, receivers):
        """Surviving fraction along each receiver-to-mirror-to-source path, or one in vacuum.

        The path is the unfolded line from the receiver to the image's centroid, broken where it
        crosses the mirror's plane; the second leg is reflected back to the real source, so a
        graded medium is integrated through the fluid the light crosses rather than through its
        mirror image.

        Parameters
        ----------
        image_centroid : jnp.ndarray, shape ``(..., 3)``
        receivers : jnp.ndarray, shape ``(..., 3)``
            Broadcasting against ``image_centroid``.
        """
        if self.absorption is None:
            return jnp.asarray(1.0)
        point, normal = self.mirror.point, self.mirror.normal
        near = dot(receivers - point, normal)
        far = dot(image_centroid - point, normal)
        # Receivers are in front of the plane and images behind it, so the two heights have
        # opposite signs and the crossing lies between the ends; the guard only keeps a
        # degenerate pair finite.
        span = near - far
        share = near / jnp.where(span == 0.0, 1.0, span)
        crossing = receivers + share[..., None] * (image_centroid - receivers)
        source = self.mirror.reflect_points(image_centroid)
        depth = self.absorption.optical_depth(receivers, crossing)
        return jnp.exp(-(depth + self.absorption.optical_depth(crossing, source)))


def _in_front(mirror: Mirror, host_points, n_points: int, readable: bool) -> np.ndarray:
    """Indices of the points strictly in front of the mirror, or all of them if unreadable.

    A point on or behind the plane sees no image through it: the mirror reflects on the side it
    faces. Leaving such a point in would cost work and change nothing, since the clip keeps none
    of an image seen from there.
    """
    if not readable:
        return np.arange(n_points)
    height = (host_points - np.asarray(mirror.point)) @ np.asarray(mirror.normal)
    return np.flatnonzero(height > 0.0)


def _sources_in_front(mirror: Mirror, geometry: Surfaces, readable: bool) -> np.ndarray:
    """Facets with any part strictly in front of the mirror, or every facet if unreadable.

    A facet wholly on or behind the plane has no image a receiver in front could see -- its own
    aperture facets and every facet coplanar with them among them -- and the clip would give it
    zero. A facet straddling the plane is kept: its part in front has an image.
    """
    if not readable:
        return np.arange(geometry.n_facets)
    corners = np.asarray(geometry.vertices) - np.asarray(mirror.point)
    height = corners @ np.asarray(mirror.normal)
    return np.flatnonzero(height.max(axis=1) > 0.0)


def _passes(receivers, pair_limit, per_receiver, body):
    """:func:`~aquaflux.radiation.work.in_passes` over receivers, with their normals if any.

    ``receivers`` is a ``(points, normals)`` pair, ``normals`` ``None`` for points in the volume;
    ``body`` takes a chunk of each.
    """
    points, normals = receivers
    if normals is None:
        return in_passes(((points, 0),), pair_limit, per_receiver, lambda p: body(p, None))
    return in_passes(((points, 0), (normals, 0)), pair_limit, per_receiver, body)


def _areal_images(sets, images, sources, path: _Path, receivers, pair_limit):
    """What the areal facets' images send each receiver through the aperture, ``(n,)``."""
    total = jnp.zeros(receivers[0].shape[0])
    image_geometry = images[0]
    n_aperture = path.aperture.shape[0]
    for group in _areal_groups(images):
        facets = np.intersect1d(group.facets, sources)
        if not facets.size:
            continue
        index = jnp.asarray(facets)
        vertices = jnp.take(image_geometry.vertices, index, axis=0)
        centroid = jnp.take(image_geometry.centroid, index, axis=0)
        normal = jnp.take(image_geometry.normal, index, axis=0)
        emission = [jnp.take(surfaces.emission, index) for surfaces in sets]

        def at(
            chunk,
            chunk_normals,
            vertices=vertices,
            centroid=centroid,
            normal=normal,
            emission=emission,
            profiles=group.profiles,
        ):
            seen = _seen_through(path, chunk, chunk_normals, vertices)
            direction, _ = _emitter_direction(centroid[None], chunk[:, None])
            weight = seen * path.transmittance(centroid[None], chunk[:, None])
            received = jnp.zeros(chunk.shape[0])
            for flux, profile in zip(emission, profiles, strict=True):
                radiance = profile.radiance_per_exitance(direction, normal[None])
                received = received + jnp.sum(flux * radiance * weight, axis=1)
            return received

        total = total + _passes(receivers, pair_limit, len(facets) * n_aperture, at)
    return total


def _seen_through(path: _Path, receivers, normals, vertices):
    """:func:`_seen_through_aperture` through ``path``'s aperture, weighted by its reflectances."""
    return _seen_through_aperture(path.aperture, path.reflectance, receivers, normals, vertices)


def _seen_through_aperture(aperture, weights, receivers, normals, vertices):
    """Solid angle of each image seen through the aperture, weighted by each facet's reflectance.

    The plain solid angle for a point in the volume, the projected one for a point with a normal.

    Parameters
    ----------
    aperture : jnp.ndarray, shape ``(a, 3, 3)``
        The mirror's facets.
    weights : jnp.ndarray, shape ``(a,)``
        What each aperture facet's share is weighted by: its specular reflectance, or one.
    receivers : jnp.ndarray, shape ``(r, 3)``
    normals : jnp.ndarray, shape ``(r, 3)``, or None
    vertices : jnp.ndarray, shape ``(s, 3, 3)``
        The images' corners.

    Returns
    -------
    jnp.ndarray, shape ``(r, s)``
        ``sum over aperture facets a of weights_a * (solid angle of the image within a's cone)``.
    """
    pair_normals = None if normals is None else normals[:, None, :]
    view = source_view(receivers[:, None, :], pair_normals, vertices[None])
    # One view per (receiver, image), shared by every aperture facet it is clipped against. The
    # clip stacks its candidates, so everything is spread to one batch shape rather than left to
    # broadcast.
    batch = (receivers.shape[0], vertices.shape[0], aperture.shape[0])

    def spread(array, trailing):
        return jnp.broadcast_to(array[:, :, None], (*batch, *trailing))

    spread_view = SourceView(
        loop=spread(view.loop, view.loop.shape[2:]),
        whole=spread(view.whole, ()),
        support=spread(view.support, (3,)),
        through=spread(view.through, (3,)),
    )
    to_aperture = jnp.broadcast_to(
        aperture[None, None] - receivers[:, None, None, None, :], (*batch, 3, 3)
    )
    triple_normals = None if normals is None else spread(pair_normals, (3,))
    fraction, _ = covered_by(spread_view, triple_normals, to_aperture)
    return jnp.abs(view.whole) * (fraction @ weights)


def _point_images(sets, images, sources, path: _Path, receivers, pair_limit):
    """What the point sources' images send each receiver through the aperture, ``(n,)``."""
    image_geometry = images[0]
    point_all = np.intersect1d(np.flatnonzero(image_geometry.is_point_source), sources)
    if not point_all.size:
        return jnp.zeros(receivers[0].shape[0])
    index = jnp.asarray(point_all)
    position = jnp.take(image_geometry.centroid, index, axis=0)
    plans = [
        [
            (profile, np.searchsorted(point_all, np.intersect1d(point, point_all)))
            for profile, _, point in _groups(surfaces)
        ]
        for surfaces in images
    ]
    power = [jnp.take(surfaces.power, index) for surfaces in sets]

    def at(chunk, chunk_normals):
        reflectance = _crossed_reflectance(path, chunk, position)
        direction, distance_squared = _emitter_direction(position[None], chunk[:, None])
        weight = reflectance * path.transmittance(position[None], chunk[:, None])
        weight = weight / jnp.where(distance_squared == 0.0, 1.0, distance_squared)
        if chunk_normals is not None:
            # The light arrives travelling along `direction`, so it strikes a surface facing back
            # along it; nothing is received from behind.
            weight = weight * jnp.maximum(-dot(direction, chunk_normals[:, None, :]), 0.0)
        received = jnp.zeros(chunk.shape[0])
        for flux, plan in zip(power, plans, strict=True):
            for profile, columns in plan:
                if not columns.size:
                    continue
                pick = jnp.asarray(columns)
                intensity = profile.intensity_fraction(
                    jnp.take(direction, pick, axis=1), jnp.zeros(3)
                )
                received = received + jnp.sum(
                    jnp.take(flux, pick) * intensity * jnp.take(weight, pick, axis=1), axis=1
                )
        return received

    return _passes(receivers, pair_limit, len(point_all) * path.aperture.shape[0], at)


def _crossed_reflectance(path: _Path, receivers, images):
    """The specular reflectance of the aperture facet each receiver-to-image line crosses.

    Zero where the line crosses none. A line through an edge or corner shared by several facets
    lies in each one's closed cone; it is credited to the first, so the aperture as a whole counts
    it exactly once.

    Parameters
    ----------
    receivers : jnp.ndarray, shape ``(r, 3)``
    images : jnp.ndarray, shape ``(p, 3)``
        Image points, behind the mirror.

    Returns
    -------
    jnp.ndarray, shape ``(r, p)``
    """
    to_aperture = path.aperture[None] - receivers[:, None, None, :]
    volume, edge_on = _orientation(to_aperture)
    facing = jnp.sign(volume)[..., None]
    direction = (images[None] - receivers[:, None])[:, :, None, None, :]
    inside = ~edge_on[:, None]
    for k in range(3):
        plane = spanning_plane(to_aperture[..., k, :], to_aperture[..., (k + 1) % 3, :]) * facing
        height = decidable_heights(direction, plane[:, None])[..., 0]
        inside = inside & (height >= 0.0)
    # The line must also cross the plane between its ends, which a receiver behind the mirror or
    # an image in front of it does not.
    near = dot(receivers - path.mirror.point, path.mirror.normal)[:, None]
    far = dot(images - path.mirror.point, path.mirror.normal)[None]
    crosses = (near > 0.0) & (far < 0.0)
    first = jnp.argmax(inside, axis=-1)
    reflectance = jnp.take(path.reflectance, first)
    return jnp.where(crosses & jnp.any(inside, axis=-1), reflectance, 0.0)


class PlaneExchange(eqx.Module):
    """Facet-to-facet transfer by one specular bounce in one mirror, and what it is attenuated by.

    The specular counterpart of the direct transfer's frozen arrays, for one plane: the fraction
    of what leaves facet ``j`` Lambertian that lands on facet ``i`` after reflecting in the mirror
    (with unit reflectance), and the two one-point quantities the live optics multiply it by.

    Attributes
    ----------
    geometric : jnp.ndarray, shape ``(n_facets, n_facets)``
        Row ``i``, column ``j``: the projected solid angle of ``j``'s image seen through the
        mirror from ``i``, over ``pi``, averaged over ``i``'s quadrature points. Zero where either
        facet is a point source or wholly on or behind the plane.
    separation : jnp.ndarray, shape ``(n_facets, n_facets)``
        Distance from ``i``'s centroid to ``j``'s image's centroid: the length of the unfolded
        path, which a uniform medium attenuates over.
    source_cosine : jnp.ndarray, shape ``(n_facets, n_facets)``
        Cosine, at ``j``'s image, between its normal and the direction to ``i``'s centroid -- what
        a non-Lambertian source's distribution is asked about.
    """

    geometric: jnp.ndarray
    separation: jnp.ndarray
    source_cosine: jnp.ndarray


def plane_exchange(
    mirror: Mirror,
    geometry: Surfaces,
    sample,
    weight,
    *,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
) -> PlaneExchange:
    """Build one mirror's facet-to-facet exchange, integrating over each receiving facet.

    The receiving half is quadrature, as for the direct transfer; the sending half -- the image
    clipped to the aperture -- is closed form. Only facets with some part in front of the plane
    can receive or send through it, so only those rows and columns are computed.

    Parameters
    ----------
    mirror : Mirror
    geometry : Surfaces
        The facets, which the mirror's facets index.
    sample : jnp.ndarray, shape ``(n_facets, n_points, 3)``
        Quadrature points on each facet.
    weight : jnp.ndarray, shape ``(n_points,)``
        Their weights, summing to one.
    pair_limit : int, optional
        Receiving-facet-by-source-by-aperture-facet triples one pass may form, per quadrature
        point.

    Returns
    -------
    PlaneExchange
    """
    n = geometry.n_facets
    areal = ~geometry.is_point_source
    in_front = np.zeros(n, dtype=bool)
    in_front[_sources_in_front(mirror, geometry, readable=True)] = True
    rows = np.flatnonzero(in_front & areal)
    columns = rows
    aperture = jnp.take(geometry.vertices, jnp.asarray(mirror.facets), axis=0)
    ones = jnp.ones(aperture.shape[0])
    image = mirror.image(geometry)
    index = jnp.asarray(columns)
    image_vertices = jnp.take(image.vertices, index, axis=0)
    image_centroid = jnp.take(image.centroid, index, axis=0)
    image_normal = jnp.take(image.normal, index, axis=0)
    weight = jnp.asarray(weight, dtype=float)

    def at(points, normals):
        total = jnp.zeros((points.shape[0], columns.size))
        for k in range(points.shape[1]):
            seen = _seen_through_aperture(aperture, ones, points[:, k], normals, image_vertices)
            total = total + weight[k] * seen
        return total / jnp.pi

    geometric = jnp.zeros((n, n))
    separation = jnp.zeros((n, n))
    source_cosine = jnp.zeros((n, n))
    if rows.size:
        row_index = jnp.asarray(rows)
        block = in_passes(
            (
                (jnp.take(jnp.asarray(sample), row_index, axis=0), 0),
                (jnp.take(geometry.normal, row_index, axis=0), 0),
            ),
            pair_limit,
            columns.size * aperture.shape[0],
            at,
        )
        offset = jnp.take(geometry.centroid, row_index, axis=0)[:, None] - image_centroid[None]
        distance_squared = dot(offset, offset)
        distance = jnp.sqrt(jnp.where(distance_squared == 0.0, 1.0, distance_squared))
        cosine = dot(offset, image_normal[None]) / distance
        grid = np.ix_(rows, columns)
        geometric = geometric.at[grid].set(block)
        separation = separation.at[grid].set(jnp.where(distance_squared == 0.0, 0.0, distance))
        source_cosine = source_cosine.at[grid].set(cosine)
    return PlaneExchange(geometric=geometric, separation=separation, source_cosine=source_cosine)
