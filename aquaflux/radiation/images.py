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

**What stands in the way of either leg** is a frozen mask per mirror
(:class:`~aquaflux.radiation.mirror_visibility.MirrorVisibility`), passed with each body's live
transmittance as the direct gather takes its mask. Without one the image is gathered as though the
receiver saw the mirror and the mirror the source unobstructed -- as
:func:`~aquaflux.radiation.gather.direct_fluence_rate` gathers without a mask -- and the medium and
the aperture are the only things between them.
"""

from __future__ import annotations

import dataclasses
import functools
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
    areal_layout,
)
from aquaflux.radiation.mirror_visibility import MirrorVisibility, reflected_surviving
from aquaflux.radiation.mirrors import Mirror
from aquaflux.radiation.silhouette import (
    SourceView,
    _orientation,
    angular_cone,
    cones_may_overlap,
    covered_by,
    source_view,
)
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
    shadows=None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
):
    """Fluence rate at each point from one specular bounce of every source off every mirror.

    The specular counterpart of :func:`~aquaflux.radiation.gather.direct_fluence_rate`: each
    source's mirror image in each mirror, seen through that mirror's facets and weighted by each
    facet's ``specular_reflectance``. One bounce only -- light reflecting off two mirrors in turn
    is not included -- and either leg of a path is occluded only through ``shadows``.

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
    shadows : sequence of MirrorVisibility, optional
        What stands across each reflected path: one mask per mirror, in the order of
        ``mirrors``, each built for ``points`` (:func:`~aquaflux.radiation.mirror_visibility.build_mirror_visibility`).
        Without them nothing does.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What each body in the masks lets through, per leg it crosses. Defaults to opaque; given
        without ``shadows``, refused.
    pair_limit : int, optional
        Receiver-by-source-by-aperture-facet triples one pass may form.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If a mirror names a facet the surface set does not have, or a mask is for other
        receivers, other sources or another number of mirrors.
    """
    return summed_mirrored_fluence_rate(
        (surfaces,),
        mirrors,
        points,
        absorption=absorption,
        shadows=shadows,
        transmittance=transmittance,
        pair_limit=pair_limit,
    )


def summed_mirrored_fluence_rate(
    sets,
    mirrors: Sequence[Mirror],
    points,
    *,
    absorption: Absorption | None = None,
    shadows=None,
    transmittance=None,
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
    mirrors, points, absorption, shadows, transmittance, pair_limit
        As for :func:`mirrored_fluence_rate`.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``

    Raises
    ------
    ValueError
        If no set is given, the sets do not share their geometry, a mirror names a facet the
        set does not have, or a mask does not fit.
    """
    return _mirrored(
        sets,
        mirrors,
        points,
        None,
        absorption,
        pair_limit,
        point_sources_only=False,
        shadows=shadows,
        transmittance=transmittance,
    )


def mirrored_irradiance(
    surfaces: Surfaces,
    mirrors: Sequence[Mirror],
    points,
    normals,
    *,
    absorption: Absorption | None = None,
    shadows=None,
    transmittance=None,
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
    surfaces, mirrors, absorption, shadows, transmittance, pair_limit
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
        shadows=shadows,
        transmittance=transmittance,
    )


def _mirrored(
    sets,
    mirrors,
    points,
    normals,
    absorption,
    pair_limit,
    *,
    point_sources_only,
    shadows=None,
    transmittance=None,
):
    """The mirrored field of several sets on one geometry: fluence rate, or irradiance given normals.

    With ``shadows``, one :class:`~aquaflux.radiation.mirror_visibility.MirrorVisibility` per
    mirror, which receivers and sources each mirror involves is read off its mask, which was built
    from concrete geometry -- so it is known even where the sets' vertices are traced.
    """
    sets = tuple(sets)
    geometry = _one_geometry(sets)
    masks = _shadow_masks(shadows, mirrors, points, transmittance)
    # Read before conversion: inside a trace a concrete array becomes a tracer once it passes
    # through jnp, and which receivers lie in front of a mirror is only worth knowing when it can
    # be read.
    readable = not any(isinstance(array, jax.core.Tracer) for array in (points, geometry.vertices))
    host_points = np.asarray(points, dtype=float) if readable else None
    points = jnp.asarray(points, dtype=float)
    total = jnp.zeros(points.shape[0])
    for mirror, mask in zip(mirrors, masks, strict=True):
        facets = np.asarray(mirror.facets)
        if facets.size and (facets.min() < 0 or facets.max() >= geometry.n_facets):
            msg = (
                f"a mirror names facets {facets.min()} to {facets.max()}, outside this surface "
                f"set's {geometry.n_facets}; it was found in a different set"
            )
            raise ValueError(msg)
        path = _Path.through(mirror, geometry, absorption)
        if mask is None:
            receivers = _in_front(mirror, host_points, points.shape[0], readable)
            sources = _sources_in_front(mirror, geometry, readable)
        else:
            receivers = mirror.in_front(mask.receivers)
            sources = mask.sources
        if not receivers.size:
            continue
        at = (points[receivers], None if normals is None else normals[receivers])
        shadow = _MaskRows.of(mask, receivers, transmittance)
        images = tuple(mirror.image(surfaces) for surfaces in sets)
        received = _point_images(sets, images, sources, path, at, shadow, pair_limit)
        if not point_sources_only:
            received = received + _areal_images(
                sets, images, sources, path, at, shadow, pair_limit, cull=readable
            )
        total = total.at[receivers].add(received)
    return total


def _shadow_masks(shadows, mirrors, points, transmittance) -> tuple:
    """One mask per mirror, checked against the receivers, or ``None`` for each if unshadowed."""
    if shadows is None:
        if transmittance is not None:
            msg = "transmittance was given without mirror visibility masks to apply it to"
            raise ValueError(msg)
        return (None,) * len(mirrors)
    shadows = tuple(shadows)
    if len(shadows) != len(mirrors):
        msg = f"{len(shadows)} mirror visibility masks for {len(mirrors)} mirrors; give one each"
        raise ValueError(msg)
    return tuple(mask.for_receivers(points) for mask in shadows)


class _Shadow(eqx.Module):
    """A mirror's mask for one group of receivers and sources, and the bodies' transmittance.

    A pytree, so a compiled gather takes it as an argument and the transmittance stays live.
    ``column`` maps a facet to its column of the mask where pairs are read by index; it is
    ``None`` where the columns are already the sources gathered, in their order.
    """

    crossings: jnp.ndarray
    hidden: jnp.ndarray | None
    transmittance: jnp.ndarray
    column: jnp.ndarray | None = None

    def surviving(self, crossings, hidden) -> jnp.ndarray:
        """The surviving fraction from one chunk's rows of :attr:`crossings` and :attr:`hidden`."""
        return reflected_surviving(crossings, hidden, self.transmittance)

    def at(self, rows, facets) -> jnp.ndarray:
        """The surviving fraction of the paths from ``facets`` to the receivers ``rows``.

        ``rows`` index the receivers the mask's rows were cut to; ``facets`` index the surface
        set, read through :attr:`column`. The two broadcast against each other.
        """
        column = jnp.take(self.column, facets, mode="clip")
        hidden = None if self.hidden is None else self.hidden[rows, column]
        return reflected_surviving(self.crossings[:, rows, column], hidden, self.transmittance)


class _MaskRows:
    """One mirror's mask, cut down to the receivers in front of it; columns are cut per group."""

    def __init__(self, mask: MirrorVisibility, receivers, transmittance):
        rows = jnp.asarray(receivers)
        self.mask = mask
        self.crossings = jnp.take(mask.crossings, rows, axis=1)
        self.hidden = None if mask.hidden is None else jnp.take(mask.hidden, rows, axis=0)
        self.transmittance = (
            jnp.zeros(mask.n_occluders)
            if transmittance is None
            else jnp.broadcast_to(jnp.asarray(transmittance, dtype=float), (mask.n_occluders,))
        )

    @classmethod
    def of(cls, mask: MirrorVisibility | None, receivers, transmittance) -> _MaskRows | None:
        """``mask``'s rows for ``receivers``, or ``None`` where nothing is masked."""
        return None if mask is None else cls(mask, receivers, transmittance)

    def with_lookup(self, n_facets: int) -> _Shadow:
        """Every column, read by facet index through a lookup of ``n_facets`` entries."""
        column = np.zeros(n_facets, dtype=int)
        column[self.mask.sources] = np.arange(len(self.mask.sources))
        return _Shadow(
            crossings=self.crossings,
            hidden=self.hidden,
            transmittance=self.transmittance,
            column=jnp.asarray(column),
        )

    def for_sources(self, facets) -> _Shadow:
        """The shadow of the paths from ``facets``, columns in their order."""
        position = jnp.asarray(self.mask.columns(facets))
        return _Shadow(
            crossings=jnp.take(self.crossings, position, axis=2),
            hidden=None if self.hidden is None else jnp.take(self.hidden, position, axis=1),
            transmittance=self.transmittance,
        )


class _Path(eqx.Module):
    """What every path through one mirror shares: its plane, its aperture and the medium.

    A pytree, so a compiled gather takes it as an argument and the reflectances and the medium
    stay live in it.
    """

    mirror: Mirror
    aperture: jnp.ndarray
    reflectance: jnp.ndarray
    absorption: Absorption | None

    @classmethod
    def through(cls, mirror: Mirror, geometry: Surfaces, absorption) -> _Path:
        """The path through ``mirror``, whose facets index ``geometry``."""
        facets = jnp.asarray(mirror.facets)
        return cls(
            mirror=mirror,
            aperture=jnp.take(geometry.vertices, facets, axis=0),
            reflectance=jnp.take(jnp.asarray(geometry.specular_reflectance, dtype=float), facets),
            absorption=absorption,
        )

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

    Leaving a point on or behind the plane in would cost work and change nothing, since the clip
    keeps none of an image seen from there.
    """
    return mirror.in_front(host_points) if readable else np.arange(n_points)


def _sources_in_front(mirror: Mirror, geometry: Surfaces, readable: bool) -> np.ndarray:
    """Facets with any part strictly in front of the mirror, or every facet if unreadable."""
    return mirror.sources_in_front(geometry) if readable else np.arange(geometry.n_facets)


def _passes(receivers, pair_limit, per_receiver, body, shadow: _Shadow | None = None):
    """:func:`~aquaflux.radiation.work.in_passes` over receivers, with their normals and shadow.

    ``receivers`` is a ``(points, normals)`` pair, ``normals`` ``None`` for points in the volume.
    ``body`` takes a chunk of the points, of the normals, and the chunk's surviving fraction --
    one where nothing is masked -- cut from ``shadow`` alongside them.
    """
    points, normals = receivers
    arrays = [(points, 0)]
    if normals is not None:
        arrays.append((normals, 0))
    if shadow is not None:
        arrays.append((shadow.crossings, 1))
        if shadow.hidden is not None:
            arrays.append((shadow.hidden, 0))

    def run(*chunks):
        chunks = list(chunks)
        chunk = chunks.pop(0)
        chunk_normals = chunks.pop(0) if normals is not None else None
        surviving = 1.0
        if shadow is not None:
            crossings = chunks.pop(0)
            hidden = chunks.pop(0) if shadow.hidden is not None else None
            surviving = shadow.surviving(crossings, hidden)
        return body(chunk, chunk_normals, surviving)

    return in_passes(arrays, pair_limit, per_receiver, run)


def _areal_images(sets, images, sources, path: _Path, receivers, shadow, pair_limit, *, cull):
    """What the areal facets' images send each receiver through the aperture, ``(n,)``.

    Laid out as the direct gather lays out its facets (:func:`~aquaflux.radiation.gather.areal_layout`):
    receivers in small blocks, each against the images that can light it. An image is an ordinary
    surface set, so a block lying wholly behind an image's plane, where a profile dark behind
    itself sends nothing, leaves the image off its list exactly as it would the facet.

    ``cull`` clips each image only against the aperture facets it overlaps; it is on wherever the
    geometry can be read, which is where no derivative with respect to it is being taken.
    """
    points, normals = receivers
    image_geometry = images[0]
    groups = [
        dataclasses.replace(group, facets=np.intersect1d(group.facets, sources))
        for group in _areal_groups(images)
    ]
    groups = [group for group in groups if group.facets.size]
    total = jnp.zeros(points.shape[0])
    if not groups:
        return total
    layout = areal_layout(points, image_geometry, groups)
    emission = tuple(surfaces.emission for surfaces in sets)
    lookup = None if shadow is None else shadow.with_lookup(image_geometry.n_facets)
    for group, segments in zip(groups, layout, strict=True):
        for segment in segments:
            total = total + _segment_through(
                path,
                image_geometry.vertices,
                image_geometry.centroid,
                image_geometry.normal,
                emission,
                group.profiles,
                points,
                normals,
                lookup,
                segment,
                pair_limit=pair_limit,
                cull=cull,
            )
    return total


@eqx.filter_jit
def _segment_through(
    path,
    vertices,
    centroid,
    normal,
    emission,
    profiles,
    points,
    normals,
    shadow,
    segment,
    *,
    pair_limit,
    cull,
):
    """What one segment's blocks receive from their listed images, ``(n_points,)``.

    As :func:`~aquaflux.radiation.gather._segment_fluence` gathers a segment of facets: each block
    against its own list, so nothing is formed for a pair off the list; entries past a list and
    padding receivers contribute exactly zero. Module level and compiled, taking everything that
    varies as an argument, so the next call of a sweep reuses the program. ``cull`` is
    :func:`_seen_pairs`'s, and must be off where the geometry is differentiated.
    """
    n_points = points.shape[0]
    rows = jnp.asarray(segment.rows)
    lists = (jnp.asarray(segment.facets), jnp.asarray(segment.valid))

    def at(block_rows, *block_lists):
        block_facets, block_valid = lists if segment.shared else block_lists
        row = jnp.minimum(block_rows, n_points - 1)
        grid = (*row.shape, block_facets.shape[-1])
        receivers = jnp.take(points, row, axis=0)[:, :, None, :]
        image = jnp.take(vertices, block_facets, axis=0, mode="clip")[:, None]
        image_centroid = jnp.take(centroid, block_facets, axis=0, mode="clip")[:, None]
        image_normal = jnp.take(normal, block_facets, axis=0, mode="clip")[:, None]
        seen = _seen_pairs(
            path.aperture,
            path.reflectance,
            jnp.broadcast_to(receivers, (*grid, 3)).reshape(-1, 3),
            None
            if normals is None
            else jnp.broadcast_to(
                jnp.take(normals, row, axis=0)[:, :, None, :], (*grid, 3)
            ).reshape(-1, 3),
            jnp.broadcast_to(image, (*grid, 3, 3)).reshape(-1, 3, 3),
            cull=cull,
        ).reshape(grid)
        direction, _ = _emitter_direction(image_centroid, receivers)
        weight = seen * path.transmittance(image_centroid, receivers)
        if shadow is not None:
            weight = weight * shadow.at(row[:, :, None], block_facets[:, None, :])
        weight = jnp.where(block_valid[:, None, :], weight, 0.0)
        total = jnp.zeros(row.shape)
        for flux, profile in zip(emission, profiles, strict=True):
            exitance = jnp.take(flux, block_facets, axis=0, mode="clip")[:, None, :]
            radiance = profile.radiance_per_exitance(direction, image_normal)
            total = total + jnp.sum(exitance * radiance * weight, axis=2)
        return total

    per_block = segment.block * segment.width * path.aperture.shape[0]
    arrays = ((rows, 0),) if segment.shared else ((rows, 0), (lists[0], 0), (lists[1], 0))
    received = in_passes(arrays, pair_limit, per_block, at)
    return jnp.zeros(n_points).at[rows.ravel()].add(received.ravel(), mode="drop")


#: Triples one step of the culled clip forms -- a receiving point, an image and an aperture facet
#: whose cones can overlap. The survivors of a pass are clipped this many at a time.
_CLIP_BATCH = 4096


def _seen_through_aperture(aperture, weights, receivers, normals, vertices, *, cull: bool = False):
    """:func:`_seen_pairs` for every receiver against every image.

    Parameters
    ----------
    aperture : jnp.ndarray, shape ``(a, 3, 3)``
    weights : jnp.ndarray, shape ``(a,)``
    receivers : jnp.ndarray, shape ``(r, 3)``
    normals : jnp.ndarray, shape ``(r, 3)``, or None
    vertices : jnp.ndarray, shape ``(s, 3, 3)``
    cull : bool, optional

    Returns
    -------
    jnp.ndarray, shape ``(r, s)``
    """
    grid = (receivers.shape[0], vertices.shape[0])
    seen = _seen_pairs(
        aperture,
        weights,
        jnp.broadcast_to(receivers[:, None, :], (*grid, 3)).reshape(-1, 3),
        None
        if normals is None
        else jnp.broadcast_to(normals[:, None, :], (*grid, 3)).reshape(-1, 3),
        jnp.broadcast_to(vertices[None], (*grid, 3, 3)).reshape(-1, 3, 3),
        cull=cull,
    )
    return seen.reshape(grid)


def _seen_pairs(aperture, weights, receivers, normals, vertices, *, cull: bool):
    """Solid angle of each pair's image seen through the aperture, weighted per aperture facet.

    The plain solid angle for a point in the volume, the projected one for a point with a normal.

    **With ``cull``, an image is clipped only against the aperture facets it can be seen
    through.** Seen from a receiver, an aperture facet whose cone of directions cannot overlap
    the image's shows none of it, so its clip would return zero; the cone test is the one the
    silhouette clip uses to drop blockers, conservative in the same direction, so the answer is
    the same. Without it every image is clipped against every aperture facet, and the cost is the
    aperture's facet count times a direct gather; with it, the facets an image actually overlaps
    -- a handful, measured on a reactor's end plates, however finely they are meshed.

    The survivors are found and clipped inside a compiled loop whose length follows their count,
    which reverse mode cannot differentiate through. So the culled clip is taken of geometry held
    constant: the aperture, the receivers and the images are cut from any derivative, while the
    weights -- the live reflectances -- multiply its result outside the loop and keep theirs.
    Where a derivative with respect to the geometry is wanted, ``cull`` must be off.

    Parameters
    ----------
    aperture : jnp.ndarray, shape ``(a, 3, 3)``
        The mirror's facets.
    weights : jnp.ndarray, shape ``(a,)``
        What each aperture facet's share is weighted by: its specular reflectance, or one.
    receivers : jnp.ndarray, shape ``(p, 3)``
        Each pair's receiver.
    normals : jnp.ndarray, shape ``(p, 3)``, or None
    vertices : jnp.ndarray, shape ``(p, 3, 3)``
        Each pair's image.
    cull : bool
        Clip each image only against the aperture facets whose cones overlap it.

    Returns
    -------
    jnp.ndarray, shape ``(p,)``
        ``sum over aperture facets a of weights_a * (solid angle of the image within a's cone)``.
    """
    if cull:
        aperture, receivers, vertices = (
            jax.lax.stop_gradient(array) for array in (aperture, receivers, vertices)
        )
        normals = None if normals is None else jax.lax.stop_gradient(normals)
    view = source_view(receivers, normals, vertices)
    if cull:
        fraction = _culled_fractions(view, aperture, receivers, normals, vertices)
        return jnp.abs(view.whole) * (fraction @ weights)
    # One view per pair, shared by every aperture facet it is clipped against. The clip stacks
    # its candidates, so everything is spread to one batch shape rather than left to broadcast.
    batch = (receivers.shape[0], aperture.shape[0])

    def spread(array, trailing):
        return jnp.broadcast_to(array[:, None], (*batch, *trailing))

    spread_view = SourceView(
        loop=spread(view.loop, view.loop.shape[1:]),
        whole=spread(view.whole, ()),
        support=spread(view.support, (3,)),
        through=spread(view.through, (3,)),
    )
    to_aperture = jnp.broadcast_to(aperture[None] - receivers[:, None, None, :], (*batch, 3, 3))
    fraction, _ = covered_by(
        spread_view, None if normals is None else spread(normals, (3,)), to_aperture
    )
    return jnp.abs(view.whole) * (fraction @ weights)


def _may_show(aperture, receivers, vertices):
    """Which aperture facets may show part of each pair's image, seen from its receiver.

    Conservatively: ``False`` only where the facet's cone of directions and the image's cannot
    overlap, so the facet shows none of the image; ``True`` may still clip to nothing. A cone too
    wide to bound anything -- a receiver near the facet's or the image's own plane -- overlaps
    everything.

    Parameters
    ----------
    aperture : jnp.ndarray, shape ``(a, 3, 3)``
    receivers : jnp.ndarray, shape ``(p, 3)``
    vertices : jnp.ndarray, shape ``(p, 3, 3)``

    Returns
    -------
    jnp.ndarray of bool, shape ``(p, a)``
    """
    image_cone = angular_cone(vertices - receivers[:, None, :])
    facet_cone = angular_cone(aperture[None] - receivers[:, None, None, :])
    return cones_may_overlap(tuple(part[:, None] for part in image_cone), facet_cone)


def _culled_fractions(view: SourceView, aperture, receivers, normals, vertices):
    """Each pair's image share within each aperture facet's cone, clipping only where cones overlap.

    Returns
    -------
    jnp.ndarray, shape ``(p, a)``
        Zero for every triple the cone test rules out.
    """
    n_pairs, n_facets = receivers.shape[0], aperture.shape[0]
    kept = _may_show(aperture, receivers, vertices).reshape(-1)
    n_triples = kept.shape[0]
    width = min(_CLIP_BATCH, n_triples)
    survivors = jnp.sum(kept)
    # The survivors' flat indices first, padded to a whole number of batches.
    (order,) = jnp.nonzero(kept, size=-(-n_triples // width) * width, fill_value=0)

    def clip(state):
        step, fraction = state
        start = step * width
        triple = jax.lax.dynamic_slice(order, (start,), (width,))
        valid = start + jnp.arange(width) < survivors
        pair, facet = jnp.divmod(triple, n_facets)
        share, _ = covered_by(
            SourceView(
                loop=view.loop[pair],
                whole=view.whole[pair],
                support=view.support[pair],
                through=view.through[pair],
            ),
            None if normals is None else normals[pair],
            aperture[facet] - receivers[pair][:, None, :],
        )
        return step + 1, fraction.at[triple].add(jnp.where(valid, share, 0.0))

    _, fraction = jax.lax.while_loop(
        lambda state: state[0] * width < survivors, clip, (0, jnp.zeros(n_triples))
    )
    return fraction.reshape(n_pairs, n_facets)


def _point_images(sets, images, sources, path: _Path, receivers, shadow, pair_limit):
    """What the point sources' images send each receiver through the aperture, ``(n,)``."""
    image_geometry = images[0]
    point_all = np.intersect1d(np.flatnonzero(image_geometry.is_point_source), sources)
    if not point_all.size:
        return jnp.zeros(receivers[0].shape[0])
    index = jnp.asarray(point_all)
    position = jnp.take(image_geometry.centroid, index, axis=0)
    plans = tuple(
        tuple(
            (
                profile,
                tuple(int(c) for c in np.searchsorted(point_all, np.intersect1d(point, point_all))),
            )
            for profile, _, point in _groups(surfaces)
        )
        for surfaces in images
    )
    power = [jnp.take(surfaces.power, index) for surfaces in sets]

    return _points_through(
        path,
        position,
        tuple(power),
        plans,
        *receivers,
        None if shadow is None else shadow.for_sources(point_all),
        pair_limit=pair_limit,
    )


@eqx.filter_jit
def _points_through(path, position, power, plans, points, normals, shadow, *, pair_limit):
    """What the point sources' images send each receiver, compiled once per shape and plan.

    ``plans`` holds, per set, each point-source profile with the columns it applies to as a tuple
    of ints, which is static: it decides which columns the program reads.
    """

    def at(chunk, chunk_normals, surviving):
        reflectance = _crossed_reflectance(path, chunk, position)
        direction, distance_squared = _emitter_direction(position[None], chunk[:, None])
        weight = reflectance * path.transmittance(position[None], chunk[:, None]) * surviving
        weight = weight / jnp.where(distance_squared == 0.0, 1.0, distance_squared)
        if chunk_normals is not None:
            # The light arrives travelling along `direction`, so it strikes a surface facing back
            # along it; nothing is received from behind.
            weight = weight * jnp.maximum(-dot(direction, chunk_normals[:, None, :]), 0.0)
        received = jnp.zeros(chunk.shape[0])
        for flux, plan in zip(power, plans, strict=True):
            for profile, columns in plan:
                if not columns:
                    continue
                pick = jnp.asarray(columns)
                intensity = profile.intensity_fraction(
                    jnp.take(direction, pick, axis=1), jnp.zeros(3)
                )
                received = received + jnp.sum(
                    jnp.take(flux, pick) * intensity * jnp.take(weight, pick, axis=1), axis=1
                )
        return received

    per_receiver = position.shape[0] * path.aperture.shape[0]
    return _passes((points, normals), pair_limit, per_receiver, at, shadow)


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
    image = mirror.image(geometry)
    index = jnp.asarray(columns)
    image_vertices = jnp.take(image.vertices, index, axis=0)
    image_centroid = jnp.take(image.centroid, index, axis=0)
    image_normal = jnp.take(image.normal, index, axis=0)
    weight = jnp.asarray(weight, dtype=float)

    geometric = jnp.zeros((n, n))
    separation = jnp.zeros((n, n))
    source_cosine = jnp.zeros((n, n))
    if rows.size:
        row_index = jnp.asarray(rows)
        block = _exchange_rows(
            aperture,
            image_vertices,
            jnp.take(jnp.asarray(sample), row_index, axis=0),
            jnp.take(geometry.normal, row_index, axis=0),
            weight,
            pair_limit=pair_limit,
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


@functools.partial(jax.jit, static_argnames="pair_limit")
def _exchange_rows(aperture, image_vertices, sample, normals, weight, *, pair_limit):
    """The projected solid angle of each image through the aperture, over ``pi``, per row.

    Module level and compiled, so the planes of one build -- and of the next -- that share their
    shapes share one program rather than each being traced afresh.

    Parameters
    ----------
    aperture : jnp.ndarray, shape ``(a, 3, 3)``
    image_vertices : jnp.ndarray, shape ``(s, 3, 3)``
    sample : jnp.ndarray, shape ``(r, q, 3)``
        Quadrature points on the receiving facets.
    normals : jnp.ndarray, shape ``(r, 3)``
    weight : jnp.ndarray, shape ``(q,)``

    Returns
    -------
    jnp.ndarray, shape ``(r, s)``
    """
    ones = jnp.ones(aperture.shape[0])

    def at(points, chunk_normals):
        # Scanned over the quadrature points rather than unrolled, so the clip is compiled once
        # however many points the rule has.
        def accumulate(total, sampled):
            weight_k, points_k = sampled
            seen = _seen_through_aperture(
                aperture, ones, points_k, chunk_normals, image_vertices, cull=True
            )
            return total + weight_k * seen, None

        total, _ = jax.lax.scan(
            accumulate,
            jnp.zeros((points.shape[0], image_vertices.shape[0])),
            (weight, jnp.swapaxes(points, 0, 1)),
        )
        return total / jnp.pi

    return in_passes(
        ((sample, 0), (normals, 0)),
        pair_limit,
        image_vertices.shape[0] * aperture.shape[0],
        at,
    )
