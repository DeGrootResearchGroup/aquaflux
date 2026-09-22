"""Summing every source's contribution at every receiver — the backward gather.

At each receiver the module adds up what every emitting facet and every point source delivers
there. It is a *backward* gather because it starts at the receiver and looks toward the
sources, the opposite of tracing photons forward, and it is deterministic: the answer at a
point is a sum, not a sample, so it carries neither stochastic noise nor the bias that comes
from scoring a photon's path length through a finite cell.

Each term may be attenuated by the water it crosses, through an ``Absorption`` supplied by the
caller; with none, the field is the vacuum one, which is what every analytic reference case is
stated in. Intervening bodies multiply the same terms by a surviving fraction, supplied as a
pre-built visibility mask together with the live transmittance of each body.

**Two quantities, two kernels, and the difference is not a convention.** The fluence rate
``G`` counts power arriving from every direction with no regard to which, because the thing it
governs — a microbe tumbling in a flow — has no orientation. The irradiance ``E`` weights each
direction by the cosine of its angle to a receiving surface, because an oblique beam spreads
over more of that surface. So ``G`` uses the plain solid angle and ``E`` the projected one, and
no scalar converts one into the other.

Facets are visited in groups of equal angular distribution. The grouping is done on the host
before anything is traced, so each group's distribution is resolved once while the program is
being built and the compiled code contains no test of which kind a facet is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility
from aquaflux.vectors import dot

__all__ = ["direct_fluence_rate", "direct_irradiance"]

_DEFAULT_CHUNK = 4096


def _groups(surfaces: Surfaces) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """Partition facets by angular distribution, and within that by areal versus point.

    Done in numpy, on the host, at trace time: the sizes are then compile-time constants and
    each group's profile is a concrete object whose methods inline, so the traced program holds
    no branch on facet kind and no gather through a profile table.
    """
    if isinstance(surfaces.profile_index, jax.core.Tracer):
        msg = (
            "the surface set's profile index must be concrete here: which angular distribution "
            "each facet emits with decides the shape of the traced program, so it cannot itself "
            "be traced. Close over the surface set and pass only the values that vary -- "
            "jit(lambda emission: direct_fluence_rate(surfaces.with_optics(emission=emission), "
            "points)) -- rather than passing the whole set as an argument. Vertices may be "
            "traced: "
            "substitute them with Surfaces.with_geometry, which keeps the labels."
        )
        raise TypeError(msg)
    index = np.asarray(surfaces.profile_index)
    # The kind of a source is read from its label and never from its area. The two agree, but
    # the area is a quantity a gradient may flow through, and reading a code path off a traced
    # quantity is what would stop a lamp from being able to move.
    is_point = surfaces.is_point_source
    partition = []
    for kind, profile in enumerate(surfaces.profiles):
        selected = index == kind
        areal = np.flatnonzero(selected & ~is_point)
        point = np.flatnonzero(selected & is_point)
        if len(areal) or len(point):
            partition.append((profile, areal, point))
    return partition


def _chunked(arrays, chunk_size: int, body):
    """Apply ``body`` to the receivers in fixed-size chunks and concatenate the results.

    Every array in ``arrays`` is indexed by receiver and is cut the same way, so a per-receiver
    quantity computed outside — a row of the visibility mask, a receiving surface's normal —
    stays lined up with its point without the body having to index anything itself.

    The receiver-by-source product is the module's whole cost and would be the whole of its
    memory too if it were formed at once: a hundred thousand cells against a thousand facets is
    a hundred million entries per intermediate. Chunking keeps the working set proportional to
    the chunk rather than to the problem, at no cost in arithmetic. The last chunk is padded
    rather than made smaller, so the traced body is compiled once.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be at least 1 receiver; got {chunk_size}"
        raise ValueError(msg)
    arrays = [jnp.asarray(array) for array in arrays]
    n_points = arrays[0].shape[0]
    if n_points == 0:
        return jnp.zeros(0)
    chunk_size = min(chunk_size, n_points)
    n_chunks = -(-n_points // chunk_size)
    padding = n_chunks * chunk_size - n_points
    shaped = [
        jnp.concatenate([array, jnp.repeat(array[-1:], padding, axis=0)]).reshape(
            n_chunks, chunk_size, *array.shape[1:]
        )
        for array in arrays
    ]
    _, out = lax.scan(lambda carry, chunk: (carry, body(*chunk)), None, tuple(shaped))
    return out.reshape(-1)[:n_points]


def _surviving_rows(visibility, transmittance, points, n_facets) -> jnp.ndarray:
    """Per-receiver rows of the fraction getting past the intervening bodies.

    Returns a full ``(n_receivers, n_facets)`` array even when nothing occludes, so that the
    chunking and the gather have one shape to deal with rather than two code paths.
    """
    if visibility is None:
        if transmittance is not None:
            msg = "transmittance was given without a visibility mask to apply it to"
            raise ValueError(msg)
        return jnp.ones((jnp.asarray(points).shape[0], n_facets))
    if not isinstance(visibility, Visibility):
        msg = f"visibility must be a Visibility; got {type(visibility).__name__}"
        raise TypeError(msg)
    visibility.for_receivers(points)
    if transmittance is None:
        transmittance = jnp.zeros(visibility.n_occluders)
    return visibility.surviving(transmittance)


def _transmittance(absorption, source: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    """Surviving fraction along each source-to-receiver segment, or one in vacuum.

    ⚠️ **The path is taken from the facet's centroid**, so a facet large enough for its far
    corner to sit at a noticeably different optical depth is attenuated as though it did not.
    That is the field's standard per-segment treatment, and it is another reason the refinement
    criterion exists: the facet width that makes the emission assumption hold makes this one
    hold too.

    There is no clamp on the optical depth. In double precision ``exp(-tau)`` reaches zero near
    ``tau = 745``, where zero is the right answer and is what is returned; a clamp would buy
    nothing and would flatten the sensitivity to absorbance across a whole region.
    """
    if absorption is None:
        return jnp.asarray(1.0)
    return jnp.exp(-absorption.optical_depth(source[None, :, :], receivers[:, None, :]))


def _emitter_cosine(surfaces: Surfaces, facets: np.ndarray, receivers: jnp.ndarray):
    """Cosine at each emitting facet of the angle to each receiver, and the separation.

    Shapes are ``(n_receivers, n_facets)``; the cosine is measured at the *source*, between its
    outward normal and the direction to the receiver, which is what an angular distribution is
    a function of.
    """
    centroid = jnp.take(surfaces.centroid, facets, axis=0)
    normal = jnp.take(surfaces.normal, facets, axis=0)
    offset = receivers[:, None, :] - centroid[None, :, :]
    distance_squared = dot(offset, offset)
    distance = jnp.sqrt(jnp.where(distance_squared == 0.0, 1.0, distance_squared))
    return dot(offset, normal[None, :, :]) / distance, distance_squared


def _streamed(
    surfaces, points, *, occluders, self_occlusion, absorption, transmittance, chunk_size
):
    """Gather chunk by chunk, building each chunk's visibility and letting it go.

    The chunk loop is on the host rather than traced, because building a mask is host work --
    it compacts candidate pairs and refuses receivers inside a body -- and a traced loop could
    not call it. Each chunk's gather is the ordinary traced one.
    """
    points = jnp.asarray(points, dtype=float)
    if chunk_size < 1:
        msg = f"chunk_size must be at least 1 receiver; got {chunk_size}"
        raise ValueError(msg)
    if points.shape[0] == 0:
        return jnp.zeros(0)
    pieces = []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        mask = build_visibility(occluders, surfaces, chunk, self_occlusion=self_occlusion)
        pieces.append(
            direct_fluence_rate(
                surfaces,
                chunk,
                absorption=absorption,
                visibility=mask,
                transmittance=transmittance,
                chunk_size=chunk_size,
            )
        )
        del mask
    return jnp.concatenate(pieces)


def direct_fluence_rate(
    surfaces: Surfaces,
    points,
    *,
    absorption: Absorption | None = None,
    visibility: Visibility | None = None,
    occluders=None,
    self_occlusion=None,
    transmittance=None,
    chunk_size: int = _DEFAULT_CHUNK,
):
    """Fluence rate at each receiver point, in vacuum.

    The zeroth angular moment of radiance over the whole sphere: the radiant power crossing a
    point from every direction, per unit area, in W/m². It carries **no receiver cosine** — see
    :func:`direct_irradiance` for the quantity that does.

    An areal facet contributes its radiance times the solid angle it subtends, exactly, with the
    solid angle in closed form rather than approximated by an inverse square. A facet whose
    outward normal points away from the receiver contributes nothing: for a convex emitting body
    that clamp *is* the visibility test, and it is exact.

    A point source contributes ``P f(omega) / r^2``.

    Parameters
    ----------
    surfaces : Surfaces
        The emitting set. Facets with zero area are point sources and carry radiant power.
    points : array_like, shape ``(n_points, 3)``
        Receiver positions — cell centres, probes, anywhere.
    absorption : Absorption, optional
        The absorbing medium between the sources and the receivers.
    visibility : Visibility, optional
        Which bodies lie between which sources and which receivers, built once for these exact
        receiver positions and checked against them here. Mutually exclusive with ``occluders``
        and ``self_occlusion``.
    occluders : sequence of Occluder, optional
        The bodies themselves, instead of a mask built from them. Each chunk's mask is then
        built here and **dropped when that chunk is done**, so peak memory is set by the chunk
        rather than by the receiver count -- which is what makes a mesh-scale field computable
        at all: a mask over a million cells and a few thousand facets is tens of gigabytes per
        body, while the chunks are a few hundred megabytes. An empty sequence is meaningful:
        it streams the emitting surface's own shadowing with no other body present.
    self_occlusion : SelfOcclusion, optional
        How the surface shadows itself while streaming, as in
        :func:`~aquaflux.radiation.visibility.build_visibility`. Passing it implies streaming.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What fraction each body lets through, in ``[0, 1]``. Differentiable, and defaulting to
        zero -- opaque -- so that a mask supplied without one blocks rather than passes.
    chunk_size : int, optional
        Receivers per traced chunk, and per streamed mask. Trades peak memory against nothing;
        the arithmetic is the same either way.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If ``chunk_size`` is less than one receiver, if the visibility mask was built for a
        different set of receivers than the points given here, or if a mask is given together
        with the bodies to build one from.

    Notes
    -----
    Streaming rebuilds each chunk's mask on every call, so a study that sweeps optics over one
    frozen scene is better served by building the mask once -- which is what
    :func:`~aquaflux.radiation.model.build_radiation_model` does, and why the frozen mask is
    what carries the derivative with respect to a body's transmittance.
    """
    if occluders is not None or self_occlusion is not None:
        if visibility is not None:
            msg = (
                "give either a built visibility mask or the bodies to build one from, not both: "
                "with both, the mask that decides the shadows would be silently the one built "
                "here from `occluders`, and the one passed in would do nothing."
            )
            raise ValueError(msg)
        return _streamed(
            surfaces,
            points,
            occluders=() if occluders is None else occluders,
            self_occlusion=self_occlusion,
            absorption=absorption,
            transmittance=transmittance,
            chunk_size=chunk_size,
        )
    points = jnp.asarray(points, dtype=float)
    partition = _groups(surfaces)
    surviving_rows = _surviving_rows(visibility, transmittance, points, surfaces.n_facets)

    def at(receivers, surviving_all):
        total = jnp.zeros(receivers.shape[0])
        for profile, areal, point in partition:
            if len(areal):
                cosine, _ = _emitter_cosine(surfaces, areal, receivers)
                radiance = jnp.take(surfaces.emission, areal) * profile.radiance_per_exitance(
                    cosine
                )
                omega = solid_angle(
                    receivers[:, None, :], jnp.take(surfaces.vertices, areal, axis=0)[None, ...]
                )
                surviving = _transmittance(
                    absorption, jnp.take(surfaces.centroid, areal, axis=0), receivers
                ) * jnp.take(surviving_all, areal, axis=1)
                total = total + jnp.sum(radiance * omega * surviving, axis=1)
            if len(point):
                cosine, distance_squared = _emitter_cosine(surfaces, point, receivers)
                fraction = profile.intensity_fraction(cosine)
                surviving = _transmittance(
                    absorption, jnp.take(surfaces.centroid, point, axis=0), receivers
                ) * jnp.take(surviving_all, point, axis=1)
                total = total + jnp.sum(
                    jnp.take(surfaces.power, point) * fraction * surviving / distance_squared,
                    axis=1,
                )
        return total

    return _chunked((points, surviving_rows), chunk_size, at)


def direct_irradiance(
    surfaces: Surfaces,
    points,
    normals,
    *,
    absorption: Absorption | None = None,
    visibility: Visibility | None = None,
    transmittance=None,
    chunk_size: int = _DEFAULT_CHUNK,
):
    """Irradiance on an oriented receiving surface at each point, in vacuum.

    The first angular moment of radiance over the receiver's hemisphere: power per unit area of
    a surface facing a given way, in W/m². Directions arriving obliquely count for less, which
    is the whole difference from :func:`direct_fluence_rate`.

    ⚠️ **``E = G cos(theta)`` is a single-source identity**, true for one point source and a
    receiver facing it. It fails for several sources at once, which is exactly when this
    function is needed: a receiver in an isotropic field has ``G = 4 pi L`` and ``E = pi L``.

    Parameters
    ----------
    surfaces : Surfaces
        The emitting set.
    points : array_like, shape ``(n_points, 3)``
        Receiver positions.
    normals : array_like, shape ``(n_points, 3)``
        Unit outward normal of the receiving surface at each point.
    absorption : Absorption, optional
        The absorbing medium between the sources and the receivers.
    visibility : Visibility, optional
        Which bodies lie between which sources and which receivers, built once for these exact
        receiver positions and checked against them here.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What fraction each body lets through, in ``[0, 1]``. Differentiable, and defaulting to
        zero -- opaque -- so that a mask supplied without one blocks rather than passes.
    chunk_size : int, optional
        Receivers per traced chunk.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Irradiance in W/m².

    Raises
    ------
    ValueError
        If ``normals`` and ``points`` disagree in shape, if ``chunk_size`` is less than one
        receiver, or if the visibility mask was built for a different set of receivers.
    """
    points = jnp.asarray(points, dtype=float)
    normals = jnp.asarray(normals, dtype=float)
    if normals.shape != points.shape:
        msg = f"normals must match points in shape; got {normals.shape} and {points.shape}"
        raise ValueError(msg)
    partition = _groups(surfaces)
    surviving_rows = _surviving_rows(visibility, transmittance, points, surfaces.n_facets)

    def at(receivers, receiver_normal, surviving_all):
        total = jnp.zeros(receivers.shape[0])
        for profile, areal, point in partition:
            if len(areal):
                cosine, _ = _emitter_cosine(surfaces, areal, receivers)
                radiance = jnp.take(surfaces.emission, areal) * profile.radiance_per_exitance(
                    cosine
                )
                projected = projected_solid_angle(
                    receivers[:, None, :],
                    jnp.broadcast_to(receiver_normal[:, None, :], (*cosine.shape, 3)),
                    jnp.take(surfaces.vertices, areal, axis=0)[None, ...],
                )
                surviving = _transmittance(
                    absorption, jnp.take(surfaces.centroid, areal, axis=0), receivers
                ) * jnp.take(surviving_all, areal, axis=1)
                total = total + jnp.sum(radiance * projected * surviving, axis=1)
            if len(point):
                centroid = jnp.take(surfaces.centroid, point, axis=0)
                offset = receivers[:, None, :] - centroid[None, :, :]
                distance_squared = dot(offset, offset)
                distance = jnp.sqrt(jnp.where(distance_squared == 0.0, 1.0, distance_squared))
                source_cosine = dot(offset, jnp.take(surfaces.normal, point, axis=0)[None, :, :])
                receiver_cosine = jnp.maximum(
                    -dot(offset, receiver_normal[:, None, :]) / distance, 0.0
                )
                fraction = profile.intensity_fraction(source_cosine / distance)
                surviving = _transmittance(absorption, centroid, receivers) * jnp.take(
                    surviving_all, point, axis=1
                )
                total = total + jnp.sum(
                    jnp.take(surfaces.power, point)
                    * fraction
                    * receiver_cosine
                    * surviving
                    / distance_squared,
                    axis=1,
                )
        return total

    return _chunked((points, normals, surviving_rows), chunk_size, at)
