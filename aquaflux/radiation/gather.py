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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import (
    Visibility,
    _unchecked_visibility,
    refuse_points_inside,
    surviving_fraction,
)
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, in_passes, receivers_per_pass
from aquaflux.vectors import dot

__all__ = [
    "direct_fluence_rate",
    "direct_irradiance",
    "streamed_fluence_rate",
    "summed_fluence_rate",
]


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


def _shadow_rows(visibility, transmittance, points, sets) -> tuple:
    """What the chunks need to form the fraction getting past the intervening bodies.

    ``(array, axis)`` pairs for :func:`~aquaflux.radiation.work.in_passes` — the mask's own layers, each cut along its
    receiver axis — and **not** the fraction itself: formed here it would be a floating-point
    array the size of the whole problem, eight bytes a pair on top of the mask, before any
    chunking could bound it. Each chunk forms its own share instead, in
    :func:`~aquaflux.radiation.visibility.surviving_fraction`, the one expression
    :meth:`~aquaflux.radiation.visibility.Visibility.surviving` also evaluates.

    **Empty when nothing occludes, and deliberately not a row of ones**, for the same reason.

    Returns
    -------
    tuple
        The ``(array, axis)`` pairs, and the transmittance to apply, defaulting to opaque.
    """
    if visibility is None:
        if transmittance is not None:
            msg = "transmittance was given without a visibility mask to apply it to"
            raise ValueError(msg)
        return (), None
    if not isinstance(visibility, Visibility):
        msg = f"visibility must be a Visibility; got {type(visibility).__name__}"
        raise TypeError(msg)
    visibility.for_receivers(points)
    if visibility.clear_behind:
        _refuse_light_from_behind(sets)
    if transmittance is None:
        transmittance = jnp.zeros(visibility.n_occluders)
    return ((visibility.blocked, 1), (visibility.hidden_by_geometry, 0)), transmittance


def _refuse_light_from_behind(sets) -> None:
    """Raise if a set lights anything from behind a facet, which a mask has recorded as clear.

    A mask built with :attr:`~aquaflux.radiation.visibility.Visibility.clear_behind` never
    tested a pair whose source faces away from its receiver, so it is right only where such a
    pair carries nothing: every areal facet must be dark behind itself. Checked here, at the
    gather, against the optics actually used rather than the ones the mask was built with.
    """
    for surfaces in sets:
        if not surfaces.dark_behind:
            msg = (
                "an areal facet emits with a profile that is not declared dark behind itself, "
                "through a shadow mask that left every pair facing away from its receiver "
                "untested. Build the mask again with these profiles in the surface set, or set "
                "dark_behind on a profile that sends nothing backwards."
            )
            raise ValueError(msg)


def _surviving(layers, transmittance):
    """A chunk's surviving fraction from its mask layers, or ``None`` where nothing occludes."""
    return surviving_fraction(*layers, transmittance) if layers else None


def _masked(surviving, facets):
    """The surviving fraction for ``facets`` from a chunk's, or 1 where nothing occludes."""
    return 1.0 if surviving is None else jnp.take(surviving, facets, axis=1)


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


def streamed_fluence_rate(
    sets,
    points,
    *,
    shadow_geometry: Surfaces,
    occluders,
    self_occlusion=None,
    visibility_options=None,
    absorption: Absorption | None = None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
):
    """The summed fluence rate of several surface sets, each chunk's shadow mask built and dropped.

    The streamed counterpart of passing a built mask to :func:`direct_fluence_rate`: memory is set
    by the chunk rather than by the receiver count, in the forward pass **and** in a gradient.

    **One mask per chunk serves every set.** The sets share their geometry — an emitted field and
    the reflected one it bounces into, say — so they cast the same shadows, and building the mask
    once per set would double the dominant cost for nothing.

    **A gradient keeps nothing per pair.** Reverse mode would ordinarily hold every chunk's
    receiver-by-facet intermediates until the backward pass, which at a mesh's cells is the size
    of the whole problem however small the chunks are. Each chunk is instead a custom
    vector-Jacobian product that saves only its inputs and, on the way back, rebuilds its mask and
    recomputes its gather. So a gradient costs a second mask build and a second gather per chunk,
    and memory for one chunk. Emission, power, reflectance, the absorbing medium and each body's
    transmittance are all reached; the bodies' geometry is not, as everywhere shadows are frozen.

    **The gather is compiled once per call and reused by every chunk**, forward and backward. Run
    eagerly it would be traced again for each chunk, with its arrays formed one operation at a
    time, and at a finely divided emitter -- tens of thousands of chunks of a few dozen receivers
    -- that overhead is most of the cost. The live values and each chunk's mask are its
    **arguments**, not constants it closes over, so a compiled program holds no copy of them;
    only the sets' labels, which decide its shape, are closed over. A shorter last chunk compiles
    once more, which is cheaper than building a mask for padding receivers.

    Parameters
    ----------
    sets : sequence of Surfaces
        The sets whose fields are summed. Their optics and their vertices are read live; their
        geometry must be ``shadow_geometry``'s, or the shadows are cast from somewhere else.
    points : array_like, shape ``(n_points, 3)``
        Receiver positions.
    shadow_geometry : Surfaces
        The geometry the masks are built from, **concrete**: building a mask is host work and
        cannot see a traced vertex. A model passes the geometry it was built for, which is what
        lets a gradient with respect to a lamp's position be taken with the shadows frozen.
    occluders : sequence of aquaflux.solids.Body
        The bodies in the way. May be empty, which still streams the surface's own shadowing.
    self_occlusion : SelfOcclusion, optional
        How the surface shadows itself, as in
        :func:`~aquaflux.radiation.visibility.build_visibility`.
    visibility_options : mapping, optional
        Further keywords for :func:`~aquaflux.radiation.visibility.build_visibility`.
    absorption, transmittance, pair_limit
        As for :func:`direct_fluence_rate`. ``pair_limit`` bounds each streamed chunk and each
        traced chunk inside it.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Notes
    -----
    Not compilable with ``jit`` as a whole: the chunk loop and the mask builds are host work,
    which a traced loop could not call.
    """
    points = jnp.asarray(points, dtype=float)
    per_chunk = receivers_per_pass(pair_limit, shadow_geometry.n_facets)
    if points.shape[0] == 0:
        return jnp.zeros(0)
    live = (tuple(sets), absorption, transmittance)
    options = {
        **({} if visibility_options is None else dict(visibility_options)),
        **({} if self_occlusion is None else {"self_occlusion": self_occlusion}),
    }
    # What depends on the scene and not on the pass is done once, here: the refusal of points
    # inside a body, over every point at once, and whatever the strategy can prepare from the
    # surface alone -- the grid over its triangles, which is otherwise rebuilt for every pass.
    refuse_points_inside(occluders, shadow_geometry, points)
    if options.get("self_occlusion") is not None:
        options["self_occlusion"] = options["self_occlusion"].prepared(shadow_geometry)
    shadows = _Shadows(
        geometry=shadow_geometry,
        occluders=tuple(occluders),
        options=options,
        gather=_compiled_gather(live, pair_limit),
    )
    return jnp.concatenate(
        [
            _shadowed_chunk(live, points[start : start + per_chunk], shadows)
            for start in range(0, points.shape[0], per_chunk)
        ]
    )


def _compiled_gather(live, pair_limit: int):
    """The summed gather of every set against one chunk's mask, as one compiled program.

    The live values are split into their floating-point arrays, which the program takes as
    arguments and a gradient reaches, and everything else, which it closes over. What is closed
    over is exactly what must stay concrete: which profile each facet emits with decides the
    traced program's shape, so it cannot be an argument. Built once per stream, so every chunk of
    that stream reuses the one program.
    """
    _, labels = eqx.partition(live, eqx.is_inexact_array)

    @jax.jit
    def gather(values, points, mask):
        sets, absorption, transmittance = eqx.combine(values, labels)
        return summed_fluence_rate(
            sets,
            points,
            absorption=absorption,
            visibility=mask,
            transmittance=transmittance,
            pair_limit=pair_limit,
        )

    return gather


class _Shadows(eqx.Module):
    """What a chunk's mask is built from, and the gather it feeds: fixed for a whole stream, and
    never differentiated."""

    geometry: Surfaces
    occluders: tuple
    options: dict
    gather: object = eqx.field(static=True)

    def mask(self, points) -> Visibility:
        """The mask for these receivers."""
        return _unchecked_visibility(self.occluders, self.geometry, points, **self.options)


def _chunk_total(live, points, shadows: _Shadows):
    """One chunk's summed field, with its own mask."""
    values, _ = eqx.partition(live, eqx.is_inexact_array)
    return shadows.gather(values, points, shadows.mask(points))


@eqx.filter_custom_vjp
def _shadowed_chunk(live, points, shadows):
    """One chunk, differentiated by recomputing it rather than by keeping its intermediates."""
    return _chunk_total(live, points, shadows)


@_shadowed_chunk.def_fwd
def _shadowed_chunk_fwd(perturbed, live, points, shadows):
    del perturbed
    return _chunk_total(live, points, shadows), None


@_shadowed_chunk.def_bwd
def _shadowed_chunk_bwd(residuals, cotangent, perturbed, live, points, shadows):
    del residuals, perturbed
    _, pull = eqx.filter_vjp(lambda value: _chunk_total(value, points, shadows), live)
    return pull(cotangent)[0]


def direct_fluence_rate(
    surfaces: Surfaces,
    points,
    *,
    absorption: Absorption | None = None,
    visibility: Visibility | None = None,
    occluders=None,
    self_occlusion=None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
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
    occluders : sequence of aquaflux.solids.Body, optional
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
    pair_limit : int, optional
        Receiver-by-facet pairs per traced chunk, and per streamed mask. The pairs are what
        cost memory, so a chunk holds as many receivers as fit and a finer emitter gets fewer of
        them per chunk rather than a larger chunk. Trades peak memory against nothing; the
        arithmetic is the same either way.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If ``pair_limit`` is less than one, if the visibility mask was built for a
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
        return streamed_fluence_rate(
            (surfaces,),
            points,
            shadow_geometry=surfaces,
            occluders=() if occluders is None else occluders,
            self_occlusion=self_occlusion,
            absorption=absorption,
            transmittance=transmittance,
            pair_limit=pair_limit,
        )
    return summed_fluence_rate(
        (surfaces,),
        points,
        absorption=absorption,
        visibility=visibility,
        transmittance=transmittance,
        pair_limit=pair_limit,
    )


def summed_fluence_rate(
    sets,
    points,
    *,
    absorption: Absorption | None = None,
    visibility: Visibility | None = None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
):
    """The summed fluence rate of several surface sets **on one geometry**, in one pass.

    What :func:`direct_fluence_rate` gives for each set, added — but everything that depends only
    on where the facets and receivers are is formed **once** and shared: the solid angle of
    every facet at every receiver, the emitter cosine, the attenuation along each path and the
    surviving fraction through the mask. Only the radiance weight differs between the sets, so a
    model's emitted field and the reflected field it bounces into cost one geometric pass rather
    than two. Under a graded medium that saves a second walk of every path through the grid.

    Each set's own terms are summed first and the sets added after, in the order given, which is
    exactly what adding :func:`direct_fluence_rate`'s results would do; the shared factors are
    the same numbers either way, so the answer is too.

    Parameters
    ----------
    sets : sequence of Surfaces
        The sets whose fields are summed. **The geometry is read from the first**: the others
        must be the same facets with other optics -- ``surfaces.with_optics(...)`` of it -- and a
        set whose concrete vertices differ is refused. Their emission, power and profiles are read
        from each set.
    points, absorption, visibility, transmittance, pair_limit
        As for :func:`direct_fluence_rate`.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If no set is given, or if the sets do not share their geometry.
    """
    sets = tuple(sets)
    geometry = _one_geometry(sets)
    layers, transmittance = _shadow_rows(visibility, transmittance, points, sets)
    points = jnp.asarray(points, dtype=float)
    plans = [_groups(surfaces) for surfaces in sets]
    areal_all = np.flatnonzero(~geometry.is_point_source)
    point_all = np.flatnonzero(geometry.is_point_source)

    def at(receivers, *chunk_layers):
        surviving_all = _surviving(chunk_layers, transmittance)
        if len(areal_all):
            areal_cosine, _ = _emitter_cosine(geometry, areal_all, receivers)
            omega = solid_angle(
                receivers[:, None, :], jnp.take(geometry.vertices, areal_all, axis=0)[None, ...]
            )
            areal_surviving = _transmittance(
                absorption, jnp.take(geometry.centroid, areal_all, axis=0), receivers
            ) * _masked(surviving_all, areal_all)
        if len(point_all):
            point_cosine, distance_squared = _emitter_cosine(geometry, point_all, receivers)
            point_surviving = _transmittance(
                absorption, jnp.take(geometry.centroid, point_all, axis=0), receivers
            ) * _masked(surviving_all, point_all)
        totals = []
        for surfaces, partition in zip(sets, plans, strict=True):
            total = jnp.zeros(receivers.shape[0])
            for profile, areal, point in partition:
                if len(areal):
                    pick = _columns(areal, areal_all)
                    radiance = jnp.take(surfaces.emission, areal) * profile.radiance_per_exitance(
                        pick(areal_cosine)
                    )
                    total = total + jnp.sum(radiance * pick(omega) * pick(areal_surviving), axis=1)
                if len(point):
                    pick = _columns(point, point_all)
                    fraction = profile.intensity_fraction(pick(point_cosine))
                    total = total + jnp.sum(
                        jnp.take(surfaces.power, point)
                        * fraction
                        * pick(point_surviving)
                        / pick(distance_squared),
                        axis=1,
                    )
            totals.append(total)
        return sum(totals)

    return in_passes(((points, 0), *layers), pair_limit, geometry.n_facets, at)


def _one_geometry(sets) -> Surfaces:
    """The geometry every set shares, refusing sets that do not share one."""
    if not sets:
        msg = "at least one surface set is needed"
        raise ValueError(msg)
    geometry = sets[0]
    for other in sets[1:]:
        same = (
            other.n_facets == geometry.n_facets
            and other.point_source_index == geometry.point_source_index
        )
        if same and not any(
            isinstance(surfaces.vertices, jax.core.Tracer) for surfaces in (geometry, other)
        ):
            same = other.vertices is geometry.vertices or np.array_equal(
                np.asarray(other.vertices), np.asarray(geometry.vertices)
            )
        if not same:
            msg = (
                "the surface sets summed in one gather must share their geometry -- the solid "
                "angles and shadows are formed once, from the first -- so each must be "
                "`with_optics(...)` of the same set"
            )
            raise ValueError(msg)
    return geometry


def _columns(subset: np.ndarray, of: np.ndarray):
    """Pick ``subset``'s columns out of arrays formed over ``of``, or pass them through whole.

    A scalar passes through too, which is what the surviving fraction is when nothing occludes and
    the medium is vacuum.
    """
    if len(subset) == len(of):
        return lambda array: array
    positions = np.searchsorted(of, subset)
    return lambda array: array if jnp.ndim(array) == 0 else jnp.take(array, positions, axis=1)


def direct_irradiance(
    surfaces: Surfaces,
    points,
    normals,
    *,
    absorption: Absorption | None = None,
    visibility: Visibility | None = None,
    transmittance=None,
    pair_limit: int = DEFAULT_PAIR_LIMIT,
    point_sources_only: bool = False,
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
    pair_limit : int, optional
        Receiver-by-facet pairs per traced chunk, as for :func:`direct_fluence_rate`.
    point_sources_only : bool, optional
        Gather the point sources alone and leave the areal facets out entirely -- not weighted
        by zero, but never visited. What a surface solve needs as the irradiance arriving from
        outside its transfer matrix, which already carries every areal facet; and the areal
        pairs are nearly all of the cost, since a clipped projected solid angle is formed for
        each.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Irradiance in W/m².

    Raises
    ------
    ValueError
        If ``normals`` and ``points`` disagree in shape, if ``pair_limit`` is less than one, or
        if the visibility mask was built for a different set of receivers.
    """
    layers, transmittance = _shadow_rows(visibility, transmittance, points, (surfaces,))
    points = jnp.asarray(points, dtype=float)
    normals = jnp.asarray(normals, dtype=float)
    if normals.shape != points.shape:
        msg = f"normals must match points in shape; got {normals.shape} and {points.shape}"
        raise ValueError(msg)
    partition = _groups(surfaces)

    def at(receivers, receiver_normal, *chunk_layers):
        surviving_all = _surviving(chunk_layers, transmittance)
        total = jnp.zeros(receivers.shape[0])
        for profile, areal, point in partition:
            if len(areal) and not point_sources_only:
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
                ) * _masked(surviving_all, areal)
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
                surviving = _transmittance(absorption, centroid, receivers) * _masked(
                    surviving_all, point
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

    return in_passes(((points, 0), (normals, 0), *layers), pair_limit, surfaces.n_facets, at)
