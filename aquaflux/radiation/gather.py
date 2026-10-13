"""Summing every source's contribution at every receiver — the backward gather.

At each receiver the module adds up what every emitting facet and every point source delivers
there. It is a *backward* gather because it starts at the receiver and looks toward the
sources, the opposite of tracing photons forward, and it is deterministic: the answer at a
point is a sum, not a sample, so it carries neither stochastic noise nor the bias that comes
from scoring a photon's path length through a finite cell.

Each term may be attenuated by the medium it crosses, through an ``Absorption`` supplied by the
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

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.morton import morton_order
from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.back_faces import BackFaces
from aquaflux.radiation.culling import culling_or_default
from aquaflux.radiation.lit_blocks import lit_segments
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import (
    Visibility,
    _unchecked_visibility,
    refuse_points_inside,
    surviving_from_layers,
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

    ``(array, axis)`` pairs for :func:`~aquaflux.radiation.work.in_passes` — the mask's own
    layers, each cut along its receiver axis — and **not** the fraction itself: formed here it
    would be a floating-point array the size of the whole problem, eight bytes a pair on top of
    the mask, before any chunking could bound it. Each chunk forms its own share instead, in
    :func:`~aquaflux.radiation.visibility.surviving_fraction`, the one expression
    :meth:`~aquaflux.radiation.visibility.Visibility.surviving` also evaluates. The surface's own
    layer is left out where the mask holds none, because the surface hides nothing.

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
        return _Layers(()), None
    if not isinstance(visibility, Visibility):
        msg = f"visibility must be a Visibility; got {type(visibility).__name__}"
        raise TypeError(msg)
    visibility.for_receivers(points)
    if visibility.clear_behind:
        _refuse_light_from_behind(sets)
    if transmittance is None:
        transmittance = jnp.zeros(visibility.n_occluders)
    kinds, layers = visibility.layers()
    return _Layers(layers, kinds), transmittance


class _Layers(tuple):
    """A mask's ``(array, axis)`` layers, with what each is, for :func:`_surviving`.

    A tuple, so it is cut into chunks, spread and tested for emptiness as the layers themselves
    are; the names ride along on the host.
    """

    def __new__(cls, layers, kinds=()):
        made = super().__new__(cls, layers)
        made.kinds = tuple(kinds)
        return made


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


def _surviving(kinds, layers, transmittance):
    """A chunk's surviving fraction from its mask layers, or ``None`` where nothing occludes.

    The layers are named by ``kinds``, as :meth:`~aquaflux.radiation.visibility.Visibility.layers`
    names them.
    """
    if not layers:
        return None
    return surviving_from_layers(kinds, layers, transmittance)


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
    return jnp.exp(-absorption.optical_depth(source, receivers))


def _emitter_direction(centroid, receivers):
    """Unit direction from each emitting facet towards each receiver, and the separation squared.

    The arrays broadcast together, their last axis the three coordinates. The direction is what a
    source's angular distribution is asked about, together with the source's own normal. A
    receiver exactly at a centroid gets a zero direction rather than a NaN one.
    """
    offset = receivers - centroid
    distance_squared = dot(offset, offset)
    distance = jnp.sqrt(jnp.where(distance_squared == 0.0, 1.0, distance_squared))
    return offset / distance[..., None], distance_squared


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
    extra=None,
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
        cannot see a traced vertex. Building the masks from it rather than from ``sets`` is what
        lets a gradient with respect to a source's position be taken here, with the shadows
        frozen; a model passes the geometry it was built for.
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
    extra : object, optional
        Further light each chunk receives, built and dropped with the chunk and recomputed with
        it on the way back: anything with a ``field(live, points)`` method returning the chunk's
        share, ``(n_chunk,)``, from ``live = (sets, absorption, transmittance)``. Its own shadows
        must be built inside ``field`` from concrete geometry, as the chunk's mask is, since the
        live values are traced on the way back. What reaches the receivers by a mirror is
        gathered this way.

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
    # inside a body, over every point at once, and whatever the strategies can prepare from the
    # surface alone -- the grid over its triangles, and the bodies' summaries of its facet
    # clusters, which are otherwise rebuilt for every pass.
    refuse_points_inside(occluders, shadow_geometry, points)
    if options.get("self_occlusion") is not None:
        options["self_occlusion"] = options["self_occlusion"].prepared(shadow_geometry)
    if occluders:
        options["body_culling"] = culling_or_default(options.get("body_culling")).prepared(
            tuple(occluders), shadow_geometry.centroid
        )
    shadows = _Shadows(
        geometry=shadow_geometry,
        occluders=tuple(occluders),
        options=options,
        gather=_compiled_parts(live, pair_limit),
        extra=extra,
    )
    # The chunks are cut from the points in space-filling-curve order, so each is a compact
    # region: its blocks' boxes are small, its tiles' shafts narrow, and both decide more.
    order = morton_order(np.asarray(points))
    ordered = points[order]
    field = jnp.concatenate(
        [
            _shadowed_chunk(live, ordered[start : start + per_chunk], shadows)
            for start in range(0, points.shape[0], per_chunk)
        ]
    )
    return jnp.zeros(points.shape[0]).at[order].set(field)


#: Segments each streamed chunk's blocks are cut into, per areal group. More pad less, fewer
#: compile less: a chunk's segments are compiled by shape, and equal segments keep the shapes to
#: the width ladder. Four padded a sampled Sozzi-like scene to 0.56 of every pair, against 0.75
#: for one and 0.59 for eight.
_STREAM_SEGMENTS = 4


class _Labels:
    """The part of a stream's live values that decides its programs' shapes, hashable by content.

    Which profile each facet emits with, which facets are point sources, the kinds of profile and
    medium: everything but the floating-point values a gradient reaches. Hashed by what it holds
    rather than by identity, so two calls on the same scene -- each with its own copy of the
    surface set -- find the same compiled programs instead of compiling their own.
    """

    def __init__(self, labels):
        self.labels = labels
        leaves, treedef = jax.tree.flatten(labels)
        self._key = (treedef, tuple(_content(leaf) for leaf in leaves))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other) -> bool:
        return isinstance(other, _Labels) and self._key == other._key


def _content(leaf):
    """A hashable stand-in for one label: an array by its type, shape and bytes."""
    if isinstance(leaf, jax.core.Tracer):
        return ("traced", id(leaf))
    if isinstance(leaf, np.ndarray | jax.Array):
        array = np.asarray(leaf)
        return (array.dtype.str, array.shape, array.tobytes())
    try:
        hash(leaf)
    except TypeError:
        return ("unhashable", id(leaf))
    return leaf


def _unpacked(values, points, mask, labels: _Labels):
    """The sets, medium, mask layers and transmittance of one chunk, inside its program."""
    sets, absorption, transmittance = eqx.combine(values, labels.labels)
    layers, transmittance = _shadow_rows(mask, transmittance, points, sets)
    return sets, absorption, layers, transmittance


@functools.partial(jax.jit, static_argnames=("labels", "pair_limit"))
def _point_part(values, points, mask, *, labels: _Labels, pair_limit: int):
    """A chunk's point sources, compiled once per scene and chunk size."""
    sets, absorption, layers, transmittance = _unpacked(values, points, mask, labels)
    return _point_fluence(sets, points, layers, absorption, transmittance, pair_limit)


@functools.partial(jax.jit, static_argnames=("labels", "group", "pair_limit"))
def _segment_part(values, points, mask, segment, *, labels: _Labels, group: int, pair_limit: int):
    """One segment of a chunk's areal blocks, compiled once per scene and segment shape."""
    sets, absorption, layers, transmittance = _unpacked(values, points, mask, labels)
    return _segment_fluence(
        sets, _areal_groups(sets)[group], points, layers, absorption, transmittance, segment,
        pair_limit,
    )  # fmt: skip


@dataclasses.dataclass(frozen=True, eq=False)
class _CompiledParts:
    """The gather of every set against one chunk's mask, in pieces each compiled once.

    ``points`` gathers the point sources; ``segment`` one segment of the areal blocks, by the
    index of its group. Both take the live values and the chunk's mask as arguments; the labels,
    which decide the programs' shapes, are static and hashed by content, and a segment's shape
    comes from the width ladder -- so a scene compiles a handful of programs however many chunks
    and however many calls it has.
    """

    labels: _Labels
    pair_limit: int

    def points(self, values, points, mask):
        return _point_part(values, points, mask, labels=self.labels, pair_limit=self.pair_limit)

    def segment(self, values, points, mask, group, segment):
        return _segment_part(
            values, points, mask, segment, labels=self.labels, group=group,
            pair_limit=self.pair_limit,
        )  # fmt: skip


def _compiled_parts(live, pair_limit: int) -> _CompiledParts:
    """The compiled pieces of a stream's gather. See :class:`_CompiledParts`."""
    _, labels = eqx.partition(live, eqx.is_inexact_array)
    return _CompiledParts(labels=_Labels(labels), pair_limit=pair_limit)


class _Shadows(eqx.Module):
    """What a chunk's mask is built from, and the gather it feeds: fixed for a whole stream, and
    never differentiated."""

    geometry: Surfaces
    occluders: tuple
    options: dict
    gather: _CompiledParts = eqx.field(static=True)
    extra: object = eqx.field(static=True, default=None)

    def mask(self, points) -> Visibility:
        """The mask for these receivers."""
        return _unchecked_visibility(self.occluders, self.geometry, points, **self.options)


def _chunk_total(live, points, shadows: _Shadows):
    """One chunk's summed field, with its own mask and its own layout.

    The layout is formed from the stream's geometry, which is concrete, and from the sets' labels
    alone, so it is the same forward and on the way back, when the values are traced.
    """
    values, _ = eqx.partition(live, eqx.is_inexact_array)
    mask = shadows.mask(points)
    groups = _areal_groups(live[0])
    layout = areal_layout(points, shadows.geometry, groups, segments=_STREAM_SEGMENTS)
    total = shadows.gather.points(values, points, mask)
    for index, segments in enumerate(layout):
        for segment in segments:
            total = total + shadows.gather.segment(values, points, mask, index, segment)
    if shadows.extra is not None:
        total = total + shadows.extra.field(live, points)
    return total


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
    """Fluence rate at each receiver point, straight from the sources, with no reflection.

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
        Receiver-by-facet pairs a traced chunk may form, and per streamed mask. The pairs are
        what cost memory, so a chunk holds as many receivers as fit and a finer emitter gets fewer
        of them per chunk rather than a larger chunk. Trades peak memory against nothing; the
        arithmetic is the same either way. A traced chunk forms no more than
        ``aquaflux.radiation.work.PASS_PAIRS`` however high this is set, because past a
        core's cache the same pairs cost more.

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
            # The masks are built from positions alone; with the gradient stopped, a traced
            # emission or profile here is not a tangent carried into every chunk's custom
            # vector-Jacobian product, which accepts one only through the live values.
            shadow_geometry=jax.lax.stop_gradient(surfaces),
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
    layout=None,
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
    layout : tuple, optional
        Which facets each block of receivers is gathered against, from :func:`areal_layout`
        for these points and these sets. Unset, it is formed here -- which reads the positions,
        and so lists every facet when they are traced. A caller whose points are traced but
        known elsewhere passes the layout formed from them.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Fluence rate in W/m².

    Raises
    ------
    ValueError
        If no set is given, or if the sets do not share their geometry.

    Notes
    -----
    **A facet is not gathered at a receiver it certainly sends nothing to.** For every facet
    whose profile in every set is dark behind itself, receivers are grouped in small compact
    blocks and a facet whose plane a block's bounding box lies wholly behind is left off that
    block's list (:mod:`~aquaflux.radiation.lit_blocks`). Around a lamp that is about half the
    pairs. What is left off is exactly zero, so the field is the full gather's up to the order
    its terms are added in -- a rounding, not a bit-for-bit identity.
    """
    sets = tuple(sets)
    geometry = _one_geometry(sets)
    layers, transmittance = _shadow_rows(visibility, transmittance, points, sets)
    groups = _areal_groups(sets)
    if layout is None:
        # Laid out from the points as given: inside a trace even a concrete array becomes a
        # tracer once it passes through jnp, and a traced position cannot be read.
        layout = areal_layout(points, geometry, groups)
    points = jnp.asarray(points, dtype=float)
    total = _point_fluence(sets, points, layers, absorption, transmittance, pair_limit)
    for group, segments in zip(groups, layout, strict=True):
        for segment in segments:
            total = total + _segment_fluence(
                sets, group, points, layers, absorption, transmittance, segment, pair_limit
            )
    return total


@dataclasses.dataclass(frozen=True, eq=False)
class _ArealGroup:
    """Areal facets every set emits from with one profile each: the unit a layout is made for.

    The sets share their geometry and may differ in their profiles -- a model's reflected set is
    Lambertian whatever the lamp emits like -- so the areal facets are split by the profile each
    set gives them, and each piece is gathered with a profile that is a concrete object per set.
    Host metadata, never traced, so a plain record rather than a pytree.
    """

    facets: np.ndarray
    profiles: tuple

    @property
    def dark_behind(self) -> bool:
        """Whether every set's profile here sends nothing behind its facets."""
        return all(profile.dark_behind for profile in self.profiles)


def _areal_groups(sets) -> tuple[_ArealGroup, ...]:
    """The areal facets split by the profile each set emits them with, on the host."""
    for surfaces in sets:
        if isinstance(surfaces.profile_index, jax.core.Tracer):
            _groups(surfaces)  # raises, saying why the index must be concrete
    geometry = sets[0]
    areal = np.flatnonzero(~geometry.is_point_source)
    keys = np.stack([np.asarray(surfaces.profile_index)[areal] for surfaces in sets], axis=1)
    kinds, which = np.unique(keys, axis=0, return_inverse=True)
    return tuple(
        _ArealGroup(
            facets=areal[np.asarray(which).ravel() == index],
            profiles=tuple(
                surfaces.profiles[int(kind)] for surfaces, kind in zip(sets, key, strict=True)
            ),
        )
        for index, key in enumerate(kinds)
    )


def areal_layout(points, geometry: Surfaces, groups, *, segments=None) -> tuple:
    """How the areal facets are gathered at ``points``: per group, its blocks and their lists.

    A facet is left off a block's list only where it certainly sends the block nothing -- the
    block lies behind its plane and every set's profile there is dark behind -- and only where
    both the positions and the geometry can be read. Otherwise every facet is listed, and the
    gather does the full work in the same layout.

    Parameters
    ----------
    points : array_like, shape ``(n_points, 3)``
    geometry : Surfaces
        The facets' positions; the planes are read from them when they are concrete.
    groups : sequence
        The areal groups, from the sets being gathered.
    segments : int, optional
        As for :func:`~aquaflux.radiation.lit_blocks.lit_segments`.

    Returns
    -------
    tuple of tuple of LitSegment
        One tuple of segments per group.
    """
    readable = not any(
        isinstance(array, jax.core.Tracer) for array in (points, geometry.centroid, geometry.normal)
    )
    facing = BackFaces.of(geometry) if readable else None
    return tuple(
        lit_segments(
            points,
            group.facets,
            facing if group.dark_behind else None,
            segments=segments,
        )
        for group in groups
    )


def _point_fluence(sets, points, layers, absorption, transmittance, pair_limit):
    """What every set's point sources deliver at ``points``, ``(n_points,)``: every pair gathered."""
    geometry = sets[0]
    point_all = np.flatnonzero(geometry.is_point_source)
    if not len(point_all) or points.shape[0] == 0:
        return jnp.zeros(points.shape[0])
    plans = [_groups(surfaces) for surfaces in sets]
    centroid = jnp.take(geometry.centroid, point_all, axis=0)
    normal = jnp.take(geometry.normal, point_all, axis=0)
    # Only the point sources' columns of the mask are read, so only they are widened.
    column_layers = tuple(
        (jnp.take(array, point_all, axis=array.ndim - 1), axis) for array, axis in layers
    )

    def at(receivers, *chunk_layers):
        direction, distance_squared = _emitter_direction(centroid[None], receivers[:, None])
        surviving = _transmittance(absorption, centroid[None], receivers[:, None, :])
        shadows = _surviving(layers.kinds, chunk_layers, transmittance)
        if shadows is not None:
            surviving = surviving * shadows
        total = jnp.zeros(receivers.shape[0])
        for surfaces, partition in zip(sets, plans, strict=True):
            for profile, _, point in partition:
                if len(point):
                    pick = _columns(point, point_all)
                    total = total + jnp.sum(
                        jnp.take(surfaces.power, point)
                        * profile.intensity_fraction(pick(direction), pick(normal[None]))
                        * pick(surviving)
                        / pick(distance_squared),
                        axis=1,
                    )
        return total

    return in_passes(((points, 0), *column_layers), pair_limit, len(point_all), at)


def _segment_fluence(sets, group, points, layers, absorption, transmittance, segment, pair_limit):
    """What one segment's blocks receive from their listed facets, as ``(n_points,)``.

    Each block is gathered against its own list: the receivers are gathered by their rows, the
    facets by their indices, and the mask at both -- so nothing is formed for a pair off the
    list. Entries past a block's list, and padding receivers, contribute exactly zero; a padding
    receiver's row is one past the last point and its result is dropped.
    """
    geometry = sets[0]
    n_points = points.shape[0]
    rows = jnp.asarray(segment.rows)
    lists = (jnp.asarray(segment.facets), jnp.asarray(segment.valid))

    def at(block_rows, *block_lists):
        block_facets, block_valid = lists if segment.shared else block_lists
        row = jnp.minimum(block_rows, n_points - 1)
        receivers = jnp.take(points, row, axis=0, mode="clip")[:, :, None, :]
        centroid = jnp.take(geometry.centroid, block_facets, axis=0, mode="clip")[:, None]
        normal = jnp.take(geometry.normal, block_facets, axis=0, mode="clip")[:, None]
        vertices = jnp.take(geometry.vertices, block_facets, axis=0, mode="clip")[:, None]
        direction, _ = _emitter_direction(centroid, receivers)
        weight = solid_angle(receivers, vertices) * _transmittance(absorption, centroid, receivers)
        if layers:
            pair = (row[:, :, None], block_facets[:, None, :])
            blocked, *pairs = (array for array, _ in layers)
            weight = weight * surviving_from_layers(
                layers.kinds,
                (blocked[:, pair[0], pair[1]], *(array[pair] for array in pairs)),
                transmittance,
            )
        weight = jnp.where(block_valid[:, None, :], weight, 0.0)
        total = jnp.zeros(block_rows.shape)
        for surfaces, profile in zip(sets, group.profiles, strict=True):
            emission = jnp.take(surfaces.emission, block_facets, axis=0, mode="clip")[:, None, :]
            total = total + jnp.sum(
                emission * profile.radiance_per_exitance(direction, normal) * weight, axis=2
            )
        return total

    per_block = segment.block * segment.width
    arrays = ((rows, 0),) if segment.shared else ((rows, 0), (lists[0], 0), (lists[1], 0))
    received = in_passes(arrays, pair_limit, per_block, at)
    return jnp.zeros(n_points).at[rows.ravel()].add(received.ravel(), mode="drop")


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
    receiver_facet=None,
):
    """Irradiance on an oriented receiving surface at each point, with no reflection.

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
    receiver_facet : array_like of int, shape ``(n_points,)`` or ``(n_points, k)``, optional
        When a receiver point lies on one of ``surfaces``' own facets, that facet's index; when it
        lies on several (on a shared edge or vertex), each of them, ``-1`` filling a row that names
        fewer. Those facets are left out of that point's sum. A flat facet sends nothing into its
        own plane, so this is the right answer and not an approximation -- the same convention as
        the zero diagonal of the facet-to-facet transfer.
        ⚠️ **Omitting it where it applies can count a facet's whole exitance at a point on it.** A
        point inside a facet's triangle is in that facet's plane, where the corners' heights above
        the receiver's plane are rounding noise. Off an axis-aligned plane the clip cannot always
        decide them, keeps the in-plane triangle, and the projected solid angle is then the full
        hemisphere, ``E = B``: measured on a rotated plane at sub-triangle centres, over half the
        points did so. A point at a facet's centroid or on a shared vertex happens to read zero, so
        this goes unseen on face-centre receivers.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Irradiance in W/m².

    Raises
    ------
    ValueError
        If ``normals`` and ``points`` disagree in shape, if ``pair_limit`` is less than one, if
        ``receiver_facet`` is not one row per point naming facets of ``surfaces`` (or ``-1``), or
        if the visibility mask was built for a different set of receivers.
    """
    layers, transmittance = _shadow_rows(visibility, transmittance, points, (surfaces,))
    points = jnp.asarray(points, dtype=float)
    normals = jnp.asarray(normals, dtype=float)
    if normals.shape != points.shape:
        msg = f"normals must match points in shape; got {normals.shape} and {points.shape}"
        raise ValueError(msg)
    # Cut into chunks only when named, so a gather with none adds nothing per receiver.
    named = (
        ()
        if receiver_facet is None
        else ((_own_facet_rows(receiver_facet, len(points), surfaces.n_facets), 0),)
    )
    partition = _groups(surfaces)

    def at(receivers, receiver_normal, *chunk_layers):
        own_facets, chunk_layers = (
            (chunk_layers[0], chunk_layers[1:]) if named else (None, chunk_layers)
        )
        surviving_all = _surviving(layers.kinds, chunk_layers, transmittance)
        total = jnp.zeros(receivers.shape[0])
        for profile, areal, point in partition:
            if len(areal) and not point_sources_only:
                direction, _ = _emitter_direction(
                    jnp.take(surfaces.centroid, areal, axis=0)[None], receivers[:, None, :]
                )
                radiance = jnp.take(surfaces.emission, areal) * profile.radiance_per_exitance(
                    direction, jnp.take(surfaces.normal, areal, axis=0)[None]
                )
                projected = projected_solid_angle(
                    receivers[:, None, :],
                    jnp.broadcast_to(receiver_normal[:, None, :], direction.shape),
                    jnp.take(surfaces.vertices, areal, axis=0)[None, ...],
                )
                surviving = _transmittance(
                    absorption,
                    jnp.take(surfaces.centroid, areal, axis=0)[None],
                    receivers[:, None, :],
                ) * _masked(surviving_all, areal)
                lit = radiance * projected * surviving
                if own_facets is not None:
                    # A facet lights nothing in its own plane; at a point inside its triangle the
                    # clip can instead return the whole hemisphere, so the point's own are dropped.
                    own = jnp.any(areal[None, :, None] == own_facets[:, None, :], axis=-1)
                    lit = jnp.where(own, 0.0, lit)
                total = total + jnp.sum(lit, axis=1)
            if len(point):
                centroid = jnp.take(surfaces.centroid, point, axis=0)
                direction, distance_squared = _emitter_direction(
                    centroid[None, :, :], receivers[:, None, :]
                )
                receiver_cosine = jnp.maximum(-dot(direction, receiver_normal[:, None, :]), 0.0)
                fraction = profile.intensity_fraction(
                    direction, jnp.take(surfaces.normal, point, axis=0)[None, :, :]
                )
                surviving = _transmittance(
                    absorption, centroid[None], receivers[:, None, :]
                ) * _masked(surviving_all, point)
                total = total + jnp.sum(
                    jnp.take(surfaces.power, point)
                    * fraction
                    * receiver_cosine
                    * surviving
                    / distance_squared,
                    axis=1,
                )
        return total

    return in_passes(
        ((points, 0), (normals, 0), *named, *layers), pair_limit, surfaces.n_facets, at
    )


def _own_facet_rows(receiver_facet, n_points: int, n_facets: int) -> np.ndarray:
    """The facets each receiver lies on, as ``(n_points, k)`` rows padded with ``-1``, checked."""
    rows = np.asarray(receiver_facet)
    if not np.issubdtype(rows.dtype, np.integer):
        msg = f"receiver_facet must hold integer facet indices; got dtype {rows.dtype}"
        raise ValueError(msg)
    rows = rows[:, None] if rows.ndim == 1 else rows
    if rows.ndim != 2 or rows.shape[0] != n_points:
        msg = (
            f"receiver_facet must have one row per point, shape ({n_points},) or ({n_points}, k); "
            f"got {np.shape(receiver_facet)}"
        )
        raise ValueError(msg)
    if rows.size and (rows.min() < -1 or rows.max() >= n_facets):
        msg = f"receiver_facet must name facets in [0, {n_facets}) or be -1"
        raise ValueError(msg)
    return rows
