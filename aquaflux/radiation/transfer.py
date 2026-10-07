"""The transfer matrix: what fraction of what leaves each facet reaches every other one.

Pure geometry. Nothing here knows what any surface emits, how much it reflects, or what the
medium between them absorbs — those are supplied per call to the solve in
:mod:`aquaflux.radiation.model`, which is what lets a design study pay this ``n^2`` build once
and sweep the optics against it with the derivatives intact.

**Both facets of every pair are integrated over, and they are integrated differently.** A
transfer factor is a double area integral. The sending facet is exact — the projected solid angle
is a closed form — and the receiving facet is quadrature, six points by default. Keeping the
source exact is what makes the row sums exact at every rule, and the row sums are what bound the
conditioning; integrating both by quadrature would trade that for exact reciprocity, which is the
worse way round. See :func:`build_transfer` for the point counts on offer and what each costs.

⚠️ **Refining the mesh does not improve reciprocity; only the quadrature does.** Shrinking the
facets of a closed box brings each one's neighbours proportionally closer, so the ratio the
quadrature error depends on never changes: ``reciprocity_residual`` measures 0.2421, 0.0281,
0.0078 and 0.0045 at 1, 3, 6 and 12 points per receiver, and the same four numbers again at every
refinement from 12 to 432 facets. It is a diagnostic and not a gate.

⚠️ **Global energy conservation follows reciprocity, and at one point per receiver it DEGRADES
under refinement.** For a small lamp in a large box, absorbed over emitted measures 1.000000,
0.999868, 0.982769, 0.975625 and 0.973077 at 12, 48, 192, 432 and 768 facets — 2.7% of the lamp's
output unaccounted for and still growing, because the facets nearest the lamp close in on it
while staying the same size relative to their separation. At the six-point default the same scene
gives 1.000000, 1.000212, 1.000171, 1.000120 and 1.000102, improving instead. ⚠️ A box in which
*every* facet emits balances to 1.000000 at every mesh and every rule, so it cannot see any of
this: each facet's error is its neighbour's and they cancel identically.

⚠️ **The SOURCE's angular distribution is still evaluated at ONE direction**, from its centroid to
the receiver's, and the receiver quadrature does not fix that. It matters only when the source is
not Lambertian — a Lambertian distribution cancels against the projected solid angle and balances
exactly at every refinement. A cosine-power source of exponent 8 balances at 1.086, 0.978, 0.982
and 0.987 at 12, 48, 192 and 432 facets, against 1.145, 0.970, 0.970 and 0.977 at one point per
receiver: the two agree to three figures from three points upward, because this error is not the
receiver's. What shrinks it is refining the mesh, which samples more directions. Subdivide a
narrow source's surroundings, or read its result as carrying a few percent of slack.

The same applies to the other three quantities that multiply the geometric term elementwise — the
centroid separation carrying absorption, the source cosine above, and the occlusion mask. All
three are one-point, necessarily: they are live and differentiable, so they cannot move inside the
frozen build. In a scene where a facet is partly shadowed, or the medium absorbs appreciably over
a facet's own width, those are the coarse approximations and not the receiver quadrature.

**``F`` is built from the PROJECTED solid angle.** A receiving facet is a surface, so light
arriving obliquely counts for less; the plain solid angle is the fluence-rate kernel and using
it here overstates every transfer by exactly a factor of two over a closed enclosure. That
distinction is the one this package has got wrong most often, and the row-sum metric below is
what catches it.

**What is frozen and what is live** is the other thing to get right, and both halves have been
wrong at some point in this design:

===========================  ==================================================================
frozen, built once           the projected solid angles, the source-side cosines, the
                             centroid separations, and the occlusion mask -- all ``n^2``
live, differentiated         reflectance, emission, radiant power, profile parameters,
                             occluder transmittance, and the absorption coefficient
===========================  ==================================================================

Freezing too much severs a gradient the module promises, and the loss is neither obvious nor
total: freezing the whole of ``F`` costs a few percent of the sensitivity to absorbance, and
freezing the visibility inside the geometry term costs about two thirds of the sensitivity to
transmittance. Both leave a finite, plausible-looking number behind. The rule that prevents it:
**anything promised a gradient must be computed outside the frozen arrays**, as an elementwise
multiply against them.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numba
import numpy as np

from aquaflux.morton import morton_order
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.clipping import _SLACK
from aquaflux.radiation.images import plane_exchange
from aquaflux.radiation.lit_blocks import rounded_width
from aquaflux.radiation.mirror_visibility import (
    MirrorVisibility,
    build_mirror_masks,
    reflected_surviving,
)
from aquaflux.radiation.mirrors import Mirror, planar_mirrors
from aquaflux.radiation.profiles import AxisymmetricProfile, Lambertian
from aquaflux.radiation.quadrature import TriangleQuadrature, triangle_quadrature
from aquaflux.radiation.self_occlusion import SelfOcclusion
from aquaflux.radiation.solid_angle import projected_solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, in_passes
from aquaflux.vectors import dot

#: The most planes one specular body may split into before it is refused as curved. Each plane is a
#: mirror whose images cost a gather of their own -- several times a direct one even with each image
#: clipped only against the aperture facets it overlaps, as measured on a reactor's flat end plates by
#: ``validation/sozzi_radiation/specular_cost.py`` -- so the bound keeps a model's cost within some
#: tens of direct gathers per body. Twelve admits any box and a body with several bends; a curved
#: body, tessellated into many strips, exceeds it and belongs to an exact curved-mirror method.
MAX_MIRROR_PLANES = 12

#: Points per receiving facet in the default transfer build. Six is where the measured
#: cost/accuracy frontier turns over -- see :func:`build_transfer`.
_DEFAULT_RECEIVER_POINTS = 6

__all__ = [
    "TransferMatrix",
    "build_transfer",
    "reciprocity_residual",
    "row_sum_error",
]


class TransferMatrix(eqx.Module):
    """The frozen geometry of facet-to-facet transfer.

    Everything here is a function of shape and position alone, costs ``n^2`` to build, and never
    changes while a design study varies emission, reflectance or the medium's absorbance. It is held
    apart from those so the expensive build happens once and the derivatives still reach the
    things that move.

    Attributes
    ----------
    geometric : jnp.ndarray, shape ``(n_facets, n_facets)``
        The form factor from facet ``i`` to facet ``j`` — the fraction of what leaves ``i``
        Lambertian that lands on ``j``, before any blocking or absorption, with the diagonal
        zeroed. Equivalently, and this is how the solve reads it, the weight with which ``j``'s
        radiosity contributes to the irradiance on ``i``: those are the same number, not two
        related by reciprocity. **The area therefore belongs to the ROW index** — ``A_i``
        multiplies row ``i`` — which is what :func:`reciprocity_residual` weights by, and what
        an equal-area fixture cannot distinguish from the transpose.
    source_cosine : jnp.ndarray, shape ``(n_facets, n_facets)``
        Cosine, at source ``j``, of the angle between its normal and the direction to receiver
        ``i``. Needed only to evaluate a non-Lambertian source's own angular distribution, which
        is live because its parameters are.
    separation : jnp.ndarray, shape ``(n_facets, n_facets)``
        Centroid-to-centroid distance, which is what turns a uniform absorption coefficient into
        a transmittance without re-walking any geometry.
    visibility : Visibility
        The frozen blocking mask, built for the facet centroids as receivers.
    specular_solids : tuple of str
        The bodies that reflect specularly, in the order of the specular arrays' first axis.
        Empty for a transfer with no specular reflection, which is then exactly the diffuse one.
    mirrors : tuple of Mirror
        Every plane those bodies reflect in, across all of them.
    specular_terms : tuple of (int, tuple of int)
        What each entry of the specular arrays' first axis stands for: the specular body, by its
        index in :attr:`specular_solids`, and the pattern of what its paths cross -- how many of
        the two legs each analytic body lies across, in the order of the bodies the visibility was
        built with. Paths are grouped by pattern so that the bodies' transmittance stays live:
        every path of one term is filtered by the same powers of the same transmittances. With
        nothing in the way there is one term per body, with an empty pattern.
    specular_geometric : jnp.ndarray, shape ``(n_terms, n_facets, n_facets)``
        Per term, the transfer from facet ``j`` to facet ``i`` by one bounce in any of its body's
        planes, at unit reflectance and before any transmittance:
        :attr:`~aquaflux.radiation.images.PlaneExchange.geometric` summed over the planes, over
        the pairs whose path has the term's pattern. A path the surface's own triangles stand
        across carries nothing and is in no term. The body's specular reflectance and the
        pattern's transmittance multiply it live.
    specular_separation, specular_source_cosine : jnp.ndarray, shape ``(n_terms, n_facets, n_facets)``
        The unfolded path length and the source cosine of each pair, averaged over the planes in
        proportion to what each carries. Exact where one plane carries the pair; where several
        reach it along different paths, a one-point value in the same sense the centroid
        separation is for the direct transfer.
    point_shadows : tuple of MirrorVisibility, or None
        What stands across the paths from each point source to each facet, one mask per mirror
        in the order of :attr:`mirrors` -- the reflected counterpart of :attr:`visibility` for the
        light the transfer cannot carry, since a point source has no area in it. ``None`` where
        nothing can stand in the way, or no source is a point.
    """

    geometric: jnp.ndarray
    source_cosine: jnp.ndarray
    separation: jnp.ndarray
    visibility: Visibility
    specular_geometric: jnp.ndarray | None = None
    specular_separation: jnp.ndarray | None = None
    specular_source_cosine: jnp.ndarray | None = None
    mirrors: tuple[Mirror, ...] = ()
    point_shadows: tuple[MirrorVisibility, ...] | None = None
    specular_solids: tuple[str, ...] = eqx.field(static=True, default=())
    specular_terms: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True, default=())

    @property
    def n_facets(self) -> int:
        """Number of facets."""
        return int(self.geometric.shape[0])

    def assemble(
        self, surfaces, absorption=None, transmittance=None, *, pair_limit=DEFAULT_PAIR_LIMIT
    ):
        """The reflected and emitted transfer matrices, for one set of optical values.

        Both are elementwise products against the frozen arrays, which is what keeps a
        derivative with respect to transmittance or absorbance from needing the ``n^2`` build
        again. They are the same array object whenever every areal source is Lambertian, which
        is the reduction that pins the profile constants.

        Parameters
        ----------
        surfaces : Surfaces
            Read for its angular distributions and, for a non-uniform medium, its centroids --
            which must therefore be the ones the matrix was built from.
        absorption : Absorption, optional
            The medium between facets. A uniform coefficient goes through the frozen
            separations in closed form; anything else re-walks every pair.
        transmittance : array_like, shape ``(n_occluders,)``, optional
            What each analytic body lets through. Defaults to opaque.
        pair_limit : int, optional
            Facet pairs one pass of a non-uniform medium's walk may form. The walk is the only
            part of this that visits geometry again, and formed whole it is several of its own
            working arrays per pair across all ``n^2`` pairs at once; walked a block of receiving
            facets at a time, it is that for one block. Unused for a uniform medium.

        Returns
        -------
        tuple of (jnp.ndarray, jnp.ndarray)
            ``F`` and ``F^M``, each ``(n_facets, n_facets)``: the weight carrying a facet's
            *reflected* output, which leaves Lambertian, and the weight carrying its own
            *emission*, which leaves with its own distribution. Each includes what reaches a facet
            by one specular bounce, weighted by the bouncing body's specular reflectance.

        Raises
        ------
        ValueError
            If a body not built as specular has a specular reflectance, or one built as specular
            has different specular reflectances on different facets.
        TypeError
            If the specular reflectance is traced and no body was built as specular.
        NotImplementedError
            If there are specular bodies and the medium is graded, which their frozen path
            lengths cannot carry.
        """
        transmittance = (
            jnp.zeros(self.visibility.n_occluders) if transmittance is None else transmittance
        )
        surviving = self.visibility.surviving(transmittance)
        if absorption is None:
            through = 1.0
        elif isinstance(absorption, UniformAbsorption):
            # Closed form off the frozen separation: no geometry is revisited, and the derivative
            # with respect to the coefficient is exact.
            through = jnp.exp(-absorption.coefficient * self.separation)
        else:
            centroid = jnp.asarray(surfaces.centroid)
            through = in_passes(
                ((centroid, 0),),
                pair_limit,
                self.n_facets,
                lambda receiving: jnp.exp(
                    -absorption.optical_depth(centroid[None, :, :], receiving[:, None, :])
                ),
            )
        common = self.geometric * surviving * through
        mirrored = self._mirrored(surfaces, absorption, transmittance)

        # The emitted component leaves with each source's own distribution; the reflected component
        # leaves Lambertian by assumption. For a Lambertian source the two coincide exactly, which
        # is worth keeping as the reduction that pins the profile constants.
        lambertian = all(
            isinstance(profile, Lambertian)
            for kind, profile in enumerate(surfaces.profiles)
            if np.any((np.asarray(surfaces.profile_index) == kind) & ~surfaces.is_point_source)
        )
        reflected = common + sum(term for term, _ in mirrored)
        if lambertian:
            return reflected, reflected
        emitted = common * self._relative_radiance(surfaces, self.source_cosine)
        for term, cosine in mirrored:
            emitted = emitted + term * self._relative_radiance(surfaces, cosine)
        return reflected, emitted

    def _mirrored(self, surfaces, absorption, transmittance) -> list:
        """Each specular term's live transfer, with the source cosine its emission is read at.

        Returns
        -------
        list of (jnp.ndarray, jnp.ndarray)
            Per term, ``rho_s * surviving * G * attenuation`` and the mean source cosine, each
            ``(n_facets, n_facets)``: the body's reflectance, its pattern's transmittance, and the
            medium along the unfolded path.
        """
        reflectance = _specular_by_solid(surfaces, self.specular_solids)
        if not self.specular_solids:
            return []
        if absorption is not None and not isinstance(absorption, UniformAbsorption):
            msg = (
                "specular reflection between facets is carried along frozen unfolded path "
                "lengths, which a uniform medium attenuates in closed form but a graded one "
                "cannot: its paths have to be walked through the fluid, and which mirror plane "
                "each one breaks at is not kept. Use a UniformAbsorption, or build the model "
                "without specular bodies."
            )
            raise NotImplementedError(msg)
        terms = []
        for index, (solid, crossings) in enumerate(self.specular_terms):
            surviving = reflected_surviving(
                jnp.asarray(crossings, dtype=jnp.uint8).reshape(-1), None, transmittance
            )
            term = reflectance[solid] * surviving * self.specular_geometric[index]
            if absorption is not None:
                term = term * jnp.exp(-absorption.coefficient * self.specular_separation[index])
            terms.append((term, self.specular_source_cosine[index]))
        return terms

    def _relative_radiance(self, surfaces, cosine):
        """Each areal source's radiance at ``cosine``, relative to a Lambertian one: ``pi L / M``.

        Point sources are left at zero: they are absent from the transfer, and asking one for a
        radiance is a category error it refuses rather than answers.

        Parameters
        ----------
        surfaces : Surfaces
        cosine : jnp.ndarray, shape ``(n_facets, n_facets)``
            The source cosine of each pair, column ``j`` at source ``j``.
        """
        index = np.asarray(surfaces.profile_index)
        areal = ~surfaces.is_point_source
        relative = jnp.zeros_like(cosine)
        for kind, profile in enumerate(surfaces.profiles):
            sources = np.flatnonzero((index == kind) & areal)
            if not len(sources):
                continue
            if not isinstance(profile, AxisymmetricProfile):
                msg = (
                    f"{type(profile).__name__} depends on more than the angle from the facet "
                    "normal, and the transfer freezes only that angle's cosine for each pair of "
                    "facets, so it cannot carry this source's emission to the other surfaces. "
                    "The direct gathers take any profile; for the surface transfer, give these "
                    "facets an AxisymmetricProfile."
                )
                raise NotImplementedError(msg)
            weight = jnp.pi * profile.radiance_per_exitance_at(jnp.take(cosine, sources, axis=1))
            relative = relative.at[:, sources].set(weight)
        return relative


class _Geometry(eqx.Module):
    """What the row blocks read of the facets, bundled so the compiled block takes one argument."""

    vertices: jnp.ndarray
    centroid: jnp.ndarray
    normal: jnp.ndarray
    areal: jnp.ndarray


#: A projected solid angle, in steradians, below which a pair is left to be zero: one rounding of a
#: transfer factor of order one. A sending triangle whose vertices all lie within rounding of a
#: receiver's plane -- the strips of a faceted cylinder, whose normals are only as exact as their
#: short edges allow -- comes out a few parts in ``1e14`` of its distance in front of it, and the
#: kernel returns a sliver of order ``1e-17`` for it; this is what lets such a pair be skipped.
_NEGLIGIBLE = 2.0**-52

#: How close to a receiver's point, relative to the lengths involved, a point must come to count as
#: lying on a facet in the receiver's own plane. Generous on purpose: a facet judged to hold the
#: point is kept and evaluated, so the tolerance can only make the test keep more.
_ON_FACET_TOLERANCE = 1e-9


@jax.jit
def _row_block(geometry: _Geometry, sample, weight, index, columns):
    """Every frozen quantity for the receiving facets ``index``, the costly one at ``columns`` only.

    One compiled pass, so nothing it forms is larger than ``len(index) x n_facets`` (or that times
    three for the offsets), and it is the only place each of the three quantities is computed.
    The geometric term, the costly one, is evaluated only against the sending facets in
    ``columns`` and scattered into rows of the full width; every other facet lies behind or in the
    plane of every receiver of the block, where the term is zero (:func:`_columns_in_front`). The
    source cosine and the separation are cheap and are formed against every facet.
    """
    facing = geometry.normal[index]
    triangles = geometry.vertices[columns]

    def accumulate(total, sampled):
        weight_q, point_q = sampled
        at_point = jax.vmap(
            lambda point, facing_i: jax.vmap(
                lambda triangle: projected_solid_angle(point, facing_i, triangle)
            )(triangles)
        )(point_q, facing)
        return total + weight_q * at_point, None

    # Scanned rather than vmapped over the quadrature points so the live intermediate stays
    # (rows, columns) whatever the rule costs, which is what lets ``chunk_size`` keep meaning
    # the same thing it did with a single point per facet.
    solid, _ = jax.lax.scan(
        accumulate,
        jnp.zeros((index.shape[0], columns.shape[0])),
        (weight, jnp.swapaxes(sample[index], 0, 1)),
    )
    # A facet cannot transfer to itself: every quadrature point lies in its own plane, where the
    # contour integral returns a whole hemisphere. Left in, every row sum is exactly one too
    # large. A planar triangle really does see none of itself, so this is the exact value and
    # not a repair. A point source has no surface to receive on and no area to emit from; it
    # reaches the facets through the ordinary gather instead, as an external irradiance.
    keep = (index[:, None] != columns[None, :]) & (
        geometry.areal[index][:, None] & geometry.areal[columns][None, :]
    )
    geometric = (
        jnp.zeros((index.shape[0], geometry.vertices.shape[0]))
        .at[:, columns]
        .set(jnp.where(keep, solid / jnp.pi, 0.0))
    )

    offset = geometry.centroid[index][:, None, :] - geometry.centroid[None, :, :]
    separation_squared = dot(offset, offset)
    separation = jnp.sqrt(jnp.where(separation_squared == 0.0, 0.0, separation_squared))
    safe = jnp.where(separation == 0.0, 1.0, separation)
    source_cosine = dot(offset, geometry.normal[None, :, :]) / safe
    return geometric, source_cosine, separation


@functools.partial(jax.jit, donate_argnums=0)
def _written(buffers, block, index):
    """``buffers`` with ``block`` written at rows ``index``, in place: the buffers are donated."""
    return tuple(buffer.at[index].set(part) for buffer, part in zip(buffers, block, strict=True))


def _columns_in_front(vertices, sample, normal, areal, index) -> np.ndarray:
    """The areal facets that can send anything to some areal receiver in ``index``.

    The kernel clips a sending triangle to the closed half-space in front of the receiving point,
    after snapping any vertex height within rounding of the plane to exactly zero
    (:func:`~aquaflux.radiation.clipping.decidable_heights`). So where **no** vertex is strictly in
    front, what survives the clip lies in the receiver's own plane, where every direction has zero
    obliquity: the projected solid angle is zero, and the kernel returns it as zero or as a
    rounding of the zero that the contour sum of an in-plane loop is. Such a facet is left out.
    That covers a facet wholly behind the plane and, around a faceted cylinder, the far larger
    number lying **in** it -- every triangle of a flat strip is coplanar with the others -- which
    a test for "strictly behind" kept and evaluated to produce zeros. A facet with a vertex in
    front is kept unless a bound on what it can send (:func:`_projected_bound`) is below
    :data:`_NEGLIGIBLE`: in floating point a coplanar strip is never quite coplanar, and without
    this its far members would all be kept for slivers of order ``1e-17``. So a skipped pair's
    transfer factor is zero or under one rounding of a factor of order one -- not bit for bit the
    kernel's, which returns such a pair as dust of either sign's size.

    ⚠️ **Except where the point lies on that in-plane part.** A receiving point inside a coplanar
    triangle sees it as a whole hemisphere (the convention that makes each facet exclude itself),
    and one on a shared edge or vertex is on the boundary of the contour. A valid mesh has
    neither -- quadrature points are interior to their own facet -- but a degenerate triangle's
    points need not be, so any such pair is kept and left to the kernel. The heights are snapped
    with the clip's own bound, so the two agree on which vertices are in the plane.

    Host arithmetic over concrete positions, one block at a time; the loop exits at the first
    point that keeps a facet, so a facet in front of a block costs little and one behind or beside
    every receiver costs the whole block.
    """
    rows = index[areal[index]]
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    keep = _any_in_front(
        np.ascontiguousarray(vertices, dtype=float),
        np.ascontiguousarray(sample[rows], dtype=float),
        np.ascontiguousarray(normal[rows], dtype=float),
        np.ascontiguousarray(areal, dtype=np.bool_),
        _SLACK,
        _ON_FACET_TOLERANCE,
    )
    return np.flatnonzero(keep)


@numba.njit(parallel=True)
def _any_in_front(vertices, points, facing, areal, slack, tolerance):
    """Per facet, whether any of ``points`` (with its receiver's normal) can receive from it."""
    n_facets = vertices.shape[0]
    keep = np.zeros(n_facets, dtype=np.bool_)
    for facet in numba.prange(n_facets):
        if not areal[facet]:
            continue
        triangle = vertices[facet]
        for row in range(points.shape[0]):
            normal = facing[row]
            for point in points[row]:
                if _can_receive(triangle, point, normal, slack, tolerance):
                    keep[facet] = True
                    break
            if keep[facet]:
                break
    return keep


@numba.njit
def _can_receive(triangle, point, normal, slack, tolerance):
    """Whether the kernel can return more than a rounding of zero for this triangle at this point."""
    on_plane = np.zeros(3, dtype=np.bool_)
    highest = 0.0
    for k in range(3):
        height = 0.0
        bound = 0.0
        for d in range(3):
            offset = triangle[k, d] - point[d]
            height += offset * normal[d]
            bound += abs(offset) * abs(normal[d])
        if abs(height) <= slack * bound:
            on_plane[k] = True
        elif height > highest:
            highest = height
    if highest > 0.0:
        return _projected_bound(triangle, point, highest) > _NEGLIGIBLE
    count = on_plane.sum()
    if count == 0:
        return False
    if count == 3:
        return _on_triangle(triangle, point, tolerance)
    if count == 2:
        first = 0 if on_plane[0] else 1
        second = 2 if on_plane[2] else 1
        return _on_segment(triangle[first], triangle[second], point, tolerance)
    for k in range(3):
        if on_plane[k]:
            return _near(triangle[k], point, tolerance)
    return True


@numba.njit
def _projected_bound(triangle, point, highest):
    """An upper bound on the projected solid angle of a triangle rising at most ``highest`` in front.

    Every point of the triangle is at least ``d`` from the receiving point, taken as the distance
    to its centroid less the radius of a sphere about the centroid holding its vertices, so it
    subtends at most ``area / d**2``; and every direction to it has an obliquity of at most
    ``highest / d``. Where the triangle may come nearer than that sphere allows, there is no bound
    and infinity is returned.
    """
    centroid = (triangle[0] + triangle[1] + triangle[2]) / 3.0
    radius = 0.0
    for k in range(3):
        radius = max(radius, np.sqrt(((triangle[k] - centroid) ** 2).sum()))
    distance = np.sqrt(((centroid - point) ** 2).sum()) - radius
    if distance <= 0.0:
        return np.inf
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    area = 0.5 * np.sqrt(
        (a[1] * b[2] - a[2] * b[1]) ** 2
        + (a[2] * b[0] - a[0] * b[2]) ** 2
        + (a[0] * b[1] - a[1] * b[0]) ** 2
    )
    return area * highest / distance**3


@numba.njit
def _near(vertex, point, tolerance):
    """Whether ``point`` is ``vertex``, to ``tolerance`` of their size."""
    distance = 0.0
    scale = 0.0
    for d in range(3):
        distance += (vertex[d] - point[d]) ** 2
        scale += vertex[d] ** 2 + point[d] ** 2
    return distance <= (tolerance**2) * scale


@numba.njit
def _on_segment(start, end, point, tolerance):
    """Whether ``point`` lies on the segment from ``start`` to ``end``, to ``tolerance`` of its size."""
    along = 0.0
    length = 0.0
    for d in range(3):
        along += (point[d] - start[d]) * (end[d] - start[d])
        length += (end[d] - start[d]) ** 2
    if length == 0.0:
        return _near(start, point, tolerance)
    fraction = min(1.0, max(0.0, along / length))
    distance = 0.0
    for d in range(3):
        distance += (start[d] + fraction * (end[d] - start[d]) - point[d]) ** 2
    return distance <= (tolerance**2) * length


@numba.njit
def _on_triangle(triangle, point, tolerance):
    """Whether ``point``, in the triangle's plane, lies inside it or on its boundary.

    Each edge's side test is the triple product of the edge, the point's offset from the edge's
    start and the triangle's own area vector, which is positive for every edge when the point is
    inside whichever way the triangle is wound. A triangle of no area has no inside to test and is
    reported as holding the point, so that the pair is kept.
    """
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    area = np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )
    area_squared = (area**2).sum()
    if area_squared == 0.0:
        return True
    for k in range(3):
        edge = triangle[(k + 1) % 3] - triangle[k]
        offset = point - triangle[k]
        cross = np.array(
            [
                edge[1] * offset[2] - edge[2] * offset[1],
                edge[2] * offset[0] - edge[0] * offset[2],
                edge[0] * offset[1] - edge[1] * offset[0],
            ]
        )
        side = (cross * area).sum()
        scale = np.sqrt((edge**2).sum() * (offset**2).sum() * area_squared)
        if side < -tolerance * scale:
            return False
    return True


def _row_blocks(geometry: _Geometry, sample, weight, rows: int):
    """The three ``n x n`` frozen arrays, filled block by block into buffers of their final size.

    ⚠️ **The whole-matrix temporaries are the cost, not the stored arrays**, and on this backend
    freed memory is not handed back, so a build's peak is the running total of everything it ever
    formed. The first version assembled each quantity whole — rows collected and concatenated, then
    masked in two more full-size passes, the separation from an ``(n, n, 3)`` offset array — and
    peaked at about eighteen of its own ``n x n`` arrays: 8.3 GB against 0.45 GB each for a
    7,516-facet lamp. Here each block is computed once, by one compiled pass, and written into its
    place by an update that donates the buffer, so nothing larger than a block exists besides the
    three arrays being kept.

    **The blocks are cut along a space-filling curve, and each evaluates the costly term only
    against the facets that can lie in front of it.** A facet with no vertex in front of a receiver's
    plane contributes nothing to that receiver (:func:`_columns_in_front`), and around a convex
    lamp that is every pair but those of a degenerate facet; a block of neighbouring receivers shares most of its planes'
    fronts, so its list is short, where a block of facets in storage order may span the whole
    surface and need nearly every column. The lists are padded to a ladder of a few widths per
    doubling, never past the facet count (:func:`~aquaflux.radiation.lit_blocks.rounded_width`),
    so the build compiles a handful of programs however the lists fall, and a list nearly as long
    as the surface costs no more than the full row.

    The last block is filled out by repeating its last receiver, and each list by repeating its
    last facet; a repeated entry is the same pair, written the same value again.
    """
    n_facets = int(geometry.vertices.shape[0])
    if n_facets == 0:
        empty = jnp.zeros((0, 0))
        return empty, empty, empty
    rows = max(1, min(int(rows), n_facets))
    vertices = np.asarray(geometry.vertices, dtype=float)
    points = np.asarray(sample, dtype=float)
    normal = np.asarray(geometry.normal, dtype=float)
    areal = np.asarray(geometry.areal, dtype=bool)
    order = morton_order(np.asarray(geometry.centroid, dtype=float))
    buffers = tuple(jnp.zeros((n_facets, n_facets)) for _ in range(3))
    for start in range(0, n_facets, rows):
        index = order[start : start + rows]
        index = np.concatenate([index, np.repeat(index[-1:], rows - len(index))])
        columns = _columns_in_front(vertices, points, normal, areal, index)
        if len(columns) == 0:
            columns = index[:1]
        width = rounded_width(len(columns), n_facets)
        columns = np.concatenate([columns, np.repeat(columns[-1:], width - len(columns))])
        index, columns = jnp.asarray(index), jnp.asarray(columns)
        block = _row_block(geometry, sample, weight, index, columns)
        buffers = _written(buffers, block, index)
    return buffers


def _specular_by_solid(surfaces, solids) -> jnp.ndarray:
    """Each specular body's specular reflectance, checked against the per-facet values.

    The transfer carries one live reflectance per specular body, so a value that varies over a
    body's facets, or one on a body not built as specular, would be silently replaced or dropped.
    Both are refused where the values can be read. A traced reflectance cannot be read: it is
    taken from each body's first facet, and refused outright when no body was built as specular,
    since a derivative with respect to it would then read zero while a mirror sends light on.

    Parameters
    ----------
    surfaces : Surfaces
    solids : tuple of str
        The specular bodies, in the transfer's order.

    Returns
    -------
    jnp.ndarray, shape ``(len(solids),)``
    """
    specular = surfaces.specular_reflectance
    solid_id = np.asarray(surfaces.solid_id)
    members = [np.flatnonzero(solid_id == surfaces.solid_names.index(name)) for name in solids]
    if isinstance(specular, jax.core.Tracer):
        if not solids:
            msg = (
                "specular_reflectance is traced, but no body was built as specular, so the model "
                "carries no specular reflection and its derivative with respect to that "
                "reflectance would read zero. Build the model with the reflecting bodies named "
                "as specular, or keep the reflectance out of the traced values."
            )
            raise TypeError(msg)
        return jnp.take(specular, jnp.asarray([facets[0] for facets in members]))
    values = np.asarray(specular, dtype=float)
    declared = np.zeros(values.shape, dtype=bool)
    for name, facets in zip(solids, members, strict=True):
        declared[facets] = True
        if np.any(values[facets] != values[facets[0]]):
            msg = (
                f"body {name!r} has specular reflectances from {values[facets].min()} to "
                f"{values[facets].max()}, but a specular body reflects with one value: give its "
                "facets the same specular_reflectance"
            )
            raise ValueError(msg)
    stray = np.flatnonzero(~declared & (values != 0.0))
    if stray.size:
        bodies = sorted({surfaces.solid_names[int(solid_id[f])] for f in stray})
        msg = (
            f"bodies {bodies} have a specular reflectance but were not built as specular, so the "
            "light they would reflect as mirrors would be dropped, leaving the field darker than "
            "the walls make it. Name them as specular when building, or give their reflectance "
            "as diffuse_reflectance."
        )
        raise ValueError(msg)
    return jnp.asarray([values[facets[0]] for facets in members])


def build_transfer(
    surfaces: Surfaces,
    *,
    occluders=(),
    self_occlusion: SelfOcclusion | None = None,
    receiver_quadrature: TriangleQuadrature | int | None = None,
    chunk_size: int = 256,
    specular: Sequence[str] = (),
    max_mirror_planes: int = MAX_MIRROR_PLANES,
    **visibility_options,
) -> TransferMatrix:
    """Compute, once, the geometry of transfer between every pair of facets.

    Parameters
    ----------
    surfaces : Surfaces
        The facets. Only their geometry is read; the optical properties are supplied per call.
    occluders : sequence of aquaflux.solids.Body, optional
        Analytic bodies between facets.
    self_occlusion : SelfOcclusion or None, optional
        How the facets are tested for blocking one another, which for a non-convex body they do.
        Defaults to one ray per pair; pass
        :class:`~aquaflux.radiation.self_occlusion.SilhouetteOcclusion` for an exact fraction.
        ``None`` here means "use the default", not "switch it off" -- to switch it off, build
        the mask yourself with :func:`~aquaflux.radiation.visibility.build_visibility`.
    receiver_quadrature : TriangleQuadrature or int, optional
        How finely to integrate over each *receiving* facet. An integer is a point count passed
        to :func:`~aquaflux.radiation.quadrature.triangle_quadrature`. Defaults to six points.
        It is the only knob that improves reciprocity, and it costs far less than its point
        count suggests — see the tables below.
    chunk_size : int, optional
        Receiving facets per block of the build, bounding the working set beside the arrays kept.
        Its meaning is unchanged by the quadrature: the points are accumulated one at a time
        within a block, so a finer rule costs time and not memory.
    specular : sequence of str, optional
        Bodies, by name, that reflect specularly. Each is split into the planes its facets lie in
        (:func:`~aquaflux.radiation.mirrors.planar_mirrors`), and one bounce in any of them is
        carried between facets. Their specular reflectance is supplied per call, one value per
        body. Both legs of each reflected path are tested against the occluders and the surface's
        own triangles, one ray per pair as the direct mask's are
        (:func:`~aquaflux.radiation.mirror_visibility.build_mirror_visibility`), whichever
        ``self_occlusion`` strategy is in force.
    max_mirror_planes : int, optional
        The most planes one specular body may split into. A curved body is one plane per facet
        strip, each a mirror whose images cost a gather of their own and which together describe
        a curved mirror poorly; a body past this is refused rather than built slowly and wrongly.
    **visibility_options
        Passed through to the visibility build.

    Returns
    -------
    TransferMatrix

    Raises
    ------
    ValueError
        If a specular body splits into more than ``max_mirror_planes`` planes.

    Notes
    -----
    A transfer factor is a double area integral, over the sending facet and over the receiving
    one. The sending half is exact — :func:`~aquaflux.radiation.solid_angle.projected_solid_angle`
    is a closed form — so the receiving half is the whole of the discretization error, and the
    quadrature above is what controls it. Two consequences of keeping the source exact are worth
    stating, because they are what make a one-sided rule the right shape here rather than a
    half-finished one:

    - **The row sums stay exact at every point count.** Each quadrature point sees a closed
      enclosure, so its own row sums to one; the weights sum to one, so the average does too.
      Measured at 1e-15 for every rule. Integrating *both* facets by quadrature would instead
      give exact reciprocity and approximate row sums — the worse trade, since the row sum is
      what bounds the conditioning and what catches a wrong kernel.
    - **Reciprocity converges in the point count, not in the mesh.** Refining a closed box
      brings each facet's neighbours proportionally closer, so the ratio the error depends on
      does not change and the residual sits flat under refinement. Measured on boxes of 12 to
      432 facets, identically at each:

      ======  ======  ====================  ==========================================
      points  degree  reciprocity residual  worst column, ``sum_i A_i F_ij / A_j - 1``
      ======  ======  ====================  ==========================================
      1       1       0.2421                8.8%
      3       2       0.0281                1.5%
      6       4       0.0078                0.38%
      12      6       0.0045                0.18%
      ======  ======  ====================  ==========================================

    ⚠️ **A finer rule costs far less than its point count**, which is why six is the default
    rather than one. The build is limited by moving geometry through memory, not by evaluating
    the kernel, and every extra point on a receiver reuses the same triangles. Against the
    one-point build, measured as the median of five warm calls on closed boxes of 192 to 3072
    facets at the default ``chunk_size``, on a CPU backend with x64 on (JAX 0.10.2, macOS arm64,
    11 cores):

    ========  ==========  ==========  ===========  ===========
    facets    1 point     3 points    6 points     12 points
    ========  ==========  ==========  ===========  ===========
    192       0.18 s      1.01x       1.15x        1.09x
    768       0.54 s      1.00x       1.31x        1.60x
    1728      1.32 s      1.18x       1.37x        2.03x
    3072      2.54 s      1.27x       1.72x        3.63x
    ========  ==========  ==========  ===========  ===========

    Read those ratios as the shape and not to two figures: wall clock on a shared desktop
    carries a spread of about 20% run to run, which is wider than the gap between neighbouring
    columns at the small sizes.

    What is *not* integrated over the receiver: the centroid separation that carries absorption,
    the source-side cosine that carries a non-Lambertian profile, and the occlusion mask. All
    three multiply the geometric term elementwise so that they can stay live and differentiable,
    and moving them inside the quadrature would put them back inside the frozen ``n^2`` build.
    They remain one-point quantities, and each carries its own rule of thumb for how fine a
    mesh it needs.

    **Absorption: keep ``a * w`` small**, where ``w`` is a facet's own length scale, the square
    root of its area. The stored separation is one centroid-to-centroid distance, while the
    factor it stands for is the average of ``exp(-a r)`` over the pair under the pair's own
    transfer kernel. To leading order the relative error is ``-a * (mean r - centroid r)``, so it
    is **first** order in ``a * w`` -- not the second-order convexity correction on ``exp``, which
    is swamped. For facets squarely facing one another that excess falls like ``w**2 / (4 d)`` at
    separation ``d``, making the error worst between neighbours, where it reaches about
    ``0.15 * a * w``. At ``a * w = 0.1`` that is under 2%; by ``a * w = 1`` it is past 10% and the
    closed form has stopped describing the scene. Water at 95% ultraviolet transmittance absorbs
    at 5.13 per metre, so 20 mm facets in it sit at ``a * w = 0.1``.

    **A non-Lambertian profile: keep ``(n - 1) * (w / d)**2`` small.** The profile is evaluated
    once per pair, source centroid to receiver centroid, while the solid angle it weights is
    integrated exactly; the mismatch is **second** order in the facet's angular width, about
    ``0.13 * (n - 1) * (w / d)**2`` for a cosine-power source of exponent ``n``. It is
    identically zero for a Lambertian source at every geometry and every distance, so nothing
    pays it unless a non-Lambertian *areal* source is in the scene, and it never touches the
    reflected component, which leaves Lambertian by assumption.

    ⚠️ **Neither error has a fixed sign.** Both run one way for near-axial pairs and the other
    way for oblique ones -- for absorption because the ``1 / r**2`` weighting concentrates on a
    slid-apart pair's facing near corners, until the separation it effectively averages drops
    *below* the centroid-to-centroid one. The two therefore partly cancel in a total over many
    pairs while cancelling not at all in a single local transfer, so a mesh should be sized
    against the per-pair figures above and not against an energy balance, which is much the more
    flattering of the two.

    The build is ``n^2`` in both time and memory: three floating-point arrays of ``n x n`` are kept,
    with a facet-side shadow mask of one boolean and one floating-point array more, so a thousand
    facets is tens of megabytes and ten thousand is gigabytes. The facet count, not the cell count,
    is what limits the surface system. Rows are computed ``chunk_size`` at a time and written in
    place into arrays of their final size, so the build's peak is what it keeps plus one block's
    work and the shadow pass, not a multiple of the matrix.
    """
    if receiver_quadrature is None:
        receiver_quadrature = _DEFAULT_RECEIVER_POINTS
    if not isinstance(receiver_quadrature, TriangleQuadrature):
        receiver_quadrature = triangle_quadrature(int(receiver_quadrature))

    # The build is frozen geometry: nothing downstream differentiates it, so it is cut from any
    # trace at its inputs rather than at its outputs, where it would be the whole matrix.
    geometry = _Geometry(
        vertices=jax.lax.stop_gradient(surfaces.vertices),
        centroid=jax.lax.stop_gradient(surfaces.centroid),
        normal=jax.lax.stop_gradient(surfaces.normal),
        areal=jnp.asarray(~surfaces.is_point_source),
    )
    sample = receiver_quadrature.points(geometry.vertices)
    weight = jnp.asarray(receiver_quadrature.weight)
    geometric, source_cosine, separation = _row_blocks(geometry, sample, weight, chunk_size)
    specular = tuple(dict.fromkeys(specular))
    mirrored = _specular_exchange(
        surfaces,
        specular,
        sample,
        weight,
        functools.partial(
            build_mirror_masks,
            occluders=occluders,
            surfaces=surfaces,
            points=surfaces.centroid,
            # The receivers ARE the facets, so each path's second leg ends on the one it is
            # aimed at and must ignore it, as the direct mask's rays do.
            receiver_facet=np.arange(surfaces.n_facets),
            # Prepared once here, not once per plane: a grid over the triangles is the same for
            # every mirror. Not at all without a mirror, when no reflected path is masked.
            self_occlusion=(
                self_occlusion.prepared(surfaces)
                if self_occlusion is not None and specular
                else self_occlusion
            ),
            **visibility_options,
        ),
        max_mirror_planes,
    )

    return TransferMatrix(
        geometric=geometric,
        source_cosine=source_cosine,
        separation=separation,
        visibility=build_visibility(
            occluders,
            surfaces,
            surfaces.centroid,
            # The receivers here ARE the facets, so each ray ends on the one it is aimed at and
            # must be told to ignore it. Without this every mutually visible pair reads as
            # blocked and a closed enclosure loses its interreflection entirely.
            receiver_facet=np.arange(surfaces.n_facets),
            self_occlusion=self_occlusion,
            **visibility_options,
        ),
        **mirrored,
    )


def _specular_exchange(surfaces, specular, sample, weight, masks, limit):
    """The specular fields of a :class:`TransferMatrix`: per body and crossing pattern, summed.

    Each plane's exchange is split by what stands across each pair's reflected path. A pair the
    surface's own triangles hide carries nothing; the rest are grouped by how many legs each
    analytic body crosses, so the bodies' transmittance can stay live: within a group every path
    is filtered by the same powers of the same transmittances, so the group's planes sum into one
    array that the live factor multiplies as a whole. ``masks`` builds the masks of a sequence of
    mirrors between the facets, or returns ``None`` when nothing can shadow a path.

    Returns
    -------
    dict
        Keyword arguments for :class:`TransferMatrix`, empty when nothing is specular.
    """
    if not specular:
        return {}
    n = surfaces.n_facets
    point_sources = np.flatnonzero(surfaces.is_point_source)
    shadowed = False
    terms: dict[tuple[int, tuple[int, ...]], list] = {}
    mirrors, point_masks = [], []
    for solid, name in enumerate(specular):
        planes = planar_mirrors(surfaces, [name])
        if len(planes) > limit:
            msg = (
                f"specular body {name!r} lies in {len(planes)} planes, past the limit of {limit}: "
                "a curved body is one mirror per flat strip of its facets, which costs a gather "
                "per strip and describes a curved mirror poorly. Name only flat bodies as "
                "specular, or raise max_mirror_planes."
            )
            raise ValueError(msg)
        for mirror in planes:
            exchange = plane_exchange(mirror, surfaces, sample, weight)
            built = masks(mirrors=(mirror,))
            mask = None if built is None else built[0]
            shadowed = built is not None
            for crossings, carried in _by_pattern(exchange.geometric, mask, n):
                total, length, angle = terms.setdefault(
                    (solid, crossings), [jnp.zeros((n, n)), jnp.zeros((n, n)), jnp.zeros((n, n))]
                )
                terms[(solid, crossings)] = [
                    total + carried,
                    length + carried * exchange.separation,
                    angle + carried * exchange.source_cosine,
                ]
            if mask is not None and point_sources.size:
                point_masks.append(mask.for_sources(np.intersect1d(point_sources, mask.sources)))
        mirrors.extend(planes)

    keys = sorted(terms)
    geometric, separation, cosine = [], [], []
    for key in keys:
        total, length, angle = terms[key]
        carried = total > 0.0
        safe = jnp.where(carried, total, 1.0)
        geometric.append(total)
        separation.append(jnp.where(carried, length / safe, 0.0))
        cosine.append(jnp.where(carried, angle / safe, 0.0))
    return {
        "specular_geometric": jnp.stack(geometric),
        "specular_separation": jnp.stack(separation),
        "specular_source_cosine": jnp.stack(cosine),
        "mirrors": tuple(mirrors),
        "specular_solids": specular,
        "specular_terms": tuple(keys),
        "point_shadows": tuple(point_masks) if shadowed and point_sources.size else None,
    }


def _by_pattern(geometric, mask, n: int):
    """One plane's exchange split by the legs each body crosses, with hidden pairs dropped.

    Yields
    ------
    tuple of (tuple of int, jnp.ndarray)
        Each pattern -- legs crossed, per body -- and the exchange of the pairs that have it,
        ``(n, n)``, zero elsewhere. A pattern no carried pair has is not yielded.
    """
    if mask is None:
        yield (), geometric
        return
    crossings = np.zeros((mask.n_occluders, n, n), dtype=np.uint8)
    crossings[:, :, mask.sources] = np.asarray(mask.crossings)
    visible = np.ones((n, n), dtype=bool)
    if mask.hidden is not None:
        visible[:, mask.sources] = ~np.asarray(mask.hidden)
    # One integer per pair naming its pattern: the legs crossed, as digits base three.
    code = np.zeros((n, n), dtype=np.int64)
    for body in range(mask.n_occluders - 1, -1, -1):
        code = 3 * code + crossings[body]
    carried = visible & (np.asarray(geometric) > 0.0)
    for value in np.unique(code[carried]):
        pattern = tuple(int(digit) for digit in _digits(int(value), mask.n_occluders))
        yield pattern, jnp.where(jnp.asarray(carried & (code == value)), geometric, 0.0)


def _digits(value: int, count: int) -> list[int]:
    """``value``'s ``count`` least significant digits in base three, least significant first."""
    digits = []
    for _ in range(count):
        value, digit = divmod(value, 3)
        digits.append(digit)
    return digits


def row_sum_error(transfer: TransferMatrix) -> float:
    """Largest departure of a row of ``F`` from one — the standard figure of merit.

    Every bit of light leaving a facet inside a **closed** enclosure lands somewhere, so each row
    of the transfer matrix sums to one. That identity is what bounds the spectral radius of
    ``diag(rho) F`` by the largest reflectance, and with it the conditioning of the whole system,
    so it is worth reporting on any built matrix rather than assumed.

    On an open surface the rows sum to less than one by the fraction that escapes, and this
    number is then a measure of the opening rather than of an error.
    """
    return float(jnp.max(jnp.abs(jnp.sum(transfer.geometric, axis=1) - 1.0)))


def reciprocity_residual(transfer: TransferMatrix, area) -> float:
    """How far ``A_i F_ij`` sits from ``A_j F_ji``, against the largest entry of the matrix.

    ⚠️ **This is a diagnostic, not a gate.** Reciprocity is exact for the double-area-integral
    form factor, where both facets are integrated over. This matrix integrates the *source*
    exactly and the *receiver* by quadrature, so the residual here is that quadrature's error:
    it falls with the point count, and stays flat under mesh refinement, because refining a
    closed box brings each facet's neighbours proportionally closer and the geometry stays
    self-similar. On boxes of 12 to 432 facets it measures 0.2421, 0.0281, 0.0078 and 0.0045 at
    1, 3, 6 and 12 points per receiver, the same at every refinement.

    So a value here says how differently the near-neighbour entries are apportioned from a fully
    integrated form factor, not whether the matrix is wrong. What is exact at every point count,
    and what the solve's conditioning actually rests on, is the row sum — see
    :func:`row_sum_error`.

    ⚠️ **The area multiplies the ROW index.** ``transfer.geometric[i, j]`` is the form factor
    *from* ``i`` *to* ``j``, so reciprocity pairs ``A_i F_ij`` with ``A_j F_ji`` — the transpose
    of ``area[:, None] * geometric``. Weighting the column instead is the same expression on any
    fixture whose facets share one area, which every closed box built by subdividing a cube
    does; on two squares of area 1 and 100 it reports 0.9999 where the correct pairing reports
    0.0022.

    The comparison is against the largest entry rather than each pair's own magnitude. Pairs
    that transfer nothing — two facets of the same flat wall — hold values around 1e-18, and a
    per-pair relative measure turns that rounding noise into a residual near one while those
    pairs carry, measurably, 0.0000 of the total transfer.
    """
    area = jnp.asarray(area, dtype=float)
    weighted = area[:, None] * transfer.geometric
    largest = jnp.max(jnp.abs(weighted))
    return float(jnp.max(jnp.abs(weighted - weighted.T)) / jnp.where(largest == 0.0, 1.0, largest))
