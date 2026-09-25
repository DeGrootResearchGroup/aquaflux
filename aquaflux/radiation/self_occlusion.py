"""How much of each source a surface's own triangles hide from each receiver.

The emitting surface is the reactor's walls and bodies, so it shadows itself: a bent duct, a
lamp sleeve, a baffle. The geometry doing the blocking *is* the geometry doing the emitting,
which is why it is answered here rather than by a body standing in the light.

⚠️ **This is the fallback, not the first choice.** Where the shape has an analytic description —
and a vessel usually does — describing the fluid it holds and letting
:class:`~aquaflux.solids.Outside` decide is both exact and orders of magnitude
cheaper, because it is a formula rather than a search over triangles. What belongs here is a
surface with no such description, or one whose description has not been written down.

Two strategies answer it, and **neither dominates the other**, which is why both are kept and
injected rather than one being chosen here:

* :class:`RayCastOcclusion` casts one ray per pair and returns a bit. Cheap, and right about
  every pair whose shadow edge it does not land on.
* :class:`SilhouetteOcclusion` clips the source's angular extent against each blocker's
  silhouette and returns the covered fraction. Exact for a sight line crossing front-facing
  geometry once -- a sleeve, a wall, a bent duct, a baffle declared two-sided -- and the only
  treatment that moves the *worst* pair rather than the average. It **errs dark** where two separate front-facing
  silhouettes overlap in angle, because it adds areas rather than unioning them, so it reports
  a count alongside the fraction and a count of one proves that pair exact.

Both return a **fraction**, so the rest of the package does not branch on which was used: the
ray test simply returns zeros and ones. Both are **frozen** -- computed once when the model is
built and stored -- which is what keeps an ``n**2`` or ``n**3`` pass off the differentiation
path. Occlusion has no derivative worth having in either form: the ray test is a staircase, and
while the fraction is continuous, evaluating it live would put the whole build on the tape at
every gradient.
"""

from __future__ import annotations

import abc
import warnings
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.checks import open_facets
from aquaflux.radiation.grid import TriangleGrid
from aquaflux.radiation.silhouette import (
    angular_cone,
    beyond_source_plane,
    cones_may_overlap,
    covered_by,
    source_view,
)
from aquaflux.radiation.triangles import padded_length, segment_is_cut
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = [
    "NoOcclusion",
    "OcclusionField",
    "RayCastOcclusion",
    "SelfOcclusion",
    "SilhouetteOcclusion",
]


class OcclusionField(eqx.Module):
    """What a self-occlusion strategy produces, for one set of receivers.

    Attributes
    ----------
    fraction : jnp.ndarray, shape ``(n_receivers, n_facets)``
        Of each source's projected solid angle, how much the surface's own triangles hide. Zero
        or one from a ray test; anywhere in between from the silhouette clip.
    overlapping : jnp.ndarray of bool, shape ``(n_receivers, n_facets)``
        Whether more than one blocker covered part of this pair, so their fractions were added
        and **may** have been double counted. Always ``False`` from a ray test, whose ``or`` is
        idempotent. It proves a pair exact where it is ``False``; where it is ``True`` it proves
        nothing, and on a meshed body it is ``True`` for nearly every hidden pair, because a
        tiled blocker covers a pair with several of its triangles without any of them
        overlapping. It is a count taken in the pass that already runs, so it costs nothing,
        but it is not a detector for the over-count.
    """

    fraction: jnp.ndarray
    overlapping: jnp.ndarray


class SelfOcclusion(eqx.Module):
    """How a surface's own triangles are tested for standing in the light."""

    #: Whether this strategy can answer for receivers lying on no facet -- points in the fluid,
    #: which have no surface normal. A model builds two masks, one between facets and one from
    #: facets to its volume receivers, and reads this to decide whether the strategy that
    #: serves the first can serve the second too.
    serves_volume_receivers: ClassVar[bool] = True

    @abc.abstractmethod
    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Work out, once, how much of each source each receiver cannot see.

        Parameters
        ----------
        surfaces : Surfaces
            The emitting set. Its triangles are both the sources and the blockers.
        points : jnp.ndarray, shape ``(n_receivers, 3)``
            Receiver positions.
        near : jnp.ndarray, shape ``(n_facets,)``
            How far along each segment to start looking, in length units -- the margin that
            stops a facet from shadowing itself.
        receiver_facet : jnp.ndarray of int, shape ``(n_receivers,)`` or None
            Which facet each receiver sits on, or ``-1`` where none, or ``None`` when the
            receivers are volume points lying on no facet at all.

        Returns
        -------
        OcclusionField
        """


class NoOcclusion(SelfOcclusion):
    """The surface does not shadow itself at all.

    A strategy rather than a ``None``, so that "unset, use the default" and "deliberately switched
    off" stay different things. Collapsing them onto one sentinel is how a caller who *meant* the
    default ends up with no self-occlusion, which produces a plausible brighter field and no error
    -- the exact failure the warning on :func:`~aquaflux.radiation.visibility.build_visibility`
    describes.

    ⚠️ **Correct only for a convex surface**, where no facet can stand in front of another and the
    source-side cosine clamp is already the exact visibility test. For anything else -- a sleeve, a
    baffle, a bend -- this silently deletes the shadows the package exists to compute.
    """

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Nothing hides anything. See :meth:`SelfOcclusion.field`."""
        shape = (int(points.shape[0]), int(surfaces.n_facets))
        return OcclusionField(fraction=jnp.zeros(shape), overlapping=jnp.zeros(shape, dtype=bool))


class RayCastOcclusion(SelfOcclusion):
    """One ray per pair, facet centroid to receiver: blocked or not, with nothing in between.

    A pair only half in shadow is recorded wholly blocked or wholly clear, and refining the mesh
    changes how *many* pairs straddle a shadow edge rather than how wrong a straddling one is.
    Cheap, and the right choice wherever the geometry is coarse relative to its shadows or where
    several separate bodies can shadow one sight line -- the case the silhouette clip adds up
    twice.

    **Every ray is tested against every triangle** unless :attr:`grid` is set, which is what
    makes this cost ``n_receivers x n_facets x n_triangles``. Set it on anything but a small
    scene: a uniform grid over the triangles turns the per-ray work from the whole surface into
    the few triangles the segment's own voxels hold, for the same answers
    (:class:`~aquaflux.radiation.grid.TriangleGrid`).

    Attributes
    ----------
    pair_limit : int
        Receiver-by-facet pairs per pass, bounding the peak memory of the build: a pass forms
        one ray per pair, and its origins, targets and exclusions with it.
    work_limit : int
        Ray-by-triangle entries per pass, which is what bounds the intersection test's memory
        and, through that, its speed. It bounds the grid's passes too: a step of the walk tests
        every live ray against everything its voxel holds, which without a bound is one array of
        every pair in that step.
    grid : bool or int or tuple of int
        Cull each ray's candidates with a uniform grid over the triangles. ``False`` (the
        default) tests everything; ``True`` sizes the grid from the triangle count; an integer
        or a triple sets its resolution per axis. **Off by default** because it is a change of
        cost, not of answers, and the answers are what the shipped path is trusted for -- but a
        real reactor is unusable without it.
    """

    pair_limit: int = DEFAULT_PAIR_LIMIT
    work_limit: int = 4_000_000
    grid: bool | int | tuple[int, int, int] = False

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Cast the rays. See :meth:`SelfOcclusion.field`."""
        n_receivers, n_facets = points.shape[0], surfaces.n_facets
        grid = (
            TriangleGrid.build(
                np.asarray(surfaces.vertices),
                resolution=None if self.grid is True else self.grid,
            )
            if self.grid is not False
            else None
        )
        facet = jnp.arange(n_facets)
        # Every ray must ignore the facet it leaves; one aimed at a facet centroid must ignore
        # that facet too, or it is blocked by its own destination.
        source_of = jnp.broadcast_to(facet[None, :], (n_receivers, n_facets))
        if receiver_facet is None:
            exclusions = source_of[..., None]
        else:
            target_of = jnp.broadcast_to(
                jnp.asarray(receiver_facet, dtype=int)[:, None], (n_receivers, n_facets)
            )
            exclusions = jnp.stack([source_of, target_of], axis=-1)

        per_pass = receivers_per_pass(self.pair_limit, n_facets)
        rows = []
        for start in range(0, n_receivers, per_pass):
            receivers = points[start : start + per_pass]
            rays = receivers.shape[0]
            flat = (rays * n_facets, 3)
            origin = surfaces.centroid[None, :, :]
            target = receivers[:, None, :]
            if grid is not None:
                rows.append(
                    jnp.asarray(
                        grid.blocks(
                            np.broadcast_to(
                                np.asarray(surfaces.centroid)[None, :, :], (rays, n_facets, 3)
                            ).reshape(flat),
                            np.broadcast_to(
                                np.asarray(receivers)[:, None, :], (rays, n_facets, 3)
                            ).reshape(flat),
                            np.broadcast_to(np.asarray(near), (rays, n_facets)).reshape(-1),
                            exclude=np.asarray(exclusions[start : start + per_pass]).reshape(
                                -1, exclusions.shape[-1]
                            ),
                            work_limit=self.work_limit,
                        ).reshape(rays, n_facets)
                    )
                )
                continue
            rows.append(
                segment_is_cut(
                    jnp.broadcast_to(origin, (rays, n_facets, 3)).reshape(flat),
                    jnp.broadcast_to(target, (rays, n_facets, 3)).reshape(flat),
                    surfaces.vertices,
                    jnp.broadcast_to(near, (rays, n_facets)).reshape(-1),
                    exclude=exclusions[start : start + per_pass].reshape(-1, exclusions.shape[-1]),
                    work_limit=self.work_limit,
                ).reshape(rays, n_facets)
            )
        blocked = (
            jnp.concatenate(rows, axis=0)
            if rows
            else jnp.zeros((n_receivers, n_facets), dtype=bool)
        )
        # A bit, widened to the fraction the rest of the package consumes. One ray can only ever
        # say all or nothing, so no pair it reports is ever an addition of two answers.
        return OcclusionField(
            fraction=blocked.astype(float),
            overlapping=jnp.zeros_like(blocked, dtype=bool),
        )


class SilhouetteOcclusion(SelfOcclusion):
    """Clip each source's angular extent against every blocker's silhouette: an exact fraction.

    No sampling anywhere, so a pair the shadow edge crosses is right rather than rounded to the
    nearer bit. See :mod:`aquaflux.radiation.silhouette` for the geometry and for the one case
    it gets wrong: overlapping front-facing silhouettes, whose covered shares are added and so
    err dark. That is rare where a nearer body hides a source completely, since the sum is
    clipped at one; it bites where two bodies each hide part of one source.
    :attr:`OcclusionField.overlapping` marks every pair where more than one blocker
    contributed, which includes every tiling, so it proves pairs exact rather than finding the
    wrong ones.

    ⚠️ **Every receiver must sit on a facet.** The fraction is of a *projected* solid angle, so
    it needs the receiver's own normal to project onto, and a point in the fluid has none. That
    is not a limitation of the clip but of the quantity: a volume gather weights sources by
    their unprojected solid angle, which is a different measure and would need a different
    kernel. A model selected with this strategy therefore serves its volume receivers with the
    ray test (see :class:`~aquaflux.radiation.model.RadiationSettings`, ``receiver_occlusion``),
    so the fluence rate in the fluid still sees each sleeve as all or nothing per pair; the
    exact fraction reaches the surface-to-surface transfer, which carries the interreflection.

    **Three passes per receiver**, and the middle one is what makes the cost bearable: build
    each triangle's bounding cone and each source's clipped view (both ``n`` per receiver, not
    ``n**2``); cull every (source, blocker) pair whose cones cannot overlap; then clip only the
    survivors. The cull's survival falls as the mesh refines -- a finer pair sweeps a narrower
    pencil -- which is what keeps this from costing ``n**3``.

    **Which blockers count, and why it has to be declared.** A blocker counts only from the
    side it faces, unless its body is named in :attr:`two_sided`. On a closed, consistently
    wound surface that is exact: a blocked sight line leaves the medium through one face and
    re-enters through another, and only the first faces the receiver, so counting back faces too
    would add the same shadow twice. A zero-thickness sheet -- a baffle given as one layer of
    triangles -- has the medium on both sides and is crossed once from either, so seen from
    behind it would hide nothing: light passes straight through it. Topology cannot settle this
    alone (see :func:`~aquaflux.radiation.checks.open_facets`): a sheet welded to a wall all the
    way round looks closed, and a duct exported without its end caps looks like a sheet. So the
    sheets are named, and any open piece of surface left undeclared is **warned about** when the
    field is built rather than silently leaking light. The ray test has no such distinction to
    make -- it asks only whether a segment crosses a triangle -- which is why it needs none.

    Attributes
    ----------
    work_chunk : int
        The largest number of surviving (source, blocker) pairs clipped per compiled call, which
        is what bounds this pass's memory.
    two_sided : tuple of str
        Bodies, by their name in :attr:`~aquaflux.radiation.surfaces.Surfaces.solid_names`, that
        block from both sides: zero-thickness sheets. Empty by default. Naming a closed body here
        makes it block twice, which errs dark.

        Chunks are padded up to a **power of two** rather than to this length. Padding every
        chunk to the cap compiles exactly one program, which sounds like the thing to want and
        is not: a body whose receivers have sixty candidates each would then clip a quarter of a
        million pairs apiece, thousands of times the real work. Rounding up to a power of two
        wastes at most half a chunk and costs a couple of dozen compiled shapes across any
        conceivable range, each of which is reused by every receiver that lands in its bucket.
    """

    serves_volume_receivers: ClassVar[bool] = False

    work_chunk: int = 262_144
    two_sided: tuple[str, ...] = ()

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Clip the survivors. See :meth:`SelfOcclusion.field`."""
        if receiver_facet is None:
            msg = (
                "SilhouetteOcclusion needs each receiver's own surface normal, and "
                "`receiver_facet` says which facet supplies it. Receivers in the volume have no "
                "normal and no projected solid angle to take a fraction of, so use "
                "RayCastOcclusion for a volume gather."
            )
            raise ValueError(msg)
        facet_of = np.asarray(receiver_facet, dtype=int)
        if np.any(facet_of < 0):
            stray = int(np.sum(facet_of < 0))
            msg = (
                f"{stray} receiver(s) lie on no facet (`receiver_facet` is -1 there). "
                "SilhouetteOcclusion takes a fraction of a projected solid angle and so needs a "
                "normal at every receiver; use RayCastOcclusion for those points."
            )
            raise ValueError(msg)

        vertices = surfaces.vertices
        normal = surfaces.normal
        centroid = surfaces.centroid
        n_facets = int(surfaces.n_facets)
        n_receivers = int(points.shape[0])
        index = np.arange(n_facets)
        either_side = self._either_side(surfaces)

        fraction = np.zeros((n_receivers, n_facets))
        blockers = np.zeros((n_receivers, n_facets), dtype=np.int32)
        for row in range(n_receivers):
            source, blocker = self._candidates(
                points[row], normal[facet_of[row]], vertices, centroid, normal, near, either_side
            )
            # A facet never blocks itself, and never blocks the facet the receiver sits on.
            legal = (source != blocker) & (blocker != facet_of[row]) & (source != facet_of[row])
            source, blocker = source[legal], blocker[legal]
            if not len(source):
                continue
            covered, hit = self._clip(points[row], normal[facet_of[row]], vertices, source, blocker)
            np.add.at(fraction[row], source, covered)
            np.add.at(blockers[row], source, hit.astype(np.int32))

        del index
        return OcclusionField(
            fraction=jnp.clip(jnp.asarray(fraction), 0.0, 1.0),
            # More than one blocker contributed, so the two areas were ADDED. They may or may
            # not overlap in angle -- a tiling of one wall does not -- so this is "possibly
            # double counted", which is the honest thing a count can say.
            overlapping=jnp.asarray(blockers > 1),
        )

    def _either_side(self, surfaces) -> np.ndarray:
        """Which facets block from both sides, warning about open pieces nobody declared."""
        names = tuple(surfaces.solid_names)
        unknown = sorted(set(self.two_sided) - set(names))
        if unknown:
            msg = (
                f"two_sided names no body in this surface set: {unknown}; have {list(names)}. "
                "A misspelt sheet would otherwise be counted from one side only, and let light "
                "through from behind without any error."
            )
            raise ValueError(msg)
        solid = np.asarray(surfaces.solid_id)
        declared = np.isin(solid, [names.index(name) for name in self.two_sided])
        undeclared = open_facets(np.asarray(surfaces.vertices)) & ~declared
        if np.any(undeclared):
            bodies = sorted({names[k] for k in np.unique(solid[undeclared])})
            warnings.warn(
                f"SilhouetteOcclusion: {int(np.count_nonzero(undeclared))} facet(s) of "
                f"{bodies} belong to pieces of surface with a free edge, and are counted as "
                "blocking only from the side they face -- right for a solid whose surface was "
                "left open, wrong for a zero-thickness sheet, which then lets light straight "
                "through from behind. Name the sheets in `two_sided`, close the surface, or use "
                "RayCastOcclusion.",
                stacklevel=3,
            )
        return declared

    @staticmethod
    @jax.jit
    def _survivors(receiver, receiver_normal, vertices, centroid, normal, near, either_side):
        """Which (source, blocker) pairs could possibly matter, as a dense mask."""
        relative = vertices - receiver
        cone = angular_cone(relative)
        source_cone = tuple(x[:, None] if x.ndim == 1 else x[:, None, :] for x in cone)
        blocker_cone = tuple(x[None, :] if x.ndim == 1 else x[None, :, :] for x in cone)

        view = source_view(receiver, receiver_normal, vertices)
        keep = cones_may_overlap(source_cone, blocker_cone)
        keep &= ~beyond_source_plane(
            relative[None, :, :, :], view.support[:, None, :], view.through[:, None, :]
        )
        # A blocker entirely behind the receiver's own tangent plane blocks nothing in front.
        keep &= ~jnp.all(jnp.sum(relative * receiver_normal, axis=-1) < 0.0, axis=-1)[None, :]
        # Front-facing only: a sight line leaving a wetted surface and re-entering crosses
        # front-facing geometry exactly once, so the back faces would double the count. A
        # declared sheet is crossed once from either side, so it counts from both.
        facing = (jnp.sum(normal * (receiver - centroid), axis=-1) > 0.0) | either_side
        # The near margin keeps a facet from shadowing its own immediate neighbourhood, the same
        # role it plays for the ray test.
        far_enough = jnp.linalg.norm(receiver - centroid, axis=-1) > near
        return keep & (facing & far_enough)[None, :]

    def _candidates(self, receiver, receiver_normal, vertices, centroid, normal, near, either_side):
        """The surviving pairs, compacted on the host -- legal because the mask is frozen."""
        keep = np.asarray(
            self._survivors(
                receiver, receiver_normal, vertices, centroid, normal, near, either_side
            )
        )
        return np.nonzero(keep)

    def _clip(self, receiver, receiver_normal, vertices, source, blocker):
        """Run the exact clip over the candidate list, in fixed-size padded chunks."""
        covered = np.zeros(len(source))
        hit = np.zeros(len(source), dtype=bool)
        for start in range(0, len(source), self.work_chunk):
            stop = min(start + self.work_chunk, len(source))
            piece = slice(start, stop)
            # Padded up to a power of two, so one compiled program serves every chunk of that
            # size; the padding repeats a real item and its answer is discarded.
            pad = padded_length(stop - start) - (stop - start)
            take_source = np.pad(source[piece], (0, pad), mode="edge")
            take_blocker = np.pad(blocker[piece], (0, pad), mode="edge")
            part, struck = self._clip_chunk(
                receiver, receiver_normal, vertices, take_source, take_blocker
            )
            covered[piece] = np.asarray(part)[: stop - start]
            hit[piece] = np.asarray(struck)[: stop - start]
        return covered, hit

    @staticmethod
    @jax.jit
    def _clip_chunk(receiver, receiver_normal, vertices, source, blocker):
        """One compiled pass over a fixed number of (source, blocker) pairs."""
        view = source_view(receiver, receiver_normal, jnp.take(vertices, source, axis=0))
        to_blocker = jnp.take(vertices, blocker, axis=0) - receiver
        return covered_by(view, receiver_normal, to_blocker)
