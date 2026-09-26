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
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.checks import open_facets
from aquaflux.radiation.clipping import decidable_heights
from aquaflux.radiation.clusters import FacetClusters
from aquaflux.radiation.grid import TriangleGrid
from aquaflux.radiation.silhouette import (
    SourceView,
    angular_cone,
    beyond_source_plane,
    cones_may_overlap,
    covered_by,
    covers_nothing,
    enclosing_cone,
    source_view,
)
from aquaflux.radiation.triangles import padded_length, pairs_are_cut
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass
from aquaflux.vectors import dot

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
    fraction : jnp.ndarray, shape ``(n_receivers, n_facets)``, or None
        Of each source's view -- its projected solid angle from a receiver on a facet, its plain
        solid angle from one in the volume -- how much the surface's own triangles hide.
        **Stored at the narrowest type that holds it**, because it is the size of the whole
        problem: floating point from the silhouette clip, which measures anything in between;
        boolean from a ray test, which can only say all or nothing, and whose eight bytes a pair
        as a float held one bit; and ``None`` where nothing is hidden at all
        (:class:`NoOcclusion`). Whoever reads it widens it; see
        :func:`~aquaflux.radiation.visibility.surviving_fraction`.
    overlapping : jnp.ndarray of bool, shape ``(n_receivers, n_facets)``, or None
        Whether more than one blocker covered part of this pair, so their fractions were added
        and **may** have been double counted. ``None`` from a ray test, whose ``or`` is
        idempotent, so no pair it reports is ever such an addition, and from
        :class:`NoOcclusion`. It proves a pair exact where it is ``False``; where it is ``True`` it proves
        nothing, and on a meshed body it is ``True`` for nearly every hidden pair, because a
        tiled blocker covers a pair with several of its triangles without any of them
        overlapping. It is a count taken in the pass that already runs, so it costs nothing,
        but it is not a detector for the over-count.
    clear_behind : bool
        Whether a pair whose source faces away from its receiver was recorded clear without
        being tested. Such a pair carries no light from a source that is dark behind itself, so
        whatever a test found there would be multiplied by zero; see
        :attr:`~aquaflux.radiation.visibility.Visibility.clear_behind` for what it asks of a
        caller.
    """

    fraction: jnp.ndarray | None
    overlapping: jnp.ndarray | None
    clear_behind: bool = eqx.field(static=True, default=False)


class SelfOcclusion(eqx.Module):
    """How a surface's own triangles are tested for standing in the light.

    Every strategy answers for both kinds of receiver: points on a facet, as the surface-to-surface
    transfer uses, and points in the fluid, as the volume gather does.
    """

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

    def prepared(self, surfaces) -> SelfOcclusion:
        """This strategy with whatever it can work out from ``surfaces`` alone done once.

        For a caller that builds many masks of one surface -- a stream building one per pass of
        receivers -- so that work depending only on the surface is not repeated for every pass.
        The answers are the same either way. Unless a strategy has such work, it is itself.

        Parameters
        ----------
        surfaces : Surfaces
            The emitting set the masks will be built for.

        Returns
        -------
        SelfOcclusion
        """
        del surfaces
        return self


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
        return OcclusionField(fraction=None, overlapping=None)


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
        Receiver-by-facet pairs per pass, bounding the peak memory of the build: a pass holds
        which pairs it casts, as two indices each, and without a grid forms their rays a chunk
        of the intersection test at a time; with one, the walk needs every ray's endpoints and
        exclusions for the whole pass.
    work_limit : int
        Ray-by-triangle entries per pass, which is what bounds the intersection test's memory
        and, through that, its speed. It bounds the grid's passes too: a step of the walk tests
        every live ray against everything its voxel holds, which without a bound is one array of
        every pair in that step.
    grid : bool or int or tuple of int or TriangleGrid
        Cull each ray's candidates with a uniform grid over the triangles. ``False`` (the
        default) tests everything; ``True`` sizes the grid from the triangle count; an integer
        or a triple sets its resolution per axis; a built
        :class:`~aquaflux.radiation.grid.TriangleGrid` is used as it is, and must be of the
        surface's own triangles. **Off by default** because it is a change of cost, not of
        answers, and the answers are what the shipped path is trusted for -- but a real reactor
        is unusable without it.

    Notes
    -----
    **For receivers in the volume, a pair whose source faces away is not cast at all** when
    every areal facet's profile is :attr:`~aquaflux.radiation.profiles.Profile.dark_behind`:
    the gather weights that pair by the source's radiance towards the receiver, which is then
    exactly zero, so the ray's answer could only ever be multiplied by nothing. It is recorded
    clear, and the mask says so (:attr:`OcclusionField.clear_behind`). A lamp sees about half of
    each receiver from behind, so this is a large share of the rays. The test is on the sign of
    the receiver's height above the facet's plane, the same quantity the gather's cosine is, and
    a height too close to zero for its sign to be trusted is cast rather than skipped. Receivers
    on facets -- the surface-to-surface transfer -- are always cast in full: the transfer's
    weights do not vanish behind a source.
    """

    pair_limit: int = DEFAULT_PAIR_LIMIT
    work_limit: int = 4_000_000
    grid: bool | int | tuple[int, int, int] | TriangleGrid = False

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Cast the rays. See :meth:`SelfOcclusion.field`."""
        n_receivers, n_facets = int(points.shape[0]), int(surfaces.n_facets)
        grid = self._triangle_grid(surfaces)
        clear_behind = receiver_facet is None and surfaces.dark_behind
        centroid = np.asarray(surfaces.centroid)
        near = np.asarray(near, dtype=float)
        on_facet = None if receiver_facet is None else np.asarray(receiver_facet, dtype=int)
        blocked = np.zeros((n_receivers, n_facets), dtype=bool)
        per_pass = receivers_per_pass(self.pair_limit, n_facets)
        for start in range(0, n_receivers, per_pass):
            receivers = np.asarray(points[start : start + per_pass], dtype=float)
            if clear_behind:
                cast = ~_facing_away(surfaces, receivers)
            else:
                cast = np.ones((len(receivers), n_facets), dtype=bool)
            row, source = np.nonzero(cast)
            if len(row) == 0:
                continue
            on_this_pass = None if on_facet is None else on_facet[start : start + per_pass]
            if grid is not None:
                # The grid walk is host code that steps every ray, so it needs every ray's
                # endpoints at once; the brute-force test below forms them a chunk at a time.
                hit = grid.blocks(
                    centroid[source],
                    receivers[row],
                    near[source],
                    exclude=_exclusions(source, row, on_this_pass),
                    work_limit=self.work_limit,
                )
            else:
                hit = pairs_are_cut(
                    receivers,
                    centroid,
                    near,
                    surfaces.vertices,
                    row,
                    source,
                    target=on_this_pass,
                    work_limit=self.work_limit,
                )
            blocked[start + row, source] = np.asarray(hit)
        # A bit, kept as one: whoever reads it widens it. One ray can only ever say all or
        # nothing, so no pair it reports is ever an addition of two answers.
        return OcclusionField(
            fraction=jnp.asarray(blocked), overlapping=None, clear_behind=clear_behind
        )

    def prepared(self, surfaces) -> RayCastOcclusion:
        """With its grid built, so a stream of masks builds it once. See :meth:`SelfOcclusion.prepared`."""
        grid = self._triangle_grid(surfaces)
        return self if grid is None else eqx.tree_at(lambda strategy: strategy.grid, self, grid)

    def _triangle_grid(self, surfaces) -> TriangleGrid | None:
        """The grid the rays are culled with, built here unless it was given built."""
        if self.grid is False:
            return None
        if isinstance(self.grid, TriangleGrid):
            vertices = np.asarray(surfaces.vertices)
            if self.grid.vertices.shape != vertices.shape or not np.array_equal(
                self.grid.vertices, vertices
            ):
                msg = (
                    "the grid given was built over other triangles than this surface's own; a "
                    "grid decides which triangles a ray is tested against, so one of other "
                    "triangles misses the shadows this surface casts"
                )
                raise ValueError(msg)
            return self.grid
        return TriangleGrid.build(
            np.asarray(surfaces.vertices), resolution=None if self.grid is True else self.grid
        )


def _exclusions(source: np.ndarray, row: np.ndarray, on_facet) -> np.ndarray:
    """What each ray ignores: the facet it leaves and, when its receiver sits on one, that facet.

    Leaving out the second blocks every ray aimed at a facet centroid on its own destination.
    Formed for one pass's rays, from their indices, rather than for the whole problem.
    """
    return source[:, None] if on_facet is None else np.stack([source, on_facet[row]], axis=1)


def _facing_away(surfaces, receivers) -> np.ndarray:
    """Which (receiver, facet) pairs are certainly behind an areal facet: ``(n_receivers, n_facets)``.

    The receiver's height above the facet's own plane, with a sign that cannot be trusted
    resolved to zero -- which keeps the pair -- so a pair is dropped only where rounding could
    not have put the receiver on the other side. A point source has a zero normal, so every
    height is zero and none of its pairs is ever dropped; it is excluded by its label as well.
    """
    behind = _behind(
        jnp.asarray(surfaces.normal), jnp.asarray(surfaces.centroid), jnp.asarray(receivers)
    )
    return np.asarray(behind) & ~np.asarray(surfaces.is_point_source)[None, :]


@jax.jit
def _behind(normal, centroid, receivers):
    """Whether each receiver lies certainly behind each facet's plane."""
    height = decidable_heights(
        receivers[:, None, None, :], normal[None, :, :], through=centroid[None, :, :]
    )
    return height[..., 0] < 0.0


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

    **The fraction is of the measure the receiver gathers in.** A receiver on a facet takes its
    share of the source's *projected* solid angle, about that facet's normal, because that is
    what an irradiance weights by; a point in the fluid has no normal and takes its share of the
    *plain* solid angle, which is what a fluence rate weights by. The clip is the same for both --
    it works in direction space -- and only the contour integral taken of the covered region
    differs. So a fluence rate in the water sees a sleeve's shadow edge as the fraction it is,
    rather than all or nothing per pair.

    **Four passes**, and the middle two are what make the cost bearable. Per receiver: build each
    triangle's bounding cone and each source's clipped view (``n`` each, not ``n**2``), keeping
    only the sources that subtend something and the blockers that could stand in front of this
    receiver; then cull every surviving (source, blocker) pair whose cones cannot overlap or whose
    blocker lies beyond the source (:func:`~aquaflux.radiation.silhouette.may_occlude`'s bounds) --
    on clusters of neighbouring facets first, so that pairs between two clusters whose bounds
    cannot meet are rejected by one test, and pair by pair only within the clusters that remain.
    Then, over the pairs of many receivers at once: reject every pair that certainly covers
    nothing (:func:`~aquaflux.radiation.silhouette.covers_nothing`) -- most of what the cones let
    through, on a meshed enclosure -- and clip only what is left. The cone cull's survival falls
    as the mesh refines, since a finer pair sweeps a narrower pencil, which is what keeps this from
    costing ``n**3``.

    **The last two passes pack pairs from many receivers into chunks of one fixed size**, each
    carrying its own receiver and its source's view, so every chunk but the very last is full and
    one compiled program serves them all; the last is padded to a power of two. Packing per
    receiver instead pads every receiver's pairs, which on a meshed body wastes a large share of
    the clip on padding. Chunks run on :attr:`threads` threads: one compiled call keeps only part
    of a multi-core processor busy, and several at once keep more of it. The answer does not depend
    on either, since every pair is clipped by the same program and summed in the same order.

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
        (Source, blocker) pairs rejected or clipped per compiled call, which bounds those passes'
        memory: each of :attr:`threads` calls in flight holds one chunk's working set.
    pair_limit : int
        Entries one compiled call of the cull may form for a receiver -- cluster pairs times the
        cluster size, or member pairs -- which bounds the cull's memory the same way
        ``pair_limit`` bounds every receiver-by-facet pass in this package.
    cluster_size : int
        Facets per cluster in the cull (:class:`~aquaflux.radiation.clusters.FacetClusters`).
        Smaller clusters bound their members more tightly but make more cluster pairs to test;
        the answer does not depend on it, only the cost.
    threads : int
        How many compiled calls of the reject and clip passes run at once, and how many receivers
        ahead the cull works.
    two_sided : tuple of str
        Bodies, by their name in :attr:`~aquaflux.radiation.surfaces.Surfaces.solid_names`, that
        block from both sides: zero-thickness sheets. Empty by default. Naming a closed body here
        makes it block twice, which errs dark.
    """

    work_chunk: int = 32_768
    pair_limit: int = DEFAULT_PAIR_LIMIT
    cluster_size: int = 32
    threads: int = 4
    two_sided: tuple[str, ...] = ()

    def field(self, surfaces, points, near, receiver_facet) -> OcclusionField:
        """Clip the survivors. See :meth:`SelfOcclusion.field`."""
        n_receivers = int(points.shape[0])
        # A receiver on no facet -- every one, when there are no facets to name -- is a point in
        # the volume: it has no normal, and takes its share of the plain solid angle instead.
        facet_of = (
            np.full(n_receivers, -1)
            if receiver_facet is None
            else np.asarray(receiver_facet, dtype=int)
        )
        n_facets = int(surfaces.n_facets)
        either_side = self._either_side(surfaces)
        vertices = jnp.asarray(surfaces.vertices, dtype=float)
        centroid = jnp.asarray(surfaces.centroid, dtype=float)
        normal = jnp.asarray(surfaces.normal, dtype=float)
        near = jnp.asarray(near, dtype=float)
        receivers = np.asarray(points, dtype=float)
        facet_normal = np.asarray(surfaces.normal, dtype=float)

        fraction = np.zeros((n_receivers, n_facets))
        blockers = np.zeros((n_receivers, n_facets), dtype=np.int32)

        clusters = FacetClusters.build(np.asarray(surfaces.vertices), self.cluster_size)

        def candidates(row):
            own = facet_of[row]
            facing = None if own < 0 else facet_normal[own]
            return self._candidates(
                receivers[row], facing, vertices, centroid, normal, near, either_side, clusters
            )

        with ThreadPoolExecutor(max_workers=max(1, self.threads)) as pool:
            # Surface and volume receivers take different measures, so they are clipped by
            # different programs and packed into different chunks.
            pipelines = {
                on_facet: _PairPipeline(
                    pool,
                    self.work_chunk,
                    max(1, self.threads),
                    vertices,
                    receivers,
                    facet_normal[np.maximum(facet_of, 0)] if on_facet else None,
                    fraction,
                    blockers,
                )
                for on_facet in (True, False)
            }
            for row, (view, source, blocker) in _ahead(pool, candidates, n_receivers, self.threads):
                own = facet_of[row]
                # A facet never blocks itself, and never blocks the facet the receiver sits on --
                # a test that is vacuous for a receiver on no facet, whose index is -1.
                legal = (source != blocker) & (blocker != own) & (source != own)
                if np.any(legal):
                    pipelines[own >= 0].add(row, view, source[legal], blocker[legal])
            for pipeline in pipelines.values():
                pipeline.finish()

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
        return jnp.asarray(declared)

    @staticmethod
    @jax.jit
    def _per_triangle(receiver, receiver_normal, vertices, centroid, normal, near, either_side):
        """Everything the cull needs that is one value per triangle, not one per pair.

        Returns each triangle's bounding cone, each source's view, which sources subtend anything
        at all, and which triangles could block from here. ``receiver_normal`` is ``None`` for a
        receiver in the volume, which sees every way and so has no tangent plane to cull behind.
        """
        relative = vertices - receiver
        cone = angular_cone(relative)
        view = source_view(receiver, receiver_normal, vertices)
        # Front-facing only: a sight line leaving a wetted surface and re-entering crosses
        # front-facing geometry exactly once, so the back faces would double the count. A
        # declared sheet is crossed once from either side, so it counts from both.
        facing = (dot(normal, receiver - centroid) > 0.0) | either_side
        # The near margin keeps a facet from shadowing its own immediate neighbourhood, the same
        # role it plays for the ray test.
        far_enough = jnp.linalg.norm(receiver - centroid, axis=-1) > near
        may_block = facing & far_enough
        # A blocker entirely behind the receiver's own tangent plane blocks nothing in front.
        if receiver_normal is not None:
            may_block &= ~jnp.all(dot(relative, receiver_normal) < 0.0, axis=-1)
        return cone, view, view.subtends(), may_block

    @staticmethod
    @jax.jit
    def _cluster_bounds(cone, members, subtends, may_block):
        """Each cluster's enclosing cone as a source and as a blocker, and whether it is either."""
        source_cone, has_source = enclosing_cone(cone, members, subtends)
        blocker_cone, has_blocker = enclosing_cone(cone, members, may_block)
        return source_cone, has_source, blocker_cone, has_blocker

    @staticmethod
    @jax.jit
    def _cluster_pairs(
        receiver, view, members, subtends, source_cone, blocker_cone, centre, radius, rows, cols
    ):
        """Which (source cluster, blocker cluster) pairs could hold a pair worth keeping.

        ``rows`` are source clusters and ``cols`` blocker clusters, so the mask is
        ``(len(rows), len(cols))``. Two reasons to reject a cluster pair, each a bound on the
        member test run after it: the enclosing cones cannot overlap, or the blocker cluster's
        bounding sphere lies wholly beyond the supporting plane of every eligible source in the
        source cluster -- so every blocker corner is beyond every such source.
        """
        keep = cones_may_overlap(
            tuple(x[rows][:, None] if x.ndim == 1 else x[rows][:, None, :] for x in source_cone),
            tuple(x[cols][None, :] if x.ndim == 1 else x[cols][None, :, :] for x in blocker_cone),
        )
        slot = jnp.maximum(members[rows], 0)
        eligible = (members[rows] >= 0) & jnp.take(subtends, slot)
        # Heights of each blocker cluster's centre above each source's plane: (rows, cols, k).
        offset = (centre[cols] - receiver)[None, :, None, :] - view.through[slot][:, None, :, :]
        height = dot(offset, view.support[slot][:, None, :, :])
        beyond = jnp.all(
            ~eligible[:, None, :] | (height + radius[cols][None, :, None] < 0.0), axis=-1
        )
        return keep & ~beyond

    @staticmethod
    @jax.jit
    def _member_pairs(
        receiver,
        vertices,
        cone,
        view,
        members,
        subtends,
        may_block,
        source_cluster,
        blocker_cluster,
    ):
        """The member test over every pair of each given cluster pair: ``(batch, k, k)``.

        The same two tests the cull has always applied to a (source, blocker) pair -- their cones
        can overlap, and the blocker is not wholly beyond the source's plane -- restricted to
        members that are eligible in their role.
        """
        source = members[source_cluster]
        blocker = members[blocker_cluster]
        s, b = jnp.maximum(source, 0), jnp.maximum(blocker, 0)
        eligible = ((source >= 0) & jnp.take(subtends, s))[:, :, None] & (
            (blocker >= 0) & jnp.take(may_block, b)
        )[:, None, :]
        keep = cones_may_overlap(
            tuple(x[s][:, :, None] if x.ndim == 1 else x[s][:, :, None, :] for x in cone),
            tuple(x[b][:, None, :] if x.ndim == 1 else x[b][:, None, :, :] for x in cone),
        )
        to_blocker = jnp.take(vertices, b, axis=0) - receiver
        keep &= ~beyond_source_plane(
            to_blocker[:, None, :, :, :],
            view.support[s][:, :, None, :],
            view.through[s][:, :, None, :],
        )
        return eligible & keep

    def _candidates(
        self, receiver, receiver_normal, vertices, centroid, normal, near, either_side, clusters
    ):
        """The pairs the cull keeps for one receiver, compacted on the host, and its views.

        Legal to compact on the host because the mask is frozen. The cull runs on clusters first
        (:class:`~aquaflux.radiation.clusters.FacetClusters`): each cluster's members are bounded
        by one cone per role, cluster pairs are rejected whole where those bounds allow, and only
        the members of the surviving cluster pairs are tested one by one. Every call forms at most
        :attr:`pair_limit` entries. The kept pairs are exactly those the member test would keep
        over all pairs -- the cluster stage only skips pairs that test would reject -- grouped by
        cluster pair.

        The compaction is on the host, which on a processor is the faster place for it: numpy's
        ``nonzero`` over a batch's mask beats a compiled one with a bounded output several times
        over, transfer included.

        Returns
        -------
        tuple of (SourceView, np.ndarray, np.ndarray)
            Every source's view, on the host, shape ``(n_facets, ...)``; and the kept pairs as
            parallel arrays of source and blocker indices.
        """
        cone, view, subtends, may_block = self._per_triangle(
            receiver, receiver_normal, vertices, centroid, normal, near, either_side
        )
        host_view = jax.tree.map(np.asarray, view)
        empty = np.zeros(0, dtype=int)
        members = jnp.asarray(clusters.members)
        source_cone, has_source, blocker_cone, has_blocker = self._cluster_bounds(
            cone, members, subtends, may_block
        )
        rows = np.flatnonzero(np.asarray(has_source))
        cols = np.flatnonzero(np.asarray(has_blocker))
        if not len(rows) or not len(cols):
            return host_view, empty, empty

        # Cluster pairs, in blocks of source clusters against every blocker cluster. Everything
        # is padded to powers of two, repeating a real index whose answers are cut off, so a
        # couple of dozen compiled shapes serve every receiver.
        k = clusters.size
        width = padded_length(len(cols))
        height = min(
            padded_length(len(rows)), _power_of_two_at_most(self.pair_limit // (width * k))
        )
        padded_cols = np.pad(cols, (0, width - len(cols)), mode="edge")
        source_cluster, blocker_cluster = [], []
        for start in range(0, len(rows), height):
            block = rows[start : start + height]
            keep = np.asarray(
                self._cluster_pairs(
                    receiver,
                    view,
                    members,
                    subtends,
                    source_cone,
                    blocker_cone,
                    jnp.asarray(clusters.centre),
                    jnp.asarray(clusters.radius),
                    np.pad(block, (0, height - len(block)), mode="edge"),
                    padded_cols,
                )
            )[: len(block), : len(cols)]
            i, j = np.nonzero(keep)
            source_cluster.append(block[i])
            blocker_cluster.append(cols[j])
        source_cluster = np.concatenate(source_cluster)
        blocker_cluster = np.concatenate(blocker_cluster)
        if not len(source_cluster):
            return host_view, empty, empty

        # Members of the surviving cluster pairs, a batch of cluster pairs per call.
        batch = min(
            padded_length(len(source_cluster)), _power_of_two_at_most(self.pair_limit // (k * k))
        )
        found_source, found_blocker = [], []
        for start in range(0, len(source_cluster), batch):
            these_sources = source_cluster[start : start + batch]
            these_blockers = blocker_cluster[start : start + batch]
            pad = batch - len(these_sources)
            keep = np.asarray(
                self._member_pairs(
                    receiver,
                    vertices,
                    cone,
                    view,
                    members,
                    subtends,
                    may_block,
                    np.pad(these_sources, (0, pad), mode="edge"),
                    np.pad(these_blockers, (0, pad), mode="edge"),
                )
            )[: len(these_sources)]
            pair, i, j = np.nonzero(keep)
            found_source.append(clusters.members[these_sources[pair], i])
            found_blocker.append(clusters.members[these_blockers[pair], j])
        # Clusters partition the facets, so each pair is found exactly once. They come out grouped
        # by cluster pair rather than sorted: sorting them costs as much as a fifth of the cull,
        # and their order only decides the order a source's shares are summed in, which moves the
        # answer by a rounding and not by a pair.
        return host_view, np.concatenate(found_source), np.concatenate(found_blocker)


def _power_of_two_at_most(count: int) -> int:
    """The largest power of two not above ``count``, and at least one."""
    return 1 << max(0, int(count).bit_length() - 1) if count > 0 else 1


def _ahead(pool, work, count: int, lookahead: int):
    """``(index, work(index))`` for every index in order, with up to ``lookahead`` computed ahead.

    Lets the per-receiver cull of the next receivers run on the pool while this one's pairs are
    handed on, without changing the order anything is consumed in.
    """
    pending = deque()
    for index in range(count):
        pending.append((index, pool.submit(work, index)))
        if len(pending) > max(0, lookahead):
            done, future = pending.popleft()
            yield done, future.result()
    while pending:
        done, future = pending.popleft()
        yield done, future.result()


class _Pairs:
    """(Receiver, source, blocker) triples in flight, with each one's source view.

    Plain host arrays, one entry per pair, so that pairs from any number of receivers can be cut
    into chunks of one size: each pair carries everything the reject and clip passes read that is
    not the global vertex table.
    """

    __slots__ = ("blocker", "row", "source", "view")

    def __init__(self, row, source, blocker, view: SourceView):
        self.row = row
        self.source = source
        self.blocker = blocker
        self.view = view

    def __len__(self) -> int:
        return len(self.row)

    def take(self, index) -> _Pairs:
        """The pairs ``index`` picks: a slice, a boolean mask or an index array."""
        return _Pairs(
            self.row[index], self.source[index], self.blocker[index], self.view.take(index)
        )

    @staticmethod
    def joined(parts) -> _Pairs:
        """One set of pairs from several, in order."""
        if len(parts) == 1:
            return parts[0]
        return _Pairs(
            np.concatenate([part.row for part in parts]),
            np.concatenate([part.source for part in parts]),
            np.concatenate([part.blocker for part in parts]),
            jax.tree.map(lambda *xs: np.concatenate(xs), *[part.view for part in parts]),
        )


class _PairPipeline:
    """Reject, then clip, the culled pairs of many receivers in chunks of one fixed size.

    Pairs accumulate as receivers are culled. Once a stage holds a chunk per thread, its whole
    chunks run -- concurrently, on the pool -- and the remainder waits for the next receivers, so
    no chunk is padded until the very last. The clip's answers are summed into the caller's arrays
    in the order the pairs arrived, whatever order the chunks finish in.
    """

    def __init__(self, pool, chunk, threads, vertices, receivers, normals, fraction, blockers):
        self._pool = pool
        self._chunk = max(1, int(chunk))
        self._wave = self._chunk * threads
        self._vertices = vertices
        self._receivers = receivers
        self._normals = normals
        self._fraction = fraction
        self._blockers = blockers
        self._culled: list[_Pairs] = []
        self._kept: list[_Pairs] = []
        self._n_culled = 0
        self._n_kept = 0

    def add(self, row, view: SourceView, source, blocker):
        """Take one receiver's culled pairs, and every source's view from it."""
        pairs = _Pairs(np.full(len(source), row), source, blocker, view.take(source))
        self._culled.append(pairs)
        self._n_culled += len(pairs)
        if self._n_culled >= self._wave:
            self._reject(final=False)

    def finish(self):
        """Run whatever is left, padding the last chunk of each stage."""
        self._reject(final=True)
        self._clip(final=True)

    def _reject(self, final: bool):
        ready, self._culled, self._n_culled = self._split(self._culled, final)
        if ready is not None:
            worth = np.concatenate(self._run(self._worth_clipping, ready))
            kept = ready.take(worth)
            if len(kept):
                self._kept.append(kept)
                self._n_kept += len(kept)
        if final or self._n_kept >= self._wave:
            self._clip(final)

    def _clip(self, final: bool):
        ready, self._kept, self._n_kept = self._split(self._kept, final)
        if ready is None:
            return
        answers = self._run(self._covered, ready)
        covered = np.concatenate([part for part, _ in answers])
        hit = np.concatenate([struck for _, struck in answers])
        np.add.at(self._fraction, (ready.row, ready.source), covered)
        np.add.at(self._blockers, (ready.row, ready.source), hit.astype(np.int32))

    def _split(self, parts, final: bool):
        """The pairs ready to run -- whole chunks, or everything when final -- and the rest."""
        if not parts:
            return None, [], 0
        pairs = _Pairs.joined(parts)
        ready = len(pairs) if final else (len(pairs) // self._chunk) * self._chunk
        if ready == 0:
            return None, [pairs], len(pairs)
        rest = pairs.take(slice(ready, None))
        return pairs.take(slice(0, ready)), ([rest] if len(rest) else []), len(rest)

    def _run(self, kernel, pairs: _Pairs):
        """``kernel`` over ``pairs`` a chunk at a time, on the pool, answers in order."""
        futures = [
            self._pool.submit(self._call, kernel, pairs.take(slice(start, start + self._chunk)))
            for start in range(0, len(pairs), self._chunk)
        ]
        return [future.result() for future in futures]

    def _call(self, kernel, pairs: _Pairs):
        """One compiled call, on a chunk padded to full size -- or, the last one, a power of two.

        The padding repeats a real pair and its answers are discarded.
        """
        count = len(pairs)
        size = self._chunk if count == self._chunk else min(self._chunk, padded_length(count))
        index = np.minimum(np.arange(size), count - 1)
        padded = pairs.take(index)
        normal = None if self._normals is None else self._normals[padded.row]
        answer = kernel(
            self._vertices,
            self._receivers[padded.row],
            normal,
            padded.view,
            padded.source,
            padded.blocker,
        )
        return jax.tree.map(lambda x: np.asarray(x)[:count], answer)

    @staticmethod
    @jax.jit
    def _worth_clipping(vertices, receiver, receiver_normal, view, source, blocker):
        """Which pairs might cover something, so are worth clipping."""
        del receiver_normal
        to_source = jnp.take(vertices, source, axis=0) - receiver[:, None, :]
        to_blocker = jnp.take(vertices, blocker, axis=0) - receiver[:, None, :]
        return ~covers_nothing(view, to_source, to_blocker)

    @staticmethod
    @jax.jit
    def _covered(vertices, receiver, receiver_normal, view, source, blocker):
        """The clip itself: each pair's covered fraction, and whether it covered anything."""
        del source
        to_blocker = jnp.take(vertices, blocker, axis=0) - receiver[:, None, :]
        return covered_by(view, receiver_normal, to_blocker)
