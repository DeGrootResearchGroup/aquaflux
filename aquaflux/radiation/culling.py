"""How the analytic-body layer of a shadow mask is decided: pair by pair, or a block at a time.

Every body is asked the same question of every source-receiver pair -- does it lie across the
segment between them -- and the answer is a hard yes or no fixed by geometry. Asked pair by pair,
the cost is receivers times sources whatever the scene looks like, and most of it is spent
re-deriving an answer the neighbouring pairs already gave: a cell in the middle of a chamber sees
every facet of the lamp across open water, and so does the cell beside it.

**Shaft culling** (Haines and Wallace, 1991) asks the question once for a whole *tile* of pairs.
The receivers are grouped into compact blocks and the sources into compact clusters. Every
segment from a source of a cluster to a receiver of a block lies inside the convex hull of the
two groups together, the *shaft*; a body that provably misses the shaft lies across none of
those segments, and the whole tile is clear with no segment tested. Only tiles no body can vouch
for are tested pair by pair, exactly as before.

**The proof is conservative, so the mask is the same mask.** A body's
:meth:`~aquaflux.solids.Body.clearance` says "clear" or "don't know", never "blocked", and
"don't know" falls back to the exact per-pair test. What changes is the cost, never the answer --
which is why it is the default wherever a mask is built, with every-pair testing kept as the
reference it is checked against.
The tempting shortcut -- one segment from a cluster's centre to a block's centre, taken as
representative -- is **not** a proof: it lights up whatever small shadow falls between the two
centres, silently.

**The proof costs nothing per pair.** A body's clearance is a handful of witness functions whose
negative region lies in a convex set disjoint from the body, and a group is summarized by the
largest value each witness takes over its points. The summary of a tile is the larger of its two
groups' summaries, so a tile is certified by comparing two short rows -- the pairs themselves are
never visited.

**The grouping is host work; the fallback test is compiled.** Ordering points along a
space-filling curve and cutting the order into groups is a sort, and deciding which tiles need
testing is a comparison over tiles; both run in numpy, off any trace. The undecided tiles are
then gathered into batches of one fixed shape and tested by the same compiled body test the
pair-by-pair strategy uses, so the two agree bit for bit on every pair either one tests.
"""

from __future__ import annotations

import abc
import dataclasses
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.triangles import padded_length
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass

__all__ = ["BodyCulling", "EveryPair", "ShaftCulling"]


@eqx.filter_jit
def _compiled_blocks(body, origin, target, near):
    """One body's answer for a batch of segments, as a single compiled expression.

    ⚠️ **Compiling an analytic body is worth a great deal, and compiling a triangulated one
    raises.** An analytic test is a few dozen arithmetic operations over a receivers-by-sources
    array, and a body assembled from several inequalities -- or a fluid described by several
    regions -- is several such arrays. Evaluated eagerly, every one of them is materialized in
    turn, hundreds of megabytes each at a production pass, and the mask's cost becomes the cost
    of writing those intermediates rather than of the arithmetic. Compiled, they fuse into the
    reduction at the end and none is ever formed. A body that answers on the host instead --
    walking a grid of triangles, dropping rays as they are settled -- cannot be traced at all,
    and deliberately so.

    So this is applied only where :attr:`~aquaflux.solids.Body.traceable` says it may be, which is
    a declaration on the body rather than a guess from its type.
    """
    return body.blocks(origin, target, near)


def _body_blocks(body, origin, target, near):
    """One body's answer, compiled where the body says it can be."""
    if body.traceable:
        return _compiled_blocks(body, origin, target, near)
    return body.blocks(origin, target, near)


class BodyCulling(eqx.Module):
    """How the analytic bodies' layer of a shadow mask is worked out.

    Each strategy returns the same boolean mask; they differ only in how many segments they
    test to find it.
    """

    @abc.abstractmethod
    def blocked(
        self, bodies, sources, near, receivers, pair_limit: int = DEFAULT_PAIR_LIMIT
    ) -> jnp.ndarray:
        """Whether each body lies across the segment from each source to each receiver.

        Parameters
        ----------
        bodies : sequence of aquaflux.solids.Body
            At least one.
        sources : array_like, shape ``(n_sources, 3)``
            Where each segment starts: the facet centroids.
        near : array_like, shape ``(n_sources,)``
            How far from its source a hit must be before it counts, per source, as a length --
            see :meth:`~aquaflux.solids.Body.blocks`.
        receivers : array_like, shape ``(n_receivers, 3)``
            Where each segment ends.
        pair_limit : int, optional
            Source-receiver pairs one compiled test may form, bounding its peak memory.

        Returns
        -------
        jnp.ndarray of bool, shape ``(n_bodies, n_receivers, n_sources)``
        """

    def prepared(self, bodies, sources) -> BodyCulling:
        """This strategy with whatever it can work out from the bodies and sources alone done once.

        For a caller that builds many masks against one set of sources -- a stream building one
        per pass of receivers -- so that work depending only on the sources is not repeated for
        every pass. The answers are the same either way. Unless a strategy has such work, it is
        itself.

        Parameters
        ----------
        bodies : sequence of aquaflux.solids.Body
        sources : array_like, shape ``(n_sources, 3)``
            The sources every later :meth:`blocked` call will be given.

        Returns
        -------
        BodyCulling
        """
        del bodies, sources
        return self


class EveryPair(BodyCulling):
    """Every body against every segment: the reference the other strategies must reproduce.

    Brute force over the bodies, which is correct practice at the count a scene has: a
    bounding-volume hierarchy over a handful of primitives is a single leaf node, and its
    traversal would cost more than the tests it saves. What it cannot exploit is coherence
    between neighbouring *pairs*, which is what :class:`ShaftCulling`, the default, is for. Pass
    this strategy explicitly to build a mask without culling -- the same mask, for more tests.
    """

    def blocked(
        self, bodies, sources, near, receivers, pair_limit: int = DEFAULT_PAIR_LIMIT
    ) -> jnp.ndarray:
        """See :meth:`BodyCulling.blocked`."""
        sources = jnp.asarray(sources, dtype=float)
        near = jnp.asarray(near, dtype=float)
        receivers = jnp.asarray(receivers, dtype=float)
        per_pass = receivers_per_pass(pair_limit, sources.shape[0])
        rows = [
            jnp.stack(
                [
                    _body_blocks(
                        body,
                        sources[None, :, :],
                        receivers[start : start + per_pass, None, :],
                        near[None, :],
                    )
                    for body in bodies
                ],
                axis=0,
            )
            for start in range(0, receivers.shape[0], per_pass)
        ]
        if not rows:
            return jnp.zeros((len(bodies), 0, sources.shape[0]), dtype=bool)
        return jnp.concatenate(rows, axis=1)


#: Levels per axis of the grid a point's place on the space-filling curve is read from. Ten bits
#: an axis: finer than any group is worth ordering within, and a key of thirty bits.
_ORDER_BITS = 10


def _spread_bits(values: np.ndarray) -> np.ndarray:
    """Each value's low :data:`_ORDER_BITS` bits, moved to every third bit position."""
    values = values.astype(np.uint64)
    spread = np.zeros_like(values)
    for bit in range(_ORDER_BITS):
        spread |= ((values >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit)
    return spread


def spatial_order(points) -> np.ndarray:
    """An ordering of points along a Morton (Z-order) curve, so neighbours in it are near in space.

    The points' bounding box is divided into ``2**10`` levels per axis, each point's three level
    numbers are interleaved bit by bit into one key, and the points are sorted by key. Any run of
    consecutive points in that order then lies in a compact region -- which is what makes a run a
    useful group to ask a question of once. Ties keep their input order, so the ordering is
    reproducible.

    Parameters
    ----------
    points : array_like, shape ``(n, 3)``

    Returns
    -------
    np.ndarray of int, shape ``(n,)``
        The indices of ``points`` in curve order.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    low = points.min(axis=0)
    extent = points.max(axis=0) - low
    levels = (1 << _ORDER_BITS) - 1
    scaled = np.where(extent > 0.0, (points - low) / np.where(extent > 0.0, extent, 1.0), 0.0)
    cell = np.minimum((scaled * levels).astype(np.int64), levels)
    key = (
        _spread_bits(cell[:, 0]) | (_spread_bits(cell[:, 1]) << 1) | (_spread_bits(cell[:, 2]) << 2)
    )
    return np.argsort(key, kind="stable")


class _Curve(eqx.Module):
    """Points ordered along the space-filling curve, and cut into groups at several sizes.

    Every size divides the one before it, so each group at one size is a whole number of groups
    at the next: a group that cannot be decided is split into its children along the same curve,
    and nothing is re-ordered on the way down.

    Attributes
    ----------
    order : np.ndarray of int, shape ``(n_padded,)``
        The point indices in curve order, padded to a whole number of the coarsest groups by
        repeating the last point. Repetition changes neither a group's summary (a maximum) nor any
        answer written back for it (the repeated pair is the same pair, written the same value
        twice).
    n_points : int
        How many of :attr:`order` are real.
    """

    order: np.ndarray
    n_points: int = eqx.field(static=True)

    @classmethod
    def of(cls, points, coarsest: int) -> _Curve:
        """``points`` in curve order, padded to a whole number of groups of ``coarsest``."""
        order = spatial_order(points)
        padded = -(-len(order) // coarsest) * coarsest
        return cls(
            order=np.concatenate([order, np.repeat(order[-1:], padded - len(order))]),
            n_points=len(order),
        )

    def members(self, size: int) -> np.ndarray:
        """The point indices of each group of ``size``, shape ``(n_groups, size)``."""
        return self.order.reshape(-1, size)

    def counts(self, size: int) -> np.ndarray:
        """How many members of each group of ``size`` are real, shape ``(n_groups,)``."""
        start = np.arange(len(self.order) // size) * size
        return np.clip(self.n_points - start, 0, size)

    def summary(self, features: np.ndarray, size: int) -> np.ndarray:
        """Each group's column-wise maximum of ``features``, shape ``(n_groups, n_features)``."""
        return features[self.members(size)].max(axis=1)


@dataclasses.dataclass(frozen=True, eq=False)
class _Groups:
    """One side of the tiles: its points along the curve, and each body's group summaries.

    Plain host arrays rather than a pytree, as the triangle grid is: nothing here is traced or
    differentiated, so a strategy carrying one through a custom gradient carries it untouched.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The points grouped.
    bodies : tuple of aquaflux.solids.Body
        The bodies the summaries are for.
    curve : _Curve
        The points in curve order, padded to whole groups of the coarsest size.
    summaries : tuple of tuple of np.ndarray
        ``summaries[b][level]`` is body ``b``'s summary of every group at that level, shape
        ``(n_groups, n_features)`` -- the column-wise maximum of its clearance features.
    """

    points: np.ndarray
    bodies: tuple
    curve: _Curve
    summaries: tuple

    @classmethod
    def of(cls, bodies, points, sizes) -> _Groups:
        """``points`` grouped at each of ``sizes``, coarsest first, and summarized per body."""
        curve = _Curve.of(points, sizes[0])
        summaries = []
        for body in bodies:
            features = np.asarray(body.clearance(points))
            summaries.append(tuple(curve.summary(features, size) for size in sizes))
        return cls(points=points, bodies=tuple(bodies), curve=curve, summaries=tuple(summaries))

    def serves(self, bodies, points) -> bool:
        """Whether these are the bodies and points the groups were formed from.

        By value, not identity: a body passed through a custom gradient comes back as a new
        object holding the same leaves. Compared on the host, leaf by leaf, since a stream asks
        once per pass and an eager device comparison per leaf would cost more than it guards.
        """
        if points.shape != self.points.shape or not np.array_equal(points, self.points):
            return False
        given, given_structure = jax.tree.flatten(tuple(bodies))
        held, held_structure = jax.tree.flatten(self.bodies)
        return given_structure == held_structure and all(
            a is b or np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(given, held, strict=True)
        )


def _vouched(body, receiver_summary, source_summary) -> np.ndarray:
    """Which tiles of every receiver group against every source group ``body`` vouches for.

    Shape ``(n_receiver_groups, n_source_groups)``. A tile's summary is the larger of its two
    groups' summaries -- see :meth:`~aquaflux.solids.Body.clearance`. Formed a band of receiver
    groups at a time, so the tiles-by-features array never holds more than
    :data:`~aquaflux.radiation.work.DEFAULT_PAIR_LIMIT` entries whatever the scene's size.
    """
    n_rows, n_cols = len(receiver_summary), len(source_summary)
    n_features = receiver_summary.shape[-1]
    clear = np.zeros((n_rows, n_cols), dtype=bool)
    if n_features == 0:
        return clear
    band = receivers_per_pass(DEFAULT_PAIR_LIMIT, n_cols * n_features)
    for start in range(0, n_rows, band):
        tile = np.maximum(
            receiver_summary[start : start + band, None, :], source_summary[None, :, :]
        )
        clear[start : start + band] = np.asarray(body.vouches(tile))
    return clear


def _vouched_pairs(body, receiver_summary, source_summary, rows, cols) -> np.ndarray:
    """Which of the listed tiles ``body`` vouches for, shape ``(n_tiles,)``.

    The same test as :func:`_vouched`, over a list of tiles rather than every combination, and
    in batches of the same bound.
    """
    clear = np.zeros(len(rows), dtype=bool)
    n_features = receiver_summary.shape[-1]
    if n_features == 0:
        return clear
    batch = receivers_per_pass(DEFAULT_PAIR_LIMIT, n_features)
    for start in range(0, len(rows), batch):
        tile = np.maximum(
            receiver_summary[rows[start : start + batch]],
            source_summary[cols[start : start + batch]],
        )
        clear[start : start + batch] = np.asarray(body.vouches(tile))
    return clear


def _check_sizes(name: str, sizes) -> None:
    """Refuse a group-size ladder that is empty, not positive, or does not nest."""
    if len(sizes) == 0:
        msg = f"ShaftCulling.{name} needs at least one group size"
        raise ValueError(msg)
    if any(size < 1 for size in sizes):
        msg = f"ShaftCulling.{name} sizes must be at least 1; got {sizes}"
        raise ValueError(msg)
    if any(coarse % fine for coarse, fine in itertools.pairwise(sizes)):
        msg = (
            f"ShaftCulling.{name} sizes must each divide the one before, so that an undecided "
            f"group splits into whole groups of the next size; got {sizes}"
        )
        raise ValueError(msg)


class ShaftCulling(BodyCulling):
    """Decide whole tiles of pairs where a body can prove it misses them; test the rest.

    Receivers and sources are each ordered along a space-filling curve and cut into groups. For
    each body and each (receiver block, source cluster) tile, the body's
    :meth:`~aquaflux.solids.Body.clearance` features are merged between the two groups'
    summaries and handed to :meth:`~aquaflux.solids.Body.vouches`; a tile it vouches for is
    recorded clear without a segment being tested. **A tile it cannot vouch for is split** into
    the tiles of the next, smaller group sizes and asked again -- a narrower shaft is easier to
    vouch for -- and only the tiles still undecided at the finest size are tested pair by pair by
    the compiled body test. The mask is the one :class:`EveryPair` builds; only the number of
    segments tested differs.

    What it saves is the share of pairs lying in vouched-for tiles, and that is a property of
    the scene: most of a chamber seen from inside its own convex region, none of a bent duct seen
    across its bend. A body that offers no features is tested pair by pair everywhere, at a
    small cost for the grouping. This is the strategy a mask is built with when none is chosen,
    at the default group sizes below.

    **Under a trace it tests every pair instead.** The grouping and the certificates are host
    work, so they need concrete positions; where a body's geometry or a point is traced -- a mask
    built inside ``jax.grad`` of a body's radius, say -- the answer is handed to
    :class:`EveryPair`, which gives the same mask and can be traced. Nothing is lost by it: the
    mask is a step function of that geometry, so its derivative is zero either way.

    Attributes
    ----------
    receiver_blocks : tuple of int
        Receivers per block, coarsest first; each divides the one before. Smaller blocks make
        narrower shafts, which more bodies can vouch for, and more tiles to compare -- which is
        why refinement only ever splits the tiles the coarser size could not decide.
    source_clusters : tuple of int
        Sources per cluster at the same levels, with the same trade. As many sizes as
        ``receiver_blocks``.
    sources : _Groups or None
        The sources' side of the tiles, formed once by :meth:`prepared`; unset, it is formed by
        every call. Either way the mask is the same.

    Notes
    -----
    **How far to refine depends on what a pair costs to test.** Each level costs a comparison per
    tile it asks about and saves the tests of the pairs it vouches for, so it pays where testing a
    pair is expensive -- a walk through a grid of triangles -- and can cost more than it saves
    where the test is a few comparisons, as an analytic body's is. The default refines to pairs of
    two, for the expensive case; a scene of analytic bodies alone is served as well or better by
    stopping at eight (``receiver_blocks=(32, 8)``, ``source_clusters=(32, 8)``).
    """

    receiver_blocks: tuple = eqx.field(static=True, default=(32, 8, 2))
    source_clusters: tuple = eqx.field(static=True, default=(32, 8, 2))
    sources: _Groups | None = None

    def __check_init__(self):
        _check_sizes("receiver_blocks", self.receiver_blocks)
        _check_sizes("source_clusters", self.source_clusters)
        if len(self.receiver_blocks) != len(self.source_clusters):
            msg = (
                "ShaftCulling needs as many source cluster sizes as receiver block sizes, one of "
                f"each per level; got {self.receiver_blocks} and {self.source_clusters}"
            )
            raise ValueError(msg)

    def blocked(
        self, bodies, sources, near, receivers, pair_limit: int = DEFAULT_PAIR_LIMIT
    ) -> jnp.ndarray:
        """See :meth:`BodyCulling.blocked`."""
        if any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree.leaves((bodies, sources, near, receivers))
        ):
            return EveryPair().blocked(bodies, sources, near, receivers, pair_limit)
        sources = np.asarray(sources, dtype=float)
        near = np.asarray(near, dtype=float)
        receivers = np.asarray(receivers, dtype=float)
        mask = np.zeros((len(bodies), len(receivers), len(sources)), dtype=bool)
        if len(receivers) == 0 or len(sources) == 0:
            return jnp.asarray(mask)
        blocks = _Groups.of(bodies, receivers, self.receiver_blocks)
        clusters = self._source_groups(bodies, sources)
        block, cluster = self.receiver_blocks[-1], self.source_clusters[-1]
        for index, body in enumerate(bodies):
            rows, cols = self._undecided(index, body, blocks, clusters)
            self._test_tiles(
                body,
                sources,
                near,
                receivers,
                blocks.curve.members(block)[rows],
                clusters.curve.members(cluster)[cols],
                pair_limit,
                out=mask[index],
            )
        return jnp.asarray(mask)

    def prepared(self, bodies, sources) -> ShaftCulling:
        """With the sources' side of the tiles formed once. See :meth:`BodyCulling.prepared`.

        What a pass needs of the sources -- their order along the curve and every body's
        clearance summary of every cluster at every level -- depends on the sources and the
        bodies alone, so a stream of passes forms it here rather than once per pass.
        """
        sources = np.asarray(sources, dtype=float)
        if len(sources) == 0 or any(
            isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((bodies, sources))
        ):
            return self
        groups = _Groups.of(bodies, sources, self.source_clusters)
        return eqx.tree_at(lambda strategy: strategy.sources, self, groups, is_leaf=_is_none)

    def _source_groups(self, bodies, sources) -> _Groups:
        """The sources' side: as prepared, if it was, else formed now."""
        if self.sources is None:
            return _Groups.of(bodies, sources, self.source_clusters)
        if not self.sources.serves(bodies, sources):
            msg = (
                "this ShaftCulling was prepared for other sources or other bodies than it was "
                "given; its clusters' summaries would certify tiles against the wrong geometry, "
                "so prepare it again for these, or use one that was not prepared"
            )
            raise ValueError(msg)
        return self.sources

    def certified_pairs(self, bodies, sources, receivers) -> np.ndarray:
        """How many source-receiver pairs each body is certified to miss without a test.

        The measure of what the strategy saves on a given scene: the pairs it does not test are
        these, per body, whichever level of refinement vouched for them.

        Parameters
        ----------
        bodies : sequence of aquaflux.solids.Body
        sources : array_like, shape ``(n_sources, 3)``
        receivers : array_like, shape ``(n_receivers, 3)``

        Returns
        -------
        np.ndarray of int, shape ``(n_bodies,)``
        """
        sources = np.asarray(sources, dtype=float)
        receivers = np.asarray(receivers, dtype=float)
        if len(receivers) == 0 or len(sources) == 0:
            return np.zeros(len(bodies), dtype=np.int64)
        blocks = _Groups.of(bodies, receivers, self.receiver_blocks)
        clusters = self._source_groups(bodies, sources)
        row_counts = blocks.curve.counts(self.receiver_blocks[-1])
        col_counts = clusters.curve.counts(self.source_clusters[-1])
        certified = []
        for index, body in enumerate(bodies):
            rows, cols = self._undecided(index, body, blocks, clusters)
            tested = int(np.sum(row_counts[rows] * col_counts[cols]))
            certified.append(len(receivers) * len(sources) - tested)
        return np.array(certified, dtype=np.int64)

    def _undecided(self, index, body, blocks, clusters):
        """The finest-level tiles ``body`` could not vouch for, as ``(rows, cols)`` group indices.

        ``index`` is the body's place in both groups' summaries. Every tile at the coarsest level
        is asked; each one refused is split into its children at the next level, those whose
        groups hold any real point are asked, and so on. A child made wholly of padding is
        dropped rather than asked: it holds no pair.
        """
        receiver_summaries, source_summaries = blocks.summaries[index], clusters.summaries[index]
        levels = list(zip(self.receiver_blocks, self.source_clusters, strict=True))
        rows, cols = np.nonzero(~_vouched(body, receiver_summaries[0], source_summaries[0]))
        for level, (coarse, (block, cluster)) in enumerate(itertools.pairwise(levels), start=1):
            split_rows, split_cols = coarse[0] // block, coarse[1] // cluster
            children_rows = rows[:, None, None] * split_rows + np.arange(split_rows)[:, None]
            children_cols = cols[:, None, None] * split_cols + np.arange(split_cols)[None, :]
            rows, cols = (
                side.ravel() for side in np.broadcast_arrays(children_rows, children_cols)
            )
            real = (blocks.curve.counts(block)[rows] > 0) & (
                clusters.curve.counts(cluster)[cols] > 0
            )
            rows, cols = rows[real], cols[real]
            refused = ~_vouched_pairs(
                body, receiver_summaries[level], source_summaries[level], rows, cols
            )
            rows, cols = rows[refused], cols[refused]
        return rows, cols

    def _test_tiles(self, body, sources, near, receivers, rows, cols, pair_limit, out):
        """Test the undecided tiles pair by pair, writing each answer into ``out`` in place.

        ``rows`` and ``cols`` hold each tile's point indices. Tiles go in batches of one shape,
        padded to a power of two by repeating the last tile, so a pass compiles a couple of
        programs however many tiles it has; the repeated tile writes its own answer again.
        """
        per_tile = rows.shape[1] * cols.shape[1]
        per_batch = max(1, pair_limit // per_tile)
        for start in range(0, len(rows), per_batch):
            batch_rows = rows[start : start + per_batch]
            batch_cols = cols[start : start + per_batch]
            width = padded_length(len(batch_rows))
            pad = np.repeat(np.arange(len(batch_rows))[-1:], width - len(batch_rows))
            take = np.concatenate([np.arange(len(batch_rows)), pad])
            batch_rows, batch_cols = batch_rows[take], batch_cols[take]
            answer = _body_blocks(
                body,
                jnp.asarray(sources[batch_cols])[:, None, :, :],
                jnp.asarray(receivers[batch_rows])[:, :, None, :],
                jnp.asarray(near[batch_cols])[:, None, :],
            )
            out[batch_rows[:, :, None], batch_cols[:, None, :]] = np.asarray(answer)


def _is_none(value) -> bool:
    return value is None


def culling_or_default(body_culling: BodyCulling | None) -> BodyCulling:
    """The strategy a mask build uses: the one given, or :class:`ShaftCulling` if none was.

    Parameters
    ----------
    body_culling : BodyCulling or None

    Returns
    -------
    BodyCulling
    """
    return ShaftCulling() if body_culling is None else body_culling
