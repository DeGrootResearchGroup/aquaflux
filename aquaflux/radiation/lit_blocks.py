"""Receivers in small blocks, each with the list of facets that can light it: the gather's layout.

A facet whose plane a receiver lies behind sends it nothing when the facet's profile is dark
behind itself (:attr:`~aquaflux.radiation.profiles.Profile.dark_behind`), and around a lamp
that is about half of every receiver's facets. A gather laid out as receivers by *every* facet
computes a solid angle, a path and a shadow for each of those pairs and multiplies it by zero.
Laid out as **blocks of neighbouring receivers, each against its own list of facets**, it does
not form them at all: a facet is left off a block's list where the block's bounding box lies
wholly behind its plane (:meth:`~aquaflux.radiation.back_faces.BackFaces.lit_facets`), which
proves it for every receiver of the block. What the lists leave out is exactly zero, so the
field is the same up to the order its terms are added in.

**The lists are padded to a few widths, not one.** Blocks differ: one beside the lamp sees a
fraction of it, one far off nearly half, one in line with its end most of it. Padding every
list to the longest would give most of the saving back, so blocks are sorted by the length of
their list and cut into **segments**, each padded to a width rounded up to one of a few steps
per doubling. A traced program's shapes are then drawn from a short ladder, so however many
blocks a call has, it compiles a handful of programs.

**The layout is decided on the host from concrete positions**, like a shadow mask, and handed
to the traced gather as index arrays. Where the positions cannot be read -- receivers that are
themselves traced -- the lists are every facet, and the gather does the full work in the same
layout, with the same answer.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import numpy as np

from aquaflux.morton import morton_order

__all__ = ["LitSegment", "lit_segments", "rounded_width"]

#: Receivers per block. Small, because a block's box has to lie behind a facet's plane for the
#: facet to be left off its list, and a small box is proven behind far more planes; and not one,
#: because each block's lists are a facet index a pair, which a block shares among its receivers.
BLOCK = 8

#: Steps per doubling of the padded widths a segment's lists are rounded up to.
_STEPS_PER_DOUBLING = 4


class LitSegment(eqx.Module):
    """Blocks of receivers whose facet lists are padded to one width.

    Attributes
    ----------
    rows : np.ndarray of int, shape ``(n_blocks, block)``
        Each block's receivers, as indices into the points. A block short of receivers is filled
        with the index one past the last point, which the gather reads as nothing.
    facets : np.ndarray of int, shape ``(n_blocks, width)`` or ``(1, width)``
        Each block's facets, as indices into the surface set; filled past its list by repeating
        its last entry. **One row shared by every block** where every block is listed against
        every facet, so a full listing costs nothing per block.
    valid : np.ndarray of bool, shape as :attr:`facets`
        Which entries of :attr:`facets` are on the block's list.
    """

    rows: np.ndarray
    facets: np.ndarray
    valid: np.ndarray

    @property
    def shared(self) -> bool:
        """Whether one list serves every block."""
        return self.facets.shape[0] == 1 and self.rows.shape[0] != 1

    @property
    def width(self) -> int:
        """How many facets each block is gathered against."""
        return int(self.facets.shape[1])

    @property
    def block(self) -> int:
        """How many receivers each block holds."""
        return int(self.rows.shape[1])


def rounded_width(count: int, cap: int) -> int:
    """``count`` rounded up to a step of a ladder with a few steps per doubling, at most ``cap``.

    Parameters
    ----------
    count : int
        The longest list in a segment.
    cap : int
        The width no list can exceed: every facet the lists are drawn from.

    Returns
    -------
    int
        At least one, so a segment of blocks with empty lists still has a shape.
    """
    count = max(1, int(count))
    step = max(1, (1 << (count - 1).bit_length()) // (2 * _STEPS_PER_DOUBLING))
    return min(max(1, int(cap)), -(-count // step) * step)


def lit_segments(points, facets, facing=None, *, block: int = BLOCK, segments=None):
    """The gather's layout of ``points`` against ``facets``: blocks, their lists, cut into segments.

    Parameters
    ----------
    points : array_like, shape ``(n_points, 3)``
        Receiver positions. Where they are traced, or where ``facing`` is not given, every block
        is listed against every facet and the blocks are the points in the order given.
    facets : np.ndarray of int, shape ``(n_facets,)``
        The facets to gather, as indices into the surface set.
    facing : BackFaces, optional
        The facets' planes. Given, the points are ordered along a space-filling curve so that a
        block is a compact cluster, and a facet whose plane a block lies wholly behind is left
        off its list. Give it only where every facet in ``facets`` emits with a profile dark
        behind itself; elsewhere a facet left off would be light not gathered.
    block : int, optional
        Receivers per block.
    segments : int, optional
        Cut the blocks into this many segments of equal block count, each padded to its own
        width -- so the shapes depend only on the point count and the ladder, which is what a
        caller compiling one program per chunk of points wants. Unset, a segment is every block
        whose list rounds to one width, which pads least.

    Returns
    -------
    tuple of LitSegment
        Every point in exactly one block of one segment.
    """
    facets = np.asarray(facets, dtype=np.int64)
    n_points = int(np.shape(points)[0])
    if n_points == 0 or len(facets) == 0:
        return ()
    traced = isinstance(points, jax.core.Tracer)
    if facing is None or traced:
        order = np.arange(n_points)
    else:
        order = morton_order(np.asarray(points, dtype=float))
    n_blocks = -(-n_points // block)
    padded = np.concatenate([order, np.full(n_blocks * block - n_points, -1)])
    rows = np.where(padded < 0, n_points, padded).reshape(n_blocks, block)
    if facing is None or traced:
        return (
            LitSegment(
                rows=rows.astype(np.int32),
                facets=facets[None, :].astype(np.int32),
                valid=np.ones((1, len(facets)), dtype=bool),
            ),
        )
    else:
        # A padded slot repeats its block's last receiver, which leaves the box unchanged.
        members = np.where(padded < 0, order[-1], padded).reshape(n_blocks, block)
        offsets, lit = facing.lit_facets(np.asarray(points, dtype=float), members, facets)
    counts = np.diff(offsets)
    by_length = np.argsort(counts, kind="stable")
    parts = _parts(counts[by_length], len(facets), segments)
    return tuple(
        _segment(rows, offsets, lit, facets, by_length[start:stop], n_points, pad_to)
        for start, stop, pad_to in parts
    )


def _parts(sorted_counts: np.ndarray, cap: int, segments):
    """Where to cut blocks sorted by list length, as ``(start, stop, blocks)`` triples.

    ``blocks`` is how many blocks the segment holds once padded: its own count, or, for equal
    segments, the same for every one, so a short last segment keeps the others' shape.
    """
    n_blocks = len(sorted_counts)
    if segments is None:
        widths = np.array([rounded_width(count, cap) for count in sorted_counts])
        cuts = np.flatnonzero(np.diff(widths)) + 1
        bounds = np.concatenate([[0], cuts, [n_blocks]])
        return [(int(a), int(b), int(b - a)) for a, b in itertools.pairwise(bounds)]
    size = -(-n_blocks // segments)
    return [(start, min(start + size, n_blocks), size) for start in range(0, n_blocks, size)]


def _segment(rows, offsets, lit, candidates, blocks, n_points, pad_to) -> LitSegment:
    """One segment: ``blocks``' rows and lists, padded to a rounded width and ``pad_to`` blocks."""
    counts = offsets[blocks + 1] - offsets[blocks]
    width = rounded_width(counts.max(initial=0), len(candidates))
    starts = offsets[blocks]
    # Past its list, a block repeats its last entry -- the first candidate, for an empty list --
    # so every index is a real facet and the entry is simply not counted.
    lit = np.concatenate([lit, candidates[:1]])
    within = np.arange(width)[None, :] < counts[:, None]
    last = np.where(counts > 0, starts + counts - 1, len(lit) - 1)
    position = np.where(within, starts[:, None] + np.arange(width)[None, :], last[:, None])
    facets = lit[position]
    block_rows = rows[blocks]
    extra = pad_to - len(blocks)
    if extra:
        block_rows = np.concatenate([block_rows, np.full((extra, rows.shape[1]), n_points)])
        facets = np.concatenate([facets, np.repeat(facets[-1:], extra, axis=0)])
        within = np.concatenate([within, np.zeros((extra, width), dtype=bool)])
    return LitSegment(
        rows=block_rows.astype(np.int32),
        facets=facets.astype(np.int32),
        valid=within,
    )
