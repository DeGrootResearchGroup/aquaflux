"""Transparent solids: where light bends, and how much of it gets across.

A lamp's quartz sleeve, a quartz window, the wall of a flow cell: a solid that light passes
**through** rather than stopping at. Two things happen at each of its surfaces, and neither can be
written as a scalar transmittance on a straight line.

- **Refraction** bends the path by Snell's law, ``n_1 sin(theta_1) = n_2 sin(theta_2)``, so the
  light reaching a point from a source behind a curved surface arrives from a different direction,
  and spread over a different solid angle, than the straight line says. That moves light from one
  place to another rather than only dimming it.
- **Fresnel reflection** turns a share of the light back at each surface, a share that grows with
  the angle of incidence and reaches all of it past the critical angle (total internal reflection).

**What is here.** A :class:`Transparent` region is a convex solid with a refractive index and an
absorbing medium of its own; regions nest (a quartz cylinder holding an air cylinder is a sleeve),
and :class:`Media` holds them with the surrounding medium's own index and absorption. Between a
source and a receiver the path crosses the boundary of each region that holds exactly one of them,
once each and in a fixed order -- out of every region round the source, into every region round the
receiver -- because a straight leg leaving a convex body never returns to it. Only where each
crossing lies is unknown, and Fermat's principle fixes it: the path makes its optical length
``sum n_j |x_{j+1} - x_j|`` stationary over the crossing points, each held to its surface. That is a
small system per path, solved by Newton's method and differentiated by the implicit function
theorem, so the derivative with respect to an index, an absorption or a surface's position is the
exact one and the iterations are never put on the tape.

**What the path carries.** The direction it leaves the source in (for the source's angular
distribution), the direction it arrives from (for the solid angle the source fills at the
receiver), the product of the Fresnel transmittances at its crossings, and the absorption along
each leg in that leg's medium.

**A region holding neither end** -- a neighbouring lamp's sleeve -- is either missed or passed
**through**: the light from a source to a point beyond a neighbour reaches it by every path that does
one or the other, so each is a **route** of its own (:meth:`Chain.routes`). A route through a region
enters it and leaves it again, at whatever depth of its nesting -- through a sleeve's quartz alone, or
across its air gap too -- each crossing bent and each counted by its Fresnel transmittance. A path on
a route that meets a region it does not pass through carries nothing: that light belongs to the route
through it. At most one such region is passed by one path; light that would have to pass two is not
followed. A route through a region can join two points by more than one path -- past either side of
a lamp's arc inside a sleeve -- so it is solved from several starting points across the region and
every distinct path found is kept.

**What is not here.** Light that reflects on its way -- off the inside of a sleeve and out the far
side, or back and forth inside the quartz -- is counted as lost at each reflection, not followed.
"""

from __future__ import annotations

import dataclasses
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.triangles import padded_length
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, PASS_PAIRS
from aquaflux.solids import ConvexSolid
from aquaflux.vectors import dot, norm, norm_squared

__all__ = [
    "Chain",
    "Media",
    "Paths",
    "Transparent",
    "fresnel_transmittance",
    "solve_paths",
    "straight_reach",
]

#: Steps allowed per path. A path that exists converges quadratically once near it, in a handful;
#: one that does not -- no transmitted path joins the two points, say beyond the critical angle --
#: stops where the optical length stops falling, or runs to this, and is reported as absent.
_NEWTON_STEPS = 60

#: Halvings of a step allowed before it is taken as unable to shorten the path.
_HALVINGS = 40

#: A Newton step this short, relative to the separation of the path's ends, is taken without
#: asking the optical length to fall: the fall would be below the length's own rounding.
_BASIN = 1e-6

#: Convergence of a path: the Newton step left, relative to the separation of its ends. Rounding
#: leaves a few parts in 1e16 of the coordinates; this leaves room for it.
_PATH_TOLERANCE = 1e-10

#: How far a crossing may stand outside the body's other faces, relative to the separation of the
#: path's ends, before it is not on the body's boundary at all but on the extension of a face.
_ON_BOUNDARY = 1e-9

#: Where a route through a region starts its search, across the region from the straight line: a
#: fraction of the way to the region's edge, either way along two directions across the line.
_START_REACH = 0.6

#: Two paths on one route closer than this, relative to the separation of their ends, are one path
#: found from two starting points, and counted once.
_SAME_PATH = 1e-6

#: How close a source or receiver may come to a region's surface, relative to the scene's size. A
#: point there has no well-defined medium, and a path from it has a leg of zero length.
_ON_INTERFACE = 1e-9


def fresnel_transmittance(cos_incident, n_from, n_to) -> jnp.ndarray:
    """Share of unpolarized light transmitted across a smooth interface.

    The mean of the two polarizations' Fresnel transmittances, ``1 - (r_s^2 + r_p^2) / 2`` with

    ``r_s = (n_1 cos(theta_i) - n_2 cos(theta_t)) / (n_1 cos(theta_i) + n_2 cos(theta_t))``,
    ``r_p = (n_2 cos(theta_i) - n_1 cos(theta_t)) / (n_2 cos(theta_i) + n_1 cos(theta_t))``,

    and zero beyond the critical angle, where all of it is reflected. The same both ways across the
    interface for one pair of angles, which is why a path's transmittance does not depend on which
    end the light started from.

    Parameters
    ----------
    cos_incident : array_like
        Cosine of the angle of incidence, measured from the surface normal; its sign is ignored.
    n_from, n_to : array_like
        Refractive indices of the medium the light arrives in and the one it enters. Broadcast
        against ``cos_incident``.

    Returns
    -------
    jnp.ndarray
        Transmittance in ``[0, 1]``.
    """
    cos_i = jnp.minimum(jnp.abs(jnp.asarray(cos_incident, dtype=float)), 1.0)
    n_from = jnp.asarray(n_from, dtype=float)
    n_to = jnp.asarray(n_to, dtype=float)
    sin_t_squared = (n_from / n_to) ** 2 * (1.0 - cos_i**2)
    total = sin_t_squared >= 1.0
    # The root is guarded inside its argument, so past the critical angle neither the value nor
    # its derivative is a NaN that the selection below would not stop.
    cos_t = jnp.sqrt(jnp.where(total, 1.0, 1.0 - sin_t_squared))
    s_sum = n_from * cos_i + n_to * cos_t
    p_sum = n_to * cos_i + n_from * cos_t
    # At grazing incidence both sums can vanish together only for a zero index, which is refused
    # where indices are given; a zero cosine on one side alone leaves them positive.
    r_s = (n_from * cos_i - n_to * cos_t) / jnp.where(s_sum == 0.0, 1.0, s_sum)
    r_p = (n_to * cos_i - n_from * cos_t) / jnp.where(p_sum == 0.0, 1.0, p_sum)
    return jnp.where(total, 0.0, 1.0 - 0.5 * (r_s**2 + r_p**2))


class Transparent(eqx.Module):
    """A convex solid light passes through, with its own refractive index and absorption.

    Attributes
    ----------
    body : aquaflux.solids.ConvexSolid
        Where it is. Convex, so a straight leg leaving it never returns and a path crosses its
        surface once on the way out or in.
    refractive_index : jnp.ndarray
        Its index, a differentiable scalar. 1.5048 for fused silica at 254 nm (Malitson, 1965).
    absorption : Absorption or None
        The medium inside it; unset, it absorbs nothing.
    inside : tuple of Transparent
        Regions wholly inside this one, each of which is its own medium there: a sleeve is the
        quartz cylinder with the air gap ``inside`` it. They must not overlap one another or reach
        out of this one; that is checked at every point classified, and a point found inside a
        region but not inside the region holding it is refused.
    """

    body: ConvexSolid
    refractive_index: jnp.ndarray
    absorption: Absorption | None = None
    inside: tuple = ()

    def __init__(self, body, refractive_index, absorption=None, inside=()):
        if not isinstance(body, ConvexSolid):
            msg = (
                f"a transparent region must be a convex solid; got {type(body).__name__}. A path "
                "crosses a convex surface once on its way out or in, which is what fixes the order "
                "of its crossings. Describe a hollow body as a solid region with another inside it."
            )
            raise TypeError(msg)
        self.body = body
        self.refractive_index = jnp.asarray(refractive_index, dtype=float)
        self.absorption = absorption
        self.inside = tuple(inside)
        for child in self.inside:
            if not isinstance(child, Transparent):
                msg = f"a region's inside holds Transparent regions; got {type(child).__name__}"
                raise TypeError(msg)


class Media(eqx.Module):
    """The medium light travels through, and the transparent regions standing in it.

    Attributes
    ----------
    refractive_index : jnp.ndarray
        The surrounding medium's index -- 1.376 for water at 254 nm (Hale & Querry, 1973) -- a
        differentiable scalar.
    regions : tuple of Transparent
        The outermost regions; each may hold others.
    absorption : Absorption or None
        The surrounding medium's absorption; unset, it absorbs nothing.
    """

    refractive_index: jnp.ndarray
    regions: tuple = ()
    absorption: Absorption | None = None

    def __init__(self, refractive_index, regions=(), absorption=None):
        self.refractive_index = jnp.asarray(refractive_index, dtype=float)
        self.regions = tuple(regions)
        self.absorption = absorption
        for region in self.regions:
            if not isinstance(region, Transparent):
                msg = f"regions are Transparent regions; got {type(region).__name__}"
                raise TypeError(msg)

    @property
    def nodes(self) -> tuple[Transparent, ...]:
        """Every region, outermost first, each followed by those inside it (depth first)."""
        return tuple(node for node, _ in _walk(self.regions, -1))

    @property
    def parents(self) -> np.ndarray:
        """For each of :attr:`nodes`, the index of the region holding it, ``-1`` for none."""
        return np.asarray([parent for _, parent in _walk(self.regions, -1)], dtype=int)

    def index_of(self, node: int) -> jnp.ndarray:
        """The refractive index of a region by its place in :attr:`nodes`, ``-1`` the surroundings."""
        return self.refractive_index if node < 0 else self.nodes[node].refractive_index

    def absorption_of(self, node: int) -> Absorption | None:
        """The absorbing medium of a region by its place in :attr:`nodes`, ``-1`` the surroundings."""
        return self.absorption if node < 0 else self.nodes[node].absorption

    def region_of(self, points, what: str = "point") -> np.ndarray:
        """Which medium each point is in: the innermost region holding it, ``-1`` for none.

        Host work on concrete positions: the answer decides which crossings a path has, which is
        the shape of the program that follows it.

        Parameters
        ----------
        points : array_like, shape ``(n, 3)``
        what : str
            What the points are, for the error message.

        Returns
        -------
        numpy.ndarray of int, shape ``(n,)``

        Raises
        ------
        ValueError
            If a point is within a rounding of a region's surface, where its medium is not
            defined, or is inside a region but not inside the region holding that one.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        region = np.full(points.shape[0], -1, dtype=int)
        if not self.regions or not points.shape[0]:
            return region
        scale = float(np.max(np.abs(points))) + float(np.ptp(points, axis=0).max())
        tolerance = _ON_INTERFACE * max(scale, 1.0)
        parents = self.parents
        depth = np.zeros(len(parents), dtype=int)
        for node, parent in enumerate(parents):
            depth[node] = 0 if parent < 0 else depth[parent] + 1
        inside = np.zeros((len(parents), points.shape[0]), dtype=bool)
        for node, region_node in enumerate(self.nodes):
            distance = np.asarray(region_node.body.signed_distance(jnp.asarray(points)))
            close = np.flatnonzero(np.abs(distance) <= tolerance)
            if len(close):
                msg = (
                    f"{len(close)} {what}(s) lie on the surface of transparent region {node} "
                    f"(first few: {close[:8].tolist()}), where which medium they are in is not "
                    "defined. Move them off the surface."
                )
                raise ValueError(msg)
            inside[node] = distance < 0.0
        for node, parent in enumerate(parents):
            if parent >= 0:
                stray = np.flatnonzero(inside[node] & ~inside[parent])
                if len(stray):
                    msg = (
                        f"{len(stray)} {what}(s) are inside transparent region {node} but not "
                        f"inside region {parent}, which holds it (first few: "
                        f"{stray[:8].tolist()}). A region must lie wholly inside the one holding "
                        "it."
                    )
                    raise ValueError(msg)
        for node in np.argsort(depth, kind="stable"):
            region[inside[node]] = node
        claimed = inside.sum(axis=0)
        expected = np.where(region < 0, 0, depth[np.maximum(region, 0)] + 1)
        overlapping = np.flatnonzero(claimed != expected)
        if len(overlapping):
            msg = (
                f"{len(overlapping)} {what}(s) are inside two transparent regions neither of which "
                f"holds the other (first few: {overlapping[:8].tolist()}). Regions must not "
                "overlap: nest one inside the other, or keep them apart."
            )
            raise ValueError(msg)
        return region

    def region_of_facets(self, surfaces) -> np.ndarray:
        """The medium each facet lies in, read at its centroid, refusing one that straddles a surface.

        Parameters
        ----------
        surfaces : Surfaces
            Read for its vertices and centroids, which must be concrete.

        Returns
        -------
        numpy.ndarray of int, shape ``(n_facets,)``

        Raises
        ------
        ValueError
            If a facet's corners are not all in the medium its centroid is in: a facet must lie in
            one medium. And as :meth:`region_of`, for a corner or centroid on a region's surface.
        """
        vertices = np.asarray(surfaces.vertices, dtype=float)
        corners = self.region_of(vertices.reshape(-1, 3), "facet vertex").reshape(-1, 3)
        centres = self.region_of(np.asarray(surfaces.centroid), "facet centroid")
        split = np.flatnonzero(np.any(corners != centres[:, None], axis=1))
        if len(split):
            msg = (
                f"{len(split)} facet(s) cross the surface of a transparent region (first few: "
                f"{split[:8].tolist()}); a facet must lie in one medium."
            )
            raise ValueError(msg)
        return centres


def _walk(regions, parent):
    """Depth-first ``(region, parent index)`` pairs, the parent indexing the same sequence."""
    order = []

    def visit(nodes, holder):
        for node in nodes:
            order.append((node, holder))
            visit(node.inside, len(order) - 1)

    visit(regions, parent)
    return order


@dataclasses.dataclass(frozen=True)
class Chain:
    """The crossings a path makes between two media, and the medium of each leg: host metadata.

    Attributes
    ----------
    crossings : tuple of tuple of (int, bool)
        In order from the source, each region crossed and whether the path leaves it (``True``)
        or enters it.
    legs : tuple of int
        The region each leg runs in, ``-1`` the surroundings; one more than the crossings.
    beside : tuple of tuple of int
        For each leg, the regions it must miss: every region that is not crossed, does not hold the
        leg's medium and is not that medium. A leg meeting one is on another route.
    passing : tuple of int
        The regions the path passes through, holding neither end, outermost first: entered in this
        order and left in the reverse. Empty for the route that passes through none.
    pass_at : int
        Where in :attr:`crossings` the pass begins; meaningless when :attr:`passing` is empty.
    """

    crossings: tuple
    legs: tuple
    beside: tuple
    passing: tuple = ()
    pass_at: int = 0

    @classmethod
    def between(
        cls, parents: np.ndarray, source: int, receiver: int, through: int | None = None
    ) -> Chain:
        """The chain from a source in region ``source`` to a receiver in region ``receiver``.

        Parameters
        ----------
        parents : numpy.ndarray of int
            :attr:`Media.parents`.
        source, receiver : int
            The regions the ends are in, ``-1`` the surroundings.
        through : int, optional
            A region holding neither end, to pass through: the path enters every region holding it
            down from one in a leg's own medium, then leaves them again. Unset, the path passes
            through none.

        Raises
        ------
        ValueError
            If ``through`` holds an end, or no leg's medium holds it.
        """
        up, down = _lineage(parents, source), _lineage(parents, receiver)
        common = next((node for node in up if node in down), -1)
        leaving = up[: up.index(common)] if common >= 0 else up
        entering = down[: down.index(common)] if common >= 0 else down
        crossings = [(node, True) for node in leaving] + [
            (node, False) for node in reversed(entering)
        ]
        legs = [source, *(int(parents[node]) if out else node for node, out in crossings)]
        passing, pass_at = (), 0
        if through is not None:
            line = _lineage(parents, through)
            if set(line) & (set(up) | set(down)):
                msg = f"region {through} holds an end of the path, so it is not passed through"
                raise ValueError(msg)
            top = next((k for k, node in enumerate(line) if int(parents[node]) in legs), None)
            if top is None:
                msg = f"no leg of the path runs in a medium holding region {through}"
                raise ValueError(msg)
            passing = tuple(reversed(line[: top + 1]))
            pass_at = legs.index(int(parents[passing[0]]))
            pass_crossings = [(node, False) for node in passing] + [
                (node, True) for node in reversed(passing)
            ]
            crossings[pass_at:pass_at] = pass_crossings
            legs = [source, *(int(parents[node]) if out else node for node, out in crossings)]
        crossed = {node for node, _ in crossings}
        beside = tuple(
            tuple(
                node
                for node in range(len(parents))
                if node not in _lineage(parents, leg) and node not in crossed
            )
            for leg in legs
        )
        return cls(
            crossings=tuple(crossings),
            legs=tuple(int(leg) for leg in legs),
            beside=beside,
            passing=passing,
            pass_at=pass_at,
        )

    @classmethod
    def routes(cls, parents: np.ndarray, source: int, receiver: int) -> tuple[Chain, ...]:
        """Every route between two media: through no region, then through each region it can pass.

        A route through a region holding neither end is one per depth of that region's nesting: a
        sleeve gives one through its quartz alone and one across its air gap too. Between two
        points in one medium there is no route through nothing -- that is a straight line, the
        direct gather's -- so only the routes through a region are listed.

        Parameters
        ----------
        parents : numpy.ndarray of int
        source, receiver : int

        Returns
        -------
        tuple of Chain
        """
        held = set(_lineage(parents, source)) | set(_lineage(parents, receiver))
        found = [] if source == receiver else [cls.between(parents, source, receiver)]
        for node in range(len(parents)):
            if node in held:
                continue
            try:
                found.append(cls.between(parents, source, receiver, through=node))
            except ValueError:
                continue
        return tuple(found)

    @property
    def n_crossings(self) -> int:
        """How many surfaces the path crosses."""
        return len(self.crossings)

    @property
    def n_starts(self) -> int:
        """How many starting points the path is solved from: four through a region, else one."""
        return 4 if self.passing else 1


def _lineage(parents, node) -> list[int]:
    """``node`` and every region holding it, innermost first; empty for the surroundings."""
    line = []
    while node >= 0:
        line.append(int(node))
        node = parents[node]
    return line


class Paths(eqx.Module):
    """Transmitted paths between pairs of points, and what each carries.

    Attributes
    ----------
    departure : jnp.ndarray, shape ``(..., 3)``
        Unit direction each path leaves its source in.
    arrival : jnp.ndarray, shape ``(..., 3)``
        Unit direction from the receiver back along the path's last leg: where the light seems to
        come from.
    transmittance : jnp.ndarray, shape ``(...)``
        What gets across: the Fresnel transmittances of the crossings and the absorption of each
        leg in its own medium.
    valid : jnp.ndarray of bool, shape ``(...)``
        Whether the path exists: the solve converged, every crossing is on its body's boundary and
        goes the way the chain says, no leg meets a region it must miss, and -- on a route through a
        region -- no earlier starting point found the same path. A path that does not exist carries
        nothing.
    points : jnp.ndarray, shape ``(..., n_crossings, 3)``
        Where it crosses each surface.
    """

    departure: jnp.ndarray
    arrival: jnp.ndarray
    transmittance: jnp.ndarray
    valid: jnp.ndarray
    points: jnp.ndarray


def solve_paths(media: Media, chain: Chain, sources, receivers) -> Paths:
    """The transmitted path between each source and receiver, for one chain of crossings.

    Parameters
    ----------
    media : Media
        Its indices, absorptions and region geometry are live: the path's derivative with respect
        to each comes from the implicit function theorem on the converged path.
    chain : Chain
        The crossings, from :meth:`Chain.between` or :meth:`Chain.routes`; every pair given must have
        it.
    sources, receivers : array_like, shape ``(..., 3)``
        Broadcast against each other.

    Returns
    -------
    Paths
        Shaped as the broadcast pairs; on a route through a region, with one more leading axis of
        :attr:`Chain.n_starts`, a path per starting point, each distinct path valid once.
    """
    return _solve_paths(
        media, chain, jnp.asarray(sources, dtype=float), jnp.asarray(receivers, dtype=float)
    )


@eqx.filter_jit
def _solve_paths(media: Media, chain: Chain, sources, receivers) -> Paths:
    """:func:`solve_paths`, compiled once per chain and shape.

    Compiled because the solve is a loop with a line search and two curvatures in it, which run
    one operation at a time cost seconds per call however few paths are asked for; inside a caller's
    own trace it is inlined.
    """
    shape = jnp.broadcast_shapes(sources.shape, receivers.shape)
    flat_sources = jnp.broadcast_to(sources, shape).reshape(-1, 3)
    flat_receivers = jnp.broadcast_to(receivers, shape).reshape(-1, 3)
    if not chain.passing:
        paths = jax.vmap(lambda s, r: _one_path(media, chain, s, r, None))(
            flat_sources, flat_receivers
        )
        return jax.tree.map(lambda leaf: leaf.reshape(*shape[:-1], *leaf.shape[1:]), paths)

    def from_every_start(source, receiver):
        aims = jax.lax.stop_gradient(_starts(media, chain, source, receiver))
        return jax.vmap(lambda aim: _one_path(media, chain, source, receiver, aim))(aims)

    paths = jax.vmap(from_every_start)(flat_sources, flat_receivers)
    paths = jax.tree.map(lambda leaf: jnp.moveaxis(leaf, 1, 0), paths)
    paths = _distinct(paths, norm(flat_receivers - flat_sources))
    return jax.tree.map(
        lambda leaf: leaf.reshape(leaf.shape[0], *shape[:-1], *leaf.shape[2:]), paths
    )


def _distinct(paths: Paths, separation) -> Paths:
    """``paths`` from every start, each valid only where no earlier start found the same path."""
    scale = jnp.where(separation > 0.0, separation, 1.0)
    valid = [paths.valid[0]]
    for start in range(1, paths.valid.shape[0]):
        repeated = jnp.zeros_like(paths.valid[start])
        for earlier in range(start):
            apart = jnp.max(norm(paths.points[start] - paths.points[earlier]), axis=-1)
            repeated = repeated | (paths.valid[earlier] & (apart <= _SAME_PATH * scale))
        valid.append(paths.valid[start] & ~repeated)
    valid = jnp.stack(valid)
    return Paths(
        departure=paths.departure,
        arrival=paths.arrival,
        transmittance=jnp.where(valid, paths.transmittance, 0.0),
        valid=valid,
        points=paths.points,
    )


def _starts(media: Media, chain: Chain, source, receiver) -> jnp.ndarray:
    """``(n_starts, 3)`` points inside the deepest region passed, where a route's search begins.

    The search starts from the polyline ``source -> aim -> receiver``. The aims stand about the
    point ``b`` where the straight line is deepest in the region passed -- the middle of its chord
    through it, or the line's nearest point to it where it misses -- moved across the line either way
    along two directions perpendicular to it: through a sleeve, past both sides of its arc. They
    reach :data:`_START_REACH` of the way to the region's edge, or, for a route that stops short of
    a region nested inside, halfway between that region's edge and the inner one's, which is where
    such a path runs.
    """
    nodes = media.nodes
    outer = nodes[chain.passing[0]].body
    deepest = nodes[chain.passing[-1]]
    straight = receiver - source
    enter, exit_ = outer.intervals(source, straight)
    meets = jnp.isfinite(enter[0]) & jnp.isfinite(exit_[0]) & (exit_[0] > enter[0])
    fraction = jnp.linspace(0.0, 1.0, 33)
    along = source + fraction[:, None] * straight
    nearest = along[jnp.argmin(outer.signed_distance(along))]
    middle = 0.5 * (jnp.where(meets, enter[0], 0.0) + jnp.where(meets, exit_[0], 0.0))
    b = jnp.where(meets, source + middle * straight, nearest)
    unit = straight / jnp.where(norm(straight) > 0.0, norm(straight), 1.0)
    axis = jnp.eye(3)[jnp.argmin(jnp.abs(unit))]
    across = jnp.cross(unit, axis)
    across = across / norm(across)
    other = jnp.cross(unit, across)

    def reach(body, direction):
        """How far from ``b`` the body's boundary is along ``direction``, the nearer way."""
        low, high = body.intervals(b, direction)
        inside = (low[0] < 0.0) & (high[0] > 0.0)
        return jnp.where(inside, jnp.minimum(-low[0], high[0]), 0.0)

    def offset(direction):
        edge = reach(deepest.body, direction)
        if deepest.inside:
            inner = jnp.max(jnp.stack([reach(child.body, direction) for child in deepest.inside]))
            return 0.5 * (edge + inner)
        return _START_REACH * edge

    distance = jnp.minimum(offset(across), offset(other))
    return jnp.stack(
        [b + distance * across, b - distance * across, b + distance * other, b - distance * other]
    )


def _one_path(media: Media, chain: Chain, source, receiver, aim) -> Paths:
    """One pair's path, its crossings found by minimizing the optical length (Fermat's principle).

    ``aim`` is ``None`` for a route through no region, whose search starts from the straight line;
    otherwise a point inside the deepest region passed, and the search starts from the polyline
    through it.
    """
    nodes = media.nodes
    bodies = [nodes[node].body for node, _ in chain.crossings]
    indices = jnp.stack([media.index_of(leg) for leg in chain.legs])
    m = chain.n_crossings
    separation = norm(receiver - source)
    length_scale = jnp.where(separation > 0.0, separation, 1.0)

    # The face each crossing is held to, read off the straight line -- or, on a route through a
    # region, off the polyline through the aim: it is the face the line leaves or enters the body
    # by, which is where the path starts its search.
    straight = receiver - source
    split = chain.pass_at + len(chain.passing)
    guess, faces = [], []
    for k, (body, (_, out)) in enumerate(zip(bodies, chain.crossings, strict=True)):
        if aim is None:
            origin, step = source, straight
        elif k < split:
            origin, step = source, aim - source
        else:
            origin, step = aim, receiver - aim
        enter, exit_ = body.intervals(origin, step)
        t = exit_[0] if out else enter[0]
        point = origin + t * step
        if aim is not None:
            # A start whose line misses the body begins at the aim, inside it, and is moved onto the
            # nearest face from there.
            point = jnp.where(jnp.isfinite(t), point, aim)
        guess.append(point)
        faces.append(jnp.argmax(body.face_distances(point)))
    faces = jnp.stack(faces)

    # Each body's faces, built once: a body builds them anew each time it is asked, and a solve
    # asks dozens of times.
    held = [body.constraints for body in bodies]

    def face(k, position):
        return jnp.take(jnp.stack([bound.signed_distance(position) for bound in held[k]]), faces[k])

    def surfaces(crossing):
        """Each crossing's distance off its face, ``(m,)``, and the face's (unnormalized) normal."""
        pairs = [jax.value_and_grad(face, argnums=1)(k, crossing[k]) for k in range(m)]
        return jnp.stack([value for value, _ in pairs]), jnp.stack([normal for _, normal in pairs])

    def legs(crossing):
        """Each leg's length and unit direction, source to receiver."""
        full = jnp.concatenate([source[None], crossing, receiver[None]])
        steps = full[1:] - full[:-1]
        lengths = norm(steps)
        return lengths, steps / jnp.where(lengths > 0.0, lengths, 1.0)[:, None]

    def optical_length(crossing):
        lengths, _ = legs(crossing)
        return jnp.sum(indices * lengths)

    def length_gradient(crossing):
        """``n_k u_k - n_{k+1} u_{k+1}`` at crossing ``k``: the pull of the legs either side of it."""
        _, units = legs(crossing)
        return indices[:-1, None] * units[:-1] - indices[1:, None] * units[1:]

    def length_curvature(crossing):
        """The optical length's second derivative, ``(3m, 3m)``, written out.

        A leg of index ``n``, length ``l`` and direction ``u`` contributes ``n (I - u u^T) / l`` --
        stiff across itself, free along itself -- to each end it moves and minus that between its
        two ends; the source and receiver are fixed. Written out rather than taken by automatic
        differentiation twice over, which made tracing the solve take seconds.
        """
        lengths, units = legs(crossing)
        across = (
            indices[:, None, None]
            * (jnp.eye(3) - units[:, :, None] * units[:, None, :])
            / jnp.where(lengths > 0.0, lengths, 1.0)[:, None, None]
        )
        curvature = jnp.zeros((m, 3, m, 3))
        for k in range(m):
            curvature = curvature.at[k, :, k, :].set(across[k] + across[k + 1])
            if k + 1 < m:
                curvature = curvature.at[k, :, k + 1, :].set(-across[k + 1])
                curvature = curvature.at[k + 1, :, k, :].set(-across[k + 1])
        return curvature.reshape(3 * m, 3 * m)

    def residual(z):
        crossing = z[: 3 * m].reshape(m, 3)
        multiplier = z[3 * m :]
        gradient = length_gradient(crossing)
        surface, normals = surfaces(crossing)
        return jnp.concatenate([(gradient + multiplier[:, None] * normals).ravel(), surface])

    def multipliers(normals, gradient):
        """The multipliers that best balance the gradient against the normals, per crossing."""
        return -dot(gradient, normals) / norm_squared(normals)

    def onto(crossing):
        """Each point moved onto its own face, along the face's normal.

        Exact in one move for a plane, a ball and a tube, whose signed distance is the distance
        with a unit gradient; the second move tightens a taper, whose distance is exact only on
        the surface.
        """
        for _ in range(2):
            surface, normals = surfaces(crossing)
            crossing = crossing - (surface / norm_squared(normals))[:, None] * normals
        return crossing

    def tangential(normals, vectors):
        """``vectors``, one per crossing, less their part along that crossing's normal."""
        return vectors - (dot(vectors, normals) / norm_squared(normals))[:, None] * normals

    def with_multipliers(crossing):
        """The crossings and their multipliers, as the unknowns of :func:`residual`."""
        _, normals = surfaces(crossing)
        return jnp.concatenate([crossing.ravel(), multipliers(normals, length_gradient(crossing))])

    start = jnp.stack(guess)
    z0 = with_multipliers(start)

    def newton_step(crossing):
        """The step a Newton iteration on the stationarity conditions takes from ``crossing``.

        The curvature is the Lagrangian's -- the optical length's, plus each face's own times its
        multiplier -- which is what makes it quadratic on a curved surface. Where that step does
        not go downhill in optical length (far from the path, on a surface whose curvature has the
        wrong sign), the tangential steepest descent is taken instead, sized to a tenth of the
        separation of the path's ends; the line search does the rest.
        """
        _, normals = surfaces(crossing)
        gradient = length_gradient(crossing)
        weights = multipliers(normals, gradient)
        curvature = length_curvature(crossing)
        faces_curvature = jax.scipy.linalg.block_diag(
            *[weights[k] * jax.hessian(face, argnums=1)(k, crossing[k]) for k in range(m)]
        )
        constraint = jax.scipy.linalg.block_diag(*[normals[k][:, None] for k in range(m)])
        system = jnp.block(
            [[curvature + faces_curvature, constraint], [constraint.T, jnp.zeros((m, m))]]
        )
        update = jnp.linalg.solve(system, jnp.concatenate([-gradient.ravel(), jnp.zeros(m)]))
        update = update[: 3 * m].reshape(m, 3)
        descent = -tangential(normals, gradient)
        descent = descent * (0.1 * length_scale / jnp.maximum(jnp.max(norm(descent)), 1e-300))
        slope = jnp.sum(update * gradient)
        # Close to the path the slope is a rounding of zero and its sign means nothing.
        close = jnp.max(norm(update)) <= _BASIN * length_scale
        usable = jnp.all(jnp.isfinite(update)) & ((slope <= 0.0) | close)
        return jnp.where(usable, update, descent), usable

    def arrived(crossing):
        """Whether a Newton step from here is a rounding of the separation: the path is found.

        Measured by the step rather than by the gradient, because the gradient's size at a given
        distance from the path scales with the curvature, which a short leg makes large; the step
        is a length, compared with the path's own.
        """
        update, usable = newton_step(crossing)
        return usable & (jnp.max(norm(update)) <= _PATH_TOLERANCE * length_scale)

    def descend(f, initial):
        """Minimize the optical length over the crossing points, each held to its face.

        A transmitted path is the shortest optical path among those crossing each surface once,
        so its length is a merit function: each step is a Newton step (or steepest descent),
        moved back onto the faces, and halved until the length falls. That is what reaches a path
        whose crossing lies far from where the straight line meets the surface -- a source seen
        at grazing incidence through an interface close to the receiver -- where Newton's method
        alone, started from the straight line, runs away.
        """
        del f
        initial_crossing = initial[: 3 * m].reshape(m, 3)

        def step(state):
            crossing, count, _ = state
            update, usable = newton_step(crossing)
            done = usable & (jnp.max(norm(update)) <= _PATH_TOLERANCE * length_scale)
            length = optical_length(crossing)

            # The line search tries a fixed ladder of halved steps at once and keeps the longest
            # that shortens the path. Not a loop: a ``while_loop`` inside this one, under the
            # ``vmap`` over pairs, stops updating a pair whose outer loop has finished while its
            # own condition still holds, and the batch never ends.
            ladder = 2.0 ** -jnp.arange(_HALVINGS + 1.0)
            tried = jax.vmap(lambda a: optical_length(onto(crossing + a * update)))(ladder)
            shortens = tried < length
            scale = jnp.where(jnp.any(shortens), ladder[jnp.argmax(shortens)], 0.0)
            # Close to the path the length changes by less than its own rounding, so a Newton step
            # there is taken whole: the basin is quadratic, and the next step is smaller still.
            close = usable & (jnp.max(norm(update)) <= _BASIN * length_scale)
            scale = jnp.where(close, 1.0, scale)
            moved = onto(crossing + scale * update)
            improved = close | (optical_length(moved) < length)
            following = jnp.where(done | ~improved, crossing, moved)
            return following, count + 1, done | ~improved

        def going(state):
            _, count, stopped = state
            return (count < _NEWTON_STEPS) & ~stopped

        crossing, _, _ = jax.lax.while_loop(going, step, (initial_crossing, 0, False))
        done = arrived(crossing) & jnp.all(jnp.isfinite(crossing))
        z = with_multipliers(crossing)
        # A path that was not found is returned as the straight line it started from: finite, so
        # the implicit derivative through it is finite too, and carrying nothing. The flag rides
        # out as a number: an auxiliary output of a differentiated solve must have a tangent, and
        # a boolean has none.
        return jnp.where(done, z, initial), done.astype(float)

    def tangent(g, y):
        return jnp.linalg.solve(jax.jacfwd(g)(y), y)

    z, found = jax.lax.custom_root(residual, z0, descend, tangent, has_aux=True)
    found = jax.lax.stop_gradient(found) > 0.5
    crossing = z[: 3 * m].reshape(m, 3)
    full = jnp.concatenate([source[None], crossing, receiver[None]])
    _, directions = legs(crossing)

    valid = found
    transmittance = jnp.asarray(1.0)
    for k, (body, (_, out)) in enumerate(zip(bodies, chain.crossings, strict=True)):
        normal = jax.grad(face, argnums=1)(k, crossing[k])
        normal = normal / norm(normal)
        before, after = dot(directions[k], normal), dot(directions[k + 1], normal)
        # Leaving, both legs head out through the outward normal; entering, both head in. A
        # stationary path that turns back at the surface is a reflection, not this path. The
        # descent finds the shortest path, which never turns back, so this is a guard on what the
        # solve returns rather than a case it reaches.
        valid = valid & ((before > 0.0) & (after > 0.0) if out else (before < 0.0) & (after < 0.0))
        valid = valid & jnp.all(body.face_distances(crossing[k]) <= _ON_BOUNDARY * length_scale)
        transmittance = transmittance * fresnel_transmittance(before, indices[k], indices[k + 1])

    depth = jnp.asarray(0.0)
    for leg, (start_point, end_point) in enumerate(itertools.pairwise(full)):
        medium = media.absorption_of(chain.legs[leg])
        if medium is not None:
            depth = depth + medium.optical_depth(start_point, end_point)
        valid = valid & ~_meets_any(media, chain.beside[leg], start_point, end_point)
    transmittance = transmittance * jnp.exp(-depth)

    return Paths(
        departure=directions[0],
        arrival=-directions[-1],
        transmittance=jnp.where(valid, transmittance, 0.0),
        valid=valid,
        points=crossing,
    )


def _meets_any(media: Media, regions, origin, target) -> jnp.ndarray:
    """Whether the segment from ``origin`` to ``target`` passes through any of ``regions``.

    Through, not touching: the part of the segment inside must be longer than a rounding of the
    segment's own length, so a leg that only grazes a region's surface is not counted as meeting it.
    """
    step = target - origin
    meets = jnp.asarray(False)
    for node in regions:
        enter, exit_ = media.nodes[node].body.intervals(origin, step)
        inside = jnp.minimum(exit_[0], 1.0) - jnp.maximum(enter[0], 0.0)
        meets = meets | (inside > _ON_BOUNDARY)
    return meets


def straight_reach(media: Media, surfaces, points, *, pair_limit: int = DEFAULT_PAIR_LIMIT):
    """Whether the straight segment from each facet's centroid to each point carries its light.

    It does where the facet and the point lie in one medium and the segment meets no transparent
    region: there the light goes straight. Where they lie in different media, or the segment passes
    through a region, their light goes by refracted paths instead
    (:mod:`~aquaflux.radiation.refracted`), and the straight segment carries none. Taken along the
    centroid's segment, as every shadow is, and frozen.

    Parameters
    ----------
    media : Media
    surfaces : Surfaces
        The sources; read for their centroids, and their corners to place each facet in one medium.
    points : array_like, shape ``(n_points, 3)``
        The receivers, concrete.
    pair_limit : int, optional
        Pairs evaluated at once, at most; a pass is also held to
        :data:`~aquaflux.radiation.work.PASS_PAIRS`.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n_points, n_facets)``

    Raises
    ------
    ValueError
        If a point or a facet corner lies on a region's surface, or a facet straddles one.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    facet_region = media.region_of_facets(surfaces)
    point_region = media.region_of(points, "receiver")
    centroid = np.asarray(surfaces.centroid, dtype=float)
    reach = np.zeros((points.shape[0], surfaces.n_facets), dtype=bool)
    parents = media.parents
    for medium in np.unique(point_region):
        rows = np.flatnonzero(point_region == medium)
        columns = np.flatnonzero(facet_region == medium)
        if not len(columns):
            continue
        beside = Chain.between(parents, int(medium), int(medium)).beside[0]
        met = _pairs_meet(media, beside, centroid[columns], points[rows], pair_limit=pair_limit)
        reach[np.ix_(rows, columns)] = ~met
    return reach


def _pairs_meet(media: Media, regions, origins, targets, *, pair_limit: int) -> np.ndarray:
    """``(n_targets, n_origins)``: whether each segment from an origin to a target meets a region.

    Host work on concrete positions, a pass of pairs at a time.
    """
    origins, targets = np.asarray(origins, dtype=float), np.asarray(targets, dtype=float)
    met = np.zeros((len(targets), len(origins)), dtype=bool)
    count = met.size
    if not regions or not count:
        return met
    regions = tuple(int(node) for node in regions)
    chunk = padded_length(min(count, max(1, pair_limit), PASS_PAIRS))
    for start in range(0, count, chunk):
        # Padded to the chunk by repeating the last pair, so every pass shares one program.
        flat = np.minimum(np.arange(start, start + chunk), count - 1)
        row, column = flat // len(origins), flat % len(origins)
        got = np.asarray(
            _segments_meet(media, regions, jnp.asarray(origins[column]), jnp.asarray(targets[row]))
        )
        kept = slice(0, min(chunk, count - start))
        met[row[kept], column[kept]] = got[kept]
    return met


@eqx.filter_jit
def _segments_meet(media: Media, regions: tuple, origin, target) -> jnp.ndarray:
    """:func:`_meets_any` for many segments, compiled per shape."""
    return jax.vmap(lambda start, end: _meets_any(media, regions, start, end))(origin, target)
