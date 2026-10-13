"""Which receivers a route through a stack of coaxial cylinders can reach: a sound cull.

A route through a neighbouring lamp's sleeve (:meth:`~aquaflux.radiation.refraction.Chain.between`
with ``through=``) carries light only to receivers a ray bent by that sleeve can reach, and for most
receivers there is none. Solving every route for every receiver costs as much as solving the ones
that carry light, so the receivers no ray can reach are left out before any path is solved. This
module decides which, from closed forms, and never leaves out a receiver a ray can reach.

**The conserved quantities.** Every surface the route crosses inside the sleeve is the side of a
cylinder on one axis, and the medium round it is uniform. Snell's law at such a surface keeps the
component of ``n u`` along the axis, ``beta = n u_z``, so in the plane across the axis the ray
behaves as a ray in media of index ``m = sqrt(n^2 - beta^2)``, and keeps ``m p``, its in-plane
index times its impact parameter ``p`` about the axis (its angular momentum about the axis). In
that plane a ray entering the stack with impact parameter ``p`` in the medium round it leaves with
the same ``p``, its direction turned by

    D = Phi - 2 acos(p / R_0),

where ``Phi`` is the angle the ray sweeps round the axis inside: through the levels ``0..k`` of the
stack, of radii ``R_0 > R_1 > ...``, with impact ``p_j = m_out p / m_j`` at level ``j``,

    Phi = 2 sum_{j<k} [acos(p_j / R_j) - acos(p_j / R_{j+1})] + 2 acos(p_k / R_k).

With equal indices ``Phi = 2 acos(p / R_0)`` and the ray goes straight. The turn is towards the axis
for a denser sleeve, and away from it for a less dense one.

**The test, per receiver.** In the plane across the axis, with the axis at the origin, a line is a
direction angle ``alpha`` and a signed impact parameter ``b`` (``x . (-sin alpha, cos alpha) = b``);
the ray leaving the sleeve towards a receiver at polar position ``(rho, theta)`` has
``alpha_out = theta - asin(b / rho)``, and the ray that entered it ``alpha_in = alpha_out + sign(b)
D``, the same ``b``. It came from the source side: where the light leaves the region holding the
source, or the source itself, which lies in a disc of centre ``c`` and radius ``a``. So the
incoming line must meet that disc, ``|c . n - b| <= a``, before it meets the sleeve,
``c . d - a < -sqrt(R_0^2 - b^2)``. The unknowns ``(b, beta)`` are covered by cells, each with
bounds on ``D`` from the closed form (every ratio ``m_out / m_j`` is monotone in ``beta`` and every
term monotone in its argument, so a cell's corners bound it), and a receiver is kept if the test
can hold anywhere in any cell. Cells that can hold are split, a few times, to tighten the bounds; a
receiver that keeps too many cells is kept without splitting further. Keeping is always sound, so
every approximation here keeps more, never less.

**Where it says nothing.** Along the axis a ray's ``u_z`` keeps its sign (``beta`` is conserved and
every index is positive), so a path's height along the axis moves one way from where it leaves the
source side to the receiver; if both lie within the length of every cylinder of the stack, the path
crosses their sides and not their ends. Where that fails, where the receiver lies in another region,
where the source side is not a cylinder parallel to the stack, or where any geometry or index is
traced, every receiver is kept.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np

from aquaflux.solids import Cylinder

#: Cells across the in-plane impact parameter, each sign, before any are split.
_IMPACT_CELLS = 32

#: Cells across the axial index ``beta`` before any are split.
_AXIAL_CELLS = 16

#: How many times a cell that can carry light to a receiver is split in four.
_SPLITS = 5

#: Cells a receiver may keep before it is kept outright, without splitting further.
_CELLS_KEPT = 512

#: Receivers tested together, which bounds the arrays formed to this many times the cells.
_RECEIVERS_AT_ONCE = 64

#: How far, relative to the axis, a cylinder may be off parallel and still count as parallel.
_PARALLEL = 1e-12


@dataclasses.dataclass(frozen=True, eq=False)
class CoaxialStack:
    """The in-plane geometry of a route through a stack of coaxial cylinders.

    Attributes
    ----------
    centre : numpy.ndarray, shape ``(3,)``
        A point on the axis.
    axis : numpy.ndarray, shape ``(3,)``
        The unit axis.
    frame : numpy.ndarray, shape ``(2, 3)``
        Two unit vectors across the axis, at right angles.
    radii : numpy.ndarray, shape ``(k + 1,)``
        The cylinders the route enters, outermost first.
    missed : float or None
        The radius of the cylinder inside the deepest one entered, which the route must miss; unset
        when it holds none.
    indices : numpy.ndarray, shape ``(k + 2,)``
        The index of the medium round the stack, then of each level entered.
    span : tuple of float
        The heights along the axis, from :attr:`centre`, within which every cylinder of the stack
        has its side.
    """

    centre: np.ndarray
    axis: np.ndarray
    frame: np.ndarray
    radii: np.ndarray
    missed: float | None
    indices: np.ndarray
    span: tuple

    @classmethod
    def of(cls, media, chain) -> CoaxialStack | None:
        """The stack a route passes through, or ``None`` where the route is not through one.

        A route through one region, or through a region and the one region it holds, each a
        :class:`~aquaflux.solids.Cylinder` on one axis, with concrete geometry and indices.
        """
        passing = chain.passing
        if not passing or len(passing) > 2:
            return None
        shell = passing[0]
        children = [node for node, parent in enumerate(media.parents) if parent == shell]
        if len(children) != 1 or (len(passing) == 2 and passing[1] != children[0]):
            return None
        bodies = [media.nodes[shell].body, media.nodes[children[0]].body]
        indices = [media.index_of(int(media.parents[shell])), *(media.index_of(n) for n in passing)]
        if not all(isinstance(body, Cylinder) for body in bodies) or _traced((bodies, indices)):
            return None
        centre = np.asarray(bodies[0].centre, dtype=float)
        axis = np.asarray(bodies[0].axis, dtype=float)
        if not all(_on_axis(centre, axis, body) for body in bodies[1:]):
            return None
        ends = [_heights(centre, axis, body) for body in bodies]
        span = (max(low for low, _ in ends), min(high for _, high in ends))
        radii = [float(body.radius) for body in bodies]
        return cls(
            centre=centre,
            axis=axis,
            frame=_across(axis),
            radii=np.asarray(radii[: len(passing)]),
            missed=radii[1] if len(passing) == 1 else None,
            indices=np.asarray([float(index) for index in indices]),
            span=span,
        )

    def plane(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Points across the axis, ``(n, 2)``, and their heights along it, ``(n,)``."""
        offset = np.asarray(points, dtype=float).reshape(-1, 3) - self.centre
        return offset @ self.frame.T, offset @ self.axis

    def disc(self, body) -> tuple[np.ndarray, float] | None:
        """A parallel cylinder's cross-section, centre ``(2,)`` and radius; ``None`` otherwise."""
        if not isinstance(body, Cylinder) or _traced(body):
            return None
        if abs(abs(float(np.dot(self.axis, np.asarray(body.axis, dtype=float)))) - 1.0) > _PARALLEL:
            return None
        centre, _ = self.plane(np.asarray(body.centre, dtype=float))
        return centre[0], float(body.radius)

    def heights(self, body) -> tuple[float, float]:
        """The heights along the axis a parallel cylinder spans."""
        return _heights(self.centre, self.axis, body)

    def turns(self, cells: np.ndarray):
        """Whether a cell can hold a transmitted ray, and bounds on its turn ``D`` there.

        Parameters
        ----------
        cells : numpy.ndarray, shape ``(n, 4)``
            Each cell's impact parameters ``p`` (unsigned) and axial indices ``beta``, as
            ``(p_low, p_high, beta_low, beta_high)``, all non-negative.

        Returns
        -------
        possible : numpy.ndarray of bool, shape ``(n,)``
        low, high : numpy.ndarray, shape ``(n,)``
            Bounds on ``D`` over every ray the cell holds; meaningless where not ``possible``.
        """
        p_low, p_high, beta_low, beta_high = cells.T
        levels = len(self.radii)

        def ratio_bounds(level):
            # m_out / m_level at each end of the cell's beta range: monotone in beta.
            if level < 0:
                one = np.ones_like(beta_low)
                return one, one
            ends = [self._ratio(level, beta) for beta in (beta_low, beta_high)]
            return np.minimum(*ends), np.maximum(*ends)

        def argument(level, radius):
            low, high = ratio_bounds(level)
            return p_low * low / radius, p_high * high / radius

        possible = p_low < self.radii[0]
        for level in range(levels):
            radius = self.radii[level]
            for side in (level - 1, level):
                low, _ = argument(side, radius)
                possible &= low <= 1.0
        if self.missed is not None:
            _, high = argument(levels - 1, self.missed)
            possible &= high >= 1.0

        terms = [(-2.0, -1, self.radii[0])]
        for level in range(levels - 1):
            terms += [(2.0, level, self.radii[level]), (-2.0, level, self.radii[level + 1])]
        terms.append((2.0, levels - 1, self.radii[levels - 1]))
        low = np.zeros_like(p_low)
        high = np.zeros_like(p_low)
        for weight, level, radius in terms:
            small, large = argument(level, radius)
            # acos falls as its argument rises; every argument of a possible ray is in [0, 1].
            least = np.arccos(np.clip(np.nan_to_num(large, nan=np.inf), 0.0, 1.0))
            most = np.arccos(np.clip(small, 0.0, 1.0))
            if weight > 0:
                low, high = low + weight * least, high + weight * most
            else:
                low, high = low + weight * most, high + weight * least
        return possible, low, high

    def _ratio(self, level, beta):
        out, inside = self.indices[0], self.indices[level + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.sqrt(
                np.maximum(out * out - beta * beta, 0.0) / (inside * inside - beta * beta)
            )
        return np.where(inside * inside - beta * beta > 0.0, ratio, np.inf)


def reachable(stack: CoaxialStack, centre, radius, span, points) -> np.ndarray:
    """Which receivers a ray through the stack from a disc on the source side can reach.

    Parameters
    ----------
    stack : CoaxialStack
    centre : numpy.ndarray, shape ``(2,)``
        The disc's centre across the axis (:meth:`CoaxialStack.plane`).
    radius : float
        The disc's radius.
    span : tuple of float
        The heights along the axis the source side lies within.
    points : numpy.ndarray, shape ``(n, 3)``
        Receivers in the medium round the stack.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n,)``
        ``False`` only where no transmitted ray can reach the receiver.
    """
    across, height = stack.plane(points)
    distance = np.hypot(across[:, 0], across[:, 1])
    low, high = stack.span
    keep = np.ones(len(distance), dtype=bool)
    if not (low <= span[0] and span[1] <= high):
        return keep
    tested = np.flatnonzero((height >= low) & (height <= high) & (distance > stack.radii[0]))
    angle = np.arctan2(across[:, 1], across[:, 0])
    cells = _first_cells(stack)
    for first in range(0, len(tested), _RECEIVERS_AT_ONCE):
        rows = tested[first : first + _RECEIVERS_AT_ONCE]
        keep[rows] = _any_cell_carries(
            stack, cells, np.asarray(centre, float), float(radius), distance[rows], angle[rows]
        )
    return keep


def _any_cell_carries(stack, cells, centre, radius, distance, angle) -> np.ndarray:
    """Per receiver, whether some cell can carry a ray from the disc to it."""
    n = len(distance)
    kept = np.zeros(n, dtype=bool)
    who = np.repeat(np.arange(n), len(cells))
    live = np.tile(cells, (n, 1))
    for split in range(_SPLITS + 1):
        able = _carries(stack, live, centre, radius, distance[who], angle[who])
        who, live = who[able], live[able]
        if split == _SPLITS:
            break
        crowded = np.bincount(who, minlength=n) > _CELLS_KEPT
        kept |= crowded
        stay = ~crowded[who]
        who, live = np.repeat(who[stay], 4), _quartered(live[stay])
    return kept | (np.bincount(who, minlength=n) > 0)


def _first_cells(stack) -> np.ndarray:
    """Cells ``(sign, p_low, p_high, beta_low, beta_high)`` covering every ray into the stack.

    Impact parameters crowd towards the rim and axial indices towards their largest, where the
    turn changes fastest.
    """
    edges_p = stack.radii[0] * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, _IMPACT_CELLS + 1))
    beta_max = float(np.min(stack.indices))
    edges_beta = beta_max * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, _AXIAL_CELLS + 1))
    p_low, beta_low = np.meshgrid(edges_p[:-1], edges_beta[:-1], indexing="ij")
    p_high, beta_high = np.meshgrid(edges_p[1:], edges_beta[1:], indexing="ij")
    cells = np.stack([p_low, p_high, beta_low, beta_high], axis=-1).reshape(-1, 4)
    return np.concatenate(
        [np.column_stack([np.full(len(cells), sign), cells]) for sign in (1.0, -1.0)]
    )


def _quartered(cells) -> np.ndarray:
    """Each cell split in two along its impact parameter and its axial index, in place order."""
    sign, p_low, p_high, beta_low, beta_high = cells.T
    p_mid, beta_mid = 0.5 * (p_low + p_high), 0.5 * (beta_low + beta_high)
    parts = [
        (p_low, p_mid, beta_low, beta_mid),
        (p_mid, p_high, beta_low, beta_mid),
        (p_low, p_mid, beta_mid, beta_high),
        (p_mid, p_high, beta_mid, beta_high),
    ]
    return np.stack([np.column_stack([sign, *part]) for part in parts], axis=1).reshape(-1, 5)


def _carries(stack, cells, centre, radius, distance, angle) -> np.ndarray:
    """Per (receiver, cell), whether a ray in the cell can run from the disc to the receiver."""
    sign = cells[:, 0]
    possible, turn_low, turn_high = stack.turns(cells[:, 1:])
    p_low, p_high = cells[:, 1], cells[:, 2]
    b_low = np.where(sign > 0, p_low, -p_high)
    b_high = np.where(sign > 0, p_high, -p_low)
    out_low = angle - np.arcsin(b_high / distance)
    out_high = angle - np.arcsin(b_low / distance)
    in_low = np.where(sign > 0, out_low + turn_low, out_low - turn_high)
    in_high = np.where(sign > 0, out_high + turn_high, out_high - turn_low)
    # The line at direction alpha: d = (cos, sin), n = (-sin, cos); with c = |c| (cos phi, sin phi),
    # c . d = |c| cos(phi - alpha) and c . n = |c| sin(phi - alpha).
    size = float(np.hypot(*centre))
    phi = float(np.arctan2(centre[1], centre[0]))
    cos_low, _ = _cosine_bounds(phi - in_high, phi - in_low)
    sin_low, sin_high = _cosine_bounds(phi - in_high - 0.5 * np.pi, phi - in_low - 0.5 * np.pi)
    meets = (size * sin_high >= b_low - radius) & (size * sin_low <= b_high + radius)
    entry = -np.sqrt(np.maximum(stack.radii[0] ** 2 - p_high**2, 0.0))
    before = size * cos_low - radius < entry
    return possible & meets & before


def _cosine_bounds(low, high) -> tuple[np.ndarray, np.ndarray]:
    """The least and greatest cosine over each interval of angles ``[low, high]``."""
    turn = 2.0 * np.pi
    whole = high - low >= turn
    peak = np.ceil(low / turn) <= np.floor(high / turn)
    trough = np.ceil((low - np.pi) / turn) <= np.floor((high - np.pi) / turn)
    ends_low = np.minimum(np.cos(low), np.cos(high))
    ends_high = np.maximum(np.cos(low), np.cos(high))
    least = np.where(whole | trough, -1.0, ends_low)
    greatest = np.where(whole | peak, 1.0, ends_high)
    return least, greatest


def _heights(centre, axis, body) -> tuple[float, float]:
    """The heights along ``axis``, from ``centre``, that a cylinder parallel to it spans."""
    middle = float(np.dot(np.asarray(body.centre, dtype=float) - centre, axis))
    return middle - float(body.half_length), middle + float(body.half_length)


def _on_axis(centre, axis, body) -> bool:
    if abs(abs(float(np.dot(axis, np.asarray(body.axis, dtype=float)))) - 1.0) > _PARALLEL:
        return False
    offset = np.asarray(body.centre, dtype=float) - centre
    across = offset - np.dot(offset, axis) * axis
    return float(np.linalg.norm(across)) <= 1e-9 * float(body.radius)


def _across(axis) -> np.ndarray:
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(axis, first)])


def _traced(tree) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


def kept_receivers(media, chain, sources, points) -> np.ndarray:
    """Which receivers a route can carry light to from sources in one medium.

    Parameters
    ----------
    media : Media
    chain : Chain
        A route from the sources' medium to the receivers'.
    sources : numpy.ndarray, shape ``(..., 3)``, or None
        Every vertex of the source facets; ``None`` where they are traced, and then every receiver
        of a route starting in the stack's own medium is kept.
    points : numpy.ndarray, shape ``(n, 3)``
        The receivers, all in the route's last medium.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n,)``
        ``False`` only where no transmitted ray on the route can reach the receiver: every
        receiver is kept on a route through no region, through anything but a stack of coaxial
        cylinders, ending in another region after the stack, or beginning on the source side on a
        surface that is not a cylinder parallel to the stack.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    keep = np.ones(len(points), dtype=bool)
    stack = CoaxialStack.of(media, chain)
    if stack is None or chain.pass_at + 2 * len(chain.passing) != chain.n_crossings:
        return keep
    if chain.pass_at == 0:
        if sources is None:
            return keep
        across, height = stack.plane(sources)
        centre = 0.5 * (across.min(axis=0) + across.max(axis=0))
        radius = float(np.max(np.linalg.norm(across - centre, axis=1)))
        span = (float(height.min()), float(height.max()))
    else:
        node, leaving = chain.crossings[chain.pass_at - 1]
        disc = stack.disc(media.nodes[node].body) if leaving else None
        if disc is None:
            return keep
        centre, radius = disc
        span = stack.heights(media.nodes[node].body)
    return reachable(stack, centre, radius, span, points)
