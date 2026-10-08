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

**Regions the path does not cross by construction** -- a transparent solid that holds neither the
source nor the receiver, such as a neighbouring lamp's sleeve -- are crossed **straight**: the leg
is attenuated by their absorption along its chord and by the Fresnel loss at the angles the straight
chord meets their surfaces, and is not bent. That is an approximation, and its size depends on the
scene; it is exact only when such a region's index matches its surroundings.

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
    "straight_through",
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
        For each leg, the regions it may pass straight through: every region that is not crossed,
        does not hold the leg's medium and is not that medium.
    """

    crossings: tuple
    legs: tuple
    beside: tuple

    @classmethod
    def between(cls, parents: np.ndarray, source: int, receiver: int) -> Chain:
        """The chain from a source in region ``source`` to a receiver in region ``receiver``."""

        def lineage(node):
            line = []
            while node >= 0:
                line.append(int(node))
                node = parents[node]
            return line

        up, down = lineage(source), lineage(receiver)
        common = next((node for node in up if node in down), -1)
        leaving = up[: up.index(common)] if common >= 0 else up
        entering = down[: down.index(common)] if common >= 0 else down
        crossings = tuple((node, True) for node in leaving) + tuple(
            (node, False) for node in reversed(entering)
        )
        legs = (source, *(parents[node] if out else node for node, out in crossings))
        legs = tuple(int(leg) for leg in legs)
        # A crossed region is never beside a leg: one leaving it starts on its surface and one
        # entering it ends there, and neither goes through it, convex as it is.
        crossed = {node for node, _ in crossings}
        beside = tuple(
            tuple(
                node
                for node in range(len(parents))
                if node not in lineage(leg) and node not in crossed
            )
            for leg in legs
        )
        return cls(crossings=crossings, legs=legs, beside=beside)

    @property
    def n_crossings(self) -> int:
        """How many surfaces the path crosses."""
        return len(self.crossings)


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
        What gets across: the Fresnel transmittances of the crossings, the absorption of each leg
        in its own medium, and whatever the legs pass straight through.
    valid : jnp.ndarray of bool, shape ``(...)``
        Whether the path exists: the solve converged, every crossing is on its body's boundary and
        goes the way the chain says. A path that does not exist carries nothing.
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
        The crossings, from :meth:`Chain.between`; every pair given must have it.
    sources, receivers : array_like, shape ``(..., 3)``
        Broadcast against each other.

    Returns
    -------
    Paths
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
    paths = jax.vmap(lambda s, r: _one_path(media, chain, s, r))(flat_sources, flat_receivers)
    return jax.tree.map(lambda leaf: leaf.reshape(*shape[:-1], *leaf.shape[1:]), paths)


def _one_path(media: Media, chain: Chain, source, receiver) -> Paths:
    """One pair's path, its crossings found by minimizing the optical length (Fermat's principle)."""
    nodes = media.nodes
    bodies = [nodes[node].body for node, _ in chain.crossings]
    indices = jnp.stack([media.index_of(leg) for leg in chain.legs])
    m = chain.n_crossings
    separation = norm(receiver - source)
    length_scale = jnp.where(separation > 0.0, separation, 1.0)

    # The face each crossing is held to, read off the straight line: it is the face the line
    # leaves or enters the body by, which is where the path starts its search.
    straight = receiver - source
    guess, faces = [], []
    for body, (_, out) in zip(bodies, chain.crossings, strict=True):
        enter, exit_ = body.intervals(source, straight)
        t = exit_[0] if out else enter[0]
        point = source + t * straight
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
        through, extra = _straight_through(media, chain.beside[leg], start_point, end_point)
        transmittance = transmittance * through
        depth = depth + extra
    transmittance = transmittance * jnp.exp(-depth)

    return Paths(
        departure=directions[0],
        arrival=-directions[-1],
        transmittance=jnp.where(valid, transmittance, 0.0),
        valid=valid,
        points=crossing,
    )


def straight_through(media: Media, surfaces, points, *, pair_limit: int = DEFAULT_PAIR_LIMIT):
    """What the straight segment from each facet's centroid to each point gets past.

    The factor a mask of straight segments carries for the transparent regions: where a facet and a
    point lie in the same medium, the product over every region the segment passes through of the
    Fresnel transmittances where the segment enters and leaves it, at the angles it meets the
    region's surface there, and of the region's absorption along its chord in excess of the
    medium's own -- which the gather already counts along the whole segment. Where the two lie in
    different media it is **zero**: their light crosses a surface that bends it, and goes by a
    refracted path (:mod:`~aquaflux.radiation.refracted`) instead.

    Taken along the centroid's segment, as every shadow is, and frozen: worked out on the host from
    the indices and absorptions given, with no derivative.

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
    numpy.ndarray, shape ``(n_points, n_facets)``

    Raises
    ------
    ValueError
        If a point or a facet corner lies on a region's surface, or a facet straddles one.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    facet_region = media.region_of_facets(surfaces)
    point_region = media.region_of(points, "receiver")
    centroid = np.asarray(surfaces.centroid, dtype=float)
    through = np.zeros((points.shape[0], surfaces.n_facets))
    parents = media.parents
    for medium in np.unique(point_region):
        rows = np.flatnonzero(point_region == medium)
        columns = np.flatnonzero(facet_region == medium)
        if not len(columns):
            continue
        beside = Chain.between(parents, int(medium), int(medium)).beside[0]
        if not beside:
            through[np.ix_(rows, columns)] = 1.0
            continue
        count = len(rows) * len(columns)
        chunk = padded_length(min(count, max(1, pair_limit), PASS_PAIRS))
        for start in range(0, count, chunk):
            # Padded to the chunk by repeating the last pair, so every pass shares one program.
            flat = np.minimum(np.arange(start, start + chunk), count - 1)
            row, column = rows[flat // len(columns)], columns[flat % len(columns)]
            got = np.asarray(
                _through_pairs(
                    media, beside, jnp.asarray(centroid[column]), jnp.asarray(points[row])
                )
            )
            kept = slice(0, min(chunk, count - start))
            through[row[kept], column[kept]] = got[kept]
    return through


@eqx.filter_jit
def _through_pairs(media: Media, beside: tuple, origin, target) -> jnp.ndarray:
    """:func:`_straight_through` for many segments, as one factor each, compiled per shape."""

    def one(start, end):
        through, depth = _straight_through(media, beside, start, end)
        return through * jnp.exp(-depth)

    return jax.vmap(one)(origin, target)


def _straight_through(media: Media, beside, origin, target):
    """What the regions a leg passes straight through let past: Fresnel losses, and extra depth.

    Each region the segment crosses is entered and left along the straight line, at the angles the
    line meets its surface there, and absorbs along its chord **less what the medium holding it
    would have absorbed there**: the leg's own medium is already counted along the whole leg, so a
    nested region's excess over its holder is what remains. Telescoping over a region and the ones
    inside it, that is each piece of the chord absorbed by its own medium.
    """
    parents = media.parents
    nodes = media.nodes
    through = jnp.asarray(1.0)
    depth = jnp.asarray(0.0)
    step = target - origin
    span = norm(step)
    for node in beside:
        region = nodes[node]
        holder = int(parents[node])
        enter, exit_ = region.body.intervals(origin, step)
        enter, exit_ = jnp.maximum(enter[0], 0.0), jnp.minimum(exit_[0], 1.0)
        crossed = exit_ > enter
        # Points to evaluate the surfaces at, moved somewhere finite where the line misses, so
        # nothing infinite reaches a gradient the selection below would not stop.
        enter_at = origin + jnp.where(crossed, enter, 0.5) * step
        exit_at = origin + jnp.where(crossed, exit_, 0.5) * step
        inner, outer = region.refractive_index, media.index_of(holder)
        unit = step / jnp.where(span > 0.0, span, 1.0)
        through = through * jnp.where(
            crossed,
            fresnel_transmittance(dot(unit, _outward(region.body, enter_at)), outer, inner)
            * fresnel_transmittance(dot(unit, _outward(region.body, exit_at)), inner, outer),
            1.0,
        )
        chord = jnp.where(crossed, (exit_ - enter) * span, 0.0)
        middle = 0.5 * (enter_at + exit_at)
        own = _coefficient(region.absorption, middle)
        surrounding = _coefficient(media.absorption_of(holder), middle)
        depth = depth + (own - surrounding) * chord
    return through, depth


def _outward(body: ConvexSolid, position) -> jnp.ndarray:
    """The outward unit normal of the face of ``body`` nearest to holding ``position``."""
    gradient = jax.grad(body.signed_distance)(position)
    return gradient / norm(gradient)


def _coefficient(absorption: Absorption | None, position) -> jnp.ndarray:
    """An absorbing medium's coefficient at a position, zero for none."""
    return jnp.asarray(0.0) if absorption is None else absorption.sample(position)
