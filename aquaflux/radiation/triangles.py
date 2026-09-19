"""Whether a segment is cut by any triangle of a surface.

This is the test that lets a reactor shadow itself: a facet on one leg of a bent duct does not
illuminate a cell in the other leg, because the wall between them is in the way. The analytic
bodies of :mod:`aquaflux.radiation.occluders` cannot express that — the geometry doing the
blocking *is* the emitting surface.

The intersection is Woop, Benthin and Wald's **watertight** ray-triangle test (*Journal of
Computer Graphics Techniques* 2(1), 2013) rather than the more familiar Möller-Trumbore form.
The two agree almost everywhere; they differ on rays that pass through an edge shared by two
triangles, where an ordinary test can report a hit on both or on neither depending on rounding.
Neither is right, and "neither" is the one that matters: it is a pinhole through a closed
surface. Woop's construction transforms the ray to the ``+z`` axis and evaluates the three edge
functions in a way that makes the two triangles sharing an edge give exactly opposite values, so
one of them always claims the ray.

⚠️ **A facet must not shadow itself**, and the exclusion is by index rather than by tolerance:
the ray starts at the facet's own centroid, so the facet is always hit, at zero distance.
Excluding its whole *solid* would be wrong — a non-convex body such as a bent duct is precisely
the case this module exists for, and there the blocking wall belongs to the same body as the
emitter. Neighbouring facets that share an edge with the source are handled by the same
near-origin exclusion the analytic bodies use.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["segment_is_cut"]


def _watertight_hit(origin, direction, vertices):
    """Woop's edge test, for one ray against many triangles.

    ``origin`` and ``direction`` broadcast as ``(..., 3)``; ``vertices`` is ``(..., 3, 3)``.
    Returns the hit distance along ``direction`` in units of its length, and whether the ray's
    line meets the triangle at all — the caller decides which distances count.
    """
    # Put the ray's largest component on the z axis, so the shear below never divides by a
    # small number.
    kz = jnp.argmax(jnp.abs(direction), axis=-1)
    kx = (kz + 1) % 3
    ky = (kz + 2) % 3
    # The published algorithm also swaps kx and ky when the chosen component is negative, to
    # keep the coordinate system right-handed. That is deliberately omitted: it is inert here.
    # Flipping handedness negates u, v, w and the determinant together, and both places they
    # are used are invariant to that — the inside test accepts all-non-negative *or*
    # all-non-positive, and the distance divides by the same determinant. Measured over 20000
    # rays against 300 triangles, adding the swap changes no hit and moves no distance by more
    # than 0.0. The swap is needed when the determinant's sign is used for back-face culling,
    # which this module does with the facet normal instead.

    def component(array, axis):
        return jnp.take_along_axis(array, axis[..., None], axis=-1)[..., 0]

    dx, dy, dz = (component(direction, axis) for axis in (kx, ky, kz))
    shear_x, shear_y, shear_z = dx / dz, dy / dz, 1.0 / dz

    relative = vertices - origin[..., None, :]
    corners = []
    for corner in range(3):
        offset = relative[..., corner, :]
        ox = component(offset, kx)
        oy = component(offset, ky)
        oz = component(offset, kz)
        corners.append((ox - shear_x * oz, oy - shear_y * oz, oz * shear_z))
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = corners

    u = cx * by - cy * bx
    v = ax * cy - ay * cx
    w = bx * ay - by * ax

    determinant = u + v + w
    same_side = ((u >= 0.0) & (v >= 0.0) & (w >= 0.0)) | ((u <= 0.0) & (v <= 0.0) & (w <= 0.0))
    edge_on = determinant == 0.0
    safe = jnp.where(edge_on, 1.0, determinant)
    distance = (u * az + v * bz + w * cz) / safe
    return distance, same_side & ~edge_on


def segment_is_cut(origin, target, vertices, min_distance, *, exclude=None, work_limit=4_000_000):
    """Whether any triangle lies across the segment from ``origin`` to ``target``.

    Parameters
    ----------
    origin, target : jnp.ndarray, shape ``(n_rays, 3)``
        Segment endpoints. By convention ``origin`` is the source.
    vertices : jnp.ndarray, shape ``(n_triangles, 3, 3)``
        The blocking triangles.
    min_distance : jnp.ndarray, shape ``(n_rays,)``
        How far from ``origin`` a hit must be before it counts, in length units.
    exclude : jnp.ndarray of int, shape ``(n_rays,)``, optional
        One triangle per ray to ignore — the source facet itself, which every ray leaving its
        centroid hits at zero distance. Pass ``-1`` to exclude nothing for that ray.
    work_limit : int, optional
        How many ray-by-triangle entries one pass may form. **This is the only tuning knob that
        matters, and it matters a great deal**: the intermediate is the whole memory cost of the
        test, and the throughput is flat at roughly 50 Mtest/s while it fits and falls off a
        cliff when it does not. Measured on 2048 triangles, double precision, an eleven-core
        machine with 19 GB: 0.5 MB through 134 MB of intermediate all run at 42-57 Mtest/s, and
        537 MB runs at **2.4** — twenty times slower. The default keeps one intermediate near
        32 MB, which leaves room for the several the test forms at once.

        Both axes are cut to honour it. Blocking only the triangles is not enough: the ray count
        is itself the product of receivers and facets, so it reaches the millions on its own and
        would blow the limit at a block size of one.

    Returns
    -------
    jnp.ndarray of bool, shape ``(n_rays,)``
    """
    origin = jnp.asarray(origin, dtype=float)
    direction = jnp.asarray(target, dtype=float) - origin
    length = jnp.sqrt(jnp.sum(direction * direction, axis=-1))
    near = jnp.asarray(min_distance) / jnp.where(length == 0.0, 1.0, length)

    vertices = jnp.asarray(vertices, dtype=float)
    exclude = None if exclude is None else jnp.asarray(exclude)
    n_rays, n_triangles = origin.shape[0], vertices.shape[0]
    if n_rays == 0 or n_triangles == 0:
        return jnp.zeros(n_rays, dtype=bool)

    ray_chunk = max(1, min(n_rays, work_limit))
    block_size = max(1, min(n_triangles, work_limit // ray_chunk))

    pieces = []
    for first in range(0, n_rays, ray_chunk):
        rays = slice(first, first + ray_chunk)
        cut = jnp.zeros(origin[rays].shape[0], dtype=bool)
        for start in range(0, n_triangles, block_size):
            block = vertices[start : start + block_size]
            distance, meets = _watertight_hit(
                origin[rays][:, None, :], direction[rays][:, None, :], block[None, ...]
            )
            hit = meets & (distance > near[rays][:, None]) & (distance <= 1.0)
            if exclude is not None:
                indices = jnp.arange(start, start + block.shape[0])
                hit = hit & (indices[None, :] != exclude[rays][:, None])
            cut = cut | jnp.any(hit, axis=-1)
        pieces.append(cut)
    return jnp.concatenate(pieces)
