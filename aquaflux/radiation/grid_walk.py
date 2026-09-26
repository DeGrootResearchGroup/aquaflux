"""The triangle grid's walk: each segment walked to its first hit by one compiled loop.

:meth:`~aquaflux.radiation.grid.TriangleGrid.blocks` walks each segment through the voxels it
crosses (Amanatides & Woo's 3D-DDA) and tests the triangles those voxels hold. Here that walk is
ordinary compiled code, one ray per iteration of a parallel loop (Numba): the walk's state stays in
registers, a ray that hits something stops at once, and no (ray, triangle) pair is ever written to
memory. Both other ways of writing it lose. A traced (``jax``) loop needs a static trip count and
so pays the longest walk against the fullest voxel on every ray; whole-array numpy passes over the
rays still in flight pay about 150-180 ns per ray per voxel step in bookkeeping, profiled on a
51,200-triangle cylindrical wall, so a finer grid made that walk slower rather than faster. On the
same rays this loop measured 17-37x faster than that array walk, with identical answers.

⚠️ **The intersection test here is a second implementation of**
:func:`~aquaflux.radiation.triangles._watertight_hit` **and**
:func:`~aquaflux.radiation.triangles._counts_as_hit`, because a Numba loop cannot call a traced
JAX function, and the dense every-triangle path still needs the traced one. The grid's tests
compare the two ray for ray, and any change to one must be made to the other. The edge function
keeps the same averaged form, which is exactly antisymmetric whatever the compiler does to a
multiply followed by a subtraction -- that property, not agreement with the traced kernel's
rounding, is what keeps a closed surface closed.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(inline="always")
def _edge_function(first, second, third, fourth):
    """``first * second - third * fourth``, exactly antisymmetric under swapping the two pairs."""
    forward = first * second - third * fourth
    return 0.5 * (forward - (third * fourth - first * second))


@numba.njit(inline="always")
def _sheared(corner, origin, kx, ky, kz, shear_x, shear_y, shear_z):
    """One corner relative to the ray's origin, sheared so the ray runs along ``+z``."""
    ox = corner[kx] - origin[kx]
    oy = corner[ky] - origin[ky]
    oz = corner[kz] - origin[kz]
    return ox - shear_x * oz, oy - shear_y * oz, oz * shear_z


@numba.njit(inline="always")
def _ray_frame(direction):
    """The ray's axis permutation and shear, which every triangle it meets is tested in.

    The largest component goes on ``z``, so the shear never divides by a small number. It depends
    on the ray alone, so the walk forms it once per ray rather than once per triangle tested.
    """
    kz = 0
    largest = abs(direction[0])
    for axis in (1, 2):
        if abs(direction[axis]) > largest:
            kz = axis
            largest = abs(direction[axis])
    kx = (kz + 1) % 3
    ky = (kz + 2) % 3
    dz = direction[kz]
    return kx, ky, kz, direction[kx] / dz, direction[ky] / dz, 1.0 / dz


@numba.njit(inline="always")
def _cuts(origin, frame, near, corners, index, exclude):
    """Whether one ray is cut by one triangle it was not told to ignore.

    Woop, Benthin and Wald's watertight test with the traced kernel's axis choice, shear, edge
    functions and distance window: a hit counts past the near margin, at or before the far end.
    ``frame`` is the ray's :func:`_ray_frame`.
    """
    kx, ky, kz, shear_x, shear_y, shear_z = frame
    ax, ay, az = _sheared(corners[0], origin, kx, ky, kz, shear_x, shear_y, shear_z)
    bx, by, bz = _sheared(corners[1], origin, kx, ky, kz, shear_x, shear_y, shear_z)
    cx, cy, cz = _sheared(corners[2], origin, kx, ky, kz, shear_x, shear_y, shear_z)
    u = _edge_function(cx, by, cy, bx)
    v = _edge_function(ax, cy, ay, cx)
    w = _edge_function(bx, ay, by, ax)
    determinant = u + v + w
    if determinant == 0.0:
        return False
    if not ((u >= 0.0 and v >= 0.0 and w >= 0.0) or (u <= 0.0 and v <= 0.0 and w <= 0.0)):
        return False
    distance = (u * az + v * bz + w * cz) / determinant
    if not (distance > near and distance <= 1.0):
        return False
    for excluded in exclude:
        if excluded == index:
            return False
    return True


@numba.njit(parallel=True)
def _walk(
    ray, voxel, until, step, delta, origin, direction, near, exclude, vertices, starts,
    triangles, resolution, max_steps, blocked,
):  # fmt: skip
    """Walk each live ray to its first hit or its end, writing ``blocked`` in place."""
    for live in numba.prange(len(ray)):
        r = ray[live]
        frame = _ray_frame(direction[r])
        at = voxel[live].copy()
        crossing = until[live].copy()
        for _ in range(max_steps):
            flat = (at[0] * resolution[1] + at[1]) * resolution[2] + at[2]
            for entry in range(starts[flat], starts[flat + 1]):
                triangle = triangles[entry]
                if _cuts(origin[r], frame, near[r], vertices[triangle], triangle, exclude[r]):
                    blocked[r] = True
                    break
            if blocked[r]:
                break
            # Step across the nearest voxel face, the first axis winning a tie.
            axis = 0
            if crossing[1] < crossing[axis]:
                axis = 1
            if crossing[2] < crossing[axis]:
                axis = 2
            leaving = crossing[axis]
            moved = at[axis] + step[live, axis]
            if not (leaving <= 1.0 and 0 <= moved < resolution[axis]):
                break
            at[axis] = moved
            crossing[axis] = leaving + delta[live, axis]


def walk_to_first_hit(
    ray, voxel, until, step, delta, origin, direction, near, exclude, grid, max_steps
) -> np.ndarray:
    """Whether each live ray meets a triangle, walking every one of them to its end.

    Parameters
    ----------
    ray : np.ndarray of int, shape ``(n_live,)``
        Which of the rays enter the grid; the rest are not blocked.
    voxel, step : np.ndarray of int, shape ``(n_live, 3)``
        Each live ray's starting voxel, and the direction it steps along each axis.
    until, delta : np.ndarray, shape ``(n_live, 3)``
        The segment parameter at which each live ray next crosses a voxel face on each axis, and
        how far that parameter moves per voxel.
    origin, direction : np.ndarray, shape ``(n_rays, 3)``
        Every ray's start and its direction, the segment running over ``[0, 1]`` of it.
    near : np.ndarray, shape ``(n_rays,)``
        How far along the segment, as a share of it, a hit must be before it counts.
    exclude : np.ndarray of int, shape ``(n_rays, k)``
        Triangles each ray ignores; ``-1`` excludes nothing.
    grid : TriangleGrid
        The triangles and the voxels that hold them.
    max_steps : int
        Most voxels one walk visits, the array walk's own bound.

    Returns
    -------
    np.ndarray of bool, shape ``(n_rays,)``
    """
    blocked = np.zeros(len(origin), dtype=bool)
    _walk(
        np.ascontiguousarray(ray, dtype=np.int64),
        np.ascontiguousarray(voxel, dtype=np.int64),
        np.ascontiguousarray(until, dtype=float),
        np.ascontiguousarray(step, dtype=np.int64),
        np.ascontiguousarray(delta, dtype=float),
        origin,
        direction,
        near,
        np.ascontiguousarray(exclude, dtype=np.int64),
        grid.vertices,
        np.ascontiguousarray(grid.starts, dtype=np.int64),
        np.ascontiguousarray(grid.triangles, dtype=np.int64),
        np.ascontiguousarray(grid.resolution, dtype=np.int64),
        max_steps,
        blocked,
    )
    return blocked
