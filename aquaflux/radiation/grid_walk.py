"""The grid walk as one compiled loop per ray, rather than one array pass per step over every ray.

:meth:`~aquaflux.radiation.grid.TriangleGrid.blocks` walks each segment through the voxels it
crosses (Amanatides & Woo's 3D-DDA) and tests the triangles those voxels hold. Written as array
code, a step of that walk is a dozen whole-array operations over every ray still in flight, plus
a compiled call for the (ray, triangle) pairs the step produces. Profiled on a 51,200-triangle
cylindrical wall, that bookkeeping costs about 150-180 ns per ray per voxel step and is most of
the walk's time, so refining the grid -- fewer triangles tested, more steps taken -- makes it
slower rather than faster.

Here each ray is walked to its end by ordinary compiled code, one ray per iteration of a parallel
loop: the walk's state stays in registers, a ray that hits something stops at once rather than at
the end of an array pass, and no (ray, triangle) pair is ever written to memory. The work each
ray does is the same as the array walk's -- the same voxels, the same triangles in the same
order, the same predicate -- so the answers are too.

The compiler is Numba, imported when a walk is first asked for, so the package imports without it.

⚠️ **The intersection test here is a second implementation of**
:func:`~aquaflux.radiation.triangles._watertight_hit` **and**
:func:`~aquaflux.radiation.triangles._counts_as_hit`, because a Numba loop cannot call a traced
JAX function. The two are compared ray for ray by the grid's tests, over both walks, and any
change to one must be made to the other. The edge function keeps the same averaged form, which
is exactly antisymmetric whatever the compiler does to a multiply followed by a subtraction --
that property, not agreement with the traced kernel's rounding, is what keeps a closed surface
closed.
"""

from __future__ import annotations

import functools

import numpy as np


@functools.cache
def _kernel():
    """The compiled walk, built on first use so that importing this module needs no Numba."""
    try:
        import numba
    except ImportError as error:
        msg = (
            "the compiled grid walk needs Numba, which is optional: install it with "
            "`pip install aquaflux[numba]`, or walk with walk='array'"
        )
        raise ImportError(msg) from error

    @numba.njit(inline="always")
    def edge_function(first, second, third, fourth):
        forward = first * second - third * fourth
        return 0.5 * (forward - (third * fourth - first * second))

    @numba.njit(inline="always")
    def sheared(corner, origin, kx, ky, kz, shear_x, shear_y, shear_z):
        ox = corner[kx] - origin[kx]
        oy = corner[ky] - origin[ky]
        oz = corner[kz] - origin[kz]
        return ox - shear_x * oz, oy - shear_y * oz, oz * shear_z

    @numba.njit(inline="always")
    def cuts(origin, direction, near, corners, index, exclude):
        # Woop, Benthin and Wald's watertight test, with the axis choice, shear, edge functions
        # and distance window of the traced kernel, evaluated for one ray and one triangle.
        kz = 0
        largest = abs(direction[0])
        for axis in (1, 2):
            if abs(direction[axis]) > largest:
                kz = axis
                largest = abs(direction[axis])
        kx = (kz + 1) % 3
        ky = (kz + 2) % 3
        dz = direction[kz]
        shear_x = direction[kx] / dz
        shear_y = direction[ky] / dz
        shear_z = 1.0 / dz
        ax, ay, az = sheared(corners[0], origin, kx, ky, kz, shear_x, shear_y, shear_z)
        bx, by, bz = sheared(corners[1], origin, kx, ky, kz, shear_x, shear_y, shear_z)
        cx, cy, cz = sheared(corners[2], origin, kx, ky, kz, shear_x, shear_y, shear_z)
        u = edge_function(cx, by, cy, bx)
        v = edge_function(ax, cy, ay, cx)
        w = edge_function(bx, ay, by, ax)
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
    def walk(
        ray, voxel, until, step, delta, origin, direction, near, exclude, vertices, starts,
        triangles, resolution, max_steps, blocked,
    ):  # fmt: skip
        for live in numba.prange(len(ray)):
            r = ray[live]
            at = voxel[live].copy()
            crossing = until[live].copy()
            for _ in range(max_steps):
                flat = (at[0] * resolution[1] + at[1]) * resolution[2] + at[2]
                for entry in range(starts[flat], starts[flat + 1]):
                    triangle = triangles[entry]
                    if cuts(
                        origin[r], direction[r], near[r], vertices[triangle], triangle, exclude[r]
                    ):
                        blocked[r] = True
                        break
                if blocked[r]:
                    break
                # The first of the smallest crossings, as the array walk's argmin takes it.
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

    return walk


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
    _kernel()(
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
