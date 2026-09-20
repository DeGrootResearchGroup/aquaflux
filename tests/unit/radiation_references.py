"""Closed-form and third-party reference values for the radiation solid-angle kernels.

Every number a kernel is judged against comes from outside the kernel. Two of the three
sources here are closed forms a reader can look up; the third is an independent
implementation whose values are recorded rather than imported, for the reason given at
:data:`PYVIEWFACTOR_PARALLEL_SQUARES`.
"""

from __future__ import annotations

import numpy as np
from aquaflux.radiation.surfaces import Surfaces

#: View factors between two identical, directly opposed, parallel unit squares, computed by
#: **pyviewfactor 1.1.0** (MIT licence) on 2026-09-18, mapping separation ``d/L`` to ``F``.
#:
#: pyviewfactor evaluates the double area integral by a contour integral — a different
#: algorithm from anything in this package, written by people with no stake in this
#: formulation, which is the property that makes it worth having. It agrees with
#: :func:`parallel_squares_view_factor` to 3.5e-16 over this range, and on a perpendicular
#: pair sharing an edge it returns 0.20004386856510536 against the 0.20004 tabulated in the
#: standard view-factor catalogues, so it was checked before it was trusted.
#:
#: The values are recorded here rather than computed by importing the package, because a test
#: hidden behind an optional import is skipped silently and a skip is indistinguishable from a
#: pass. Regenerate them with ``pip install pyviewfactor`` and the recipe in this module's
#: accompanying test, and update the date above if they move.
PYVIEWFACTOR_PARALLEL_SQUARES: dict[float, float] = {
    0.25: 0.6320364300138601,
    0.5: 0.41525328357714675,
    1.0: 0.19982489569838735,
    2.0: 0.0685895888185526,
    4.0: 0.019106958038708648,
}

#: Same source and date: the view factor between two unit squares meeting at a right angle
#: along a shared edge. This one earns its place by being strongly oblique — the receiver's
#: obliquity sweeps the full range across the emitter, which is exactly the configuration an
#: unweighted solid angle gets wrong and an on-axis test cannot see.
PYVIEWFACTOR_PERPENDICULAR_SQUARES = 0.20004386856510536


def axial_rectangle_solid_angle(half_width: float, half_height: float, distance: float) -> float:
    """Solid angle of a rectangle at a point on its axis, in steradians.

    The standard closed form for a ``2a x 2b`` rectangle whose centre lies a distance ``d``
    along its normal from the observer::

        Omega = 4 arctan( a b / (d sqrt(a^2 + b^2 + d^2)) )
    """
    a, b, d = half_width, half_height, distance
    return 4.0 * np.arctan(a * b / (d * np.sqrt(a * a + b * b + d * d)))


def parallel_squares_view_factor(side: float, distance: float) -> float:
    """View factor between identical, directly opposed, parallel squares.

    The classical closed form for two coaxial rectangles, here with both in-plane dimensions
    equal. ``F`` is the fraction of everything leaving one square, diffusely, that lands on
    the other — an area-to-area average, not a value at a point, which is why a kernel
    evaluated at a single receiver point only approaches it as the receiver is refined.
    """
    x = y = side / distance
    terms = (
        np.log(np.sqrt((1 + x**2) * (1 + y**2) / (1 + x**2 + y**2)))
        + x * np.sqrt(1 + y**2) * np.arctan(x / np.sqrt(1 + y**2))
        + y * np.sqrt(1 + x**2) * np.arctan(y / np.sqrt(1 + x**2))
        - x * np.arctan(x)
        - y * np.arctan(y)
    )
    return 2.0 / (np.pi * x * y) * terms


def rectangle_triangles(centre, first_edge, second_edge) -> np.ndarray:
    """Two triangles covering the rectangle spanned by the two half-edge vectors."""
    centre, first_edge, second_edge = (
        np.asarray(v, dtype=float) for v in (centre, first_edge, second_edge)
    )
    corners = np.array(
        [
            centre - first_edge - second_edge,
            centre + first_edge - second_edge,
            centre + first_edge + second_edge,
            centre - first_edge + second_edge,
        ]
    )
    return np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]])


def box_enclosure(divisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed unit box triangulated ``divisions x divisions`` per face.

    Returns the vertices ``(n, 3, 3)``, the **inward** unit normals ``(n, 3)``, and the facet
    centroids ``(n, 3)``. The windings are deliberately left inconsistent between faces, so a
    kernel that quietly depends on vertex order cannot pass on this fixture.
    """
    edges = np.linspace(0.0, 1.0, divisions + 1)
    triangles, normals = [], []
    for axis in range(3):
        for side in (0.0, 1.0):
            inward = [0.0, 0.0, 0.0]
            inward[axis] = 1.0 if side == 0.0 else -1.0
            for i in range(divisions):
                for j in range(divisions):

                    def corner(first, second, axis=axis, side=side):
                        point = [0.0, 0.0, 0.0]
                        point[axis] = side
                        point[(axis + 1) % 3] = first
                        point[(axis + 2) % 3] = second
                        return point

                    a = corner(edges[i], edges[j])
                    b = corner(edges[i + 1], edges[j])
                    c = corner(edges[i + 1], edges[j + 1])
                    d = corner(edges[i], edges[j + 1])
                    triangles += [[a, b, c], [a, c, d]]
                    normals += [inward, inward]
    vertices = np.array(triangles)
    return vertices, np.array(normals), vertices.mean(axis=1)


def disc_triangles(radius: float, rings: int = 24, sectors: int = 64, height: float = 0.0):
    """A flat disc in the ``z = height`` plane, triangulated and wound counter-clockwise.

    Built vectorized rather than by looping over cells: the fixtures here run in the always-on
    gate, and a Python loop over a few thousand triangles costs more than the gather it feeds.
    """
    radii = np.linspace(0.0, radius, rings + 1)
    angles = np.linspace(0.0, 2.0 * np.pi, sectors + 1)
    inner, outer = radii[:-1, None], radii[1:, None]
    start, end = angles[None, :-1], angles[None, 1:]

    def corner(r, a):
        return np.stack(
            [
                r * np.cos(a) * np.ones_like(a * r),
                r * np.sin(a) * np.ones_like(a * r),
                np.full(np.broadcast(r, a).shape, height),
            ],
            axis=-1,
        )

    a = corner(inner, start)
    b = corner(outer, start)
    c = corner(outer, end)
    d = corner(inner, end)
    lower = np.stack([a, b, c], axis=-2).reshape(-1, 3, 3)
    upper = np.stack([a, c, d], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([lower, upper])


def cylinder_triangles(radius: float, half_length: float, sectors: int = 160, slices: int = 160):
    """A tube of the given radius about the z axis, outward-facing, without end caps.

    A cylinder is convex, so from any outside point the facets a receiver can see are exactly
    those whose outward normal faces it — which makes this the fixture that tests the
    source-side clamp without needing an occluder.

    ⚠️ **Its seam does not close, so capping it does not give a watertight body.** The angles
    run ``linspace(0, 2 * pi, sectors + 1)``, and the last sector ends at ``2 * pi``, whose sine
    is ``-2.4e-16`` rather than zero — so the first and last columns of vertices differ in the
    last bits and the surface has a hair-width slit down it. That is invisible to everything
    this fixture is used for, and fatal to a ray-tightness sweep: rays aimed at the seam escape
    through it whatever the intersection test does. Use :func:`closed_prism` when the body has
    to be closed.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sectors + 1)
    heights = np.linspace(-half_length, half_length, slices + 1)
    start, end = angles[:-1, None], angles[1:, None]
    low, high = heights[None, :-1], heights[None, 1:]

    def corner(a, z):
        shape = np.broadcast(a, z).shape
        return np.stack(
            [
                np.broadcast_to(radius * np.cos(a), shape),
                np.broadcast_to(radius * np.sin(a), shape),
                np.broadcast_to(z, shape),
            ],
            axis=-1,
        )

    a = corner(start, low)
    b = corner(end, low)
    c = corner(end, high)
    d = corner(start, high)
    lower = np.stack([a, b, c], axis=-2).reshape(-1, 3, 3)
    upper = np.stack([a, c, d], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([lower, upper])


def finite_line_fluence_rate(power_per_length: float, half_length: float, radius: float) -> float:
    """Closed form for a finite isotropic line source at perpendicular distance ``radius``.

    ``G = P' (alpha_2 - alpha_1) / (4 pi r)`` with ``alpha = arctan(z / r)``, measured at the
    line's mid-plane so the two angles are symmetric. The infinite limit is ``P' / (4 r)``.
    """
    alpha = np.arctan(half_length / radius)
    return power_per_length * (2.0 * alpha) / (4.0 * np.pi * radius)


def inward_box(divisions: int = 2) -> np.ndarray:
    """A closed unit box triangulated ``divisions`` per face, wound to face **inward**.

    :func:`box_enclosure` deliberately leaves its windings inconsistent, which is what makes it
    a good fixture for a kernel that must not depend on them. A surface set derives its normals
    *from* the winding, so anything that reasons about which way a facet faces needs this
    version instead: every triangle's right-hand-rule normal points into the box.
    """
    vertices, inward, _ = box_enclosure(divisions)
    derived = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    facing_out = np.sum(derived * inward, axis=1) < 0.0
    corrected = vertices.copy()
    corrected[facing_out] = corrected[facing_out][:, ::-1, :]
    return corrected


def box(divisions: int = 2, **optics) -> Surfaces:
    """A closed unit box as a surface set, inward-facing."""
    return Surfaces.from_triangles(inward_box(divisions), **optics)


def stretched_box(divisions: int = 2, **optics) -> Surfaces:
    """A closed box whose facets do **not** all have the same area.

    Stretching the unit box along one axis is affine and positive, so it stays closed and stays
    consistently wound, but its long walls carry triangles three times the area of its ends.
    Every equal-area fixture is blind to which index of the transfer matrix an area belongs on,
    because both choices are then the same expression.
    """
    return Surfaces.from_triangles(inward_box(divisions) * np.array([1.0, 1.0, 3.0]), **optics)


def facing_plates(n: int, *, half: float = 1.0, gap: float = 1.0) -> np.ndarray:
    """Two square plates facing each other, each meshed ``n x n`` quads of two triangles.

    The fixture for partial occlusion: put a body between them and some facet pairs are half
    shadowed, which is the one configuration the closed boxes elsewhere in this module cannot
    produce — a box has nothing in the way.

    Facet order is plate-major, then row-major over the quads, then the **two triangles of a
    quad adjacent to one another**. :func:`area_average_onto` depends on that last part, and
    getting it wrong still leaves every coarse patch owning two facets of the right plate, so
    an ownership check passes and only a control measurement catches it.
    """
    edges = np.linspace(-half, half, n + 1)
    corners = np.array(
        [
            [
                [edges[i], edges[j], 0.0],
                [edges[i + 1], edges[j], 0.0],
                [edges[i + 1], edges[j + 1], 0.0],
                [edges[i], edges[j + 1], 0.0],
            ]
            for i in range(n)
            for j in range(n)
        ]
    )
    flat = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]], axis=1)
    flat = flat.reshape(-1, 2, 3, 3).reshape(-1, 3, 3)
    lower, upper = flat.copy(), flat.copy()
    lower[:, :, 2] = -gap
    upper[:, :, 2] = gap
    return np.concatenate([lower, upper[:, ::-1, :]])


def quad_of_facet(n: int, n_coarse: int) -> np.ndarray:
    """Which coarse quad each facet of :func:`facing_plates` belongs to.

    Raises
    ------
    ValueError
        If ``n_coarse`` does not divide ``n``. Integer division would otherwise run the owner
        index off the end of the coarse grid, which surfaces as an out-of-range scatter deep
        inside :func:`area_average_onto` rather than as a statement about the two meshes.
    """
    if n_coarse <= 0 or n % n_coarse:
        msg = f"the coarse grid must divide the fine one; {n_coarse} does not divide {n}"
        raise ValueError(msg)
    step = n // n_coarse
    row, column = np.divmod(np.arange(n * n), n)
    per_plate = np.repeat((row // step) * n_coarse + (column // step), 2)
    return np.concatenate([per_plate, per_plate + n_coarse**2])


def area_average_onto(matrix, area, n: int, n_coarse: int) -> np.ndarray:
    """Area-average a facet-by-facet transfer onto the coarse quads it refines.

    This *is* the coarse form factor: the source index is averaged over its patch weighted by
    area, because a patch's form factor is the area-weighted mean of its parts'; the receiver
    index is summed, because what lands on a patch is what lands on all of it. So a refined
    transfer aggregated this way is the right thing to compare a coarse one against, rather
    than a finer answer to a different question.

    ⚠️ **Two things here are inert on the plate fixture and are written for correctness rather
    than because a test would catch them.** Every triangle of :func:`facing_plates` is congruent,
    so the area weighting is the same as a plain mean; and a coarse patch and its fine cover have
    the same total area by construction, so dividing by it scales both sides of any comparison
    equally and cancels under a normalized metric. Both matter the moment this is pointed at a
    mesh whose facets differ in size.
    """
    matrix, area = np.asarray(matrix), np.asarray(area)
    owner = quad_of_facet(n, n_coarse)
    n_quads = 2 * n_coarse**2
    columns = np.zeros((len(owner), n_quads))
    for quad in range(n_quads):
        columns[:, quad] = matrix[:, owner == quad].sum(axis=1)
    weighted = np.zeros((n_quads, n_quads))
    total = np.zeros(n_quads)
    np.add.at(weighted, owner, area[:, None] * columns)
    np.add.at(total, owner, area)
    return weighted / total[:, None]


def closed_prism(outline: np.ndarray, half_height: float) -> np.ndarray:
    """A closed, inward-sealed prism over a 2D ``outline``, built from one vertex table.

    Every vertex is written once and reused by index, so two faces meeting at an edge carry
    *the same numbers* rather than numbers that agree to a tolerance. That is what a
    ray-tightness sweep needs and what a body assembled from independently evaluated
    trigonometry cannot offer (see :func:`cylinder_triangles`): a seam whose two sides differ in
    the last bits is a real pinhole, and it will be blamed on the intersection test.

    Parameters
    ----------
    outline : np.ndarray, shape ``(n, 2)``
        The cross-section, in order. It may be non-convex — an L gives a reflex edge, which is
        where an interior ray can graze two faces at once.
    half_height : float
        Half the extrusion along z; the prism spans ``-half_height`` to ``+half_height``.

    Returns
    -------
    np.ndarray, shape ``(4 * n, 3, 3)``
        Two triangles per side wall plus one per cap sector.
    """
    outline = np.asarray(outline, dtype=float)
    count = len(outline)
    low = np.column_stack([outline, np.full(count, -half_height)])
    high = np.column_stack([outline, np.full(count, half_height)])
    centre_low = np.array([*outline.mean(axis=0), -half_height])
    centre_high = np.array([*outline.mean(axis=0), half_height])

    faces = []
    for k in range(count):
        following = (k + 1) % count
        faces.append([low[k], low[following], high[following]])
        faces.append([low[k], high[following], high[k]])
        faces.append([centre_low, low[following], low[k]])
        faces.append([centre_high, high[k], high[following]])
    return np.array(faces)


def closed_drum(sectors: int, radius: float = 1.0, half_height: float = 1.0) -> np.ndarray:
    """A closed circular prism — a curved seam that actually meets itself."""
    angle = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
    return closed_prism(radius * np.column_stack([np.cos(angle), np.sin(angle)]), half_height)


#: A non-convex cross-section: the reflex corner at (0.8, 0.8) is the feature a closed-body
#: tightness sweep wants, because an interior ray can leave through two faces that meet there.
L_OUTLINE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.8], [0.8, 0.8], [0.8, 2.0], [0.0, 2.0]])
