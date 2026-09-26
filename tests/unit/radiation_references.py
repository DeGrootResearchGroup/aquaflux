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


def mid_box_sheet(divisions: int, *, span: float) -> np.ndarray:
    """A sheet across the unit box at ``x = 0.5``, meshed ``divisions`` a side over ``span``.

    With ``span`` 1 its rim lies on grid lines of :func:`inward_box` of the same divisions, so
    every rim edge is shared with a wall: a sheet welded in all the way round.
    """
    edges = np.linspace(0.5 - span / 2, 0.5 + span / 2, divisions + 1)
    triangles = []
    for i in range(divisions):
        for j in range(divisions):
            a, b = edges[i], edges[i + 1]
            c, d = edges[j], edges[j + 1]
            triangles.append([[0.5, a, c], [0.5, b, c], [0.5, b, d]])
            triangles.append([[0.5, a, c], [0.5, b, d], [0.5, a, d]])
    return np.array(triangles)


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
        The cross-section, in order.

        ⚠️ **A non-convex outline gives OVERLAPPING CAP TRIANGLES, and the body is then closed
        but not a valid surface.** The caps are fanned from the outline's mean point, which only
        tiles a convex outline; on :data:`L_OUTLINE` two cap triangles overlap over 76% of one
        of them. That is harmless for a ray-tightness sweep, where all that matters is that no
        ray escapes — which is what this fixture was built for, and an L does give the reflex
        edge that sweep wants. It is *not* harmless for anything integrating over the surface: a
        radiosity or occlusion fixture built on it has two coplanar triangles genuinely hiding
        one another, which no real surface does.
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


# ---------------------------------------------------------------------------------------------
# Dense reference integrals for the factors the transfer evaluates at ONE point per pair.
#
# ``build_transfer`` integrates the geometric term over the receiving facet but multiplies it by
# absorption and by a source's angular profile evaluated at a single centroid-to-centroid
# direction. Both are therefore biased by however much the factor varies across a facet, and
# both biases are second order in that variation. These integrate the same quantities densely,
# so the bias can be measured rather than argued about.
# ---------------------------------------------------------------------------------------------

#: Sub-triangles per edge when a facet is integrated densely. Convergence in this number is
#: checked rather than assumed -- a reference that is still moving judges nothing.
DENSE_SUBDIVISIONS = 24


def uniform_samples(triangle: np.ndarray, subdivisions: int = DENSE_SUBDIVISIONS):
    """Equal-area sample points over one triangle, with the area each stands for.

    Splitting every edge into ``k`` parts cuts the triangle into ``k**2`` sub-triangles of
    exactly equal area, so each centroid carries the same weight and no quadrature rule's own
    bias enters a reference built from them.

    Parameters
    ----------
    triangle : np.ndarray, shape ``(3, 3)``
        The corners.
    subdivisions : int, optional
        Parts per edge, ``k``.

    Returns
    -------
    tuple of (np.ndarray of shape ``(k**2, 3)``, float)
        The sample points, and the area per sample.
    """
    first, second, third = np.asarray(triangle, dtype=float)
    edge_a, edge_b = second - first, third - first
    k = int(subdivisions)
    offsets = []
    for i in range(k):
        for j in range(k - i):
            # The upward sub-triangle of each lattice cell, plus the downward one that
            # completes the rhombus where there is room for it.
            offsets.append(((3 * i + 1) * edge_a + (3 * j + 1) * edge_b) / (3.0 * k))
            if i + j < k - 1:
                offsets.append(((3 * i + 2) * edge_a + (3 * j + 2) * edge_b) / (3.0 * k))
    area = 0.5 * float(np.linalg.norm(np.cross(edge_a, edge_b)))
    return first + np.array(offsets), area / k**2


def unit_facet(centre, edge: float, facing_down: bool = False) -> np.ndarray:
    """A right triangle of area exactly ``edge**2``, centred on ``centre``, normal along z.

    Its length scale ``sqrt(area)`` is exactly ``edge``, so a sweep over separation or
    absorbance reads directly against ``a * w`` or ``w / r`` with no shape factor in the way.
    ``facing_down`` reverses the winding, and with it the normal, for the far side of a pair.
    """
    along, across = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if facing_down:
        along, across = across, along
    corners = np.array(
        [
            -edge * (along + across) / 2.0,
            edge * (along - across) / 2.0,
            edge * (3.0 * across - along) / 2.0,
        ]
    )
    # Shifted so ``centre`` is the CENTROID rather than an arbitrary reference point. Without
    # it the two facets of a "facing" pair sit laterally offset from one another, and a sweep
    # labelled by separation is quietly measuring an oblique geometry.
    return np.asarray(centre, dtype=float) + corners - corners.mean(axis=0)


def _geometry(triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal, taken from the shipped surface geometry rather than re-derived.

    A reference that derived its own normal could differ from the code under test in winding
    convention and measure that difference instead of the thing being asked about.
    """
    surfaces = Surfaces.from_triangles(np.asarray(triangle, dtype=float)[None, ...])
    return np.asarray(surfaces.centroid)[0], np.asarray(surfaces.normal)[0]


def _pair_kernel(source, receiver, subdivisions):
    """Sample-by-sample transfer kernel over a facet pair, and each sample pair's separation.

    The kernel is ``cos_s cos_r / (pi r**2)`` times the two sample areas -- the integrand of the
    pair's transfer, with the absorption and the profile left out so either can be weighted by
    it. Sample pairs turned away from one another carry nothing between them and are clamped to
    zero rather than dropped, so the array shape does not depend on the geometry.
    """
    points_s, area_s = uniform_samples(source, subdivisions)
    points_r, area_r = uniform_samples(receiver, subdivisions)
    _, normal_s = _geometry(source)
    _, normal_r = _geometry(receiver)

    offset = points_r[None, :, :] - points_s[:, None, :]
    distance = np.linalg.norm(offset, axis=-1)
    cos_s = np.einsum("srd,d->sr", offset, normal_s) / distance
    cos_r = -np.einsum("srd,d->sr", offset, normal_r) / distance
    kernel = np.clip(cos_s, 0.0, None) * np.clip(cos_r, 0.0, None) / (np.pi * distance**2)
    return kernel * area_s * area_r, distance


def centroid_separation(source, receiver) -> float:
    """Centroid-to-centroid distance -- the one number the shipped build carries per pair."""
    centroid_s, _ = _geometry(source)
    centroid_r, _ = _geometry(receiver)
    return float(np.linalg.norm(centroid_r - centroid_s))


def mean_separation_excess(source, receiver, subdivisions=DENSE_SUBDIVISIONS) -> float:
    """``<r> - r_centroid``: how far the centroid separation sits from the mean path length.

    The mean is weighted by the pair's own transfer kernel, because that is the weighting the
    absorption factor is averaged under. This quantity is the whole of the absorption bias to
    first order -- the bias is ``-a`` times it, at every geometry.

    ⚠️ **It is NOT always positive, and assuming it was cost a wrong claim in three files.**
    Jensen's inequality on the norm says the distance between two mean positions cannot exceed
    the mean of the distances, which is true and is about the *kernel-weighted* mean positions,
    not the geometric centroids the build actually stores. Head-on the two nearly coincide and
    the excess is positive; slide the pair sideways and the kernel -- which goes like
    ``1 / r**2`` -- concentrates on the facing near corners until the typical separation it
    weights falls *below* the centroid-to-centroid distance. That sign change is exactly the
    sign change in the absorption bias.
    """
    kernel, distance = _pair_kernel(source, receiver, subdivisions)
    mean = float(np.sum(kernel * distance) / np.sum(kernel))
    return mean - centroid_separation(source, receiver)


def absorption_bias(source, receiver, coefficient: float, subdivisions=DENSE_SUBDIVISIONS):
    """Exact absorbed transfer over the closed form taken at the centroid separation.

    The build multiplies a pair's geometric term by ``exp(-a * r_centroid)``; the honest factor
    is the kernel-weighted average of ``exp(-a * r)`` over both facets. This returns their
    ratio, so **below one means the shipped form transmits too much**.

    ⚠️ The leading term is *first* order in ``a * w``, not the second-order Jensen correction on
    ``exp``, and its sign is not fixed across geometries. Both facts come from
    :func:`mean_separation_excess`, which predicts this to four figures wherever it is checked.
    """
    kernel, distance = _pair_kernel(source, receiver, subdivisions)
    exact = float(np.sum(kernel * np.exp(-coefficient * distance)))
    gap = centroid_separation(source, receiver)
    return exact / (float(np.sum(kernel)) * float(np.exp(-coefficient * gap)))


def _source_samples(source, receiver_point, subdivisions):
    """Per-sample source cosines and the solid angle each sample subtends at the receiver."""
    points, area = uniform_samples(source, subdivisions)
    _, normal = _geometry(source)
    offset = np.asarray(receiver_point, dtype=float)[None, :] - points
    distance = np.linalg.norm(offset, axis=-1)
    cosine = np.clip(np.einsum("sd,d->s", offset, normal) / distance, 0.0, None)
    return cosine, cosine * area / distance**2


def profile_bias(source, receiver_point, exponent: float, subdivisions=DENSE_SUBDIVISIONS):
    """Exact emitted transfer over the closed form taken at the centroid direction.

    The solid angle a source facet subtends is integrated exactly, but the profile weighting it
    is evaluated once, at the centroid direction. This returns the ratio of the honest integral
    to that product, so **above one means the shipped form is too dark**. It is exactly one for
    a Lambertian source at every geometry, which is the control the sweep needs.
    """
    from aquaflux.radiation.profiles import CosinePower

    profile = CosinePower(exponent)
    cosine, omega = _source_samples(source, receiver_point, subdivisions)
    centroid, normal = _geometry(source)
    to_receiver = np.asarray(receiver_point, dtype=float) - centroid
    at_centroid = float(np.dot(to_receiver, normal) / np.linalg.norm(to_receiver))

    exact = float(np.sum(np.asarray(profile.radiance_per_exitance(cosine)) * omega))
    return exact / (float(profile.radiance_per_exitance(at_centroid)) * float(np.sum(omega)))


def profile_cumulant_bias(source, receiver_point, exponent: float, subdivisions=DENSE_SUBDIVISIONS):
    """What is left of the profile bias under a two-moment frozen/live split.

    Freezing a *set* of directions per pair and evaluating the profile at each would multiply
    the frozen ``n**2`` array by the sample count. There is a cheaper split: the profile is
    ``c**(n - 1)`` up to constants, so with ``l = log c`` the solid-angle-weighted average is
    ``<exp((n - 1) l)>``, whose cumulant expansion is

        exp( (n - 1) <l>  +  (n - 1)**2 Var(l) / 2  +  ... )

    Truncating after the variance needs **two** frozen numbers per pair instead of one per
    sample, and leaves the exponent outside them, whole and differentiable. Both correction
    terms vanish at ``n = 1`` along with the error itself.

    Returns the residual ratio after that correction, to be read against
    :func:`profile_bias`'s.
    """
    cosine, omega = _source_samples(source, receiver_point, subdivisions)
    lit = cosine > 0.0
    share = omega[lit] / np.sum(omega[lit])
    log_cosine = np.log(cosine[lit])
    mean = float(np.sum(share * log_cosine))
    variance = float(np.sum(share * (log_cosine - mean) ** 2))

    power = float(exponent) - 1.0
    exact = float(np.sum(cosine[lit] ** power * omega[lit]) / np.sum(omega[lit]))
    return exact / float(np.exp(power * mean + 0.5 * power**2 * variance))


def sampled_fraction(receiver, receiver_normal, source, blockers, samples=200_000, seed=0):
    """Brute force: area-sample the source, weight by its solid-angle measure, ray-test.

    The independent reference for the analytic silhouette clip -- a different algorithm
    entirely, so agreement between them is evidence rather than a tautology. It converges like
    ``1 / sqrt(samples)``, so a disagreement is only meaningful once it stops shrinking as the
    sample count rises.

    ⚠️ **The weight is the PROJECTED SOLID ANGLE measure, not area.** Area-uniform samples
    weighted by the receiver cosine alone answer a different question, and the gap does not
    vanish with more samples -- it plateaus, which reads exactly like a real discrepancy in
    whatever is being judged. Both source and receiver cosines and the inverse square are
    needed. For a receiver in the volume -- ``receiver_normal`` of ``None`` -- the measure is the
    plain solid angle, and only the source cosine and the inverse square remain.

    Parameters
    ----------
    receiver : array_like, shape ``(3,)``
    receiver_normal : array_like, shape ``(3,)``, or None
    source : array_like, shape ``(3, 3)``
    blockers : array_like, shape ``(3, 3)`` or ``(n, 3, 3)``
    samples : int, optional
    seed : int, optional

    Returns
    -------
    float
        The blocked share of the source's projected solid angle -- or of its plain solid angle,
        for a receiver in the volume -- in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    a, b, c = np.asarray(source, dtype=float)
    u, v = rng.random(samples), rng.random(samples)
    outside = u + v > 1.0
    u, v = np.where(outside, 1 - u, u), np.where(outside, 1 - v, v)
    points = a + u[:, None] * (b - a) + v[:, None] * (c - a)

    receiver = np.asarray(receiver, dtype=float)
    offset = points - receiver
    squared = np.sum(offset * offset, axis=1)
    unit = offset / np.sqrt(squared)[:, None]
    source_normal = np.cross(b - a, c - a)
    source_normal /= np.linalg.norm(source_normal)
    weight = np.abs(unit @ source_normal) / squared
    if receiver_normal is not None:
        weight = weight * np.abs(unit @ np.asarray(receiver_normal))

    blocked = np.zeros(samples, dtype=bool)
    for p0, p1, p2 in np.atleast_3d(np.asarray(blockers, dtype=float)).reshape(-1, 3, 3):
        e1, e2 = p1 - p0, p2 - p0
        h = np.cross(unit, e2)
        det = e1 @ h.T
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = receiver - p0
        bu = inv * (s @ h.T)
        q = np.cross(s, e1)
        bv = inv * (unit @ q)
        t = inv * (e2 @ q)
        inside = ok & (bu >= 0) & (bu <= 1) & (bv >= 0) & (bu + bv <= 1)
        # Strictly between the receiver and the sample: a triangle beyond the source does not
        # occlude it, and one at the origin is the receiver's own facet.
        blocked |= inside & (t > 1e-12) & (t < np.sqrt(squared))
    return float(np.sum(weight * blocked) / np.sum(weight))
