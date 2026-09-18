"""Closed-form and third-party reference values for the radiation solid-angle kernels.

Every number a kernel is judged against comes from outside the kernel. Two of the three
sources here are closed forms a reader can look up; the third is an independent
implementation whose values are recorded rather than imported, for the reason given at
:data:`PYVIEWFACTOR_PARALLEL_SQUARES`.
"""

from __future__ import annotations

import numpy as np

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
