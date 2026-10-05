"""Plane mirrors: the planes a specular surface reflects in, and the images it forms.

A flat facet that reflects specularly sends light off along the mirror image of the direction it
arrived from. Seen from the other side of the reflection, that light comes in a straight line from
the **mirror image** of its source -- the source reflected in the facet's plane -- so a specular
bounce can be gathered as an ordinary direct contribution from an image, with every closed-form
kernel, shadow test and attenuation the direct gather already has. That is the image-source method,
and it is exact for a plane.

A :class:`Mirror` is one plane and the facets of a surface set that lie in it. The facets are the
mirror's **aperture**: an image is seen only through them, never through the rest of the plane.
Every facet of one plane shares one image of each source, so a flat wall triangulated into many
pieces is one mirror, not many -- which is what keeps the number of images, and with it the cost,
set by the scene's planes rather than by its triangulation.

:func:`planar_mirrors` finds the mirrors of the bodies a caller names, grouping each body's facets
by the plane they lie in. Only a flat body is the right input: a curved one is many small planes,
one mirror each, which is exact for the faceted surface but describes a curved mirror poorly and
expensively, so it is a different method's job.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.surfaces import Surfaces
from aquaflux.vectors import reflect

__all__ = ["Mirror", "planar_mirrors"]

#: Default coplanarity tolerance, as a fraction of the surface set's extent. Far above the rounding
#: of a single-precision STL (about 6e-8 of a coordinate) and the error of a computed normal, so a
#: flat wall read from a file is one plane; and an error of this size in where an image stands is
#: negligible however a curved surface's facets end up grouped.
_RELATIVE_TOLERANCE = 1e-6


class Mirror(eqx.Module):
    """A plane that reflects specularly, and the facets of a surface set that lie in it.

    Built by :func:`planar_mirrors`. The plane is stored as a point on it and its unit normal,
    which points to the side its facets face -- the side a mirror reflects on.

    Attributes
    ----------
    point : jnp.ndarray, shape ``(3,)``
        A point on the plane: the area-weighted mean of its facets' centroids.
    normal : jnp.ndarray, shape ``(3,)``
        Unit normal of the plane, on the side its facets face.
    facets : numpy.ndarray of int, shape ``(n_aperture,)``
        Indices, into the surface set the mirror was found in, of the facets in its plane, in
        ascending order. They are its aperture. A **numpy** array, because it is a label that
        says which facets a later step treats as this mirror, as
        :attr:`~aquaflux.radiation.surfaces.Surfaces.profile_index` is.
    """

    point: jnp.ndarray
    normal: jnp.ndarray
    facets: np.ndarray

    def reflect_points(self, points) -> jnp.ndarray:
        """Reflect positions in the plane.

        Parameters
        ----------
        points : array_like, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray, shape ``(..., 3)``
        """
        points = jnp.asarray(points, dtype=float)
        return self.point + reflect(points - self.point, self.normal)

    def reflect_directions(self, directions) -> jnp.ndarray:
        """Reflect directions -- or any free vector, such as a normal -- in the plane.

        Parameters
        ----------
        directions : array_like, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray, shape ``(..., 3)``
        """
        return reflect(jnp.asarray(directions, dtype=float), self.normal)

    def heights(self, points) -> np.ndarray:
        """Signed distance of each point from the plane, positive on the side the mirror faces.

        Host work, for deciding which points and facets a path through the mirror can involve:
        the points must be concrete.

        Parameters
        ----------
        points : array_like, shape ``(..., 3)``

        Returns
        -------
        numpy.ndarray, shape ``(...)``
        """
        return (np.asarray(points, dtype=float) - np.asarray(self.point)) @ np.asarray(self.normal)

    def in_front(self, points) -> np.ndarray:
        """Indices of the points strictly in front of the mirror.

        Only those see anything in it: the mirror reflects on the side it faces, and a point on
        the plane sees the mirror edge on.

        Parameters
        ----------
        points : array_like, shape ``(n_points, 3)``
            Concrete.

        Returns
        -------
        numpy.ndarray of int
        """
        return np.flatnonzero(self.heights(points) > 0.0)

    def sources_in_front(self, surfaces: Surfaces) -> np.ndarray:
        """Indices of the facets with any part strictly in front of the mirror.

        Only those have an image a point in front can see. A facet wholly on or behind the plane
        has none -- the mirror's own facets and every facet coplanar with them among them -- and
        a facet straddling the plane is kept, since its part in front has an image.

        Parameters
        ----------
        surfaces : Surfaces
            Concrete.

        Returns
        -------
        numpy.ndarray of int
        """
        return np.flatnonzero(self.heights(surfaces.vertices).max(axis=1) > 0.0)

    def image(self, surfaces: Surfaces) -> Surfaces:
        """The mirror image of a surface set in this plane: what a viewer sees in the mirror.

        Each triangle is reflected and its winding reversed, which reflects its outward normal
        rather than turning it inwards, and each angular distribution is replaced by its own
        mirror image (:meth:`~aquaflux.radiation.profiles.Profile.mirrored`). So the image
        sends towards each reflected direction exactly what its source sends towards the
        direction itself, and every optical property and label is carried across unchanged.
        Reflecting the whole set does not decide which facets an image is formed for; a facet
        lying in or behind the plane has no image a viewer in front of it could see, and the
        caller chooses.

        The vertices may be traced: the image follows them, as
        :meth:`~aquaflux.radiation.surfaces.Surfaces.with_geometry` does for a moved set.

        Parameters
        ----------
        surfaces : Surfaces

        Returns
        -------
        Surfaces
            Facet ``i`` of the result is the image of facet ``i`` of ``surfaces``.
        """
        # Swapping two corners reverses the winding, which negates the normal the corners give.
        # A reflection reverses handedness too, so the two together give the reflected normal.
        reflected = self.reflect_points(surfaces.vertices)[:, jnp.array([0, 2, 1])]
        return surfaces.with_geometry(reflected).with_optics(
            profiles=tuple(profile.mirrored(self.normal) for profile in surfaces.profiles),
            profile_index=surfaces.profile_index,
        )


def planar_mirrors(
    surfaces: Surfaces, solids: Sequence[str], *, tolerance: float | None = None
) -> tuple[Mirror, ...]:
    """Group the facets of the named bodies into the planes they lie in, one mirror per plane.

    A facet joins a plane when all three of its corners lie within ``tolerance`` of it and it
    faces the same side. Facets are taken largest first, each one not yet placed starting a
    plane of its own, so a plane is set by a well-shaped facet rather than by a sliver whose
    computed normal is mostly rounding. Each plane's normal and point are then the area-weighted
    means over every facet placed in it. Two facets in one plane but facing opposite ways -- the
    two sides of a thin sheet -- are two mirrors, since each reflects only on the side it faces.

    Point sources are skipped: a facet with no area has no plane.

    Parameters
    ----------
    surfaces : Surfaces
        The surface set the mirrors are found in.
    solids : sequence of str
        Names of the bodies, from :attr:`~aquaflux.radiation.surfaces.Surfaces.solid_names`,
        that reflect specularly. Each is grouped on its own, so two bodies never share a mirror
        even where they share a plane.
    tolerance : float, optional
        How far a corner may lie from a plane and still be in it, in the surface set's length
        unit. Defaults to a millionth of the set's extent.

    Returns
    -------
    tuple of Mirror
        Ordered by the lowest facet index each holds.

    Raises
    ------
    KeyError
        If a name is not a body of the surface set; a misspelled body would otherwise reflect
        nothing, with no error.
    ValueError
        If ``tolerance`` is not positive.
    """
    unknown = sorted(set(solids) - set(surfaces.solid_names))
    if unknown:
        msg = f"no such body in this surface set: {unknown}; have {list(surfaces.solid_names)}"
        raise KeyError(msg)
    vertices = np.asarray(surfaces.vertices, dtype=float)
    if tolerance is None:
        extent = float(np.ptp(vertices.reshape(-1, 3), axis=0).max()) if len(vertices) else 0.0
        tolerance = _RELATIVE_TOLERANCE * extent
    if not tolerance > 0.0:
        msg = f"tolerance must be positive; got {tolerance}"
        raise ValueError(msg)

    solid_id = np.asarray(surfaces.solid_id)
    areal = ~surfaces.is_point_source
    mirrors = []
    for name in dict.fromkeys(solids):
        body = surfaces.solid_names.index(name)
        facets = np.flatnonzero((solid_id == body) & areal)
        mirrors += _planes(
            facets,
            vertices,
            np.asarray(surfaces.centroid, dtype=float),
            np.asarray(surfaces.normal, dtype=float),
            np.asarray(surfaces.area, dtype=float),
            tolerance,
        )
    return tuple(sorted(mirrors, key=lambda mirror: int(mirror.facets[0])))


def _planes(facets, vertices, centroid, normal, area, tolerance) -> list[Mirror]:
    """One body's facets grouped into the planes they lie in, largest facet first.

    Parameters
    ----------
    facets : numpy.ndarray of int, shape ``(m,)``
        The body's areal facets.
    vertices : numpy.ndarray, shape ``(n_facets, 3, 3)``
    centroid, normal : numpy.ndarray, shape ``(n_facets, 3)``
    area : numpy.ndarray, shape ``(n_facets,)``
    tolerance : float

    Returns
    -------
    list of Mirror
    """
    unplaced = facets[np.argsort(-area[facets], kind="stable")]
    planes = []
    while unplaced.size:
        seed = unplaced[0]
        heights = np.einsum("fkd,d->fk", vertices[unplaced] - centroid[seed], normal[seed])
        member = (np.abs(heights).max(axis=1) <= tolerance) & (normal[unplaced] @ normal[seed] > 0)
        # The seed lies in its own plane by construction, but a sliver's corners can sit further
        # from its own computed plane than the tolerance; it still starts this plane.
        member[0] = True
        placed = np.sort(unplaced[member])
        weight = area[placed]
        summed = weight @ normal[placed]
        planes.append(
            Mirror(
                point=jnp.asarray(weight @ centroid[placed] / weight.sum()),
                normal=jnp.asarray(summed / np.linalg.norm(summed)),
                facets=placed,
            )
        )
        unplaced = unplaced[~member]
    return planes
