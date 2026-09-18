"""The emitting and reflecting surface set: triangles plus the per-facet optical properties.

A :class:`Surfaces` is the one description of "what is emitting and what is reflecting" that
every later stage reads. It carries the triangle vertices themselves, not merely a centroid
and an area, because the closed-form solid angles that the gather is built on integrate over
the triangle rather than approximating it by a point — the near-field error of that
approximation runs to hundreds of percent at the distances a reactor annulus puts its
receivers at.

Three of its fields are the quantities a design study varies — emission, radiant power and
reflectance — and they are ordinary array leaves so that a derivative with respect to any of
them flows straight through. The geometry is a leaf too: marking it static would put whole
arrays in the compilation cache key, which equinox warns about and which fails outright on
the second call with a different array.

**Emission and power are separate fields, and zero area is legal.** A point source is
represented as a zero-area facet carrying radiant power in watts; an areal facet carries
emission in watts per square metre and no power. Guarding against a zero-area triangle would
therefore delete the point sources, which is why :meth:`Surfaces.from_triangles` does not.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.vectors import dot

__all__ = ["Surfaces"]


class Surfaces(eqx.Module):
    """Triangular emitting and reflecting facets with their per-facet optical properties.

    Build one with :meth:`from_triangles`, which derives the geometry from the vertices so the
    centroid, normal and area cannot disagree with them. Construct the class directly only when
    every field is already known to be consistent.

    Attributes
    ----------
    vertices : jnp.ndarray, shape ``(n_facets, 3, 3)``
        Triangle vertices; the second axis indexes the three corners.
    centroid : jnp.ndarray, shape ``(n_facets, 3)``
        Vertex mean — the receiver point used when a facet acts as a receiving surface.
    normal : jnp.ndarray, shape ``(n_facets, 3)``
        Unit outward normal from the vertex winding by the right-hand rule. **Exactly zero on a
        zero-area facet**, which has no orientation to report.
    area : jnp.ndarray, shape ``(n_facets,)``
        Triangle area. Zero marks a point source and is legal.
    solid_id : jnp.ndarray of int, shape ``(n_facets,)``
        Which named body in the source file the facet came from, indexing :attr:`solid_names`.
    emission : jnp.ndarray, shape ``(n_facets,)``
        Prescribed emission ``M`` in W/m², zero on a point source. Differentiable.
    power : jnp.ndarray, shape ``(n_facets,)``
        Radiant power ``P`` in W, zero on an areal facet. Differentiable.
    reflectance : jnp.ndarray, shape ``(n_facets,)``
        Diffuse reflectance ``rho`` in ``[0, 1]``. Differentiable.
    solid_names : tuple of str
        Body names in :attr:`solid_id` order (static metadata, not a leaf).
    """

    vertices: jnp.ndarray
    centroid: jnp.ndarray
    normal: jnp.ndarray
    area: jnp.ndarray
    solid_id: jnp.ndarray
    emission: jnp.ndarray
    power: jnp.ndarray
    reflectance: jnp.ndarray
    solid_names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def from_triangles(
        cls,
        vertices,
        *,
        solid_id=None,
        solid_names=("surface",),
        emission=0.0,
        power=0.0,
        reflectance=0.0,
    ) -> Surfaces:
        """Build a surface set from triangle vertices, deriving all of its geometry.

        Parameters
        ----------
        vertices : array_like, shape ``(n_facets, 3, 3)``
            Triangle vertices.
        solid_id : array_like of int, shape ``(n_facets,)``, optional
            Body index per facet. Defaults to all zeros — one unnamed body.
        solid_names : tuple of str, optional
            Body names in ``solid_id`` order.
        emission, power, reflectance : float or array_like, shape ``(n_facets,)``, optional
            Per-facet optical properties; a scalar is broadcast to every facet.

        Returns
        -------
        Surfaces
        """
        vertices = jnp.asarray(vertices, dtype=float)
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3):
            msg = f"vertices must have shape (n_facets, 3, 3); got {vertices.shape}"
            raise ValueError(msg)
        n_facets = vertices.shape[0]

        edge_a = vertices[:, 1] - vertices[:, 0]
        edge_b = vertices[:, 2] - vertices[:, 0]
        twice_vector_area = jnp.cross(edge_a, edge_b)
        twice_area = jnp.sqrt(dot(twice_vector_area, twice_vector_area))
        # A zero-area facet is a point source, not an error, so it keeps a zero normal rather
        # than a NaN one; guarding the division away entirely would delete the point sources.
        degenerate = twice_area == 0.0
        normal = twice_vector_area / jnp.where(degenerate, 1.0, twice_area)[:, None]

        def spread(value, name):
            spread_value = jnp.broadcast_to(jnp.asarray(value, dtype=float), (n_facets,))
            if spread_value.shape != (n_facets,):  # pragma: no cover - broadcast_to raises first
                msg = f"{name} must be a scalar or have shape ({n_facets},)"
                raise ValueError(msg)
            return spread_value

        if solid_id is None:
            solid_id = jnp.zeros(n_facets, dtype=jnp.int32)
        else:
            solid_id = jnp.asarray(solid_id, dtype=jnp.int32)
            if solid_id.shape != (n_facets,):
                msg = f"solid_id must have shape ({n_facets},); got {solid_id.shape}"
                raise ValueError(msg)
            if int(jnp.max(solid_id, initial=-1)) >= len(solid_names):
                msg = (
                    f"solid_id indexes body {int(jnp.max(solid_id))} but only "
                    f"{len(solid_names)} name(s) were given"
                )
                raise ValueError(msg)

        return cls(
            vertices=vertices,
            centroid=jnp.mean(vertices, axis=1),
            normal=normal,
            area=0.5 * twice_area,
            solid_id=solid_id,
            emission=spread(emission, "emission"),
            power=spread(power, "power"),
            reflectance=spread(reflectance, "reflectance"),
            solid_names=tuple(solid_names),
        )

    @property
    def n_facets(self) -> int:
        """Number of facets in the set."""
        return int(self.vertices.shape[0])

    def per_facet(self, by_solid: dict[str, float], default: float | None = None):
        """Expand a per-body mapping to a per-facet array.

        Optical properties are set per named body in a surface file — a lamp sleeve, a wall, a
        baffle — while every array here is per facet. This does the expansion in one place, so
        the several properties that need it cannot each grow their own loop over bodies.

        Parameters
        ----------
        by_solid : dict of str to float
            Value per body name.
        default : float, optional
            Value for bodies the mapping omits. Without one, an omission raises — the common
            mistake is a misspelled body name, which would otherwise silently emit nothing.

        Returns
        -------
        jnp.ndarray, shape ``(n_facets,)``

        Raises
        ------
        KeyError
            If a body has no entry and no ``default`` was given, or the mapping names a body
            the surface set does not contain.
        """
        unknown = set(by_solid) - set(self.solid_names)
        if unknown:
            msg = f"no such body in this surface set: {sorted(unknown)}; have {list(self.solid_names)}"
            raise KeyError(msg)
        if default is None:
            missing = [name for name in self.solid_names if name not in by_solid]
            if missing:
                msg = f"no value given for body {missing}; pass a default to allow omissions"
                raise KeyError(msg)
        values = np.array([by_solid.get(name, default) for name in self.solid_names], dtype=float)
        return jnp.asarray(values)[self.solid_id]

    def with_optics(self, *, emission=None, power=None, reflectance=None) -> Surfaces:
        """A copy carrying different optical properties and the same geometry.

        The geometry is what a build freezes and the optics are what a study varies, so
        replacing the latter is the common edit. Substituting only the fields given keeps the
        derived geometry consistent by never touching it.
        """
        replacements = {
            "emission": emission,
            "power": power,
            "reflectance": reflectance,
        }
        updated = self
        for name, value in replacements.items():
            if value is None:
                continue
            spread = jnp.broadcast_to(jnp.asarray(value, dtype=float), (self.n_facets,))
            updated = eqx.tree_at(lambda s, n=name: getattr(s, n), updated, spread)
        return updated
