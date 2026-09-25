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

**Which facets are point sources is recorded explicitly, not inferred from the area.** The two
say the same thing when a set is first built, and they must not be conflated all the same: the
area is a number the solver may differentiate through, while the kind is a *label* that decides
which code path a facet takes and therefore has to be known before anything is traced. Deriving
the label from the number ties them together, and the knot shows up the first time someone
differentiates with respect to vertex positions -- moving a lamp -- at which point the area
becomes a traced quantity and the label becomes unavailable. Keeping them apart costs one
static tuple and leaves the geometry free to move.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.profiles import Lambertian, Profile
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
    profiles : tuple of Profile
        The distinct angular distributions present in the set. A **collection** rather than one
        profile for the whole set, because each facet emits with its own: a lamp sleeve and a
        reflector in the same geometry are different sources, and the transfer of *emitted*
        light needs each facet's own distribution even though every *reflected* ray leaves
        Lambertian. The host-side partition this induces is also what lets the gather resolve
        the distribution once per kind at trace time rather than branching per facet.
    profile_index : numpy.ndarray of int, shape ``(n_facets,)``
        Which entry of :attr:`profiles` each facet emits with. A **numpy** array rather than a
        JAX one because it is a label that decides the shape of the traced program, like
        :attr:`point_source_index`: inside a trace every ``jnp`` operation is staged even on a
        concrete input, so a JAX array built there would arrive at the gather as a tracer and
        could not be partitioned on.
    point_source_index : tuple of int
        Which facets are point sources rather than emitting surfaces — static metadata, not a
        leaf, because it selects a code path. Stored as the indices rather than a mask so that
        the usual case, a handful of lamps among many facets, costs almost nothing to carry.
    """

    vertices: jnp.ndarray
    centroid: jnp.ndarray
    normal: jnp.ndarray
    area: jnp.ndarray
    solid_id: jnp.ndarray
    emission: jnp.ndarray
    power: jnp.ndarray
    reflectance: jnp.ndarray
    profile_index: np.ndarray
    solid_names: tuple[str, ...] = eqx.field(static=True)
    profiles: tuple[Profile, ...] = ()
    point_source_index: tuple[int, ...] = eqx.field(static=True, default=())

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
        profiles=None,
        profile_index=None,
        point_sources=None,
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
        profiles : tuple of Profile, optional
            The distinct angular distributions present. Defaults to a single
            :class:`~aquaflux.radiation.profiles.Lambertian`, which is what a diffuse surface
            emits with and what every reflected ray leaves by.
        profile_index : array_like of int, shape ``(n_facets,)``, optional
            Which profile each facet uses. Defaults to all zeros.
        point_sources : sequence of int, optional
            Indices of the facets that are point sources. Defaults to every facet whose
            triangle has no area, which is what they are built as — pass it explicitly only to
            record a different set, never to work around a geometry that came out degenerate.

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

        def in_range(indices, limit, name, what):
            """Check an index array against a table size, when the values are available.

            Skipped for a traced array rather than forced concrete: the same surface set gets
            rebuilt inside a traced function whenever its vertices move, and a validation that
            reads a value would turn a range check into a tracer leak. The indices do not
            change when geometry does, so the check has already run on the way in.

            ⚠️ The comparison is done in numpy and **not** with ``jnp``. Inside a trace, every
            ``jnp`` operation is staged out whether or not its inputs are concrete, so
            ``int(jnp.max(concrete_array))`` raises there while ``int(np.max(...))`` does not.
            A build-time check written with ``jnp`` works perfectly until the first time the
            object is rebuilt inside a traced function.
            """
            if isinstance(indices, jax.core.Tracer):
                return
            largest = int(np.max(np.asarray(indices), initial=-1))
            if largest >= limit:
                msg = f"{name} selects {what} {largest} but only {limit} were given"
                raise ValueError(msg)

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
            in_range(solid_id, len(solid_names), "solid_id", "body")

        if profiles is None:
            profiles = (Lambertian(),)
        profiles = tuple(profiles)
        if not profiles:
            msg = "profiles must contain at least one Profile"
            raise ValueError(msg)
        if profile_index is None:
            profile_index = np.zeros(n_facets, dtype=np.int32)
        elif not isinstance(profile_index, jax.core.Tracer):
            # Left traced when it is: the gather then refuses it with the reason, where a numpy
            # conversion here would fail with no explanation.
            profile_index = np.asarray(profile_index, dtype=np.int32)
        if profile_index.shape != (n_facets,):
            msg = f"profile_index must have shape ({n_facets},); got {profile_index.shape}"
            raise ValueError(msg)
        in_range(profile_index, len(profiles), "profile_index", "profile")

        if point_sources is None:
            if isinstance(twice_area, jax.core.Tracer):
                msg = (
                    "point_sources must be given when the vertices are traced: which facets "
                    "are point sources is a label that decides a code path, so it cannot be "
                    "read off a traced area. Use Surfaces.with_geometry to move a set, which "
                    "carries the labels across."
                )
                raise TypeError(msg)
            point_sources = np.flatnonzero(np.asarray(twice_area) == 0.0)
        point_sources = tuple(int(index) for index in point_sources)
        out_of_range = [index for index in point_sources if not 0 <= index < n_facets]
        if out_of_range:
            msg = f"point_sources indexes facets outside the set: {out_of_range[:8]}"
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
            profile_index=profile_index,
            solid_names=tuple(solid_names),
            profiles=profiles,
            point_source_index=point_sources,
        )

    @property
    def n_facets(self) -> int:
        """Number of facets in the set."""
        return int(self.vertices.shape[0])

    @property
    def is_point_source(self) -> np.ndarray:
        """Boolean mask of the point sources, shape ``(n_facets,)``.

        Built from :attr:`point_source_index` rather than from the area, and returned as a
        plain array because it is known before anything is traced. Everything that has to
        distinguish the two kinds of source reads this, so there is one answer to the question.
        """
        mask = np.zeros(self.n_facets, dtype=bool)
        mask[list(self.point_source_index)] = True
        return mask

    def with_geometry(self, vertices) -> Surfaces:
        """A copy on moved vertices, with the derived geometry recomputed and the labels kept.

        This is how a lamp moves. The centroid, normal and area all follow the vertices and are
        recomputed here rather than carried over, because substituting the vertices alone would
        leave the three of them describing the old shape — silently, since nothing downstream
        can tell. The optical properties and the point-source labels are preserved, so the
        result is the same surface set in a new position.

        The vertices may be traced, which is the point: it is what lets a gradient reach a
        source's position.
        """
        vertices = jnp.asarray(vertices, dtype=float)
        if vertices.shape[0] != self.n_facets:
            msg = (
                f"expected {self.n_facets} triangles to move, got {vertices.shape[0]}; "
                "with_geometry moves a surface set, it does not rebuild one"
            )
            raise ValueError(msg)
        return Surfaces.from_triangles(
            vertices,
            solid_id=self.solid_id,
            solid_names=self.solid_names,
            emission=self.emission,
            power=self.power,
            reflectance=self.reflectance,
            profiles=self.profiles,
            profile_index=self.profile_index,
            point_sources=self.point_source_index,
        )

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

    def area_by_solid(self) -> dict[str, float]:
        """Total facet area of each named body, in the order of :attr:`solid_names`.

        The counterpart of :meth:`per_facet`, which goes the other way. A body's area is the
        sum of its triangles and **not** whatever closed form the shape was drawn from: an
        inscribed triangulation of a cylinder undershoots ``pi d L`` by 2.5% at eight sectors
        and 0.16% at thirty-two, so a rated power divided by the analytic area radiates
        measurably less than the rating once it reaches these facets. Dividing by this instead
        makes the two agree exactly at any refinement.

        A body of point sources totals zero, which is not an error — it has no area for an
        exitance to be defined on, and carries radiant power instead.

        Returns
        -------
        dict of str to float
        """
        totals = np.bincount(
            np.asarray(self.solid_id),
            weights=np.asarray(self.area),
            minlength=len(self.solid_names),
        )
        return {name: float(total) for name, total in zip(self.solid_names, totals, strict=True)}

    def with_optics(
        self, *, emission=None, power=None, reflectance=None, profiles=None, profile_index=None
    ) -> Surfaces:
        """A copy carrying different optical properties and the same geometry.

        The geometry is what a build freezes and the optics are what a study varies, so
        replacing the latter is the common edit. Substituting only the fields given keeps the
        derived geometry consistent by never touching it.

        Parameters
        ----------
        emission, power, reflectance : array_like, optional
            Per-facet values, broadcast to ``(n_facets,)``.
        profiles : tuple of Profile, optional
            A new catalogue of angular distributions. Supplying one without ``profile_index``
            is only meaningful when it holds a single profile, which is then given to every
            facet — that is how a set is re-read as purely Lambertian, which is what the
            *reflected* part of a radiosity solution leaves with whatever the source emitted
            like.
        profile_index : array_like of int, optional
            Which entry of ``profiles`` each facet uses.

        Raises
        ------
        ValueError
            If ``profiles`` is given without ``profile_index`` and holds more than one entry,
            leaving the existing indices pointing into a catalogue that has changed under them.
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
        if profiles is None and profile_index is None:
            return updated
        if profiles is not None and profile_index is None:
            if len(profiles) != 1:
                msg = (
                    f"profiles has {len(profiles)} entries and no profile_index was given; the "
                    "existing indices would point into a different catalogue. Pass both."
                )
                raise ValueError(msg)
            profile_index = np.zeros(self.n_facets, dtype=int)
        index = np.asarray(profile_index, dtype=int)
        catalogue = self.profiles if profiles is None else tuple(profiles)
        if index.shape != (self.n_facets,):
            msg = f"profile_index must be ({self.n_facets},); got {index.shape}"
            raise ValueError(msg)
        if len(catalogue) and (index.min() < 0 or index.max() >= len(catalogue)):
            msg = (
                f"profile_index runs from {index.min()} to {index.max()}, outside the "
                f"{len(catalogue)} profiles given"
            )
            raise ValueError(msg)
        # Replacing the catalogue changes the pytree's structure when its length changes, which
        # is past what tree_at substitutes, so the whole record is rebuilt instead. Every other
        # field is carried across by reference, so the geometry is shared rather than recomputed.
        return dataclasses.replace(
            updated, profiles=catalogue, profile_index=index.astype(np.int32, copy=False)
        )
