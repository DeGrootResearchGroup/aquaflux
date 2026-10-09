"""Slope limiters for bounded second-order reconstruction.

A limited linear reconstruction takes the face value ``phi_f = phi_C + psi_C grad phi_C .
(x_f - x_C)`` from the upwind cell ``C``. The limiter ``psi_C in [0, 1]`` throttles the
gradient term so the reconstructed face values stay within the range of the surrounding cell
values — ``psi = 1`` is full second-order, ``psi = 0`` collapses to first-order upwind. It is a
**per-cell** quantity: the minimum, over the cell's faces, of a per-face limiter function.

:class:`VenkatakrishnanLimiter` is the smooth limiter of Venkatakrishnan (1993). For each face
of cell ``i`` with unlimited increment ``d- = grad phi_i . (x_f - x_i)`` and available headroom
``d+`` (``phi_max - phi_i`` when ``d- > 0``, else ``phi_min - phi_i``, over the cell's stencil):

    psi_face = [ (d+^2 + eps^2) d- + 2 d-^2 d+ ] / [ (d+^2 + 2 d-^2 + d+ d- + eps^2) d- ],

and ``psi_i = min_face psi_face``. The softening ``eps`` switches limiting off where the field varies
by less than it -- in smooth regions, where the classic (non-smooth) min/max limiters would clip and
stall convergence, which is the whole reason to prefer this one.

**The softening is a fraction of the field's reference magnitude**: ``eps = K phi_ref``, with ``K``
dimensionless and ``phi_ref`` the size of the variations the field carries (an inlet speed for a
velocity, an inlet concentration for a species). ``eps^2`` is then in the units of ``d+^2``, which it is
added to, so ``K`` means the same thing in any system of units and on any mesh of the same problem:
changing the units of the field changes ``phi_ref`` with it, and refining the mesh changes neither.
This replaces Venkatakrishnan's mesh-length form ``eps^2 = (K dx)^3`` (``K^3`` times a cell volume),
which adds a volume to a squared field increment and so makes ``K`` depend on the mesh's absolute
size and the field's units. ``phi_ref`` is a fixed property of the problem rather than the
range of the current iterate: a range read from the state would change the residual as the solve
progresses, and its derivative would couple every cell to the two cells holding the extrema.

Unlike the classic implementation, which **freezes** ``psi`` at the previous iterate and adds
the limited correction as an explicit source, here ``psi(phi, grad phi)`` is written into the
residual. It is smooth in its arguments (the only non-smoothness is the ``min`` over faces and
the stencil ``min``/``max``, continuous with measure-zero kinks), so automatic differentiation
linearizes it and places it in the Jacobian.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import TYPE_CHECKING, ClassVar

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot

if TYPE_CHECKING:
    from collections.abc import Callable

    from aquaflux.context import FieldContext


def _optional_float_leaf(value: object) -> jnp.ndarray | None:
    """A reference scale as a floating array leaf, or ``None`` for unset."""
    return None if value is None else jnp.asarray(value, dtype=float)


class Limiter(eqx.Module):
    """Strategy interface: a per-cell slope limiter ``psi in [0, 1]``."""

    @abc.abstractmethod
    def limit(self, field: jnp.ndarray, context: FieldContext) -> jnp.ndarray:
        """Per-cell limiter values, shape ``(n_cells,)``.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        context : FieldContext
            The shared per-field context this field was reconstructed into: ``context.gradient`` is
            this field's own cell gradient, and ``context.mesh.face_cells`` / ``context.mesh.geometry``
            supply the connectivity and the face/cell metrics (face centroids; cell centroids and
            volumes). A limiter reads no boundary value and no property, so it is one of the
            strategies a plain :class:`~aquaflux.context.MeshContext` would also suffice for -- it
            takes the full :class:`~aquaflux.context.FieldContext` only because that is what its
            caller (:class:`~aquaflux.discretization.advection.LimitedUpwind`) already holds.
        """

    def with_reference_scale(self, scale: Callable[[], float]) -> Limiter:
        """This limiter for a field of reference magnitude ``scale()``; unchanged if it needs none.

        The assembler of an equation knows the magnitude of the field it solves for (the speed of a
        flow, the inlet value of a species); a limiter whose softening is a fraction of that
        magnitude takes it from there. ``scale`` is a function rather than a number so the assembler
        derives the magnitude only for a limiter that reads it -- deriving it can fail on a problem
        that states none, which must not refuse a limiter that never needed one.

        Parameters
        ----------
        scale : callable
            Returns the field's reference magnitude, in the field's own units.

        Returns
        -------
        Limiter
            This limiter, by default: a limiter with no softening needs no scale.
        """
        return self


class VenkatakrishnanLimiter(Limiter):
    """The smooth Venkatakrishnan (1993) limiter, softened by a fraction of the field's magnitude.

    Attributes
    ----------
    softening : float
        The dimensionless coefficient ``K`` in ``eps = K scale``: the size of a variation the limiter
        leaves alone, as a fraction of the field's reference magnitude. Larger ``K`` limits less
        (smoother, less bounded); ``K -> 0`` recovers a strict limiter. An ordinary pytree leaf: the
        limiter only does arithmetic with it, so it can be a differentiation target.
    scale : jnp.ndarray or None
        The field's reference magnitude ``phi_ref``, in the field's units, stored as a floating
        array leaf (so a new value is not a recompile). Unset, the assembler of
        the equation the limiter is used in supplies it (see :meth:`with_reference_scale`); a value
        given here is kept. Evaluated with no scale it raises: a softening of zero would be a strict
        limiter, which divides zero by zero on a uniform field.
    """

    softening: float = 0.05
    scale: jnp.ndarray | None = eqx.field(default=None, converter=_optional_float_leaf)

    #: The assembler supplies the scale, so a case file does not state it.
    not_settings: ClassVar[tuple[str, ...]] = ("scale",)

    def with_reference_scale(self, scale: Callable[[], float]) -> VenkatakrishnanLimiter:
        """This limiter with its reference magnitude set to ``scale()``, unless one is set already.

        Parameters
        ----------
        scale : callable
            Returns the field's reference magnitude.

        Returns
        -------
        VenkatakrishnanLimiter
            ``self`` when :attr:`scale` is set; otherwise a copy carrying ``scale()``.

        Raises
        ------
        ValueError
            If the magnitude is not positive and finite.
        """
        if self.scale is not None:
            return self
        magnitude = float(scale())
        if not magnitude > 0.0 or not math.isfinite(magnitude):
            raise ValueError(
                f"VenkatakrishnanLimiter: a reference scale must be positive and finite, got "
                f"{magnitude!r}."
            )
        return dataclasses.replace(self, scale=magnitude)

    def limit(self, field, context):
        if self.scale is None:
            raise ValueError(
                "VenkatakrishnanLimiter has no reference scale: its softening is a fraction of the "
                "field's magnitude. Build the equation through its assembler, which supplies the "
                "magnitude of the field it solves for, or give the limiter a scale."
            )
        face_cells = context.mesh.face_cells
        gradient = context.gradient
        face_geometry, cell_geometry = context.mesh.geometry.face, context.mesh.geometry.cell
        owner = face_cells.owner
        neighbour = face_cells.safe_neighbour
        phi = field

        # Stencil extrema over each cell and its interior-face neighbours: each cell takes the
        # extremum of the adjacent cells' values (owner ← neighbour's phi, neighbour ← owner's),
        # via the connectivity's max/min scatter, which excludes a boundary face's neighbour side
        # structurally.
        phi_max = jnp.maximum(phi, face_cells.scatter_max(phi[neighbour], phi[owner]))
        phi_min = jnp.minimum(phi, face_cells.scatter_min(phi[neighbour], phi[owner]))

        # eps = K phi_ref, so eps^2 is in the units of the squared headroom it is added to.
        eps2 = (self.softening * self.scale) ** 2
        # A vanishing increment is moved away from zero by an amount negligible against the field.
        tiny = 1e-12 * self.scale
        x_face = face_geometry.centroid

        def face_limiter(cell, x_cell):
            """Venkatakrishnan psi for each face as seen from ``cell`` at position ``x_cell``.

            ``x_cell`` is a gathered per-face position, not derived from ``cell`` by indexing:
            across a periodic seam the neighbour side's *position* is the neighbour's periodic
            image, not its raw centroid, while every field gather (``phi``, ``gradient``, ...) off
            ``cell`` stays unshifted, since field values are periodic and positions are not.
            """
            delta_minus = dot(gradient[cell], x_face - x_cell)
            # Regularize away from zero, treating zero as positive (sign of +1 at x == 0) so a
            # vanishing increment (constant field) gives psi -> 1 rather than 0/0.
            sign = jnp.where(delta_minus >= 0.0, 1.0, -1.0)
            delta_minus = sign * (jnp.abs(delta_minus) + tiny)
            headroom = jnp.where(
                delta_minus > 0.0, phi_max[cell] - phi[cell], phi_min[cell] - phi[cell]
            )
            numerator = (headroom**2 + eps2) * delta_minus + 2.0 * delta_minus**2 * headroom
            denominator = (
                headroom**2 + 2.0 * delta_minus**2 + headroom * delta_minus + eps2
            ) * delta_minus
            return numerator / denominator

        # Per-cell limiter = min over the cell's incident faces. Neither side needs masking: the
        # owner is a real cell on every face, and the connectivity's scatter_min already excludes a
        # boundary face's neighbour side from the reduction, whatever value face_limiter gives it
        # there (a boundary face reads safe_neighbour == owner, so its neighbour-side value is just
        # a harmless second copy of the owner-side one, discarded before it can matter).
        owner_limiter = face_limiter(owner, cell_geometry.centroid[owner])
        neighbour_limiter = face_limiter(
            neighbour, face_cells.neighbour_centroid(cell_geometry.centroid)
        )
        psi = face_cells.scatter_min(owner_limiter, neighbour_limiter)
        return jnp.clip(psi, 0.0, 1.0)
