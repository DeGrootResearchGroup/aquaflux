"""How much of a source's light survives the water between it and a receiver.

Water absorbs ultraviolet light, and along a straight path the surviving fraction is
``exp(-tau)`` with the **optical depth** ``tau`` the absorption coefficient integrated over the
path. Everything in this module computes that one number for a segment; the gather multiplies
each source's contribution by it.

Two cases, and the difference is not one of accuracy alone:

:class:`UniformAbsorption` is ``tau = a r``, in closed form. It is the field-standard treatment
rather than a simplification — the ultraviolet-reactor literature attenuates analytically per
segment, stating the assumption of uniform optical properties outright — and it is exact for
the model it describes.

:class:`VoxelAbsorption` carries a coefficient that varies in space, sampled on a grid, and
integrates it along each segment by walking the grid cell by cell. **This is not ray
marching**, and the distinction is the reason the module is written this way. Marching takes
fixed steps and assumes the coefficient is constant across each one, which systematically
overestimates the surviving fraction — Jensen's inequality, and a bias that shrinks only as the
step does. Walking the actual cell boundaries finds every crossing exactly, so the only error
left is in the field's own representation. It also removes a convergence parameter: there is no
step size to choose, and the grid spacing is the single remaining axis.

Two further choices follow from wanting gradients as well as values. The field is **interpolated
trilinearly rather than sampled at the nearest cell**, because a nearest-neighbour lookup is
piecewise constant in position, so its derivative with respect to a source's position is zero
almost everywhere and undefined on the cell faces. And the integral within each cell is taken
with **Simpson's rule**, which is *exact* here: a trilinear field restricted to a straight line
is a cubic in the path parameter, and Simpson integrates cubics exactly. So the optical depth is
the exact integral of the interpolated field, not a quadrature of it.

Stochastic alternatives — delta tracking and its relatives — are deliberately absent. They are
unbiased in expectation but depend on a random stream, and the derivative of a null-collision
estimate is not an estimate of the derivative without further machinery.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from aquaflux.vectors import dot

__all__ = ["Absorption", "UniformAbsorption", "VoxelAbsorption"]


class Absorption(eqx.Module):
    """The optical depth along a straight segment through an absorbing medium."""

    @abc.abstractmethod
    def optical_depth(self, origin: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Integrate the absorption coefficient from ``origin`` to ``target``.

        Parameters
        ----------
        origin, target : jnp.ndarray
            Segment endpoints, shape ``(..., 3)``, broadcast against each other.

        Returns
        -------
        jnp.ndarray
            Optical depth, shape ``(...)``, dimensionless and non-negative. The surviving
            fraction is its negative exponential.
        """


class UniformAbsorption(Absorption):
    """A single absorption coefficient everywhere: ``tau = a r``.

    Attributes
    ----------
    coefficient : jnp.ndarray
        Absorption coefficient ``a`` in inverse metres, a differentiable scalar leaf. Water
        transmittance is usually quoted as a percentage over ten millimetres, which converts as
        ``a = -ln(T10) / 0.01``; 70% per centimetre is about 35.7 per metre.
    """

    coefficient: jnp.ndarray

    def __init__(self, coefficient):
        self.coefficient = jnp.asarray(coefficient, dtype=float)

    def optical_depth(self, origin: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """``a`` times the straight-line distance."""
        offset = jnp.asarray(target, dtype=float) - jnp.asarray(origin, dtype=float)
        return self.coefficient * jnp.sqrt(dot(offset, offset))


class VoxelAbsorption(Absorption):
    """An absorption coefficient sampled on a regular grid, integrated exactly along a segment.

    The coefficient is stored at **cell centres**, so a grid of shape ``(nx, ny, nz)`` with the
    given spacing covers the box from ``origin`` to ``origin + spacing * (nx, ny, nz)``, and the
    sample ``[i, j, k]`` sits at ``origin + spacing * (i + 0.5, j + 0.5, k + 0.5)``.

    ⚠️ **The grid does not have to match the flow mesh, and usually should not.** Absorbance
    varies far more smoothly than velocity, and the cost of a segment is set by how many cells it
    crosses: the traversal runs a fixed ``nx + ny + nz`` steps, because a traced loop cannot have
    a data-dependent length. A coarse absorbance grid over a fine flow mesh is both cheaper and
    no less accurate.

    Attributes
    ----------
    coefficient : jnp.ndarray, shape ``(nx, ny, nz)``
        Absorption coefficient per cell, in inverse metres. A differentiable leaf: a gradient
        with respect to it is a sensitivity to the water quality, cell by cell.
    origin : jnp.ndarray, shape ``(3,)``
        Lower corner of the grid.
    spacing : jnp.ndarray, shape ``(3,)``
        Cell size along each axis.
    """

    coefficient: jnp.ndarray
    origin: jnp.ndarray
    spacing: jnp.ndarray

    def __init__(self, coefficient, origin, spacing):
        self.coefficient = jnp.asarray(coefficient, dtype=float)
        self.origin = jnp.broadcast_to(jnp.asarray(origin, dtype=float), (3,))
        self.spacing = jnp.broadcast_to(jnp.asarray(spacing, dtype=float), (3,))
        if self.coefficient.ndim != 3:
            msg = (
                f"coefficient must be a three-dimensional grid; got shape {self.coefficient.shape}"
            )
            raise ValueError(msg)

    @property
    def dims(self) -> tuple[int, int, int]:
        """Cell counts along each axis."""
        return tuple(int(n) for n in self.coefficient.shape)

    @property
    def max_crossings(self) -> int:
        """Most pieces a straight segment is cut into — the traversal's fixed step count.

        The cuts are the planes through the sample points, of which there are ``nx`` on the x
        axis and likewise for the others, so a segment meets at most ``nx + ny + nz`` of them.
        The bound is computed here on the host, from the shape alone, which is what makes a
        traced loop over it legal: it carries no dependence on where any particular ray goes.
        """
        nx, ny, nz = self.dims
        return nx + ny + nz + 1

    @property
    def _lattice_origin(self) -> jnp.ndarray:
        """Where the traversal's cuts start — the first sample plane, not the grid's corner.

        ⚠️ **The traversal must be cut on the SAMPLE planes, not on the cell faces**, and the
        two are half a cell apart. The interpolated field is a separate cubic between each pair
        of neighbouring samples; a piece that straddles a sample plane straddles a kink, and
        Simpson's rule integrates a kink no better than any other rule. Cutting on the cell
        faces instead is wrong by a fraction of a percent on a smooth field -- small enough to
        read as discretization error and large enough to matter, which is the worst size for a
        mistake to be. Measured on a linear field, where the answer is known in closed form:
        0.43% wrong on the faces, exact on the sample planes.
        """
        return self.origin + 0.5 * self.spacing

    def sample(self, position: jnp.ndarray) -> jnp.ndarray:
        """Trilinearly interpolate the coefficient at a position, clamped to the grid.

        Clamped rather than zero-padded: a position outside the grid takes the value of the
        nearest boundary cell, so a segment that strays just past the edge attenuates like the
        water at the edge rather than like vacuum. Interpolating rather than picking the nearest
        cell is what makes the result differentiable in position — a nearest-cell lookup is
        piecewise constant, so its derivative is zero almost everywhere and undefined on the
        faces between.
        """
        nx, ny, nz = self.dims
        counts = jnp.asarray([nx, ny, nz], dtype=float)
        # Samples sit at cell centres, hence the half-cell shift into interpolation coordinates.
        grid = (jnp.asarray(position, dtype=float) - self.origin) / self.spacing - 0.5
        grid = jnp.clip(grid, 0.0, counts - 1.0)
        lower = jnp.floor(grid)
        fraction = grid - lower
        lower = jnp.clip(lower, 0.0, counts - 2.0).astype(jnp.int32)
        fraction = jnp.clip(grid - lower, 0.0, 1.0)

        total = 0.0
        for corner in range(8):
            offsets = jnp.asarray([(corner >> axis) & 1 for axis in range(3)], dtype=jnp.int32)
            weight = jnp.prod(jnp.where(offsets == 1, fraction, 1.0 - fraction), axis=-1)
            index = lower + offsets
            total = total + weight * self.coefficient[index[..., 0], index[..., 1], index[..., 2]]
        return total

    def optical_depth(self, origin: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Walk the grid from ``origin`` to ``target``, integrating exactly cell by cell.

        The segment is parameterized on ``[0, 1]``. Each step advances to the nearer of the next
        cell boundary and the segment's end, and adds Simpson's rule over that piece — exact,
        because the interpolated field along a straight line is a cubic. Steps past the end
        contribute nothing, so the loop runs its fixed length without affecting the answer.
        """
        origin = jnp.asarray(origin, dtype=float)
        target = jnp.asarray(target, dtype=float)
        origin, target = jnp.broadcast_arrays(origin, target)
        offset = target - origin
        length = jnp.sqrt(dot(offset, offset))

        # In cell units, the parameter distance between successive face crossings on each axis.
        step_in_cells = offset / self.spacing
        moving = jnp.abs(step_in_cells) > 0.0
        crossing_interval = jnp.where(
            moving, 1.0 / jnp.where(moving, jnp.abs(step_in_cells), 1.0), jnp.inf
        )

        start_in_cells = (origin - self._lattice_origin) / self.spacing
        counts = jnp.asarray(self.dims, dtype=float)
        forward = step_in_cells > 0.0

        # The cuts are the planes through the sample points, indexed 0 to n-1 on each axis, and
        # ⚠️ ONLY THOSE. Beyond the outermost sample the interpolated field is clamped and has
        # no further kinks, so a cut out there would spend a step of the fixed budget on a piece
        # that needs not exist. Both ends of the range matter: a segment starting far outside
        # the grid meets its first *real* plane a long way in, and generating the imaginary
        # planes before it exhausts the budget and silently drops the segment's tail.
        next_index = jnp.where(forward, jnp.ceil(start_in_cells), jnp.floor(start_in_cells))
        on_a_plane = next_index == start_in_cells
        next_index = jnp.where(on_a_plane, next_index + jnp.where(forward, 1.0, -1.0), next_index)
        # Skip forward to the first plane that exists, in the direction of travel.
        next_index = jnp.where(
            forward, jnp.maximum(next_index, 0.0), jnp.minimum(next_index, counts - 1.0)
        )
        outermost = jnp.where(forward, counts - 1.0, 0.0)
        ahead = jnp.where(forward, next_index <= outermost, next_index >= outermost)

        def parameter_at(index):
            return (index - start_in_cells) / jnp.where(moving, step_in_cells, 1.0)

        first_crossing = jnp.where(moving & ahead, parameter_at(next_index), jnp.inf)
        last_crossing = jnp.where(moving, parameter_at(outermost), -jnp.inf)

        def at(parameter):
            return self.sample(origin + offset * parameter[..., None])

        # ⚠️ The running total and the field at the piece's near end are CARRIED, not collected.
        # Returning each piece from the scan stacks all of them, one per step of the fixed budget,
        # before a single sum -- the walk's largest array by far -- and each piece's near end is
        # the previous one's far end, so sampling it again repeats a third of the lookups for the
        # same value. Each step is also checkpointed: a gradient then keeps the carry per step and
        # recomputes the lookups on the way back, rather than keeping every lookup's gather
        # indices and weights for all the steps at once.
        @jax.checkpoint
        def advance(carry, _):
            here, next_face, near, total = carry
            there = jnp.minimum(jnp.min(next_face, axis=-1), 1.0)
            there = jnp.maximum(there, here)
            far = at(there)
            weights = near + 4.0 * at(0.5 * (here + there)) + far
            total = total + (there - here) * length * weights / 6.0
            used = next_face <= there[..., None]
            next_face = jnp.where(used, next_face + crossing_interval, next_face)
            next_face = jnp.where(next_face > last_crossing + 1e-12, jnp.inf, next_face)
            return (there, next_face, far, total), None

        start = jnp.zeros_like(length)
        (_, _, _, total), _ = lax.scan(
            advance,
            (start, first_crossing, at(start), jnp.zeros_like(length)),
            None,
            length=self.max_crossings,
        )
        return total


def grid_covers(field: VoxelAbsorption, points) -> bool:
    """Whether every one of ``points`` lies inside the grid's box.

    A build-time check, in numpy. Outside the grid the coefficient is clamped to its boundary
    value, which is a reasonable thing to do for a segment that strays a little past the edge
    and a poor description of a reactor that does not fit in its own absorbance grid.
    """
    points = np.asarray(points, dtype=float)
    lower = np.asarray(field.origin)
    upper = lower + np.asarray(field.spacing) * np.asarray(field.dims)
    return bool(np.all(points >= lower) and np.all(points <= upper))
