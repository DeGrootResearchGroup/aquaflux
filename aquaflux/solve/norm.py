"""Residual norms for the Newton convergence test and forward globalization.

A single scalar summary of a residual vector drives three decisions in a nonlinear solve: the
stopping test (``||R|| <= atol + rtol ||R0||``), the switched-evolution-relaxation shift
(``beta = beta0 (||R||/||R0||)^p``), and the line-search / divergence acceptance. The plain
Euclidean norm is the default and is correct when every degree of freedom is on a comparable scale.

It is the **wrong** measure for a strongly heterogeneous block system -- e.g. a coupled RANS state
whose ``omega`` residual is O(1e5) while its ``k`` residual is O(1e-3). The Euclidean norm is then
almost entirely ``omega``, so the line search cannot see -- and therefore cannot protect -- the ``k``
block: a step that lets ``k`` blow up or collapse is accepted (the ``omega``-dominated norm barely
moves), while a step that would reduce ``k`` is vetoed because ``omega`` ticked up. Both the stopping
test and the globalization then judge only one field.

:class:`BlockScaledNorm` is the fix: it scales each contiguous block by its own reference magnitude
before combining, so every block contributes comparably and the measure judges the whole system.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np

# A residual norm maps a flat residual vector to a non-negative scalar. The default everywhere is the
# plain Euclidean norm; a heterogeneous block system injects :class:`BlockScaledNorm` instead.
ResidualNorm = Callable[[jnp.ndarray], jnp.ndarray]


class BlockScaledNorm(eqx.Module):
    """A residual norm that scales each contiguous block by its own reference magnitude.

    Splits the flat residual into blocks of the given ``sizes`` (in order), divides each block's
    Euclidean norm by its reference ``scale``, and returns the Euclidean norm of those per-block
    relative residuals,

        ``||R|| = sqrt( sum_b ( ||R_b|| / scale_b )^2 )``.

    With a single block whose ``scale`` is its own ``||R0||`` this is the plain relative residual;
    with several disparate-scale blocks it prevents the largest-magnitude block from dominating, so
    the forward march's stopping test and globalization judge **every** block rather than only the
    one with the largest residual. It is used only on the forward path (the convergence test and the
    pseudo-transient / line-search decisions); the implicit-function-theorem adjoint never forms a
    residual norm, so the choice of norm does not touch the gradient.

    Attributes
    ----------
    sizes : tuple of int
        Length of each contiguous block, in order; must sum to the residual length (static).
    scales : tuple of float
        The positive per-block reference magnitude each block's norm is divided by (static);
        typically the block's initial residual norm ``||R0_block||``.
    """

    sizes: tuple[int, ...] = eqx.field(static=True)
    scales: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The block-scaled Euclidean norm of ``residual`` (shape ``(sum(sizes),)``)."""
        split_points = tuple(int(p) for p in np.cumsum(self.sizes)[:-1])
        blocks = jnp.split(residual, split_points)
        relative = jnp.stack(
            [
                jnp.linalg.norm(block) / scale
                for block, scale in zip(blocks, self.scales, strict=True)
            ]
        )
        return jnp.linalg.norm(relative)


class RowScaledNorm(eqx.Module):
    """A residual measure that equilibrates each row, then normalizes by each field's own magnitude.

    Built in two stages, and it is the two together that make the result a *fractional change* rather
    than a raw magnitude:

    1. **Row equilibration.** Each cell's residual is divided by that row's own diagonal coefficient,
       ``|R_i| / a_i``. Since ``a_i`` is (to a Jacobi approximation) the derivative of the row with
       respect to its own unknown, the quotient is *how far that cell's value would move if its row
       were solved alone* — so it carries the units of the field, whatever units the equation was
       written in.
    2. **Field normalization.** Each block's mean is divided by that field's own mean magnitude,
       giving a dimensionless fractional change. A block whose residual is already dimensionless
       (a mass imbalance divided by a mass throughput, say) passes a scale of ``1``.

    The per-cell values are combined with an **L1 mean**, not a root-mean-square, and that is
    load-bearing rather than stylistic. A squared measure is dominated by its largest entries, and on
    a converged turbulent field the residual concentrates into a handful of cells with the sharpest
    near-wall gradients — so a root-mean-square keeps reporting those few cells while the field as a
    whole converges. Measured on a separating-flow benchmark: under a root-mean-square, one cell
    carried up to 72 % of a block's total and the measure ranked a badly wrong state *better* than a
    converged one; under the L1 mean the worst cell carried a few per cent and the ranking matched the
    physics. Blocks are then combined in the Euclidean sense, which is safe because they are only a
    handful of comparable, already-dimensionless numbers.

    **The scales are rebuilt every outer iteration, so they are traced values over a static shape.**
    Only :attr:`sizes` is static -- it sets where the blocks are split. The two scale arrays are
    ordinary leaves, which is what makes re-deriving the measure at a new state a **compilation cache
    hit** rather than a fresh compile: the block structure is unchanged, only the numbers move. (The
    frozen multigrid hierarchies are laid out the same way, and for the same reason.) Baking the
    scales in as constants instead would make a per-iteration rebuild cost a recompile per step, which
    is far more than the measure is worth.

    **They must be held FIXED across a line search, though.** The scales depend on the state, so
    re-deriving them per trial step would let a candidate be preferred for shrinking its own
    denominator rather than its residual -- the search would no longer be comparing like with like.
    Build once per outer iteration, use for every trial step within it.

    Attributes
    ----------
    sizes : tuple of int
        Length of each contiguous block, in order; must sum to the residual length (static).
    row_scale : jnp.ndarray
        The per-row divisor, shape ``(sum(sizes),)``, strictly positive. Each entry is its row's own
        diagonal coefficient -- for a row written directly in the solved variable (a value fixation in
        log form, say) that derivative is one, so such rows pass through unscaled.
    field_scale : jnp.ndarray
        The per-block divisor for stage 2, shape ``(len(sizes),)``, strictly positive; the field's mean
        absolute magnitude, or ``1`` for a block already dimensionless after stage 1.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> norm = RowScaledNorm(sizes=(2, 2), row_scale=jnp.array([2.0, 2.0, 1.0, 1.0]),
    ...                      field_scale=jnp.array([1.0, 4.0]))
    >>> # block 0: mean(|[1,3]|/2)/1 = 1.0 ; block 1: mean(|[2,6]|/1)/4 = 1.0
    >>> float(norm(jnp.array([1.0, 3.0, 2.0, 6.0])))
    1.4142135623730951
    """

    sizes: tuple[int, ...] = eqx.field(static=True)
    row_scale: jnp.ndarray
    field_scale: jnp.ndarray

    def __call__(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The row-equilibrated, field-normalized measure of ``residual``.

        Parameters
        ----------
        residual : jnp.ndarray
            The flat residual, shape ``(sum(sizes),)``.

        Returns
        -------
        jnp.ndarray
            A non-negative scalar: the Euclidean combination of the per-block mean fractional changes.
        """
        equilibrated = jnp.abs(residual) / self.row_scale
        split_points = tuple(int(p) for p in np.cumsum(self.sizes)[:-1])
        blocks = jnp.split(equilibrated, split_points)
        fractional = jnp.stack(
            [jnp.mean(block) / scale for block, scale in zip(blocks, self.field_scale, strict=True)]
        )
        return jnp.linalg.norm(fractional)

    def per_block(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The per-block fractional changes, shape ``(len(sizes),)`` -- the reporting view.

        The solver steers on the single scalar :meth:`__call__` returns; this exposes the individual
        equations' convergence, which is what a user reads to see *which* equation is limiting and
        what a march reports per step.
        """
        equilibrated = jnp.abs(residual) / self.row_scale
        split_points = tuple(int(p) for p in np.cumsum(self.sizes)[:-1])
        return jnp.stack(
            [
                jnp.mean(block) / self.field_scale[i]
                for i, block in enumerate(jnp.split(equilibrated, split_points))
            ]
        )
