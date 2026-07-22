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
