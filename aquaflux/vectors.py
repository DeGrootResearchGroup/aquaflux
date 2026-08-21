"""Vector algebra on fields of vectors: per-element dot products, magnitudes, and scaling.

Finite-volume kernels work with *fields* of small spatial vectors — one ``(dim,)`` vector per
face or per cell, stored as a ``(..., dim)`` array whose leading axes index the faces/cells and
whose last axis holds the spatial components. The elementary operations on such a field — the
per-element dot product ``a·b``, the squared magnitude ``|a|²``, and scaling each vector by a
per-element scalar — recur throughout the geometry, the reconstruction schemes, and the flux
operators. Spelled out each time as ``jnp.sum(a * b, axis=-1)`` or ``s[..., None] * v`` they bury
the intent under axis and broadcasting bookkeeping, so they are defined once here and imported
wherever a vector field is contracted or scaled. Because they are written here and nowhere else,
*how* each is spelled is decided once rather than at every call site — :func:`dot` in particular is
deliberately not the reduction a reader would reach for first, for the reason its own docstring
gives.

Every function treats the **last axis** as the spatial component and broadcasts over any leading
batch axes, so each applies unchanged to a single vector ``(dim,)``, a per-face field
``(n_faces, dim)``, or any higher-rank batch ``(..., dim)``.
"""

from __future__ import annotations

import functools
import operator

import jax.numpy as jnp


def dot(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Per-element dot product ``a·b`` contracted over the last (spatial) axis.

    The spatial components are multiplied together as one array and then summed **explicitly**,
    component by component, rather than by a reduction over the last axis. The two spell the same
    contraction, but they compile differently on the CPU backend: a fusion rooted at a reduction is
    emitted as a custom kernel that runs on a single thread, whereas the explicit sum stays a plain
    elementwise fusion, which the backend partitions across cores like the neighbouring kernels. As
    a dot product over a two- or three-component axis is memory-bound, running it on one thread
    while the operators around it run on several is what made it worth spelling out. The components
    are summed left to right, in index order; the reduction it replaces is free to associate the
    same terms differently, and to contract a multiply and an add into one fused operation. That
    freedom, rather than any change of formula, is why the two agree to within a rounding of the
    summed terms and not bit for bit — a difference that is unbounded *relative* to the result when
    the contraction nearly cancels, so compare against the magnitudes summed, not against the answer.

    Multiplying first and unrolling only the sum keeps the broadcasting of the reduction it
    replaces: an operand carrying a trailing axis of length one against the other's ``dim`` is
    broadcast by the product, so the components are indexed out of the already-broadcast result.
    The product is formed with :func:`jax.numpy.multiply` rather than ``*`` so that a pair of NumPy
    operands still yields a JAX array, as contracting them with a reduction did.

    Parameters
    ----------
    a, b : jnp.ndarray
        Vector fields of shape ``(..., dim)``, broadcast against each other.

    Returns
    -------
    jnp.ndarray
        The contracted product, shape ``(...)`` — one scalar per leading index.
    """
    product = jnp.multiply(a, b)
    dim = product.shape[-1]
    if dim == 0:
        # An empty contraction sums no terms; the reduction this replaces returns zero.
        return jnp.zeros(product.shape[:-1], dtype=product.dtype)
    return functools.reduce(operator.add, (product[..., i] for i in range(dim)))


def norm_squared(a: jnp.ndarray) -> jnp.ndarray:
    """Squared magnitude ``|a|² = a·a`` over the last (spatial) axis.

    Cheaper than squaring :func:`jnp.linalg.norm` and avoids the non-differentiable ``sqrt`` at the
    origin — use it wherever only the squared length is needed (a distance weight, a positivity
    guard).

    Parameters
    ----------
    a : jnp.ndarray
        Vector field, shape ``(..., dim)``.

    Returns
    -------
    jnp.ndarray
        The squared magnitude, shape ``(...)``.
    """
    return dot(a, a)


def scale(vectors: jnp.ndarray, scalars: jnp.ndarray) -> jnp.ndarray:
    """Scale each vector by its own per-element scalar: ``scalars[..., None] * vectors``.

    The scalar field carries one weight per leading index; it is broadcast across the spatial
    component axis so every ``(dim,)`` vector is multiplied by the matching scalar. Reads at the
    call site as "scale ``vectors`` by ``scalars``".

    Parameters
    ----------
    vectors : jnp.ndarray
        Vector field, shape ``(..., dim)``.
    scalars : jnp.ndarray
        Per-element scalars, shape ``(...)`` — the shape of ``vectors`` without its trailing axis.

    Returns
    -------
    jnp.ndarray
        The scaled vector field, shape ``(..., dim)``.
    """
    return scalars[..., None] * vectors
