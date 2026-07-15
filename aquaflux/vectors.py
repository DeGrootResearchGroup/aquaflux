"""Vector algebra on fields of vectors: per-element dot products, magnitudes, and scaling.

Finite-volume kernels work with *fields* of small spatial vectors — one ``(dim,)`` vector per
face or per cell, stored as a ``(..., dim)`` array whose leading axes index the faces/cells and
whose last axis holds the spatial components. The elementary operations on such a field — the
per-element dot product ``a·b``, the squared magnitude ``|a|²``, and scaling each vector by a
per-element scalar — recur throughout the geometry, the reconstruction schemes, and the flux
operators. Spelled out each time as ``jnp.sum(a * b, axis=-1)`` or ``s[..., None] * v`` they bury
the intent under axis and broadcasting bookkeeping, so they are defined once here and imported
wherever a vector field is contracted or scaled.

Every function treats the **last axis** as the spatial component and broadcasts over any leading
batch axes, so each applies unchanged to a single vector ``(dim,)``, a per-face field
``(n_faces, dim)``, or any higher-rank batch ``(..., dim)``.
"""

from __future__ import annotations

import jax.numpy as jnp


def dot(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Per-element dot product ``a·b`` contracted over the last (spatial) axis.

    Parameters
    ----------
    a, b : jnp.ndarray
        Vector fields of matching shape ``(..., dim)``.

    Returns
    -------
    jnp.ndarray
        The contracted product, shape ``(...)`` — one scalar per leading index.
    """
    return jnp.sum(a * b, axis=-1)


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
