"""Prescribed face-flux builders for verification cases (transport by a known velocity field).

These construct the injected inputs a scalar advection operator needs when the flow is *not*
being solved — a manufactured or analytic velocity field carrying a passive scalar. Production
runs never use them: a scalar transported by a solved flow consumes that flow's Rhie--Chow face
mass flux (the conservative flux that also closes continuity), never a velocity re-projected here.
"""

from __future__ import annotations

import jax.numpy as jnp
from aquaflux.mesh import FaceGeometry
from aquaflux.vectors import dot


def face_mass_flux(face_geometry: FaceGeometry, velocity: jnp.ndarray) -> jnp.ndarray:
    """Owner-outward face mass flux ``mdot_f = (u . n) A`` for a prescribed velocity field.

    Parameters
    ----------
    face_geometry : FaceGeometry
        Per-face area and owner-outward unit normal.
    velocity : jnp.ndarray
        Velocity, either a single ``(dim,)`` vector (uniform flow) or a per-face
        ``(n_faces, dim)`` field.

    Returns
    -------
    jnp.ndarray
        Owner-outward mass flux per face, shape ``(n_faces,)``.

    Notes
    -----
    A conservative advection operator needs a *discretely divergence-free* face-flux field
    (the per-cell owner-outward fluxes sum to zero). A uniform velocity satisfies this on any
    closed mesh; a general prescribed field does not, and would introduce a spurious source. This
    is why it is a verification helper rather than a solver operator: only analytically
    divergence-free fields (uniform flow, a stream function) are safe inputs. A scalar advected by
    a solved flow must instead reuse that flow's Rhie--Chow mass flux.
    """
    u = jnp.asarray(velocity)
    u_dot_n = dot(u, face_geometry.normal) if u.ndim == 2 else face_geometry.normal @ u
    return u_dot_n * face_geometry.area
