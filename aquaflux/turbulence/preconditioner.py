"""Convection-diffusion AMG preconditioner for the SST scalar transport (k, omega) solves.

The k and omega equations are convection-diffusion-reaction scalars. At high Reynolds number their
unpreconditioned Krylov solve stagnates for the same reason the momentum block did before it was made
convection-aware: the first-order-upwind advection dominates the diffusion, so the linear operator is
strongly nonsymmetric and far from the diagonal an unpreconditioned Krylov method implicitly assumes.

This builds a frozen nonsymmetric aggregation multigrid on the scalar's ``viscous + first-order-upwind``
operator -- the same reduction the velocity block uses -- and applies it as a matrix-free V-cycle
preconditioner. The three ingredients the multigrid builder needs are read straight off the transport:

* the diffusion face coefficient ``Gamma_face * A / (d . n)`` from the effective diffusivity
  ``Gamma = nu + sigma nu_t`` (interpolated to faces exactly as the diffusion flux forms it),
* the convective volume flux ``mdot / rho`` per face, and
* the reaction-plus-boundary diagonal -- the part of the true diagonal the interior stencil does not
  supply (the local source terms' linearization and the Dirichlet boundary stiffness) -- got in a
  single Jacobian-vector product ``J . 1``, minus the interior convective imbalance so a
  boundary-adjacent cell's absent boundary face is not double-counted onto the diagonal.

The hierarchy is built once off the jit path (integer/scipy graph work) and applied as a fixed,
``stop_gradient``-ed V-cycle, so it only accelerates the Krylov iteration and leaves the converged
field and the segregated solve's adjoint untouched (the same guarantee as the flow preconditioner).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.mesh import Mesh
from aquaflux.mesh.geometry import MeshGeometry
from aquaflux.schemes.interpolation import interpolate_owner_neighbour, interpolation_factor
from aquaflux.solve.multigrid import (
    air_multigrid_solve,
    build_convection_air_hierarchy,
    build_convection_hierarchy,
    convection_multigrid_solve,
)
from aquaflux.vectors import dot

_Factory = Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]


def scalar_transport_preconditioner(
    mesh: Mesh,
    geometry: MeshGeometry,
    diffusivity: jnp.ndarray,
    volume_flux: jnp.ndarray,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    reference: jnp.ndarray,
    *,
    method: str = "twolevel",
    v_cycles: int = 1,
    fixed_cells: jnp.ndarray | None = None,
) -> _Factory:
    """A frozen convection-diffusion V-cycle preconditioner for a scalar transport equation.

    Parameters
    ----------
    mesh, geometry : Mesh, MeshGeometry
        The transport mesh and its metrics.
    diffusivity : jnp.ndarray
        The per-cell effective diffusivity ``Gamma = nu + sigma nu_t``, shape ``(n_cells,)``.
    volume_flux : jnp.ndarray
        The per-face convective flux the equation advects on (``mdot / rho``), shape ``(n_faces,)``.
    residual_fn : callable
        The scalar residual ``phi -> R(phi)`` whose Jacobian is preconditioned. Only its ``J . 1``
        row sum is used (for the reaction/boundary diagonal), so its sources are read without being
        re-implemented here.
    reference : jnp.ndarray
        The field the frozen operator linearizes at, shape ``(n_cells,)`` (e.g. the current sweep's
        field). For a linear transport equation any reference gives the same operator; a piecewise
        source (a limiter) is captured at its reference branch.
    method : {"twolevel", "air"}
        The convection hierarchy: the stable two-level aggregation (default) or the reduction-based
        (lAIR) hierarchy that coarsens fully and stays mesh-independent at large sizes.
    v_cycles : int
        V-cycles per apply.
    fixed_cells : jnp.ndarray, optional
        Cells whose residual is a value fixation ``phi - target`` (e.g. the omega near-wall cells):
        their rows are the identity, so they are detached from the aggregation (their incident edges
        dropped, unit diagonal) to match the operator the solve actually inverts.

    Returns
    -------
    callable
        A ``phi -> M`` factory giving the (frozen, ``phi``-independent) preconditioner matvec ``M``.
    """
    if method not in ("twolevel", "air"):
        raise ValueError(f"unknown method {method!r}; use 'twolevel' or 'air'")
    face_cells = mesh.face_cells
    owner_e, nb_e, interior_faces = face_cells.interior_edges()
    n = mesh.n_cells

    # Diffusion face coefficient Gamma_face * A / (d . n), interpolated to faces as the flux forms it.
    gamma = jax.lax.stop_gradient(diffusivity)
    gamma_face = interpolate_owner_neighbour(
        gamma, interpolation_factor(face_cells, geometry), face_cells
    )
    d = (
        face_cells.neighbour_centroid(geometry.cell.centroid)
        - geometry.cell.centroid[face_cells.owner]
    )
    normal_distance = dot(d, geometry.face.normal)
    viscous = gamma_face * geometry.face.area / normal_distance
    visc_int = np.asarray(jax.lax.stop_gradient(viscous))[interior_faces]
    mdot_int = np.asarray(jax.lax.stop_gradient(volume_flux))[interior_faces]

    # Reaction + boundary diagonal. The aggregation operator carries only the *interior* stencil
    # (the edges above), so its diagonal must be corrected by everything else on the true diagonal:
    # the local source linearization and the Dirichlet boundary-face stiffness. That is J . 1 minus
    # the interior convective imbalance -- a conservative interior operator has zero row sums, but a
    # boundary-adjacent cell's interior faces do not sum to zero (the boundary face carrying the rest
    # is absent from the edge list), and J . 1 would otherwise fold that imbalance into the diagonal
    # (double-counting the boundary flux). Subtracting the interior net outflow removes it exactly.
    _, j_dot_one = jax.jvp(residual_fn, (reference,), (jnp.ones_like(reference),))
    interior_outflow = np.zeros(n)
    np.add.at(interior_outflow, owner_e, mdot_int)
    np.add.at(interior_outflow, nb_e, -mdot_int)
    boundary_diagonal = np.asarray(jax.lax.stop_gradient(j_dot_one)) - interior_outflow
    # Clamp non-negative: any residual anti-diffusive source (e.g. an active production limiter not
    # already made explicit) would make the operator indefinite and its V-cycle diverge; dropping it
    # keeps an M-matrix and only softens the preconditioner (it approximates the Jacobian).
    boundary_diagonal = np.maximum(boundary_diagonal, 0.0)

    if fixed_cells is not None:
        fixed = np.asarray(fixed_cells)
        is_fixed = np.zeros(n, dtype=bool)
        is_fixed[fixed] = True
        keep = ~(is_fixed[owner_e] | is_fixed[nb_e])
        owner_e, nb_e = owner_e[keep], nb_e[keep]
        visc_int, mdot_int = visc_int[keep], mdot_int[keep]
        boundary_diagonal = boundary_diagonal.copy()
        boundary_diagonal[fixed] = 1.0  # identity rows: residual is phi - target

    if method == "air":
        hierarchy = build_convection_air_hierarchy(
            owner_e, nb_e, visc_int, mdot_int, n, boundary_diagonal=boundary_diagonal
        )

        def solve(r: jnp.ndarray) -> jnp.ndarray:
            return air_multigrid_solve(hierarchy, r, cycles=v_cycles)
    else:
        hierarchy = build_convection_hierarchy(
            owner_e, nb_e, visc_int, mdot_int, n, boundary_diagonal=boundary_diagonal, max_levels=2
        )

        def solve(r: jnp.ndarray) -> jnp.ndarray:
            return convection_multigrid_solve(hierarchy, r, cycles=v_cycles)

    return lambda phi: solve
