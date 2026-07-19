"""Rhie--Chow momentum interpolation: the face mass flux and the momentum diagonal.

Collocated (cell-centred) storage of pressure and velocity admits a checkerboard pressure mode
because a cell's continuity balance never sees its own pressure. Rhie--Chow interpolation cures
this by adding a pressure-difference term to the interpolated face velocity, scaled by the
momentum "d" coefficient ``V / a_P`` (cell volume over the momentum-matrix diagonal):

    u_f . n = interp(u) . n  -  (V/a_P)_f [ (p_N - p_P)/(d.n)  -  interp(grad p) . n ],

so the face flux ``mdot_f = rho (u_f . n) A`` depends on the *compact* pressure difference and
couples pressure implicitly in continuity — the extra term is the difference between the compact
and the interpolated pressure gradient, which vanishes as the mesh resolves, so consistency is
preserved.

``a_P`` is the diagonal of the momentum equation (viscous + convective central coefficient, plus
transient). Its convective part uses a velocity-flux *estimate* for the mass flux (not the
Rhie--Chow ``mdot`` itself), which breaks the ``a_P`` <-> ``mdot`` circularity and makes ``a_P`` a
non-circular function of the velocity. It is assembled here from the central-coefficient formula
but, unlike a classical lagged coefficient, it is **differentiated** in the residual: ``a_P``
enters ``V / a_P``, whose damping term is non-zero for a non-linear pressure field, so it genuinely
affects the converged solution's sensitivity — freezing it (``stop_gradient``) would leave the
implicit-function-theorem adjoint linearizing a different residual than the one solved. The block
preconditioner, which needs a constant operator, freezes ``a_P`` on its side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.schemes.interpolation import interpolate_owner_neighbour, interpolate_to_face
from aquaflux.vectors import dot, scale

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


def viscous_face_coefficient(
    mu: jnp.ndarray,
    normal_distance: jnp.ndarray,
    interp_factor: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
) -> jnp.ndarray:
    """Central viscous face coefficient ``mu_f A / (d.n)`` per face, shape ``(n_faces,)``.

    The face viscosity is the owner/neighbour interpolation ``mu_f`` (the same blend every other face
    value uses), times the face area over the normal distance. The single definition of the viscous
    coupling shared by the momentum diagonal (:func:`momentum_diagonal`) and the frozen
    convection-diffusion operator the velocity-block preconditioner coarsens, so the two cannot drift.

    Parameters
    ----------
    mu : jnp.ndarray
        Per-cell dynamic viscosity, shape ``(n_cells,)``.
    normal_distance : jnp.ndarray
        Owner-to-neighbour (or owner-to-face on boundaries) normal distance ``d . n`` per face,
        shape ``(n_faces,)``.
    interp_factor : jnp.ndarray
        Face interpolation factor ``g``, shape ``(n_faces,)``.
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads face areas.
    """
    mu_face = interpolate_owner_neighbour(mu, interp_factor, face_cells)
    return mu_face * geometry.face.area / normal_distance


def momentum_diagonal(
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
    mu: jnp.ndarray,
    normal_distance: jnp.ndarray,
    interp_factor: jnp.ndarray,
    mdot_lagged: jnp.ndarray | None = None,
    dt: float | None = None,
    boundary_owner_coeff: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-cell momentum-matrix diagonal ``a_P`` (viscous + convective + transient central coeff).

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads cell volumes and face areas.
    mu : jnp.ndarray
        Per-cell dynamic viscosity, shape ``(n_cells,)``.
    normal_distance : jnp.ndarray
        Owner-to-neighbour (or owner-to-face on boundaries) normal distance ``d . n`` per face,
        shape ``(n_faces,)``.
    interp_factor : jnp.ndarray
        Face interpolation factor ``g``, shape ``(n_faces,)`` — the same owner/neighbour blend used
        for every other face value, applied here to the viscosity.
    mdot_lagged : jnp.ndarray, optional
        Lagged face mass flux for the convective contribution, shape ``(n_faces,)``; omit for
        Stokes flow (no convection).
    dt : float, optional
        Timestep for the transient contribution ``V / dt``; omit for steady flow.
    boundary_owner_coeff : jnp.ndarray, optional
        Per-face owner diagonal contribution on boundary faces (zero on interior faces), shape
        ``(n_faces,)`` — each patch's :meth:`~aquaflux.flow.boundary.FlowBoundary.momentum_diagonal_coefficient`,
        so a zero-gradient outlet drops the viscous term and a wall drops the convective one. When
        omitted, every boundary face contributes the full interior-style ``viscous + max(mdot, 0)``
        (the leading-order form; only the pure geometry/unit tests rely on this default).

    Returns
    -------
    jnp.ndarray
        ``a_P`` per cell **per velocity component**, shape ``(n_cells, dim)``. The viscous,
        convective, and transient contributions are the same for every component (isotropic
        viscosity, a component-independent mass flux), so the columns are equal here; the
        per-component shape is the seam a directional momentum source (e.g. anisotropic porous
        resistance) would fill differently. The Rhie--Chow coefficient projects it onto the face
        normal as ``sum_i n_i^2 (V / a_P_i)``, which reduces to the scalar ``V / a_P`` for equal
        components.
    """
    viscous = viscous_face_coefficient(mu, normal_distance, interp_factor, face_cells, geometry)

    # The scatter masks the neighbour coefficient to zero on boundary faces, so pass it unmasked.
    owner_coeff = viscous
    neighbour_coeff = viscous
    if mdot_lagged is not None:  # convective upwind: outflow leaves the owner, inflow the neighbour
        owner_coeff = owner_coeff + jnp.maximum(mdot_lagged, 0.0)
        neighbour_coeff = neighbour_coeff + jnp.maximum(-mdot_lagged, 0.0)

    # On boundary faces the owner contribution is the patch's own diagonal coefficient (a wall drops
    # the convective term, a zero-gradient outlet the viscous one), not the interior-style sum above;
    # the scatter already zeros the neighbour coefficient on boundary faces.
    if boundary_owner_coeff is not None:
        owner_coeff = jnp.where(face_cells.interior, owner_coeff, boundary_owner_coeff)

    a_p_isotropic = face_cells.scatter(owner_coeff, neighbour_coeff)
    if dt is not None:
        a_p_isotropic = a_p_isotropic + geometry.cell.volume / dt
    dim = geometry.face.normal.shape[-1]
    return jnp.broadcast_to(a_p_isotropic[:, None], (*a_p_isotropic.shape, dim))


def advective_momentum_flux(
    velocity: jnp.ndarray,
    rho: jnp.ndarray,
    interp_factor: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
    grad_velocity: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Owner-outward advective momentum flux ``(rho u)_f . n`` per face, shape ``(n_faces,)``.

    Reconstructs the per-cell momentum ``rho u`` to each face and projects it on the owner-outward
    normal — the single definition of the convective face flux shared by the Rhie--Chow mass flux
    (:func:`interior_mass_flux`) and the momentum diagonal's convective estimate
    (:meth:`~aquaflux.flow.MomentumContinuity.momentum_matrix_diagonal`). Keeping them one function is
    what makes the estimate that breaks the ``a_P`` <-> ``mdot`` circularity consistent with the mass
    flux it stands in for. Density rides **with** the velocity as the conserved momentum ``rho u`` (not
    a separate face density times an interpolated velocity); the two agree exactly for uniform ``rho``.

    Parameters
    ----------
    velocity : jnp.ndarray
        Per-cell velocity, shape ``(n_cells, dim)``.
    rho : jnp.ndarray
        Per-cell density, shape ``(n_cells,)``.
    interp_factor : jnp.ndarray
        Face interpolation factor ``g``, shape ``(n_faces,)``.
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads face normals (and centroids, for the reconstruction).
    grad_velocity : jnp.ndarray, optional
        Per-cell velocity gradient ``(n_cells, dim, dim)``. When given, the momentum is reconstructed
        to the face integration point (2nd order, :func:`interpolate_to_face`); when omitted, the cheap
        leading-order owner/neighbour blend (:func:`interpolate_owner_neighbour`).

    Returns
    -------
    jnp.ndarray
        The normal-projected face momentum flux **before** the area factor; multiply by
        ``geometry.face.area`` for the mass flux ``mdot``.
    """
    momentum = scale(velocity, rho)
    if grad_velocity is None:
        momentum_face = interpolate_owner_neighbour(momentum, interp_factor, face_cells)
    else:
        grad_momentum = (
            rho[:, None, None] * grad_velocity
        )  # grad(rho u) = rho grad(u) for uniform rho
        momentum_face = interpolate_to_face(
            momentum, grad_momentum, interp_factor, face_cells, geometry
        )
    return dot(momentum_face, geometry.face.normal)


def interior_mass_flux(
    velocity: jnp.ndarray,
    grad_velocity: jnp.ndarray,
    pressure: jnp.ndarray,
    grad_pressure: jnp.ndarray,
    d_coeff: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
    interp_factor: jnp.ndarray,
    normal_distance: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Owner-outward Rhie--Chow mass flux on interior faces, shape ``(n_faces,)``.

    Reconstructs the per-cell momentum ``rho u`` to each face's integration point and adds the
    compact pressure-jump term that couples pressure implicitly and kills checkerboarding. Boundary
    faces are handled by the per-patch closures; the caller masks them out of this interior result.

    Two skewness-consistent pieces keep the flux second-order on non-orthogonal meshes: the advective
    momentum is reconstructed to the face centroid (:func:`interpolate_to_face`, carrying the
    ``grad·(x_ip − x_g)`` correction rather than stopping at the projection foot ``x_g``), and the
    pressure damping compares the compact and gradient-reconstructed pressure jumps **along the
    owner→neighbour vector d** (not the face normal), so the two are the same directional derivative
    and cancel exactly on a linear pressure field.

    The advective flux is ``(rho u)_f . n`` — the density is reconstructed **with** the velocity as
    the momentum ``rho u`` (the conserved mass flux), not as a separate face density times an
    interpolated velocity; the two agree exactly for uniform ``rho``.

    Parameters
    ----------
    velocity : jnp.ndarray
        Per-cell velocity, shape ``(n_cells, dim)``.
    grad_velocity : jnp.ndarray
        Per-cell velocity gradient ``grad_velocity[c, i, j] = d u_i / d x_j``, shape
        ``(n_cells, dim, dim)`` — reconstructs the momentum to the integration point.
    pressure : jnp.ndarray
        Per-cell pressure, shape ``(n_cells,)``.
    grad_pressure : jnp.ndarray
        Per-cell pressure gradient, shape ``(n_cells, dim)``.
    d_coeff : jnp.ndarray
        Per-cell, per-component Rhie--Chow coefficient ``V / a_P``, shape ``(n_cells, dim)``. It is
        projected onto the face normal as ``sum_i n_i^2 (V/a_P_i)`` (the directional damping
        strength), which reduces to the scalar ``V/a_P`` for isotropic ``a_P``.
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads face normals/areas and the cell/face centroids.
    interp_factor : jnp.ndarray
        Face interpolation factor ``g``, shape ``(n_faces,)``.
    normal_distance : jnp.ndarray
        Owner-to-neighbour normal distance ``d . n``, shape ``(n_faces,)``.
    rho : jnp.ndarray
        Per-cell density, shape ``(n_cells,)``.
    """
    face_geometry = geometry.face
    normal = face_geometry.normal

    # Advective mass flux: the shared momentum reconstruction rho*u -> face integration point (2nd
    # order), projected on the normal — the same estimate the momentum diagonal's convective term uses.
    mom_normal = advective_momentum_flux(
        velocity, rho, interp_factor, face_cells, geometry, grad_velocity
    )

    rho_face = interpolate_owner_neighbour(rho, interp_factor, face_cells)
    grad_face = interpolate_owner_neighbour(grad_pressure, interp_factor, face_cells)

    # Directional Rhie--Chow coefficient d_hat = sum_i n_i^2 (V/a_P_i)_f, the momentum-diagonal
    # damping projected onto the face normal (scalar V/a_P for isotropic a_P).
    d_coeff_face = interpolate_owner_neighbour(d_coeff, interp_factor, face_cells)
    d_hat = dot(normal * normal, d_coeff_face)

    # Rhie--Chow damping: (actual pressure jump − gradient-reconstructed jump) along the connection
    # vector d, per normal distance — a consistent directional derivative that vanishes on a linear
    # pressure field and suppresses checkerboarding.
    cell_centroid = geometry.cell.centroid
    d = face_cells.neighbour_centroid(cell_centroid) - cell_centroid[face_cells.owner]  # seam image
    dp = pressure[face_cells.safe_neighbour] - pressure[face_cells.owner]
    damping = (dp - dot(grad_face, d)) / normal_distance
    return (mom_normal - rho_face * d_hat * damping) * face_geometry.area
