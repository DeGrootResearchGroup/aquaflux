"""SIMPLE-type block preconditioner for the coupled p--U Newton solve.

The coupled saddle-point Jacobian ``J = [[F, G],[D, -C]]`` (momentum block ``F``, pressure-gradient
``G``, divergence ``D``, Rhie--Chow pressure block ``-C``) is *indefinite*, so unpreconditioned
GMRES stalls and its iteration count grows like ``h^-2`` with the mesh. The cure is a preconditioner
that captures the **pressure Schur complement** ``S = -C - D F^{-1} G``: with a spectrally-equivalent
``S``-approximation the iteration count becomes mesh-independent (Murphy--Golub--Wathen 2000: a
block-diagonal preconditioner with the exact Schur gives GMRES convergence in <= 3 iterations).

This module builds the SIMPLE approximation, whose defining move is to replace ``F^{-1}`` by its
diagonal -- which is exactly the momentum diagonal ``a_P`` the flow solver already computes and lags.
The pressure Schur is then the compact Rhie--Chow **pressure Laplacian** with face coefficient
``c_f = rho (V/a_P)_f A_f / (d.n)_f`` -- the same ``d = V/a_P`` coefficient the Rhie--Chow mass flux
uses (:mod:`aquaflux.flow.rhie_chow`).

Everything here is a **frozen** (``stop_gradient``-ed) linear operator: the preconditioner only
accelerates the Krylov iteration and, by the implicit-function-theorem adjoint of the converged
solve, leaves the solution and its gradient untouched (see ``tests/unit/test_preconditioning.py``).
The inner Schur solve is a **fixed** number of damped-Jacobi sweeps, so the whole preconditioner is a
constant linear operator and plain (left-preconditioned) GMRES suffices -- no FGMRES needed. A
mesh-independent inner solve (multigrid) is the scalable upgrade.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from aquaflux.schemes.interpolation import interpolate_owner_neighbour

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


def schur_face_coefficient(
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
    interp_factor: jnp.ndarray,
    normal_distance: jnp.ndarray,
    a_p: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Per-face SIMPLE Schur coefficient ``c_f = rho (V/a_P)_f A_f / (d.n)_f`` (0 on boundary faces).

    The pressure-Poisson face coefficient, shared by the assembled Laplacian
    (:func:`pressure_schur_laplacian`) and the multigrid inner solve. ``(V/a_P)_f`` is the interpolated
    Rhie--Chow ``d`` coefficient, the same one the mass flux uses; ``rho`` is interpolated to the face
    likewise.

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads cell volumes and face areas.
    interp_factor, normal_distance : jnp.ndarray
        Face interpolation factor ``g`` and normal distance ``d.n``, shape ``(n_faces,)`` each.
    a_p : jnp.ndarray
        Momentum diagonal per cell, shape ``(n_cells,)``.
    rho : jnp.ndarray
        Per-cell density, shape ``(n_cells,)``.

    Returns
    -------
    jnp.ndarray
        ``c_f`` per face, shape ``(n_faces,)`` (zero on boundary faces).
    """
    d_coeff = geometry.cell.volume / a_p  # V/a_P per cell
    d_coeff_face = interpolate_owner_neighbour(d_coeff, interp_factor, face_cells)
    rho_face = interpolate_owner_neighbour(rho, interp_factor, face_cells)
    return jnp.where(
        face_cells.interior, rho_face * d_coeff_face * geometry.face.area / normal_distance, 0.0
    )


def pressure_schur_laplacian(
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
    interp_factor: jnp.ndarray,
    normal_distance: jnp.ndarray,
    a_p: jnp.ndarray,
    rho: jnp.ndarray,
    pressure_pin: int | None = None,
    boundary_diagonal: jnp.ndarray | None = None,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """The compact Rhie--Chow pressure Laplacian ``Ŝ`` (the SIMPLE Schur approximation).

    Each interior face contributes a symmetric coupling with coefficient
    ``c_f = rho (V/a_P)_f A_f / (d.n)_f`` (``(V/a_P)_f`` the interpolated Rhie--Chow ``d`` coefficient),
    giving the M-matrix Laplacian ``(Ŝ p)_P = sum_f c_f (p_P - p_N)`` with diagonal ``sum_f c_f`` and
    off-diagonal ``-c_f``. Boundary faces carry no pressure--pressure coupling (a no-flux/velocity
    wall does not couple pressures), which is exact for the closed all-wall domains this first cut
    targets; pressure-outlet coupling is a later addition.

    The coefficients are ``stop_gradient``-ed: ``Ŝ`` is a frozen preconditioner operator, not part of
    the AD Jacobian.

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    geometry : MeshGeometry
        The mesh metrics; reads cell volumes and face areas.
    interp_factor : jnp.ndarray
        Face interpolation factor ``g`` (owner weight ``1 - g``), shape ``(n_faces,)``.
    normal_distance : jnp.ndarray
        Owner-to-neighbour normal distance ``d.n`` per face, shape ``(n_faces,)``.
    a_p : jnp.ndarray
        Momentum diagonal per cell, shape ``(n_cells,)`` (already lagged/frozen by the caller).
    rho : jnp.ndarray
        Density.
    pressure_pin : int, optional
        Index of a pinned pressure cell (its continuity row is replaced by ``p = const``); its Laplacian
        row is set to identity so the inner solve leaves it fixed.
    boundary_diagonal : jnp.ndarray, optional
        Per-cell pressure--pressure stiffness added to the diagonal from pressure-fixing boundary faces
        (a :class:`~aquaflux.flow.PressureOutlet`). This is the boundary coupling that makes the
        open-domain Schur non-singular; ``None`` (or all-zero, for a closed domain) leaves the
        pure-Neumann interior Laplacian, regularised instead by ``pressure_pin``.

    Returns
    -------
    matvec : callable
        ``p -> Ŝ p`` of shape ``(n_cells,)``.
    diagonal : jnp.ndarray
        Diagonal of ``Ŝ``, shape ``(n_cells,)`` (for a Jacobi/Chebyshev inner smoother).
    """
    owner = face_cells.owner
    nb = face_cells.safe_neighbour

    coeff = jax.lax.stop_gradient(
        schur_face_coefficient(face_cells, geometry, interp_factor, normal_distance, a_p, rho)
    )

    # Symmetric M-matrix Laplacian: each interior face adds c_f to both incident diagonals; a
    # pressure-fixing boundary face adds its coupling to its owner's diagonal only (the boundary
    # pressure is prescribed, so there is no off-diagonal term).
    diagonal = face_cells.scatter_symmetric(coeff)
    if boundary_diagonal is not None:
        diagonal = diagonal + jax.lax.stop_gradient(boundary_diagonal)
    if pressure_pin is not None:  # pinned row is identity (continuity replaced by p = const)
        diagonal = diagonal.at[pressure_pin].set(1.0)

    def matvec(p: jnp.ndarray) -> jnp.ndarray:
        flux = coeff * (p[owner] - p[nb])  # owner-outward; zero on boundary faces
        result = face_cells.scatter_conservative(flux)
        if boundary_diagonal is not None:
            result = result + jax.lax.stop_gradient(boundary_diagonal) * p
        if pressure_pin is not None:
            result = result.at[pressure_pin].set(p[pressure_pin])
        return result

    return matvec, diagonal


def damped_jacobi_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    diagonal: jnp.ndarray,
    rhs: jnp.ndarray,
    sweeps: int,
    omega: float,
    pressure_pin: int | None = None,
) -> jnp.ndarray:
    """A **fixed** number of damped-Jacobi sweeps for ``A x = rhs`` -- a constant linear operator.

    Each sweep is ``x <- x + omega D^{-1} (rhs - A x)``. A fixed sweep count (no convergence check)
    makes the map ``rhs -> x`` a fixed linear operator with frozen coefficients, so it can serve as a
    left preconditioner under plain GMRES (a *variable*, convergence-based inner would instead require
    FGMRES). This is the placeholder inner solve; a fixed-cycle multigrid is the mesh-independent
    upgrade.

    Parameters
    ----------
    matvec : callable
        The operator ``x -> A x`` of shape ``(n_cells,)``.
    diagonal : jnp.ndarray
        Diagonal of ``A``, shape ``(n_cells,)``.
    rhs : jnp.ndarray
        Right-hand side, shape ``(n_cells,)``.
    sweeps : int
        Number of Jacobi sweeps (static).
    omega : float
        Damping factor in ``(0, 1]`` (~0.7 for a Laplacian).
    pressure_pin : int, optional
        Pinned cell held at ``x = rhs`` throughout (its row is identity).

    Returns
    -------
    jnp.ndarray
        The approximate solution ``x``, shape ``(n_cells,)``.
    """
    inv_diagonal = 1.0 / diagonal
    x = jnp.zeros_like(rhs)
    if pressure_pin is not None:
        x = x.at[pressure_pin].set(rhs[pressure_pin])
    for _ in range(sweeps):
        x = x + omega * inv_diagonal * (rhs - matvec(x))
        if pressure_pin is not None:
            x = x.at[pressure_pin].set(rhs[pressure_pin])
    return x
