"""The diffusion face-flux operator: a flux-continuous non-orthogonal diffusion flux.

The diffusive flux through a face is ``Gamma (grad phi . n) A``. Its normal derivative is
built to be **flux-continuous across the face**, so that a jump in the diffusion coefficient
(different properties / zones sharing a face) is handled natively. Requiring

    Gamma_P grad phi|_P . n  =  Gamma_N grad phi|_N . n

and extrapolating a one-sided normal derivative from each cell centroid to the face
integration point,

    grad phi|_P . n  =  (phi_ip - phi_P - corr_P) / (D_P . n),   corr_P = grad phi_P . (D_P - (D_P.n) n),

(and likewise from ``N``), then eliminating the common face value ``phi_ip`` gives the
normal derivative in terms of cell-centred quantities:

    grad phi|_ip . n  =  [ (phi_N - phi_P) + corr_N - corr_P ] / denom,
    denom = (D_P . n) - (Gamma_P / Gamma_N)(D_N . n),

where ``D_P = x_ip - x_P`` and ``D_N = x_ip - x_N`` are the owner/neighbour centroid →
face-centroid displacements. The face flux is then ``Gamma_P (grad phi|_ip . n) A``, which —
by construction of ``denom`` from the continuity condition — is the single conservative flux
both cells share (owner ``+``, neighbour ``-``).

The two ``corr`` terms are the non-orthogonal correction: each is the owner/neighbour cell
gradient dotted with the tangential part of its centroid-to-face displacement, and both
vanish on an orthogonal grid, where ``denom -> (x_N - x_P) . n`` and the whole expression
reduces to the harmonic-mean flux of Patankar. Because the correction is written directly
into this residual term (not deferred to an explicit source), automatic differentiation
places it *in the Jacobian* — a consistently linearized non-orthogonal operator with no
hand-derived coefficients.

At a boundary face the neighbour side is replaced by the weak boundary value ``phi_ip``
supplied by a :class:`~aquaflux.boundary.conditions.BoundaryCondition`, giving the
one-sided flux ``Gamma_P (phi_ip - phi_P - corr_P) / (D_P . n) A``.

An optional per-face ``boundary_coefficient`` overrides ``Gamma_P`` on boundary faces only, for a
surface whose effective transport coefficient differs from its owner cell's — a wall-function eddy
viscosity, a contact resistance, a surface film. Interior faces and the ``None`` default are
untouched, so it is behaviour-neutral wherever it is not supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale

from .face_flux import FaceContext, FaceFluxOperator

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


def flux_continuous_denominator(
    dpn: jnp.ndarray,
    dnn: jnp.ndarray,
    gamma_owner: jnp.ndarray,
    gamma_neighbour: jnp.ndarray,
) -> jnp.ndarray:
    """Coefficient-jump-corrected normal distance ``(D_P.n) - (Gamma_P/Gamma_N)(D_N.n)``.

    The denominator of the flux-continuous normal derivative: eliminating the common face value from
    the two one-sided extrapolations under ``Gamma_P dphi/dn|_P = Gamma_N dphi/dn|_N`` leaves this in
    the denominator, so a coefficient jump is carried natively (it reduces to ``(x_N - x_P).n`` when
    ``Gamma_P = Gamma_N``, the orthogonal-limit harmonic mean). Defined once here so the residual flux
    (:class:`DiffusionFlux`) and the operator-diagonal conductance
    (:func:`flux_continuous_conductance`) cannot use different denominators.

    Parameters
    ----------
    dpn, dnn : jnp.ndarray
        The owner/neighbour centroid → face-centroid normal displacements ``D_P.n`` (``> 0``) and
        ``D_N.n`` (``< 0`` on interior faces), shape ``(n_faces,)`` each.
    gamma_owner, gamma_neighbour : jnp.ndarray
        The owner/neighbour cell diffusion coefficient gathered to each face, shape ``(n_faces,)``.
    """
    return dpn - (gamma_owner / gamma_neighbour) * dnn


def flux_continuous_conductance(
    gamma: jnp.ndarray,
    geometry: MeshGeometry,
    face_cells: FaceCellConnectivity,
) -> jnp.ndarray:
    """Per-face flux-continuous diffusion conductance ``Gamma_P A / denom``, shape ``(n_faces,)``.

    This is exactly the contribution each face makes to the diffusion operator's diagonal — and the
    magnitude of its symmetric off-diagonal coupling: differentiating :class:`DiffusionFlux`'s
    owner-outward face flux ``-Gamma_P (grad phi . n) A`` w.r.t. the owner value gives ``Gamma_P A /
    denom``, and (through the ``owner +`` / ``neighbour -`` scatter) w.r.t. the neighbour value gives the
    same magnitude to the neighbour's diagonal — so scattering this one per-face value to *both* incident
    cells reproduces the operator diagonal. On an orthogonal face it is the harmonic mean ``2
    Gamma_P Gamma_N / (Gamma_P + Gamma_N) . A / h``; it reduces to ``Gamma A / h`` for a constant
    coefficient. Boundary faces use the one-sided ``D_P.n`` (there is no neighbour), matching the
    one-sided boundary flux.

    The single definition of the diffusion coupling shared by the momentum diagonal ``a_P``, the scalar
    pseudo-time shift, and the frozen convection-diffusion operators the AMG hierarchies coarsen — so
    none can drift from the residual's :class:`DiffusionFlux`. It uses the owner coefficient ``Gamma_P``
    (not an owner/neighbour interpolation), which is what makes the assembled coupling equal to the
    operator's actual diagonal contribution rather than an arithmetic-mean approximation.

    Parameters
    ----------
    gamma : jnp.ndarray
        Per-cell diffusion coefficient (viscosity for momentum, ``nu + sigma nu_t`` for a scalar), shape
        ``(n_cells,)``.
    geometry : MeshGeometry
        The mesh metrics; reads face normals/areas and the cell/face centroids.
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).
    """
    fg = geometry.face
    x_cell = geometry.cell.centroid
    normal = fg.normal
    d_p = fg.centroid - x_cell[face_cells.owner]
    d_n = fg.centroid - face_cells.neighbour_centroid(
        x_cell
    )  # periodic-image neighbour across a seam
    dpn = dot(d_p, normal)  # D_P . n  (> 0)
    dnn = dot(d_n, normal)  # D_N . n  (< 0 on interior faces)
    gamma_owner = gamma[face_cells.owner]
    gamma_neighbour = gamma[face_cells.safe_neighbour]
    # Boundary faces have no neighbour (safe_neighbour is the owner, so D_N.n collapses to D_P.n and the
    # interior denom to zero); use the one-sided D_P.n there, as the one-sided boundary flux does.
    denom = face_cells.combine_face_values(
        flux_continuous_denominator(dpn, dnn, gamma_owner, gamma_neighbour), dpn
    )
    return gamma_owner * fg.area / denom


class DiffusionFlux(FaceFluxOperator):
    """Flux-continuous non-orthogonal diffusion flux.

    The full physical flux (orthogonal part + non-orthogonal correction + coefficient-jump
    handling) is written as one residual term, so its linearization comes entirely from
    automatic differentiation. It consumes the injected per-cell gradient (from a
    :class:`~aquaflux.schemes.GradientScheme`, carried on the context) for the correction; on an
    orthogonal grid the correction is identically zero and the gradient is inert.

    Attributes
    ----------
    coefficient : str
        The name of the property this operator uses as its diffusion coefficient
        ``Gamma`` (``"diffusivity"`` for a generic scalar, ``"conductivity"`` for heat,
        ``"viscosity"`` for momentum) — read from ``context.properties``. Static.
    boundary_coefficient : jnp.ndarray or None
        Optional per-face coefficient ``(n_faces,)`` that replaces the owner-cell ``Gamma`` **on
        boundary faces only** (interior entries are ignored). Its home is a surface whose effective
        transport coefficient differs from its owner cell's — the wall-function eddy viscosity, where
        the momentum wall-face ``mu_eff`` is the wall model's value rather than ``k/omega``. ``None``
        (default) leaves every face on the owner-cell coefficient, so the operator is unchanged.
    """

    coefficient: str = eqx.field(static=True, default="diffusivity")
    boundary_coefficient: jnp.ndarray | None = None

    def face_flux(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        fc = context.face_cells
        owner, neighbour = fc.owner, fc.safe_neighbour
        fg = context.geometry.face
        n, area, x_ip = fg.normal, fg.area, fg.centroid
        x_cell = context.geometry.cell.centroid

        phi_owner, phi_neighbour = field[owner], field[neighbour]
        grad_owner, grad_neighbour = context.gradient[owner], context.gradient[neighbour]
        gamma = context.properties[self.coefficient]
        gamma_owner, gamma_neighbour = gamma[owner], gamma[neighbour]

        d_p = x_ip - x_cell[owner]
        d_n = x_ip - fc.neighbour_centroid(x_cell)  # periodic-image neighbour across a seam
        dpn = dot(d_p, n)  # D_P . n  (> 0)
        dnn = dot(d_n, n)  # D_N . n  (< 0 on interior faces)

        corr_p = dot(grad_owner, d_p - scale(n, dpn))
        corr_n = dot(grad_neighbour, d_n - scale(n, dnn))

        # Interior: two-sided, flux-continuous normal derivative (Gamma-jump in denom).
        denom = flux_continuous_denominator(dpn, dnn, gamma_owner, gamma_neighbour)
        denom_safe = fc.combine_face_values(denom, 1.0)  # boundary branch unused; keep grad finite
        normal_grad_interior = ((phi_neighbour - phi_owner) + corr_n - corr_p) / denom_safe

        # Boundary: one-sided normal derivative to the weak face value.
        normal_grad_boundary = (context.boundary_values - phi_owner - corr_p) / dpn

        normal_grad = fc.combine_face_values(normal_grad_interior, normal_grad_boundary)
        # The face coefficient is the owner-cell value, except where a per-face boundary coefficient
        # overrides it on boundary faces (combine_face_values keeps the owner value on interior
        # faces, so interior fluxes and the None default are untouched).
        gamma_face = gamma_owner
        if self.boundary_coefficient is not None:
            gamma_face = fc.combine_face_values(gamma_owner, self.boundary_coefficient)
        # Owner-outward flux of phi is down-gradient (Fourier): -Gamma (grad phi . n) A.
        return -gamma_face * normal_grad * area
