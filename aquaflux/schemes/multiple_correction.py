"""Gradient and Hessian by successive correction of Green--Gauss sums, with no system to solve.

:class:`~aquaflux.schemes.HessianCorrectedGradient` reaches exactness for quadratic fields by
solving a globally coupled gradient--Hessian system: the gradient equation needs the Hessian
through a face-curvature term, the Hessian equation needs the gradient, and the cycle is closed by
iterating. On a heavily warped mesh that costs twelve to fifteen sweeps of two face-kernel passes
each, and the count is a property of the mesh that has to be measured.

This module reaches the same exactness by the route of Pont, Brenner, Cinnella, Maugars and Robinet
(*Multiple-correction hybrid k-exact schemes for high-order compressible RANS-LES simulations on
fully unstructured grids*, J. Comput. Phys. 350, 2017), and it has no cycle in it. The whole method
follows from one observation about the Green--Gauss sum:

**Handed a linear field of gradient ``a``, the raw sum does not return ``a``.** It returns ``M1 a``
for a per-cell matrix ``M1`` that depends only on the mesh. So recover ``M1`` by applying the sum to
the coordinate fields, and ``D1 = M1^-1 R`` is exact for linear fields by construction. Apply ``D1``
twice and the same argument repeats one order up: the result is an inconsistent Hessian, ``M2`` is
what that double application returns for each quadratic basis field, and ``M2^-1`` repairs it.
Finally ``D1`` on a quadratic carries a first-order error which is a fixed linear function of the
Hessian, so subtracting it lifts the gradient to second order.

The sequence is therefore ``phi -> g1 -> H -> g``: **two passes over the faces and three per-cell
matrix products**, against the coupled scheme's twelve to fifteen sweeps of two passes. Every
correction matrix is obtained by running the operators on coordinate monomials, so there are no
hand-derived geometric formulas here and no volume moments to compute — the same probe-the-operator
device :func:`~aquaflux.schemes.cell_diagonal_block` already uses to recover per-cell blocks.

Three properties are worth stating because they are what make the method usable here rather than
merely fast:

* **It is exactly linear in the field.** A fixed sequence of fixed linear maps, with no convergence
  test and no data-dependent branching, so the tangent is the same sequence applied to the tangent
  — an unrolled apply, not an implicit-function solve. Handed an
  :class:`~aquaflux.schemes.ImposedGradient` it becomes *affine* rather than linear, the imposed
  values being an added constant; the tangent is still that same unrolled apply, which is the half
  of the property a differentiated solve depends on.
* **The matrices it inverts are small, local and well conditioned.** Measured on perturbed
  tetrahedra, ``cond(M1) <= 4.8`` and ``cond(M2) <= 10``, where the coupled scheme's own per-cell
  Hessian block under an owner boundary closure is ``1e18``. That is why no iteration is needed:
  there is no globally coupled system to converge.
* **Its data exchange is one ring per pass**, so unlike the coupled scheme it is not structurally
  barred from running domain-decomposed (see :meth:`MultipleCorrectionGradient.gradients`).

⚠️ **The boundary closure is not a detail here, and the rule governing it is not obvious.** The
correction matrices are probed on the *quadratic* basis, so they absorb whatever a closure does to a
quadratic — but an error the closure makes at *linear* order lands in a term those probes never see,
and nothing downstream removes it. A closure must therefore reproduce linear fields exactly. A raw
one-sided difference ``(phi_face - phi_owner) / (d.n)`` does not, on a skewed mesh, and using it
costs the whole reconstruction its accuracy; the non-orthogonal correction that repairs it is
:func:`~aquaflux.schemes.non_orthogonal_correction`, shared with the diffusion flux that has always
needed the same term for the same reason.

⚠️ **A closure is also the wrong instrument where the boundary value is not data.** Every closure
here reads the field on the boundary face, and a *zero-gradient* boundary condition does not supply
one -- it returns the owner cell's own value. Differencing against that does not estimate the normal
derivative, it *imposes* a near-zero one, which on the near-wall ``omega`` of a k--omega closure
contradicts a modelled profile that diverges like ``1 / d**2``. Measured on a backward-facing step
this leaves the march with no descent direction at all rather than a degraded one. The seam for such
a field is :class:`~aquaflux.schemes.ImposedGradient`: the caller states the gradient it knows, and
it is used in place of both the reconstruction and the closure. Where a gradient is genuinely
unknown a closure is still the answer -- the two solve different problems.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale

from .gradient import (
    GradientScheme,
    ImposedGradient,
    contract_symmetric,
    expand_symmetric,
    symmetric_components,
)
from .interpolation import (
    interpolate_owner_neighbour,
    interpolation_factor,
    non_orthogonal_correction,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from aquaflux.mesh import FaceCellConnectivity, Mesh, MeshGeometry


class GradientBoundaryClosure(eqx.Module):
    """Strategy: the **gradient's** value on a boundary face.

    A boundary condition supplies the field on a boundary face; it never supplies the field's
    gradient there. But a reconstruction that differentiates a gradient — which is what producing a
    Hessian from Green--Gauss sums amounts to — needs exactly that. This is the closure of that gap,
    and it is a strategy because the choice is consequential and mesh-dependent rather than a
    formula everyone agrees on.

    ⚠️ **An implementation must reproduce linear fields exactly.** Where the field varies linearly
    the true face gradient is the (constant) cell gradient, so a closure that returns anything else
    injects an error at linear order — and the correction matrices, calibrated on quadratics, cannot
    remove it. This is the one hard requirement on an implementation; being crude is survivable,
    being linear-inexact is not.

    ⚠️ **A closure receives the boundary field values as an array, with no boundary condition
    attached, so it cannot tell data from a closure of its own.** A zero-gradient condition returns
    the owner cell's value, and any closure that reads it is then differencing a number that carries
    no information. Where that matters the answer is not a cleverer closure but
    :class:`~aquaflux.schemes.ImposedGradient`, through which a caller states the gradient it
    knows; a closure's job is the case where nobody knows it.
    """

    @abc.abstractmethod
    def face_gradient(
        self,
        gradient: jnp.ndarray,
        field: jnp.ndarray,
        boundary_values: jnp.ndarray,
        face_cells: FaceCellConnectivity,
        geometry: MeshGeometry,
    ) -> jnp.ndarray:
        """The gradient on every face, shape ``(n_faces, dim)``.

        Only the boundary entries are used by the caller; interior faces are interpolated from the
        cells, so an implementation may return anything there.

        Parameters
        ----------
        gradient : jnp.ndarray
            The cell gradient already reconstructed, shape ``(n_cells, dim)``.
        field : jnp.ndarray
            Cell values of the field, shape ``(n_cells,)``.
        boundary_values : jnp.ndarray
            Face values of the field, shape ``(n_faces,)``; interior entries are unused.
        face_cells : FaceCellConnectivity
            The face→cell incidence.
        geometry : MeshGeometry
            The mesh metrics.
        """
        raise NotImplementedError


class OwnerGradient(GradientBoundaryClosure):
    """Take the owner cell's own gradient, unchanged.

    The cheapest closure that satisfies the linear-exactness requirement — where the field is linear
    the cell gradient *is* the face gradient — and on hexahedral meshes it costs nothing measurable:
    exactness holds and ``cond(M2)`` runs 4.8--5.3 against an exact-boundary reference's 2.5--2.9.

    ⚠️ **It fails on boundary tetrahedra.** Such a cell has four faces and one or two of them then
    carry no direction the owner's own gradient has not already supplied, which leaves the six
    Hessian components underdetermined: measured ``cond(M2)`` of ``9e17`` and a reconstruction no
    longer exact for quadratics. Prefer :class:`SkewCorrectedGradient` on any mesh with tetrahedra
    at a boundary.
    """

    def face_gradient(self, gradient, field, boundary_values, face_cells, geometry):
        return gradient[face_cells.owner]


class SkewCorrectedGradient(GradientBoundaryClosure):
    """Owner gradient tangentially, one-sided difference to the boundary value normally.

    A boundary condition does supply information the owner's gradient does not: the field's value on
    the face. Differencing it against the owner value gives the **normal** derivative, while the
    tangential part still comes from the owner. That extra direction is what a boundary tetrahedron
    is short of, and supplying it restores exactness there (measured ``5e-14`` against the owner
    closure's ``7e-2``) while *improving* conditioning on hexahedra (``cond(M2)`` 2.9 against 5.3 —
    the exact-boundary reference's own value).

    ⚠️ **The non-orthogonal correction is what makes it legal, not what makes it accurate.** Without
    it the one-sided difference is not exact for a linear field on a skewed mesh, and the closure
    then destroys the reconstruction outright rather than degrading it — measured worse than the
    owner closure it was meant to improve on. It is shared with the diffusion flux, which has always
    needed the same term to extrapolate the same derivative to the same place.

    ⚠️ **It is only as good as the boundary value, and a zero-gradient condition supplies none.**
    Such a condition returns the owner cell's own value, so the difference this closure takes is
    ``-correction`` rather than a rise, and the normal derivative it produces is near zero by
    construction. On a field whose modelled profile diverges at the wall -- the near-wall ``omega``
    of a k--omega closure -- that is not an approximation but a contradiction, and it is measured to
    cost a march its descent direction outright. Give such a field an
    :class:`~aquaflux.schemes.ImposedGradient`, which overrides this closure on the faces it covers.
    """

    def face_gradient(self, gradient, field, boundary_values, face_cells, geometry):
        owner = face_cells.owner
        normal = geometry.face.normal
        owner_gradient = gradient[owner]
        displacement = geometry.face.centroid - geometry.cell.centroid[owner]
        along = dot(displacement, normal)
        # A boundary face's centroid never coincides with its owner's, so `along` is nonzero there;
        # interior entries are discarded by the caller and are guarded only to keep the tangent
        # finite under differentiation.
        safe = jnp.where(jnp.abs(along) > 0.0, along, 1.0)
        rise = (
            boundary_values
            - field[owner]
            - non_orthogonal_correction(owner_gradient, displacement, normal)
        )
        tangential = owner_gradient - scale(normal, dot(owner_gradient, normal))
        return tangential + scale(normal, rise / safe)


class Corrections(eqx.Module):
    """The per-cell correction matrices, built once for one geometry.

    All three are geometry-only, so they are the whole of what
    :meth:`MultipleCorrectionGradient.bind` holds.

    Attributes
    ----------
    m1_inverse : jnp.ndarray
        ``M1^-1``, shape ``(n_cells, dim, dim)``. ``M1`` is what the raw Green--Gauss sum returns
        for a linear field of unit gradient in each coordinate direction.
    m2_inverse : jnp.ndarray
        ``M2^-1``, shape ``(n_cells, n_sym, n_sym)``. ``M2`` is what the doubly-applied 1-exact
        operator returns for each quadratic basis field.
    gradient_defect : jnp.ndarray
        The 1-exact gradient's first-order error per unit Hessian, shape ``(n_cells, dim, n_sym)``;
        subtracting it lifts that gradient to second order.
    """

    m1_inverse: jnp.ndarray
    m2_inverse: jnp.ndarray
    gradient_defect: jnp.ndarray


class MultipleCorrectionGradient(GradientScheme):
    """Second-order gradient and first-order Hessian, in two face passes and no solve.

    See the module docstring for the construction. In use this is a drop-in
    :class:`~aquaflux.schemes.GradientScheme`: it reconstructs the same quantity to the same
    contract — exact for quadratic fields — as
    :class:`~aquaflux.schemes.HessianCorrectedGradient`, without that scheme's coupled system, its
    sweep count, or the per-mesh calibration that count needs.

    Attributes
    ----------
    boundary_closure : GradientBoundaryClosure
        How the gradient is closed on boundary faces, where a boundary condition gives the field but
        not its derivative. Defaults to :class:`SkewCorrectedGradient`, which is the only shipped
        closure that holds up on boundary tetrahedra; :class:`OwnerGradient` is cheaper and adequate
        where cells are hexahedral.
    prepared : Corrections or None
        The geometry-only correction matrices, present once :meth:`bind` has been called. ``None``
        rebuilds them on every reconstruction, which is correct but wasteful — an assembler binds
        the scheme it is handed, so ordinary use never pays that.
    """

    boundary_closure: GradientBoundaryClosure = eqx.field(default_factory=SkewCorrectedGradient)
    prepared: Corrections | None = None

    def bind(self, mesh: Mesh, geometry: MeshGeometry) -> MultipleCorrectionGradient:
        """This scheme carrying the correction matrices it would otherwise rebuild every call.

        They depend only on the geometry, so this is pure bookkeeping: a bound scheme returns the
        same reconstruction bit for bit. See
        :meth:`~aquaflux.schemes.HessianCorrectedGradient.bind` for the staleness warning, which
        applies here identically — a binding is valid for the geometry it was made against and no
        other.
        """
        return MultipleCorrectionGradient(
            boundary_closure=self.boundary_closure,
            prepared=_build_corrections(mesh, geometry, self.boundary_closure),
        )

    def _reconstruct_gradient(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        imposed: ImposedGradient | None = None,
    ) -> jnp.ndarray:
        if operator_hook is not None:
            raise NotImplementedError(
                "MultipleCorrectionGradient cannot yet run domain-decomposed. Unlike the coupled "
                "Hessian scheme this is not a structural bar -- the reconstruction exchanges one "
                "ring per pass, so a correct distributed build needs the reconstructed gradient "
                "halo-exchanged once between the two passes. `operator_hook` is the wrong seam for "
                "that: it refreshes a solve's unknown, and there is no solve here."
            )
        return self.reconstruct(field, mesh, geometry, boundary_values, imposed=imposed)[0]

    def reconstruct(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
        *,
        imposed: ImposedGradient | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Both reconstructed quantities: the gradient and the Hessian.

        The Hessian is a genuine output here rather than an eliminated intermediate, and it costs
        nothing extra — the gradient's own second-order correction is built from it.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        mesh : Mesh
            Provides owner/neighbour connectivity.
        geometry : MeshGeometry
            Face and cell metrics.
        boundary_values : jnp.ndarray
            Face values of the field, shape ``(n_faces,)``; interior entries are ignored.
        imposed : ImposedGradient, optional
            Cells whose gradient is a model quantity rather than something to reconstruct. It is
            imposed on the first-order-exact gradient **before** the second pass differentiates it,
            and on the boundary faces those cells own -- so the Hessian is built from the imposed
            gradient rather than from the estimate it replaces. That ordering is the whole reason
            this is an argument to the reconstruction and not a correction applied to its result.

        Returns
        -------
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``; exact for quadratic fields where nothing is
            imposed, and exactly the imposed value where something is.
        hessian : jnp.ndarray
            Cell Hessians in independent components, shape ``(n_cells, n_sym)``, in
            :func:`~aquaflux.schemes.expand_symmetric`'s order; exact for quadratic fields and
            first-order accurate in general.

        Notes
        -----
        With ``imposed`` given the reconstruction is **affine** in ``field`` rather than linear: the
        imposed values are an added constant, so the map no longer sends zero to zero. Its tangent
        is still the same fixed sequence of fixed linear maps applied to the tangent, with no
        implicit-function solve -- which is the property that matters inside a differentiated solve.
        What is given up is only that the map can no longer be recovered by evaluating it on a
        tangent.
        """
        prepared = self.prepared
        if prepared is None:
            prepared = _build_corrections(mesh, geometry, self.boundary_closure)
        face_cells, dim = mesh.face_cells, mesh.dim
        factor = interpolation_factor(face_cells, geometry)
        area = scale(geometry.face.normal, geometry.face.area)

        first = _one_exact(
            field, boundary_values, prepared.m1_inverse, factor, face_cells, area, geometry
        )
        face_gradient = self.boundary_closure.face_gradient(
            first, field, boundary_values, face_cells, geometry
        )
        if imposed is not None:
            # Before the second pass, not after it. That pass differentiates `first` and closes the
            # boundary from it, so an imposition applied to the returned gradient would arrive after
            # both of its consumers had already read the estimate it replaces.
            first = imposed.impose(first)
            face_gradient = imposed.impose_on_faces(face_gradient, face_cells)
        raw = _one_exact(
            first, face_gradient, prepared.m1_inverse, factor, face_cells, area, geometry
        )
        hessian = jnp.einsum(
            "nab,nb->na", prepared.m2_inverse, contract_symmetric(_symmetrize(raw), dim)
        )
        gradient = first - jnp.einsum("nia,na->ni", prepared.gradient_defect, hessian)
        # Again at the end: the second-order correction is a defect of the *reconstruction*, and
        # subtracting it from an imposed value would corrupt the very number the caller imposed.
        return (gradient if imposed is None else imposed.impose(gradient)), hessian


def _symmetrize(tensor: jnp.ndarray) -> jnp.ndarray:
    """Average a tensor with its transpose.

    The numerical cross-derivatives do not satisfy the equality of mixed partials to machine
    accuracy on an irregular grid — differentiating in one order and then the other traverses
    different cells — so the two halves are averaged before the symmetric components are read off.
    """
    return 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))


def _green_gauss(
    cell_values: jnp.ndarray,
    face_values: jnp.ndarray,
    factor: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    area: jnp.ndarray,
    geometry: MeshGeometry,
) -> jnp.ndarray:
    """The raw Green--Gauss sum ``(1/V) sum_f interp(u) (x) A_f``, appending a gradient axis.

    Not consistent on a distorted mesh, and deliberately so: what it returns for a known field is
    precisely what the correction matrices are recovered from.
    """
    interior = interpolate_owner_neighbour(cell_values, factor, face_cells)
    value = face_cells.combine_face_values(interior, face_values)
    contribution = value[..., None] * area.reshape(area.shape[0], *(1,) * (value.ndim - 1), -1)
    volume = geometry.cell.volume
    total = face_cells.scatter_conservative(contribution)
    return total / volume.reshape(volume.shape[0], *(1,) * (total.ndim - 1))


def _one_exact(
    cell_values: jnp.ndarray,
    face_values: jnp.ndarray,
    m1_inverse: jnp.ndarray,
    factor: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    area: jnp.ndarray,
    geometry: MeshGeometry,
) -> jnp.ndarray:
    """``D1 = M1^-1 R`` — the Green--Gauss sum made exact for linear fields."""
    raw = _green_gauss(cell_values, face_values, factor, face_cells, area, geometry)
    return jnp.einsum("nij,n...j->n...i", m1_inverse, raw)


def _build_corrections(
    mesh: Mesh, geometry: MeshGeometry, closure: GradientBoundaryClosure
) -> Corrections:
    """Recover the three correction matrices by running the operators on coordinate monomials.

    Nothing here is a derived geometric formula. ``M1`` is *defined* as what the raw sum returns for
    each coordinate field, and ``M2`` and the gradient defect as what the composed operators return
    for each quadratic basis field — so the corrections cannot drift from the operators they
    correct, in the way a separately-derived expression could.

    ⚠️ **The probes run through the same boundary closure the reconstruction will use.** Building a
    correction with exact boundary data and applying it with a real closure corrects an operator
    nobody evaluates; the mismatch reads as the closure destroying exactness, which is a
    considerably more alarming symptom than its cause.

    The coordinates are centred on the mesh before the monomials are formed. The raw sum annihilates
    constants, so this cannot change any correction — it only keeps the monomial magnitudes
    comparable to the cell size instead of to the distance from an arbitrary origin.
    """
    dim = mesh.dim
    n_sym = symmetric_components(dim)
    face_cells = mesh.face_cells
    factor = interpolation_factor(face_cells, geometry)
    area = scale(geometry.face.normal, geometry.face.area)

    centroid = geometry.cell.centroid
    origin = jnp.mean(centroid, axis=0)
    cell_x = centroid - origin
    face_x = geometry.face.centroid - origin

    def raw(cell_values, face_values):
        return _green_gauss(cell_values, face_values, factor, face_cells, area, geometry)

    # M1: what the raw sum returns for each coordinate field. Its columns are indexed by the
    # direction probed, its rows by the gradient component returned.
    m1 = jnp.stack([raw(cell_x[:, i], face_x[:, i]) for i in range(dim)], axis=-1)
    m1_inverse = jnp.linalg.inv(m1)

    def one_exact(cell_values, face_values):
        return jnp.einsum("nij,n...j->n...i", m1_inverse, raw(cell_values, face_values))

    basis = expand_symmetric(jnp.eye(n_sym), dim)  # (n_sym, dim, dim)
    defect_columns, m2_columns = [], []
    for component in basis:
        # psi(x) = 1/2 x . E . x, whose exact gradient is E x and whose exact Hessian is E.
        psi_cell = 0.5 * jnp.einsum("ni,ij,nj->n", cell_x, component, cell_x)
        psi_face = 0.5 * jnp.einsum("ni,ij,nj->n", face_x, component, face_x)
        first = one_exact(psi_cell, psi_face)
        defect_columns.append(first - cell_x @ component.T)
        face_gradient = closure.face_gradient(first, psi_cell, psi_face, face_cells, geometry)
        second = one_exact(first, face_gradient)
        m2_columns.append(contract_symmetric(_symmetrize(second), dim))

    return Corrections(
        m1_inverse=m1_inverse,
        m2_inverse=jnp.linalg.inv(jnp.stack(m2_columns, axis=-1)),
        gradient_defect=jnp.stack(defect_columns, axis=-1),
    )
