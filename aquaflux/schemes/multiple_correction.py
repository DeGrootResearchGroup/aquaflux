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

⚠️ **A closure that differences the boundary value needs that value to be a PRESCRIBED one.** A
reconstruction is fed boundary values evaluated at *zero* gradient, so the residual stays a
single-pass function of the field. A prescribed value is unaffected by that; a gradient-type
condition is not, since its whole content is a correction term which is then evaluated away --
leaving the boundary value equal to the owner cell's own. A one-sided closure that subtracts the
non-orthogonal correction anyway has corrected twice, and dividing the residue by the wall-normal
distance turns it into a large spurious derivative. ``boundary_values_at`` is the seam that repairs
that, by asking the caller to re-evaluate its closures at the reconstruction's own gradient.

⚠️ **Repairing it does not make such a closure safe on a gradient-type patch, and the reason is
structural.** Once the boundary value carries its correction the one-sided difference is exactly
zero, so the closure replaces that face's **normal** gradient component with a hard zero -- and a cell
owning two boundary faces has two independent normal directions replaced, over-constraining the very
Hessian :class:`OwnerGradient` leaves *under*-constrained at such a cell for the opposite reason.
Neither the corrections nor the check on them can see it, because both are probed against exact face
values rather than the boundary conditions'. See :class:`SkewCorrectedGradient` for what that costs on
a real march, and why :class:`OwnerGradient` is the safe choice on any mesh whose patches are not all
Dirichlet.

Separately: some cells' gradient is not to be closed at all but *known*, the near-wall ``omega`` of a
k--omega closure being the standing case. The seam for that is
:class:`~aquaflux.schemes.ImposedGradient`, which supersedes both the reconstruction at those cells
and the closure on the boundary faces they own. It is not a remedy for the paragraph above -- it
settles one field, where that defect is in the closure's arithmetic and reaches every field.
"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import equinox as eqx
import jax
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
    attached, so it cannot tell a prescribed value from a gradient-type one evaluated at leading
    order.** The two need different treatment and look identical here: see
    :class:`SkewCorrectedGradient`, where differencing the second kind corrects twice. Until a
    closure can be told the patch kinds, an implementation that reads the boundary value is
    restricted to problems whose patches all prescribe values.
    """

    #: Whether :meth:`face_gradient` reads ``boundary_values`` at all. A closure that does needs them
    #: **corrected** -- re-evaluated at the reconstruction's own gradient -- and a reconstruction then
    #: pays one boundary-value evaluation to supply them. A closure that does not is spared that, and
    #: is also immune to the leading-order trap the correction exists to route around.
    #:
    #: A plain class attribute rather than a dataclass field, deliberately: a defaulted field here
    #: would force every field of every subclass to carry a default too, so a closure that needs
    #: required state -- a set of cells, say -- could not be written at all.
    reads_boundary_values = True

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

    ⚠️ **It fails on a tetrahedron with TWO OR MORE boundary faces** -- a corner or edge tet, not
    merely a tet at a boundary. A boundary face closed with the owner's own gradient carries no
    direction the cell did not already have, so such a cell is left with two informative faces
    against six Hessian components and ``M2`` is singular to working precision. Measured on a
    perturbed tetrahedral mesh, resolved by boundary-face count: cells with **0 or 1** boundary face
    reconstruct a quadratic to ``7e-15`` with ``max |M2^-1|`` of 2--4.5; the eighteen cells with
    **2** are wrong by **173 %** at ``1.6e16``. No other cell is affected.

    That distinction is worth keeping, because the common case is on the safe side of it: on the
    1.6M-cell snappyHexMesh reactor **all 3,063 four-faced cells have exactly one boundary face**,
    and this closure reconstructs a quadratic there to ``4.9e-12`` -- as well as
    :class:`SkewCorrectedGradient` does, and better conditioned. :meth:`MultipleCorrectionGradient.bind`
    measures ``M2`` and warns rather than leaving this to be inferred from cell shapes.
    """

    reads_boundary_values = False

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

    ⚠️⚠️ **IT LOSES A COUPLED RANS MARCH THAT THREE OTHER RECONSTRUCTIONS COMPLETE.** On a separating
    turbulent benchmark it clears two Reynolds rungs at a full step and then loses the target rung,
    where corrected Green--Gauss, the Hessian-corrected scheme and this scheme under
    :class:`OwnerGradient` all reach the same answer. Measured at the iterate where that happens, all
    four at the same state and the same pseudo-time shift, the largest correction to ``log omega``
    resolved by how many boundary faces a cell owns -- beside the step length the line search then
    keeps:

    ============================  ==============  ==============  ==============  =============
    reconstruction                interior        1 face          corners         kept step
    ============================  ==============  ==============  ==============  =============
    corrected Green--Gauss        0.456           1.36            1.91            1 -> 0.128
    Hessian-corrected             0.438           4.71            9.83            0.5 -> 0.605
    this scheme, owner closure    0.454           6.64            15.6            0.25 -> 0.783
    this scheme, one-sided        0.454           16.1            58.4            0.0625 -> 0.955
    ============================  ==============  ==============  ==============  =============

    **The interior column is the control and it is flat** -- a 4 % spread over four structurally
    different reconstructions -- while the boundary columns span thirty-fold and the kept step orders
    identically. So the difference is made entirely at cells owning a boundary face, and worst at
    cells owning two. Since ``omega`` is transported in log form, an entry of 58 means that cell's
    ``omega`` is multiplied by ``e**58``, and halving the step length only takes the square root of
    that factor -- so the search can reach an admissible step but not one long enough to be worth
    taking, which is what the march rejects.

    **Those wall-adjacent rows are not the omega equation, and that is what closes the chain.** They
    are replaced by the algebraic near-wall fixation ``log omega = log omega_wall(k)``, whose
    log-layer branch carries ``sqrt(k)`` -- so the row's only two entries are a one on its own unknown
    and ``-d(log omega_target)/dk``, and the correction there is the identity
    ``d log omega = -R_row + (d log omega_target / dk) * dk``. Measured on that benchmark, the ``k``
    term reproduces the whole correction to within 1e-5 in every one of the four arms: the wall cells'
    ``omega`` correction is simply whatever the ``k`` correction asks for. What the reconstruction
    changes is the velocity gradient those wall cells see, hence their ``k`` production, hence ``dk``
    -- which on that state reaches **four thousand times** the local ``k``, in the **increasing**
    direction that a fraction-to-the-boundary positivity limit does not bound (it guards only entries
    that could cross zero, and the decreasing side is an identical ``-0.27`` in all four arms).

    **The reading of that ordering** -- offered as a reading, not separately isolated -- is that the
    four differ in how much of the returned gradient comes from a *boundary face's* gradient.
    Corrected Green--Gauss forms no Hessian, so a boundary face contributes its value and nothing
    else. The other three build a Hessian from a second pass over the gradient field, where a boundary
    face contributes ``face_gradient * A / V``, and at a wall cell ``A / V`` is of order the inverse
    wall-normal distance. What the two closures differ in is exactly the boundary face gradient they
    supply: :class:`OwnerGradient` hands over the full cell gradient, while this one -- once the
    boundary value carries its own correction -- differences to **identically zero** on a
    gradient-type patch, replacing that face's normal component with a hard zero. A cell owning two
    boundary faces has two independent normal directions so replaced.

    ⚠️ :func:`_undetermined_cells` cannot see any of this: ``M2`` is probed with the exact quadratic
    evaluated at face centroids on every patch, i.e. face values carrying real nonzero normal
    derivatives, so the probed operator and the applied one differ on exactly the gradient-type
    patches.

    Use :class:`OwnerGradient`, which is the default, unless the mesh has boundary **tetrahedra**,
    where the owner closure leaves the Hessian underdetermined and this is the only shipped
    alternative. On such a mesh, watch convergence.

    **Three mechanisms were proposed and refuted while this was being chased**; do not re-propose
    them. All three were measured either at a cold initial condition or on a Reynolds-rung anchor, and
    the two closures are *identical* at every such state -- which is why they all missed:

    * *The near-wall* ``omega`` *gradient.* Imposing the analytical one on the wall cells and their
      boundary faces, and again inside the omega equation's own assembler, reproduces the stall to
      three figures.
    * *The correction matrices' conditioning.* On that mesh ``cond(M2^-1)`` is 2.14 under this
      closure against 3.49 under the owner one -- this is the better-conditioned of the two.
    * *Correcting twice on a gradient-type patch.* That was a real defect and it is now fixed (see
      below); the artifact it produced fell by eighteen orders and the march is unchanged.

    ⚠️ **The double correction, since fixed, is still worth understanding, because a closure that
    reads a boundary value inherits it.** A reconstruction is fed boundary values evaluated at
    **zero** gradient, so the residual stays a single-pass function of the field. A prescribed value
    does not depend on the gradient and is unaffected. A gradient-type condition is: ``ZeroGradient``
    returns ``phi_owner + tangential correction``, whose correction is evaluated away, leaving the
    boundary value exactly ``phi_owner``. Differencing that subtracts a correction nothing added, and
    dividing by ``d.n`` -- the smallest distance in the mesh at a wall -- turns the residue into a
    large spurious derivative: measured at ``3e7`` on an outlet and ``1e5`` at a wall. Declaring
    :attr:`~GradientBoundaryClosure.reads_boundary_values` is what asks the reconstruction for the
    **corrected** values instead, which collapses those to roundoff while leaving a Dirichlet patch
    untouched (``validation/multiple_correction/boundary_closure_probe.py`` measures both).
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


class CellwiseFallback(GradientBoundaryClosure):
    """One closure everywhere, another on named cells.

    The two shipped closures fail in opposite regimes, and the regimes are *local*:
    :class:`OwnerGradient` is undetermined on a tetrahedron with two or more boundary faces and
    exact everywhere else, while :class:`SkewCorrectedGradient` supplies the missing direction but
    reads a boundary value, which is the thing a coupled RANS march is measured to stall under.
    Since the cells that need the second one can be identified -- by measuring the correction the
    first produces, not by counting faces -- neither has to be chosen for the whole mesh.

    Built by :meth:`MultipleCorrectionGradient.bind` when it finds cells its closure cannot
    determine, so it is not usually constructed directly. On a mesh with no such cells nothing is
    built and the reconstruction is byte-identical to the primary alone.

    Attributes
    ----------
    cells : jnp.ndarray
        Cells whose boundary faces take :attr:`secondary`, shape ``(n_fallback,)``.
    primary : GradientBoundaryClosure
        Used on every other boundary face.
    secondary : GradientBoundaryClosure
        Used on the boundary faces of :attr:`cells`.
    """

    cells: jnp.ndarray
    primary: GradientBoundaryClosure
    secondary: GradientBoundaryClosure

    @property
    def reads_boundary_values(self) -> bool:
        """``True`` if either side reads them -- the corrected values are then needed for that side."""
        return self.primary.reads_boundary_values or self.secondary.reads_boundary_values

    def face_gradient(self, gradient, field, boundary_values, face_cells, geometry):
        primary = self.primary.face_gradient(gradient, field, boundary_values, face_cells, geometry)
        secondary = self.secondary.face_gradient(
            gradient, field, boundary_values, face_cells, geometry
        )
        take = jnp.zeros(face_cells.n_cells, dtype=bool).at[self.cells].set(True)
        return jnp.where(take[face_cells.owner][:, None], secondary, primary)


class Corrections(eqx.Module):
    """The per-cell correction matrices, built once for one geometry.

    All three are geometry-only, so they are the whole of what
    :meth:`MultipleCorrectionGradient.bind` holds.

    Attributes
    ----------
    m1 : jnp.ndarray
        ``M1``, shape ``(n_cells, dim, dim)`` -- what the raw Green--Gauss sum returns for a linear
        field of unit gradient in each coordinate direction. Carried as well as its inverse because
        :meth:`MultipleCorrectionGradient.reconstruct`'s boundary extrapolation adds a per-cell term
        to it, and *which* patches that term covers is a property of the field being reconstructed
        rather than of the geometry -- so it cannot be folded in here.
    m1_inverse : jnp.ndarray
        ``M1^-1``, shape ``(n_cells, dim, dim)``; used directly when no extrapolation is asked for.
    m2_inverse : jnp.ndarray
        ``M2^-1``, shape ``(n_cells, n_sym, n_sym)``. ``M2`` is what the doubly-applied 1-exact
        operator returns for each quadratic basis field.
    gradient_defect : jnp.ndarray
        The 1-exact gradient's first-order error per unit Hessian, shape ``(n_cells, dim, n_sym)``;
        subtracting it lifts that gradient to second order.
    closure : GradientBoundaryClosure
        The closure these were probed through, and therefore the one the reconstruction must apply --
        which is not always the one that was asked for, since a mesh with cells the requested closure
        cannot determine is repaired with a :class:`CellwiseFallback`. Carrying it here is what keeps
        the correction and the operator it corrects from ever disagreeing.
    """

    m1: jnp.ndarray
    m1_inverse: jnp.ndarray
    m2_inverse: jnp.ndarray
    gradient_defect: jnp.ndarray
    closure: GradientBoundaryClosure


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
        not its derivative. Defaults to :class:`OwnerGradient`, which never reads a boundary value
        and is the one that marches a coupled RANS case (measured; see :class:`SkewCorrectedGradient`
        for what happens otherwise, and for the caveat that it is the closure to reach for on a mesh
        with boundary **tetrahedra**, where the owner closure leaves the Hessian underdetermined).
    fallback : GradientBoundaryClosure or None
        Used on the boundary faces of any cell :attr:`boundary_closure` cannot determine, leaving
        every other cell alone (:class:`CellwiseFallback`). Defaults to ``None``, which leaves such
        cells amplifying and warns rather than repairing them. :class:`SkewCorrectedGradient`
        supplies the direction a corner tetrahedron is short of and repairs them, but it also reads
        a boundary value on every boundary face it is installed on — including every one of the
        undetermined cells' own — and that closure is measured to destabilize a coupled march badly
        at an unconverged iterate on exactly those cells (see its own docstring). A repair chosen
        without knowing the mesh needs it is therefore not a safe default: pass it explicitly once
        you have looked at the mesh and decided the corner-tetrahedron accuracy is worth that risk.
        On a mesh with no undetermined cells nothing is built either way and the reconstruction is
        byte-identical regardless of this setting — measured on quadrilateral and hexahedral meshes,
        and on a 1.6M-cell snappyHexMesh mesh.
    prepared : Corrections or None
        The geometry-only correction matrices, present once :meth:`bind` has been called. ``None``
        rebuilds them on every reconstruction, which is correct but wasteful — an assembler binds
        the scheme it is handed, so ordinary use never pays that. It also carries the closure they
        were probed through, which is what the reconstruction applies — not always the one asked
        for, since a repaired mesh gets a :class:`CellwiseFallback`.
    """

    boundary_closure: GradientBoundaryClosure = eqx.field(default_factory=OwnerGradient)
    fallback: GradientBoundaryClosure | None = None
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
            fallback=self.fallback,
            prepared=_build_corrections(mesh, geometry, self.boundary_closure, self.fallback),
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
        boundary_values_at: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        boundary_chain: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if operator_hook is not None:
            raise NotImplementedError(
                "MultipleCorrectionGradient cannot yet run domain-decomposed. Unlike the coupled "
                "Hessian scheme this is not a structural bar -- the reconstruction exchanges one "
                "ring per pass, so a correct distributed build needs the reconstructed gradient "
                "halo-exchanged once between the two passes. `operator_hook` is the wrong seam for "
                "that: it refreshes a solve's unknown, and there is no solve here."
            )
        return self.reconstruct(
            field,
            mesh,
            geometry,
            boundary_values,
            imposed=imposed,
            boundary_values_at=boundary_values_at,
            boundary_chain=boundary_chain,
        )[0]

    def reconstruct(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
        *,
        imposed: ImposedGradient | None = None,
        boundary_values_at: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        boundary_chain: jnp.ndarray | None = None,
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
        boundary_chain : jnp.ndarray, optional
            ``d(boundary value)/d(phi_owner)`` per face, shape ``(n_faces,)`` -- one on a patch whose
            value is derived from the owner cell (zero-gradient, Neumann, Robin), zero on a prescribed
            one. **Supply it whenever any patch is not a prescribed value.** The first pass then reads
            the value consistent with the cell's own field on those patches instead of one asserting a
            normal derivative the iterate does not yet have, which is otherwise a **linear**-order
            error the correction matrices cannot remove -- 57--62 % of the boundary-cell gradient,
            independently of mesh spacing, with a Hessian that doubles on every refinement. See
            :func:`_extrapolated_first_pass`; it costs one per-cell inverse and no extra pass.
            ``None`` (default) is byte-identical to not passing it.

            ⚠️ It reaches the **first pass only**. ``M2`` and the gradient defect are still probed
            against exact face values, so a quadratic keeps a second-order inconsistency there --
            measured at 1.6 % of the boundary-cell gradient at 8 cells across and falling as ``h``
            (0.76 %, 0.37 % at 16 and 32), against the 58 % that does *not* fall without this.
        boundary_values_at : callable, optional
            ``gradient -> boundary_values``: the caller's boundary closures re-evaluated at a
            reconstructed gradient. **Supply this whenever the boundary values passed in were
            evaluated at zero gradient**, which is what a residual assembler does. The correction
            matrices are probed against *exact* face values, so a closure fed anything else is
            correcting an operator the probes never saw -- and on a gradient-type patch the
            difference is precisely the term :class:`SkewCorrectedGradient` then divides by the
            wall-normal distance. ``None`` uses ``boundary_values`` as given.

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
            prepared = _build_corrections(mesh, geometry, self.boundary_closure, self.fallback)
        # The closure the corrections were PROBED through, which a repaired mesh makes different
        # from the one asked for. Applying the requested one here would correct an operator nobody
        # evaluates -- the exact mistake this module warns about elsewhere.
        closure = prepared.closure
        face_cells, dim = mesh.face_cells, mesh.dim
        factor = interpolation_factor(face_cells, geometry)
        area = scale(geometry.face.normal, geometry.face.area)

        # The first pass reads boundary VALUES, so a value asserting a normal derivative the iterate
        # does not have is a linear-order error here -- before any closure is consulted, and in every
        # closure alike. `boundary_chain` is what lets it read the cell's own extrapolation instead.
        m1_inverse, first_values = (
            (prepared.m1_inverse, boundary_values)
            if boundary_chain is None
            else _extrapolated_first_pass(
                prepared.m1, boundary_chain, field, boundary_values, face_cells, geometry
            )
        )
        first = _one_exact(field, first_values, m1_inverse, factor, face_cells, area, geometry)
        # The closure DIFFERENTIATES a boundary value, so it needs the corrected one: a
        # gradient-type condition carries its whole content in a correction that a zero-gradient
        # evaluation throws away, leaving the face value equal to the owner's and the closure
        # subtracting a term nothing added. The first pass above deliberately keeps the values as
        # given -- there the error is a value of order the correction, not one divided by `d.n`.
        # Skipped entirely for a closure that reads no boundary value: the re-evaluation is a
        # scatter over every boundary face, and `OwnerGradient` would discard the result.
        needs_correcting = boundary_values_at is not None and closure.reads_boundary_values
        closure_values = boundary_values_at(first) if needs_correcting else boundary_values
        if boundary_chain is not None and closure.reads_boundary_values:
            # The closure gets the SAME extrapolation the first pass got, for the same reason and
            # with a sharper consequence. Differencing a value that asserts a normal derivative the
            # iterate does not have makes the closure report that asserted derivative rather than the
            # field's: on a zero-gradient patch it returns the owner gradient with its normal
            # component replaced by a hard zero. Fed `phi_P + grad phi . d` instead, the difference is
            # `(grad phi . n)(d . n)` and the closure returns the owner gradient entire -- linear-exact,
            # which is the one property this module states a closure may not do without.
            chain = jnp.where(face_cells.interior, 0.0, boundary_chain)
            displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
            extrapolated = field[face_cells.owner] + dot(first[face_cells.owner], displacement)
            closure_values = chain * extrapolated + (1.0 - chain) * closure_values
        face_gradient = closure.face_gradient(first, field, closure_values, face_cells, geometry)
        if imposed is not None:
            # Before the second pass, not after it. That pass differentiates `first` and closes the
            # boundary from it, so an imposition applied to the returned gradient would arrive after
            # both of its consumers had already read the estimate it replaces.
            first = imposed.impose(first)
            face_gradient = imposed.impose_on_faces(face_gradient, face_cells)
        # The geometry's own `M1^-1` here, NOT the extrapolated one: `M2` and the gradient defect were
        # probed through this operator, and correcting an operator nobody evaluates is the mistake this
        # module warns about. The extrapolation is a statement about the FIELD's boundary values; the
        # second pass differentiates a gradient, whose boundary faces the closure supplies.
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


def _extrapolated_first_pass(
    m1: jnp.ndarray,
    boundary_chain: jnp.ndarray,
    field: jnp.ndarray,
    boundary_values: jnp.ndarray,
    face_cells: FaceCellConnectivity,
    geometry: MeshGeometry,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(M1 - B)^-1`` and the boundary values that go with it, for the extrapolated first pass.

    **Why the first pass needs anything at all.** A reconstruction is handed the field's boundary
    values, and on a patch whose value is derived from the owner cell -- zero-gradient, and any
    Neumann or Robin condition -- that value asserts a normal derivative the *iterate* does not have.
    An iterate does not satisfy its own boundary conditions until it has converged, so the value and
    the interior field disagree, the Green--Gauss sum averages the two, and the cell gradient comes
    out wrong at **linear** order. Measured on a linear field over a zero-gradient patch, the
    reconstructed boundary-cell gradient is wrong by 57 % under
    :class:`OwnerGradient` and 62 % under :class:`SkewCorrectedGradient`, **independently of mesh
    spacing**, while the Hessian the sum reports *doubles with every refinement* -- it is faithfully
    reporting an ever-sharper kink that the field does not have.

    The repair is to give the first pass the value consistent with the cell's own field,
    ``phi_P + grad phi . d``, on exactly those patches, and to leave a prescribed value alone. That is
    the correction the boundary condition would carry if the iterate satisfied it, and the two agree
    at convergence -- the condition is still imposed, by the flux, which is where it belongs.

    **Why it costs nothing.** That value depends on the gradient being reconstructed, so read
    literally it is a fixed point. But it is *linear* in the gradient: each such face contributes
    ``(grad phi_P . d) A_f`` to the sum, i.e. ``B_P grad phi_P`` for a per-cell matrix. So it moves to
    the other side, and the pass is ``(M1 - B) grad phi = raw`` -- one per-cell inverse, exactly what
    ``M1`` already was. Iterating it instead converges at a mesh-independent 0.571 per pass and would
    need eight or ten of them.

    Parameters
    ----------
    m1 : jnp.ndarray
        The geometry's ``M1``, shape ``(n_cells, dim, dim)``.
    boundary_chain : jnp.ndarray
        ``d(boundary value)/d(phi_owner)`` per face, shape ``(n_faces,)`` -- one on a patch whose
        value is derived from the owner cell, zero on a prescribed one, and in between for a Robin
        condition, which is then blended in the same proportion. Interior entries are ignored.
    field : jnp.ndarray
        Cell values, shape ``(n_cells,)``.
    boundary_values : jnp.ndarray
        The field's boundary values as the caller formed them, shape ``(n_faces,)``.
    face_cells, geometry
        The connectivity and metrics.

    Returns
    -------
    inverse : jnp.ndarray
        ``(M1 - B)^-1``, shape ``(n_cells, dim, dim)``.
    values : jnp.ndarray
        The gradient-independent part of the extrapolated boundary values, shape ``(n_faces,)`` --
        the rest of the extrapolation is what ``B`` accounts for.
    """
    chain = jnp.where(face_cells.interior, 0.0, boundary_chain)
    displacement = geometry.face.centroid - geometry.cell.centroid[face_cells.owner]
    area = scale(geometry.face.normal, geometry.face.area)
    # B_P = sum over the cell's own boundary faces of chain * (A_f (x) d) / V, scattered to the owner.
    # `scatter_conservative` is the same accumulation the sum itself uses, so the two cannot disagree
    # about which faces belong to which cell.
    outer = displacement[:, :, None] * area[:, None, :]
    b = face_cells.scatter_conservative(chain[:, None, None] * outer)
    b = jnp.swapaxes(b, 1, 2) / geometry.cell.volume[:, None, None]
    values = chain * field[face_cells.owner] + (1.0 - chain) * boundary_values
    return jnp.linalg.inv(m1 - b), values


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


#: Above this, a cell's Hessian correction is amplifying rather than repairing. The matrices are
#: probed on quadratic monomials of order the cell size, so a healthy inverse is order one --
#: measured at 2--19 across quadrilateral, hexahedral and tetrahedral meshes, and 1.86e+01 on a
#: 1.6M-cell snappyHexMesh mesh. A cell left underdetermined runs to 1e16 instead, so anything in
#: between is a wide margin rather than a tuned threshold.
_UNDETERMINED_CORRECTION = 1e8

_FALLBACK_WARNED = False


def _undetermined_cells(m2_inverse: jnp.ndarray) -> jnp.ndarray | None:
    """Which cells' Hessian correction is singular enough to amplify, or ``None`` if none are.

    Measured on the correction itself rather than inferred from cell shape, which is the distinction
    that matters: "a boundary tetrahedron" over-predicts badly. Resolved by boundary-face count on a
    perturbed tetrahedral mesh, cells with **one** boundary face reconstruct a quadratic to 7.2e-15
    while those with **two** are wrong by 173 %, and on a real snappyHexMesh mesh every four-faced
    cell has exactly one -- so a shape count would repair thousands of cells that are already exact
    and miss nothing in return.

    Returns ``None`` when the values are traced, since a traced build cannot be inspected without
    forcing it, and when nothing is wrong -- which is the common case and the one that must cost
    nothing.
    """
    if isinstance(jnp.asarray(m2_inverse), jax.core.Tracer):
        return None
    worst = jnp.max(jnp.abs(m2_inverse), axis=(1, 2))
    # ⚠️ NOT `worst > limit`: a singular inverse is large on one platform and NON-FINITE on another
    # (measured -- the same tetrahedral mesh gives 1e16 under macOS Accelerate and NaN under the
    # BLAS on CI), and `NaN > limit` is False. Testing the negation catches both, where the obvious
    # comparison would silently decline to repair exactly the cells that need it most.
    cells = jnp.flatnonzero(~(worst <= _UNDETERMINED_CORRECTION))
    return cells if cells.size else None


def _warn_repaired(cells: jnp.ndarray, primary, secondary, total: int) -> None:
    """Say once that the closure was repaired, since it is not what the caller asked for."""
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(
        f"MultipleCorrectionGradient: {cells.size} of {total} cells leave the Hessian "
        f"underdetermined under {type(primary).__name__} -- a tetrahedron with two or more boundary "
        f"faces is the usual cause -- so {type(secondary).__name__} is used on those cells' boundary "
        f"faces and the correction rebuilt. Every other cell is unchanged. Pass `fallback=None` to "
        f"get the unrepaired reconstruction and this warning instead.",
        stacklevel=2,
    )


def _warn_unrepairable(cells: jnp.ndarray, closure, total: int) -> None:
    """Say once that the correction is singular and nothing here can repair it."""
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    warnings.warn(
        f"MultipleCorrectionGradient: {cells.size} of {total} cells leave the Hessian "
        f"underdetermined under {type(closure).__name__} (a tetrahedron with two or more boundary "
        f"faces is the usual cause), and no fallback closure was given (or the fallback is the same "
        f"closure). The reconstruction amplifies in those cells instead of being exact for "
        f"quadratics there. Passing `fallback=SkewCorrectedGradient()` repairs it, at the cost of "
        f"reading a boundary value on those cells' faces -- which SkewCorrectedGradient's own "
        f"docstring documents destabilizing a coupled march at an unconverged iterate, so weigh that "
        f"before opting in.",
        stacklevel=2,
    )


def _build_corrections(
    mesh: Mesh,
    geometry: MeshGeometry,
    closure: GradientBoundaryClosure,
    fallback: GradientBoundaryClosure | None = None,
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

    m2_inverse = jnp.linalg.inv(jnp.stack(m2_columns, axis=-1))

    # Repair, rather than merely report: the cells a closure cannot determine are identifiable, so
    # neither closure has to be chosen for the whole mesh. Nothing is rebuilt when nothing is wrong,
    # which is the common case -- so a healthy mesh pays one comparison, not a second build.
    undetermined = _undetermined_cells(m2_inverse)
    if undetermined is not None:
        if fallback is None or type(fallback) is type(closure):
            _warn_unrepairable(undetermined, closure, mesh.n_cells)
        else:
            _warn_repaired(undetermined, closure, fallback, mesh.n_cells)
            return _build_corrections(
                mesh, geometry, CellwiseFallback(undetermined, closure, fallback)
            )

    return Corrections(
        m1=m1,
        m1_inverse=m1_inverse,
        m2_inverse=m2_inverse,
        gradient_defect=jnp.stack(defect_columns, axis=-1),
        closure=closure,
    )
