"""A compact gradient reconstruction: per-cell stencil weights, exact for quadratics.

Every linear gradient reconstruction is a fixed set of weights per cell — ``g_P = sum_j w_Pj phi_j``
over the cells of a stencil and the data on the boundary faces they own. Two facts about those
weights decide whether a reconstruction is usable in a collocated pressure–velocity solve, and they
are separate:

* **What it reproduces.** Weights that integrate every polynomial up to quadratic exactly make the
  reconstruction second-order.
* **How large they are.** The Rhie--Chow pressure coupling and the non-orthogonal diffusion correction
  are both *cancelling* differences: a compact two-point difference minus the reconstructed gradient's
  contribution. When a cell's weights are large and opposing, the correction overshoots the difference
  it corrects, the coefficient changes sign, and the term anti-damps instead of damping.

Exactness leaves the weights far from determined — on a tetrahedron's two-hop stencil it fixes ten
numbers out of about thirteen per gradient component — so a scheme is free to choose badly within it,
and one can be exact and still anti-damp. This scheme makes the choice explicitly: of all the weights
on the stencil that are exact for quadratics, take the ones nearest to a reference reconstruction
known to damp, and stay compact.

The reference is one block sweep of the coupled gradient--Hessian system of Betchen and Straatman
(2010) (:class:`~aquaflux.schemes.HessianCorrectedGradient`'s system), which costs one pass over the
faces and a per-cell block solve. :attr:`ProjectedStencilGradient.blend` is how much of it the weights
aim at: ``0`` gives the minimum-norm exact weights (an unweighted quadratic least-squares fit on the
stencil), ``1`` the nearest exact weights to the reference itself.

Everything is built once per field in :meth:`ProjectedStencilGradient.bind` and applied at run time as
one gather and one contraction per cell — no solve, and the same stencil a two-pass compact scheme
already couples, so a residual's Jacobian gains no reach.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from aquaflux.mesh import Mesh, MeshGeometry
from aquaflux.vectors import scale

from .gradient import (
    AveragedNeighbourHessian,
    BoundaryLinearization,
    GradientScheme,
    HessianBoundaryClosure,
    HessianCorrectedGradient,
    ImposedGradient,
)
from .interpolation import interpolation_factor

#: Above this, a stencil's monomials are too nearly dependent for the exactness constraints to be
#: solved reliably, and the weights they produce can be large. Reported, not fatal: the conditioning
#: is a property of the mesh, and a caller with a worse mesh than this threshold assumes should see
#: the number rather than have the scheme refuse.
_ILL_CONDITIONED = 1e6

_WARNED: set[str] = set()


def _warn_once(kind: str, message: str) -> None:
    """Warn once per process for each distinct ``kind`` — a scheme is bound once per field, and the
    same mesh property would otherwise be reported once per field."""
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    warnings.warn(message, UserWarning, stacklevel=3)


def polynomial_basis(offsets: jnp.ndarray, dim: int) -> jnp.ndarray:
    """The monomials up to quadratic at ``offsets``, shape ``(..., n_terms)``.

    ``n_terms`` is ``1 + dim + dim (dim + 1) / 2`` — six in two dimensions, ten in three: the constant,
    the coordinates, and the distinct products of two coordinates.

    Parameters
    ----------
    offsets : jnp.ndarray
        Positions relative to the cell, scaled by its size, shape ``(..., dim)``.
    dim : int
        Spatial dimension.

    Returns
    -------
    jnp.ndarray
        The monomials, shape ``(..., n_terms)``.
    """
    terms = [jnp.ones_like(offsets[..., 0])]
    terms.extend(offsets[..., i] for i in range(dim))
    terms.extend(offsets[..., i] * offsets[..., j] for i in range(dim) for j in range(i, dim))
    return jnp.stack(terms, axis=-1)


def polynomial_basis_gradient(offsets: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Each monomial's gradient at ``offsets``, shape ``(..., n_terms, dim)`` — the companion of
    :func:`polynomial_basis`, for a face whose datum is a normal derivative."""
    zero = jnp.zeros_like(offsets[..., 0])
    rows = [jnp.stack([zero] * dim, axis=-1)]
    for i in range(dim):
        rows.append(jnp.stack([jnp.ones_like(zero) if k == i else zero for k in range(dim)], -1))
    for i in range(dim):
        for j in range(i, dim):
            row = []
            for k in range(dim):
                entry = zero
                if k == i:
                    entry = entry + offsets[..., j]
                if k == j:
                    entry = entry + offsets[..., i]
                row.append(entry)
            rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


class _Stencil(eqx.Module):
    """Which cells and boundary faces each cell reconstructs from, padded to one width.

    Attributes
    ----------
    cells : jnp.ndarray
        Cell indices per stencil, shape ``(n_cells, width)``; padded entries repeat the cell itself
        and carry zero weight, so a gather never reads out of range.
    cell_used, face_used : jnp.ndarray
        Which entries are real, shapes ``(n_cells, width)`` / ``(n_cells, face_width)``.
    faces : jnp.ndarray
        Boundary-face indices per stencil, shape ``(n_cells, face_width)``.
    """

    cells: jnp.ndarray
    cell_used: jnp.ndarray
    faces: jnp.ndarray
    face_used: jnp.ndarray


def build_stencil(mesh: Mesh, reach: int) -> _Stencil:
    """Every cell's neighbourhood out to ``reach`` face hops, with the boundary faces it owns.

    Parameters
    ----------
    mesh : Mesh
        The mesh, read for its face-to-cell connectivity.
    reach : int
        Number of face hops (1 gives the face neighbours, 2 their neighbours too).

    Returns
    -------
    _Stencil
        The padded index arrays.
    """
    face_cells = mesh.face_cells
    owner = np.asarray(face_cells.owner)
    neighbour = np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    n = mesh.n_cells
    rows = np.concatenate([owner[interior], neighbour[interior], np.arange(n)])
    columns = np.concatenate([neighbour[interior], owner[interior], np.arange(n)])
    # Sparse boolean reach: the neighbourhood matrix raised to `reach` is the hop-distance pattern,
    # and multiplying patterns is what keeps this linear in the cell count rather than a per-cell
    # breadth-first search in Python.
    adjacency = csr_matrix((np.ones(rows.size, dtype=bool), (rows, columns)), shape=(n, n))
    pattern = adjacency
    for _ in range(reach - 1):
        pattern = pattern @ adjacency
    pattern = pattern.tocsr()
    pattern.sort_indices()
    counts = np.diff(pattern.indptr)
    width = int(counts.max())
    cells = np.repeat(np.arange(n)[:, None], width, axis=1)
    cell_used = np.arange(width)[None, :] < counts[:, None]
    cells[cell_used] = pattern.indices

    # Each stencil cell brings the boundary faces it owns. Built by index arithmetic rather than a
    # per-cell loop: `owned` lists every cell's boundary faces contiguously, and one gather then
    # expands every (cell, stencil entry) pair into that entry's share of it.
    boundary_faces = np.flatnonzero(~interior)
    owned = boundary_faces[np.argsort(owner[boundary_faces], kind="stable")]
    per_cell = np.bincount(owner[boundary_faces], minlength=n)
    owned_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(per_cell, out=owned_start[1:])
    pair_cell = np.repeat(np.arange(n), counts)
    pair_counts = per_cell[pattern.indices]
    offsets = np.zeros(pair_counts.size + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=offsets[1:])
    within = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], pair_counts)
    flat_faces = owned[np.repeat(owned_start[pattern.indices], pair_counts) + within]
    flat_rows = np.repeat(pair_cell, pair_counts)
    face_counts = np.bincount(flat_rows, minlength=n)
    face_width = max(1, int(face_counts.max()))
    row_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(face_counts, out=row_start[1:])
    slot = np.arange(flat_rows.size) - np.repeat(row_start[:-1], face_counts)
    faces = np.zeros((n, face_width), dtype=np.int64)
    faces[flat_rows, slot] = flat_faces
    face_used = np.arange(face_width)[None, :] < face_counts[:, None]
    return _Stencil(
        cells=jnp.asarray(cells),
        cell_used=jnp.asarray(cell_used),
        faces=jnp.asarray(faces),
        face_used=jnp.asarray(face_used),
    )


def _entry_of(stencil: _Stencil, n_cells: int, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Which stencil slot of ``rows`` holds ``columns`` -- a search over the whole padded array at
    once, since each row's entries are sorted and a row-major key makes them globally sorted."""
    cells = np.asarray(stencil.cells)
    used = np.asarray(stencil.cell_used)
    keys = (np.arange(cells.shape[0])[:, None] * n_cells + cells)[used]
    slots = np.broadcast_to(np.arange(cells.shape[1]), cells.shape)[used]
    return slots[np.searchsorted(keys, rows * n_cells + columns)]


def _reference_weights(
    mesh: Mesh,
    geometry: MeshGeometry,
    stencil: _Stencil,
    closure: HessianBoundaryClosure,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One block sweep of the gradient--Hessian system, as weights on the stencil.

    From a zero start that sweep is ``g = P_g^-1 b_g``: the gradient equation's right-hand side, which
    is a Green--Gauss sum of interpolated face values, under the cell's own block inverse. Both pieces
    come from :class:`~aquaflux.schemes.HessianCorrectedGradient`'s own systems, so the reference this
    scheme aims at is that scheme's, not a second copy of it.

    Returns the weights on the stencil's cells, ``(n_cells, dim, width)``, and on its boundary faces,
    ``(n_cells, dim, face_width)``.
    """
    face_cells = mesh.face_cells
    dim, n = mesh.dim, mesh.n_cells
    systems = HessianCorrectedGradient._systems(mesh, geometry, closure)
    inverse = systems.outer_preconditioner(systems.inner(), True).inverse  # (n, dim, dim)

    factor = interpolation_factor(face_cells, geometry)
    area = scale(geometry.face.normal, geometry.face.area)
    owner = np.asarray(face_cells.owner)
    neighbour = np.asarray(face_cells.neighbour)
    interior = neighbour >= 0
    factor_np, area_np = np.asarray(factor), np.asarray(area)

    # `b_g` is a conservative scatter of `area * phi_face`, with `phi_face` the linear interpolation
    # on an interior face and the boundary datum on a boundary face. Its coefficients therefore live
    # per face, and each lands on one (cell, stencil entry) pair.
    cells = np.asarray(stencil.cells)
    faces = np.asarray(stencil.faces)
    inner_owner, inner_neighbour = owner[interior], neighbour[interior]
    cell_coefficients = np.zeros((n, dim, cells.shape[1]))
    contribution = area_np[interior] * (1.0 - factor_np[interior])[:, None]
    other = area_np[interior] * factor_np[interior][:, None]
    # The owner's row takes `+area * phi_face` and the neighbour's `-area * phi_face`, and each of
    # the two cells' values appears in both rows.
    for row, column, coefficient in (
        (inner_owner, inner_owner, contribution),
        (inner_owner, inner_neighbour, other),
        (inner_neighbour, inner_owner, -contribution),
        (inner_neighbour, inner_neighbour, -other),
    ):
        slots = _entry_of(stencil, n, row, column)
        np.add.at(cell_coefficients, (row, slice(None), slots), coefficient)

    face_coefficients = np.zeros((n, dim, faces.shape[1]))
    boundary = np.flatnonzero(~interior)
    face_owner = owner[boundary]
    face_keys = (np.arange(n)[:, None] * mesh.n_faces + faces)[np.asarray(stencil.face_used)]
    face_slots = np.broadcast_to(np.arange(faces.shape[1]), faces.shape)[
        np.asarray(stencil.face_used)
    ]
    slots = face_slots[np.searchsorted(face_keys, face_owner * mesh.n_faces + boundary)]
    np.add.at(face_coefficients, (face_owner, slice(None), slots), area_np[boundary])

    cell_weights = jnp.einsum("nij,njk->nik", inverse, jnp.asarray(cell_coefficients))
    face_weights = jnp.einsum("nij,njk->nik", inverse, jnp.asarray(face_coefficients))
    return cell_weights, face_weights


class ProjectedStencilGradient(GradientScheme):
    """Compact weights, exact for quadratics, nearest to a reference that damps.

    See the module docstring for what the construction is for. The weights are built per field in
    :meth:`bind` — which boundary face carries a value and which a normal derivative comes from that
    field's :class:`~aquaflux.schemes.BoundaryLinearization` — and applied as one gather and
    contraction, so a reconstruction costs no solve and no iteration.

    Exact for linear and quadratic fields on any mesh whose stencils determine them, and second-order
    for smooth fields.

    Attributes
    ----------
    blend : float
        How much of the reference reconstruction the weights aim at, in ``[0, 1]``. ``0`` takes the
        minimum-norm exact weights, which are an unweighted quadratic least-squares fit on the
        stencil; ``1`` takes the nearest exact weights to the reference itself. The default ``0.75``
        is what ``validation/tetrahedral_gradient_ab/betchen_projected_probe.py`` measures as keeping
        the reference's accuracy while leaving the Rhie--Chow damping's sign intact on a tetrahedral
        duct; re-measure it with that harness on a mesh unlike one.
    reach : int
        Face hops in the stencil (default ``2``). Two is the smallest that can determine a quadratic
        on a tetrahedral mesh: a tetrahedron has four face neighbours against the nine coefficients a
        quadratic needs in three dimensions.
    boundary_weight : float
        The reference system's neighbour-averaged boundary Hessian weight (default ``0.5``). It
        affects only the reference, and through it only the weights' starting point.
    prepared : tuple, optional
        This geometry's weights and stencil, built by :meth:`bind`. A scheme that has not been bound
        cannot reconstruct: the weights depend on the field's boundary conditions, which are not
        knowable from the mesh.
    """

    blend: float = eqx.field(static=True, default=0.75)
    reach: int = eqx.field(static=True, default=2)
    boundary_weight: float = eqx.field(static=True, default=0.5)
    prepared: tuple | None = None

    def bind(
        self,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_linearization: BoundaryLinearization | None = None,
    ) -> ProjectedStencilGradient:
        """This scheme carrying the weights for one field's conditions on this geometry.

        Parameters
        ----------
        mesh : Mesh
            The mesh to bind to; its geometry must be concrete.
        geometry : MeshGeometry
            That mesh's face and cell metrics.
        boundary_linearization : BoundaryLinearization, optional
            The field's conditions, linearized: a face whose value does not move with its owner's
            value carries a prescribed **value**, and one whose value follows it carries a prescribed
            **normal derivative**. Omitted, every boundary face is read as carrying a value, which is
            what a condition-free binding means.

        Returns
        -------
        ProjectedStencilGradient
            The same scheme carrying this field's weights.

        Raises
        ------
        ValueError
            If a condition is neither of those two — a Robin or convective condition mixes a value
            and a derivative in one datum, and this scheme has no stencil constraint for that yet.

        Notes
        -----
        ⚠️ **A bound scheme belongs to the geometry and the conditions it was bound to.** The weights
        are constants after binding, so a stale binding returns a wrong gradient rather than a slower
        one; a cell-count mismatch is refused, a different geometry of the same size is not. For the
        same reason, do not bind inside a region being differentiated with respect to the *geometry*:
        the weights are then constants the differentiation cannot see through. Binding is transparent
        to differentiation with respect to the field, which is what a flow solve differentiates.
        """
        dim, n = mesh.dim, mesh.n_cells
        face_cells = mesh.face_cells
        if isinstance(face_cells.interior, jax.core.Tracer):
            raise NotImplementedError(
                "ProjectedStencilGradient.bind needs a concrete mesh: its stencil comes from the "
                "connectivity and its weights from a per-cell fit, neither of which can be traced. "
                "A traced mesh means an assembler is being built INSIDE a residual evaluation -- "
                "which the turbulence closure does for k and omega, binding their schemes once per "
                "call. Bind outside the residual and pass the bound scheme in, or use "
                "MultipleCorrectionGradient for those fields, whose bind is traceable."
            )
        interior = np.asarray(face_cells.interior)
        if boundary_linearization is None:
            extrapolates = np.zeros(mesh.n_faces, dtype=bool)
        else:
            value_weight = np.asarray(boundary_linearization.value_weight)
            mixed = (~interior) & (value_weight > 1e-8) & (value_weight < 1.0 - 1e-8)
            if mixed.any():
                raise ValueError(
                    f"{int(mixed.sum())} boundary faces carry a condition whose face value is "
                    "neither prescribed nor a pure extrapolation (a Robin or convective condition, "
                    "whose datum mixes a value and a normal derivative). ProjectedStencilGradient "
                    "constrains its stencil with one or the other and has no constraint for the "
                    "mixture yet; use another gradient scheme for such a field."
                )
            extrapolates = (~interior) & (value_weight >= 1.0 - 1e-8)

        stencil = build_stencil(mesh, self.reach)
        closure = AveragedNeighbourHessian(weight=self.boundary_weight)
        reference_cells, reference_faces = _reference_weights(mesh, geometry, stencil, closure)

        centroid = geometry.cell.centroid
        face_centroid = geometry.face.centroid
        normal = geometry.face.normal
        # A per-cell length, so the monomials are evaluated on offsets of order one and their normal
        # equations stay well scaled whatever the mesh's units.
        size = geometry.cell.volume ** (1.0 / dim)

        offsets_cells = (centroid[stencil.cells] - centroid[:, None, :]) / size[:, None, None]
        offsets_faces = (face_centroid[stencil.faces] - centroid[:, None, :]) / size[:, None, None]
        basis_cells = polynomial_basis(offsets_cells, dim)  # (n, width, terms)
        value_rows = polynomial_basis(offsets_faces, dim)
        derivative_rows = (
            jnp.einsum(
                "nktd,nkd->nkt",
                polynomial_basis_gradient(offsets_faces, dim),
                normal[stencil.faces],
            )
            / size[:, None, None]
        )
        takes_derivative = jnp.asarray(extrapolates)[stencil.faces]
        basis_faces = jnp.where(takes_derivative[:, :, None], derivative_rows, value_rows)

        # Padding contributes nothing: a zero column leaves the normal equations and the weights
        # untouched, so a short stencil behaves exactly as if it had been built at its own width.
        basis = jnp.concatenate(
            [
                jnp.where(stencil.cell_used[:, :, None], basis_cells, 0.0),
                jnp.where(stencil.face_used[:, :, None], basis_faces, 0.0),
            ],
            axis=1,
        )  # (n, width + face_width, terms)
        start = self.blend * jnp.concatenate(
            [
                jnp.where(stencil.cell_used[:, None, :], reference_cells, 0.0),
                jnp.where(stencil.face_used[:, None, :], reference_faces, 0.0),
            ],
            axis=2,
        )  # (n, dim, width + face_width)

        # What the weights must reproduce: each monomial's own gradient at the cell, which is the
        # identity on the linear terms and zero on the rest, in the monomials' scaled coordinates.
        n_terms = basis.shape[-1]
        exact = jnp.zeros((n, dim, n_terms))
        for i in range(dim):
            exact = exact.at[:, i, 1 + i].set(1.0 / size)

        normal_equations = jnp.einsum("nkt,nks->nts", basis, basis)
        defect = exact - jnp.einsum("nkt,nik->nit", basis, start)
        # One small solve per cell, over the monomials: `solve` batches the leading axis, taking the
        # right-hand sides in its last axis, so the gradient components ride there.
        multipliers = jnp.linalg.solve(
            _regularized(normal_equations), jnp.swapaxes(defect, 1, 2)
        )  # (n, terms, dim)
        weights = start + jnp.einsum("nkt,nti->nik", basis, multipliers)
        _report_conditioning(normal_equations, n)
        return ProjectedStencilGradient(
            blend=self.blend,
            reach=self.reach,
            boundary_weight=self.boundary_weight,
            prepared=(
                stencil,
                weights[:, :, : stencil.cells.shape[1]],
                weights[:, :, stencil.cells.shape[1] :],
                jnp.asarray((~interior) & ~extrapolates),
                jnp.asarray(extrapolates),
            ),
        )

    def _reconstruct_gradient(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
        *,
        operator_hook=None,
        imposed: ImposedGradient | None = None,
        boundary_values_at=None,
        boundary_gradient_weight=None,
    ) -> jnp.ndarray:
        # `imposed` is applied by `gradients`, which is the whole of what this scheme owes it: the
        # weights are a fixed linear map, so overwriting rows of the answer is exactly overwriting
        # those cells' reconstructions.
        del imposed, boundary_values_at, boundary_gradient_weight
        if operator_hook is not None:
            raise NotImplementedError(
                "ProjectedStencilGradient cannot run domain-decomposed: its stencil reaches two face "
                "hops, so a cell next to a partition boundary reads values a single halo exchange of "
                "the field does not carry. A distributed build needs a two-deep halo -- not yet "
                "built. Use CorrectedGreenGauss with SweptGradientSolve for a distributed "
                "non-orthogonal gradient."
            )
        if self.prepared is None:
            raise ValueError(
                "this gradient scheme has not been bound. Its weights depend on the field's boundary "
                "conditions as well as the geometry, so call bind(mesh, geometry, linearization) "
                "before reconstructing -- a residual assembler does this once per field."
            )
        stencil, cell_weights, face_weights, prescribed, extrapolates = self.prepared
        if cell_weights.shape[0] != mesh.n_cells:
            raise ValueError(
                f"this gradient scheme was bound to a geometry of {cell_weights.shape[0]} cells and "
                f"is being asked to reconstruct on one of {mesh.n_cells}. Its weights are that "
                "geometry's; call bind() again for this one."
            )
        owner = mesh.face_cells.owner
        along = jnp.sum(
            (geometry.face.centroid - geometry.cell.centroid[owner]) * geometry.face.normal, axis=-1
        )
        # A prescribed face contributes its value. An extrapolating one contributes the normal
        # derivative its condition prescribes, which is what the boundary value evaluated at zero
        # gradient carries: `phi_f = phi_owner + (d . n) * dphi/dn` there.
        derivative = jnp.where(
            extrapolates,
            (boundary_values - field[owner]) / jnp.where(along != 0.0, along, 1.0),
            0.0,
        )
        data = jnp.where(prescribed, boundary_values, derivative)
        return jnp.einsum("nik,nk->ni", cell_weights, field[stencil.cells]) + jnp.einsum(
            "nik,nk->ni", face_weights, data[stencil.faces]
        )


def _regularized(normal_equations: jnp.ndarray) -> jnp.ndarray:
    """The normal equations with a relative floor on the diagonal, so a stencil that cannot determine
    every monomial still yields finite weights (the ones it can determine, and no correction along the
    directions it cannot) rather than a NaN that propagates into every later reconstruction."""
    scale_per_cell = jnp.max(jnp.abs(jnp.diagonal(normal_equations, axis1=1, axis2=2)), axis=1)
    floor = 1e-12 * jnp.where(scale_per_cell > 0.0, scale_per_cell, 1.0)
    return normal_equations + floor[:, None, None] * jnp.eye(normal_equations.shape[-1])


def _report_conditioning(normal_equations: jnp.ndarray, n_cells: int) -> None:
    """Report cells whose stencil barely determines the monomials, which is where large weights come
    from — a property of the mesh, so it is reported rather than treated as an error."""
    if isinstance(normal_equations, jax.core.Tracer):  # a trace carries no concrete conditioning
        return
    try:
        values = np.linalg.eigvalsh(np.asarray(normal_equations))
    except np.linalg.LinAlgError:  # pragma: no cover - a degenerate stencil
        return
    largest, smallest = values[:, -1], values[:, 0]
    condition = np.where(smallest > 0, largest / np.where(smallest > 0, smallest, 1.0), np.inf)
    bad = int((condition > _ILL_CONDITIONED).sum())
    if bad:
        _warn_once(
            "conditioning",
            f"ProjectedStencilGradient: {bad} of {n_cells} cells have a stencil whose monomials are "
            f"nearly dependent (worst condition number {condition.max():.2e}), so the weights there "
            "are large and the reconstruction is sensitive. A larger `reach` gives those cells more "
            "cells to fit, at a wider Jacobian.",
        )
