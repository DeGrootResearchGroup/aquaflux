"""Gradient reconstruction schemes — reconstruct cell gradients from a cell field.

A ``GradientScheme`` is the swappable numerics object the flow terms consume for their
non-orthogonal corrections (and later Rhie–Chow). It is defined and verified *independently
of any physics*: the exact test is to reconstruct the gradient of a known analytic field and
compare to its analytic gradient (order-of-accuracy study).

:class:`CompactGreenGauss` is the base, one-shot Green–Gauss reconstruction:

    grad(phi)_P = (1 / V_P) * sum_faces  phi_ip * S_f          (S_f = A_f n_f, owner-outward)

with a linearly-interpolated interior face value ``phi_ip = (1-g) phi_P + g phi_N`` (``g`` the
projection factor of the face centroid onto the P–N line) and the supplied boundary value on
boundary faces. It is 2nd-order and linear-exact on orthogonal grids but **inconsistent**
(order ~0) on irregular grids — the known Green–Gauss deficiency. :class:`CorrectedGreenGauss`
adds the non-orthogonal correction (a coupled system): linear-exact on any mesh, consistent
on irregular grids, but capped near 1st order there (the accuracy ceiling the implicit
gradient later removes).
"""

from __future__ import annotations

import abc
import dataclasses
import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from aquaflux.mesh.face import face_geometry_scheme
from aquaflux.vectors import dot, scale

from .interpolation import (
    blend_owner_neighbour,
    interpolate_owner_neighbour,
    interpolation_factor,
)

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, Mesh, MeshGeometry


_GRADIENT_UNCONVERGED_WARNED = False

_Tree = TypeVar("_Tree")

# Default relative-residual threshold for the fixed-sweep solve's under-resolution warning. Named
# rather than written twice, because `SweepCalibration.solver` builds that solver too and a
# calibrated count must not arrive with a different diagnostic than a hand-written one.
_DEFAULT_WARN_TOL = 5e-2


def _warn_gradient_unconverged(sweeps: int, tol: float) -> None:
    """Host-side diagnostic: warn (once per process) if the fixed-sweep gradient solve is under-resolved.

    Invoked from a ``jax.debug.callback`` inside :meth:`SweptGradientSolve.solve`, gated by a
    ``lax.cond`` so it fires only when the residual (which the sweep already computed) exceeds
    ``tol``. Because that callback runs on every under-resolved gradient solve (many per Newton
    step), a module-level flag guarantees a single emission — the mesh conditioning is fixed, so one
    warning is the whole message.
    """
    global _GRADIENT_UNCONVERGED_WARNED
    if _GRADIENT_UNCONVERGED_WARNED:
        return
    _GRADIENT_UNCONVERGED_WARNED = True
    warnings.warn(
        f"SweptGradientSolve: the corrected-gradient sweeps are under-resolved on this mesh "
        f"(relative residual exceeded {tol:.0e} after {sweeps} sweeps). Increase `sweeps` for this "
        f"non-orthogonality, or set `warn_tol=None` to silence.",
        stacklevel=1,
    )


class _CorrectedTerms(NamedTuple):
    """Geometry-only intermediates shared by the corrected-gradient operator ``A_g`` and RHS ``B``.

    Bundling them lets one face-geometry computation feed both the operator (which is
    field-independent) and the right-hand side (which carries the field), so both linear-solve
    strategies (:class:`GmresGradientSolve`, :class:`SweptGradientSolve`) build on the same system.
    """

    face_cells: FaceCellConnectivity  # face→cell gather/scatter operators (owner / neighbour)
    g: jnp.ndarray  # (n_faces,) projection factor of the face centroid onto the P–N line
    skew: jnp.ndarray  # (n_faces, dim) skewness offset D_g,ip from the P–N line to the face
    area_vector: jnp.ndarray  # (n_faces, dim) owner-outward S_f = A_f n_f
    volume: jnp.ndarray  # (n_cells,) cell volumes


class ImposedGradient(eqx.Module):
    """A gradient that is *known* on some cells, imposed on a reconstruction rather than reconstructed.

    Some cells' gradient is a model quantity rather than something a stencil should estimate. The
    near-wall ``omega`` of a k--omega closure is the standing example: those cells do not solve a
    transport balance at all, their value being fixed by a profile going like ``1 / d**2``, so the
    honest gradient there is that profile's analytical derivative. Reconstructing it instead is not
    merely imprecise -- a linear fit over cells whose wall distance varies sharply returns about a
    quarter of the analytical magnitude, and the neighbouring ring roughly twice too much.

    Overwriting the gradient a scheme *returns* is not enough for every scheme, which is why this is
    passed **in** rather than applied **after**. A reconstruction that consumes its own first
    estimate -- to build a Hessian by differentiating it, say -- would consume the value being
    replaced and correct it only once the damage was done.

    ⚠️ **Imposing a gradient substitutes a model for a reconstruction, so a scheme's exactness
    contract stops at the imposed cells and the cells that read them.** Correction matrices are
    calibrated against the operator a scheme would otherwise apply, and an imposed value is by
    construction not what that operator returns. That is the trade taken deliberately: a consistent
    operator around a value measured four times too small is worse than an inconsistent one around
    the right value.

    Attributes
    ----------
    cells : jnp.ndarray
        Indices of the cells whose gradient is imposed, shape ``(n_imposed,)``.
    gradient : jnp.ndarray
        The gradient imposed there, shape ``(n_imposed, dim)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> imposed = ImposedGradient(jnp.array([1]), jnp.array([[3.0, 4.0]]))
    >>> imposed.impose(jnp.zeros((3, 2)))
    Array([[0., 0.],
           [3., 4.],
           [0., 0.]], dtype=float64)
    """

    cells: jnp.ndarray
    gradient: jnp.ndarray

    def impose(self, gradient: jnp.ndarray) -> jnp.ndarray:
        """``gradient`` with the imposed rows overwritten.

        Parameters
        ----------
        gradient : jnp.ndarray
            A cell gradient, shape ``(n_cells, dim)``.

        Returns
        -------
        jnp.ndarray
            The same array with rows :attr:`cells` set to :attr:`gradient`; shape unchanged.
        """
        return gradient.at[self.cells].set(self.gradient)

    def impose_on_faces(
        self, face_gradient: jnp.ndarray, face_cells: FaceCellConnectivity
    ) -> jnp.ndarray:
        """``face_gradient`` with every boundary face of an imposed cell carrying that cell's gradient.

        A boundary closure exists to supply a derivative a boundary condition does not carry. Where
        a caller has imposed one it is not missing, and closing it from the field values there
        contradicts the model that imposed it: on a wall ``omega`` face the boundary value is itself
        a zero-gradient closure, so differencing against it imposes a near-zero normal derivative
        exactly where the modelled profile diverges.

        Interior faces are untouched -- they interpolate from two cells, both of which already carry
        the imposition.

        Parameters
        ----------
        face_gradient : jnp.ndarray
            The gradient on every face as a closure left it, shape ``(n_faces, dim)``.
        face_cells : FaceCellConnectivity
            The face->cell incidence.

        Returns
        -------
        jnp.ndarray
            The same array with the imposed cells' boundary faces overwritten.
        """
        owner = face_cells.owner
        shape = (face_cells.n_cells, face_gradient.shape[-1])
        values = jnp.zeros(shape, dtype=face_gradient.dtype).at[self.cells].set(self.gradient)
        flagged = jnp.zeros(face_cells.n_cells, dtype=bool).at[self.cells].set(True)
        take = flagged[owner] & ~face_cells.interior
        return jnp.where(take[:, None], values[owner], face_gradient)


class GradientScheme(eqx.Module):
    """Strategy interface: reconstruct cell gradients from a cell field."""

    def bind(self, mesh: Mesh, geometry: MeshGeometry) -> GradientScheme:
        """Return this scheme prepared for one geometry, ready to reconstruct on it repeatedly.

        A reconstruction may do work that depends only on the geometry, and a solver calls it over
        and over on the same mesh — once per field, and once per Krylov matvec, since a matvec
        re-evaluates the residual. This is where a scheme hoists that work out of the per-call path.

        The default returns the scheme unchanged, which is the honest answer for a reconstruction
        that has nothing geometry-only worth holding: a single-pass Green--Gauss sum, or a swept
        solve whose preconditioner is a per-cell scalar it is cheaper to recompute than to carry.
        Assemblers call this when they are built, so a scheme that overrides it is prepared once by
        every consumer without any of them knowing which schemes benefit.

        Parameters
        ----------
        mesh : Mesh
            The mesh to prepare for.
        geometry : MeshGeometry
            That mesh's face and cell metrics.

        Returns
        -------
        GradientScheme
            A scheme equivalent to this one on that geometry. Implementations must return the *same*
            reconstruction, not an approximation of it -- preparing is a cost change, never an
            accuracy one.
        """
        return self

    def gradients(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        imposed: ImposedGradient | None = None,
        boundary_values_at: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Cell gradients of ``field``, shape ``(n_cells, dim)``.

        The reconstruction itself is :meth:`_reconstruct_gradient`; this wrapper is where an
        imposed gradient is guaranteed to reach the answer, so that every scheme honours it whether
        or not it has anywhere earlier to put it.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        mesh : Mesh
            Provides owner/neighbour connectivity.
        geometry : MeshGeometry
            Face and cell metrics (areas, owner-outward normals, centroids, volumes).
        boundary_values : jnp.ndarray
            Face values on boundary faces, shape ``(n_faces,)`` (interior entries ignored).
        operator_hook : callable, optional
            A ghost-cell exchange threaded into an iterative reconstruction's linear solve, applied
            to the unknown before each operator apply (see :meth:`GradientSolve.solve`). The identity
            when omitted (the serial path). A single-pass scheme reconstructs owned rows exactly from
            the already-exchanged ``field`` and so ignores it; a scheme whose reconstruction couples
            across partitions in a way this per-apply exchange cannot make serial-exact must raise
            when it is not ``None``, never silently return a wrong owned gradient.
        imposed : ImposedGradient, optional
            Cells whose gradient the caller knows analytically and wants used instead of a
            reconstruction. ``None`` (the default) reconstructs everywhere, and leaves every scheme
            here byte-identical to one that had never heard of the argument.
        boundary_values_at : callable, optional
            ``gradient -> boundary_values``: the caller's boundary closures re-evaluated at a
            reconstructed gradient. ``boundary_values`` above is those closures evaluated at *zero*
            gradient, which keeps the residual a single pass over the field but leaves a
            gradient-type condition carrying none of its own correction. A scheme that
            **differentiates** a boundary value needs the corrected one, and this is how it asks.
            ``None`` (the default) leaves every scheme reconstructing exactly as before.

        Returns
        -------
        jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``; the imposed rows carry exactly what was
            imposed.
        """
        gradient = self._reconstruct_gradient(
            field,
            mesh,
            geometry,
            boundary_values,
            operator_hook=operator_hook,
            imposed=imposed,
            boundary_values_at=boundary_values_at,
        )
        return gradient if imposed is None else imposed.impose(gradient)

    @abc.abstractmethod
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
    ) -> jnp.ndarray:
        """The reconstruction itself; see :meth:`gradients` for the arguments.

        The extension point every scheme implements. ``imposed`` is passed down rather than left to
        :meth:`gradients` because a scheme that *consumes* its own reconstructed gradient -- to
        differentiate it into a Hessian, say -- must impose before that consumer reads it, not after.
        A scheme with no such internal consumer may ignore the argument entirely: :meth:`gradients`
        applies it to whatever comes back either way.
        """


class CompactGreenGauss(GradientScheme):
    """One-shot Green–Gauss with linearly-interpolated interior face values."""

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
    ) -> jnp.ndarray:
        # No iterative solve: an owned cell's one-shot gradient is exact once its `field` halo is
        # filled, so the per-apply ghost exchange (`operator_hook`) has nothing to correct here. The
        # distributed residual still exchanges the *final* gradient for the flux (its `gradient_hook`).
        del operator_hook, imposed, boundary_values_at
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        g = interpolation_factor(face_cells, geometry)
        phi_interior = interpolate_owner_neighbour(field, g, face_cells)
        phi_face = face_cells.combine_face_values(phi_interior, boundary_values)

        area_vector = scale(face_geometry.normal, face_geometry.area)  # owner-outward S_f
        grad_sum = face_cells.scatter_conservative(scale(area_vector, phi_face))
        return scale(grad_sum, 1.0 / cell_geometry.volume)


def symmetric_components(dim: int) -> int:
    """Number of independent components of a symmetric ``(dim, dim)`` tensor — 3 in 2D, 6 in 3D.

    Parameters
    ----------
    dim : int
        Spatial dimension.

    Returns
    -------
    int
        ``dim (dim + 1) / 2``.
    """
    return dim * (dim + 1) // 2


def _symmetric_pairs(dim: int) -> list[tuple[int, int]]:
    """The ``(i, j)`` index pairs with ``i <= j``, in the order the packed components use."""
    return [(i, j) for i in range(dim) for j in range(i, dim)]


def expand_symmetric(packed: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Expand packed independent components into a full symmetric tensor field.

    For a field of second derivatives of a twice-continuously-differentiable function the tensor is
    symmetric, so only ``dim (dim + 1) / 2`` of its ``dim**2`` entries are independent. Carrying only
    those is what makes the reconstruction's unknown smaller than the tensor it represents; this is
    the map back, applied wherever a face kernel wants the tensor itself.

    Parameters
    ----------
    packed : jnp.ndarray
        Independent components, shape ``(n_cells, dim (dim + 1) / 2)``, ordered by ``(i, j)`` with
        ``i <= j``, ``i`` varying slowest.
    dim : int
        Spatial dimension.

    Returns
    -------
    jnp.ndarray
        The symmetric tensor field, shape ``(n_cells, dim, dim)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> expand_symmetric(jnp.array([[1.0, 2.0, 3.0]]), 2)
    Array([[[1., 2.],
            [2., 3.]]], dtype=float64)
    """
    rows = [
        jnp.stack(
            [packed[:, _symmetric_pairs(dim).index((min(i, j), max(i, j)))] for j in range(dim)],
            axis=-1,
        )
        for i in range(dim)
    ]
    return jnp.stack(rows, axis=-2)


def contract_symmetric(tensor: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Contract a tensor field onto the packed symmetric components — the adjoint of
    :func:`expand_symmetric`.

    This is the transpose of the expansion, not its inverse: an off-diagonal component appears in the
    tensor twice, so the entry it receives is the **sum** of the two, which is what makes
    ``<expand(u), T> == <u, contract(T)>`` hold for every ``T``. That identity is the reason a
    residual is reduced with this rather than by reading off the upper triangle — the reduced system
    is then the projection of the full one onto the symmetric subspace, and inherits its structure.

    Parameters
    ----------
    tensor : jnp.ndarray
        A tensor field, shape ``(n_cells, dim, dim)``. It need not be symmetric.
    dim : int
        Spatial dimension.

    Returns
    -------
    jnp.ndarray
        Packed components, shape ``(n_cells, dim (dim + 1) / 2)``, in :func:`expand_symmetric`'s
        order.
    """
    return jnp.stack(
        [
            tensor[:, i, j] if i == j else tensor[:, i, j] + tensor[:, j, i]
            for i, j in _symmetric_pairs(dim)
        ],
        axis=-1,
    )


class GradientPreconditioner(eqx.Module):
    """Strategy: apply an approximate ``A⁻¹`` to a residual, cell-locally and in one pass.

    The reconstruction systems here are all **volume-dominated** — a per-cell diagonal block plus a
    weaker coupling to the face neighbours — so what an iterative solve needs is a cheap, exactly
    parallel approximation of the inverse of that per-cell block. This is that approximation, and it
    is a strategy rather than a fixed formula because *how good* the per-cell block has to be varies
    by system: the corrected-gradient system's block is close enough to the cell volume that scaling
    by ``1/V`` converges, while the gradient--Hessian system's is not (the gradient and the Hessian
    couple to each other *within* a cell, which ``1/V`` cannot see).

    An implementation is a pure function of the residual with no cross-cell reduction, so it costs
    one pass over the cells and is correct unchanged under domain decomposition.
    """

    @abc.abstractmethod
    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        """Return the preconditioned residual ``P⁻¹·residual`` (the shape of ``residual``).

        Parameters
        ----------
        residual : jnp.ndarray
            The residual to precondition, shape ``(n_cells, ...)`` — the trailing axes carry the
            unknown's per-cell components (a gradient ``(dim,)``, a Hessian ``(dim, dim)``).

        Returns
        -------
        jnp.ndarray
            The preconditioned residual, the same shape as ``residual``.
        """


class InverseVolume(GradientPreconditioner):
    """Scale each cell's residual by ``1/V`` — the cell volume as a stand-in for the diagonal.

    The cheapest useful preconditioner, and an accurate one wherever the operator's per-cell block is
    dominated by the volume that sits on its diagonal: the corrected-gradient operator
    ``A_g = V ⊙ I − (skewness coupling)`` is of that kind, so ``1/V`` is within the skewness of the
    true block and the Richardson iteration it drives converges quickly. It is **not** accurate where
    a cell's own unknowns couple to each other at leading order — see :class:`CellBlockJacobi`.

    Attributes
    ----------
    inverse_volume : jnp.ndarray
        Reciprocal cell volumes, shape ``(n_cells,)``.
    """

    inverse_volume: jnp.ndarray

    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        # Broadcast over however many component axes the unknown carries (a gradient has one, a
        # Hessian two), rather than the exactly-one-trailing-axis `vectors.scale` assumes.
        weight = self.inverse_volume.reshape(self.inverse_volume.shape + (1,) * (residual.ndim - 1))
        return weight * residual


class CellBlockJacobi(GradientPreconditioner):
    """Apply the exact inverse of each cell's own diagonal block — block Jacobi over the cells.

    Where a cell's unknowns couple to *each other* at leading order, scaling by the volume is not an
    approximate inverse of the diagonal block, it ignores the block's off-diagonal entries entirely.
    That is the gradient--Hessian system's situation: the Hessian enters the gradient equation through
    a face-curvature term and the gradient enters the Hessian equation through a Green--Gauss sum, and
    both are the same order as the volume term — so an inverse-volume iteration on it converges slowly
    **even on a perfectly orthogonal mesh**, where the skewness coupling that the corrected-gradient
    system worries about is identically zero. Inverting the per-cell block instead removes exactly
    that intra-cell coupling and leaves only the weak inter-cell one.

    The inverse is stored as one small square matrix per cell and contracted against the residual's
    **last** axis, which covers both unknowns this serves: a gradient residual ``(n_cells, dim)``
    takes a ``(dim, dim)`` inverse, and the Hessian system's residual — carried as the independent
    components of a symmetric tensor — takes a ``(n_sym, n_sym)`` one, ``n_sym`` being
    ``dim (dim + 1) / 2``. Both are exact per-cell inverses, not approximations.

    Attributes
    ----------
    inverse : jnp.ndarray
        Per-cell inverse of the diagonal block, shape ``(n_cells, dim, dim)``.
    """

    inverse: jnp.ndarray

    def apply(self, residual: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("cjl,c...l->c...j", self.inverse, residual)


def cell_diagonal_block(
    owner_column: Callable[[jnp.ndarray], jnp.ndarray],
    neighbour_column: Callable[[jnp.ndarray], jnp.ndarray],
    diagonal: jnp.ndarray,
    n_cells: int,
    dim: int,
) -> jnp.ndarray:
    """Recover a face-assembled operator's per-cell diagonal block **exactly**, by probing each side
    of the face separately.

    A face-assembled operator ``A u = D ⊙ u − scatter(F(u_owner, u_nb))`` couples a cell only to its
    face neighbours, so its per-cell diagonal block is the part of that scatter which lands back on
    the cell the value was gathered from. Reading it off a plain matrix--vector product is not
    possible — probing with a component set in *every* cell returns the whole row sum, neighbours
    included — and separating it normally costs a graph colouring, so that no two adjacent cells
    share a probe.

    Passing the unknown's two sides as separate fields removes that need. Evaluating the face kernel
    with the neighbour side zeroed and keeping only the owner-side scatter leaves each cell reading
    exclusively its own value, so **one probe per component** gives that component's column of every
    cell's block at once, whatever the mesh. The two callables here are those two half-evaluations;
    summing them gives the full block. This costs ``dim`` operator applies rather than
    ``n_colours × dim``, needs no adjacency graph, and — because both halves come from the same face
    kernel the full operator is built from — cannot drift from the operator it preconditions.

    Parameters
    ----------
    owner_column, neighbour_column : callable
        Given a per-cell probe field of shape ``(n_cells, dim)``, return the owner-side / neighbour-side
        scatter of the face kernel evaluated with only that side live, shape ``(n_cells, dim)``.
    diagonal : jnp.ndarray
        The operator's explicit diagonal term (the cell volume), shape ``(n_cells,)``; it enters the
        block as ``diagonal · I``.
    n_cells, dim : int
        Cell count and the block's size.

    Returns
    -------
    jnp.ndarray
        The per-cell diagonal block, shape ``(n_cells, dim, dim)``.
    """
    basis = jnp.eye(dim)

    def column(unit: jnp.ndarray) -> jnp.ndarray:
        probe = jnp.broadcast_to(unit, (n_cells, dim))
        return scale(probe, diagonal) - (owner_column(probe) + neighbour_column(probe))

    # columns[k] is the k-th column of every cell's block, shape (n_cells, dim); move the probe axis
    # last so the result indexes as block[cell, row, column].
    columns = jax.vmap(column)(basis)
    return jnp.moveaxis(columns, 0, -1)


class GradientSystem(NamedTuple):
    """One reconstruction system as a solve strategy sees it: the operator, its preconditioner, and
    the shape of the unknown.

    A reconstruction scheme assembles its system from the geometry once and then does two different
    things with it — solves it against a right-hand side (every reconstruction) and *measures* it
    (:func:`contraction_rate`, once per case). Both need exactly this triple and nothing scheme-
    specific, which is why it is a value rather than three parameters threaded separately: it is what
    lets one estimator serve the corrected-gradient system and both of the Hessian-corrected scheme's
    systems unchanged. The right-hand side is deliberately absent — it is the only field-dependent
    part, and neither the operator nor its convergence rate depends on it.

    Attributes
    ----------
    preconditioner : GradientPreconditioner
        Approximate inverse of the operator's per-cell diagonal block.
    operator : callable
        The matrix-free operator ``A`` (a matvec ``x -> A·x``) on an unknown of shape ``shape``.
    shape : tuple of int
        Shape of the unknown — ``(n_cells, dim)`` for a gradient, ``(n_cells, dim, dim)`` for a
        Hessian.
    """

    preconditioner: GradientPreconditioner
    operator: Callable[[jnp.ndarray], jnp.ndarray]
    shape: tuple[int, ...]


class GradientSolve(eqx.Module):
    """Strategy: apply ``A⁻¹`` to solve a volume-dominated reconstruction system ``A·x = rhs``.

    A reconstruction scheme reduces to a sparse linear system whose operator ``A`` is geometry-only
    and **volume-dominated** — a per-cell diagonal block, with the discretization coupling to the face
    neighbours as an off-diagonal correction. *How* that system is inverted — a Krylov solve, a fixed
    sweep — is orthogonal to the discretization, so it is an injected strategy rather than a separate
    scheme. The strategy receives only the matrix-free operator ``A``, the right-hand side, and a
    :class:`GradientPreconditioner` — nothing scheme-specific — so the same strategy serves
    :class:`CorrectedGreenGauss` (the gradient system) and both of
    :class:`HessianCorrectedGradient`'s systems (the outer Schur system on the gradient and the inner
    system on the Hessian).
    """

    @property
    def requires_linear_operator(self) -> bool:
        """Whether this strategy needs ``operator`` to be *transposable* — i.e. strictly linear in
        the unknown, with no host-side effects.

        A Krylov solve differentiated by the implicit function theorem forms its tangent from the
        transpose of the operator it was given, and ``jax.linear_transpose`` rejects a function
        containing anything nonlinear. A fixed-sweep strategy differentiates by unrolling and so
        imposes no such requirement. This matters when one system's solve runs *inside* another's
        operator, as :class:`HessianCorrectedGradient`'s inner Hessian solve does.
        """
        return False

    @property
    def emits_host_diagnostics(self) -> bool:
        """Whether :meth:`solve` evaluates a host-side diagnostic alongside the solve.

        Such a diagnostic measures convergence, so it is nonlinear in the unknown — which makes the
        solve unusable inside an operator handed to a strategy whose
        :attr:`requires_linear_operator` is set.
        """
        return False

    @abc.abstractmethod
    def solve(
        self,
        preconditioner: GradientPreconditioner,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Solve ``A·x = rhs`` for ``x`` (the shape of ``rhs``).

        Parameters
        ----------
        preconditioner : GradientPreconditioner
            Approximate inverse of the operator's per-cell diagonal block, supplied by the scheme that
            owns the system (it knows the block's structure). :class:`InverseVolume` where the volume
            dominates the block; :class:`CellBlockJacobi` where the cell's own unknowns couple to each
            other at leading order.
        operator : callable
            The matrix-free operator ``A`` (a matvec ``x -> A·x``). The unknown may carry any trailing
            shape — a per-cell gradient ``(n_cells, dim)`` or a per-cell Hessian ``(n_cells, dim, dim)``
            — and the preconditioner broadcasts or contracts over it accordingly.
        rhs : jnp.ndarray
            The right-hand side, matching the shape of the unknown ``x``.
        operator_hook : callable, optional
            A transform ``x -> x`` applied to the unknown **before every operator apply**. The
            identity when omitted (the serial path). A domain-decomposed solve passes the ghost-cell
            exchange here: each partition holds owned + ghost rows of ``x``, and the operator's owned
            output rows are only correct once the ghost rows carry their owning partition's current
            values, so the exchange must run once per iteration. A strategy that cannot honour this
            correctly (one whose iteration forms cross-cell reductions over the whole local vector)
            must **raise** when it is not ``None`` rather than silently return a wrong owned solution.
        """


class CellPreconditioner(eqx.Module):
    """Strategy: which approximate inverse of ``A_g`` drives the corrected-gradient sweep.

    The sweep converges at a rate set by ``rho(I - P^-1 A_g)``, so this choice decides how many
    sweeps a given accuracy costs -- and, on a mesh with a degenerate cell, whether the iteration
    converges at all. A strategy rather than a flag because the options differ in what they compute,
    not merely in a constant: one reads a per-cell scalar already to hand, the other probes the
    operator.
    """

    @abc.abstractmethod
    def build(self, terms: _CorrectedTerms) -> GradientPreconditioner:
        """Return the preconditioner for the system these ``terms`` describe.

        Parameters
        ----------
        terms : _CorrectedTerms
            The geometry intermediates the operator is built from, so the preconditioner cannot be
            derived from a different geometry than the operator it preconditions.
        """


class InverseCellVolume(CellPreconditioner):
    """``1/V`` per cell -- the default, and the right choice on a mesh of reasonable quality.

    ``A_g``'s per-cell block is the cell volume less the skewness coupling, so on a well-shaped cell
    this is within the skewness of the true block and the sweep it drives converges quickly. It costs
    a reciprocal and nothing else.

    ⚠️ **It degrades exactly as the cell does.** The neglected coupling scales with face area while
    the volume does not, so as a cell flattens ``1/V`` stops approximating anything. Measured on a
    single squashed cell, the reconstruction error there grows without bound with the volume ratio:
    3.6e+04 at a ratio of 2.5e3, 2.9e+12 at 3.6e4, 2.9e+28 at 3.6e8. Prefer :class:`ExactCellBlock`
    on a mesh carrying such cells, or on any case that diverges.
    """

    def build(self, terms: _CorrectedTerms) -> GradientPreconditioner:
        return InverseVolume(1.0 / terms.volume)


class ExactCellBlock(CellPreconditioner):
    """The operator's **true** per-cell block, recovered by probing and inverted per cell.

    Costs ``dim`` extra operator applies to extract, plus a small dense inverse per cell, and its
    application is a per-cell matrix product rather than a scalar multiply -- together roughly 3x the
    default's cost at four sweeps, falling as a share as the sweep count rises since the extraction is
    a fixed prologue. In exchange the iteration is insensitive to cell shape: on the squashed-cell
    sweep above, where ``1/V`` reaches 2.9e+28, this holds a flat 1.9e+01 across five orders of volume
    ratio -- the scheme's own discretization error on a degenerate cell, rather than a diverging
    iteration on top of it.

    On a healthy mesh the two agree to four significant figures, so this buys robustness and not
    accuracy: it is the choice for a mesh with poor cells, not a better default.
    """

    def build(self, terms: _CorrectedTerms) -> GradientPreconditioner:
        fc = terms.face_cells
        owner, neighbour = fc.owner, fc.safe_neighbour
        n_cells, dim = terms.volume.shape[0], terms.area_vector.shape[-1]
        no_face = jnp.zeros((fc.n_faces, dim))

        # The two half-evaluations `cell_diagonal_block` needs: each returns the part of the
        # correction scatter that lands back on the cell the probe was gathered from. The correction
        # is a conservative scatter, so the owner receives it and the neighbour its negation -- and
        # it is masked to zero on boundary faces, which carry no P-N line for the skewness to be
        # measured along.
        def owner_column(probe: jnp.ndarray) -> jnp.ndarray:
            w = (1.0 - terms.g) * dot(terms.skew, probe[owner])
            flux = fc.combine_face_values(scale(terms.area_vector, w), 0.0)
            return fc.scatter(flux, no_face)

        def neighbour_column(probe: jnp.ndarray) -> jnp.ndarray:
            w = terms.g * dot(terms.skew, probe[neighbour])
            flux = fc.combine_face_values(scale(terms.area_vector, w), 0.0)
            return fc.scatter(no_face, -flux)

        block = cell_diagonal_block(owner_column, neighbour_column, terms.volume, n_cells, dim)
        return CellBlockJacobi(jnp.linalg.inv(block))


class GmresGradientSolve(GradientSolve):
    """Solve the corrected-gradient system with matrix-free GMRES, differentiated by implicit diff.

    Robust to any conditioning — GMRES converges to the requested tolerance regardless of skew —
    and exact to that tolerance, self-tuning where the fixed-sweep count of
    :class:`SweptGradientSolve` would have to be raised for a badly-skewed mesh. That robustness comes
    at a price that rules it out as the default: a nested Krylov solve carrying its own implicit-diff
    tangent, re-entered on **every** reconstruction, which dominates the cost when the gradient is
    reconstructed inside a nonlinear (e.g. coupled RANS) Newton solve — where each Jacobian--vector
    product then differentiates through a full inner GMRES. :class:`SweptGradientSolve` (the default)
    replaces that with a short unrolled sparse apply; reach for this strategy only when a mesh is
    skewed enough that the swept sweep count would have to grow impractically.

    The injected preconditioner is applied on the **right** — GMRES is run on ``A P⁻¹ y = b`` and the
    answer recovered as ``x = P⁻¹y``. The solution is the same for any invertible ``P``, so this
    changes the iteration count and not the answer.

    ⚠️ **It must be the right and not the left, and the difference is not cosmetic.** Left
    preconditioning solves ``P⁻¹A x = P⁻¹b``, whose residual is measured in ``P⁻¹``'s norm rather than
    the problem's — and these preconditioners scale by roughly the inverse cell volume, which on a real
    mesh is a factor of ~1e6. The solver's convergence *and stagnation* tests then operate on a
    quantity six orders away from the true residual, and lineax's stagnation detector fires on a system
    it is about to solve: measured on this package's own backward-facing-step benchmark, reconstructing
    the omega gradient at the initial state, left preconditioning **raised** where the unpreconditioned
    solve converged in 7 iterations to a true relative residual of 6e-17. Right preconditioning leaves
    the residual at the problem's own scale, so both tests see exactly what they saw before.

    Attributes
    ----------
    rtol, atol : float
        GMRES relative / absolute tolerances (static).
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-10)

    @property
    def requires_linear_operator(self) -> bool:
        return True  # the implicit-diff tangent transposes the operator

    def solve(
        self,
        preconditioner: GradientPreconditioner,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if operator_hook is not None:
            raise NotImplementedError(
                "GmresGradientSolve cannot run domain-decomposed: GMRES forms inner products over "
                "the whole local vector, which double-counts a partition's ghost rows and is not "
                "reduced across partitions, so refreshing the ghost rows before each apply is not "
                "enough to make it converge to the serial solution. Use SweptGradientSolve for a "
                "distributed gradient solve (its preconditioned-Richardson sweeps form no global "
                "inner product, so a per-sweep ghost exchange is exact)."
            )
        # Right-preconditioned: the Krylov space is built on `A P⁻¹` and the right-hand side is left
        # alone, so the residual the solver measures IS the true residual. See the class docstring —
        # preconditioning this on the left instead moves the convergence and stagnation tests into a
        # norm ~1e6 away from the problem's own, and it breaks solves that otherwise converge.
        op = lx.FunctionLinearOperator(
            lambda v: operator(preconditioner.apply(v)),
            jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
        )
        solution = lx.linear_solve(op, rhs, solver=lx.GMRES(rtol=self.rtol, atol=self.atol)).value
        return preconditioner.apply(solution)


class SweptGradientSolve(GradientSolve):
    """Solve a reconstruction system by a fixed number of matrix-free preconditioned-Richardson
    sweeps — a sparse, ``O(n)``, scalable way to apply the constant ``A⁻¹``.

    These operators are diagonal-block-dominated (the per-cell block dominates the coupling to the
    face neighbours), so the preconditioned Richardson iteration

        x_{k+1} = x_k + P⁻¹ (b − A·x_k)

    converges geometrically with rate ``ρ(I − P⁻¹A) < 1`` for the injected
    :class:`GradientPreconditioner` ``P``. **The rate is a property of that pairing, not of the
    sweep**: on the corrected-gradient system ``A_g = V ⊙ I − C``, the cheap :class:`InverseVolume`
    already gives a small ``ρ``, while on the gradient--Hessian system it does not and
    :class:`CellBlockJacobi` is what makes the same sweep converge (see those classes). A **fixed** ``sweeps`` count reaches
    machine precision for this well-conditioned operator with no dense matrix and no nested Krylov
    solve; a sweep costs one operator apply (the first needs none — its iterate is zero, so its
    residual is ``B·φ`` outright, and ``sweeps`` sweeps cost ``sweeps - 1`` applies), so the cost is
    **linear in the mesh** — where a dense LU of ``A_g`` would be ``O((n·dim)²)`` per apply and cross
    over to a loss on finer meshes.
    Differentiated by simply unrolling the short, static-length loop, so the gradient's response to
    ``φ`` is carried implicitly into the flow Jacobian **without** an implicit-diff tangent solve.

    This is the **default** ``GradientSolve`` for :class:`CorrectedGreenGauss`: the unrolled sparse
    apply carries no nested Krylov solve and no implicit-diff tangent, so reconstructing the gradient
    inside a nonlinear (e.g. coupled RANS) Newton solve stays cheap — each Jacobian--vector product
    differentiates only through a handful of matvecs, not through a full inner GMRES (which made the
    :class:`GmresGradientSolve` alternative impractical there).

    Because ``A_g`` is volume-dominated the iteration converges in few sweeps, and the count needed is
    set by the **skewness, not the mesh size** — which is what makes the fixed-sweep apply ``O(n)``. The
    default ``sweeps=4`` stays well within discretization error but is **not** the exact solve: on a
    randomly perturbed 16x16 grid the relative departure from :class:`GmresGradientSolve` is ~3e-7 at 5%
    perturbation and ~3e-4 at 20%, reaching the floating-point floor at ~12 sweeps (5%) and ~20 (20%).
    That is why the tests asserting machine-precision properties of the *discretization* pin the exact
    solve rather than this one. A too-skewed mesh needs more; rather than pay for a data-dependent stop
    (which would defeat the cheap unrolled differentiation), the residual the last sweep already
    computed is checked against ``warn_tol`` and a **warning** is emitted once if the sweeps are
    under-resolved — a diagnostic, not a termination.

    **The count also sets how far a residual built on this reaches across the cell graph**, because
    every sweep after the first applies ``A_g`` and ``A_g`` couples a cell to its face neighbours —
    so ``sweeps`` sweeps reach ``sweeps - 1`` cells beyond ``B·φ``'s own. That is a constraint
    on anything assembling an operator by coloured probing at a fixed distance, and
    :func:`narrow_gradient_sweeps` is how such a consumer caps it without touching the solve.

    Attributes
    ----------
    sweeps : int
        Number of preconditioned-Richardson sweeps (static).
    relaxation : float
        Damping on each sweep's correction, in ``(0, 1]``. ``1.0`` (the default) is the undamped
        iteration; smaller values trade sweeps for robustness and are required for convergence on a
        sufficiently skewed mesh. A differentiable leaf, not a static field, so it can be swept or
        fitted without retracing.
    warn_tol : float or None
        Emit a one-time warning if the relative gradient residual after ``sweeps`` exceeds this
        (default ``5e-2``, i.e. the sweep is clearly stalling — the converged field stays accurate
        well below this, so it flags only a genuinely under-resolved mesh). ``None`` disables the
        check entirely, and it is skipped at ``sweeps=1`` where the measured ratio is exactly 1 on
        any mesh (see :meth:`solve`), so it would report nothing but its own construction.
    """

    sweeps: int = eqx.field(static=True, default=4)
    warn_tol: float | None = eqx.field(static=True, default=_DEFAULT_WARN_TOL)
    relaxation: float = 1.0

    @property
    def emits_host_diagnostics(self) -> bool:
        # The check norms the residual, which is nonlinear in the unknown; `warn_tol=None` removes it.
        return self.warn_tol is not None and self.sweeps > 1

    def solve(
        self,
        preconditioner: GradientPreconditioner,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        # Domain-decomposed: refresh the ghost rows of the current iterate before every operator
        # apply, so each partition's owned output rows equal the serial matvec restricted to owned
        # cells. The Richardson update writes garbage into the ghost/null rows, but the next apply's
        # hook overwrites them, so the owned rows converge exactly to the serial solution.
        op = operator if operator_hook is None else lambda v: operator(operator_hook(v))
        if self.sweeps <= 0:
            return jnp.zeros_like(rhs)

        # THE FIRST SWEEP'S OPERATOR APPLY IS PEELED, and it is exact rather than an approximation.
        # The iteration starts from a zero iterate, so that sweep's residual is `rhs - A·0`, which is
        # `rhs` itself -- the apply forming it multiplies a vector known to be zero at full price.
        # Nothing downstream removes it: the compiler folds the gathers against the zero constant but
        # not the scatters, so it costs a whole operator apply out of `sweeps` on every reconstruction,
        # and this one runs inside every residual evaluation and every Jacobian--vector product.
        # UNDER-RELAXATION IS NOT OPTIONAL ON A SUFFICIENTLY SKEWED MESH. Betchen and Straatman solve
        # this reconstruction by under-relaxed block-Jacobi and state that on an arbitrary grid the
        # relaxation must be strictly less than one for the iteration to converge at all; their
        # experiments run it at 0.8, converging in 33 iterations. Undamped (`relaxation=1`) is the
        # special case, safe only where the iteration's error operator is already a contraction.
        #
        # ⚠️ THEIR PAPER'S PARAMETER IS THE COMPLEMENT OF THIS ONE, so the two numbers look unrelated.
        # They write the update as a blend holding back a fraction `a` of the previous iterate,
        # `G <- a G + (1 - a) (block-Jacobi update)`, and require `a > 0`; this field is the weight on
        # the correction, `relaxation = 1 - a`. Their reported `a = 0.2` is `relaxation = 0.8`, and
        # their `a > 0` requirement is `relaxation < 1`. Read either number without its convention and
        # it inverts.
        #
        # THE SWEEPS ARE A `lax.scan`, NOT A PYTHON LOOP, AND THAT IS A SCALING DECISION. Unrolling
        # emits one copy of the operator apply per sweep, so the compiled program grows with the sweep
        # count -- and this reconstruction nests, since the Hessian solve runs once per outer apply,
        # making the program `outer x inner` applies. Measured on a 1.6M-cell mesh: at 468 applies the
        # scanned form compiles and runs in 230 s at 7.6 GB while the unrolled one is killed by the
        # operating system during COMPILATION, not execution -- runtime memory is flat in the sweep
        # count either way. A scan compiles one body whatever the depth.
        #
        # `sweeps` stays a static field, which is exactly what `length` wants: it never becomes a
        # tracer, so the calibration and `narrow_gradient_sweeps` keep working on it unchanged.
        peeled = self.relaxation * preconditioner.apply(rhs)
        if self.sweeps == 1:
            # A single sweep IS the peel, and returning it here rather than scanning zero times keeps
            # the operator out of the traced program entirely. `lax.scan` traces its body even at
            # `length=0`, so without this the operator would be traced for a solve that never applies
            # it -- harmless at run time, but it costs a trace and makes the peel invisible to
            # anything reading the program.
            return peeled

        def sweep(carry, _):
            x, _previous = carry
            residual = rhs - op(x)
            return (x + self.relaxation * preconditioner.apply(residual), residual), None

        # The carry holds the residual as well as the iterate, purely so the diagnostic below can
        # still read it: it is the residual that *formed* the final update, so the reported ratio is
        # the same quantity the unrolled loop reported, one apply already spent and free.
        (x, residual), _ = jax.lax.scan(
            sweep,
            (peeled, rhs),
            None,
            length=self.sweeps - 1,
        )
        # The convergence diagnostic norms the residual over the whole local vector. Under domain
        # decomposition that vector holds each partition's ghost/null rows too, so a faithful global
        # norm would need an owned-only cross-partition reduction the operator-wrapping seam does not
        # carry; the sweep count is a static, mesh-property-driven choice, so the distributed path
        # drops the (unreliable) diagnostic rather than report a per-partition norm.
        # ...and it is skipped at ONE sweep, where it carries no information rather than a little. The
        # residual below is the one entering the final update, so at a single sweep it is the initial
        # `rhs` and the ratio is *exactly* 1 whatever the mesh — it fires on a perfectly orthogonal
        # grid whose answer is exact. A single sweep is `g = V⁻¹Bφ`, the uncorrected Green–Gauss
        # reconstruction, so there is no correction being under-resolved to report on.
        if operator_hook is None and self.warn_tol is not None and self.sweeps > 1:
            # `residual` is rhs - A·x from the last sweep (one apply already spent) — a free,
            # slightly conservative convergence indicator. The host-side warning is gated behind a
            # `lax.cond` on the tolerance so the (host-synchronizing) callback fires *only* when the
            # sweeps are actually under-resolved; on a converged mesh no callback runs, so the check
            # is free in the common case.
            relative = jnp.linalg.norm(residual) / (
                jnp.linalg.norm(rhs) + jnp.finfo(rhs.dtype).tiny
            )
            jax.lax.cond(
                relative > self.warn_tol,
                lambda: jax.debug.callback(
                    _warn_gradient_unconverged, self.sweeps, self.warn_tol, ordered=False
                ),
                lambda: None,
            )
        return x


def _calibrated_solver(
    system: GradientSystem,
    *,
    tol: float,
    iters: int,
    floor: int,
    cap: int,
    seed: int,
    warn_tol: float | None = _DEFAULT_WARN_TOL,
) -> SweptGradientSolve:
    """Measure ``system`` and return the fixed-sweep strategy reaching ``tol`` on it.

    Every scheme factory that calibrates itself is written against this, so none of them constructs
    a :class:`SweptGradientSolve` itself: the schemes differ in *which* systems they have to
    calibrate and in nothing else, and the calibration surface they expose is one configuration
    rather than one per scheme. Split out because two copies of that surface drift a keyword at a
    time, and no single change to either looks wrong.

    See :class:`SweepCalibration` for what the parameters mean; ``warn_tol`` reaches the built
    solver and is ``None`` for a system solved inside another system's operator.
    """
    calibration = SweepCalibration(tol=tol, iters=iters, floor=floor, cap=cap, seed=seed)
    return SweptGradientSolve(sweeps=calibration.sweeps(system), warn_tol=warn_tol)


_TRACED_GEOMETRY_MESSAGE = (
    "geometry is traced; calibrate outside the differentiated region and pass `sweeps=` explicitly. "
    "The sweep count is a Python int held in the solve strategy's static configuration, and a tracer "
    "cannot become one. This is a property of what the count is, not a limitation of the estimator: "
    "an integer count has zero derivative almost everywhere, and the count is part of the "
    "discretization -- the state solves the residual assembled at that count and the adjoint "
    "differentiates the same one, consistently, whatever it is."
)


def _reject_traced_geometry(system: GradientSystem) -> None:
    """Raise before the estimate if the system was built from traced geometry.

    Left alone, the tracer surfaces as a concretization error from inside a logarithm several frames
    down, naming neither the geometry nor the way out.
    """
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(system.preconditioner)):
        raise ValueError(_TRACED_GEOMETRY_MESSAGE)


class ContractionRate(NamedTuple):
    """The measured convergence rate of a preconditioned-Richardson sweep, with its own self-check.

    Attributes
    ----------
    rate : float
        The estimate at the full apply budget -- ``rho(I - P⁻¹A)``, the factor by which each sweep
        multiplies the remaining error.
    half_budget_rate : float
        The same estimate at half the budget, which is what :attr:`settling_ratio` compares against.
    annihilated : bool
        Whether some iterate's norm reached zero, which makes :attr:`rate` a floor rather than a
        measurement. It is reported instead of being folded into the rate because the two causes
        cannot be told apart from the number alone: a correction that is identically zero on the mesh
        really does have rate zero, while a **singular** system annihilates the probe through
        arithmetic and has no rate at all. A caller comparing candidates must reject the second --
        see :func:`fastest_boundary_closure` -- and one measuring a single known-good system can
        ignore this.

        ⚠️ Which of the two happens is platform-dependent, so a caller that ignores this on a
        degenerate mesh is making a platform-dependent decision. Measured on the same perturbed
        tetrahedral mesh under an owner Hessian closure: a rate of **8.64** under macOS Accelerate
        against **2.2e-308** (the smallest normal double, this floor) under the BLAS on the CI
        runners -- a diverging closure reported as a perfect one.
    """

    rate: float
    half_budget_rate: float
    annihilated: bool = False

    @property
    def settling_ratio(self) -> float:
        """How far the estimate moved over the budget's second half -- ``1`` means it has settled.

        The estimate approaches the true rate **from below**, so a ratio well above one says the
        budget was too short and the rate is being under-reported (which would under-resolve the
        sweep count derived from it). Measured between 1.03 and 1.17 at the default budget, across
        meshes from orthogonal to 40 % perturbed and over both systems of the Hessian-corrected
        scheme, so a ratio in that neighbourhood is the healthy reading and one of, say, 2 is not. Returns infinity if the half-budget estimate is zero,
        which happens only when the iteration annihilates the probe outright.
        """
        if self.half_budget_rate == 0.0:
            return math.inf if self.rate > 0.0 else 1.0
        return self.rate / self.half_budget_rate


def contraction_rate(
    system: GradientSystem,
    *,
    iters: int = 24,
    seed: int = 0,
    norm: Callable[[jnp.ndarray], jnp.ndarray] = jnp.linalg.norm,
) -> ContractionRate:
    """Measure ``rho(I - P⁻¹A)`` for a reconstruction system -- the rate its Richardson sweep converges at.

    The preconditioned-Richardson iteration started from zero has error ``e_k = M^k e_0`` with
    ``M = I - P⁻¹A``, so the relative error after ``k`` sweeps is ``rho^k`` asymptotically. Both
    ``A`` and ``P`` are geometry-only here, so ``rho`` is a **property of the mesh** and can be
    measured once and reused for every reconstruction on it. It spans some eighty-fold across the
    meshes this solver runs on -- measured at ``9e-17`` on an orthogonal grid (where the correction
    vanishes and one sweep is already exact), ``5e-3`` on a backward-facing-step mesh, ``0.11`` at
    20 % random grid perturbation and ``0.44`` at 40 % -- which is why a single fixed sweep count is
    simultaneously far past machine precision on one mesh and short of engineering accuracy on
    another.

    The estimate is the Gelfand form ``rho ~ (prod_k ||M^k v|| / ||M^(k-1) v||)^(1/iters)``: the
    ``iters``-th root of the accumulated growth, rather than a ratio of successive norms. That
    matters because ``M`` is nonsymmetric and its dominant eigenvalue may be a complex-conjugate
    pair, on which successive-norm ratios oscillate indefinitely while the root averages the
    oscillation out.

    **The budget is fixed rather than data-dependent, deliberately.** A stopping test reads the norm
    back to the host on every apply, and a host round trip costs far more than the apply it is
    measuring (measured here at some thirty times the cost of the same apply inside a compiled
    loop). The whole budget therefore runs inside one compiled fixed-length loop with a single host
    synchronization, and :attr:`ContractionRate.settling_ratio` reports whether it was long enough
    instead of a stopping test deciding.

    **Cost.** ``iters`` operator applies, once, plus one preconditioner application each. A
    ``k``-sweep reconstruction spends ``k - 1`` applies, so the default budget is the work of about
    eight four-sweep reconstructions -- paid at build time against a saving on every reconstruction
    thereafter. The ratio is mesh-size-independent, both sides being linear in the mesh.

    Parameters
    ----------
    system : GradientSystem
        The system to measure. Its geometry must be **concrete**: the count derived from this is a
        static Python int, so calibration belongs outside any differentiated or jitted region.
    iters : int, optional
        Apply budget (default 24); at least 2, since half of it is the settledness check.
    seed : int, optional
        Seed for the random starting vector (default 0). A generic start is deficient in no
        eigendirection, so the estimate is insensitive to it; it is exposed so a result can be
        reproduced or a repeat can confirm that insensitivity.
    norm : callable, optional
        Vector norm (default the Euclidean norm over the whole array). **Under domain decomposition
        this default is wrong**: it norms each partition's ghost rows alongside its owned ones, so
        the ghost rows -- which the iteration fills with values their owning partition has not
        supplied -- are counted into the growth. Calibrate on the global mesh **before** partitioning,
        or pass an owned-only cross-partition reduction here *and* a ``system.operator`` that already
        performs its halo exchange (a plain callable, so ``lambda v: operator(exchange(v))`` composes
        it without a further parameter).

    Returns
    -------
    ContractionRate
        The rate and its settledness self-check.

    Raises
    ------
    ValueError
        If ``iters`` is below 2, if the system was built from traced geometry, or if the estimate is
        not finite (which means the iteration overflowed rather than that the mesh is difficult -- a
        divergent iteration reports a rate above one and is not an error here).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from aquaflux.mesh import structured_grid_2d
    >>> from aquaflux.schemes import CorrectedGreenGauss, contraction_rate
    >>> mesh = structured_grid_2d(4, 4, 1.0, 1.0)
    >>> scheme = CorrectedGreenGauss()
    >>> rate = contraction_rate(scheme.system(scheme.terms(mesh, mesh.geometry())))
    >>> bool(rate.rate < 1e-10)  # an orthogonal grid: the correction vanishes, one sweep is exact
    True
    """
    if iters < 2:
        raise ValueError(
            f"contraction_rate needs iters >= 2 (the settledness check halves it); got {iters}."
        )
    _reject_traced_geometry(system)

    preconditioner, operator = system.preconditioner, system.operator
    half = iters // 2

    @jax.jit
    def run(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(k, carry):
            v, log_full, log_half, annihilated = carry
            v = v - preconditioner.apply(operator(v))
            n = norm(v)
            # An iteration that annihilates the probe (a mesh whose correction is identically zero)
            # has rate zero, not one: renormalizing by 1 would leave the vector at zero and every
            # further step would contribute log(1) = 0, reporting a *perfect* mesh as a divergent
            # one. Accumulating log(tiny) instead carries it to zero, which is the true rate.
            positive = n > 0
            v = v / jnp.where(positive, n, 1.0)
            log_n = jnp.log(jnp.where(positive, n, jnp.finfo(n.dtype).tiny))
            return (
                v,
                log_full + log_n,
                log_half + jnp.where(k < half, log_n, 0.0),
                annihilated | ~positive,
            )

        zero = jnp.zeros((), v.dtype)
        _, log_full, log_half, annihilated = jax.lax.fori_loop(
            0, iters, body, (v, zero, zero, jnp.zeros((), bool))
        )
        return jnp.exp(log_full / iters), jnp.exp(log_half / half), annihilated

    v = jax.random.normal(jax.random.PRNGKey(seed), system.shape)
    v = v / norm(v)
    try:
        full_a, half_a, annihilated_a = run(v)
        full, halfway = float(full_a), float(half_a)
        annihilated = bool(annihilated_a)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError) as exc:
        raise ValueError(_TRACED_GEOMETRY_MESSAGE) from exc
    if not (math.isfinite(full) and math.isfinite(halfway)):
        raise ValueError(
            f"the contraction-rate estimate is not finite (rate {full}, half-budget {halfway}); the "
            "iteration overflowed rather than merely diverging, so the operator or preconditioner is "
            "likely mis-assembled."
        )
    return ContractionRate(full, halfway, annihilated)


@dataclasses.dataclass(frozen=True)
class SweepCalibration:
    """How a measured contraction rate becomes a sweep count -- the settings both scheme factories share.

    ``sweeps = ceil(log(tol) / log(rho))`` is the smallest ``k`` with ``rho^k <= tol``, clamped to
    ``[floor, cap]``. It carries **no safety margin**, and that is a measured choice rather than an
    oversight: across 180 combinations (twelve meshes, three fields, five tolerances) the count this
    returns met the tolerance every time. The two errors cancel -- the estimate approaches the rate
    from below, which would under-count, but the same early transient makes the first few sweeps
    reduce the error *faster* than ``rho^k``, which over-delivers by about as much.

    **What the tolerance means, and the one trap in it.** It bounds the reconstructed gradient's
    relative error in the **Euclidean (L2) norm over all cells**, not cell by cell -- and the
    difference is not cosmetic. Measured at the calibrated count for ``tol = 1e-4`` over 26
    combinations (thirteen meshes from 5 % to 40 % grid perturbation plus a backward-facing-step
    mesh, two fields each): the L2 error met the tolerance in **every** one, while the worst single
    cell's relative error ran 5 to 60 times it (median 12) and **exceeded the requested tolerance in
    half of them**. The rate this is derived from is the iteration's dominant eigenvalue, which
    governs the norm and not the extremes, so a mesh whose skewness is concentrated in a few cells
    reaches the L2 target with those cells still short of it.

    So: ask for an L2 tolerance one to two orders tighter than the per-cell accuracy actually wanted,
    and do not read this as a per-cell bound. Note also which way the error does *not* run -- an
    earlier reading of this held that a global rate would size the count for the worst cell and so
    over-resolve the bulk. It is the other way about.

    **Why the default is 1e-4.** The gradient enters the residual as a *correction*, and the two
    reconstruction schemes here differ from each other by one to two percent in L2 on the same mesh,
    so ``1e-4`` sits two orders below the difference between the schemes themselves -- well inside
    the discretization's own uncertainty while still cheap. It picks one sweep on a skew-free mesh,
    two on a mildly non-orthogonal industrial one, five at 20 % grid perturbation, seven at 30 % and
    eleven or twelve at 40 %. A looser ``1e-2`` is *not* defensible: it drops the industrial mesh to
    a single sweep, which is the uncorrected compact reconstruction on a mesh whose worst face
    carries appreciable skewness.

    Attributes
    ----------
    tol : float
        Target relative error of the reconstructed gradient in the L2 norm (default ``1e-4``);
        strictly between 0 and 1.
    iters : int
        Apply budget for the rate estimate (default 24); see :func:`contraction_rate`.
    floor : int
        Smallest count to return (default 1 -- one sweep is the uncorrected reconstruction, which is
        exact where the skewness correction vanishes).
    cap : int
        Largest count to return (default 64). A system whose rate is at or above one cannot be solved
        by this sweep at any count and returns the cap; such a mesh wants
        :class:`GmresGradientSolve` instead.
    seed : int
        Seed for the estimate's starting vector (default 0).
    """

    tol: float = 1e-4
    iters: int = 24
    floor: int = 1
    cap: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        # Checked here rather than where the count is derived, so an inconsistent calibration cannot
        # be constructed at all and every path through it is covered by the one check.
        if not 0.0 < self.tol < 1.0:
            raise ValueError(f"tol must be strictly between 0 and 1; got {self.tol}.")
        if self.iters < 2:
            raise ValueError(f"iters must be at least 2; got {self.iters}.")
        if self.floor < 1:
            raise ValueError(
                f"floor must be at least 1 (a solve runs at least one sweep); got {self.floor}."
            )
        if self.cap < self.floor:
            raise ValueError(f"cap ({self.cap}) must be at least floor ({self.floor}).")

    def sweeps_for(self, rate: float) -> int:
        """Sweep count reaching :attr:`tol` at a contraction rate of ``rate``.

        Parameters
        ----------
        rate : float
            A measured contraction rate, e.g. :attr:`ContractionRate.rate`.

        Returns
        -------
        int
            ``ceil(log(tol) / log(rate))`` clamped to ``[floor, cap]``; the floor at ``rate <= 0``
            (an exactly-solved system) and the cap at ``rate >= 1`` (an iteration that does not
            converge).

        Raises
        ------
        ValueError
            If ``rate`` is negative or not finite.
        """
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"a contraction rate must be finite and non-negative; got {rate}.")
        if rate <= 0.0:
            return self.floor
        if rate >= 1.0:
            return self.cap
        return int(min(self.cap, max(self.floor, math.ceil(math.log(self.tol) / math.log(rate)))))

    def sweeps(self, system: GradientSystem) -> int:
        """Measure ``system``'s contraction rate and return the sweep count reaching :attr:`tol`.

        Parameters
        ----------
        system : GradientSystem
            The system to calibrate, built from concrete geometry.

        Returns
        -------
        int
            The calibrated sweep count.
        """
        rate = contraction_rate(system, iters=self.iters, seed=self.seed)
        return self.sweeps_for(rate.rate)


def fastest_boundary_closure(
    mesh: Mesh,
    geometry: MeshGeometry,
    *,
    candidates: tuple[HessianBoundaryClosure, ...] | None = None,
    local_schur_block: bool = True,
    relaxation: float = 1.0,
    iters: int = SweepCalibration.iters,
    seed: int = SweepCalibration.seed,
) -> HessianBoundaryClosure:
    """The Hessian boundary closure whose coupled sweep converges fastest **on this mesh**.

    The closure is not a global ranking — it is a property of the mesh, and the two shipped choices
    swap places. On a well-shaped mesh the owner closure is the faster (a contraction rate of 0.198
    against 0.304 on the pitzDaily benchmark, six sweeps against eight); on a heavily warped one the
    neighbour-averaged closure is (0.4385 against 0.5378 on a 1.6M-cell reactor mesh with interior
    face skewness p99 0.20 and face planarity down to 0.877, twelve sweeps against fifteen). Picking
    one in advance is therefore wrong on half the meshes it will meet, which is what this measures
    away — the same argument, and the same instrument, as calibrating the sweep count rather than
    assuming it.

    **The contraction rate subsumes solvability, which is why it is the only criterion here.** Both
    shipped closures reproduce a constant Hessian exactly, so neither trades accuracy for speed; what
    separates them on bad cells is that the owner closure can leave a cell's Hessian block singular,
    and a closure whose block is singular does not merely reconstruct badly — it fails to contract at
    all, and so loses on rate by a wide margin rather than winning on it.

    Parameters
    ----------
    mesh : Mesh
        The mesh to measure against; its geometry must be concrete, since the comparison is made
        outside any traced or differentiated region.
    geometry : MeshGeometry
        That mesh's face and cell metrics.
    candidates : tuple of HessianBoundaryClosure, optional
        The closures to compare (default: :class:`OwnerHessian` and
        :class:`AveragedNeighbourHessian`, the two that are exact for a quadratic). Pass a tuple to
        compare a calibrated blend weight, or to add a closure of your own.
    local_schur_block, relaxation : optional
        The configuration the scheme will run with; both change the rate, so both must match what
        will run or this measures a system nobody solves. Defaults are the shipped ones.
    iters, seed : int, optional
        Passed to :func:`contraction_rate`.

    Returns
    -------
    HessianBoundaryClosure
        The candidate with the lowest measured rate. Ties go to the earlier candidate, so the default
        order prefers the cheaper owner closure when the two are indistinguishable.

    Examples
    --------
    >>> from aquaflux.mesh import structured_grid_2d
    >>> from aquaflux.schemes import OwnerHessian, fastest_boundary_closure
    >>> mesh = structured_grid_2d(4, 4, 1.0, 1.0)
    >>> isinstance(fastest_boundary_closure(mesh, mesh.geometry()), OwnerHessian)
    True
    """
    if candidates is None:
        candidates = (OwnerHessian(), AveragedNeighbourHessian())
    if not candidates:
        raise ValueError("fastest_boundary_closure needs at least one candidate closure.")

    # ⚠️ Seeded with NaN-safety in mind: a closure that leaves a cell's Hessian block singular can
    # produce a NON-FINITE rate rather than a large one, and `NaN < best` is False -- so a plain
    # comparison would let such a closure lose *by accident* rather than on its merits, and would
    # rank two broken closures by their order in the list. A non-finite rate is mapped to infinity
    # below, which makes the ordering total and the choice independent of that order.
    best, best_rate = None, math.inf
    for closure in candidates:
        systems = HessianCorrectedGradient._systems(mesh, geometry, closure)
        inner = systems.inner()
        measured = contraction_rate(
            systems.coupled_error(
                relaxation,
                systems.outer_preconditioner(inner, local_schur_block),
                inner.preconditioner,
            ),
            iters=iters,
            seed=seed,
        )
        rate = float(measured.rate)
        # A candidate whose probe was annihilated has no measured rate, only the floor the estimator
        # reports in its place -- and on a closure that leaves a cell's Hessian block singular that
        # floor reads as a PERFECT contraction, which would make the broken closure win. Both
        # candidates here are approximations, so neither can annihilate the probe legitimately; the
        # one case where that would be genuine (a correction identically zero on the mesh) makes the
        # closures equivalent anyway, so ranking them both unusable loses nothing.
        if measured.annihilated or not math.isfinite(rate):
            rate = math.inf
        if best is None or rate < best_rate:
            best, best_rate = closure, rate
    return best


def narrow_gradient_sweeps(tree: _Tree, sweeps: int) -> _Tree:
    """Copy ``tree`` with every :class:`SweptGradientSolve` and :class:`CoupledBlockSweep` inside it
    capped at ``sweeps``.

    **What this is for: capping how far a residual's Jacobian reaches on the cell graph.** Each
    Richardson sweep applies ``A_g`` once, and ``A_g`` couples a cell to its face neighbours — so an
    ``n``-sweep reconstruction reads cell values ``n`` cells away, and a residual built on it reads
    them ``n + 1`` cells away (a face flux gathers the gradient of the cells on both sides), wherever
    the mesh is skewed enough for the correction to be live. On an orthogonal mesh the correction
    vanishes and the extra sweeps have nothing to add, so narrowing is free there — in reach exactly,
    and in value to round-off.

    That matters to a preconditioner assembled by **coloured probing**, which recovers the Jacobian
    over the cell graph out to a fixed distance. Where the residual reaches further than the probe
    does, the far couplings are not dropped: a colouring is collision-free only for the pattern it
    was built at, so two same-coloured cells can both couple to one row and the whole response is
    charged to whichever of them lies inside the pattern. The far entry is **folded onto a near one**,
    which perturbs entries the factorization turns into pivots rather than merely omitting small
    terms. Probing a narrowed copy instead keeps the recovered matrix exact for the residual it was
    taken from, leaving a bounded, stated approximation in place of a corrupted one.

    The narrowed copy is for the **preconditioner only**. The solve's own operator stays the exact
    Jacobian--vector product of the full residual, so neither the converged state nor its adjoint is
    affected — a preconditioner changes how a Krylov solve gets to the answer, never where it lands.

    **On :class:`HessianCorrectedGradient` this reaches both of its solvers, and they are not alike.**
    Narrowing the outer Schur solve trades reach for accuracy the same way it does for a corrected
    Green--Gauss gradient. Narrowing the *inner* Hessian solve additionally changes which operator is
    being solved — the eliminated system's ``A_HH⁻¹`` is what the sweep approximates — so the narrowed
    copy stops being exact for quadratic fields far sooner than the sweep count alone suggests. That
    is legitimate in a preconditioner, which is a stated approximation either way, and it is *why* the
    inner count is worth narrowing: those sweeps carry stencil reach exactly as the outer ones do.

    Parameters
    ----------
    tree : Any
        Any object holding gradient schemes — an assembled case, a residual assembler, a scheme.
        Traversed through ``equinox.Module`` fields and through tuples and lists; anything else is
        returned as it stands.
    sweeps : int
        The sweep ceiling. A solve already at or below it is left alone (and returned by identity),
        so this only ever **narrows** — it cannot hand a preconditioner a wider stencil than the
        residual it approximates.

    Returns
    -------
    Any
        The rewritten tree, of the same type as ``tree``; ``tree`` itself when it holds no swept
        solve above the ceiling.

    Raises
    ------
    ValueError
        If ``sweeps`` is below one. A single sweep is ``g = P⁻¹ B phi`` — the uncorrected
        Green–Gauss reconstruction under the inverse-volume preconditioner — and there is nothing
        narrower to ask for.

    Examples
    --------
    >>> scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=4))
    >>> narrow_gradient_sweeps(scheme, 2).solver.sweeps
    2
    >>> narrow_gradient_sweeps(scheme, 8) is scheme  # never widens
    True
    """
    if sweeps < 1:
        raise ValueError(f"narrow_gradient_sweeps: sweeps must be at least 1, got {sweeps}.")

    def rewrite(node: object) -> object:
        if isinstance(node, SweptGradientSolve):
            if node.sweeps <= sweeps:
                return node
            # Carry `relaxation` as well as `warn_tol`: a narrowed copy must differ from its original
            # in the sweep count and in nothing else. Rebuilding it at the class default silently
            # un-damps an under-relaxed solve, and on a mesh skewed enough to need that damping the
            # undamped iteration does not converge at all -- so the copy would diverge where the
            # original was fine, with no sign that a setting had been dropped.
            return SweptGradientSolve(
                sweeps=sweeps, warn_tol=node.warn_tol, relaxation=node.relaxation
            )
        # ⚠️ THE COUPLED SWEEP MUST BE NARROWED TOO, and missing it would fail SILENTLY. Its count
        # sets the reconstruction's stencil exactly as the swept solve's does, and it *replaces* the
        # two swept solvers rather than sitting beside them — so a scheme using it has no
        # `SweptGradientSolve` left for this to find, and narrowing would return the tree unchanged
        # while a caller believed the stencil had been bounded. That is the same shape as the trap a
        # Krylov outer solver already carries here, and the reason to state it at both classes.
        if isinstance(node, CoupledBlockSweep):
            if node.sweeps <= sweeps:
                return node
            return CoupledBlockSweep(sweeps=sweeps, relaxation=node.relaxation)
        if isinstance(node, eqx.Module):
            # `sweeps` is a static field, so it lives in the pytree's structure rather than among its
            # leaves and `tree_at` cannot reach it; rebuilding each Module along the path is how a
            # static field is replaced. Every Module here takes its fields as constructor arguments,
            # which is what makes `dataclasses.replace` faithful.
            changed = {}
            for field in dataclasses.fields(node):
                if not field.init:  # not a constructor argument, so `replace` cannot carry it
                    continue
                value = getattr(node, field.name)
                rewritten = rewrite(value)
                if rewritten is not value:
                    changed[field.name] = rewritten
            return dataclasses.replace(node, **changed) if changed else node
        if isinstance(node, tuple | list):
            rewritten = [rewrite(item) for item in node]
            if all(new is old for new, old in zip(rewritten, node, strict=True)):
                return node
            # A named tuple takes its entries positionally; a plain tuple or list takes the sequence.
            return type(node)(*rewritten) if hasattr(node, "_fields") else type(node)(rewritten)
        return node

    return rewrite(tree)


class HessianSolve(eqx.Module):
    """Strategy: how the gradient is obtained from the coupled gradient-and-Hessian system.

    The scheme assembles two systems, and there is more than one defensible way to get a gradient out
    of them — sweep both blocks together, eliminate the Hessian and solve the two separately, or solve
    the un-eliminated system whole. Each is one of these, and a scheme carries **exactly one**.

    That is the point of the interface rather than a consequence of it. These paths do not share
    settings: an inner-solve strategy means nothing to a coupled sweep, and a sweep count means
    nothing to a nested solve. Held as separate fields on the scheme they would all be present at
    once, most of them inert, and a caller configuring the wrong one would be ignored in silence —
    which is exactly what happened when the coupled sweep first became the default, and cost four
    call sites in this repository their meaning. Under one field the mistake cannot be written down.
    """

    def gradients(
        self,
        systems: _HessianSystems,
        field: jnp.ndarray,
        boundary_values: jnp.ndarray,
        *,
        local_schur_block: bool,
    ) -> jnp.ndarray:
        """Reconstruct the cell-centred gradient from the assembled systems.

        Parameters
        ----------
        systems : _HessianSystems
            The scheme's systems at one geometry.
        field : jnp.ndarray
            The field being reconstructed, shape ``(n_cells,)``.
        boundary_values : jnp.ndarray
            Its values at boundary-face centroids, shape ``(n_faces,)``.
        local_schur_block : bool
            Whether the outer preconditioner is built from the Schur complement's own per-cell block
            rather than from ``A_gg``'s alone.

        Returns
        -------
        jnp.ndarray
            The gradient, shape ``(n_cells, dim)``.
        """
        raise NotImplementedError


class CoupledBlockSweep(HessianSolve):
    """Sweep the gradient and Hessian blocks together, instead of nesting a solve inside each apply.

    The eliminated system is solved today by a sweep on the gradient whose every operator apply runs
    a *complete* Hessian solve inside it — so the Hessian is re-converged from zero once per outer
    sweep, discarding what the previous one learned. Sweeping both blocks alternately keeps that
    work:

    .. code-block:: text

        h <- h + P_H^-1 (A_Hg g  -  A_HH h)
        g <- g + w P_g^-1 (b_g  -  A_gg g  +  A_gH h)

    **The fixed point is the same Schur solution, and that is provable rather than measured.** At a
    joint fixed point the first line forces ``A_HH h = A_Hg g``, hence ``h = A_HH^-1 A_Hg g``;
    substituting into the second gives ``(A_gg - A_gH A_HH^-1 A_Hg) g = b_g``, which is ``S g = b_g``.
    Both preconditioners are invertible, so each step is an equivalence and not merely an implication.

    **The system stays gradient-sized.** No enlarged unknown is formed and no solve is run on the
    packed ``[g, H]`` vector; ``h`` is an iterate of this sweep, not an unknown of a larger system.

    ⚠️ **``h`` STARTS AT ZERO ON EVERY CALL, and that is a correctness requirement.** Carried between
    calls it would make the reconstruction history-dependent -- the same field would give a slightly
    different gradient depending on what was reconstructed before it -- which breaks the linearity
    everything downstream assumes and puts bias into the adjoint. It is the distinction between this
    and warm-starting a solve from a previous *step's* answer, which is a different proposal and a
    refuted one.

    Attributes
    ----------
    sweeps : int
        Number of coupled sweeps (static). One sweep costs **two** face-kernel passes against the
        nested path's ``1 + inner``, so a given accuracy is reached for far less work -- but the count
        needed is not the nested path's outer count and has to be calibrated on its own.

        ⚠️ **The count is what this costs, not the work inside one sweep, and the two are easy to
        confuse.** On a 12225-cell mesh the sweeps are ~82 % of a reconstruction, and *halving* the
        face-kernel passes per sweep (from four to two, by merging each row into one evaluation) moved
        the whole reconstruction by only 9 %: the compiler was already folding away most of the
        redundant work, because the arguments being wasted were literal zeros rather than runtime
        values. Cutting the count is worth several times as much -- on that mesh the sweep contracts
        at 0.315, so twelve sweeps reproduce twenty to ``1e-9`` for a third less time. Calibrate the
        count before optimizing the sweep.
    relaxation : float
        Damping on the gradient update, in ``(0, 1]``. A differentiable leaf.
    """

    sweeps: int = eqx.field(static=True, default=20)
    relaxation: float = 1.0

    def gradients(
        self,
        systems: _HessianSystems,
        field: jnp.ndarray,
        boundary_values: jnp.ndarray,
        *,
        local_schur_block: bool,
    ) -> jnp.ndarray:
        inner = systems.inner()
        # The outer PRECONDITIONER, not the outer system: this sweep never applies the Schur
        # operator, whose construction would want an inner solve it would then discard.
        return systems.block_sweep(
            self,
            systems.outer_preconditioner(inner, local_schur_block),
            inner.preconditioner,
            systems.gradient_rhs(field, boundary_values),
        )

    @classmethod
    def calibrated(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        tol: float = SweepCalibration.tol,
        iters: int = SweepCalibration.iters,
        floor: int = SweepCalibration.floor,
        cap: int = SweepCalibration.cap,
        seed: int = SweepCalibration.seed,
        local_schur_block: bool = True,
        boundary_closure: HessianBoundaryClosure | None = None,
    ) -> CoupledBlockSweep:
        """Build this sweep with its count **measured from the mesh** rather than assumed.

        Its count is not the nested solve's outer count and cannot be borrowed from it: a coupled
        sweep costs about a third as much and converges at its own rate, so a count that is right for
        one is wrong for the other. The rate is a property of the mesh — both operators and both
        preconditioners are geometry-only — so it is measured once here and reused for every
        reconstruction on that mesh.

        Parameters
        ----------
        mesh, geometry
            The mesh to measure, and its geometry. Must be **concrete**: the count derived from this
            is a static Python int, so calibration belongs outside any differentiated or jitted
            region.
        tol, iters, floor, cap, seed
            The shared calibration settings; see :class:`SweepCalibration`.
        local_schur_block : bool, optional
            As on :class:`HessianCorrectedGradient` (default ``True``). ⚠️ **It must match the value
            the scheme will run with, because it selects the preconditioner and so the rate.** It is
            not a detail on a skewed mesh: on perturbed tetrahedra under
            :class:`AveragedNeighbourHessian` the two preconditioners contract at 0.6620 and 0.8980,
            which is 23 sweeps against 86 for the same tolerance. Where the cell shapes are good it
            barely registers (0.2551 against 0.2570 at 35 % hex perturbation), which is why a
            mismatch here hides on easy meshes and surfaces on the ones this scheme exists for.

        Returns
        -------
        CoupledBlockSweep
            The sweep, at the calibrated count and this class's default relaxation. ⚠️ A non-default
            relaxation changes the rate and so the count, and is **not** calibrated here — measure it
            with :func:`contraction_rate` on
            ``HessianCorrectedGradient._systems(...).coupled_error(...)`` if you use one.

        Notes
        -----
        ⚠️ **The count this returns is CONSERVATIVE for the gradient, and knowingly so.** The rate is
        measured on the packed ``[g, h]`` error, because the two blocks converge together and the
        asymptotic rate belongs to the pair — but the Hessian's error dominates that estimate while
        the gradient's falls faster. Measured at the shared default tolerance, the reconstruction
        comes in five to seven times *better* than asked: 5.0e-07 against a nested solve's 3.4e-06 on
        an orthogonal grid, 1.1e-06 against 5.2e-06 at 30 % perturbation. Read a comparison against
        another scheme at equal *tolerance* accordingly — this one is not spending its budget the
        same way.

        Examples
        --------
        >>> from aquaflux.mesh import structured_grid_2d
        >>> from aquaflux.schemes import CoupledBlockSweep
        >>> mesh = structured_grid_2d(4, 4, 1.0, 1.0)
        >>> CoupledBlockSweep.calibrated(mesh, mesh.geometry()).sweeps
        6
        """
        # The closure and the preconditioner both change the operator whose rate is being measured,
        # so both have to be the ones the scheme will run with -- the same reason the scheme's own
        # factory takes them. Reached through `outer_preconditioner` rather than `outer`, which would
        # additionally build the Schur operator this sweep never applies.
        systems = HessianCorrectedGradient._systems(
            mesh, geometry, OwnerHessian() if boundary_closure is None else boundary_closure
        )
        inner = systems.inner()
        rate = contraction_rate(
            systems.coupled_error(
                cls.relaxation,
                systems.outer_preconditioner(inner, local_schur_block),
                inner.preconditioner,
            ),
            iters=iters,
            seed=seed,
        )
        settings = SweepCalibration(tol=tol, iters=iters, floor=floor, cap=cap, seed=seed)
        return cls(sweeps=settings.sweeps_for(rate.rate))


class CorrectedGreenGauss(GradientScheme):
    """Green–Gauss with the non-orthogonal skewness correction — a coupled sparse system.

    The corrected face value adds a gradient-based extrapolation from the P–N line to the
    face centroid:

        phi_ip = (1-g) phi_P + g phi_N  +  [(1-g) grad(phi)_P + g grad(phi)_N] . D_g,ip

    where ``g`` is the projection factor of the face centroid onto the P–N line and
    ``D_g,ip = x_ip - x_g`` is the skewness offset. Because the correction depends on the
    *gradients* of the cell and its neighbours, substituting into Green–Gauss gives a
    nearest-neighbour-coupled linear system

        A_g . G = B . phi ,     A_g = V (.) I  -  (the correction coupling)

    with ``A_g`` **geometry-only** and well-conditioned (``V`` dominates for mild skew). *How* the
    system is solved is an injected :class:`GradientSolve` strategy — :class:`SweptGradientSolve`
    (default; fixed sweeps, ``O(n)``, and cheap to differentiate inside a nonlinear solve) or
    :class:`GmresGradientSolve` (exact via ``lineax`` + implicit diff, for a mesh skewed enough that
    the swept sweep count would grow impractically); the discretization is identical either way. This
    is the standalone,
    physics-free form; coupling ``A_g``/``B`` into a flow Newton solve later is the Schur step (same
    ``A_g``/``B``). The correction makes the face value exact for linear fields, so the
    reconstruction is **linear-exact on any mesh** — the fix for :class:`CompactGreenGauss`'s
    inconsistency on irregular grids.

    Attributes
    ----------
    solver : GradientSolve
        The strategy applying ``A_g⁻¹`` to solve ``A_g·G = B·φ`` (default
        :class:`SweptGradientSolve`, the scalable unrolled sweep; use :class:`GmresGradientSolve` for
        a mesh skewed enough to need an exact Krylov solve).
    """

    solver: GradientSolve = eqx.field(default_factory=SweptGradientSolve)
    preconditioner: CellPreconditioner = eqx.field(default_factory=InverseCellVolume)

    @staticmethod
    def terms(mesh: Mesh, geometry: MeshGeometry) -> _CorrectedTerms:
        """Geometry-only intermediates of the corrected-gradient system (operator + RHS share them).

        Parameters
        ----------
        mesh : Mesh
            Owner/neighbour connectivity.
        geometry : MeshGeometry
            Face and cell metrics (centroids, owner-outward area vectors, volumes).

        Returns
        -------
        _CorrectedTerms
            The bundled per-face/per-cell geometry the operator and RHS both consume.
        """
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        x_p = cell_geometry.centroid[face_cells.owner]
        d = (
            face_cells.neighbour_centroid(cell_geometry.centroid) - x_p
        )  # periodic-image across seam
        g = interpolation_factor(face_cells, geometry)
        skew = face_geometry.centroid - (x_p + scale(d, g))  # D_g,ip: offset from P–N line to face
        area_vector = scale(face_geometry.normal, face_geometry.area)  # owner-outward S_f
        return _CorrectedTerms(face_cells, g, skew, area_vector, cell_geometry.volume)

    @classmethod
    def operator(cls, t: _CorrectedTerms) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The field-independent, geometry-only linear operator ``A_g`` (a matvec on the gradient).

        ``A_g = V ⊙ I − (correction coupling)``; it depends only on ``t``, never on the field, and is
        volume-dominated — which is exactly what lets :class:`SweptGradientSolve` invert it by a
        few fixed matrix-free sweeps.
        """
        fc = t.face_cells
        owner, nb = fc.owner, fc.safe_neighbour

        def matvec(grad: jnp.ndarray) -> jnp.ndarray:
            w = (1.0 - t.g) * dot(t.skew, grad[owner]) + t.g * dot(t.skew, grad[nb])
            # the correction vanishes on boundary faces (owner side too), so pre-mask before scatter
            correction = fc.scatter_conservative(
                fc.combine_face_values(scale(t.area_vector, w), 0.0)
            )
            return scale(grad, t.volume) - correction

        return matvec

    @classmethod
    def system(
        cls, t: _CorrectedTerms, preconditioner: CellPreconditioner | None = None
    ) -> GradientSystem:
        """The gradient system ``A_g·G = B·φ`` as a solve strategy sees it — operator and preconditioner.

        This is where the choice of preconditioner for *this* system is made: ``A_g``'s per-cell
        block is the cell volume less the skewness coupling, so :class:`InverseVolume` is within the
        skewness of the true block and the sweep it drives converges quickly. Bundling it with the
        operator gives the reconstruction and the calibration one assembly to share, so a count
        calibrated here cannot be measured against a different pairing than the one that runs.

        Parameters
        ----------
        t : _CorrectedTerms
            The geometry intermediates from :meth:`terms`.

        Returns
        -------
        GradientSystem
            Operator, preconditioner, and the ``(n_cells, dim)`` shape of the gradient.
        """
        n_cells, dim = t.volume.shape[0], t.area_vector.shape[-1]
        chosen = InverseCellVolume() if preconditioner is None else preconditioner
        return GradientSystem(chosen.build(t), cls.operator(t), (n_cells, dim))

    @classmethod
    def calibrated(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        tol: float = SweepCalibration.tol,
        iters: int = SweepCalibration.iters,
        floor: int = SweepCalibration.floor,
        cap: int = SweepCalibration.cap,
        seed: int = SweepCalibration.seed,
        preconditioner: CellPreconditioner | None = None,
    ) -> CorrectedGreenGauss:
        """Build this scheme with the sweep count **measured from the mesh** rather than assumed.

        The sweep count that reaches a given accuracy is set by the mesh's non-orthogonality and
        spans some fifty-fold across the meshes this solver runs on, so a count chosen once for all
        meshes is far past machine precision on some and short of engineering accuracy on others —
        and a fixed sweep carries no convergence test, so being short of it is silent. This measures
        the system's contraction rate once (``iters`` operator applies, about seven reconstructions)
        and returns the smallest count reaching ``tol``.

        The count is a concrete Python int, so this must run **outside** any differentiated or
        traced region — build the scheme here, then use it inside. That is a property of the count
        rather than a restriction: an integer has zero derivative almost everywhere, and the count is
        part of the discretization, with the state and its adjoint using the same one.

        Under domain decomposition, call this on the **global** mesh before partitioning; see
        :func:`contraction_rate` for why a per-partition estimate is not the same measurement.

        Parameters
        ----------
        mesh : Mesh
            The mesh to calibrate against; its geometry must be concrete.
        geometry : MeshGeometry
            That mesh's face and cell metrics.
        tol : float, optional
            Target relative gradient error in the **L2 norm over all cells** (default ``1e-4``) — not
            a per-cell bound. The worst single cell was measured at 5 to 60 times this and exceeded
            it in half the meshes tried; see :class:`SweepCalibration` before choosing a value.
        iters : int, optional
            Apply budget for the rate estimate (default 24).
        floor, cap : int, optional
            Bounds on the returned count (defaults 1 and 64).
        seed : int, optional
            Seed for the estimate's starting vector (default 0).

        Returns
        -------
        CorrectedGreenGauss
            The scheme, carrying a :class:`SweptGradientSolve` at the calibrated count.

        Examples
        --------
        >>> from aquaflux.mesh import structured_grid_2d
        >>> from aquaflux.schemes import CorrectedGreenGauss
        >>> mesh = structured_grid_2d(4, 4, 1.0, 1.0)
        >>> scheme = CorrectedGreenGauss.calibrated(mesh, mesh.geometry())
        >>> scheme.solver.sweeps  # orthogonal: the correction vanishes, one sweep is exact
        1
        """
        chosen = InverseCellVolume() if preconditioner is None else preconditioner
        return cls(
            solver=_calibrated_solver(
                cls.system(cls.terms(mesh, geometry), chosen),
                tol=tol,
                iters=iters,
                floor=floor,
                cap=cap,
                seed=seed,
            ),
            # The measured count belongs to the pairing it was measured on, so the preconditioner
            # travels with it: calibrating under one and running under another would report a count
            # for a system that never runs.
            preconditioner=chosen,
        )

    @classmethod
    def rhs(
        cls, t: _CorrectedTerms, field: jnp.ndarray, boundary_values: jnp.ndarray
    ) -> jnp.ndarray:
        """The right-hand side ``B·φ``: base (interpolated) Green–Gauss with exact boundary values."""
        fc = t.face_cells
        phi_base = fc.combine_face_values(
            interpolate_owner_neighbour(field, t.g, fc), boundary_values
        )
        return fc.scatter_conservative(scale(t.area_vector, phi_base))

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
    ) -> jnp.ndarray:
        # No internal consumer: this scheme solves for the gradient and returns it, so an imposed
        # row is applied by `gradients` and nothing here reads the row it replaces. It never
        # differentiates a boundary value either, so the corrected ones are of no use to it.
        del imposed, boundary_values_at
        t = self.terms(mesh, geometry)
        system = self.system(t, self.preconditioner)
        return self.solver.solve(
            system.preconditioner,
            system.operator,
            self.rhs(t, field, boundary_values),
            operator_hook=operator_hook,
        )


class PreparedBoundaryClosure(NamedTuple):
    """A boundary closure with its geometry bound in: what the operator applies, and what the
    per-cell block probes should see.

    Attributes
    ----------
    apply : callable
        ``(n_cells, dim, dim) -> (n_cells, dim, dim)``, linear in its argument — the Hessian a
        boundary face's gradient extrapolation carries.
    diagonal : callable
        ``probe -> the part of `apply` that lands on each cell's OWN diagonal block``. The blocks are
        recovered by probing the face kernels with a **uniform** field, and that recovery is exact
        only because zeroing one side of a face leaves each cell reading its own value. A closure
        that reads *neighbours'* Hessians breaks that: the gather returns the probe's own value and
        is indistinguishable from a diagonal term. So each closure states its diagonal contribution
        rather than letting the probe infer one — and, because that contribution can be per-cell (a
        closure that falls back to the owner's own Hessian on some cells and not others), it is a
        callable over the same bound geometry rather than a constant.
    """

    apply: Callable[[jnp.ndarray], jnp.ndarray]
    diagonal: Callable[[jnp.ndarray], jnp.ndarray]


class HessianBoundaryClosure(eqx.Module):
    """Strategy: which first-order Hessian a boundary face's gradient extrapolation carries.

    A boundary face has no neighbour to interpolate with, so the gradient there is extrapolated from
    the owner and needs *some* estimate of the Hessian to carry it the remaining distance. Only
    first-order accuracy is required of that estimate, which leaves a genuine choice — and the two
    options here trade against each other rather than one dominating, which is why this is a strategy
    and not a setting.
    """

    def prepare(
        self, face_cells: FaceCellConnectivity, separation: jnp.ndarray
    ) -> PreparedBoundaryClosure:
        """Bind the geometry once, returning the pair the operator and the block probes need.

        Both halves come from one call because a closure's diagonal contribution can depend on the
        same geometry its map does — which cell falls back to its own Hessian, for instance — and
        deriving them separately is how the two drift apart.

        Parameters
        ----------
        face_cells : FaceCellConnectivity
            The mesh's face-to-cell connectivity.
        separation : jnp.ndarray
            Owner-to-neighbour centroid vector per face, shape ``(n_faces, dim)``.

        Returns
        -------
        PreparedBoundaryClosure
            Its ``apply`` and ``diagonal``, both closed over this geometry.
        """
        raise NotImplementedError


class OwnerHessian(HessianBoundaryClosure):
    """Carry the cell's **own** Hessian across the boundary extrapolation — the default.

    Exact for a quadratic field on any mesh where the resulting system is solvable, which is the
    property this whole scheme exists for: the Hessian of a quadratic is what the extrapolation needs
    and this supplies it without approximation.

    ⚠️ **It can leave the Hessian system under-determined, and on a wholly tetrahedral mesh it
    does.** The measured condition number of ``A_HH`` there is ~1e17 against ~4 under
    :class:`AveragedInteriorHessian` — numerically singular, so no solver recovers it. Betchen &
    Straatman's own illustration is a mesh one cell thick in some direction, where the second
    derivative across that direction appears in no equation at all and is therefore arbitrary; a
    tetrahedral mesh is a less obvious instance of the same thing.
    """

    def prepare(
        self, face_cells: FaceCellConnectivity, separation: jnp.ndarray
    ) -> PreparedBoundaryClosure:
        # The identity, and wholly diagonal: every cell reads its own Hessian and no other.
        return PreparedBoundaryClosure(apply=lambda h: h, diagonal=lambda probe: probe)


def _inverse_distance_average(
    face_cells: FaceCellConnectivity,
    separation: jnp.ndarray,
    eligible_cell: jnp.ndarray,
    *,
    fall_back_to_owner: bool,
) -> PreparedBoundaryClosure:
    """Average a cell's neighbours' Hessians by inverse distance, over a given eligible set.

    Shared by both averaging closures, which differ **only** in which neighbours they count — so the
    weighting, the empty case and the diagonal declaration are stated once here rather than twice.

    ``fall_back_to_owner`` decides the empty case, and it is not a detail: falling back to the cell's
    own Hessian keeps the weights a partition of unity on **every** cell, and with it the exactness
    for a quadratic — a quadratic's Hessian is constant, and any weighted average of a constant whose
    weights sum to one returns it unchanged. Falling back to zero drops the curvature term on those
    cells and loses exactness there. ⚠️ The two cannot be mixed and matched with the eligible set:
    with the narrow set an owner fallback re-admits the cell's own Hessian to its own closure often
    enough to make the Hessian system singular again on a tetrahedral mesh (measured ``cond(A_HH)``
    ``1.7e+18``), which is why the narrow closure keeps the zero fallback and pays for it in accuracy.

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The mesh's face-to-cell connectivity.
    separation : jnp.ndarray
        Owner-to-neighbour centroid vector per face, shape ``(n_faces, dim)``.
    eligible_cell : jnp.ndarray
        Per-cell boolean, shape ``(n_cells,)``: may this cell be averaged over?
    fall_back_to_owner : bool
        Where a cell has no eligible neighbour, take its own Hessian (``True``) or zero (``False``).

    Returns
    -------
    PreparedBoundaryClosure
        The averaging map and its per-cell diagonal contribution.
    """
    owner, nb = face_cells.owner, face_cells.safe_neighbour
    interior = face_cells.interior
    inverse_distance = 1.0 / jnp.linalg.norm(separation, axis=-1)
    # One weight per direction of travel: a cell averages across faces it owns and faces it
    # neighbours, and the eligibility test applies to whichever cell is on the far side.
    from_neighbour = jnp.where(interior & eligible_cell[nb], inverse_distance, 0.0)
    from_owner = jnp.where(interior & eligible_cell[owner], inverse_distance, 0.0)
    total = face_cells.scatter(from_neighbour, from_owner)
    found = total > 0.0
    inverse_total = jnp.where(found, 1.0 / jnp.where(found, total, 1.0), 0.0)

    def apply(h: jnp.ndarray) -> jnp.ndarray:
        gathered = face_cells.scatter(
            from_neighbour[:, None, None] * h[nb], from_owner[:, None, None] * h[owner]
        )
        averaged = gathered * inverse_total[:, None, None]
        if not fall_back_to_owner:
            return averaged
        return jnp.where(found[:, None, None], averaged, h)

    # A cell's own Hessian appears only on the fallback branch, so only those cells contribute to
    # their own diagonal block. Declaring the probe everywhere would report the neighbour gather as
    # diagonal; declaring zero everywhere would drop the fallback cells' real diagonal term.
    def diagonal(probe: jnp.ndarray) -> jnp.ndarray:
        if not fall_back_to_owner:
            return jnp.zeros_like(probe)
        return jnp.where(found[:, None, None], 0.0, probe)

    return PreparedBoundaryClosure(apply=apply, diagonal=diagonal)


class AveragedNeighbourHessian(HessianBoundaryClosure):
    """Average the Hessian over **all** of the cell's face neighbours — exact *and* solvable.

    This is the closure to reach for when :class:`OwnerHessian` will not solve. It keeps every
    property that matters:

    * **Exact for a quadratic**, like :class:`OwnerHessian` and unlike
      :class:`AveragedInteriorHessian`. Every cell on a connected mesh has at least one face
      neighbour, so the weights sum to one everywhere and a constant Hessian passes through
      unchanged. Measured under a converged solve: ``2.6e-15`` on a tetrahedral mesh, ``6.0e-16``
      on a perturbed hexahedral one, ``9.4e-16`` in 2D.
    * **Leaves the Hessian system solvable**, because a cell's own Hessian never enters its own
      boundary closure. On a wholly tetrahedral mesh ``cond(A_HH)`` is ``8.4`` here against
      ``1.0e+18`` under :class:`OwnerHessian`.

    ⚠️ **The coupling it adds is invisible to the per-cell preconditioner, so it costs inner sweeps.**
    The Hessian system is solved by preconditioned Richardson over a per-cell block — block Jacobi by
    construction — and the neighbour coupling this introduces is precisely what such a block cannot
    represent. Calibrated to ``tol = 1e-10``, the inner count rises from 8 to 23 at ``weight = 1``.
    ⚠️ **That cost does NOT amortize with mesh size**: it is 23 at 27 cells and 23 at 1728, even as
    the share of cells owning a boundary face falls from 96 % to 42 %, because the slowest mode lives
    in the coupled rows however few of them there are.

    ``weight`` is what that cost is traded against, and it trades well. The closure is
    ``(1 - weight)`` of the cell's own Hessian plus ``weight`` of the neighbour average, which is a
    partition of unity for **any** weight and so exact for a quadratic at all of them — only the
    strength of the decoupling changes. Measured on a tetrahedral mesh and a 25 %-perturbed
    hexahedral one:

    ======  ================  ================  =========================
    weight  ``cond(A_HH)``    inner sweeps      quadratic at 20/10 sweeps
    ======  ================  ================  =========================
    1.0     8.4               23                5.1e-07
    0.4     35.7              15                2.1e-09
    0.2     142               11                9.8e-12
    0.1     ~480              —                 3.7e-14
    0.0     1.0e+18           10                *unsolvable*
    ======  ================  ================  =========================

    A little coupling is enough to break the degeneracy, and a small weight keeps both the sweep count
    and the fixed-sweep accuracy near the default closure's. **The default is ``0.2``**, which on the
    meshes measured is where that trade sits — but the table above is one tetrahedral mesh, and the
    right weight is a property of the mesh rather than a constant. **Prefer
    :meth:`calibrated`, which measures it** (and returns ``0.0``, costing nothing at all, on a mesh
    whose cells close their own boundaries perfectly well).
    """

    weight: float = eqx.field(static=True, default=0.2)

    def prepare(
        self, face_cells: FaceCellConnectivity, separation: jnp.ndarray
    ) -> PreparedBoundaryClosure:
        averaged = _inverse_distance_average(
            face_cells,
            separation,
            jnp.ones(face_cells.n_cells, dtype=bool),
            fall_back_to_owner=True,
        )
        if self.weight == 1.0:
            return averaged
        if self.weight == 0.0:
            # Identical to `OwnerHessian`, and short-circuited so it also costs the same: with no
            # averaging in the blend there is no reason to pay for the gather and scatter that
            # builds it. `calibrated` returns this weight on any mesh the owner closure already
            # closes, so it is a live path rather than a defensive branch.
            return PreparedBoundaryClosure(apply=lambda h: h, diagonal=lambda probe: probe)
        # The retained fraction of the cell's own Hessian is diagonal, and the averaged part carries
        # its own declaration — so the blend's diagonal is the blend of the two.
        own = self.weight
        return PreparedBoundaryClosure(
            apply=lambda h: (1.0 - own) * h + own * averaged.apply(h),
            diagonal=lambda probe: (1.0 - own) * probe + own * averaged.diagonal(probe),
        )

    @classmethod
    def calibrated(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        tol: float = SweepCalibration.tol,
        iters: int = SweepCalibration.iters,
        floor: int = SweepCalibration.floor,
        cap: int = SweepCalibration.cap,
        seed: int = SweepCalibration.seed,
        weights: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 1.0),
    ) -> AveragedNeighbourHessian:
        """Choose the blend weight **from the mesh** rather than assuming one.

        The weight trades a well-conditioned Hessian system against the sweeps needed to solve it, and
        which way that trades is a property of the mesh — so it is measured, by the same estimator the
        sweep counts use. For each candidate weight this measures the Hessian system's contraction
        rate and converts it to a sweep count, then takes the weight needing the fewest.

        On a mesh where the cell's own Hessian closes the system perfectly well the ladder's smallest
        weight is fastest and is chosen, so the closure adds as little coupling as it offers; on a
        mesh where it does not, the weights that leave the system singular saturate at ``cap`` and
        lose to any weight that works.

        ⚠️ **The ladder deliberately excludes weight ``0``, and the reason is reproducibility rather
        than taste.** At zero this closure *is* :class:`OwnerHessian`, which on the meshes that need
        this closure at all leaves the Hessian system numerically singular — and the rate of a
        singular system is not a stable measurement. It depends on how the platform's linear algebra
        inverts a near-singular block: the same mesh measured ``15.4`` (expansive, correctly rejected)
        on one machine and comfortably below one on another, where it was then chosen as the cheapest
        option. A calibration must not decide anything from that number. Ask for no coupling by naming
        :class:`OwnerHessian`, which is a choice rather than a measurement.

        Costs ``iters`` operator applies per candidate weight, once, off the differentiated path.

        Parameters
        ----------
        mesh : Mesh
            The mesh to measure on.
        geometry : MeshGeometry
            Its geometry. Must be concrete — see :meth:`HessianCorrectedGradient.calibrated`.
        tol, iters, floor, cap, seed
            The calibration settings, as elsewhere: ``tol`` is the residual reduction a sweep count is
            sized for, and the rest configure the rate estimate and bound the count.
        weights : tuple of float
            Candidate weights, in ``(0, 1]``. Ties are broken toward the **smallest**, which prefers
            the weaker coupling and so the cheaper solve. Zero is excluded by default — see above.

        Returns
        -------
        AveragedNeighbourHessian
            The closure carrying the chosen weight.

        Examples
        --------
        >>> from aquaflux.mesh import structured_grid_3d
        >>> from aquaflux.schemes import AveragedNeighbourHessian
        >>> mesh = structured_grid_3d(3, 3, 3)
        >>> AveragedNeighbourHessian.calibrated(mesh, mesh.geometry()).weight
        0.05
        """
        settings = SweepCalibration(tol=tol, iters=iters, floor=floor, cap=cap, seed=seed)
        best, fewest = weights[0], None
        for weight in weights:
            system = HessianCorrectedGradient._systems(mesh, geometry, cls(weight=weight)).inner()
            needed = settings.sweeps(system)
            if fewest is None or needed < fewest:
                best, fewest = weight, needed
        return cls(weight=best)


class AveragedInteriorHessian(HessianBoundaryClosure):
    """Betchen & Straatman's closure: the inverse-distance average of the Hessian over the cell's
    neighbours that are themselves **clear of the boundary**.

    ⚠️ **Kept for comparison against the source, and not recommended — use
    :class:`AveragedNeighbourHessian` instead**, which is exact for a quadratic where this is not and
    keeps the system just as solvable (both reach ``cond(A_HH) ~ 8`` on a tetrahedral mesh).

    Where :class:`OwnerHessian` can leave the system under-determined, this cannot: the boundary
    extrapolation stops depending on the very Hessian component the boundary fails to constrain, and
    reaches for neighbours that are constrained instead. On a tetrahedral mesh that is the difference
    between ``cond(A_HH) ~ 1e17`` and ``~ 8``.

    ⚠️ **It gives up exactness for quadratics, and that is not a small print.** Where a cell has no
    boundary-clear neighbour at all — a corner, an edge, or any cell on a mesh too coarse to have an
    interior — the average is empty and the closure is zero, so the extrapolation drops its curvature
    term entirely and that cell is first-order. Measured on a quadratic over a 25 %-perturbed
    hexahedral grid: median relative error ``3.3e-03`` at 3³, ``2.0e-03`` at 5³ and ``4.0e-05`` at 8³
    as the interior grows, against ``~1e-15`` under :class:`OwnerHessian` at every size. That is
    consistent with the source, which reports second-*order* gradients rather than exact ones.

    ⚠️ **That loss is a property of THIS eligible set, not of averaging.** Restricting to
    boundary-clear neighbours is what empties the average on some cells; averaging over *all* face
    neighbours never does, and so stays exact — see :class:`AveragedNeighbourHessian`. The narrow set
    does need its zero fallback, though: pairing it with an owner fallback re-admits enough of the
    cell's own Hessian to make the tetrahedral system singular again (``cond(A_HH) ~ 1.7e+18``).
    """

    def prepare(
        self, face_cells: FaceCellConnectivity, separation: jnp.ndarray
    ) -> PreparedBoundaryClosure:
        # Eligible: a cell owning no boundary face of its own. The zero fallback is load-bearing for
        # THIS eligible set -- see `_inverse_distance_average`.
        boundary_faces_per_cell = face_cells.scatter(
            jnp.where(face_cells.interior, 0.0, 1.0), jnp.zeros(separation.shape[0])
        )
        return _inverse_distance_average(
            face_cells, separation, boundary_faces_per_cell == 0.0, fall_back_to_owner=False
        )


class NestedHessianSolve(HessianSolve):
    """Eliminate the Hessian and solve the two systems separately — the Hessian re-converged from
    zero inside every apply of the outer one.

    The two systems are not alike, so they take separate strategies: ``solver`` drives the outer Schur
    system on the gradient, ``hessian_solver`` the inner ``A_HH`` system that runs inside every one of
    its operator applies.

    ⚠️ **This is the slower path on the meshes this scheme exists for**, because it discards what the
    previous outer apply learned about the Hessian: 1.4x the cost of :class:`CoupledBlockSweep` at
    30 % grid perturbation, 1.8x at 40 %, and up to 12x under :class:`AveragedNeighbourHessian`,
    whose Hessian system is the expensive one to re-converge. Those ratios were measured while one
    coupled sweep cost four face-kernel passes rather than today's two, so they understate the gap. It is
    faster only on an orthogonal mesh, where the outer solve is nearly trivial and there is nothing to
    save — and where this scheme has no advantage worth its cost anyway.

    Kept because it is the arrangement every measurement in this scheme's history was taken on, and
    the control any comparison against the coupled sweep needs.

    Attributes
    ----------
    solver : GradientSolve
        Drives the outer Schur system.
    hessian_solver : GradientSolve
        Drives the inner ``A_HH`` system. Its ``warn_tol`` is ``None`` by default — see below.
    """

    solver: GradientSolve = eqx.field(default_factory=lambda: SweptGradientSolve(sweeps=20))
    hessian_solver: GradientSolve = eqx.field(
        default_factory=lambda: SweptGradientSolve(sweeps=10, warn_tol=None)
    )

    def gradients(
        self,
        systems: _HessianSystems,
        field: jnp.ndarray,
        boundary_values: jnp.ndarray,
        *,
        local_schur_block: bool,
    ) -> jnp.ndarray:
        if self.solver.requires_linear_operator and self.hessian_solver.emits_host_diagnostics:
            raise ValueError(
                "hessian_solver runs inside the outer Schur operator, and this outer solver forms "
                "its implicit-diff tangent by transposing that operator — which requires the "
                "operator be strictly linear. The inner solver's under-resolution diagnostic norms "
                "the residual, which is not, so the transpose fails deep inside the linear solver "
                "with an uninformative error. Pass warn_tol=None on the hessian_solver (e.g. "
                "SweptGradientSolve(sweeps=6, warn_tol=None)), or use a fixed-sweep outer `solver`, "
                "which differentiates by unrolling and imposes no such requirement. The outer "
                "`solver`'s own diagnostic is unaffected."
            )
        inner = systems.inner()
        outer = systems.outer(self.hessian_solver, inner, local_schur_block)
        return self.solver.solve(
            outer.preconditioner, outer.operator, systems.gradient_rhs(field, boundary_values)
        )

    @classmethod
    def calibrated(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        tol: float = SweepCalibration.tol,
        iters: int = SweepCalibration.iters,
        floor: int = SweepCalibration.floor,
        cap: int = SweepCalibration.cap,
        seed: int = SweepCalibration.seed,
        local_schur_block: bool = True,
        boundary_closure: HessianBoundaryClosure | None = None,
    ) -> NestedHessianSolve:
        """Build this path with **both** counts measured from the mesh rather than assumed.

        The two systems converge at very different rates and neither is well served by a count chosen
        in advance, so both are measured: the inner Hessian system first, then the outer system
        *using that calibrated inner solve*, which is the composition that will actually run.

        Parameters
        ----------
        mesh, geometry
            The mesh to measure on and its geometry. The geometry must be concrete.
        tol, iters, floor, cap, seed
            Calibration settings, as elsewhere.
        local_schur_block : bool
            Which outer preconditioner the outer count is measured against — it changes the operator.
        boundary_closure : HessianBoundaryClosure, optional
            The closure to measure under; it changes the operator too.

        Returns
        -------
        NestedHessianSolve
            Carrying the two measured counts.
        """
        settings = dict(tol=tol, iters=iters, floor=floor, cap=cap, seed=seed)
        closure = OwnerHessian() if boundary_closure is None else boundary_closure
        systems = HessianCorrectedGradient._systems(mesh, geometry, closure)
        inner_system = systems.inner()
        # The inner solve runs inside the outer operator, whose transpose an outer Krylov strategy
        # would form, so its diagnostic is disabled — the same pairing the field default carries.
        hessian_solver = _calibrated_solver(inner_system, warn_tol=None, **settings)
        return cls(
            solver=_calibrated_solver(
                systems.outer(hessian_solver, inner_system, local_schur_block), **settings
            ),
            hessian_solver=hessian_solver,
        )


class PackedSystemSolve(HessianSolve):
    """Solve the un-eliminated ``[g, h]`` system whole, without eliminating the Hessian.

    This exists to check the elimination against the system it eliminates from: the two must agree to
    machine precision, and that they do is a property of the discretization rather than of either
    solver. It is not the production path — the whole point of the elimination is that the gradient is
    the only primary unknown, which is what lets it Schur-couple into a flow solve at gradient size.

    Deliberately plain: the packed system is preconditioned by the cell volume, which serves both
    blocks because both diagonal blocks carry it.

    Attributes
    ----------
    solver : GradientSolve
        Drives the packed system.
    """

    solver: GradientSolve = eqx.field(default_factory=lambda: SweptGradientSolve(sweeps=20))

    def gradients(
        self,
        systems: _HessianSystems,
        field: jnp.ndarray,
        boundary_values: jnp.ndarray,
        *,
        local_schur_block: bool,
    ) -> jnp.ndarray:
        packed = self.solver.solve(
            systems.coupled.preconditioner,
            systems.coupled.operator,
            systems.coupled_rhs(field, boundary_values),
        )
        return packed[:, : systems.dim]


class _HessianSystems(NamedTuple):
    """The linear systems of the Hessian-corrected reconstruction at one geometry.

    Everything here is geometry-only except the two right-hand sides, which take the field. The
    scheme has two shapes — the Hessian eliminated (an inner system on the Hessian inside an outer
    system on the gradient) or the two solved together as one packed system — so this carries both,
    and a consumer takes the pair it needs.

    Attributes
    ----------
    gradient_rhs : callable
        ``(field, boundary_values) -> b_g``, the right-hand side of the gradient equation, which is
        also the eliminated system's right-hand side unreduced.
    coupled_rhs : callable
        ``(field, boundary_values) -> [b_g, 0]``, the same packed for the un-eliminated system.
    coupled : GradientSystem
        The un-eliminated system on the packed unknown ``[g, H]``.
    inner : callable
        ``() -> GradientSystem`` for the Hessian system ``A_HH``. A call builds the per-cell block
        its preconditioner inverts, which is why it is deferred rather than a field.
    block_sweep : callable
        ``(sweep, gradient_preconditioner, hessian_preconditioner, b_g) -> g``, the coupled
        block sweep of :class:`CoupledBlockSweep` over these same operators.
    coupled_error : callable
        ``(relaxation, gradient_preconditioner, hessian_preconditioner) -> GradientSystem`` whose
        contraction rate is that block sweep's.
    outer : callable
        ``(hessian_solver, inner_system) -> GradientSystem`` for the Schur system on the gradient.
        The inner solve runs inside its operator, so the strategy solving it is part of the operator
        and is injected here — which is what lets the operator be measured with the same inner solve
        that will run inside it.
    """

    dim: int
    gradient_rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    coupled_rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    coupled: GradientSystem
    inner: Callable[[], GradientSystem]
    outer: Callable[[GradientSolve, GradientSystem], GradientSystem]
    outer_preconditioner: Callable[..., GradientPreconditioner]
    hessian_row_defect: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    gradient_row_defect: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    block_sweep: Callable[..., jnp.ndarray]
    coupled_error: Callable[..., GradientSystem]


class HessianCorrectedGradient(GradientScheme):
    """Second-order gradient via Betchen's coupled gradient + Hessian reconstruction, with
    the **Hessian Schur-eliminated** so only the gradient is the primary unknown.

    Betchen & Straatman (2010) reconstruct the gradient by a Green–Gauss sum with a
    face-curvature correction, and the Hessian by a Green–Gauss sum of the gradient
    components — a coupled linear system in ``[g, H]`` per cell. The Hessian is needed only
    to lift the gradient to 2nd order; it is not wanted as an output. So the coupled system

        [ A_gg  A_gH ] [ g ]   [ b_g ]
        [ A_Hg  A_HH ] [ H ] = [  0  ]

    is reduced by **Schur elimination of ``H``** to a gradient-only system
    ``S·g = b_g`` with ``S = A_gg − A_gH · A_HH⁻¹ · A_Hg`` (``A_HH`` geometry-only,
    well-conditioned). Every block comes from **AD** — the residual is the forward
    reconstruction (a few interpolations and Green–Gauss sums), never hand-derived
    coefficient matrices — and the ``A_HH⁻¹`` is applied matrix-free by the injected solver.
    Pass ``hessian_solve=PackedSystemSolve()`` to solve the full ``[g, H]`` system instead, which
    is how the elimination is checked against the system it eliminates from.

    The right-hand side of the eliminated system is ``b_g`` **unreduced**: the Hessian equation is a
    Green–Gauss sum of gradient components and so carries no term in the reconstructed field at all,
    making ``b_H`` identically zero for every field and mesh. There is therefore no ``A_gH·A_HH⁻¹·b_H``
    correction to form, and no inner solve to run for it.

    **The Hessian is carried as its independent components, not as a full tensor.** The Hessian of a
    twice-continuously-differentiable field is symmetric, so ``dim (dim + 1) / 2`` components carry it
    — six in three dimensions rather than nine. Solving for those leaves ``dim**2`` equations for
    fewer unknowns, and the surplus is removed in the least-squares sense weighted by the cell's own
    block, as Betchen & Straatman (2010) do. Three consequences, of which the second is the one to
    know:

    * the unknown, and every Hessian iterate on the reverse-mode tape, is a third smaller;
    * the reduced per-cell block is **symmetric positive definite by construction**, where the
      unsymmetrized block is neither symmetric nor definite. ⚠️ That property is *not* what makes
      this work: an unweighted projection, whose block carries no such guarantee, fixes the same
      reactor mesh equally well (0 diverging cells either way). The symmetry is doing the work, and
      the weighting is kept because it is the source's formulation and costs one contraction; and
    * that block no longer factors. An unsymmetrized ``H`` enters its own equation only as ``H·a``,
      which touches one tensor index and leaves the other alone, making the block ``I ⊗ C`` and
      storable as ``(dim, dim)``. Symmetry couples the two indices, so the block is a dense
      ``(n_sym, n_sym)`` — four times the storage in three dimensions, against a third less for the
      unknown itself.

    *How* each system is solved is an injected :class:`GradientSolve`, exactly as in
    :class:`CorrectedGreenGauss` — but there are **two** systems here and they are not alike, so they
    take separate strategies rather than sharing one:

    - :class:`NestedHessianSolve`'s ``solver`` drives the **outer** Schur system on the gradient. It is well conditioned (the
      measured condition number of ``S`` is between 1 and 7 across mild-to-heavy skew), so a
      preconditioned fixed sweep solves it without a Krylov method — which is the default, and the
      reason is what a *march* pays rather than what one reconstruction costs. A Krylov solve is
      differentiated by the implicit function theorem, so every Jacobian--vector product solves a
      second Schur system, each with its own inner solve inside every one of its iterations; a fixed
      sweep is differentiated by unrolling. Measured per reconstruction against a corrected
      Green--Gauss baseline of 1.0, at equal accuracy: a Krylov outer costs ``92.9x`` forward and
      ``158.6x`` as a jvp, this default ``66.5x`` and ``59.9x`` — **2.6x cheaper on the jvp path**,
      which is the one a Krylov flow solve pays per iteration.
    - its ``hessian_solver`` drives the **inner** ``A_HH`` system on the Hessian, once per outer operator
      apply. Its cost is multiplied by the outer iteration count, so it is the one that decides
      whether the scheme is affordable — and it needs **no Krylov solve at all**: paired with the
      per-cell :class:`CellBlockJacobi` block below, a fixed handful of Richardson sweeps takes it to
      machine precision. Its accuracy sets how faithfully the applied operator matches ``S``, and
      hence the reconstruction's exactness for quadratic fields; the outer *rate* is almost
      independent of it (measured: the outer ``ρ`` moves by 0.3% between one inner sweep and twelve).

    **Choosing the outer sweep count.** Twenty, because that is what preserves the scheme's defining
    property on a skewed mesh rather than what is cheapest. Departure from an exactly-solved
    reconstruction of the same system, on a quadratic field:

    ==========================  ========  ========  ========  ========
    mesh                        outer 5   outer 10  outer 15  outer 20
    ==========================  ========  ========  ========  ========
    2D grid, 20% perturbed      4.8e-05   1.2e-08   5.0e-12   1.7e-15
    2D grid, 30% perturbed      1.2e-04   7.4e-08   6.2e-11   7.9e-14
    3D hex grid, 25% perturbed  1.3e-04   1.5e-07   1.7e-10   1.9e-13
    ==========================  ========  ========  ========  ========

    Cost is linear in this count, so a lower one is available and is a reasonable *case* choice where
    the mesh is mild: five sweeps costs a quarter as much and still reconstructs two orders closer to
    the exact gradient than a corrected Green--Gauss reconstruction differs from it, which is the
    comparison that matters when the two schemes are being weighed against each other. What it gives
    up is exactness for quadratics, so it is not the library default.

    **Choosing the inner sweep count.** The inner iteration contracts geometrically at a rate set by
    the mesh's skewness alone, so the count is a mesh property and not a size one. Measured on a
    heavily skewed hexahedral grid, the reconstruction's departure from exactness for a quadratic
    falls by about two and a half orders per two sweeps — ``4.8e-04`` at two sweeps, ``1.3e-06`` at
    four, ``3.0e-09`` at six, ``8.6e-12`` at eight — and reaches the exactly-solved answer to machine
    precision at ten, which is the default. A mesh skewed enough to need more will lose quadratic
    exactness *quietly*, because the fixed sweep carries no convergence test; the inner solver's own
    under-resolution warning cannot help here (it is disabled by default for the transposability
    reason below), so on an unfamiliar mesh check the count directly by comparing against
    ``hessian_solver=GmresGradientSolve()``, which solves the same system exactly.

    **The inner solver must not emit host diagnostics.** It runs inside the outer Schur operator, and
    an outer Krylov ``solver`` forms its implicit-diff tangent by transposing that operator — which
    requires the operator be strictly linear. A convergence diagnostic is not, so the default inner
    solver sets ``warn_tol=None`` and the combination that would fail is rejected with an explanation
    rather than left to fail inside the linear solver. A fixed-sweep outer ``solver`` differentiates
    by unrolling instead and carries no such restriction.

    **Why the inner system needs a block preconditioner and not ``1/V``.** ``A_HH``'s per-cell block
    is not close to ``V·I``: the Hessian enters its own equation through the gradient it reconstructs,
    at the same order as the volume term. Inverse-volume Richardson on it has a spectral radius of
    ``0.5`` — *on an orthogonal mesh*, where there is no skewness at all — which is why an inner
    Krylov solve looked necessary. Inverting the cell's own block instead drops that to ``0`` on an
    orthogonal mesh and to ~``0.1`` at heavy skew. The block is ``(n_sym, n_sym)`` per cell, where
    ``n_sym = dim (dim + 1) / 2`` is the number of independent components of the symmetric Hessian
    this scheme solves for — six in three dimensions rather than nine.

    Exact for linear *and* quadratic fields on any mesh (the Hessian captures the exact
    second derivative), and 2nd-order for smooth fields — the reconstruction that removes
    :class:`CorrectedGreenGauss`'s ~1st-order cap on irregular grids.

    Attributes
    ----------
    hessian_solve : HessianSolve
        **How** the two systems are solved, as one injected strategy — the scheme carries exactly
        one, so a path's settings travel with the path. :class:`CoupledBlockSweep` (the default)
        sweeps both blocks together, carrying the Hessian from one sweep to the next;
        :class:`NestedHessianSolve` eliminates the Hessian and re-converges it from zero inside every
        apply of the outer solve; :class:`PackedSystemSolve` solves the un-eliminated ``[g, h]``
        system whole, which is the check that the elimination changes nothing.

        The default is the coupled sweep because it matches the nested pair's accuracy at a fraction
        of the face-kernel passes -- 40 against 220 for the ``20`` / ``20+10`` pair -- and is worth
        1.4x at 30 % grid perturbation, 1.8x at 40 %, and up to 12x under
        :class:`AveragedNeighbourHessian`. ⚠️ Those speed ratios were measured while one sweep cost
        **four** passes rather than today's two, so they understate the coupled path; the pass counts
        themselves are exact. And read either against the caveat on
        :attr:`CoupledBlockSweep.sweeps`: a face-pass count is a poor predictor of wall clock here. ⚠️ It is *slower* on an orthogonal mesh, where
        the nested outer solve is nearly trivial — which is also a mesh on which this scheme has no
        advantage worth its cost.
    local_schur_block : bool
        Build the outer preconditioner from the Schur complement's own per-cell block rather than
        from ``A_gg``'s alone. **Default ``True``.**

        It is the better approximation — on a 1.6M-cell reactor mesh its block sits within 3.4 % of
        the true Schur block where ``A_gg``'s is 21 % away, and on synthetic meshes it improves the
        reconstruction everywhere, by some four orders on a cell squashed nearly flat — and it is
        also the convergent one. Setting it ``False`` on that mesh raises the sweep's contraction
        rate from ``0.31`` to ``2.04`` and the worst cell's error from ``5.2e-03`` to ``5.8e+05``.

        ⚠️ **IT DIVERGED ON A REAL MESH WHILE THE HESSIAN WAS SOLVED UNSYMMETRIZED, and solving the
        six independent components instead removed that entirely.** On a 1.6M-cell reactor mesh the
        worst cell went from **5.6e+18** to **5.2e-03** and the count above 100 % error from **4599
        of 1 635 909** to **none** -- while ``A_gg``'s block, which had been the safe arm, went the
        other way and now reaches 5.8e+05 on the same mesh. The reading that stood here, that a
        stationary iteration wants a diagonally dominant preconditioner rather than an accurate one,
        survives as a description of what an unsymmetrized reduction did; it is not a reason to avoid
        this block, which is now both the accurate arm and the convergent one.

        The mechanism this is consistent with -- and it is **not** isolated by that measurement -- is
        that the elimination term is ``A_gH A_HH⁻¹ A_Hg`` and the reduction changes ``A_HH``. ⚠️ The
        definiteness is **not** the mechanism: an unweighted projection, which gives no such
        guarantee, removes the divergence on the same mesh just as completely, so what matters is
        that the Hessian is symmetric rather than how the surplus equations are removed.

        The cells that remain hardest are four-faced and sit **against a boundary** (all twelve of the
        worst own a boundary face, against 23 % of the mesh doing so), which is where this scheme
        closes the Hessian by the simpler of the two treatments in the literature.

        ``validation/uvreactor_openfoam/schur_block_diagnosis.py`` reproduces all of it.
    """

    hessian_solve: HessianSolve = eqx.field(default_factory=CoupledBlockSweep)
    local_schur_block: bool = eqx.field(static=True, default=True)
    boundary_closure: HessianBoundaryClosure = eqx.field(default_factory=OwnerHessian)
    prepared_outer: GradientPreconditioner | None = None

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
    ) -> jnp.ndarray:
        if operator_hook is not None:
            raise NotImplementedError(
                "HessianCorrectedGradient cannot run domain-decomposed: the gradient couples to the "
                "Hessian through the nested Schur and inner A_HH solves, whose operators read ghost "
                "gradients and ghost Hessians that a single per-apply exchange of the outer gradient "
                "does not refresh. A correct distributed build would exchange inside each nested "
                "solve — not yet built. Use CorrectedGreenGauss with SweptGradientSolve for a "
                "distributed non-orthogonal gradient."
            )
        # ⚠️ Deliberately applied by `gradients` to the converged answer, and NOT projected onto the
        # sweep's gradient iterate, which would be the analogue of what the multiple-correction
        # scheme does. Overwriting rows of an iterate changes the fixed point rather than the path
        # to it: the coupled sweep's whole justification is that its fixed point is the Schur
        # solution of the gradient--Hessian system, and a projected sweep converges to a different
        # system's answer. Imposing a gradient here would have to enter as a constraint on that
        # system -- a real piece of work, and not one this seam licenses.
        del imposed, boundary_values_at
        if self.prepared_outer is not None and self.prepared_outer.inverse.shape[0] != mesh.n_cells:
            raise ValueError(
                "this gradient scheme was bound to a geometry of "
                f"{self.prepared_outer.inverse.shape[0]} cells and is being asked to reconstruct on "
                f"one of {mesh.n_cells}. A bound scheme carries that geometry's outer preconditioner, "
                "which with a fixed sweep count changes the reconstructed gradient and not merely "
                "how fast it converges -- so it cannot be reused across meshes. Call bind() again "
                "for this geometry, or drop the binding."
            )
        return self.hessian_solve.gradients(
            self._systems(mesh, geometry, self.boundary_closure, self.prepared_outer),
            field,
            boundary_values,
            local_schur_block=self.local_schur_block,
        )

    def bind(self, mesh: Mesh, geometry: MeshGeometry) -> HessianCorrectedGradient:
        """This scheme carrying the outer preconditioner it would otherwise rebuild every call.

        The outer preconditioner is geometry-only, yet it is rebuilt on every reconstruction --
        every field, and every Krylov matvec, since a matvec re-executes the residual. Within one
        compiled residual the compiler already shares it across the fields, so what binding
        collects is the repetition *across* residual evaluations.

        It is worth most where the sweep count is lowest, because the sweeps are what it is
        competing with: on pitzDaily the whole geometry prologue is ~18 % of a reconstruction at
        twenty sweeps and ~48 % at the seven a 1e-4 calibration asks for, of which this
        preconditioner is over half. It is also the cheapest part of that prologue to hold --
        ``(n_cells, dim, dim)``, against the ``(n_cells, n_sym, n_sym)`` inner inverse that is four
        times larger in three dimensions and that this scheme now rebuilds by algebra anyway.

        Parameters
        ----------
        mesh : Mesh
            The mesh to bind to; its geometry must be concrete.
        geometry : MeshGeometry
            That mesh's face and cell metrics.

        Returns
        -------
        HessianCorrectedGradient
            The same scheme carrying this geometry's outer preconditioner.

        Notes
        -----
        ⚠️ **A bound scheme is valid for the geometry it was bound to and no other.** With a fixed
        sweep count the preconditioner determines the answer, not merely the rate at which the sweep
        reaches it, so a stale binding returns a subtly wrong gradient rather than a slower one. A
        cell-count mismatch is refused outright; a *different* geometry with the same cell count
        cannot be detected and is the caller's responsibility.

        For the same reason, do not bind outside a region being differentiated with respect to the
        **geometry** (node positions, say): the bound preconditioner is then a constant that the
        differentiation cannot see through, and the shape derivative comes back wrong rather than
        failing. Binding is transparent to differentiation with respect to the *field*, which is
        what a flow solve differentiates.

        Examples
        --------
        >>> from aquaflux.schemes import HessianCorrectedGradient
        >>> from aquaflux.mesh import structured_grid_2d
        >>> mesh = structured_grid_2d(4, 4, 1.0, 1.0)
        >>> scheme = HessianCorrectedGradient().bind(mesh, mesh.geometry())
        >>> scheme.prepared_outer is None
        False
        """
        systems = self._systems(mesh, geometry, self.boundary_closure)
        prepared = systems.outer_preconditioner(systems.inner(), self.local_schur_block)
        return HessianCorrectedGradient(
            hessian_solve=self.hessian_solve,
            local_schur_block=self.local_schur_block,
            boundary_closure=self.boundary_closure,
            prepared_outer=prepared,
        )

    @classmethod
    def calibrated(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        tol: float = SweepCalibration.tol,
        iters: int = SweepCalibration.iters,
        floor: int = SweepCalibration.floor,
        cap: int = SweepCalibration.cap,
        seed: int = SweepCalibration.seed,
        schur: bool = True,
        local_schur_block: bool = True,
        boundary_closure: HessianBoundaryClosure | None = None,
        coupled: bool = True,
    ) -> HessianCorrectedGradient:
        """Build this scheme with **both** sweep counts measured from the mesh rather than assumed.

        The two systems here converge at very different rates and neither is well served by a count
        chosen in advance, so both are measured: the inner Hessian system first, then the outer
        system *using that calibrated inner solve*, which is the composition that will actually run.
        The outer rate is nearly a constant of the scheme rather than of the mesh — its difficulty is
        the gradient--Hessian coupling **within** a cell, not the mesh's skewness — so the outer count
        barely moves between a mild mesh and a heavily perturbed one, while the inner count tracks
        the skewness in the usual way.

        Read the counts this returns against the class defaults with their targets in mind, not
        side by side: the defaults are sized to preserve exactness for quadratic fields (an error
        around ``1e-10``), whereas ``tol`` here defaults to ``1e-4``. The two answer different
        questions, and a calibrated count *below* the default is not evidence the default was wrong.
        Pass a tighter ``tol`` to calibrate for exactness instead.

        The inner truncation sets an error floor no number of outer sweeps removes, but it reaches
        the gradient attenuated by three to five orders, so calibrating both at one tolerance leaves
        margin. That attenuation was measured on two meshes only — check the composition against
        ``hessian_solver=GmresGradientSolve()`` on an unfamiliar one rather than assuming it.

        As with :meth:`CorrectedGreenGauss.calibrated`, the counts are concrete Python ints, so this
        must run **outside** any differentiated or traced region, and under domain decomposition it
        must run on the global mesh before partitioning.

        Parameters
        ----------
        mesh : Mesh
            The mesh to calibrate against; its geometry must be concrete.
        geometry : MeshGeometry
            That mesh's face and cell metrics.
        tol : float, optional
            Target relative gradient error in the L2 norm over all cells (default ``1e-4``), applied
            to both systems. See :class:`SweepCalibration`.
        iters : int, optional
            Apply budget for each rate estimate (default 24). The outer estimate runs an inner solve
            inside every apply, so it is the dearer of the two.
        floor, cap : int, optional
            Bounds on the returned counts (defaults 1 and 64).
        seed : int, optional
            Seed for the estimates' starting vectors (default 0).
        schur : bool, optional
            As on the class (default ``True``). When ``False`` there is no inner system and the
            single calibrated count is for the full packed system.

        Returns
        -------
        HessianCorrectedGradient
            The scheme, carrying :class:`SweptGradientSolve` strategies at the calibrated counts.
        """
        settings = {"tol": tol, "iters": iters, "floor": floor, "cap": cap, "seed": seed}
        # The closure changes the operator, so it must be in force while the counts are measured
        # -- calibrating one system and running another is the whole failure this factory exists
        # to prevent.
        closure = OwnerHessian() if boundary_closure is None else boundary_closure
        if not schur:
            return cls(
                hessian_solve=PackedSystemSolve(
                    solver=_calibrated_solver(
                        cls._systems(mesh, geometry, closure).coupled, **settings
                    )
                ),
                boundary_closure=closure,
            )
        # One field, so one strategy is built and it is the one that will run -- there is no second
        # path left carrying counts nobody measured.
        solve: HessianSolve = (
            CoupledBlockSweep.calibrated(
                mesh,
                geometry,
                local_schur_block=local_schur_block,
                boundary_closure=closure,
                **settings,
            )
            if coupled
            else NestedHessianSolve.calibrated(
                mesh,
                geometry,
                local_schur_block=local_schur_block,
                boundary_closure=closure,
                **settings,
            )
        )
        return cls(
            hessian_solve=solve,
            # Travels with the counts, for the same reason the counts and the preconditioner belong
            # together: it changes which system was measured.
            local_schur_block=local_schur_block,
            boundary_closure=closure,
        )

    @staticmethod
    def _systems(
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_closure: HessianBoundaryClosure | None = None,
        prepared_outer: GradientPreconditioner | None = None,
    ) -> _HessianSystems:
        """Assemble this scheme's linear systems from the geometry — everything but the field.

        The reconstruction solves these and the calibration measures them, so they are assembled
        here once for both: a count measured against a different assembly than the one that runs
        would be calibrating the wrong operator.

        Parameters
        ----------
        prepared_outer : GradientPreconditioner, optional
            The outer system's preconditioner, already built for this geometry by
            :meth:`HessianCorrectedGradient.bind`. Supplied, ``outer_preconditioner`` returns it
            instead of rebuilding it, which is the point of binding — it is geometry-only and is
            otherwise rebuilt on every reconstruction. It is threaded here, at the one place the
            preconditioner is constructed, so every :class:`HessianSolve` strategy picks it up
            without any of them changing shape.
        """
        closure = OwnerHessian() if boundary_closure is None else boundary_closure
        dim = mesh.dim
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        owner = face_cells.owner
        nb = face_cells.safe_neighbour
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        n_sym = symmetric_components(dim)

        x_own = cell_geometry.centroid[owner]
        x_ip = face_geometry.centroid
        x_nb = face_cells.neighbour_centroid(cell_geometry.centroid)  # periodic-image across a seam
        s = x_nb - x_own
        f = interpolation_factor(face_cells, geometry)
        skew = x_ip - (x_own + scale(s, f))  # D_f,ip
        nhat = face_geometry.normal  # owner-outward unit normal
        area_vector = scale(nhat, face_geometry.area)  # S_f
        d_own = x_ip - x_own  # owner centroid → face centroid
        d_nb = x_ip - x_nb  # neighbour centroid → face centroid
        vol = cell_geometry.volume

        # The face-curvature tensor the Hessian contracts against in the gradient equation. Geometry
        # only — no field, no unknown — so it is formed once here rather than inside the face kernel,
        # which runs once per operator apply and so once per iteration of both solves.
        curvature = skew[:, :, None] * skew[:, None, :] - (f * (1.0 - f))[:, None, None] * (
            s[:, :, None] * s[:, None, :]
        )

        # ⚠️ THE FACE'S WARP MOMENT -- zero on a planar face, and the term a second-order
        # Green–Gauss reconstruction silently drops on one that is not. The derivation replaces the
        # face integral of `x (x) n` with `x_ip (x) S`, which holds only when the normal is constant
        # over the face; the residue is exactly this moment. Geometry only, so formed once here.
        #
        # It matters far more than "1% of faces are warped" suggests, because the term it corrects is
        # the curvature correction itself -- the whole reason to run this scheme rather than a
        # corrected Green–Gauss one. Measured on a warped grid at planarity 0.89, dropping it costs
        # the reconstruction of a quadratic ~13 orders (8.6e-15 -> 4.1e-02), the entire advantage.
        warp = face_geometry_scheme(dim).warp_first_moment(
            mesh.node_coords, mesh.face_nodes, x_ip, nhat
        )

        def _warp_moment(h, d):
            # ½ Pᵀ(H d + Hᵀ d) — the Hessian contracted against the face's warp moment, the moment
            # entering the cell sum contracted on each of the Hessian's indices in turn. The
            # reconstruction carries only the independent components of a symmetric tensor, so the
            # two contractions coincide and the half cancels the doubling.
            return jnp.einsum("fjk,fj->fk", warp, jnp.einsum("fij,fj->fi", h, d))

        def _face_gradient(g_own, h_own, g_nb, h_nb, h_bnd):
            """The gradient AT THE FACE CENTROID -- interpolated, then carried across the skewness.

            Exact for a linear gradient field (so, for a quadratic ``phi``): the blend lands on the
            point ``x_own + f s`` and the Hessian term carries it the remaining ``skew`` to ``x_ip``.
            Boundary faces have no neighbour to blend with and extrapolate from the owner instead.

            Both equations need this same quantity -- the Hessian equation sums it over faces, and
            the gradient equation's warp term contracts it against the face's warp moment -- so it is
            defined once here rather than written out in each.
            """
            g_face = blend_owner_neighbour(g_own, g_nb, f, face_cells)
            h_face = blend_owner_neighbour(h_own, h_nb, f, face_cells)
            interior = g_face + jnp.einsum("fij,fj->fi", h_face, skew)
            boundary = g_own[owner] + jnp.einsum("fij,fj->fi", h_bnd[owner], d_own)
            return face_cells.combine_face_values(interior, boundary)

        def _hessian_moment(h, d):
            # ½ dᵀ H d — the Hessian's correction to the mean of φ over the face, so the Green–Gauss
            # face integral is exact for a quadratic. Written from each cell's centroid-to-face
            # vector d (not an explicit face second-moment tensor), so it is dimension-general.
            return 0.5 * jnp.einsum("fi,fij,fj->f", d, h, d)

        # The two equations are kept as separate face kernels because the solve uses them
        # separately far more often than together: the inner Hessian sweep applies only the Hessian
        # equation, and it is the innermost loop in the whole scheme. Each takes the unknown's two
        # sides as separate fields — passing the same pair to both is the operator, and zeroing one
        # side isolates the other's dependence, which is how the per-cell diagonal blocks below come
        # out of these same kernels rather than from a second derivation of the coefficients.
        def gradient_face_terms(g_own, h_own, g_nb, h_nb, h_bnd, fld, bvals):
            """Owner- and neighbour-side face contributions to the gradient equation.

            The face value carried to the Green–Gauss sum is the 2nd-order interpolation plus the
            face-curvature correction, less each side's own Hessian moment about the face.
            """
            h_o, h_n = h_own[owner], h_nb[nb]
            g_face = blend_owner_neighbour(g_own, g_nb, f, face_cells)
            h_face = blend_owner_neighbour(h_own, h_nb, f, face_cells)
            phi_int = (
                interpolate_owner_neighbour(fld, f, face_cells)
                + dot(skew, g_face)
                + 0.5 * jnp.sum(curvature * h_face, axis=(1, 2))
            )
            phi_ip = face_cells.combine_face_values(phi_int, bvals)
            # ⚠️ THE FIRST-ORDER WARP TERM, much the larger of the two. The face integral of `phi n`
            # expands about the face centroid as `phi_ip S + grad(phi).P + ½ H:Q`, and the middle
            # term is dropped by any derivation assuming a planar face, since `P` vanishes there. Its
            # `grad(phi)` is the value AT the face centroid -- the skewness-carried one, and on a
            # boundary face the owner extrapolation -- not the raw blend.
            warp_gradient = jnp.einsum(
                "fjk,fj->fk", warp, _face_gradient(g_own, h_own, g_nb, h_nb, h_bnd)
            )
            owner_side = (
                scale(area_vector, phi_ip - _hessian_moment(h_o, d_own))
                + warp_gradient
                - _warp_moment(h_o, d_own)
            )
            neighbour_side = (
                scale(area_vector, phi_ip - _hessian_moment(h_n, d_nb))
                + warp_gradient
                - _warp_moment(h_n, d_nb)
            )
            # The neighbour's outward normal is the opposite one, which flips the vector area and the
            # warp moment together — so the whole contribution flips, not just the area term.
            return owner_side, -neighbour_side

        def hessian_face_terms(g_own, h_own, g_nb, h_nb, h_bnd):
            """Owner- and neighbour-side face contributions to the Hessian equation.

            A Green–Gauss sum of the gradient components: interior faces take the 2nd-order
            interpolation of the gradient, boundary faces extrapolate it from the owner. It takes no
            field and no boundary values — the Hessian equation carries no term in the reconstructed
            field at all, which is why its right-hand side is identically zero and the eliminated
            system's is ``b_g`` unreduced.
            """
            h_face = blend_owner_neighbour(h_own, h_nb, f, face_cells)
            gi = _face_gradient(g_own, h_own, g_nb, h_nb, h_bnd)
            # ⚠️ THE SAME WARP TERM THE GRADIENT EQUATION NEEDS, for the same reason: this equation is
            # itself a Green–Gauss sum -- of the gradient rather than the field -- so it inherits the
            # planar-face assumption identically. The exact face integral is
            # `grad(phi)(x_ip) (x) S + H.P`. Correcting only the gradient equation leaves the Hessian
            # first-order-wrong on a warped face, and the gradient equation then applies an accurate
            # correction built from an inaccurate Hessian.
            h_at_face = face_cells.combine_face_values(h_face, h_own[owner])
            contribution = gi[:, :, None] * area_vector[:, None, :] + jnp.einsum(
                "fij,fjk->fik", h_at_face, warp
            )
            return contribution, -contribution

        zero_g = jnp.zeros((n_cells, dim))
        zero_h = jnp.zeros((n_cells, dim, dim))
        zero_u = jnp.zeros((n_cells, n_sym))
        zero_f = jnp.zeros(n_cells)
        zero_b = jnp.zeros(n_faces)
        no_face_g = jnp.zeros((n_faces, dim))  # a scatter's unused half needs a real zero array
        no_face_h = jnp.zeros((n_faces, dim, dim))

        # ---- THE HESSIAN A BOUNDARY EXTRAPOLATION CARRIES, an injected strategy.
        #
        # ⚠️ IT IS PASSED INTO THE FACE KERNELS RATHER THAN COMPUTED INSIDE THEM, and that is a
        # correctness requirement rather than a style choice. A closure may read the cell's
        # NEIGHBOURS' Hessians, so a per-cell diagonal block probed through a kernel that computed it
        # internally would pick up neighbour coupling and report it as diagonal — the "compose the
        # blocks, not the operators" trap. Each closure states its own diagonal contribution through
        # `diagonal_probe`, which is what the block extractions below pass.
        prepared_closure = closure.prepare(face_cells, s)
        boundary_hessian = prepared_closure.apply

        # The φ-only right-hand side. Only the gradient equation has one: `hessian_face_terms` takes
        # no field, so the Hessian equation's is identically zero for any field and any mesh — which
        # is also why the eliminated system's right-hand side is this one unreduced.
        def gradient_rhs(fld, bvals):
            return face_cells.scatter(
                *gradient_face_terms(zero_g, zero_h, zero_g, zero_h, zero_h, fld, bvals)
            )

        # ---- The Hessian equation's own block, and the reduction onto symmetric components.
        #
        # The Hessian of a twice-continuously-differentiable field is symmetric, so only `n_sym` of
        # its `dim**2` entries are independent and only those are solved for. That leaves `dim**2`
        # equations for `n_sym` unknowns — over-determined — and they are reduced in the least-squares
        # sense weighted by the cell's own block: the reduced row is `(A_P E)ᵀ` applied to the full
        # row, `E` being the expansion of the packed components. Two consequences, the second being
        # why the weighting earns its cost over an unweighted projection:
        #
        #  * it minimizes the residual of the equation as written, rather than of whatever an
        #    unweighted projection happens to leave; and
        #  * the reduced per-cell diagonal block is `(A_P E)ᵀ(A_P E)`, **symmetric positive definite
        #    by construction**, where the unreduced block is neither symmetric nor definite.
        #
        # `contract_symmetric`'s adjoint identity `<E u, R> = <u, contract(R)>` turns `(A_P E)ᵀ R`
        # into `contract(R A_P)`, so the `(dim², n_sym)` matrix is never formed.
        def _hessian_probe(unit_row):
            """A Hessian field whose first row is ``unit_row`` in every cell, the rest zero."""
            return jnp.zeros((n_cells, dim, dim)).at[:, 0, :].set(unit_row)

        def hh_owner(probe):
            lit = _hessian_probe(probe)
            owner_side, _ = hessian_face_terms(
                zero_g, lit, zero_g, zero_h, prepared_closure.diagonal(lit)
            )
            return face_cells.scatter(owner_side, no_face_h)[:, 0, :]

        def hh_neighbour(probe):
            _, neighbour_side = hessian_face_terms(
                zero_g, zero_h, zero_g, _hessian_probe(probe), zero_h
            )
            return face_cells.scatter(no_face_h, neighbour_side)[:, 0, :]

        # `A_P`: the UNREDUCED per-cell block, exactly `I_dim ⊗ C` for this `(dim, dim)` `C`, because
        # an unsymmetrized Hessian enters its own equation only as `H·a` for per-face vectors `a` —
        # which contracts its second index and leaves the first untouched, so one row's probe gives
        # every row's. Geometry-only, and now needed by the un-eliminated system as well as the
        # eliminated one (the reduction is part of the equation, not part of the elimination), so it
        # is built here rather than inside `inner()`.
        hessian_cell_block = cell_diagonal_block(hh_owner, hh_neighbour, vol, n_cells, dim)

        def reduce_hessian(residual):
            """`(A_P E)ᵀ` applied to a full ``(n_cells, dim, dim)`` Hessian row, then row-scaled.

            The `1 / vol` is a positive per-cell row scaling: it changes neither the solution nor the
            block's definiteness, and it restores the volume scaling the unreduced equation carried.
            Without it the weighting squares that scaling, and every consumer that assumes a
            volume-scaled Hessian row — the packed system's inverse-volume preconditioner among them
            — degrades silently by a factor of the cell volume.
            """
            weighted = jnp.einsum("nij,njl->nil", residual, hessian_cell_block)
            return contract_symmetric(weighted, dim) / vol[:, None]

        # Full coupled system on the packed unknown [g, h] of shape (n_cells, dim + n_sym); both
        # diagonal blocks carry the cell volume, so one inverse-volume preconditioner covers both.
        # This path exists to check the elimination against the un-eliminated system, so it is
        # kept deliberately plain — the block preconditioner below is for the Schur path.
        def pack(g, u):
            return jnp.concatenate([g, u], axis=1)

        def coupled(packed):
            g, u = packed[:, :dim], packed[:, dim:]
            h = expand_symmetric(u, dim)
            rhs_g = face_cells.scatter(
                *gradient_face_terms(g, h, g, h, boundary_hessian(h), zero_f, zero_b)
            )
            rhs_h = face_cells.scatter(*hessian_face_terms(g, h, g, h, boundary_hessian(h)))
            return pack(scale(g, vol) - rhs_g, reduce_hessian(vol[:, None, None] * h - rhs_h))

        def coupled_rhs(fld, bvals):
            # The Hessian equation's right-hand side is zero (see `gradient_rhs` above).
            return pack(gradient_rhs(fld, bvals), zero_u)

        # ---- Schur elimination of the Hessian block.
        def a_gg(g):
            """``A_gg·g`` — the gradient equation's own row, with the Hessian held at zero."""
            return scale(g, vol) - face_cells.scatter(
                *gradient_face_terms(g, zero_h, g, zero_h, zero_h, zero_f, zero_b)
            )

        def gradient_and_hessian_rows(g):
            """``(A_gg·g, A_Hg·g)`` — the outer operator needs both, from one pass over the faces."""
            rhs_h = face_cells.scatter(*hessian_face_terms(g, zero_h, g, zero_h, zero_h))
            return a_gg(g), reduce_hessian(-rhs_h)

        def hessian_row_defect(g, u):
            """``A_Hg·g − A_HH·u`` in ONE evaluation of the Hessian equation rather than two.

            This is the quantity the coupled sweep drives to zero, and writing it as a difference of
            two rows spends two passes over the faces on something one pass produces — over the
            **largest** field in the scheme, the Hessian equation carrying a tensor per face where
            the gradient equation carries a vector.

            The saving is exact rather than an approximation: `hessian_face_terms` is linear in the
            gradient and the Hessian jointly, so evaluating it at ``(-g, h)`` gives
            ``H(0, h) − H(g, 0)`` — precisely the combination wanted. The boundary closure is linear
            too and the ``g`` half contributes nothing to it, so ``boundary_hessian(h)`` is the right
            argument for the merged call.
            """
            h = expand_symmetric(u, dim)
            return reduce_hessian(
                -vol[:, None, None] * h
                + face_cells.scatter(*hessian_face_terms(-g, h, -g, h, boundary_hessian(h)))
            )

        def a_hh(u):
            # The innermost loop: only the Hessian equation, so the gradient equation's face-curvature
            # work is not done and then discarded.
            h = expand_symmetric(u, dim)
            return reduce_hessian(
                vol[:, None, None] * h
                - face_cells.scatter(*hessian_face_terms(zero_g, h, zero_g, h, boundary_hessian(h)))
            )

        def a_gh(u):
            h = expand_symmetric(u, dim)
            return -face_cells.scatter(
                *gradient_face_terms(zero_g, h, zero_g, h, boundary_hessian(h), zero_f, zero_b)
            )

        def gradient_row_defect(g, u):
            """``A_gg·g − A_gH·u`` in ONE evaluation of the gradient equation rather than two.

            The companion of :func:`hessian_row_defect`, on the other row and for the same reason:
            `gradient_face_terms` is linear in the gradient and the Hessian jointly, so evaluating it
            once at ``(g, −h)`` gives ``G(g, 0) − G(0, h)`` — precisely the combination the sweep's
            gradient update needs. The boundary closure is linear too, so ``boundary_hessian(−h)`` is
            the right argument for the merged call.

            This is the more valuable of the two merges even though it saves the same one pass, because
            the gradient equation is evaluated **twice** per sweep against the Hessian equation's once,
            and it is the pass that carries the face-curvature and warp terms.
            """
            h = expand_symmetric(-u, dim)
            return scale(g, vol) - face_cells.scatter(
                *gradient_face_terms(g, h, g, h, boundary_hessian(h), zero_f, zero_b)
            )

        # `A_gg`'s per-cell diagonal block, probed from the same face kernel the operator is built
        # from, so the two cannot drift. The reduced Hessian equation's block is `(n_sym, n_sym)` and
        # no longer factors — imposing symmetry couples the two tensor indices that the Kronecker
        # structure above kept apart — but it needs no probe at all; see `inner()`.
        def gg_owner(probe):
            owner_side, _ = gradient_face_terms(
                probe, zero_h, zero_g, zero_h, zero_h, zero_f, zero_b
            )
            return face_cells.scatter(owner_side, no_face_g)

        def gg_neighbour(probe):
            _, neighbour_side = gradient_face_terms(
                zero_g, zero_h, probe, zero_h, zero_h, zero_f, zero_b
            )
            return face_cells.scatter(no_face_g, neighbour_side)

        def inner():
            """The reduced Hessian system, whose per-cell block costs NO pass over the faces.

            The unreduced Hessian equation's per-cell action is exactly ``H ↦ H·C`` for the
            ``(dim, dim)`` block ``C`` built above — that is the Kronecker structure `A_P` already
            relies on — and the reduction is the fixed linear map ``(A_P E)ᵀ``. So the reduced block's
            column for the unit component ``e_a`` is ``contract_symmetric(E(e_a)·C·C) / vol``:
            per-cell arithmetic on a small dense matrix already in hand.

            Probing it instead costs ``n_sym`` evaluations of the face kernel — twelve passes over the
            faces in three dimensions, each gathering a tensor per face — to recover a block that the
            algebra gives for nothing. It is the same quantity either way, and a unit test pins the
            two against each other; this route simply does not pay for it.
            """
            # `E(e_a)` for each unit component, cell-independent, so the einsum below contracts a
            # fixed `(dim, dim)` against the per-cell block rather than broadcasting a tensor field.
            basis = expand_symmetric(jnp.eye(n_sym), dim)
            columns = jax.vmap(
                lambda unit: reduce_hessian(jnp.einsum("ij,nlj->nil", unit, hessian_cell_block))
            )(basis)
            block = jnp.moveaxis(columns, 0, -1)
            return GradientSystem(CellBlockJacobi(jnp.linalg.inv(block)), a_hh, (n_cells, n_sym))

        def gh_owner(probe_h):
            side, _ = gradient_face_terms(
                zero_g,
                probe_h,
                zero_g,
                zero_h,
                prepared_closure.diagonal(probe_h),
                zero_f,
                zero_b,
            )
            return -face_cells.scatter(side, no_face_g)

        def gh_neighbour(probe_h):
            _, side = gradient_face_terms(zero_g, zero_h, zero_g, probe_h, zero_h, zero_f, zero_b)
            return -face_cells.scatter(no_face_g, side)

        def hg_owner(probe_g):
            side, _ = hessian_face_terms(probe_g, zero_h, zero_g, zero_h, zero_h)
            return -reduce_hessian(face_cells.scatter(side, no_face_h))

        def hg_neighbour(probe_g):
            _, side = hessian_face_terms(zero_g, zero_h, probe_g, zero_h, zero_h)
            return -reduce_hessian(face_cells.scatter(no_face_h, side))

        def local_schur_block(gradient_block, hessian_inverse):
            """``A_gg``'s block less the elimination term's own, contracted cell by cell.

            **The outer operator is the Schur complement, so its preconditioner should approximate
            the Schur complement's diagonal block -- not ``A_gg``'s.** The neglected term is
            ``A_gH A_HH⁻¹ A_Hg``, which on a well-shaped cell is a small perturbation of ``A_gg`` and
            on a flattened one is not: the cell's volume vanishes while its face couplings do not, so
            the term it was safe to drop becomes the dominant part of that cell's row.

            Each factor is replaced by its own per-cell block and the three are contracted per cell --
            the same approximation a pressure Schur usually gets. It is an approximation: the true
            block also carries paths out to a neighbour and back, which this drops.

            Costs ``dim`` probes for ``A_Hg`` and ``n_sym`` for ``A_gH``. Both are plain matrices once
            the Hessian is carried as its independent components, so the elimination term is an
            ordinary triple product rather than the rank-three contraction the full tensor needed.

            ⚠️ It diverged on a real mesh under a fixed-sweep outer solve while the Hessian was
            solved unsymmetrized; see the class docstring for what changed and what that does and
            does not establish.
            """

            def hessian_row_block(unit):  # A_Hg: g_k -> u_a
                probe = jnp.broadcast_to(unit, (n_cells, dim))
                return hg_owner(probe) + hg_neighbour(probe)

            def gradient_row_block(unit):  # A_gH: u_a -> g_m
                probe = expand_symmetric(jnp.broadcast_to(unit, (n_cells, n_sym)), dim)
                return gh_owner(probe) + gh_neighbour(probe)

            _, hg = jax.lax.scan(
                lambda carry, unit: (carry, hessian_row_block(unit)), None, jnp.eye(dim)
            )
            hg = jnp.moveaxis(hg, 0, -1)

            # ⚠️ PROBED UNDER `lax.scan`, WHICH IS THE ONLY CONSTRUCT THAT ACTUALLY SEQUENCES THEM.
            # A Python loop over the probes followed by a `stack` does NOT: to the compiler that is
            # `n_sym` INDEPENDENT computations feeding one consumer, free to be scheduled together,
            # so every probe's face intermediates can be live at once. Each probe gathers a
            # `(n_cells, dim, dim)` field to `(n_faces, dim, dim)` -- 376 MB at 1.6M cells, twice per
            # probe -- so six of them concurrently is several gigabytes, which on a real mesh is the
            # difference between fitting in memory and swapping. Measured: a reconstruction went from
            # ~200 s (swapping, and FLAT in the sweep count because the prologue dominated everything)
            # to the sweeps mattering again.
            #
            # ⚠️ It is invisible on a small mesh, which is how it shipped: at 13824 cells six live
            # probes is a few MB, and a Python loop measured 1.29 GB against a `vmap`'s 1.39 GB --
            # a real difference, for a reason that does not survive to the size that matters.
            # `scan` bounds the intermediates by construction rather than by the scheduler's choice.
            def probe_column(carry, unit):
                return carry, gradient_row_block(unit)

            _, gh = jax.lax.scan(probe_column, None, jnp.eye(n_sym))
            gh = jnp.moveaxis(gh, 0, -1)
            # `g_k -> u_a` by `hg`, then the reduced `A_HH⁻¹`, then `u_a -> g_m` by `gh`.
            return gradient_block - jnp.einsum("nma,nab,nbk->nmk", gh, hessian_inverse, hg)

        def outer_preconditioner(inner_system, use_local_schur_block=False):
            """The outer system's per-cell preconditioner, without building its operator.

            The coupled sweep needs this and not the Schur operator, whose construction takes an
            inner solve it would never run — so reaching it through `outer` would mean handing that
            call a strategy chosen only to be discarded.
            """
            if prepared_outer is not None:
                return prepared_outer
            block = cell_diagonal_block(gg_owner, gg_neighbour, vol, n_cells, dim)
            if use_local_schur_block:
                block = local_schur_block(block, inner_system.preconditioner.inverse)
            return CellBlockJacobi(jnp.linalg.inv(block))

        def outer(hessian_solver, inner_system, use_local_schur_block=False):
            preconditioner = outer_preconditioner(inner_system, use_local_schur_block)

            def schur(g):
                a_gg_g, a_hg_g = gradient_and_hessian_rows(g)
                return a_gg_g - a_gh(
                    hessian_solver.solve(inner_system.preconditioner, inner_system.operator, a_hg_g)
                )

            return GradientSystem(preconditioner, schur, (n_cells, dim))

        def block_sweep(sweep, gradient_preconditioner, hessian_preconditioner, rhs_g):
            """Run :class:`CoupledBlockSweep` over these systems and return the gradient."""

            # From `(g, h) = (0, 0)` the first sweep reduces to `w P_g^-1 b_g` -- the Hessian update
            # sees `A_Hg 0 - A_HH 0` and stays at zero, and the gradient update sees `b_g` outright.
            # Peeled for the same reason the swept solve peels its first apply: those passes multiply
            # vectors known to be zero.
            def step(carry, _):
                g, h = carry
                # TWO face-kernel evaluations, not four: each row is evaluated ONCE, at the argument
                # that yields the defect that row's update needs, rather than once per block and then
                # subtracted. Both rows are linear in `(g, H)` jointly, which is what makes the
                # merged argument exact rather than an approximation -- see `hessian_row_defect` and
                # `gradient_row_defect`.
                h_next = h + hessian_preconditioner.apply(hessian_row_defect(g, h))
                # Gauss-Seidel, not Jacobi: the gradient update uses the Hessian just computed. That
                # is what lets a single Hessian sweep per gradient sweep converge at all.
                g_next = g + sweep.relaxation * gradient_preconditioner.apply(
                    rhs_g - gradient_row_defect(g, h_next)
                )
                return (g_next, h_next), None

            start = (
                sweep.relaxation * gradient_preconditioner.apply(rhs_g),
                zero_u,
            )
            if sweep.sweeps <= 1:
                return start[0]
            (g, _), _ = jax.lax.scan(step, start, None, length=sweep.sweeps - 1)
            return g

        def coupled_error(relaxation, gradient_preconditioner, hessian_preconditioner):
            """The block sweep's error operator, as a system :func:`contraction_rate` can measure.

            Propagating the error is the sweep with the right-hand side set to zero, so this is the
            same two updates with ``b_g`` dropped. It is presented as a `GradientSystem` with an
            IDENTITY preconditioner and the operator ``v - M v``, because the estimator forms
            ``v - P⁻¹(A v)`` -- which is then exactly ``M v``. That reuse is the point: the Gelfand
            averaging, the settledness check and the traced-geometry guard are subtle enough that a
            second copy of them for this operator would be a liability.

            The error is carried on the packed ``[g, h]`` vector because the two blocks converge
            together and the rate is a property of the pair; the gradient's own error is bounded by
            it asymptotically, which is what the sweep count is derived from.
            """

            def error(packed):
                g = packed[:, :dim]
                h = packed[:, dim:]
                a_gg_g, a_hg_g = gradient_and_hessian_rows(g)
                h_next = h + hessian_preconditioner.apply(a_hg_g - a_hh(h))
                g_next = g + relaxation * gradient_preconditioner.apply(-a_gg_g + a_gh(h_next))
                return jnp.concatenate([g_next, h_next], axis=1)

            return GradientSystem(
                InverseVolume(jnp.ones(n_cells)),  # identity: `v - I(v - Mv)` is `Mv`
                lambda v: v - error(v),
                (n_cells, dim + n_sym),
            )

        return _HessianSystems(
            dim=dim,
            gradient_rhs=gradient_rhs,
            coupled_rhs=coupled_rhs,
            coupled=GradientSystem(InverseVolume(1.0 / vol), coupled, (n_cells, dim + n_sym)),
            inner=inner,
            outer=outer,
            outer_preconditioner=outer_preconditioner,
            hessian_row_defect=hessian_row_defect,
            gradient_row_defect=gradient_row_defect,
            block_sweep=block_sweep,
            coupled_error=coupled_error,
        )
