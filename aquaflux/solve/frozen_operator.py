"""Assemble a frozen transport operator as a sparse matrix, for the algebraic-multigrid setup.

The preconditioners freeze an approximate linearization of a transport equation — a symmetric
diffusive coupling on the interior faces, optionally plus first-order-upwind convection at a
reference flux — and coarsen it once, off the jit path. It sits beside the coarsening it feeds, and
deliberately outside :mod:`aquaflux.solve.multigrid`, which stays a pure operator-coarsening library.
The first-order-upwind stencil is the *preconditioner's* choice, not the model's: whatever scheme the
residual discretizes advection with, the frozen operator always upwinds first-order, because that is
what makes it a diagonally dominant M-matrix an aggregation hierarchy can coarsen.

Every consumer of a frozen operator builds it through this one assembler: the pressure Schur and the
viscous velocity block (symmetric, ``flux=None``), the convection-aware velocity block, and the k/omega
scalar-transport preconditioner. The multigrid builders in :mod:`aquaflux.solve.multigrid` then take
the assembled matrix, so they stay a pure operator-coarsening library.

This module imports only ``numpy`` and ``scipy.sparse`` — it holds no mesh, no field, and no
``jax`` — so it stays testable on a bare graph and adds no dependency to any subsystem that already
builds a hierarchy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def require_valid_graph(n: int, owner: np.ndarray, nb: np.ndarray, where: str) -> None:
    """Reject a malformed interior-face graph before it is assembled into an operator.

    The frozen operators are assembled and coarsened once, off the jit path, and then held fixed; a
    bad graph would otherwise bake ``inf``/``NaN`` into the frozen preconditioner (via a later
    zero-diagonal inversion) and only show up as a silently stalling runtime V-cycle. Checks the
    invariants that must hold for *any* mesh: at least one cell, matched edge arrays, and every edge
    index in range.

    Parameters
    ----------
    n : int
        Number of cells.
    owner, nb : np.ndarray
        Interior-face edge endpoints, shape ``(n_edges,)`` each.
    where : str
        Caller name, included in the error message.

    Raises
    ------
    ValueError
        If ``n < 1``, ``owner`` and ``nb`` differ in length, or any endpoint is outside ``[0, n)``.
    """
    if n < 1:
        raise ValueError(f"{where}: need at least one cell, got n={n}.")
    owner, nb = np.asarray(owner), np.asarray(nb)
    if owner.shape != nb.shape:
        raise ValueError(
            f"{where}: owner and nb must have the same shape, got {owner.shape} and {nb.shape}."
        )
    if owner.size and (owner.min() < 0 or owner.max() >= n or nb.min() < 0 or nb.max() >= n):
        raise ValueError(f"{where}: edge endpoints out of range for n={n} cells.")


def convection_diffusion_operator(
    owner: np.ndarray,
    nb: np.ndarray,
    coefficient: np.ndarray,
    n: int,
    *,
    flux: np.ndarray | None = None,
    boundary_diagonal: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Frozen convection-diffusion operator ``A`` on an interior-face graph, as a scipy CSR matrix.

    Each interior edge ``(owner P, neighbour N)`` carries a symmetric diffusive coupling
    ``coefficient`` (e.g. ``Gamma_face A / (d.n)``). With a ``flux`` it also carries a
    first-order-upwind convective coupling from the owner-outward face flux: an outflow
    (``flux > 0``) advects the owner value, an inflow the neighbour value. The entries are

        A[P, N] = -(coefficient + max(-flux, 0)),   A[N, P] = -(coefficient + max(flux, 0)),

    with the matching diagonal contributions ``coefficient + max(flux, 0)`` at P and
    ``coefficient + max(-flux, 0)`` at N — so ``A`` is a diagonally dominant M-matrix, nonsymmetric
    exactly where the flux is non-zero. Without a ``flux`` the convective terms vanish and ``A`` is
    the symmetric graph Laplacian of the edge coefficients (the pressure-Schur and viscous-momentum
    case). ``boundary_diagonal`` adds the per-cell boundary-face contributions the interior edges do
    not carry (Dirichlet wall/inlet stiffness, outflow convection, a reaction linearization).

    Parameters
    ----------
    owner, nb : np.ndarray
        Interior-face edge endpoints, shape ``(n_edges,)`` each.
    coefficient : np.ndarray
        Per-edge symmetric diffusive coefficient, shape ``(n_edges,)``.
    n : int
        Number of cells.
    flux : np.ndarray, optional
        Per-edge owner-outward convective face flux (the frozen convective linearization), shape
        ``(n_edges,)``. Omit for a symmetric (pure diffusion) operator.
    boundary_diagonal : np.ndarray, optional
        Per-cell boundary-face diagonal contribution, shape ``(n_cells,)``.

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled operator, shape ``(n, n)``.
    """
    require_valid_graph(n, owner, nb, "convection_diffusion_operator")
    o, m = np.asarray(owner), np.asarray(nb)
    c = np.asarray(coefficient)
    zero = np.zeros_like(c)
    f = zero if flux is None else np.asarray(flux)
    up_out = np.maximum(f, 0.0)  # outflow leaves the owner: owner value is upwind
    up_in = np.maximum(-f, 0.0)  # inflow enters the owner: neighbour value is upwind
    rows = np.concatenate([o, m, o, m])
    cols = np.concatenate([m, o, o, m])
    vals = np.concatenate([-(c + up_in), -(c + up_out), c + up_out, c + up_in])
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if boundary_diagonal is not None:
        a = a + sp.diags(np.asarray(boundary_diagonal))
    return a.tocsr()


def decouple_dof(a: sp.csr_matrix, index: int) -> sp.csr_matrix:
    """Decouple one degree of freedom from ``a``: zero its row and column, unit diagonal.

    The regularization for a closed-domain pressure system, whose operator is otherwise singular (a
    pure-Neumann Laplacian defines pressure only up to a constant). Decoupling the pinned cell — as
    opposed to handling the pin after the fact — leaves the operator nonsingular and makes the pinned
    cell a singleton in any subsequent aggregation, so the coarse space's null space matches the
    pinned outer Jacobian and no post-hoc pin handling is needed in the V-cycle.

    Parameters
    ----------
    a : scipy.sparse matrix
        The assembled operator, shape ``(n, n)``.
    index : int
        The pinned cell index.

    Returns
    -------
    scipy.sparse.csr_matrix
        The operator with row/column ``index`` decoupled.
    """
    a = a.tolil()
    a[index, :] = 0
    a[:, index] = 0
    a[index, index] = 1.0
    return a.tocsr()
