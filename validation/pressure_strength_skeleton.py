"""A static strength skeleton for the pressure column, and a probe plan that colours p for it.

The coupled step cases' Jacobian reaches pressure at cell-graph distance 3 only through the Rhie--Chow
interior mass flux, ``m_f ~ -rho d_f |S_f| [(p_N - p_P) - d . ((1 - g) grad p_P + g grad p_N)] / (d . n)``:
the multiple-correction gradient's second pass makes ``grad p`` at a face neighbour reach two rings, so
the face flux reaches three from the row cell. On a graded mesh some of those distance-3 couplings are
as strong as a face coupling (across a thin cell the cell three faces away can couple more strongly than
the face neighbour along its long side), and dropping them is what stagnates bfs3d's solve under a
reach-2 pressure probe. This builds, from geometry alone, which distance-3 pairs those are, and a probe
plan that keeps them:

  strength  ``S_ij = sum over the faces f of i of |d m_f / d p_j|``, with the mass-flux derivative formed
            from the bound gradient scheme's own operator (probed on the geometry, zero boundary values),
            the face conductance ``|S_f| / (d . n)`` and a per-cell ``V / sum_f conductance`` standing in
            for ``V / a_P``. Static: nothing here reads a state.
  pattern   ``R2 + S(tau)``: every pair within graph distance 2, plus each distance-3 pair whose strength
            is at least ``tau`` times row ``i``'s weakest face coupling in the proxy continuity block
            ``C = B F`` (``B`` the face-to-cell sum, ``F`` the weighted mass-flux derivative).
  colouring a saturation-degree colouring of the pattern's conflict graph (two columns conflict when a
            row couples to both), so the pressure column can be probed with it collision-free for the
            pattern.
  plan      a :class:`~aquaflux.precondition.sparse_jacobian.ColumnProbePlan` whose stored pattern is
            the uniform reach-3 one, every column but pressure probed at reach 3, and pressure probed
            with the colouring above, reading only the entries of ``R2 + S(tau)`` (the rest of its
            reach-3 ring written as explicit zeros, as any short-probed column's is).

The proxy covers the continuity rows, simplifies boundary closures (every boundary face is handed a zero
value, whatever the pressure's condition there) and replaces ``V / a_P`` by a geometric stand-in;
``reach_option1_audit.py`` measures the real Jacobian against the pattern this selects.

Run directly to print the colour counts and the kept distance-3 entries per row for one or both meshes::

    python validation/pressure_strength_skeleton.py [bfs3d|pitzDaily] [tau ...]
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import aquaflux  # noqa: F401,E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.precondition import JacobianProbe  # noqa: E402
from aquaflux.precondition.sparse_jacobian import (  # noqa: E402
    ColumnProbePlan,
    _saturation_colouring,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
)
from aquaflux.schemes import MultipleCorrectionGradient  # noqa: E402
from aquaflux.schemes.interpolation import interpolation_factor  # noqa: E402
from aquaflux.turbulence.coupled import _CoupledNarrowing  # noqa: E402
from mesh_anisotropy import MESHES, spacing_ratio  # noqa: E402

#: Seeds per batched gradient evaluation when the gradient operator is probed.
GRADIENT_BATCH = 8


def _binary(matrix):
    matrix = matrix.tocsr().copy()
    matrix.data[:] = 1.0
    return matrix


@dataclasses.dataclass(frozen=True)
class CellGraph:
    """Cell-graph distance masks on the interior-face graph, each an ``(n, n)`` 0/1 CSR matrix.

    Attributes
    ----------
    faces : scipy.sparse.csr_matrix
        Face neighbours (distance exactly 1).
    within : tuple of scipy.sparse.csr_matrix
        ``within[r]`` holds every pair within distance ``r`` (``within[0]`` the identity), up to 3.
    """

    faces: sp.csr_matrix
    within: tuple

    @classmethod
    def of(cls, owner, nb, n):
        owner, nb = np.asarray(owner), np.asarray(nb)
        eye = sp.identity(n, format="csr")
        faces = _binary(
            sp.csr_matrix(
                (np.ones(2 * owner.size), (np.r_[owner, nb], np.r_[nb, owner])), shape=(n, n)
            )
        )
        step = _binary(faces + eye)
        within = [eye, step]
        for _ in range(2):
            within.append(_binary(within[-1] @ step))
        return cls(faces, tuple(within))

    def at_distance(self, reach):
        """Pairs at distance exactly ``reach`` (1, 2 or 3), as a 0/1 CSR matrix."""
        exact = (self.within[reach] - self.within[reach - 1]).tocsr()
        exact.eliminate_zeros()
        return _binary(exact)


def row_max(matrix, n):
    """Per row, the largest stored value (zero for an empty row), shape ``(n,)``."""
    matrix = matrix.tocsr()
    out = np.zeros(n)
    filled = np.diff(matrix.indptr) > 0
    if matrix.nnz:
        out[filled] = np.maximum.reduceat(matrix.data, matrix.indptr[:-1][filled])
    return out


def row_min(matrix, n):
    """Per row, the smallest stored value (``inf`` for an empty row), shape ``(n,)``."""
    matrix = matrix.tocsr()
    out = np.full(n, np.inf)
    filled = np.diff(matrix.indptr) > 0
    if matrix.nnz:
        out[filled] = np.minimum.reduceat(matrix.data, matrix.indptr[:-1][filled])
    return out


def restricted(matrix, mask):
    """``|matrix|`` on the positions of the 0/1 ``mask`` only, explicit zeros removed."""
    out = abs(matrix).multiply(mask).tocsr()
    out.eliminate_zeros()
    return out


def gradient_operator(mesh, geom, colouring):
    """The bound multiple-correction gradient as ``dim`` sparse ``(n, n)`` matrices, probed by colour.

    Geometry-only: every boundary face is handed a zero value. The recovery is checked against a direct
    evaluation on a random field and refused if it is not exact to round-off.
    """
    n, dim = mesh.n_cells, mesh.dim
    scheme = MultipleCorrectionGradient().bind(mesh, geom)
    zeros = jnp.zeros(mesh.face_cells.owner.shape[0])
    gradient = jax.jit(jax.vmap(lambda f: scheme.gradients(f, mesh, geom, zeros)))
    colour, rows, cols = colouring.colour, colouring.pattern_rows, colouring.pattern_cols
    parts = []
    for start in range(0, colouring.n_colours, GRADIENT_BATCH):
        colours = np.arange(start, min(start + GRADIENT_BATCH, colouring.n_colours))
        seeds = (colour[None, :] == colours[:, None]).astype(np.float64)
        responses = np.asarray(gradient(jnp.asarray(seeds)))
        for k, c in enumerate(colours):
            hit = colour[cols] == c
            parts.append((rows[hit], cols[hit], responses[k][rows[hit]]))
    r, c, v = (np.concatenate(p) for p in zip(*parts, strict=True))
    operator = [sp.csr_matrix((v[:, a], (r, c)), shape=(n, n)) for a in range(dim)]
    field = np.random.default_rng(0).standard_normal(n)
    direct = np.asarray(gradient(jnp.asarray(field[None])))[0]
    error = max(
        np.abs(operator[a] @ field - direct[:, a]).max() / np.abs(direct[:, a]).max()
        for a in range(dim)
    )
    if error > 1e-10:
        raise RuntimeError(
            f"the probed gradient operator misses its direct evaluation by {error:.2e}"
        )
    return operator


@dataclasses.dataclass(frozen=True)
class PressureSkeleton:
    """The static pressure-coupling skeleton of one mesh.

    Attributes
    ----------
    graph : CellGraph
        The cell graph's distance masks.
    strength : scipy.sparse.csr_matrix
        ``S_ij``, shape ``(n, n)``: per row cell, the summed ``|d m_f / d p_j|`` over its interior faces.
    continuity : scipy.sparse.csr_matrix
        The proxy continuity block ``C = B F``, shape ``(n, n)``.
    weakest_face : np.ndarray
        Per row, the smallest face-neighbour ``|C_ij|``, shape ``(n,)``.
    spacing : np.ndarray
        Per cell, the spacing ratio ``max / min |d . n|`` over its faces, shape ``(n,)``.
    owner, nb : np.ndarray
        The interior-face cell-graph edges, shape ``(n_edges,)`` each.
    """

    graph: CellGraph
    strength: sp.csr_matrix
    continuity: sp.csr_matrix
    weakest_face: np.ndarray
    spacing: np.ndarray
    owner: np.ndarray
    nb: np.ndarray

    @property
    def n(self):
        return self.weakest_face.size

    @classmethod
    def of(cls, mesh, geom=None):
        geom = mesh.geometry() if geom is None else geom
        fc = mesh.face_cells
        owner_all, interior = np.asarray(fc.owner), np.asarray(fc.interior)
        owner, nb, _ = fc.interior_edges()
        owner, nb = np.asarray(owner), np.asarray(nb)
        n, dim = mesh.n_cells, mesh.dim
        faces = np.flatnonzero(interior)
        if not (
            np.array_equal(owner_all[faces], owner)
            and np.array_equal(np.asarray(fc.neighbour)[faces], nb)
        ):
            raise RuntimeError("interior_edges() is not the interior faces in face order")
        gradient = gradient_operator(mesh, geom, block_stencil_colouring(owner, nb, n, 3))
        centroid = np.asarray(geom.cell.centroid)
        volume = np.asarray(geom.cell.volume)
        area = np.asarray(geom.face.area)
        area = np.linalg.norm(area, axis=-1) if area.ndim == 2 else area
        normal = np.asarray(geom.face.normal)
        g = np.asarray(interpolation_factor(fc, geom))[faces]
        d = centroid[nb] - centroid[owner]
        dn = np.einsum("fi,fi->f", d, normal[faces])
        conductance = area[faces] / np.abs(dn)
        boundary = ~interior
        to_face = np.abs(
            np.einsum(
                "fi,fi->f",
                np.asarray(geom.face.centroid)[boundary] - centroid[owner_all[boundary]],
                normal[boundary],
            )
        )
        total = np.bincount(owner, conductance, n) + np.bincount(nb, conductance, n)
        total += np.bincount(owner_all[boundary], area[boundary] / np.maximum(to_face, 1e-300), n)
        d_cell = volume / total  # the geometric stand-in for V / a_P
        nf = faces.size
        at_owner = sp.csr_matrix((np.ones(nf), (np.arange(nf), owner)), shape=(nf, n))
        at_neighbour = sp.csr_matrix((np.ones(nf), (np.arange(nf), nb)), shape=(nf, n))
        flux = at_neighbour - at_owner
        for a in range(dim):
            flux = flux - sp.diags(d[:, a] * (1 - g)) @ at_owner @ gradient[a]
            flux = flux - sp.diags(d[:, a] * g) @ at_neighbour @ gradient[a]
        weight = ((1 - g) * d_cell[owner] + g * d_cell[nb]) * area[faces] / dn
        weighted = (sp.diags(weight) @ flux).tocsr()
        to_cells = sp.csr_matrix(
            (
                np.r_[np.ones(nf), -np.ones(nf)],
                (np.r_[owner, nb], np.r_[np.arange(nf), np.arange(nf)]),
            ),
            shape=(n, nf),
        )
        continuity = (to_cells @ weighted).tocsr()
        strength = (abs(to_cells) @ abs(weighted)).tocsr()
        graph = CellGraph.of(owner, nb, n)
        weakest = row_min(restricted(continuity, graph.faces), n)
        return cls(graph, strength, continuity, weakest, spacing_ratio(mesh, geom), owner, nb)

    def strong_distance_three(self, tau):
        """The distance-3 pairs ``S(tau)``: strength at least ``tau`` times the row's weakest face coupling."""
        coo = restricted(self.strength, self.graph.at_distance(3)).tocoo()
        keep = coo.data >= tau * self.weakest_face[coo.row]
        return sp.csr_matrix(
            (np.ones(int(keep.sum())), (coo.row[keep], coo.col[keep])), shape=(self.n, self.n)
        )

    def pattern(self, tau):
        """The 0/1 pattern ``R2 + S(tau)`` (``tau=None``: reach 2 alone)."""
        within2 = self.graph.within[2]
        return within2 if tau is None else _binary(within2 + self.strong_distance_three(tau))


def pattern_colouring(pattern):
    """A collision-free probe colouring for ``pattern`` (rows couple to columns), as ``(colour, n_colours)``.

    Two columns may share a colour only if no row couples to both, so the conflict graph is
    ``pattern^T pattern``; it is coloured by saturation degree, as the library colours a stencil. The
    result is checked: no row of the pattern holds two columns of one colour.
    """
    pattern = _binary(pattern)
    n = pattern.shape[0]
    conflict = _binary(pattern.T @ pattern)
    colour = _saturation_colouring(conflict, n)
    n_colours = int(colour.max()) + 1
    coo = pattern.tocoo()
    key = coo.row.astype(np.int64) * n_colours + colour[coo.col]
    if np.unique(key).size != key.size:
        raise RuntimeError("the colouring puts two columns of one row in one colour")
    return colour, n_colours


def pattern_probe_plan(
    owner, nb, n, n_fields, field, pattern, colour, n_colours, active_rows=None, others_reach=3
):
    """A reach-3 probe plan in which column ``field`` is probed with ``colour``, reading ``pattern`` only.

    The stored pattern is the uniform reach-3 plan's; every other column is probed as the library's
    per-column plan probes it at ``others_reach``. Column ``field`` reads the reach-3 entries that lie
    in ``pattern`` (which must lie inside reach 3) and writes the rest as explicit zeros, as a
    short-probed column's out-of-reach entries are.

    Parameters
    ----------
    owner, nb : np.ndarray
        Interior-face cell-graph edges, shape ``(n_edges,)`` each.
    n, n_fields : int
        Cells and fields per cell.
    field : int
        The column field probed with the pattern's colouring.
    pattern : scipy.sparse.csr_matrix
        The 0/1 cell pattern that column reads, shape ``(n, n)``.
    colour : np.ndarray
        Its colouring (:func:`pattern_colouring`), shape ``(n,)``.
    n_colours : int
        Colours that colouring uses.
    active_rows : np.ndarray, optional
        Field-pair blocks to store, shape ``(n_fields, n_fields)``, as in ``ColumnProbePlan``.
    others_reach : int
        The reach every other column is probed at, ``2`` or ``3``.

    Returns
    -------
    ColumnProbePlan
        The plan.
    """
    if others_reach not in (2, 3):
        raise ValueError(f"others_reach={others_reach}: the other columns are probed at 2 or 3")
    others = column_probe_plan(owner, nb, n, [others_reach] * n_fields, pattern_reach=3)
    wide = block_stencil_colouring(owner, nb, n, 3)
    flat_wide = wide.pattern_rows.astype(np.int64) * n + wide.pattern_cols
    coo = _binary(pattern).tocoo()
    flat_pattern = np.sort(coo.row.astype(np.int64) * n + coo.col)
    if not np.isin(flat_pattern, flat_wide).all():
        raise ValueError("the pattern reaches beyond the reach-3 pattern it is stored in")
    where = np.searchsorted(flat_pattern, flat_wide)
    inside = (where < flat_pattern.size) & (
        flat_pattern[np.minimum(where, flat_pattern.size - 1)] == flat_wide
    )
    # The library plan stores the same reach-3 pattern, in the same order, so its rows line up.
    assert np.array_equal(others.pattern_rows, wide.pattern_rows)
    assert np.array_equal(others.pattern_cols, wide.pattern_cols)
    in_reach = np.array(others.in_reach, copy=True)
    in_reach[field] = inside
    colours = np.array(others.colour, copy=True)
    colours[field] = colour
    counts = list(others.n_colours)
    counts[field] = n_colours
    reach = list(others.reach)
    reach[field] = 2  # the plan's label only; the entries read are ``in_reach``'s
    return ColumnProbePlan(
        n,
        n_fields,
        wide.pattern_rows,
        wide.pattern_cols,
        colours,
        tuple(counts),
        tuple(reach),
        in_reach,
        active_rows=active_rows,
    )


_SKELETONS: dict = {}


def pattern_jacobian_probe(
    coupled, groups, field, tau, gradient_sweeps, production_viscosity, others_reach=3
):
    """A coupled-RANS probe whose column ``field`` is coloured for ``R2 + S(tau)``.

    The plan is :func:`pattern_probe_plan` over the case mesh's skeleton (built once per mesh in this
    process), the other columns at ``others_reach``; it is wrapped in a :class:`JacobianProbe` laid out
    for ``groups`` (a field split's, or ``None``), with the coupled residual's own narrowing, so it can
    stand where a session would have built its probe from the case's settings.
    """
    mesh = coupled.momentum.mesh
    skeleton = _SKELETONS.get(id(mesh))
    if skeleton is None:
        skeleton = _SKELETONS[id(mesh)] = PressureSkeleton.of(mesh)
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    pattern = skeleton.pattern(tau)
    colour, count = pattern_colouring(pattern)
    plan = pattern_probe_plan(
        skeleton.owner,
        skeleton.nb,
        n,
        n_fields,
        field,
        pattern,
        colour,
        count,
        active_rows=None if groups is None else groups.active_rows(),
        others_reach=others_reach,
    )
    print(
        f"[probe] column {field} coloured for R2 + S({tau:g}), the rest at {others_reach}: {count} "
        f"colours, {plan.n_probes} probes per fit (kept distance-3 entries per row "
        f"{skeleton.strong_distance_three(tau).nnz / n:.2f})",
        flush=True,
    )
    leading = None if groups is None else groups.n_leading_fields
    return JacobianProbe(
        plan,
        block_stencil_gather_map(plan, leading_fields=leading),
        _CoupledNarrowing(gradient_sweeps, production_viscosity),
    )


def main():
    from aquaflux.io import read_openfoam

    names = [a for a in sys.argv[1:] if a in MESHES] or list(MESHES)
    taus = [float(a) for a in sys.argv[1:] if a not in MESHES] or [0.03, 0.01]
    for name in names:
        mesh = read_openfoam(MESHES[name])
        skeleton = PressureSkeleton.of(mesh)
        n = skeleton.n
        print(f"[{name}] {n} cells", flush=True)
        for label, tau in [("reach 2", None)] + [(f"S({t:g})", t) for t in taus]:
            kept = 0 if tau is None else skeleton.strong_distance_three(tau).nnz / n
            print(
                f"  {label:9s} colours {pattern_colouring(skeleton.pattern(tau))[1]:3d}, kept d3 per row {kept:.2f}",
                flush=True,
            )
        print(
            f"  reach 3   colours {block_stencil_colouring(skeleton.owner, skeleton.nb, n, 3).n_colours:3d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
