"""Is assembling the coupled Jacobian from local derivatives worth building? A pre-implementation probe.

The materialized-Jacobian preconditioner recovers the coupled Jacobian by coloured probing: one batched
``jax.jvp`` of the whole residual per (colour, column field). The proposal this case prices is to build
the same matrix from *local* derivatives instead -- differentiate each stage of the residual with
respect to its own immediate inputs and compose the stages by sparse products. It is not built here; this
harness measures the quantities that decide whether building it could pay.

The residual is ``R(x) = F(x, g(x))`` with ``g`` the reconstructed cell gradients. On a mesh-fixed
gradient scheme ``g`` is a linear map ``G`` of the state, so the chain rule gives

    J = F_x + F_g G

where ``G`` depends on the mesh alone and is built once per case, never per refresh. The staged route
therefore pays, per materialization, for ``F_x`` and ``F_g`` (each a probe of ``F`` with the gradient
held fixed, at a shorter reach than ``R``'s) plus one sparse product. What it can save is (i) the
gradient reconstruction inside every probe and (ii) any colours a shorter reach removes; what it adds is
the extra columns ``F_g`` carries (``dim`` per reconstructed field) and the product.

Arms, all at one state (the time-accurate OpenFOAM field, mapped cell for cell onto this mesh):

``shipped``
    The case as it runs: ``R``'s probe at the case's reach, timed per materialize and checked exact.
``frozen-gradient``
    The same residual with every reconstructed gradient wrapped in ``stop_gradient``: its primal is
    identical, and its Jacobian is exactly ``F_x``. Its own reach is measured, and its materialize is
    timed at that reach, so the ``F_x`` half of the staged route is a measurement rather than an estimate.
``F_g``, ``G`` and the product
    Not materialized (that needs ``F`` split at the gradient, i.e. the implementation). ``F_g``'s probe
    count is taken from the measured colour counts and its per-probe cost from ``F_x``'s; the product is
    timed in SciPy on matrices with exactly the block patterns it would have, filled with random values.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/local_jacobian_probe.py

``PITZ_LOCAL_JAC_REPEATS`` sets the timing repeats (default 3; the minimum is reported).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402  (the validated benchmark: mesh, physics, reference fields)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.schemes import DEFAULT_GRADIENT_SCHEME, GradientScheme  # noqa: E402
from aquaflux.solve import (JacobianProbe, materialize_block_jacobian,)
from aquaflux.solve.materialized_session import (PROBE_BATCH_SIZE, batched_jacobian_matvec, jacobian_matvec,)
from aquaflux.solve.sparse_jacobian import (block_stencil_colouring, jacobian_relative_error,)

REPEATS = int(os.environ.get("PITZ_LOCAL_JAC_REPEATS", "3"))

#: The reach the shipped case probes at (``case.yaml``: the multiple-correction residual reaches 3).
SHIPPED_REACH = 3

#: Reaches whose colourings are reported; the staged route's ``F_g`` is costed at reach 1.
REACHES = (1, 2, 3)


class FrozenGradient(GradientScheme):
    """A gradient scheme whose reconstruction carries no derivative: ``stop_gradient`` of another's.

    The residual built on it has the shipped residual's value at every state, and its Jacobian is the
    shipped Jacobian with the path through the reconstructed gradients removed -- ``F_x`` in the chain
    rule ``J = F_x + F_g G``.

    Attributes
    ----------
    inner : GradientScheme
        The reconstruction whose value is used.
    """

    inner: GradientScheme
    #: Python-side count of reconstructions traced, so the number of gradient fields is read off the
    #: residual rather than assumed. A class-level list, deliberately outside the pytree.
    traced: list = eqx.field(static=True, default_factory=list)

    def bind(self, mesh, geometry, boundary_linearization=None):
        return FrozenGradient(self.inner.bind(mesh, geometry, boundary_linearization), self.traced)

    def _reconstruct_gradient(self, field, mesh, geometry, boundary_values, **kwargs):
        self.traced.append(field.shape)
        return jax.lax.stop_gradient(
            self.inner.gradients(field, mesh, geometry, boundary_values, **kwargs)
        )


def _timed(fn):
    """Minimum wall time of ``fn`` over ``REPEATS`` calls, after one warm-up call (which compiles)."""
    fn()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples)


def _block(x):
    return jax.block_until_ready(x)


def reference_state(coupled):
    """The time-accurate OpenFOAM field, packed into the coupled state in the solved variables."""
    of = compare.read_openfoam_reference()
    flow = coupled.momentum.layout.pack(jnp.asarray(of["U"]), jnp.asarray(of["p"]))
    return coupled.state_from_physical(flow, jnp.asarray(of["k"]), jnp.asarray(of["omega"]))


def adjacency(owner, nb, n):
    ones = np.ones(2 * owner.size + n)
    a = sp.coo_matrix(
        (
            ones,
            (np.concatenate([owner, nb, np.arange(n)]), np.concatenate([nb, owner, np.arange(n)])),
        ),
        shape=(n, n),
    ).tocsr()
    a.data[:] = 1.0
    return a


def pattern_at(a, reach):
    p = a
    for _ in range(reach - 1):
        p = p @ a
        p.data[:] = 1.0
    return p.tocsr()


def random_block_matrix(cell_pattern, rows_per_cell, cols_per_cell, rng):
    """A CSR matrix whose (row-cell, column-cell) blocks are dense ``rows x cols`` on ``cell_pattern``."""
    coo = cell_pattern.tocoo()
    n = cell_pattern.shape[0]
    r = (np.arange(rows_per_cell)[:, None, None] * n + coo.row[None, None, :]).repeat(
        cols_per_cell, axis=1
    )
    c = (np.arange(cols_per_cell)[None, :, None] * n + coo.col[None, None, :]).repeat(
        rows_per_cell, axis=0
    )
    return sp.csr_matrix(
        (rng.standard_normal(r.size), (r.ravel(), c.ravel())),
        shape=(rows_per_cell * n, cols_per_cell * n),
    )


def materialize_timed(assembler, state, plan_reach, n_fields, face_cells, n):
    """Materialize ``assembler``'s Jacobian at ``plan_reach``; return (matrix, seconds, probes)."""
    probe = JacobianProbe.build(face_cells, n, n_fields, plan_reach)
    matvec = eqx.filter_jit(lambda v: jacobian_matvec(assembler, state, v))
    batched = eqx.filter_jit(lambda vs: batched_jacobian_matvec(assembler, state, vs))

    def run():
        return materialize_block_jacobian(
            matvec,
            probe.plan,
            batched_matvec=batched,
            probe_batch_size=PROBE_BATCH_SIZE,
            structure=probe.structure,
        )

    seconds = _timed(run)
    return run(), seconds, probe.plan.n_probes, matvec


def main():
    print(f"[configuration] repeats {REPEATS}, probe batch {PROBE_BATCH_SIZE}", flush=True)
    print(f"  jax {jax.__version__}, backend {jax.default_backend()}, devices {jax.devices()}")
    print(f"  cpu count {os.cpu_count()}", flush=True)

    shipped = compare.build_case()["coupled"]
    tracer = FrozenGradient(DEFAULT_GRADIENT_SCHEME)
    frozen = compare.build_case(gradient_scheme=tracer)["coupled"]
    scheme = shipped.momentum.gradient_scheme
    print(f"  shipped gradient scheme: {type(scheme).__name__}", flush=True)

    mesh = shipped.momentum.mesh
    n = shipped.layout.n_cells
    nf = shipped.layout.n_fields
    dim = mesh.dim
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    print(f"  cells {n}, fields {nf}, dim {dim}, interior faces {owner.size}", flush=True)

    state = reference_state(shipped)
    r_shipped = np.asarray(shipped.residual(state))
    r_frozen = np.asarray(frozen.residual(state))
    print(
        f"  |R| at the OpenFOAM state {np.linalg.norm(r_shipped):.4e}; frozen-gradient residual "
        f"differs by {np.linalg.norm(r_frozen - r_shipped):.1e} (must be ~0: same primal)",
        flush=True,
    )
    tracer.traced.clear()
    jax.make_jaxpr(frozen.residual)(state)
    n_grad_fields = len(tracer.traced)
    print(f"  gradient reconstructions per residual evaluation: {n_grad_fields}", flush=True)

    # -- colourings -----------------------------------------------------------------------------------
    print(
        "\n[colourings: reach -> colours, cells per row (mean / max), cell-block nnz]", flush=True
    )
    a = adjacency(owner, nb, n)
    colours = {}
    for reach in REACHES:
        c = block_stencil_colouring(owner, nb, n, reach)
        row = np.diff(pattern_at(a, reach).indptr)
        colours[reach] = c.n_colours
        print(
            f"  reach {reach}: {c.n_colours:3d} colours, {row.mean():5.1f} / {row.max():3d} cells per "
            f"row, {row.sum():9d} blocks",
            flush=True,
        )

    # -- residual and jvp costs -----------------------------------------------------------------------
    print("\n[single evaluations, jit-compiled, warm, min of repeats]", flush=True)
    tangent = jnp.asarray(np.random.default_rng(0).standard_normal(state.shape))
    for label, case in (("shipped R", shipped), ("frozen-gradient F", frozen)):
        res = eqx.filter_jit(lambda s, case=case: case.residual(s))
        jvp = eqx.filter_jit(lambda s, t, case=case: jacobian_matvec(case, s, t))
        t_r = _timed(lambda res=res: _block(res(state)))
        t_j = _timed(lambda jvp=jvp: _block(jvp(state, tangent)))
        print(f"  {label:<20s} residual {t_r * 1e3:7.2f} ms   jvp {t_j * 1e3:7.2f} ms", flush=True)

    # -- shipped materialize --------------------------------------------------------------------------
    print("\n[materialize, batched coloured probe]", flush=True)
    j_shipped, t_shipped, p_shipped, mv_shipped = materialize_timed(
        shipped, state, SHIPPED_REACH, nf, mesh.face_cells, n
    )
    err = jacobian_relative_error(j_shipped, mv_shipped)
    print(
        f"  shipped R at reach {SHIPPED_REACH}: {p_shipped} probes, {t_shipped:.3f} s "
        f"({t_shipped / p_shipped * 1e3:.2f} ms/probe), nnz {j_shipped.nnz}, error vs jvp {err:.1e}",
        flush=True,
    )

    # -- F_x: reach, then materialize at it ----------------------------------------------------------
    fx_reach = None
    for reach in REACHES:
        j_fx, t_fx, p_fx, mv_fx = materialize_timed(frozen, state, reach, nf, mesh.face_cells, n)
        err = jacobian_relative_error(j_fx, mv_fx)
        print(
            f"  frozen F_x at reach {reach}: {p_fx} probes, {t_fx:.3f} s "
            f"({t_fx / p_fx * 1e3:.2f} ms/probe), nnz {j_fx.nnz}, error vs its jvp {err:.1e}",
            flush=True,
        )
        if err < 1e-12:
            fx_reach = reach
            break
    if fx_reach is None:
        print("  F_x is not exact at any reach probed -- the staged estimate below is not valid.")
        return
    per_probe_fx = t_fx / p_fx

    # -- F_g estimate and the product -----------------------------------------------------------------
    g_cols = n_grad_fields * dim
    p_fg = colours[1] * g_cols
    t_fg = p_fg * per_probe_fx
    print(
        f"\n[F_g, estimated] {n_grad_fields} gradient fields x dim {dim} = {g_cols} columns per cell; "
        f"at reach 1 ({colours[1]} colours) that is {p_fg} probes, ~{t_fg:.3f} s at F_x's "
        f"{per_probe_fx * 1e3:.2f} ms/probe",
        flush=True,
    )

    rng = np.random.default_rng(1)
    fg = random_block_matrix(pattern_at(a, 1), nf, g_cols, rng)
    # G: each reconstructed field's gradient (dim rows) reads one state field at the gradient's own
    # reach. Two face passes reach distance 2. Which state field each reconstruction reads is not
    # recovered here; spreading them over the fields gives the right pattern size and product work.
    g_reach = 2
    gp = pattern_at(a, g_reach).tocoo()
    blocks_r, blocks_c = [], []
    for gfield in range(n_grad_fields):
        source = gfield % nf
        for d in range(dim):
            blocks_r.append((gfield * dim + d) * n + gp.row)
            blocks_c.append(source * n + gp.col)
    rows, cols = np.concatenate(blocks_r), np.concatenate(blocks_c)
    g = sp.csr_matrix((rng.standard_normal(rows.size), (rows, cols)), shape=(g_cols * n, nf * n))
    fx_random = random_block_matrix(pattern_at(a, fx_reach), nf, nf, rng)
    t_product = _timed(lambda: fx_random + fg @ g)
    composed = fx_random + fg @ g
    print(
        f"[product, SciPy, random values on the real patterns] F_g {fg.nnz} nnz @ G {g.nnz} nnz "
        f"+ F_x -> {composed.nnz} nnz in {t_product:.3f} s",
        flush=True,
    )

    t_staged = t_fx + t_fg + t_product
    print("\n[summary, per materialize]", flush=True)
    print(f"  shipped coloured probe of R      {t_shipped:7.3f} s   ({p_shipped} probes)")
    print(f"  staged: F_x probe (measured)     {t_fx:7.3f} s   ({p_fx} probes, reach {fx_reach})")
    print(f"          F_g probe (estimated)    {t_fg:7.3f} s   ({p_fg} probes, reach 1)")
    print(f"          F_x + F_g G (measured)   {t_product:7.3f} s")
    print(f"          total                    {t_staged:7.3f} s")
    print(f"  ratio shipped / staged           {t_shipped / t_staged:7.2f}x", flush=True)


if __name__ == "__main__":
    main()
