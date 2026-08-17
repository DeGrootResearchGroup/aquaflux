"""Why does a zero-fill ILU fail on pitzDaily and not on bfs3d? A one-variable skewness sweep.

Every arm is judged the same way: assemble the coupled Jacobian at a named state, add the
pseudo-transient shift, equilibrate + reorder cell-major (exactly what the library hands PETSc),
then run PETSc GMRES with KSP_NORM_UNPRECONDITIONED and read the TRUE residual computed
independently in scipy.  ILU(0) and ILU(1) differ by ONE PETSc option, so the arms cannot
differ by implementation.

Cases:
  pitz / pitz-compact   pitzDaily, corrected vs compact Green-Gauss (isolates the correction)
  grid2d-<p>            coupled RANS on a perturbed structured 2D grid, perturbation p
  grid3d-<p>            the 3D counterpart (isolates 2D-vs-3D)
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

# Run directly from anywhere: resolve the repository root from THIS file rather than pinning a
# checkout, so the harness survives being run from another worktree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "pitzdaily_openfoam"))

import jax.numpy as jnp  # noqa: E402
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient  # noqa: E402
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind  # noqa: E402
from aquaflux.flow import (  # noqa: E402
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.properties import Constant, PropertyModel  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    CompactGreenGauss,
    CorrectedGreenGauss,
    VenkatakrishnanLimiter,
)
from aquaflux.solve import (  # noqa: E402
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    equilibrate_cell_major,
)
from aquaflux.turbulence import (  # noqa: E402
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)
from aquaflux.vectors import scale  # noqa: E402

FIELDS2 = ("u", "v", "p", "k", "omega")
FIELDS3 = ("u", "v", "w", "p", "k", "omega")


# --------------------------------------------------------------------------- cases


def build_pitz(gradient="corrected"):
    """pitzDaily exactly as `compare.build_case` builds it, with the gradient scheme swappable.

    Monkeypatching the name compare.py reads keeps every other choice — mesh, BCs, model
    constants, schemes, log-omega — literally the case's own, so the arm is the gradient
    scheme and nothing else.
    """
    import compare

    original = compare.CorrectedGreenGauss
    if gradient == "compact":
        compare.CorrectedGreenGauss = CompactGreenGauss
    try:
        return compare.build_case()["coupled"]
    finally:
        compare.CorrectedGreenGauss = original


# The perturbed-grid channel: pitzDaily's SCHEME bundle (corrected Green-Gauss, limited
# second-order momentum upwind, first-order scalars, log-omega) on a structured grid whose
# interior nodes are displaced by a controlled fraction of the cell size.  Perturbation is
# then the only variable across the sweep.
RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H


def build_grid(perturb, dim=2, nx=24, ny=16, nz=8, seed=1, gradient="corrected", growth=1.0):
    """The channel. `growth` grades the wall-normal spacing, which moves the ASPECT RATIO alone.

    Grading is applied by displacing the y coordinates of the whole lattice AFTER the perturbation,
    so a graded arm and an ungraded one differ in cell aspect ratio and in nothing else -- the
    topology, the interior-node noise and every scheme are identical.
    """
    import equinox as eqx
    from aquaflux.mesh import graded_nodes
    from tests.support.meshes import perturbed_grid_2d, perturbed_grid_3d

    if dim == 2:
        mesh = perturbed_grid_2d(nx, ny, lx=L, ly=H, perturb=perturb, seed=seed,
                                 named_boundaries=True)
        walls = ["bottom", "top"]
        inlet_velocity = (U_IN, 0.0)
    else:
        mesh = perturbed_grid_3d(nx, ny, nz, lx=L, ly=H, lz=H, perturb=perturb, seed=seed,
                                 named_boundaries=True)
        walls = ["bottom", "top", "front", "back"]
        inlet_velocity = (U_IN, 0.0, 0.0)
    if growth != 1.0:
        # Map the uniform y lattice onto a graded one by interpolation, so the perturbation's
        # relative displacement is carried through rather than being flattened.
        graded = np.asarray(graded_nodes(ny, H, growth))
        uniform = np.linspace(0.0, H, ny + 1)
        coords = np.array(mesh.node_coords, dtype=np.float64, copy=True)
        coords[:, 1] = np.interp(coords[:, 1], uniform, graded)
        mesh = eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(coords))
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    grad = CorrectedGreenGauss() if gradient == "corrected" else CompactGreenGauss()
    flow_bc = {
        "left": VelocityInlet(velocity=inlet_velocity),
        "right": PressureOutlet(pressure=0.0),
    }
    scalar_k_bc = {"left": Dirichlet(k_in), "right": ZeroGradient()}
    scalar_w_bc = {"left": Dirichlet(omega_in), "right": ZeroGradient()}
    for wall in walls:
        flow_bc[wall] = NoSlipWall()
        scalar_k_bc[wall] = Dirichlet(0.0)
        scalar_w_bc[wall] = ZeroGradient()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(flow_bc),
        advection_scheme=LimitedUpwind(limiter=VenkatakrishnanLimiter()),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        grad,
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=walls,
        k_boundary=BoundaryConditions(scalar_k_bc),
        omega_boundary=BoundaryConditions(scalar_w_bc),
    )
    return CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())


# --------------------------------------------------------------------------- mesh metrics


def skew_metrics(coupled):
    """Per-interior-face |x_f - (x_P + g d)| / |d| -- the same measure the record quotes."""
    from aquaflux.schemes.interpolation import interpolation_factor

    mesh = coupled.momentum.mesh
    geom = mesh.geometry()
    fc = mesh.face_cells
    g = interpolation_factor(fc, geom)
    x_p = geom.cell.centroid[fc.owner]
    d = fc.neighbour_centroid(geom.cell.centroid) - x_p
    skew = geom.face.centroid - (x_p + scale(d, g))
    ratio = np.asarray(jnp.linalg.norm(skew, axis=-1) / jnp.linalg.norm(d, axis=-1))
    interior = np.asarray(fc.interior)
    return ratio, interior


def per_cell_skew(coupled):
    """The largest interior-face skewness ratio touching each cell."""
    ratio, interior = skew_metrics(coupled)
    mesh = coupled.momentum.mesh
    fc = mesh.face_cells
    owner, nb = np.asarray(fc.owner), np.asarray(fc.neighbour)
    out = np.zeros(mesh.n_cells)
    idx = np.flatnonzero(interior)
    np.maximum.at(out, owner[idx], ratio[idx])
    np.maximum.at(out, nb[idx], ratio[idx])
    return out


def boundary_faces_per_cell(coupled):
    mesh = coupled.momentum.mesh
    fc = mesh.face_cells
    interior = np.asarray(fc.interior)
    owner = np.asarray(fc.owner)
    out = np.zeros(mesh.n_cells, dtype=int)
    np.add.at(out, owner[~interior], 1)
    return out


def cell_aspect_ratio(coupled):
    """A crude per-cell aspect ratio: max/min face-centroid distance from the cell centroid."""
    mesh = coupled.momentum.mesh
    geom = mesh.geometry()
    fc = mesh.face_cells
    centroid = np.asarray(geom.cell.centroid)
    fcentroid = np.asarray(geom.face.centroid)
    owner, nb = np.asarray(fc.owner), np.asarray(fc.neighbour)
    interior = np.asarray(fc.interior)
    lo = np.full(mesh.n_cells, np.inf)
    hi = np.zeros(mesh.n_cells)
    for cells, faces in ((owner, np.arange(len(owner))),
                         (nb[interior], np.flatnonzero(interior))):
        dist = np.linalg.norm(fcentroid[faces] - centroid[cells], axis=-1)
        np.minimum.at(lo, cells, dist)
        np.maximum.at(hi, cells, dist)
    return hi / np.maximum(lo, 1e-300)


# --------------------------------------------------------------------------- measurement


def seed_state(coupled):
    flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    state = coupled.state_from_physical(flow, k, omega)
    residual = coupled.residual(state)
    assert bool(jnp.all(jnp.isfinite(residual))), "the seed state's residual is NOT finite"
    return state, residual


def materialize(coupled, state, reach):
    plan = _coupled_jacobian_plan(coupled, reach, None)
    structure = block_stencil_gather_map(plan)
    started = time.time()
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    print(f"    reach {reach}: {jacobian.shape[0]} dofs, {jacobian.nnz / 1e6:.2f}M nnz "
          f"in {time.time() - started:.0f}s", flush=True)
    return jacobian


def block_jacobian_error(coupled, state, jacobian, n_fields, seed=0):
    """Per (row field, column field) relative error of the recovered matrix vs the true jvp.

    A whole-matrix single-vector check collapses over row fields and cannot see a wrong
    pressure block; this reads the error per PAIR.
    """
    n = coupled.layout.n_cells
    rng = np.random.default_rng(seed)
    rows = np.zeros((n_fields, n_fields))
    norms = np.zeros((n_fields, n_fields))
    for col in range(n_fields):
        v = np.zeros(n_fields * n)
        v[col * n:(col + 1) * n] = rng.standard_normal(n)
        exact = np.asarray(_jacobian_matvec(coupled, state, jnp.asarray(v)))
        got = jacobian @ v
        for row in range(n_fields):
            sl = slice(row * n, (row + 1) * n)
            rows[row, col] = np.linalg.norm(got[sl] - exact[sl])
            norms[row, col] = np.linalg.norm(exact[sl])
    return rows / np.maximum(norms, 1e-300)


def assemble(jacobian, shift, n_fields):
    return equilibrate_cell_major(MonolithicAmgPreconditioner._shifted(jacobian, shift), n_fields)


def ilu_pivots(cell_major, n_fields, levels):
    """min |pivot|, negative count, and the negative pivots' row indices, via PETSc's own ILU(k)."""
    from petsc4py import PETSc

    mat = PETSc.Mat().createAIJWithArrays(
        size=cell_major.shape,
        csr=(cell_major.indptr.astype(PETSc.IntType),
             cell_major.indices.astype(PETSc.IntType),
             cell_major.data.astype(PETSc.ScalarType)),
    )
    mat.setBlockSize(n_fields)
    mat.assemble()
    pc = PETSc.PC().create()
    try:
        pc.setOperators(mat)
        pc.setType("ilu")
        pc.setFactorLevels(levels)
        pc.setUp()
        factor = pc.getFactorMatrix()
        reciprocal = factor.getDiagonal().getArray().copy()
        nnz = int(factor.getInfo()["nz_used"])
    except Exception as failure:
        return {"failed": f"{type(failure).__name__}: {failure}"}
    finally:
        pc.destroy()
        mat.destroy()
    finite = reciprocal != 0.0
    pivots = np.where(finite, 1.0 / np.where(finite, reciprocal, 1.0), np.inf)
    magnitude = np.abs(pivots)
    return {
        "min": float(magnitude.min()),
        "negative": int((pivots < 0.0).sum()),
        "negative_rows": np.flatnonzero(pivots < 0.0),
        "median": float(np.median(magnitude)),
        "nnz": nnz,
    }


def ksp_solve(cell_major, rhs_eq, n_fields, arm, rtol=1e-8, max_it=400):
    """GMRES(30) preconditioned by `arm`, stopping on the UNPRECONDITIONED residual.

    Returns (iterations, true relative residual of the equilibrated system, reason, x).
    """
    from petsc4py import PETSc

    prefix = f"probe{abs(hash((arm, cell_major.nnz))) % 10**6}_"
    mat = PETSc.Mat().createAIJWithArrays(
        size=cell_major.shape,
        csr=(cell_major.indptr.astype(PETSc.IntType),
             cell_major.indices.astype(PETSc.IntType),
             cell_major.data.astype(PETSc.ScalarType)),
    )
    mat.setBlockSize(n_fields)
    mat.assemble()
    b = mat.createVecLeft()
    b.setArray(np.ascontiguousarray(rhs_eq, dtype=PETSc.ScalarType))
    x = mat.createVecRight()
    ksp = PETSc.KSP().create()
    opts = PETSc.Options()
    settings = {}
    try:
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(mat)
        ksp.setType("gmres")
        ksp.setNormType(PETSc.KSP.NormType.NORM_UNPRECONDITIONED)
        ksp.setTolerances(rtol=rtol, atol=1e-50, divtol=1e12, max_it=max_it)
        settings.update({"ksp_gmres_restart": 30})
        if arm.startswith("gamg"):
            fill, sweeps = (int(part) for part in arm.split("-")[1].split("x"))
            settings |= {
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "pc_gamg_agg_nsmooths": 0,
                "pc_gamg_coarse_eq_limit": 2000,
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "lu",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_ksp_max_it": sweeps,
                "mg_levels_pc_type": "ilu",
                "mg_levels_pc_factor_levels": fill,
            }
        else:
            settings |= {"pc_type": "ilu", "pc_factor_levels": int(arm[-1])}
        for key, value in settings.items():
            opts[prefix + key] = value
        ksp.setFromOptions()
        ksp.getPC().setOptionsPrefix(prefix)
        started = time.time()
        ksp.solve(b, x)
        elapsed = time.time() - started
        solution = x.getArray().copy()
        return {
            "its": int(ksp.getIterationNumber()),
            "reason": int(ksp.getConvergedReason()),
            "seconds": elapsed,
            "x": solution,
        }
    except Exception as failure:
        return {"failed": f"{type(failure).__name__}: {failure}"}
    finally:
        for key in settings:
            opts.delValue(prefix + key)
        ksp.destroy()
        mat.destroy()
        b.destroy()
        x.destroy()


ARMS = ("ilu0", "ilu1", "gamg-0x4", "gamg-1x4")


def run_case(name, coupled, betas, reach=3, arms=ARMS, localize=False):
    dim = coupled.layout.dim
    n_fields = dim + 3
    names = FIELDS2 if dim == 2 else FIELDS3
    n = coupled.layout.n_cells
    ratio, interior = skew_metrics(coupled)
    live = ratio[interior]
    print(f"\n{'=' * 96}\nCASE {name}: {n} cells, {dim}D, {n_fields} fields, {n_fields * n} dofs")
    print(f"  interior-face skewness |x_f-(x_P+g d)|/|d|: median {np.median(live):.3e}  "
          f"p99 {np.quantile(live, 0.99):.3e}  max {live.max():.3e}  "
          f"above 1e-6: {int((live > 1e-6).sum())} of {live.size}", flush=True)

    state, residual = seed_state(coupled)
    rhs = -np.asarray(residual, dtype=np.float64)
    print(f"  seed state: |R| {np.linalg.norm(rhs):.4e}, all finite", flush=True)

    jacobian = materialize(coupled, state, reach)
    err = block_jacobian_error(coupled, state, jacobian, n_fields)
    print("  probe error per (row field, column field), worst per column:")
    print("    " + "  ".join(f"{names[c]}:{err[:, c].max():.2e}" for c in range(n_fields)),
          flush=True)

    base = _coupled_shift_policy(coupled, state, "twolevel")
    results = {}
    for beta in betas:
        shift = (_frozen_shift_diagonal(base, beta, state) if beta > 0
                 else np.zeros(n_fields * n))
        cell_major, scaling, perm = assemble(jacobian, np.asarray(shift), n_fields)
        rhs_eq = (np.asarray(scaling) * rhs)[perm]
        print(f"\n  -- beta {beta}: nnz {cell_major.nnz / 1e6:.2f}M", flush=True)
        for levels in (0, 1):
            census = ilu_pivots(cell_major, n_fields, levels)
            if "failed" in census:
                print(f"     ILU({levels}) census REFUSED  {census['failed']}", flush=True)
                continue
            print(f"     ILU({levels}) pivots: min |p| {census['min']:.3e}  negative "
                  f"{census['negative']:>6}  median {census['median']:.3e}  "
                  f"factor nnz {census['nnz'] / 1e6:.2f}M", flush=True)
            if localize and census["negative"]:
                report_negative(name, beta, levels, census["negative_rows"], coupled, perm,
                                n_fields, names)
        for arm in arms:
            out = ksp_solve(cell_major, rhs_eq, n_fields, arm)
            if "failed" in out:
                print(f"     {arm:<10} FAILED  {out['failed']}", flush=True)
                results[(beta, arm)] = None
                continue
            y = out["x"]
            true_eq = np.linalg.norm(cell_major @ y - rhs_eq) / np.linalg.norm(rhs_eq)
            print(f"     {arm:<10} its {out['its']:>4}  reason {out['reason']:>3}  "
                  f"TRUE rel {true_eq:.3e}  {out['seconds']:>5.1f}s", flush=True)
            results[(beta, arm)] = (out["its"], true_eq, out["reason"])
        del cell_major
        gc.collect()
    del jacobian
    gc.collect()
    return results


def report_negative(name, beta, levels, rows, coupled, perm, n_fields, names):
    """Which rows carry the negative pivots, and how those cells compare with the base rate."""
    # cell-major: row = cell * n_fields + field
    cells = rows // n_fields
    fields = rows % n_fields
    counts = np.bincount(fields, minlength=n_fields)
    print("       negative pivots by field: "
          + "  ".join(f"{names[f]} {counts[f]}" for f in range(n_fields)), flush=True)
    unique = np.unique(cells)
    skew = per_cell_skew(coupled)
    walls = boundary_faces_per_cell(coupled)
    ar = cell_aspect_ratio(coupled)
    n = len(skew)
    for label, quantity in (("skew", skew), ("wall faces", walls), ("aspect ratio", ar)):
        hit = quantity[unique]
        print(f"       {label:<13} hit median {np.median(hit):.3e}  "
              f"mesh median {np.median(quantity):.3e}  "
              f"hit share above mesh p90 {float((hit > np.quantile(quantity, 0.9)).mean()):.2f} "
              f"(base rate 0.10)", flush=True)
    print(f"       {len(unique)} distinct cells of {n}", flush=True)


# --------------------------------------------------------------------------- main

BETAS = tuple(float(b) for b in os.environ.get("PROBE_BETAS", "0.0,0.05,0.5,2.0").split(","))
REACH = int(os.environ.get("PROBE_REACH", "3"))


def main():
    which = sys.argv[1:] or ["grid2d-0.0"]
    for spec in which:
        started = time.time()
        if spec == "pitz":
            coupled = build_pitz("corrected")
        elif spec == "pitz-compact":
            coupled = build_pitz("compact")
        elif spec.startswith("grid2d-"):
            coupled = build_grid(float(spec.split("-")[1]), dim=2)
        elif spec.startswith("grid2dc-"):  # compact gradient control
            coupled = build_grid(float(spec.split("-")[1]), dim=2, gradient="compact")
        elif spec.startswith("grid3d-"):
            coupled = build_grid(float(spec.split("-")[1]), dim=3)
        else:
            raise SystemExit(f"unknown case {spec!r}")
        run_case(spec, coupled, BETAS, reach=REACH, localize=True)
        del coupled
        gc.collect()
        print(f"  [{spec} done in {time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
