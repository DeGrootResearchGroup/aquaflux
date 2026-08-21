# Gradient reconstruction

A finite-volume solver stores one value per cell, but almost everything it computes needs a
**gradient**: the non-orthogonal correction in a diffusion flux, the reconstruction of a face
value for second-order advection, and the pressure gradient in a momentum balance. On a mesh
of perfect cubes any reasonable estimate will do. On a real mesh — graded, skewed, cut by a
CAD surface — the gradient scheme is often what sets the accuracy of the whole solve.

This page covers the three reconstructions `aquaflux` ships, how each one is solved, and how
to choose between them.

If you only want the short answer: use {class}`~aquaflux.schemes.CorrectedGreenGauss`, which
is the default everywhere, and read on if your mesh is skewed enough that you suspect the
gradient is the limiting error.

```{note}
The examples below use a `mesh` and its `geometry`, plus a cell field `phi` and the field's
values on the boundary faces. Every scheme has the same interface, so they are interchangeable
in any of the places a scheme is injected — a `ResidualAssembler`,
a {class}`~aquaflux.flow.MomentumContinuity`, a turbulence closure.
```

## The three schemes

```python
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    HessianCorrectedGradient,
)

gradient = CorrectedGreenGauss().gradients(phi, mesh, geometry, boundary_values)
```

### Compact Green–Gauss

{class}`~aquaflux.schemes.CompactGreenGauss` applies the divergence theorem directly:

```text
grad(phi)_P = (1 / V_P) * sum_faces  phi_ip * S_f
```

with the face value linearly interpolated between the two cells. It is one pass over the
faces, with no linear system, and it is second-order accurate on an orthogonal grid.

Its weakness is specific and worth knowing: the interpolation gives the value at the point
where the face centroid **projects onto the line joining the two cell centroids**, not at the
face centroid itself. On a skewed mesh those are different points, and the resulting error
does not shrink as the mesh refines — the scheme is *inconsistent* there, converging at
roughly zeroth order. Use it on near-orthogonal meshes, or where the gradient enters only as a
small correction.

### Corrected Green–Gauss

{class}`~aquaflux.schemes.CorrectedGreenGauss` closes exactly that gap. It adds the offset
from the projection foot to the true face centroid, evaluated with the gradient itself:

```text
phi_ip = (1-g) phi_P + g phi_N  +  [(1-g) grad(phi)_P + g grad(phi)_N] . D_g,ip
```

Because the correction uses the gradient being solved for, this is no longer a single pass —
it is a sparse linear system `A_g G = B phi` coupling each cell to its face neighbours. The
operator `A_g` depends only on the mesh, never on the field.

The payoff is that the reconstruction becomes **exact for linear fields on any mesh**, which
restores consistency on irregular grids. It remains capped near first order there for general
fields, because a Green–Gauss sum reproduces the mean of the face value, not its value at the
centroid, and that distinction survives the linear correction. This is the default scheme.

### Hessian-corrected (Betchen)

{class}`~aquaflux.schemes.HessianCorrectedGradient` implements the coupled gradient-and-Hessian
reconstruction of Betchen & Straatman (2010). Reconstructing the second derivative alongside
the gradient lets the face integral be exact for a quadratic, which lifts the gradient itself
to second order on any mesh:

```text
[ A_gg  A_gH ] [ g ]   [ b_g ]
[ A_Hg  A_HH ] [ H ] = [  0  ]
```

The Hessian is wanted only to correct the gradient, never as an output, so it is eliminated:
the system actually solved is `S g = b_g` with `S = A_gg - A_gH A_HH⁻¹ A_Hg`. Every block is
formed by automatic differentiation of the forward reconstruction rather than from
hand-derived coefficient matrices, so there is one statement of the discretization and no
second copy to drift.

It is exact for linear **and** quadratic fields on any mesh with planar faces. Where the
gradient enters a face value at leading order — advection, Rhie–Chow — that is the difference
between a scheme that caps near first order on a skewed mesh and one that does not.

```{warning}
A **warped** face breaks Green–Gauss exactness for a quadratic for every scheme in this
family, this one included, because the face integral itself is no longer exact. On a mesh
with badly non-planar faces the reconstruction is limited by the faces, not by the scheme;
{func}`~aquaflux.mesh.face_planarity` reports how close each face is to planar.
```

## How the system is solved

Two of the three schemes reduce to a linear system, and *how* that system is inverted is a
separate, injected choice — the discretization is identical either way. It matters because
the reconstruction runs inside every residual evaluation, so its cost is multiplied by
everything above it.

{class}`~aquaflux.schemes.SweptGradientSolve` applies a fixed number of preconditioned
Richardson sweeps:

```text
x <- x + P⁻¹ (b - A x)
```

There is no matrix, no nested Krylov solve, and no implicit-diff tangent: the loop has a
static length, so differentiating it is just unrolling. Its cost is linear in the mesh, which
is what makes it the default.

{class}`~aquaflux.schemes.GmresGradientSolve` solves the same system with matrix-free GMRES to
a requested tolerance, and is differentiated by implicit differentiation. It is exact to that
tolerance regardless of conditioning, at the cost of a nested Krylov solve inside every
reconstruction.

That cost is easy to underestimate, because it does not fall where you look for it. A flow solve
spends most of its time in Krylov iterations, and each one is a **Jacobian–vector product** through
the residual — so each one differentiates the reconstruction. A fixed sweep is differentiated by
unrolling, which costs about what the forward pass costs. Implicit differentiation instead solves a
*second* linear system per product. Measured on a coupled RANS case, per reconstruction against a
corrected Green–Gauss baseline of 1.0, at matched accuracy: a Krylov solve of the Hessian-corrected
system costs `93×` forward and `159×` as a Jacobian–vector product; a fixed sweep costs `67×` and
`60×`. The gap on the forward pass is modest; on the path the solver actually repeats, it is 2.6×.

This is why both schemes default to a fixed sweep, and why the Krylov strategy is kept for what it is
genuinely better at: a mesh skewed enough that the sweep count would grow impractically, and providing
an exact reference to calibrate a sweep count against.

```python
from aquaflux.schemes import CorrectedGreenGauss, GmresGradientSolve, SweptGradientSolve

CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=8))   # more sweeps for a skewed mesh
CorrectedGreenGauss(solver=GmresGradientSolve())           # exact, for a reference value

# The Hessian-corrected scheme has two systems and takes a strategy for each.
HessianCorrectedGradient(hessian_solver=GmresGradientSolve())   # exact inner, to calibrate against
```

### Preconditioning, and why it is a separate choice

Both solves take a {class}`~aquaflux.schemes.GradientPreconditioner` — an approximate inverse
of the operator's **per-cell diagonal block**. There are two, and the difference between them
is not a tuning knob:

{class}`~aquaflux.schemes.InverseVolume` scales each cell's residual by `1/V`. The corrected
gradient's operator is `A_g = V ⊙ I − (skewness coupling)`, so the volume really is most of
its diagonal block and this converges quickly.

{class}`~aquaflux.schemes.CellBlockJacobi` inverts each cell's own diagonal block exactly. The
gradient–Hessian system needs this, because there a cell's gradient and its Hessian couple to
*each other* at the same order as the volume term — something `1/V` cannot represent at all,
rather than merely approximate. The consequence is sharp: on a perfectly orthogonal mesh,
where there is no skewness whatsoever, inverse-volume Richardson on that system still has a
spectral radius of `0.5`, while inverting the cell block gives `0`. That is why the scheme
looked as though it needed a Krylov solve, and why it no longer does.

Each scheme supplies the preconditioner its own system needs, so this is not usually something
you set. It is described here because it explains the sweep counts below.

## Choosing the sweep count

A fixed sweep count carries no convergence test, so it is worth knowing what sets it: the
**mesh's skewness, not its size**. That is what keeps the cost linear in the number of cells,
and it also means a count calibrated on a small test mesh does not transfer to a skewed one.

For the corrected gradient, the default of four sweeps reaches the exactly-solved gradient to
well within discretization error on mildly non-orthogonal meshes, and a strongly skewed mesh
wants more. For the Hessian-corrected scheme, the inner Hessian solve contracts geometrically
at a rate set by skewness alone: measured on a heavily skewed hexahedral grid, the
reconstruction's departure from exactness for a quadratic falls by about two and a half orders
per two sweeps, reaching the exactly-solved answer to machine precision at the default of ten.

The way to check a mesh you have not used before is to compare against an exact solve of the
same system, which is what `GmresGradientSolve` provides:

```python
approx = HessianCorrectedGradient().gradients(phi, mesh, geometry, boundary_values)
exact = HessianCorrectedGradient(
    hessian_solver=GmresGradientSolve()
).gradients(phi, mesh, geometry, boundary_values)
```

If those agree to well below your discretization error, the default is sufficient on that
mesh. A quadratic field is the sharpest probe, since the scheme is defined to reproduce it
exactly and any departure is the solve rather than the discretization.

```{warning}
Because the sweep is fixed, an under-resolved mesh loses that exactness **silently** — there
is no residual test to fail. Check it once per new mesh rather than assuming.
```

## Choosing a scheme

| | mesh it suits | exact for | solve cost |
|---|---|---|---|
| {class}`~aquaflux.schemes.CompactGreenGauss` | near-orthogonal | linear, on orthogonal grids | one pass, no system |
| {class}`~aquaflux.schemes.CorrectedGreenGauss` | any | linear, on any mesh | a few sparse sweeps |
| {class}`~aquaflux.schemes.HessianCorrectedGradient` | skewed, where the gradient leads | linear and quadratic | an inner and an outer solve |

Two practical points beyond accuracy.

**The Hessian-corrected scheme has a much longer stencil.** Its gradient couples to the
Hessian, which couples to the neighbours' Hessians, so the reconstruction at a cell depends on
cells far across the mesh — where the corrected gradient's dependence stops a fixed number of
rings away, set by its sweep count. That is invisible to the solve itself, but it matters to
anything that assembles an approximate Jacobian over a bounded stencil: such an approximation
does not simply drop the couplings it cannot see. Preconditioners that build their own
operator from the mesh — the block preconditioners described in
[Preconditioning](preconditioning.md) — are unaffected, because they never try to represent
the gradient's coupling in the first place.

**Only the corrected gradient runs under domain decomposition.** Its Richardson sweeps form no
global inner product, so a partitioned solve can refresh ghost values once per sweep and get
owned gradients identical to a serial run. The Krylov solve and the Hessian-corrected scheme
raise rather than return a quietly wrong answer.

## A reference

Betchen, L. J. and Straatman, A. G. (2010). "An accurate gradient and Hessian reconstruction
method for cell-centered finite volume discretizations on general unstructured grids."
*International Journal for Numerical Methods in Fluids* 62(9), 945–962.
