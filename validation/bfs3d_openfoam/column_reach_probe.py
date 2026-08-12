"""How far one column of the coupled Jacobian reaches, and which scheme decides it.

The coloured probe is charged per (colour, column field) and the colour count climbs steeply with the
stencil reach, so a column that reaches further is directly more expensive to materialize. On this case
the velocity columns reach graph distance three while pressure and the turbulence scalars stop at two,
which is worth understanding rather than just recording: the two momentum and scalar transport equations
look alike, so something other than the equation is setting it.

This answers it **without materializing anything**. Perturbing one degree of freedom and taking a single
directional derivative gives that whole column; how far the response spreads on the cell graph *is* the
column's reach. One jvp costs a few megabytes against a full materialize's gigabytes, so this runs
against a busy machine and can sweep schemes freely.

Arms are chosen to separate causes a single measurement would confound -- three momentum advection
schemes, the gradient reconstruction, and two ``stop_gradient`` arms that each remove one velocity path
from the Jacobian:

* ``limited`` -- the benchmark's own Venkatakrishnan-limited linear upwind.
* ``unlimited`` -- the same second-order reconstruction with the limiter removed. If the limiter were
  what widens the stencil, this arm would come back narrower.
* ``first-order`` -- no reconstruction at all, the scheme the k/omega equations already use.

**Measured on this case at ``state-00077``: the EDDY VISCOSITY's strain-rate dependence is what sets
the velocity reach.** Only one arm moves it — ``nu_t`` with the strain rate held constant takes velocity
from three to two; every scheme arm, the compact gradient, and the ``a_P`` arm all leave it at three.

``nu_t = a1 k / max(a1 omega, S F2)`` reads the strain-rate magnitude, which is built from the velocity
gradient, so the eddy viscosity is a *velocity-dependent coefficient* of a flux that is already
gradient-based; the composition spends the extra ring. That also accounts for the columns that stay at
two: ``k`` and ``omega`` enter ``nu_t`` **pointwise**, and pressure does not enter it at all.

Three explanations are **refuted** and should not be re-proposed: the flux limiter and the second-order
reconstruction (first-order upwind on momentum -- the scheme k and omega already use -- still reaches
three); the gradient scheme's own stencil (a one-shot compact Green-Gauss still reaches three); and the
velocity gradient inside the momentum diagonal's lagged convective flux.

The two ``stop_gradient`` arms are **diagnostics, not proposals**: they change the Jacobian rather than
the residual, so they measure a stencil rather than a model.

Usage::

    python3 -u validation/bfs3d_openfoam/column_reach_probe.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind  # noqa: E402
from aquaflux.flow.momentum import MomentumContinuity  # noqa: E402
from aquaflux.schemes import CompactGreenGauss  # noqa: E402
from aquaflux.turbulence import SSTModel  # noqa: E402

FIELDS = ("u", "v", "w", "p", "k", "omega")

#: Relative to the column's largest entry. Below this a response is float64 noise, not a coupling; the
#: materialized audit uses the same bar and finds the same split.
TOLERANCE = 1e-12

#: A developed state; see the note where it is loaded.
STATE = "state-00077"

ARMS = {
    "limited (the case)": None,
    "unlimited 2nd order": LimitedUpwind(limiter=None),
    "first-order upwind": FirstOrderUpwind(),
}


@contextlib.contextmanager
def eddy_viscosity_blind_to_strain():
    """Run the body with the eddy viscosity carrying no dependence on the strain rate.

    ``nu_t = a1 k / max(a1 omega, S F2)`` reads the strain-rate magnitude, which is built from the
    velocity gradient — so the eddy viscosity is a velocity-dependent coefficient of the viscous flux,
    and a coefficient one ring out multiplies a flux that is already one ring out. Cutting the
    derivative through ``S`` leaves the value untouched and removes exactly that path.

    A **diagnostic**: it changes the Jacobian, not the residual, so the arm measures a stencil rather
    than a model. (``stop_gradient`` on the strain rate is also what a lagged-``nu_t`` linearization
    would do, so it is not an exotic operator.)
    """
    original = SSTModel.eddy_viscosity

    def blind(self, k, omega, strain_rate, nu, d):
        return original(self, k, omega, jax.lax.stop_gradient(strain_rate), nu, d)

    SSTModel.eddy_viscosity = blind
    try:
        yield
    finally:
        SSTModel.eddy_viscosity = original


@contextlib.contextmanager
def a_p_flux_without_the_gradient():
    """Run the body with the momentum diagonal's lagged flux taking no velocity gradient.

    **Kept as a REFUTATION.** The account it was built to test: velocity reaches one ring further than
    its own equation because the mass flux carries ``V / a_P`` averaged across the face -- so a cell's
    residual reads its *neighbour's* momentum diagonal -- and that diagonal is built from a lagged
    convective flux which is itself a gradient-reconstructed velocity, adding a ring on the far side.
    Dropping the gradient there should then cost exactly that ring.

    **It does not: the velocity columns still reach three.** The ring is not in the pressure-velocity
    coupling at all; it is the eddy viscosity (see the module docstring). The arm stays because the
    account is a natural one to arrive at from reading the source, and re-deriving it is cheaper to
    refute from here than from scratch.

    This is a **diagnostic, not a proposed configuration**: it changes ``a_P`` and therefore the
    operator. It is only defensible as a probe because that flux is already a lagged stand-in for the
    mass flux, so this makes it more lagged rather than wrong.
    """
    original = MomentumContinuity._lagged_mdot_estimate

    def first_order(self, velocity, grad_velocity=None):
        return original(self, velocity, None)

    MomentumContinuity._lagged_mdot_estimate = first_order
    try:
        yield
    finally:
        MomentumContinuity._lagged_mdot_estimate = original


def graph_distance(owner, nb, n, sources, max_reach=5):
    """Breadth-first cell-graph distance from every source cell, capped at ``max_reach``."""
    adjacency = sp.coo_matrix(
        (np.ones(2 * owner.size), (np.concatenate([owner, nb]), np.concatenate([nb, owner]))),
        shape=(n, n),
    ).tocsr()
    distance = np.full((len(sources), n), max_reach + 1, dtype=np.int64)
    for row, source in enumerate(sources):
        front = np.zeros(n, dtype=bool)
        front[source] = True
        seen = front.copy()
        distance[row, source] = 0
        for step in range(1, max_reach + 1):
            front = (adjacency @ front.astype(np.float64) > 0) & ~seen
            distance[row, front] = step
            seen |= front
    return distance


def column_reach(coupled, state, cell, field, distance, n, n_fields):
    """The greatest cell-graph distance at which one column has a non-negligible entry."""
    seed = np.zeros(n_fields * n)
    seed[field * n + cell] = 1.0
    response = np.asarray(jax.jvp(coupled.residual, (jnp.asarray(state),), (jnp.asarray(seed),))[1])
    rows = np.abs(response).reshape(n_fields, n)
    # Threshold each ROW FIELD against its own largest entry. The coupled residual is ~100% omega --
    # omega rows run orders above the rest -- so one global threshold would put every velocity and
    # pressure row under the bar and report a reach of zero.
    live = (rows > TOLERANCE * rows.max(axis=1, keepdims=True)).any(axis=0)
    reached = distance[live]
    return int(reached.max()) if reached.size else 0


def main():
    print("[case] importing the mesh once; each arm rebuilds only the schemes", flush=True)
    reference = compare.build_case()
    mesh = reference["coupled"].momentum.mesh
    n = mesh.n_cells
    n_fields = reference["coupled"].layout.dim + 3
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)

    # Interior cells, far from every boundary: a wall-adjacent cell's stencil is truncated by the patch
    # closure and would understate the reach the mesh interior actually pays for.
    centroid = np.asarray(mesh.geometry().cell.centroid)
    span = centroid.max(axis=0) - centroid.min(axis=0)
    middle = centroid.min(axis=0) + 0.5 * span
    interior = np.argsort(np.linalg.norm((centroid - middle) / span, axis=1))[:4]
    print(f"  probing from cells {interior.tolist()} (deepest in the interior)", flush=True)

    distances = graph_distance(owner, nb, n, interior)

    # A DEVELOPED state, and this is load-bearing rather than tidiness. On a cold field the velocity
    # gradients are small enough that every second-order reconstruction contributes nothing above the
    # noise floor, so all three arms report the same reach -- the materialized audit sees exactly that,
    # measuring reach 2 for every column at a cold iterate and 3 for the velocities once the flow
    # develops. Probing cold would "show" the scheme does not matter, for a reason that is an artifact
    # of the state.
    checkpoint = CASE / "checkpoints" / f"{STATE}.npz"
    if not checkpoint.exists():
        raise SystemExit(
            f"{STATE}: no such checkpoint (these rotate; present: "
            f"{sorted(p.stem for p in checkpoint.parent.glob('state-*.npz'))})"
        )
    with np.load(checkpoint) as data:
        state = np.asarray(data["state"])
        print(
            f"  state {STATE}: step {int(data['step'])}, |R| {float(data['residual_norm']):.3e},"
            f" shift {float(data['shift']):.5g}",
            flush=True,
        )

    def report(label, coupled, note=""):
        reaches = [
            max(
                column_reach(coupled, state, int(cell), field, distances[row], n, n_fields)
                for row, cell in enumerate(interior)
            )
            for field in range(n_fields)
        ]
        cells = "  ".join(f"{FIELDS[f]} {reaches[f]}" for f in range(n_fields))
        print(f"  {label:24s} {cells}{note}", flush=True)

    print("\n  -- momentum advection scheme --", flush=True)
    for label, scheme in ARMS.items():
        case = reference if scheme is None else compare.build_case(momentum_advection=scheme)
        report(label, case["coupled"])
        del case

    print("\n  -- isolating arms: each removes ONE velocity path from the Jacobian --", flush=True)
    # The gradient reconstruction's own stencil. A corrected Green-Gauss gradient is not one-shot, so a
    # gradient evaluated at a neighbour can read further than that neighbour's own ring; the compact
    # scheme is one-shot and cannot.
    case = compare.build_case(gradient=CompactGreenGauss())
    report("compact Green-Gauss", case["coupled"])
    del case

    for label, patch in (
        ("a_P flux, no gradient", a_p_flux_without_the_gradient),
        ("nu_t blind to strain", eddy_viscosity_blind_to_strain),
    ):
        with patch():  # rebuilt INSIDE the patch, or the case keeps the unpatched closure
            case = compare.build_case()
            report(label, case["coupled"])
            del case


if __name__ == "__main__":
    main()
