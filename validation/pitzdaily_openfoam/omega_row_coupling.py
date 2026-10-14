"""Which term of the omega equation couples it to the cell's own pressure and velocity?

The coupled operator's per-cell blocks are near-singular in a handful of near-wall, low-``k`` cells, and
the reason is a large entry in the omega ROW on the cell's own pressure and velocity (see
``validation/bfs3d_openfoam/cell_block_conditioning.py``, which removes couplings to tell that from an
empty omega column). This script says where that entry comes from. It splits the omega residual into its
five terms -- advection, diffusion, production, destruction, cross-diffusion -- each assembled with the
same operator, boundary closures, gradient scheme and imposed wall gradient the residual uses, checks
that they sum to the residual's omega rows, and differentiates each one along a unit pressure or velocity
perturbation in the chosen cells (``jax.jvp``, so every entry is exact).

It also reports, per cell, the advection entry rewritten in the **bounded** (non-conservative) form
``R_omega - omega_P * R_continuity / rho``: the omega advection less omega times the net volume flux, the
form a segregated finite-volume code assembles for a scalar so that a continuity error is not convected
as a source (OpenFOAM's ``bounded Gauss`` scheme). The two forms share every root -- the difference is
``omega_P`` times a continuity residual that vanishes there -- and differ in the Jacobian by exactly
``omega_P`` times the continuity row. Its pressure entry is ``sum_f D_f (omega_upwind,f - omega_P)``
rather than ``sum_f D_f omega_upwind,f``, with ``D_f`` the face flux's pressure sensitivity, so it
measures how much of the coupling is omega's *level* and how much its *variation* across the faces.

Finally it repeats the per-cell block census of ``cell_block_conditioning.py`` with the omega row
rewritten that way (transported rows only: a wall-fixed row is a value fixation with no advection), which
says whether the near-singular blocks are a property of the conservative form.

Measured on pitzDaily ``state-00017`` at shift 1e-4 (2026-10-14): every pressure entry and almost every
velocity entry of the flagged cells' omega rows is advection. In the bulk it is ``omega_P`` times the
continuity row exactly, which the bounded form removes. In the flagged cells -- the second ring off a
wall -- about 60 % survives it: the wall-fixed neighbour's large omega convected across one face.

**Usage** -- checkpoint index, shift, then the cells to decompose::

    python3 validation/pitzdaily_openfoam/omega_row_coupling.py 17 0.0001 721 1081 10619 5000
    python3 validation/pitzdaily_openfoam/omega_row_coupling.py --case bfs3d_openfoam 11 0.0173 22480 12880

``--case`` names another case directory under ``validation/`` with a ``compare.build_case`` and a
``checkpoints/`` directory. The census materializes the coupled Jacobian, so run it on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _case_directory(argv: list[str]) -> tuple[Path, list[str]]:
    """The case directory ``--case`` names (this script's own by default), and the remaining arguments."""
    if argv[:1] == ["--case"]:
        return ROOT / "validation" / argv[1], argv[2:]
    return Path(__file__).resolve().parent, argv


CASE, ARGUMENTS = _case_directory(sys.argv[1:])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "bfs3d_openfoam"))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.discretization import AdvectionFlux, DiffusionFlux  # noqa: E402
from aquaflux.precondition import MaterializedJacobianPreconditioner  # noqa: E402
from aquaflux.precondition.frozen_operator import equilibrate_cell_major  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
    frozen_shift_diagonal,
    jacobian_matvec,
)
from aquaflux.precondition.sparse_jacobian import block_stencil_gather_map  # noqa: E402
from aquaflux.properties import PropertyModel  # noqa: E402
from aquaflux.turbulence import ScalarTwoLevel  # noqa: E402
from aquaflux.turbulence.coupled import _coupled_jacobian_plan, _coupled_shift_policy  # noqa: E402
from aquaflux.turbulence.sources import (  # noqa: E402
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from cell_block_conditioning import (  # noqa: E402
    FIELDS,
    SINGULAR_BELOW,
    conditioning,
    diagonal_blocks,
)

TERMS = ("advection", "diffusion", "production", "destruction", "cross-diffusion")


def omega_terms(coupled, state: jnp.ndarray) -> jnp.ndarray:
    """The omega residual's five terms, shape ``(5, n_cells)``, in :data:`TERMS` order.

    Mirrors ``CoupledRANS.residual`` and ``SSTTurbulence.omega_residual`` term by term: the closure and
    the Rhie--Chow mass flux are formed once at ``state`` exactly as the residual forms them, and each
    term is its own assembler over the omega equation's boundary closures, gradient scheme and imposed
    wall gradient. The wall-fixed rows are not replaced here; :func:`main` masks them.
    """
    turbulence = coupled.turbulence
    flow, k, omega = coupled.physical_fields(state)
    closure, momentum = coupled.effective_momentum(flow, k, omega)
    mdot = momentum.flow_fields(flow).mdot
    diffusivity = turbulence._diffusivity(
        closure.nu_t, closure.f1, turbulence.model.sigma_omega1, turbulence.model.sigma_omega2
    )
    model = turbulence.model
    operators = (
        ((AdvectionFlux(turbulence._volume_flux(mdot), turbulence.omega_advection_scheme),), ()),
        ((DiffusionFlux(),), ()),
        (
            (),
            (
                OmegaProduction(
                    closure.strain_rate, closure.nu_t, closure.k, closure.omega, closure.f1, model
                ),
            ),
        ),
        ((), (OmegaDestruction(closure.f1, model),)),
        (
            (),
            (
                OmegaCrossDiffusion(
                    closure.omega, closure.grad_k, closure.grad_omega, closure.f1, model
                ),
            ),
        ),
    )
    rows = []
    for fluxes, sources in operators:
        assembler = turbulence._assembler(
            PropertyModel({"diffusivity": diffusivity}),
            fluxes,
            turbulence.omega_boundary,
            gradient_scheme=turbulence.omega_gradient_scheme,
            source_operators=sources,
            imposed_gradient=closure.imposed_omega_gradient,
        )
        rows.append(assembler.residual(omega))
    return jnp.stack(rows)


def unit(n: int, index: int) -> jnp.ndarray:
    """The unit vector ``e_index`` of length ``n``."""
    return jnp.zeros(n).at[index].set(1.0)


def bounded_row_operator(
    coupled, state: jnp.ndarray, n_fields: int, n_cells: int, transported: np.ndarray
) -> sp.csr_matrix:
    """``L = I - diag(omega) P``: the transported omega rows less ``omega_P / rho`` times continuity.

    ``P`` maps a cell's continuity row onto its omega row, for the cells in ``transported`` only: a
    wall-fixed omega row is a value fixation, not a transport balance, and has no advection to rewrite.
    Applied on the left of the field-major Jacobian it gives the Jacobian of the bounded form up to the
    term ``-R_continuity d(omega)/dw`` on the omega diagonal, which :func:`main` adds separately because
    it is not a row combination.
    """
    _, _, omega = coupled.physical_fields(state)
    rho = float(np.asarray(coupled.turbulence.density))
    pressure = n_fields - 3  # [vel..., p, k, omega]
    cells = np.flatnonzero(transported)
    rows = (n_fields - 1) * n_cells + cells
    columns = pressure * n_cells + cells
    combination = sp.csr_matrix(
        (np.asarray(omega)[cells] / rho, (rows, columns)), shape=(n_fields * n_cells,) * 2
    )
    return sp.identity(n_fields * n_cells, format="csr") - combination


def main():
    index = int(ARGUMENTS[0])
    beta = float(ARGUMENTS[1])
    cells = [int(argument) for argument in ARGUMENTS[2:]]
    data = np.load(CASE / f"checkpoints/state-{index:05d}.npz")
    state = jnp.asarray(data["state"])
    coupled = compare.build_case()["coupled"]
    layout = coupled.layout
    n_fields, n_cells = layout.n_fields, layout.n_cells
    fields = FIELDS[n_fields]
    pressure = fields.index("p")
    omega_field = fields.index("omega")
    wall = np.zeros(n_cells, bool)
    wall[np.asarray(coupled.turbulence.wall_cells)] = True

    terms_at = jax.jit(lambda s: omega_terms(coupled, s))
    residual = np.asarray(coupled.residual(state))
    _, k, omega = (np.asarray(f) for f in coupled.physical_fields(state))
    terms = np.asarray(terms_at(state))
    omega_rows = residual[omega_field * n_cells : (omega_field + 1) * n_cells]
    continuity = residual[pressure * n_cells : (pressure + 1) * n_cells]
    interior = ~wall
    mismatch = np.abs(terms.sum(0) - omega_rows)[interior].max()
    scale = np.abs(terms).max(0)[interior].max()
    print(
        f"state-{index:05d}: {n_cells} cells, shift {beta:g}; {int(wall.sum())} wall-fixed omega rows\n"
        f"term sum against the residual's omega rows (unfixed cells): max |diff| {mismatch:.2e} "
        f"against a largest term of {scale:.2e}",
        flush=True,
    )

    centroids = np.asarray(coupled.momentum.geometry.cell.centroid)
    distance = np.asarray(coupled.turbulence.wall_distance)
    for cell in cells:
        print(
            f"\ncell {cell}: x = {np.array2string(centroids[cell], precision=5)}, wall distance "
            f"{distance[cell]:.3e}, k {k[cell]:.3e}, omega {omega[cell]:.4e} (log {np.log(omega[cell]):.3f}),"
            f" wall-fixed {bool(wall[cell])}"
        )
        print(
            "  terms at the state: "
            + ", ".join(f"{t} {v:+.3e}" for t, v in zip(TERMS, terms[:, cell], strict=True))
        )
        tangents = {
            f: unit(state.size, fields.index(f) * n_cells + cell) for f in fields[: n_fields - 2]
        }
        tangents["omega"] = unit(state.size, omega_field * n_cells + cell)
        full = {
            name: jax.jvp(coupled.residual, (state,), (tangent,))[1]
            for name, tangent in tangents.items()
        }
        print(
            f"  {'d R_omega / d':>16}"
            + "".join(f"{t:>16}" for t in TERMS)
            + f"{'sum':>12}{'residual':>12}"
        )
        for name, tangent in tangents.items():
            parts = np.asarray(jax.jvp(terms_at, (state,), (tangent,))[1][:, cell])
            whole = float(full[name][omega_field * n_cells + cell])
            print(
                f"  {name:>16}"
                + "".join(f"{v:>16.4e}" for v in parts)
                + f"{parts.sum():>12.4e}{whole:>12.4e}"
            )
        dp = full["p"]
        d_continuity = float(dp[pressure * n_cells + cell])
        d_omega = float(dp[omega_field * n_cells + cell])
        rho = float(np.asarray(coupled.turbulence.density))
        bounded = d_omega - omega[cell] / rho * d_continuity
        print(
            f"  pressure: d R_continuity/dp {d_continuity:.4e}; d R_omega/dp {d_omega:.4e} = "
            f"{d_omega / (d_continuity / rho):.4e} x (d R_continuity/dp / rho), against omega_P "
            f"{omega[cell]:.4e}\n  bounded form d R_omega/dp = {bounded:.4e} "
            f"({bounded / d_omega:+.3f} of the conservative)"
        )
        for name in fields[: n_fields - 3]:
            d_c = float(full[name][pressure * n_cells + cell])
            d_o = float(full[name][omega_field * n_cells + cell])
            print(
                f"  {name}: d R_omega/d{name} {d_o:.4e}, bounded {d_o - omega[cell] / rho * d_c:.4e}"
            )

    print("\nblock census, conservative form against the bounded form of the omega row", flush=True)
    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    policy = _coupled_shift_policy(coupled, state, ScalarTwoLevel())
    jacobian = MaterializedJacobianPreconditioner._materialize_jacobian(
        lambda v: jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: batched_jacobian_matvec(coupled, state, seeds),
        PROBE_BATCH_SIZE,
        structure,
    )
    jacobian = sp.csr_matrix(jacobian)
    bounded = bounded_row_operator(coupled, state, n_fields, n_cells, ~wall) @ jacobian
    # The bounded row's omega diagonal also carries -R_continuity * d(omega)/dw = -R_c * omega (log).
    correction = np.zeros(n_fields * n_cells)
    rho = float(np.asarray(coupled.turbulence.density))
    correction[omega_field * n_cells : (omega_field + 1) * n_cells] = np.where(
        wall, 0.0, -continuity * omega / rho
    )
    combined = bounded
    bounded = bounded + sp.diags(correction)
    unfixed = ~wall
    conservative_norm = np.linalg.norm(omega_rows[unfixed])
    carried = np.linalg.norm((omega * continuity / rho)[unfixed])
    print(
        f"omega rows (unfixed cells): |R_omega| {conservative_norm:.4e}, of which omega_P R_continuity / rho "
        f"{carried:.4e}; bounded |R_omega - omega_P R_c / rho| "
        f"{np.linalg.norm((omega_rows - omega * continuity / rho)[unfixed]):.4e}",
        flush=True,
    )
    print(
        f"{'shift':>10} {'form':>13} {'median':>11} {'min':>11} {'below ' + format(SINGULAR_BELOW, 'g'):>11} {'< 1e-2':>8}"
    )
    for shift in (beta, 0.0):
        diagonal = (
            frozen_shift_diagonal(policy, shift, state) if shift else np.zeros(n_cells * n_fields)
        )
        for label, matrix in (
            ("conservative", jacobian),
            ("row combined", combined),
            ("bounded", bounded),
        ):
            cell_major, _, _ = equilibrate_cell_major(
                MaterializedJacobianPreconditioner._shifted(matrix, diagonal), n_fields
            )
            smallest, _ = conditioning(diagonal_blocks(cell_major, n_fields, n_cells))
            print(
                f"{shift:>10.4g} {label:>13} {np.median(smallest):>11.3e} {smallest.min():>11.3e} "
                f"{int((smallest < SINGULAR_BELOW).sum()):>11} {int((smallest < 1e-2).sum()):>8}",
                flush=True,
            )
            if label != "conservative" and shift == beta:
                order = np.argsort(smallest)[:6]
                raw = sp.csr_matrix(matrix).diagonal()[
                    omega_field * n_cells : (omega_field + 1) * n_cells
                ]
                original = jacobian.diagonal()[omega_field * n_cells : (omega_field + 1) * n_cells]
                print(
                    f"   worst {label} blocks (cell: sigma_min, raw omega diagonal before -> after, "
                    "R_continuity, wall-fixed):"
                )
                for c in order:
                    print(
                        f"     {c}: {smallest[c]:.1e}, {original[c]:+.3e} -> {raw[c]:+.3e}, "
                        f"{continuity[c]:+.3e}, {bool(wall[c])}"
                    )


if __name__ == "__main__":
    main()
