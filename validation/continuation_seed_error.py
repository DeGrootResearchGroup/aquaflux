"""How large a Reynolds-continuation step the solution path will tolerate, measured before taking one.

A continuation hands each rung the previous rung's converged fields and asks it to solve at a new
Reynolds number. How big that jump may safely be is the schedule's central question, and the shipped
answer is a constant: one decade per rung, chosen a priori. This measures the quantity that should be
choosing it instead, from residual evaluations alone -- no Jacobian, no linear solve, and no march.

Parameterize the homotopy by the log viscosity scale ``lam = ln(scale)``, so a geometric schedule of
ratio ``r`` takes constant steps ``|dlam| = ln(r)``. Write ``u*`` for a converged state at ``lam0``, so
``R(u*, lam0) = 0``. For a candidate step ``dlam`` two numbers are available for the price of a few
residual evaluations:

``E0(dlam) = || R(u*, lam0 + dlam) ||``
    The seed error the next rung would actually start from -- what the present schedule accepts blind.

``E1(dlam) = || R(u*, lam0 + dlam) - dlam dR/dlam ||``
    What is left of that after the first-order model of the path is subtracted. ``dR/dlam`` is one
    forward-mode derivative in the *parameter*, so this costs nothing extra.

Their ratio ``tau = E1 / E0`` is the fraction of the step the linear model fails to explain -- a
dimensionless, self-normalizing measure of how far the path has departed from its own tangent over that
step. It goes to zero with ``dlam`` and reaches one when the neglected curvature has grown to match the
entire first-order term, which is the point past which the step is no longer a small perturbation of
anything. Being a ratio of two norms of the same measure it carries no units, no mesh size and no
Reynolds number, so a threshold on it is a property of the method rather than of a case.

The practical reading: the present schedule holds ``dlam`` constant, and a schedule that instead held
``tau`` constant would take large steps where the path is straight and small ones where it bends. This
script reports ``tau`` against spacing so the two can be compared on a case.

⚠️ **``tau`` measures the PATH, not the march's ability to follow it.** That a step the tangent explains
poorly is also a step the solver finds hard is the assumption the whole idea rests on; it is not
established by this measurement, and the way to check it is to see whether ``tau`` ranks rungs in the
order the marches actually found difficult.

⚠️ **The state must be converged at ``lam0``.** Everything here expands about a root, and ``E1`` is a
leading-order bound in any case: it is what an exactly-solved tangent predictor would leave, so a real
predictor -- which solves against the unshifted Jacobian, loosely -- can only fall short of it.

Usage -- a case directory, a checkpoint index, the ``lam0`` that checkpoint is converged at, and
optionally the spacings to report as viscosity-scale ratios::

    python3 validation/continuation_seed_error.py bfs3d_openfoam 69 0
    python3 validation/continuation_seed_error.py pitzdaily_openfoam 28 4.60517
    python3 validation/continuation_seed_error.py pitzdaily_openfoam 47 2.302585 10 5 2 1.5

A rung of a ramp built by ``GeometricReynoldsSchedule`` sits at ``lam0 = ln(ratio ** k)`` counting down
from the anchor, so on the default decade ladder the rungs are ``ln(100) = 4.60517``, ``ln(10) =
2.302585`` and ``0`` at the target.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

VALIDATION = Path(__file__).resolve().parent
ROOT = VALIDATION.parent
sys.path.insert(0, str(ROOT))

#: Viscosity-scale ratios to report, as a geometric schedule's ``ratio``. ``10`` is the shipped
#: one-decade spacing; the rest are the finer ladders it would be traded for.
DEFAULT_RATIOS = (10.0, 5.0, 3.1623, 2.0, 1.5, 1.2)


def load_case(case: str):
    """Import a validation case's ``compare`` module and build its assembled coupled system.

    Both cases expose the same two things this needs -- a ``build_case()`` that assembles without
    solving, and a ``checkpoints/`` directory of saved states -- so the probe is written against that
    pair rather than against either case.

    Parameters
    ----------
    case : str
        The case directory name under ``validation/`` (e.g. ``"pitzdaily_openfoam"``).

    Returns
    -------
    tuple
        ``(coupled, case_directory)``.
    """
    directory = VALIDATION / case
    if not directory.is_dir():
        raise SystemExit(f"no such case directory: {directory}")
    sys.path.insert(0, str(directory))
    compare = importlib.import_module("compare")
    return compare.build_case()["coupled"], directory


def judged_norm(coupled, state):
    """The row-equilibrated residual measure at ``state`` -- the one the march stops on.

    Built as a shift source only, since scoring a residual needs the shift diagonal the scales come
    from and not the flow preconditioner a full policy would also construct.
    """
    from aquaflux.turbulence.coupled import _coupled_shift_policy, coupled_scaled_norm

    return coupled_scaled_norm(
        coupled, _coupled_shift_policy(coupled, state, None, build_flow_block=False), state
    )


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: continuation_seed_error.py <case> <checkpoint-index> <lam0> [ratios...]"
        )
    import jax
    import jax.numpy as jnp

    case, index, lam0_value = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    ratios = [float(a) for a in sys.argv[4:]] or list(DEFAULT_RATIOS)

    coupled, directory = load_case(case)
    state = jnp.asarray(np.load(directory / f"checkpoints/state-{index:05d}.npz")["state"])
    lam0 = jnp.asarray(lam0_value)

    def residual_at(lam):
        return coupled.with_scaled_molecular_viscosity(jnp.exp(lam)).residual(state)

    anchor, tangent = jax.jvp(residual_at, (lam0,), (jnp.ones_like(lam0),))
    at_anchor = judged_norm(
        coupled.with_scaled_molecular_viscosity(float(np.exp(lam0_value))), state
    )

    print(
        f"[{case}] checkpoint {index}, {coupled.layout.n_cells} cells, "
        f"lam0 = {lam0_value:.6f} (viscosity scale {np.exp(lam0_value):.4g})"
    )
    print(f"  |R(u*, lam0)| judged = {at_anchor(anchor):.4e}   (0 at a converged root)")
    print()
    print(f"{'ratio':>7} {'dlam':>9} {'E0':>11} {'E1':>11} {'tau=E1/E0':>10}")
    print("-" * 52)
    for ratio in ratios:
        # The ramp travels toward the target, i.e. toward a SMALLER viscosity scale: lam decreases.
        dlam = -float(np.log(ratio))
        measure = judged_norm(
            coupled.with_scaled_molecular_viscosity(float(np.exp(lam0_value + dlam))), state
        )
        zeroth = residual_at(lam0 + dlam)
        e0 = float(measure(zeroth))
        e1 = float(measure(zeroth - dlam * tangent))
        print(f"{ratio:7.3f} {dlam:9.4f} {e0:11.4e} {e1:11.4e} {e1 / e0:10.3f}")


if __name__ == "__main__":
    main()
