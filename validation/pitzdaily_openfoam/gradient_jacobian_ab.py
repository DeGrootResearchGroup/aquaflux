"""How accurate does the gradient reconstruction have to be -- in ``R``, and separately in ``J``?

The coupled march solves each shifted Newton system to a relative residual of ``0.3``: a deliberately
30 %-accurate step, because an inexact-Newton iteration converges to the root of its residual for any
sufficiently accurate operator. Meanwhile the gradient reconstruction inside that residual is solved to
near machine precision, and ``jax.jvp`` differentiates through the same sweep count, so the exact
gradient is paid again on **every matrix-vector product** -- of which a step takes many.

That is only inconsistent if the two accuracies buy the same thing, and they do not:

* the reconstruction inside ``R`` decides **which discrete equations are being solved**. Loosen it and
  the root moves. With a fixed sweep count ``R`` is still an exactly linear, deterministic, history-free
  function of the field, so Newton converges perfectly well -- just to the root of a slightly different
  discretization. On a poor-quality mesh that difference is the answer being wrong.
* the reconstruction inside ``J`` decides only **how fast** the iteration reaches whichever root ``R``
  defines. That is the same latitude the ``0.3`` already takes.

So this case measures the two separately, on the validated benchmark next door, whose case definition,
physics, preconditioner and stopping tolerances it imports rather than restates:

``residual`` arms
    vary the reconstruction's own sweep count and compare the **converged fields** and the reattachment
    length the benchmark is judged by. This prices the accuracy of the residual's gradient directly --
    the question "what is all that exactness buying?" -- rather than by the reconstruction error, which
    says nothing about the answer. ⚠️ Each arm's probing reach moves with its sweep count, because the
    sweeps are what carry the residual's stencil across the cell graph and a probe shorter than that
    stencil folds the far coupling onto near entries instead of capturing it. So these arms differ in
    their preconditioner too and their **wall clocks are not comparable**; their converged fields are.

``jacobian`` arms
    hold the residual fixed and cap the sweeps only in the copy the **Jacobian** is differentiated from
    (``jacobian_gradient_sweeps``). Every arm therefore solves the identical discrete problem and must
    land on the identical root; what can differ is the cost of getting there. Judge these on outer
    steps and cumulative Krylov cycles first and wall clock second -- an operator too far from the true
    Jacobian costs more iterations than the cheaper product saves, and the cheaper product is only worth
    having if the iteration count holds.

The adjoint is untouched by either. The implicit-function-theorem reverse rule differentiates the
residual it was handed at the converged state, without consulting the forward step, so a sensitivity
stays exact however approximate the operator that marched to that state.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/gradient_jacobian_ab.py

    PITZ_AB_ARMS=jacobian validation/run_case.sh validation/pitzdaily_openfoam/gradient_jacobian_ab.py

Every arm runs in **one process**, back to back on one machine, so the wall clocks within a group are
comparable to whatever this machine's noise floor allows -- N invocations would not be.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402  (the validated benchmark: mesh, physics, preconditioner, metric)
import numpy as np  # noqa: E402
from aquaflux.schemes import CorrectedGreenGauss, SweptGradientSolve  # noqa: E402

#: Which groups to run: ``residual``, ``jacobian``, or both (comma-separated).
GROUPS = tuple(os.environ.get("PITZ_AB_ARMS", "jacobian,residual").split(","))

#: ``(label, residual sweeps, probe reach, jacobian sweeps)``.
#:
#: The **jacobian** group is the point of the case and holds the residual at the shipped ``4`` and the
#: probe at the shipped ``5``, varying only the copy the Krylov operator differentiates. Its first arm
#: is the control: ``None`` means the exact Jacobian of the residual, i.e. the case exactly as it ships.
#:
#: The **residual** group varies the discretization, so each arm's reach moves with its sweep count.
#: On this mesh the pressure column reaches ``sweeps + 1`` and the velocity columns ``sweeps + 2`` (the
#: eddy viscosity's strain-rate dependence spends a ring the gradient does not), so the reach is set to
#: ``sweeps + 2`` throughout -- which is the shipped ``5`` at the shipped ``4``, and is why the control
#: arm is shared between the groups.
ARMS = {
    "jacobian": (
        ("J full (control)", 4, 5, None),
        ("J swept-2", 4, 5, 2),
        ("J swept-1", 4, 5, 1),
    ),
    "residual": (
        ("R swept-4 (control)", 4, 5, None),
        ("R swept-2", 2, 4, None),
        ("R swept-6", 6, 7, None),
    ),
}

#: Compared between the arms. ``nut`` is included because a turbulence case feels a gradient change most
#: in the eddy viscosity, which is built from velocity gradients.
FIELDS = ("U", "p", "k", "omega", "nut")

#: A row of the march log's outer summary table, ``| step | t(s) | beta | in | cyc | R | a_min | flg |``
#: -- captured as ``(step, cyc)``.
#:
#: ⚠️ Anchored at the line start and matched on the whole row, because the log carries a second,
#: indented per-inner table of the same broad shape. A looser pattern matches both and silently reads
#: the outer table's ``t(s)`` column as a cycle count -- a plausible-looking number that grows with the
#: march.
_STEP_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*[\d.eE+-]+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
    re.MULTILINE,
)


def march_cost(log_path):
    """``(steps, cumulative Krylov cycles, worst step)`` read back out of one arm's march log."""
    rows = _STEP_ROW.findall(Path(log_path).read_text())
    cycles = [int(c) for _, _, c in rows]
    return len(rows), sum(cycles), (max(cycles) if cycles else 0)


def solve_arm(label, residual_sweeps, reach, jacobian_sweeps, log_path):
    """March one arm to convergence and return its fields alongside what it cost."""
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=residual_sweeps))
    started = time.perf_counter()
    fields = compare.solve_aquaflux(
        log_path=log_path,
        gradient_scheme=scheme,
        stencil_reach=reach,
        jacobian_gradient_sweeps=jacobian_sweeps,
    )
    elapsed = time.perf_counter() - started
    steps, cycles, worst = march_cost(log_path)
    return dict(
        label=label,
        wall=elapsed,
        steps=steps,
        cycles=cycles,
        worst=worst,
        xr=compare.reattachment_length(fields["centroid"], fields["U"][:, 0]),
        **{name: fields[name] for name in FIELDS},
    )


def relative_difference(a, b):
    """``(relative L2, relative max)`` of ``a - b``, each normalized by ``b``'s own magnitude.

    Normalized so fields spanning many orders -- a pressure against an omega -- read off one table.
    """
    scale = float(np.linalg.norm(b)) or 1.0
    peak = float(np.abs(b).max()) or 1.0
    return float(np.linalg.norm(a - b)) / scale, float(np.abs(a - b).max()) / peak


def report(group, results):
    control = results[0]
    print(f"\n{'=' * 78}\n{group} arms -- against '{control['label']}'\n{'=' * 78}", flush=True)
    print(
        f"{'arm':<22} {'steps':>6} {'cycles':>7} {'worst':>6} {'wall (s)':>9} {'x_r/h':>7}",
        flush=True,
    )
    for r in results:
        print(
            f"{r['label']:<22} {r['steps']:>6} {r['cycles']:>7} {r['worst']:>6} "
            f"{r['wall']:>9.1f} {r['xr']:>7.3f}",
            flush=True,
        )
    print(
        f"\n{'field difference against the control (relative L2 / relative max)':<78}", flush=True
    )
    print(f"{'arm':<22} " + " ".join(f"{n:>21}" for n in FIELDS), flush=True)
    for r in results[1:]:
        cells = []
        for name in FIELDS:
            l2, peak = relative_difference(r[name], control[name])
            cells.append(f"{l2:>9.2e} /{peak:>9.2e}")
        print(f"{r['label']:<22} " + " ".join(cells), flush=True)


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # State what every number below was taken under, in the run's own output, before any result: a
    # measurement whose configuration is not written beside it cannot be re-adjudicated later.
    print(f"[configuration] {stamp}", flush=True)
    print(f"  host ILU kernel: {'compiled' if compare.ILU0_COMPILED else 'PURE PYTHON (void)'}")
    # ⚠️ NOT the case's shipped `petsc` default, and that is deliberate rather than a preference: on
    # this tree the shipped bundle collapses at the first step of the second Reynolds rung (alpha 0,
    # beta escalating to the ladder's top, residual to `inf`), reproducibly and independently of
    # anything measured here. `simplesmooth` marches the same case to the same reattachment length in
    # comparable wall clock, so it is what these arms are compared on. Every arm uses it, so the
    # comparison between them is unaffected -- but a number here is not comparable to one taken on the
    # shipped bundle.
    print(f"  flow (leading) inverse: {compare.FLOW_INVERSE}")
    print(f"  Reynolds continuation points: {compare.N_POINTS}")
    print(
        f"  forward rtol (row-scaled) / restart: {compare.FORWARD_RTOL} / {compare.FORWARD_RESTART}"
    )
    print(f"  dual-time inner steps / tol: {compare.INNER_STEPS} / {compare.INNER_TOL}")
    print(f"  stop (rtol, atol): {compare.RTOL}, {compare.ATOL}", flush=True)

    for group in GROUPS:
        if group not in ARMS:
            raise SystemExit(f"unknown arm group {group!r}; expected one of {sorted(ARMS)}")
        results = []
        for label, residual_sweeps, reach, jacobian_sweeps in ARMS[group]:
            log_path = HERE / f"runs/ab-{stamp}-{group}-{label.split()[1]}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"\n[{group}: {label}] residual swept-{residual_sweeps}, probe reach {reach}, "
                f"jacobian {jacobian_sweeps or 'full (exact)'} -> {log_path.name}",
                flush=True,
            )
            results.append(solve_arm(label, residual_sweeps, reach, jacobian_sweeps, log_path))
            r = results[-1]
            print(
                f"  {r['steps']} steps, {r['cycles']} cycles, {r['wall']:.1f} s, "
                f"x_r/h {r['xr']:.3f}",
                flush=True,
            )
        report(group, results)


if __name__ == "__main__":
    main()
