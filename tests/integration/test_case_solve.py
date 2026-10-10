"""A case runs from its file alone: read, checked, built and solved with no settings in code.

The unit tests pin what a case hands each library solve; this pins that the handover produces a
solve -- that a file stating no solver, or one with settings, reaches a converged root, and the root
the library solve reaches when it is called with the same settings directly.
"""

from __future__ import annotations

import dataclasses

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from aquaflux.case import case_spec_from_mapping, read_case
from aquaflux.flow import solve_flow_march
from aquaflux.solve import Convergence, DualTimeLoop
from aquaflux.turbulence import sst_initial_fields

#: A laminar channel at a Reynolds number of about 50, fed a uniform velocity at its left side.
_CHANNEL = """
mesh: {kind: StructuredGrid, cells: [8, 4], lengths: [2.0, 1.0]}
fluid: {density: 1.0, kinematic_viscosity: 2.0e-2}
physics: {kind: Laminar}
boundaries:
  left: {kind: Inlet, velocity: [1.0, 0.0]}
  right: {kind: Outlet, pressure: 0.0}
  bottom: {kind: Wall}
  top: {kind: Wall}
numerics:
  momentum_advection: {kind: FirstOrderUpwind}
"""

#: A RANS channel fed at its left side, solved by the segregated loop: an unconstrained flow, so its
#: flow solve is :func:`~aquaflux.flow.reused_flow_solve` rather than the bulk-velocity one.
_SEGREGATED_RANS_CHANNEL = """
mesh: {kind: StructuredGrid, cells: [8, 4], lengths: [2.0, 1.0]}
fluid: {density: 1.0, kinematic_viscosity: 1.0e-3}
physics: {kind: RANS}
boundaries:
  left:
    kind: Inlet
    velocity: [1.0, 0.0]
    turbulence: {kind: FixedTurbulence, k: 1.0e-3, omega: 1.0}
  right: {kind: Outlet, pressure: 0.0}
  bottom: {kind: Wall}
  top: {kind: Wall}
numerics:
  momentum_advection: {kind: FirstOrderUpwind}
  turbulence_advection: {kind: FirstOrderUpwind}
solver: {kind: Segregated, sweeps: 60}
"""

#: A dual-time march at a tight absolute stop, as a file states it.
_SOLVER = {
    "kind": "FlowMarch",
    "max_steps": 80,
    "convergence": {"kind": "Convergence", "rtol": 0.0, "atol": 1e-10},
    "dual_time": {"kind": "DualTimeLoop", "inner_steps": 3},
}


@pytest.fixture(scope="module")
def channel(tmp_path_factory):
    """The channel's file, checked and built once: the solver section changes the solve, not the problem."""
    path = tmp_path_factory.mktemp("case") / "case.yaml"
    path.write_text(_CHANNEL)
    checked = read_case(path).check()
    return checked, checked.build()


def test_a_laminar_case_stating_no_solver_marches_to_a_root_from_its_file(channel) -> None:
    checked, problem = channel
    assert checked.spec.solver is None
    state = checked.solve(problem)
    assert float(jnp.linalg.norm(problem.residual(state))) < 1e-8
    velocity, _ = problem.unpack(state)
    # A developed channel carries its inflow through and is fastest on the centreline: not the seed.
    assert float(jnp.max(velocity[:, 0])) > 1.1
    np.testing.assert_array_equal(np.asarray(state), np.asarray(solve_flow_march(problem)))


def test_a_solver_section_reaches_the_march_it_describes(channel) -> None:
    """The file's settings, not the defaults, are what ran: the root is the one they give directly."""
    checked, problem = channel
    document = yaml.safe_load(_CHANNEL) | {"solver": _SOLVER}
    stated = dataclasses.replace(checked, spec=case_spec_from_mapping(document))
    state = stated.solve(problem)
    np.testing.assert_array_equal(
        np.asarray(state),
        np.asarray(
            solve_flow_march(
                problem,
                max_steps=80,
                convergence=Convergence(rtol=0.0, atol=1e-10),
                dual_time=DualTimeLoop(inner_steps=3),
            )
        ),
    )
    # Both are roots, reached by different marches, so they agree to the stop and not to the bit.
    default = checked.solve(problem)
    assert not np.array_equal(np.asarray(default), np.asarray(state))
    np.testing.assert_allclose(np.asarray(default), np.asarray(state), rtol=0, atol=1e-6)


@pytest.mark.slow
def test_an_inlet_driven_rans_case_runs_the_segregated_loop_from_its_file(tmp_path) -> None:
    """The segregated loop calls its flow solve as ``(momentum, state) -> (momentum, state)``, and a
    case hands it the library's own builder for an unconstrained flow, with no adapter between them.

    The unit tests replace the loop with a recorder, so they cannot see whether the builder they
    check is handed over and what the loop then does with its result; this runs both. A flow solve
    returning the state alone fails on the loop's first sweep, so reaching a root at all is the check.
    """
    path = tmp_path / "case.yaml"
    path.write_text(_SEGREGATED_RANS_CHANNEL)
    checked = read_case(path).check()
    problem = checked.build()

    flow, k, omega = checked.solve(problem)

    start = problem.pack_state(*sst_initial_fields(problem.momentum, problem.turbulence))
    residual = float(jnp.linalg.norm(problem.residual(problem.pack_state(flow, k, omega))))
    assert residual < 1e-3 * float(jnp.linalg.norm(problem.residual(start)))
