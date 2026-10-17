"""A census of the compiled coupled residual: what the CPU backend turns the residual into.

Two performance defects in this package were invisible to every test and plain in the compiled
program's text. Both are properties of how XLA (the compiler JAX lowers to) emits the residual on its
CPU backend, so this module compiles the coupled k-omega SST residual and its Jacobian-vector product
(JVP) -- the program every Krylov iteration and every coloured Jacobian probe runs -- and counts three
kinds of instruction in the optimized HLO (XLA's high-level operation graph) text:

* **Library factorizations.** A batched ``jnp.linalg.inv`` or ``solve`` of the per-cell ``(dim, dim)``
  blocks lowers to LAPACK (Linear Algebra PACKage) custom calls -- ``lapack_dgetrf_ffi`` and
  ``lapack_dtrsm_ffi`` -- that XLA cannot fuse into the kernels around them. One such call inside the
  multiple-correction gradient's first pass cost most of a pitzDaily residual evaluation before it was
  replaced by the closed-form adjugate (:func:`aquaflux.schemes.gradient.small_inverse`). Per-evaluation
  code must carry none, so the count is pinned at zero.
* **Custom fusions** (``kind=kCustom``). On this backend every reduction lowers to one; nothing else in
  the residual does today.
* **Single-threaded custom fusions** -- those whose ``outer_dimension_partitions`` is empty, so the
  kernel runs on one thread while the loop fusions around it are split across cores. A dot product
  written ``jnp.sum(a * b, axis=-1)`` lowers to exactly this; :func:`aquaflux.vectors.dot` multiplies and
  then sums the components explicitly so that it does not.

Why counts and not wall time
----------------------------
A timing on a shared machine, or a four-core continuous-integration runner, moves with whatever else
is running, while the instruction mix of a compiled program does not. The count is also the cause,
where a time is only its symptom: the defects above were found by reading these lines, and a regression
of either kind adds instructions of exactly these kinds. The counts are ratchets, pinned at the values
measured below, so any move fails:

* **A count went up.** A per-evaluation path has gained a factorization or a single-threaded
  reduction. Find it by diffing the compiled text before and after the change (the ``metadata=`` field
  of each instruction names the source operation); if it is deliberate, measure what it costs on a real
  case before raising the number here.
* **A count went down.** Something removed one; lower the number to the new value, so the improvement
  is kept.
* **A count moved with no change to the package.** A JAX or XLA release changed how the backend lowers
  the same program. Read the differing lines before re-pinning; a new release that emits more
  single-threaded kernels is worth knowing about rather than absorbing.

Configuration the numbers were measured under
---------------------------------------------
jax and jaxlib 0.11.2, CPU backend, x64, Linux x86_64, four cores, 2026-10-17. Identical with the fast
tier's thread pinning (``OMP_NUM_THREADS=1``, ``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false``) and with
the process restricted to one or two cores. **The mesh size is load-bearing:** XLA partitions a kernel
and lowers a reduction to a custom fusion only above some size, and on a 768-cell grid neither happens
whatever the source says -- the ``dot`` mutation below changed no count there. At 60 x 40 cells both
classes are visible, and :func:`test_the_fixture_is_large_enough_for_the_backend_to_partition` fails if
a backend change makes the fixture too small to see them.

Each assertion was checked against a broken package (each break reverted exactly):

* ``_boundary_condition_first_pass`` in ``schemes/multiple_correction.py`` inverting with
  ``jnp.linalg.inv`` instead of ``small_inverse``: 15 LAPACK custom calls in each program, so the
  LAPACK test fails.
* ``vectors.dot`` written as ``jnp.sum(jnp.multiply(a, b), axis=-1)``: custom fusions 2 -> 27 in the
  residual and 3 -> 46 in the JVP, all single-threaded, so both ratchets fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.case import CaseFile, case_spec_from_mapping

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="the compiled program counted here is the CPU backend's",
)

REPO = Path(__file__).resolve().parents[2]

#: pitzDaily's shipped numerics on a structured channel: limited upwind momentum advection, first-order
#: k and omega, the multiple-correction gradient, log omega and the frozen production limiter. The size
#: is set by what the backend partitions, not by the physics (see the module docstring).
SECTIONS = {
    "mesh": {"kind": "StructuredGrid", "cells": [60, 40], "lengths": [2.0, 1.0]},
    "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-5},
    "physics": {
        "kind": "RANS",
        "omega_variable": {"kind": "LogScalars"},
        "explicit_production_limiter": True,
    },
    "boundaries": {
        "left": {
            "kind": "Inlet",
            "velocity": [10.0, 0.0],
            "turbulence": {"kind": "FixedTurbulence", "k": 0.375, "omega": 440.15},
        },
        "right": {"kind": "Outlet", "pressure": 0.0},
        "bottom": {"kind": "Wall", "k": "zero_gradient"},
        "top": {"kind": "Wall", "k": "zero_gradient"},
    },
    "numerics": {
        "momentum_advection": {
            "kind": "LimitedUpwind",
            "limiter": {"kind": "VenkatakrishnanLimiter"},
        },
        "turbulence_advection": {"kind": "FirstOrderUpwind"},
        "gradient": {"kind": "MultipleCorrectionGradient"},
    },
}

#: The ratchets: program -> (custom fusions, single-threaded custom fusions). See the module docstring
#: for the configuration and for what to do when one moves.
RATCHETS = {"residual": (2, 2), "jvp": (3, 3)}

_PARTITIONS_KEY = '"outer_dimension_partitions"'
_UNPARTITIONED = '"outer_dimension_partitions":[]'


@dataclass(frozen=True)
class ProgramCensus:
    """The instruction counts of one compiled program that this module pins."""

    lapack_calls: int
    custom_fusions: int
    single_threaded_custom_fusions: int
    partitioned_loop_fusions: int


def census(hlo_text: str) -> ProgramCensus:
    """Count the instructions of interest in an optimized HLO module's text.

    Parameters
    ----------
    hlo_text : str
        What ``jax.jit(f).lower(*args).compile().as_text()`` returns.

    Returns
    -------
    ProgramCensus
        One count per kind of instruction, each read line by line.
    """
    lines = hlo_text.splitlines()
    custom = [line for line in lines if "kind=kCustom" in line]
    return ProgramCensus(
        lapack_calls=sum(
            1 for line in lines if "custom-call(" in line and re.search(r'"lapack_\w+"', line)
        ),
        custom_fusions=len(custom),
        single_threaded_custom_fusions=sum(1 for line in custom if _UNPARTITIONED in line),
        partitioned_loop_fusions=sum(
            1
            for line in lines
            if "kind=kLoop" in line and _PARTITIONS_KEY in line and _UNPARTITIONED not in line
        ),
    )


def _compiled_text(function, *args) -> str:
    return jax.jit(function).lower(*args).compile().as_text()


@pytest.fixture(scope="module")
def programs() -> dict[str, ProgramCensus]:
    """The census of the coupled residual and of its JVP, each compiled once for the module."""
    coupled = CaseFile(case_spec_from_mapping(SECTIONS), REPO).check().build()
    flow_size, n_cells, _ = coupled.layout.sizes
    # Only the shapes reach the compiled program; the values are a positive, plausible state.
    state = coupled.state_from_physical(
        jnp.full(flow_size, 0.1), jnp.full(n_cells, 0.05), jnp.full(n_cells, 10.0)
    )
    # A plain `jax.jit` over the module's arrays: an `eqx.filter_jit` program has no `as_text`.
    arrays, static = eqx.partition(coupled, eqx.is_array)

    def residual(arrays, state):
        return eqx.combine(arrays, static).residual(state)

    def jvp(arrays, state, tangent):
        return jax.jvp(lambda x: residual(arrays, x), (state,), (tangent,))[1]

    return {
        "residual": census(_compiled_text(residual, arrays, state)),
        "jvp": census(_compiled_text(jvp, arrays, state, jnp.ones_like(state))),
    }


@pytest.mark.parametrize("program", sorted(RATCHETS))
def test_no_library_factorization_is_compiled_into_the_residual(programs, program) -> None:
    """A LAPACK call per evaluation cannot be fused, and on pitzDaily one cost most of the residual."""
    assert programs[program].lapack_calls == 0, (
        f"the compiled {program} calls LAPACK {programs[program].lapack_calls} times; a per-cell "
        "block inverse on the per-evaluation path should use `schemes.gradient.small_inverse`"
    )


def _ratchet(what: str, measured: int, pinned: int) -> None:
    assert measured <= pinned, (
        f"{what}: {measured}, up from {pinned}. A per-evaluation path has gained one; diff the "
        "compiled text to find it (see this module's docstring)"
    )
    assert measured >= pinned, (
        f"{what}: {measured}, down from {pinned}. Lower the ratchet in RATCHETS to {measured}"
    )


@pytest.mark.parametrize("program", sorted(RATCHETS))
def test_the_custom_fusion_count_does_not_rise(programs, program) -> None:
    _ratchet(
        f"custom fusions in the {program}", programs[program].custom_fusions, RATCHETS[program][0]
    )


@pytest.mark.parametrize("program", sorted(RATCHETS))
def test_the_single_threaded_reduction_count_does_not_rise(programs, program) -> None:
    """The class a reduction-form ``dot`` falls into: a kernel on one thread among partitioned ones."""
    _ratchet(
        f"single-threaded custom fusions in the {program}",
        programs[program].single_threaded_custom_fusions,
        RATCHETS[program][1],
    )


@pytest.mark.parametrize("program", sorted(RATCHETS))
def test_the_fixture_is_large_enough_for_the_backend_to_partition(programs, program) -> None:
    """Below some size XLA partitions nothing, and the ratchets above could then never fail.

    The single-threaded count means something only beside kernels the backend did thread. If this
    fails, the census no longer sees the class it guards: enlarge the mesh until it passes and the
    ``dot`` mutation in the module docstring moves the count again, then re-pin.
    """
    assert programs[program].partitioned_loop_fusions > 0


def test_the_census_reads_each_kind_of_line_it_is_shown() -> None:
    """The counts above are only as good as the line matching, so the matching is pinned directly.

    Synthetic lines in the compiled text's own format: each kind counted where it appears, and the
    near misses -- a partitioned custom fusion, an unpartitioned loop fusion, a custom call to another
    library -- not counted.
    """
    text = "\n".join(
        [
            '  %custom-call.1 = f64[4,2,2]{2,1,0} custom-call(%a), custom_call_target="lapack_dgetrf_ffi"',
            '  %custom-call.2 = f64[4]{0} custom-call(%b), custom_call_target="other_library_call"',
            '  %ynn_fusion = f64[9]{0} fusion(%c), kind=kCustom, calls=%f, backend_config={"fusion_config":'
            '{"kind":"__ynn_fusion"},"outer_dimension_partitions":[]}',
            '  %ynn_fusion.1 = f64[9]{0} fusion(%c), kind=kCustom, calls=%g, backend_config={"fusion_config":'
            '{"kind":"__ynn_fusion"},"outer_dimension_partitions":["2"]}',
            '  %loop_fusion = f64[9]{0} fusion(%d), kind=kLoop, calls=%h, backend_config={"outer_dimension_partitions":["4"]}',
            "  %loop_fusion.1 = f64[9]{0} fusion(%d), kind=kLoop, calls=%i",
        ]
    )
    assert census(text) == ProgramCensus(
        lapack_calls=1,
        custom_fusions=2,
        single_threaded_custom_fusions=1,
        partitioned_loop_fusions=1,
    )
