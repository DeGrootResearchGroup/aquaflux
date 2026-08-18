"""Validation: a passive tracer on the 3D backward-facing step, against OpenFOAM.

The test tier's view of ``validation/bfs3d_species``. That case's data -- the OpenFOAM mesh, its
converged flow, and the reference tracer field -- is generated locally and is not in the repository,
so every test here **skips** when it is absent rather than failing. What they add over running the
case by hand is that the properties the comparison rests on are asserted rather than eyeballed in a
log: if the injection stops being sub-patch, or the tracer stops being conservative on the imported
flux, the case's numbers are meaningless and these say so.

The same-flux arm is the one exercised. It needs only the reference's ``phi`` and ``nut`` -- not the
OpenFOAM tracer run, and not an aquaflux flow solve -- so it is the part of the case that can be
checked without an hour of compute.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[2] / "validation"
SPECIES = VALIDATION / "bfs3d_species"
OF_FLOW = VALIDATION / "bfs3d_openfoam" / "of_case"

pytestmark = pytest.mark.validation

_REQUIRED = (
    OF_FLOW / "constant" / "polyMesh" / "owner",
    OF_FLOW / "2000" / "phi",
    OF_FLOW / "2000" / "nut",
)


def _case():
    """The case module, imported by path (two ``compare`` modules exist under ``validation/``)."""
    for path in (VALIDATION / "bfs3d_openfoam", SPECIES):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("bfs3d_species_case", SPECIES / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def same_flux():
    """The same-flux arm: aquaflux's tracer on the reference's own flux and eddy viscosity."""
    missing = [p for p in _REQUIRED if not p.exists()]
    if missing:
        pytest.skip(f"bfs3d case data not generated: {missing[0].name} is absent")
    case = _case()
    mesh = case.read_openfoam(OF_FLOW)
    geometry = mesh.geometry()
    flux, nut = case.openfoam_flux_and_nut(mesh)
    transport, diffusivity = case.build_transport(mesh, geometry, nut)
    values = np.asarray(case.solve_tracer(transport, mesh, geometry, flux, diffusivity))
    return case, mesh, geometry, flux, transport, values


def test_the_injection_is_sub_patch(same_flux) -> None:
    """The tracer enters over PART of the inlet -- which is what makes this a mixing problem.

    A uniform inlet is reproduced by any scheme and would measure nothing, so this pins the
    property the whole case rests on: a minority of the inlet's flux carries tracer, and the
    injected band is neither the whole patch nor a single face.
    """
    case, mesh, geometry, flux, _transport, _values = same_flux
    indices = np.asarray(mesh.face_patches.indices("inlet"))
    centroids = np.asarray(geometry.face.centroid)[indices]
    injected = np.asarray(case.injected_value(centroids))

    assert injected.max() == pytest.approx(1.0, abs=1e-6)
    assert (injected > 1e-6).sum() < 0.5 * injected.size  # genuinely part of the patch
    assert (injected > 1e-6).sum() > 10  # and genuinely resolved, not one face

    carried = case.injected_throughput(mesh, geometry, flux)
    total = abs(float(np.sum(np.asarray(flux)[indices])))
    assert 0.02 < carried / total < 0.30


def test_the_tracer_is_conservative_on_the_imported_flux(same_flux) -> None:
    """What enters leaves, to solver tolerance -- the property the imported flux exists to give.

    Summing the converged residual telescopes every interior face away, so what survives is the net
    boundary flux, which must vanish at steady state with no source. It holds only because the
    scalar rides OpenFOAM's own ``phi``, on which OpenFOAM's continuity closes; on a flux rebuilt
    from cell velocities this is the assertion that would fail.
    """
    import jax.numpy as jnp

    case, mesh, geometry, flux, transport, values = same_flux

    net = case.conservation_error(transport, flux, jnp.asarray(values))
    injected = case.injected_throughput(mesh, geometry, flux)
    assert net < 1e-6 * abs(injected)


def test_the_tracer_stays_bounded(same_flux) -> None:
    """A concentration cannot leave the range imposed on the boundary by more than the limiter's slack.

    The injected value is 1 and nothing is removed anywhere, so the field belongs in ``[0, 1]``. A
    limited second-order scheme is not formally bounded, so a small undershoot is allowed and a
    large one is a defect -- an unbounded tracer would make every mixing number downstream fiction.
    """
    _case, _mesh, _geometry, _flux, _transport, values = same_flux
    assert values.min() > -1e-3
    assert values.max() < 1.0 + 1e-6


def test_the_plume_mixes_downstream(same_flux) -> None:
    """Unmixedness falls with distance -- the case measures mixing, so mixing must be happening.

    Normalized variance in a slab: 1 for a completely segregated stream, 0 once uniform. If it did
    not fall, the tracer would be passing through without interacting with the recirculation and
    there would be nothing for the two codes to disagree about.
    """
    case, _mesh, geometry, _flux, _transport, values = same_flux
    centroid = np.asarray(geometry.cell.centroid)
    volume = np.asarray(geometry.cell.volume)

    _near_mean, near = case.slab_profile(centroid, volume, values, 1.0)
    _far_mean, far = case.slab_profile(centroid, volume, values, 16.0)

    assert np.isfinite(near) and np.isfinite(far)
    assert far < near


def test_it_agrees_with_openfoam(same_flux) -> None:
    """Cellwise agreement with the reference, on the identical flux.

    Skipped until the OpenFOAM tracer run exists. The bar is deliberately loose: mesh, flux,
    diffusivity and boundary values are common to both codes, so what is left is the scalar
    discretization -- nominally the same order in each, not the same scheme -- and the case's job is
    to report that difference, not to assert it away. A failure here means something structural
    (a placement, a diffusivity convention, a boundary closure), not a scheme difference.
    """
    case, _mesh, _geometry, _flux, _transport, values = same_flux
    reference = case.OF_SPECIES / "s"
    if not reference.exists():
        pytest.skip("OpenFOAM tracer run not present (of_case/run_of.sh)")

    of_values = case.read_volume_scalar_field(reference, _mesh)
    delta = values - np.asarray(of_values)
    assert np.sqrt(np.mean(delta**2)) < 0.05
