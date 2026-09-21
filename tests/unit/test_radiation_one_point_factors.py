"""The two live factors the transfer evaluates at one point per pair, and what bounds each.

The full sweeps live in ``validation/radiation_one_point_factors.py``. What is here is the
cheap half those numbers rest on: the two controls that must read exactly zero, the mechanism
that predicts the absorption bias, the order each bias carries in the facet's extent, and that
the shipped build really does apply absorption the way the bound assumes. A change to the
transfer cannot quietly invalidate the recorded bounds without one of these failing.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.self_occlusion import NoOcclusion
from aquaflux.radiation.solid_angle import solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer

from tests.unit.radiation_references import (
    _pair_kernel,
    _source_samples,
    absorption_bias,
    centroid_separation,
    mean_separation_excess,
    profile_bias,
    profile_cumulant_bias,
    unit_facet,
)

#: Coarser than the harness's 24, and checked against it below: these assertions are about
#: orders and signs, not about the fourth figure, and the pair kernel is quadratic in this.
COARSE = 12

SOURCE = unit_facet([0.0, 0.0, 0.0], 1.0)


def _receiver(gap: float, offset: float = 0.0) -> np.ndarray:
    return unit_facet([offset, 0.0, gap], 1.0, facing_down=True)


def _point(distance: float, degrees: float = 0.0) -> np.ndarray:
    radians = np.radians(degrees)
    return distance * np.array([np.sin(radians), 0.0, np.cos(radians)])


@pytest.mark.parametrize("gap", [1.0, 2.0, 4.0])
def test_nothing_absorbing_leaves_no_absorption_bias(gap):
    """The control. With ``a = 0`` both sides are the bare kernel and the ratio is exactly one.

    Without this the sweep's numbers could be measuring the sampling, the clamp, or the
    geometry helper rather than the absorption, and nothing in the table would say so.
    """
    assert absorption_bias(SOURCE, _receiver(gap), 0.0, COARSE) == 1.0


@pytest.mark.parametrize("distance", [1.0, 2.0, 8.0])
@pytest.mark.parametrize("degrees", [0.0, 45.0])
def test_a_lambertian_source_has_no_profile_bias_at_any_geometry(distance, degrees):
    """The other control, and it is exact rather than small.

    A Lambertian radiance is constant over direction, so evaluating it at the centroid direction
    is evaluating a constant and the one-point form is not an approximation at all. It holds at
    every distance and every angle, which is what makes the non-Lambertian numbers attributable
    to the profile and not to the geometry they were measured on.
    """
    assert profile_bias(SOURCE, _point(distance, degrees), 1.0, COARSE) == pytest.approx(1.0)


@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("offset", [0.0, 0.5, 1.0, 2.0, 4.0])
def test_the_mean_separation_excess_predicts_the_absorption_bias(gap, offset):
    """The mechanism, not just the magnitude: ``bias -> -a (<r> - r_centroid)`` as ``a -> 0``.

    This is what licenses stating the bound in terms of a facet's extent rather than quoting a
    table, and it is asserted over the whole geometry family rather than the axial corner --
    including the offsets where the excess is *negative*, because those are the ones that would
    expose the prediction as an axial coincidence if it were one. If the leading term were the
    Jensen correction on ``exp`` the bias would be second order in ``a`` and this slope would
    come out zero.
    """
    receiver = _receiver(gap, offset)
    slope = (absorption_bias(SOURCE, receiver, 1e-3, COARSE) - 1.0) / 1e-3
    excess = mean_separation_excess(SOURCE, receiver, COARSE)
    # An absolute floor as well as a relative one: the family crosses zero between offsets, and
    # a purely relative tolerance on a quantity passing through zero asks for infinite precision
    # at the crossing rather than accuracy.
    assert slope == pytest.approx(-excess, rel=2e-3, abs=1e-4)


def test_the_mean_separation_excess_changes_sign_with_lateral_offset():
    """Head-on it is positive; slid sideways it goes negative, and that IS the bias sign flip.

    The naive reading -- that the distance between two mean positions can never exceed the mean
    of the distances, so the stored number is always the shorter one -- is Jensen's inequality
    about the *kernel-weighted* mean positions, not about the geometric centroids the build
    stores. This pins the difference, which an earlier version of these findings asserted away.
    """
    assert mean_separation_excess(SOURCE, _receiver(1.0, 0.0), COARSE) > 0.0
    assert mean_separation_excess(SOURCE, _receiver(1.0, 4.0), COARSE) < 0.0


def test_the_absorption_bias_is_first_order_in_the_absorbance_across_a_facet():
    """Tripling ``a * w`` roughly triples the bias. A second-order law would multiply it by nine.

    Pinned as a band rather than a number: the point is the ORDER, and an assertion tight enough
    to name the coefficient would fail on a harmless change to the sampling.
    """
    receiver = _receiver(1.0)
    small = abs(absorption_bias(SOURCE, receiver, 0.1, COARSE) - 1.0)
    large = abs(absorption_bias(SOURCE, receiver, 0.3, COARSE) - 1.0)
    assert 2.5 < large / small < 3.5


def test_the_profile_bias_is_second_order_in_the_angular_width():
    """Doubling the distance quarters the bias. A first-order law would halve it.

    The two factors therefore need different rules of thumb, which is the practical reason for
    establishing each order separately rather than quoting one bound for both.
    """
    near = abs(profile_bias(SOURCE, _point(2.0), 8.0, COARSE) - 1.0)
    far = abs(profile_bias(SOURCE, _point(4.0), 8.0, COARSE) - 1.0)
    assert 3.0 < near / far < 5.0


@pytest.mark.parametrize("exponent", [2.0, 4.0, 8.0])
def test_two_frozen_moments_cut_the_profile_bias_by_orders(exponent):
    """The candidate frozen/live split works, and by how much is the reason to record it.

    Freezing the solid-angle-weighted mean and variance of ``log cos`` per pair -- two numbers,
    not one per sample -- leaves the exponent live and removes most of the bias. This pins that
    it is a large improvement, not a marginal one.
    """
    point = _point(2.0)
    shipped = abs(profile_bias(SOURCE, point, exponent, COARSE) - 1.0)
    corrected = abs(profile_cumulant_bias(SOURCE, point, exponent, COARSE) - 1.0)
    assert corrected < shipped / 20.0


def test_neither_bias_has_a_fixed_sign():
    """Both run one way near the axis and the other way off it, inside one geometry family.

    This is the claim that stops either being treated as a one-way correction, and it is the
    one the first version of the harness got wrong from otherwise-sound convexity reasoning.
    """
    axial = absorption_bias(SOURCE, _receiver(1.0, 0.0), 0.3, COARSE)
    oblique = absorption_bias(SOURCE, _receiver(1.0, 4.0), 0.3, COARSE)
    assert axial < 1.0 < oblique

    head_on = profile_bias(SOURCE, _point(2.0, 0.0), 8.0, COARSE)
    grazing = profile_bias(SOURCE, _point(2.0, 75.0), 8.0, COARSE)
    assert head_on < 1.0 < grazing


@pytest.mark.parametrize(
    "measure, argument",
    [(absorption_bias, 0.3), (profile_bias, 8.0)],
    ids=["absorption", "profile"],
)
def test_the_dense_reference_is_converged_in_its_sample_count(measure, argument):
    """Halving the sampling must not move the answer, or the sweeps report the sampling."""
    target = _receiver(1.0) if measure is absorption_bias else _point(1.0)
    assert measure(SOURCE, target, argument, COARSE) == pytest.approx(
        measure(SOURCE, target, argument, 2 * COARSE), rel=2e-3
    )


def test_the_shipped_transfer_applies_absorption_at_the_centroid_separation():
    """The bound describes the real code path, not an idealization of it.

    Everything above measures a closed form against a dense integral. This checks that the
    closed form is what ``assemble`` actually evaluates -- the ratio of the absorbing transfer
    to the clear one is exactly ``exp(-a * centroid separation)``, pair by pair. If the build
    ever integrated absorption over the receiver, this is the test that would notice, and every
    recorded bound would need re-measuring.
    """
    vertices = np.stack([SOURCE, _receiver(1.5)])
    surfaces = Surfaces.from_triangles(vertices, emission=1.0, reflectance=0.0)
    transfer = build_transfer(surfaces, self_occlusion=NoOcclusion())

    coefficient = 0.37
    clear, _ = transfer.assemble(surfaces)
    absorbing, _ = transfer.assemble(surfaces, absorption=UniformAbsorption(coefficient))

    pair = (0, 1)
    ratio = float(np.asarray(absorbing)[pair] / np.asarray(clear)[pair])
    assert ratio == pytest.approx(
        np.exp(-coefficient * centroid_separation(SOURCE, _receiver(1.5))), rel=1e-12
    )


# --- that the reference integrates the RIGHT thing -------------------------------------------
#
# Every assertion above is a ratio in which the sampling weight appears on both sides, so all of
# them are blind to that weight being wrong: drop the inverse-square from the pair kernel, or the
# cosine from the per-sample solid angle, and the controls still read exactly zero and the
# mechanism identity still holds. A mutation pass is how that was found. These two close it, by
# checking each weight against the shipped kernel it is supposed to mirror -- which is also the
# only independent reference available for it.


def test_the_per_sample_solid_angles_sum_to_the_shipped_kernel():
    """The profile reference's weight is a solid angle, and the package already computes one."""
    point = _point(1.5)
    _, omega = _source_samples(SOURCE, point, COARSE)
    assert float(np.sum(omega)) == pytest.approx(float(solid_angle(point, SOURCE)), rel=5e-3)


def test_the_reference_pair_kernel_reproduces_the_shipped_geometric_term():
    """The absorption reference's weight is the pair's own transfer, so it must equal it.

    ``geometric[i, j]`` is averaged over the receiving facet, so it is compared against the
    kernel integrated over both facets and divided by the receiver's area.
    """
    receiver = _receiver(1.5)
    surfaces = Surfaces.from_triangles(np.stack([SOURCE, receiver]))
    transfer = build_transfer(surfaces, self_occlusion=NoOcclusion())

    kernel, _ = _pair_kernel(SOURCE, receiver, COARSE)
    expected = float(np.asarray(transfer.geometric)[1, 0] * np.asarray(surfaces.area)[1])
    assert float(np.sum(kernel)) == pytest.approx(expected, rel=5e-3)


def test_the_centroid_separation_uses_both_centroids():
    """With the source at the origin, dropping either centroid is invisible. So move it.

    Every other fixture here puts the source at the origin, where its centroid is the zero
    vector and ``centroid_r - centroid_s`` is indistinguishable from ``centroid_r`` -- a whole
    family of wrong implementations that no other assertion in this file can see.
    """
    source = unit_facet([3.0, -2.0, 5.0], 1.0)
    receiver = unit_facet([3.0, -2.0, 7.5], 1.0, facing_down=True)
    assert centroid_separation(source, receiver) == pytest.approx(2.5, rel=1e-12)
