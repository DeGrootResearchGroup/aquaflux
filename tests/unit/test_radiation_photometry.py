"""IES LM-63 photometry: reading a file, completing its symmetry, and the profile it defines."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    Lambertian,
    Surfaces,
    build_transfer,
    check_profiles,
    direct_fluence_rate,
    direct_irradiance,
    read_ies,
)
from aquaflux.radiation.photometry import MIN_COSINE, Photometry


def _ies(vertical, horizontal, rows, *, multiplier=1.0, ballast=1.0, kind=1, tilt="NONE"):
    """The text of an LM-63 file holding ``rows[h][gamma]``."""
    rows = np.asarray(rows, dtype=float)
    lines = [
        "IESNA:LM-63-2002",
        "[TEST] synthetic",
        "[_INTENSITYUNITS] W/sr",
        f"TILT={tilt}",
        f"1 -1 {multiplier} {len(vertical)} {len(horizontal)} {kind} 2 0.1 0.2 0.0",
        f"{ballast} 1 10",
        " ".join(map(str, vertical)),
        " ".join(map(str, horizontal)),
        *(" ".join(map(str, row)) for row in rows),
    ]
    return "\n".join(lines) + "\n"


def _write(tmp_path, text, name="lamp.ies") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def b1(tmp_path_factory) -> Path:
    """A synthetic full-table luminaire shaped like the measured Ushio B1 module.

    Same layout as the file the ray-effects case reads (37 gamma columns over 0 to 90 degrees,
    17 horizontal rows over 0 to 360, no assumed symmetry, mW/sr, a 45 x 60 mm opening), so these
    tests do not depend on ``validation/``. The intensity is a smooth cosine lobe with an azimuthal
    modulation that vanishes on the axis, so no two rows disagree at ``gamma = 0``.
    """
    vertical = np.arange(37) * 2.5
    horizontal = np.arange(17) * 22.5
    g, h = np.radians(vertical)[None, :], np.radians(horizontal)[:, None]
    rows = (
        120.0 * np.cos(g) ** 6 * (1 + 0.2 * np.sin(g) * np.sin(h) + 0.1 * np.sin(g) * np.cos(2 * h))
    )
    text = (
        _ies(vertical.tolist(), horizontal.tolist(), rows)
        .replace("1 -1 1.0 37 17 1 2 0.1 0.2 0.0", "1 -1 1.0 37 17 1 2 0.045 0.06 0.0")
        .replace("[_INTENSITYUNITS] W/sr", "[_INTENSITYUNITS] mW/sr")
    )
    return _write(tmp_path_factory.mktemp("ies"), text, name="b1_like.ies")


def _bilinear_reference(photometry: Photometry, gamma_deg, h_deg):
    """The stored table at ``(gamma, h)`` in degrees, by np.interp on a full-circle table.

    Written against the LM-63 folding rules directly, not through ``full_circle``, so it is an
    independent reading of the same file.
    """
    h = np.asarray(h_deg) % 360.0
    kind = photometry.symmetry
    if kind == "quadrant":
        h = np.where(h > 270, 360 - h, np.where(h > 180, h - 180, np.where(h > 90, 180 - h, h)))
    elif kind == "bilateral":
        h = np.where(h > 180, 360 - h, h)
    stored_h = photometry.horizontal
    table = photometry.intensity
    if kind == "rotational":
        return np.interp(gamma_deg, photometry.vertical, table[0])
    if kind == "full":
        stored_h = np.append(stored_h, stored_h[0] + 360.0)
        table = np.vstack([table, table[:1]])
    columns = np.stack([np.interp(gamma_deg, photometry.vertical, row) for row in table])
    return np.array([np.interp(hh, stored_h, columns[:, k]) for k, hh in enumerate(np.ravel(h))])


def _hemisphere(normal, up, n=1200):
    """World directions over the front hemisphere of ``normal``, and each one's solid angle.

    A midpoint rule in ``(cos gamma, phi)``, whose measure is uniform in solid angle; built in its
    own frame so the profile's frame is not reused.
    """
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    first = np.cross(normal, [0.3, -0.7, 0.2])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    mu = (np.arange(n) + 0.5) / n
    phi = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * np.pi
    m, p = np.meshgrid(mu, phi, indexing="ij")
    s = np.sqrt(1 - m**2)
    directions = (
        s[..., None] * np.cos(p)[..., None] * first
        + s[..., None] * np.sin(p)[..., None] * second
        + m[..., None] * normal
    )
    return directions.reshape(-1, 3), (1.0 / n) * (2 * np.pi / (2 * n))


# -- reading ------------------------------------------------------------------------------------


def test_a_b1_shaped_file_reads_as_a_full_table_in_milliwatts_per_steradian(b1):
    """A file laid out like the ray-effects case's: every header field the case relies on, read back."""
    photometry = read_ies(b1)
    assert photometry.symmetry == "full"
    assert photometry.intensity.shape == (17, 37)
    assert photometry.vertical[[0, -1]].tolist() == [0.0, 90.0]
    assert photometry.horizontal[[0, -1]].tolist() == [0.0, 360.0]
    assert photometry.opening == (0.045, 0.06, 0.0)
    assert photometry.opening_unit == "metres"
    assert photometry.keywords["_INTENSITYUNITS"] == "mW/sr"
    assert photometry.intensity[0, 0] == pytest.approx(120.0, rel=1e-12)


def test_the_multiplier_and_ballast_factor_scale_the_table(tmp_path):
    rows = [[4.0, 2.0, 0.0]]
    plain = read_ies(_write(tmp_path, _ies([0, 45, 90], [0], rows)))
    scaled = read_ies(_write(tmp_path, _ies([0, 45, 90], [0], rows, multiplier=3, ballast=0.5)))
    np.testing.assert_allclose(scaled.intensity, 1.5 * plain.intensity, rtol=1e-15)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_ies([0, 90], [0], [[1, 0]], tilt="INCLUDE"), "only TILT=NONE"),
        (_ies([0, 90], [0], [[1, 0]], kind=2), "only Type C"),
        (_ies([0, 90], [0], [[1, 0]])[:-4] + "\n", "expected 18 numbers"),
        ("IESNA:LM-63-2002\n1 2 3\n", "no TILT= line"),
    ],
    ids=["tilt", "type-b", "truncated", "no-tilt-line"],
)
def test_a_file_it_cannot_read_faithfully_is_refused(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        read_ies(_write(tmp_path, text))


# -- symmetry -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizontal", "kind"),
    [([0.0], "rotational"), ([0, 30, 60, 90], "quadrant"), ([0, 60, 120, 180], "bilateral")],
)
def test_a_partial_table_is_completed_by_the_symmetry_its_last_angle_declares(
    tmp_path, horizontal, kind
):
    """The completed circle interpolates to the value the LM-63 fold rule gives, everywhere.

    The reference folds each angle onto the stored range and interpolates there, which is how the
    standard defines a symmetric table; the profile's table is built differently (by expanding the
    rows once), so the two agreeing at off-grid angles pins the expansion. Rows differ from one
    another, so a fold landing on the wrong stored row is visible.
    """
    vertical = [0, 30, 60, 90]
    rows = [[10 + 3 * i, 8 + i, 4 - 0.5 * i, 0.5 * i] for i in range(len(horizontal))]
    photometry = read_ies(_write(tmp_path, _ies(vertical, horizontal, rows)))
    assert photometry.symmetry == kind
    full_h, full_table = photometry.full_circle()
    assert full_h[-1] - full_h[0] == pytest.approx(360.0)
    rng = np.random.default_rng(0)
    gamma, h = rng.uniform(0, 90, 200), rng.uniform(0, 360, 200)
    columns = np.stack([np.interp(gamma, vertical, row) for row in full_table])
    ours = np.array([np.interp(hh, full_h, columns[:, k]) for k, hh in enumerate(h)])
    np.testing.assert_allclose(ours, _bilinear_reference(photometry, gamma, h), rtol=1e-12)


def test_a_quadrant_table_mirrors_into_the_second_quadrant(tmp_path):
    """A value checked by hand: h = 135 degrees reads the stored 45-degree row."""
    rows = [[1.0, 1.0], [2.0, 2.0], [7.0, 7.0]]
    photometry = read_ies(_write(tmp_path, _ies([0, 90], [0, 45, 90], rows)))
    full_h, full_table = photometry.full_circle()
    assert full_table[np.flatnonzero(np.isclose(full_h, 135.0))[0], 0] == 2.0
    assert full_table[np.flatnonzero(np.isclose(full_h, 180.0))[0], 0] == 1.0


# -- flux and normalization ---------------------------------------------------------------------


def test_the_flux_is_the_exact_integral_of_the_bilinear_table(b1):
    """Against a dense midpoint integral of an independent bilinear reading of the file."""
    photometry = read_ies(b1)
    n = 1500
    gamma = (np.arange(n) + 0.5) / n * 90.0
    h = (np.arange(2 * n) + 0.5) / (2 * n) * 360.0
    g, hh = np.meshgrid(gamma, h, indexing="ij")
    values = _bilinear_reference(photometry, g.ravel(), hh.ravel()).reshape(g.shape)
    dense = (
        np.sum(values * np.sin(np.radians(g))) * np.radians(90.0 / n) * np.radians(360 / (2 * n))
    )
    assert photometry.flux == pytest.approx(dense, rel=2e-6)


def test_a_table_stored_short_of_ninety_degrees_is_continued_at_its_last_value(tmp_path):
    """Clamped, not extrapolated: the band from 60 to 90 degrees emits at the 60-degree value."""
    photometry = read_ies(_write(tmp_path, _ies([0, 60], [0], [[2.0, 3.0]])))
    # Linear from 2 to 3 over [0, 60] degrees, then 3 to 90: each band in closed form.
    a, b = 0.0, np.pi / 3
    s0 = np.cos(a) - np.cos(b)
    s1 = (np.sin(b) - b * np.cos(b)) - (np.sin(a) - a * np.cos(a))
    stored = 2 * np.pi * (2.0 * (b * s0 - s1) + 3.0 * (s1 - a * s0)) / (b - a)
    assert photometry.flux == pytest.approx(stored, rel=1e-12)
    assert photometry.front_hemisphere_flux() == pytest.approx(
        stored + 3.0 * 2 * np.pi * np.cos(b), rel=1e-12
    )
    # The profile evaluates the same continuation, and is normalized over it.
    profile = photometry.profile(up=(1.0, 0.0, 0.0))
    normal = jnp.array([0.0, 0.0, -1.0])
    at = [
        float(profile.intensity_fraction(jnp.array([np.sin(g), 0.0, -np.cos(g)]), normal))
        for g in np.radians([60.0, 75.0, 89.0])
    ]
    np.testing.assert_allclose(at, at[0], rtol=1e-12)
    directions, weight = _hemisphere(normal, (1.0, 0.0, 0.0), n=600)
    total = np.sum(np.asarray(profile.intensity_fraction(jnp.asarray(directions), normal)))
    assert total * weight == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize(
    ("normal", "up"),
    [((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)), ((0.3, -0.5, 0.81), (0.0, 1.0, 0.2))],
    ids=["downward", "oblique"],
)
def test_the_profile_integrates_to_one_over_the_facets_front_hemisphere(b1, normal, up):
    """Normalized in the facet's own frame, whichever way the facet faces.

    The oblique case puts the normal and ``up`` off every axis, so a frame built from the world
    axes rather than from the normal would integrate to something else.
    """
    normal = np.asarray(normal) / np.linalg.norm(normal)
    profile = read_ies(b1).profile(up=up)
    directions, weight = _hemisphere(normal, up)
    total = np.sum(np.asarray(profile.intensity_fraction(jnp.asarray(directions), normal)))
    assert total * weight == pytest.approx(1.0, rel=2e-6)


def test_the_profile_is_the_table_over_its_flux_along_every_tabulated_direction(tmp_path):
    """Each stored direction, sent as a world vector, reads its own stored value back.

    Normal +x and up +z, so ``h = 90`` lies along ``normal x up = -y``. A frame whose
    horizontal angle ran the other way, or started from the wrong reference, reads a different
    row at every ``h`` not on a mirror line of the table.
    """
    vertical = [0.0, 20.0, 50.0, 90.0]
    horizontal = [0.0, 45.0, 135.0, 200.0, 290.0]
    rows = [[5.0, 4.0 + i, 2.0 + 0.3 * i, 0.0] for i in range(len(horizontal))]
    photometry = read_ies(_write(tmp_path, _ies(vertical, horizontal, rows)))
    profile = photometry.profile(up=(0.0, 0.0, 1.0))
    normal, first, second = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])
    for i, h in enumerate(np.radians(horizontal)):
        for k, g in enumerate(np.radians(vertical[1:-1]), start=1):
            direction = np.cos(g) * normal + np.sin(g) * (np.cos(h) * first + np.sin(h) * second)
            value = float(profile.intensity_fraction(jnp.asarray(direction), jnp.asarray(normal)))
            assert value * photometry.front_hemisphere_flux() == pytest.approx(
                rows[i][k], rel=1e-12
            )


def test_the_horizontal_angle_runs_towards_normal_cross_up(tmp_path):
    """The sign convention, as of-optical-radiation's iesEmitter fixes it: ``F(h) = 5 + 4 sin h``.

    With normal +x and up +z the brightest direction (h = 90) is towards -y and the dimmest
    (h = 270) towards +y, and +z and -z (h = 0 and 180) tie between them.
    """
    horizontal = np.arange(0, 360, 45.0)
    rows = [[5 + 4 * np.sin(np.radians(h))] * 4 for h in horizontal]
    profile = read_ies(_write(tmp_path, _ies([0, 30, 60, 90], horizontal, rows))).profile(
        up=(0.0, 0.0, 1.0)
    )
    normal = jnp.array([1.0, 0.0, 0.0])

    def towards(y, z):
        direction = np.array([1.0, y, z])
        return float(
            profile.intensity_fraction(jnp.asarray(direction / np.linalg.norm(direction)), normal)
        )

    minus_y, plus_y, plus_z, minus_z = towards(-1, 0), towards(1, 0), towards(0, 1), towards(0, -1)
    assert minus_y / plus_y == pytest.approx(9.0, rel=1e-12)
    assert plus_z == pytest.approx(minus_z, rel=1e-12)
    assert minus_y > plus_z > plus_y


def test_only_the_part_of_up_in_the_facets_plane_matters(b1):
    """An ``up`` tilted towards the normal measures ``h`` from its projection, not from itself.

    The two profiles differ only in ``up``'s component along the normal (+x), so they must agree
    to rounding at every direction; one that used ``up`` unprojected would measure every angle from
    a reference that is neither unit length nor in the plane.
    """
    photometry = read_ies(b1)
    tilted, flat = photometry.profile(up=(0.7, 0.0, 1.0)), photometry.profile(up=(0.0, 0.0, 1.0))
    normal = jnp.array([1.0, 0.0, 0.0])
    directions, _ = _hemisphere(np.asarray(normal), None, n=40)
    np.testing.assert_allclose(
        tilted.intensity_fraction(jnp.asarray(directions), normal),
        flat.intensity_fraction(jnp.asarray(directions), normal),
        rtol=1e-12,
        atol=1e-15,
    )


def test_a_rotational_cosine_table_is_lambertian(tmp_path):
    """``I = I0 cos gamma`` tabulated finely is a Lambertian emitter: radiance ``1/pi`` of exitance.

    Bilinear interpolation of a cosine at half-degree spacing errs by under 1e-5 relative, so the
    reduction is checked to that; a normalization over the whole sphere instead of the front
    hemisphere would halve it, and a missing cosine in the radiance would make it vary as cos.
    """
    vertical = np.arange(0, 90.5, 0.5)
    profile = read_ies(
        _write(tmp_path, _ies(vertical, [0.0], [np.cos(np.radians(vertical))]))
    ).profile(up=(1.0, 0.0, 0.0))
    normal = jnp.array([0.0, 0.0, -1.0])
    gamma = np.radians(np.linspace(0.0, 85.0, 30))
    directions = np.stack([np.sin(gamma), np.zeros_like(gamma), -np.cos(gamma)], axis=-1)
    np.testing.assert_allclose(
        profile.radiance_per_exitance(jnp.asarray(directions), normal),
        Lambertian().radiance_per_exitance(jnp.asarray(directions), normal),
        rtol=2e-5,
    )


def test_the_profile_is_dark_behind_and_bounded_along_the_facets_plane(tmp_path):
    """Nothing behind the facet; along its plane the radiance is the intensity over MIN_COSINE."""
    profile = read_ies(_write(tmp_path, _ies([0, 90], [0], [[2.0, 1.0]]))).profile(
        up=(1.0, 0.0, 0.0)
    )
    normal = jnp.array([0.0, 0.0, -1.0])
    behind = jnp.array([[0.0, 0.6, 0.8], [1.0, 0.0, 1e-9]])
    assert np.all(np.asarray(profile.radiance_per_exitance(behind, normal)) == 0.0)
    assert np.all(np.asarray(profile.intensity_fraction(behind, normal)) == 0.0)
    grazing = jnp.array([1.0, 0.0, -1e-7])
    radiance = float(profile.radiance_per_exitance(grazing / jnp.linalg.norm(grazing), normal))
    fraction = float(profile.intensity_fraction(grazing / jnp.linalg.norm(grazing), normal))
    assert radiance == pytest.approx(fraction / MIN_COSINE, rel=1e-9)
    assert type(profile).dark_behind


# -- in a scene ---------------------------------------------------------------------------------


def _window(width, normal_down=True):
    """A square of side ``width`` centred on the origin, facing -z (two triangles)."""
    w = width / 2
    a, b, c, d = (-w, -w, 0.0), (w, -w, 0.0), (w, w, 0.0), (-w, w, 0.0)
    triangles = np.array([[a, c, b], [a, d, c]] if normal_down else [[a, b, c], [a, c, d]])
    return triangles


def test_a_small_window_delivers_power_times_intensity_over_distance_squared(b1):
    """Far from a small window the fluence rate is ``P f(d) / r^2`` along every direction.

    The receivers sit 2 m below a 1 cm window at a spread of angles and azimuths, so the table's
    variation in both angles is exercised; the finite window's own correction is of order
    ``(w / r)^2``, about 1e-5. They stay 3 degrees or more off the axis: a measured table can disagree
    with itself at ``gamma = 0`` across ``h``, so within a fraction of a degree of the pole the
    window's two triangles could read different rows of it and the point-source limit would not be
    the right reference there.
    """
    photometry = read_ies(b1)
    profile = photometry.profile(up=(1.0, 0.0, 0.0))
    power = 0.1188
    triangles = _window(0.01)
    area = 0.01**2
    surfaces = Surfaces.from_triangles(triangles, emission=power / area, profiles=(profile,))
    rng = np.random.default_rng(3)
    gamma, h = np.radians(rng.uniform(3, 70, 25)), rng.uniform(0, 2 * np.pi, 25)
    r = 2.0
    points = r * np.stack([np.sin(gamma) * np.cos(h), np.sin(gamma) * np.sin(h), -np.cos(gamma)], 1)
    fluence = np.asarray(direct_fluence_rate(surfaces, jnp.asarray(points)))
    expected = (
        power
        * np.asarray(profile.intensity_fraction(jnp.asarray(points / r), jnp.array([0.0, 0, -1])))
        / r**2
    )
    np.testing.assert_allclose(fluence, expected, rtol=1e-4)
    # Irradiance on a floor facing up is the same times the cosine at the receiver.
    irradiance = np.asarray(
        direct_irradiance(surfaces, jnp.asarray(points), jnp.tile(jnp.array([0.0, 0, 1]), (25, 1)))
    )
    np.testing.assert_allclose(irradiance, expected * np.cos(gamma), rtol=1e-4)


def test_check_profiles_refuses_an_up_direction_along_the_window_normal(b1):
    profile = read_ies(b1).profile(up=(0.0, 0.0, 1.0))
    surfaces = Surfaces.from_triangles(_window(0.01), emission=1.0, profiles=(profile,))
    with pytest.raises(ValueError, match="parallel to the profile's up"):
        check_profiles(surfaces)
    check_profiles(
        Surfaces.from_triangles(
            _window(0.01), emission=1.0, profiles=(read_ies(b1).profile(up=(1.0, 0, 0)),)
        )
    )


def test_the_surface_transfer_refuses_a_profile_it_cannot_freeze(b1):
    """The transfer freezes one source cosine per pair; an azimuthal table needs more."""
    lamp = _window(0.1)
    floor = _window(1.0, normal_down=False) + np.array([0.0, 0.0, -1.0])
    profile = read_ies(b1).profile(up=(1.0, 0.0, 0.0))
    surfaces = Surfaces.from_triangles(
        np.concatenate([lamp, floor]),
        emission=np.array([1.0, 1.0, 0.0, 0.0]),
        reflectance=np.array([0.0, 0.0, 0.5, 0.5]),
        profiles=(profile, Lambertian()),
        profile_index=np.array([0, 0, 1, 1]),
    )
    transfer = build_transfer(surfaces)
    with pytest.raises(NotImplementedError, match="freezes only that angle's cosine"):
        transfer.assemble(surfaces)


def test_the_fluence_rate_is_differentiable_in_the_table(b1):
    """The table is a live leaf: its derivative matches a finite difference and is not zero."""
    profile = read_ies(b1).profile(up=(1.0, 0.0, 0.0))
    triangles = _window(0.01)
    point = jnp.array([[0.3, -0.2, -1.5]])

    def fluence(scale):
        scaled = type(profile)(
            profile.vertical, profile.horizontal, profile.table * scale, profile.up
        )
        return direct_fluence_rate(
            Surfaces.from_triangles(triangles, emission=1000.0, profiles=(scaled,)), point
        )[0]

    derivative = float(jax.grad(fluence)(1.0))
    step = 1e-4
    assert derivative != 0.0
    assert derivative == pytest.approx(
        (float(fluence(1.0 + step)) - float(fluence(1.0 - step))) / (2 * step), rel=1e-8
    )
