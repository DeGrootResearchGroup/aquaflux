"""Unit tests for the Venkatakrishnan slope limiter (physics-free)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.context import FieldContext, MeshContext
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter


def _context(mesh, geom, gradient):
    """A field context carrying only what the limiter reads: the gradient and the mesh."""
    mesh_context = MeshContext(face_cells=mesh.face_cells, geometry=geom, properties={})
    return FieldContext(
        mesh=mesh_context, boundary_values=jnp.zeros(mesh.n_faces), gradient=gradient
    )


def _psi(field, boundary_values):
    mesh = structured_grid_2d(24, 24)
    geom = mesh.geometry()
    grad = CorrectedGreenGauss().gradients(
        field(geom.cell.centroid), mesh, geom, field(geom.face.centroid)
    )
    psi = VenkatakrishnanLimiter(scale=1.0).limit(
        field(geom.cell.centroid), _context(mesh, geom, grad)
    )
    boundary_cells = set(
        np.asarray(mesh.face_cells.owner)[np.asarray(mesh.face_cells.neighbour) < 0].tolist()
    )
    interior = np.array([c not in boundary_cells for c in range(mesh.n_cells)])
    return np.asarray(psi), interior, np.asarray(geom.cell.centroid)


def _linear(x):
    return 2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0


def test_limiter_is_one_on_smooth_field() -> None:
    """A smooth (linear) field is reconstructed without limiting: psi -> 1 in the interior."""
    psi, interior, _ = _psi(_linear, _linear)
    assert psi[interior].min() > 0.999


def test_limiter_activates_near_discontinuity() -> None:
    """Near a step the limiter drops below 1; far from it, it is ~1."""

    def step(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    psi, _, centroid = _psi(step, step)
    near = np.abs(centroid[:, 0] - 0.5) < 0.1
    far = np.abs(centroid[:, 0] - 0.5) > 0.3
    assert psi[near].min() < 0.9
    assert psi[far].min() > 0.99


def test_limiter_stays_in_unit_interval() -> None:
    def step(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    psi, _, _ = _psi(step, step)
    assert psi.min() >= 0.0
    assert psi.max() <= 1.0 + 1e-12


def test_limiter_uses_the_periodic_image_across_a_seam() -> None:
    """A smooth periodic field must not be limited at a periodic seam.

    Before the fix, the neighbour side's unlimited increment was formed against the raw
    (non-periodic-image) neighbour centroid, which across a periodic seam sits a full domain
    length away rather than one cell width -- collapsing `psi` toward 0 there for any field, no
    matter how smooth. `face_cells.neighbour_centroid` is the fix: it gathers the neighbour's
    periodic image instead, matching the displacement `LimitedUpwind.face_value` reconstructs on.
    """
    lx = 1.0
    mesh = structured_grid_2d(8, 4, lx=lx, ly=1.0, periodic=("x",), named_boundaries=True)
    geom = mesh.geometry()

    # A sine, so the field is monotone across the seam: a smooth extremum is limited by design (its
    # headroom is ~0), so a cosine, peaked on the seam, would read as limited whatever the seam did.
    def field(x):
        return jnp.sin(2.0 * jnp.pi * x[..., 0] / lx)

    grad = CorrectedGreenGauss().gradients(
        field(geom.cell.centroid), mesh, geom, field(geom.face.centroid)
    )
    psi = VenkatakrishnanLimiter(scale=1.0).limit(
        field(geom.cell.centroid), _context(mesh, geom, grad)
    )

    fc = mesh.face_cells
    seam_faces = np.asarray(jnp.any(fc.neighbour_offset != 0.0, axis=-1))
    assert seam_faces.any(), "fixture must actually carry a periodic seam"
    seam_cells = set(np.asarray(fc.owner)[seam_faces].tolist()) | set(
        np.asarray(fc.neighbour)[seam_faces].tolist()
    )
    # 0.9 separates "correctly unlimited" (0.947 here, the curvature limiting the non-seam cells
    # beside them see too) from the pre-fix bug, which on this fixture collapses the seam cells' psi
    # to 0.11 by forming the unlimited increment against a raw neighbour centroid a full domain length
    # away instead of the periodic image.
    assert np.asarray(psi)[list(seam_cells)].min() > 0.9


def test_limiter_is_differentiable() -> None:
    """``jax.grad`` through the limiter (the stencil min/max and the smooth ratio) matches a
    central finite difference at several interior cells -- not merely finite. A ``stop_gradient``
    around the stencil ``phi_max``/``phi_min`` extrema passes a bare NaN check (headroom is still a
    function of ``field`` through its own cell's value) while moving several of these values well
    outside a finite-difference tolerance.
    """
    mesh = structured_grid_2d(8, 8)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss()
    limiter = VenkatakrishnanLimiter(scale=1.0)
    field0 = jnp.sin(geom.cell.centroid[:, 0] * 3.0)

    def loss(field):
        grad = scheme.gradients(field, mesh, geom, jnp.zeros(mesh.n_faces))
        return jnp.sum(limiter.limit(field, _context(mesh, geom, grad)) ** 2)

    sens = jax.grad(loss)(field0)
    assert not bool(jnp.any(jnp.isnan(sens)))

    step = 1e-6
    # Cells away from a stencil-min/max kink (crossing one under the central difference's +-step
    # would spuriously fail); chosen by inspection of this fixture's smooth interior.
    for index in (2, 5, 10, 30):
        finite_difference = float(
            (loss(field0.at[index].add(step)) - loss(field0.at[index].add(-step))) / (2.0 * step)
        )
        assert float(sens[index]) == pytest.approx(finite_difference, rel=1e-4, abs=1e-8)
    assert float(jnp.abs(sens).max()) > 1e-6, (
        "a severed adjoint would report zero and pass a finiteness check"
    )


def test_the_softening_constant_is_a_differentiable_leaf() -> None:
    """``softening`` is an ordinary pytree leaf, so a gradient taken with respect to the limiter reaches it.

    Held as a static field it would be part of the tree's structure instead: it would not appear among
    the leaves, the gradient module would carry no derivative for it, and changing it would recompile.
    The value is checked against a central finite difference on a step, where the limiter is active
    and ``psi`` therefore depends on the softening -- on a smooth field ``psi`` is ~1 whatever it is.
    """
    mesh = structured_grid_2d(12, 12)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss()
    field = jnp.where(geom.cell.centroid[:, 0] < 0.5, 1.0, 0.0)
    context = _context(mesh, geom, scheme.gradients(field, mesh, geom, jnp.zeros(mesh.n_faces)))

    def loss(limiter):
        return jnp.sum(limiter.limit(field, context))

    limiter = VenkatakrishnanLimiter(softening=jnp.asarray(0.1), scale=1.0)
    assert any(leaf is limiter.softening for leaf in jax.tree_util.tree_leaves(limiter))

    sensitivity = float(eqx.filter_grad(loss)(limiter).softening)
    step = 1e-6
    finite_difference = float(
        (
            loss(VenkatakrishnanLimiter(softening=0.1 + step, scale=1.0))
            - loss(VenkatakrishnanLimiter(softening=0.1 - step, scale=1.0))
        )
        / (2.0 * step)
    )
    assert abs(finite_difference) > 1e-3, "the fixture must make psi depend on the softening"
    assert sensitivity == pytest.approx(finite_difference, rel=1e-5)


def _step_psi(*, length=1.0, amplitude=1.0, scale=1.0, softening=0.05, n=24):
    """``psi`` of a smoothed step of height ``amplitude`` across a square of side ``length``.

    The step is a ``tanh`` a few cells wide, so the limiter is genuinely active and partly softened:
    a fixture where ``psi`` is 1 or 0 everywhere could not tell one softening from another.
    """
    mesh = structured_grid_2d(n, n, lx=length, ly=length)
    geom = mesh.geometry()

    def field(x):
        return amplitude * 0.5 * (1.0 + jnp.tanh((x[..., 0] / length - 0.5) * 12.0))

    values = field(geom.cell.centroid)
    gradient = CorrectedGreenGauss().gradients(values, mesh, geom, field(geom.face.centroid))
    limiter = VenkatakrishnanLimiter(softening=softening, scale=scale)
    return np.asarray(limiter.limit(values, _context(mesh, geom, gradient)))


def test_psi_does_not_depend_on_the_units_of_the_field() -> None:
    """The same field stated in units 1000x smaller, with its scale stated likewise, limits identically.

    The softening ``eps = K phi_ref`` carries the field's units, so it scales with the field and
    ``psi`` (a ratio of increments) is unchanged. A softening with units of its own -- the
    ``eps^2 = vol K^3`` this replaced -- holds still while the field's increments grow, and ``psi``
    moves.
    """
    reference = _step_psi()
    assert 0.05 < reference.min() < 0.95, "the fixture must put the limiter in its softened range"
    np.testing.assert_allclose(_step_psi(amplitude=1e3, scale=1e3), reference, rtol=1e-10)


def test_psi_does_not_depend_on_the_size_of_the_mesh() -> None:
    """The same problem on a domain 1000x larger (the same mesh, in millimetres) limits identically.

    Neither the softening nor the ratio of increments depends on a length, so stretching the mesh
    with the field changes nothing. A cell-volume softening grows with the domain and would switch
    the limiter off on the larger one.
    """
    np.testing.assert_allclose(_step_psi(length=1e3), _step_psi(), rtol=1e-10)


def test_the_softening_decides_how_much_of_a_variation_is_left_alone() -> None:
    """A softening larger than the step leaves it unlimited; one far smaller limits it.

    Pins that the softening reaches ``psi`` at all, in the direction it should: an implementation
    that dropped ``eps`` from either the numerator or the denominator would fail one side.
    """
    assert _step_psi(softening=10.0).min() > 0.99
    assert _step_psi(softening=1e-4).min() < _step_psi(softening=0.05).min() - 0.05


def test_a_limiter_with_no_scale_refuses_to_evaluate() -> None:
    """With no scale the softening would be zero, a strict limiter dividing 0/0 on a flat field."""
    mesh = structured_grid_2d(4, 4)
    geom = mesh.geometry()
    field = jnp.zeros(mesh.n_cells)
    context = _context(mesh, geom, jnp.zeros((mesh.n_cells, 2)))
    with pytest.raises(ValueError, match="has no reference scale"):
        VenkatakrishnanLimiter().limit(field, context)


def test_a_reference_scale_is_filled_in_only_where_none_was_given() -> None:
    """The assembler's scale fills an unset one; a scale stated on the limiter wins."""
    assert float(VenkatakrishnanLimiter().with_reference_scale(lambda: 3.0).scale) == 3.0
    kept = VenkatakrishnanLimiter(scale=2.0).with_reference_scale(lambda: 3.0)
    assert float(kept.scale) == 2.0


def test_a_scale_already_given_is_never_derived() -> None:
    """A limiter that carries a scale does not ask for one, so an underivable scale is no refusal."""

    def underivable():
        raise AssertionError("asked for a scale it already has")

    assert float(VenkatakrishnanLimiter(scale=2.0).with_reference_scale(underivable).scale) == 2.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_reference_scale_must_be_positive_and_finite(bad) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        VenkatakrishnanLimiter().with_reference_scale(lambda: bad)


def test_schemes_that_read_no_scale_never_ask_for_one() -> None:
    """First-order and unlimited upwind are returned as they are, without deriving a scale."""
    from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind

    def underivable():
        raise AssertionError("a scheme that reads no scale asked for one")

    first = FirstOrderUpwind()
    unlimited = LimitedUpwind()
    assert first.with_reference_scale(underivable) is first
    assert unlimited.with_reference_scale(underivable) is unlimited
    limited = LimitedUpwind(limiter=VenkatakrishnanLimiter()).with_reference_scale(lambda: 4.0)
    assert float(limited.limiter.scale) == 4.0


def test_the_scale_is_not_a_case_file_setting() -> None:
    """A case file states the softening; the scale is the assembler's, so a file cannot name it."""
    from aquaflux.case import case_schema

    fields = {field["name"] for field in case_schema()["kinds"]["VenkatakrishnanLimiter"]["fields"]}
    assert fields == {"softening"}
