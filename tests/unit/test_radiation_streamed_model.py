"""A model whose receiver shadows are built per chunk at every call rather than held whole.

Streaming is a memory strategy, so the first thing pinned is that it changes nothing about the
answer: the field and every gradient agree with the held mask's. The rest pins the memory claims
mechanically — by counting the mask builds and what each is handed — because they are invisible in
the answer, which is the reason they need tests of their own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import gather
from aquaflux.radiation.absorption import UniformAbsorption
from aquaflux.radiation.model import RadiationSettings, build_radiation_model, fluence_rate
from aquaflux.radiation.receiver_shadows import FrozenShadows, StreamedShadows
from aquaflux.radiation.self_occlusion import NoOcclusion
from aquaflux.solids import Cylinder

from tests.unit.radiation_references import box

SLEEVE = Cylinder(centre=[0.5, 0.5, 0.5], axis=[0, 0, 1], radius=0.15, half_length=0.3)


def _receivers(count=11):
    """Points in the box, outside the sleeve, some of them behind it from most walls."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0.05, 0.95, (4 * count, 3))
    outside = np.hypot(points[:, 0] - 0.5, points[:, 1] - 0.5) > 0.2
    return points[outside][:count]


def _scene():
    return box(2, emission=1.0, reflectance=0.5)


def _model(stream, **settings):
    surfaces = _scene()
    model = build_radiation_model(
        _receivers(),
        surfaces,
        occluders=[SLEEVE],
        settings=RadiationSettings(
            self_occlusion=NoOcclusion(), stream_receiver_mask=stream, **settings
        ),
    )
    return model, surfaces


def _watch_mask_builds(monkeypatch):
    """Record how many receivers each streamed mask build is handed."""
    handed = []
    real = gather._unchecked_visibility

    def watched(occluders, surfaces, points, **options):
        handed.append(np.asarray(points).shape[0])
        return real(occluders, surfaces, points, **options)

    monkeypatch.setattr(gather, "_unchecked_visibility", watched)
    return handed


def test_the_mask_is_held_unless_streaming_is_asked_for():
    held, _ = _model(stream=None)
    streamed, _ = _model(stream=True)
    assert isinstance(held.receiver_shadows, FrozenShadows)
    assert isinstance(streamed.receiver_shadows, StreamedShadows)


def test_streaming_gives_the_field_a_held_mask_gives():
    held, surfaces = _model(stream=None)
    streamed, _ = _model(stream=True, gather_pair_limit=3 * surfaces.n_facets)
    water = UniformAbsorption(0.8)
    expected, _ = fluence_rate(held, surfaces, absorption=water, transmittance=jnp.array([0.2]))
    got, _ = fluence_rate(streamed, surfaces, absorption=water, transmittance=jnp.array([0.2]))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12)
    assert float(np.ptp(np.asarray(expected))) > 0.0, "a uniform field would hide a misplaced mask"


def test_every_gradient_a_held_mask_gives_a_streamed_one_gives_too():
    """Emission, reflectance, the medium and the sleeve's transmittance, all through the solve."""
    held, surfaces = _model(stream=None)
    streamed, _ = _model(stream=True, gather_pair_limit=3 * surfaces.n_facets)

    def total(model, emission, reflectance, coefficient, transmittance):
        lit = surfaces.with_optics(emission=emission, reflectance=reflectance)
        field, _ = fluence_rate(
            model, lit, absorption=UniformAbsorption(coefficient), transmittance=transmittance
        )
        return jnp.sum(field)

    arguments = (
        jnp.asarray(surfaces.emission),
        jnp.asarray(surfaces.reflectance),
        jnp.asarray(0.8),
        jnp.array([0.2]),
    )
    wanted = jax.grad(lambda *a: total(held, *a), argnums=(0, 1, 2, 3))(*arguments)
    got = jax.grad(lambda *a: total(streamed, *a), argnums=(0, 1, 2, 3))(*arguments)
    for expected, found in zip(wanted, got, strict=True):
        np.testing.assert_allclose(np.asarray(found), np.asarray(expected), rtol=1e-10)
        assert np.any(np.asarray(expected) != 0.0), "a severed adjoint reads zero, not NaN"

    # And one against a difference, not only against the other path.
    step = 1e-6
    shifted = [total(streamed, *arguments[:3], arguments[3] + sign * step) for sign in (1.0, -1.0)]
    finite = (shifted[0] - shifted[1]) / (2.0 * step)
    np.testing.assert_allclose(float(got[3][0]), float(finite), rtol=1e-6)


def test_one_mask_per_chunk_serves_the_emitted_and_the_reflected_gathers(monkeypatch):
    """The two gathers share the geometry, so building a mask for each would double the cost."""
    streamed, surfaces = _model(stream=True, gather_pair_limit=4 * _scene().n_facets)
    handed = _watch_mask_builds(monkeypatch)
    fluence_rate(streamed, surfaces)
    assert handed == [4, 4, 3], handed


def test_a_gradient_rebuilds_each_chunk_s_mask_rather_than_keeping_its_intermediates(monkeypatch):
    """What keeps a mesh-scale gradient to one chunk's memory, pinned by the builds it causes.

    Reverse mode would otherwise hold every chunk's receiver-by-facet intermediates until the
    backward pass -- the size of the whole problem. Each chunk instead saves only its inputs and
    rebuilds its mask on the way back, so a gradient sees every chunk's build twice.
    """
    streamed, surfaces = _model(stream=True, gather_pair_limit=4 * _scene().n_facets)
    handed = _watch_mask_builds(monkeypatch)

    def total(emission):
        field, _ = fluence_rate(streamed, surfaces.with_optics(emission=emission))
        return jnp.sum(field)

    jax.grad(total)(jnp.asarray(surfaces.emission))
    # Forward chunk by chunk, then back through them in reverse, each rebuilding its own mask.
    assert handed == [4, 4, 3, 3, 4, 4], handed


def test_a_lamp_can_be_moved_under_a_gradient_with_the_shadows_frozen():
    """The masks come from the geometry the model was built for, not from the call's vertices.

    Under a gradient with respect to a vertex those vertices are traced, and a mask cannot be built
    from a traced position -- so a stream that built its masks from the call's surface set could
    not be differentiated in geometry at all, where a held mask can.
    """
    held, surfaces = _model(stream=None)
    streamed, _ = _model(stream=True, gather_pair_limit=4 * surfaces.n_facets)

    def total(model, lift):
        moved = surfaces.with_geometry(jnp.asarray(surfaces.vertices).at[0, :, 2].add(lift))
        field, _ = fluence_rate(model, moved)
        return jnp.sum(field)

    expected = jax.grad(lambda z: total(held, z))(0.0)
    got = jax.grad(lambda z: total(streamed, z))(0.0)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-10)
    assert float(expected) != 0.0


def test_a_receiver_inside_a_body_is_refused_at_build_not_at_the_first_call():
    """A held mask refuses it while being built; a streamed one builds nothing then, and must."""
    surfaces = _scene()
    inside = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="receiver"):
        build_radiation_model(
            inside,
            surfaces,
            occluders=[SLEEVE],
            settings=RadiationSettings(self_occlusion=NoOcclusion(), stream_receiver_mask=True),
        )
