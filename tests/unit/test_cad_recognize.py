"""The recognition rules, on hand-built face records — no CAD kernel needed.

Each fixture states the faces a kernel would report for a known shape, so every rule is checked
against the body that shape actually is. The two things a rule most easily gets backwards are
exercised directly: which side of a curved face the solid is on (a pipe wall against a bore), and
faces of one surface split by an exporter (two half-turn faces are one whole cylinder).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.io.cad import (
    ConeFace,
    ConvexPolyhedron,
    CurvedPieces,
    CylinderFace,
    OtherFace,
    PlaneFace,
    SolidDescription,
    SphereFace,
    UnrecognizedSolid,
    recognize,
)
from aquaflux.solids import Cone, Cylinder, Intersection, Sphere, Union

TURN = 2.0 * math.pi
X, Y, Z = np.eye(3)


def solid(*faces, vertices=None, extent=1.0, name="part"):
    """A description with the given faces."""
    corners = np.zeros((0, 3)) if vertices is None else np.asarray(vertices, dtype=float)
    return SolidDescription(name=name, faces=tuple(faces), vertices=corners, extent=extent)


def caps(start, stop, axis):
    """The two flat ends of a stretch of axis, outward."""
    axis = np.asarray(axis, dtype=float)
    return (
        PlaneFace(point=np.asarray(start, float), outward_normal=-axis),
        PlaneFace(point=np.asarray(stop, float), outward_normal=axis),
    )


def cylinder_parameters(body):
    """``(centre, axis, radius, half_length)`` of a Cylinder, as plain numbers."""
    return (
        np.asarray(body.centre),
        np.asarray(body.axis),
        float(body.radius),
        float(body.half_length),
    )


# ---------------------------------------------------------------------------------------------
# Curved pieces
# ---------------------------------------------------------------------------------------------


def test_a_capped_cylinder_is_the_cylinder_it_wraps():
    wall = CylinderFace(
        origin=np.array([1.0, 2.0, 3.0]), axis=Z, radius=0.5, axial_range=(0.25, 2.25),
        angle=TURN, solid_inside=True,
    )  # fmt: skip
    found = recognize(solid(wall, *caps([1, 2, 3.25], [1, 2, 5.25], Z)))
    assert isinstance(found.body, Cylinder)
    centre, axis, radius, half = cylinder_parameters(found.body)
    np.testing.assert_allclose(centre, [1.0, 2.0, 4.25], atol=1e-15)
    np.testing.assert_allclose(np.abs(axis), Z, atol=1e-15)
    assert (radius, half) == (0.5, 1.0)
    assert found.stands_alone and not found.proven


def test_a_cylinder_split_into_two_half_turns_is_one_cylinder():
    """Exporters split a periodic face; the halves must be recognized as one surface.

    The two halves are reported from different origins and with opposite axis directions, which is
    exactly what a kernel is free to do, so the merge must compare the axis line itself.
    """
    first = CylinderFace(
        origin=np.array([0.0, 0.0, 0.0]), axis=X, radius=0.2, axial_range=(0.0, 1.0),
        angle=math.pi, solid_inside=True,
    )  # fmt: skip
    second = CylinderFace(
        origin=np.array([1.0, 0.0, 0.0]), axis=-X, radius=0.2, axial_range=(0.0, 1.0),
        angle=math.pi, solid_inside=True,
    )  # fmt: skip
    found = recognize(solid(first, second, *caps([0, 0, 0], [1, 0, 0], X)))
    assert isinstance(found.body, Cylinder)
    centre, _, radius, half = cylinder_parameters(found.body)
    np.testing.assert_allclose(centre, [0.5, 0.0, 0.0], atol=1e-15)
    assert (radius, half) == (0.2, 0.5)


def test_two_half_turns_on_different_cylinders_are_not_merged():
    """Same radius, parallel axes, different lines: two partial faces, not one whole one."""
    first = CylinderFace(
        origin=np.zeros(3), axis=X, radius=0.2, axial_range=(0.0, 1.0), angle=math.pi,
        solid_inside=True,
    )  # fmt: skip
    second = CylinderFace(
        origin=np.array([0.0, 0.01, 0.0]), axis=X, radius=0.2, axial_range=(0.0, 1.0),
        angle=math.pi, solid_inside=True,
    )  # fmt: skip
    outcome = CurvedPieces().propose(solid(first, second))
    assert outcome == "none of its curved faces goes all the way round"


def test_a_lamp_with_a_hemispherical_tip_is_a_cylinder_and_a_sphere():
    wall = CylinderFace(
        origin=np.zeros(3), axis=X, radius=0.01, axial_range=(0.0, 0.8), angle=TURN,
        solid_inside=True,
    )  # fmt: skip
    tip = SphereFace(centre=np.array([0.8, 0.0, 0.0]), radius=0.01, solid_inside=True)
    base = PlaneFace(point=np.zeros(3), outward_normal=-X)
    found = recognize(solid(wall, tip, base))
    assert isinstance(found.body, Union)
    kinds = sorted(type(part).__name__ for part in found.body.bodies)
    assert kinds == ["Cylinder", "Sphere"]
    sphere = next(part for part in found.body.bodies if isinstance(part, Sphere))
    np.testing.assert_allclose(np.asarray(sphere.centre), [0.8, 0.0, 0.0])
    assert found.stands_alone


def test_a_pipe_cut_to_fit_a_vessel_is_its_cylinder_and_does_not_stand_alone():
    """The saddle end is a partial face of the VESSEL's cylinder, with the pipe outside it."""
    wall = CylinderFace(
        origin=np.array([0.05, 0.0, 0.9]), axis=-Z, radius=0.01, axial_range=(0.0, 0.86),
        angle=TURN, solid_inside=True,
    )  # fmt: skip
    saddle = CylinderFace(
        origin=np.zeros(3), axis=X, radius=0.045, axial_range=(0.04, 0.06), angle=0.43,
        solid_inside=False,
    )  # fmt: skip
    top = PlaneFace(point=np.array([0.05, 0.0, 0.9]), outward_normal=Z)
    found = recognize(solid(wall, saddle, top))
    assert isinstance(found.body, Cylinder)
    assert not found.stands_alone
    centre, _, _, half = cylinder_parameters(found.body)
    np.testing.assert_allclose(centre, [0.05, 0.0, 0.47], atol=1e-15)
    assert half == pytest.approx(0.43)


def test_a_bore_is_declined_rather_than_read_as_a_piece():
    """A whole-turn face with the solid OUTSIDE it is material removed, not a convex piece."""
    bore = CylinderFace(
        origin=np.zeros(3), axis=Z, radius=0.1, axial_range=(0.0, 1.0), angle=TURN,
        solid_inside=False,
    )  # fmt: skip
    outer = CylinderFace(
        origin=np.zeros(3), axis=Z, radius=0.2, axial_range=(0.0, 1.0), angle=TURN,
        solid_inside=True,
    )  # fmt: skip
    outcome = CurvedPieces().propose(solid(outer, bore, *caps([0, 0, 0], [0, 0, 1], Z)))
    assert isinstance(outcome, str) and "bore" in outcome


def test_a_cone_keeps_its_radii_at_the_ends_they_belong_to():
    """The canonical axis may point the other way from the kernel's, and the radii must follow."""
    wall = ConeFace(
        origin=np.array([0.0, 0.0, 2.0]), axis=-Z, axial_range=(0.0, 2.0), radii=(0.5, 1.5),
        angle=TURN, solid_inside=True,
    )  # fmt: skip
    found = recognize(solid(wall, *caps([0, 0, 0], [0, 0, 2], Z)))
    assert isinstance(found.body, Cone)
    # Radius 1.5 at z = 0 and 0.5 at z = 2, whichever way the axis is stored.
    for z, radius in ((0.0, 1.5), (2.0, 0.5)):
        height = z + (1e-3 if z == 0 else -1e-3)  # just inside the end cap
        inside = np.array([[radius - 2e-3, 0.0, height]])
        outside = np.array([[radius + 2e-3, 0.0, height]])
        assert bool(found.body.contains(jnp.asarray(inside))[0])
        assert not bool(found.body.contains(jnp.asarray(outside))[0])


def test_an_unknown_surface_is_declined_by_name():
    torus = OtherFace(kind="Torus")
    with pytest.raises(UnrecognizedSolid, match="Torus"):
        recognize(solid(torus, name="elbow"))


def test_a_refusal_names_every_rule_and_the_solid():
    wall = CylinderFace(
        origin=np.zeros(3), axis=Z, radius=0.1, axial_range=(0.0, 1.0), angle=1.0,
        solid_inside=True,
    )  # fmt: skip
    with pytest.raises(UnrecognizedSolid) as refused:
        recognize(solid(wall, name="sliver"))
    message = str(refused.value)
    assert "'sliver'" in message
    assert "CurvedPieces: none of its curved faces goes all the way round" in message
    assert "ConvexPolyhedron: not all of its faces are planar" in message


# ---------------------------------------------------------------------------------------------
# Convex polyhedra
# ---------------------------------------------------------------------------------------------

CUBE_CORNERS = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)


def cube_faces():
    """The six faces of the unit cube, outward."""
    faces = []
    for axis in np.eye(3):
        faces.append(PlaneFace(point=np.zeros(3), outward_normal=-axis))
        faces.append(PlaneFace(point=axis.copy(), outward_normal=axis))
    return faces


def test_a_convex_planar_solid_is_the_intersection_of_its_half_spaces_and_needs_no_check():
    found = recognize(solid(*cube_faces(), vertices=CUBE_CORNERS, extent=math.sqrt(3.0)))
    assert isinstance(found.body, Intersection) and len(found.body.bodies) == 6
    assert found.proven and found.stands_alone
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 1.5, (2000, 3))
    truth = np.all((points > 0.0) & (points < 1.0), axis=1)
    assert np.array_equal(np.asarray(found.body.contains(jnp.asarray(points))), truth)


def test_a_face_split_into_two_coplanar_pieces_is_one_half_space():
    halves = [*cube_faces(), PlaneFace(point=np.array([0.0, 0.5, 0.0]), outward_normal=-X)]
    found = ConvexPolyhedron().propose(solid(*halves, vertices=CUBE_CORNERS, extent=1.7))
    assert len(found.body.bodies) == 6


def test_a_planar_solid_that_is_not_convex_is_declined():
    """An L-shaped prism: one corner lies outside the plane of a face."""
    corners = np.array(
        [[x, y, z] for (x, y) in [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)] for z in (0, 1)],
        dtype=float,
    )
    faces = [
        PlaneFace(point=np.zeros(3), outward_normal=-Z),
        PlaneFace(point=np.array([0, 0, 1.0]), outward_normal=Z),
        PlaneFace(point=np.zeros(3), outward_normal=-Y),
        PlaneFace(point=np.array([2.0, 0, 0]), outward_normal=X),
        PlaneFace(point=np.array([2.0, 1, 0]), outward_normal=Y),  # the inner corner's face
        PlaneFace(point=np.array([1.0, 1, 0]), outward_normal=X),
        PlaneFace(point=np.array([1.0, 2, 0]), outward_normal=Y),
        PlaneFace(point=np.zeros(3), outward_normal=-X),
    ]
    outcome = ConvexPolyhedron().propose(solid(*faces, vertices=corners, extent=3.0))
    assert outcome == "it is bounded by planes but is not convex"
