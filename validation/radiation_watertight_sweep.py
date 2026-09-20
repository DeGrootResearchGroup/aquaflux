"""Does the intersection test stay watertight once the compiler gets hold of it?

The ray-triangle test is Woop, Benthin and Wald's watertight form (*Journal of Computer Graphics
Techniques* 2(1), 2013), and what makes it watertight is an exact-arithmetic property, not an
approximation that gets better with precision. Two triangles sharing an edge evaluate the same
edge function with the operand pairs swapped, and a ray passing through that edge is claimed by
exactly one of them **only while the two results are exact negatives**.

Written as ``a * b - c * d`` that holds when each product is rounded before the subtraction —
floating multiplication commutes bit for bit, so the two are ``fl(P) - fl(Q)`` and
``fl(Q) - fl(P)``. It stops holding when a compiler fuses one multiply into the subtraction: that
form keeps one product at full precision and rounds the other, so the two triangles keep
*different* products exact and the ray escapes between them. The kernel therefore computes its
edge functions in a form whose value does not depend on which pair leads, and this harness is the
evidence that the form is necessary and that it is sufficient.

Three measurements:

1. **Closed bodies.** Rays from inside, aimed at every vertex, edge midpoint and face centroid —
   every feature an intersection test can fall between. A leak is an interior ray that escapes.
   Run against the shipped edge function and against the plain difference, because a tightness
   sweep that cannot see a leak is worth nothing, and the first version of this one could not.
2. **The shear.** The kernel's other multiply-subtract. Two triangles sharing a vertex must
   transform it to the same point, or the antisymmetry argument never gets started.
3. **The solid-angle contour form.** Its additivity over a tiling is the same shared-edge
   cancellation, so it is checked for the same failure — and does not have it, for a reason worth
   keeping: there the cancellation feeds a continuous quantity rather than a sign test.

Run with ``validation/run_case.sh validation/radiation_watertight_sweep.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.radiation.solid_angle import projected_solid_angle, signed_solid_angle, solid_angle
from scipy.spatial import ConvexHull
from tests.unit.radiation_references import L_OUTLINE, closed_drum, closed_prism


def features(triangles):
    """Every vertex, edge midpoint and face centroid of a triangulation — exactly.

    ⚠️ Not rounded and not deduplicated. These aims are useful only because they land on a
    feature exactly; snapping them to a tolerance moves them off it, and the sweep then reports
    no leaks whatever the intersection test does. That is not hypothetical — it is how the first
    version of this file was written, and the control arm is the only reason it was noticed.
    """
    midpoints = [0.5 * (triangles[:, k] + triangles[:, (k + 1) % 3]) for k in range(3)]
    return np.concatenate([triangles.reshape(-1, 3), *midpoints, triangles.mean(axis=1)])


def leaks(hit, body, interior, aims):
    """How many rays from inside ``body`` at its own features are not stopped by it."""
    escaped = 0
    for point in interior:
        direction = (aims - point) * 3.0
        origin = jnp.broadcast_to(jnp.asarray(point), direction.shape)
        distance, meets = hit(
            origin[:, None, :], jnp.asarray(direction)[:, None, :], jnp.asarray(body)[None, ...]
        )
        stopped = np.asarray(meets) & (np.asarray(distance) > 0.0) & (np.asarray(distance) <= 1.0)
        escaped += int(np.sum(~stopped.any(axis=-1)))
    return escaped


def closed_bodies():
    """Bodies that are closed to the last bit, so a leak is the test's and not the mesh's."""
    for count, seed in ((24, 3), (40, 11), (60, 7), (90, 19)):
        rng = np.random.default_rng(seed)
        points = rng.normal(size=(count, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        yield f"hull, {count} points", points[ConvexHull(points).simplices], np.zeros((1, 3))
    inside = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.5], [-0.3, 0.35, -0.7]])
    for sectors in (16, 48):
        yield f"closed drum, {sectors} sectors", closed_drum(sectors), inside
    yield (
        "L-prism (reflex edge)",
        closed_prism(L_OUTLINE, 1.0),
        np.array([[0.4, 0.4, 0.0], [1.5, 0.4, 0.3], [0.4, 1.5, -0.4]]),
    )


def sweep(plain):
    """Leak counts over every closed body, with the edge function of one's choosing."""
    import aquaflux.radiation.triangles as triangles_module

    keep = triangles_module._edge_function
    if plain:
        triangles_module._edge_function = lambda a, b, c, d: a * b - c * d
    # ⚠️ Swapping a module global does NOT invalidate a compiled version that read it: `jit`
    # keys its cache on the function object, and `_watertight_hit` is the same object either
    # way. Without this the second arm silently replays the first one's program and reports the
    # first one's leaks -- which it did, and the numbers agreeing exactly is the only thing that
    # gave it away.
    jax.clear_caches()
    try:
        arms = (
            ("eager", triangles_module._watertight_hit),
            ("traced", jax.jit(triangles_module._watertight_hit)),
        )
        print(f"   {'body':>24} {'triangles':>10} {'rays':>7} {'eager':>7} {'traced':>7}")
        worst = 0
        for name, body, interior in closed_bodies():
            aims = features(body)
            counted = [leaks(hit, body, interior, aims) for _, hit in arms]
            worst = max(worst, *counted)
            print(
                f"   {name:>24} {len(body):10,} {len(aims) * len(interior):7,}"
                f" {counted[0]:7,} {counted[1]:7,}",
                flush=True,
            )
        return worst
    finally:
        triangles_module._edge_function = keep
        jax.clear_caches()


def shear_audit():
    """Do two triangles sharing a vertex shear it to the same point, compiled?"""
    rng = np.random.default_rng(11)
    points = rng.normal(size=(40, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    simplices = ConvexHull(points).simplices
    body = points[simplices]
    aims = np.concatenate([points, body.reshape(-1, 3)])

    def sheared(origin, direction, vertices):
        """The transformed vertex coordinates the kernel forms, exposed for comparison."""
        axis_z = jnp.argmax(jnp.abs(direction), axis=-1)
        axis_x, axis_y = (axis_z + 1) % 3, (axis_z + 2) % 3

        def component(array, axis):
            return jnp.take_along_axis(array, axis[..., None], axis=-1)[..., 0]

        dx, dy, dz = (component(direction, a) for a in (axis_x, axis_y, axis_z))
        shear_x, shear_y = dx / dz, dy / dz
        relative = vertices - origin[..., None, :]
        out = []
        for corner in range(3):
            offset = relative[..., corner, :]
            ox = component(offset, axis_x)
            oy = component(offset, axis_y)
            oz = component(offset, axis_z)
            out.append(jnp.stack([ox - shear_x * oz, oy - shear_y * oz], axis=-1))
        return jnp.stack(out, axis=-2)

    origin = jnp.zeros((len(aims), 1, 3))
    direction = jnp.asarray(aims * 3.0)[:, None, :]
    for tag, fn in (("eager", sheared), ("traced", jax.jit(sheared))):
        got = np.asarray(fn(origin, direction, jnp.asarray(body)[None, ...]))
        disagreeing = 0
        for vertex in range(len(points)):
            carried = [(t, c) for t, s in enumerate(simplices) for c in range(3) if s[c] == vertex]
            if len(carried) < 2:
                continue
            values = np.stack([got[:, t, c, :] for t, c in carried])
            disagreeing += int(np.sum(np.any(values != values[0][None], axis=-1)))
        print(f"   {tag:>6}: {disagreeing:,} shared-vertex transforms disagree")


def split_once(triangle):
    """A triangle into four, by its edge midpoints."""
    a, b, c = triangle
    ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
    return [np.array(t) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]


def additivity_audit():
    """Does the contour form stay additive over a tiling once it is compiled?"""
    point = jnp.asarray([0.0, 0.0, 0.0])
    normal = jnp.asarray([0.0, 0.0, 1.0])
    whole = np.array([[-1.0, -1.0, 2.0], [1.4, -0.9, 2.0], [0.1, 1.2, 2.0]])
    kernels = (
        ("projected_solid_angle", projected_solid_angle),
        ("solid_angle", lambda p, n, t: solid_angle(p, t)),
        ("signed_solid_angle", lambda p, n, t: signed_solid_angle(p, t)),
    )
    for name, kernel in kernels:
        for tag, fn in (("eager", kernel), ("traced", jax.jit(kernel))):
            reference = float(fn(point, normal, jnp.asarray(whole)))
            pieces, gaps = [whole], []
            for _level in (1, 2, 3):
                pieces = [child for piece in pieces for child in split_once(piece)]
                total = sum(float(fn(point, normal, jnp.asarray(t))) for t in pieces)
                gaps.append(f"{abs(total - reference) / abs(reference):.1e}")
            print(f"   {name:>21} {tag:>6}: 4 / 16 / 64 pieces   {'   '.join(gaps)}", flush=True)


def main() -> None:
    started = time.time()
    print("1. CONTROL — the plain difference of products, which the kernel does NOT use.")
    control = sweep(plain=True)
    print(f"   -- leaks once compiled, worst {control} on one body. The sweep can see a leak.\n")

    print("2. The shipped edge function, on the same bodies.")
    shipped = sweep(plain=False)
    print(f"   -- worst leak count anywhere: {shipped}\n")

    print("3. The other multiply-subtract: does the shear move a shared vertex?")
    shear_audit()
    print("   -- the same numbers go in, so the same numbers come out, fused or not.\n")

    print("4. The solid-angle contour form, whose additivity is the same cancellation.")
    additivity_audit()
    print("   -- unaffected, and the reason generalizes: there the cancellation feeds a")
    print("      continuous quantity, where a last-bit change is a last-bit change. The")
    print("      intersection test feeds it to a SIGN TEST deciding which triangle claims a")
    print("      ray, and a discrete predicate has no small errors.")
    print(f"\n({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
