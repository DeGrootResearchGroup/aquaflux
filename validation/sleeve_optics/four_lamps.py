"""The square of four sleeved lamps the rim harnesses share: arcs, sleeves and water.

Four lamps 50 mm apart in 95 % UVT water, each an arc of radius ``tracer.Scene``'s inside a quartz
sleeve holding an air gap, as ``check_array.py`` traces them. ``scene(sectors, slices)`` returns the
arcs as one surface set (each lamp emitting 1 W per metre) and the media.
"""

import jax.numpy as jnp
import numpy as np
from aquaflux.radiation import Surfaces, UniformAbsorption
from aquaflux.radiation.refraction import Media, Transparent
from aquaflux.solids import Cylinder
from check_refraction import graded_tube
from tracer import AIR, QUARTZ, WATER
from tracer import Scene as Traced

PITCH = 0.025
ABSORPTION = -np.log(0.95) / 0.01
CENTRES = PITCH * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def sliced(triangles, parts):
    """Each triangle split into ``parts**2`` by an even lattice on its own corners, winding kept."""
    if parts == 1:
        return triangles
    out = []
    for first, second, third in triangles:
        u, v = (second - first) / parts, (third - first) / parts
        for i in range(parts):
            for j in range(parts - i):
                a = first + i * u + j * v
                out.append([a, a + u, a + v])
                if j < parts - i - 1:
                    out.append([a + u, a + u + v, a + v])
    return np.asarray(out)


def scene(sectors, slices=1):
    """The four arcs, ``sectors`` round each and each triangle in ``slices**2``, and the media."""
    traced = Traced(centres=CENTRES, wall=0.1, absorption=ABSORPTION, neighbours="optics")
    arc = sliced(graded_tube(traced.arc, 1.5, sectors), slices)
    triangles = np.concatenate([arc + np.array([x, y, 0.0]) for x, y in CENTRES])
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    lamps = Surfaces.from_triangles(jnp.asarray(triangles), emission=3.0 / (area.sum() / 4))
    sleeves = tuple(
        Transparent(
            Cylinder([x, y, 0.0], [0, 0, 1], traced.outer, 1.75),
            QUARTZ,
            inside=(Transparent(Cylinder([x, y, 0.0], [0, 0, 1], traced.inner, 1.75), AIR),),
        )
        for x, y in CENTRES
    )
    return lamps, Media(WATER, sleeves, UniformAbsorption(ABSORPTION))
