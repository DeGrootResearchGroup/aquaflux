"""Does Netgen fill a supplied closed surface mesh with tets and leave that surface exactly as given?

This is the conformal gap fill the mesher needs: the faces bounding the gap (prism-layer caps,
pyramid sides on a hex core) are built by aquaflux, and Netgen may only add interior points. No
CAD is involved, and the process never imports OCP.

Each arm is a unit box with ``N`` x ``N`` cells per face, wound counter-clockwise seen from outside
the domain unless flipped. One line per arm: the volume elements by vertex count (4 = tet), whether
the boundary came back unchanged, and whether any supplied point moved.

Usage::

    python validation/meshing_netgen/netgen_fill.py
"""

import numpy as np
from netgen.meshing import Element2D, FaceDescriptor, MeshPoint
from netgen.meshing import Mesh as NetgenMesh

N = 4
MAXH = 0.3
ARMS = {
    "box, triangles": {},
    "box, triangles wound inward": {"flip": True},
    "box, top face as quadrilaterals": {"quads": True},
    "box with a box-shaped hole, triangles": {"hole": True},
}


class _Surface:
    """A Netgen mesh holding only boundary elements, with each supplied point recorded."""

    def __init__(self):
        self.mesh = NetgenMesh(dim=3)
        self.descriptor = self.mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
        self.points = {}

    def point(self, xyz):
        key = tuple(np.round(xyz, 12))
        if key not in self.points:
            self.points[key] = self.mesh.Add(MeshPoint(tuple(float(c) for c in xyz)))
        return self.points[key]

    def box(self, lo, hi, *, inward, quads_on_top=False):
        """Six faces of the box [lo, hi]^3; ``inward`` winds them into the box (a hole's walls)."""
        ticks = np.linspace(lo, hi, N + 1)
        for axis in range(3):
            a, b = (axis + 1) % 3, (axis + 2) % 3  # a x b = +axis
            for side in (lo, hi):
                for i in range(N):
                    for j in range(N):
                        ring = []
                        for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            xyz = np.zeros(3)
                            xyz[axis], xyz[a], xyz[b] = side, ticks[i + di], ticks[j + dj]
                            ring.append(self.point(xyz))
                        if (side == lo) != inward:
                            ring = ring[::-1]
                        if quads_on_top and axis == 2 and side == hi:
                            self.mesh.Add(Element2D(self.descriptor, ring))
                        else:
                            self.mesh.Add(Element2D(self.descriptor, ring[:3]))
                            self.mesh.Add(Element2D(self.descriptor, [ring[0], ring[2], ring[3]]))


def run(flip=False, quads=False, hole=False) -> str:
    surface = _Surface()
    surface.box(0.0, 1.0, inward=flip, quads_on_top=quads)
    if hole:
        surface.box(0.35, 0.65, inward=not flip)
    supplied = len(surface.mesh.Elements2D())
    surface.mesh.GenerateVolumeMesh(maxh=MAXH)
    kinds: dict[int, int] = {}
    for element in surface.mesh.Elements3D():
        kinds[len(element.vertices)] = kinds.get(len(element.vertices), 0) + 1
    moved = sum(
        np.linalg.norm(np.array(surface.mesh[index].p) - np.array(xyz)) > 1e-12
        for xyz, index in surface.points.items()
    )
    unchanged = len(surface.mesh.Elements2D()) == supplied
    return f"elements {kinds or 'none'}; boundary unchanged {unchanged}; points moved {moved}"


def main() -> None:
    for name, arm in ARMS.items():
        print(f"{name:40s} {run(**arm)}", flush=True)


if __name__ == "__main__":
    main()
