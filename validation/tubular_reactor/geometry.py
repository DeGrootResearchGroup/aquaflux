"""A closed-vessel tubular ultraviolet reactor: ten lamps along its axis, a perforated plate at its inlet.

Every length is in metres and the reactor's axis is x. The water enters along the axis through the
inlet pipe (x < 0), passes the perforated plate, flows along the ten lamp sleeves and leaves through a
side pipe near the far end wall, through which the lamps are mounted.

Each lamp is the Sozzi & Taghipour (2006) lamp: an emitting sleeve 20 mm across, 800 mm long from the
end wall to the centre of its hemispherical tip, 35.4 W of ultraviolet-C. Only its outer surface is
modelled -- the quartz and the air inside it are not -- so it emits diffusely from that surface and
absorbs whatever lands on it.

The geometry is described once, by :class:`Reactor`, and handed out three ways:

- as exact bodies (:meth:`Reactor.water`, :meth:`Reactor.plate`, :meth:`Reactor.lamp_bodies`),
  which decide every shadow by formula -- the vessel and its pipes as the water they hold, the
  perforated plate as a slab a sight line crosses only within one hole (:class:`PerforatedPlate`),
  and each lamp as a cylinder with a hemispherical tip;
- as triangles of the lamps' surfaces (:meth:`Reactor.lamp_triangles`), every vertex on the true
  surface, which emit;
- as triangles of the walls the water touches (:func:`wall_triangles`, through gmsh's OpenCASCADE
  kernel), which reflect diffusely. The same model is written as a STEP drawing of the water.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from itertools import pairwise
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.solids import Body, Cylinder, Outside, Sphere, Union

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])


@dataclasses.dataclass(frozen=True)
class Reactor:
    """The reactor's dimensions, in metres.

    The perforated plate's holes are on a hexagonal lattice of ``hole_pitch`` whose hundred points
    nearest the axis lie within the plate; their diameter makes the holes' area exactly
    ``open_fraction`` of the plate's.
    """

    radius: float = 0.150
    length: float = 1.100
    inlet_radius: float = 0.050
    inlet_length: float = 0.250
    outlet_radius: float = 0.050
    outlet_x: float = 1.000
    outlet_reach: float = 0.400
    plate_x: float = 0.100
    plate_thickness: float = 0.006
    n_holes: int = 100
    open_fraction: float = 0.5
    hole_pitch: float = 0.026
    lamp_radius: float = 0.010
    lamp_length: float = 0.800
    lamp_power: float = 35.4
    inner_ring: tuple[float, int, float] = (0.050, 3, 90.0)
    outer_ring: tuple[float, int, float] = (0.110, 7, 90.0 + 180.0 / 7)

    # --- the lamps ---------------------------------------------------------------------------------

    @cached_property
    def lamp_centres(self) -> np.ndarray:
        """Each lamp's axis in the (y, z) plane, shape (n_lamps, 2): two rings."""
        centres = []
        for ring_radius, count, start_degrees in (self.inner_ring, self.outer_ring):
            angles = np.radians(start_degrees) + 2.0 * np.pi * np.arange(count) / count
            centres.append(ring_radius * np.column_stack([np.cos(angles), np.sin(angles)]))
        return np.concatenate(centres)

    @property
    def n_lamps(self) -> int:
        return len(self.lamp_centres)

    @property
    def lamp_names(self) -> tuple[str, ...]:
        return tuple(f"lamp{k}" for k in range(self.n_lamps))

    @property
    def tip_x(self) -> float:
        """Where each lamp's hemispherical tip is centred; the sleeve runs from here to the end wall."""
        return self.length - self.lamp_length

    def lamp_bodies(self, shrink: float) -> tuple[Union, ...]:
        """Each lamp as an exact solid, ``shrink`` thinner than the sleeve.

        The emitting triangles' centroids lie a chord's depth inside the true sleeve, and a body may
        hold no facet, so the shadow is cast by a body just inside the facets: the shadow of a lamp
        is ``shrink`` narrower on each side than the sleeve.
        """
        radius = self.lamp_radius - shrink
        # Stopped a millimetre short of the end wall, so the wall's triangles round each lamp's
        # mounting hole -- whose centroids can lie inside the sleeve's circle -- are not inside it.
        half = 0.5 * (self.lamp_length - 0.001)
        return tuple(
            Union(
                Cylinder(
                    centre=[self.tip_x + half, y, z], axis=X_AXIS, radius=radius, half_length=half
                ),
                Sphere(centre=[self.tip_x, y, z], radius=radius),
            )
            for y, z in self.lamp_centres
        )

    def lamp_distance(self, points: np.ndarray) -> np.ndarray:
        """Each point's signed distance from the nearest lamp's surface, negative inside one."""
        along = np.clip(points[:, 0], self.tip_x, self.length)
        nearest = np.full(len(points), np.inf)
        for y, z in self.lamp_centres:
            gap = np.sqrt(
                (points[:, 0] - along) ** 2 + (points[:, 1] - y) ** 2 + (points[:, 2] - z) ** 2
            )
            nearest = np.minimum(nearest, gap - self.lamp_radius)
        return nearest

    def lamp_triangles(self, sectors: int, axial_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Every lamp's emitting surface: its sleeve and hemispherical tip, every vertex on them.

        Parameters
        ----------
        sectors : int
            Triangle columns round the sleeve.
        axial_size : float
            The largest extent of a sleeve triangle along the axis; the tip is cut into rings of
            about the same arc length.

        Returns
        -------
        triangles : np.ndarray, shape (n, 3, 3)
            Wound so each faces out of its lamp, into the water.
        lamp : np.ndarray of int, shape (n,)
            Which lamp each triangle belongs to.
        """
        one = _capsule_triangles(self.lamp_radius, self.tip_x, self.length, sectors, axial_size)
        triangles, lamp = [], []
        for k, (y, z) in enumerate(self.lamp_centres):
            triangles.append(one + np.array([0.0, y, z]))
            lamp.append(np.full(len(one), k))
        return np.concatenate(triangles), np.concatenate(lamp)

    # --- the perforated plate ----------------------------------------------------------------------

    @cached_property
    def hole_radius(self) -> float:
        return self.radius * np.sqrt(self.open_fraction / self.n_holes)

    @cached_property
    def hole_centres(self) -> np.ndarray:
        """The holes' centres in the (y, z) plane, shape (n_holes, 2): the points of
        :func:`hex_lattice` nearest the axis. Refuses a pitch that leaves holes overlapping, one
        cutting the vessel wall, or a tie for the last hole."""
        pitch = self.hole_pitch
        span = int(np.ceil(self.radius / pitch)) + 2
        i, j = np.meshgrid(np.arange(-span, span + 1), np.arange(-span, span + 1))
        points = np.column_stack(hex_lattice(pitch, i.ravel(), j.ravel()))
        radius = np.hypot(points[:, 0], points[:, 1])
        order = np.argsort(radius, kind="stable")
        if radius[order[self.n_holes]] - radius[order[self.n_holes - 1]] < 1e-9:
            raise ValueError(f"the {self.n_holes} lattice points nearest the axis are not unique")
        chosen = points[order[: self.n_holes]]
        rim = self.radius - (np.hypot(chosen[:, 0], chosen[:, 1]).max() + self.hole_radius)
        if pitch <= 2.0 * self.hole_radius or rim <= 0.0:
            raise ValueError(
                f"a hole pitch of {pitch} m leaves {pitch - 2 * self.hole_radius:.4f} m between "
                f"holes and {rim:.4f} m to the wall"
            )
        return chosen

    def plate(self) -> PerforatedPlate:
        """The perforated plate as an exact body."""
        ring = float(np.hypot(*self.hole_centres.T).max())
        return PerforatedPlate(
            near_x=self.plate_x,
            far_x=self.plate_far_x,
            pitch=self.hole_pitch,
            hole_radius=float(self.hole_radius),
            # Halfway to the next lattice point out, so the hundred are in and no other is.
            ring_radius=ring + 1e-6,
        )

    @property
    def plate_far_x(self) -> float:
        return self.plate_x + self.plate_thickness

    # --- the water ---------------------------------------------------------------------------------

    @cached_property
    def _regions(self) -> tuple[Cylinder, ...]:
        """Convex pieces whose union is the water, the plate and lamps aside: the chamber, and the
        two pipes carried into it so that each overlaps it."""
        overlap = 0.001
        half = 0.5 * self.length
        regions = [
            Cylinder(centre=[half, 0, 0], axis=X_AXIS, radius=self.radius, half_length=half),
            Cylinder(
                centre=[0.5 * (overlap - self.inlet_length), 0, 0],
                axis=X_AXIS,
                radius=self.inlet_radius,
                half_length=0.5 * (self.inlet_length + overlap),
            ),
        ]
        # The outlet pipe starts inside the chamber, where its whole section is water.
        start = np.sqrt(self.radius**2 - self.outlet_radius**2) - 0.01
        regions.append(
            Cylinder(
                centre=[self.outlet_x, 0.5 * (start + self.outlet_reach), 0],
                axis=Y_AXIS,
                radius=self.outlet_radius,
                half_length=0.5 * (self.outlet_reach - start),
            )
        )
        return tuple(regions)

    def water(self, tolerance: float) -> Outside:
        """Everything outside the vessel and its pipes."""
        return Outside(*self._regions, tolerance=tolerance)

    def in_water(self, points: np.ndarray, margin: float) -> np.ndarray:
        """Whether each point (n, 3) is in the water at least ``margin`` from every wall and lamp."""
        inside = np.zeros(len(points), bool)
        for region in self._regions:
            axis = np.asarray(region.axis)
            offset = points - np.asarray(region.centre)
            along = offset @ axis
            across = np.linalg.norm(offset - along[:, None] * axis, axis=1)
            inside |= (np.abs(along) <= float(region.half_length) - margin) & (
                across <= float(region.radius) - margin
            )
        in_plate = np.asarray(self.plate().contains(jnp.asarray(points), margin=margin))
        return inside & ~in_plate & (self.lamp_distance(points) >= margin)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([-self.inlet_length, -self.radius, -self.radius])
        high = np.array([self.length, self.outlet_reach, self.radius])
        return low, high

    def summary(self) -> dict:
        """The numbers a reader of the results needs, for the run record."""
        plate_area = np.pi * self.radius**2
        holes = self.n_holes * np.pi * self.hole_radius**2
        between = np.linalg.norm(self.lamp_centres[:, None] - self.lamp_centres[None], axis=2)
        np.fill_diagonal(between, np.inf)
        return {
            "vessel_inner_diameter_m": 2 * self.radius,
            "vessel_length_m": self.length,
            "inlet_pipe_diameter_m": 2 * self.inlet_radius,
            "outlet_pipe_diameter_m": 2 * self.outlet_radius,
            "plate_x_m": [self.plate_x, self.plate_far_x],
            "holes": self.n_holes,
            "hole_diameter_m": 2 * self.hole_radius,
            "hole_pitch_m": self.hole_pitch,
            "open_area_fraction": holes / plate_area,
            "lamps": self.n_lamps,
            "lamp_power_W": self.lamp_power,
            "sleeve_diameter_m": 2 * self.lamp_radius,
            "lamp_length_m": self.lamp_length,
            "lamp_centres_yz_m": self.lamp_centres.round(6).tolist(),
            "closest_sleeve_gap_m": float(between.min() - 2 * self.lamp_radius),
            "closest_sleeve_to_wall_m": float(
                self.radius - np.hypot(*self.lamp_centres.T).max() - self.lamp_radius
            ),
        }


def hex_lattice(pitch, i, j):
    """The point (y, z) of a hexagonal lattice of spacing ``pitch`` with indices ``i``, ``j``: rows
    along y, each odd row shifted half a pitch, and no point on the axis."""
    y = pitch * (i + 0.5 * (j % 2) + 0.5)
    z = pitch * np.sqrt(3.0) / 2.0 * j
    return y, z


class PerforatedPlate(Body):
    """A plate normal to x, from ``near_x`` to ``far_x``, with a round hole at each point of
    :func:`hex_lattice` within ``ring_radius`` of the axis.

    A segment crossing the plate is clear exactly when the part of it inside the plate lies in one
    hole: the holes are convex, so that is both ends of the part lying in the same hole, and the
    hole holding a point is the lattice point nearest it (each hole is narrower than half a pitch,
    so it lies in its point's own lattice cell). One lookup per segment, whatever the number of
    holes. The plate's own extent across the vessel is left to the vessel's body.
    """

    near_x: float = eqx.field(static=True)
    far_x: float = eqx.field(static=True)
    pitch: float = eqx.field(static=True)
    hole_radius: float = eqx.field(static=True)
    ring_radius: float = eqx.field(static=True)
    #: How far inside the plate a point must be before :meth:`contains` calls it embedded: a wall
    #: triangle's centroid on the plate's face sits a rounding to either side of it.
    tolerance: float = eqx.field(static=True, default=1e-9)

    traceable = True

    def _hole_of(self, y, z):
        """The centre of the hole each point (y, z) lies in, and whether it lies in one."""
        row = self.pitch * np.sqrt(3.0) / 2.0
        best = None
        for shift in (-1, 0, 1):
            j = jnp.round(z / row) + shift
            odd = jnp.mod(j, 2.0)
            i = jnp.round(y / self.pitch - 0.5 * odd - 0.5)
            cy = self.pitch * (i + 0.5 * odd + 0.5)
            cz = row * j
            gap = (y - cy) ** 2 + (z - cz) ** 2
            if best is None:
                best = (gap, cy, cz)
            else:
                nearer = gap < best[0]
                best = tuple(
                    jnp.where(nearer, new, old)
                    for new, old in zip((gap, cy, cz), best, strict=True)
                )
        gap, cy, cz = best
        holed = (gap <= self.hole_radius**2) & (cy**2 + cz**2 <= self.ring_radius**2)
        return cy, cz, holed

    def blocks(self, origin, target, min_distance):
        origin = jnp.asarray(origin, dtype=float)
        target = jnp.asarray(target, dtype=float)
        along = target - origin
        length = jnp.sqrt(jnp.sum(along**2, axis=-1))
        margin = jnp.asarray(min_distance) / jnp.where(length > 0, length, 1.0)
        dx = along[..., 0]
        safe = jnp.where(dx == 0.0, 1.0, dx)
        t_near = (self.near_x - origin[..., 0]) / safe
        t_far = (self.far_x - origin[..., 0]) / safe
        enter = jnp.where(dx == 0.0, -jnp.inf, jnp.minimum(t_near, t_far))
        leave = jnp.where(dx == 0.0, jnp.inf, jnp.maximum(t_near, t_far))
        inside_slab = (origin[..., 0] > self.near_x) & (origin[..., 0] < self.far_x)
        enter = jnp.where((dx == 0.0) & ~inside_slab, jnp.inf, enter)
        # The part of the segment in the plate, less the margin at both ends.
        start = jnp.maximum(enter, margin)
        stop = jnp.minimum(leave, 1.0 - margin)
        crosses = stop > start
        a = origin + start[..., None] * along
        b = origin + stop[..., None] * along
        ay, az, a_holed = self._hole_of(a[..., 1], a[..., 2])
        by, bz, b_holed = self._hole_of(b[..., 1], b[..., 2])
        through = a_holed & b_holed & (ay == by) & (az == bz)
        return crosses & ~through

    def contains(self, position, margin: float | None = None):
        """Whether each point is in the plate's metal; ``margin`` grows the metal by that much
        (unset, it shrinks by the tolerance)."""
        grow = -self.tolerance if margin is None else margin
        position = jnp.asarray(position, dtype=float)
        x = position[..., 0]
        in_slab = (x > self.near_x - grow) & (x < self.far_x + grow)
        cy, cz, _ = self._hole_of(position[..., 1], position[..., 2])
        gap = (position[..., 1] - cy) ** 2 + (position[..., 2] - cz) ** 2
        in_hole = (gap < (self.hole_radius - grow) ** 2) & (cy**2 + cz**2 <= self.ring_radius**2)
        return in_slab & ~in_hole

    def clearance(self, position):
        """Two witnesses: wholly on the near side of the plate, or wholly on the far side."""
        position = jnp.asarray(position, dtype=float)
        x = position[..., 0]
        slack = 1e-10 * (jnp.abs(x) + abs(self.far_x))
        return jnp.stack([x - self.near_x + slack, self.far_x - x + slack], axis=-1)


def _capsule_triangles(radius, tip_x, end_x, sectors, axial_size) -> np.ndarray:
    """A sleeve along x from ``tip_x`` to ``end_x`` closed at ``tip_x`` by a hemisphere, as
    triangles facing outward; the open end is the end wall's."""
    angles = 2.0 * np.pi * np.arange(sectors + 1) / sectors
    angles[-1] = 0.0  # close the ring on the first vertex exactly
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    stations = np.linspace(tip_x, end_x, max(1, int(np.ceil((end_x - tip_x) / axial_size))) + 1)
    rings = [np.column_stack([np.full(sectors + 1, x), radius * ring]) for x in stations]
    # The tip: rings of latitude from the equator to the pole, the pole a fan.
    n_lat = max(sectors // 4, int(np.ceil(0.5 * np.pi * radius / axial_size)))
    cap = []
    for polar in np.linspace(0.5 * np.pi, 0.0, n_lat + 1)[1:-1]:
        cap.append(
            np.column_stack(
                [
                    np.full(sectors + 1, tip_x - radius * np.cos(polar)),
                    radius * np.sin(polar) * ring,
                ]
            )
        )
    loops = cap[::-1] + rings  # from the ring nearest the pole to the end wall
    triangles = []
    for a, b in pairwise(loops):
        for s in range(sectors):
            triangles.append([a[s], b[s], b[s + 1]])
            triangles.append([a[s], b[s + 1], a[s + 1]])
    pole = np.array([tip_x - radius, 0.0, 0.0])
    first = loops[0]
    for s in range(sectors):
        triangles.append([pole, first[s], first[s + 1]])
    triangles = np.asarray(triangles)
    return _facing_out(triangles, tip_x)


def _facing_out(triangles: np.ndarray, tip_x: float) -> np.ndarray:
    """Reverse each triangle whose normal points towards the capsule's axis (or tip centre)."""
    centroid = triangles.mean(axis=1)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    axis_point = np.column_stack(
        [np.maximum(centroid[:, 0], tip_x), np.zeros(len(centroid)), np.zeros(len(centroid))]
    )
    inward = np.einsum("ij,ij->i", normal, centroid - axis_point) < 0.0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    return triangles


def wall_triangles(
    reactor: Reactor, wall_size: float, step_file: Path | None = None, per_turn: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """The walls the water touches, as triangles facing into the water, through gmsh.

    The water is built as solids -- the chamber and its two pipes fused, less the perforated plate
    and the lamps -- and its boundary meshed in two dimensions. The lamps' surfaces and the pipes'
    open ends are left out: the lamps emit and are described separately, and light leaving through
    an open end leaves the reactor.

    Parameters
    ----------
    reactor : Reactor
    wall_size : float
        The longest triangle edge.
    step_file : Path, optional
        Where to write the water as a STEP solid.
    per_turn : int
        Triangle edges per full turn of a curved edge: each hole in the plate is a polygon of this
        many sides.

    Returns
    -------
    triangles : np.ndarray, shape (n, 3, 3)
    part : np.ndarray of int, shape (n,)
        0 for the vessel and pipe walls, 1 for the perforated plate (its faces and holes).
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("tubular_reactor")
        occ = gmsh.model.occ
        r = reactor
        vessel = occ.addCylinder(0, 0, 0, r.length, 0, 0, r.radius)
        inlet = occ.addCylinder(-r.inlet_length, 0, 0, r.inlet_length + 0.01, 0, 0, r.inlet_radius)
        outlet = occ.addCylinder(r.outlet_x, 0, 0, 0, r.outlet_reach, 0, r.outlet_radius)
        chamber, _ = occ.fuse([(3, vessel)], [(3, inlet), (3, outlet)])
        plate = occ.addCylinder(r.plate_x, 0, 0, r.plate_thickness, 0, 0, r.radius)
        holes = [
            (
                3,
                occ.addCylinder(
                    r.plate_x - 0.001, y, z, r.plate_thickness + 0.002, 0, 0, r.hole_radius
                ),
            )
            for y, z in r.hole_centres
        ]
        plate_solid, _ = occ.cut([(3, plate)], holes)
        lamps = []
        for y, z in r.lamp_centres:
            sleeve = occ.addCylinder(r.tip_x, y, z, r.lamp_length + 0.01, 0, 0, r.lamp_radius)
            tip = occ.addSphere(r.tip_x, y, z, r.lamp_radius)
            lamps.extend([(3, sleeve), (3, tip)])
        water, _ = occ.cut(chamber, plate_solid + lamps)
        occ.synchronize()
        if step_file is not None:
            gmsh.write(str(step_file))

        gmsh.option.setNumber("Mesh.MeshSizeMax", wall_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 5e-3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", per_turn)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        boundary = gmsh.model.getBoundary(water, oriented=False, combined=True)
        kept, parts = [], []
        for dim, tag in boundary:
            low = np.array(gmsh.model.getBoundingBox(dim, tag)[:3])
            high = np.array(gmsh.model.getBoundingBox(dim, tag)[3:])
            if _is_lamp(r, low, high) or _is_opening(r, low, high):
                continue
            kept.append(tag)
            on_plate = low[0] >= r.plate_x - 1e-6 and high[0] <= r.plate_far_x + 1e-6
            parts.append(1 if on_plate else 0)
        gmsh.model.mesh.generate(2)
        triangles, part = [], []
        for tag, which in zip(kept, parts, strict=True):
            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            position = dict(zip(node_tags, coords.reshape(-1, 3), strict=True))
            types, _, element_nodes = gmsh.model.mesh.getElements(2, tag)
            for kind, nodes in zip(types, element_nodes, strict=True):
                if kind != 2:  # three-node triangles only
                    continue
                for a, b, c in nodes.reshape(-1, 3):
                    triangles.append([position[a], position[b], position[c]])
                    part.append(which)
        triangles = np.asarray(triangles)
        part = np.asarray(part)
    finally:
        gmsh.finalize()
    return _facing_water(reactor, triangles), part


def _is_lamp(r: Reactor, low, high) -> bool:
    """A face lying within a sleeve's reach of a lamp axis and not on the end wall."""
    if high[0] - low[0] < 1e-9 and abs(low[0] - r.length) < 1e-6:
        return False
    middle = 0.5 * (low[1:] + high[1:])
    near = np.min(np.hypot(*(r.lamp_centres - middle).T))
    return near < 1e-6 and (high[1:] - low[1:]).max() <= 2 * r.lamp_radius + 1e-6


def _is_opening(r: Reactor, low, high) -> bool:
    """The inlet pipe's end disc and the outlet pipe's."""
    inlet = abs(low[0] + r.inlet_length) < 1e-6 and abs(high[0] + r.inlet_length) < 1e-6
    outlet = abs(low[1] - r.outlet_reach) < 1e-6 and abs(high[1] - r.outlet_reach) < 1e-6
    return inlet or outlet


def _facing_water(reactor: Reactor, triangles: np.ndarray) -> np.ndarray:
    """Turn each triangle to face into the water: a point a little along its normal must be water."""
    centroid = triangles.mean(axis=1)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    # A step must be further than the triangle's chord lies from the curved wall it stands for,
    # and less than the thinnest solid (the plate's ligaments between holes): the smallest of a
    # few that decides.
    behind = np.zeros(len(triangles), bool)
    undecided = np.ones(len(triangles), bool)
    for step in (5e-4, 1.5e-3, 3e-3, 6e-3, 1.2e-2):
        ahead = reactor.in_water(centroid + step * normal, margin=0.0)
        back = reactor.in_water(centroid - step * normal, margin=0.0)
        decided = undecided & (ahead != back)
        behind[decided] = back[decided]
        undecided &= ~decided
    if np.any(undecided):
        raise ValueError(
            f"{int(undecided.sum())} wall triangles have water on both sides or neither"
        )
    flip = behind
    triangles = triangles.copy()
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return triangles
