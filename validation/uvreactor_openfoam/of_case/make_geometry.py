#!/usr/bin/env python3
"""Build snappyHexMesh STL geometry from the SozziTaghipour.step CAD model.

Source: SozziTaghipour.step (OnShape export, four solids, mm units, body axis
along +y). Produces four watertight ASCII STL files under
constant/triSurface/ corresponding to the patch names that
snappyHexMeshDict expects:

    bodyWall.stl   - reactor body wall, end caps, and pipe walls
    lampWall.stl   - lamp+sleeve outer surface (cylinder + hemispherical tip
                     + base disc)
    inlet.stl      - open face at the upstream end of the inlet pipe
    outlet.stl     - open face at the downstream end of the outlet pipe

Operations performed on the STEP geometry (gmsh + OpenCASCADE backend):

  1. Boolean: fluid = (body union inletPipe union outletPipe) - lamp.
     This yields a single watertight surface for the fluid region with
     proper holes at the pipe attachments.

  2. Classify each face of the resulting fluid solid by location:
       - inlet  = planar disc at the largest x in the model
       - outlet = planar disc at the largest z in the model
       - lampWall = faces whose bounding cylinder (around the body axis)
                    has radius <= R_LAMP + small_tol
       - bodyWall = everything else

  3. Coordinate transform applied at STL write time:
       (x_step, y_step, z_step) [mm]  ->  (y_step/1000, x_step/1000, z_step/1000) [m]
     so the body axis aligns with +x (matching this case's convention).

The STEP file dimensions are taken from Sozzi & Taghipour 2006 (Section 2.2,
Table 1, L-shape geometry with lamp holder).
"""

import math
import os
import sys

import gmsh


# Geometric constants used for surface classification (mm in STEP coords).
R_LAMP_MM = 10.0
R_BODY_MM = 44.5
R_PIPE_MM = 9.55
TOL_MM = 0.5


HERE = os.path.dirname(os.path.abspath(__file__))
STEP = os.path.join(HERE, "SozziTaghipour.step")
OUTDIR = os.path.join(HERE, "constant", "triSurface")


# ---------------------------------------------------------------------------
# STL emission with coordinate transform
# ---------------------------------------------------------------------------

def transform(p_mm):
    """STEP (x, y, z) [mm] -> case (y/1000, x/1000, z/1000) [m].

    Swaps x and y so the body axis (originally +y in the STEP) aligns with
    +x in the case, then scales mm to m.
    """
    x, y, z = p_mm
    return (y*1e-3, x*1e-3, z*1e-3)


def _normal(v0, v1, v2):
    e0 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e1 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    n = (
        e0[1]*e1[2] - e0[2]*e1[1],
        e0[2]*e1[0] - e0[0]*e1[2],
        e0[0]*e1[1] - e0[1]*e1[0],
    )
    nm = math.sqrt(sum(c*c for c in n))
    if nm == 0:
        return (0.0, 0.0, 0.0)
    return (n[0]/nm, n[1]/nm, n[2]/nm)


def write_ascii_stl(path, name, triangles_mm):
    """triangles_mm: list of ((x0,y0,z0), (x1,y1,z1), (x2,y2,z2)) in mm."""
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for v0, v1, v2 in triangles_mm:
            t0 = transform(v0)
            t1 = transform(v1)
            t2 = transform(v2)
            nx, ny, nz = _normal(t0, t1, t2)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for v in (t0, t1, t2):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

def classify_surface(mesh_size, surface_tag):
    """Return one of {'inlet', 'outlet', 'lampWall', 'bodyWall'}."""
    bounds = gmsh.model.getBoundingBox(2, surface_tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bounds

    # The inlet face is the planar disc at the maximum y in the model
    # (far end of the inlet pipe, originally at y=1739 mm in STEP coords).
    if ymin > 1700.0 and ymax > 1700.0:
        return "inlet"

    # The outlet face is the planar disc at the maximum z in the model
    # (top of the outlet pipe, originally at z=894.5 mm).
    if zmin > 800.0 and zmax > 800.0:
        return "outlet"

    # lampWall: the only surfaces of the fluid solid that lie entirely inside
    # the cylinder of radius R_LAMP+tol around the body axis (y in STEP) AND
    # whose y-extent stays within the body (y in [0, 889] mm). The latter
    # check excludes the inlet pipe wall, which is also a thin cylinder
    # concentric with the y-axis but at y > 889 mm.
    rmax = max(abs(xmin), abs(xmax), abs(zmin), abs(zmax))
    if rmax < R_LAMP_MM + TOL_MM and ymax <= 889.0 + TOL_MM:
        return "lampWall"

    return "bodyWall"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(STEP):
        sys.stderr.write(f"STEP file not found: {STEP}\n")
        return 1
    os.makedirs(OUTDIR, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Load STEP. The OpenCASCADE backend supports boolean operations.
    gmsh.merge(STEP)
    gmsh.model.occ.synchronize()

    # Identify the four volumes by their bounding box.
    # Convention from inspection of the STEP file:
    #   vol 1 = lamp        bounds y[0, 810]      (small radius)
    #   vol 2 = outlet pipe bounds y[38.1, 57.2]  z[43.5, 894.5]
    #   vol 3 = inlet pipe  bounds y[889, 1739]   (long, +y)
    #   vol 4 = body        bounds y[0, 889]      (large radius)
    vols = {}
    for dim, tag in gmsh.model.getEntities(3):
        b = gmsh.model.getBoundingBox(dim, tag)
        xmax_abs = max(abs(b[0]), abs(b[3]))
        zmax_abs = max(abs(b[2]), abs(b[5]))
        # Per-axis extent (not sqrt sum), to test cylinder radius
        # consistently with a cylinder centred on the y-axis.
        rmax = max(xmax_abs, zmax_abs)
        if b[4] > 1700.0:
            vols["inlet"] = tag
        elif b[5] > 800.0:
            vols["outlet"] = tag
        elif rmax < R_LAMP_MM + TOL_MM:
            vols["lamp"] = tag
        elif rmax > R_BODY_MM - TOL_MM:
            vols["body"] = tag

    for k in ("body", "lamp", "inlet", "outlet"):
        if k not in vols:
            sys.stderr.write(f"Could not identify volume: {k}\n")
            return 2
    print(f"Identified volumes: {vols}")

    # Boolean: fluid = (body union inlet union outlet) - lamp.
    union, _ = gmsh.model.occ.fuse(
        [(3, vols["body"])],
        [(3, vols["inlet"]), (3, vols["outlet"])],
    )
    gmsh.model.occ.synchronize()

    fluid, _ = gmsh.model.occ.cut(union, [(3, vols["lamp"])])
    gmsh.model.occ.synchronize()
    if len(fluid) != 1:
        sys.stderr.write(f"Expected 1 fluid volume, got {len(fluid)}\n")
        return 3
    fluid_tag = fluid[0][1]

    # Classify the surfaces of the fluid volume
    boundary = gmsh.model.getBoundary([(3, fluid_tag)], oriented=False)
    classified = {"bodyWall": [], "lampWall": [], "inlet": [], "outlet": []}
    for dim, tag in boundary:
        cls = classify_surface(None, tag)
        classified[cls].append(tag)

    print()
    print("Surface classification:")
    for k, tags in classified.items():
        print(f"  {k:10s}  {len(tags):>3d} surface(s)  tags={tags}")

    # Mesh the surface, then extract triangles per group and write STLs
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)   # mm
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)   # mm
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(2)

    print()
    print("STL output:")
    for name, tags in classified.items():
        path = os.path.join(OUTDIR, f"{name}.stl")
        triangles = []
        for tag in tags:
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, tag)
            # Get all node coordinates for this surface
            for elem_type, elem_tags, node_tags in zip(etypes, etags, enodes):
                if elem_type != 2:  # 3-node triangle
                    continue
                # node_tags is a flat list: 3 entries per triangle
                for i in range(0, len(node_tags), 3):
                    nts = node_tags[i:i + 3]
                    verts = []
                    for nt in nts:
                        coords = gmsh.model.mesh.getNode(nt)[0]
                        verts.append(tuple(coords))
                    triangles.append(tuple(verts))
        write_ascii_stl(path, name, triangles)
        size_kib = os.path.getsize(path)/1024
        print(f"  {name:10s}  {len(triangles):>6d} triangles   {size_kib:>7.1f} KiB")

    gmsh.finalize()

    # Summary in case coordinates (m)
    print()
    print("Geometry summary (case coords, after STEP transform):")
    print(f"  Body:    x in [0, 0.889] m, OD = 8.9 cm")
    print(f"  Lamp:    x in [0, 0.810] m physical (free-tip apex at x=0.810);"
          f" OD = 2.0 cm")
    print(f"  Inlet:   axial pipe, x in [0.889, 1.739] m,  dia = 1.91 cm")
    print(f"  Outlet:  perpendicular pipe at x = 0.0477,"
          f" z in [0.0435, 0.8945] m, dia = 1.91 cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
