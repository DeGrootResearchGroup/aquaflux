"""Build an unstructured tetrahedral duct mesh with gmsh's OpenCASCADE + Delaunay backends.

A rectangular duct (streamwise x, cross-section y-z), meshed as tetrahedra rather than the
hexahedra every other 3D case here uses. Any tetrahedral mesh of a box has cells owning two or
more boundary faces wherever an element touches an edge of the box -- the corner tetrahedra
issue #432 is about -- so this needs no special construction to produce them, only a genuinely
unstructured tet mesh of a domain with edges.

Three boundary patches: `inlet` (x = 0), `outlet` (x = LX), `walls` (the other four faces, one
patch -- a corner cell's two boundary faces are typically both on `walls`, which is the regime
`OwnerGradient` leaves underdetermined). Written as Gmsh MSH format 2.2 ASCII, the format
`gmshToFoam` reads.
"""

import os

import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "duct.msh")

# Duct dimensions (m): streamwise x, cross-section y by z.
LX, LY, LZ = 0.25, 0.025, 0.025
# Target element size (m), chosen by a small sweep (0.005-0.008) for the smallest mesh with zero
# cells whose Hessian-correction normal equations are near-singular in the INTERIOR -- a Delaunay
# tessellation occasionally leaves a cell whose immediate neighbourhood is nearly coplanar, unrelated
# to the boundary-face question this case exists to test, and a few sizes hit it (0.006, 0.008: ~10
# cells each) while others do not. 0.007 has none, at 2462 cells / 176 corner cells (>=2 boundary
# faces) -- re-run the sweep in this docstring's history if the geometry changes.
MESH_SIZE = 0.007

TOL = 1e-6 * max(LX, LY, LZ)


def classify(tag):
    xmin, _ymin, _zmin, xmax, _ymax, _zmax = gmsh.model.getBoundingBox(2, tag)
    if abs(xmin) < TOL and abs(xmax) < TOL:
        return "inlet"
    if abs(xmin - LX) < TOL and abs(xmax - LX) < TOL:
        return "outlet"
    return "walls"


def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("tetrahedral_duct")

    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, LX, LY, LZ)
    gmsh.model.occ.synchronize()

    faces = gmsh.model.getBoundary([(3, box)], oriented=False)
    groups: dict[str, list[int]] = {"inlet": [], "outlet": [], "walls": []}
    for _dim, tag in faces:
        groups[classify(tag)].append(tag)
    for name, tags in groups.items():
        if not tags:
            raise RuntimeError(f"no surfaces classified as '{name}'")
        ptag = gmsh.model.addPhysicalGroup(2, tags, name=name)
        print(f"  physical surface '{name}': {len(tags)} face(s), tag {ptag}")
    vtag = gmsh.model.addPhysicalGroup(3, [box], name="fluid")
    print(f"  physical volume 'fluid': tag {vtag}")

    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.model.mesh.generate(3)
    # Deliberately NOT running gmsh's own mesh optimizer (Netgen or otherwise): it restructures the
    # mesh near every edge specifically to remove sliver-adjacent cells, which also eliminates every
    # cell owning two or more boundary faces -- exactly the population this case exists to mesh. The
    # element size above was chosen instead (see MESH_SIZE's comment) to reach zero degenerate cells
    # without optimizing away the corner cells.

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    print(f"  {n_nodes} nodes, {n_tets} tetrahedra")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(OUT)
    print(f"  wrote {OUT}")
    gmsh.finalize()


if __name__ == "__main__":
    main()
