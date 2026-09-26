# Reactor geometry from CAD

An ultraviolet reactor is drawn in computer-aided design (CAD) before it is meshed, and the
radiation model needs its shape twice: as the geometry that **shadows** the light — the vessel
walls, the pipes a cell may sit down — and as the lamp that **emits** it. Both can be read
straight from the drawing. {func}`~aquaflux.io.cad.read_step` reads a STEP file (ISO 10303, the
exchange format every CAD tool writes) into a {class}`~aquaflux.io.cad.CadModel`, which hands out

- **exact bodies** from `aquaflux.solids` — a vessel as the cylinders it is rather than as a
  faceted triangle soup — each checked against the drawing before it is returned, and
- **triangles** of a solid's surface for an emitter, every vertex on the true surface and the facet
  size under your control.

Reading needs the optional CAD kernel, OpenCASCADE through its Python binding:

```bash
pip install "aquaflux[cad]"
```

## Why exact bodies rather than triangles

A shadow test asks, for every pair of a lamp facet and a cell, whether the straight line between
them leaves the water. Against a cylinder that is a formula — a few dozen arithmetic operations, the
same for every pair, compiled into one expression. Against a triangulated wall it is a search over
thousands of triangles, and it is also *wrong* by the sliver between a round pipe and the polygon
that stands in for it. On the Sozzi & Taghipour (2006) reactor the triangulated wall's mask costs
several hundred times the rest of the calculation put together; described as three cylinders, the
same mask is exact and costs about as much as the gather it shadows. The CAD reader is how a real
drawing gets that description without anyone typing radii into a script.

## Reading a drawing

```python
import aquaflux
from aquaflux.io.cad import Placement, read_step

cad = read_step(
    "SozziTaghipour.step",
    Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]),   # drawn along y, meshed along x
)
cad.names        # ('lamp', 'outlet_pipe', 'inlet_pipe', 'reactor_body')
```

- **Names** are the solids' part names in the file; a part placed several times in an assembly
  gives one solid per placement, `name[0]`, `name[1]`, ….
- **Lengths are always metres**, whatever unit the drawing was made in.
- **A {class}`~aquaflux.io.cad.Placement`** maps the drawing's frame to the mesh's,
  `x_mesh = matrix @ x_drawing + offset`. The matrix must be orthogonal — a rotation, or a
  reflection such as the axis swap above — because a stretch would turn a cylinder into an elliptic
  one, which no body describes.

## The water, as the vessel's shadow

A vessel's wall is awkward to write down as a solid; the water it holds is easy. So the shadowing
body for a vessel is *everything that is not water*:

```python
water = cad.fluid("reactor_body", "inlet_pipe", "outlet_pipe")   # an aquaflux.solids.Outside
```

A sight line is clear exactly when the named solids cover it end to end. A pipe that joins the
vessel ends, in the drawing, in a saddle-shaped face cut to the vessel's curve; the reader describes
it by its full cylinder carried on into the vessel, which is exact once the two are taken together
— which is why such a pipe is accepted by {meth}`~aquaflux.io.cad.CadModel.fluid` and refused on its
own by {meth}`~aquaflux.io.cad.CadModel.solid`.

## The lamp, as an emitting surface

```python
import numpy as np
from aquaflux.radiation import (
    NoOcclusion, Surfaces, UniformAbsorption, absorption_from_uvt,
    direct_fluence_rate, lamp_exitance,
)

triangles = cad.triangles("lamp", chord=1e-4, facet_size=2.5e-3)
# The lamp's base disc sits against the end wall: it lights no cell, but it would take a share of
# the rating, so it is removed before the rating is spread over the lamp.
triangles = triangles[~np.all(np.isclose(triangles[:, :, 0], 0.0), axis=1)]

lamp = Surfaces.from_triangles(triangles, solid_names=("lamp",))
lamp = lamp.with_optics(emission=lamp_exitance(lamp, {"lamp": 35.0}))   # a 35 W lamp

G = direct_fluence_rate(
    lamp,
    cell_centres,                                   # (n_cells, 3), e.g. from the mesh
    absorption=UniformAbsorption(absorption_from_uvt(70.0)),
    occluders=[water],
    self_occlusion=NoOcclusion(),                   # a convex lamp cannot shadow itself
)
```

`chord` bounds how far a facet may lie from the true surface, and sets the spacing *around* a
curved face; `facet_size` bounds how large a facet may be, and sets the spacing *along* a straight
one. **For a lamp, keep the chord coarse and choose the facet size for accuracy**: the cells nearest
a lamp are sensitive to the spacing along it. On the Sozzi reactor
(`validation/sozzi_radiation/lamp_resolution.py`), `chord=1e-4, facet_size=2.5e-3` gives 66,011
facets and a median error of 0.16% within 5 mm of the lamp and 0.04% elsewhere, against a
270,336-facet reference — as accurate near the lamp as an analytic lamp of the same facet count.
Because the vertices lie on the true surface, the lamp's area is not inscribed, and
{func}`~aquaflux.radiation.lamp_exitance` then makes it radiate exactly its rating.

### Without a drawing: the lamp from the mesh's own patch

A case meshed around the lamp already describes its surface: the lamp's boundary patch. Its faces
can be used directly, which removes the second description of the lamp an STL or a drawing would
otherwise be, and any disagreement between the two:

```python
from aquaflux.mesh import patch_triangles
from aquaflux.radiation import coarsen_surfaces

patch = patch_triangles(mesh, mesh.geometry(), ["lampWall"])     # facing into the water
exact = Surfaces.from_triangles(patch.vertices, solid_id=patch.patch_id, solid_names=patch.patch_names)
exact = exact.with_optics(emission=exact.per_facet({"lampWall": 696.42}))   # W/m^2

lamp, record = coarsen_surfaces(exact, max_edge=4e-3, chord=1e-4)
```

{func}`~aquaflux.mesh.patch_triangles` cuts each face into the same triangles the mesh computes its
face geometry from, so a planar face's triangles have exactly its area, and faces them into the
domain from the mesh's own face normals rather than from the order the nodes were listed in. A patch
snapped to a lamp carries the mesh's resolution rather than one chosen for radiation, and the gather
costs cells times facets, so {func}`~aquaflux.radiation.coarsen_surfaces` coarsens it first: edges are
collapsed until no triangle has an edge longer than `max_edge`, lies further than `chord` from the
patch, or turns further than `angle` from it, with the patch's rim and any sharp crease kept. Every
vertex it keeps is one of the patch's. Each body's exitance is raised by the small area the coarse
facets lose, so the lamp radiates exactly what the patch did; `record` reports the realized edge,
chord and angle per facet.

On the Sozzi reactor (`validation/sozzi_radiation/lamp_resolution.py`) the 48,550-face lamp patch is
194,636 triangles; at `max_edge=4e-3, chord=1e-4` it coarsens to 18,432 facets in about twenty
seconds, changing the fluence rate by a median 0.52% within 5 mm of the lamp and 0.14% elsewhere. As
for the drawing's lamp, the spacing along the lamp matters more than the chord. The patch itself is
a snapped approximation of the lamp: against the true cylinder it is about as accurate as the STL
the mesh was snapped to, so where a drawing is available its lamp is the more accurate emitter, and
where the comparison is against a finite-volume radiation solve on the same mesh, the patch is
exactly the surface that solve emits from.

### With reflecting walls: the whole model, its shadows streamed

The gather above is the whole field when the walls are black. Where they reflect, build the model,
which also solves the interreflection between facets. At a mesh's cells its receiver shadow mask —
one entry per cell, facet and body — is too large to hold, so ask for it to be built chunk by chunk
at each call instead:

```python
from aquaflux.radiation import RadiationSettings, build_radiation_model, fluence_rate

model = build_radiation_model(
    cell_centres, lamp, occluders=[water],
    settings=RadiationSettings(self_occlusion=NoOcclusion(), stream_receiver_mask=True),
)
G, cycles = fluence_rate(model, lamp, absorption=UniformAbsorption(absorption_from_uvt(70.0)))
```

Streaming changes memory, not the answer or its derivatives: a gradient through `fluence_rate` is
kept to one chunk's memory as well, by rebuilding each chunk's mask on the way back. On the Sozzi
reactor the streamed model reproduces the hand-built field above to 4.4e-16 relative on all
1,635,909 cells. Leave `stream_receiver_mask` unset for a scene whose mask fits, where holding it
once makes every later call cheaper.

Most of those segments never leave the water: a cell in the chamber and a facet of the lamp both lie
inside the chamber's cylinder, and so does every segment between them. To use that, give the
settings a {class}`~aquaflux.radiation.ShaftCulling` as `body_culling`. Cells and facets are grouped
into compact blocks, and a body that can prove it misses the whole region between a block of cells
and a block of facets clears every pair of that tile without testing one. A tile it cannot vouch for
is split into smaller tiles and asked again, and only what is still undecided at the smallest size
is tested pair by pair. The mask is identical, bit for bit, to the one testing every pair gives;
only the number of segments tested changes. On receivers sampled uniformly in the Sozzi reactor's
water, `validation/sozzi_radiation/body_culling.py` certifies about nine pairs in ten this way and
builds the mask five to nine times faster -- the most when refinement stops at groups of eight,
which is what the class documentation recommends for a scene of analytic bodies alone. A mesh
refined towards its walls and lamp puts more of its cells where a tile cannot be vouched for, so
expect less there.

Geometry that exists only as triangles -- a vessel wall from an STL file, a sculpted baffle -- goes
in the same list of occluders as a {class}`~aquaflux.radiation.TriangleBody`. Its segments are
tested through a uniform grid over its triangles, and under shaft culling it vouches for any tile
whose bounding box holds no occupied voxel of a finer grid. Testing a segment against triangles is
far dearer than against a cylinder, which is where refinement pays: on the Sozzi chamber's wall as
51,328 triangles (`validation/sozzi_radiation/triangle_culling.py`), refining to groups of two
vouches for nine pairs in ten and builds that wall's mask about five times faster, against less
than twofold when refinement stops at eight. Which side of the body is solid is read from the
triangles: a piece with a free edge is a sheet with no inside, and a closed piece is solid on the
side its normals point away from, so a vessel wound to face its water is solid outside it. Pass
`sheet=True` when a closed surface is really a sheet.

## What is checked, and what is refused

Recognition reads each solid's faces — which surface each lies on, which side of it the solid is
on — and proposes a body:

- a solid wrapped in whole-turn cylinders, cones and spheres is their **union** (a flat-ended pipe,
  a lamp with a hemispherical tip, a reducer; a cylinder an exporter split into two half-turn faces
  is recognized as one);
- a convex solid bounded only by planes is the **intersection of its faces' half-spaces** (a box, a
  rectangular channel, a plate), which is exact by construction.

**Every other proposal is checked against the drawing before it is handed out.** Points are sampled
on both boundaries — the proposal's and the drawing's — and every one must lie within the drawing's
own declared geometric tolerance of the other; a proposal that fails raises
{class}`~aquaflux.io.cad.InexactBody`. The tolerance is the precision the drawing was exported at,
because nothing can be compared with a drawing more finely than that: a drawing exported at 10 µm
cannot tell a pipe from one 0.1% thinner, and the check does not pretend to. A distance rather than
a volume is compared because a thin feature — a baffle, a fin — holds almost no volume but is far
from any boundary that leaves it out. {meth}`~aquaflux.io.cad.CadModel.discrepancy` reports the
measured distance beside the tolerance it is held to.

A solid no rule can describe exactly — a torus elbow, a free-form spline surface, an L-shaped
plate — raises {class}`~aquaflux.io.cad.UnrecognizedSolid`, naming what each rule found. It is
refused rather than approximated, because an approximate shadow is not reported as one: a reactor
shadowed by the wrong geometry produces a plausible field.

## Verified on a real drawing

The Sozzi & Taghipour reactor's own drawing (an Onshape export; three vessel solids and the lamp)
reads in under a second. Its water, checked against the drawing to 0.75 µm against a declared
10 µm, shadows the reactor on 180 million lamp-to-cell sight lines **identically** to an occluder
derived by hand for that reactor — not one sight line differs — even though the drawing's pipes run
well beyond the meshed domain where the hand-typed ones stop at it. The harness is
`validation/sozzi_radiation/primitive_occlusion.py`.
