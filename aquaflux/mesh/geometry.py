"""The derived geometric metrics of a mesh, bundled as one value.

:class:`MeshGeometry` pairs the face metrics (:class:`~aquaflux.mesh.face.FaceGeometry`:
areas, centroids, owner-outward normals) with the cell metrics
(:class:`~aquaflux.mesh.cell.CellGeometry`: volumes, centroids). The two are always computed
together and consumed together — the residual assembly, the gradient/interpolation schemes,
and the coupled-flow terms all read both — so they are carried as a single object rather than
as a loose pair of arguments threaded through every signature.

Why geometry is *derived on demand* (returned by :meth:`~aquaflux.mesh.Mesh.geometry`), not
stored as fields on the :class:`~aquaflux.mesh.Mesh`
-----------------------------------------------------------------------------------------------
The metrics here are a pure function of the mesh's node coordinates and topology. Keeping them
as a separate, recomputed product — rather than caching them alongside the coordinates they are
derived from — is deliberate, and rests on two properties of this solver:

1. **Differentiability with respect to node positions.** The node coordinates are a
   differentiable leaf of the mesh: reverse-mode gradients (mesh-sensitivity diagnostics) flow
   through them. If the metrics were stored as sibling leaves of the coordinates, automatic
   differentiation would treat the two as *independent* values and silently skip the
   coordinate → metric dependency, yielding wrong gradients with no error. Producing the metrics
   through a function of the coordinates instead keeps that dependency on the tape, so a gradient
   with respect to node positions correctly chains through area, normal, volume, and centroid.

2. **Staleness under topology transforms.** Build-time preprocessing steps such as cell
   renumbering and domain partitioning rewrite the connectivity and cell order. A stored copy of
   the metrics would have to be re-permuted in lock-step with every such transform or go quietly
   out of date; a derived product is simply recomputed from the transformed mesh, with nothing to
   keep in sync.

For a solver that neither differentiates with respect to node positions nor transforms topology
after construction, caching the metrics on the mesh would be unobjectionable. It is precisely
those two capabilities that make a cached copy a liability here, so geometry stays a derived
product and this class gives that product a name and a home.

Attributes
----------
face : FaceGeometry
    Face areas, centroids, and owner-outward unit normals.
cell : CellGeometry
    Cell volumes and centroids.
"""

from __future__ import annotations

import equinox as eqx

from .cell import CellGeometry
from .face import FaceGeometry


class MeshGeometry(eqx.Module):
    """Bundled face and cell metrics derived from a mesh (see the module docstring).

    Attributes
    ----------
    face : FaceGeometry
        Face areas, centroids, and owner-outward unit normals.
    cell : CellGeometry
        Cell volumes and centroids.
    """

    face: FaceGeometry
    cell: CellGeometry
