"""The per-evaluation context every strategy family gathers its inputs from.

A residual, a gradient reconstruction, a slope limiter and a boundary closure all evaluate against
the *same* mesh at the *same* instant: the connectivity, the face/cell geometry, and the physical
properties never differ between them within one residual evaluation. What genuinely varies is the
*field* -- its values, the weak boundary values a set of closures resolves it to, and its
reconstructed gradient. Splitting the two is what lets a vector field (three velocity components, a
Reynolds-stress tensor, ...) be evaluated as several per-field views over one shared mesh context,
rather than forcing every consumer to agree on a single flat bundle scoped to one scalar.

:class:`MeshContext` carries the shared half -- connectivity, geometry, and the evaluated property
map -- and :class:`FieldContext` wraps it with one field's own reconstructed state. A strategy that
needs only the mesh (a slope limiter, a boundary closure evaluating no state-dependent property)
takes a :class:`MeshContext` directly; one that needs a field's own value, boundary value or
gradient too takes the :class:`FieldContext` that field was reconstructed into.

This module imports only mesh types, deliberately: it sits below every strategy family (residual
assembly, reconstruction schemes, boundary closures) so each can consume it without importing
another, which is what makes one context object usable by all of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


class MeshContext(eqx.Module):
    """The mesh-level inputs shared by every field evaluated in one residual evaluation.

    Attributes
    ----------
    face_cells : FaceCellConnectivity
        The face->cell incidence and its gather/scatter operators -- the gather primitive
        (``field[face_cells.owner]`` / ``field[face_cells.safe_neighbour]``).
    geometry : MeshGeometry
        Face metrics (areas, owner-outward normals, centroids) and cell metrics (volumes,
        centroids).
    properties : mapping of {str: jnp.ndarray}
        The evaluated per-cell properties, ``{name: (n_cells,) array}`` (density, viscosity,
        conductivity, ...). One field regardless of how many properties exist, so adding a
        property never changes this object's shape; a consumer reads the property it names.
    """

    face_cells: FaceCellConnectivity
    geometry: MeshGeometry
    properties: Mapping[str, jnp.ndarray]


class FieldContext(eqx.Module):
    """One field's reconstructed state, sharing a :class:`MeshContext` with any field alongside it.

    A vector field (velocity's components, say) is several ``FieldContext``s built from one shared
    ``mesh`` -- the connectivity, geometry and properties are formed once and referenced by each
    component's view, rather than recomputed or duplicated per component.

    Attributes
    ----------
    mesh : MeshContext
        The shared mesh-level inputs.
    boundary_values : jnp.ndarray
        Weak boundary face values ``phi_ip`` for this field, shape ``(n_faces,)`` (interior entries
        ignored).
    gradient : jnp.ndarray
        Reconstructed cell gradient of this field, shape ``(n_cells, dim)``. Formed once per
        evaluation (a linear solve on skewed grids), so it is a context field rather than
        re-solved per consumer; zeros when no gradient scheme is injected.
    """

    mesh: MeshContext
    boundary_values: jnp.ndarray
    gradient: jnp.ndarray
