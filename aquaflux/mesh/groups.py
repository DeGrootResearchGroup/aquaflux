"""Named groupings of mesh elements: cell zones and face patches.

A **partition** of elements into named groups — one ``int`` label per element, each
element in exactly one group, stored struct-of-arrays as a single label array.
:class:`LabelledGroups` carries the shared machinery; the
specific :class:`CellZones` and :class:`FacePatches` add the type-specific helpers.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .connectivity import FaceCellConnectivity


class LabelledGroups(eqx.Module):
    """A partition of ``n_elements`` into named groups (each element in exactly one group).

    Attributes
    ----------
    label : jnp.ndarray
        Group id per element, shape ``(n_elements,)``.
    names : tuple of str
        Group name per id (static); ``label`` values index into it.
    """

    label: jnp.ndarray
    names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def from_dict(
        cls,
        n_elements: int,
        groups: dict[str, object],
        default: str = "default",
    ) -> LabelledGroups:
        """Build a partition from ``{name: indices}``, unlisted elements → the ``default`` group.

        The ``default`` group is always id 0 (even when empty), so every element has a home.
        Raises ``ValueError`` if two groups claim the same element (partition violated) or an
        index is out of range.
        """
        names = [default] + [n for n in groups if n != default]
        name_to_id = {n: i for i, n in enumerate(names)}
        label = np.zeros(n_elements, dtype=np.int64)
        assigned = np.zeros(n_elements, dtype=bool)
        for name, indices in groups.items():
            idx = np.asarray(list(indices), dtype=np.int64)
            if idx.size == 0:
                continue
            if idx.min() < 0 or idx.max() >= n_elements:
                raise ValueError(f"group '{name}' has an index outside [0, {n_elements})")
            if np.any(assigned[idx]):
                raise ValueError(f"group '{name}' overlaps another group (partition violated)")
            assigned[idx] = True
            label[idx] = name_to_id[name]
        return cls(label=jnp.asarray(label), names=tuple(names))

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.names)

    def id_of(self, name: str) -> int:
        """Group id for a name."""
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(f"no group named '{name}'; groups are {self.names}") from None

    def mask(self, name: str) -> jnp.ndarray:
        """Boolean selector for the named group, shape ``(n_elements,)``."""
        return self.label == self.id_of(name)

    def indices(self, name: str) -> jnp.ndarray:
        """Element indices in the named group."""
        return jnp.where(self.mask(name))[0]

    def size(self, name: str) -> int:
        """Number of elements in the named group."""
        return int(jnp.sum(self.mask(name)))


class CellZones(LabelledGroups):
    """A partition of cells into named zones (fluid / solid / porous / …)."""

    @classmethod
    def default(cls, n_cells: int) -> CellZones:
        """A single ``"default"`` zone containing every cell."""
        return cls(label=jnp.zeros(n_cells, dtype=jnp.int64), names=("default",))

    def interface_mask(self, face_cells: FaceCellConnectivity) -> jnp.ndarray:
        """Interior faces whose two cells lie in *different* zones, shape ``(n_faces,)``.

        Zone interfaces are derived from the labelling — no hand-maintained lists.

        Parameters
        ----------
        face_cells : FaceCellConnectivity
            The face→cell incidence (``mesh.face_cells``).
        """
        # boundary faces read as owner == neighbour → never an interface
        owner, nb = face_cells.owner, face_cells.safe_neighbour
        return face_cells.interior & (self.label[owner] != self.label[nb])

    def interface_mask_between(
        self, face_cells: FaceCellConnectivity, zone_a: str, zone_b: str
    ) -> jnp.ndarray:
        """Faces between zones ``zone_a`` and ``zone_b`` (either orientation).

        Raises ``ValueError`` if the two names are equal — an interface is *between* two distinct
        zones, so ``zone_a == zone_b`` (which would return every intra-zone interior face) is
        almost certainly a caller mistake.

        Parameters
        ----------
        face_cells : FaceCellConnectivity
            The face→cell incidence (``mesh.face_cells``).
        zone_a, zone_b : str
            The two (distinct) zone names to find the interface between.
        """
        if zone_a == zone_b:
            raise ValueError(
                f"interface_mask_between needs two distinct zones; got '{zone_a}' twice"
            )
        a, b = self.id_of(zone_a), self.id_of(zone_b)
        zo, zn = self.label[face_cells.owner], self.label[face_cells.safe_neighbour]
        between = ((zo == a) & (zn == b)) | ((zo == b) & (zn == a))
        return face_cells.interior & between


class FacePatches(LabelledGroups):
    """A partition of faces into named patches.

    Boundary faces default to a ``"boundary"`` patch and interior faces to ``"interior"``;
    named patches overlay these. A patch may name **interior** faces too — that is how a
    baffle (an interior face acting as a wall) or an explicitly-treated interface is
    represented.
    """

    @classmethod
    def from_dict(cls, neighbour: jnp.ndarray, groups: dict[str, object]) -> FacePatches:
        """Build face patches: ``"interior"``/``"boundary"`` by default, then overlay ``groups``.

        Named patches move their faces off the default; two named patches may not claim the
        same face. Out-of-range indices raise ``ValueError``. The names ``"interior"`` and
        ``"boundary"`` are reserved (assigned automatically from the boundary mask) and may not
        be used for a named patch.
        """
        nb = np.asarray(neighbour)
        n_faces = nb.shape[0]
        base = ["interior", "boundary"]
        reserved = [n for n in groups if n in base]
        if reserved:
            raise ValueError(
                f"patch name(s) {reserved} are reserved (interior/boundary are assigned "
                "automatically); use a different name"
            )
        names = base + [n for n in groups if n not in base]
        name_to_id = {n: i for i, n in enumerate(names)}
        label = np.where(nb < 0, name_to_id["boundary"], name_to_id["interior"]).astype(np.int64)
        assigned = np.zeros(n_faces, dtype=bool)  # tracks explicit (named) assignment only
        for name, indices in groups.items():
            idx = np.asarray(list(indices), dtype=np.int64)
            if idx.size == 0:
                continue
            if idx.min() < 0 or idx.max() >= n_faces:
                raise ValueError(f"patch '{name}' has an index outside [0, {n_faces})")
            if np.any(assigned[idx]):
                raise ValueError(f"patch '{name}' overlaps another named patch")
            assigned[idx] = True
            label[idx] = name_to_id[name]
        return cls(label=jnp.asarray(label), names=tuple(names))

    @classmethod
    def default(cls, neighbour: jnp.ndarray) -> FacePatches:
        """The default ``"interior"`` + ``"boundary"`` split."""
        return cls.from_dict(neighbour, {})

    def is_boundary_patch(self, name: str, face_cells: FaceCellConnectivity) -> bool:
        """Whether the named patch is non-empty and every face in it is a boundary face.

        An empty patch returns ``False`` (rather than a vacuously-``True`` "all faces are
        boundary faces"), so the answer is never misleading.

        Parameters
        ----------
        name : str
            The patch name.
        face_cells : FaceCellConnectivity
            The face→cell incidence (``mesh.face_cells``); a boundary face is ``~interior``.
        """
        mask = self.mask(name)
        boundary = ~face_cells.interior
        return bool(jnp.any(mask)) and bool(jnp.all(boundary[mask]))
