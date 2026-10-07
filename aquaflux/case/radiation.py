"""What a radiation case states besides its patches: the lamps' light, the medium, where each surface
and body comes from, and where the light is wanted.

A radiation case is a mesh whose boundary patches are lamps and walls (:class:`~aquaflux.case.Lamp`,
:class:`~aquaflux.case.Wall`), plus the values here:

* how a lamp distributes its light (:class:`LambertianProfile`, :class:`CosinePowerProfile`, or a
  measured luminaire's table, :class:`IesProfile`);
* the medium the light crosses (:class:`UniformMedium`);
* where a lamp's or a reflecting wall's triangles come from. The drawing the mesh was made from is
  the best source -- a STEP file (:class:`CadSurface`, every vertex on the true surface) or an STL file
  (:class:`StlSurface`) -- and the mesh's own patch (:class:`MeshPatch`) is what is used when neither is
  given, since a mesh is only an approximation of that drawing and a reflecting surface needs far
  fewer, larger facets than a mesh has faces;
* what stands in the way: the drawing's solids as exact bodies (:class:`CadSolid`, :class:`CadFluid`),
  an STL file's triangles (:class:`StlBody`), or mesh patches (:class:`PatchBody`);
* where the light is wanted (:class:`Receivers`): the fluence rate at the cell centres, the irradiance
  at the centres of the faces of chosen patches.

A surface read from a file is matched to the way its patch faces: each triangle's normal is compared
with the inward normal of the nearest face of that patch, and the whole surface is turned round if it
faces out of the domain. A surface whose triangles disagree with one another is refused, since one of
them is wound the wrong way and would emit or reflect nothing.

Every file is named relative to the case file.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.spatial import cKDTree

from aquaflux.io.cad.model import CadModel
from aquaflux.radiation import (
    Absorption,
    CosinePower,
    Lambertian,
    Profile,
    TriangleBody,
    UniformAbsorption,
    absorption_from_uvt,
    coarsen_to_size,
    read_ies,
    read_stl,
)

__all__ = [
    "CadFluid",
    "CadPlacement",
    "CadSolid",
    "CadSurface",
    "Coarsen",
    "CosinePowerProfile",
    "IesProfile",
    "LambertianProfile",
    "LampProfile",
    "MeshPatch",
    "OccluderSpec",
    "PatchBody",
    "Receivers",
    "StlBody",
    "StlSurface",
    "SurfaceSource",
    "UniformMedium",
]

#: A triangle whose normal lies within this angle of its patch's nearest face, or of its opposite,
#: says which way the surface faces; one nearer a right angle (at a crease, where the nearest face
#: is on the other side of it) says nothing and is not counted.
_DECISIVE_COSINE = 0.5

#: The radiant intensity units an IES file may state in its ``[_INTENSITYUNITS]`` keyword, in W/sr.
_RADIANT_INTENSITY_UNITS = {"W/sr": 1.0, "mW/sr": 1e-3, "uW/sr": 1e-6, "\u00b5W/sr": 1e-6}


def _refuse_non_finite(owner: str, name: str, values) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{owner}.{name} must be finite, got {values!r}.")


def _refuse_non_positive(owner: str, name: str, value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value > 0):
        raise ValueError(f"{owner}.{name} must be a positive, finite number, got {value!r}.")


# --- How a lamp distributes its light --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LampProfile(abc.ABC):
    """How a lamp distributes its light over direction: Lambertian, a cosine power, or a measured table."""

    #: Whether the profile can state the lamp's power itself (:meth:`radiant_power`).
    can_state_power: ClassVar[bool] = False

    @abc.abstractmethod
    def profile(self, directory: Path) -> Profile:
        """The angular distribution its facets emit with.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in; a file named here is relative to it.
        """

    def radiant_power(self, directory: Path) -> float | None:
        """The lamp's power in W if the profile itself states it; ``None`` (the default) if not."""
        del directory
        return None


@dataclasses.dataclass(frozen=True)
class LambertianProfile(LampProfile):
    """A diffuse emitter: the same radiance in every direction it faces."""

    def profile(self, directory: Path) -> Lambertian:
        """:class:`~aquaflux.radiation.Lambertian`."""
        del directory
        return Lambertian()


@dataclasses.dataclass(frozen=True)
class CosinePowerProfile(LampProfile):
    """A beam: intensity proportional to ``cos(theta)^exponent`` about each facet's normal.

    Attributes
    ----------
    exponent : float
        ``>= 1``; one is Lambertian.
    """

    exponent: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.exponent) and self.exponent >= 1.0):
            raise ValueError(
                f"CosinePowerProfile.exponent must be at least one, got {self.exponent!r}."
            )

    def profile(self, directory: Path) -> CosinePower:
        """:class:`~aquaflux.radiation.CosinePower` at :attr:`exponent`."""
        del directory
        return CosinePower(self.exponent)


@dataclasses.dataclass(frozen=True)
class IesProfile(LampProfile):
    """A measured luminaire's photometry, read from an IES LM-63 file.

    Attributes
    ----------
    file : str
        The ``.ies`` file, relative to the case file.
    up : tuple of float
        The direction, in the case's frame, of the table's horizontal angle zero (the fixture's
        "up"). It must not be parallel to any lamp facet's normal.

    Raises
    ------
    ValueError
        If ``up`` is not three finite components, not all zero.
    """

    can_state_power: ClassVar[bool] = True
    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    up: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("IesProfile.file names the photometry file.")
        if len(self.up) != 3:
            raise ValueError(f"IesProfile.up has three components, got {self.up!r}.")
        _refuse_non_finite("IesProfile", "up", self.up)
        if not any(self.up):
            raise ValueError("IesProfile.up is a direction, so it cannot be zero.")

    def profile(self, directory: Path) -> Profile:
        """The table's :class:`~aquaflux.radiation.PhotometricProfile`, oriented by :attr:`up`."""
        return read_ies(Path(directory) / self.file).profile(up=self.up)

    def radiant_power(self, directory: Path) -> float | None:
        """The table's own flux in W, if the file states its intensities in a radiant unit.

        LM-63 intensities are candela unless the file says otherwise, and a candela table carries no
        radiant power without a spectrum. A file measured for ultraviolet may state its unit in an
        ``[_INTENSITYUNITS]`` keyword -- ``W/sr``, ``mW/sr`` or ``uW/sr`` -- and its flux is then a
        power; for any other file this is ``None`` and the lamp must state its power.
        """
        photometry = read_ies(Path(directory) / self.file)
        scale = _RADIANT_INTENSITY_UNITS.get(photometry.keywords.get("_INTENSITYUNITS"))
        return None if scale is None else photometry.flux * scale


# --- The medium ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UniformMedium:
    """A medium of uniform absorption, stated as a coefficient or as a transmittance.

    Exactly one of the two is given.

    Attributes
    ----------
    absorption : float or None
        The napierian absorption coefficient, per metre (light falls as ``exp(-a r)``).
    transmittance : float or None
        The ultraviolet transmittance over one centimetre, in percent (water's UVT); converted by
        :func:`~aquaflux.radiation.absorption_from_uvt`.

    Raises
    ------
    ValueError
        If neither or both is given, or the value given is out of range.
    """

    absorption: float | None = None
    transmittance: float | None = None

    def __post_init__(self) -> None:
        if (self.absorption is None) == (self.transmittance is None):
            raise ValueError(
                "a UniformMedium states exactly one of absorption (per metre) and transmittance "
                f"(percent over a centimetre), got {'both' if self.absorption is not None else 'neither'}."
            )
        if self.absorption is not None and not (
            math.isfinite(self.absorption) and self.absorption >= 0
        ):
            raise ValueError(
                f"UniformMedium.absorption must be a non-negative, finite number, got {self.absorption!r}."
            )
        if self.transmittance is not None and not 0 < self.transmittance <= 100:
            raise ValueError(
                f"UniformMedium.transmittance is a percentage in (0, 100], got {self.transmittance!r}."
            )

    def absorption_model(self) -> Absorption:
        """The medium as the gathers take it."""
        if self.absorption is not None:
            return UniformAbsorption(self.absorption)
        return UniformAbsorption(absorption_from_uvt(self.transmittance))


# --- Where a surface's triangles come from ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Coarsen:
    """Coarsen a surface as far as these bounds allow (:func:`~aquaflux.radiation.coarsen_to_size`).

    Attributes
    ----------
    max_edge : float
        The longest edge a coarse triangle may have, in metres.
    chord : float
        The farthest an original vertex may lie from the coarse triangle that replaces it, in metres.
    angle : float or None
        The largest angle, in radians, between a coarse triangle and an original one it replaces;
        unset, the coarsener's own.
    """

    max_edge: float
    chord: float
    angle: float | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (coarsen_to_size,)

    def __post_init__(self) -> None:
        for name in ("max_edge", "chord", "angle"):
            _refuse_non_positive("Coarsen", name, getattr(self, name))

    def applied(self, triangles: np.ndarray) -> np.ndarray:
        """``triangles`` coarsened, each surviving vertex one of theirs."""
        options = {} if self.angle is None else {"angle": self.angle}
        return coarsen_to_size(
            triangles, max_edge=self.max_edge, chord=self.chord, **options
        ).vertices


@dataclasses.dataclass(frozen=True)
class CadPlacement:
    """The rigid map from a drawing's frame to the case's: ``x -> matrix @ x + offset``, in metres.

    Attributes
    ----------
    matrix : tuple of tuple of float or None
        Three rows of three; a rotation, or a rotation and a reflection. Unset, the identity.
    offset : tuple of float or None
        Three components; unset, none.
    """

    matrix: tuple[tuple[float, ...], ...] | None = None
    offset: tuple[float, ...] | None = None

    def placement(self):
        """The CAD reader's :class:`~aquaflux.io.cad.Placement`, which checks the matrix."""
        from aquaflux.io.cad import Placement

        options = {}
        if self.matrix is not None:
            options["matrix"] = np.asarray(self.matrix, dtype=float)
        if self.offset is not None:
            options["offset"] = np.asarray(self.offset, dtype=float)
        return Placement(**options)


class _Drawings:
    """The CAD files a build reads, each read once however many surfaces and bodies name it."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._read: dict = {}

    def model(self, file: str, placement: CadPlacement | None):
        from aquaflux.io.cad import read_step

        key = (file, placement)
        if key not in self._read:
            self._read[key] = read_step(
                self._directory / file, None if placement is None else placement.placement()
            )
        return self._read[key]


@dataclasses.dataclass(frozen=True)
class PatchSurface:
    """A boundary patch of the mesh, as the reference a surface standing for it is checked against.

    Attributes
    ----------
    name : str
    triangles : np.ndarray, shape ``(n, 3, 3)``
        The patch's faces as triangles facing into the domain.
    centres : np.ndarray, shape ``(m, 3)``
        Its faces' centres.
    inward : np.ndarray, shape ``(m, 3)``
        Its faces' unit normals, into the domain.
    """

    name: str
    triangles: np.ndarray
    centres: np.ndarray
    inward: np.ndarray


@dataclasses.dataclass(frozen=True)
class SurfaceSource(abc.ABC):
    """Where a lamp's or a reflecting wall's triangles come from: :class:`CadSurface`,
    :class:`StlSurface`, or the mesh's own patch, :class:`MeshPatch`."""

    def triangles(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        """The surface's triangles, facing into the domain as ``patch`` does.

        Parameters
        ----------
        patch : PatchSurface
            The patch the surface stands for.
        directory : pathlib.Path
            The directory the case file sits in.
        drawings : _Drawings
            The CAD files already read.

        Returns
        -------
        np.ndarray, shape ``(n, 3, 3)``

        Raises
        ------
        ValueError
            If the surface's triangles disagree with one another about which way they face, or none of
            them lies square enough to its patch to say.
        """
        return _facing(self._read(patch, directory, drawings), patch, type(self).__name__)

    @abc.abstractmethod
    def _read(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        """The triangles as the source gives them, in either winding."""


@dataclasses.dataclass(frozen=True)
class MeshPatch(SurfaceSource):
    """The mesh's own patch: its faces, each cut into the fan of triangles about its centre.

    The fallback when the drawing is not to hand. A mesh has far more faces than a reflecting surface
    needs facets -- the facet-to-facet transfer is a dense matrix over them -- so a reflecting patch
    is normally coarsened.

    Attributes
    ----------
    coarsen : Coarsen or None
        Coarsen the patch's triangles; unset, they are used as they are.
    """

    coarsen: Coarsen | None = None

    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = ("coarsen",)

    def triangles(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        """The patch's triangles, already facing into the domain, coarsened if asked."""
        del directory, drawings
        return patch.triangles if self.coarsen is None else self.coarsen.applied(patch.triangles)

    def _read(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        return self.triangles(patch, directory, drawings)


@dataclasses.dataclass(frozen=True)
class StlSurface(SurfaceSource):
    """Triangles read from an STL file, in metres.

    Attributes
    ----------
    file : str
        The STL file, relative to the case file. Lengths are read as metres, unscaled.
    solids : tuple of str
        The file's named solids to take; empty, all of them.
    coarsen : Coarsen or None
        Coarsen the triangles read; unset, they are used as they are.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solids: tuple[str, ...] = ()
    coarsen: Coarsen | None = None

    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = ("coarsen",)

    def _read(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        del patch, drawings
        triangles = _stl_triangles(Path(directory) / self.file, self.solids, "StlSurface")
        return triangles if self.coarsen is None else self.coarsen.applied(triangles)


@dataclasses.dataclass(frozen=True)
class CadSurface(SurfaceSource):
    """A solid of a STEP drawing, triangulated with every vertex on its true surface.

    Needs the optional CAD kernel (``pip install aquaflux[cad]``).

    Attributes
    ----------
    file : str
        The STEP file, relative to the case file.
    solid : str
        The solid's name in the drawing.
    chord : float
        The farthest a triangle may lie from the surface, in metres.
    facet_size : float or None
        The facet size wanted, in metres (see :meth:`~aquaflux.io.cad.CadModel.triangles`); unset,
        only the chord is bounded.
    angle : float or None
        The largest angle, in radians, between the surface normals at a triangle's corners; unset,
        the reader's own.
    placement : CadPlacement or None
        Where the drawing sits in the case's frame; unset, where it was drawn.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solid: str
    chord: float
    facet_size: float | None = None
    angle: float | None = None
    placement: CadPlacement | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (CadModel.triangles,)
    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = ("facet_size",)

    def __post_init__(self) -> None:
        for name in ("chord", "facet_size", "angle"):
            _refuse_non_positive("CadSurface", name, getattr(self, name))

    def _read(self, patch: PatchSurface, directory: Path, drawings: _Drawings) -> np.ndarray:
        del patch, directory
        options = {"chord": self.chord}
        if self.facet_size is not None:
            options["facet_size"] = self.facet_size
        if self.angle is not None:
            options["angle"] = self.angle
        return np.asarray(
            drawings.model(self.file, self.placement).triangles(self.solid, **options)
        )


def _stl_triangles(path: Path, solids: tuple[str, ...], owner: str) -> np.ndarray:
    """The triangles of the named solids of an STL file, or all of them."""
    soup = read_stl(path)
    if not solids:
        return np.asarray(soup.vertices)
    missing = [name for name in solids if name not in soup.solid_names]
    if missing:
        raise ValueError(
            f"{owner}: {path} has no solid {', '.join(map(repr, missing))}; its solids are "
            f"{list(soup.solid_names)}."
        )
    wanted = np.isin(np.asarray(soup.solid_id), [soup.solid_names.index(name) for name in solids])
    return np.asarray(soup.vertices)[wanted]


def _facing(triangles: np.ndarray, patch: PatchSurface, source: str) -> np.ndarray:
    """``triangles`` turned to face into the domain, as the patch they stand for does.

    Each triangle's normal is compared with the inward normal of the patch's nearest face; a
    triangle at a crease, whose nearest face may be across it, is not counted.
    """
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError(f"{source} for patch {patch.name!r} gave no triangles.")
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    areal = length > 0.0
    _, nearest = cKDTree(patch.centres).query(triangles.mean(axis=1)[areal])
    cosine = np.sum(normal[areal] / length[areal, None] * patch.inward[nearest], axis=1)
    agree = int(np.sum(cosine > _DECISIVE_COSINE))
    disagree = int(np.sum(cosine < -_DECISIVE_COSINE))
    if agree and disagree:
        raise ValueError(
            f"{source} for patch {patch.name!r}: {agree} of its triangles face into the domain and "
            f"{disagree} face out of it, so some are wound the wrong way (they would emit or reflect "
            "nothing). Fix the winding in the file."
        )
    if not (agree or disagree):
        raise ValueError(
            f"{source} for patch {patch.name!r}: none of its triangles lies square to the patch's "
            "nearest face, so it cannot be told which way it faces -- is it the surface of this patch, "
            "placed where the mesh is?"
        )
    return triangles if agree else triangles[:, ::-1, :]


# --- What stands in the way ------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OccluderSpec(abc.ABC):
    """A body that shadows: a drawing's solid, an STL file's triangles, or mesh patches."""

    @abc.abstractmethod
    def body(self, directory: Path, drawings: _Drawings, patch_triangles):
        """The body, as the gathers take it.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.
        drawings : _Drawings
            The CAD files already read.
        patch_triangles : callable
            ``(names, allow_folded=...) -> (n, 3, 3)`` triangles of those mesh patches, facing into the
            domain (see :func:`~aquaflux.mesh.patch_triangles`).
        """

    def mesh_patches(self) -> tuple[str, ...]:
        """The mesh patches this body is made of; none by default."""
        return ()


@dataclasses.dataclass(frozen=True)
class PatchBody(OccluderSpec):
    """Mesh patches as a body of triangles (:class:`~aquaflux.radiation.TriangleBody`).

    The irradiance is not gathered on these patches by default: their face centres lie on the body.

    Attributes
    ----------
    patches : tuple of str
        The patches.
    sheet : bool or None
        ``True`` treats the patches as a sheet with no inside, as suits a surface with the domain on
        both sides of it, such as a perforated object hung in a room. ``None`` decides from each
        connected piece's topology.
    """

    patches: tuple[str, ...]
    sheet: bool | None = None

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("PatchBody names the patches the body is made of.")

    def body(self, directory: Path, drawings: _Drawings, patch_triangles) -> TriangleBody:
        """A :class:`~aquaflux.radiation.TriangleBody` of the patches' triangles."""
        del directory, drawings
        # Facing into the domain is facing out of the solid the patches bound, the body's convention. A
        # face whose centre fan folds over itself -- common on a snapped surface -- is kept: the fan
        # still covers the face, and a body only has to stand in the way.
        return TriangleBody.build(
            patch_triangles(self.patches, allow_folded=True), sheet=self.sheet
        )

    def mesh_patches(self) -> tuple[str, ...]:
        """:attr:`patches`."""
        return self.patches


@dataclasses.dataclass(frozen=True)
class StlBody(OccluderSpec):
    """An STL file's triangles as a body (:class:`~aquaflux.radiation.TriangleBody`).

    Attributes
    ----------
    file : str
        The STL file, relative to the case file, in metres.
    solids : tuple of str
        The file's named solids to take; empty, all of them.
    sheet : bool or None
        As for :class:`PatchBody`.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solids: tuple[str, ...] = ()
    sheet: bool | None = None

    def body(self, directory: Path, drawings: _Drawings, patch_triangles) -> TriangleBody:
        """A :class:`~aquaflux.radiation.TriangleBody` of the file's triangles."""
        del drawings, patch_triangles
        return TriangleBody.build(
            _stl_triangles(Path(directory) / self.file, self.solids, "StlBody"), sheet=self.sheet
        )


@dataclasses.dataclass(frozen=True)
class CadSolid(OccluderSpec):
    """A solid of a STEP drawing, recognized as an exact body (:meth:`~aquaflux.io.cad.CadModel.solid`).

    Attributes
    ----------
    file : str
        The STEP file, relative to the case file.
    solid : str
        The solid's name in the drawing.
    placement : CadPlacement or None
        Where the drawing sits in the case's frame.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solid: str
    placement: CadPlacement | None = None

    def body(self, directory: Path, drawings: _Drawings, patch_triangles):
        """The solid as an exact body."""
        del directory, patch_triangles
        return drawings.model(self.file, self.placement).solid(self.solid)


@dataclasses.dataclass(frozen=True)
class CadFluid(OccluderSpec):
    """The medium held by solids of a STEP drawing: everything they are not shadows
    (:meth:`~aquaflux.io.cad.CadModel.fluid`) -- a vessel and its pipes, drawn as the water they hold.

    Attributes
    ----------
    file : str
        The STEP file, relative to the case file.
    solids : tuple of str
        The solids that together are the medium.
    placement : CadPlacement or None
        Where the drawing sits in the case's frame.
    tolerance : float or None
        How far outside the medium, in metres, a point may lie before it is called embedded in the
        wall; a mesh's cell centres sit a rounding off the surface they were snapped to. Unset, none.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solids: tuple[str, ...]
    placement: CadPlacement | None = None
    tolerance: float | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (CadModel.fluid,)

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("CadFluid names the solids that hold the medium.")

    def body(self, directory: Path, drawings: _Drawings, patch_triangles):
        """The medium's complement as one exact body."""
        del directory, patch_triangles
        options = {} if self.tolerance is None else {"tolerance": self.tolerance}
        return drawings.model(self.file, self.placement).fluid(*self.solids, **options)


# --- Where the light is wanted ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Receivers:
    """Where the light is gathered.

    Attributes
    ----------
    cells : bool
        The fluence rate at every cell centre.
    patches : tuple of str or None
        The patches whose face centres the irradiance is gathered at. Unset, every wall patch that
        is not part of a :class:`PatchBody`; a lamp's own faces are never gathered at, since they lie
        on the emitting surface.
    """

    cells: bool = True
    patches: tuple[str, ...] | None = None
