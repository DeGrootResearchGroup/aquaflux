"""A computer-aided design (CAD) model, read once and asked for exact bodies and triangles.

:func:`read_step` reads a STEP file into a :class:`CadModel`, and the model hands out what the rest
of the package consumes, all in the case's frame and in metres:

* :meth:`CadModel.solid` — one named solid as an exact body from :mod:`aquaflux.solids`;
* :meth:`CadModel.fluid` — several named solids as the fluid they hold, an
  :class:`~aquaflux.solids.Outside`, for shadowing a vessel by what it contains;
* :meth:`CadModel.triangles` — a solid's surface as triangles, for an emitting surface.

**Every body handed out has been checked against the solid it came from, and a body that fails is
refused rather than returned.** The recognition rules propose a body; the kernel then samples points
on both boundaries — the proposal's and the solid's — and measures how far each lies from the other
boundary. Every point must be within the file's own declared geometric tolerance, the precision the
drawing was exported at, beyond which no comparison with it means anything. That is what makes a
short list of rules safe on an arbitrary file: a rule that is wrong about a shape is caught here,
where the alternative is a vessel shadowed by the wrong geometry with nothing to say so.

⚠️ **Why distance between boundaries and not difference in volume.** A volume test cannot be set.
Tight enough to notice a thin missing feature — a baffle, a fin — it fails a correct file, because
an exported intersection curve is a spline fitted to the file's tolerance and moves the volume by
more than such a feature holds. Loose enough to pass the file, it waves the baffle through. A missing
baffle is far from the proposal's boundary however little volume it has, and every face is sampled
at least at its own corners, so the distance sees it.

**A solid whose end is cut to fit a neighbour is checked with the neighbour.** A pipe standing on a
vessel ends in a saddle-shaped face lying on the vessel's wall. Its cylinder is not the pipe — it
runs on into the vessel — but the union of the two cylinders is exactly the union of the two
solids, and a fluid is a union. So :meth:`CadModel.fluid` checks the union, and
:meth:`CadModel.solid` refuses such a solid on its own.
"""

from __future__ import annotations

import os

import numpy as np

from aquaflux.io.cad.faces import SolidDescription
from aquaflux.io.cad.placement import Placement
from aquaflux.io.cad.recognize import DEFAULT_RULES, Recognition, recognize
from aquaflux.solids import Outside, Solid

__all__ = ["CadModel", "InexactBody", "read_step"]

#: How many sample spacings span a region, when the spacing is not given: the boundaries are cut into
#: pieces this many times smaller than their extent before their mesh nodes are taken as samples.
_SAMPLES_ACROSS = 16


class InexactBody(ValueError):
    """A proposed body's boundary is further from the solid's than the tolerance allows."""


class CadModel:
    """Named solids read from a CAD file, and the bodies and triangles made from them.

    Build one with :func:`read_step`. The model holds the kernel's solids and the kernel that reads
    them; everything it returns is plain arrays and :mod:`aquaflux.solids` bodies.

    Parameters
    ----------
    shapes : mapping of str to kernel solid
        The solids, by name, already in the case's frame.
    kernel : object
        What answers questions about ``shapes`` (see :mod:`aquaflux.io.cad.kernel`).
    rules : sequence of RecognitionRule, optional
        The recognition rules, tried in order.
    boundary_tolerance : float, optional
        How far, in metres, a sampled point of either boundary may lie from the other before a
        proposal is refused. Unset, it is the largest geometric tolerance the file declares on the
        solids being compared: a drawing exported at 10 micrometres cannot be checked to one.
    sample_spacing : float, optional
        How far apart, in metres, the sampled boundary points are. Unset, a sixteenth of the
        compared region's extent. Every face is sampled at its corners whatever this is, so a
        missing feature is seen however small; the spacing decides how finely a smooth departure
        in the middle of a face is looked for.
    """

    def __init__(
        self,
        shapes,
        kernel,
        rules=DEFAULT_RULES,
        boundary_tolerance: float | None = None,
        sample_spacing: float | None = None,
    ):
        self._shapes = dict(shapes)
        self._kernel = kernel
        self._rules = tuple(rules)
        self._boundary_tolerance = boundary_tolerance
        self._sample_spacing = sample_spacing
        self._descriptions: dict[str, SolidDescription] = {}
        self._recognitions: dict[str, Recognition] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """The solids' names, in the order the file lists them."""
        return tuple(self._shapes)

    def _shape(self, name: str):
        try:
            return self._shapes[name]
        except KeyError:
            msg = f"no solid named {name!r}; the model has {', '.join(map(repr, self.names))}"
            raise KeyError(msg) from None

    def describe(self, name: str) -> SolidDescription:
        """What a solid's faces are, as :mod:`aquaflux.io.cad.faces` records."""
        if name not in self._descriptions:
            self._descriptions[name] = self._kernel.describe(name, self._shape(name))
        return self._descriptions[name]

    def recognition(self, name: str) -> Recognition:
        """The proposal the recognition rules make for a solid, before it is checked."""
        if name not in self._recognitions:
            self._recognitions[name] = recognize(self.describe(name), self._rules)
        return self._recognitions[name]

    def discrepancy(self, names) -> tuple[float, float]:
        """How far the recognized bodies of ``names`` are from the solids, together.

        Returns
        -------
        distance : float
            The largest distance, in metres, from a sampled point of either boundary — the union of
            the proposed bodies, the union of the solids — to the other. Zero when the description
            is exact, to the precision of the sampling.
        tolerance : float
            The distance it is held to: the given ``boundary_tolerance``, or the file's own.
        """
        names = tuple(names)
        solids = [self._shape(name) for name in names]
        proposed = []
        for name, shape in zip(names, solids, strict=True):
            recognition = self.recognition(name)
            # A proven body is its solid by construction, so the solid stands in for it.
            proposed.append(
                shape if recognition.proven else self._kernel.shape_of(recognition.body)
            )
        truth, claim = self._kernel.fuse(solids), self._kernel.fuse(proposed)
        spacing = self._sample_spacing
        if spacing is None:
            spacing = max(self.describe(name).extent for name in names) / _SAMPLES_ACROSS
        distance = 0.0
        for sampled, other in ((truth, claim), (claim, truth)):
            points = self._kernel.triangulate(sampled, spacing, 0.5, spacing).reshape(-1, 3)
            points = np.unique(points, axis=0)
            distance = max(distance, float(np.max(self._kernel.distances(points, other))))
        tolerance = self._boundary_tolerance
        if tolerance is None:
            tolerance = self._kernel.tolerance(solids)
        return distance, float(tolerance)

    def _checked(self, names) -> None:
        distance, tolerance = self.discrepancy(names)
        if distance > tolerance:
            rules = ", ".join(f"{name!r}: {self.recognition(name).rule}" for name in names)
            msg = (
                f"the bodies proposed for {', '.join(map(repr, names))} lie up to {distance:.3g} m "
                f"from the solids' surface, against a tolerance of {tolerance:.3g} m, so they are "
                f"refused rather than used ({rules})"
            )
            raise InexactBody(msg)

    def solid(self, name: str) -> Solid:
        """One solid as an exact body.

        Raises
        ------
        UnrecognizedSolid
            If no recognition rule applies.
        InexactBody
            If the proposed body is not the solid, or is exact only together with a neighbour —
            for which use :meth:`fluid`.
        """
        recognition = self.recognition(name)
        if not recognition.stands_alone:
            msg = (
                f"{name!r} has a face cut to fit a neighbour, so its body is exact only together "
                "with that neighbour; describe them together with CadModel.fluid"
            )
            raise InexactBody(msg)
        if not recognition.proven:
            self._checked((name,))
        return recognition.body

    def fluid(self, *names: str, tolerance: float = 0.0) -> Outside:
        """The fluid held by several solids — everything they are not — as one body.

        Parameters
        ----------
        *names : str
            The solids that together are the fluid: a vessel and the pipes joining it, say.
        tolerance : float, optional
            Passed to :class:`~aquaflux.solids.Outside`: how far outside the fluid, in metres, a
            point may lie before it is called embedded in the wall. A mesh's cell centres sit a
            rounding off the surface they were snapped to.

        Raises
        ------
        UnrecognizedSolid
            If a solid has no recognition rule.
        InexactBody
            If the proposed bodies' union is not the solids' union.
        """
        if not names:
            msg = "fluid needs the names of the solids holding it"
            raise ValueError(msg)
        self._checked(names)
        return Outside(*(self.recognition(name).body for name in names), tolerance=tolerance)

    def triangles(
        self, name: str, *, chord: float, facet_size: float | None = None, angle: float = 0.5
    ) -> np.ndarray:
        """A solid's surface as triangles wound outward, every vertex on the true surface.

        Parameters
        ----------
        name : str
        chord : float
            The largest distance, in metres, a triangle may lie from the surface.
        facet_size : float, optional
            The size of facet wanted, in metres. The surface is cut by a grid of planes this far
            apart before it is meshed, so every facet fits in a cube of that side: no edge is longer
            than ``sqrt(3)`` times it, and in practice edges come out close to it. Unset, only the
            chord is bounded, and a straight cylinder then comes out as slivers running its whole
            length — one absorption sample and one emission value along the entire tube.
        angle : float, optional
            The largest angle, in radians, between the surface normals at a triangle's corners.

        Returns
        -------
        np.ndarray, shape ``(n_triangles, 3, 3)``
            Ready for :meth:`aquaflux.radiation.Surfaces.from_triangles`.
        """
        return self._kernel.triangulate(self._shape(name), chord, angle, facet_size)


def read_step(path: str | os.PathLike, placement: Placement | None = None, **options) -> CadModel:
    """Read a STEP file (the ISO 10303 CAD exchange format) into a :class:`CadModel`.

    Needs the optional CAD kernel: ``pip install aquaflux[cad]``.

    Parameters
    ----------
    path : str or path-like
    placement : Placement, optional
        The rigid map from the drawing's frame to the case's; the identity if unset. Lengths are
        always read in metres, whatever unit the file was drawn in.
    **options
        Passed to :class:`CadModel`: ``rules``, ``boundary_tolerance`` and ``sample_spacing``.

    Returns
    -------
    CadModel

    Examples
    --------
    A reactor drawn with its axis along ``y`` and meshed with it along ``x``::

        cad = read_step("reactor.step", Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        water = cad.fluid("reactor_body", "inlet_pipe", "outlet_pipe")
        lamp = cad.triangles("lamp", chord=2e-5, facet_size=4e-3)
    """
    try:
        from aquaflux.io.cad.kernel import OpenCascade
    except ImportError as error:
        msg = (
            "reading a CAD file needs the OpenCASCADE kernel's Python binding, which is optional: "
            "install it with `pip install aquaflux[cad]`"
        )
        raise ImportError(msg) from error
    kernel = OpenCascade()
    shapes = kernel.read_step(path, placement or Placement())
    return CadModel(shapes, kernel, **options)
