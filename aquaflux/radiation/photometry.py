"""A measured luminaire: IES LM-63 photometric files and the angular profile they describe.

An LM-63 file (the IESNA standard photometric format, "an IES file") tabulates a luminaire's
radiant or luminous **intensity** -- power per steradian -- over a grid of directions. In the
**Type C** convention used here, a direction is given by two angles in the luminaire's own frame:

- the **vertical angle** ``gamma``, measured from the luminaire's axis (``gamma = 0`` is straight
  along the beam), and
- the **horizontal angle** ``h``, measured around that axis from a reference direction ``up``
  (``h = 0``), increasing towards ``axis x up`` (``h = 90``).

:func:`read_ies` parses a file into a :class:`Photometry`, which knows the table, its symmetry
and its total flux; :meth:`Photometry.profile` turns it into a :class:`PhotometricProfile`, the
angular distribution an emitting facet of a :class:`~aquaflux.radiation.surfaces.Surfaces` set
can carry. The profile's axis is the facet's own outward normal, so a flat lamp window facing
down emits the table about the downward direction; ``up`` fixes the rotation about it.

**Between the tabulated directions the intensity is bilinear in** ``(gamma, h)`` **in degrees**,
clamped (not extrapolated) outside the tabulated vertical range. A table covering only part of
the circle is completed by the symmetry its last horizontal angle declares: a single angle is
rotationally symmetric, a last angle of 90 degrees is symmetric in each quadrant, one of 180 is
mirror-symmetric about the 0-180 plane, and anything else is a full table closed periodically.
These are the LM-63 conventions, and the ones of-optical-radiation's ``iesEmitter`` boundary
condition follows, so the two solvers read one file as one continuous distribution.

**Normalization is exact.** A bilinear table weighted by ``sin gamma`` integrates in closed form
cell by cell, so the flux, and the constant that makes the profile integrate to one, carry no
quadrature error. The profile is normalized over the facet's **front hemisphere** -- the
directions a flat emitter can send light into -- so facets given exitance ``P / A`` radiate
exactly ``P`` whatever the table holds behind its axis.

⚠️ **Units are the file's.** LM-63 intensities are candela by default, and a far-ultraviolet file
may state milliwatts per steradian in a keyword instead; :attr:`Photometry.flux` is in the file's
intensity unit times a steradian, and converting it is the caller's decision, never this module's.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.profiles import Profile
from aquaflux.vectors import dot, norm_squared, reflect

__all__ = ["MIN_COSINE", "PhotometricProfile", "Photometry", "read_ies"]

# Below this cosine a facet's radiance is taken as the intensity over this cosine rather than over
# a cosine going to zero: a table with light at gamma = 90 degrees describes a finite intensity
# along the facet's plane, which no finite radiance from a flat facet can deliver. The same floor
# of-optical-radiation's iesEmitter applies.
MIN_COSINE = 1e-3

# How close a last horizontal angle must be to 90 or 180 degrees to declare that symmetry.
_SYMMETRY_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class Photometry:
    """A Type C photometric table, as read from an LM-63 file.

    Attributes
    ----------
    vertical : np.ndarray, shape ``(n_vertical,)``
        Vertical angles ``gamma`` in degrees, strictly ascending, within ``[0, 180]``.
    horizontal : np.ndarray, shape ``(n_horizontal,)``
        Horizontal angles ``h`` in degrees as stored, strictly ascending.
    intensity : np.ndarray, shape ``(n_horizontal, n_vertical)``
        Intensity at each ``(h, gamma)``, in the file's unit, with the file's multiplier and
        ballast factor applied and negative entries clipped to zero.
    keywords : dict
        The ``[KEYWORD] value`` lines of the header, keyword to value.
    opening : tuple
        Width, length and height of the luminous opening, in ``opening_unit``.
    opening_unit : str
        ``"feet"`` or ``"metres"``, from the file's units type.
    """

    vertical: np.ndarray
    horizontal: np.ndarray
    intensity: np.ndarray
    keywords: dict
    opening: tuple
    opening_unit: str

    def __post_init__(self):
        if self.vertical.ndim != 1 or len(self.vertical) < 2:
            msg = "a photometric table needs at least two vertical angles"
            raise ValueError(msg)
        if self.horizontal.ndim != 1 or len(self.horizontal) < 1:
            msg = "a photometric table needs at least one horizontal angle"
            raise ValueError(msg)
        if np.any(np.diff(self.vertical) <= 0) or np.any(np.diff(self.horizontal) <= 0):
            msg = "vertical and horizontal angles must be strictly ascending"
            raise ValueError(msg)
        if self.intensity.shape != (len(self.horizontal), len(self.vertical)):
            msg = (
                f"intensity has shape {self.intensity.shape}; expected "
                f"({len(self.horizontal)}, {len(self.vertical)})"
            )
            raise ValueError(msg)
        if self.vertical[0] < 0.0 or self.vertical[-1] > 180.0:
            msg = f"vertical angles must lie in [0, 180] degrees; got {self.vertical[[0, -1]]}"
            raise ValueError(msg)
        if self.symmetry == "full" and self.horizontal[-1] - self.horizontal[0] > 360.0:
            msg = f"horizontal angles span more than a turn: {self.horizontal[[0, -1]]}"
            raise ValueError(msg)

    @property
    def symmetry(self) -> str:
        """``"rotational"``, ``"quadrant"``, ``"bilateral"`` or ``"full"``, from the stored range."""
        if len(self.horizontal) == 1:
            return "rotational"
        last = self.horizontal[-1]
        if abs(last - 90.0) < _SYMMETRY_TOLERANCE:
            return "quadrant"
        if abs(last - 180.0) < _SYMMETRY_TOLERANCE:
            return "bilateral"
        return "full"

    def full_circle(self) -> tuple[np.ndarray, np.ndarray]:
        """The table completed round the whole circle by its symmetry.

        Returns
        -------
        horizontal : np.ndarray, shape ``(m,)``
            Horizontal angles in degrees, ascending from the first angle to that angle plus 360,
            so the last row repeats the first.
        intensity : np.ndarray, shape ``(m, n_vertical)``
            The intensity at each of them. Interpolating linearly in ``h`` between these rows is
            the same as folding ``h`` by the symmetry and interpolating in the stored table.
        """
        h, table = self.horizontal, self.intensity
        kind = self.symmetry
        if kind == "rotational":
            return np.array([h[0], h[0] + 360.0]), np.vstack([table, table])
        if kind == "full":
            if abs(h[-1] - h[0] - 360.0) < _SYMMETRY_TOLERANCE:
                return h, table
            return np.append(h, h[0] + 360.0), np.vstack([table, table[:1]])
        # A mirror symmetry: every angle of the circle folds exactly onto a stored one.
        if kind == "bilateral":
            images = [h, 360.0 - h]
        else:
            images = [h, 180.0 - h, 180.0 + h, 360.0 - h]
        full = np.unique(np.round(np.concatenate(images), 9))
        folded = np.array([_fold(angle, kind) for angle in full])
        rows = np.minimum(np.searchsorted(h, folded - 1e-9), len(h) - 1)
        if not np.allclose(h[rows], folded, atol=1e-6):
            msg = f"a {kind}-symmetric table's folded angles do not land on its stored angles"
            raise ValueError(msg)
        return full, table[rows]

    @property
    def flux(self) -> float:
        """Total flux over the tabulated vertical range, in the file's intensity unit x sr.

        The exact integral of the bilinear table round the circle and across the stored range of
        ``gamma``. For a table covering ``[0, 90]`` degrees it equals
        :meth:`front_hemisphere_flux`.
        """
        h, table = self.full_circle()
        return _bilinear_integral(np.radians(h), np.radians(self.vertical), table)

    def front_hemisphere_flux(self) -> float:
        """The flux a flat emitter with this table sends into its front hemisphere.

        As :attr:`flux`, but over ``gamma`` in ``[0, 90]`` degrees with the table clamped outside
        its stored range -- the distribution the profile evaluates -- so a table stored only to
        60 degrees is continued at its 60-degree values, and one stored to 180 is cut at 90.
        """
        h, table = self.full_circle()
        vertical, table = _clamped_to(self.vertical, table, 0.0, 90.0)
        return _bilinear_integral(np.radians(h), np.radians(vertical), table)

    def profile(self, up) -> PhotometricProfile:
        """The table as a normalized angular distribution for emitting facets.

        Parameters
        ----------
        up : array_like, shape ``(3,)``
            The direction of ``h = 0`` in the scene's frame. Only its part perpendicular to each
            facet's normal is used, so it must not be parallel to one;
            :func:`~aquaflux.radiation.checks.check_profiles` refuses a set where it is.

        Returns
        -------
        PhotometricProfile
            Integrating to one over each facet's front hemisphere.
        """
        h, table = self.full_circle()
        return PhotometricProfile(
            vertical=jnp.asarray(np.radians(self.vertical)),
            horizontal=jnp.asarray(np.radians(h)),
            table=jnp.asarray(table / self.front_hemisphere_flux()),
            up=jnp.asarray(up, dtype=float),
        )


class PhotometricProfile(Profile):
    """A luminaire's measured intensity, tabulated over both angles about the facet normal.

    ``gamma`` is the angle from the facet's outward normal, and ``h`` the angle round it from
    ``up`` projected onto the facet's plane, positive towards ``normal x up`` -- or, in a mirror
    image (:meth:`mirrored`), towards its opposite. Bilinear in the two angles between tabulated
    directions, clamped beyond the tabulated vertical range; built by :meth:`Photometry.profile`,
    which completes the table round the circle and normalizes it.

    Emits nothing behind the facet. Where ``cos gamma`` is below :data:`MIN_COSINE` the radiance
    is the intensity over :data:`MIN_COSINE`, so the contract
    ``radiance_per_exitance * cos gamma == intensity_fraction`` holds for
    ``cos gamma >= MIN_COSINE`` and the radiance stays finite along the facet's plane.

    Attributes
    ----------
    vertical : jnp.ndarray, shape ``(n_vertical,)``
        Vertical angles in radians, ascending.
    horizontal : jnp.ndarray, shape ``(m,)``
        Horizontal angles in radians, ascending, spanning exactly one turn.
    table : jnp.ndarray, shape ``(m, n_vertical)``
        Fraction of the total power per steradian at each tabulated direction. A differentiable
        leaf.
    up : jnp.ndarray, shape ``(3,)``
        The direction of ``h = 0``.
    handedness : int
        ``1`` when ``h`` increases towards ``normal x up``, as the table was measured; ``-1`` in
        a mirror image, which turns the other way. A label rather than a leaf: it is a sign,
        and nothing is differentiated with respect to it.
    """

    dark_behind: ClassVar[bool] = True

    vertical: jnp.ndarray
    horizontal: jnp.ndarray
    table: jnp.ndarray
    up: jnp.ndarray
    handedness: int = eqx.field(static=True, default=1)

    def angles(self, direction, normal) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``cos gamma``, and ``gamma`` and ``h`` in radians, of each direction.

        Parameters
        ----------
        direction : jnp.ndarray, shape ``(..., 3)``
            Unit vectors from the source.
        normal : jnp.ndarray, shape ``(..., 3)``
            The facets' outward unit normals, broadcasting against ``direction``.

        Returns
        -------
        tuple of jnp.ndarray, each shape ``(...)``
            ``h`` lies in one turn from the table's first horizontal angle.
        """
        direction = jnp.asarray(direction, dtype=float)
        normal = jnp.asarray(normal, dtype=float)
        cosine = dot(direction, normal)
        # gamma from atan2 of the sine and cosine, not arccos, which loses accuracy near the axis
        # and has an unbounded derivative there; the sine is the length of the direction's part
        # in the facet's plane, under a guarded root so its derivative is finite at zero.
        in_plane = direction - cosine[..., None] * normal
        sine_squared = norm_squared(in_plane)
        positive = sine_squared > 0.0
        sine = jnp.where(positive, jnp.sqrt(jnp.where(positive, sine_squared, 1.0)), 0.0)
        gamma = jnp.arctan2(sine, cosine)
        # h from up and normal x up. Both are dotted with the in-plane part only, which is normal
        # to the facet, so up's own component along the normal drops out; and atan2 needs no unit
        # vectors, so neither is normalized. The handedness turns it the other way in a mirror
        # image, where the cross product of two reflected vectors is the reflection negated.
        # On the axis the in-plane part is zero and h is arctan2(0, 0): zero in value, as the
        # table's convention for the pole wants, but its derivative is 0/0. A where() after the
        # arctan2 would still differentiate it, so the operands are substituted before it instead,
        # with (0, 1), whose angle is the same zero and whose derivative is finite.
        across = jnp.where(
            positive, self.handedness * dot(in_plane, jnp.cross(normal, self.up)), 0.0
        )
        along = jnp.where(positive, dot(in_plane, self.up), 1.0)
        h = jnp.arctan2(across, along)
        start = self.horizontal[0]
        return cosine, gamma, start + jnp.mod(h - start, 2.0 * jnp.pi)

    def _interpolate(self, gamma, h):
        """The bilinear table at ``(gamma, h)``, clamped in ``gamma``."""
        vertical, horizontal = self.vertical, self.horizontal
        gamma = jnp.clip(gamma, vertical[0], vertical[-1])
        iv = jnp.clip(jnp.searchsorted(vertical, gamma, side="right") - 1, 0, len(vertical) - 2)
        fv = (gamma - vertical[iv]) / (vertical[iv + 1] - vertical[iv])
        ih = jnp.clip(jnp.searchsorted(horizontal, h, side="right") - 1, 0, len(horizontal) - 2)
        fh = jnp.clip((h - horizontal[ih]) / (horizontal[ih + 1] - horizontal[ih]), 0.0, 1.0)
        low = (1.0 - fv) * self.table[ih, iv] + fv * self.table[ih, iv + 1]
        high = (1.0 - fv) * self.table[ih + 1, iv] + fv * self.table[ih + 1, iv + 1]
        return (1.0 - fh) * low + fh * high

    def intensity_fraction(self, direction, normal) -> jnp.ndarray:
        """The tabulated fraction per steradian, zero behind the facet."""
        cosine, gamma, h = self.angles(direction, normal)
        return jnp.where(cosine > 0.0, self._interpolate(gamma, h), 0.0)

    def radiance_per_exitance(self, direction, normal) -> jnp.ndarray:
        """The fraction over ``max(cos gamma, MIN_COSINE)``, zero behind the facet."""
        cosine, gamma, h = self.angles(direction, normal)
        value = self._interpolate(gamma, h) / jnp.maximum(cosine, MIN_COSINE)
        return jnp.where(cosine > 0.0, value, 0.0)

    def mirrored(self, normal) -> PhotometricProfile:
        """The table seen in a plane mirror: ``up`` reflected, and ``h`` turning the other way.

        Reflecting ``up`` alone is not enough. A reflection reverses handedness, so a direction
        that sat at ``h`` towards ``normal x up`` from the source sits, in the image, at ``h``
        towards the *opposite* of the image's own ``normal x up`` -- read with the same sense of
        rotation, every asymmetric table would come back turned the wrong way round its axis.
        """
        return dataclasses.replace(
            self,
            up=reflect(self.up, jnp.asarray(normal, dtype=float)),
            handedness=-self.handedness,
        )

    def refuse_normals(self, normals) -> str | None:
        """Why facets with these normals cannot carry this profile, or ``None`` if they can.

        ``h`` is measured from ``up``'s part in the facet's plane, which does not exist when ``up``
        is parallel to the normal: the angle would silently read zero.
        """
        up = np.asarray(self.up, dtype=float)
        up = up / np.linalg.norm(up)
        normals = np.asarray(normals, dtype=float)
        parallel = np.linalg.norm(np.cross(normals, up), axis=-1) < 1e-6
        if parallel.any():
            return (
                f"{int(parallel.sum())} facet(s) have a normal parallel to the profile's up "
                f"direction {np.asarray(self.up).tolist()}, so the horizontal angle is undefined "
                "there; give an up direction lying in the emitting window's plane"
            )
        return None


def read_ies(path) -> Photometry:
    """Parse an IES LM-63 file with Type C photometry and no tilt.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    Photometry
        The table, with the file's candela multiplier and ballast factor applied.

    Raises
    ------
    ValueError
        If the file has no ``TILT=`` line or a tilt table, is not Type C, or holds more or fewer
        numbers than its header declares.
    """
    text = Path(path).read_text(errors="replace")
    head, marker, body = text.partition("TILT=")
    if not marker:
        msg = f"{path}: no TILT= line, so this is not an LM-63 file"
        raise ValueError(msg)
    tilt, _, body = body.partition("\n")
    if tilt.strip().upper() != "NONE":
        msg = f"{path}: only TILT=NONE is supported; got TILT={tilt.strip()}"
        raise ValueError(msg)
    keywords = dict(re.findall(r"^\[([^\]]+)\][ \t]*(.*?)[ \t]*\r?$", head, re.M))
    try:
        values = [float(token) for token in body.split()]
    except ValueError as error:
        msg = f"{path}: a non-numeric token follows the TILT line ({error})"
        raise ValueError(msg) from None
    if len(values) < 13:
        msg = f"{path}: the header block after the TILT line is truncated"
        raise ValueError(msg)
    multiplier = values[2]
    n_vertical, n_horizontal, kind, units = (int(v) for v in values[3:7])
    ballast = values[10]
    if kind != 1:
        msg = f"{path}: only Type C photometry (type 1) is supported; got type {kind}"
        raise ValueError(msg)
    expected = 13 + n_vertical + n_horizontal + n_vertical * n_horizontal
    if len(values) != expected:
        msg = (
            f"{path}: expected {expected} numbers after the TILT line for a {n_vertical} x "
            f"{n_horizontal} table; found {len(values)}"
        )
        raise ValueError(msg)
    table = np.asarray(values[13:])
    intensity = table[n_vertical + n_horizontal :].reshape(n_horizontal, n_vertical)
    return Photometry(
        vertical=table[:n_vertical],
        horizontal=table[n_vertical : n_vertical + n_horizontal],
        intensity=np.maximum(intensity * multiplier * ballast, 0.0),
        keywords=keywords,
        opening=tuple(values[7:10]),
        opening_unit={1: "feet", 2: "metres"}.get(units, f"units type {units}"),
    )


def _fold(angle: float, kind: str) -> float:
    """An angle in ``[0, 360]`` degrees folded onto a symmetric table's stored range."""
    if kind == "quadrant":
        if angle > 270.0:
            return 360.0 - angle
        if angle > 180.0:
            return angle - 180.0
        if angle > 90.0:
            return 180.0 - angle
        return angle
    return 360.0 - angle if angle > 180.0 else angle


def _clamped_to(vertical, table, low, high):
    """The table's vertical axis cut or extended to ``[low, high]`` degrees, values clamped."""
    inside = vertical[(vertical > low) & (vertical < high)]
    points = np.unique(np.concatenate([[low, high], inside]))
    return points, np.stack([np.interp(points, vertical, row) for row in table])


def _bilinear_integral(horizontal, vertical, table) -> float:
    """``integral of I(gamma, h) sin gamma dgamma dh`` over the table, exactly, for bilinear ``I``.

    Angles in radians. Over ``h`` the integrand is linear in each interval, so the trapezoid rule
    is exact; over ``gamma``, each interval's two linear hat functions times ``sin gamma``
    integrate in closed form.
    """
    a, b = vertical[:-1], vertical[1:]
    s0 = np.cos(a) - np.cos(b)
    s1 = (np.sin(b) - b * np.cos(b)) - (np.sin(a) - a * np.cos(a))
    lower, upper = (b * s0 - s1) / (b - a), (s1 - a * s0) / (b - a)
    per_row = table[:, :-1] @ lower + table[:, 1:] @ upper
    return float(np.sum(0.5 * np.diff(horizontal) * (per_row[:-1] + per_row[1:])))
