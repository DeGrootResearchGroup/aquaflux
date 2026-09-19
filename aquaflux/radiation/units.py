"""The two conversions between how ultraviolet equipment is specified and what the model wants.

Everything else in this package works in SI: an exitance in W/m², an absorption coefficient in
reciprocal metres, a radiant power in watts. Ultraviolet reactors are not specified that way.
A lamp is bought by its **rated ultraviolet output in watts** and a water is characterized by
its **ultraviolet transmittance**, a percentage through a one-centimetre cell. Both conversions
are short, both are easy to get wrong in a way that produces a plausible field, and both were
being done by hand at every call site before this module existed.

Neither conversion is differentiable-by-accident: they are ordinary host-side arithmetic run
once while a scene is being set up, before anything is traced. A study that sweeps lamp power
should vary the watts and re-run :func:`lamp_exitance`, not differentiate through it — though
nothing stops the latter, since both are plain arithmetic on their arguments.
"""

from __future__ import annotations

import numpy as np

from aquaflux.radiation.surfaces import Surfaces

__all__ = ["absorption_from_uvt", "lamp_exitance"]

#: Path length of the cell an ultraviolet transmittance is quoted through, in metres. The
#: one-centimetre cell is the convention the whole water industry quotes against; a
#: transmittance given through any other path is a different number for the same water.
_QUOTED_PATH = 0.01


def absorption_from_uvt(uvt, *, path_length: float = _QUOTED_PATH):
    """Napierian absorption coefficient, in reciprocal metres, from an ultraviolet transmittance.

    ``UVT`` is the percentage of 254 nm light surviving a one-centimetre cell of the water —
    the number a water-quality report carries, and the one an ultraviolet reactor is sized
    against. Beer's law inverts it::

        a = -ln(UVT / 100) / path_length

    The result is what :class:`~aquaflux.radiation.absorption.UniformAbsorption` wants: a
    **napierian** coefficient per **metre**, matching the ``exp(-a r)`` this package attenuates
    with and the metres its geometry is in.

    ⚠️ **Two other conventions are in circulation and both are silently wrong here.** A
    *decadic* coefficient, paired with ``10^(-A r)``, is smaller by ``ln 10`` — a factor of
    2.3, which reads as clearer water rather than as an error. And a coefficient per
    *centimetre* is smaller by a hundred. A 95% UVT water is 5.129 per metre here; the same
    water is 2.228 decadic per metre, or 0.05129 napierian per centimetre.

    Parameters
    ----------
    uvt : float or array_like
        Ultraviolet transmittance as a **percentage**, in ``(0, 100]``. A value of 95 means 95%
        of the light survives the cell, so pure water approaches 100 and the coefficient
        approaches zero.
    path_length : float, optional
        The cell the transmittance was measured through, in metres. Defaults to one centimetre,
        which is what a report means unless it says otherwise.

    Returns
    -------
    numpy.ndarray or float
        The coefficient in reciprocal metres, the same shape as ``uvt``.

    Raises
    ------
    ValueError
        If ``uvt`` is outside ``(0, 100]``, or if ``path_length`` is not positive. A value at or
        below 1 also raises: it is far more often a fraction handed over as a percentage — 0.95
        for 95% — than it is a genuine sub-1% water, and the two readings give 466 against 5.13
        per metre, ninety times apart, which no later result would flag. A water that really is
        that opaque is outside the range ultraviolet reactors are built for; invert Beer's law
        by hand if you need one.

    Examples
    --------
    >>> float(absorption_from_uvt(95.0))
    5.129329438755057
    >>> float(absorption_from_uvt(100.0))
    0.0
    """
    if path_length <= 0.0:
        msg = f"path_length must be a positive length in metres; got {path_length}"
        raise ValueError(msg)
    fraction = np.asarray(uvt, dtype=float)
    if np.any(fraction > 100.0) or np.any(fraction <= 0.0):
        msg = f"uvt is a percentage in (0, 100]; got {np.asarray(uvt)}"
        raise ValueError(msg)
    if np.any(fraction <= 1.0):
        msg = (
            f"uvt is a percentage, not a fraction, and {np.min(fraction)} is almost certainly "
            "the latter -- 0.95 for 95% rather than a water transmitting under one percent. "
            "Pass 95.0: read as a fraction it gives 5.13 per metre and read as written it "
            "gives 466, ninety times apart, and nothing downstream would report it."
        )
        raise ValueError(msg)
    # The trailing zero is not redundant: at exactly 100% the logarithm is +0.0 and its
    # negation is -0.0, which is numerically the right answer and reads as a wrong one.
    return -np.log(fraction / 100.0) / path_length + 0.0


def lamp_exitance(surfaces: Surfaces, by_solid: dict[str, float], *, default: float = 0.0):
    """Per-facet exitance in W/m², from each lamp body's rated ultraviolet output in watts.

    A lamp is specified by the ultraviolet power it emits, and a surface facet by the power it
    emits per unit area, so somewhere the rating has to be divided by the emitting area. Doing
    it here, against the **triangulation's own** area, is what makes the model radiate exactly
    the rating: ``sum(M A)`` over a body comes back to its watts at any refinement, because the
    same areas appear in both.

    Dividing by the shape's analytic area instead does not. An inscribed triangulation of a
    cylinder undershoots ``pi d L`` — measured on a 23 mm by 400 mm sleeve, by 2.55% at eight
    sectors, 0.64% at sixteen and 0.16% at thirty-two — so a hand-computed exitance makes the
    model quietly radiate that much less than the lamp it represents, always in the same
    direction, and no later check reports it.

    ⚠️ **Which body you hand the rating to is a modelling decision this cannot make for you.**
    A lamp's rating is the power leaving its envelope, while the geometry in a reactor model is
    usually the quartz sleeve around it, which is slightly larger. Giving the sleeve the rating
    is right if you intend the sleeve to be the emitter and wrong if you meant the envelope, and
    the two differ by the area ratio. Nothing in the signature can tell them apart.

    Parameters
    ----------
    surfaces : Surfaces
        The set the exitance is for. Only its areas and body labels are read.
    by_solid : dict of str to float
        Rated ultraviolet output per named body, in watts. Bodies the mapping omits take
        ``default``.
    default : float, optional
        Exitance for bodies with no rating, in W/m². Zero — a wall does not emit — which is why
        this is an exitance rather than another power: there is no area to divide by.

    Returns
    -------
    jnp.ndarray, shape ``(n_facets,)``
        Exitance per facet, ready for
        :meth:`Surfaces.with_optics <aquaflux.radiation.surfaces.Surfaces.with_optics>`.

    Raises
    ------
    KeyError
        If ``by_solid`` names a body the surface set does not contain — a misspelled name would
        otherwise leave the lamp dark.
    ValueError
        If a rated body has no area. Its facets are point sources, which carry radiant power
        rather than exitance; pass the watts to ``power`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> from aquaflux.radiation.surfaces import Surfaces
    >>> square = np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0]],
    ...                    [[0, 0, 0], [2, 2, 0], [0, 2, 0]]], dtype=float)
    >>> lamp = Surfaces.from_triangles(square, solid_id=[0, 0], solid_names=("lamp",))
    >>> exitance = lamp_exitance(lamp, {"lamp": 8.0})     # 8 W over 4 m^2
    >>> float(np.sum(np.asarray(exitance) * np.asarray(lamp.area)))
    8.0
    """
    area = surfaces.area_by_solid()
    unrated = set(by_solid) - set(area)
    if unrated:
        msg = f"no such body in this surface set: {sorted(unrated)}; have {list(area)}"
        raise KeyError(msg)
    flat = [name for name in by_solid if area[name] == 0.0]
    if flat:
        msg = (
            f"body {flat} has no area, so it has no exitance: its facets are point sources, "
            "which carry radiant power in watts rather than an exitance in W/m^2. Pass those "
            "watts to Surfaces.with_optics(power=...) instead."
        )
        raise ValueError(msg)
    return surfaces.per_facet(
        {name: power / area[name] for name, power in by_solid.items()}, default=default
    )
