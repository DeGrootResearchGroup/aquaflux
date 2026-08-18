"""The tracer injector: one definition of *where* the species enters, shared by both codes.

The case injects a passive tracer over **part of the existing inlet patch**, which is what makes the
downstream field a genuine three-dimensional mixing problem rather than a uniform profile that any
scheme reproduces. The injected band sits low in the inlet channel, so the tracer is entrained into
the shear layer shed from the step lip, and is offset in span, so the side-wall corner flow acts on
it asymmetrically -- both features the flow already has and a uniform inlet would not probe.

**Why this lives in its own module.** The same profile has to reach two codes: aquaflux applies it
through a :class:`~aquaflux.boundary.DirichletField` on the inlet patch, and the OpenFOAM case gets
it as an explicit per-face list written by ``write_inlet_field.py``. Defined twice it would drift,
and a drifted injector is indistinguishable from a transport discrepancy -- exactly the confound the
case exists to avoid. So it is defined once, here, and the OpenFOAM values are *generated* from this
function rather than restated in the case dictionary.

**Why the edges are tapered rather than sharp.** A discontinuous top-hat is the more obvious choice
and makes the comparison worse: at a jump the two codes' gradient limiters break ties differently,
so the leading difference between them would be limiter behaviour at one cell rather than the
scalar transport being measured. A raised-cosine taper a few cells wide keeps the injection local
and sub-patch while leaving the profile resolved, so a difference downstream is attributable. The
taper is not a physical claim -- it is a choice to make the measurement mean something, and it costs
nothing in fidelity because both codes receive the identical face values.

The inlet plane is ``x = -0.03``, ``y`` in ``[0, 0.01]`` (the inlet channel, one step high) and
``z`` in ``[0, 0.04]`` (the span, ``4h`` between the side walls), discretized ``16 x 16``.
"""

from __future__ import annotations

import jax.numpy as jnp

#: Injected value inside the band (dimensionless tracer; the equation is linear in it, so the scale
#: is arbitrary and 1 makes "fraction of the injected concentration" the natural reading).
INJECTED_VALUE = 1.0

#: Wall-normal extent of the band, in metres. The inlet channel spans ``y`` in ``[0, 0.01]``; this
#: keeps the tracer in its lower half so it feeds the shear layer at the step lip (``y = 0``).
Y_LOW, Y_HIGH = 0.000, 0.006

#: Spanwise extent, in metres. The span is ``[0, 0.04]``, so its centre is ``0.02``: this band is
#: offset toward the ``z = 0`` side wall, which breaks the spanwise symmetry the geometry otherwise
#: has and makes the corner flow part of the problem.
Z_LOW, Z_HIGH = 0.004, 0.016

#: Raised-cosine taper width at each edge, in metres (see the module docstring on why not a jump).
Y_TAPER, Z_TAPER = 0.002, 0.003


def _window(coord, low: float, high: float, taper: float):
    """A smooth top-hat in one coordinate: 1 well inside ``[low, high]``, 0 outside, cosine between.

    Parameters
    ----------
    coord : array_like
        Coordinate values, any shape.
    low, high : float
        The band's outer edges, where the window reaches zero.
    taper : float
        Width of the raised-cosine edge, so the window is exactly 1 on ``[low + taper, high - taper]``.

    Returns
    -------
    jnp.ndarray
        The window value, in ``[0, 1]``, same shape as ``coord``.
    """
    # Distance into the band from whichever edge is nearer, clipped to the taper and mapped through
    # a raised cosine -- one expression rather than a three-way branch, which also keeps it
    # differentiable and vectorized.
    inward = jnp.minimum(coord - low, high - coord)
    ramp = jnp.clip(inward / taper, 0.0, 1.0)
    return 0.5 * (1.0 - jnp.cos(jnp.pi * ramp))


def injected_value(face_centroid):
    """The imposed tracer value at inlet face centroids -- the case's boundary profile.

    Parameters
    ----------
    face_centroid : array_like
        Face centroids, shape ``(n, 3)``, in metres.

    Returns
    -------
    jnp.ndarray
        Imposed value per face, shape ``(n,)``, in ``[0, INJECTED_VALUE]``.
    """
    y = face_centroid[..., 1]
    z = face_centroid[..., 2]
    return INJECTED_VALUE * _window(y, Y_LOW, Y_HIGH, Y_TAPER) * _window(z, Z_LOW, Z_HIGH, Z_TAPER)
