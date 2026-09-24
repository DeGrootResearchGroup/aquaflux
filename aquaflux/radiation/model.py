"""The assembled radiation model, and the three quantities a user asks it for.

Everything below this module is a piece: a solid-angle kernel, a surface set, a transfer matrix,
a shadow mask, an absorbing medium. Assembling them correctly is not obvious — the frozen
``n^2`` geometry has to be built for *these* receivers and *these* occluders, the point sources
have to be kept out of the facet-to-facet transfer and fed in as an external irradiance instead,
and the reflected part of every facet's output has to be re-gathered into the volume as a
Lambertian source whatever the source emitted like. Getting any of that wrong leaves a plausible
field rather than an error. :func:`build_radiation_model` does it once, and the three entry
points read off the result.

    model = build_radiation_model(cell_centres, surfaces, occluders=(...))   # once, n^2

    B = radiosity(model, surfaces)             # what each facet sends out,  (n_facets,)
    H = surface_irradiance(model, surfaces)    # what lands on each facet,   (n_facets,)
    G = fluence_rate(model, surfaces)          # fluence rate in the volume, (n_receivers,)

A wall lit by a lamp re-emits, and what it re-emits lights every other wall, including the
ones that lit it. The ultraviolet-reactor literature usually stops short of solving that:
reflection is neglected, or replaced by mirror images of the source, or a Monte Carlo run is
capped at a handful of bounces. Measured against those treatments, ordinary stainless walls
account for errors of up to a third, and the uniformity of the field depends strongly on the
*diffuse* part of the wall's reflectance.

Here it is a linear system, so **the number of bounces is not a parameter**: the inverse of
``I - diag(rho) F`` is the infinite bounce sum, and solving it costs no more than a few dozen
matrix-vector products.

    H = F^M M + F (B - M) + H_point + H_external    irradiance on each facet
    B = M + rho * H                                 what each facet sends back out

with ``F_ij`` the fraction of what leaves facet ``i`` that lands on facet ``j``, which is also
the weight with which ``j``'s radiosity lights ``i`` — one number, not two related by
reciprocity. Eliminating ``H`` gives
``(I - diag(rho) F) B = M + rho * ((F^M - F) M + H_point + H_external)``, which is what is
actually solved.

⚠️ **Reflection here is purely DIFFUSE, and a scalar reflectance does not say that.** A wall
described only by the number 0.95 could scatter that light in every direction or send it off
like a mirror, and the two are not close: Hassanpour et al. (2023) measure a **10-47% spread in
log reduction between fully specular and fully diffuse walls at the same reflectivity of 0.95**.
Diffuse is the right default rather than merely the convenient one — Li et al. (2017) find that
diffuse reflection raises the reduction-equivalent fluence above specular, and the measurement
literature emphasizes it — but the assumption belongs beside the number, because a reflectance
supplied without it is an under-specified input.

**Occluder geometry is a build argument; occluder transmittance is a call argument.** The split
is by when the value is needed rather than by what it describes: the shadow mask is frozen
``n^2`` geometry, and there is no later point in the data flow at which it could be frozen, while
what each body lets through is a number a study varies and a gradient must reach.

**The surface set is passed again at call time, and it must be the geometry the model was built
for.** The transfer matrix and both shadow masks were frozen from the build-time geometry, while
the direct gather reads the call-time vertices live -- so a moved surface set would give a field
lit from the new position through shadows cast from the old one, plausible and wrong. The model
therefore records a fingerprint of the geometry it was built for, and every call **refuses** a
surface set whose concrete geometry differs: ``surfaces.with_optics(...)`` is the cheap way to
sweep emission or reflectance, and moving a lamp with ``with_geometry`` needs a *new model*. The
one exception is geometry under tracing -- a gradient with respect to a lamp's position -- which
cannot be inspected and is what the live gather exists for: that derivative is taken with the
shadows frozen, as every other frozen quantity here is.

⚠️ **``G`` is a bare ``(n_receivers,)`` array in the receivers' own order**, because a cell field
is a bare array in every other signature in this library. There is no field type to wrap it in
and one should not be invented here.

**What this module does NOT do** is inject the result into a transport equation. A fluence rate
becomes a reaction source through the transport package's own volume-source seam, and that
subclass belongs to the consumer: radiation knows about surfaces and points in space, and
nothing about cells, fluxes or residuals. Keeping that fence one-way is why
:func:`build_radiation_model` takes an array of receiver positions rather than a mesh.
"""

from __future__ import annotations

import hashlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.absorption import Absorption
from aquaflux.radiation.gather import direct_fluence_rate, direct_irradiance
from aquaflux.radiation.profiles import Lambertian
from aquaflux.radiation.quadrature import TriangleQuadrature
from aquaflux.radiation.self_occlusion import SelfOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import TransferMatrix, build_transfer
from aquaflux.radiation.visibility import Visibility, build_visibility
from aquaflux.solve import relative_residual_gmres, solve_linear

__all__ = [
    "RadiationModel",
    "RadiationSettings",
    "build_radiation_model",
    "fluence_rate",
    "radiosity",
    "surface_irradiance",
]

#: Relative residual the default radiosity solve stops at.
_DEFAULT_RTOL = 1e-10


class RadiationSettings(eqx.Module):
    """Build-time choices that are not physics.

    **Every field defaults to** ``None``, **meaning "not set here"** — the value is then whatever
    the function it reaches decides, so each default is written down once beside its own
    reasoning rather than a second time here where it would drift. ``RadiationSettings(
    receiver_quadrature=12)`` changes one thing and nothing else.

    The membership test is whether a setting's reason can be stated without naming the thing it
    is attached to. It can for all of these: they describe how hard the build works and how much
    memory it may use, not what is being modelled. Anything whose reason names a lamp, a wall or
    a medium is physics and belongs on :class:`~aquaflux.radiation.surfaces.Surfaces` or on an
    :class:`~aquaflux.radiation.absorption.Absorption`, not here.

    Attributes
    ----------
    receiver_quadrature : int or TriangleQuadrature or None
        Points per receiving facet in the transfer build. The knob that controls reciprocity and
        with it the conservation of energy between facets; unset, six points. See
        :func:`~aquaflux.radiation.transfer.build_transfer`.
    transfer_chunk_size : int or None
        Receiving facets per pass of the ``n^2`` transfer build, bounding its peak memory.
    gather_chunk_size : int or None
        Receivers per pass of the volume gather, bounding its peak memory. A separate number
        from the one above because the two loops are over different things — facets against
        facets, and receivers against facets — and a scene may have far more of one than the
        other.
    self_occlusion : SelfOcclusion or None
        How the emitting facets are tested for shadowing one another, which for any non-convex
        body they do. Unset, one ray is cast per pair. Pass
        :class:`~aquaflux.radiation.self_occlusion.SilhouetteOcclusion` to clip exact fractions
        instead, which resolves a partly shadowed pair rather than rounding it to the nearer
        answer, at a cost that rises steeply with facet count. It also governs the volume
        receivers unless ``receiver_occlusion`` says otherwise -- see there.
    receiver_occlusion : SelfOcclusion or None
        How the facets are tested for shadowing the volume receivers. Unset, the receivers
        follow ``self_occlusion`` wherever that strategy can serve a point in the fluid, so
        switching self-occlusion off, or choosing the ray test, applies to both masks alike.
        Where it cannot -- the silhouette clip takes a share of a *projected* solid angle and
        needs a receiver normal that a point in the fluid does not have -- the receivers fall
        to the volume mask's own default, one ray per pair. Set this to choose differently.
    """

    receiver_quadrature: int | TriangleQuadrature | None = eqx.field(static=True, default=None)
    transfer_chunk_size: int | None = eqx.field(static=True, default=None)
    gather_chunk_size: int | None = eqx.field(static=True, default=None)
    self_occlusion: SelfOcclusion | None = eqx.field(static=True, default=None)
    receiver_occlusion: SelfOcclusion | None = eqx.field(static=True, default=None)

    def _passed(self, **named):
        """Drop the unset entries, so each reaches its own default rather than a copy of it."""
        return {name: value for name, value in named.items() if value is not None}

    def visibility_options(self) -> dict:
        """The subset the facet-to-facet shadow mask reads."""
        return self._passed(self_occlusion=self.self_occlusion)

    def receiver_visibility_options(self) -> dict:
        """The subset the volume-receiver shadow mask reads.

        Derived from ``self_occlusion`` unless ``receiver_occlusion`` is set, because building
        the two masks from separate choices is how they come to disagree about whether the
        surface shadows itself -- which is not visible in either mask on its own. A strategy
        that cannot serve a point in the fluid is not passed on at all, so the volume mask
        reaches its own default rather than a second copy of it written here.
        """
        if self.receiver_occlusion is not None:
            return {"self_occlusion": self.receiver_occlusion}
        if self.self_occlusion is None or not self.self_occlusion.serves_volume_receivers:
            return {}
        return {"self_occlusion": self.self_occlusion}

    def transfer_options(self) -> dict:
        """The subset :func:`~aquaflux.radiation.transfer.build_transfer` reads."""
        return {
            **self._passed(
                receiver_quadrature=self.receiver_quadrature,
                chunk_size=self.transfer_chunk_size,
            ),
            **self.visibility_options(),
        }

    def gather_options(self) -> dict:
        """The subset the volume gather reads."""
        return self._passed(chunk_size=self.gather_chunk_size)


class RadiationModel(eqx.Module):
    """Everything about a scene that its shape alone decides, computed once.

    Built by :func:`build_radiation_model`. Holds no optical property and no medium: those are
    supplied per call, so a design study that sweeps lamp power, wall reflectance or water
    quality pays the ``n^2`` build once and the solve many times, with the derivatives reaching
    every one of the swept values.

    Attributes
    ----------
    receivers : jnp.ndarray, shape ``(n_receivers, 3)``
        Where the fluence rate is wanted — cell centres, usually. Their order is the order of
        everything :func:`fluence_rate` returns.
    transfer : TransferMatrix
        Facet-to-facet geometry, including the facet-side shadow mask.
    receiver_visibility : Visibility
        Which bodies stand between which facets and which *receivers*. A second mask from the
        one inside ``transfer``, built for these points rather than for the facet centroids.
    settings : RadiationSettings
        What the build was told, kept so a later call chunks the gather the same way and so a
        result can say what produced it.
    geometry : str
        A fingerprint of the surface set's vertices and of which facets are point sources, the
        two things the frozen arrays were built from. Every call checks the surface set it is
        given against it.
    """

    receivers: jnp.ndarray
    transfer: TransferMatrix
    receiver_visibility: Visibility
    settings: RadiationSettings
    geometry: str = eqx.field(static=True)

    @property
    def n_facets(self) -> int:
        """Number of facets the transfer was built for."""
        return self.transfer.n_facets

    @property
    def n_receivers(self) -> int:
        """Number of points the fluence rate is evaluated at."""
        return int(self.receivers.shape[0])


def build_radiation_model(
    receivers,
    surfaces: Surfaces,
    *,
    occluders=(),
    settings: RadiationSettings | None = None,
    **visibility_options,
) -> RadiationModel:
    """Freeze everything a scene's geometry decides, once.

    Parameters
    ----------
    receivers : array_like, shape ``(n_receivers, 3)``
        Where the fluence rate is wanted. Cell centroids, for a field on a mesh; the whole mesh
        is not asked for, because nothing here reads anything else from one.
    surfaces : Surfaces
        The emitting set. Only its geometry is read — the optics are supplied per call.
    occluders : sequence of aquaflux.solids.Body, optional
        Analytic bodies standing between things. Their *geometry* is frozen here; what each
        lets through is a call argument.
    settings : RadiationSettings, optional
        Build-time choices. Unset fields take each function's own default.
    **visibility_options
        Passed through to both visibility builds.

    Returns
    -------
    RadiationModel

    Raises
    ------
    ValueError
        If a facet centroid or a receiver lies inside one of the bodies, which is a geometry
        error rather than a shadow — the raise comes from the visibility build.

    Notes
    -----
    The cost is ``n_facets^2`` for the transfer plus ``n_facets * n_receivers`` for the receiver
    mask, in both time and memory. The facet count is what limits the surface system; the
    receiver count only multiplies the cheaper of the two.

    Both masks are built against the same bodies, so a body that shadows a facet also shadows
    the cells behind it. Building them separately — the trap this function exists to close — is
    how a field ends up lit through a lamp sleeve that the surface solve correctly treated as
    opaque.
    """
    settings = RadiationSettings() if settings is None else settings
    receivers = jnp.asarray(receivers, dtype=float)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        msg = f"receivers must be (n_receivers, 3); got {receivers.shape}"
        raise ValueError(msg)

    transfer = build_transfer(
        surfaces, occluders=occluders, **settings.transfer_options(), **visibility_options
    )
    receiver_visibility = build_visibility(
        occluders,
        surfaces,
        receivers,
        **settings.receiver_visibility_options(),
        **visibility_options,
    )
    return RadiationModel(
        receivers=receivers,
        transfer=transfer,
        receiver_visibility=receiver_visibility,
        settings=settings,
        geometry=_geometry_fingerprint(surfaces),
    )


def _geometry_fingerprint(surfaces: Surfaces) -> str | None:
    """A digest of what the frozen arrays depend on, or ``None`` if the geometry is traced.

    Exact rather than toleranced: :meth:`~aquaflux.radiation.surfaces.Surfaces.with_optics`
    carries the same vertex array over, so a legitimate call matches bit for bit, and a surface
    set that differs by any amount was not the one the shadows were cast from.
    """
    if isinstance(surfaces.vertices, jax.core.Tracer):
        return None
    digest = hashlib.sha256(np.ascontiguousarray(surfaces.vertices, dtype=float).tobytes())
    digest.update(surfaces.is_point_source.tobytes())
    return digest.hexdigest()


def _check_geometry(model: RadiationModel, surfaces: Surfaces) -> None:
    """Refuse a surface set whose concrete geometry is not the one the model was built for."""
    found = _geometry_fingerprint(surfaces)
    if found is None or found == model.geometry:
        return
    msg = (
        f"this surface set's geometry is not the one the model was built for "
        f"({surfaces.n_facets} facets given, {model.n_facets} built). The transfer matrix and "
        "the shadow masks are frozen from the build-time geometry while the gather reads the "
        "vertices given here, so the field would be lit from one geometry through the shadows "
        "of another. To change optics, pass `surfaces.with_optics(...)` of the set the model was "
        "built from; to change geometry, build a new model."
    )
    raise ValueError(msg)


def radiosity(
    model: RadiationModel,
    surfaces: Surfaces,
    *,
    absorption: Absorption | None = None,
    transmittance=None,
    external_irradiance=None,
    solver=None,
):
    """Solve for the radiosity of every facet — what it sends out, emission plus reflection.

    Parameters
    ----------
    model : RadiationModel
        The frozen geometry, built for these facets.
    surfaces : Surfaces
        Supplies the live optical properties: emission, reflectance and profile parameters. Its
        geometry must be the one ``model`` was built from, and a different one is refused: the
        point-source arrivals read it live, while the transfer and its mask are frozen.
    absorption : Absorption, optional
        The medium between facets. A uniform coefficient is applied in closed form against the
        frozen separations; any other kind re-walks the geometry for every pair on every call,
        which is correct but costs the ``n^2`` build again.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What each analytic body lets through. Defaults to opaque.
    external_irradiance : array_like, shape ``(n_facets,)``, optional
        Irradiance on the facets from sources outside the surface system — point sources, which
        have no area to participate in the transfer. Obtain it from the ordinary gather.
    solver : lineax.AbstractLinearSolver, optional
        How to solve the system. The default is a matrix-free generalized minimal residual
        method stopping at a **global relative** residual of 1e-10. Supply your own to change
        the tolerance or the restart length.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        The radiosity per facet in W/m², and the solver's restart-cycle count.

        ⚠️ **Cycles, not matrix-vector products** — one cycle is up to the solver's ``restart``
        (120 by default), so a three is some hundreds of products. The count is the *cost* of
        the solve rather than part of its answer: a non-convergent solve raises, so what it
        tells you is how hard this scene was, which is also what makes it the honest reading to
        quote beside a field. Drop it where only the field is wanted, ``B, _ = radiosity(...)``.

    Notes
    -----
    The solve is matrix-free and stops on a **global relative** residual. The stock componentwise
    test is wrong for this system: most facets do not emit, so most rows of the right-hand side
    are zero, which turns a relative tolerance into an absolute one and stalls the solve.

    The system is not symmetrized to use a conjugate-gradient method. Doing so needs a left scale
    by ``1/(rho A)``, and zero reflectance is both the default and the value on every non-lamp
    surface of a real reactor.
    """
    _check_geometry(model, surfaces)
    emission = jnp.asarray(surfaces.emission, dtype=float)
    reflectance = jnp.asarray(surfaces.reflectance, dtype=float)
    reflected, emitted = model.transfer.assemble(surfaces, absorption, transmittance)

    source = emission + reflectance * ((emitted - reflected) @ emission)
    arriving = _point_source_irradiance(model, surfaces, absorption, transmittance)
    if external_irradiance is not None:
        arriving = (
            jnp.asarray(external_irradiance, dtype=float)
            if arriving is None
            else arriving + jnp.asarray(external_irradiance, dtype=float)
        )
    if arriving is not None:
        source = source + reflectance * arriving

    def matvec(x):
        return x - reflectance * (reflected @ x)

    return solve_linear(matvec, source, solver=solver or relative_residual_gmres(_DEFAULT_RTOL))


def surface_irradiance(
    model: RadiationModel,
    surfaces: Surfaces,
    *,
    absorption: Absorption | None = None,
    transmittance=None,
    external_irradiance=None,
    solver=None,
):
    """Irradiance landing on each facet, from the solved radiosity.

    Returned as the solve already forms it — ``F^M M + F (B - M) + H_point + H_external`` —
    rather than as a second expression. Writing it as ``F B`` would be wrong wherever a source
    is not Lambertian, and would drift from the system actually solved.

    Point-source facets get ``NaN``: they have no surface for an irradiance to land on, and a
    zero there would read as a shadowed surface rather than as a category error.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        Irradiance per facet in W/m², and the solver's restart-cycle count.
    """
    emission = jnp.asarray(surfaces.emission, dtype=float)
    reflected, emitted = model.transfer.assemble(surfaces, absorption, transmittance)
    outgoing, cycles = radiosity(
        model,
        surfaces,
        absorption=absorption,
        transmittance=transmittance,
        external_irradiance=external_irradiance,
        solver=solver,
    )
    landing = emitted @ emission + reflected @ (outgoing - emission)
    # The same two arrivals the solve's right-hand side carries, and for the same reason: the
    # set's own point sources are not in the transfer matrix, so their contribution has to be
    # added here too or a lamp-lit wall reads as dark.
    arriving = _point_source_irradiance(model, surfaces, absorption, transmittance)
    if arriving is not None:
        landing = landing + arriving
    if external_irradiance is not None:
        landing = landing + jnp.asarray(external_irradiance, dtype=float)
    return jnp.where(jnp.asarray(surfaces.is_point_source), jnp.nan, landing), cycles


def _point_source_irradiance(model, surfaces, absorption, transmittance):
    """What the point sources land on each facet, which the facet-to-facet transfer cannot.

    A point source has no area to emit from and no surface to receive on, so it is absent from
    the transfer matrix entirely. It still lights every facet, and in a real reactor it is
    usually the *only* thing that does. Wiring that in is the assembly step most easily left
    out, and leaving it out gives a dark enclosure rather than an error.

    Returns ``None`` when there are no point sources, so the common case adds no work.
    """
    if not surfaces.is_point_source.any():
        return None
    # Zeroing the areal exitance leaves only the point branch of the gather; the facets' own
    # emission reaches them through the transfer matrix and must not be counted twice.
    lamps = surfaces.with_optics(emission=jnp.zeros(surfaces.n_facets))
    landing = direct_irradiance(
        lamps,
        surfaces.centroid,
        surfaces.normal,
        absorption=absorption,
        visibility=model.transfer.visibility,
        transmittance=transmittance,
    )
    # A point source has no surface for an arrival to land on, and a zero-length normal, so its
    # own entry is meaningless rather than small.
    return jnp.where(jnp.asarray(surfaces.is_point_source), 0.0, landing)


def fluence_rate(
    model: RadiationModel,
    surfaces: Surfaces,
    *,
    absorption: Absorption | None = None,
    transmittance=None,
    external_irradiance=None,
    solver=None,
):
    """Fluence rate at the model's receivers, with interreflection closed.

    The quantity ultraviolet dose is computed from: radiant power crossing a point from every
    direction, per unit area, in W/m², carrying **no receiver cosine** because the organism it
    acts on has no orientation.

    Two contributions, gathered separately because they leave with different angular
    distributions:

    - what each facet **emits**, with its own profile, plus what the point sources radiate;
    - what each facet **reflects**, which is Lambertian by the diffuse-reflection assumption
      whatever the light that arrived was like.

    The second is the one a direct gather cannot give and the surface solve exists to supply. On
    ordinary stainless it is not a correction: reflection accounts for errors of up to a third
    when it is neglected.

    Parameters
    ----------
    model : RadiationModel
        The frozen geometry, built for these facets and these receivers.
    surfaces : Surfaces
        Supplies the live optics. Its geometry must be the one ``model`` was built from, and a
        different one is refused: the direct gather reads it live, while the shadows are frozen.
    absorption : Absorption, optional
        The medium. Applied both between facets and between facets and receivers.
    transmittance : array_like, shape ``(n_occluders,)``, optional
        What each body lets through. Defaults to opaque.
    external_irradiance : array_like, shape ``(n_facets,)``, optional
        Irradiance on the facets from sources outside this surface set entirely. The set's own
        point sources are **not** this — they are added automatically.
    solver : lineax.AbstractLinearSolver, optional
        How to solve the interreflection system.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        Fluence rate at each receiver in W/m², in the receivers' own order, and the solver's
        restart-cycle count. The count is returned for the same reason its two siblings return
        it — a field is not evidence of anything until the solve behind it is known to have
        converged.

    Notes
    -----
    The volume gather runs **twice**, once for the emitted field and once for the reflected one,
    and the second pass is paid even when every reflectance is zero. It cannot be skipped on
    that condition: reflectance is a traced value a gradient must reach, so there is nothing to
    branch on at trace time. A scene that genuinely has no reflecting surface is better served
    by :func:`~aquaflux.radiation.gather.direct_fluence_rate` on its own.
    """
    outgoing, cycles = radiosity(
        model,
        surfaces,
        absorption=absorption,
        transmittance=transmittance,
        external_irradiance=external_irradiance,
        solver=solver,
    )
    gather = model.settings.gather_options()
    common = {
        "absorption": absorption,
        "visibility": model.receiver_visibility,
        "transmittance": transmittance,
        **gather,
    }
    emitted = direct_fluence_rate(surfaces, model.receivers, **common)

    # What is left over after each facet's own emission is its reflected part, and that leaves
    # Lambertian whatever the source emitted like -- so it is re-gathered as a separate set with
    # a single Lambertian profile rather than with the emitters' own. Point sources carry power
    # rather than exitance and do not reflect, so they are zeroed out of this pass entirely, or
    # they would radiate a second time.
    #
    # The two lines overlap today and deliberately both stay: a point source has a zero normal,
    # so the Lambertian distribution imposed above already returns zero intensity towards every
    # receiver and the power would contribute nothing even if it were left in place. That is a
    # property of Lambertian rather than of this function, and it would stop holding the moment
    # the reflected distribution became anything else.
    areal = jnp.asarray(~surfaces.is_point_source)
    bounced = surfaces.with_optics(
        emission=jnp.where(areal, outgoing - surfaces.emission, 0.0),
        power=jnp.zeros(surfaces.n_facets),
        profiles=(Lambertian(),),
    )
    return emitted + direct_fluence_rate(bounced, model.receivers, **common), cycles
