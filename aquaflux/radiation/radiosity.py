"""Light bouncing between surfaces, resolved to convergence rather than truncated.

A wall lit by a lamp re-emits, and what it re-emits lights every other wall, including the
ones that lit it. The ultraviolet-reactor literature usually stops short of solving that:
reflection is neglected, or replaced by mirror images of the source, or a Monte Carlo run is
capped at a handful of bounces. Measured against those treatments, ordinary stainless walls
account for errors of up to a third, and the uniformity of the field depends strongly on the
*diffuse* part of the wall's reflectance.

Here it is a linear system, so **the number of bounces is not a parameter**: the inverse of
``I - diag(rho) F`` is the infinite bounce sum, and solving it costs no more than a few dozen
matrix-vector products.

    H = F^M M + F (B - M) + H_external          irradiance on each facet
    B = M + rho * H                             what each facet sends back out

with ``F_ij`` the fraction of what leaves facet ``j`` that lands on facet ``i``. Eliminating
``H`` gives ``(I - diag(rho) F) B = M + rho * ((F^M - F) M + H_external)``, which is what is
actually solved.

⚠️ **Reflection here is purely DIFFUSE, and a scalar reflectance does not say that.** A wall
described only by the number 0.95 could scatter that light in every direction or send it off
like a mirror, and the two are not close: Hassanpour et al. (2023) measure a **10-47% spread in
log reduction between fully specular and fully diffuse walls at the same reflectivity of 0.95**.
Diffuse is the right default rather than merely the convenient one — Li et al. (2017) find that
diffuse reflection raises the reduction-equivalent fluence above specular, and the measurement
literature emphasizes it — but the assumption belongs beside the number, because a reflectance
supplied without it is an under-specified input.

⚠️ **The SOURCE's angular distribution is evaluated at the centroid direction too**, which
matters only when it is not Lambertian. A Lambertian source's emitted transfer conserves energy
exactly — the total landing on a closed box equals the total leaving, to 1.000000 at every
refinement — because its distribution cancels against the projected solid angle. A cosine-power
source's does not: on the same boxes, an exponent of 8 balances at 1.145, 0.970, 0.970 and 0.981
at 12, 48, 192 and 768 facets, and an exponent of 2 at 1.069, 1.007, 0.995 and 0.995. The error
shrinks as the receiving facets shrink and more directions are sampled, but it is a few percent
at usable resolutions. Subdivide a narrow source's surroundings, or read its result as carrying
that much slack.

⚠️ **The receiver is evaluated at one point — its centroid — and that is the model's remaining
approximation here.** The emitting facet is integrated exactly, so the row sums are exact to
about 1e-15 and the conditioning bound they give is real. Reciprocity is not: it sits around
24% on a closed box and **does not improve with refinement**, because shrinking the facets
brings their neighbours proportionally closer. Near-neighbour transfer is therefore apportioned
slightly differently from a fully integrated form factor, while the total leaving each facet is
exact. See :func:`reciprocity_residual`.

**What that costs in practice is much less than it sounds.** Global energy conservation — all the
light a lamp emits being absorbed somewhere — needs reciprocity, and per column the identity
``sum_i A_i F_ij = A_j`` is violated by up to 8.9%. But those column errors carry mixed signs and
very nearly cancel when summed over an enclosure: measured on closed boxes of 12, 48, 192, 432 and
768 facets, absorbed over emitted comes to 1.000000, 0.999868, 1.000112, 1.000083 and 1.000062.
So the field is conservative to about one part in ten thousand while any single pair of nearby
facets may exchange a quarter more or less than it should.

**``F`` is built from the PROJECTED solid angle.** A receiving facet is a surface, so light
arriving obliquely counts for less; the plain solid angle is the fluence-rate kernel and using
it here overstates every transfer by exactly a factor of two over a closed enclosure. That
distinction is the one this package has got wrong most often, and the row-sum metric below is
what catches it.

**What is frozen and what is live** is the other thing to get right, and both halves have been
wrong at some point in this design:

===========================  ==================================================================
frozen, built once           the projected solid angles, the source-side cosines, the
                             centroid separations, and the occlusion mask -- all ``n^2``
live, differentiated         reflectance, emission, radiant power, profile parameters,
                             occluder transmittance, and the absorption coefficient
===========================  ==================================================================

Freezing too much severs a gradient the module promises, and the loss is neither obvious nor
total: freezing the whole of ``F`` costs a few percent of the sensitivity to absorbance, and
freezing the visibility inside the geometry term costs about two thirds of the sensitivity to
transmittance. Both leave a finite, plausible-looking number behind. The rule that prevents it:
**anything promised a gradient must be computed outside the frozen arrays**, as an elementwise
multiply against them.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.absorption import Absorption, UniformAbsorption
from aquaflux.radiation.profiles import Lambertian
from aquaflux.radiation.solid_angle import projected_solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility
from aquaflux.solve import relative_residual_gmres, solve_linear
from aquaflux.vectors import dot

#: Relative residual the default solve stops at.
_DEFAULT_RTOL = 1e-10

__all__ = [
    "TransferMatrix",
    "build_transfer",
    "radiosity",
    "reciprocity_residual",
    "row_sum_error",
    "surface_irradiance",
]


class TransferMatrix(eqx.Module):
    """The frozen geometry of facet-to-facet transfer.

    Everything here is a function of shape and position alone, costs ``n^2`` to build, and never
    changes while a design study varies emission, reflectance or water quality. It is held
    apart from those so the expensive build happens once and the derivatives still reach the
    things that move.

    Attributes
    ----------
    geometric : jnp.ndarray, shape ``(n_facets, n_facets)``
        ``Omega_proj_ij / pi`` — the fraction of a *Lambertian* facet ``j``'s output reaching
        facet ``i``, before any blocking or absorption, with the diagonal zeroed. Row ``i`` is
        the receiver.
    source_cosine : jnp.ndarray, shape ``(n_facets, n_facets)``
        Cosine, at source ``j``, of the angle between its normal and the direction to receiver
        ``i``. Needed only to evaluate a non-Lambertian source's own angular distribution, which
        is live because its parameters are.
    separation : jnp.ndarray, shape ``(n_facets, n_facets)``
        Centroid-to-centroid distance, which is what turns a uniform absorption coefficient into
        a transmittance without re-walking any geometry.
    visibility : Visibility
        The frozen blocking mask, built for the facet centroids as receivers.
    """

    geometric: jnp.ndarray
    source_cosine: jnp.ndarray
    separation: jnp.ndarray
    visibility: Visibility

    @property
    def n_facets(self) -> int:
        """Number of facets."""
        return int(self.geometric.shape[0])


def build_transfer(
    surfaces: Surfaces,
    *,
    occluders=(),
    self_occlusion: bool = True,
    chunk_size: int = 256,
    **visibility_options,
) -> TransferMatrix:
    """Compute, once, the geometry of transfer between every pair of facets.

    Parameters
    ----------
    surfaces : Surfaces
        The facets. Only their geometry is read; the optical properties are supplied per call.
    occluders : sequence of Occluder, optional
        Analytic bodies between facets.
    self_occlusion : bool, optional
        Whether the facets block one another, which for a non-convex body they do.
    chunk_size : int, optional
        Receiving facets per pass of the solid-angle build, bounding its peak memory.
    **visibility_options
        Passed through to the visibility build.

    Returns
    -------
    TransferMatrix

    Notes
    -----
    The build is ``n^2`` in both time and memory: a thousand facets is a million entries per
    array and four such arrays, which is megabytes; ten thousand is a hundred million, which is
    gigabytes. The facet count, not the cell count, is what limits the surface system.
    """
    vertices = surfaces.vertices
    centroid = surfaces.centroid
    normal = surfaces.normal
    n_facets = surfaces.n_facets

    def solid_angles(receiver_slice):
        return jax.vmap(
            lambda point, facing: jax.vmap(
                lambda triangle: projected_solid_angle(point, facing, triangle)
            )(vertices)
        )(centroid[receiver_slice], normal[receiver_slice])

    rows = [
        solid_angles(slice(start, start + chunk_size)) for start in range(0, n_facets, chunk_size)
    ]
    geometric = jnp.concatenate(rows, axis=0) / jnp.pi

    # A facet cannot transfer to itself: its centroid lies in its own plane, where the contour
    # integral returns a whole hemisphere. Left in, every row sum is exactly one too large.
    geometric = geometric * ~jnp.eye(n_facets, dtype=bool)
    # A point source has no surface to receive on and no area to emit from; it reaches the
    # facets through the ordinary gather instead, as an external irradiance.
    areal = jnp.asarray(~surfaces.is_point_source)
    geometric = geometric * (areal[:, None] & areal[None, :])

    offset = centroid[:, None, :] - centroid[None, :, :]
    separation_squared = dot(offset, offset)
    separation = jnp.sqrt(jnp.where(separation_squared == 0.0, 0.0, separation_squared))
    safe = jnp.where(separation == 0.0, 1.0, separation)
    source_cosine = dot(offset, normal[None, :, :]) / safe

    return TransferMatrix(
        geometric=jax.lax.stop_gradient(geometric),
        source_cosine=jax.lax.stop_gradient(source_cosine),
        separation=jax.lax.stop_gradient(separation),
        visibility=build_visibility(
            occluders,
            surfaces,
            centroid,
            self_occlusion=self_occlusion,
            **visibility_options,
        ),
    )


def _live_transfer(transfer, surfaces, absorption, transmittance):
    """Assemble ``F`` and ``F^M`` from the frozen geometry and the values that vary.

    Both are elementwise products against frozen arrays, which is what keeps a derivative with
    respect to transmittance or absorbance from needing the ``n^2`` build again.
    """
    surviving = transfer.visibility.surviving(
        jnp.zeros(transfer.visibility.n_occluders) if transmittance is None else transmittance
    )
    if absorption is None:
        through = 1.0
    elif isinstance(absorption, UniformAbsorption):
        # Closed form off the frozen separation: no geometry is revisited, and the derivative
        # with respect to the coefficient is exact.
        through = jnp.exp(-absorption.coefficient * transfer.separation)
    else:
        through = jnp.exp(
            absorption.optical_depth(surfaces.centroid[None, :, :], surfaces.centroid[:, None, :])
            * -1.0
        )
    common = transfer.geometric * surviving * through

    # The emitted component leaves with each source's own distribution; the reflected component
    # leaves Lambertian by assumption. For a Lambertian source the two coincide exactly, which
    # is worth keeping as the reduction that pins the profile constants.
    lambertian = all(
        isinstance(profile, Lambertian)
        for kind, profile in enumerate(surfaces.profiles)
        if np.any((np.asarray(surfaces.profile_index) == kind) & ~surfaces.is_point_source)
    )
    if lambertian:
        return common, common
    index = np.asarray(surfaces.profile_index)
    # Point sources are already absent from the transfer, and asking one for a radiance is a
    # category error it refuses rather than answers — so they are skipped here too, or a scene
    # with a point lamp in it could not assemble at all.
    areal = ~surfaces.is_point_source
    relative = jnp.zeros_like(common)
    for kind, profile in enumerate(surfaces.profiles):
        sources = np.flatnonzero((index == kind) & areal)
        if not len(sources):
            continue
        weight = jnp.pi * profile.radiance_per_exitance(
            jnp.take(transfer.source_cosine, sources, axis=1)
        )
        relative = relative.at[:, sources].set(weight)
    return common, common * relative


def radiosity(
    transfer: TransferMatrix,
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
    transfer : TransferMatrix
        The frozen geometry, built for these facets.
    surfaces : Surfaces
        Supplies the live optical properties: emission, reflectance and profile parameters. Its
        geometry is not read here; that was frozen into ``transfer``.
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

    Notes
    -----
    The solve is matrix-free and stops on a **global relative** residual. The stock componentwise
    test is wrong for this system: most facets do not emit, so most rows of the right-hand side
    are zero, which turns a relative tolerance into an absolute one and stalls the solve.

    The system is not symmetrized to use a conjugate-gradient method. Doing so needs a left scale
    by ``1/(rho A)``, and zero reflectance is both the default and the value on every non-lamp
    surface of a real reactor.
    """
    emission = jnp.asarray(surfaces.emission, dtype=float)
    reflectance = jnp.asarray(surfaces.reflectance, dtype=float)
    reflected, emitted = _live_transfer(transfer, surfaces, absorption, transmittance)

    source = emission + reflectance * ((emitted - reflected) @ emission)
    if external_irradiance is not None:
        source = source + reflectance * jnp.asarray(external_irradiance, dtype=float)

    def matvec(x):
        return x - reflectance * (reflected @ x)

    return solve_linear(matvec, source, solver=solver or relative_residual_gmres(_DEFAULT_RTOL))


def surface_irradiance(
    transfer: TransferMatrix,
    surfaces: Surfaces,
    *,
    absorption: Absorption | None = None,
    transmittance=None,
    external_irradiance=None,
    solver=None,
):
    """Irradiance landing on each facet, from the solved radiosity.

    Returned as the solve already forms it — ``F^M M + F (B - M) + H_external`` — rather than as
    a second expression. Writing it as ``F B`` would be wrong wherever a source is not
    Lambertian, and would drift from the system actually solved.

    Point-source facets get ``NaN``: they have no surface for an irradiance to land on, and a
    zero there would read as a shadowed surface rather than as a category error.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        Irradiance per facet in W/m², and the solver's restart-cycle count.
    """
    emission = jnp.asarray(surfaces.emission, dtype=float)
    reflected, emitted = _live_transfer(transfer, surfaces, absorption, transmittance)
    outgoing, steps = radiosity(
        transfer,
        surfaces,
        absorption=absorption,
        transmittance=transmittance,
        external_irradiance=external_irradiance,
        solver=solver,
    )
    landing = emitted @ emission + reflected @ (outgoing - emission)
    if external_irradiance is not None:
        landing = landing + jnp.asarray(external_irradiance, dtype=float)
    return jnp.where(jnp.asarray(surfaces.is_point_source), jnp.nan, landing), steps


def row_sum_error(transfer: TransferMatrix) -> float:
    """Largest departure of a row of ``F`` from one — the standard figure of merit.

    Every bit of light leaving a facet inside a **closed** enclosure lands somewhere, so each row
    of the transfer matrix sums to one. That identity is what bounds the spectral radius of
    ``diag(rho) F`` by the largest reflectance, and with it the conditioning of the whole system,
    so it is worth reporting on any built matrix rather than assumed.

    On an open surface the rows sum to less than one by the fraction that escapes, and this
    number is then a measure of the opening rather than of an error.
    """
    return float(jnp.max(jnp.abs(jnp.sum(transfer.geometric, axis=1) - 1.0)))


def reciprocity_residual(transfer: TransferMatrix, area) -> float:
    """How far ``A_j F_ij`` sits from ``A_i F_ji``, against the largest entry of the matrix.

    ⚠️ **This is a diagnostic, not a gate, and it does not converge.** Reciprocity is exact for
    the double-area-integral form factor, where both facets are integrated over. This matrix
    evaluates the *receiver* at a single point — its centroid — and that one-point rule breaks
    reciprocity by an amount that refinement does not reduce: on a closed box it sits at
    **0.2421 at 12, 48, 192, 432 and 768 facets alike**, because shrinking the facets brings
    their neighbours proportionally closer and the geometry stays self-similar.

    So a large value here says the near-neighbour entries are apportioned differently from a
    fully-integrated form factor, not that the matrix is wrong. What *is* exact, and what the
    solve's conditioning actually rests on, is the row sum — see :func:`row_sum_error`, which
    holds to about 1e-15 on the same boxes. Reducing this one needs quadrature over the
    receiving facet as well, which costs another factor in the build.

    The comparison is against the largest entry rather than each pair's own magnitude. Pairs
    that transfer nothing — two facets of the same flat wall — hold values around 1e-18, and a
    per-pair relative measure turns that rounding noise into a residual near one while those
    pairs carry, measurably, 0.0000 of the total transfer.
    """
    # F[i, j] is what reaches receiver i from source j, so the emitting area sits on the
    # COLUMN index: A_j F[i, j] is compared with A_i F[j, i].
    area = jnp.asarray(area, dtype=float)
    weighted = area[None, :] * transfer.geometric
    largest = jnp.max(jnp.abs(weighted))
    return float(jnp.max(jnp.abs(weighted - weighted.T)) / jnp.where(largest == 0.0, 1.0, largest))
