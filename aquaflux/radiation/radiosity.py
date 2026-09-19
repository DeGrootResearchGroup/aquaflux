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

with ``F_ij`` the fraction of what leaves facet ``i`` that lands on facet ``j``, which is also
the weight with which ``j``'s radiosity lights ``i`` — one number, not two related by
reciprocity. Eliminating ``H`` gives
``(I - diag(rho) F) B = M + rho * ((F^M - F) M + H_external)``, which is what is actually solved.

⚠️ **Reflection here is purely DIFFUSE, and a scalar reflectance does not say that.** A wall
described only by the number 0.95 could scatter that light in every direction or send it off
like a mirror, and the two are not close: Hassanpour et al. (2023) measure a **10-47% spread in
log reduction between fully specular and fully diffuse walls at the same reflectivity of 0.95**.
Diffuse is the right default rather than merely the convenient one — Li et al. (2017) find that
diffuse reflection raises the reduction-equivalent fluence above specular, and the measurement
literature emphasizes it — but the assumption belongs beside the number, because a reflectance
supplied without it is an under-specified input.

**Both facets of every pair are integrated over, and they are integrated differently.** A
transfer factor is a double area integral. The sending facet is exact — the projected solid angle
is a closed form — and the receiving facet is quadrature, six points by default. Keeping the
source exact is what makes the row sums exact at every rule, and the row sums are what bound the
conditioning; integrating both by quadrature would trade that for exact reciprocity, which is the
worse way round. See :func:`build_transfer` for the point counts on offer and what each costs.

⚠️ **Refining the mesh does not improve reciprocity; only the quadrature does.** Shrinking the
facets of a closed box brings each one's neighbours proportionally closer, so the ratio the
quadrature error depends on never changes: ``reciprocity_residual`` measures 0.2421, 0.0281,
0.0078 and 0.0045 at 1, 3, 6 and 12 points per receiver, and the same four numbers again at every
refinement from 12 to 432 facets. It is a diagnostic and not a gate.

⚠️ **Global energy conservation follows reciprocity, and at one point per receiver it DEGRADES
under refinement.** For a small lamp in a large box, absorbed over emitted measures 1.000000,
0.999868, 0.982769, 0.975625 and 0.973077 at 12, 48, 192, 432 and 768 facets — 2.7% of the lamp's
output unaccounted for and still growing, because the facets nearest the lamp close in on it
while staying the same size relative to their separation. At the six-point default the same scene
gives 1.000000, 1.000212, 1.000171, 1.000120 and 1.000102, improving instead. ⚠️ A box in which
*every* facet emits balances to 1.000000 at every mesh and every rule, so it cannot see any of
this: each facet's error is its neighbour's and they cancel identically.

⚠️ **The SOURCE's angular distribution is still evaluated at ONE direction**, from its centroid to
the receiver's, and the receiver quadrature does not fix that. It matters only when the source is
not Lambertian — a Lambertian distribution cancels against the projected solid angle and balances
exactly at every refinement. A cosine-power source of exponent 8 balances at 1.086, 0.978, 0.982
and 0.987 at 12, 48, 192 and 432 facets, against 1.145, 0.970, 0.970 and 0.977 at one point per
receiver: the two agree to three figures from three points upward, because this error is not the
receiver's. What shrinks it is refining the mesh, which samples more directions. Subdivide a
narrow source's surroundings, or read its result as carrying a few percent of slack.

The same applies to the other three quantities that multiply the geometric term elementwise — the
centroid separation carrying absorption, the source cosine above, and the occlusion mask. All
three are one-point, necessarily: they are live and differentiable, so they cannot move inside the
frozen build. In a scene where a facet is partly shadowed, or the medium absorbs appreciably over
a facet's own width, those are the coarse approximations and not the receiver quadrature.

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
from aquaflux.radiation.quadrature import TriangleQuadrature, triangle_quadrature
from aquaflux.radiation.solid_angle import projected_solid_angle
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility
from aquaflux.solve import relative_residual_gmres, solve_linear
from aquaflux.vectors import dot

#: Relative residual the default solve stops at.
_DEFAULT_RTOL = 1e-10

#: Points per receiving facet in the default transfer build. Six is where the measured
#: cost/accuracy frontier turns over -- see :func:`build_transfer`.
_DEFAULT_RECEIVER_POINTS = 6

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
        The form factor from facet ``i`` to facet ``j`` — the fraction of what leaves ``i``
        Lambertian that lands on ``j``, before any blocking or absorption, with the diagonal
        zeroed. Equivalently, and this is how the solve reads it, the weight with which ``j``'s
        radiosity contributes to the irradiance on ``i``: those are the same number, not two
        related by reciprocity. **The area therefore belongs to the ROW index** — ``A_i``
        multiplies row ``i`` — which is what :func:`reciprocity_residual` weights by, and what
        an equal-area fixture cannot distinguish from the transpose.
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
    receiver_quadrature: TriangleQuadrature | int | None = None,
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
    receiver_quadrature : TriangleQuadrature or int, optional
        How finely to integrate over each *receiving* facet. An integer is a point count passed
        to :func:`~aquaflux.radiation.quadrature.triangle_quadrature`. Defaults to six points.
        It is the only knob that improves reciprocity, and it costs far less than its point
        count suggests — see the tables below.
    chunk_size : int, optional
        Receiving facets per pass of the solid-angle build, bounding its peak memory. Its
        meaning is unchanged by the quadrature: the points are accumulated one at a time within
        a pass, so a finer rule costs time and not memory.
    **visibility_options
        Passed through to the visibility build.

    Returns
    -------
    TransferMatrix

    Notes
    -----
    A transfer factor is a double area integral, over the sending facet and over the receiving
    one. The sending half is exact — :func:`~aquaflux.radiation.solid_angle.projected_solid_angle`
    is a closed form — so the receiving half is the whole of the discretization error, and the
    quadrature above is what controls it. Two consequences of keeping the source exact are worth
    stating, because they are what make a one-sided rule the right shape here rather than a
    half-finished one:

    - **The row sums stay exact at every point count.** Each quadrature point sees a closed
      enclosure, so its own row sums to one; the weights sum to one, so the average does too.
      Measured at 1e-15 for every rule. Integrating *both* facets by quadrature would instead
      give exact reciprocity and approximate row sums — the worse trade, since the row sum is
      what bounds the conditioning and what catches a wrong kernel.
    - **Reciprocity converges in the point count, not in the mesh.** Refining a closed box
      brings each facet's neighbours proportionally closer, so the ratio the error depends on
      does not change and the residual sits flat under refinement. Measured on boxes of 12 to
      432 facets, identically at each:

      ======  ======  ====================  ==========================================
      points  degree  reciprocity residual  worst column, ``sum_i A_i F_ij / A_j - 1``
      ======  ======  ====================  ==========================================
      1       1       0.2421                8.8%
      3       2       0.0281                1.5%
      6       4       0.0078                0.38%
      12      6       0.0045                0.18%
      ======  ======  ====================  ==========================================

    ⚠️ **A finer rule costs far less than its point count**, which is why six is the default
    rather than one. The build is limited by moving geometry through memory, not by evaluating
    the kernel, and every extra point on a receiver reuses the same triangles. Against the
    one-point build, measured as the median of five warm calls on closed boxes of 192 to 3072
    facets at the default ``chunk_size``, on a CPU backend with x64 on (JAX 0.10.2, macOS arm64,
    11 cores):

    ========  ==========  ==========  ===========  ===========
    facets    1 point     3 points    6 points     12 points
    ========  ==========  ==========  ===========  ===========
    192       0.18 s      1.01x       1.15x        1.09x
    768       0.54 s      1.00x       1.31x        1.60x
    1728      1.32 s      1.18x       1.37x        2.03x
    3072      2.54 s      1.27x       1.72x        3.63x
    ========  ==========  ==========  ===========  ===========

    Read those ratios as the shape and not to two figures: wall clock on a shared desktop
    carries a spread of about 20% run to run, which is wider than the gap between neighbouring
    columns at the small sizes.

    What is *not* integrated over the receiver: the centroid separation that carries absorption,
    the source-side cosine that carries a non-Lambertian profile, and the occlusion mask. All
    three multiply the geometric term elementwise so that they can stay live and differentiable,
    and moving them inside the quadrature would put them back inside the frozen ``n^2`` build.
    They remain one-point quantities, which is the coarser approximation in any scene where a
    facet is partly shadowed or the medium is strongly absorbing over a facet's own width.

    The build is ``n^2`` in both time and memory: a thousand facets is a million entries per
    array and four such arrays, which is megabytes; ten thousand is a hundred million, which is
    gigabytes. The facet count, not the cell count, is what limits the surface system.
    """
    if receiver_quadrature is None:
        receiver_quadrature = _DEFAULT_RECEIVER_POINTS
    if not isinstance(receiver_quadrature, TriangleQuadrature):
        receiver_quadrature = triangle_quadrature(int(receiver_quadrature))

    vertices = surfaces.vertices
    centroid = surfaces.centroid
    normal = surfaces.normal
    n_facets = surfaces.n_facets

    # (n_facets, n_points, 3) -- where on each receiving facet the transfer is sampled.
    sample = receiver_quadrature.points(vertices)
    weight = jnp.asarray(receiver_quadrature.weight)

    def solid_angles(receiver_slice):
        facing = normal[receiver_slice]

        def accumulate(total, sampled):
            weight_q, point_q = sampled
            at_point = jax.vmap(
                lambda point, facing_i: jax.vmap(
                    lambda triangle: projected_solid_angle(point, facing_i, triangle)
                )(vertices)
            )(point_q, facing)
            return total + weight_q * at_point, None

        # Scanned rather than vmapped over the quadrature points so the live intermediate stays
        # (receivers, n_facets) whatever the rule costs, which is what lets ``chunk_size`` keep
        # meaning the same thing it did with a single point per facet.
        rows, _ = jax.lax.scan(
            accumulate,
            jnp.zeros((facing.shape[0], n_facets)),
            (weight, jnp.swapaxes(sample[receiver_slice], 0, 1)),
        )
        return rows

    rows = [
        solid_angles(slice(start, start + chunk_size)) for start in range(0, n_facets, chunk_size)
    ]
    geometric = jnp.concatenate(rows, axis=0) / jnp.pi

    # A facet cannot transfer to itself: every quadrature point lies in its own plane, where the
    # contour integral returns a whole hemisphere. Left in, every row sum is exactly one too
    # large. A planar triangle really does see none of itself, so this is the exact value and
    # not a repair.
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
            # The receivers here ARE the facets, so each ray ends on the one it is aimed at and
            # must be told to ignore it. Without this every mutually visible pair reads as
            # blocked and a closed enclosure loses its interreflection entirely.
            receiver_facet=np.arange(surfaces.n_facets),
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
    """How far ``A_i F_ij`` sits from ``A_j F_ji``, against the largest entry of the matrix.

    ⚠️ **This is a diagnostic, not a gate.** Reciprocity is exact for the double-area-integral
    form factor, where both facets are integrated over. This matrix integrates the *source*
    exactly and the *receiver* by quadrature, so the residual here is that quadrature's error:
    it falls with the point count, and stays flat under mesh refinement, because refining a
    closed box brings each facet's neighbours proportionally closer and the geometry stays
    self-similar. On boxes of 12 to 432 facets it measures 0.2421, 0.0281, 0.0078 and 0.0045 at
    1, 3, 6 and 12 points per receiver, the same at every refinement.

    So a value here says how differently the near-neighbour entries are apportioned from a fully
    integrated form factor, not whether the matrix is wrong. What is exact at every point count,
    and what the solve's conditioning actually rests on, is the row sum — see
    :func:`row_sum_error`.

    ⚠️ **The area multiplies the ROW index.** ``transfer.geometric[i, j]`` is the form factor
    *from* ``i`` *to* ``j``, so reciprocity pairs ``A_i F_ij`` with ``A_j F_ji`` — the transpose
    of ``area[:, None] * geometric``. Weighting the column instead is the same expression on any
    fixture whose facets share one area, which every closed box built by subdividing a cube
    does; on two squares of area 1 and 100 it reports 0.9999 where the correct pairing reports
    0.0022.

    The comparison is against the largest entry rather than each pair's own magnitude. Pairs
    that transfer nothing — two facets of the same flat wall — hold values around 1e-18, and a
    per-pair relative measure turns that rounding noise into a residual near one while those
    pairs carry, measurably, 0.0000 of the total transfer.
    """
    area = jnp.asarray(area, dtype=float)
    weighted = area[:, None] * transfer.geometric
    largest = jnp.max(jnp.abs(weighted))
    return float(jnp.max(jnp.abs(weighted - weighted.T)) / jnp.where(largest == 0.0, 1.0, largest))
