"""What drives the momentum equation, and what that makes of the state.

A flow is set in motion in one of two ways, and the difference is not a flag on an otherwise common
solve -- it decides what the unknowns *are*:

* :class:`BoundaryDriven` -- the motion comes from the boundary conditions (a prescribed inlet
  velocity, a moving wall) and from whatever :class:`~aquaflux.flow.MomentumSource` terms act on the
  interior, a uniform driving force among them. The unknowns are the fields, and nothing about the
  solve is special.
* :class:`MassFlow` -- a uniform streamwise force held to a target bulk velocity. The force is not
  known in advance: it is an unknown **of the same solve**, a scalar Lagrange multiplier for the
  constraint ``<U_dir> - target = 0``. The state therefore carries one degree of freedom belonging to
  no cell, and the residual one row belonging to no cell, so the layout, the stopping measure and the
  converged answer all differ.

That second one is why a flow assembler carries a drive at all. A force that is simply *prescribed*
is a :class:`~aquaflux.flow.UniformBodyForce` source and needs nothing here; a force that is
**solved for** cannot be a source, because the residual has to write the current iterate into it on
every evaluation and the border column ``dR/dbeta = -V`` has to come from somewhere. Keeping the two
apart is what stops one field meaning a fixed input on one case and a live unknown on another.

Everything the bordered form needs is here rather than at the two solves that run it -- the flow-only
one (:func:`~aquaflux.flow.bulk_velocity_flow_solve`) and the coupled RANS one
(:func:`~aquaflux.turbulence.solve_coupled_mass_flow`). They differ only in which assembler they
border and how the velocity is read out of their state; the layout, the seed, the split, the border
vectors, the bulk-velocity average and the stopping test are one definition used by both.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.solve import Convergence, Euclidean, FieldLayout, GlobalDofs

if TYPE_CHECKING:
    from .momentum import MomentumContinuity

__all__ = [
    "BODY_FORCE",
    "BOUNDARY_DRIVEN",
    "MASS_FLOW_CONVERGENCE",
    "BoundaryDriven",
    "Drive",
    "MassFlow",
    "mass_flow_drive",
    "refuse_a_constraint_this_solve_cannot_hold",
]

#: The bordered state's name for the solved body force -- the block :class:`MassFlow` appends.
BODY_FORCE = "body_force"

#: The stopping test a mass-flow-constrained solve falls back on. The row-scaled measure an
#: unconstrained coupled solve defaults to has no form here: the constraint's border row has no
#: diagonal to equilibrate by. A caller may still ask for :class:`~aquaflux.solve.BlockScaled`, whose
#: border block shares the flow block's scale.
MASS_FLOW_CONVERGENCE = Convergence(measure=Euclidean(), rtol=1e-10, atol=1e-12)


class Drive(eqx.Module):
    """How a flow's momentum is forced, and the shape of state that makes.

    Two implementations: :class:`BoundaryDriven` and :class:`MassFlow`. Every member is abstract,
    because the whole point of the distinction is that a solve must not assume the unbordered shape
    -- a default would let a bordered drive inherit an answer that silently drops its multiplier.
    """

    @abc.abstractmethod
    def volumetric_force(self, dim: int) -> jnp.ndarray | None:
        """The uniform force per unit volume this drive applies, shape ``(dim,)``, or ``None``.

        ``None`` means this drive adds no force term at all -- distinct from a force that happens to
        be zero, which a mass-flow drive has at the start of every solve.

        Parameters
        ----------
        dim : int
            Spatial dimension.

        Returns
        -------
        jnp.ndarray or None
            The force per unit volume, shape ``(dim,)``, or ``None`` for no force term.
        """

    @abc.abstractmethod
    def layout(self, fields: FieldLayout) -> FieldLayout:
        """The solved state's layout, given the layout of the fields alone.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout -- the fields, with no border.

        Returns
        -------
        FieldLayout
            The layout the solve iterates on.
        """

    @abc.abstractmethod
    def driven_state(self, fields: FieldLayout, state: jnp.ndarray) -> jnp.ndarray:
        """The solve's initial vector, given an initial state of the fields alone.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout.
        state : jnp.ndarray
            The fields' initial state, shape ``(fields.size,)``.

        Returns
        -------
        jnp.ndarray
            The vector the solve starts from, shape ``(self.layout(fields).size,)``.
        """

    @abc.abstractmethod
    def fields_state(self, fields: FieldLayout, driven: jnp.ndarray) -> jnp.ndarray:
        """The fields alone, out of a vector in this drive's layout.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout.
        driven : jnp.ndarray
            A vector in ``self.layout(fields)``.

        Returns
        -------
        jnp.ndarray
            The fields, shape ``(fields.size,)``.
        """

    @abc.abstractmethod
    def settled(self, fields: FieldLayout, driven: jnp.ndarray) -> Drive:
        """This drive carrying whatever ``driven`` says its forcing converged to.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout.
        driven : jnp.ndarray
            A vector in ``self.layout(fields)``.

        Returns
        -------
        Drive
            The drive as the solve left it.
        """


class BoundaryDriven(Drive):
    """The flow is driven by its boundary conditions and its sources; the unknowns are the fields.

    An inlet-and-outlet duct, a lid-driven cavity, and a periodic channel pushed by a prescribed
    :class:`~aquaflux.flow.UniformBodyForce` are all this drive: whatever sets the flow in motion is
    already in the boundary conditions or in the source terms, so the solve is the ordinary one and
    the state is the fields.
    """

    def volumetric_force(self, dim: int) -> jnp.ndarray | None:
        """``None`` -- this drive adds no force of its own; a prescribed one is a source."""
        del dim
        return None

    def layout(self, fields: FieldLayout) -> FieldLayout:
        """``fields`` unchanged -- there is no border."""
        return fields

    def driven_state(self, fields: FieldLayout, state: jnp.ndarray) -> jnp.ndarray:
        """``state`` unchanged."""
        del fields
        return state

    def fields_state(self, fields: FieldLayout, driven: jnp.ndarray) -> jnp.ndarray:
        """``driven`` unchanged."""
        del fields
        return driven

    def settled(self, fields: FieldLayout, driven: jnp.ndarray) -> BoundaryDriven:
        """This drive -- it has no forcing for a solve to settle."""
        del fields, driven
        return self


class MassFlow(Drive):
    """A uniform streamwise force solved to hold the bulk velocity at ``target``.

    A streamwise-periodic channel prescribes velocity nowhere; it is driven to a target bulk
    (volume-averaged) velocity by a uniform body force along one axis. Holding that target with an
    *outer* controller -- solve at a fixed force, measure the bulk velocity, nudge the force -- can
    overshoot badly when the momentum operator changes between solves, so the force is instead a
    scalar Lagrange multiplier ``beta`` appended to the state, with the residual bordered by the
    constraint row:

        R_aug([w, beta]) = [ R(w; beta) ; <U_dir>(w) - target ].

    ``<U_dir> = target`` then holds at the converged root by construction, the bordered Jacobian
    ``[[J, a], [c^T, 0]]`` is assembled by automatic differentiation like any other, and the
    implicit-function-theorem adjoint carries the constraint -- so a gradient through the solve is
    the sensitivity *at fixed bulk velocity*.

    Attributes
    ----------
    target : float
        The bulk (volume-averaged) velocity component to hold.
    flow_direction : int
        The axis the bulk velocity is measured and the force applied along, static.
    force : jnp.ndarray
        The current value of the solved force per unit volume, a scalar along ``flow_direction``. A
        differentiable leaf: the residual writes each Newton iterate into it, and the converged solve
        leaves the answer here (:meth:`settled`). It is the multiplier's value, never a prescribed
        input -- a force that is prescribed is a :class:`~aquaflux.flow.UniformBodyForce` source.
    """

    target: float
    flow_direction: int = eqx.field(static=True, default=0)
    # Converted on the way in, so a drive written with a Python float and one written with an array
    # are the same pytree: the march compiles its step with the assembler as an argument, and a leaf
    # that is sometimes a float and sometimes an array recompiles it.
    force: jnp.ndarray = eqx.field(converter=jnp.asarray, default=0.0)

    def volumetric_force(self, dim: int) -> jnp.ndarray:
        """The current multiplier as a force vector, shape ``(dim,)``, zero off ``flow_direction``."""
        return jnp.zeros(dim).at[self.flow_direction].set(self.force)

    def layout(self, fields: FieldLayout) -> FieldLayout:
        """``fields`` with one more block: the single global dof holding the force."""
        return fields.appended(GlobalDofs(BODY_FORCE, 1))

    def _border(self, fields: FieldLayout) -> slice:
        """Where the force sits in the bordered vector, read off the declared layout."""
        return self.layout(fields).slice_of(BODY_FORCE)

    def join(self, fields: FieldLayout, state: jnp.ndarray, border: jnp.ndarray) -> jnp.ndarray:
        """A vector in this drive's layout, from a field-sized part and the scalar border entry.

        The one place the border is attached. Every consumer of the bordered form -- the seed, the
        augmented residual, the constraint preconditioner's output -- goes through it, so the declared
        layout is what decides where the entry sits rather than each site appending by hand.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout.
        state : jnp.ndarray
            The field part, shape ``(fields.size,)``.
        border : jnp.ndarray
            The scalar border entry.

        Returns
        -------
        jnp.ndarray
            The bordered vector, shape ``(self.layout(fields).size,)``.
        """
        return self.layout(fields).pack(*fields.unpack(state), jnp.atleast_1d(border))

    def split(self, fields: FieldLayout, driven: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The field part and the scalar border entry of a vector in this drive's layout.

        Parameters
        ----------
        fields : FieldLayout
            The assembler's own layout.
        driven : jnp.ndarray
            A vector in ``self.layout(fields)``.

        Returns
        -------
        tuple of jnp.ndarray
            The fields, shape ``(fields.size,)``, and the scalar border entry.
        """
        border = self._border(fields)
        return driven[: border.start], driven[border][0]

    def driven_state(self, fields: FieldLayout, state: jnp.ndarray) -> jnp.ndarray:
        """``state`` with the current force attached -- what the constrained solve starts from."""
        return self.join(fields, state, self.force)

    def fields_state(self, fields: FieldLayout, driven: jnp.ndarray) -> jnp.ndarray:
        """The fields, with the attached force dropped."""
        return self.split(fields, driven)[0]

    def settled(self, fields: FieldLayout, driven: jnp.ndarray) -> MassFlow:
        """This drive carrying the force ``driven`` ends at."""
        return eqx.tree_at(lambda d: d.force, self, self.split(fields, driven)[1])

    def forced(self, momentum: MomentumContinuity, beta: jnp.ndarray) -> MomentumContinuity:
        """``momentum`` with its drive carrying ``beta`` -- the assembler one Newton iterate sees.

        Parameters
        ----------
        momentum : MomentumContinuity
            The flow assembler, whose drive is this one.
        beta : jnp.ndarray
            The multiplier's value for this evaluation.

        Returns
        -------
        MomentumContinuity
            The assembler forced at ``beta``.
        """
        return eqx.tree_at(lambda m: m.drive.force, momentum, beta)

    def bulk_velocity(self, velocity: jnp.ndarray, volume: jnp.ndarray) -> jnp.ndarray:
        """The volume-averaged velocity component along ``flow_direction``.

        Parameters
        ----------
        velocity : jnp.ndarray
            Cell velocities, shape ``(n_cells, dim)``.
        volume : jnp.ndarray
            Cell volumes, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The scalar bulk velocity ``<U_dir>``.
        """
        return jnp.sum(velocity[:, self.flow_direction] * volume) / jnp.sum(volume)

    def constraint_vectors(self, momentum: MomentumContinuity) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The border column ``a`` and row ``c`` of the augmented Jacobian, as flat flow vectors.

        ``a = dR/dbeta = -V`` on the flow-direction velocity rows (the force enters as
        ``R = flux - beta V``); ``c = d<U_dir>/dw = V / sum(V)`` there, so ``c . w = <U_dir>``. Both are
        fixed by the geometry, which is why the constraint preconditioner can be built once from a
        concrete reference.

        Parameters
        ----------
        momentum : MomentumContinuity
            The flow assembler whose geometry sets both vectors.

        Returns
        -------
        tuple of jnp.ndarray
            ``(a, c)``, each shape ``((dim + 1) n_cells,)`` in the flow layout.
        """
        volume = momentum.geometry.cell.volume
        n_cells, dim = momentum.mesh.n_cells, momentum.mesh.dim
        pressure_zero = jnp.zeros(n_cells)
        force = jnp.zeros((n_cells, dim)).at[:, self.flow_direction].set(-volume)
        average = jnp.zeros((n_cells, dim)).at[:, self.flow_direction].set(volume / jnp.sum(volume))
        return momentum.pack(force, pressure_zero), momentum.pack(average, pressure_zero)

    def constraint_vectors_in(
        self, momentum: MomentumContinuity, layout: FieldLayout
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """:meth:`constraint_vectors` in an outer layout whose **first** block is the flow.

        ``beta`` enters only the momentum block, and ``<U_dir>`` reads only the velocity, so in a
        coupled state the border column and row are the flow block's own, padded with zeros over every
        other block. Written once here rather than at each residual that borders itself, and derived
        from the outer layout rather than from a count of its blocks, so a state that grows a field
        does not silently pad the wrong width.

        Parameters
        ----------
        momentum : MomentumContinuity
            The flow assembler whose geometry sets both vectors.
        layout : FieldLayout
            The unbordered outer layout, whose first block is the flow sub-state.

        Returns
        -------
        tuple of jnp.ndarray
            ``(a, c)``, each shape ``(layout.size,)``.
        """
        force_flow, average_flow = self.constraint_vectors(momentum)
        zeros = [
            block.unflatten(jnp.zeros(size), layout.n_cells)
            for block, size in zip(layout.blocks[1:], layout.sizes[1:], strict=True)
        ]
        return layout.pack(force_flow, *zeros), layout.pack(average_flow, *zeros)


#: The default drive: nothing forces the interior, so the boundary conditions and the sources do. A
#: shared instance, so an assembler built without naming a drive carries the same value every time
#: rather than a fresh object that compares equal but hashes apart.
BOUNDARY_DRIVEN = BoundaryDriven()


def mass_flow_drive(momentum: MomentumContinuity, caller: str) -> MassFlow:
    """``momentum``'s mass-flow drive, refused by name if it has none.

    Every entry point that borders a residual with a bulk-velocity constraint needs the same thing
    from the assembler, and the failure is worth catching here rather than where it would otherwise
    surface: the bordered residual writes each Newton iterate into ``momentum.drive.force``, which on a
    boundary-driven assembler is a tree path that does not exist, so the message would name a pytree
    rather than the mistake -- and it would arrive from inside a traced step.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler, or the momentum block of a coupled one.
    caller : str
        The public function to name in the message.

    Returns
    -------
    MassFlow
        The drive.

    Raises
    ------
    TypeError
        If the assembler is not driven by a :class:`MassFlow`.
    """
    drive = momentum.drive
    if not isinstance(drive, MassFlow):
        raise TypeError(
            f"{caller} holds a bulk velocity, so its momentum assembler must be built with "
            f"drive=MassFlow(target=...), got {type(drive).__name__}. A force that is prescribed "
            "rather than solved for is a UniformBodyForce source instead."
        )
    return drive


def refuse_a_constraint_this_solve_cannot_hold(momentum: MomentumContinuity, caller: str) -> None:
    """Refuse a :class:`MassFlow`-driven assembler at a solve that marches the fields alone.

    Such a solve would run perfectly well and answer the wrong question: the drive's force is a valid
    force, so the march would hold it **fixed at the seed** and converge to whatever bulk velocity that
    happens to produce, with nothing anywhere saying the constraint was dropped. That is the failure
    this whole distinction exists to make impossible, so it is refused rather than solved.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler, or the momentum block of a coupled one.
    caller : str
        The public function to name in the message.

    Raises
    ------
    TypeError
        If the assembler is driven by a :class:`MassFlow`.
    """
    if isinstance(momentum.drive, MassFlow):
        raise TypeError(
            f"{caller} marches the fields alone, so it cannot hold the bulk velocity this assembler's "
            "MassFlow drive asks for -- it would solve at the seed force and report nothing. Use the "
            "constrained solve instead, or, to solve at a force you are prescribing, build with "
            "sources=(UniformBodyForce(...),) and no drive."
        )
