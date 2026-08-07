"""Named physical fields of a coupled RANS state, for march-log diagnostics.

A residual norm says how far the discrete equations are from being satisfied; it does not say which
field is still moving, or whether the solution has stopped changing at all. Those are different
questions, and a scalar case metric (a reattachment length, a drag coefficient) answers neither --
it is one number, often quantized by the mesh, so it can sit perfectly still while the fields behind
it drift by several per cent.

This module answers both, per equation, under one set of names (:func:`coupled_equation_names`):

* :func:`coupled_fields` exposes the coupled state as **named physical fields**, which
  :func:`~aquaflux.solve.field_change_metrics` turns into a per-field relative-change column -- has
  this field stopped moving?
* :func:`coupled_residuals` reports the **per-equation residual** on the march's own row-equilibrated
  measure -- is this equation satisfied?

Reading them together is what separates the two ways a march ends: every residual small means the
equations are solved, while a residual that will not fall beside a relative change already at ~1e-8
means the iterates have converged and something other than the step is limiting.

The split from the logger is deliberate: the *extraction* is specific to this state layout and lives
here; the *change measure* is generic and lives with the logger, so it can be tested on plain arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp

from .coupled import coupled_scaled_norm

__all__ = ["coupled_equation_names", "coupled_fields", "coupled_residuals"]

#: Velocity-component names in axis order, so a 2D case reads ``u, v`` and a 3D one ``u, v, w``.
_VELOCITY_NAMES = ("u", "v", "w")


def coupled_equation_names(dim: int) -> tuple[str, ...]:
    """The coupled state's solved equations, named, **in the order the flat state lays them out**.

    ``(u, v, w, p, k, omega)`` in three dimensions and ``(u, v, p, k, omega)`` in two: one name per
    equal-sized block of the flat layout ``[vel_0..vel_{dim-1}, pressure, k, omega]``. The single home
    for these names, so a per-block residual and a per-field change report the same equation under the
    same label and cannot drift apart.

    Parameters
    ----------
    dim : int
        Number of velocity components (the spatial dimension), at most 3.

    Returns
    -------
    tuple of str
        The ``dim + 3`` block names, in block order.

    Raises
    ------
    ValueError
        If ``dim`` exceeds the three named velocity components.

    Examples
    --------
    >>> coupled_equation_names(3)
    ('u', 'v', 'w', 'p', 'k', 'omega')
    >>> coupled_equation_names(2)
    ('u', 'v', 'p', 'k', 'omega')
    """
    if dim > len(_VELOCITY_NAMES):
        raise ValueError(f"dim {dim} exceeds the named velocity components {_VELOCITY_NAMES}")
    return (*_VELOCITY_NAMES[:dim], "p", "k", "omega")


def coupled_fields(coupled) -> Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]]:
    """Build ``state -> {"u": ..., "v": ..., "p": ..., "k": ..., "omega": ..., "nut": ...}``.

    The fields are the **physical** ones (each solved scalar is mapped back through its variable
    transform), so a log reads the same whether ``omega`` is solved directly or in log form.

    Three field-specific choices are made here rather than in the generic change measure:

    - **Velocity is split per component** (``u``, ``v``, ``w``), under
      :func:`coupled_equation_names`, so each component's change lines up with its own momentum
      equation's residual. A single vector entry would average the three, hiding a component that has
      stopped moving behind two that have not -- and on a flow with a weak cross-stream component
      that is exactly the one worth watching.
    - **Pressure is returned gauge-free** (its mean removed). Incompressible pressure is determined
      only up to an additive constant unless a boundary pins it, so a raw ``p`` could report a large
      relative change from a shift that carries no physical content at all.
    - **``nu_t`` is included** even though it is derived rather than solved. It is the field the
      momentum equations actually see from the turbulence model, so it is the one whose drift
      explains a stalling coupled solve -- and it moves when ``k`` and ``omega`` move in ways their
      individual norms can hide. Having no equation of its own, it is the one entry here with no
      counterpart in :func:`coupled_residuals`.

    Note that ``k``, ``omega`` and ``nu_t`` span orders of magnitude across a boundary layer, so a
    2-norm over them is dominated by the near-wall peak. That is usually the right emphasis (the
    near-wall region is where the stiffness is), but it does mean these columns report the near-wall
    field, not a domain average.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled case, used only for its state layout and closure.

    Returns
    -------
    callable
        ``state -> mapping of name to array``, ready for
        :func:`~aquaflux.solve.field_change_metrics`.

    Examples
    --------
    >>> metrics = field_change_metrics(coupled_fields(coupled))  # doctest: +SKIP
    >>> MarchLogger(metrics=metrics)  # doctest: +SKIP
    """
    momentum, turbulence = coupled.momentum, coupled.turbulence
    dim = coupled.layout.dim
    velocity_names = coupled_equation_names(dim)[:dim]

    def fields(state: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        flow, k, omega = coupled.physical_fields(state)
        velocity, pressure = momentum.unpack(flow)
        nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
        components = {name: velocity[:, i] for i, name in enumerate(velocity_names)}
        return {
            **components,
            "p": pressure - jnp.mean(pressure),  # gauge-free: see the docstring
            "k": k,
            "omega": omega,
            "nut": nu_t,
        }

    return fields


def coupled_residuals(
    coupled, continuation, reference_state: jnp.ndarray | None = None
) -> Callable[[jnp.ndarray], Mapping[str, float]]:
    """Build ``state -> {equation name: residual}`` on the march's own row-equilibrated measure.

    The march steers on a single scalar residual, which cannot say *which* equation is holding the
    solve up. This reports the same measure per equation: the row-equilibrated, field-normalized
    fractional change of each block (:meth:`~aquaflux.solve.RowScaledNorm.per_block`), named by
    :func:`coupled_equation_names`. Because the scalar measure is the Euclidean combination of exactly
    these numbers, they are read on the same scale as the residual column beside them -- a block near
    the total owns the residual, and the rest are already converged.

    **The scales are built at the PREVIOUS state, which is what makes these compose into the reported
    residual exactly.** The march re-derives the measure at the state each outer iteration *starts*
    from and holds it for the whole iteration -- every trial step, the acceptance test, and the norm it
    reports -- so a step's residual is ``norm_at_start(R(state_at_end))``. Building the scales at the
    state handed in here instead would measure the right residual vector in the *wrong* scales, and the
    per-equation rows would not add up to the number printed above them. So this holds the previous
    state and equilibrates against that.

    **Stateful by construction**, therefore: it must be called once per step, in order, from
    :meth:`~aquaflux.solve.MarchLogger.on_checkpoint` -- the same contract
    :func:`~aquaflux.solve.field_change_metrics` carries, and for the same reason. Calling it from an
    arbitrary probe corrupts the sequence.

    ``nu_t`` has no equation and so appears in :func:`coupled_fields` but not here.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled case, for its residual, layout and dimension.
    continuation
        The march's continuation engine, read at call time for its ``shift_policy`` -- the source of
        the per-row diagonal the equilibration divides by. Read late, not captured, so a refreshed
        segment's rebuilt diagonals are the ones used.
    reference_state : jnp.ndarray, optional
        The state the march's first observed step will start from -- its segment seed. Supply it
        whenever a fresh reporter is built partway through a march (a continuation rung rebuilds the
        case and its continuation, so it rebuilds this too), or that first step alone would be
        equilibrated at its own end state. ``None`` falls back to the first state seen, which is right
        only when the reporter is built for a march that has not started.

    Returns
    -------
    callable
        ``state -> mapping of equation name to float``, for :class:`~aquaflux.solve.MarchLogger`'s
        ``residuals``.

    Examples
    --------
    >>> MarchLogger(residuals=coupled_residuals(coupled, engine, seed))  # doctest: +SKIP
    """
    names = coupled_equation_names(coupled.layout.dim)
    # One-element list rather than `nonlocal`: the closure only ever rebinds it, and a mutable cell
    # keeps that visible at the point of use.
    equilibrate_at = [reference_state]

    def residuals(state: jnp.ndarray) -> Mapping[str, float]:
        at = state if equilibrate_at[0] is None else equilibrate_at[0]
        measure = coupled_scaled_norm(coupled, continuation.shift_policy, at)
        per_block = measure.per_block(coupled.residual(state))
        equilibrate_at[0] = state
        return {name: float(value) for name, value in zip(names, per_block, strict=True)}

    return residuals
