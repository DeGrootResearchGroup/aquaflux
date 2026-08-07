"""Named physical fields of a coupled RANS state, for march-log diagnostics.

A residual norm says how far the discrete equations are from being satisfied; it does not say which
field is still moving, or whether the solution has stopped changing at all. Those are different
questions, and a scalar case metric (a reattachment length, a drag coefficient) answers neither --
it is one number, often quantized by the mesh, so it can sit perfectly still while the fields behind
it drift by several per cent.

This module exposes the coupled state as **named physical fields**, which
:func:`~aquaflux.solve.field_change_metrics` turns into a per-field relative-change column in the
march log. The split is deliberate: the *extraction* is specific to this state layout and lives here;
the *change measure* is generic and lives with the logger, so it can be tested on plain arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp

__all__ = ["coupled_fields"]


def coupled_fields(coupled) -> Callable[[jnp.ndarray], Mapping[str, jnp.ndarray]]:
    """Build ``state -> {"u": ..., "p": ..., "k": ..., "w": ..., "nut": ...}`` for a coupled RANS case.

    The fields are the **physical** ones (each solved scalar is mapped back through its variable
    transform), so a log reads the same whether ``omega`` is solved directly or in log form.

    Two field-specific choices are made here rather than in the generic change measure:

    - **Pressure is returned gauge-free** (its mean removed). Incompressible pressure is determined
      only up to an additive constant unless a boundary pins it, so a raw ``p`` could report a large
      relative change from a shift that carries no physical content at all.
    - **``nu_t`` is included** even though it is derived rather than solved. It is the field the
      momentum equations actually see from the turbulence model, so it is the one whose drift
      explains a stalling coupled solve -- and it moves when ``k`` and ``omega`` move in ways their
      individual norms can hide.

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

    def fields(state: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        flow, k, omega = coupled.physical_fields(state)
        velocity, pressure = momentum.unpack(flow)
        nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
        return {
            "u": velocity,
            "p": pressure - jnp.mean(pressure),  # gauge-free: see the docstring
            "k": k,
            "w": omega,
            "nut": nu_t,
        }

    return fields
