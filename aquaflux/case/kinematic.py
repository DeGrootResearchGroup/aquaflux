"""Pressure per unit density: the form OpenFOAM's incompressible solvers carry.

An incompressible OpenFOAM solver writes and reads ``p`` as the pressure divided by the density (its
dimensions are ``[0 2 -2 0 0 0 0]``, a velocity squared), whereas aquaflux solves for, checkpoints and
writes to VTK the pressure itself. The two agree only at a density of one. A case file's OpenFOAM
writer divides by the density and its OpenFOAM starting state multiplies by it, and both do it here,
so the two directions cannot disagree about what the factor is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kinematic_pressure", "pressure_from_kinematic"]


def kinematic_pressure(pressure: np.ndarray, density: float) -> np.ndarray:
    """The pressure per unit density an OpenFOAM field file holds.

    Parameters
    ----------
    pressure : np.ndarray
        The pressure, ``(n_cells,)``.
    density : float
        The fluid's density.

    Returns
    -------
    np.ndarray
        ``pressure / density``, ``(n_cells,)``.
    """
    return np.asarray(pressure) / density


def pressure_from_kinematic(kinematic: np.ndarray, density: float) -> np.ndarray:
    """The pressure a case solves for, from the pressure per unit density an OpenFOAM file holds.

    Parameters
    ----------
    kinematic : np.ndarray
        ``p`` as an OpenFOAM field file holds it, ``(n_cells,)``.
    density : float
        The fluid's density.

    Returns
    -------
    np.ndarray
        ``kinematic * density``, ``(n_cells,)``.
    """
    return np.asarray(kinematic) * density
