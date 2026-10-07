"""A browser interface for aquaflux, run on this machine: ``aquaflux-ui <results>``.

The page has three sections. **Results** opens a run's output directory (or a VTK file), renders it
here with VTK and shows the images beside controls for what is drawn -- the field, its colour scale,
slice planes, threshold ranges, the snapshot -- and the march's convergence history. **Setup** opens,
edits, checks and saves a case file, and shows its mesh. **Run** holds the place of starting and
following a run.

This package imports neither the solver nor JAX, and the solver never imports it, so it is installed
separately (``pip install "aquaflux[ui]"``) and Results runs wherever results have been copied to.
Setup asks the solver installed beside it, through a separate solver process.
"""

from __future__ import annotations

from .history import ConvergenceHistory
from .scene import Layer, Pipeline, Scene, View
from .sources import ResultSource, RunDirectory, VtkFiles, open_source

__all__ = [
    "ConvergenceHistory",
    "Layer",
    "Pipeline",
    "ResultSource",
    "RunDirectory",
    "Scene",
    "View",
    "VtkFiles",
    "open_source",
]
